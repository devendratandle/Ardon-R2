//! A complete decoder-only transformer, wired for INFERENCE.
//!
//! This assembles the verified primitives (ops + infer) into a model that
//! actually generates: embedding → N × (attention, SwiGLU FFN) → final
//! norm → output projection, with a KV cache per layer so each new token
//! costs O(context) instead of O(context²).
//!
//! Architecture is the modern (Llama-family) one, which is what public
//! checkpoints ship: pre-normalization with RMSNorm, RoPE applied to
//! queries and keys, SwiGLU feed-forward, residual connections. That
//! matters practically — it is the layout a GGUF/safetensors loader can
//! fill directly, so this struct is the target format for weight loading
//! rather than an internal shape someone must convert to.
//!
//! Weights are plain `Vec<f32>` public fields with a documented layout:
//! no autograd tape, no graph, no builder ceremony. A loader fills them;
//! `forward_step` reads them.

use crate::infer::{attend_step, KvCache, Sampler};
use crate::ops::{matmul, rmsnorm, rope_inplace, swiglu};

/// Model shape. Everything the forward pass needs to know.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Model width (embedding size).
    pub dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab: usize,
    /// FFN inner width (typically ~8/3 × dim in Llama-family models).
    pub ffn_hidden: usize,
    /// Longest context this model will serve; sizes the KV caches.
    pub max_seq: usize,
    /// RoPE frequency base (10000 is the near-universal default).
    pub rope_base: f32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl Config {
    /// Per-head dimension. Requires `dim % n_heads == 0`, checked by
    /// [`Model::validate`].
    #[inline] pub fn head_dim(&self) -> usize { self.dim / self.n_heads }

    /// Total parameter count — the honest measure of model size, and what
    /// makes "how big can this get" a concrete question rather than a
    /// claim. Reported in the same units people quote (7B, 70B, 1T).
    pub fn n_params(&self) -> usize {
        let d = self.dim;
        let per_layer = 2 * d            // two RMSNorm weight vectors
            + 4 * d * d                  // Wq, Wk, Wv, Wo
            + 3 * d * self.ffn_hidden;   // W1, W2, W3
        self.vocab * d                   // token embedding
            + self.n_layers * per_layer
            + d                          // final norm
            + self.vocab * d             // output projection
    }
}

/// One transformer block's weights. Matrices are row-major `[in × out]`,
/// so a `1 × in` activation times an `in × out` matrix is a plain matmul
/// with no transpose — the layout checkpoints already use.
#[derive(Debug, Clone)]
pub struct Layer {
    /// RMSNorm weight before attention. `[dim]`
    pub attn_norm: Vec<f32>,
    /// Query/key/value/output projections. `[dim × dim]` each.
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    /// RMSNorm weight before the FFN. `[dim]`
    pub ffn_norm: Vec<f32>,
    /// SwiGLU FFN: gate `[dim × hidden]`, down `[hidden × dim]`, up `[dim × hidden]`.
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

impl Layer {
    pub fn zeros(cfg: &Config) -> Self {
        let (d, h) = (cfg.dim, cfg.ffn_hidden);
        Layer {
            attn_norm: vec![0.0; d],
            wq: vec![0.0; d * d], wk: vec![0.0; d * d],
            wv: vec![0.0; d * d], wo: vec![0.0; d * d],
            ffn_norm: vec![0.0; d],
            w1: vec![0.0; d * h], w2: vec![0.0; h * d], w3: vec![0.0; d * h],
        }
    }
}

/// The model: embedding table, blocks, final norm, output projection.
#[derive(Debug, Clone)]
pub struct Model {
    pub cfg: Config,
    /// Token embedding, `[vocab × dim]`.
    pub tok_embed: Vec<f32>,
    pub layers: Vec<Layer>,
    /// Final RMSNorm weight, `[dim]`.
    pub final_norm: Vec<f32>,
    /// Output projection to logits, `[dim × vocab]`.
    pub output: Vec<f32>,
}

impl Model {
    /// Allocate a correctly-shaped model with zero weights — the target a
    /// weight loader fills.
    pub fn zeros(cfg: Config) -> Self {
        Model {
            tok_embed: vec![0.0; cfg.vocab * cfg.dim],
            layers: (0..cfg.n_layers).map(|_| Layer::zeros(&cfg)).collect(),
            final_norm: vec![0.0; cfg.dim],
            output: vec![0.0; cfg.dim * cfg.vocab],
            cfg,
        }
    }

    /// Check the shape invariants a loader could violate. Called by
    /// `forward_step` callers via `new_caches`; cheap and worth it —
    /// a mis-shaped weight otherwise shows up as silent garbage output.
    pub fn validate(&self) -> Result<(), String> {
        let c = &self.cfg;
        if c.dim == 0 || c.n_heads == 0 || c.vocab == 0 {
            return Err("Config: dim, n_heads and vocab must be non-zero".into());
        }
        if c.dim % c.n_heads != 0 {
            return Err(format!("Config: dim {} not divisible by n_heads {}", c.dim, c.n_heads));
        }
        if c.head_dim() % 2 != 0 {
            return Err(format!("Config: head_dim {} must be even (RoPE rotates pairs)", c.head_dim()));
        }
        if self.tok_embed.len() != c.vocab * c.dim {
            return Err(format!("tok_embed: expected {}, got {}", c.vocab * c.dim, self.tok_embed.len()));
        }
        if self.output.len() != c.dim * c.vocab {
            return Err(format!("output: expected {}, got {}", c.dim * c.vocab, self.output.len()));
        }
        if self.layers.len() != c.n_layers {
            return Err(format!("layers: expected {}, got {}", c.n_layers, self.layers.len()));
        }
        for (i, l) in self.layers.iter().enumerate() {
            let (d, h) = (c.dim, c.ffn_hidden);
            for (name, got, want) in [
                ("attn_norm", l.attn_norm.len(), d), ("ffn_norm", l.ffn_norm.len(), d),
                ("wq", l.wq.len(), d * d), ("wk", l.wk.len(), d * d),
                ("wv", l.wv.len(), d * d), ("wo", l.wo.len(), d * d),
                ("w1", l.w1.len(), d * h), ("w2", l.w2.len(), h * d), ("w3", l.w3.len(), d * h),
            ] {
                if got != want {
                    return Err(format!("layer {i} {name}: expected {want}, got {got}"));
                }
            }
        }
        Ok(())
    }

    /// One KV cache per layer, sized to `max_seq`. Validates the model
    /// first so a shape error surfaces before any compute.
    pub fn new_caches(&self) -> Result<Vec<KvCache>, String> {
        self.validate()?;
        let c = &self.cfg;
        Ok((0..c.n_layers)
            .map(|_| KvCache::new(c.n_heads, c.head_dim(), c.max_seq))
            .collect())
    }

    /// Run ONE token at position `pos` and return logits over the vocab.
    ///
    /// `caches` must come from [`new_caches`] and carries all history —
    /// this is what makes the step O(context) rather than O(context²).
    /// `pos` must equal the current cache length (the next free slot);
    /// a mismatch is an error rather than silently corrupt attention.
    pub fn forward_step(&self, token: usize, pos: usize, caches: &mut [KvCache])
        -> Result<Vec<f32>, String>
    {
        let c = &self.cfg;
        if token >= c.vocab {
            return Err(format!("token {} out of vocab {}", token, c.vocab));
        }
        if caches.len() != c.n_layers {
            return Err(format!("caches: expected {}, got {}", c.n_layers, caches.len()));
        }
        let (d, hd) = (c.dim, c.head_dim());

        // Embedding row for this token.
        let mut x = self.tok_embed[token * d..token * d + d].to_vec();

        for (li, layer) in self.layers.iter().enumerate() {
            let cache = &mut caches[li];
            if cache.len() != pos {
                return Err(format!(
                    "layer {li}: pos {pos} does not match cache length {} \
                     (tokens must be fed in order)", cache.len()));
            }

            // ── Attention (pre-norm) ──
            let h = rmsnorm(&x, &layer.attn_norm, c.eps);
            let mut q = matmul(&h, &layer.wq, 1, d, d);
            let mut k = matmul(&h, &layer.wk, 1, d, d);
            let v = matmul(&h, &layer.wv, 1, d, d);

            // RoPE encodes position by ROTATING q and k, per head. Because
            // the rotation is applied before caching, cached keys already
            // carry their own position — which is exactly why a cached key
            // stays valid for every later query.
            for head in 0..c.n_heads {
                rope_inplace(&mut q[head * hd..(head + 1) * hd], pos, c.rope_base);
                rope_inplace(&mut k[head * hd..(head + 1) * hd], pos, c.rope_base);
            }

            // Cache THEN attend, so the token attends to itself as well —
            // the causal mask, expressed structurally.
            cache.append(&k, &v)?;
            let a = attend_step(&q, cache);
            let o = matmul(&a, &layer.wo, 1, d, d);
            for (xi, oi) in x.iter_mut().zip(&o) { *xi += oi; }   // residual

            // ── SwiGLU feed-forward (pre-norm) ──
            let h2 = rmsnorm(&x, &layer.ffn_norm, c.eps);
            let gate = matmul(&h2, &layer.w1, 1, d, c.ffn_hidden);
            let up   = matmul(&h2, &layer.w3, 1, d, c.ffn_hidden);
            let act  = swiglu(&gate, &up);
            let down = matmul(&act, &layer.w2, 1, c.ffn_hidden, d);
            for (xi, di) in x.iter_mut().zip(&down) { *xi += di; } // residual
        }

        let x = rmsnorm(&x, &self.final_norm, c.eps);
        Ok(matmul(&x, &self.output, 1, d, c.vocab))
    }

    /// Feed a prompt and generate `max_new` tokens.
    ///
    /// Returns only the NEW tokens. The prompt is processed one token at a
    /// time through the same path as generation ("prefill"), so there is a
    /// single code path to get right — and the cache makes re-reading the
    /// prompt unnecessary for every subsequent token.
    ///
    /// `stop` ends generation early (an end-of-sequence id).
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new: usize,
        sampler: &mut Sampler,
        stop: Option<usize>,
    ) -> Result<Vec<usize>, String> {
        if prompt.is_empty() {
            return Err("generate: prompt must contain at least one token".into());
        }
        let mut caches = self.new_caches()?;
        if prompt.len() + max_new > self.cfg.max_seq {
            return Err(format!(
                "generate: prompt {} + max_new {} exceeds max_seq {}",
                prompt.len(), max_new, self.cfg.max_seq));
        }

        // Prefill: every prompt token except the last only fills the cache.
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.forward_step(t, pos, &mut caches)?;
            pos += 1;
        }

        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let next = sampler.sample(&logits);
            if Some(next) == stop { break; }
            out.push(next);
            if pos >= self.cfg.max_seq { break; }
            logits = self.forward_step(next, pos, &mut caches)?;
            pos += 1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::SamplerConfig;

    fn cfg() -> Config {
        Config { dim: 16, n_heads: 4, n_layers: 3, vocab: 11,
                 ffn_hidden: 24, max_seq: 32, rope_base: 10000.0, eps: 1e-5 }
    }

    /// Deterministic pseudo-random weights — reproducible without an RNG
    /// dependency, and varied enough that a wrong wiring shows up.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.113 + seed).sin()) * 0.35).collect()
    }

    fn demo_model() -> Model {
        let c = cfg();
        let mut m = Model::zeros(c);
        m.tok_embed = fill(c.vocab * c.dim, 0.7);
        m.final_norm = vec![1.0; c.dim];
        m.output = fill(c.dim * c.vocab, 1.9);
        for (i, l) in m.layers.iter_mut().enumerate() {
            let s = i as f32 * 3.1;
            l.attn_norm = vec![1.0; c.dim];
            l.ffn_norm = vec![1.0; c.dim];
            l.wq = fill(c.dim * c.dim, s + 0.2);
            l.wk = fill(c.dim * c.dim, s + 0.5);
            l.wv = fill(c.dim * c.dim, s + 0.9);
            l.wo = fill(c.dim * c.dim, s + 1.3);
            l.w1 = fill(c.dim * c.ffn_hidden, s + 1.7);
            l.w2 = fill(c.ffn_hidden * c.dim, s + 2.1);
            l.w3 = fill(c.dim * c.ffn_hidden, s + 2.5);
        }
        m
    }

    #[test]
    fn cached_generation_matches_recompute_from_scratch() {
        // THE end-to-end invariant: the KV cache changes cost, never
        // output. At every step, incremental decoding must produce the
        // same logits as replaying the whole prefix into a fresh cache.
        let m = demo_model();
        let tokens = [3usize, 7, 1, 0, 9, 4, 2];

        let mut caches = m.new_caches().unwrap();
        for (t, &tok) in tokens.iter().enumerate() {
            let incremental = m.forward_step(tok, t, &mut caches).unwrap();

            // Reference: fresh cache, replay 0..=t, take the last logits.
            let mut fresh = m.new_caches().unwrap();
            let mut reference = Vec::new();
            for (p, &tk) in tokens[..=t].iter().enumerate() {
                reference = m.forward_step(tk, p, &mut fresh).unwrap();
            }

            assert_eq!(incremental.len(), m.cfg.vocab);
            for (i, (a, b)) in incremental.iter().zip(&reference).enumerate() {
                assert!((a - b).abs() < 1e-4,
                    "step {t}, logit {i}: cached {a} != recomputed {b}");
            }
        }
    }

    #[test]
    fn generate_is_reproducible_and_bounded() {
        let m = demo_model();
        let cfgs = SamplerConfig { temperature: 0.8, top_k: 3, ..Default::default() };
        let run = |seed| {
            let mut s = Sampler::new(seed, cfgs);
            m.generate(&[2, 5], 6, &mut s, None).unwrap()
        };
        let a = run(99);
        assert_eq!(a.len(), 6, "must produce exactly max_new tokens");
        assert!(a.iter().all(|&t| t < m.cfg.vocab), "every token in vocab");
        assert_eq!(a, run(99), "same seed must replay identically");
    }

    #[test]
    fn greedy_generation_is_deterministic() {
        let m = demo_model();
        let mut s1 = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
        let mut s2 = Sampler::new(2, SamplerConfig { temperature: 0.0, ..Default::default() });
        // Greedy ignores the seed entirely — different seeds, same output.
        assert_eq!(m.generate(&[1], 5, &mut s1, None).unwrap(),
                   m.generate(&[1], 5, &mut s2, None).unwrap());
    }

    #[test]
    fn stop_token_ends_generation() {
        let m = demo_model();
        let mut s = Sampler::new(5, SamplerConfig { temperature: 0.0, ..Default::default() });
        let full = m.generate(&[4], 8, &mut s, None).unwrap();
        // Greedy is deterministic, so stopping on its first token yields none.
        let mut s2 = Sampler::new(5, SamplerConfig { temperature: 0.0, ..Default::default() });
        let stopped = m.generate(&[4], 8, &mut s2, Some(full[0])).unwrap();
        assert!(stopped.is_empty(), "generation must stop at the stop token");
    }

    #[test]
    fn out_of_order_positions_are_rejected() {
        // Feeding a wrong position would silently corrupt attention;
        // it must be a loud error instead.
        let m = demo_model();
        let mut caches = m.new_caches().unwrap();
        m.forward_step(1, 0, &mut caches).unwrap();
        let err = m.forward_step(2, 5, &mut caches).unwrap_err();
        assert!(err.contains("does not match cache length"));
    }

    #[test]
    fn shape_and_range_errors_are_caught() {
        let m = demo_model();
        let mut caches = m.new_caches().unwrap();
        assert!(m.forward_step(999, 0, &mut caches).unwrap_err().contains("out of vocab"));

        let mut bad = demo_model();
        bad.layers[1].wq.truncate(3);
        assert!(bad.validate().unwrap_err().contains("layer 1 wq"));

        let mut odd = Model::zeros(Config { n_heads: 5, dim: 15, ..cfg() });
        odd.cfg.n_heads = 5; odd.cfg.dim = 15;
        assert!(odd.validate().is_err(), "odd head_dim must be rejected (RoPE pairs)");
    }

    #[test]
    fn context_limit_is_enforced_before_compute() {
        let m = demo_model();
        let mut s = Sampler::new(1, SamplerConfig::default());
        let err = m.generate(&[1, 2], m.cfg.max_seq, &mut s, None).unwrap_err();
        assert!(err.contains("exceeds max_seq"));
    }

    #[test]
    fn param_count_matches_the_allocated_weights() {
        // n_params is quoted as model size, so it must equal reality.
        let m = demo_model();
        let actual = m.tok_embed.len() + m.final_norm.len() + m.output.len()
            + m.layers.iter().map(|l| {
                l.attn_norm.len() + l.ffn_norm.len()
                + l.wq.len() + l.wk.len() + l.wv.len() + l.wo.len()
                + l.w1.len() + l.w2.len() + l.w3.len()
            }).sum::<usize>();
        assert_eq!(m.cfg.n_params(), actual);
    }
}
