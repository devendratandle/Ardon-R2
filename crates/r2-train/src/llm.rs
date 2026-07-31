//! Train the model that R2 actually serves.
//!
//! The point of this module is that there is **one architecture**, not
//! two. It trains exactly the network `r2_tensor::model::Model` runs —
//! pre-norm RMSNorm, RoPE, grouped-query attention, SwiGLU — and exports
//! straight into that struct. No conversion layer, no "training format"
//! versus "serving format", and therefore no class of bug where a model
//! learns one thing and serves another.
//!
//! The proof that this holds is a test, not a comment: after training,
//! the tape's logits and the served model's logits must agree.

use r2_autograd::{Tape, Var};
use r2_tensor::model::{Config, Layer, Model};

use crate::optim::Adam;

/// A trainable model: the same shape as [`Model`], held as flat parameter
/// blocks so the optimizer sees one contiguous vector per tensor.
pub struct Trainer {
    pub cfg: Config,
    /// Parameter blocks in a fixed order (see [`Trainer::block_shapes`]).
    pub params: Vec<Vec<f32>>,
    pub opt: Adam,
    pub step: u64,
}

/// Per-layer parameter blocks, in the order they are stored.
const PER_LAYER: [&str; 9] =
    ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3"];

impl Trainer {
    /// Block layout: `[tok_embed, final_norm, output, then 9 per layer]`.
    pub fn block_shapes(cfg: &Config) -> Vec<usize> {
        let (d, kv, h) = (cfg.dim, cfg.kv_dim(), cfg.ffn_hidden);
        let mut s = vec![cfg.vocab * d, d, d * cfg.vocab];
        for _ in 0..cfg.n_layers {
            s.extend_from_slice(&[d, d * d, d * kv, d * kv, d * d, d, d * h, h * d, d * h]);
        }
        s
    }

    /// Initialize with small random weights and norms at 1.0.
    ///
    /// Scale is `1/sqrt(fan_in)`: too large and the residual stream
    /// explodes through the layers, too small and gradients vanish before
    /// they reach the early blocks. RMSNorm weights start at 1 (identity),
    /// which is what every reference implementation does — starting them
    /// at 0 would zero the signal and the model would never train.
    pub fn new(cfg: Config, lr: f32, seed: u64) -> Result<Trainer, String> {
        cfg.validate()?;
        let shapes = Self::block_shapes(&cfg);
        let mut rng = seed | 1;
        let mut next = || {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            ((rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        };
        let d = cfg.dim;
        let params: Vec<Vec<f32>> = shapes.iter().enumerate().map(|(i, &n)| {
            let is_norm = i == 1 || (i >= 3 && {
                let k = (i - 3) % 9;
                k == 0 || k == 5
            });
            if is_norm {
                vec![1.0; n]
            } else {
                let scale = 1.0 / (d as f32).sqrt();
                (0..n).map(|_| next() * 2.0 * scale).collect()
            }
        }).collect();

        let total: usize = shapes.iter().sum();
        let opt = Adam::new(total, lr);
        Ok(Trainer { cfg, params, opt, step: 0 })
    }

    /// Total trainable parameters — equals `Config::n_params`.
    pub fn n_params(&self) -> usize { self.params.iter().map(|p| p.len()).sum() }

    fn layer_block(&self, layer: usize, part: usize) -> usize { 3 + layer * 9 + part }

    /// Forward one sequence through the tape, returning the logits var.
    /// Mirrors `Model::forward_step` over a whole sequence at once —
    /// same ops, same order, same RoPE.
    fn forward(&self, tape: &mut Tape, tokens: &[usize], leaves: &[Var]) -> Var {
        let c = &self.cfg;
        let (d, kv, hd, t) = (c.dim, c.kv_dim(), c.head_dim(), tokens.len());

        // Embedding as a one-hot matmul, so gradients reach the table.
        let mut onehot = vec![0.0f32; t * c.vocab];
        for (i, &tok) in tokens.iter().enumerate() { onehot[i * c.vocab + tok] = 1.0; }
        let oh = tape.leaf(onehot, false);
        let mut x = tape.matmul(oh, leaves[0], t, c.vocab, d);

        for l in 0..c.n_layers {
            let b = |p: usize| leaves[self.layer_block(l, p)];

            // ── Attention ──
            let h = tape.rmsnorm(x, b(0), d, c.eps);
            let q = tape.matmul(h, b(1), t, d, d);
            let k = tape.matmul(h, b(2), t, d, kv);
            let v = tape.matmul(h, b(3), t, d, kv);
            let q = tape.rope(q, t, c.n_heads, hd, c.rope_base);
            let k = tape.rope(k, t, c.n_kv_heads, hd, c.rope_base);

            // Per-head attention, with K/V heads shared across a group of
            // query heads (GQA) — the same mapping the server uses.
            let group = c.n_heads / c.n_kv_heads;
            let mut heads: Vec<Var> = Vec::with_capacity(c.n_heads);
            for qh in 0..c.n_heads {
                let kvh = qh / group;
                let qs = tape.slice_cols(q, t, c.n_heads * hd, qh * hd, hd);
                let ks = tape.slice_cols(k, t, c.n_kv_heads * hd, kvh * hd, hd);
                let vs = tape.slice_cols(v, t, c.n_kv_heads * hd, kvh * hd, hd);
                let kt = tape.transpose(ks, t, hd);
                let sc = tape.matmul(qs, kt, t, hd, t);
                let sc = tape.scale_mask_causal(sc, t, 1.0 / (hd as f32).sqrt());
                let at = tape.softmax_rows(sc, t);
                heads.push(tape.matmul(at, vs, t, t, hd));
            }
            let ctx = tape.concat_cols(&heads, t, hd);
            let o = tape.matmul(ctx, b(4), t, d, d);
            x = tape.add(x, o);

            // ── SwiGLU FFN ──
            let h2 = tape.rmsnorm(x, b(5), d, c.eps);
            let gate = tape.matmul(h2, b(6), t, d, c.ffn_hidden);
            let up = tape.matmul(h2, b(8), t, d, c.ffn_hidden);
            let act = tape.silu(gate);
            let gated = tape.mul(act, up);
            let down = tape.matmul(gated, b(7), t, c.ffn_hidden, d);
            x = tape.add(x, down);
        }

        let xn = tape.rmsnorm(x, leaves[1], d, c.eps);
        tape.matmul(xn, leaves[2], t, d, c.vocab)
    }

    /// One optimizer step over a batch of (input, target) sequences.
    /// Returns the mean loss.
    pub fn train_step(&mut self, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<f32, String> {
        if batch.is_empty() { return Err("train_step: empty batch".into()); }
        let shapes = Self::block_shapes(&self.cfg);
        let mut grads: Vec<Vec<f32>> = shapes.iter().map(|&n| vec![0.0; n]).collect();
        let mut total = 0.0f32;

        for (inp, tgt) in batch {
            let mut tape = Tape::new();
            let leaves: Vec<Var> = self.params.iter()
                .map(|p| tape.leaf(p.clone(), true)).collect();
            let logits = self.forward(&mut tape, inp, &leaves);
            let loss = tape.softmax_ce(logits, self.cfg.vocab, tgt.clone());
            total += tape.value(loss)[0];
            tape.backward(loss);
            for (g, lv) in grads.iter_mut().zip(&leaves) {
                for (a, b) in g.iter_mut().zip(tape.grad(*lv)) { *a += b; }
            }
        }

        let n = batch.len() as f32;
        let mut flat: Vec<f32> = Vec::with_capacity(self.opt.len());
        for g in &grads { flat.extend(g.iter().map(|x| x / n)); }
        let mut params_flat: Vec<f32> = self.params.iter().flatten().copied().collect();
        self.opt.step(&mut params_flat, &flat)?;

        let mut off = 0;
        for p in self.params.iter_mut() {
            let n = p.len();
            p.copy_from_slice(&params_flat[off..off + n]);
            off += n;
        }
        self.step += 1;
        Ok(total / n)
    }

    /// Export into the model R2 serves. This is the whole point: the
    /// trained parameters go straight into the serving struct with no
    /// reshaping, because they were trained in that layout.
    pub fn to_model(&self) -> Result<Model, String> {
        let c = self.cfg;
        let mut m = Model::zeros(c);
        m.tok_embed = self.params[0].clone();
        m.final_norm = self.params[1].clone();
        m.output = self.params[2].clone();
        for l in 0..c.n_layers {
            let g = |p: usize| self.params[self.layer_block(l, p)].clone();
            m.layers[l] = Layer {
                attn_norm: g(0), wq: g(1), wk: g(2), wv: g(3), wo: g(4),
                ffn_norm: g(5), w1: g(6), w2: g(7), w3: g(8),
            };
        }
        m.validate()?;
        Ok(m)
    }

    /// Names of the parameter blocks, for diagnostics.
    pub fn block_names(&self) -> Vec<String> {
        let mut v = vec!["tok_embed".into(), "final_norm".into(), "output".into()];
        for l in 0..self.cfg.n_layers {
            for p in PER_LAYER { v.push(format!("layer.{l}.{p}")); }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_tensor::infer::{Sampler, SamplerConfig};

    fn tiny() -> Config {
        Config { dim: 16, n_heads: 4, n_kv_heads: 2, n_layers: 2, vocab: 10,
                 ffn_hidden: 32, max_seq: 16, rope_base: 10000.0, eps: 1e-5 }
    }

    #[test]
    fn block_layout_matches_the_config_parameter_count() {
        let c = tiny();
        let t = Trainer::new(c, 0.01, 1).unwrap();
        assert_eq!(t.n_params(), c.n_params(),
            "the trainer must hold exactly the parameters the model declares");
    }

    #[test]
    fn training_reduces_loss_and_learns_the_task() {
        // Learn to continue a fixed sequence — small, but it can only be
        // solved by actually using attention over previous positions.
        let c = tiny();
        let mut t = Trainer::new(c, 0.02, 7).unwrap();
        let batch = vec![
            (vec![1, 2, 3, 4], vec![2, 3, 4, 5]),
            (vec![5, 6, 7, 8], vec![6, 7, 8, 9]),
        ];
        let first = t.train_step(&batch).unwrap();
        let mut last = first;
        for _ in 0..120 { last = t.train_step(&batch).unwrap(); }
        assert!(last < first * 0.5,
            "loss should fall substantially: {first} -> {last}");
    }

    #[test]
    fn trained_model_serves_identically_to_the_tape() {
        // THE UNIFICATION INVARIANT: what was trained is what is served.
        // The tape's logits for a sequence must equal the served model's
        // logits, step by step — one architecture, not two.
        let c = tiny();
        let mut t = Trainer::new(c, 0.02, 3).unwrap();
        let batch = vec![(vec![1, 2, 3, 4], vec![2, 3, 4, 5])];
        for _ in 0..30 { t.train_step(&batch).unwrap(); }

        let tokens = [1usize, 2, 3, 4];
        // Tape logits for the whole sequence.
        let mut tape = Tape::new();
        let leaves: Vec<Var> = t.params.iter().map(|p| tape.leaf(p.clone(), false)).collect();
        let lg = t.forward(&mut tape, &tokens, &leaves);
        let tape_logits = tape.value(lg).to_vec();

        // Served logits, one token at a time through the KV cache.
        let m = t.to_model().unwrap();
        let mut caches = m.new_caches().unwrap();
        for (i, &tok) in tokens.iter().enumerate() {
            let served = m.forward_step(tok, i, &mut caches).unwrap();
            let row = &tape_logits[i * c.vocab..(i + 1) * c.vocab];
            for (j, (&a, &b)) in served.iter().zip(row.iter()).enumerate() {
                assert!((a - b).abs() < 1e-3,
                    "position {i}, logit {j}: served {a} != trained {b}");
            }
        }
    }

    #[test]
    fn a_trained_model_round_trips_through_the_r2_format() {
        let c = tiny();
        let mut t = Trainer::new(c, 0.02, 11).unwrap();
        let batch = vec![(vec![1, 2, 3], vec![2, 3, 4])];
        for _ in 0..20 { t.train_step(&batch).unwrap(); }
        let m = t.to_model().unwrap();

        let dir = std::env::temp_dir().join(format!("r2trained-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        m.save_dir(&dir).unwrap();
        let back = Model::load_dir(&dir).unwrap();

        let mut s1 = Sampler::new(5, SamplerConfig { temperature: 0.0, ..Default::default() });
        let mut s2 = Sampler::new(5, SamplerConfig { temperature: 0.0, ..Default::default() });
        assert_eq!(m.generate(&[1], 5, &mut s1, None).unwrap(),
                   back.generate(&[1], 5, &mut s2, None).unwrap(),
                   "train -> save -> load -> serve must be lossless");
        let _ = std::fs::remove_dir_all(dir);
    }
}
