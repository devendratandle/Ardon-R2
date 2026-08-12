//! Inference — the KV cache, incremental attention, and sampling.
//!
//! This is the DEPLOY half of "train and deploy". Training (r2-autograd,
//! r2-train) computes gradients over a whole sequence at once; serving a
//! model is a different shape: tokens arrive one at a time and each new
//! token must attend to everything before it. Recomputing the whole
//! prefix per token is O(n²) work per step and makes generation
//! quadratic overall — unusable past a few hundred tokens. The fix, and
//! the reason every serving stack has one, is a **KV cache**: the keys
//! and values for past positions are computed once and kept, so a step
//! costs O(n) instead of O(n²).
//!
//! THE CORRECTNESS INVARIANT (tested): incremental decoding with the
//! cache produces bit-comparable output to recomputing the full sequence
//! from scratch. The cache is an optimization, never a change in
//! behaviour — same property the distributed trainer holds against
//! single-device (see r2-train::distributed).
//!
//! No allocator tricks, no unsafe, no dependencies: capacity is reserved
//! up front so a generation loop performs no reallocation, which is what
//! keeps latency flat as the context grows.

/// Per-layer cache of past keys and values for one sequence.
///
/// Layout is `[pos][head][dim]` flattened — position-major so appending a
/// step is a contiguous write and attention over the prefix walks memory
/// forward. Capacity is fixed at construction (`max_seq`), so a decode
/// loop never reallocates.
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Heads and per-head dim.
    pub n_heads: usize,
    pub head_dim: usize,
    /// Positions currently held.
    len: usize,
    /// Maximum positions this cache can hold.
    pub max_seq: usize,
}

impl KvCache {
    pub fn new(n_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let stride = n_heads * head_dim;
        KvCache {
            k: vec![0.0; stride * max_seq],
            v: vec![0.0; stride * max_seq],
            n_heads, head_dim, len: 0, max_seq,
        }
    }

    /// Positions currently cached.
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Elements per position across all heads.
    #[inline] pub fn stride(&self) -> usize { self.n_heads * self.head_dim }

    /// Forget everything — reuse the allocation for the next sequence.
    pub fn clear(&mut self) { self.len = 0; }

    /// Append one position's keys and values (each `n_heads * head_dim`).
    /// Errors instead of growing: a serving loop must know its context
    /// limit up front rather than silently reallocating mid-stream.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), String> {
        let stride = self.stride();
        if k.len() != stride || v.len() != stride {
            return Err(format!(
                "KvCache::append: expected {} values per step, got k={} v={}",
                stride, k.len(), v.len()));
        }
        if self.len >= self.max_seq {
            return Err(format!(
                "KvCache::append: context full ({} positions)", self.max_seq));
        }
        let off = self.len * stride;
        self.k[off..off + stride].copy_from_slice(k);
        self.v[off..off + stride].copy_from_slice(v);
        self.len += 1;
        Ok(())
    }

    /// Key vector for (position, head).
    #[inline]
    pub fn key(&self, pos: usize, head: usize) -> &[f32] {
        let off = pos * self.stride() + head * self.head_dim;
        &self.k[off..off + self.head_dim]
    }

    /// Value vector for (position, head).
    #[inline]
    pub fn value(&self, pos: usize, head: usize) -> &[f32] {
        let off = pos * self.stride() + head * self.head_dim;
        &self.v[off..off + self.head_dim]
    }
}

/// One decode step of multi-head attention against the cache.
///
/// `q` is the current token's query for all heads (`n_heads * head_dim`);
/// the cache must ALREADY contain this position's key/value (append
/// first, then attend — that is what makes the token attend to itself,
/// matching the causal mask used in training).
///
/// Returns the attention output, `n_heads * head_dim`, ready for the
/// output projection. Softmax is max-shifted for numerical stability, so
/// long contexts cannot overflow the exponential.
pub fn attend_step(q: &[f32], cache: &KvCache) -> Vec<f32> {
    // Plain multi-head attention is the special case where every query
    // head has its own K/V head.
    attend_step_grouped(q, cache, cache.n_heads)
}

/// Grouped-query attention (GQA): `n_q_heads` query heads share the
/// cache's `cache.n_heads` key/value heads, in contiguous groups.
///
/// Modern models (Llama-3, Mistral, 70B-class) do this because the KV
/// cache — not the weights — dominates serving memory: 32 query heads
/// over 8 K/V heads cuts the cache 4× at almost no quality cost. Query
/// head `i` reads K/V head `i / (n_q_heads / n_kv_heads)`, which is the
/// grouping the checkpoints are trained with.
///
/// With `n_q_heads == cache.n_heads` this is exactly multi-head attention
/// (group size 1), so one code path serves both — proven by test.
pub fn attend_step_grouped(q: &[f32], cache: &KvCache, n_q_heads: usize) -> Vec<f32> {
    let (kvh, d, n) = (cache.n_heads, cache.head_dim, cache.len());
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; n_q_heads * d];
    if n == 0 || kvh == 0 || n_q_heads == 0 { return out; }
    // Queries per K/V head. Integer division: callers validate divisibility
    // (Model::validate), and a non-divisible split would mis-group heads.
    let group = (n_q_heads / kvh).max(1);

    let mut scores = vec![0.0f32; n];
    for head in 0..n_q_heads {
        let qh = &q[head * d..head * d + d];
        let kv_head = (head / group).min(kvh - 1);
        // Scores against every cached position (causal by construction:
        // the cache only ever holds positions <= current).
        for (p, s) in scores.iter_mut().enumerate() {
            let kh = cache.key(p, kv_head);
            *s = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        // Max-shifted softmax.
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
        let inv = 1.0 / sum;
        // Weighted sum of values.
        let o = &mut out[head * d..head * d + d];
        for (p, &w) in scores.iter().enumerate() {
            let w = w * inv;
            let vh = cache.value(p, kv_head);
            for (oi, &vi) in o.iter_mut().zip(vh) { *oi += w * vi; }
        }
    }
    out
}

// ─── Sampling ────────────────────────────────────────────────────────

/// Deterministic decoding: the highest-scoring token. Ties resolve to the
/// lowest index so a run is reproducible.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best = i; }
    }
    best
}

/// Sampling controls. `temperature <= 0` means greedy (argmax), which is
/// also the exact limit of temperature → 0, so the knob is continuous.
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability reaches `top_p` (1.0 = disabled).
    pub top_p: f32,
    /// Keep only the `top_k` highest-probability tokens (0 = disabled).
    pub top_k: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig { temperature: 1.0, top_p: 1.0, top_k: 0 }
    }
}

/// Reproducible sampler. Same seed + same logits ⇒ same tokens, which is
/// what makes a generation bug reproducible and a regression testable.
#[derive(Debug, Clone)]
pub struct Sampler {
    state: u64,
    pub cfg: SamplerConfig,
}

impl Sampler {
    pub fn new(seed: u64, cfg: SamplerConfig) -> Self {
        // Any non-zero state; xorshift64* is fine for sampling.
        Sampler { state: seed | 1, cfg }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.state = x;
        // Top 24 bits → [0,1).
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Pick one token from `logits` under the configured policy.
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() { return 0; }
        if self.cfg.temperature <= 0.0 { return argmax(logits); }

        // Temperature-scaled, max-shifted softmax.
        let inv_t = 1.0 / self.cfg.temperature;
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
            .map(|(i, &l)| (i, ((l - m) * inv_t).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        for (_, p) in probs.iter_mut() { *p /= sum; }

        // Highest probability first — required by both top-k and top-p.
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.cfg.top_k > 0 && self.cfg.top_k < probs.len() {
            probs.truncate(self.cfg.top_k);
        }
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (_, p) in probs.iter() {
                cum += *p;
                keep += 1;
                if cum >= self.cfg.top_p { break; }
            }
            probs.truncate(keep.max(1)); // never leave an empty set
        }

        // Renormalize the surviving set and draw.
        let total: f32 = probs.iter().map(|(_, p)| *p).sum();
        let r = self.next_f32() * total;
        let mut acc = 0.0f32;
        for (i, p) in &probs {
            acc += *p;
            if r <= acc { return *i; }
        }
        probs.last().map(|(i, _)| *i).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: full-sequence attention for the LAST position, computed
    /// from scratch over all keys/values (what training does). The cache
    /// must reproduce this exactly.
    fn attend_full(q: &[f32], ks: &[Vec<f32>], vs: &[Vec<f32>],
                   h: usize, d: usize) -> Vec<f32> {
        let scale = 1.0 / (d as f32).sqrt();
        let n = ks.len();
        let mut out = vec![0.0f32; h * d];
        for head in 0..h {
            let qh = &q[head * d..head * d + d];
            let mut sc: Vec<f32> = (0..n).map(|p| {
                let kh = &ks[p][head * d..head * d + d];
                qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale
            }).collect();
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in sc.iter_mut() { *s = (*s - m).exp(); sum += *s; }
            for (p, s) in sc.iter().enumerate() {
                let w = s / sum;
                let vh = &vs[p][head * d..head * d + d];
                for (oi, &vi) in out[head * d..head * d + d].iter_mut().zip(vh) {
                    *oi += w * vi;
                }
            }
        }
        out
    }

    // Deterministic pseudo-data so the test is reproducible.
    fn vecf(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
    }

    #[test]
    fn incremental_decode_matches_full_recompute() {
        // THE invariant: the cache is an optimization, not a behaviour change.
        let (h, d, steps) = (3usize, 8usize, 12usize);
        let stride = h * d;
        let mut cache = KvCache::new(h, d, 64);
        let mut ks = Vec::new();
        let mut vs = Vec::new();

        for t in 0..steps {
            let k = vecf(stride, t as f32 * 1.1);
            let v = vecf(stride, t as f32 * 2.3 + 0.5);
            let q = vecf(stride, t as f32 * 0.7 + 9.0);
            cache.append(&k, &v).unwrap();
            ks.push(k); vs.push(v);

            let got  = attend_step(&q, &cache);
            let want = attend_full(&q, &ks, &vs, h, d);
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-5,
                    "step {t}, elem {i}: cached {a} != full {b}");
            }
        }
        assert_eq!(cache.len(), steps);
    }

    #[test]
    fn gqa_with_one_kv_head_per_query_head_is_plain_attention() {
        // Group size 1 must reduce EXACTLY to multi-head attention, so
        // one code path can serve both without a behaviour difference.
        let (h, d) = (4usize, 8usize);
        let mut cache = KvCache::new(h, d, 8);
        for t in 0..4 {
            cache.append(&vecf(h * d, t as f32), &vecf(h * d, t as f32 + 0.5)).unwrap();
        }
        let q = vecf(h * d, 9.0);
        assert_eq!(attend_step(&q, &cache), attend_step_grouped(&q, &cache, h));
    }

    #[test]
    fn gqa_shares_kv_heads_in_contiguous_groups() {
        // 4 query heads over 2 K/V heads: queries 0,1 read K/V head 0 and
        // queries 2,3 read K/V head 1. Verified by making the two K/V
        // heads carry distinguishable values — if the grouping were wrong
        // (e.g. interleaved, or head-index reuse) the outputs would swap.
        let (n_q, n_kv, d) = (4usize, 2usize, 4usize);
        let mut cache = KvCache::new(n_kv, d, 4);
        // K/V head 0 → value all 1.0 ; K/V head 1 → value all 5.0
        let k = vecf(n_kv * d, 1.0);
        let mut v = vec![1.0f32; n_kv * d];
        for x in v[d..].iter_mut() { *x = 5.0; }
        cache.append(&k, &v).unwrap();

        let out = attend_step_grouped(&vecf(n_q * d, 2.0), &cache, n_q);
        // Single cached position ⇒ output is exactly the value vector of
        // whichever K/V head that query head reads.
        for head in 0..n_q {
            let want = if head < 2 { 1.0 } else { 5.0 };
            for i in 0..d {
                assert!((out[head * d + i] - want).abs() < 1e-6,
                    "query head {head} must read K/V head {}", head / 2);
            }
        }
    }

    #[test]
    fn attention_weights_form_a_distribution() {
        // A single cached position must return exactly that value vector
        // (softmax over one element is 1.0) — catches scale/normalization
        // errors that averaging would hide.
        let (h, d) = (2usize, 4usize);
        let mut cache = KvCache::new(h, d, 8);
        let v = vecf(h * d, 3.0);
        cache.append(&vecf(h * d, 1.0), &v).unwrap();
        let out = attend_step(&vecf(h * d, 5.0), &cache);
        for (a, b) in out.iter().zip(&v) {
            assert!((a - b).abs() < 1e-6, "single-position attention must return v");
        }
    }

    #[test]
    fn cache_reports_full_context_instead_of_growing() {
        let mut cache = KvCache::new(1, 2, 2);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
        let err = cache.append(&[9.0, 9.0], &[9.0, 9.0]).unwrap_err();
        assert!(err.contains("context full"));
        // Wrong-sized step is rejected too.
        let mut c2 = KvCache::new(1, 2, 2);
        assert!(c2.append(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn clear_reuses_the_allocation() {
        let mut cache = KvCache::new(1, 2, 4);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.clear();
        assert!(cache.is_empty());
        cache.append(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
        assert_eq!(cache.key(0, 0), &[5.0, 6.0]);
    }

    #[test]
    fn greedy_is_deterministic_and_zero_temperature_matches_argmax() {
        let logits = vec![0.1, 3.5, -2.0, 3.4];
        assert_eq!(argmax(&logits), 1);
        let mut s = Sampler::new(42, SamplerConfig { temperature: 0.0, ..Default::default() });
        for _ in 0..5 { assert_eq!(s.sample(&logits), 1); }
    }

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = SamplerConfig { temperature: 1.0, ..Default::default() };
        let mut a = Sampler::new(7, cfg);
        let mut b = Sampler::new(7, cfg);
        let sa: Vec<usize> = (0..20).map(|_| a.sample(&logits)).collect();
        let sb: Vec<usize> = (0..20).map(|_| b.sample(&logits)).collect();
        assert_eq!(sa, sb, "same seed must replay identically");
    }

    #[test]
    fn top_k_and_top_p_restrict_the_candidate_set() {
        // One token dominates; top_k=1 and a tight top_p must both pin to it.
        let logits = vec![0.0, 10.0, 0.1, 0.2];
        let mut k1 = Sampler::new(3, SamplerConfig { temperature: 1.0, top_k: 1, ..Default::default() });
        let mut p1 = Sampler::new(3, SamplerConfig { temperature: 1.0, top_p: 0.5, ..Default::default() });
        for _ in 0..10 {
            assert_eq!(k1.sample(&logits), 1);
            assert_eq!(p1.sample(&logits), 1);
        }
    }

    #[test]
    fn sampler_never_returns_out_of_range() {
        // Degenerate inputs must still yield a valid token index.
        let mut s = Sampler::new(1, SamplerConfig { temperature: 0.5, top_p: 0.0, top_k: 0 });
        let logits = vec![-1.0, -1.0, -1.0];
        for _ in 0..20 { assert!(s.sample(&logits) < logits.len()); }
        assert_eq!(s.sample(&[]), 0);
    }
}
