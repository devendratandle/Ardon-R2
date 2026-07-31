//! Segment-recurrent attention — long context on a CPU budget.
//!
//! # Why this exists
//!
//! Standard attention lets every token see every earlier token directly.
//! That costs `n²` score computations and an `n`-long KV cache. On a GPU,
//! with ~10⁵ GFLOP/s, quadratic cost is affordable to tens of thousands of
//! tokens. On a CPU, with ~10 GFLOP/s, it is not: 160k tokens is
//! 1.3×10¹⁰ score-pairs per layer per head, which is hours, not seconds.
//!
//! So the CPU-favourable restructuring is not a faster kernel — it is
//! **changing the cost class**. Split the stream into fixed segments; each
//! segment attends to its own tokens plus a small carried **memory** that
//! summarizes everything before it. Cost becomes
//!
//! ```text
//!   segments × segment_len × (segment_len + memory)  =  n · (C + M)
//! ```
//!
//! which is LINEAR in `n`. At n = 160k, C = 256, M = 64 that is ~250×
//! fewer score computations than full attention, and the cache holds
//! `C + M` positions instead of `n` — bounded memory for unbounded input.
//!
//! # What it costs (stated honestly)
//!
//! Recurrence is not free. Information from an old segment reaches the
//! present only through the carried memory, so precise recall of a
//! specific distant token is lossy in a way full attention is not. This
//! is the same trade-off Transformer-XL and Recurrent Memory Transformer
//! make, and it is a real one: on tasks that need exact long-range lookup,
//! full attention wins at equal parameter count.
//!
//! The honest framing is therefore: this buys **unbounded context on
//! hardware that cannot afford quadratic attention**, at some loss of
//! long-range precision — not a free improvement.

use crate::infer::KvCache;

/// Cost of attention over `n` tokens, in score computations per head per
/// layer. Exposed because the whole argument for this module is a cost
/// comparison, and a reader should be able to check it rather than trust it.
pub fn full_attention_pairs(n: usize) -> u64 {
    // Causal: position i attends to i+1 positions → n(n+1)/2.
    let n = n as u64;
    n * (n + 1) / 2
}

/// Same measure for segment-recurrent attention.
pub fn segment_attention_pairs(n: usize, segment: usize, memory: usize) -> u64 {
    if segment == 0 { return 0; }
    let segs = (n + segment - 1) / segment;
    let per_seg = (segment as u64) * (segment as u64 + 1) / 2   // causal within
        + (segment as u64) * memory as u64;                     // plus memory
    segs as u64 * per_seg
}

/// A bounded cache: the current segment's keys/values plus a carried
/// memory of the past.
///
/// Memory slots sit at positions `0..memory` and the live segment grows
/// after them, so a single contiguous cache serves both and the existing
/// attention kernel works unchanged.
#[derive(Debug)]
pub struct SegmentCache {
    cache: KvCache,
    /// Number of memory slots reserved at the front.
    pub memory: usize,
    /// Tokens per segment.
    pub segment: usize,
    /// Memory slots actually filled (0 until the first segment rolls).
    filled_memory: usize,
    /// Tokens in the live segment.
    live: usize,
    /// Total tokens ever seen — the logical position, which keeps growing
    /// even though storage does not.
    pub seen: usize,
}

impl SegmentCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, segment: usize, memory: usize) -> Self {
        SegmentCache {
            cache: KvCache::new(n_kv_heads, head_dim, memory + segment),
            memory, segment, filled_memory: 0, live: 0, seen: 0,
        }
    }

    /// Positions currently attendable: filled memory + live segment.
    pub fn len(&self) -> usize { self.cache.len() }
    pub fn is_empty(&self) -> bool { self.cache.is_empty() }
    /// The underlying cache, for the attention kernel.
    pub fn cache(&self) -> &KvCache { &self.cache }
    /// Peak positions this will ever hold — the bounded-memory guarantee.
    pub fn capacity(&self) -> usize { self.memory + self.segment }

    /// Append one token. When the live segment fills, it is summarized
    /// into memory and cleared — the recurrence step.
    ///
    /// Summarization here is a MEAN over the segment's keys and values per
    /// head. That is the simplest faithful summary and it is deliberately
    /// not learned: a learned compressor is strictly better, but it must be
    /// trained, and this module's claim is about COST CLASS, which a mean
    /// demonstrates without confounding the measurement.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), String> {
        if self.live >= self.segment {
            self.roll()?;
        }
        self.cache.append(k, v)?;
        self.live += 1;
        self.seen += 1;
        Ok(())
    }

    /// Fold the live segment into memory and start a fresh one.
    fn roll(&mut self) -> Result<(), String> {
        let (h, d) = (self.cache.n_heads, self.cache.head_dim);
        let stride = h * d;

        // Mean-pool the live segment per head.
        let start = self.filled_memory;
        let mut ks = vec![0.0f32; stride];
        let mut vs = vec![0.0f32; stride];
        for p in start..self.cache.len() {
            for head in 0..h {
                let ko = self.cache.key(p, head);
                let vo = self.cache.value(p, head);
                for j in 0..d {
                    ks[head * d + j] += ko[j];
                    vs[head * d + j] += vo[j];
                }
            }
        }
        let n = (self.cache.len() - start).max(1) as f32;
        for x in ks.iter_mut() { *x /= n; }
        for x in vs.iter_mut() { *x /= n; }

        // Rebuild: existing memory (evicting the oldest if full) + summary.
        let mut mem_k: Vec<Vec<f32>> = Vec::new();
        let mut mem_v: Vec<Vec<f32>> = Vec::new();
        for p in 0..self.filled_memory {
            let mut kk = vec![0.0f32; stride];
            let mut vv = vec![0.0f32; stride];
            for head in 0..h {
                kk[head * d..(head + 1) * d].copy_from_slice(self.cache.key(p, head));
                vv[head * d..(head + 1) * d].copy_from_slice(self.cache.value(p, head));
            }
            mem_k.push(kk); mem_v.push(vv);
        }
        mem_k.push(ks); mem_v.push(vs);
        // Oldest memory is dropped first — a fixed-size window over history.
        while mem_k.len() > self.memory { mem_k.remove(0); mem_v.remove(0); }

        self.cache.clear();
        for (kk, vv) in mem_k.iter().zip(&mem_v) { self.cache.append(kk, vv)?; }
        self.filled_memory = mem_k.len();
        self.live = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::attend_step_grouped;

    fn vecf(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin())).collect()
    }

    #[test]
    fn cost_class_is_linear_not_quadratic() {
        // THE ARGUMENT, checked as arithmetic: full attention grows with
        // n²; segment-recurrent grows with n. This is the entire reason
        // the architecture is worth considering on a CPU.
        let (seg, mem) = (256usize, 64usize);
        let a = segment_attention_pairs(16_000, seg, mem);
        let b = segment_attention_pairs(160_000, seg, mem);
        // 10x the tokens ⇒ ~10x the work (linear), not 100x.
        let ratio = b as f64 / a as f64;
        assert!((ratio - 10.0).abs() < 0.1, "segment cost should be linear, got {ratio}x");

        // And it must be dramatically cheaper than full attention at length.
        let full = full_attention_pairs(160_000);
        assert!(full / b > 100, "expected >100x saving, got {}x", full / b);
    }

    #[test]
    fn memory_is_bounded_however_long_the_stream() {
        // The other half of the claim: storage stops growing. A cache that
        // grows with input cannot serve an unbounded stream on any machine.
        let (h, d, seg, mem) = (2usize, 4usize, 8usize, 4usize);
        let mut c = SegmentCache::new(h, d, seg, mem);
        for t in 0..500 {
            c.append(&vecf(h * d, t as f32), &vecf(h * d, t as f32 + 0.5)).unwrap();
            assert!(c.len() <= c.capacity(),
                "cache exceeded its bound at token {t}: {} > {}", c.len(), c.capacity());
        }
        assert_eq!(c.seen, 500, "logical position keeps counting");
        assert!(c.len() <= mem + seg);
    }

    #[test]
    fn attention_still_works_against_the_bounded_cache() {
        // The existing kernel must work unchanged — memory slots and live
        // tokens are just positions.
        let (h, d, seg, mem) = (2usize, 4usize, 8usize, 4usize);
        let mut c = SegmentCache::new(h, d, seg, mem);
        for t in 0..40 {
            c.append(&vecf(h * d, t as f32), &vecf(h * d, t as f32 + 0.5)).unwrap();
        }
        let out = attend_step_grouped(&vecf(h * d, 9.0), c.cache(), h);
        assert_eq!(out.len(), h * d);
        assert!(out.iter().all(|x| x.is_finite()), "attention output must be finite");
    }

    #[test]
    fn short_streams_behave_exactly_like_full_attention() {
        // Below one segment there is no recurrence, so this must be
        // identical to the ordinary cache — no behaviour change for the
        // common short case.
        let (h, d) = (2usize, 4usize);
        let mut seg_c = SegmentCache::new(h, d, 16, 4);
        let mut plain = KvCache::new(h, d, 16);
        for t in 0..10 {
            let (k, v) = (vecf(h * d, t as f32), vecf(h * d, t as f32 + 0.5));
            seg_c.append(&k, &v).unwrap();
            plain.append(&k, &v).unwrap();
        }
        let q = vecf(h * d, 3.0);
        assert_eq!(attend_step_grouped(&q, seg_c.cache(), h),
                   attend_step_grouped(&q, &plain, h));
    }
}
