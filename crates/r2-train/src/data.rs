//! Batches from a token stream.
//!
//! A batch is a function of the step index alone: no iterator state, nothing
//! to save. So a resumed run continues on exactly the data it would have
//! seen, and a corpus of any size costs O(batch x seq) resident memory —
//! shuffling included, because the order is computed, not stored.

use r2_autograd::IGNORE;

use crate::tokens::TokenStore;

/// `(input, target)` token sequences, the shape the trainers take.
pub type Batch = Vec<(Vec<usize>, Vec<usize>)>;

/// Where windows of tokens come from: a memory map or a slice in RAM.
pub trait TokenSource {
    fn n_tokens(&self) -> usize;
    /// Ids `[start, start + n)`, or `None` past the end.
    fn ids(&self, start: usize, n: usize) -> Option<Vec<usize>>;
}

impl TokenSource for TokenStore {
    fn n_tokens(&self) -> usize { self.len() }
    fn ids(&self, start: usize, n: usize) -> Option<Vec<usize>> { self.window(start, n) }
}

impl TokenSource for [u32] {
    fn n_tokens(&self) -> usize { self.len() }
    fn ids(&self, start: usize, n: usize) -> Option<Vec<usize>> {
        Some(self.get(start..start.checked_add(n)?)?.iter().map(|&i| i as usize).collect())
    }
}

/// The order windows are visited in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Consecutive, non-overlapping windows, wrapping at the end. The
    /// PyTorch half of the benchmark reproduces this in two lines.
    Sequential,
    /// A new permutation of the windows every epoch, from this seed.
    Shuffled(u64),
}

/// `batch` windows of `seq` tokens per step.
#[derive(Debug, Clone, Copy)]
pub struct Batcher {
    pub batch: usize,
    pub seq: usize,
    pub order: Order,
}

impl Batcher {
    /// The batch for optimizer step `step`. The source must hold at least
    /// `seq + 2` tokens.
    pub fn at<S: TokenSource + ?Sized>(&self, src: &S, step: usize) -> Batch {
        let span = src.n_tokens() - self.seq - 1;
        (0..self.batch).map(|b| {
            let o = self.offset(step * self.batch + b, span);
            let mut w = src.ids(o, self.seq + 1).expect("window inside the stream");
            let target = w[1..].to_vec();
            w.truncate(self.seq);
            (w, target)
        }).collect()
    }

    /// Start of the `pos`-th window visited.
    fn offset(&self, pos: usize, span: usize) -> usize {
        match self.order {
            Order::Sequential => (pos * self.seq) % span,
            Order::Shuffled(seed) => {
                let windows = (span / self.seq).max(1);
                let key = mix(seed ^ mix((pos / windows) as u64));
                // the shift moves window boundaries between epochs, so a
                // token pair split by a boundary once is not split always
                let shift = mix(key) as usize % self.seq;
                (permute(pos % windows, windows, key) * self.seq + shift) % span
            }
        }
    }
}

/// Prompt-and-completion examples: the model is shown the whole text but is
/// only marked on the completion. Each example's targets are [`IGNORE`] over
/// the prompt, so the loss teaches what follows a prompt, not the prompts.
///
/// A batch is padded on the right to its own longest example. Attention is
/// causal, so a padding token cannot affect any position before it, and its
/// own targets are [`IGNORE`]: padding changes neither loss nor gradient (a
/// test holds this to the bit), and no attention mask is needed.
pub struct PairSet {
    /// Whole examples, shortest first, so neighbours make tight batches.
    examples: Vec<(Vec<usize>, usize)>,
    pad: usize,
}

impl PairSet {
    /// `pairs` are `(prompt, completion)` ids. An example longer than
    /// `max_len` loses the end of its completion; one with no completion
    /// left to learn is an error, because the model would be silently taught
    /// nothing from it.
    pub fn new(pairs: &[(Vec<u32>, Vec<u32>)], max_len: usize, pad: u32) -> Result<PairSet, String> {
        let mut examples = Vec::with_capacity(pairs.len());
        for (i, (p, c)) in pairs.iter().enumerate() {
            if p.is_empty() { return Err(format!("example {}: empty prompt", i + 1)); }
            if c.is_empty() { return Err(format!("example {}: empty completion", i + 1)); }
            if p.len() >= max_len {
                return Err(format!("example {}: a prompt of {} tokens leaves no room in seq = {}; \
                                    raise seq or shorten it", i + 1, p.len(), max_len - 1));
            }
            let ids: Vec<usize> = p.iter().chain(c).take(max_len).map(|&t| t as usize).collect();
            examples.push((ids, p.len()));
        }
        if examples.is_empty() { return Err("no examples".into()); }
        examples.sort_by_key(|(ids, _)| ids.len());
        Ok(PairSet { examples, pad: pad as usize })
    }

    pub fn len(&self) -> usize { self.examples.len() }
    pub fn is_empty(&self) -> bool { self.examples.is_empty() }

    /// The batch for `step`. With `Order::Shuffled`, batches of similar
    /// lengths are visited in a new order each epoch: little padding, so
    /// little wasted work, and still a different order every time. With
    /// `Order::Sequential` they come shortest first.
    pub fn at(&self, batch: usize, step: usize, order: Order) -> Batch {
        let batches = self.examples.len().div_ceil(batch.max(1));
        let g = step % batches;
        let g = match order {
            Order::Sequential => g,
            Order::Shuffled(seed) => permute(g, batches, mix(seed ^ mix((step / batches) as u64))),
        };
        let group = &self.examples[g * batch..((g + 1) * batch).min(self.examples.len())];
        let longest = group.iter().map(|(ids, _)| ids.len()).max().unwrap_or(1);
        group.iter().map(|(ids, prompt)| {
            let n = longest - 1;
            let mut input = ids[..ids.len() - 1].to_vec();
            let mut target: Vec<usize> = ids[1..].to_vec();
            for t in target.iter_mut().take(prompt - 1) { *t = IGNORE; }
            input.resize(n, self.pad);
            target.resize(n, IGNORE);
            (input, target)
        }).collect()
    }
}

/// splitmix64's finalizer.
fn mix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A bijection on `0..n`, keyed: a Feistel network over the next even power
/// of two, cycle-walking values that land outside `0..n` (at most ~4 tries
/// on average). Shuffling a billion windows therefore stores nothing, where
/// a permutation table would hold 8 GB in RAM.
fn permute(i: usize, n: usize, key: u64) -> usize {
    if n <= 1 { return 0; }
    let half = (usize::BITS - (n - 1).leading_zeros()).div_ceil(2);
    let mask = (1u64 << half) - 1;
    let mut x = i as u64;
    loop {
        let (mut l, mut r) = (x >> half, x & mask);
        for round in 0..4u64 {
            (l, r) = (r, l ^ (mix(r ^ key.rotate_left(round as u32 * 13) ^ round) & mask));
        }
        x = (l << half) | r;
        if (x as usize) < n { return x as usize; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(n: u32) -> Vec<u32> { (0..n).collect() }

    #[test]
    fn sequential_walks_consecutive_windows_and_wraps() {
        let s = stream(100);
        let b = Batcher { batch: 2, seq: 10, order: Order::Sequential };
        let first = b.at(&s[..], 0);
        assert_eq!(first[0].0, (0..10).collect::<Vec<usize>>());
        assert_eq!(first[0].1, (1..11).collect::<Vec<usize>>());
        assert_eq!(first[1].0[0], 10);
        assert_eq!(b.at(&s[..], 1)[0].0[0], 20, "step 1 starts where step 0 stopped");
        assert_eq!(b.at(&s[..], 5)[0].0[0], 100 % 89, "wraps modulo the span");
    }

    #[test]
    fn permute_is_a_bijection_at_every_size() {
        for n in [1usize, 2, 3, 7, 8, 9, 100, 1000, 4097] {
            let mut seen = vec![false; n];
            for i in 0..n {
                let p = permute(i, n, 0xABCD);
                assert!(p < n && !seen[p], "n={n}: {i} -> {p}");
                seen[p] = true;
            }
        }
    }

    #[test]
    fn shuffled_visits_every_window_once_per_epoch_and_reorders_between_epochs() {
        let s = stream(1000);
        let b = Batcher { batch: 1, seq: 10, order: Order::Shuffled(7) };
        let windows = (1000 - 10 - 1) / 10;
        let epoch = |e: usize| -> Vec<usize> { (0..windows).map(|i| b.at(&s[..], e * windows + i)[0].0[0]).collect() };
        let (e0, e1) = (epoch(0), epoch(1));
        let mut sorted = e0.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), windows, "an epoch repeats no window");
        assert_ne!(e0, e1, "each epoch has its own order");
        assert_eq!(e0, epoch(0), "and that order is a function of the step alone");
        let other = Batcher { order: Order::Shuffled(8), ..b };
        let e8: Vec<usize> = (0..windows).map(|i| other.at(&s[..], i)[0].0[0]).collect();
        assert_ne!(e0, e8, "and the seed picks it");
    }

    fn pairs() -> Vec<(Vec<u32>, Vec<u32>)> {
        vec![(vec![1, 2], vec![3, 4, 5]), (vec![6], vec![7]), (vec![8, 9, 10], vec![11]),
             (vec![12], vec![13, 14, 15, 16])]
    }

    #[test]
    fn only_the_completion_is_marked_and_padding_is_ignored() {
        let s = PairSet::new(&pairs(), 16, 0).unwrap();
        // shortest first: [6,7] then the three-to-five token examples
        let b = s.at(2, 0, Order::Sequential);
        assert_eq!(b[0].0, vec![6, 0, 0], "padded on the right to the batch's longest");
        assert_eq!(b[0].1, vec![7, IGNORE, IGNORE], "predicting 7 from 6 is the only mark");
        assert_eq!(b[1].0, vec![8, 9, 10]);
        assert_eq!(b[1].1, vec![IGNORE, IGNORE, 11], "the prompt's own tokens are not marked");
    }

    #[test]
    fn every_example_is_seen_each_epoch_in_a_new_order() {
        let s = PairSet::new(&pairs(), 16, 0).unwrap();
        for order in [Order::Sequential, Order::Shuffled(3)] {
            let mut firsts: Vec<usize> = (0..2).flat_map(|st| s.at(2, st, order)).map(|(i, _)| i[0]).collect();
            firsts.sort();
            assert_eq!(firsts, vec![1, 6, 8, 12], "{order:?}");
        }
        let orders: std::collections::HashSet<Vec<usize>> = (0..10)
            .map(|e| s.at(2, e * 2, Order::Shuffled(3)).iter().map(|(i, _)| i[0]).collect()).collect();
        assert!(orders.len() > 1, "shuffling must reorder batches between epochs");
    }

    #[test]
    fn long_examples_lose_their_end_and_useless_ones_are_refused() {
        let s = PairSet::new(&pairs(), 4, 0).unwrap();
        for st in 0..2 {
            assert!(s.at(4, st, Order::Sequential).iter().all(|(i, _)| i.len() <= 3));
        }
        assert!(PairSet::new(&[(vec![1, 2, 3, 4], vec![5])], 4, 0).err().unwrap().contains("seq"));
        assert!(PairSet::new(&[(vec![], vec![5])], 4, 0).err().unwrap().contains("empty prompt"));
        assert!(PairSet::new(&[(vec![5], vec![])], 4, 0).err().unwrap().contains("empty completion"));
        assert!(PairSet::new(&[], 4, 0).is_err());
    }

    #[test]
    fn a_batch_stays_inside_the_stream_at_the_smallest_size() {
        let s = stream(6);
        for order in [Order::Sequential, Order::Shuffled(1)] {
            let b = Batcher { batch: 3, seq: 4, order };
            for step in 0..20 {
                for (i, t) in b.at(&s[..], step) {
                    assert_eq!((i.len(), t.len()), (4, 4));
                    assert_eq!(t[0], i[0] + 1);
                }
            }
        }
    }
}
