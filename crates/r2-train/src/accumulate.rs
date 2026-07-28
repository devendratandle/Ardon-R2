//! Gradient accumulation — train a big batch in small sequential parts.
//!
//! This is the exact answer to "can I train in two or three parts on one
//! machine and join them?". Yes — because the gradient of a sum is the sum
//! of gradients, so processing micro-batches **sequentially** and applying
//! **one** optimizer step at the end is mathematically identical to
//! processing the whole batch at once. Memory falls to one micro-batch;
//! the update is unchanged.
//!
//! It is also the honest way to rehearse a cluster on a single machine:
//! the same reconciliation that `distributed::sync_grads` performs across
//! ranks in parallel, performed here across time. Deterministic, no
//! races — so a distributed bug shows up reproducibly before any cluster
//! is involved.
//!
//! **Unequal parts are weighted by sample count.** This is the subtle bug
//! the accumulator exists to prevent: if the last micro-batch is smaller
//! (the usual case — datasets rarely divide evenly) then averaging the
//! per-micro-batch means gives every *batch* equal say instead of every
//! *sample*, and the result silently differs from full-batch training.
//! Weighting by sample count keeps the equivalence exact.

/// Accumulates per-sample-mean gradients across micro-batches.
#[derive(Debug, Clone)]
pub struct GradAccumulator {
    /// Running Σ (mean_grad_i × n_samples_i), one entry per parameter tensor.
    sums: Vec<Vec<f32>>,
    /// Total samples seen — the denominator that makes unequal parts exact.
    total: usize,
    /// Micro-batches accumulated.
    parts: usize,
}

impl GradAccumulator {
    /// Allocate for parameter tensors of the given lengths.
    pub fn new(shapes: &[usize]) -> Self {
        GradAccumulator {
            sums: shapes.iter().map(|&n| vec![0.0f32; n]).collect(),
            total: 0,
            parts: 0,
        }
    }

    /// Samples accumulated so far.
    #[inline] pub fn samples(&self) -> usize { self.total }
    /// Micro-batches accumulated so far.
    #[inline] pub fn parts(&self) -> usize { self.parts }
    #[inline] pub fn is_empty(&self) -> bool { self.total == 0 }

    /// Add one micro-batch's **mean** gradient, weighted by how many
    /// samples produced it. Errors on a shape mismatch rather than
    /// accumulating a prefix — a silently wrong gradient is the worst
    /// possible failure mode, since training still "works".
    pub fn add(&mut self, grads: &[Vec<f32>], n_samples: usize) -> Result<(), String> {
        if n_samples == 0 {
            return Err("GradAccumulator::add: micro-batch has no samples".into());
        }
        if grads.len() != self.sums.len() {
            return Err(format!(
                "GradAccumulator::add: expected {} tensors, got {}",
                self.sums.len(), grads.len()));
        }
        for (i, (acc, g)) in self.sums.iter_mut().zip(grads).enumerate() {
            if acc.len() != g.len() {
                return Err(format!(
                    "GradAccumulator::add: tensor {} expected {} values, got {}",
                    i, acc.len(), g.len()));
            }
        }
        let w = n_samples as f32;
        for (acc, g) in self.sums.iter_mut().zip(grads) {
            for (a, &x) in acc.iter_mut().zip(g) { *a += x * w; }
        }
        self.total += n_samples;
        self.parts += 1;
        Ok(())
    }

    /// The accumulated gradient, per sample — identical to what a single
    /// pass over the whole batch would have produced.
    pub fn mean(&self) -> Result<Vec<Vec<f32>>, String> {
        if self.total == 0 {
            return Err("GradAccumulator::mean: nothing accumulated".into());
        }
        let inv = 1.0 / self.total as f32;
        Ok(self.sums.iter()
            .map(|s| s.iter().map(|&x| x * inv).collect())
            .collect())
    }

    /// Clear for the next batch, keeping the allocation.
    pub fn reset(&mut self) {
        for s in self.sums.iter_mut() { for x in s.iter_mut() { *x = 0.0; } }
        self.total = 0;
        self.parts = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Adam;
    use r2_autograd::Tape;

    /// Mean gradient of a linear model y = x·w under MSE, over `samples`,
    /// via the autograd tape — the same path training uses, so the test
    /// exercises real gradients rather than a hand-derived formula.
    fn mean_grad(w: &[f32], samples: &[(Vec<f32>, f32)], d: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; w.len()];
        for (x, y) in samples {
            let mut t = Tape::new();
            let wv = t.leaf(w.to_vec(), true);
            let xv = t.leaf(x.clone(), false);
            let pred = t.matmul(xv, wv, 1, d, 1);
            let loss = t.mse(pred, vec![*y]);
            t.backward(loss);
            for (a, g) in acc.iter_mut().zip(t.grad(wv)) { *a += g; }
        }
        let n = samples.len() as f32;
        for a in acc.iter_mut() { *a /= n; }
        acc
    }

    fn data() -> Vec<(Vec<f32>, f32)> {
        vec![
            (vec![1.0, 2.0, -1.0], 3.0),
            (vec![0.5, -1.0, 2.0], -1.0),
            (vec![2.0, 1.0, 0.0], 4.0),
            (vec![-1.0, 0.5, 1.5], 0.5),
            (vec![0.0, 3.0, -2.0], 2.0),
            (vec![1.5, -0.5, 1.0], -0.5),
        ]
    }

    #[test]
    fn three_equal_parts_equal_one_big_batch() {
        // THE INVARIANT: sequential micro-batches, one update — identical
        // to the full batch. This is what makes "train in parts" exact.
        let (d, w) = (3usize, vec![0.1f32, -0.2, 0.3]);
        let all = data();
        let full = mean_grad(&w, &all, d);

        let mut acc = GradAccumulator::new(&[d]);
        for chunk in all.chunks(2) {
            acc.add(&[mean_grad(&w, chunk, d)], chunk.len()).unwrap();
        }
        assert_eq!(acc.parts(), 3);
        assert_eq!(acc.samples(), 6);

        let got = acc.mean().unwrap();
        for (a, b) in got[0].iter().zip(&full) {
            assert!((a - b).abs() < 1e-6, "accumulated {a} != full-batch {b}");
        }
    }

    #[test]
    fn unequal_parts_are_still_exact_when_weighted() {
        // Datasets rarely divide evenly. Splitting 6 as 1+2+3 must still
        // match the full batch — that only holds because add() weights by
        // sample count.
        let (d, w) = (3usize, vec![0.1f32, -0.2, 0.3]);
        let all = data();
        let full = mean_grad(&w, &all, d);

        let mut acc = GradAccumulator::new(&[d]);
        for part in [&all[0..1], &all[1..3], &all[3..6]] {
            acc.add(&[mean_grad(&w, part, d)], part.len()).unwrap();
        }
        for (a, b) in acc.mean().unwrap()[0].iter().zip(&full) {
            assert!((a - b).abs() < 1e-6, "uneven split must stay exact: {a} != {b}");
        }
    }

    #[test]
    fn unweighted_averaging_of_uneven_parts_is_wrong() {
        // The counter-example that justifies the weighting: averaging the
        // per-micro-batch means gives every BATCH equal say instead of
        // every SAMPLE, and silently differs from full-batch training.
        let (d, w) = (3usize, vec![0.1f32, -0.2, 0.3]);
        let all = data();
        let full = mean_grad(&w, &all, d);
        let parts = [&all[0..1], &all[1..3], &all[3..6]];

        let mut naive = vec![0.0f32; d];
        for part in parts {
            for (a, g) in naive.iter_mut().zip(mean_grad(&w, part, d)) { *a += g / 3.0; }
        }
        let differs = naive.iter().zip(&full).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "unweighted averaging should NOT match full-batch");
    }

    #[test]
    fn accumulated_training_step_matches_full_batch_training() {
        // End to end: an Adam update from accumulated gradients lands on
        // the same parameters as one from a full-batch gradient.
        let (d, all) = (3usize, data());

        let mut w_full = vec![0.1f32, -0.2, 0.3];
        let mut opt_full = Adam::new(d, 0.05);
        for _ in 0..5 {
            let g = mean_grad(&w_full, &all, d);
            opt_full.step(&mut w_full, &g).unwrap();
        }

        let mut w_acc = vec![0.1f32, -0.2, 0.3];
        let mut opt_acc = Adam::new(d, 0.05);
        let mut acc = GradAccumulator::new(&[d]);
        for _ in 0..5 {
            acc.reset();
            for chunk in all.chunks(2) {
                acc.add(&[mean_grad(&w_acc, chunk, d)], chunk.len()).unwrap();
            }
            let g = acc.mean().unwrap();
            opt_acc.step(&mut w_acc, &g[0]).unwrap();
        }

        for (a, b) in w_acc.iter().zip(&w_full) {
            assert!((a - b).abs() < 1e-5,
                "accumulated training must match full-batch training: {a} != {b}");
        }
    }

    #[test]
    fn multi_tensor_models_accumulate_independently() {
        let mut acc = GradAccumulator::new(&[2, 3]);
        acc.add(&[vec![1.0, 2.0], vec![1.0, 1.0, 1.0]], 1).unwrap();
        acc.add(&[vec![3.0, 4.0], vec![3.0, 3.0, 3.0]], 3).unwrap();
        // Weighted mean: (1*1 + 3*3)/4 = 2.5 ; (2*1 + 4*3)/4 = 3.5
        let m = acc.mean().unwrap();
        assert!((m[0][0] - 2.5).abs() < 1e-6 && (m[0][1] - 3.5).abs() < 1e-6);
        assert!(m[1].iter().all(|&x| (x - 2.5).abs() < 1e-6));
    }

    #[test]
    fn reset_clears_state_but_keeps_capacity() {
        let mut acc = GradAccumulator::new(&[2]);
        acc.add(&[vec![5.0, 5.0]], 2).unwrap();
        acc.reset();
        assert!(acc.is_empty() && acc.parts() == 0);
        assert!(acc.mean().is_err(), "empty accumulator must not fabricate a gradient");
        acc.add(&[vec![1.0, 1.0]], 1).unwrap();
        assert_eq!(acc.mean().unwrap()[0], vec![1.0, 1.0]);
    }

    #[test]
    fn shape_and_empty_errors_are_loud() {
        let mut acc = GradAccumulator::new(&[2, 3]);
        assert!(acc.add(&[vec![1.0, 2.0]], 1).unwrap_err().contains("expected 2 tensors"));
        assert!(acc.add(&[vec![1.0], vec![1.0, 1.0, 1.0]], 1)
            .unwrap_err().contains("tensor 0 expected 2 values"));
        assert!(acc.add(&[vec![1.0, 2.0], vec![1.0, 1.0, 1.0]], 0)
            .unwrap_err().contains("no samples"));
        assert!(acc.is_empty(), "no rejected call may leave partial state");
    }
}
