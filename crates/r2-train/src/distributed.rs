//! Shard-aware gradient synchronization — the seam that turns "trained on
//! one machine" into "trains across the mesh." This is what closes the
//! r2-autograd ↔ r2-mesh gap (docs/LLM_TRILLION_ARCHITECTURE.md, the
//! Layer-3/Layer-4 integration point).
//!
//! After each worker computes LOCAL gradients (its own micro-batch through
//! the autograd tape), the gradients must be reconciled across the mesh
//! ACCORDING TO each parameter's shard spec, so every worker applies a
//! consistent update:
//!
//!   Replicated (data-parallel)   → all_reduce the gradient (mean), so
//!                                   every replica sees the global gradient.
//!   TensorParallel (Megatron)    → the WEIGHT gradient is already local to
//!                                   the shard (each rank owns its slice);
//!                                   no reduction on the weight grad. (The
//!                                   ACTIVATION grad crossing a column/row
//!                                   split is all_reduced inside the matmul
//!                                   backward — a kernel concern, not here.)
//!   ZeroSharded (ZeRO)           → gradients are reduce_scattered so each
//!                                   rank ends holding only ITS shard's grad
//!                                   (the optimizer then updates that shard).
//!   PipelineStage                → grads are local to the stage.
//!
//! THE CORRECTNESS INVARIANT (tested): data-parallel training across G
//! workers, each on an equal slice of the batch with all_reduced-mean
//! gradients, produces the SAME parameters as single-device training on
//! the full batch. If that holds, the distributed path is provably
//! equivalent to the reference — the property that makes scaling safe.

use r2_mesh::{Collective, ReduceOp, Shard};

/// Reconcile one worker's gradients across the group per the shard specs.
/// `grads[i]` is the local gradient for parameter `i`; `shards[i]` is its
/// placement. Equal-shard data parallelism (the standard DDP assumption):
/// each worker passes its LOCAL-MEAN gradient; the all_reduce+divide yields
/// the global mean.
pub fn sync_grads(
    coll: &dyn Collective,
    rank: usize,
    grads: &mut [Vec<f32>],
    shards: &[Shard],
) {
    let g = coll.group_size() as f32;
    for (grad, shard) in grads.iter_mut().zip(shards) {
        match shard {
            Shard::Replicated => {
                coll.all_reduce(rank, grad, ReduceOp::Sum);
                for x in grad.iter_mut() { *x /= g; } // mean of the local means
            }
            Shard::ZeroSharded { .. } => {
                // ZeRO: each rank keeps only its shard of the summed grad.
                let shard_len = grad.len() / coll.group_size();
                let mut owned = vec![0.0f32; shard_len];
                coll.reduce_scatter(rank, grad, &mut owned, ReduceOp::Sum);
                for x in owned.iter_mut() { *x /= g; }
                // Write the owned shard back into the rank's slice; other
                // slices are not this rank's responsibility (optimizer +
                // all_gather reassemble the full param — see r2_mesh::zero_*).
                let off = rank * shard_len;
                grad[off..off + shard_len].copy_from_slice(&owned);
            }
            // Weight grads for tensor/pipeline parallelism are already the
            // correct local quantity — nothing to reduce here.
            Shard::TensorParallel { .. } | Shard::PipelineStage { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_autograd::Tape;
    use r2_mesh::ThreadCollective;
    use std::sync::Arc;
    use std::thread;

    /// Tiny model: linear y = x·W (d→1), MSE loss. Per-sample gradient via
    /// the autograd tape. Returns the MEAN gradient over `samples`.
    fn mean_grad(w: &[f32], samples: &[(Vec<f32>, f32)], d: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; w.len()];
        for (x, y) in samples {
            let mut t = Tape::new();
            let wv = t.leaf(w.to_vec(), true);
            let xv = t.leaf(x.clone(), false);
            let pred = t.matmul(xv, wv, 1, d, 1); // (1×d)·(d×1) = 1×1
            let loss = t.mse(pred, vec![*y]);
            t.backward(loss);
            for (a, g) in acc.iter_mut().zip(t.grad(wv)) { *a += g; }
        }
        let n = samples.len() as f32;
        for a in acc.iter_mut() { *a /= n; }
        acc
    }

    #[test]
    fn data_parallel_matches_single_device() {
        let d = 3;
        // Fixed dataset of 4 samples.
        let data = vec![
            (vec![1.0, 2.0, -1.0], 3.0),
            (vec![0.5, -1.0, 2.0], -1.0),
            (vec![2.0, 1.0, 0.0], 4.0),
            (vec![-1.0, 0.5, 1.5], 0.5),
        ];
        let lr = 0.03;
        let steps = 60;

        // ── Reference: single device, full batch ──
        let mut w_ref = vec![0.1f32; d];
        for _ in 0..steps {
            let g = mean_grad(&w_ref, &data, d);
            for (wi, gi) in w_ref.iter_mut().zip(&g) { *wi -= lr * gi; }
        }

        // ── Distributed: 2 workers, 2 samples each, sync_grads all-reduce ──
        let coll = ThreadCollective::new(2);
        let data = Arc::new(data);
        let shards = Arc::new(vec![Shard::Replicated]); // W is data-parallel
        let mut handles = Vec::new();
        for rank in 0..2 {
            let c = coll.clone();
            let data = data.clone();
            let shards = shards.clone();
            handles.push(thread::spawn(move || {
                let local = if rank == 0 { &data[0..2] } else { &data[2..4] };
                let mut w = vec![0.1f32; d];
                for _ in 0..steps {
                    let g = mean_grad(&w, local, d);   // local-MEAN gradient
                    let mut grads = vec![g];
                    sync_grads(c.as_ref(), rank, &mut grads, &shards); // → global mean
                    for (wi, gi) in w.iter_mut().zip(&grads[0]) { *wi -= lr * gi; }
                }
                w
            }));
        }
        let results: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Both workers must agree, AND match the single-device reference.
        for r in &results {
            assert_eq!(r.len(), w_ref.len());
            for (got, want) in r.iter().zip(&w_ref) {
                assert!((got - want).abs() < 1e-4,
                    "distributed {:?} != single-device {:?}", r, w_ref);
            }
        }
        // Sanity: the two workers ended identical (consensus).
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn zero_sharded_sync_reduces_to_owned_shard() {
        // 2 ranks, 4-param grad. Each rank's local grad = [1,2,3,4];
        // summed = [2,4,6,8], mean = [1,2,3,4]; rank r keeps its 2-slice.
        let coll = ThreadCollective::new(2);
        let shards = Arc::new(vec![Shard::ZeroSharded { group: r2_mesh::GroupId(0) }]);
        let mut handles = Vec::new();
        for rank in 0..2 {
            let c = coll.clone();
            let shards = shards.clone();
            handles.push(thread::spawn(move || {
                let mut grads = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
                sync_grads(c.as_ref(), rank, &mut grads, &shards);
                grads.pop().unwrap()
            }));
        }
        let out: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // rank 0 owns slice [0,2): mean [1,2]; rank 1 owns [2,4): mean [3,4].
        assert_eq!(&out[0][0..2], &[1.0, 2.0]);
        assert_eq!(&out[1][2..4], &[3.0, 4.0]);
    }

    #[test]
    fn tensor_parallel_weight_grad_is_untouched() {
        // TensorParallel weight grads are already local — sync is a no-op.
        let coll = ThreadCollective::new(2);
        let shards = vec![Shard::TensorParallel { dim: 0, group: r2_mesh::GroupId(0) }];
        let mut grads = vec![vec![5.0f32, 6.0, 7.0]];
        sync_grads(coll.as_ref(), 0, &mut grads, &shards);
        assert_eq!(grads[0], vec![5.0, 6.0, 7.0]);
    }
}
