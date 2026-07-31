//! Strided reduction kernel — Phase K.6.

use crate::{SerialBackend, RayonBackend};
use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;
use crate::ReduceOp;

// ════════════════════════════════════════════════════════════════════
// Strided reduction kernel — Phase K.6 (Tier 4).
// ════════════════════════════════════════════════════════════════════
//
// Reduce over a strided view of a slice without copying. Index walk is
//   `data[offset], data[offset+stride], ..., data[offset+(count-1)*stride]`.
//
// Motivation: column-major matrices store columns contiguously (stride 1)
// but rows non-contiguously (stride = nrow). Reducing a row required
// a copy-into-`Vec` round-trip; this kernel skips that allocation.
//
// Implementation notes:
//   - Same NA propagation as `reduce`: any `None` in the walked positions
//     → `None`.
//   - Variance/Sd use the same two-pass algorithm but iterate strided
//     indices, not over a copied buffer.
//   - Median still materialises a `Vec<f64>` because select_nth_unstable
//     needs contiguous storage; the win for Median is amortised by
//     avoiding the option-unwrap pass on a temporary.

pub trait StridedReduceBackend: Send + Sync {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64>;
}

impl StridedReduceBackend for SerialBackend {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64> {
        debug_assert!(stride > 0, "stride must be > 0");
        if count == 0 { return None; }
        // Index iterator for the walk.
        let idx = |k: usize| offset + k * stride;
        // Bounds-check the last index up front.
        if idx(count - 1) >= data.len() { return None; }

        match op {
            ReduceOp::Sum => {
                let mut acc = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc += v, None => return None }
                }
                Some(acc)
            }
            ReduceOp::Mean => {
                let mut acc = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc += v, None => return None }
                }
                Some(acc / count as f64)
            }
            ReduceOp::Min => {
                let mut m = f64::INFINITY;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => m = m.min(v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Max => {
                let mut m = f64::NEG_INFINITY;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => m = m.max(v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Prod => {
                let mut acc = 1.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc *= v, None => return None }
                }
                Some(acc)
            }
            ReduceOp::Var | ReduceOp::Sd => {
                if count < 2 { return None; }
                let mut sum = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => sum += v, None => return None }
                }
                let mean = sum / count as f64;
                let mut ss = 0.0_f64;
                for k in 0..count {
                    let v = data[idx(k)].unwrap();
                    let d = v - mean; ss += d * d;
                }
                let var = ss / (count - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                let mut buf = r2_memory::scratch_acquire(count);
                let result: Option<f64> = (|| {
                    for k in 0..count {
                        match data[idx(k)] { Some(v) => buf.push(v), None => return None }
                    }
                    let m = buf.len();
                    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                    if m % 2 == 1 {
                        let (_, mid, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        Some(*mid)
                    } else {
                        let (_, hi, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        let hi_v = *hi;
                        let lo_v = buf[..m / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        Some((lo_v + hi_v) / 2.0)
                    }
                })();
                r2_memory::scratch_release(buf);
                result
            }
        }
    }
}

impl StridedReduceBackend for RayonBackend {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64> {
        debug_assert!(stride > 0, "stride must be > 0");
        if count == 0 { return None; }
        if offset + (count - 1).saturating_mul(stride) >= data.len() { return None; }
        // Two-pass: scan for any NA in parallel; if clean, parallel reduce.
        // (Rayon's try_reduce works on `Try`-implementing items, so we
        // wrap as `Result<f64, ()>` and unwrap via `.ok()` at the end.)
        let any_na = (0..count).into_par_iter()
            .any(|k| data[offset + k * stride].is_none());
        if any_na { return None; }
        match op {
            ReduceOp::Sum => {
                Some((0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .sum::<f64>())
            }
            ReduceOp::Mean => {
                let s: f64 = (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .sum();
                Some(s / count as f64)
            }
            ReduceOp::Min => {
                (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .reduce(|| f64::INFINITY, f64::min)
                    .into()
            }
            ReduceOp::Max => {
                (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .reduce(|| f64::NEG_INFINITY, f64::max)
                    .into()
            }
            ReduceOp::Prod => {
                Some((0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .product::<f64>())
            }
            _ => SerialBackend.reduce_strided(op, data, offset, stride, count),
        }
    }
}

/// Strided reduction dispatcher. Backend chosen by Oracle on the walked
/// element count (`count`), not the underlying slice length.
pub fn reduce_strided(
    op: ReduceOp,
    data: &[Option<f64>],
    offset: usize,
    stride: usize,
    count: usize,
) -> Option<f64> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(count)) {
        Backend::Serial => SerialBackend.reduce_strided(op, data, offset, stride, count),
        // The Oracle only returns Gpu for matmul, never for these ops;

        // if that ever changes, parallel CPU is the safe equivalent.

        Backend::Gpu | Backend::Rayon => RayonBackend.reduce_strided(op, data, offset, stride, count),
    }
}
