//! Ternary kernel — Phase K.5 (fused multiply-add and friends).

use crate::{SerialBackend, RayonBackend};
use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════
// Ternary kernel — Phase K.5 (Tier 4): fused multiply-add and friends.
// ════════════════════════════════════════════════════════════════════
//
// Three-input element-wise ops. Initial members: `MulAdd` (`a*b + c`).
// Useful for BLAS-like inner loops, polynomial evaluation, weighted
// sums, gemm row-update kernels, and as a JIT specialisation target.
// NA propagation: any None among the three inputs at position i → None.

/// Element-wise ternary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryOp {
    /// Fused multiply-add: `a*b + c`. Uses `f64::mul_add` so on hardware
    /// with an FMA instruction the multiply and add are a single rounded
    /// operation (one ulp instead of two). Falls back to scalar `a*b + c`
    /// on platforms without FMA.
    MulAdd,
}

#[inline]
fn apply_ternop(op: TernaryOp, a: f64, b: f64, c: f64) -> f64 {
    match op {
        TernaryOp::MulAdd => a.mul_add(b, c),
    }
}

pub trait TernaryBackend: Send + Sync {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl TernaryBackend for SerialBackend {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "ternary kernel: a/b length mismatch");
        debug_assert_eq!(a.len(), c.len(), "ternary kernel: a/c length mismatch");
        a.iter().zip(b.iter()).zip(c.iter()).map(|((x, y), z)| match (x, y, z) {
            (Some(xv), Some(yv), Some(zv)) => Some(apply_ternop(op, *xv, *yv, *zv)),
            _ => None,
        }).collect()
    }
}

impl TernaryBackend for RayonBackend {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "ternary kernel: a/b length mismatch");
        debug_assert_eq!(a.len(), c.len(), "ternary kernel: a/c length mismatch");
        (0..a.len()).into_par_iter().map(|i| match (a[i], b[i], c[i]) {
            (Some(xv), Some(yv), Some(zv)) => Some(apply_ternop(op, xv, yv, zv)),
            _ => None,
        }).collect()
    }
}

/// Element-wise ternary dispatcher. Backend chosen by Oracle.
/// All three inputs must have equal length — caller handles recycling.
pub fn ternary(op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(a.len())) {
        Backend::Serial => SerialBackend.ternary(op, a, b, c),
        Backend::Rayon => RayonBackend.ternary(op, a, b, c),
    }
}
