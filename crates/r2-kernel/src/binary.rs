//! Binary kernel — Phase K.3.

use crate::{SerialBackend, RayonBackend};
use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════
// Binary kernel — Phase K.3
// ════════════════════════════════════════════════════════════════════
//
// Element-wise vector⊗vector arithmetic. Both inputs must have matching
// length (R-style recycling is the *caller's* responsibility for now —
// recycling lives at a higher layer because it depends on R-syntax
// semantics, not on numerical kernels). NA in either input → NA out.

/// Element-wise binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

#[inline]
fn apply_binop(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Pow => a.powf(b),
        BinaryOp::Mod => a.rem_euclid(b),
    }
}

pub trait BinaryBackend: Send + Sync {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl BinaryBackend for SerialBackend {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "binary kernel: length mismatch (caller must recycle)");
        a.iter().zip(b.iter()).map(|(x, y)| match (x, y) {
            (Some(xv), Some(yv)) => Some(apply_binop(op, *xv, *yv)),
            _ => None,
        }).collect()
    }
}

impl BinaryBackend for RayonBackend {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "binary kernel: length mismatch (caller must recycle)");
        a.par_iter().zip(b.par_iter()).map(|(x, y)| match (x, y) {
            (Some(xv), Some(yv)) => Some(apply_binop(op, *xv, *yv)),
            _ => None,
        }).collect()
    }
}

/// Element-wise binary dispatcher. Backend chosen by Oracle.
/// Inputs must have equal length — recycling is a higher-layer concern.
pub fn binary(op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(a.len())) {
        Backend::Serial => SerialBackend.binary(op, a, b),
        // The Oracle only returns Gpu for matmul, never for these ops;

        // if that ever changes, parallel CPU is the safe equivalent.

        Backend::Gpu | Backend::Rayon => RayonBackend.binary(op, a, b),
    }
}
