//! Element-wise map kernel — Phase K.2.

use crate::{SerialBackend, RayonBackend};
use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════
// Element-wise map kernel — Phase K.2
// ════════════════════════════════════════════════════════════════════
//
// `MapOp` covers unary functions that produce one output per input.
// NA preserved: `None` in → `None` out at the same index. Domain errors
// (sqrt of negative, log of non-positive) yield `NaN` per IEEE 754;
// builtins decide whether to surface them as warnings.

/// Element-wise unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOp {
    Sqrt,
    Abs,
    Exp,
    Ln,
    Log2,
    Log10,
    Neg,
    // Phase R.M.1 — trig and transcendental ops. NA-aware, IEEE-754 NaN
    // propagation through every operation. Match CRAN R 4.5 behavior:
    // sin/cos/tan accept radians, asin/acos return NaN outside [-1,1].
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Sign,
    Trunc,
    Expm1,
    Log1p,
}

pub trait MapBackend: Send + Sync {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>>;
}

#[inline]
fn apply_op(op: MapOp, v: f64) -> f64 {
    match op {
        MapOp::Sqrt  => v.sqrt(),
        MapOp::Abs   => v.abs(),
        MapOp::Exp   => v.exp(),
        MapOp::Ln    => v.ln(),
        MapOp::Log2  => v.log2(),
        MapOp::Log10 => v.log10(),
        MapOp::Neg   => -v,
        MapOp::Sin   => v.sin(),
        MapOp::Cos   => v.cos(),
        MapOp::Tan   => v.tan(),
        MapOp::Asin  => v.asin(),
        MapOp::Acos  => v.acos(),
        MapOp::Atan  => v.atan(),
        MapOp::Sinh  => v.sinh(),
        MapOp::Cosh  => v.cosh(),
        MapOp::Tanh  => v.tanh(),
        MapOp::Sign  => if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else if v == 0.0 { 0.0 } else { f64::NAN },
        MapOp::Trunc => v.trunc(),
        MapOp::Expm1 => v.exp_m1(),
        MapOp::Log1p => v.ln_1p(),
    }
}

impl MapBackend for SerialBackend {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        data.iter().map(|x| x.map(|v| apply_op(op, v))).collect()
    }
}

impl MapBackend for RayonBackend {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        data.par_iter().map(|x| x.map(|v| apply_op(op, v))).collect()
    }
}

/// Element-wise map dispatcher. Backend chosen by Oracle (Op::PerElementMap).
pub fn map(op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.map(op, data),
        // The Oracle only returns Gpu for matmul, never for these ops;

        // if that ever changes, parallel CPU is the safe equivalent.

        Backend::Gpu | Backend::Rayon => RayonBackend.map(op, data),
    }
}
