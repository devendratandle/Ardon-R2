//! Reduction kernel (Phase K) + shared backend marker types.
//!
//! `SerialBackend`/`RayonBackend` are zero-sized markers shared by every
//! kernel family; each family adds its own `impl <Family>Backend` for them
//! in its module. They live here (re-exported from the crate root) so the
//! whole kernel still presents one flat public surface.

use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

/// Reduction operations supported by the kernel.
///
/// `Var` / `Sd` use Bessel's correction (sample variance, n-1 divisor) —
/// matches R's `var()` / `sd()`. NA propagates: any null in the input
/// returns `None` (matching R's default `na.rm = FALSE`).
///
/// `Median` requires sorting; serial uses quickselect O(n), Rayon uses
/// `par_sort_by` O(n log n / p).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
    Prod,
    Var,
    Sd,
    Median,
}

/// Backend trait — one impl per execution model. Each backend reduces a
/// buffer of nullable f64s to a scalar. `None` propagates NA.
pub trait ReduceBackend: Send + Sync {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64>;
}

// ── Serial backend ───────────────────────────────────────────────────

pub struct SerialBackend;

impl ReduceBackend for SerialBackend {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
        let n = data.len();
        match op {
            ReduceOp::Sum => data.iter().try_fold(0.0_f64, |acc, x| x.map(|v| acc + v)),
            ReduceOp::Mean => {
                if n == 0 { return None; }
                data.iter().try_fold(0.0_f64, |acc, x| x.map(|v| acc + v)).map(|s| s / n as f64)
            }
            ReduceOp::Min => {
                if n == 0 { return None; }
                let mut m = f64::INFINITY;
                for x in data {
                    match x { Some(v) => m = m.min(*v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Max => {
                if n == 0 { return None; }
                let mut m = f64::NEG_INFINITY;
                for x in data {
                    match x { Some(v) => m = m.max(*v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Prod => data.iter().try_fold(1.0_f64, |acc, x| x.map(|v| acc * v)),
            ReduceOp::Var | ReduceOp::Sd => {
                if n < 2 { return None; }
                // Two-pass: mean, then sum of squared deviations.
                let mut sum = 0.0; let mut count = 0usize;
                for x in data {
                    match x { Some(v) => { sum += v; count += 1; } None => return None }
                }
                let mean = sum / count as f64;
                let ss: f64 = data.iter().map(|x| {
                    let v = x.unwrap(); let d = v - mean; d * d
                }).sum();
                let var = ss / (count - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                if n == 0 { return None; }
                // Reject if any NA — matches R's default na.rm=FALSE.
                // Use the scratch pool: median is a one-shot
                // materialise-and-discard, perfect fit for buffer
                // recycling.
                let mut buf = r2_memory::scratch_acquire(n);
                let result: Option<f64> = (|| {
                    for x in data {
                        match x { Some(v) => buf.push(*v), None => return None }
                    }
                    let m = buf.len();
                    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                    if m % 2 == 1 {
                        let (_, mid, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        Some(*mid)
                    } else {
                        let upper_idx = m / 2;
                        let (lower, upper, _) = buf.select_nth_unstable_by(upper_idx, cmp);
                        let upper_val = *upper;
                        let lower_val = lower.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        Some((lower_val + upper_val) / 2.0)
                    }
                })();
                r2_memory::scratch_release(buf);
                result
            }
        }
    }
}

// ── Rayon backend ────────────────────────────────────────────────────
//
// Single-pass NaN-propagation pattern: map None→NaN, reduce, check for NaN
// at the end. Avoids two passes (NA-check + sum) at the cost of producing
// NaN as the in-band null marker during reduction.

pub struct RayonBackend;

impl ReduceBackend for RayonBackend {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
        let n = data.len();
        match op {
            ReduceOp::Sum => {
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { None } else { Some(s) }
            }
            ReduceOp::Mean => {
                if n == 0 { return None; }
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { None } else { Some(s / n as f64) }
            }
            ReduceOp::Min => {
                if n == 0 { return None; }
                let r = data.par_iter().map(|x| x.unwrap_or(f64::NAN))
                    .reduce(|| f64::INFINITY, f64::min);
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Max => {
                if n == 0 { return None; }
                let r = data.par_iter().map(|x| x.unwrap_or(f64::NAN))
                    .reduce(|| f64::NEG_INFINITY, f64::max);
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Prod => {
                let r: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).product();
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Var | ReduceOp::Sd => {
                if n < 2 { return None; }
                // Single-pass any-NA detection using NaN propagation, then
                // a parallel mean and a parallel sum of squared deviations.
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { return None; }
                let mean = s / n as f64;
                let ss: f64 = data.par_iter()
                    .map(|x| { let v = x.unwrap(); let d = v - mean; d * d }).sum();
                let var = ss / (n - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                if n == 0 { return None; }
                // Strip NAs (defaults match serial path); par_sort the rest.
                let mut buf: Vec<f64> = Vec::with_capacity(n);
                for x in data {
                    match x { Some(v) => buf.push(*v), None => return None }
                }
                let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                buf.par_sort_by(cmp);
                let m = buf.len();
                Some(if m % 2 == 1 { buf[m/2] } else { (buf[m/2-1] + buf[m/2]) / 2.0 })
            }
        }
    }
}

// ── Top-level dispatcher ─────────────────────────────────────────────

/// Reduce a slice of nullable f64 to a scalar. Backend is chosen by Oracle.
/// This is the public entry point — builtins call this and never see Rayon.
pub fn reduce(op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.reduce(op, data),
        // The Oracle only returns Gpu for matmul, never for these ops;

        // if that ever changes, parallel CPU is the safe equivalent.

        Backend::Gpu | Backend::Rayon => RayonBackend.reduce(op, data),
    }
}
