//! R2 Stats — canonical statistical math.
//!
//! Per docs/ARCHITECTURE.md §5 Phase R:
//!   - Each statistical operation is defined ONCE here.
//!   - r2-engine's `bi_*` functions are thin wrappers that handle
//!     argument parsing/coercion, then delegate to this crate.
//!   - This crate has no dependency on r2-engine — it's reusable by
//!     any Rust caller (other R2 builtins, addon packages, third-party
//!     consumers, the future general-purpose stats framework).
//!
//! Backend dispatch (Serial / Rayon / future GPU) lives below this layer
//! in `r2-kernel`. This crate is pure math + NA semantics; it doesn't
//! know about parallelism.
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only — depends on `r2-types` and `r2-kernel`.
//!   §4.9 Parallelism stays below the kernel layer; this crate uses the
//!        kernel's public API and never touches Rayon directly.

use r2_kernel::{ReduceOp, MapOp, BinaryOp};
use r2_types::*;

// ── Routed output macros ─────────────────────────────────────────────
// Drop-in replacements for `println!` / `print!` that send through the
// engine/GUI-capturable sink (r2_types::out) instead of raw stdout, so
// formatted results (t.test, aov, manova, summary, …) appear in the
// desktop GUI console — not just a terminal. Defined before the module
// declarations so every submodule can use them.
macro_rules! soutln {
    () => { $crate::__rout("\n") };
    ($($arg:tt)*) => { $crate::__rout(&format!("{}\n", format_args!($($arg)*))) };
}
macro_rules! sout {
    ($($arg:tt)*) => { $crate::__rout(&format!("{}", format_args!($($arg)*))) };
}
#[allow(unused_macros)]
macro_rules! serrln {
    () => { $crate::__rerr("\n") };
    ($($arg:tt)*) => { $crate::__rerr(&format!("{}\n", format_args!($($arg)*))) };
}

#[doc(hidden)]
pub fn __rout(s: &str) { r2_types::out::rout(s); }
#[doc(hidden)]
pub fn __rerr(s: &str) { r2_types::out::rerr(s); }

pub mod dist;
pub mod summary;
pub mod htest;
pub mod models;
pub mod rng;
pub mod multivariate;
pub mod mixed;
pub mod time;

// Re-export numerical helpers for engine-side callers (model summaries
// like lm/glm still print inline using these).
pub use dist::{erf, phi, qnorm_approx};
pub use htest::{
    chi_sq_cdf, fmt_pval, gamma_approx, incomplete_beta, ln_gamma,
    signif_stars, t_cdf,
};

// ── Reductions ───────────────────────────────────────────────────────
//
// Each reduction is one line — the kernel handles serial/parallel
// dispatch and NA propagation internally. The engine's `bi_*` wrappers
// call these.

pub fn sum(v: &[Real])    -> Real { r2_kernel::reduce(ReduceOp::Sum,    v) }
pub fn mean(v: &[Real])   -> Real { r2_kernel::reduce(ReduceOp::Mean,   v) }
pub fn min(v: &[Real])    -> Real { r2_kernel::reduce(ReduceOp::Min,    v) }
pub fn max(v: &[Real])    -> Real { r2_kernel::reduce(ReduceOp::Max,    v) }
pub fn prod(v: &[Real])   -> Real { r2_kernel::reduce(ReduceOp::Prod,   v) }
pub fn var(v: &[Real])    -> Real { r2_kernel::reduce(ReduceOp::Var,    v) }
pub fn sd(v: &[Real])     -> Real { r2_kernel::reduce(ReduceOp::Sd,     v) }
pub fn median(v: &[Real]) -> Real { r2_kernel::reduce(ReduceOp::Median, v) }

// ── Element-wise math (re-exposed via the map kernel) ────────────────

pub fn sqrt(v: &[Real]) -> Vec<Real> { r2_kernel::map(MapOp::Sqrt, v) }
pub fn abs(v: &[Real])  -> Vec<Real> { r2_kernel::map(MapOp::Abs,  v) }
pub fn exp(v: &[Real])  -> Vec<Real> { r2_kernel::map(MapOp::Exp,  v) }
pub fn ln(v: &[Real])   -> Vec<Real> { r2_kernel::map(MapOp::Ln,   v) }

// ── Two-vector statistics ────────────────────────────────────────────
//
// Correlation and covariance — Bessel-corrected (sample, n-1 divisor).
// Matches R's `cor()` and `cov()`.

pub fn cov(x: &[Real], y: &[Real]) -> Real {
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter_map(|(a, b)| match (a, b) { (Some(a), Some(b)) => Some((*a, *b)), _ => None })
        .collect();
    let n = pairs.len();
    if n < 2 { return None; }
    let nf = n as f64;
    let mx = pairs.iter().map(|(a, _)| a).sum::<f64>() / nf;
    let my = pairs.iter().map(|(_, b)| b).sum::<f64>() / nf;
    let c = pairs.iter().map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (nf - 1.0);
    Some(c)
}

pub fn cor(x: &[Real], y: &[Real]) -> Real {
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter_map(|(a, b)| match (a, b) { (Some(a), Some(b)) => Some((*a, *b)), _ => None })
        .collect();
    let n = pairs.len();
    if n < 2 { return None; }
    let nf = n as f64;
    let mx = pairs.iter().map(|(a, _)| a).sum::<f64>() / nf;
    let my = pairs.iter().map(|(_, b)| b).sum::<f64>() / nf;
    let c  = pairs.iter().map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (nf - 1.0);
    let sx = (pairs.iter().map(|(a, _)| (a - mx).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    let sy = (pairs.iter().map(|(_, b)| (b - my).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    if sx == 0.0 || sy == 0.0 { None } else { Some(c / (sx * sy)) }
}

// ── Element-wise binary (passthroughs to kernel) ─────────────────────

pub fn add(a: &[Real], b: &[Real]) -> Vec<Real> { r2_kernel::binary(BinaryOp::Add, a, b) }
pub fn sub(a: &[Real], b: &[Real]) -> Vec<Real> { r2_kernel::binary(BinaryOp::Sub, a, b) }
pub fn mul(a: &[Real], b: &[Real]) -> Vec<Real> { r2_kernel::binary(BinaryOp::Mul, a, b) }
pub fn div(a: &[Real], b: &[Real]) -> Vec<Real> { r2_kernel::binary(BinaryOp::Div, a, b) }

// ════════════════════════════════════════════════════════════════════
// Builtin wrappers — Phase R completion
// ════════════════════════════════════════════════════════════════════
//
// One generic type-coercion helper handles RVal → Vec<Real> for the
// common numeric inputs. Each `bi_*` becomes a 3-line wrapper.
//
// Builtins return `Result<RVal, R2Err>`. Both types live in r2-types.
// The function signature matches r2-engine's `BuiltinFn` — no engine
// dep needed because we don't use the `&mut Engine` parameter for
// pure stats operations.

fn coerce_reals(arg: &RVal) -> Result<Vec<Real>, R2Err> {
    match arg {
        RVal::Numeric(v, _) => Ok(v.as_vec().clone()),
        RVal::Integer(v, _) => Ok(v.iter().map(|x| x.map(|n| n as f64)).collect()),
        RVal::Logical(v, _) => Ok(v.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect()),
        RVal::Matrix(m) => Ok(m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect()),
        _ => Err(R2Err {
            msg: format!("cannot reduce non-numeric '{}'", arg.type_name()),
            kind: ErrKind::Type,
        }),
    }
}

fn first_arg(args: &[EvalArg]) -> RVal {
    args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

/// Columnar-aware reduction: for `RVal::Numeric` input, dispatches to
/// `ColumnarF64::{sum,mean,min,max,prod}` on the cached `&[f64]` slice
/// — no `Vec<Option<f64>>` materialisation. For other input types
/// (Integer/Logical/Matrix) falls back to coercing into `Vec<Real>` and
/// then the legacy `&[Real]` kernel path.
///
/// `na_rm` is the R-style `na.rm=` flag; not yet wired to call-site —
/// reductions in v0.1.x default to `na_rm=false` (NA propagates), matching
/// the previous behavior.
macro_rules! reduce_builtin {
    ($name:ident, $stats_fn:path, $col_method:ident) => {
        pub fn $name(args: &[EvalArg]) -> Result<RVal, R2Err> {
            // F.3 columnar fast path: skip the Reals Deref-into-Vec<Real>
            // materialisation and reduce on the cached `&[f64]` directly.
            if let Some(arg) = args.first() {
                if let RVal::Numeric(v, _) = &arg.value {
                    let col = v.columnar();
                    let result = col.$col_method(false);
                    return Ok(RVal::Numeric(vec![result].into(), Attrs::default()));
                }
            }
            let arg = first_arg(args);
            let opts = coerce_reals(&arg)?;
            Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()))
        }
    };
}

reduce_builtin!(bi_sum,  sum,  sum);
reduce_builtin!(bi_mean, mean, mean);
reduce_builtin!(bi_min,  min,  min);
reduce_builtin!(bi_max,  max,  max);

/// Var/Sd/Median don't have ColumnarF64 implementations yet — keep the
/// legacy `Vec<Real>` path via the kernel `reduce` dispatcher. (Migrating
/// these is a follow-up; their cost is dominated by the algorithm, not
/// the boxed-form conversion.)
macro_rules! reduce_builtin_legacy {
    ($name:ident, $stats_fn:path) => {
        pub fn $name(args: &[EvalArg]) -> Result<RVal, R2Err> {
            if let Some(arg) = args.first() {
                if let RVal::Numeric(v, _) = &arg.value {
                    return Ok(RVal::Numeric(vec![$stats_fn(v)].into(), Attrs::default()));
                }
            }
            let arg = first_arg(args);
            let opts = coerce_reals(&arg)?;
            Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()))
        }
    };
}

reduce_builtin_legacy!(bi_prod,   prod);
reduce_builtin_legacy!(bi_var,    var);
reduce_builtin_legacy!(bi_sd,     sd);
reduce_builtin_legacy!(bi_median, median);

/// Returns the list of (name, function-pointer) pairs this crate exports.
/// r2-engine's startup calls this and adds each entry to its registry.
/// Pattern locks in: every domain crate (`r2-ml`, `r2-data`, `r2-graphics`)
/// will export the same shape — `pub fn register_builtins()`.
///
/// Note: the returned signature is `fn(&[EvalArg]) -> Result<RVal, R2Err>`
/// — pure-stats builtins do not need `&mut Engine`. r2-engine wraps these
/// to match its `BuiltinFn` signature at registration time.
pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("sum",    bi_sum),
        ("mean",   bi_mean),
        ("min",    bi_min),
        ("max",    bi_max),
        ("prod",   bi_prod),
        ("var",    bi_var),
        ("sd",     bi_sd),
        ("median", bi_median),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_mean_known() {
        let v: Vec<Real> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        assert_eq!(sum(&v),  Some(10.0));
        assert_eq!(mean(&v), Some(2.5));
        assert_eq!(min(&v),  Some(1.0));
        assert_eq!(max(&v),  Some(4.0));
    }

    #[test]
    fn test_sd_var_match() {
        // [2,4,4,4,5,5,7,9] — mean=5, sum-of-squared-dev=32, sample var=32/7≈4.5714.
        let v: Vec<Real> = vec![Some(2.0), Some(4.0), Some(4.0), Some(4.0), Some(5.0), Some(5.0), Some(7.0), Some(9.0)];
        let s = sd(&v).unwrap();
        let va = var(&v).unwrap();
        assert!((va - 32.0/7.0).abs() < 1e-10, "var={}", va);
        assert!((s - (32.0/7.0_f64).sqrt()).abs() < 1e-10, "sd={}", s);
        assert!((va.sqrt() - s).abs() < 1e-10);
    }

    #[test]
    fn test_cor_perfect_linear() {
        let x: Vec<Real> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let y: Vec<Real> = vec![Some(2.0), Some(4.0), Some(6.0), Some(8.0), Some(10.0)];
        let r = cor(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cov_matches_var() {
        // cov(x, x) == var(x)
        let x: Vec<Real> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        assert!((cov(&x, &x).unwrap() - var(&x).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_na_propagates() {
        let v: Vec<Real> = vec![Some(1.0), None, Some(3.0)];
        assert_eq!(sum(&v),  None);
        assert_eq!(mean(&v), None);
    }

    #[test]
    fn test_map_ops() {
        let v: Vec<Real> = vec![Some(4.0), Some(9.0), Some(16.0)];
        let s = sqrt(&v);
        assert_eq!(s, vec![Some(2.0), Some(3.0), Some(4.0)]);
    }
}
