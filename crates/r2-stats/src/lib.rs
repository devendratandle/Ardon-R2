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
pub mod moments;
pub mod summary;
pub mod htest;
pub mod models;
pub mod rng;
pub mod multivariate;
pub mod mixed;
pub mod time;
pub mod plssem;

// Re-export numerical helpers for engine-side callers (model summaries
// like lm/glm still print inline using these).
pub use dist::{erf, erfc, phi, phi_upper, qnorm_approx};
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

/// Whether `na.rm=TRUE` (or `=T`, or a non-zero numeric) was passed.
fn na_rm_flag(args: &[EvalArg]) -> bool {
    args.iter()
        .find(|a| a.name.as_deref() == Some("na.rm"))
        .map(|a| match &a.value {
            RVal::Logical(l, _) => matches!(l.as_vec().first(), Some(Some(true))),
            RVal::Numeric(n, _) => matches!(n.as_vec().first(), Some(Some(x)) if *x != 0.0),
            RVal::Integer(i, _) => matches!(i.as_vec().first(), Some(Some(x)) if *x != 0),
            _ => false,
        })
        .unwrap_or(false)
}

/// Columnar-aware reduction: for `RVal::Numeric` input, dispatches to
/// `ColumnarF64::{sum,mean,min,max,prod}` on the cached `&[f64]` slice
/// — no `Vec<Option<f64>>` materialisation. For other input types
/// (Integer/Logical/Matrix) falls back to coercing into `Vec<Real>` and
/// then the legacy `&[Real]` kernel path.
///
/// The R-style `na.rm=` flag is honored: `na.rm=TRUE` drops NAs before
/// reducing (the columnar path passes the flag to `ColumnarF64`; the
/// fallback path filters `None` out of the boxed form). Default `false`
/// → NA propagates, matching R.
macro_rules! reduce_builtin {
    ($name:ident, $stats_fn:path, $col_method:ident, $variadic:expr) => {
        pub fn $name(args: &[EvalArg]) -> Result<RVal, R2Err> {
            let na_rm = na_rm_flag(args);
            let data_args: Vec<&EvalArg> =
                args.iter().filter(|a| a.name.as_deref() != Some("na.rm")).collect();
            // Single data arg: F.3 columnar fast path on the cached &[f64].
            if data_args.len() <= 1 {
                if let Some(arg) = data_args.first() {
                    if let RVal::Numeric(v, _) = &arg.value {
                        let col = v.columnar();
                        return Ok(RVal::Numeric(vec![col.$col_method(na_rm)].into(), Attrs::default()));
                    }
                }
                let arg = data_args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null);
                let mut opts = coerce_reals(&arg)?;
                if na_rm { opts.retain(|x| x.is_some()); }
                return Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()));
            }
            // Multiple data args: variadic fns (sum/min/max) combine ALL of
            // them; non-variadic (mean) use only the first, matching R.
            let mut opts: Vec<Real> = Vec::new();
            if $variadic { for a in &data_args { opts.extend(coerce_reals(&a.value)?); } }
            else { opts = coerce_reals(&data_args[0].value)?; }
            if na_rm { opts.retain(|x| x.is_some()); }
            Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()))
        }
    };
}

reduce_builtin!(bi_sum,  sum,  sum,  true);
reduce_builtin!(bi_mean, mean, mean, false);
reduce_builtin!(bi_min,  min,  min,  true);
reduce_builtin!(bi_max,  max,  max,  true);

/// Var/Sd/Median don't have ColumnarF64 implementations yet — keep the
/// legacy `Vec<Real>` path via the kernel `reduce` dispatcher. (Migrating
/// these is a follow-up; their cost is dominated by the algorithm, not
/// the boxed-form conversion.)
macro_rules! reduce_builtin_legacy {
    ($name:ident, $stats_fn:path, $variadic:expr) => {
        pub fn $name(args: &[EvalArg]) -> Result<RVal, R2Err> {
            let na_rm = na_rm_flag(args);
            let data_args: Vec<&EvalArg> =
                args.iter().filter(|a| a.name.as_deref() != Some("na.rm")).collect();
            if data_args.len() <= 1 {
                if let Some(arg) = data_args.first() {
                    if let RVal::Numeric(v, _) = &arg.value {
                        if na_rm {
                            let opts: Vec<Real> =
                                v.as_vec().iter().copied().filter(|x| x.is_some()).collect();
                            return Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()));
                        }
                        return Ok(RVal::Numeric(vec![$stats_fn(v)].into(), Attrs::default()));
                    }
                }
                let arg = data_args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null);
                let mut opts = coerce_reals(&arg)?;
                if na_rm { opts.retain(|x| x.is_some()); }
                return Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()));
            }
            let mut opts: Vec<Real> = Vec::new();
            if $variadic { for a in &data_args { opts.extend(coerce_reals(&a.value)?); } }
            else { opts = coerce_reals(&data_args[0].value)?; }
            if na_rm { opts.retain(|x| x.is_some()); }
            Ok(RVal::Numeric(vec![$stats_fn(&opts)].into(), Attrs::default()))
        }
    };
}

reduce_builtin_legacy!(bi_prod,   prod,   true);
reduce_builtin_legacy!(bi_var,    var,    false);
reduce_builtin_legacy!(bi_sd,     sd,     false);
reduce_builtin_legacy!(bi_median, median, false);

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
    fn na_rm_flag_is_honored() {
        fn val(r: RVal) -> Option<f64> {
            if let RVal::Numeric(v, _) = r { v.as_vec().first().copied().flatten() } else { None }
        }
        // data = c(1, 2, NA, 4)
        let data = EvalArg {
            name: None,
            value: RVal::Numeric(vec![Some(1.0), Some(2.0), None, Some(4.0)].into(), Attrs::default()),
        };
        let narm = EvalArg {
            name: Some(std::sync::Arc::from("na.rm")),
            value: RVal::Logical(Logicals::new(vec![Some(true)]), Attrs::default()),
        };
        let with = vec![data.clone(), narm];
        assert_eq!(val(bi_sum(&with).unwrap()),  Some(7.0));
        assert_eq!(val(bi_mean(&with).unwrap()), Some(7.0 / 3.0));
        assert_eq!(val(bi_min(&with).unwrap()),  Some(1.0));
        assert_eq!(val(bi_max(&with).unwrap()),  Some(4.0));
        assert_eq!(val(bi_median(&with).unwrap()), Some(2.0));
        // Default (no na.rm): NA propagates.
        assert_eq!(val(bi_mean(&[data]).unwrap()), None);
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
