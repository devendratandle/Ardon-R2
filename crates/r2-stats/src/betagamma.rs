//! The beta and gamma families, as R's `d/p/q` functions: every parameter
//! recycled, `lower.tail` and `log.p` honoured, and the small tail computed
//! directly so p-values keep their digits. The comparison with R itself is in
//! tests/differential/cases/beta_gamma.R.
//!
//! Gamma shares its machinery with χ² (`crate::htest`), which is the same
//! family. Beta is built the same way: the prefactor in Loader's saddle-point
//! form rather than from `lgamma`, whose large terms nearly cancel at large
//! shapes, and both tails on the log scale, the smaller one computed and the
//! other its complement.

use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal, Real};

use crate::dist::{first, flag, param_vec};
use crate::htest::{bd0, chi_sq_quantile, dgamma, ln_gamma, pgamma_log_tails, stirlerr};

const LN_2PI: f64 = 1.837_877_066_409_345_483_56;
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_741_78;

/// `ln(1 - e^l)` for `l <= 0`, without cancellation.
pub(crate) fn ln1mexp(l: f64) -> f64 {
    if l > -std::f64::consts::LN_2 { (-l.exp_m1()).ln() } else { (-l.exp()).ln_1p() }
}

/// `ln B(a, b)`, as R's `lbeta`: where a shape is at least 10, Stirling's
/// corrections replace the `lgamma`s, whose large terms would nearly cancel.
pub fn lbeta(a: f64, b: f64) -> f64 {
    let (p, q) = if a < b { (a, b) } else { (b, a) };
    if p >= 10.0 {
        let corr = stirlerr(p) + stirlerr(q) - stirlerr(p + q);
        -0.5 * q.ln() + LN_SQRT_2PI + corr + (p - 0.5) * (p / (p + q)).ln() + q * (-p / (p + q)).ln_1p()
    } else if q >= 10.0 {
        let corr = stirlerr(q) - stirlerr(p + q);
        ln_gamma(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p()
    } else {
        ln_gamma(p) + ln_gamma(q) - ln_gamma(p + q)
    }
}

/// ln of the binomial density at a real "count" `x` of `n = x + rest`, with
/// `p + q = 1`, in Loader's saddle-point form. Takes `x` and `rest` rather than
/// `x` and `n`: forming `rest` as `n - x` loses it when it is tiny beside `x`
/// (10000.001 - 10000 keeps nine digits), which cost `pbeta` 2e-10.
pub(crate) fn ln_dbinom_raw(x: f64, rest: f64, p: f64, q: f64) -> f64 {
    let n = x + rest;
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(rest) - bd0(x, n * p) - bd0(rest, n * q);
    lc - 0.5 * (LN_2PI + x.ln() + (rest / n).ln())
}

/// ln of the Beta(a, b) density at `x`, for finite positive shapes.
fn ln_dbeta(x: f64, a: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) { return f64::NEG_INFINITY; }
    ln_dbeta_xy(x, 1.0 - x, a, b)
}

/// The same with `y = 1 - x` given separately, which matters when `x` rounds
/// to 1 and only `y` still says how far off it is.
fn ln_dbeta_xy(x: f64, y: f64, a: f64, b: f64) -> f64 {
    if x == 0.0 {
        if a > 1.0 { return f64::NEG_INFINITY; }
        if a < 1.0 { return f64::INFINITY; }
        return b.ln();
    }
    if y == 0.0 {
        if b > 1.0 { return f64::NEG_INFINITY; }
        if b < 1.0 { return f64::INFINITY; }
        return a.ln();
    }
    if a <= 2.0 || b <= 2.0 {
        let (lx, ly) = if x < 0.5 { (x.ln(), (-x).ln_1p()) } else { ((-y).ln_1p(), y.ln()) };
        (a - 1.0) * lx + (b - 1.0) * ly - lbeta(a, b)
    } else {
        (a + b - 1.0).ln() + ln_dbinom_raw(a - 1.0, b - 1.0, x, y)
    }
}

/// `ln Gamma(b + a) - ln Gamma(b)` for `0 < a <= 1`, without subtracting two
/// large numbers: `b` is raised past 15 by the recurrence, then the difference
/// is taken inside Stirling's formula, term by term.
fn lgamma_diff(b: f64, a: f64) -> f64 {
    let (mut bb, mut shift) = (b, 0.0);
    while bb < 15.0 {
        shift += (a / bb).ln_1p();
        bb += 1.0;
    }
    (bb - 0.5) * (a / bb).ln_1p() + a * ((bb + a).ln() - 1.0) + stirlerr(bb + a) - stirlerr(bb) - shift
}

/// `ln I_x(a, b)` for `a <= 1` by the power series
/// `I_x = x^a / (a B(a, b)) * (1 + a sum_j (1-b)_j / j! x^j / (a + j))`.
///
/// With a small shape `I_x` is close to 1 and its complement is the answer
/// wanted, so `ln I_x` needs its small value exactly. Every piece here is
/// itself small, where the continued fraction's prefactor reaches it by
/// cancelling terms near `ln(1/a)`.
fn ln_ix_small_shape(x: f64, a: f64, b: f64) -> f64 {
    let (mut c, mut sum) = (1.0f64, 0.0f64);
    for j in 1..1_000_000 {
        let j = j as f64;
        c *= (j - b) / j * x;
        let t = c / (a + j);
        sum += t;
        if t.abs() <= 1e-18 * sum.abs() { break; }
    }
    // ln(a B(a, b)) = ln Gamma(1 + a) + ln Gamma(b) - ln Gamma(a + b)
    let ln_ab = lgamma_diff(1.0, a) - lgamma_diff(b, a);
    (a * x.ln() - ln_ab + (a * sum).ln_1p()).min(0.0)
}

/// Lentz's continued fraction for the incomplete beta (Numerical Recipes
/// §6.4). The caller keeps `x` in the fast-converging region. It takes
/// O(sqrt(max(a, b))) terms near the mode; the cap is far past that, not a
/// cutoff.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN { d = FPMIN; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..2_000_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 { break; }
    }
    h
}

/// Both tails of the regularised incomplete beta on the log scale,
/// `(ln I_x(a, b), ln(1 - I_x(a, b)))`, with `y = 1 - x` given separately so a
/// caller who has it exactly (the F family) keeps its digits.
pub fn pbeta_log_tails(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    if x <= 0.0 { return (f64::NEG_INFINITY, 0.0); }
    if y <= 0.0 { return (0.0, f64::NEG_INFINITY); }
    if x < (a + 1.0) / (a + b + 2.0) {
        let li = if a <= 1.0 { ln_ix_small_shape(x, a, b) }
            else { (ln_prefactor(x, y, a, b) - a.ln() + betacf(a, b, x).ln()).min(0.0) };
        (li, ln1mexp(li))
    } else {
        let lq = if b <= 1.0 { ln_ix_small_shape(y, b, a) }
            else { (ln_prefactor(x, y, a, b) - b.ln() + betacf(b, a, y).ln()).min(0.0) };
        (ln1mexp(lq), lq)
    }
}

/// `ln(x^a y^b / B(a, b))`, as `dbinom_raw(a, b, x, y) * a b / (a + b)`.
fn ln_prefactor(x: f64, y: f64, a: f64, b: f64) -> f64 {
    ln_dbinom_raw(a, b, x, y) + (a * b / (a + b)).ln()
}

/// One step of the search for a quantile: Newton while it stays in the
/// bracket, else bisection, geometric when the bracket spans decades, and a
/// large fixed jump from a bracket that still starts at 0 (halving from 1
/// could not reach 1e-300 in any sensible number of steps).
fn next_guess(newton: f64, lo: f64, hi: f64) -> f64 {
    if newton > lo && newton < hi { newton }
    else if lo == 0.0 { hi * 1e-6 }
    else if hi > 4.0 * lo { (lo * hi).sqrt() }
    else { 0.5 * (lo + hi) }
}

/// `(x, 1 - x)` with `I_x(a, b) = p`, for a tail `p <= 1/2` given as `lp`.
///
/// It searches in `x` when the root is below 1/2 and in `y = 1 - x` when it is
/// above, so a root within 1e-50 of either end keeps its digits.
fn qbeta_small_tail(lp: f64, a: f64, b: f64) -> (f64, f64) {
    if lp == f64::NEG_INFINITY { return (0.0, 1.0); }
    let lb = lbeta(a, b);
    let i_half = pbeta_log_tails(0.5, 0.5, a, b).0;
    if i_half >= lp {
        // I_x ~ x^a / (a B(a, b)) as x -> 0. When that underflows the true
        // quantile is below the smallest double, and 0 is the right answer.
        let start = ((lp + a.ln() + lb) / a).exp();
        if start == 0.0 { return (0.0, 1.0); }
        let (mut lo, mut hi) = (0.0f64, 0.5f64);
        let mut x = if start < 0.5 { start } else { 0.25 };
        for _ in 0..500 {
            let li = pbeta_log_tails(x, 1.0 - x, a, b).0;
            let f = li - lp;
            if f == 0.0 { break; }
            if f < 0.0 { lo = x; } else { hi = x; }
            let xn = next_guess(x - f / (ln_dbeta(x, a, b) - li).exp(), lo, hi);
            let done = (xn - x).abs() <= 2.0 * f64::EPSILON * x;
            x = xn;
            if done { break; }
        }
        (x, 1.0 - x)
    } else {
        // The upper tail near y -> 0 is y^b / (b B(a, b)), and it is close to 1.
        let start = ((ln1mexp(lp) + b.ln() + lb) / b).exp();
        if start == 0.0 { return (1.0, 0.0); }
        let (mut lo, mut hi) = (0.0f64, 0.5f64);
        let mut y = if start < 0.5 { start } else { 0.25 };
        for _ in 0..500 {
            let li = pbeta_log_tails(1.0 - y, y, a, b).0;
            let f = li - lp;
            if f == 0.0 { break; }
            if f > 0.0 { lo = y; } else { hi = y; }
            let yn = next_guess(y + f / (ln_dbeta_xy(1.0 - y, y, a, b) - li).exp(), lo, hi);
            let done = (yn - y).abs() <= 2.0 * f64::EPSILON * y;
            y = yn;
            if done { break; }
        }
        (1.0 - y, y)
    }
}

/// The `x` whose tail (`lower` or upper) has log probability `lp`. Solves for
/// whichever tail is the smaller; `Q_x(a, b) = I_{1-x}(b, a)` turns an upper
/// tail into a lower one, and the search returns both `x` and `1 - x`, so
/// neither side loses digits by subtracting from 1.
pub fn qbeta_log(lp: f64, a: f64, b: f64, lower: bool) -> f64 {
    if lp.is_nan() || lp > 0.0 { return f64::NAN; }
    let small_is_given = lp < -std::f64::consts::LN_2;
    let lt = if small_is_given { lp } else { ln1mexp(lp) };
    if lower == small_is_given { qbeta_small_tail(lt, a, b).0 } else { qbeta_small_tail(lt, b, a).1 }
}

/// The pair `(x, 1 - x)` of the quantile with log tail `lp`, for a caller that
/// divides by one of them.
pub(crate) fn qbeta_xy(lp: f64, a: f64, b: f64, lower: bool) -> (f64, f64) {
    let small_is_given = lp < -std::f64::consts::LN_2;
    let lt = if small_is_given { lp } else { ln1mexp(lp) };
    if lower == small_is_given {
        qbeta_small_tail(lt, a, b)
    } else {
        let (u, v) = qbeta_small_tail(lt, b, a);
        (v, u)
    }
}

// ── the builtins ────────────────────────────────────────────────────────

pub(crate) fn missing(f: &str, name: &str) -> R2Err {
    R2Err { msg: format!("{f}: argument \"{name}\" is missing, with no default"), kind: ErrKind::Runtime }
}

/// Map the first argument and the parameters together, every one recycled to
/// the longest, as R does; NA in any of them gives NA.
pub(crate) fn map_recycled(a: &[EvalArg], params: &[Vec<Real>], f: impl Fn(f64, &[f64]) -> f64) -> Result<RVal, R2Err> {
    let x: Vec<Real> = first(a).as_reals()?.iter().copied().collect();
    if x.is_empty() || params.iter().any(Vec::is_empty) {
        return Ok(RVal::Numeric(Vec::<Real>::new().into(), Attrs::default()));
    }
    let n = params.iter().map(Vec::len).max().unwrap_or(0).max(x.len());
    let mut p = vec![0.0; params.len()];
    let out: Vec<Real> = (0..n).map(|i| {
        let xv = x[i % x.len()]?;
        for (slot, v) in p.iter_mut().zip(params) { *slot = v[i % v.len()]?; }
        Some(f(xv, &p))
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}

/// `shape` and the scale from `rate` or `scale`. Both given must agree: R
/// stops otherwise.
fn gamma_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    let shape = param_vec(a, "shape", 1)?.ok_or_else(|| missing(f, "shape"))?;
    let rate = param_vec(a, "rate", 2)?;
    let scale = param_vec(a, "scale", 3)?;
    let scale = match (rate, scale) {
        (None, None) => vec![Some(1.0)],
        (None, Some(s)) => s,
        (Some(r), None) => r.iter().map(|v| v.map(|x| 1.0 / x)).collect(),
        (Some(r), Some(s)) => {
            for i in 0..r.len().max(s.len()).min(r.len() * s.len()) {
                if let (Some(rv), Some(sv)) = (r[i % r.len()], s[i % s.len()]) {
                    if (rv * sv - 1.0).abs() >= 1e-15 {
                        return Err(R2Err { msg: format!("{f}: specify 'rate' or 'scale' but not both"), kind: ErrKind::Runtime });
                    }
                }
            }
            s
        }
    };
    Ok(vec![shape, scale])
}

/// `dgamma(x, shape, rate = 1, scale = 1/rate, log = FALSE)`.
pub fn bi_dgamma(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = gamma_params(a, "dgamma")?;
    let log = flag(a, "log", 4, false);
    map_recycled(a, &params, move |x, p| dgamma(x, p[0], p[1], log))
}

/// `pgamma(q, shape, rate = 1, scale = 1/rate, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pgamma(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = gamma_params(a, "pgamma")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |q, p| {
        let (shape, scale) = (p[0], p[1]);
        if q.is_nan() || shape.is_nan() || scale.is_nan() || shape < 0.0 || scale <= 0.0 { return f64::NAN; }
        let (lp, lq) = if q <= 0.0 { (f64::NEG_INFINITY, 0.0) }
            else if shape == 0.0 { (0.0, f64::NEG_INFINITY) }
            else { pgamma_log_tails(shape, q / scale) };
        let l = if lower { lp } else { lq };
        if logp { l } else { l.exp() }
    })
}

/// `qgamma(p, shape, rate = 1, scale = 1/rate, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qgamma(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = gamma_params(a, "qgamma")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |pr, p| {
        let (shape, scale) = (p[0], p[1]);
        if pr.is_nan() || shape.is_nan() || scale.is_nan() || shape < 0.0 || scale <= 0.0 { return f64::NAN; }
        let lp = if logp { pr } else if !(0.0..=1.0).contains(&pr) { return f64::NAN } else { pr.ln() };
        if lp > 0.0 { return f64::NAN; }
        if shape == 0.0 { return 0.0; }
        // Gamma(shape, scale) is chi-squared(2 shape) * scale / 2
        chi_sq_quantile(lp, 2.0 * shape, lower) * (scale / 2.0)
    })
}

/// `shape1` and `shape2`. A non-central request is an error, not a silently
/// central answer.
fn beta_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    let s1 = param_vec(a, "shape1", 1)?.ok_or_else(|| missing(f, "shape1"))?;
    let s2 = param_vec(a, "shape2", 2)?.ok_or_else(|| missing(f, "shape2"))?;
    if let Some(n) = param_vec(a, "ncp", 3)? {
        if n.iter().any(|r| r.map(|x| x != 0.0).unwrap_or(false)) {
            return Err(R2Err { msg: format!("{f}: the non-central beta (ncp != 0) is not implemented"), kind: ErrKind::Runtime });
        }
    }
    Ok(vec![s1, s2])
}

/// Which limit a beta with a zero or infinite shape is: all its mass at 0, at
/// 1, at 1/2, or half at each end. `None` for an ordinary beta.
#[derive(Clone, Copy, PartialEq)]
enum BetaLimit { AtZero, AtOne, AtHalf, BothEnds }

fn beta_limit(a: f64, b: f64) -> Option<BetaLimit> {
    if a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() { return None; }
    Some(if a == 0.0 && b == 0.0 { BetaLimit::BothEnds }
         else if a == 0.0 || a / b == 0.0 { BetaLimit::AtZero }
         else if b == 0.0 || b / a == 0.0 { BetaLimit::AtOne }
         else { BetaLimit::AtHalf })
}

fn bad_shapes(a: f64, b: f64) -> bool { a.is_nan() || b.is_nan() || a < 0.0 || b < 0.0 }

/// `dbeta(x, shape1, shape2, ncp = 0, log = FALSE)`.
pub fn bi_dbeta(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = beta_params(a, "dbeta")?;
    let log = flag(a, "log", 4, false);
    map_recycled(a, &params, move |x, p| {
        let (s1, s2) = (p[0], p[1]);
        if x.is_nan() || bad_shapes(s1, s2) { return f64::NAN; }
        let out = |l: f64| if log { l } else { l.exp() };
        if let Some(limit) = beta_limit(s1, s2) {
            let at = match limit {
                BetaLimit::AtZero => x == 0.0,
                BetaLimit::AtOne => x == 1.0,
                BetaLimit::AtHalf => x == 0.5,
                BetaLimit::BothEnds => x == 0.0 || x == 1.0,
            };
            return if at { f64::INFINITY } else { out(f64::NEG_INFINITY) };
        }
        out(ln_dbeta(x, s1, s2))
    })
}

/// `pbeta(q, shape1, shape2, ncp = 0, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pbeta(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = beta_params(a, "pbeta")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |q, p| {
        let (s1, s2) = (p[0], p[1]);
        if q.is_nan() || bad_shapes(s1, s2) { return f64::NAN; }
        let (zero, half) = (f64::NEG_INFINITY, -std::f64::consts::LN_2);
        let (lp, lq) = if q <= 0.0 { (zero, 0.0) }
            else if q >= 1.0 { (0.0, zero) }
            else { match beta_limit(s1, s2) {
                Some(BetaLimit::AtZero) => (0.0, zero),
                Some(BetaLimit::AtOne) => (zero, 0.0),
                Some(BetaLimit::AtHalf) => if q < 0.5 { (zero, 0.0) } else { (0.0, zero) },
                Some(BetaLimit::BothEnds) => (half, half),
                None => pbeta_log_tails(q, 1.0 - q, s1, s2),
            } };
        let l = if lower { lp } else { lq };
        if logp { l } else { l.exp() }
    })
}

/// `qbeta(p, shape1, shape2, ncp = 0, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qbeta(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = beta_params(a, "qbeta")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |pr, p| {
        let (s1, s2) = (p[0], p[1]);
        if pr.is_nan() || bad_shapes(s1, s2) { return f64::NAN; }
        let lp = if logp { pr } else if !(0.0..=1.0).contains(&pr) { return f64::NAN } else { pr.ln() };
        if lp > 0.0 { return f64::NAN; }
        match beta_limit(s1, s2) {
            Some(BetaLimit::AtZero) => 0.0,
            Some(BetaLimit::AtOne) => 1.0,
            Some(BetaLimit::AtHalf) => 0.5,
            Some(BetaLimit::BothEnds) => {
                let half = -std::f64::consts::LN_2;
                let low = if lower { lp < half } else { lp > half };
                if lp == half { 0.5 } else if low { 0.0 } else { 1.0 }
            }
            None => qbeta_log(lp, s1, s2, lower),
        }
    })
}

#[cfg(test)]
mod tests {
    //! Mathematics that holds whatever any other program prints. The
    //! comparison with R runs R itself: tests/differential/cases/beta_gamma.R.
    use super::*;

    const SHAPES: [f64; 8] = [0.1, 0.5, 1.0, 1.5, 2.0, 7.5, 40.0, 1000.0];

    fn rel(a: f64, b: f64) -> f64 { if a == b { 0.0 } else { ((a - b) / b).abs() } }
    fn tails(x: f64, a: f64, b: f64) -> (f64, f64) {
        let (l, u) = pbeta_log_tails(x, 1.0 - x, a, b);
        (l.exp(), u.exp())
    }

    #[test]
    fn the_two_tails_sum_to_one_and_reflect() {
        for a in SHAPES { for b in SHAPES { for x in [1e-6, 0.01, 0.2, 0.5, 0.77, 0.99] {
            let (l, u) = tails(x, a, b);
            assert!((l + u - 1.0).abs() < 1e-13, "a {a} b {b} x {x}: tails sum to {}", l + u);
            // I_x(a, b) = 1 - I_{1-x}(b, a), with y given exactly: recomputing
            // 1 - (1 - x) would lose the digits being checked
            let (lr, ur) = pbeta_log_tails(1.0 - x, x, b, a);
            assert!(rel(l, ur.exp()) < 1e-11 || l < 1e-290, "a {a} b {b} x {x}: {l} vs {}", ur.exp());
            assert!(rel(u, lr.exp()) < 1e-11 || u < 1e-290, "a {a} b {b} x {x}: {u} vs {}", lr.exp());
        } } }
    }

    /// Closed forms, by independent routes.
    #[test]
    fn closed_forms_hold() {
        for x in [1e-8f64, 0.001, 0.3, 0.5, 0.9] {
            // Beta(1, 1) is uniform; Beta(a, 1) is x^a; Beta(1, b) is 1 - (1-x)^b
            assert!(rel(tails(x, 1.0, 1.0).0, x) < 1e-14, "uniform at {x}");
            assert!(rel(tails(x, 3.5, 1.0).0, x.powf(3.5)) < 1e-13, "x^a at {x}");
            let want = -(3.5 * (-x).ln_1p()).exp_m1();
            assert!(rel(tails(x, 1.0, 3.5).0, want) < 1e-13, "1 - (1-x)^b at {x}: {} vs {want}", tails(x, 1.0, 3.5).0);
            // Beta(1/2, 1/2) is the arcsine law
            let arc = 2.0 / std::f64::consts::PI * x.sqrt().asin();
            assert!(rel(tails(x, 0.5, 0.5).0, arc) < 1e-13, "arcsine at {x}");
        }
    }

    /// The density is the slope of the CDF, on the smaller tail.
    #[test]
    fn the_density_is_the_derivative_of_the_cdf() {
        for a in [0.5, 2.0, 7.5, 40.0] { for b in [0.5, 2.0, 7.5, 40.0] { for x in [0.1f64, 0.3, 0.5, 0.8] {
            let h = 1e-5 * x.min(1.0 - x);
            let slope = if tails(x, a, b).0 < 0.5 {
                (tails(x + h, a, b).0 - tails(x - h, a, b).0) / (2.0 * h)
            } else {
                (tails(x - h, a, b).1 - tails(x + h, a, b).1) / (2.0 * h)
            };
            let d = ln_dbeta(x, a, b).exp();
            if d > 1e-250 { assert!(rel(slope, d) < 1e-5, "a {a} b {b} x {x}: slope {slope} vs {d}"); }
        } } }
    }

    /// The quantile inverts the CDF in each tail, down to p = 1e-300.
    #[test]
    fn the_quantile_inverts_the_cdf_in_both_tails() {
        for a in SHAPES { for b in SHAPES { for p in [1e-300f64, 1e-100, 1e-10, 1e-3, 0.1, 0.4] {
            let (x, y) = qbeta_small_tail(p.ln(), a, b);
            if x > 1e-290 && y > 0.0 {
                let i = pbeta_log_tails(x, y, a, b).0.exp();
                assert!(rel(i, p) < 1e-9, "a {a} b {b} p {p}: x {x} y {y}, I = {i}");
            }
            // the four ways of asking, one answer
            assert_eq!(qbeta_log(p.ln(), a, b, true), x);
            assert_eq!(qbeta_log(p.ln(), b, a, false), y);
            assert_eq!(qbeta_log((-p).ln_1p(), a, b, false), x);
            assert_eq!(qbeta_log((-p).ln_1p(), b, a, true), y);
        } } }
    }

    /// A quantile too small for a double is 0, not a value that misses its
    /// target. `x` is about e^-4500 here.
    #[test]
    fn a_quantile_below_the_smallest_double_is_zero() {
        assert_eq!(qbeta_log(0.01f64.ln(), 0.001, 0.01, true), 0.0);
        assert_eq!(qbeta_log(0.99f64.ln(), 0.001, 0.01, false), 0.0);
        // and the same quantile seen from the other end is exactly 1
        assert_eq!(qbeta_log(0.01f64.ln(), 0.01, 0.001, false), 1.0);
    }

    #[test]
    fn large_shapes_lose_no_digits() {
        // I_{1/2}(n, n) = 1/2 exactly, however large n is
        for n in [10.0, 1e3, 1e5, 1e7] {
            assert!(rel(tails(0.5, n, n).0, 0.5) < 1e-12, "n {n}: {}", tails(0.5, n, n).0);
        }
    }
}
