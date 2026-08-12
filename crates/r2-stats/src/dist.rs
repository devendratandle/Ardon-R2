//! Probability distributions — Phase R.9.
//!
//! Density, CDF, and quantile functions for the standard normal,
//! plus the small numerical helpers (`erf`, `qnorm_approx`, `phi`) that
//! are also used by other r2 builtins (t-test, chisq-test, etc.).
//!
//! Random-variate generators (`rnorm`, `runif`, `sample`, `rbinom`,
//! `rpois`) are NOT in this module. They share an RNG state with
//! `r2_ml::tree::SEED_STATE` and currently live in r2-engine pending
//! a separate decision on where the RNG primitive should live.

use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal, Real};

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

// ─────────────────────────────────────────────────────────────────────
// Numerical primitives — used by dnorm/pnorm/qnorm AND by t-test /
// chisq-test in r2-engine. Re-exported from r2-stats so engine helpers
// can drop the duplicated definitions.
// ─────────────────────────────────────────────────────────────────────

/// Cody's rational-Chebyshev approximation of `erf`/`erfc`
/// (W. J. Cody, "Rational Chebyshev Approximation for the Error Function",
/// Math. Comp. 23 (1969), 631–637). Full double precision — max relative
/// error ≈ 1e-16, the same algorithm class R and Boost use. `jint`:
/// 0 → erf(x), 1 → erfc(x), 2 → exp(x²)·erfc(x) (scaled, tail-stable).
fn calerf(x: f64, jint: i32) -> f64 {
    const A: [f64; 5] = [3.16112374387056560e0, 1.13864154151050156e2,
        3.77485237685302021e2, 3.20937758913846947e3, 1.85777706184603153e-1];
    const B: [f64; 4] = [2.36012909523441209e1, 2.44024637934444173e2,
        1.28261652607737228e3, 2.84423683343917062e3];
    const C: [f64; 9] = [5.64188496988670089e-1, 8.88314979438837594e0,
        6.61191906371416295e1, 2.98635138197400131e2, 8.81952221241769090e2,
        1.71204761263407058e3, 2.05107837782607147e3, 1.23033935479799725e3,
        2.15311535474403846e-8];
    const D: [f64; 8] = [1.57449261107098347e1, 1.17693950891312499e2,
        5.37181101862009858e2, 1.62138957456669019e3, 3.29079923573345963e3,
        4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3];
    const P: [f64; 6] = [3.05326634961232344e-1, 3.60344899949804439e-1,
        1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4,
        1.63153871373020978e-2];
    const Q: [f64; 5] = [2.56852019228982242e0, 1.87295284992346047e0,
        5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
    const SQRPI: f64 = 5.6418958354775628695e-1; // 1/sqrt(pi)
    const THRESH: f64 = 0.46875;
    const XNEG: f64 = -26.628; // exp underflow bound region

    let y = x.abs();
    let mut result;
    if y <= THRESH {
        // erf on the central interval.
        let z = if y > 1.11e-16 { y * y } else { 0.0 };
        let mut xnum = A[4] * z; let mut xden = z;
        for i in 0..3 { xnum = (xnum + A[i]) * z; xden = (xden + B[i]) * z; }
        result = x * (xnum + A[3]) / (xden + B[3]);
        if jint != 0 { result = 1.0 - result; }
        if jint == 2 { result *= (z).exp(); }
        return result;
    } else if y <= 4.0 {
        // erfc on the mid interval.
        let mut xnum = C[8] * y; let mut xden = y;
        for i in 0..7 { xnum = (xnum + C[i]) * y; xden = (xden + D[i]) * y; }
        result = (xnum + C[7]) / (xden + D[7]);
        let ysq = (y * 16.0).trunc() / 16.0;
        let del = (y - ysq) * (y + ysq);
        result *= (-ysq * ysq).exp() * (-del).exp();
    } else {
        // erfc on the tail via asymptotic series in 1/y².
        result = 0.0;
        if y < XNEG.abs() {
            let z = 1.0 / (y * y);
            let mut xnum = P[5] * z; let mut xden = z;
            for i in 0..4 { xnum = (xnum + P[i]) * z; xden = (xden + Q[i]) * z; }
            result = z * (xnum + P[4]) / (xden + Q[4]);
            result = (SQRPI - result) / y;
            let ysq = (y * 16.0).trunc() / 16.0;
            let del = (y - ysq) * (y + ysq);
            result *= (-ysq * ysq).exp() * (-del).exp();
        }
    }
    // Assemble the requested function, honoring sign of x.
    match jint {
        0 => { let mut r = 0.5 - result + 0.5; if x < 0.0 { r = -r; } r }        // erf
        1 => if x < 0.0 { 2.0 - result } else { result },                        // erfc
        _ => { // scaled erfc
            if x < 0.0 {
                let ysq = (x * 16.0).trunc() / 16.0;
                let del = (x - ysq) * (x + ysq);
                2.0 * (ysq * ysq).exp() * (del).exp() - result
            } else { result }
        }
    }
}

/// Error function, full double precision (Cody 1969).
pub fn erf(x: f64) -> f64 { calerf(x, 0) }
/// Complementary error function, tail-stable to ~1e-16.
pub fn erfc(x: f64) -> f64 { calerf(x, 1) }

/// Standard normal CDF `P(Z ≤ x)`, full double precision — Cody's SPECFUN
/// `pnorm_both` (the exact algorithm R uses in `pnorm.c`). Works DIRECTLY
/// on the z-score `x` (never `erf(x/√2)`), which is what recovers the last
/// two digits: no `÷√2` rounding of the argument, and the tail's
/// `exp(-x²/2)` is split on the z-score itself so e.g. `x=8` uses
/// `exp(-32)` exactly. Returns the lower tail. `phi_upper` returns the
/// upper tail directly (no `1 - Φ` cancellation), so extreme-tail p-values
/// keep full relative precision. Every p-value in the engine inherits this.
fn pnorm_both(x: f64) -> (f64, f64) {  // (lower = Φ(x), upper = 1-Φ(x))
    // R's pnorm.c coefficients — fitted SPECIFICALLY for the normal CDF
    // (1/√(2π) baked in, exp(-x²/2) factor), NOT the generic erf set. This
    // is what makes the direct z-score evaluation full-precision.
    const A: [f64; 5] = [2.2352520354606839287, 161.02823106855587881,
        1067.6894854603709582, 18154.981253343561249, 0.065682337918207449113];
    const B: [f64; 4] = [47.20258190468824187, 976.09855173777669322,
        10260.932208618978205, 45507.789335026729956];
    const C: [f64; 9] = [0.39894151208813466764, 8.8831497943883759412,
        93.506656132177855979, 597.27027639480026226, 2494.5375852903726711,
        6848.1904505362823326, 11602.651437647350124, 9842.7148383839780218,
        1.0765576773720192317e-8];
    const D: [f64; 8] = [22.266688044328115691, 235.38790178262499861,
        1519.377599407554805, 6485.558298266760755, 18615.571640885098091,
        34900.952721145977266, 38912.003286093271411, 19685.429676859990727];
    const P: [f64; 6] = [0.21589853405795699, 0.1274011611602473639,
        0.022235277870649807, 0.001421619193227893466, 2.9112874951168792e-5,
        0.02307344176494017303];
    const Q: [f64; 5] = [1.28426009614491121, 0.468238212480865118,
        0.0659881378689285515, 0.00378239633202758244, 7.29751555083966205e-5];
    const M_1_SQRT_2PI: f64 = 0.398942280401432677939946059934; // 1/√(2π)
    const M_SQRT_32: f64 = 5.656854249492380195206754896838;    // √32
    let eps = f64::EPSILON * 0.5;

    let y = x.abs();
    if y <= 0.67448975 {
        // Central region: rational approx gives Φ(x) - ½ directly.
        let mut xsq = 0.0;
        if y > eps { xsq = x * x; }
        let mut xnum = A[4] * xsq; let mut xden = xsq;
        for i in 0..3 { xnum = (xnum + A[i]) * xsq; xden = (xden + B[i]) * xsq; }
        let temp = x * (xnum + A[3]) / (xden + B[3]);
        return (0.5 + temp, 0.5 - temp);
    }
    let (cum, del, xsq);
    if y <= M_SQRT_32 {
        // Moderate tail: erfc-form rational approx on the z-score.
        let mut xnum = C[8] * y; let mut xden = y;
        for i in 0..7 { xnum = (xnum + C[i]) * y; xden = (xden + D[i]) * y; }
        let temp = (xnum + C[7]) / (xden + D[7]);
        xsq = (y * 16.0).trunc() / 16.0;
        del = (y - xsq) * (y + xsq);
        cum = (-xsq * xsq * 0.5).exp() * (-del * 0.5).exp() * temp;
    } else {
        // Extreme tail (|x| > √32): asymptotic series in 1/x².
        let z = 1.0 / (x * x);
        let mut xnum = P[5] * z; let mut xden = z;
        for i in 0..4 { xnum = (xnum + P[i]) * z; xden = (xden + Q[i]) * z; }
        let mut temp = z * (xnum + P[4]) / (xden + Q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;
        xsq = (y * 16.0).trunc() / 16.0;      // split-exp on the z-score →
        del = (y - xsq) * (y + xsq);          // exp(-32) exact at x=8, etc.
        cum = (-xsq * xsq * 0.5).exp() * (-del * 0.5).exp() * temp;
    }
    // `cum` is the SMALL tail (mass beyond |x|); assign by sign of x.
    if x > 0.0 { (1.0 - cum, cum) } else { (cum, 1.0 - cum) }
}

/// Standard normal CDF `P(Z ≤ x)` — full double precision.
#[inline]
pub fn phi(x: f64) -> f64 { pnorm_both(x).0 }

/// Upper tail `P(Z > x)` directly — avoids the `1 - Φ(x)` cancellation, so
/// small right-tail p-values keep full relative precision.
#[inline]
pub fn phi_upper(x: f64) -> f64 { pnorm_both(x).1 }

/// Inverse standard-normal CDF via **Wichura's algorithm AS241**
/// (`PPND16`) — the same rational approximation R uses. Accurate to
/// ~1e-15 across the whole open interval (0, 1), including the tails.
/// (The previous Abramowitz-Stegun 26.2.23 fit was only ~4.5e-4.)
pub fn qnorm_approx(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        // Central region.
        let r = 0.180625 - q * q;
        let num = ((((((2509.0809287301226727_f64 * r + 33430.575583588128105) * r
            + 67265.770927008700853) * r + 45921.953931549871457) * r
            + 13731.693765509461125) * r + 1971.5909503065514427) * r
            + 133.14166789178437745) * r + 3.387132872796366608;
        let den = ((((((5226.495278852854561_f64 * r + 28729.085735721942674) * r
            + 39307.89580009271061) * r + 21213.794301586595867) * r
            + 5394.1960214247511077) * r + 687.1870074920579083) * r
            + 42.313330701600911252) * r + 1.0;
        return q * num / den;
    }

    // Tail regions.
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        let num = ((((((7.7454501427834140764e-4_f64 * r + 0.0227238449892691845833) * r
            + 0.24178072517745061177) * r + 1.27045825245236838258) * r
            + 3.64784832476320460504) * r + 5.7694972214606914055) * r
            + 4.6303378461565452959) * r + 1.42343711074968357734;
        let den = ((((((1.05075007164441684324e-9_f64 * r + 5.475938084995344946e-4) * r
            + 0.0151986665636164571966) * r + 0.14810397642748007459) * r
            + 0.68976733498510000455) * r + 1.6763848301838038494) * r
            + 2.05319162663775882187) * r + 1.0;
        num / den
    } else {
        let r = r - 5.0;
        let num = ((((((2.01033439929228813265e-7_f64 * r + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386) * r + 0.026532189526576123093) * r
            + 0.29656057182850489123) * r + 1.7848265399172913358) * r
            + 5.4637849111641143699) * r + 6.6579046435011037772;
        let den = ((((((2.04426310338993978564e-15_f64 * r + 1.4215117583164458887e-7) * r
            + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
            + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
            + 0.59983220655588793769) * r + 1.0;
        num / den
    };
    if q < 0.0 { -val } else { val }
}

// ─────────────────────────────────────────────────────────────────────
// Builtins
// ─────────────────────────────────────────────────────────────────────

fn mean_sd(a: &[EvalArg]) -> (f64, f64) {
    let mean = arg_named(a, "mean").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);
    let sd = arg_named(a, "sd").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0);
    (mean, sd)
}

pub fn bi_dnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    let result: Vec<Real> = x.iter().map(|v| v.map(|x| {
        let z = (x - mean) / sd;
        (1.0 / (sd * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
    })).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_pnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    // Use the full-precision direct CDF (`phi` = Cody SPECFUN pnorm_both),
    // NOT `0.5*(1+erf(z/√2))` — the erf route loses ~1 ULP to the ÷√2 and
    // mishandles the tail split-exp.
    let result: Vec<Real> = x.iter().map(|v| v.map(|x| phi((x - mean) / sd))).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_qnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let p = first(a).as_reals()?;
    let (mean, sd) = mean_sd(a);
    let result: Vec<Real> = p.iter().map(|v| v.map(|p| mean + sd * qnorm_approx(p))).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Additional distributions (Tier-2): exponential, binomial, Poisson,
// Student-t, chi-squared, F. d/p forms (+ q for exponential). Reuse the
// special functions in `crate::htest`.
// ─────────────────────────────────────────────────────────────────────

use crate::htest::{t_cdf, chi_sq_cdf, incomplete_beta, ln_gamma};

/// `x * ln(y)` with the convention `0 * ln(0) = 0` (avoids NaN in pmf logs).
#[inline]
fn xlogy(x: f64, y: f64) -> f64 { if x == 0.0 { 0.0 } else { x * y.ln() } }

/// A scalar parameter by `name=`, else the `pos`-th unnamed argument, else default.
fn param(a: &[EvalArg], name: &str, pos: usize, dflt: f64) -> f64 {
    arg_named(a, name).and_then(|v| v.scalar_f64().ok().flatten())
        .or_else(|| a.iter().filter(|x| x.name.is_none()).nth(pos)
            .and_then(|x| x.value.scalar_f64().ok().flatten()))
        .unwrap_or(dflt)
}
/// Map the first argument's vector element-wise through `f`.
fn map_x(a: &[EvalArg], f: impl Fn(f64) -> f64) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    Ok(RVal::Numeric(x.iter().map(|v| v.map(&f)).collect::<Vec<Real>>().into(), Attrs::default()))
}
fn ln_binom(n: f64, x: f64, p: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(x + 1.0) - ln_gamma(n - x + 1.0) + xlogy(x, p) + xlogy(n - x, 1.0 - p)
}

pub fn bi_dexp(a: &[EvalArg]) -> Result<RVal, R2Err> { let r = param(a,"rate",1,1.0); map_x(a, move |x| if x < 0.0 { 0.0 } else { r * (-r * x).exp() }) }
pub fn bi_pexp(a: &[EvalArg]) -> Result<RVal, R2Err> { let r = param(a,"rate",1,1.0); map_x(a, move |x| if x < 0.0 { 0.0 } else { 1.0 - (-r * x).exp() }) }
pub fn bi_qexp(a: &[EvalArg]) -> Result<RVal, R2Err> { let r = param(a,"rate",1,1.0); map_x(a, move |p| -(1.0 - p).ln() / r) }

pub fn bi_dbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = param(a,"size",1,0.0); let p = param(a,"prob",2,0.5);
    map_x(a, move |x| { let xi = x.round(); if xi < 0.0 || xi > n { 0.0 } else { ln_binom(n, xi, p).exp() } })
}
pub fn bi_pbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = param(a,"size",1,0.0); let p = param(a,"prob",2,0.5);
    map_x(a, move |q| { let qi = q.floor(); let mut s = 0.0; let mut i = 0.0;
        while i <= qi && i <= n { s += ln_binom(n, i, p).exp(); i += 1.0; } s.min(1.0) })
}
pub fn bi_dpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let lam = param(a,"lambda",1,1.0);
    map_x(a, move |x| { let xi = x.round(); if xi < 0.0 { 0.0 } else { (xlogy(xi, lam) - lam - ln_gamma(xi + 1.0)).exp() } })
}
pub fn bi_ppois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let lam = param(a,"lambda",1,1.0);
    map_x(a, move |q| { let qi = q.floor(); let mut s = 0.0; let mut i = 0.0;
        while i <= qi { s += (xlogy(i, lam) - lam - ln_gamma(i + 1.0)).exp(); i += 1.0; } s.min(1.0) })
}
pub fn bi_dt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = param(a,"df",1,1.0);
    let c = (ln_gamma((df + 1.0) / 2.0) - ln_gamma(df / 2.0)).exp() / (df * std::f64::consts::PI).sqrt();
    map_x(a, move |x| c * (1.0 + x * x / df).powf(-(df + 1.0) / 2.0))
}
pub fn bi_pt(a: &[EvalArg]) -> Result<RVal, R2Err> { let df = param(a,"df",1,1.0); map_x(a, move |x| t_cdf(x, df)) }
pub fn bi_dchisq(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = param(a,"df",1,1.0); let k = df / 2.0;
    map_x(a, move |x| if x < 0.0 { 0.0 } else {
        (xlogy(k - 1.0, x) - x / 2.0 - k * std::f64::consts::LN_2 - ln_gamma(k)).exp()
    })
}
pub fn bi_pchisq(a: &[EvalArg]) -> Result<RVal, R2Err> { let df = param(a,"df",1,1.0); map_x(a, move |x| chi_sq_cdf(x, df)) }
pub fn bi_pf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let d1 = param(a,"df1",1,1.0); let d2 = param(a,"df2",2,1.0);
    map_x(a, move |x| if x <= 0.0 { 0.0 } else { incomplete_beta(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2)) })
}

/// Invert a monotone CDF on `[lo, hi]` by bisection (continuous quantiles).
fn quantile_bisect(p: f64, mut lo: f64, mut hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    if p <= 0.0 { return lo; }
    if p >= 1.0 { return hi; }
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if cdf(m) < p { lo = m; } else { hi = m; }
    }
    0.5 * (lo + hi)
}
pub fn bi_qt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = param(a,"df",1,1.0);
    map_x(a, move |p| quantile_bisect(p, -1.0e7, 1.0e7, |x| t_cdf(x, df)))
}
pub fn bi_qchisq(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = param(a,"df",1,1.0);
    map_x(a, move |p| quantile_bisect(p, 0.0, 1.0e7, |x| chi_sq_cdf(x, df)))
}
pub fn bi_qf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let d1 = param(a,"df1",1,1.0); let d2 = param(a,"df2",2,1.0);
    map_x(a, move |p| quantile_bisect(p, 0.0, 1.0e7, |x|
        if x <= 0.0 { 0.0 } else { incomplete_beta(d1 / 2.0, d2 / 2.0, d1 * x / (d1 * x + d2)) }))
}
pub fn bi_qbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let size = param(a,"size",1,0.0); let pb = param(a,"prob",2,0.5);
    map_x(a, move |q| {
        if q <= 0.0 { return 0.0; }
        if q >= 1.0 { return size; }
        let (mut cum, mut k) = (0.0, 0.0);
        while k <= size { cum += ln_binom(size, k, pb).exp(); if cum >= q { return k; } k += 1.0; }
        size
    })
}
pub fn bi_qpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let lam = param(a,"lambda",1,1.0);
    map_x(a, move |q| {
        if q <= 0.0 { return 0.0; }
        let (mut cum, mut k) = (0.0, 0.0);
        while k < 1.0e7 { cum += (xlogy(k, lam) - lam - ln_gamma(k + 1.0)).exp(); if cum >= q { return k; } k += 1.0; }
        k
    })
}

/// `density(x)` — Gaussian kernel density estimate on a 512-point grid.
/// Returns a list with `$x` and `$y` (R's density object, simplified).
pub fn bi_density(a: &[EvalArg]) -> Result<RVal, R2Err> {
    use std::sync::Arc;
    let x: Vec<f64> = first(a).as_reals()?.into_iter().flatten().collect();
    if x.is_empty() {
        return Err(R2Err { msg: "density: need a non-empty numeric vector".into(), kind: ErrKind::Runtime });
    }
    let n = x.len() as f64;
    let (_, _, sxx) = crate::moments::centred1_dense(&x);
    let sd = (sxx / (n - 1.0).max(1.0)).sqrt();
    let h = (0.9 * sd * n.powf(-0.2)).max(1e-6); // Silverman's rule of thumb
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = (xmin - 3.0 * h, xmax + 3.0 * h);
    let ng = 512usize;
    let norm = 1.0 / (n * h * (2.0 * std::f64::consts::PI).sqrt());
    let mut gx = Vec::with_capacity(ng);
    let mut gy = Vec::with_capacity(ng);
    for i in 0..ng {
        let xi = lo + (hi - lo) * (i as f64) / ((ng - 1) as f64);
        let dens: f64 = x.iter().map(|&xj| { let z = (xi - xj) / h; (-0.5 * z * z).exp() }).sum::<f64>() * norm;
        gx.push(Some(xi));
        gy.push(Some(dens));
    }
    Ok(RVal::List(vec![
        (Some(Arc::from("x")), RVal::Numeric(gx.into(), Attrs::default())),
        (Some(Arc::from("y")), RVal::Numeric(gy.into(), Attrs::default())),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn dnorm_at_zero_is_one_over_sqrt_2pi() {
        let r = bi_dnorm(&[evarg(nums(&[0.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got = v[0].unwrap();
                let want = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
                assert!((got - want).abs() < 1e-12, "got {} want {}", got, want);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn pnorm_zero_is_half() {
        let r = bi_pnorm(&[evarg(nums(&[0.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert!((v[0].unwrap() - 0.5).abs() < 1e-7);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn qnorm_half_is_zero() {
        let r = bi_qnorm(&[evarg(nums(&[0.5]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                assert!(v[0].unwrap().abs() < 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn qnorm_matches_r_as241() {
        // Wichura AS241 is accurate to ~1e-15; compare to R's qnorm.
        let cases = [
            (0.975, 1.959963984540054),
            (0.025, -1.959963984540054),
            (0.99, 2.3263478740408408),
            (0.999, 3.0902323061678132),
            (0.9999999, 5.199337582187471), // deep-tail region (r > 5)
        ];
        for (p, want) in cases {
            let got = qnorm_approx(p);
            assert!((got - want).abs() < 1e-9, "qnorm({}) = {}, want {}", p, got, want);
        }
        assert_eq!(qnorm_approx(0.5), 0.0);
    }

    #[test]
    fn pnorm_qnorm_round_trip() {
        // Cody's erf makes pnorm full-precision, so the round trip is now
        // tight to ~1e-14 (was ~1e-7 with the old polynomial).
        for p in [0.1, 0.5, 0.975, 0.999, 1e-6, 0.999999] {
            let q = qnorm_approx(p);
            let back = phi(q);
            assert!((back - p).abs() < 1e-13, "round-trip p={} -> {}", p, back);
        }
    }

    #[test]
    fn pnorm_matches_r_to_full_precision() {
        // Reference values from R 4.5.3 pnorm(); Cody's algorithm must
        // agree to ~1e-14 (the old A&S polynomial was only ~1e-7).
        let cases = [
            (0.0_f64,  0.5_f64),
            (1.96,     0.97500210485177956),
            (-3.0,     0.0013498980316300954),
            (2.5,      0.99379033467422384),
            (-1.0,     0.15865525393145705),
            (5.0,      0.99999971334842808),
        ];
        for (x, want) in cases {
            let got = phi(x);
            // Full double precision now (direct z-score, R's pnorm coeffs):
            // relative error at/under 1e-15.
            let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            assert!(rel < 1e-15, "pnorm({}) = {:.17e} want {:.17e} (rel {:e})", x, got, want, rel);
        }
    }

    #[test]
    fn pnorm_extreme_tail_full_relative_precision() {
        // The tail that was ~2% wrong via erf(x/√2): direct z-score with
        // exp(-32) exact at x=8. R: pnorm(-8) = 6.2209605742717405e-16.
        let got = phi(-8.0);
        let want = 6.2209605742717405e-16;
        let rel = (got - want).abs() / want;
        assert!(rel < 1e-12, "pnorm(-8) = {:e} want {:e} (rel {:e})", got, want, rel);
        // Upper tail directly avoids 1-Φ cancellation.
        let up = phi_upper(8.0);
        assert!((up - want).abs() / want < 1e-12, "phi_upper(8) rel off: {:e}", up);
    }

    #[test]
    fn erf_erfc_identities() {
        assert!((erf(0.0)).abs() < 1e-16);            // erf(0) = 0 exactly
        assert!((erfc(0.0) - 1.0).abs() < 1e-16);     // erfc(0) = 1
        for x in [-2.0, -0.3, 0.5, 1.7, 3.0] {
            assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14, "erf+erfc != 1 at {}", x);
        }
    }
}
