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

/// Abramowitz & Stegun 7.1.26 polynomial approximation. Max error ≈ 1.5e-7.
pub fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

/// Standard normal CDF: P(Z ≤ x).
#[inline]
pub fn phi(x: f64) -> f64 { 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2)) }

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
    let result: Vec<Real> = x.iter().map(|v| v.map(|x| {
        let z = (x - mean) / sd;
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    })).collect();
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
        // qnorm then pnorm should return the input to ~1e-7 (pnorm is the
        // limiting factor at ~1e-7; qnorm itself is ~1e-15).
        for p in [0.1, 0.5, 0.975, 0.999] {
            let q = qnorm_approx(p);
            let back = phi(q);
            assert!((back - p).abs() < 1e-7, "round-trip p={} -> {}", p, back);
        }
    }
}
