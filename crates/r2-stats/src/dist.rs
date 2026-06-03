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

use r2_types::{Attrs, EvalArg, R2Err, RVal, Real};

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
        let num = (((((((2509.0809287301226727_f64 * r + 33430.575583588128105) * r
            + 67265.770927008700853) * r + 45921.953931549871457) * r
            + 13731.693765509461125) * r + 1971.5909503065514427) * r
            + 133.14166789178437745) * r + 3.387132872796366608);
        let den = (((((((5226.495278852854561_f64 * r + 28729.085735721942674) * r
            + 39307.89580009271061) * r + 21213.794301586595867) * r
            + 5394.1960214247511077) * r + 687.1870074920579083) * r
            + 42.313330701600911252) * r + 1.0);
        return q * num / den;
    }

    // Tail regions.
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        let num = (((((((7.7454501427834140764e-4_f64 * r + 0.0227238449892691845833) * r
            + 0.24178072517745061177) * r + 1.27045825245236838258) * r
            + 3.64784832476320460504) * r + 5.7694972214606914055) * r
            + 4.6303378461565452959) * r + 1.42343711074968357734);
        let den = (((((((1.05075007164441684324e-9_f64 * r + 5.475938084995344946e-4) * r
            + 0.0151986665636164571966) * r + 0.14810397642748007459) * r
            + 0.68976733498510000455) * r + 1.6763848301838038494) * r
            + 2.05319162663775882187) * r + 1.0);
        num / den
    } else {
        let r = r - 5.0;
        let num = (((((((2.01033439929228813265e-7_f64 * r + 2.71155556874348757815e-5) * r
            + 0.0012426609473880784386) * r + 0.026532189526576123093) * r
            + 0.29656057182850489123) * r + 1.7848265399172913358) * r
            + 5.4637849111641143699) * r + 6.6579046435011037772);
        let den = (((((((2.04426310338993978564e-15_f64 * r + 1.4215117583164458887e-7) * r
            + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
            + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
            + 0.59983220655588793769) * r + 1.0);
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
