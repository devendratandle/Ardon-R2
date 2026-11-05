//! Random variates for the families in `crate::betagamma` and `crate::families`,
//! as R's `r` functions: `n` (or `length(n)` when it is a vector), every
//! parameter recycled, and NaN for invalid parameters. R's streams differ by
//! design; the distribution is what is promised, and a test holds each to its
//! exact CDF.

use r2_types::{Attrs, ErrKind, EvalArg, Integer, R2Err, RVal, Real};

use crate::betagamma::missing;
use crate::dist::{first, param_vec};
use crate::rng::{check_alloc, next_random};

/// A uniform variate strictly inside (0, 1).
fn u01() -> f64 { next_random().clamp(1e-300, 1.0 - f64::EPSILON / 2.0) }

/// A standard normal variate (Box-Muller).
pub(crate) fn rnorm1() -> f64 {
    let (u1, u2) = (next_random().max(1e-15), next_random());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `ln G` for `G ~ Gamma(shape, 1)`: Marsaglia and Tsang's squeeze method, with
/// `G(a) = G(a + 1) U^(1/a)` below shape 1. On the log scale because a small
/// shape's variates underflow, and `rbeta` needs their ratio all the same.
fn ln_rgamma(shape: f64) -> f64 {
    if shape < 1.0 { return ln_rgamma(shape + 1.0) + u01().ln() / shape; }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = rnorm1();
        let v = 1.0 + c * z;
        if v <= 0.0 { continue; }
        let v3 = v * v * v;
        if u01().ln() < 0.5 * z * z + d - d * v3 + d * v3.ln() { return d.ln() + v3.ln(); }
    }
}

/// A Poisson variate. Knuth's product below mean 30, which keeps a seeded
/// script's small-mean draws as they were. His `exp(-mean)` underflows past
/// about 700, so large means use Hormann's PTRS transformed rejection.
pub(crate) fn rpois1(mean: f64) -> f64 {
    if mean <= 0.0 { return 0.0; }
    if mean < 30.0 {
        let l = (-mean).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            k += 1.0;
            p *= next_random();
            if p <= l { return k - 1.0; }
        }
    }
    let b = 0.931 + 2.53 * mean.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    let ln_mean = mean.ln();
    loop {
        let u = next_random() - 0.5;
        let v = u01();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mean + 0.43).floor();
        if us >= 0.07 && v <= vr { return k; }
        if k < 0.0 || (us < 0.013 && v > us) { continue; }
        if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln() <= -mean + k * ln_mean - crate::htest::ln_gamma(k + 1.0) {
            return k;
        }
    }
}

/// How many variates: `n`, or the length of `n` when it is a vector, as R.
fn count(a: &[EvalArg], f: &str) -> Result<usize, R2Err> {
    let v = param_vec(a, "n", 0)?.ok_or_else(|| missing(f, "n"))?;
    if v.len() > 1 { return Ok(v.len()); }
    match v.first().copied().flatten() {
        Some(x) if x >= 0.0 => Ok(x as usize),
        _ => Err(R2Err { msg: format!("{f}: invalid arguments"), kind: ErrKind::Runtime }),
    }
}

/// The parameter at position `i + 1` for each `(name, default)`. A parameter with
/// no default is required.
fn gen_params(a: &[EvalArg], f: &str, spec: &[(&str, Option<f64>)]) -> Result<Vec<Vec<Real>>, R2Err> {
    let mut out = Vec::with_capacity(spec.len());
    for (i, (name, d)) in spec.iter().enumerate() {
        let v = match (param_vec(a, name, i + 1)?, d) {
            (Some(v), _) => v,
            (None, Some(d)) => vec![Some(*d)],
            (None, None) => return Err(missing(f, name)),
        };
        if v.is_empty() { return Err(R2Err { msg: format!("{f}: invalid arguments"), kind: ErrKind::Runtime }); }
        out.push(v);
    }
    Ok(out)
}

/// Draw `n` values, every parameter recycled; NA in a parameter gives NaN.
fn draw(n: usize, params: &[Vec<Real>], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut p = vec![0.0; params.len()];
    (0..n).map(|i| {
        for (slot, v) in p.iter_mut().zip(params) { *slot = v[i % v.len()].unwrap_or(f64::NAN); }
        if p.iter().any(|x| x.is_nan()) { f64::NAN } else { f(&p) }
    }).collect()
}

fn numeric(v: Vec<f64>) -> RVal { RVal::Numeric(r2_types::Reals::from_dense_f64(v), Attrs::default()) }

/// A count distribution's result: integers, NA where a parameter was invalid,
/// as R's `rgeom`, `rnbinom`, `rhyper` and `rpois` return.
fn integers(v: Vec<f64>) -> RVal {
    let out: Vec<Integer> = v.iter().map(|&x| if x.is_nan() || x.abs() > i32::MAX as f64 { None } else { Some(x as i32) }).collect();
    RVal::Integer(out.into(), Attrs::default())
}

/// Scale from `rate` or `scale`; both given must agree.
fn gamma_scale(a: &[EvalArg], f: &str) -> Result<Vec<Real>, R2Err> {
    let rate = param_vec(a, "rate", 2)?;
    let scale = param_vec(a, "scale", 3)?;
    Ok(match (rate, scale) {
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
    })
}

/// `rnorm(n, mean = 0, sd = 1)`, recycled. These five once kept only the first
/// of a vector parameter, silently; a single value draws exactly as it did.
pub fn bi_rnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rnorm")?;
    let params = gen_params(a, "rnorm", &[("mean", Some(0.0)), ("sd", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[1] < 0.0 { f64::NAN } else { p[0] + p[1] * rnorm1() })))
}

/// `runif(n, min = 0, max = 1)`, recycled.
pub fn bi_runif(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "runif")?;
    let params = gen_params(a, "runif", &[("min", Some(0.0)), ("max", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[1] < p[0] { f64::NAN } else { p[0] + (p[1] - p[0]) * next_random() })))
}

/// `rexp(n, rate = 1)`, recycled.
pub fn bi_rexp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rexp")?;
    let params = gen_params(a, "rexp", &[("rate", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[0] <= 0.0 { f64::NAN } else { -next_random().max(1e-15).ln() / p[0] })))
}

/// `rbinom(n, size, prob)`, recycled: a Bernoulli sum for small sizes, and past
/// that a beta recursion (Devroye), which halves the problem each step instead
/// of spending a draw per trial.
pub fn bi_rbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rbinom")?;
    let params = gen_params(a, "rbinom", &[("size", None), ("prob", None)])?;
    check_alloc(n, 4)?;
    Ok(integers(draw(n, &params, |p| {
        let (size, pb) = (p[0], p[1]);
        if size < 0.0 || !(0.0..=1.0).contains(&pb) { return f64::NAN; }
        let (mut left, mut q, mut count) = (size.round(), pb, 0.0);
        while left > 1000.0 {
            let k = (left / 2.0).floor() + 1.0;
            let b = 1.0 / (1.0 + (ln_rgamma(left + 1.0 - k) - ln_rgamma(k)).exp());
            if b <= q {
                count += k;
                left -= k;
                q = (q - b) / (1.0 - b);
            } else {
                left = k - 1.0;
                q /= b;
            }
        }
        for _ in 0..left as u64 { if next_random() < q { count += 1.0; } }
        count
    })))
}

/// `rpois(n, lambda)`, recycled.
pub fn bi_rpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rpois")?;
    let params = gen_params(a, "rpois", &[("lambda", None)])?;
    check_alloc(n, 4)?;
    Ok(integers(draw(n, &params, |p| if p[0] < 0.0 { f64::NAN } else { rpois1(p[0]) })))
}

/// `rgamma(n, shape, rate = 1, scale = 1/rate)`.
pub fn bi_rgamma(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rgamma")?;
    let shape = param_vec(a, "shape", 1)?.ok_or_else(|| missing("rgamma", "shape"))?;
    let scale = gamma_scale(a, "rgamma")?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &[shape, scale], |p| {
        if p[0] < 0.0 || p[1] <= 0.0 { f64::NAN } else if p[0] == 0.0 { 0.0 } else { ln_rgamma(p[0]).exp() * p[1] }
    })))
}

/// `rbeta(n, shape1, shape2)`, as the ratio of two gammas taken on the log
/// scale, so small shapes (mass near 0 and 1) do not divide 0 by 0.
pub fn bi_rbeta(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rbeta")?;
    let params = gen_params(a, "rbeta", &[("shape1", None), ("shape2", None)])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| {
        if p[0] <= 0.0 || p[1] <= 0.0 { f64::NAN } else { 1.0 / (1.0 + (ln_rgamma(p[1]) - ln_rgamma(p[0])).exp()) }
    })))
}

/// `rchisq(n, df)`.
pub fn bi_rchisq(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rchisq")?;
    let params = gen_params(a, "rchisq", &[("df", None)])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[0] < 0.0 { f64::NAN } else if p[0] == 0.0 { 0.0 } else { 2.0 * ln_rgamma(p[0] / 2.0).exp() })))
}

/// `rt(n, df)`.
pub fn bi_rt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rt")?;
    let params = gen_params(a, "rt", &[("df", None)])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| {
        if p[0] <= 0.0 { return f64::NAN; }
        if !p[0].is_finite() { return rnorm1(); }
        rnorm1() / (2.0 * ln_rgamma(p[0] / 2.0).exp() / p[0]).sqrt()
    })))
}

/// `rf(n, df1, df2)`, R's F generator. The name is shared with R2's random
/// forest, `rf(y ~ x, data)`; the engine tells them apart by their arguments.
pub fn bi_rf_dist(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rf")?;
    let params = gen_params(a, "rf", &[("df1", None), ("df2", None)])?;
    check_alloc(n, 8)?;
    let chi = |d: f64| if d.is_finite() { 2.0 * ln_rgamma(d / 2.0).exp() / d } else { 1.0 };
    Ok(numeric(draw(n, &params, |p| if p[0] <= 0.0 || p[1] <= 0.0 { f64::NAN } else { chi(p[0]) / chi(p[1]) })))
}

/// Whether arguments to `rf` mean R's F generator: `df1` by name, or `n` and
/// `df1` as single numbers, which no random forest takes.
pub fn rf_is_f_generator(a: &[EvalArg]) -> bool {
    if a.iter().any(|x| x.name.as_deref() == Some("df1")) { return true; }
    let unnamed: Vec<&EvalArg> = a.iter().filter(|x| x.name.is_none()).collect();
    let scalar = |v: &RVal| matches!(v, RVal::Numeric(..) | RVal::Integer(..)) && v.as_reals().map(|r| r.len() == 1).unwrap_or(false);
    unnamed.len() >= 2 && scalar(&unnamed[0].value) && scalar(&unnamed[1].value)
}

/// `rweibull(n, shape, scale = 1)`.
pub fn bi_rweibull(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rweibull")?;
    let params = gen_params(a, "rweibull", &[("shape", None), ("scale", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[0] <= 0.0 || p[1] <= 0.0 { f64::NAN } else { p[1] * (-u01().ln()).powf(1.0 / p[0]) })))
}

/// `rlnorm(n, meanlog = 0, sdlog = 1)`.
pub fn bi_rlnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rlnorm")?;
    let params = gen_params(a, "rlnorm", &[("meanlog", Some(0.0)), ("sdlog", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| if p[1] < 0.0 { f64::NAN } else { (p[0] + p[1] * rnorm1()).exp() })))
}

/// `rcauchy(n, location = 0, scale = 1)`.
pub fn bi_rcauchy(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rcauchy")?;
    let params = gen_params(a, "rcauchy", &[("location", Some(0.0)), ("scale", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| {
        if p[1] < 0.0 { f64::NAN } else { p[0] + p[1] * (std::f64::consts::PI * u01()).tan() }
    })))
}

/// `rlogis(n, location = 0, scale = 1)`.
pub fn bi_rlogis(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rlogis")?;
    let params = gen_params(a, "rlogis", &[("location", Some(0.0)), ("scale", Some(1.0))])?;
    check_alloc(n, 8)?;
    Ok(numeric(draw(n, &params, |p| {
        if p[1] < 0.0 { return f64::NAN; }
        let u = u01();
        p[0] + p[1] * (u / (1.0 - u)).ln()
    })))
}

/// `rgeom(n, prob)`.
pub fn bi_rgeom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rgeom")?;
    let params = gen_params(a, "rgeom", &[("prob", None)])?;
    check_alloc(n, 4)?;
    Ok(integers(draw(n, &params, |p| {
        if p[0] <= 0.0 || p[0] > 1.0 { return f64::NAN; }
        if p[0] == 1.0 { return 0.0; }
        (u01().ln() / (-p[0]).ln_1p()).floor()
    })))
}

/// `rnbinom(n, size, prob, mu)`: Poisson counts with a gamma mean.
pub fn bi_rnbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = count(a, "rnbinom")?;
    let size = param_vec(a, "size", 1)?.ok_or_else(|| missing("rnbinom", "size"))?;
    let prob = param_vec(a, "prob", 2)?;
    let mu = crate::dist::named_vec(a, "mu")?;
    check_alloc(n, 4)?;
    match (prob, mu) {
        (Some(_), Some(_)) => Err(R2Err { msg: "rnbinom: 'prob' and 'mu' both specified".into(), kind: ErrKind::Runtime }),
        (Some(p), None) => Ok(integers(draw(n, &[size, p], |q| {
            let (s, pb) = (q[0], q[1]);
            if s < 0.0 || pb <= 0.0 || pb > 1.0 { return f64::NAN; }
            if pb == 1.0 || s == 0.0 { return 0.0; }
            rpois1(ln_rgamma(s).exp() * (1.0 - pb) / pb)
        }))),
        (None, Some(m)) => Ok(integers(draw(n, &[size, m], |q| {
            let (s, mean) = (q[0], q[1]);
            if s < 0.0 || mean < 0.0 { return f64::NAN; }
            if mean == 0.0 || s == 0.0 { return 0.0; }
            rpois1(ln_rgamma(s).exp() * mean / s)
        }))),
        (None, None) => Err(missing("rnbinom", "prob")),
    }
}

/// `rhyper(nn, m, n, k)`, by inversion of a uniform draw.
pub fn bi_rhyper(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let nn = {
        let v = param_vec(a, "nn", 0)?.ok_or_else(|| missing("rhyper", "nn"))?;
        if v.len() > 1 { v.len() } else {
            match v.first().copied().flatten() {
                Some(x) if x >= 0.0 => x as usize,
                _ => return Err(R2Err { msg: "rhyper: invalid arguments".into(), kind: ErrKind::Runtime }),
            }
        }
    };
    let params = gen_params(a, "rhyper", &[("m", None), ("n", None), ("k", None)])?;
    check_alloc(nn, 4)?;
    let _ = first(a);
    Ok(integers(draw(nn, &params, |p| {
        let (m, n, k) = (p[0], p[1], p[2]);
        if m < 0.0 || n < 0.0 || k < 0.0 || k > m + n { return f64::NAN; }
        let u = u01();
        crate::families::qhyper_tails(u.ln(), (-u).ln_1p(), m.round(), n.round(), k.round())
    })))
}

#[cfg(test)]
mod tests {
    //! Each generator against its exact CDF, by Kolmogorov-Smirnov at a level
    //! no correct generator should reach by chance, and the discrete ones by
    //! their means.
    use super::*;
    use crate::betagamma::pbeta_log_tails;
    use crate::htest::{chi_sq_cdf, pgamma_log_tails, t_cdf};

    fn arg(v: f64) -> EvalArg { EvalArg { name: None, value: RVal::Numeric(vec![Some(v)].into(), Attrs::default()) } }
    fn args(v: &[f64]) -> Vec<EvalArg> { v.iter().map(|x| arg(*x)).collect() }
    fn reals(v: &RVal) -> Vec<f64> { v.as_reals().unwrap().iter().map(|x| x.unwrap()).collect() }

    const N: f64 = 50_000.0;

    fn ks(mut xs: Vec<f64>, cdf: impl Fn(f64) -> f64) -> f64 {
        xs.sort_by(f64::total_cmp);
        let n = xs.len() as f64;
        xs.iter().enumerate().map(|(i, &x)| {
            let f = cdf(x);
            (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
        }).fold(0.0, f64::max)
    }
    fn limit() -> f64 { 2.5 / N.sqrt() }

    #[test]
    fn continuous_generators_draw_from_their_distributions() {
        let v = reals(&bi_rgamma(&args(&[N, 2.5, 2.0])).unwrap());
        assert!(ks(v, |x| pgamma_log_tails(2.5, x * 2.0).0.exp()) < limit(), "rgamma");
        let v = reals(&bi_rgamma(&args(&[N, 0.1, 2.0])).unwrap());
        assert!(ks(v, |x| pgamma_log_tails(0.1, x * 2.0).0.exp()) < limit(), "rgamma small shape");
        for (a, b) in [(0.3, 0.3), (0.5, 3.0), (30.0, 4.0)] {
            let v = reals(&bi_rbeta(&args(&[N, a, b])).unwrap());
            assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
            assert!(ks(v, |x| pbeta_log_tails(x, 1.0 - x, a, b).0.exp()) < limit(), "rbeta {a} {b}");
        }
        for df in [0.5, 3.0, 40.0] {
            let v = reals(&bi_rchisq(&args(&[N, df])).unwrap());
            assert!(ks(v, |x| chi_sq_cdf(x, df)) < limit(), "rchisq {df}");
        }
        for df in [1.0, 4.5, 60.0] {
            let v = reals(&bi_rt(&args(&[N, df])).unwrap());
            assert!(ks(v, |x| t_cdf(x, df)) < limit(), "rt {df}");
        }
        let v = reals(&bi_rf_dist(&args(&[N, 5.0, 12.0])).unwrap());
        assert!(ks(v, |x| { let d = 5.0 * x + 12.0; pbeta_log_tails(5.0 * x / d, 12.0 / d, 2.5, 6.0).0.exp() }) < limit(), "rf");
        let v = reals(&bi_rweibull(&args(&[N, 1.7, 3.0])).unwrap());
        assert!(ks(v, |x| -(-(x / 3.0).powf(1.7)).exp_m1()) < limit(), "rweibull");
        let v = reals(&bi_rlnorm(&args(&[N, 0.5, 0.8])).unwrap());
        assert!(ks(v, |x| crate::dist::phi((x.ln() - 0.5) / 0.8)) < limit(), "rlnorm");
        let v = reals(&bi_rcauchy(&args(&[N, 1.0, 2.0])).unwrap());
        assert!(ks(v, |x| 0.5 + ((x - 1.0) / 2.0).atan() / std::f64::consts::PI) < limit(), "rcauchy");
        let v = reals(&bi_rlogis(&args(&[N, 1.0, 2.0])).unwrap());
        assert!(ks(v, |x| 1.0 / (1.0 + (-(x - 1.0) / 2.0).exp())) < limit(), "rlogis");
    }

    fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
    fn var(v: &[f64]) -> f64 { let m = mean(v); v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0) }

    /// Mean within 5 standard errors; variance within 10%.
    fn moments(v: &[f64], want_mean: f64, want_var: f64, what: &str) {
        let se = (want_var / v.len() as f64).sqrt();
        assert!((mean(v) - want_mean).abs() < 5.0 * se, "{what}: mean {} vs {want_mean}", mean(v));
        assert!((var(v) / want_var - 1.0).abs() < 0.1, "{what}: variance {} vs {want_var}", var(v));
    }

    #[test]
    fn discrete_generators_have_the_right_moments() {
        let v = reals(&bi_rgeom(&args(&[N, 0.2])).unwrap());
        moments(&v, 4.0, 20.0, "rgeom");
        let v = reals(&bi_rnbinom(&args(&[N, 3.0, 0.4])).unwrap());
        moments(&v, 4.5, 11.25, "rnbinom");
        let v = reals(&bi_rhyper(&args(&[N, 30.0, 20.0, 10.0])).unwrap());
        moments(&v, 6.0, 10.0 * 0.6 * 0.4 * 40.0 / 49.0, "rhyper");
    }

    /// Knuth's method underflows past mean ~700 and stops near there.
    #[test]
    fn large_poisson_means_are_not_capped() {
        for m in [29.0, 31.0, 1000.0, 1e6] {
            let v: Vec<f64> = (0..50_000).map(|_| rpois1(m)).collect();
            moments(&v, m, m, "rpois1");
        }
    }

    #[test]
    fn large_binomials_have_the_right_moments() {
        for (size, p) in [(10.0, 0.3), (1001.0, 0.5), (1e6, 0.1), (1e8, 0.99)] {
            let v = reals(&bi_rbinom(&args(&[20_000.0, size, p])).unwrap());
            moments(&v, size * p, size * p * (1.0 - p), "rbinom");
        }
    }

    /// A vector parameter must reach every draw, not only the first.
    #[test]
    fn the_older_generators_recycle() {
        let two = |x: f64, y: f64| EvalArg { name: None, value: RVal::Numeric(vec![Some(x), Some(y)].into(), Attrs::default()) };
        let v = reals(&bi_rnorm(&[arg(6.0), two(0.0, 1000.0)]).unwrap());
        assert!(v.iter().step_by(2).all(|x| x.abs() < 100.0) && v.iter().skip(1).step_by(2).all(|x| *x > 900.0), "{v:?}");
        let v = reals(&bi_runif(&[arg(6.0), two(0.0, 1000.0), two(1.0, 2000.0)]).unwrap());
        assert!(v.iter().step_by(2).all(|x| *x <= 1.0) && v.iter().skip(1).step_by(2).all(|x| *x >= 1000.0), "{v:?}");
        let v = reals(&bi_rpois(&[arg(6.0), two(0.0, 1e5)]).unwrap());
        assert!(v.iter().step_by(2).all(|x| *x == 0.0) && v.iter().skip(1).step_by(2).all(|x| *x > 9e4), "{v:?}");
        let v = reals(&bi_rbinom(&[arg(6.0), two(0.0, 1e6), arg(0.5)]).unwrap());
        assert!(v.iter().step_by(2).all(|x| *x == 0.0) && v.iter().skip(1).step_by(2).all(|x| *x > 4e5), "{v:?}");
        let v = reals(&bi_rexp(&[arg(4.0), two(1e6, 1e-6)]).unwrap());
        assert!(v[0] < 1.0 && v[1] > 1.0, "{v:?}");
    }

    #[test]
    fn parameters_recycle_and_invalid_ones_give_nan() {
        let v = reals(&bi_rgamma(&[arg(6.0), EvalArg { name: None, value: RVal::Numeric(vec![Some(1.0), Some(1000.0)].into(), Attrs::default()) }]).unwrap());
        assert!(v[1] > 100.0 && v[3] > 100.0 && v[5] > 100.0, "{v:?}");
        assert!(reals(&bi_rbeta(&args(&[1.0, -1.0, 2.0])).unwrap())[0].is_nan());
        let vec_n = EvalArg { name: None, value: RVal::Numeric(vec![Some(9.0), Some(9.0), Some(9.0)].into(), Attrs::default()) };
        assert_eq!(reals(&bi_rt(&[vec_n, arg(5.0)]).unwrap()).len(), 3, "a vector n means its length");
    }

    #[test]
    fn rf_tells_the_generator_from_the_forest() {
        assert!(rf_is_f_generator(&args(&[10.0, 3.0, 4.0])));
        let vector = EvalArg { name: None, value: RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), Attrs::default()) };
        assert!(!rf_is_f_generator(&[vector.clone(), vector]));
    }
}
