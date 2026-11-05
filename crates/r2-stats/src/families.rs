//! The other standard distributions, as R's `d/p/q` functions: uniform,
//! Weibull, lognormal, Cauchy, logistic, geometric, negative binomial,
//! hypergeometric, the F density and the F quantile. Every parameter recycles,
//! and `lower.tail`, `log.p` and `log` are honoured. Generators are in
//! `crate::rfamilies`; the comparison with R is
//! tests/differential/cases/families.R.

use r2_types::{ErrKind, EvalArg, R2Err, RVal, Real};

use crate::betagamma::{ln1mexp, ln_dbinom_raw, map_recycled, missing, pbeta_log_tails, qbeta_xy};
use crate::dist::{flag, param_vec, phi_upper, pnorm_log_tails, qnorm_approx};
use crate::htest::chi_sq_quantile;

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_741_78;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_677_94;
const PI: f64 = std::f64::consts::PI;

/// A parameter by name or position: its default if absent, an error if it has
/// none.
pub(crate) fn pv(a: &[EvalArg], f: &str, name: &str, pos: usize, dflt: Option<f64>) -> Result<Vec<Real>, R2Err> {
    match (param_vec(a, name, pos)?, dflt) {
        (Some(v), _) => Ok(v),
        (None, Some(d)) => Ok(vec![Some(d)]),
        (None, None) => Err(missing(f, name)),
    }
}

/// The chosen tail from the pair of log tails.
#[inline]
pub(crate) fn tail(lp: f64, lq: f64, lower: bool, logp: bool) -> f64 {
    let l = if lower { lp } else { lq };
    if logp { l } else { l.exp() }
}

/// A probability argument as the log of its lower and upper tail; `None` when
/// it is not a probability.
pub(crate) fn p_in(p: f64, lower: bool, logp: bool) -> Option<(f64, f64)> {
    if p.is_nan() { return None; }
    let l = if logp { if p > 0.0 { return None; } p }
        else { if !(0.0..=1.0).contains(&p) { return None; } p.ln() };
    let other = ln1mexp(l);
    Some(if lower { (l, other) } else { (other, l) })
}

/// A density value from its log.
#[inline]
pub(crate) fn d_out(l: f64, log: bool) -> f64 { if log { l } else { l.exp() } }

/// Whether `x` is a whole number, within R's tolerance for one.
pub(crate) fn is_whole(x: f64) -> bool { (x - x.round()).abs() <= 1e-9 * x.abs().max(1.0) }

/// A non-central request is an error, not a silently central answer.
pub(crate) fn reject_ncp(a: &[EvalArg], f: &str, pos: usize, family: &str) -> Result<(), R2Err> {
    if let Some(n) = param_vec(a, "ncp", pos)? {
        if n.iter().any(|r| r.map(|x| x != 0.0).unwrap_or(false)) {
            return Err(R2Err { msg: format!("{f}: the non-central {family} (ncp != 0) is not implemented"), kind: ErrKind::Runtime });
        }
    }
    Ok(())
}

// ── uniform ─────────────────────────────────────────────────────────────

fn unif_params(a: &[EvalArg]) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, "", "min", 1, Some(0.0))?, pv(a, "", "max", 2, Some(1.0))?])
}

/// `dunif(x, min = 0, max = 1, log = FALSE)`.
pub fn bi_dunif(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &unif_params(a)?, move |x, p| {
        let (lo, hi) = (p[0], p[1]);
        if x.is_nan() || lo.is_nan() || hi.is_nan() || hi <= lo { return f64::NAN; }
        if x >= lo && x <= hi { d_out(-(hi - lo).ln(), log) } else { d_out(f64::NEG_INFINITY, log) }
    })
}

/// `punif(q, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_punif(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &unif_params(a)?, move |q, p| {
        let (lo, hi) = (p[0], p[1]);
        if q.is_nan() || lo.is_nan() || hi.is_nan() || hi < lo || !lo.is_finite() || !hi.is_finite() { return f64::NAN; }
        if q >= hi { return tail(0.0, f64::NEG_INFINITY, lower, logp); }
        if q <= lo { return tail(f64::NEG_INFINITY, 0.0, lower, logp); }
        let v = if lower { (q - lo) / (hi - lo) } else { (hi - q) / (hi - lo) };
        if logp { v.ln() } else { v }
    })
}

/// `qunif(p, min = 0, max = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qunif(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &unif_params(a)?, move |pr, p| {
        let (lo, hi) = (p[0], p[1]);
        if lo.is_nan() || hi.is_nan() || hi < lo { return f64::NAN; }
        let Some((l, _)) = p_in(pr, lower, logp) else { return f64::NAN };
        if hi == lo { return lo; }
        lo + l.exp() * (hi - lo)
    })
}

// ── Weibull ─────────────────────────────────────────────────────────────

fn weibull_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, f, "shape", 1, None)?, pv(a, f, "scale", 2, Some(1.0))?])
}

/// `dweibull(x, shape, scale = 1, log = FALSE)`.
pub fn bi_dweibull(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &weibull_params(a, "dweibull")?, move |x, p| {
        let (k, s) = (p[0], p[1]);
        if x.is_nan() || k.is_nan() || s.is_nan() || k <= 0.0 || s <= 0.0 { return f64::NAN; }
        if x < 0.0 || !x.is_finite() { return d_out(f64::NEG_INFINITY, log); }
        if x == 0.0 && k < 1.0 { return f64::INFINITY; }
        let t1 = (x / s).powf(k - 1.0);
        let t2 = t1 * (x / s);
        if log { -t2 + (k * t1 / s).ln() } else { k * t1 * (-t2).exp() / s }
    })
}

/// `pweibull(q, shape, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pweibull(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &weibull_params(a, "pweibull")?, move |q, p| {
        let (k, s) = (p[0], p[1]);
        if q.is_nan() || k.is_nan() || s.is_nan() || k <= 0.0 || s <= 0.0 { return f64::NAN; }
        if q <= 0.0 { return tail(f64::NEG_INFINITY, 0.0, lower, logp); }
        let lq = -(q / s).powf(k);
        tail(ln1mexp(lq), lq, lower, logp)
    })
}

/// `qweibull(p, shape, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qweibull(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &weibull_params(a, "qweibull")?, move |pr, p| {
        let (k, s) = (p[0], p[1]);
        if k.is_nan() || s.is_nan() || k <= 0.0 || s <= 0.0 { return f64::NAN; }
        let Some((_, lq)) = p_in(pr, lower, logp) else { return f64::NAN };
        s * (-lq).powf(1.0 / k)
    })
}

// ── lognormal ───────────────────────────────────────────────────────────

fn lnorm_params(a: &[EvalArg]) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, "", "meanlog", 1, Some(0.0))?, pv(a, "", "sdlog", 2, Some(1.0))?])
}

/// `dlnorm(x, meanlog = 0, sdlog = 1, log = FALSE)`.
pub fn bi_dlnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &lnorm_params(a)?, move |x, p| {
        let (m, s) = (p[0], p[1]);
        if x.is_nan() || m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        if x <= 0.0 { return d_out(f64::NEG_INFINITY, log); }
        let y = (x.ln() - m) / s;
        if log { -(LN_SQRT_2PI + 0.5 * y * y + (x * s).ln()) } else { INV_SQRT_2PI * (-0.5 * y * y).exp() / (x * s) }
    })
}

/// `plnorm(q, meanlog = 0, sdlog = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_plnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &lnorm_params(a)?, move |q, p| {
        let (m, s) = (p[0], p[1]);
        if q.is_nan() || m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        if q <= 0.0 { return tail(f64::NEG_INFINITY, 0.0, lower, logp); }
        let z = (q.ln() - m) / s;
        if logp {
            let (lp, lq) = pnorm_log_tails(z);
            if lower { lp } else { lq }
        } else if lower { phi_upper(-z) } else { phi_upper(z) }
    })
}

/// `qlnorm(p, meanlog = 0, sdlog = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qlnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &lnorm_params(a)?, move |pr, p| {
        let (m, s) = (p[0], p[1]);
        if m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        let Some((l, u)) = p_in(pr, lower, logp) else { return f64::NAN };
        // the normal is symmetric, so the small tail decides the quantile
        let z = if l <= u { qnorm_approx(l.exp()) } else { -qnorm_approx(u.exp()) };
        (m + s * z).exp()
    })
}

// ── Cauchy ──────────────────────────────────────────────────────────────

fn loc_scale(a: &[EvalArg]) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, "", "location", 1, Some(0.0))?, pv(a, "", "scale", 2, Some(1.0))?])
}

/// `dcauchy(x, location = 0, scale = 1, log = FALSE)`.
pub fn bi_dcauchy(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &loc_scale(a)?, move |x, p| {
        let (l, s) = (p[0], p[1]);
        if x.is_nan() || l.is_nan() || s.is_nan() || s <= 0.0 { return f64::NAN; }
        let y = (x - l) / s;
        if log { -(PI * s * (1.0 + y * y)).ln() } else { 1.0 / (PI * s * (1.0 + y * y)) }
    })
}

/// `pcauchy(q, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
/// Past one scale from the centre it goes through `atan(1/x)`, so the small
/// tail keeps its digits.
pub fn bi_pcauchy(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &loc_scale(a)?, move |q, p| {
        let (l, s) = (p[0], p[1]);
        if q.is_nan() || l.is_nan() || s.is_nan() || s <= 0.0 { return f64::NAN; }
        let mut x = (q - l) / s;
        if x.is_nan() { return f64::NAN; }
        if !x.is_finite() {
            return if x < 0.0 { tail(f64::NEG_INFINITY, 0.0, lower, logp) } else { tail(0.0, f64::NEG_INFINITY, lower, logp) };
        }
        if !lower { x = -x; }
        if x.abs() > 1.0 {
            let y = (1.0 / x).atan() / PI;
            if x > 0.0 { if logp { (-y).ln_1p() } else { 0.5 - y + 0.5 } }
            else if logp { (-y).ln() } else { -y }
        } else {
            let v = 0.5 + x.atan() / PI;
            if logp { v.ln() } else { v }
        }
    })
}

/// `qcauchy(p, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qcauchy(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &loc_scale(a)?, move |pr, p| {
        let (l, s) = (p[0], p[1]);
        if l.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        let Some((lo, up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if s == 0.0 { return l; }
        // the smaller tail, with the sign of the side it is on
        let (t, sign) = if lo <= up { (lo.exp(), -1.0) } else { (up.exp(), 1.0) };
        if t == 0.5 { return l; }
        if t == 0.0 { return sign * f64::INFINITY; }
        l + sign * s / (PI * t).tan()
    })
}

// ── logistic ────────────────────────────────────────────────────────────

/// `ln(1 + e^x)` without overflow or cancellation.
fn log1pexp(x: f64) -> f64 {
    if x <= 18.0 { x.exp().ln_1p() } else if x > 33.3 { x } else { x + (-x).exp() }
}

/// `dlogis(x, location = 0, scale = 1, log = FALSE)`.
pub fn bi_dlogis(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &loc_scale(a)?, move |x, p| {
        let (l, s) = (p[0], p[1]);
        if x.is_nan() || l.is_nan() || s.is_nan() || s <= 0.0 { return f64::NAN; }
        let y = ((x - l) / s).abs();
        let e = (-y).exp();
        let f = 1.0 + e;
        if log { -(y + (s * f * f).ln()) } else { e / (s * f * f) }
    })
}

/// `plogis(q, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_plogis(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &loc_scale(a)?, move |q, p| {
        let (l, s) = (p[0], p[1]);
        if q.is_nan() || l.is_nan() || s.is_nan() || s <= 0.0 { return f64::NAN; }
        let x = (q - l) / s;
        if x.is_nan() { return f64::NAN; }
        let t = if lower { -x } else { x };
        if logp { -log1pexp(t) } else { 1.0 / (1.0 + t.exp()) }
    })
}

/// `qlogis(p, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qlogis(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &loc_scale(a)?, move |pr, p| {
        let (l, s) = (p[0], p[1]);
        if l.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        let Some((lo, up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if s == 0.0 { return l; }
        l + s * (lo - up)
    })
}

// ── binomial pieces, for the discrete families and the F density ────────

/// `ln C(n, x) p^x q^(n-x)` over `x` and `rest = n - x`, either of which may be 0.
pub(crate) fn ln_db(x: f64, rest: f64, p: f64, q: f64) -> f64 {
    let n = x + rest;
    if x == 0.0 {
        if rest == 0.0 { return 0.0; }
        return if q == 0.0 { f64::NEG_INFINITY } else if p < 0.1 { rest * (-p).ln_1p() } else { rest * q.ln() };
    }
    if rest == 0.0 {
        return if p == 0.0 { f64::NEG_INFINITY } else if q < 0.1 { n * (-q).ln_1p() } else { n * p.ln() };
    }
    if p == 0.0 || q == 0.0 { return f64::NEG_INFINITY; }
    ln_dbinom_raw(x, rest, p, q)
}

/// The smallest whole `x` in `[lo, hi]` for which `ok(x)`, which is false below
/// the answer and true from it on: gallop from `start`, then bisect.
pub(crate) fn discrete_q(start: f64, lo: f64, hi: f64, ok: impl Fn(f64) -> bool) -> f64 {
    let mut x = start.round().clamp(lo, hi);
    let mut step = 1.0;
    if ok(x) {
        // down to a value that fails
        loop {
            if x <= lo { return lo; }
            let below = (x - step).max(lo);
            if !ok(below) { return bisect_whole(below, x, &ok); }
            x = below;
            step *= 2.0;
        }
    }
    loop {
        if x >= hi { return hi; }
        let above = (x + step).min(hi);
        if ok(above) { return bisect_whole(x, above, &ok); }
        x = above;
        step *= 2.0;
    }
}

/// With `ok(fail)` false and `ok(pass)` true, the smallest whole value passing.
fn bisect_whole(mut fail: f64, mut pass: f64, ok: &impl Fn(f64) -> bool) -> f64 {
    while pass - fail > 1.0 {
        let mid = ((fail + pass) / 2.0).floor();
        if ok(mid) { pass = mid; } else { fail = mid; }
    }
    pass
}

/// R's fuzz on a discrete quantile: a cumulative probability within this many
/// epsilons of the target counts as reaching it.
const FUZZ: f64 = 64.0 * f64::EPSILON;

/// Whether `(lp, lq)` at some `x` has reached a target given by its tails.
#[inline]
pub(crate) fn reached(lp: f64, lq: f64, t_lo: f64, t_up: f64) -> bool {
    if t_lo <= t_up { lp >= t_lo - FUZZ } else { lq <= t_up + FUZZ }
}

// ── geometric ───────────────────────────────────────────────────────────

fn prob_ok(p: f64) -> bool { p > 0.0 && p <= 1.0 }

/// `dgeom(x, prob, log = FALSE)`.
pub fn bi_dgeom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let prob = pv(a, "dgeom", "prob", 1, None)?;
    let log = flag(a, "log", 2, false);
    map_recycled(a, &[prob], move |x, p| {
        let pb = p[0];
        if x.is_nan() || pb.is_nan() || !prob_ok(pb) { return f64::NAN; }
        if x < 0.0 || !x.is_finite() || !is_whole(x) { return d_out(f64::NEG_INFINITY, log); }
        d_out(pb.ln() + ln_db(0.0, x.round(), pb, 1.0 - pb), log)
    })
}

fn pgeom_tails(q: f64, pb: f64) -> (f64, f64) {
    if q < 0.0 { return (f64::NEG_INFINITY, 0.0); }
    if !q.is_finite() || pb == 1.0 { return (0.0, f64::NEG_INFINITY); }
    let lq = (-pb).ln_1p() * ((q + 1e-7).floor() + 1.0);
    (ln1mexp(lq), lq)
}

/// `pgeom(q, prob, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pgeom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let prob = pv(a, "pgeom", "prob", 1, None)?;
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &[prob], move |q, p| {
        if q.is_nan() || p[0].is_nan() || !prob_ok(p[0]) { return f64::NAN; }
        let (lp, lq) = pgeom_tails(q, p[0]);
        tail(lp, lq, lower, logp)
    })
}

/// `qgeom(p, prob, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qgeom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let prob = pv(a, "qgeom", "prob", 1, None)?;
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &[prob], move |pr, p| {
        let pb = p[0];
        if pb.is_nan() || !prob_ok(pb) { return f64::NAN; }
        let Some((_, lq)) = p_in(pr, lower, logp) else { return f64::NAN };
        if lq == 0.0 { return 0.0; }
        if pb == 1.0 { return 0.0; }
        (lq / (-pb).ln_1p() - 1.0 - 1e-12).ceil().max(0.0)
    })
}

// ── negative binomial ───────────────────────────────────────────────────

/// `size` with `prob` and `1 - prob` as a pair, from `prob` or `mu`. From `mu`
/// the pair is `size / (size + mu)` and `mu / (size + mu)`, so a small `mu`
/// does not lose `1 - prob`.
fn nb_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    let size = pv(a, f, "size", 1, None)?;
    let prob = param_vec(a, "prob", 2)?;
    let mu = crate::dist::named_vec(a, "mu")?;
    match (prob, mu) {
        (Some(_), Some(_)) => Err(R2Err { msg: format!("{f}: 'prob' and 'mu' both specified"), kind: ErrKind::Runtime }),
        (Some(p), None) => {
            let q = p.iter().map(|v| v.map(|x| 1.0 - x)).collect();
            Ok(vec![size, p, q])
        }
        (None, Some(m)) => {
            // both of the pair need `size`, so the recycling happens here
            let n = size.len().max(m.len());
            if size.is_empty() || m.is_empty() { return Ok(vec![Vec::new(), Vec::new(), Vec::new()]); }
            let pair = |i: usize| -> Option<(f64, f64)> {
                let (s, mv) = (size[i % size.len()]?, m[i % m.len()]?);
                Some((s / (s + mv), mv / (s + mv)))
            };
            let p = (0..n).map(|i| pair(i).map(|t| t.0)).collect();
            let q = (0..n).map(|i| pair(i).map(|t| t.1)).collect();
            Ok(vec![(0..n).map(|i| size[i % size.len()]).collect(), p, q])
        }
        (None, None) => Err(missing(f, "prob")),
    }
}

/// `dnbinom(x, size, prob, mu, log = FALSE)`.
pub fn bi_dnbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = nb_params(a, "dnbinom")?;
    let log = flag(a, "log", 4, false);
    map_recycled(a, &params, move |x, p| {
        let (s, pb, q) = (p[0], p[1], p[2]);
        if x.is_nan() || s.is_nan() || pb.is_nan() || s < 0.0 || !prob_ok(pb) { return f64::NAN; }
        if x < 0.0 || !x.is_finite() || !is_whole(x) { return d_out(f64::NEG_INFINITY, log); }
        let x = x.round();
        if x == 0.0 && s == 0.0 { return d_out(0.0, log); }
        if s == 0.0 { return d_out(f64::NEG_INFINITY, log); }
        if x == 0.0 { return d_out(if q < 0.1 { s * (-q).ln_1p() } else { s * pb.ln() }, log); }
        d_out((s / (s + x)).ln() + ln_db(s, x, pb, q), log)
    })
}

fn nb_tails(k: f64, s: f64, pb: f64, q: f64) -> (f64, f64) {
    if k < 0.0 { return (f64::NEG_INFINITY, 0.0); }
    if !k.is_finite() || s == 0.0 { return (0.0, f64::NEG_INFINITY); }
    pbeta_log_tails(pb, q, s, (k + 1e-7).floor() + 1.0)
}

/// `pnbinom(q, size, prob, mu, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pnbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = nb_params(a, "pnbinom")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |k, p| {
        let (s, pb, q) = (p[0], p[1], p[2]);
        if k.is_nan() || s.is_nan() || pb.is_nan() || s < 0.0 || !prob_ok(pb) { return f64::NAN; }
        let (lp, lq) = nb_tails(k, s, pb, q);
        tail(lp, lq, lower, logp)
    })
}

/// `qnbinom(p, size, prob, mu, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qnbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = nb_params(a, "qnbinom")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |pr, p| {
        let (s, pb, q) = (p[0], p[1], p[2]);
        if s.is_nan() || pb.is_nan() || s < 0.0 || !prob_ok(pb) { return f64::NAN; }
        let Some((t_lo, t_up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if pb == 1.0 || s == 0.0 || t_lo == f64::NEG_INFINITY { return 0.0; }
        if t_up == f64::NEG_INFINITY { return f64::INFINITY; }
        let start = s * q / pb;
        discrete_q(start, 0.0, f64::INFINITY, |k| { let (lp, lq) = nb_tails(k, s, pb, q); reached(lp, lq, t_lo, t_up) })
    })
}

// ── hypergeometric ──────────────────────────────────────────────────────

fn hyper_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, f, "m", 1, None)?, pv(a, f, "n", 2, None)?, pv(a, f, "k", 3, None)?])
}

fn hyper_bad(m: f64, n: f64, k: f64) -> bool {
    m.is_nan() || n.is_nan() || k.is_nan() || m < 0.0 || n < 0.0 || k < 0.0
        || !is_whole(m) || !is_whole(n) || !is_whole(k) || k.round() > m.round() + n.round()
}

/// ln of the hypergeometric probability of `x`, as three binomial densities so
/// no huge factorials cancel.
fn ln_dhyper(x: f64, m: f64, n: f64, k: f64) -> f64 {
    if x < (k - n).max(0.0) || x > k.min(m) { return f64::NEG_INFINITY; }
    if k == 0.0 { return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY }; }
    let t = m + n;
    let (p, q) = (k / t, (t - k) / t);
    ln_db(x, m - x, p, q) + ln_db(k - x, n - (k - x), p, q) - ln_db(k, t - k, p, q)
}

/// Both tails at `x`: the side away from the mean is summed outward from `x` by
/// the term ratio, and the other is its complement.
fn hyper_tails(x: f64, m: f64, n: f64, k: f64) -> (f64, f64) {
    let (lo, hi) = ((k - n).max(0.0), k.min(m));
    if x < lo { return (f64::NEG_INFINITY, 0.0); }
    if x >= hi { return (0.0, f64::NEG_INFINITY); }
    let x = (x + 1e-7).floor();
    let mean = if m + n > 0.0 { k * m / (m + n) } else { 0.0 };
    if x < mean {
        let (mut sum, mut term, mut i) = (1.0f64, 1.0f64, x);
        while i > lo {
            term *= i * (n - k + i) / ((m - i + 1.0) * (k - i + 1.0));
            sum += term;
            if term < sum * 1e-18 { break; }
            i -= 1.0;
        }
        let lp = ln_dhyper(x, m, n, k) + sum.ln();
        (lp.min(0.0), ln1mexp(lp.min(0.0)))
    } else {
        let (mut sum, mut term, mut i) = (1.0f64, 1.0f64, x + 1.0);
        while i < hi {
            term *= (m - i) * (k - i) / ((i + 1.0) * (n - k + i + 1.0));
            sum += term;
            if term < sum * 1e-18 { break; }
            i += 1.0;
        }
        let lq = ln_dhyper(x + 1.0, m, n, k) + sum.ln();
        (ln1mexp(lq.min(0.0)), lq.min(0.0))
    }
}

/// `dhyper(x, m, n, k, log = FALSE)`.
pub fn bi_dhyper(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = hyper_params(a, "dhyper")?;
    let log = flag(a, "log", 4, false);
    map_recycled(a, &params, move |x, p| {
        let (m, n, k) = (p[0], p[1], p[2]);
        if x.is_nan() || hyper_bad(m, n, k) { return f64::NAN; }
        if x < 0.0 || !is_whole(x) { return d_out(f64::NEG_INFINITY, log); }
        d_out(ln_dhyper(x.round(), m.round(), n.round(), k.round()), log)
    })
}

/// `phyper(q, m, n, k, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_phyper(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = hyper_params(a, "phyper")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |q, p| {
        let (m, n, k) = (p[0], p[1], p[2]);
        if q.is_nan() || hyper_bad(m, n, k) { return f64::NAN; }
        let (lp, lq) = hyper_tails(q, m.round(), n.round(), k.round());
        tail(lp, lq, lower, logp)
    })
}

/// The quantile from a target's two log tails.
pub(crate) fn qhyper_tails(t_lo: f64, t_up: f64, m: f64, n: f64, k: f64) -> f64 {
    let (lo, hi) = ((k - n).max(0.0), k.min(m));
    if t_lo == f64::NEG_INFINITY { return lo; }
    if t_up == f64::NEG_INFINITY { return hi; }
    let mean = if m + n > 0.0 { k * m / (m + n) } else { 0.0 };
    discrete_q(mean, lo, hi, |x| { let (lp, lq) = hyper_tails(x, m, n, k); reached(lp, lq, t_lo, t_up) })
}

/// `qhyper(p, m, n, k, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qhyper(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = hyper_params(a, "qhyper")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |pr, p| {
        let (m, n, k) = (p[0], p[1], p[2]);
        if hyper_bad(m, n, k) { return f64::NAN; }
        let Some((t_lo, t_up)) = p_in(pr, lower, logp) else { return f64::NAN };
        qhyper_tails(t_lo, t_up, m.round(), n.round(), k.round())
    })
}

// ── the F density and quantile ──────────────────────────────────────────

fn f_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    let (d1, d2) = (pv(a, f, "df1", 1, None)?, pv(a, f, "df2", 2, None)?);
    if let Some(n) = param_vec(a, "ncp", 3)? {
        if n.iter().any(|r| r.map(|x| x != 0.0).unwrap_or(false)) {
            return Err(R2Err { msg: format!("{f}: the non-central F (ncp != 0) is not implemented"), kind: ErrKind::Runtime });
        }
    }
    Ok(vec![d1, d2])
}

/// ln of the gamma density with the given shape and scale.
fn ln_dgamma(x: f64, shape: f64, scale: f64) -> f64 { crate::htest::dgamma(x, shape, scale, true) }

/// `df(x, df1, df2, ncp, log = FALSE)`: the F density, through the binomial
/// form so large degrees of freedom do not cancel.
pub fn bi_df(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = f_params(a, "df")?;
    let log = flag(a, "log", 4, false);
    map_recycled(a, &params, move |x, p| {
        let (m, n) = (p[0], p[1]);
        if x.is_nan() || m.is_nan() || n.is_nan() || m <= 0.0 || n <= 0.0 { return f64::NAN; }
        if x < 0.0 { return d_out(f64::NEG_INFINITY, log); }
        if x == 0.0 {
            return if m > 2.0 { d_out(f64::NEG_INFINITY, log) } else if m == 2.0 { d_out(0.0, log) } else { f64::INFINITY };
        }
        if !m.is_finite() && !n.is_finite() {
            return if x == 1.0 { f64::INFINITY } else { d_out(f64::NEG_INFINITY, log) };
        }
        if !n.is_finite() { return d_out(ln_dgamma(x, m / 2.0, 2.0 / m), log); }
        if m > 1e14 { return d_out(ln_dgamma(1.0 / x, n / 2.0, 2.0 / n) - 2.0 * x.ln(), log); }
        let f = 1.0 / (n + x * m);
        let (q, pp) = (n * f, x * m * f);
        let l = if m >= 2.0 {
            (m * q / 2.0).ln() + ln_db((m - 2.0) / 2.0, n / 2.0, pp, q)
        } else {
            (m * m * q / (2.0 * pp * (m + n))).ln() + ln_db(m / 2.0, n / 2.0, pp, q)
        };
        d_out(l, log)
    })
}

/// `qf(p, df1, df2, ncp, lower.tail = TRUE, log.p = FALSE)`: through the beta
/// quantile, whose pair `(x, 1 - x)` gives `(1/x - 1)` without subtracting.
pub fn bi_qf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let params = f_params(a, "qf")?;
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |pr, p| {
        let (m, n) = (p[0], p[1]);
        if m.is_nan() || n.is_nan() || m <= 0.0 || n <= 0.0 { return f64::NAN; }
        let Some((t_lo, t_up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if t_lo == f64::NEG_INFINITY { return 0.0; }
        if t_up == f64::NEG_INFINITY { return f64::INFINITY; }
        if m <= n && n > 4e5 {
            if !m.is_finite() { return 1.0; }
            return chi_sq_quantile(t_lo, m, true) / m;
        }
        if m > 4e5 { return n / chi_sq_quantile(t_up, n, true); }
        // the lower tail of F is the upper tail of Beta(n/2, m/2)
        let (x, y) = qbeta_xy(t_up, n / 2.0, m / 2.0, true);
        if x == 0.0 { f64::INFINITY } else { (y / x) * (n / m) }
    })
}
