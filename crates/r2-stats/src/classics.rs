//! The normal, exponential, binomial, Poisson, Student-t and F-tail functions,
//! on the same base as `crate::families`: every parameter recycled, `log`,
//! `lower.tail` and `log.p` honoured, a required parameter required.
//!
//! These once took each parameter as a single number, read `log = TRUE` nowhere
//! (`dnorm(1, log = TRUE)` returned the density itself), and silently kept the
//! first of a vector parameter. tests/differential/cases/families.R compares
//! them with R.

use r2_types::{EvalArg, R2Err, RVal, Real};

use crate::betagamma::{ln1mexp, map_recycled, pbeta_log_tails};
use crate::dist::{flag, phi, phi_upper, pnorm_log_tails, qnorm_approx};
use crate::families::{d_out, discrete_q, is_whole, ln_db, p_in, pv, reached, reject_ncp, tail};
use crate::htest::{bd0, ln_dpois_raw, pgamma_log_tails, qt, stirlerr};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_741_78;
const LN_2: f64 = std::f64::consts::LN_2;

// ── normal ──────────────────────────────────────────────────────────────

fn norm_params(a: &[EvalArg]) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, "", "mean", 1, Some(0.0))?, pv(a, "", "sd", 2, Some(1.0))?])
}

/// `dnorm(x, mean = 0, sd = 1, log = FALSE)`.
pub fn bi_dnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &norm_params(a)?, move |x, p| {
        let (m, s) = (p[0], p[1]);
        if x.is_nan() || m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        if s == 0.0 { return if x == m { f64::INFINITY } else { d_out(f64::NEG_INFINITY, log) }; }
        if !x.is_finite() && m == x { return f64::NAN; }
        let z = (x - m) / s;
        if !z.is_finite() { return d_out(f64::NEG_INFINITY, log); }
        d_out(-(LN_SQRT_2PI + 0.5 * z * z + s.ln()), log)
    })
}

/// `pnorm(q, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &norm_params(a)?, move |q, p| {
        let (m, s) = (p[0], p[1]);
        if q.is_nan() || m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        let z = (q - m) / s;
        if z.is_nan() {
            if s == 0.0 { return tail(0.0, f64::NEG_INFINITY, lower, logp); }
            return f64::NAN;
        }
        if logp {
            let (lp, lq) = pnorm_log_tails(z);
            if lower { lp } else { lq }
        } else if lower { phi(z) } else { phi_upper(z) }
    })
}

/// `qnorm(p, mean = 0, sd = 1, lower.tail = TRUE, log.p = FALSE)`. The normal is
/// symmetric, so the smaller tail decides it and no `1 - p` loses digits.
pub fn bi_qnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &norm_params(a)?, move |pr, p| {
        let (m, s) = (p[0], p[1]);
        if m.is_nan() || s.is_nan() || s < 0.0 { return f64::NAN; }
        let Some((l, u)) = p_in(pr, lower, logp) else { return f64::NAN };
        if l == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
        if u == f64::NEG_INFINITY { return f64::INFINITY; }
        let z = if l <= u { qnorm_approx(l.exp()) } else { -qnorm_approx(u.exp()) };
        m + s * z
    })
}

// ── exponential ─────────────────────────────────────────────────────────

fn rate_param(a: &[EvalArg]) -> Result<Vec<Vec<Real>>, R2Err> { Ok(vec![pv(a, "", "rate", 1, Some(1.0))?]) }

/// `dexp(x, rate = 1, log = FALSE)`.
pub fn bi_dexp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 2, false);
    map_recycled(a, &rate_param(a)?, move |x, p| {
        let r = p[0];
        if x.is_nan() || r.is_nan() || r <= 0.0 { return f64::NAN; }
        if x < 0.0 { return d_out(f64::NEG_INFINITY, log); }
        d_out(r.ln() - r * x, log)
    })
}

/// `pexp(q, rate = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pexp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &rate_param(a)?, move |q, p| {
        let r = p[0];
        if q.is_nan() || r.is_nan() || r <= 0.0 { return f64::NAN; }
        if q <= 0.0 { return tail(f64::NEG_INFINITY, 0.0, lower, logp); }
        let lq = -r * q;
        tail(ln1mexp(lq), lq, lower, logp)
    })
}

/// `qexp(p, rate = 1, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qexp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &rate_param(a)?, move |pr, p| {
        let r = p[0];
        if r.is_nan() || r <= 0.0 { return f64::NAN; }
        let Some((_, lq)) = p_in(pr, lower, logp) else { return f64::NAN };
        -lq / r
    })
}

// ── binomial ────────────────────────────────────────────────────────────

fn binom_params(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    Ok(vec![pv(a, f, "size", 1, None)?, pv(a, f, "prob", 2, None)?])
}

fn binom_bad(n: f64, pb: f64) -> bool { n.is_nan() || pb.is_nan() || n < 0.0 || !is_whole(n) || !(0.0..=1.0).contains(&pb) }

/// `dbinom(x, size, prob, log = FALSE)`.
pub fn bi_dbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &binom_params(a, "dbinom")?, move |x, p| {
        let (n, pb) = (p[0], p[1]);
        if x.is_nan() || binom_bad(n, pb) { return f64::NAN; }
        let n = n.round();
        if x < 0.0 || !is_whole(x) || x.round() > n { return d_out(f64::NEG_INFINITY, log); }
        let x = x.round();
        d_out(ln_db(x, n - x, pb, 1.0 - pb), log)
    })
}

/// `P(X <= k)` and `P(X > k)` on the log scale: `I_{1-p}(n - k, k + 1)`.
fn binom_tails(k: f64, n: f64, pb: f64) -> (f64, f64) {
    if k < 0.0 { return (f64::NEG_INFINITY, 0.0); }
    let k = (k + 1e-7).floor();
    if k >= n { return (0.0, f64::NEG_INFINITY); }
    pbeta_log_tails(1.0 - pb, pb, n - k, k + 1.0)
}

/// `pbinom(q, size, prob, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_pbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &binom_params(a, "pbinom")?, move |q, p| {
        let (n, pb) = (p[0], p[1]);
        if q.is_nan() || binom_bad(n, pb) { return f64::NAN; }
        let (lp, lq) = binom_tails(q, n.round(), pb);
        tail(lp, lq, lower, logp)
    })
}

/// `qbinom(p, size, prob, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &binom_params(a, "qbinom")?, move |pr, p| {
        let (n, pb) = (p[0], p[1]);
        if binom_bad(n, pb) { return f64::NAN; }
        let n = n.round();
        let Some((t_lo, t_up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if pb == 0.0 || n == 0.0 || t_lo == f64::NEG_INFINITY { return 0.0; }
        if pb == 1.0 || t_up == f64::NEG_INFINITY { return n; }
        discrete_q(n * pb, 0.0, n, |k| { let (lp, lq) = binom_tails(k, n, pb); reached(lp, lq, t_lo, t_up) })
    })
}

// ── Poisson ─────────────────────────────────────────────────────────────

fn lambda_param(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> { Ok(vec![pv(a, f, "lambda", 1, None)?]) }

/// `dpois(x, lambda, log = FALSE)`.
pub fn bi_dpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 2, false);
    map_recycled(a, &lambda_param(a, "dpois")?, move |x, p| {
        let l = p[0];
        if x.is_nan() || l.is_nan() || l < 0.0 { return f64::NAN; }
        if x < 0.0 || !x.is_finite() || !is_whole(x) { return d_out(f64::NEG_INFINITY, log); }
        d_out(ln_dpois_raw(x.round(), l), log)
    })
}

/// `P(X <= k)` and `P(X > k)` on the log scale: the two tails of the gamma,
/// swapped, since `P(X <= k) = Q(k + 1, lambda)`.
fn pois_tails(k: f64, l: f64) -> (f64, f64) {
    if k < 0.0 { return (f64::NEG_INFINITY, 0.0); }
    if !k.is_finite() || l == 0.0 { return (0.0, f64::NEG_INFINITY); }
    let (lp, lq) = pgamma_log_tails((k + 1e-7).floor() + 1.0, l);
    (lq, lp)
}

/// `ppois(q, lambda, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_ppois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &lambda_param(a, "ppois")?, move |q, p| {
        if q.is_nan() || p[0].is_nan() || p[0] < 0.0 { return f64::NAN; }
        let (lp, lq) = pois_tails(q, p[0]);
        tail(lp, lq, lower, logp)
    })
}

/// `qpois(p, lambda, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 2, true), flag(a, "log.p", 3, false));
    map_recycled(a, &lambda_param(a, "qpois")?, move |pr, p| {
        let l = p[0];
        if l.is_nan() || l < 0.0 { return f64::NAN; }
        let Some((t_lo, t_up)) = p_in(pr, lower, logp) else { return f64::NAN };
        if l == 0.0 || t_lo == f64::NEG_INFINITY { return 0.0; }
        if t_up == f64::NEG_INFINITY { return f64::INFINITY; }
        discrete_q(l, 0.0, f64::INFINITY, |k| { let (lp, lq) = pois_tails(k, l); reached(lp, lq, t_lo, t_up) })
    })
}

// ── Student-t ───────────────────────────────────────────────────────────

fn df_param(a: &[EvalArg], f: &str) -> Result<Vec<Vec<Real>>, R2Err> {
    reject_ncp(a, f, 2, "t")?;
    Ok(vec![pv(a, f, "df", 1, None)?])
}

/// `dt(x, df, ncp, log = FALSE)`, in Loader's form as R's, so large `df` and far
/// tails do not overflow before they cancel.
pub fn bi_dt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let log = flag(a, "log", 3, false);
    map_recycled(a, &df_param(a, "dt")?, move |x, p| {
        let n = p[0];
        if x.is_nan() || n.is_nan() || n <= 0.0 { return f64::NAN; }
        if !x.is_finite() { return d_out(f64::NEG_INFINITY, log); }
        if !n.is_finite() { return d_out(-(LN_SQRT_2PI + 0.5 * x * x), log); }
        let t = -bd0(n / 2.0, (n + 1.0) / 2.0) + stirlerr((n + 1.0) / 2.0) - stirlerr(n / 2.0);
        let x2n = x * x / n;
        let (l_x2n, u) = if x2n > 1.0 / f64::EPSILON {
            let l = x.abs().ln() - n.ln() / 2.0;
            (l, n * l)
        } else if x2n > 0.2 {
            let l = (1.0 + x2n).ln() / 2.0;
            (l, n * l)
        } else {
            (x2n.ln_1p() / 2.0, -bd0(n / 2.0, (n + x * x) / 2.0) + x * x / 2.0)
        };
        d_out(t - u - (LN_SQRT_2PI + l_x2n), log)
    })
}

/// `pt(q, df, ncp, lower.tail = TRUE, log.p = FALSE)`, from the identity whose
/// value is the small probability, on the log scale so a far tail reaches past
/// underflow.
pub fn bi_pt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &df_param(a, "pt")?, move |t, p| {
        let n = p[0];
        if t.is_nan() || n.is_nan() || n <= 0.0 { return f64::NAN; }
        if !t.is_finite() {
            return if t < 0.0 { tail(f64::NEG_INFINITY, 0.0, lower, logp) } else { tail(0.0, f64::NEG_INFINITY, lower, logp) };
        }
        if !n.is_finite() {
            let (lp, lq) = pnorm_log_tails(t);
            return tail(lp, lq, lower, logp);
        }
        let t2 = t * t;
        // (ln P(T below |t|), ln P(T above |t|))
        let (big, small) = if t2 < 1.0 {
            // the middle: P(|T| <= |t|) = I_y(1/2, n/2), y = t^2 / (n + t^2)
            let (li, lq) = pbeta_log_tails(t2 / (n + t2), n / (n + t2), 0.5, n / 2.0);
            (li.exp().ln_1p() - LN_2, lq - LN_2)
        } else {
            // the tail: P(|T| > |t|) = I_x(n/2, 1/2), x = n / (n + t^2)
            let li = pbeta_log_tails(n / (n + t2), t2 / (n + t2), n / 2.0, 0.5).0 - LN_2;
            (ln1mexp(li), li)
        };
        let (lp, lq) = if t >= 0.0 { (big, small) } else { (small, big) };
        tail(lp, lq, lower, logp)
    })
}

/// `qt(p, df, ncp, lower.tail = TRUE, log.p = FALSE)`.
pub fn bi_qt(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (lower, logp) = (flag(a, "lower.tail", 3, true), flag(a, "log.p", 4, false));
    map_recycled(a, &df_param(a, "qt")?, move |pr, p| {
        let n = p[0];
        if n.is_nan() || n <= 0.0 { return f64::NAN; }
        let Some((l, u)) = p_in(pr, lower, logp) else { return f64::NAN };
        if l == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
        if u == f64::NEG_INFINITY { return f64::INFINITY; }
        if !n.is_finite() { return if l <= u { qnorm_approx(l.exp()) } else { -qnorm_approx(u.exp()) }; }
        if l <= u { qt(l.exp(), n) } else { -qt(u.exp(), n) }
    })
}

// ── the F tail ──────────────────────────────────────────────────────────

/// `pf(q, df1, df2, ncp, lower.tail = TRUE, log.p = FALSE)`: both tails from the
/// beta, with `1 - x` formed directly as `df2 / (df1 q + df2)`.
pub fn bi_pf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    reject_ncp(a, "pf", 3, "F")?;
    let params = vec![pv(a, "pf", "df1", 1, None)?, pv(a, "pf", "df2", 2, None)?];
    let (lower, logp) = (flag(a, "lower.tail", 4, true), flag(a, "log.p", 5, false));
    map_recycled(a, &params, move |q, p| {
        let (m, n) = (p[0], p[1]);
        if q.is_nan() || m.is_nan() || n.is_nan() || m <= 0.0 || n <= 0.0 { return f64::NAN; }
        if q <= 0.0 { return tail(f64::NEG_INFINITY, 0.0, lower, logp); }
        if q.is_infinite() { return tail(0.0, f64::NEG_INFINITY, lower, logp); }
        if n.is_infinite() {
            if m.is_infinite() {
                return if q < 1.0 { tail(f64::NEG_INFINITY, 0.0, lower, logp) }
                    else if q == 1.0 { tail(-LN_2, -LN_2, lower, logp) }
                    else { tail(0.0, f64::NEG_INFINITY, lower, logp) };
            }
            let (lp, lq) = pgamma_log_tails(m / 2.0, q * m / 2.0);
            return tail(lp, lq, lower, logp);
        }
        if m.is_infinite() {
            let (lp, lq) = pgamma_log_tails(n / 2.0, n / q / 2.0);
            return tail(lq, lp, lower, logp);
        }
        let den = m * q + n;
        let (lp, lq) = pbeta_log_tails(m * q / den, n / den, m / 2.0, n / 2.0);
        tail(lp, lq, lower, logp)
    })
}
