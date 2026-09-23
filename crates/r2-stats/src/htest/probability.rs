//! Statistical primitives — p-value formatting, log-gamma, incomplete
//! beta, and the t / F / chi-squared CDFs used by the hypothesis tests.


// ─────────────────────────────────────────────────────────────────────
// Numerical primitives (pure math, re-exported at crate root for
// engine-side callers like lm/glm summary printers).
// ─────────────────────────────────────────────────────────────────────

/// Significance stars next to a p-value.
pub fn signif_stars(p: f64) -> &'static str {
    if p < 0.001 { "***" }
    else if p < 0.01 { "**" }
    else if p < 0.05 { "*" }
    else if p < 0.1 { "." }
    else { " " }
}

/// Format p-value: "<2e-16" for very small, scientific for very small,
/// 4 significant digits otherwise.
pub fn fmt_pval(p: f64) -> String {
    if p < 2e-16 { "<2e-16".into() }
    else if p < 0.001 { format!("{:.3e}", p) }
    else if p < 1.0 {
        let s = format!("{:.4}", p);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
    else { "1".into() }
}

/// log-gamma via Lanczos (g=7) approximation.
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let xx = x - 1.0;
    let mut ag = coeffs[0];
    for i in 1..9 { ag += coeffs[i] / (xx + i as f64); }
    let t = xx + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xx + 0.5) * t.ln() - t + ag.ln()
}

/// Stirling approximation to gamma. Used by the simple incomplete-beta
/// integrator in `t_cdf` for df ≤ 30.
pub fn gamma_approx(x: f64) -> f64 {
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_approx(1.0 - x));
    }
    let x = x - 1.0;
    (2.0 * std::f64::consts::PI / (x + 1.0)).sqrt() * ((x + 1.0) / std::f64::consts::E).powf(x + 1.0)
}

/// Regularised incomplete beta `I_x(a, b)` via the Lentz continued
/// fraction (Numerical Recipes §6.4). Accurate to ~1e-12 across the
/// full parameter range, including `b < 1` (exercised by `t_cdf`,
/// which calls `incomplete_beta(df/2, 0.5, x)`).
///
/// The symmetry relation `I_x(a, b) = 1 − I_{1−x}(b, a)` is used so the
/// continued fraction is always evaluated in its fast-converging region
/// (`x < (a+1)/(a+b+2)`), which is also what keeps the `b < 1` boundary
/// well-conditioned — the issue that sank the earlier CF attempt.
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if a <= 0.0 || b <= 0.0 { return f64::NAN; }

    // Leading factor  x^a (1-x)^b / B(a, b), in log space.
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Lentz's modified continued fraction for the incomplete beta
/// (Numerical Recipes §6.4, `betacf`). Caller guarantees `x` is in the
/// fast-converging region via the symmetry swap in `incomplete_beta`.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN { d = FPMIN; }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c; if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if d.abs() < FPMIN { d = FPMIN; }
        c = 1.0 + aa / c; if c.abs() < FPMIN { c = FPMIN; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS { break; }
    }
    h
}

/// Upper-tail probability of the F distribution: `P(F > f)` for
/// `F(df1, df2)`, via the exact incomplete-beta identity
/// `P(F > f) = I_{df2/(df2 + df1·f)}(df2/2, df1/2)`.
///
/// Replaces the Wilson-Hilferty approximation previously inlined in the
/// ANOVA tables (~1e-3 error). `f = +∞` returns 0 (so a zero
/// within-residual gives p = 0, not the spurious p = 1 the approximation
/// produced).
pub fn f_sf(f: f64, df1: f64, df2: f64) -> f64 {
    if df1 <= 0.0 || df2 <= 0.0 { return f64::NAN; }
    if f <= 0.0 { return 1.0; }
    if !f.is_finite() { return 0.0; }
    let x = df2 / (df2 + df1 * f);
    incomplete_beta(df2 / 2.0, df1 / 2.0, x)
}

/// Student-t quantile via bisection on `t_cdf`. ~50 iterations to f64
/// precision; fast enough for one-shot CI computation in t.test.
pub fn qt(p: f64, df: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }
    let mut lo = -50.0_f64;
    let mut hi = 50.0_f64;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p { lo = mid; } else { hi = mid; }
    }
    0.5 * (lo + hi)
}

/// Student-t CDF using the regularised incomplete-beta identity:
///   P(T ≤ t) = 1 − ½ · I_{x}(df/2, ½)     where x = df / (df + t²)
/// for `t ≥ 0`; reflect for negative `t`. The Lentz CF reaches ~1e-7
/// across all df (the previous shortcut to a normal-approx for df > 30
/// produced ~1e-3 error at moderate df, which mattered for printed
/// p-values around the 0.05 threshold).
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 { return f64::NAN; }
    let x = df / (df + t * t);
    let half_ib = 0.5 * incomplete_beta(df / 2.0, 0.5, x);
    if t >= 0.0 { 1.0 - half_ib } else { half_ib }
}

// ─────────────────────────────────────────────────────────────────────
// The gamma family — χ² is `Gamma(df/2, scale 2)` — built the way R's
// nmath builds it.
//
// This replaced a χ² CDF that measured, against R 4.5.3 on a 492-value
// grid, up to 10% wrong (`qchisq(0.999999, 1000)`: 1100 vs 1227), and it
// failed in four independent ways, each of which the pieces below remove:
//   * `|df - 2| < 0.01` was treated as df = 2 (and likewise df = 1):
//     `pchisq(1, 2.005)` came out 0.39347 against R's 0.39225;
//   * past `x > df + 100` it returned exactly 1 — only 2.2 standard
//     deviations out at df = 1000, so `pchisq(1101, 1000)` was 1, not
//     0.98615, and `qchisq` could never exceed df + 100;
//   * one algorithm (the lower-tail series) for both tails, the upper
//     tail taken as `1 - lower`, which cancels small p-values away;
//   * the df = 1 / df = 2 closed forms cancelled at small x (`1 - exp`),
//     costing `qchisq(1e-10, 1)` six digits.
// ─────────────────────────────────────────────────────────────────────

/// `stirlerr(n) = ln n! - [(n + 1/2) ln n - n + ln sqrt(2 pi)]`, the
/// error of Stirling's formula — Loader's table for half-integers up to
/// 15, the asymptotic series above. It is what lets a Poisson/gamma
/// density be computed as a difference of SMALL numbers instead of the
/// large, nearly cancelling ones `lgamma` gives (at shape 5000, `x ln x`,
/// `x` and `lgamma` are each ~4e4, and their difference ~1e-2).
fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 0.083333333333333333333;         // 1/12
    const S1: f64 = 0.00277777777777777777778;       // 1/360
    const S2: f64 = 0.00079365079365079365079365;    // 1/1260
    const S3: f64 = 0.000595238095238095238095238;   // 1/1680
    const S4: f64 = 0.0008417508417508417508417508;  // 1/1188
    const HALVES: [f64; 31] = [
        0.0, 0.1534264097200273452913848, 0.0810614667953272582196702, 0.0548141210519176538961390,
        0.0413406959554092940938221, 0.03316287351993628748511048, 0.02767792568499833914878929,
        0.02374616365629749597132920, 0.02079067210376509311152277, 0.01848845053267318523077934,
        0.01664469118982119216319487, 0.01513497322191737887351255, 0.01387612882307074799874573,
        0.01281046524292022692424986, 0.01189670994589177009505572, 0.01110455975820691732662991,
        0.010411265261972096497478567, 0.009799416126158803298389475, 0.009255462182712732917728637,
        0.008768700134139385462952823, 0.008330563433362871256469318, 0.007934114564314020547248100,
        0.007573675487951840794972024, 0.007244554301320383179543912, 0.006942840107209529865664152,
        0.006665247032707682442354394, 0.006408994188004207068439631, 0.006171712263039457647532867,
        0.005951370112758847735624416, 0.005746216513010115682023589, 0.005554733551962801371038690,
    ];
    if n <= 15.0 {
        let nn = n + n;
        if nn == nn.trunc() { return HALVES[nn as usize]; }
        return ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - 0.918_938_533_204_672_741_78;   // ln sqrt(2 pi)
    }
    let nn = n * n;
    if n > 500.0 { return (S0 - S1 / nn) / n; }
    if n > 80.0 { return (S0 - (S1 - S2 / nn) / nn) / n; }
    if n > 35.0 { return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n; }
    (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
}

/// `bd0(x, np) = x ln(x/np) + np - x`, computed without the cancellation
/// of that formula when `x` is near `np` (Loader's series).
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE { return s; }
        let mut ej = 2.0 * x * v;
        v *= v;
        for j in 1..1000 {
            ej *= v;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s { return s1; }
            s = s1;
        }
    }
    x * (x / np).ln() + np - x
}

/// ln of the Poisson density at a real "count" `x` with mean `lambda`:
/// `x ln(lambda) - lambda - ln Gamma(x + 1)`, in Loader's saddle-point
/// form. Also `ln(lambda^x e^-lambda / Gamma(x+1))`, the prefactor of both
/// incomplete-gamma tails.
fn ln_dpois_raw(x: f64, lambda: f64) -> f64 {
    if lambda == 0.0 { return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY }; }
    if !lambda.is_finite() || x < 0.0 { return f64::NEG_INFINITY; }
    if x <= lambda * f64::MIN_POSITIVE { return -lambda; }
    if lambda < x * f64::MIN_POSITIVE { return -lambda + x * lambda.ln() - ln_gamma(x + 1.0); }
    -stirlerr(x) - bd0(x, lambda) - 0.5 * (2.0 * std::f64::consts::PI * x).ln()
}

/// Density of `Gamma(shape, scale)` at `x`, as R's `dgamma`, on the log
/// scale when `log` is set.
pub fn dgamma(x: f64, shape: f64, scale: f64, log: bool) -> f64 {
    let out = |v: f64| if log { v } else { v.exp() };
    if x.is_nan() || shape.is_nan() || scale.is_nan() { return f64::NAN; }
    if shape < 0.0 || scale <= 0.0 { return f64::NAN; }
    if x < 0.0 { return out(f64::NEG_INFINITY); }
    if shape == 0.0 { return if x == 0.0 { f64::INFINITY } else { out(f64::NEG_INFINITY) }; }
    if x == 0.0 {
        if shape < 1.0 { return f64::INFINITY; }
        if shape > 1.0 { return out(f64::NEG_INFINITY); }
        return out(-scale.ln());
    }
    if shape < 1.0 {
        out(ln_dpois_raw(shape, x / scale) + (shape / x).ln())
    } else {
        out(ln_dpois_raw(shape - 1.0, x / scale) - scale.ln())
    }
}

/// Both tails of the regularised incomplete gamma `P(a, x)` / `Q(a, x)`
/// on the log scale: `(ln P, ln Q)`.
///
/// The tail that is at most about one half is computed DIRECTLY — the
/// series for `P` when `x < a + 1`, the Lentz continued fraction for `Q`
/// otherwise — each times the prefactor `x^a e^-x / Gamma(a+1)` in
/// Loader's form; the other tail is its complement, which by the choice
/// of branch never cancels. So a p-value of 1e-300 comes out with full
/// relative precision, and the log scale reaches past underflow.
///
/// Both loops converge in O(sqrt(a)) terms near the mode; the cap is far
/// beyond what shape 1e8 needs, not a cutoff.
pub fn pgamma_log_tails(a: f64, x: f64) -> (f64, f64) {
    if x <= 0.0 { return (f64::NEG_INFINITY, 0.0); }
    if x.is_infinite() { return (0.0, f64::NEG_INFINITY); }
    let ln1m = |l: f64| if l > -std::f64::consts::LN_2 { (-l.exp_m1()).ln() } else { (-l.exp()).ln_1p() };
    let lpre = ln_dpois_raw(a, x);                       // ln(x^a e^-x / Gamma(a+1))
    if x < a + 1.0 {
        // P = pre * sum_{n>=0} x^n / ((a+1)...(a+n))
        let (mut sum, mut term) = (1.0f64, 1.0f64);
        for n in 1..2_000_000 {
            term *= x / (a + n as f64);
            sum += term;
            if term < sum * 1e-17 { break; }
        }
        let lp = lpre + sum.ln();
        (lp, ln1m(lp))
    } else {
        // Q = pre * a/x * CF,  CF = 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a-...)))
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..2_000_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny { d = tiny; }
            c = b + an / c;
            if c.abs() < tiny { c = tiny; }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-17 { break; }
        }
        // x^a e^-x / Gamma(a) = a * pre
        let lq = lpre + a.ln() + h.ln();
        (ln1m(lq), lq)
    }
}

/// χ² CDF, the lower tail: `P(X <= x)` for `X ~ χ²(df)`.
pub fn chi_sq_cdf(x: f64, df: f64) -> f64 {
    if x.is_nan() || df.is_nan() { return f64::NAN; }
    if df < 0.0 { return f64::NAN; }
    if x <= 0.0 { return 0.0; }
    if df == 0.0 { return 1.0; }
    pgamma_log_tails(df / 2.0, x / 2.0).0.exp()
}

/// χ² upper tail `P(X > x)` — computed directly, so small p-values keep
/// full relative precision (what `chisq.test` reports).
pub fn chi_sq_sf(x: f64, df: f64) -> f64 {
    if x.is_nan() || df.is_nan() { return f64::NAN; }
    if df < 0.0 { return f64::NAN; }
    if x <= 0.0 { return 1.0; }
    if df == 0.0 { return 0.0; }
    pgamma_log_tails(df / 2.0, x / 2.0).1.exp()
}

/// χ² quantile: the `x` with the requested tail equal to `p` (`lp` is ln p).
///
/// Inverts whichever tail is the smaller one at the target — the lower
/// tail for p <= 1/2, the upper tail otherwise — on the LOG scale, with a
/// bracketed Newton step (`d ln P / dx = density / P`) and bisection when
/// Newton leaves the bracket. Converges to a few ULP of x; no range cap.
pub fn chi_sq_quantile(lp: f64, df: f64, lower: bool) -> f64 {
    if lp.is_nan() || df.is_nan() || df < 0.0 || lp > 0.0 { return f64::NAN; }
    if df == 0.0 { return 0.0; }
    // work with the tail that is <= 1/2
    let (use_lower, lt) = if lp < -std::f64::consts::LN_2 { (lower, lp) }
        else if lower { (false, (-lp.exp_m1()).ln()) } else { (true, (-lp.exp_m1()).ln()) };
    if lt == f64::NEG_INFINITY { return if use_lower { 0.0 } else { f64::INFINITY }; }
    let a = df / 2.0;
    let ltail = |x: f64| { let (lp_, lq_) = pgamma_log_tails(a, x / 2.0); if use_lower { lp_ } else { lq_ } };
    // bracket [lo, hi] in x: the lower tail rises with x, the upper falls
    let (mut lo, mut hi) = (0.0f64, df.max(1.0));
    while (ltail(hi) < lt) == use_lower { lo = hi; hi *= 2.0; if hi > 1e300 { break; } }
    // A small lower-tail target: P(x) ~ (x/2)^a / Gamma(a+1) as x -> 0, so
    // x ~ 2 exp((ln p + ln Gamma(a+1)) / a). When that underflows the true
    // quantile is below the smallest double and 0 is the right answer —
    // bisection from the bracket could never get there (200 halvings of 1
    // reach 1e-60; qchisq(1e-300, 0.5) is ~1e-1200) and would stop on a
    // wrong nonzero value. Otherwise it is the starting point.
    let small = if use_lower { 2.0 * ((lt + ln_gamma(a + 1.0)) / a).exp() } else { f64::INFINITY };
    if small == 0.0 { return 0.0; }
    // else start from Wilson–Hilferty, clamped into the bracket
    let z = crate::dist::qnorm_approx(if use_lower { lt.exp() } else { -lt.exp_m1() });
    let wh = df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3);
    let mut x = if small < 0.1 * df.min(1.0) && small > lo && small < hi { small }
        else if wh > lo && wh < hi { wh } else { 0.5 * (lo + hi) };
    for _ in 0..200 {
        let f = ltail(x) - lt;
        if f == 0.0 { break; }
        if (f < 0.0) == use_lower { lo = x; } else { hi = x; }
        // d ln(tail)/dx = +-density/tail
        let ld = dgamma(x, a, 2.0, true) - ltail(x);
        let step = if use_lower { f / ld.exp() } else { -f / ld.exp() };
        let mut xn = x - step;
        if !(xn > lo && xn < hi) || !xn.is_finite() {
            // bisect — geometrically when the bracket spans decades
            xn = if lo > 0.0 && hi > 4.0 * lo { (lo * hi).sqrt() } else { 0.5 * (lo + hi) };
        }
        if (xn - x).abs() <= 4.0 * f64::EPSILON * x.abs() { x = xn; break; }
        x = xn;
    }
    x
}

#[cfg(test)]
mod chisq_properties {
    //! The chi-squared family checked against MATHEMATICS — identities that
    //! hold whatever any other program prints. The comparison with R lives
    //! outside the code, in tests/differential/cases/distributions.R, which
    //! runs R itself.
    use super::*;
    use crate::dist::{erf, erfc};

    const DFS: [f64; 10] = [0.5, 1.0, 1.5, 2.0, 2.005, 3.0, 10.0, 100.0, 1000.0, 10000.0];

    /// Points over the body of chi-squared(df): mean df, sd sqrt(2 df).
    fn bulk(df: f64) -> Vec<f64> {
        let sd = (2.0 * df).sqrt();
        (-6..=12).map(|k| df + k as f64 * sd / 2.0).filter(|&x| x > 0.02).collect()
    }

    fn rel(a: f64, b: f64) -> f64 { if a == b { 0.0 } else { ((a - b) / b).abs() } }

    /// P(X <= x) + P(X > x) = 1.
    #[test]
    fn the_two_tails_sum_to_one() {
        for df in DFS { for x in bulk(df) {
            let s = chi_sq_cdf(x, df) + chi_sq_sf(x, df);
            assert!((s - 1.0).abs() < 1e-14, "df {df}, x {x}: tails sum to {s}");
        } }
    }

    /// qchisq inverts pchisq, in each tail, down to p = 1e-300.
    #[test]
    fn the_quantile_inverts_the_cdf_in_both_tails() {
        for df in DFS { for p in [1e-300f64, 1e-100, 1e-10, 1e-3, 0.1, 0.5, 0.9] {
            let x = chi_sq_quantile(p.ln(), df, true);
            // where the true quantile is below the smallest double
            // (P(x) ~ (x/2)^a / Gamma(a+1) near 0), the answer is exactly 0
            let a = df / 2.0;
            if 2.0 * ((p.ln() + ln_gamma(a + 1.0)) / a).exp() < 1e-300 {
                assert_eq!(x, 0.0, "df {df}, p {p}: the quantile underflows, so it is 0");
                continue;
            }
            assert!(rel(chi_sq_cdf(x, df), p) < 1e-11, "lower: df {df}, p {p}: P(q) = {}", chi_sq_cdf(x, df));
            let x = chi_sq_quantile(p.ln(), df, false);
            assert!(rel(chi_sq_sf(x, df), p) < 1e-11, "upper: df {df}, p {p}: Q(q) = {}", chi_sq_sf(x, df));
        } }
    }

    /// The density is the slope of the CDF (central difference) — taken on
    /// whichever tail is the smaller, since differencing two values of a
    /// probability near 1 loses the digits the check needs.
    #[test]
    fn the_density_is_the_derivative_of_the_cdf() {
        for df in DFS { for x in bulk(df) {
            // the step scales with the distribution's width (sd) and, near 0,
            // with x: the difference's truncation error goes as (h / scale)^2
            let h = 1e-4 * (2.0 * df).sqrt().min(x);
            let slope = if chi_sq_cdf(x, df) < 0.5 {
                (chi_sq_cdf(x + h, df) - chi_sq_cdf(x - h, df)) / (2.0 * h)
            } else {
                (chi_sq_sf(x - h, df) - chi_sq_sf(x + h, df)) / (2.0 * h)
            };
            let d = dgamma(x, df / 2.0, 2.0, false);
            assert!(rel(slope, d) < 1e-6, "df {df}, x {x}: slope {slope} vs density {d}");
        } }
    }

    /// The closed forms: chi-squared(2) is the exponential with mean 2, and
    /// chi-squared(1) is the square of a standard normal — computed here by
    /// independent routes (expm1, and Cody's erf / erfc).
    #[test]
    fn df_one_and_two_equal_their_closed_forms() {
        for i in 0..200 {
            let x = 10f64.powf(-10.0 + i as f64 * 0.06);        // 1e-10 .. 1e2
            assert!(rel(chi_sq_cdf(x, 2.0), -(-x / 2.0).exp_m1()) < 1e-14, "df 2 lower at {x}");
            assert!(rel(chi_sq_sf(x, 2.0), (-x / 2.0).exp()) < 1e-14, "df 2 upper at {x}");
            let z = (x / 2.0).sqrt();
            assert!(rel(chi_sq_cdf(x, 1.0), erf(z)) < 1e-13, "df 1 lower at {x}");
            assert!(rel(chi_sq_sf(x, 1.0), erfc(z)) < 1e-13, "df 1 upper at {x}");
        }
    }

    /// The CDF never decreases.
    #[test]
    fn the_cdf_is_monotone() {
        for df in DFS {
            let xs = bulk(df);
            for w in xs.windows(2) { assert!(chi_sq_cdf(w[1], df) >= chi_sq_cdf(w[0], df), "df {df} at {}", w[1]); }
        }
    }

    /// The values the definitions fix at the edges.
    #[test]
    fn edges_follow_the_definitions() {
        assert_eq!(dgamma(0.0, 0.5, 2.0, false), f64::INFINITY);   // density of chi-squared(1) at 0
        assert_eq!(dgamma(0.0, 1.0, 2.0, false), 0.5);             // chi-squared(2) = Exp(mean 2)
        assert_eq!(dgamma(0.0, 1.5, 2.0, false), 0.0);             // chi-squared(3) at 0
        assert_eq!(chi_sq_cdf(0.0, 3.0), 0.0);
        assert_eq!(chi_sq_sf(0.0, 3.0), 1.0);
        assert_eq!(chi_sq_cdf(f64::INFINITY, 3.0), 1.0);
        assert!(chi_sq_cdf(1.0, -1.0).is_nan());
        assert_eq!(chi_sq_quantile(0.0, 3.0, true), f64::INFINITY);        // p = 1
        assert_eq!(chi_sq_quantile(f64::NEG_INFINITY, 3.0, true), 0.0);    // p = 0
    }
}
