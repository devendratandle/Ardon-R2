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
    // start from Wilson–Hilferty, clamped into the bracket
    let z = crate::dist::qnorm_approx(if use_lower { lt.exp() } else { -lt.exp_m1() });
    let wh = df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3);
    let mut x = if wh > lo && wh < hi { wh } else { 0.5 * (lo + hi) };
    for _ in 0..200 {
        let f = ltail(x) - lt;
        if f == 0.0 { break; }
        if (f < 0.0) == use_lower { lo = x; } else { hi = x; }
        // d ln(tail)/dx = +-density/tail
        let ld = dgamma(x, a, 2.0, true) - ltail(x);
        let step = if use_lower { f / ld.exp() } else { -f / ld.exp() };
        let mut xn = x - step;
        if !(xn > lo && xn < hi) || !xn.is_finite() { xn = 0.5 * (lo + hi); }
        if (xn - x).abs() <= 4.0 * f64::EPSILON * x.abs() { x = xn; break; }
        x = xn;
    }
    x
}

#[cfg(test)]
mod chisq_vs_r {

        //! The χ² family against R 4.5.3, value for value (R's own `sprintf("%%.17e")` output):
        //! every row a case the old implementation got wrong or a branch of the new one.
        //! Tolerance 1e-12 relative — R sums its series in another order.
        use super::*;

    // (x, df, pchisq, pchisq upper, dchisq)
    const P: &[(f64, f64, f64, f64, f64)] = &[
        (0.001, 0.5, 1.64959750768412822e-01, 8.35040249231587262e-01, 4.12234446493734765e+01),
        (0.5, 1.0, 5.20499877813046519e-01, 4.79500122186953481e-01, 4.39391289467722435e-01),
        (1.0, 2.005, 3.92246955341085501e-01, 6.07753044658914443e-01, 3.03175891008803322e-01),
        (2.0, 1.0, 8.42700792949715560e-01, 1.57299207050284467e-01, 1.03776874355148666e-01),
        (5.0, 3.0, 8.28202855703266794e-01, 1.71797144296733206e-01, 7.32249128096324475e-02),
        (10.0, 10.0, 5.59506714934787541e-01, 4.40493285065212459e-01, 8.77336848839253697e-02),
        (50.0, 5.0, 9.99999998614202634e-01, 1.38579733670095934e-09, 6.52952772125722279e-10),
        (100.0, 50.0, 9.99965450686170154e-01, 3.45493138298486413e-05, 9.26446496012619852e-06),
        (300.0, 100.0, 1.00000000000000000e+00, 7.41210085732287457e-22, 2.50705894152777400e-22),
        (1050.0, 1000.0, 8.67525943178522940e-01, 1.32474056821477087e-01, 4.63896262901107755e-03),
        (1101.0, 1000.0, 9.86145026081295728e-01, 1.38549739187042806e-02, 7.41862666809810960e-04),
        (1200.0, 1000.0, 9.99987744057669325e-01, 1.22559423306229079e-05, 1.07724943014962010e-06),
        (10100.0, 10000.0, 7.60984674759824897e-01, 2.39015325240175158e-01, 2.17876943448854551e-03),
        (9900.0, 10000.0, 2.40479914316020643e-01, 7.59520085683979329e-01, 2.21538758734431736e-03),
        (0.1, 0.5, 5.16555320830465625e-01, 4.83444679169534375e-01, 1.24064263562715382e+00),
        (1e-8, 1.0, 7.97884559473057786e-05, 9.99920211544052751e-01, 3.98942278406721289e+03),
        (30.0, 1.0, 9.99999956795369460e-01, 4.32046305782749682e-08, 2.22808733452496618e-08),
        (70.0, 3.0, 9.99999999999995781e-01, 4.26833633549231165e-15, 2.10451593849574330e-15),
    ];

    // (p, df, qchisq)
    const Q: &[(f64, f64, f64)] = &[
        (1e-10, 0.5, 1.34993957862235127e-40), (0.01, 0.5, 1.34993958591169451e-08),
        (0.5, 0.5, 8.73476047057468730e-02), (0.95, 0.5, 2.42023227488951775e+00),
        (0.999999, 0.5, 2.13756351528219604e+01),
        (1e-10, 1.0, 1.57079632679489347e-20), (0.01, 1.0, 1.57087857909702055e-04),
        (0.5, 1.0, 4.54936423119572830e-01), (0.95, 1.0, 3.84145882069412403e+00),
        (0.999999, 1.0, 2.39281269768794722e+01),
        (1e-10, 2.0, 2.00000000009999768e-10), (0.01, 2.0, 2.01006717070028838e-02),
        (0.5, 2.0, 1.38629436111989035e+00), (0.95, 2.0, 5.99146454710797993e+00),
        (0.999999, 2.0, 2.76310211158710359e+01),
        (1e-10, 2.005, 2.12044206806942537e-10), (0.01, 2.005, 2.03554978987766366e-02),
        (0.5, 2.005, 1.39113503169397634e+00), (0.95, 2.005, 6.00114488080492592e+00),
        (0.999999, 2.005, 2.76473675418424882e+01),
        (1e-10, 5.0, 3.23355714624969331e-04), (0.01, 5.0, 5.54298076728277134e-01),
        (0.5, 5.0, 4.35146019109552640e+00), (0.95, 5.0, 1.10704976935163533e+01),
        (0.999999, 5.0, 3.58881868796104229e+01),
        (1e-10, 100.0, 3.43998239091248195e+01), (0.01, 100.0, 7.00648949253997984e+01),
        (0.5, 100.0, 9.93341292359884847e+01), (0.95, 100.0, 1.24342113404004053e+02),
        (0.999999, 100.0, 1.82126777119426151e+02),
        (1e-10, 1000.0, 7.41268071719353088e+02), (0.01, 1000.0, 8.98912446929613225e+02),
        (0.5, 1000.0, 9.99333412403380976e+02), (0.95, 1000.0, 1.07467944880344089e+03),
        (0.999999, 1000.0, 1.22715242118727770e+03),
        (1e-10, 10000.0, 9.12651180242325609e+03), (0.01, 10000.0, 9.67394883957763523e+03),
        (0.5, 10000.0, 9.99933334123514396e+03), (0.95, 10000.0, 1.02337488976779368e+04),
        (0.999999, 10000.0, 1.06866898298660381e+04),
    ];

    // (upper-tail p, df, qchisq(p, df, lower.tail = FALSE)) — tails far below
    // what `1 - p` can express, which is what `lower.tail = FALSE` is FOR
    const QU: &[(f64, f64, f64)] = &[
        (1e-20, 0.5, 8.38881937663919217e+01), (1e-100, 0.5, 4.49810824490212497e+02), (0.01, 0.5, 4.86777084439405616e+00),
        (1e-20, 1.0, 8.71617334269098052e+01), (1e-100, 1.0, 4.53943082238798979e+02), (0.01, 1.0, 6.63489660102121270e+00),
        (1e-20, 5.0, 1.03428977237757962e+02), (1e-100, 5.0, 4.76379437064162744e+02), (0.01, 5.0, 1.50862724693889927e+01),
        (1e-20, 100.0, 2.92259133871445613e+02), (1e-100, 100.0, 7.52877564727371805e+02), (0.01, 100.0, 1.35806723171026789e+02),
        (1e-20, 1000.0, 1.47245690210755743e+03), (1e-100, 1000.0, 2.27313605385415758e+03), (0.01, 1000.0, 1.10696899435221735e+03),
    ];

    fn close(got: f64, want: f64, what: &str) {
        let rel = if want == 0.0 { got.abs() } else { ((got - want) / want).abs() };
        assert!(rel <= 1e-12, "{what}: R2 {got:e} vs R {want:e} (rel {rel:.1e})");
    }

    #[test]
    fn pchisq_both_tails_and_dchisq_match_r() {
        for &(x, df, p, q, d) in P {
            close(chi_sq_cdf(x, df), p, &format!("pchisq({x}, {df})"));
            close(chi_sq_sf(x, df), q, &format!("pchisq({x}, {df}, lower.tail = FALSE)"));
            close(dgamma(x, df / 2.0, 2.0, false), d, &format!("dchisq({x}, {df})"));
        }
    }

    #[test]
    fn qchisq_matches_r_in_both_tails() {
        for &(p, df, want) in Q {
            close(chi_sq_quantile(p.ln(), df, true), want, &format!("qchisq({p}, {df})"));
        }
        for &(q, df, want) in QU {
            close(chi_sq_quantile(q.ln(), df, false), want, &format!("qchisq({q}, {df}, lower.tail = FALSE)"));
        }
    }

    #[test]
    fn edges_are_rs() {
        assert_eq!(dgamma(0.0, 0.5, 2.0, false), f64::INFINITY);   // dchisq(0, 1)
        assert_eq!(dgamma(0.0, 1.0, 2.0, false), 0.5);             // dchisq(0, 2)
        assert_eq!(dgamma(0.0, 1.5, 2.0, false), 0.0);             // dchisq(0, 3)
        assert_eq!(chi_sq_cdf(0.0, 3.0), 0.0);
        assert_eq!(chi_sq_sf(0.0, 3.0), 1.0);
        assert_eq!(chi_sq_cdf(f64::INFINITY, 3.0), 1.0);
        assert!(chi_sq_cdf(1.0, -1.0).is_nan());
        assert_eq!(chi_sq_quantile(0.0, 3.0, true), f64::INFINITY);        // p = 1
        assert_eq!(chi_sq_quantile(f64::NEG_INFINITY, 3.0, true), 0.0);    // p = 0
    }
}
