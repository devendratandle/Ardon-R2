//! Statistical primitives — p-value formatting, log-gamma, incomplete
//! beta, and the t / F / chi-squared CDFs used by the hypothesis tests.

use crate::dist::phi;

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

/// χ² CDF for x ≥ 0, df > 0. Closed forms for df=1, df=2; series for the
/// rest via the regularised lower incomplete gamma.
pub fn chi_sq_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if df <= 0.0 { return 1.0; }
    if (df - 2.0).abs() < 0.01 { return 1.0 - (-x / 2.0).exp(); }
    if (df - 1.0).abs() < 0.01 { return 2.0 * phi(x.sqrt()) - 1.0; }
    let a = df / 2.0;
    let z = x / 2.0;
    if z > a + 50.0 { return 1.0; }
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..500 {
        term *= z / (a + k as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() { break; }
    }
    let log_result = -z + a * z.ln() - ln_gamma(a + 1.0) + sum.ln();
    if log_result > 0.0 { 1.0 } else { log_result.exp().min(1.0) }
}
