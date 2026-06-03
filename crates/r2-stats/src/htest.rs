//! Hypothesis tests — Phase R.10.
//!
//! Hit-and-get hypothesis tests: caller invokes the function, the
//! formatted result prints to stdout, the return value is a small
//! `RVal::TypeInstance` carrying `statistic`, `p.value`, and so on.
//! Unlike fitted-model functions (`lm`, `glm`, `aov`), these tests do
//! NOT need a separate `summary()` step — the `t.test`/`chisq.test`
//! call is the one-shot interaction.
//!
//! Therefore they migrate as plain pure builtins
//! (`fn(&[EvalArg]) -> Result<RVal, R2Err>`); no split-handler pattern,
//! no EngineCtx, no engine state.
//!
//! Hosted: `t.test`, `chisq.test`, `cor.test`, `shapiro.test`,
//! `wilcox.test`, `fisher.test`. The numerical primitives they share
//! (`t_cdf`, `chi_sq_cdf`, `ln_gamma`, `gamma_approx`, `incomplete_beta`,
//! `fmt_pval`, `signif_stars`) live in this module too and are
//! re-exported for engine-internal callers (`lm` summary print uses
//! `fmt_pval`/`signif_stars`).
//!
//! **t.test status (v0.1.0):** R-style output (data:/CI/alt-hypothesis/
//! sample-estimates), Welch–Satterthwaite df for unequal-variance two-
//! sample, formula syntax `t.test(x ~ y)`, paired test with Pearson r,
//! `id =` named arg for within-subject auto-pairing. p-value uses a
//! trapezoidal-rule incomplete-beta integration (~1e-4 accuracy);
//! closure path to LAPACK-grade is Lentz CF (tracked in KNOWN_LIMITATIONS).
//!
//! **fisher.test status (v0.1.0):** exact hypergeometric (via `lchoose`
//! / `hypergeom_pmf`) replacing the earlier χ² approximation.

use crate::dist::{phi, qnorm_approx};
use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal, TypeInstance};
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

#[inline]
fn rnum(n: f64) -> RVal { RVal::Numeric(vec![Some(n)].into(), Attrs::default()) }

#[inline]
fn rnums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }

#[inline]
fn rstr(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }

#[inline]
fn runtime_err(msg: String) -> R2Err { R2Err { msg, kind: ErrKind::Runtime } }

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

#[inline]
fn fmt_n(n: f64) -> String { r2_types::fmt_num(n) }

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

// ── helpers for the four t.test paths ────────────────────────────────

fn extract_formula(v: &RVal) -> Option<(RVal, RVal)> {
    if let RVal::List(items) = v {
        let is_formula = items.iter().any(|(n, val)| {
            n.as_ref().map(|s| s.as_ref()) == Some("~class")
                && matches!(val, RVal::Character(c, _)
                    if c.first().and_then(|x| x.as_ref()).map(|s| s.as_ref()) == Some("formula"))
        });
        if !is_formula { return None; }
        let lhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
            .map(|(_, v)| v.clone())?;
        let rhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
            .map(|(_, v)| v.clone())?;
        Some((lhs, rhs))
    } else { None }
}

/// Phase R.S.1 — return the `~error` stratum from a formula list, if any.
/// Used by t.test to enable `t.test(y ~ x + Error(subject), paired=T)`
/// as a formula-shaped alias for `t.test(y ~ x, id=subject, paired=T)`.
fn extract_error_stratum(v: &RVal) -> Option<RVal> {
    if let RVal::List(items) = v {
        items.iter()
            .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~error"))
            .map(|(_, val)| val.clone())
    } else { None }
}

/// Phase R.S.1 — pair observations by subject in row-of-appearance order.
///
/// Used for the `t.test(response ~ Error(subject), paired=T)` shortcut
/// where there is no explicit treatment grouping on the RHS. Each
/// subject must have exactly 2 observations; we take the first as
/// "obs1" and the second as "obs2" in original row order. Returns the
/// paired vectors plus the count, so the caller can format a clean
/// data-line for the t-test output.
///
/// Errors if any subject has != 2 observations, since the pairing is
/// otherwise undefined.
fn pair_by_subject_order(
    values: &[f64],
    ids: &[String],
) -> Result<(Vec<f64>, Vec<f64>), R2Err> {
    if values.len() != ids.len() {
        return Err(runtime_err(format!(
            "t.test paired-by-subject-order: values ({}) and subject ({}) must be the same length",
            values.len(), ids.len()
        )));
    }
    let mut per_subject: std::collections::HashMap<String, Vec<f64>> = Default::default();
    let mut order: Vec<String> = Vec::new();
    for (v, id) in values.iter().zip(ids) {
        if !per_subject.contains_key(id) {
            order.push(id.clone());
        }
        per_subject.entry(id.clone()).or_default().push(*v);
    }
    let mut xs = Vec::with_capacity(order.len());
    let mut ys = Vec::with_capacity(order.len());
    for id in &order {
        let obs = &per_subject[id];
        if obs.len() != 2 {
            return Err(runtime_err(format!(
                "t.test paired-by-subject-order: subject '{}' has {} observations, expected exactly 2 (one 'before' and one 'after' in row order)",
                id, obs.len()
            )));
        }
        xs.push(obs[0]);
        ys.push(obs[1]);
    }
    if xs.len() < 2 {
        return Err(runtime_err(format!(
            "t.test paired-by-subject-order: need ≥ 2 subjects (got {})",
            xs.len()
        )));
    }
    Ok((xs, ys))
}

/// Unwrap a stratum value (which arrives as `List([(Some("subject"), <column>)])`
/// from the formula construction code) into the bare column value. Robust to
/// the column being a direct RVal::Character/Factor/Numeric as well.
fn unwrap_stratum_column(v: &RVal) -> RVal {
    match v {
        RVal::List(items) if !items.is_empty() => items[0].1.clone(),
        other => other.clone(),
    }
}

/// Split `values` by the 2-level grouping vector `group`. Returns
/// (group1_label, group1_values, group2_label, group2_values).
fn split_by_group(values: &[f64], group: &RVal) -> Result<(String, Vec<f64>, String, Vec<f64>), R2Err> {
    let group_strs: Vec<String> = match group {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))
                .unwrap_or_else(|| "NA".into())).collect(),
        RVal::Numeric(v, _) => v.iter()
            .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Integer(v, _) => v.iter()
            .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Logical(v, _) => v.iter()
            .map(|x| x.map(|b| if b { "TRUE".into() } else { "FALSE".into() })
                .unwrap_or_else(|| "NA".into())).collect(),
        _ => return Err(runtime_err(
            "t.test formula RHS must be a 2-level grouping vector".into())),
    };
    if group_strs.len() != values.len() {
        return Err(runtime_err(format!(
            "t.test: LHS length ({}) != RHS length ({})", values.len(), group_strs.len())));
    }
    // Discover levels in order of first appearance — matches R's behaviour
    // for character vectors without explicit factor ordering.
    let mut levels: Vec<String> = Vec::new();
    for s in &group_strs {
        if !levels.contains(s) { levels.push(s.clone()); }
    }
    if levels.len() != 2 {
        return Err(runtime_err(format!(
            "t.test formula needs exactly 2 groups, got {}: {:?}", levels.len(), levels)));
    }
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();
    for (val, gs) in values.iter().zip(group_strs.iter()) {
        if gs == &levels[0] { g1.push(*val); }
        else if gs == &levels[1] { g2.push(*val); }
    }
    Ok((levels[0].clone(), g1, levels[1].clone(), g2))
}

/// Pearson correlation between two equal-length slices.
fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return f64::NAN; }
    let nf = n as f64;
    let mx = x.iter().take(n).sum::<f64>() / nf;
    let my = y.iter().take(n).sum::<f64>() / nf;
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx; let dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { f64::NAN }
}

fn welch_two_sample(
    x: &[f64], y: &[f64], lab_x: &str, lab_y: &str,
    conf_level: f64, data_line: &str,
) -> Result<RVal, R2Err> {
    if x.len() < 2 || y.len() < 2 {
        return Err(runtime_err("t.test: each group needs ≥ 2 observations".into()));
    }
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let mx = x.iter().sum::<f64>() / nx;
    let my = y.iter().sum::<f64>() / ny;
    let sx2 = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / (nx - 1.0);
    let sy2 = y.iter().map(|v| (v - my).powi(2)).sum::<f64>() / (ny - 1.0);
    let vx = sx2 / nx;
    let vy = sy2 / ny;
    let se = (vx + vy).sqrt();
    let diff = mx - my;
    let t_stat = diff / se;
    let df = (vx + vy).powi(2) / (vx.powi(2) / (nx - 1.0) + vy.powi(2) / (ny - 1.0));
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));
    let alpha = 1.0 - conf_level;
    let t_crit = qt(1.0 - alpha / 2.0, df);
    let ci_lo = diff - t_crit * se;
    let ci_hi = diff + t_crit * se;
    let conf_pct = (conf_level * 100.0).round() as i64;

    println!("\n\tWelch Two Sample t-test\n");
    println!("data:  {}", data_line);
    println!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    println!("alternative hypothesis: true difference in means is not equal to 0");
    println!("{} percent confidence interval:", conf_pct);
    println!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    println!("sample estimates:");
    println!("mean of {} = {}, mean of {} = {}", lab_x, fmt_n(mx), lab_y, fmt_n(my));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), rnum(t_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("parameter"), rnum(df));
    fields.insert(Arc::from("estimate"), rnums(&[mx, my]));
    fields.insert(Arc::from("conf.int"), rnums(&[ci_lo, ci_hi]));
    fields.insert(Arc::from("conf.level"), rnum(conf_level));
    fields.insert(Arc::from("method"), rstr("Welch Two Sample t-test"));
    fields.insert(Arc::from("group1"), rstr(lab_x));
    fields.insert(Arc::from("group2"), rstr(lab_y));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields }))
}

/// Match observations across two groups by subject `id`. Returns
/// `(x_paired, y_paired, dropped_count)` where each paired vector has
/// one entry per id that appears in both groups exactly once.
///
/// This is the equivalent of R's repeated-measures `Error(id/factor)`
/// extension. R itself doesn't support that in `t.test`; here it's
/// surfaced via the explicit `id =` argument because the engine NSE
/// would otherwise try to evaluate `Error()` as a function.
fn pair_by_id(
    values: &[f64], group_labels: &[String], ids: &[String],
    level1: &str, level2: &str,
) -> Result<(Vec<f64>, Vec<f64>, usize), R2Err> {
    if values.len() != group_labels.len() || values.len() != ids.len() {
        return Err(runtime_err(format!(
            "t.test paired-by-id: values ({}), group ({}), id ({}) must all be the same length",
            values.len(), group_labels.len(), ids.len())));
    }

    // For each id, collect (group, value) pairs.
    let mut per_id: std::collections::BTreeMap<String, (Option<f64>, Option<f64>)> = Default::default();
    for ((v, g), i) in values.iter().zip(group_labels).zip(ids) {
        let entry = per_id.entry(i.clone()).or_insert((None, None));
        if g == level1 {
            if entry.0.is_some() {
                return Err(runtime_err(format!(
                    "t.test paired-by-id: subject '{}' has duplicate '{}' observation", i, level1)));
            }
            entry.0 = Some(*v);
        } else if g == level2 {
            if entry.1.is_some() {
                return Err(runtime_err(format!(
                    "t.test paired-by-id: subject '{}' has duplicate '{}' observation", i, level2)));
            }
            entry.1 = Some(*v);
        }
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut dropped = 0;
    for (_, (a, b)) in per_id {
        match (a, b) {
            (Some(va), Some(vb)) => { xs.push(va); ys.push(vb); }
            _ => dropped += 1,
        }
    }
    if xs.len() < 2 {
        return Err(runtime_err(format!(
            "t.test paired-by-id: need ≥ 2 subjects with both '{}' and '{}' observations (got {})",
            level1, level2, xs.len())));
    }
    Ok((xs, ys, dropped))
}

/// Coerce an `id` argument to a vector of strings (for set-membership
/// matching). Accepts Character, Factor, Integer, Numeric.
fn id_to_strings(v: &RVal) -> Option<Vec<String>> {
    match v {
        RVal::Character(c, _) => Some(c.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into())).collect()),
        RVal::Factor(f) => Some(f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))
                .unwrap_or_else(|| "NA".into())).collect()),
        RVal::Integer(v, _) => Some(v.iter()
            .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect()),
        RVal::Numeric(v, _) => Some(v.iter()
            .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect()),
        _ => None,
    }
}

/// Paired t-test on `(x[i], y[i])` pairs. Reports the Pearson correlation
/// between the paired observations alongside the standard test fields —
/// a small extension over R's `t.test(..., paired=TRUE)` output, useful
/// for within-subject designs where the strength of pairing matters.
fn paired_t_test(
    x: &[f64], y: &[f64], lab_x: &str, lab_y: &str, mu: f64,
    conf_level: f64, data_line: &str,
) -> Result<RVal, R2Err> {
    if x.len() != y.len() {
        return Err(runtime_err(format!(
            "t.test paired: x and y must be the same length ({} vs {})", x.len(), y.len())));
    }
    let n = x.len();
    if n < 2 { return Err(runtime_err("t.test paired: need ≥ 2 pairs".into())); }
    let nf = n as f64;

    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    let mean_d = d.iter().sum::<f64>() / nf;
    let sd_d = (d.iter().map(|v| (v - mean_d).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    let se = sd_d / nf.sqrt();
    let t_stat = (mean_d - mu) / se;
    let df = nf - 1.0;
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));
    let alpha = 1.0 - conf_level;
    let t_crit = qt(1.0 - alpha / 2.0, df);
    let ci_lo = mean_d - t_crit * se;
    let ci_hi = mean_d + t_crit * se;
    let conf_pct = (conf_level * 100.0).round() as i64;

    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().sum::<f64>() / nf;
    let cor = pearson_r(x, y);

    println!("\n\tPaired t-test\n");
    println!("data:  {}", data_line);
    println!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    println!("alternative hypothesis: true mean difference is not equal to {}", fmt_n(mu));
    println!("{} percent confidence interval:", conf_pct);
    println!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    println!("sample estimates:");
    println!("mean of {} = {}, mean of {} = {}", lab_x, fmt_n(mx), lab_y, fmt_n(my));
    println!("mean of differences ({} - {}) = {}", lab_x, lab_y, fmt_n(mean_d));
    println!("correlation between pairs (Pearson r) = {}", fmt_n(cor));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), rnum(t_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("parameter"), rnum(df));
    fields.insert(Arc::from("estimate"), rnum(mean_d));
    fields.insert(Arc::from("conf.int"), rnums(&[ci_lo, ci_hi]));
    fields.insert(Arc::from("conf.level"), rnum(conf_level));
    fields.insert(Arc::from("cor"), rnum(cor));
    fields.insert(Arc::from("method"), rstr("Paired t-test"));
    fields.insert(Arc::from("group1"), rstr(lab_x));
    fields.insert(Arc::from("group2"), rstr(lab_y));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields }))
}

fn one_sample_t_test(
    x: &[f64], lab: &str, mu: f64, conf_level: f64,
) -> Result<RVal, R2Err> {
    if x.len() < 2 {
        return Err(runtime_err("t.test: need ≥ 2 observations".into()));
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let se = sd / n.sqrt();
    let t_stat = (mean - mu) / se;
    let df = n - 1.0;
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));
    let alpha = 1.0 - conf_level;
    let t_crit = qt(1.0 - alpha / 2.0, df);
    let ci_lo = mean - t_crit * se;
    let ci_hi = mean + t_crit * se;
    let conf_pct = (conf_level * 100.0).round() as i64;

    println!("\n\tOne Sample t-test\n");
    println!("data:  {}", lab);
    println!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    println!("alternative hypothesis: true mean is not equal to {}", fmt_n(mu));
    println!("{} percent confidence interval:", conf_pct);
    println!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    println!("sample estimates:");
    println!("mean of {} = {}", lab, fmt_n(mean));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), rnum(t_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("parameter"), rnum(df));
    fields.insert(Arc::from("estimate"), rnum(mean));
    fields.insert(Arc::from("conf.int"), rnums(&[ci_lo, ci_hi]));
    fields.insert(Arc::from("conf.level"), rnum(conf_level));
    fields.insert(Arc::from("method"), rstr("One Sample t-test"));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields }))
}

/// `t.test(x [, y] [, mu=] [, paired=] [, id=])` — one/two-sample/paired.
///
/// Accepted call shapes:
///   • `t.test(x)`                  — one-sample against `mu` (default 0).
///   • `t.test(x, y)`               — two-sample Welch.
///   • `t.test(x, y, paired=TRUE)`  — paired test on (x[i], y[i]) diffs.
///                                    Output also reports Pearson r between
///                                    the paired observations.
///   • `t.test(value ~ group)`      — formula form: split `value` by the
///                                    2-level `group` vector. Labels appear
///                                    in printed output and as `$group1`/
///                                    `$group2`.
///   • `t.test(value ~ group,        — within-subject auto-pairing: matches
///       id = subject,                 observations across the two `group`
///       paired = TRUE)`               levels by `subject` id, then runs a
///                                    paired test. Subjects without one
///                                    observation in each group are dropped
///                                    with a printed count.
///                                    (R uses `Error(subject/group)` in
///                                    aov() for this; t.test in R doesn't
///                                    support it. Here `id =` provides the
///                                    same capability with a syntax the
///                                    formula parser already handles.)
pub fn bi_t_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mu = arg_named(a, "mu").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);
    let paired = arg_named(a, "paired").and_then(|v| v.as_logicals().ok())
        .and_then(|v| v.first().copied().flatten())
        .unwrap_or(false);
    let conf_level = arg_named(a, "conf.level")
        .and_then(|v| v.scalar_f64().ok().flatten())
        .unwrap_or(0.95);
    if !(0.0 < conf_level && conf_level < 1.0) {
        return Err(runtime_err(format!(
            "t.test: conf.level must be in (0, 1), got {}", conf_level)));
    }
    let id_arg = arg_named(a, "id");

    // Formula form: t.test(value ~ group)
    if let Some((lhs, rhs)) = extract_formula(&first(a)) {
        // Phase R.S.1 — when the formula came in via the `data=df` path,
        // resolve_formula_term wraps each side as `List([(name, col)])`.
        // Unwrap so as_reals()/split_by_group see the raw column. The
        // non-data form (where eval_in gave back the raw vector directly)
        // is unchanged because unwrap_stratum_column is a no-op for it.
        let lhs = unwrap_stratum_column(&lhs);
        let rhs = unwrap_stratum_column(&rhs);
        let values: Vec<f64> = lhs.as_reals()?.into_iter().filter_map(|v| v).collect();

        // Phase R.S.1 — `t.test(y ~ Error(subject), paired=T)` shortcut.
        // When the formula RHS has no treatment grouping (just an Error
        // stratum), pair observations by subject in row-of-appearance
        // order: each subject must have exactly 2 observations; the
        // first becomes "obs1" and the second becomes "obs2". Cleaner
        // than asking the user to manually split into x and y vectors.
        let error_stratum_for_order = extract_error_stratum(&first(a))
            .map(|v| unwrap_stratum_column(&v));
        let rhs_is_null = matches!(rhs, RVal::Null);
        if rhs_is_null && error_stratum_for_order.is_some() {
            if !paired {
                return Err(runtime_err(
                    "t.test(y ~ Error(subject)) requires paired=TRUE — without paired, there are no groups to compare".into(),
                ));
            }
            let ids = id_to_strings(error_stratum_for_order.as_ref().unwrap())
                .ok_or_else(|| runtime_err(
                    "t.test: Error(...) stratum must be Character/Factor/Integer/Numeric".into()))?;
            let (xs, ys) = pair_by_subject_order(&values, &ids)?;
            let dl = format!("response paired by subject row-order (n = {})", xs.len());
            return paired_t_test(&xs, &ys, "obs1", "obs2", mu, conf_level, &dl);
        }

        let (lab1, g1, lab2, g2) = split_by_group(&values, &rhs)?;
        let data_line = format!("values by group ({} vs {})", lab1, lab2);

        // Phase R.S.1 — Error(subject) inside the formula RHS acts as an
        // implicit id= argument when paired=TRUE. Explicit id= still wins
        // if both are supplied.
        let error_id = if id_arg.is_some() { None } else {
            extract_error_stratum(&first(a)).map(|v| unwrap_stratum_column(&v))
        };

        if paired {
            let id_source = id_arg.or(error_id);
            if let Some(id_val) = id_source {
                let ids = id_to_strings(&id_val)
                    .ok_or_else(|| runtime_err(
                        "t.test: id= must be Character/Factor/Integer/Numeric".into()))?;
                let group_strs = match &rhs {
                    RVal::Character(v, _) => v.iter()
                        .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into()))
                        .collect::<Vec<_>>(),
                    RVal::Factor(f) => f.codes.iter()
                        .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))
                            .unwrap_or_else(|| "NA".into())).collect(),
                    RVal::Numeric(v, _) => v.iter()
                        .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect(),
                    RVal::Integer(v, _) => v.iter()
                        .map(|x| x.map(|n| format!("{}", n)).unwrap_or_else(|| "NA".into())).collect(),
                    RVal::Logical(v, _) => v.iter()
                        .map(|x| x.map(|b| if b { "TRUE".into() } else { "FALSE".into() })
                            .unwrap_or_else(|| "NA".into())).collect(),
                    _ => return Err(runtime_err(
                        "t.test paired-by-id: group vector type unsupported".into())),
                };
                let (xp, yp, dropped) = pair_by_id(&values, &group_strs, &ids, &lab1, &lab2)?;
                if dropped > 0 {
                    println!("# t.test paired-by-id: dropped {} subject(s) without both '{}' and '{}' observations",
                        dropped, lab1, lab2);
                }
                let dl = format!("{} (paired by id, n = {})", data_line, xp.len());
                return paired_t_test(&xp, &yp, &lab1, &lab2, mu, conf_level, &dl);
            }
            return paired_t_test(&g1, &g2, &lab1, &lab2, mu, conf_level, &data_line);
        }
        return welch_two_sample(&g1, &g2, &lab1, &lab2, conf_level, &data_line);
    }

    let x: Vec<f64> = first(a).as_reals()?.into_iter().filter_map(|v| v).collect();
    let two_sample = a.len() >= 2 && a[1].name.is_none();
    if two_sample {
        let y: Vec<f64> = nth(a, 1).as_reals()?.into_iter().filter_map(|v| v).collect();
        if paired {
            return paired_t_test(&x, &y, "x", "y", mu, conf_level, "x and y");
        }
        return welch_two_sample(&x, &y, "x", "y", conf_level, "x and y");
    }
    one_sample_t_test(&x, "x", mu, conf_level)
}

pub fn bi_chisq_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first(a) {
        RVal::Matrix(mat) => {
            let (nr, nc) = (mat.nrow, mat.ncol);
            let n: f64 = mat.data.iter().sum();
            let correct = arg_named(a, "correct").and_then(|v| v.as_logicals().ok())
                .map(|v| v.first().copied().flatten() == Some(true))
                .unwrap_or(nr == 2 && nc == 2);
            let row_totals: Vec<f64> = (0..nr).map(|r| (0..nc).map(|c| mat.get(r, c)).sum()).collect();
            let col_totals: Vec<f64> = (0..nc).map(|c| (0..nr).map(|r| mat.get(r, c)).sum()).collect();
            let mut chi_sq = 0.0;
            for r in 0..nr {
                for c in 0..nc {
                    let observed = mat.get(r, c);
                    let expected = row_totals[r] * col_totals[c] / n;
                    if expected > 0.0 {
                        let diff = if correct { (observed - expected).abs() - 0.5 } else { observed - expected };
                        chi_sq += diff.max(0.0).powi(2) / expected;
                    }
                }
            }
            let df = ((nr - 1) * (nc - 1)) as f64;
            let p_value = 1.0 - chi_sq_cdf(chi_sq, df);
            let method = if correct { "Pearson's Chi-squared test with Yates' continuity correction" }
                         else { "Pearson's Chi-squared test" };
            println!("\n  {}\n", method);
            println!("X-squared = {}, df = {}, p-value = {}", fmt_n(chi_sq), df as i32, fmt_pval(p_value));

            let mut fields = HashMap::new();
            fields.insert(Arc::from("statistic"), rnum(chi_sq));
            fields.insert(Arc::from("p.value"), rnum(p_value));
            fields.insert(Arc::from("parameter"), rnum(df));
            fields.insert(Arc::from("method"), rstr(method));
            Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields }))
        }
        v => {
            let obs: Vec<f64> = v.as_reals()?.into_iter().filter_map(|x| x).collect();
            let k = obs.len();
            let total: f64 = obs.iter().sum();
            let probs: Vec<f64> = arg_named(a, "p").and_then(|v| v.as_reals().ok())
                .map(|v| v.into_iter().filter_map(|x| x).collect())
                .unwrap_or_else(|| vec![1.0 / k as f64; k]);
            if probs.len() != k {
                return Err(runtime_err("chisq.test: length of p must equal length of x".into()));
            }
            let p_sum: f64 = probs.iter().sum();
            let expected: Vec<f64> = probs.iter().map(|p| total * p / p_sum).collect();
            let chi_sq: f64 = obs.iter().zip(expected.iter())
                .map(|(o, e)| if *e > 0.0 { (o - e).powi(2) / e } else { 0.0 }).sum();
            let df = (k - 1) as f64;
            let p_value = 1.0 - chi_sq_cdf(chi_sq, df);
            println!("\n  Chi-squared test for given probabilities\n");
            println!("X-squared = {}, df = {}, p-value = {}", fmt_n(chi_sq), df as i32, fmt_pval(p_value));

            let mut fields = HashMap::new();
            fields.insert(Arc::from("statistic"), rnum(chi_sq));
            fields.insert(Arc::from("p.value"), rnum(p_value));
            fields.insert(Arc::from("parameter"), rnum(df));
            fields.insert(Arc::from("method"), rstr("Chi-squared test for given probabilities"));
            Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields }))
        }
    }
}

pub fn bi_cor_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = first(a).as_reals()?.into_iter().filter_map(|v| v).collect();
    let y: Vec<f64> = nth(a, 1).as_reals()?.into_iter().filter_map(|v| v).collect();
    let n = x.len().min(y.len());
    if n < 3 { return Err(runtime_err("cor.test needs at least 3 observations".into())); }

    let mx = x.iter().take(n).sum::<f64>() / n as f64;
    let my = y.iter().take(n).sum::<f64>() / n as f64;
    let mut sxy = 0.0; let mut sxx = 0.0; let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx; let dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    let r = if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { 0.0 };
    let df = (n - 2) as f64;
    let t_stat = if (1.0 - r * r).abs() > 1e-15 { r * (df / (1.0 - r * r)).sqrt() } else { f64::INFINITY };
    let p_value = 2.0 * (1.0 - phi(t_stat.abs()));

    println!("\n  Pearson's product-moment correlation\n");
    println!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), n - 2, fmt_pval(p_value));
    println!("alternative hypothesis: true correlation is not equal to 0");
    println!("sample estimate:");
    println!("      cor");
    println!("{:>9}", fmt_n(r));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("estimate"), rnum(r));
    fields.insert(Arc::from("statistic"), rnum(t_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("df"), rnum(df));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("cor.test"), fields }))
}

pub fn bi_shapiro_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut x: Vec<f64> = first(a).as_reals()?.into_iter().filter_map(|v| v).collect();
    let n = x.len();
    if n < 3 { return Err(runtime_err("shapiro.test needs at least 3 observations".into())); }
    if n > 5000 { return Err(runtime_err("shapiro.test: sample size must be <= 5000".into())); }

    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f64> = (0..n).map(|i| {
        let p = (i as f64 + 0.375) / (n as f64 + 0.25);
        qnorm_approx(p)
    }).collect();

    let mean = x.iter().sum::<f64>() / n as f64;
    let ss = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    let me = expected.iter().sum::<f64>() / n as f64;
    let mut sxe = 0.0; let mut see = 0.0;
    for i in 0..n {
        sxe += (x[i] - mean) * (expected[i] - me);
        see += (expected[i] - me).powi(2);
    }
    let w = if ss > 0.0 && see > 0.0 { (sxe * sxe) / (ss * see) } else { 1.0 };

    let ln_n = (n as f64).ln();
    let ln_w = (1.0 - w).max(1e-15).ln();
    let mu = 0.0038915 * ln_n.powi(3) - 0.083751 * ln_n.powi(2) - 0.31082 * ln_n - 1.5861;
    let sigma = (0.0030302 * ln_n.powi(2) - 0.082676 * ln_n - 0.4803).exp();
    let z = (ln_w - mu) / sigma;
    let p_value = (1.0 - phi(z)).clamp(0.0, 1.0);

    println!("\n  Shapiro-Wilk normality test\n");
    println!("W = {}, p-value = {}", fmt_n(w), fmt_pval(p_value));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), rnum(w));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("shapiro.test"), fields }))
}

pub fn bi_wilcox_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = first(a).as_reals()?.into_iter().filter_map(|v| v).collect();
    let y_raw = arg_named(a, "y").or(Some(nth(a, 1)));
    let mu = arg_named(a, "mu").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);

    let n = x.len();
    if n < 2 { return Err(runtime_err("wilcox.test needs at least 2 observations".into())); }

    if let Some(y_val) = &y_raw {
        if let Ok(y_reals) = y_val.as_reals() {
            let y: Vec<f64> = y_reals.into_iter().filter_map(|v| v).collect();
            if !y.is_empty() {
                let m = y.len();
                let mut combined: Vec<(f64, bool)> = Vec::new();
                for v in &x { combined.push((*v, true)); }
                for v in &y { combined.push((*v, false)); }
                combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let _total = combined.len();
                let rank_sum_x: f64 = combined.iter().enumerate()
                    .filter(|(_, (_, is_x))| *is_x)
                    .map(|(i, _)| (i + 1) as f64)
                    .sum();

                let u = rank_sum_x - (n * (n + 1)) as f64 / 2.0;
                let mean_u = (n * m) as f64 / 2.0;
                let sd_u = ((n * m * (n + m + 1)) as f64 / 12.0).sqrt();
                let z = if sd_u > 0.0 { (u - mean_u) / sd_u } else { 0.0 };
                let p_value = 2.0 * (1.0 - phi(z.abs()));

                println!("\n  Wilcoxon rank sum test\n");
                println!("W = {}, p-value = {}", fmt_n(u), fmt_pval(p_value));
                println!("alternative hypothesis: true location shift is not equal to 0");

                let mut fields = HashMap::new();
                fields.insert(Arc::from("statistic"), rnum(u));
                fields.insert(Arc::from("p.value"), rnum(p_value));
                return Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("wilcox.test"), fields }));
            }
        }
    }

    let diffs: Vec<f64> = x.iter().map(|v| v - mu).filter(|d| d.abs() > 1e-15).collect();
    let nd = diffs.len();
    if nd < 2 { return Err(runtime_err("wilcox.test: not enough non-zero differences".into())); }

    let mut abs_diffs: Vec<(f64, f64)> = diffs.iter().map(|d| (d.abs(), d.signum())).collect();
    abs_diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let w_plus: f64 = abs_diffs.iter().enumerate()
        .filter(|(_, (_, sign))| *sign > 0.0)
        .map(|(i, _)| (i + 1) as f64).sum();

    let mean_w = (nd * (nd + 1)) as f64 / 4.0;
    let sd_w = ((nd * (nd + 1) * (2 * nd + 1)) as f64 / 24.0).sqrt();
    let z = if sd_w > 0.0 { (w_plus - mean_w) / sd_w } else { 0.0 };
    let p_value = 2.0 * (1.0 - phi(z.abs()));

    println!("\n  Wilcoxon signed rank test\n");
    println!("V = {}, p-value = {}", fmt_n(w_plus), fmt_pval(p_value));
    println!("alternative hypothesis: true location is not equal to {}", mu);

    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), rnum(w_plus));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("wilcox.test"), fields }))
}

/// log C(n, k) via log-gamma. Returns -∞ when k > n or k < 0.
fn lchoose(n: i64, k: i64) -> f64 {
    if k < 0 || k > n { return f64::NEG_INFINITY; }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

/// Hypergeometric PMF for 2×2 tables with fixed margins.
/// `k` = count in cell (0,0); other cells determined by row/col totals.
/// P(X=k) = C(n1, k) · C(n2, m1-k) / C(n, m1)
fn hypergeom_pmf(k: i64, n1: i64, n2: i64, m1: i64) -> f64 {
    let n = n1 + n2;
    let log_p = lchoose(n1, k) + lchoose(n2, m1 - k) - lchoose(n, m1);
    if log_p.is_finite() { log_p.exp() } else { 0.0 }
}

pub fn bi_fisher_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mat = match &first(a) {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(runtime_err("fisher.test needs a 2x2 matrix".into())),
    };
    if mat.nrow != 2 || mat.ncol != 2 {
        return Err(runtime_err("fisher.test needs a 2x2 matrix".into()));
    }

    let aa = mat.get(0, 0).round() as i64;
    let bb = mat.get(0, 1).round() as i64;
    let cc = mat.get(1, 0).round() as i64;
    let dd = mat.get(1, 1).round() as i64;
    if aa < 0 || bb < 0 || cc < 0 || dd < 0 {
        return Err(runtime_err("fisher.test: counts must be non-negative".into()));
    }

    // Sample odds ratio for the report (NOT the conditional MLE — matches
    // R's `estimate` field semantics for fisher.test 2x2).
    let or = if bb > 0 && cc > 0 {
        (aa as f64 * dd as f64) / (bb as f64 * cc as f64)
    } else {
        f64::INFINITY
    };

    // Fixed-margins exact test. Cell (0,0) ~ Hypergeometric(n, m1, n1).
    let n1 = aa + bb;        // row 0 total
    let n2 = cc + dd;        // row 1 total
    let m1 = aa + cc;        // col 0 total
    let k_min = (m1 - n2).max(0);
    let k_max = m1.min(n1);

    // Two-sided p: sum over the conditional distribution of all outcomes
    // at least as extreme as observed (P(X=k) <= P(X=aa)). This is R's
    // default `alternative = "two.sided"` semantics.
    let p_obs = hypergeom_pmf(aa, n1, n2, m1);
    // Tolerance trims floating-point ties that should be counted in.
    let tol = 1e-7 * p_obs.max(1.0);
    let mut p_value = 0.0_f64;
    for k in k_min..=k_max {
        let p_k = hypergeom_pmf(k, n1, n2, m1);
        if p_k <= p_obs + tol {
            p_value += p_k;
        }
    }
    let p_value = p_value.clamp(0.0, 1.0);

    println!("\n  Fisher's Exact Test for Count Data\n");
    println!("p-value = {}", fmt_pval(p_value));
    println!("alternative hypothesis: true odds ratio is not equal to 1");
    println!("sample estimate:");
    println!("odds ratio: {}", fmt_n(or));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("estimate"), rnum(or));
    fields.insert(Arc::from("method"), rstr("Fisher's Exact Test for Count Data"));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("fisher.test"), fields }))
}

#[cfg(test)]
mod test_suite {
    use super::*;

    fn nums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    fn chs(items: &[&str]) -> RVal {
        RVal::Character(items.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }

    fn formula(lhs: RVal, rhs: RVal) -> RVal {
        RVal::List(vec![
            (Some(Arc::from("~lhs")), lhs),
            (Some(Arc::from("~rhs")), rhs),
            (Some(Arc::from("~class")), RVal::Character(vec![Some(Arc::from("formula"))], Attrs::default())),
        ])
    }

    #[test]
    fn t_test_formula_splits_by_two_level_group() {
        // t.test(c(1,2,3,10,20,30) ~ c("a","a","a","b","b","b"))
        // R: Welch t-test, equivalent to t.test(c(1,2,3), c(10,20,30))
        let values = nums(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let groups = chs(&["a", "a", "a", "b", "b", "b"]);
        let r = bi_t_test(&[evarg(formula(values, groups))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let g1 = inst.fields.get("group1").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                let g2 = inst.fields.get("group2").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(g1, "a");
                assert_eq!(g2, "b");
                let est = inst.fields.get("estimate").unwrap();
                let means = est.as_reals().unwrap();
                assert!((means[0].unwrap() - 2.0).abs() < 1e-12);
                assert!((means[1].unwrap() - 20.0).abs() < 1e-12);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_formula_with_id_pairs_within_subject() {
        // 4 subjects, each measured pre and post. Strong increase post.
        // value: 10, 12, 8, 11, 11, 14, 9, 13   (alternating subj/time)
        // time:  pre, post, pre, post, pre, post, pre, post
        // subj:  s1, s1,  s2, s2,  s3, s3,  s4, s4
        let values = nums(&[10.0, 12.0, 8.0, 11.0, 11.0, 14.0, 9.0, 13.0]);
        let times  = chs(&["pre","post","pre","post","pre","post","pre","post"]);
        let subj   = chs(&["s1","s1","s2","s2","s3","s3","s4","s4"]);
        let r = bi_t_test(&[
            evarg(formula(values, times)),
            EvalArg { name: Some(Arc::from("id")), value: subj },
            EvalArg { name: Some(Arc::from("paired")),
                      value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) },
        ]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let method = inst.fields.get("method").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(method, "Paired t-test");
                // df = n_subjects - 1 = 3
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 3.0).abs() < 1e-12, "expected df=3, got {}", df);
                // Mean of (pre - post) differences. With pairs
                // (10,12) (8,11) (11,14) (9,13), diffs = -2, -3, -3, -4 → mean = -3.
                let est = inst.fields.get("estimate")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((est - (-3.0)).abs() < 1e-12, "expected mean diff=-3, got {}", est);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_paired_reports_pearson_r_and_uses_n_minus_1_df() {
        // Strongly-correlated paired data: y = 2x + small noise.
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[2.1, 4.05, 5.95, 8.1, 9.9]);  // ~2x
        let r = bi_t_test(&[
            evarg(x), evarg(y),
            EvalArg { name: Some(Arc::from("paired")), value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) },
        ]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                // Paired df = n - 1 = 4.
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 4.0).abs() < 1e-12, "paired df should be n-1 = 4, got {}", df);
                // Pearson r should be near 1.0 (y ≈ 2x).
                let cor = inst.fields.get("cor")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(cor > 0.999, "expected r ≈ 1, got {}", cor);
                // Method label flips to "Paired t-test".
                let method = inst.fields.get("method").and_then(|v| match v {
                    RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
                    _ => None,
                }).unwrap();
                assert_eq!(method, "Paired t-test");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_two_sample_uses_welch_df_not_pooled() {
        // x ~ tight cluster, y ~ wider — unequal variances.
        // R: t.test(c(1,2,3,4,5), c(10,20,30))$parameter ≈ 2.0602 (Welch df)
        // Pooled df would be n1+n2-2 = 6 — much larger and wrong.
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[10.0, 20.0, 30.0]);
        let r = bi_t_test(&[evarg(x), evarg(y)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let df = inst.fields.get("parameter")
                    .and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((df - 2.0602).abs() < 1e-3,
                    "Welch df should match R's 2.0602, got {} (pooled would be 6)", df);
                // Sanity: must NOT be the pooled value.
                assert!(df < 5.0, "df {} looks like pooled n1+n2-2", df);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_test_one_sample_zero_mean_against_zero() {
        // x ~ centered, mu=0 → t ≈ 0, p ≈ 1.
        let r = bi_t_test(&[evarg(nums(&[-1.0, 0.0, 1.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p > 0.5, "expected p > 0.5, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn cor_test_perfect_correlation_p_zero() {
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let r = bi_cor_test(&[evarg(x), evarg(y)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let est = inst.fields.get("estimate").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!((est - 1.0).abs() < 1e-12, "estimate should be 1, got {}", est);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn chisq_test_uniform_observations_high_p() {
        // Equal observations → chi² = 0 → p ≈ 1.
        let r = bi_chisq_test(&[evarg(nums(&[10.0, 10.0, 10.0, 10.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p > 0.99, "expected p ≈ 1, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn t_cdf_uses_beta_identity_no_normal_shortcut() {
        // The pre-R.10 implementation switched to a normal approximation
        // for df > 30, which gave ~5e-3 absolute error at moderate df.
        // The new code routes through `incomplete_beta` for all df.
        // R: pt(1.96, df=30) = 0.97032884. Lentz CF reaches ~1e-9.
        // (The pre-Lentz test used an imprecise 0.9703358 reference that
        // only survived because the trapezoidal rule ran at 2e-3.)
        let p = t_cdf(1.96, 30.0);
        assert!((p - 0.97032884).abs() < 1e-6, "got {}", p);
        // Reflection symmetry across t = 0 holds exactly.
        let p_neg = t_cdf(-1.96, 10.0);
        let p_pos = t_cdf(1.96, 10.0);
        assert!(((p_neg + p_pos) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn incomplete_beta_exact_via_lentz() {
        // pbeta(0.5, 2, 3) = 0.6875 exactly; Lentz CF is accurate to ~1e-12.
        let v = incomplete_beta(2.0, 3.0, 0.5);
        assert!((v - 0.6875).abs() < 1e-10, "got {}", v);
        // Symmetry I_x(a,b) = 1 - I_{1-x}(b,a), incl. the b < 1 path
        // (the case t_cdf exercises with b = 0.5).
        let lhs = incomplete_beta(0.5, 3.0, 0.3);
        let rhs = 1.0 - incomplete_beta(3.0, 0.5, 0.7);
        assert!((lhs - rhs).abs() < 1e-12, "symmetry: {} vs {}", lhs, rhs);
    }

    #[test]
    fn f_sf_matches_closed_form_and_handles_infinity() {
        // For df1 = 2 the F survival has the closed form
        // S(f) = (1 + f/3)^(-3) at df2 = 6.
        let f = 12.39623_f64;
        let expected = (1.0 + f / 3.0).powi(-3);
        let got = f_sf(f, 2.0, 6.0);
        assert!((got - expected).abs() < 1e-9, "f_sf {} vs closed form {}", got, expected);
        // f = +∞ (zero residual) → p = 0, not the old approximation's p = 1.
        assert_eq!(f_sf(f64::INFINITY, 2.0, 6.0), 0.0);
        // f = 0 → whole mass above → p = 1.
        assert_eq!(f_sf(0.0, 3.0, 10.0), 1.0);
    }

    #[test]
    fn fisher_test_classic_2x2_matches_r() {
        // R: fisher.test(matrix(c(3,1,1,3), nrow=2))$p.value ≈ 0.4857
        // Exact two-sided hypergeometric.
        use r2_types::Matrix;
        let m = Matrix::new(vec![3.0, 1.0, 1.0, 3.0], 2, 2);
        let r = bi_fisher_test(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                // R reports 0.4857; our sum-of-equally-or-less-likely tail
                // gives the same family of outcomes. Allow ±0.02.
                assert!((p - 0.4857).abs() < 0.02, "got p = {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn fisher_test_independent_zero_off_diag() {
        // 2x2 with one zero cell: c(8,0,0,8). R: p = 0.0001554...
        use r2_types::Matrix;
        let m = Matrix::new(vec![8.0, 0.0, 0.0, 8.0], 2, 2);
        let r = bi_fisher_test(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").and_then(|v| v.scalar_f64().ok().flatten()).unwrap();
                assert!(p < 0.001, "expected very small p, got {}", p);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn shapiro_test_returns_w_and_p() {
        let r = bi_shapiro_test(&[evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0]))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                assert!(inst.fields.contains_key("statistic"));
                assert!(inst.fields.contains_key("p.value"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn fmt_pval_scales() {
        assert_eq!(fmt_pval(1e-20), "<2e-16");
        assert!(fmt_pval(0.0001).contains("e"));
        assert_eq!(fmt_pval(0.05), "0.05");
        assert_eq!(fmt_pval(1.0), "1");
    }

    #[test]
    fn signif_stars_thresholds() {
        assert_eq!(signif_stars(0.0001), "***");
        assert_eq!(signif_stars(0.005), "**");
        assert_eq!(signif_stars(0.02), "*");
        assert_eq!(signif_stars(0.08), ".");
        assert_eq!(signif_stars(0.5), " ");
    }
}
