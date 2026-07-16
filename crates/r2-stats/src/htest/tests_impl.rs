//! Hypothesis-test builtins (t / chisq / cor / shapiro / wilcox /
//! fisher) and their formula/grouping/pairing helpers.

use super::*;
use r2_types::{EvalArg, R2Err, RVal, TypeInstance};
use std::collections::HashMap;
use std::sync::Arc;
use crate::dist::{phi_upper, qnorm_approx};

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

/// Pearson correlation between two equal-length slices — delegates to
/// the shared centred-moment primitive so cor.test, cor(), and cor(X)
/// agree numerically by construction.
fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    if x.len().min(y.len()) < 2 { return f64::NAN; }
    let (_, _, _, sxx, syy, sxy) = crate::moments::centred2_dense(x, y);
    crate::moments::pearson_from(sxx, syy, sxy)
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

    soutln!("\n\tWelch Two Sample t-test\n");
    soutln!("data:  {}", data_line);
    soutln!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    soutln!("alternative hypothesis: true difference in means is not equal to 0");
    soutln!("{} percent confidence interval:", conf_pct);
    soutln!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    soutln!("sample estimates:");
    soutln!("mean of {} = {}, mean of {} = {}", lab_x, fmt_n(mx), lab_y, fmt_n(my));

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

    soutln!("\n\tPaired t-test\n");
    soutln!("data:  {}", data_line);
    soutln!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    soutln!("alternative hypothesis: true mean difference is not equal to {}", fmt_n(mu));
    soutln!("{} percent confidence interval:", conf_pct);
    soutln!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    soutln!("sample estimates:");
    soutln!("mean of {} = {}, mean of {} = {}", lab_x, fmt_n(mx), lab_y, fmt_n(my));
    soutln!("mean of differences ({} - {}) = {}", lab_x, lab_y, fmt_n(mean_d));
    soutln!("correlation between pairs (Pearson r) = {}", fmt_n(cor));

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

    soutln!("\n\tOne Sample t-test\n");
    soutln!("data:  {}", lab);
    soutln!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), fmt_n(df), fmt_n(p_value));
    soutln!("alternative hypothesis: true mean is not equal to {}", fmt_n(mu));
    soutln!("{} percent confidence interval:", conf_pct);
    soutln!("  {}  {}", fmt_n(ci_lo), fmt_n(ci_hi));
    soutln!("sample estimates:");
    soutln!("mean of {} = {}", lab, fmt_n(mean));

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
                    soutln!("# t.test paired-by-id: dropped {} subject(s) without both '{}' and '{}' observations",
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
            soutln!("\n  {}\n", method);
            soutln!("X-squared = {}, df = {}, p-value = {}", fmt_n(chi_sq), df as i32, fmt_pval(p_value));

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
            soutln!("\n  Chi-squared test for given probabilities\n");
            soutln!("X-squared = {}, df = {}, p-value = {}", fmt_n(chi_sq), df as i32, fmt_pval(p_value));

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
    let p_value = 2.0 * phi_upper(t_stat.abs());

    soutln!("\n  Pearson's product-moment correlation\n");
    soutln!("t = {}, df = {}, p-value = {}", fmt_n(t_stat), n - 2, fmt_pval(p_value));
    soutln!("alternative hypothesis: true correlation is not equal to 0");
    soutln!("sample estimate:");
    soutln!("      cor");
    soutln!("{:>9}", fmt_n(r));

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
    let p_value = phi_upper(z).clamp(0.0, 1.0);

    soutln!("\n  Shapiro-Wilk normality test\n");
    soutln!("W = {}, p-value = {}", fmt_n(w), fmt_pval(p_value));

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
                let p_value = 2.0 * phi_upper(z.abs());

                soutln!("\n  Wilcoxon rank sum test\n");
                soutln!("W = {}, p-value = {}", fmt_n(u), fmt_pval(p_value));
                soutln!("alternative hypothesis: true location shift is not equal to 0");

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
    let p_value = 2.0 * phi_upper(z.abs());

    soutln!("\n  Wilcoxon signed rank test\n");
    soutln!("V = {}, p-value = {}", fmt_n(w_plus), fmt_pval(p_value));
    soutln!("alternative hypothesis: true location is not equal to {}", mu);

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

    soutln!("\n  Fisher's Exact Test for Count Data\n");
    soutln!("p-value = {}", fmt_pval(p_value));
    soutln!("alternative hypothesis: true odds ratio is not equal to 1");
    soutln!("sample estimate:");
    soutln!("odds ratio: {}", fmt_n(or));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("estimate"), rnum(or));
    fields.insert(Arc::from("method"), rstr("Fisher's Exact Test for Count Data"));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("fisher.test"), fields }))
}
