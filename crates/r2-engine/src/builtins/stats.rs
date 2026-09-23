//! Engine-side builtins for statistics. Most are one-line delegators into
//! `r2_stats::bi_*` or `r2_stats::<submod>::bi_*`; row/column summaries and
//! `confusion.matrix()` are implemented here. Complex stats fns that use
//! Engine helpers (`bi_median`, `bi_summary`, etc.) remain in `lib.rs`.

#![allow(clippy::all)]
use std::collections::HashMap;
use std::sync::Arc;
use r2_types::*;
use crate::{gv, gn, err};

use crate::Engine;

// ─── summary stats ──────────────────────────────────────────────────
pub(crate) fn bi_sum(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "sum") { return r; } }
    r2_stats::bi_sum(a)
}
pub(crate) fn bi_mean(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "mean") { return r; } }
    r2_stats::bi_mean(a)
}
pub(crate) fn bi_sd(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "sd") { return r; } }
    r2_stats::bi_sd(a)
}
pub(crate) fn bi_var(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "var") { return r; } }
    r2_stats::bi_var(a)
}
pub(crate) fn bi_max(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "max") { return r; } }
    r2_stats::bi_max(a)
}
pub(crate) fn bi_min(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "min") { return r; } }
    r2_stats::bi_min(a)
}
pub(crate) fn bi_prod(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let Some(v) = a.first() { if let Some(r) = super::ml_data::mmap_reduce(&v.value, "prod") { return r; } }
    r2_stats::bi_prod(a)
}
pub(crate) fn bi_cor(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_cor(a)
}
pub(crate) fn bi_cov(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_cov(a)
}

// ─── cumulative / rolling ───────────────────────────────────────────
pub(crate) fn bi_cumsum(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_cumsum(a)
}
pub(crate) fn bi_cumprod(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_cumprod(a)
}
pub(crate) fn bi_diff(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_diff(a)
}
pub(crate) fn bi_rollsum(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_rollsum(a)
}
pub(crate) fn bi_rollmean(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_rollmean(a)
}
pub(crate) fn bi_rollmax(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_rollmax(a)
}
pub(crate) fn bi_rollmin(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_rollmin(a)
}
pub(crate) fn bi_rollsd(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::summary::bi_rollsd(a)
}
pub(crate) fn bi_quantile(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Out-of-core column: streaming approximate quantiles over the mmap.
    if let Some(arg) = a.first() {
        if matches!(&arg.value, RVal::TypeInstance(i) if i.type_name.as_ref() == "mmapcol") {
            // probs from `probs=` or a positional 2nd arg, else R's default.
            let pv = a.iter().find(|x| x.name.as_deref() == Some("probs")).map(|x| &x.value)
                .or_else(|| a.get(1).filter(|x| x.name.is_none()).map(|x| &x.value));
            let probs: Vec<f64> = match pv {
                Some(RVal::Numeric(v, _)) => v.as_vec().iter().filter_map(|o| *o).collect(),
                _ => vec![0.0, 0.25, 0.5, 0.75, 1.0],
            };
            if let Some(r) = super::ml_data::mmap_quantile(&arg.value, &probs) {
                return r.map(|q| {
                    let names: Vec<std::sync::Arc<str>> =
                        probs.iter().map(|p| std::sync::Arc::from(format!("{}%", p * 100.0).as_str())).collect();
                    let mut attrs = r2_types::Attrs::default();
                    attrs.names = Some(names);
                    RVal::Numeric(q.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), attrs)
                });
            }
        }
    }
    r2_stats::summary::bi_quantile(a)
}

// ─── distributions / RNG ────────────────────────────────────────────
pub(crate) fn bi_rnorm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::rng::bi_rnorm(a)
}
pub(crate) fn bi_dnorm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::dist::bi_dnorm(a)
}
pub(crate) fn bi_pnorm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::dist::bi_pnorm(a)
}
pub(crate) fn bi_qnorm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::dist::bi_qnorm(a)
}
pub(crate) fn bi_runif(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::rng::bi_runif(a)
}
pub(crate) fn bi_sample(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::rng::bi_sample(a)
}
pub(crate) fn bi_rbinom(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::rng::bi_rbinom(a)
}
pub(crate) fn bi_rpois(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::rng::bi_rpois(a)
}
// Tier-2 distributions (d/p/q) — delegators into r2_stats::dist.
pub(crate) fn bi_dexp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_dexp(a) }
pub(crate) fn bi_pexp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_pexp(a) }
pub(crate) fn bi_qexp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qexp(a) }
pub(crate) fn bi_dbinom(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_dbinom(a) }
pub(crate) fn bi_pbinom(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_pbinom(a) }
pub(crate) fn bi_dpois(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_dpois(a) }
pub(crate) fn bi_ppois(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_ppois(a) }
pub(crate) fn bi_dt(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_dt(a) }
pub(crate) fn bi_pt(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_pt(a) }
pub(crate) fn bi_dchisq(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_dchisq(a) }
pub(crate) fn bi_pchisq(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_pchisq(a) }
pub(crate) fn bi_pf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_pf(a) }
pub(crate) fn bi_rexp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::rng::bi_rexp(a) }
pub(crate) fn bi_qt(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qt(a) }
pub(crate) fn bi_qchisq(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qchisq(a) }
pub(crate) fn bi_qf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qf(a) }
pub(crate) fn bi_qbinom(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qbinom(a) }
pub(crate) fn bi_qpois(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_qpois(a) }
pub(crate) fn bi_density(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::dist::bi_density(a) }

// ─── models / hypothesis tests / time series ────────────────────────
pub(crate) fn bi_lm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::models::bi_lm(a)
}
pub(crate) fn bi_plssem(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::plssem::bi_plssem(a)
}
pub(crate) fn bi_t_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::htest::bi_t_test(a)
}
pub(crate) fn bi_chisq_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::htest::bi_chisq_test(a)
}
pub(crate) fn bi_ts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_ts(a)
}

// ── set.seed() — RNG primitives live in r2_stats::rng ──────────────────

pub(crate) fn bi_set_seed(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::rng::bi_set_seed(a) }

// ── rowSums(), colSums(), rowMeans(), colMeans() ─────────────────────

pub(crate) fn bi_rowSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut sums = vec![0.0f64; nrow];
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    for (i, v) in vals.iter().enumerate() { if let Some(n) = v { sums[i] += n; } }
                }
            }
            Ok(rnums(&sums))
        }
        RVal::Matrix(m) => {
            let sums: Vec<f64> = (0..m.nrow).map(|r| (0..m.ncol).map(|c| m.get(r, c)).sum()).collect();
            Ok(rnums(&sums))
        }
        _ => err!(Runtime, "rowSums needs data.frame or matrix"),
    }
}

pub(crate) fn bi_colSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let mut results = Vec::new();
            for (_name, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    let s: f64 = vals.iter().filter_map(|x| *x).sum();
                    results.push(s);
                }
            }
            let mut attrs = Attrs::default();
            attrs.names = Some(df.columns.iter().filter_map(|(n, col)| {
                if e.as_reals(col).is_ok() { Some(n.clone()) } else { None }
            }).collect());
            Ok(RVal::Numeric(results.iter().map(|x| Some(*x)).collect(), attrs))
        }
        RVal::Matrix(m) => {
            let sums: Vec<f64> = (0..m.ncol).map(|c| (0..m.nrow).map(|r| m.get(r, c)).sum()).collect();
            Ok(rnums(&sums))
        }
        _ => err!(Runtime, "colSums needs data.frame or matrix"),
    }
}

pub(crate) fn bi_rowMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let ncol_num = df.columns.iter().filter(|(_, col)| e.as_reals(col).is_ok()).count();
            let mut sums = vec![0.0f64; nrow];
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    for (i, v) in vals.iter().enumerate() { if let Some(n) = v { sums[i] += n; } }
                }
            }
            Ok(rnums(&sums.iter().map(|s| s / ncol_num as f64).collect::<Vec<_>>()))
        }
        _ => err!(Runtime, "rowMeans needs data.frame or matrix"),
    }
}

pub(crate) fn bi_colMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow() as f64;
            let mut results = Vec::new();
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    let s: f64 = vals.iter().filter_map(|x| *x).sum();
                    results.push(s / nrow);
                }
            }
            Ok(rnums(&results))
        }
        RVal::Matrix(m) => {
            let nrow = m.nrow;
            let results: Vec<f64> = (0..m.ncol).map(|j| {
                let s: f64 = (0..nrow).map(|i| m.data[j * nrow + i]).sum();
                s / nrow as f64
            }).collect();
            Ok(rnums(&results))
        }
        _ => err!(Runtime, "colMeans needs data.frame or matrix"),
    }
}

// ── confusion.matrix() — classification evaluation ───────────────────

pub(crate) fn bi_confusion_matrix(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // confusion.matrix(predicted, actual) or confusion.matrix(model)
    let pred: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|x| x).collect();
    let actual: Vec<f64> = e.as_reals(&gv(a,1))?.into_iter().filter_map(|x| x).collect();

    if pred.len() != actual.len() { return err!(Runtime, "confusion.matrix: lengths must match"); }

    // Find unique classes
    let mut classes: Vec<i64> = Vec::new();
    for v in pred.iter().chain(actual.iter()) {
        let c = *v as i64;
        if !classes.contains(&c) { classes.push(c); }
    }
    classes.sort();
    let k = classes.len();

    // Build confusion matrix
    let mut cm = vec![0i32; k * k];
    for i in 0..pred.len() {
        let pi = classes.iter().position(|&c| c == pred[i] as i64).unwrap_or(0);
        let ai = classes.iter().position(|&c| c == actual[i] as i64).unwrap_or(0);
        cm[ai * k + pi] += 1; // row = actual, col = predicted
    }

    // Print
    soutln!("\nConfusion Matrix:");
    sout!("{:>12}", "Predicted→");
    for c in &classes { sout!("{:>8}", c); }
    soutln!("{:>10}", "Total");
    

    let n = pred.len();
    let mut correct = 0;
    for (ai, ac) in classes.iter().enumerate() {
        sout!("Actual {:>4} ", ac);
        let mut row_total = 0;
        for pi in 0..k {
            sout!("{:>8}", cm[ai * k + pi]);
            row_total += cm[ai * k + pi];
            if ai == pi { correct += cm[ai * k + pi]; }
        }
        soutln!("{:>10}", row_total);
    }

    
    let accuracy = correct as f64 / n as f64;
    soutln!("Accuracy: {}/{} ({}%)", correct, n, fmt_num(accuracy * 100.0));

    // Per-class precision and recall
    soutln!("\n{:>8} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1");
    for (ci, c) in classes.iter().enumerate() {
        let tp = cm[ci * k + ci] as f64;
        let pred_total: f64 = (0..k).map(|ai| cm[ai * k + ci] as f64).sum();
        let actual_total: f64 = (0..k).map(|pi| cm[ci * k + pi] as f64).sum();
        let precision = if pred_total > 0.0 { tp / pred_total } else { 0.0 };
        let recall = if actual_total > 0.0 { tp / actual_total } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        soutln!("{:>8} {:>10} {:>10} {:>10}", c, fmt_num(precision), fmt_num(recall), fmt_num(f1));
    }

    let mut fields = HashMap::new();
    fields.insert(Arc::from("accuracy"), rnum(accuracy));
    fields.insert(Arc::from("matrix"), RVal::Matrix(Matrix::new(cm.iter().map(|&x| x as f64).collect(), k, k)));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("confusion"), fields }))
}

// ── aov() / anova() and the hypothesis tests ─────────────────────────

pub(crate) fn bi_aov(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_aov(a) }

pub(crate) fn bi_anova(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_anova(a) }

// ═══════════════════════════════════════════════════════════════════════
// Additional Statistical Tests
// ═══════════════════════════════════════════════════════════════════════

// ── cor.test() — test if correlation is significant ──────────────────

pub(crate) fn bi_cor_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_cor_test(a) }

// ── shapiro.test() — test for normality ──────────────────────────────

pub(crate) fn bi_shapiro_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_shapiro_test(a) }

// ── wilcox.test() — Wilcoxon rank-sum / signed-rank test ─────────────

pub(crate) fn bi_wilcox_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_wilcox_test(a) }

// ── fisher.test() — Fisher's exact test for 2×2 tables ──────────────

pub(crate) fn bi_fisher_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_fisher_test(a) }

// ── weighted.mean() ──────────────────────────────────────────────────

pub(crate) fn bi_weighted_mean(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|v| v).collect();
    let w: Vec<f64> = gn(a, "w").or(Some(gv(a, 1)))
        .and_then(|v| e.as_reals(&v).ok())
        .unwrap_or(vec![Some(1.0); x.len()])
        .into_iter().filter_map(|v| v).collect();
    let n = x.len().min(w.len());
    let sum_w: f64 = w[..n].iter().sum();
    let wm: f64 = x[..n].iter().zip(w[..n].iter()).map(|(x, w)| x * w).sum::<f64>() / sum_w;
    Ok(rnum(wm))
}

// ── IQR() — interquartile range ──────────────────────────────────────

pub(crate) fn bi_iqr(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|v| v).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len();
    if n < 2 { return err!(Runtime, "IQR needs at least 2 values"); }
    let q1 = x[n / 4];
    let q3 = x[3 * n / 4];
    Ok(rnum(q3 - q1))
}
