//! Engine-side shims for statistics builtins. Every fn here is a
//! one-line delegator into `r2_stats::bi_*` or `r2_stats::<submod>::bi_*`.
//! Complex stats fns that use Engine helpers (`bi_median` with its
//! oracle-dispatched dead code, `bi_summary`, etc.) remain in
//! `lib.rs` and are NOT moved here.

use r2_types::{EnvRef, EvalArg, R2Err, RVal};

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
