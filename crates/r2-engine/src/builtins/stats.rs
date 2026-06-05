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

// ─── models / hypothesis tests / time series ────────────────────────
pub(crate) fn bi_lm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::models::bi_lm(a)
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
