//! Engine-side shims for ML builtins. Each delegates one-line to
//! `r2_ml::dispatch::bi_*`. The engine retains the `tree` helpers
//! that recursive-call `r2_ml::tree::build_tree` directly because
//! they reference Engine-internal types.

use r2_types::{EnvRef, EvalArg, R2Err, RVal};

use crate::Engine;

pub(crate) fn bi_prcomp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_prcomp(a)
}
pub(crate) fn bi_kmeans(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_kmeans(a)
}
pub(crate) fn bi_knn(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_knn(a)
}
pub(crate) fn bi_naive_bayes(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_naive_bayes(a)
}
pub(crate) fn bi_rpart(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_rpart(a)
}
pub(crate) fn bi_rf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_rf(a)
}
pub(crate) fn bi_gbm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_gbm(a)
}
pub(crate) fn bi_cv(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> {
    r2_ml::dispatch::bi_cv(a)
}
