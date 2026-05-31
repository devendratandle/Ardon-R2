//! Engine-side shims for data-manipulation builtins. Each delegates
//! one-line to `r2_data::<submod>::bi_*`. Functions that take `e`
//! and `env` (like `bi_do_call`) thread them through unchanged.

use r2_types::{EnvRef, EvalArg, R2Err, RVal};

use crate::Engine;

pub(crate) fn bi_c(_e: &mut Engine, args: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::concat::bi_c(args)
}
pub(crate) fn bi_head(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_head(a)
}
pub(crate) fn bi_tail(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_tail(a)
}
pub(crate) fn bi_unique(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::order::bi_unique(a)
}
pub(crate) fn bi_nrow(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_nrow(a)
}
pub(crate) fn bi_ncol(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_ncol(a)
}
pub(crate) fn bi_dim(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_dim(a)
}
pub(crate) fn bi_colnames(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_colnames(a)
}
pub(crate) fn bi_rownames(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_rownames(a)
}
pub(crate) fn bi_is_data_frame(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_is_data_frame(a)
}
pub(crate) fn bi_as_data_frame(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::meta::bi_as_data_frame(a)
}
pub(crate) fn bi_table(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::table::bi_table(a)
}
pub(crate) fn bi_merge(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::clean::bi_merge(a)
}
pub(crate) fn bi_na_omit(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::clean::bi_na_omit(a)
}
pub(crate) fn bi_complete_cases(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::clean::bi_complete_cases(a)
}
pub(crate) fn bi_do_call(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::apply::bi_do_call(e, a, env)
}
pub(crate) fn bi_duplicated(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::order::bi_duplicated(a)
}
pub(crate) fn bi_order(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::order::bi_order(a)
}
pub(crate) fn bi_rank(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::order::bi_rank(a)
}
