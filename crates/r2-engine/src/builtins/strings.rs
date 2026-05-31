//! Engine-side shims for string builtins. Every fn here is a
//! one-line delegator into `r2_strings::bi_*`.

use r2_types::{EnvRef, EvalArg, R2Err, RVal};

use crate::Engine;

pub(crate) fn bi_paste(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_paste(a)
}
pub(crate) fn bi_paste0(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_paste0(a)
}
pub(crate) fn bi_nchar(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_nchar(a)
}
pub(crate) fn bi_toupper(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_toupper(a)
}
pub(crate) fn bi_tolower(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_tolower(a)
}
pub(crate) fn bi_substr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_substr(a)
}
pub(crate) fn bi_grep(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_grep(a)
}
pub(crate) fn bi_gsub(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_gsub(a)
}
pub(crate) fn bi_strsplit(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_strsplit(a)
}
pub(crate) fn bi_sprintf(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_sprintf(a)
}
pub(crate) fn bi_trimws(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_trimws(a)
}
pub(crate) fn bi_sub(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_sub(a)
}
pub(crate) fn bi_grepl(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_grepl(a)
}
pub(crate) fn bi_regexpr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_strings::bi_regexpr(a)
}
