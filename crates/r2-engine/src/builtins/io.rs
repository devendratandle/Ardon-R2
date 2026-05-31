//! Engine-side shims for file-I/O builtins. Each delegates one-line
//! to `r2_io::bi_*`. The richer `bi_read_csv_v2` (with header /
//! sep / NA inference) stays in `lib.rs` because it uses Engine
//! helpers like `as_logicals`; only the thin pass-throughs live
//! here.

use r2_types::{EnvRef, EvalArg, R2Err, RVal};

use crate::Engine;

pub(crate) fn bi_write_csv(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_write_csv(a)
}
pub(crate) fn bi_read_table(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_read_table(a)
}
pub(crate) fn bi_read_delim(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_read_delim(a)
}
pub(crate) fn bi_write_table(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_write_table(a)
}
pub(crate) fn bi_file_exists(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_file_exists(a)
}
pub(crate) fn bi_list_files(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_list_files(a)
}
