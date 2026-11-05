//! `qr()` and its accessors. The work is in `r2_base::qr_ops`; only turning a
//! data frame into a matrix needs the engine.

use r2_base::qr_ops as ops;
use r2_types::{EnvRef, EvalArg, R2Err, RVal};

use super::coerce::bi_as_matrix;
use crate::Engine;

/// `a` with its first data-frame argument turned into a matrix, as `as.matrix`.
fn with_matrix(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<Vec<EvalArg>, R2Err> {
    let mut out = a.to_vec();
    if let Some(arg) = out.iter_mut().find(|x| matches!(x.value, RVal::DataFrame(_))) {
        let one = [EvalArg { name: None, value: arg.value.clone() }];
        arg.value = bi_as_matrix(e, &one, env)?;
    }
    Ok(out)
}

pub(crate) fn bi_qr(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    ops::bi_qr(&with_matrix(e, a, env)?)
}
pub(crate) fn bi_qr_solve(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    ops::bi_qr_solve(&with_matrix(e, a, env)?)
}
pub(crate) fn bi_is_qr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_is_qr(a) }
pub(crate) fn bi_qr_q(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_q(a) }
pub(crate) fn bi_qr_r(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_r(a) }
pub(crate) fn bi_qr_x(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_x(a) }
pub(crate) fn bi_qr_coef(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_coef(a) }
pub(crate) fn bi_qr_qy(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_qy(a) }
pub(crate) fn bi_qr_qty(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_qty(a) }
pub(crate) fn bi_qr_resid(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_resid(a) }
pub(crate) fn bi_qr_fitted(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { ops::bi_qr_fitted(a) }
