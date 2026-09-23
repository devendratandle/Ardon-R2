//! Engine-side builtins for data manipulation. Most delegate one line to
//! `r2_data::<submod>::bi_*` (functions that take `e` and `env`, like
//! `bi_do_call`, thread them through unchanged); the factor, `data()` and
//! predicate builtins at the end are implemented here.

use r2_types::*;
use crate::{gv, err};

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

// ── factors, data sets, predicates ───────────────────────────────────

pub(crate) fn bi_as_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Levels are sorted (numbers numerically), NA is never a level, and a
    // factor passes through untouched — see r2_types::factor.
    let val = gv(a, 0);
    match Factor::from_values(&val) {
        Some(f) => Ok(RVal::Factor(f)),
        None => err!(Type, "cannot coerce {} to factor", val.type_name()),
    }
}

pub(crate) fn bi_levels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Factor(f) => Ok(RVal::Character(f.levels.iter().map(|l| Some(l.clone())).collect(), Attrs::default())),
        _ => Ok(RVal::Null),
    }
}

pub(crate) fn bi_nlevels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) { RVal::Factor(f) => Ok(rint(f.levels.len() as i32)), _ => Ok(rint(0)) }
}

pub(crate) fn bi_as_logical(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a, 0))?;
    Ok(RVal::Logical(v.into(), Attrs::default()))
}

pub(crate) fn bi_data(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let val = gv(a, 0);
    match &val {
        RVal::Character(v, _) => {
            let name = v[0].as_ref().map(|s| s.to_string()).unwrap_or_default();
            if e.global_env.lookup(&name).is_some() {
                soutln!("Dataset '{}' is already loaded", name);
            } else {
                soutln!("Dataset '{}' not found", name);
            }
        }
        RVal::DataFrame(_) => {
            soutln!("Dataset is already loaded in the environment");
        }
        _ => {
            soutln!("Available datasets: iris, mtcars, airquality");
        }
    }
    Ok(RVal::Null)
}





pub(crate) fn bi_is_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Factor(_))))
}

pub(crate) fn bi_is_matrix(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Matrix(_))))
}

pub(crate) fn bi_mutate(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::dplyr::bi_mutate(a)
}
