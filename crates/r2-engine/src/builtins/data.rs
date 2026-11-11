//! Engine-side builtins for data manipulation. Most delegate one line to
//! `r2_data::<submod>::bi_*` (functions that take `e` and `env`, like
//! `bi_do_call`, thread them through unchanged); the factor, `data()` and
//! predicate builtins at the end are implemented here.

use std::sync::Arc;
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
    // `do.call("sum", args)`: a function by name, as R allows
    if let Some(RVal::Character(v, _)) = a.first().map(|x| &x.value) {
        if let Some(Some(name)) = v.first() {
            let f = env.lookup(name).filter(|f| matches!(f, RVal::Closure(_) | RVal::BuiltinFn(_)))
                .unwrap_or_else(|| RVal::BuiltinFn(name.clone()));
            let mut args = a.to_vec();
            args[0].value = f;
            return r2_data::apply::bi_do_call(e, &args, env);
        }
    }
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

/// `data()` lists R's datasets with their titles; `data(iris, CO2)` or
/// `data(list = c("iris", "CO2"))` binds a fresh copy of each in the
/// workspace and returns their names (invisibly), as R does. Every dataset
/// is reachable without `data()` too — they live in the base environment.
pub(crate) fn bi_data(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut names: Vec<Arc<str>> = Vec::new();
    for arg in a.iter().filter(|x| x.name.is_none() || x.name.as_deref() == Some("list")) {
        if let RVal::Character(v, _) = &arg.value { names.extend(v.iter().flatten().cloned()); }
    }
    if names.is_empty() {
        soutln!("Data sets in package 'datasets':\n");
        for d in r2_base::datasets::index() {
            soutln!("{:<24}{}", d.name, d.title);
        }
        return Ok(RVal::Null);
    }
    for n in &names {
        match r2_base::datasets::get(n) {
            Some(v) => e.global_env.set(n.clone(), v),
            None => r2_types::out::rerr(&format!("Warning message:\ndata set '{}' not found", n)),
        }
    }
    Ok(RVal::Character(names.into_iter().map(Some).collect(), Attrs::default()))
}





pub(crate) fn bi_is_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Factor(_))))
}
pub(crate) fn bi_is_ordered(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Factor(f) if f.ordered)))
}

pub(crate) fn bi_is_matrix(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(match gv(a, 0) {
        RVal::Matrix(_) => true,
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => at.dim.as_ref().is_some_and(|d| d.len() == 2),
        _ => false,
    }))
}

pub(crate) fn bi_mutate(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::dplyr::bi_mutate(a)
}
