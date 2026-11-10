//! Classes and attributes: `class<-`, `oldClass`, `unclass`, `attr<-`, and what a
//! classed result looks like as the list it is in R. Also `is.integer`, `row`, `col`.

use std::sync::Arc;

use r2_types::*;

use super::coerce::set_one_attr;
use super::core::str_vec;
use crate::{err, gv, Engine};

/// The value being assigned to a replacement function.
fn assigned(a: &[EvalArg]) -> RVal {
    a.iter().find(|x| x.name.as_deref() == Some("value")).map(|x| x.value.clone()).unwrap_or_else(|| gv(a, 1))
}

/// A classed result of a builtin (`lm`, `htest`, `qr`) as against a type the
/// user declared with `type`, or an out-of-core column: only the first is a list
/// in R.
pub(crate) fn is_result_object(e: &Engine, inst: &TypeInstance) -> bool {
    inst.type_name.as_ref() != "mmapcol" && !e.is_user_type(&inst.type_name)
}

/// The fields of a result, in R's order where the result's order is known here
/// and sorted otherwise. A leading dot marks what R2 keeps for itself.
pub(crate) fn field_names(inst: &TypeInstance) -> Vec<Arc<str>> {
    if inst.type_name.as_ref() == "qr" {
        return ["qr", "rank", "qraux", "pivot"].iter().map(|s| Arc::from(*s)).collect();
    }
    // a result that knows R's order for its fields says so
    if let Some(RVal::Character(order, _)) = inst.fields.get(".order") {
        return order.iter().flatten().filter(|n| inst.fields.contains_key(n.as_ref())).cloned().collect();
    }
    if inst.type_name.as_ref() == "anova" && inst.fields.contains_key("Df") {
        return ["Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)"].iter().map(|s| Arc::from(*s)).collect();
    }
    // R's order for a test result; any other field follows, sorted
    const HTEST: [&str; 10] = ["statistic", "parameter", "p.value", "conf.int", "estimate", "null.value",
        "stderr", "alternative", "method", "data.name"];
    let rank = |k: &str| HTEST.iter().position(|h| *h == k).unwrap_or(HTEST.len());
    let mut names: Vec<Arc<str>> = inst.fields.keys().filter(|k| !k.starts_with('.')).cloned().collect();
    names.sort_by(|x, y| rank(x).cmp(&rank(y)).then_with(|| x.cmp(y)));
    names
}

fn without_class(v: RVal) -> RVal {
    match v {
        RVal::Numeric(d, mut at) => { at.set_classes(Vec::new()); RVal::Numeric(d, at) }
        RVal::Integer(d, mut at) => { at.set_classes(Vec::new()); RVal::Integer(d, at) }
        RVal::Character(d, mut at) => { at.set_classes(Vec::new()); RVal::Character(d, at) }
        RVal::Logical(d, mut at) => { at.set_classes(Vec::new()); RVal::Logical(d, at) }
        RVal::List(d, mut at) => { at.set_classes(Vec::new()); RVal::List(d, at) }
        RVal::TypeInstance(inst) => {
            let names = field_names(&inst);
            RVal::list(names.into_iter().filter_map(|n| inst.fields.get(&n).cloned().map(|f| (Some(n), f))).collect())
        }
        RVal::DataFrame(df) => RVal::list(df.columns.into_iter().map(|(n, c)| (Some(n), c)).collect()),
        RVal::Factor(f) => {
            let mut at = Attrs::default();
            at.custom.insert(Arc::from("levels"), RVal::Character(f.levels.iter().cloned().map(Some).collect(), Attrs::default()));
            RVal::Integer(f.codes.iter().map(|c| c.map(|i| i as i32 + 1)).collect(), at)
        }
        other => other,
    }
}

/// `class(x) <- value`. A class that only names what `x` already is removes the
/// class attribute, as in R; one that names another basic type coerces.
pub(crate) fn bi_class_set(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = gv(a, 0);
    let value = assigned(a);
    let classes: Vec<Arc<str>> = if matches!(value, RVal::Null) { Vec::new() } else { str_vec(&value).into_iter().flatten().collect() };
    if classes.is_empty() { return Ok(without_class(x)); }
    if let [one] = classes.as_slice() {
        let setter = match one.as_ref() {
            "numeric" => Some("as.numeric"), "integer" => Some("as.integer"),
            "character" => Some("as.character"), "logical" => Some("as.logical"),
            "list" => Some("as.list"),
            _ => None,
        };
        if let Some(name) = setter {
            let bare = without_class(x);
            if bare.type_name() == one.as_ref() { return Ok(bare); }
            let Some((f, _)) = e.registry.resolve(name) else { return err!(Runtime, "could not find function \"{}\"", name) };
            return f(e, &[EvalArg { name: None, value: bare }], env);
        }
        if matches!(one.as_ref(), "matrix" | "array" | "function" | "data.frame" | "factor") {
            return err!(Runtime, "class(x) <- \"{}\" is not implemented; use as.{} or {}", one, one, one);
        }
    }
    match x {
        RVal::Numeric(..) | RVal::Integer(..) | RVal::Character(..) | RVal::Logical(..) | RVal::List(..) => {
            let mut x = x;
            set_one_attr(&mut x, "class", &value);
            Ok(x)
        }
        RVal::TypeInstance(mut inst) => { inst.type_name = classes[0].clone(); Ok(RVal::TypeInstance(inst)) }
        other => err!(Runtime, "class(x) <- value is not implemented for a {}", other.type_name()),
    }
}

/// `oldClass(x)`: the class attribute only, `NULL` for an implicit class.
pub(crate) fn bi_old_class(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let classes = match &v {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Character(_, at) | RVal::Logical(_, at) | RVal::List(_, at) => at.classes(),
        RVal::TypeInstance(i) => vec![i.type_name.clone()],
        RVal::DataFrame(_) => vec![Arc::from("data.frame")],
        RVal::Factor(_) => vec![Arc::from("factor")],
        _ => Vec::new(),
    };
    if classes.is_empty() { return Ok(RVal::Null); }
    Ok(RVal::Character(classes.into_iter().map(Some).collect(), Attrs::default()))
}

pub(crate) fn bi_unclass(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(without_class(gv(a, 0)))
}

/// `attr(x, which) <- value`; a `NULL` value removes the attribute.
pub(crate) fn bi_attr_set(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let which = match gv(a, 1) {
        RVal::Character(v, _) => v.first().cloned().flatten(),
        _ => None,
    };
    let Some(which) = which else { return err!(Runtime, "'name' must be non-null character string") };
    if which.as_ref() == "class" { return bi_class_set(e, &[a[0].clone(), EvalArg { name: None, value: assigned(a) }], env); }
    let mut x = gv(a, 0);
    set_one_attr(&mut x, &which, &assigned(a));
    Ok(x)
}

pub(crate) fn bi_is_integer(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Integer(..))))
}

fn index_matrix(a: &[EvalArg], name: &str, of: fn(usize, usize) -> usize) -> Result<RVal, R2Err> {
    let (nrow, ncol) = match &gv(a, 0) {
        RVal::Matrix(m) => (m.nrow, m.ncol),
        RVal::DataFrame(df) => (df.nrow(), df.ncol()),
        _ => return err!(Runtime, "a matrix-like object is required as argument to '{}'", name),
    };
    let mut data = Vec::with_capacity(nrow * ncol);
    for j in 0..ncol { for i in 0..nrow { data.push(of(i, j) as f64 + 1.0); } }
    Ok(RVal::Matrix(Matrix::new(data, nrow, ncol)))
}

pub(crate) fn bi_row(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { index_matrix(a, "row", |i, _| i) }
pub(crate) fn bi_col(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { index_matrix(a, "col", |_, j| j) }

/// `as.difftime(tim, units = )` for numbers: a time difference in `units`.
pub(crate) fn bi_as_difftime(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let units = a.iter().find(|x| x.name.as_deref() == Some("units")).map(|x| str_vec(&x.value))
        .and_then(|v| v.into_iter().next().flatten());
    let Some(units) = units.filter(|u| u.as_ref() != "auto") else {
        return err!(Runtime, "need explicit units for numeric conversion");
    };
    let Some(per) = crate::ops_attrs::unit_secs(&units) else { return err!(Runtime, "invalid units specified") };
    let tim = a.iter().find(|x| x.name.as_deref() == Some("tim")).or_else(|| a.iter().find(|x| x.name.is_none()))
        .map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let secs = e.as_reals(&tim)?.into_iter().map(|x| x.map(|v| v * per)).collect();
    Ok(crate::ops_attrs::difftime_in(secs, &units))
}

const WEEKDAYS: [&str; 7] = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"];
const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"];

/// The days since 1970-01-01 of each element of a date or a date-time.
fn day_numbers(e: &Engine, v: &RVal, name: &str) -> Result<Vec<Option<i64>>, R2Err> {
    let per_day = match v {
        RVal::Numeric(_, at) if at.class.as_deref() == Some("Date") => 1.0,
        RVal::Numeric(_, at) if at.class.as_deref() == Some("POSIXct") => 86400.0,
        _ => return err!(Runtime, "no applicable method for '{}' applied to an object of class \"{}\"", name, v.type_name()),
    };
    Ok(e.as_reals(v)?.into_iter().map(|x| x.map(|d| (d / per_day).floor() as i64)).collect())
}

fn calendar_names(e: &Engine, a: &[EvalArg], name: &str, pick: impl Fn(i64) -> &'static str) -> Result<RVal, R2Err> {
    let abbreviate = a.iter().find(|x| x.name.as_deref() == Some("abbreviate")).map(|x| e.truthy(&x.value)).transpose()?.unwrap_or(false);
    let days = day_numbers(e, &gv(a, 0), name)?;
    Ok(RVal::Character(days.into_iter().map(|d| d.map(|d| {
        let full = pick(d);
        Arc::from(if abbreviate { &full[..3] } else { full })
    })).collect(), Attrs::default()))
}

/// `weekdays(x, abbreviate = FALSE)` of a date or a date-time.
pub(crate) fn bi_weekdays(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    calendar_names(e, a, "weekdays", |d| WEEKDAYS[d.rem_euclid(7) as usize])
}

/// `months(x, abbreviate = FALSE)` of a date or a date-time.
pub(crate) fn bi_months(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    calendar_names(e, a, "months", |d| MONTHS[r2_stats::time::civil_from_days(d).1 as usize - 1])
}

/// `conditionMessage(c)`: the message of a condition caught by `tryCatch`.
pub(crate) fn bi_condition_message(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::List(items, _) => Ok(items.into_iter().find(|(n, _)| n.as_deref() == Some("message")).map(|(_, v)| v).unwrap_or(RVal::Null)),
        RVal::Character(s, at) => Ok(RVal::Character(s, at)),
        other => err!(Runtime, "no applicable method for 'conditionMessage' applied to an object of class \"{}\"", other.type_name()),
    }
}

/// `get(x)`: the value named `x` where the caller sees it, a builtin or an
/// operator included (`get("+")(1, 2)`, `get(.Generic)` in a group method).
pub(crate) fn bi_get(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let name = match gv(a, 0) {
        RVal::Character(s, _) => s.first().cloned().flatten(),
        _ => None,
    };
    let Some(name) = name else { return err!(Runtime, "invalid first argument") };
    if let Some(v) = env.lookup(&name).or_else(|| e.global_env.lookup(&name)) { return Ok(v); }
    if e.registry.resolve(&name).is_some() { return Ok(RVal::BuiltinFn(name)); }
    err!(Runtime, "object '{}' not found", name)
}

/// `I(x)`: `x` as it is, so a model formula takes `I(wt^2)` as arithmetic.
pub(crate) fn bi_asis(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(gv(a, 0))
}

fn names_or_null(n: Option<Vec<Arc<str>>>) -> RVal {
    match n {
        Some(n) => RVal::Character(n.into_iter().map(Some).collect(), Attrs::default()),
        None => RVal::Null,
    }
}

/// `dimnames(x)`: the row and column names of a matrix, a data frame or a
/// table result such as `anova(fit)`, as a list of two; `NULL` when neither has any.
pub(crate) fn bi_dimnames(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let pair = |r: RVal, c: RVal| -> RVal {
        if matches!((&r, &c), (RVal::Null, RVal::Null)) { RVal::Null } else { RVal::list(vec![(None, r), (None, c)]) }
    };
    Ok(match gv(a, 0) {
        RVal::Matrix(m) => pair(names_or_null(m.row_names.clone()), names_or_null(m.col_names.clone())),
        RVal::DataFrame(df) => {
            let rows = df.row_names.clone().unwrap_or_else(|| (1..=df.nrow()).map(|i| Arc::from(i.to_string().as_str())).collect());
            pair(names_or_null(Some(rows)), names_or_null(Some(df.columns.iter().map(|(n, _)| n.clone()).collect())))
        }
        RVal::TypeInstance(inst) if inst.fields.contains_key(".rownames") => {
            pair(inst.fields[".rownames"].clone(), names_or_null(Some(field_names(&inst))))
        }
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) =>
            at.custom.get("dimnames").cloned().unwrap_or(RVal::Null),
        _ => RVal::Null,
    })
}

/// `nobs(fit)`: the number of observations a model was fitted to.
pub(crate) fn bi_nobs(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::TypeInstance(inst) => match inst.fields.get("residuals") {
            Some(r) => Ok(rint(rval_length(r) as i32)),
            None => err!(Runtime, "no 'nobs' method is available for an object of class \"{}\"", inst.type_name),
        },
        RVal::Numeric(..) | RVal::Integer(..) | RVal::Logical(..) => Ok(rint(gv(a, 0).as_reals()?.iter().filter(|x| x.is_some()).count() as i32)),
        other => err!(Runtime, "no 'nobs' method is available for an object of class \"{}\"", other.type_name()),
    }
}
