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
    let mut names: Vec<Arc<str>> = inst.fields.keys().filter(|k| !k.starts_with('.')).cloned().collect();
    names.sort();
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
