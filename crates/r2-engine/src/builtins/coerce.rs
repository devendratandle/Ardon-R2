//! Builtins: constructors / coercions / predicates, special-function math
//! (gamma/choose/beta/…), and attribute/format. Split from core.rs; shares
//! core's helpers via `use super::core::*`.
#![allow(clippy::all)]
use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;
use r2_stats::htest::{fmt_pval, signif_stars, ln_gamma};
use r2_types::*;
use super::core::*;
use crate::{gv, gn, val_to_str, Engine};
use crate::err;

// ── Tier-3 constructors / coercions / predicates ───────────────────

pub(crate) fn bi_numeric(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = e.scalar_f64(&gv(a,0)).ok().flatten().unwrap_or(0.0).max(0.0) as usize;
    Ok(RVal::Numeric(vec![Some(0.0); n].into(), Attrs::default()))
}
pub(crate) fn bi_integer(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = e.scalar_f64(&gv(a,0)).ok().flatten().unwrap_or(0.0).max(0.0) as usize;
    Ok(RVal::Integer(vec![Some(0); n].into(), Attrs::default()))
}
pub(crate) fn bi_character(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = e.scalar_f64(&gv(a,0)).ok().flatten().unwrap_or(0.0).max(0.0) as usize;
    Ok(RVal::Character(vec![Some(Arc::from("")); n], Attrs::default()))
}
pub(crate) fn bi_logical(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = e.scalar_f64(&gv(a,0)).ok().flatten().unwrap_or(0.0).max(0.0) as usize;
    Ok(RVal::Logical(vec![Some(false); n].into(), Attrs::default()))
}
pub(crate) fn bi_as_matrix(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::Matrix(m) => Ok(RVal::Matrix(m.clone())),
        RVal::DataFrame(df) => {
            let nrow = df.columns.first().map(|(_, v)| rval_length(v)).unwrap_or(0);
            let mut data = Vec::new();
            let mut names = Vec::new();
            let mut ncol = 0;
            for (nm, v) in &df.columns {
                if let Ok(r) = v.as_reals() {
                    data.extend(r.into_iter().map(|o| o.unwrap_or(f64::NAN)));
                    names.push(nm.clone());
                    ncol += 1;
                }
            }
            let mut m = Matrix::new(data, nrow, ncol);
            m.col_names = Some(names);
            Ok(RVal::Matrix(m))
        }
        other => {
            let r = e.as_reals(other)?;
            let n = r.len();
            Ok(RVal::Matrix(Matrix::new(r.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect(), n, 1)))
        }
    }
}
pub(crate) fn bi_as_vector(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(match gv(a,0) {
        RVal::Matrix(m) => RVal::Numeric(m.data.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default()),
        RVal::Numeric(v, _) => RVal::Numeric(v, Attrs::default()),
        RVal::Integer(v, _) => RVal::Integer(v, Attrs::default()),
        RVal::Character(v, _) => RVal::Character(v, Attrs::default()),
        RVal::Logical(v, _) => RVal::Logical(v, Attrs::default()),
        other => other,
    })
}
pub(crate) fn bi_as_list(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(match gv(a,0) {
        RVal::List(items) => RVal::List(items),
        RVal::DataFrame(df) => RVal::List(df.columns.iter().map(|(n, v)| (Some(n.clone()), v.clone())).collect()),
        other => RVal::List(elements(&other).into_iter().map(|v| (None, v)).collect()),
    })
}
pub(crate) fn bi_is_function(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Closure(_)))) }

/// Operators as first-class functions: `` `+` ``, `` `*` ``, … so they can be
/// passed to `Reduce`/`Map`/`do.call` (idiomatic functional R). Normal
/// `a + b` still goes through `Expr::Binary`, not the registry, so this
/// only affects the backtick-quoted / passed-as-value form.
macro_rules! op_fn {
    ($name:ident, $op:ident) => {
        pub(crate) fn $name(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
            e.binary_op(r2_types::BinOp::$op, &gv(a,0), &gv(a,1))
        }
    };
}
op_fn!(bi_op_add, Add);  op_fn!(bi_op_sub, Sub);  op_fn!(bi_op_mul, Mul);
op_fn!(bi_op_div, Div);  op_fn!(bi_op_pow, Pow);  op_fn!(bi_op_mod, Mod);
op_fn!(bi_op_eq, Eq);    op_fn!(bi_op_ne, Ne);    op_fn!(bi_op_lt, Lt);
op_fn!(bi_op_gt, Gt);    op_fn!(bi_op_le, Le);     op_fn!(bi_op_ge, Ge);

/// Read the `value =` argument of a replacement function as a name vector.
pub(crate) fn setter_names(a: &[EvalArg]) -> Vec<Arc<str>> {
    let val = gn(a, "value").or_else(|| a.iter().filter(|x| x.name.is_none()).nth(1).map(|x| x.value.clone()));
    match val {
        Some(RVal::Character(v, _)) => v.into_iter().map(|x| x.unwrap_or_else(|| Arc::from("NA"))).collect(),
        Some(other) => crate::val_to_str(&other).split(' ').map(|s| Arc::from(s)).collect(),
        None => Vec::new(),
    }
}

/// `names(x) <- value` — set names on a vector/list/data.frame.
pub(crate) fn bi_names_set(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let names = setter_names(a);
    Ok(match gv(a, 0) {
        RVal::DataFrame(mut df) => { for (i, c) in df.columns.iter_mut().enumerate() { if let Some(nm) = names.get(i) { c.0 = nm.clone(); } } RVal::DataFrame(df) }
        RVal::List(mut items) => { for (i, it) in items.iter_mut().enumerate() { it.0 = names.get(i).cloned(); } RVal::List(items) }
        RVal::Numeric(v, mut at)   => { at.names = Some(names); RVal::Numeric(v, at) }
        RVal::Integer(v, mut at)   => { at.names = Some(names); RVal::Integer(v, at) }
        RVal::Character(v, mut at) => { at.names = Some(names); RVal::Character(v, at) }
        RVal::Logical(v, mut at)   => { at.names = Some(names); RVal::Logical(v, at) }
        other => other,
    })
}
/// `colnames(x) <- value` — rename data.frame / matrix columns.
pub(crate) fn bi_colnames_set(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let names = setter_names(a);
    Ok(match gv(a, 0) {
        RVal::DataFrame(mut df) => { for (i, c) in df.columns.iter_mut().enumerate() { if let Some(nm) = names.get(i) { c.0 = nm.clone(); } } RVal::DataFrame(df) }
        RVal::Matrix(mut m) => { m.col_names = Some(names); RVal::Matrix(m) }
        other => other,
    })
}
/// `rownames(x) <- value` — set data.frame / matrix row names.
pub(crate) fn bi_rownames_set(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let names = setter_names(a);
    Ok(match gv(a, 0) {
        RVal::DataFrame(mut df) => { df.row_names = Some(names); RVal::DataFrame(df) }
        RVal::Matrix(mut m) => { m.row_names = Some(names); RVal::Matrix(m) }
        other => other,
    })
}

/// `diag(x)` — matrix → its diagonal vector; vector → a diagonal matrix;
/// a single integer k → the k×k identity matrix (R's overloaded `diag`).
pub(crate) fn bi_diag(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Matrix(m) => {
            let n = m.nrow.min(m.ncol);
            let d: Vec<Real> = (0..n).map(|i| Some(m.get(i, i))).collect();
            Ok(RVal::Numeric(d.into(), Attrs::default()))
        }
        v => {
            let r = v.as_reals()?;
            if r.len() == 1 {
                // diag(k) → k×k identity.
                let k = r[0].unwrap_or(0.0) as usize;
                let mut data = vec![0.0; k * k];
                for i in 0..k { data[i * k + i] = 1.0; }
                Ok(RVal::Matrix(Matrix::new(data, k, k)))
            } else {
                // diag(v) → diagonal matrix with v on the diagonal.
                let n = r.len();
                let mut data = vec![0.0; n * n];
                for i in 0..n { data[i * n + i] = r[i].unwrap_or(0.0); }
                Ok(RVal::Matrix(Matrix::new(data, n, n)))
            }
        }
    }
}

/// `isTRUE(x)` / `isFALSE(x)` — TRUE only for a length-1, non-NA logical of
/// that value. Common in `if (isTRUE(...))` guards and test code.
pub(crate) fn bi_is_true(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(&gv(a,0), RVal::Logical(v,_) if v.as_vec().len()==1 && v.as_vec()[0]==Some(true))))
}
pub(crate) fn bi_is_false(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(&gv(a,0), RVal::Logical(v,_) if v.as_vec().len()==1 && v.as_vec()[0]==Some(false))))
}

/// Structural (type-strict) equality for `identical()`.
pub(crate) fn rval_identical(a: &RVal, b: &RVal) -> bool {
    match (a, b) {
        (RVal::Numeric(x,_),   RVal::Numeric(y,_))   => x.as_vec() == y.as_vec(),
        (RVal::Integer(x,_),   RVal::Integer(y,_))   => x.as_vec() == y.as_vec(),
        (RVal::Character(x,_), RVal::Character(y,_)) => x == y,
        (RVal::Logical(x,_),   RVal::Logical(y,_))   => x.as_vec() == y.as_vec(),
        (RVal::Null,           RVal::Null)           => true,
        (RVal::List(x),        RVal::List(y))        =>
            x.len()==y.len() && x.iter().zip(y).all(|((n1,v1),(n2,v2))| n1==n2 && rval_identical(v1,v2)),
        _ => false,
    }
}
pub(crate) fn bi_identical(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(rval_identical(&gv(a,0), &gv(a,1))))
}

/// `all.equal(x, y, tolerance=)` — near-equality. Returns TRUE within
/// tolerance (numeric, relative), else a character difference message —
/// so the idiomatic `isTRUE(all.equal(a, b))` works. Non-numeric falls
/// back to `identical`.
pub(crate) fn bi_all_equal(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = gv(a,0); let y = gv(a,1);
    let tol = gn(a, "tolerance").and_then(|v| v.as_reals().ok())
        .and_then(|r| r.first().copied().flatten()).unwrap_or(1.5e-8);
    if let (Ok(xr), Ok(yr)) = (x.as_reals(), y.as_reals()) {
        if xr.len() != yr.len() {
            return Ok(rstr(&format!("Lengths ({}, {}) differ", xr.len(), yr.len())));
        }
        let (mut sumabs, mut sumtarget) = (0.0f64, 0.0f64);
        for (xi, yi) in xr.iter().zip(yr.iter()) {
            match (xi, yi) {
                (Some(xv), Some(yv)) => { sumabs += (xv - yv).abs(); sumtarget += xv.abs(); }
                (None, None) => {}
                _ => return Ok(rstr("NA mismatch")),
            }
        }
        let rel = if sumtarget > 0.0 { sumabs / sumtarget } else { sumabs };
        return Ok(if rel <= tol { rbool(true) } else { rstr(&format!("Mean relative difference: {}", rel)) });
    }
    Ok(if rval_identical(&x, &y) { rbool(true) } else { rstr("objects differ") })
}
pub(crate) fn bi_is_list(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::List(_)))) }
pub(crate) fn bi_is_vector(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a,0), RVal::Numeric(..) | RVal::Integer(..) | RVal::Character(..) | RVal::Logical(..))))
}

// ── Tier-2 math / special functions ────────────────────────────────

/// Γ(x) via lgamma, with Euler reflection for x ≤ 0.
pub(crate) fn gamma_fn(x: f64) -> f64 {
    if x > 0.0 { ln_gamma(x).exp() }
    else {
        let s = (std::f64::consts::PI * x).sin();
        if s == 0.0 { f64::NAN } else { std::f64::consts::PI / (s * ln_gamma(1.0 - x).exp()) }
    }
}
pub(crate) fn choose_fn(n: f64, k: f64) -> f64 {
    let k = k.round();
    if k < 0.0 { return 0.0; }
    if n >= 0.0 && (n - n.round()).abs() < 1e-9 && k <= n {
        (ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)).exp().round()
    } else {
        let mut r = 1.0;
        for i in 0..(k as i64) { r *= (n - i as f64) / ((i + 1) as f64); }
        r
    }
}
pub(crate) fn median_sorted(mut v: Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}
pub(crate) fn map1(e: &mut Engine, a: &[EvalArg], f: impl Fn(f64) -> f64) -> Result<RVal, R2Err> {
    let xs = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(xs.into_iter().map(|o| o.map(&f)).collect::<Vec<_>>().into(), Attrs::default()))
}
pub(crate) fn bi_gamma(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { map1(e, a, gamma_fn) }
pub(crate) fn bi_lgamma(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { map1(e, a, ln_gamma) }
pub(crate) fn bi_factorial(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { map1(e, a, |x| gamma_fn(x + 1.0)) }
pub(crate) fn bi_beta(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let aa = e.as_reals(&gv(a,0))?; let bb = e.as_reals(&gv(a,1))?;
    let m = aa.len().max(bb.len());
    if aa.is_empty() || bb.is_empty() { return Ok(RVal::Numeric(vec![].into(), Attrs::default())); }
    let out: Vec<Option<f64>> = (0..m).map(|i| match (aa[i % aa.len()], bb[i % bb.len()]) {
        (Some(x), Some(y)) => Some((ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)).exp()),
        _ => None,
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
pub(crate) fn bi_choose(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ns = e.as_reals(&gv(a,0))?; let ks = e.as_reals(&gv(a,1))?;
    if ns.is_empty() || ks.is_empty() { return Ok(RVal::Numeric(vec![].into(), Attrs::default())); }
    let m = ns.len().max(ks.len());
    let out: Vec<Option<f64>> = (0..m).map(|i| match (ns[i % ns.len()], ks[i % ks.len()]) {
        (Some(n), Some(k)) => Some(choose_fn(n, k)),
        _ => None,
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
pub(crate) fn bi_combn(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let xr: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().flatten().collect();
    let x: Vec<f64> = if xr.len() == 1 { (1..=xr[0] as i64).map(|i| i as f64).collect() } else { xr };
    let m = e.scalar_f64(&gv(a,1))?.unwrap_or(2.0) as usize;
    let n = x.len();
    if m == 0 || m > n { return err!(Runtime, "combn(): need 1 <= m <= length(x)"); }
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut idx: Vec<usize> = (0..m).collect();
    loop {
        cols.push(idx.iter().map(|&i| x[i]).collect());
        let mut i = m as isize - 1;
        while i >= 0 && idx[i as usize] == n - m + i as usize { i -= 1; }
        if i < 0 { break; }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..m { idx[j] = idx[j - 1] + 1; }
    }
    let ncol = cols.len();
    let mut data = vec![0.0; m * ncol];
    for (c, col) in cols.iter().enumerate() { for r in 0..m { data[c * m + r] = col[r]; } }
    Ok(RVal::Matrix(Matrix::new(data, m, ncol)))
}
pub(crate) fn bi_mad(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let xs: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().flatten().collect();
    let constant = gn(a,"constant").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(1.4826);
    let med = median_sorted(xs.clone());
    let dev: Vec<f64> = xs.iter().map(|v| (v - med).abs()).collect();
    Ok(rnum(median_sorted(dev) * constant))
}
pub(crate) fn bi_fivenum(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().flatten().collect();
    if x.is_empty() { return Ok(RVal::Numeric(vec![None; 5].into(), Attrs::default())); }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len() as f64;
    let n4 = ((n + 3.0) / 2.0).floor() / 2.0;
    let d = [1.0, n4, (n + 1.0) / 2.0, n + 1.0 - n4, n];
    let out: Vec<Option<f64>> = d.iter().map(|&di| {
        let lo = (di.floor() as usize).saturating_sub(1).min(x.len() - 1);
        let hi = (di.ceil() as usize).saturating_sub(1).min(x.len() - 1);
        Some(0.5 * (x[lo] + x[hi]))
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}

// ── Tier-1 attribute / format builtins ─────────────────────────────

pub(crate) fn attrs_of(v: &RVal) -> Option<&Attrs> {
    match v {
        RVal::Numeric(_, at) | RVal::Integer(_, at)
        | RVal::Character(_, at) | RVal::Logical(_, at) => Some(at),
        _ => None,
    }
}
pub(crate) fn names_to_rval(names: &[Arc<str>]) -> RVal {
    RVal::Character(names.iter().cloned().map(Some).collect(), Attrs::default())
}
pub(crate) fn dim_to_rval(dim: &[usize]) -> RVal {
    rints(&dim.iter().map(|x| *x as i32).collect::<Vec<_>>())
}

pub(crate) fn bi_attr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    let which = val_to_str(&gv(a,1));
    let at = match attrs_of(&v) { Some(a) => a, None => return Ok(RVal::Null) };
    Ok(match which.as_str() {
        "names" => at.names.as_ref().map(|n| names_to_rval(n)).unwrap_or(RVal::Null),
        "class" => at.class.as_ref().map(|c| rstr(c)).unwrap_or(RVal::Null),
        "dim"   => at.dim.as_ref().map(|d| dim_to_rval(d)).unwrap_or(RVal::Null),
        other   => at.custom.get(&Arc::from(other)).cloned().unwrap_or(RVal::Null),
    })
}
pub(crate) fn bi_attributes(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    let at = match attrs_of(&v) { Some(a) => a, None => return Ok(RVal::Null) };
    let mut items: Vec<(Option<Arc<str>>, RVal)> = Vec::new();
    if let Some(n) = &at.names { items.push((Some(Arc::from("names")), names_to_rval(n))); }
    if let Some(c) = &at.class { items.push((Some(Arc::from("class")), rstr(c))); }
    if let Some(d) = &at.dim   { items.push((Some(Arc::from("dim")), dim_to_rval(d))); }
    for (k, val) in &at.custom { items.push((Some(k.clone()), val.clone())); }
    if items.is_empty() { Ok(RVal::Null) } else { Ok(RVal::List(items)) }
}
pub(crate) fn set_one_attr(v: &mut RVal, name: &str, val: &RVal) {
    let at = match v {
        RVal::Numeric(_, a) | RVal::Integer(_, a)
        | RVal::Character(_, a) | RVal::Logical(_, a) => a,
        _ => return,
    };
    match name {
        "names" | ".Names" =>
            at.names = Some(str_vec(val).into_iter().map(|o| o.unwrap_or_else(|| Arc::from(""))).collect()),
        "class" => at.class = Some(Arc::from(val_to_str(val).as_str())),
        "dim" => if let Ok(r) = val.as_reals() {
            at.dim = Some(r.into_iter().flatten().map(|x| x as usize).collect());
        },
        other => { at.custom.insert(Arc::from(other), val.clone()); }
    }
}
pub(crate) fn bi_structure(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut v = gv(a,0);
    for arg in &a[1..] {
        if let Some(name) = &arg.name {
            // R accepts `.Data` as the value itself; ignore here.
            if name.as_ref() == ".Data" { continue; }
            set_one_attr(&mut v, name, &arg.value);
        }
    }
    Ok(v)
}
pub(crate) fn bi_format(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    let nsmall = gn(a,"nsmall").and_then(|x| e.scalar_f64(&x).ok().flatten()).unwrap_or(0.0) as usize;
    let out: Vec<Option<Arc<str>>> = match &v {
        RVal::Numeric(nv, _) => nv.iter().copied().map(|o| o.map(|x| {
            let s = if nsmall > 0 { format!("{:.*}", nsmall, x) } else { fmt_f64(x) };
            Arc::from(s.as_str())
        })).collect(),
        other => str_vec(other),
    };
    Ok(RVal::Character(out, Attrs::default()))
}

pub(crate) fn bi_is_num(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Numeric(..)|RVal::Integer(..)))) }
pub(crate) fn bi_is_chr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Character(..)))) }
pub(crate) fn bi_is_lgl(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Logical(..)))) }
pub(crate) fn bi_as_num(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // as.numeric("153") parses the string (NA if not a number), like R.
    if let RVal::Character(v, _) = &gv(a,0) {
        let nums: Vec<Real> = v.iter().map(|x| x.as_ref().and_then(|s| s.trim().parse::<f64>().ok())).collect();
        return Ok(RVal::Numeric(nums.into(), Attrs::default()));
    }
    Ok(RVal::Numeric(e.as_reals(&gv(a,0))?.into(), Attrs::default()))
}
/// `as.single(x)` — coerce to f32 single-precision storage (Phase F.7).
/// Halves memory footprint vs `as.numeric`; arithmetic with `numeric`
/// promotes back to f64.
pub(crate) fn bi_as_single(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let singles = v.as_singles()?;
    Ok(RVal::Single(Singles::new(singles), Attrs::default()))
}
pub(crate) fn bi_is_single(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a,0), RVal::Single(..))))
}
pub(crate) fn bi_as_chr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::Character(v, _) => Ok(RVal::Character(v.clone(), Attrs::default())),
        RVal::Numeric(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|n| Arc::from(fmt_num(n).as_str()))).collect(), Attrs::default())),
        RVal::Integer(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|n| Arc::from(format!("{}", n).as_str()))).collect(), Attrs::default())),
        RVal::Logical(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|b| Arc::from(if b { "TRUE" } else { "FALSE" }))).collect(), Attrs::default())),
        RVal::Factor(f) => Ok(RVal::Character(f.codes.iter().map(|c| c.and_then(|i| f.levels.get(i as usize).cloned())).collect(), Attrs::default())),
        _ => Ok(rstr(&val_to_str(&gv(a,0)))),
    }
}
pub(crate) fn bi_as_int(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let RVal::Character(cv, _) = &gv(a,0) {
        let ints: Vec<Integer> = cv.iter().map(|x| x.as_ref().and_then(|s| s.trim().parse::<f64>().ok()).map(|n| n as i32)).collect();
        return Ok(RVal::Integer(ints.into(), Attrs::default()));
    }
    let v = e.as_reals(&gv(a,0))?; Ok(RVal::Integer(v.into_iter().map(|x| x.map(|n| n as i32)).collect(), Attrs::default()))
}
pub(crate) fn bi_strict(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { e.mode = ErrorMode::Strict; soutln!("Mode: strict"); Ok(RVal::Null) }
pub(crate) fn bi_lenient(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { e.mode = ErrorMode::Lenient; soutln!("Mode: lenient"); Ok(RVal::Null) }
pub(crate) fn bi_df(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let cols: Vec<(Arc<str>, RVal)> = a.iter().enumerate().map(|(i,arg)| { let n = arg.name.clone().unwrap_or_else(|| Arc::from(format!("V{}",i+1).as_str())); (n, arg.value.clone()) }).collect(); Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: None })) }
pub(crate) fn bi_list(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(RVal::List(a.iter().map(|x| (x.name.clone(), x.value.clone())).collect())) }

/// `list.meta(lst)` — introspect a list's per-component shape.
///
/// Returns a list with three named fields:
///   - `$kinds`: character vector of RVal-variant tags per component
///   - `$lens`: integer vector of component lengths
///   - `$total_work`: integer scalar — aggregate work across components
///   - `$homogeneous`: character scalar (`""` if mixed types) — same kind
///                    everywhere when non-empty
///
/// User code can use this to decide whether/how to parallelize over a
/// list's components, mirroring what the engine's auto-dispatch does.
/// Maps onto `r2_types::list_meta()`.
pub(crate) fn bi_list_meta(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let lst = a.first().map(|x| &x.value).ok_or_else(|| R2Err {
        msg: "list.meta: needs a list argument".into(),
        kind: ErrKind::Runtime,
    })?;
    let items = match lst {
        RVal::List(items) => items.clone(),
        _ => return Err(R2Err {
            msg: format!("list.meta: not a list (got {})", lst.type_name()),
            kind: ErrKind::Type,
        }),
    };
    let meta = r2_types::list_meta(&items);
    let kinds: Vec<Character> = meta.components.iter()
        .map(|c| Some(std::sync::Arc::from(c.kind))).collect();
    let lens: Vec<Integer> = meta.components.iter()
        .map(|c| Some(c.len as i32)).collect();
    let homog = match meta.homogeneous_kind {
        Some(k) => std::sync::Arc::from(k),
        None => std::sync::Arc::from(""),
    };
    let mut fields: HashMap<Arc<str>, RVal> = HashMap::new();
    fields.insert(Arc::from("kinds"),       RVal::Character(kinds, Attrs::default()));
    fields.insert(Arc::from("lens"),        RVal::Integer(lens.into(), Attrs::default()));
    fields.insert(Arc::from("total_work"),  RVal::Integer(vec![Some(meta.total_work as i32)].into(), Attrs::default()));
    fields.insert(Arc::from("homogeneous"), RVal::Character(vec![Some(homog)], Attrs::default()));
    Ok(RVal::List(fields.into_iter().map(|(k, v)| (Some(k), v)).collect()))
}

/// GLM family constructors. R's `glm(..., family = binomial())` calls
/// `binomial()` as a function returning a family descriptor. Engine's
/// `bi_glm` consumes either the descriptor list or the bare string
/// `"binomial"` / `"gaussian"` / `"poisson"`. Returning a tagged list
/// keeps the call path R-compatible.
pub(crate) fn make_family(name: &'static str, link: &'static str) -> RVal {
    RVal::List(vec![
        (Some(Arc::from("family")), rstr(name)),
        (Some(Arc::from("link")), rstr(link)),
        (Some(Arc::from("~class")), rstr("family")),
    ])
}
pub(crate) fn bi_binomial(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("binomial", "logit")) }
pub(crate) fn bi_gaussian(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("gaussian", "identity")) }
pub(crate) fn bi_poisson(_:  &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("poisson", "log")) }

/// `subset(df, mask)` — keep rows where `mask` is TRUE.
///
/// NSE form `subset(df, x > 2)` (where `x` resolves against df columns) is
/// supported: the engine pre-processor (see `Expr::Call` dispatch above)
/// evaluates the condition expression in a child env that binds the
/// data-frame's columns as variables, then passes the resulting logical
/// vector to this builtin. Compound conditions like `subset(df, x > 1 & y < 50)`
/// work too. Integration tests live in `tests/nse_subset_transform.rs`.
pub(crate) fn bi_subset(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return err!(Type, "subset() needs a data.frame"),
    };
    let mask: Vec<bool> = e.as_logicals(&gv(a, 1))?
        .into_iter().map(|x| x == Some(true)).collect();
    if mask.len() != df.nrow() {
        return err!(Runtime, "subset: mask length ({}) != nrow ({})", mask.len(), df.nrow());
    }
    fn pick<T: Clone>(v: &[T], m: &[bool]) -> Vec<T> {
        v.iter().zip(m).filter_map(|(x, k)| if *k { Some(x.clone()) } else { None }).collect()
    }
    let cols: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let filtered = match col {
            RVal::Numeric(v, _)   => RVal::Numeric(pick(v, &mask).into(), Attrs::default()),
            RVal::Integer(v, _)   => RVal::Integer(pick(v, &mask).into(), Attrs::default()).into(),
            RVal::Character(v, _) => RVal::Character(pick(v, &mask), Attrs::default()).into(),
            RVal::Logical(v, _)   => RVal::Logical(pick(v, &mask).into(), Attrs::default()).into(),
            _ => col.clone().into(),
        };
        (name.clone(), filtered)
    }).collect();
    Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: None }))
}

/// `transform(df, name = expr)` — append/overwrite named columns.
///
/// NSE form `transform(df, z = x + y)` is supported: the engine
/// pre-processor evaluates each `name = expr` value in a child env binding
/// df columns, so `x` and `y` resolve to the data-frame's columns rather
/// than the global env. Integration tests in `tests/nse_subset_transform.rs`.
pub(crate) fn bi_transform(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return err!(Type, "transform() needs a data.frame"),
    };
    for arg in a.iter().skip(1) {
        let name = match &arg.name {
            Some(n) => n.clone(),
            None => continue, // unnamed extras ignored
        };
        // Replace if column already exists, else append.
        if let Some(pos) = df.columns.iter().position(|(n, _)| n == &name) {
            df.columns[pos] = (name, arg.value.clone());
        } else {
            df.columns.push((name, arg.value.clone()));
        }
    }
    Ok(RVal::DataFrame(df))
}
pub(crate) fn bi_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R's `factor()` coerces numeric/integer/logical to character first.
    // We do the same — converting to string keys and building the levels
    // in order of first appearance.
    let strs: Vec<Option<Arc<str>>> = match &gv(a, 0) {
        RVal::Character(v, _) => v.clone(),
        RVal::Numeric(v, _) => v.iter()
            .map(|x| x.map(|n| Arc::from(fmt_num(n).as_str()))).collect(),
        RVal::Integer(v, _) => v.iter()
            .map(|x| x.map(|n| Arc::from(format!("{}", n).as_str()))).collect(),
        RVal::Logical(v, _) => v.iter()
            .map(|x| x.map(|b| Arc::from(if b { "TRUE" } else { "FALSE" }))).collect(),
        other => return err!(Type, "factor() not supported for {}", other.type_name()),
    };
    // R default: levels are the sorted (alphabetical) unique values — this
    // determines the integer codes and what `levels()`/`as.numeric()` return.
    // (Was first-appearance order, which disagreed with R.)
    let mut levels: Vec<Arc<str>> = Vec::new();
    for x in strs.iter().flatten() {
        if !levels.iter().any(|l| l == x) { levels.push(x.clone()); }
    }
    levels.sort();
    let codes: Vec<Option<u32>> = strs.iter().map(|x| x.as_ref().map(|s| {
        levels.iter().position(|l| l == s).unwrap() as u32
    })).collect();
    Ok(RVal::Factor(Factor { codes, levels, ordered: false }))
}
pub(crate) fn bi_names(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => Ok(RVal::Character(df.columns.iter().map(|(n,_)| Some(n.clone())).collect(), Attrs::default())),
        // Named list: R returns the names, with "" for any unnamed element,
        // or NULL if the list has no names at all.
        RVal::List(items) => {
            if items.iter().all(|(n,_)| n.is_none()) { return Ok(RVal::Null); }
            Ok(RVal::Character(
                items.iter().map(|(n,_)| Some(n.clone().unwrap_or_else(|| std::sync::Arc::from("")))).collect(),
                Attrs::default(),
            ))
        }
        // Atomic vectors carry names in their Attrs (set via `names(v)<-`).
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Character(_, at) | RVal::Logical(_, at) => {
            match &at.names {
                Some(ns) => Ok(RVal::Character(ns.iter().map(|n| Some(n.clone())).collect(), Attrs::default())),
                None => Ok(RVal::Null),
            }
        }
        _ => Ok(RVal::Null),
    }
}
pub(crate) fn bi_str(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    match &v {
        RVal::DataFrame(df) => {
            soutln!("'data.frame':  {} obs. of  {} variables:", df.nrow(), df.ncol());
            for (n, c) in &df.columns {
                let preview = match c {
                    RVal::Numeric(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect();
                        format!(" num  {}", vals.join(" "))
                    }
                    RVal::Integer(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect();
                        format!(" int  {}", vals.join(" "))
                    }
                    RVal::Character(v, _) => {
                        let vals: Vec<String> = v.iter().take(4).map(|x| match x { Some(s) => format!("\"{}\"", s), None => "NA".into() }).collect();
                        format!(" chr  {}", vals.join(" "))
                    }
                    RVal::Logical(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect();
                        format!(" logi {}", vals.join(" "))
                    }
                    RVal::Factor(f) => {
                        let vals: Vec<String> = f.codes.iter().take(6).map(|x| match x { Some(c) => format!("{}", c + 1), None => "NA".into() }).collect();
                        format!(" Factor w/ {} levels {:?}: {}", f.levels.len(), f.levels.iter().take(4).map(|s| s.to_string()).collect::<Vec<_>>(), vals.join(" "))
                    }
                    _ => format!(" {}", c.type_name()),
                };
                soutln!(" $ {:15}:{}", n, preview);
            }
        }
        RVal::Numeric(v, _) => { let vals: Vec<String> = v.iter().take(10).map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect(); soutln!(" num [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::Integer(v, _) => { let vals: Vec<String> = v.iter().take(10).map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(); soutln!(" int [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::Character(v, _) => { let vals: Vec<String> = v.iter().take(5).map(|x| match x { Some(s) => format!("\"{}\"", s), None => "NA".into() }).collect(); soutln!(" chr [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::List(items) => { soutln!("List of {}", items.len()); for (i, (n, v)) in items.iter().enumerate().take(10) { let label = n.as_ref().map(|s| format!("${}", s)).unwrap_or(format!("[[{}]]", i+1)); soutln!(" {} : {} [1:{}]", label, v.type_name(), rval_length(v)); } }
        _ => soutln!(" {} [1:{}]", v.type_name(), rval_length(&v)),
    }
    Ok(RVal::Null)
}
pub(crate) fn bi_summary(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    // Phase R.2 step 5: data-shaped paths (DataFrame, Numeric) handled by
    // r2-data::summary. Returns Some(()) if handled; falls through here
    // for TypeInstance (model summaries) and other inputs.
    if r2_data::summary::try_summary(&v).is_some() {
        return Ok(RVal::Null);
    }
    match &v {
        RVal::DataFrame(df) => {
            // [DEAD: handled by r2-data::summary::try_summary above. Kept
            // for #[allow(unreachable_code)] body-balance.]
            let mut headers: Vec<String> = Vec::new();
            // Pre-extracted per-column work item.
            enum ColData {
                Numeric(Vec<f64>),                      // pre-filtered, NA-stripped
                Char(Vec<Option<Arc<str>>>),            // raw values for counting
                AllNA,
                Other(&'static str),                    // type name to display
            }
            let mut prepped: Vec<ColData> = Vec::with_capacity(df.columns.len());
            for (name, col) in &df.columns {
                headers.push(format!("{:^18}", name));
                let item = match col {
                    RVal::Numeric(_, _) | RVal::Integer(_, _) => {
                        let n: Vec<f64> = e.as_reals(col).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        if n.is_empty() { ColData::AllNA } else { ColData::Numeric(n) }
                    }
                    RVal::Character(vals, _) => ColData::Char(vals.clone()),
                    other => ColData::Other(other.type_name()),
                };
                prepped.push(item);
            }

            // Stage 2: parallel per-column compute (no engine borrow needed).
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(prepped.len() * 100), // weight columns; threshold avoids parallelizing tiny frames
            );
            let compute_one = |item: &ColData| -> Vec<String> {
                let fs = |v: f64| -> String {
                    if (v - v.round()).abs() < 1e-10 { format!("{}", v as i64) }
                    else { let s = format!("{:.4}", v); s.trim_end_matches('0').trim_end_matches('.').to_string() }
                };
                match item {
                    ColData::Numeric(data) => {
                        let mut n = data.clone();
                        n.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let len = n.len();
                        let mean = n.iter().sum::<f64>() / len as f64;
                        let median = if len % 2 == 0 { (n[len/2-1] + n[len/2]) / 2.0 } else { n[len/2] };
                        vec![
                            format!(" Min.   :{:>8}", fs(n[0])),
                            format!(" 1st Qu.:{:>8}", fs(n[len/4])),
                            format!(" Median :{:>8}", fs(median)),
                            format!(" Mean   :{:>8}", fs(mean)),
                            format!(" 3rd Qu.:{:>8}", fs(n[3*len/4])),
                            format!(" Max.   :{:>8}", fs(n[len-1])),
                        ]
                    }
                    ColData::Char(vals) => {
                        let mut counts: Vec<(String, usize)> = Vec::new();
                        for x in vals {
                            if let Some(s) = x {
                                if let Some(entry) = counts.iter_mut().find(|(k, _)| k == s.as_ref()) { entry.1 += 1; }
                                else { counts.push((s.to_string(), 1)); }
                            }
                        }
                        counts.sort_by(|a, b| b.1.cmp(&a.1));
                        let mut lines: Vec<String> = counts.iter().take(6).map(|(k, v)| format!(" {}:{}", k, v)).collect();
                        while lines.len() < 6 { lines.push(String::new()); }
                        lines
                    }
                    ColData::AllNA => vec!["all NA".into(); 6],
                    ColData::Other(t) => vec![format!(" {}", t); 6],
                }
            };
            let col_summaries: Vec<Vec<String>> = if go_par {
                prepped.par_iter().map(|item| compute_one(item)).collect()
            } else {
                prepped.iter().map(|item| compute_one(item)).collect()
            };

            // Print columns side by side
            for h in &headers { sout!("{}", h); }
            soutln!();
            for row in 0..6 {
                for (ci, _) in headers.iter().enumerate() {
                    let s = col_summaries.get(ci).and_then(|c| c.get(row)).map(|s| s.as_str()).unwrap_or("");
                    sout!("{:<18}", s);
                }
                soutln!();
            }
            Ok(RVal::Null)
        }
        RVal::Numeric(v,_) => {
            let mut n: Vec<f64> = v.iter().filter_map(|x| *x).collect();
            if n.is_empty() { soutln!("No data"); return Ok(RVal::Null); }
            n.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let len = n.len();
            let mean = n.iter().sum::<f64>() / len as f64;
            let median = if len % 2 == 0 { (n[len/2-1] + n[len/2]) / 2.0 } else { n[len/2] };
            soutln!("   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.");
            soutln!("{:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                fmt_num(n[0]), fmt_num(n[len/4]), fmt_num(median),
                fmt_num(mean), fmt_num(n[3*len/4]), fmt_num(n[len-1]));
            Ok(RVal::Null)
        }
        RVal::TypeInstance(inst) => {
            // Phase R.S.3 — `summary(lmer)` dispatches to the verbose
            // R-style formatter (scaled residuals + variance/Std.Dev. columns
            // + t-values + p-values + correlation matrix).
            if inst.type_name.as_ref() == "lmer" {
                r2_stats::mixed::format_lmer_summary(inst)?;
                return Ok(RVal::Null);
            }
            match inst.type_name.as_ref() {
                "lm" | "glm" => {
                    // Show the captured original call (`lm(y ~ x, data = df)`)
                    // when available; fall back to the generic placeholder
                    // for old-style positional calls without NSE capture.
                    let call = inst.fields.get("call")
                        .map(|v| val_to_str(v))
                        .unwrap_or_else(|| format!("{}(formula)", inst.type_name));
                    soutln!("\nCall:\n{}", call);
                    // Residuals summary
                    if let Some(res) = inst.fields.get("residuals") {
                        let r: Vec<f64> = e.as_reals(res).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        if !r.is_empty() {
                            let mut sorted = r.clone();
                            sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
                            let n = sorted.len();
                            soutln!("\nResiduals:");
                            soutln!("      Min        1Q    Median        3Q       Max");
                            soutln!("{:>9} {:>9} {:>9} {:>9} {:>9}",
                                fmt_num(sorted[0]), fmt_num(sorted[n/4]),
                                fmt_num(sorted[n/2]), fmt_num(sorted[3*n/4]),
                                fmt_num(sorted[n-1]));
                        }
                    }
                    // Coefficient table with Std.Error, t value, Pr(>|t|)
                    let coefs_val = inst.fields.get("coefficients");
                    let se_val = inst.fields.get("std.errors");
                    let is_glm = inst.type_name.as_ref() == "glm";
                    // glm stores z.values; lm stores t.values. Both use p.values.
                    let stat_val = if is_glm {
                        inst.fields.get("z.values").or_else(|| inst.fields.get("t.values"))
                    } else {
                        inst.fields.get("t.values")
                    };
                    let pv_val = inst.fields.get("p.values");
                    if let Some(cv) = coefs_val {
                        let coeffs: Vec<f64> = e.as_reals(cv).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let se: Vec<f64> = se_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let stat: Vec<f64> = stat_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let pv: Vec<f64> = pv_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let names: Vec<String> = match cv {
                            RVal::Numeric(_, at) => at.names.as_ref().map(|n| n.iter().map(|s| s.to_string()).collect()).unwrap_or_else(|| (0..coeffs.len()).map(|i| format!("X{}", i)).collect()),
                            _ => (0..coeffs.len()).map(|i| format!("X{}", i)).collect(),
                        };
                        let (stat_label, pval_label) = if is_glm {
                            ("z value", "Pr(>|z|)")
                        } else {
                            ("t value", "Pr(>|t|)")
                        };
                        soutln!("\nCoefficients:");
                        soutln!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "", "Estimate", "Std. Error", stat_label, pval_label);
                        for i in 0..coeffs.len() {
                            let s = se.get(i).copied().unwrap_or(0.0);
                            let t = stat.get(i).copied().unwrap_or(0.0);
                            let p = pv.get(i).copied().unwrap_or(1.0);
                            let stars = signif_stars(p);
                            let p_str = fmt_pval(p);
                            soutln!("{:<15} {:>12} {:>12} {:>10} {:>10} {}",
                                names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                                fmt_num(coeffs[i]), fmt_num(s), fmt_num(t), p_str, stars);
                        }
                        soutln!("---");
                        soutln!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
                    }
                    soutln!();
                    // Residual standard error / R² / F-statistic are LM-specific
                    // (gaussian linear model with closed-form OLS). For GLM the
                    // analogous diagnostics are residual deviance + AIC, printed
                    // in the glm-specific block below.
                    if !is_glm {
                        if let Some(sig) = inst.fields.get("sigma") {
                            let sv = e.scalar_f64(sig).ok().flatten().unwrap_or(0.0);
                            sout!("Residual standard error: {}", fmt_num(sv));
                            if let Some(df) = inst.fields.get("df") {
                                let dv = e.scalar_f64(df).ok().flatten().unwrap_or(0.0);
                                sout!(" on {} degrees of freedom", dv as i32);
                            }
                            soutln!();
                        }
                        if let Some(r2) = inst.fields.get("r.squared") {
                            let rv = e.scalar_f64(r2).ok().flatten().unwrap_or(0.0);
                            sout!("Multiple R-squared:  {},", fmt_num(rv));
                        }
                        if let Some(ar2) = inst.fields.get("adj.r.squared") {
                            let av = e.scalar_f64(ar2).ok().flatten().unwrap_or(0.0);
                            soutln!("  Adjusted R-squared:  {}", fmt_num(av));
                        }
                    }
                    if !is_glm { if let Some(fs) = inst.fields.get("f.statistic") {
                        let fv = e.scalar_f64(fs).ok().flatten().unwrap_or(0.0);
                        if let Some(df) = inst.fields.get("df") {
                            let dv = e.scalar_f64(df).ok().flatten().unwrap_or(0.0) as i32;
                            let coefs: Vec<f64> = inst.fields.get("coefficients").and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                            let p_1 = coefs.len().saturating_sub(1);
                            soutln!("F-statistic: {} on {} and {} DF", fmt_num(fv), p_1, dv);
                        } else {
                            soutln!("F-statistic: {}", fmt_num(fv));
                        }
                    } }
                    // GLM-specific diagnostics: Null/Residual deviance + AIC + Fisher iterations.
                    if is_glm {
                        if let Some(d) = inst.fields.get("dispersion") {
                            let dv = e.scalar_f64(d).ok().flatten().unwrap_or(1.0);
                            let fam = inst.fields.get("family").map(|v| val_to_str(v)).unwrap_or_default();
                            soutln!();
                            soutln!("(Dispersion parameter for {} family taken to be {})", fam, fmt_num(dv));
                        }
                        if let (Some(nd), Some(dfn)) = (inst.fields.get("null.deviance"), inst.fields.get("df.null")) {
                            let ndv = e.scalar_f64(nd).ok().flatten().unwrap_or(0.0);
                            let dfn = e.scalar_f64(dfn).ok().flatten().unwrap_or(0.0) as i32;
                            soutln!();
                            soutln!("    Null deviance: {} on {} degrees of freedom", fmt_num(ndv), dfn);
                        }
                        if let (Some(rd), Some(dfr)) = (inst.fields.get("deviance"), inst.fields.get("df.residual")) {
                            let rdv = e.scalar_f64(rd).ok().flatten().unwrap_or(0.0);
                            let dfr = e.scalar_f64(dfr).ok().flatten().unwrap_or(0.0) as i32;
                            soutln!("Residual deviance: {} on {} degrees of freedom", fmt_num(rdv), dfr);
                        }
                        if let Some(aic) = inst.fields.get("aic") {
                            let av = e.scalar_f64(aic).ok().flatten().unwrap_or(0.0);
                            soutln!("AIC: {}", fmt_num(av));
                        }
                        if let Some(it) = inst.fields.get("iter") {
                            let iv = e.scalar_f64(it).ok().flatten().unwrap_or(0.0) as i32;
                            soutln!();
                            soutln!("Number of Fisher Scoring iterations: {}", iv);
                        }
                    }
                }
                "rpart" => {
                    soutln!("\nDecision Tree Summary:");
                    if let Some(tp) = inst.fields.get("type") { soutln!("Type: {}", tp); }
                    if let Some(md) = inst.fields.get("max_depth") { soutln!("Max depth: {}", md); }
                    if let Some(pred) = inst.fields.get("predictions") { soutln!("Training samples: {}", rval_length(pred)); }
                }
                "rf" => {
                    soutln!("\nRandom Forest Summary:");
                    if let Some(nt) = inst.fields.get("ntrees") { soutln!("Number of trees: {}", nt); }
                    if let Some(tp) = inst.fields.get("type") { soutln!("Type: {}", tp); }
                    if let Some(pred) = inst.fields.get("predictions") { soutln!("Training samples: {}", rval_length(pred)); }
                }
                "gbm" => {
                    soutln!("\nGradient Boosted Trees Summary:");
                    if let Some(nt) = inst.fields.get("ntrees") { soutln!("Number of trees: {}", nt); }
                    if let Some(lr) = inst.fields.get("learning_rate") { soutln!("Learning rate: {}", lr); }
                    if let Some(loss) = inst.fields.get("loss") { soutln!("Loss function: {}", loss); }
                    if let Some(tl) = inst.fields.get("train.loss") {
                        let losses = e.as_reals(tl).unwrap_or_default();
                        if let Some(last) = losses.last().and_then(|x| *x) { soutln!("Final training loss: {}", fmt_num(last)); }
                    }
                    if let Some(imp) = inst.fields.get("importance") {
                        soutln!("Feature importance:");
                        let vals = e.as_reals(imp).unwrap_or_default();
                        let names: Vec<String> = inst.fields.get("xnames")
                            .and_then(|v| if let RVal::Character(cs, _) = v { Some(cs.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()) } else { None })
                            .unwrap_or_else(|| (0..vals.len()).map(|i| format!("X{}", i + 1)).collect());
                        let mut indexed: Vec<(usize, f64)> = vals.iter().enumerate().filter_map(|(i, x)| x.map(|v| (i, v * 100.0))).collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        for (i, pct) in indexed.iter().take(10) {
                            if *pct > 0.0 {
                                let label = names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                                soutln!("  {}: {}%", label, fmt_num(*pct));
                            }
                        }
                    }
                }
                "kmeans" => {
                    soutln!("\nK-means Clustering Summary:");
                    if let Some(sz) = inst.fields.get("size") { soutln!("Cluster sizes: {}", sz); }
                    if let Some(tw) = inst.fields.get("tot.withinss") { soutln!("Total within-SS: {}", tw); }
                    if let Some(bs) = inst.fields.get("betweenss") { soutln!("Between-SS: {}", bs); }
                    if let Some(ts) = inst.fields.get("totss") {
                        if let Some(bs) = inst.fields.get("betweenss") {
                            let tot = e.scalar_f64(ts).ok().flatten().unwrap_or(1.0);
                            let bet = e.scalar_f64(bs).ok().flatten().unwrap_or(0.0);
                            soutln!("Between/Total: {}%", fmt_num(bet / tot * 100.0));
                        }
                    }
                }
                "prcomp" => {
                    soutln!("\nPCA Summary:");
                    if let Some(sd) = inst.fields.get("sdev") { soutln!("Standard deviations: {}", sd); }
                    if let Some(pv) = inst.fields.get("prop.variance") { soutln!("Proportion of variance: {}", pv); }
                }
                "cv" => {
                    soutln!("\nCross-Validation Summary:");
                    if let Some(k) = inst.fields.get("k") { soutln!("Folds: {}", k); }
                    if let Some(mm) = inst.fields.get("mean.mse") { soutln!("Mean MSE: {}", mm); }
                    if let Some(sd) = inst.fields.get("sd.mse") { soutln!("SD MSE: {}", sd); }
                }
                "confusion" => {
                    soutln!("\nConfusion Matrix Summary:");
                    if let Some(acc) = inst.fields.get("accuracy") { soutln!("Accuracy: {}", acc); }
                }
                "aov" | "anova" => {
                    // Already printed by aov()/anova() — just suppress field dump
                    let fv = inst.fields.get("f.statistic").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(0.0);
                    let pv = inst.fields.get("p.value").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(1.0);
                    soutln!("\nANOVA: F = {}, p-value = {}", fmt_num(fv), fmt_pval(pv));
                }
                "cor.test" | "shapiro.test" | "wilcox.test" | "fisher.test" | "htest" => {
                    // Already printed by test function — show key result
                    if let Some(pv) = inst.fields.get("p.value") {
                        let p = e.scalar_f64(pv).ok().flatten().unwrap_or(1.0);
                        soutln!("p-value: {}", fmt_pval(p));
                    }
                    if let Some(est) = inst.fields.get("estimate") {
                        let ev = e.scalar_f64(est).ok().flatten().unwrap_or(0.0);
                        soutln!("estimate: {}", fmt_num(ev));
                    }
                }
                _ => {
                    soutln!("\n<{}>", inst.type_name);
                    for (k, v) in &inst.fields {
                        if !k.starts_with('_') { soutln!("  ${}: {}", k, v); }
                    }
                }
            }
            Ok(RVal::Null)
        }
        _ => { soutln!("{}", v); Ok(RVal::Null) }
    }
}
pub(crate) fn bi_search(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { for p in e.registry.search_path() { soutln!("{}", p); } Ok(RVal::Null) }
