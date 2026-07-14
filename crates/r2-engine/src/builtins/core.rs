//! Core builtins — the freestanding `bi_*` functions that take
//! `&mut Engine` but contain self-sufficient logic (no eval-loop
//! coupling). Covers: length/print/cat/typeof/class/is.na/seq/rep/
//! which/abs/sqrt/round/sort/rev, the is.*/as.* coercion family,
//! data.frame(), strict/lenient mode toggles, glm family helpers
//! (binomial/gaussian/poisson), summary(), search(), and friends.
//!
//! Extracted from lib.rs BUILTINS section (engine-split, opus-4.8
//! session, content-anchored). Two module-private helpers
//! (`quoted_vec`, `make_family`) live here too — used only by
//! functions in this file.

#![allow(clippy::needless_pass_by_value)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;

pub(crate) fn bi_length(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Out-of-core column: report its element count from the stored field
    // (no need to open/scan the file).
    if let RVal::TypeInstance(i) = &gv(a,0) {
        if i.type_name.as_ref() == "mmapcol" {
            if let Some(RVal::Numeric(n, _)) = i.fields.get("length") {
                if let Some(Some(len)) = n.as_vec().first() { return Ok(rint(*len as i32)); }
            }
        }
    }
    Ok(rint(rval_length(&gv(a,0)) as i32))
}
pub(crate) fn bi_print(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    // Phase R.T.1 — class-aware print for Date / POSIXct. R prints these in
    // human form (`"2024-03-15"`) rather than the raw days/seconds f64. We
    // dispatch here instead of inside Display because the formatter lives in
    // r2-stats and we don't want r2-types depending on r2-stats.
    if let RVal::Numeric(xs, attrs) = &v {
        match attrs.class.as_deref() {
            Some("Date") => {
                let strs: Vec<String> = xs.iter()
                    .map(|x| match x { Some(d) => format!("\"{}\"", r2_stats::time::format_date(*d, "%Y-%m-%d")), None => "NA".into() })
                    .collect();
                e.emit_output(&quoted_vec(&strs));
                return Ok(v);
            }
            Some("xts") => {
                let dim = attrs.dim.as_ref();
                let idx_v = attrs.custom.get(&std::sync::Arc::from("index"));
                let cls_v = attrs.custom.get(&std::sync::Arc::from("index.class"));
                if let (Some(d), Some(RVal::Numeric(idx, _)), Some(RVal::Character(cls, _))) = (dim, idx_v, cls_v) {
                    if d.len() == 2 {
                        let nrow = d[0];
                        let ncol = d[1];
                        let xs_vec: Vec<Option<f64>> = xs.iter().copied().collect();
                        let idx_vec: Vec<f64> = idx.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
                        let cls_s = cls.first().and_then(|x| x.as_ref().map(|s| s.to_string())).unwrap_or_else(|| "POSIXct".into());
                        e.emit_output(&r2_stats::time::format_xts(&xs_vec, nrow, ncol, &idx_vec, &cls_s, attrs.names.as_deref()));
                        return Ok(v);
                    }
                }
            }
            Some("ts") => {
                if let Some(RVal::Numeric(tsp, _)) = attrs.custom.get(&std::sync::Arc::from("tsp")) {
                    if tsp.len() == 3 {
                        let s = tsp[0].unwrap_or(f64::NAN);
                        let e2 = tsp[1].unwrap_or(f64::NAN);
                        let f = tsp[2].unwrap_or(1.0);
                        let xs_vec: Vec<Option<f64>> = xs.iter().copied().collect();
                        e.emit_output(&r2_stats::time::format_ts(&xs_vec, s, e2, f));
                        return Ok(v);
                    }
                }
            }
            Some("POSIXct") => {
                let strs: Vec<String> = xs.iter()
                    .map(|x| match x { Some(s) => format!("\"{}\"", r2_stats::time::format_posixct(*s, "%Y-%m-%d %H:%M:%S")), None => "NA".into() })
                    .collect();
                e.emit_output(&quoted_vec(&strs));
                return Ok(v);
            }
            _ => {}
        }
    }
    e.emit_output(&format!("{}", v));
    Ok(v)
}

// R-style `[1] "a" "b"` formatter for vector of already-quoted strings.
pub(crate) fn quoted_vec(strs: &[String]) -> String {
    let mut s = String::from("[1] ");
    for (i, x) in strs.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(x);
    }
    s
}
pub(crate) fn bi_cat(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let sep = gn(a,"sep").map(|v| val_to_str(&v)).unwrap_or(" ".into());
    let parts: Vec<String> = a.iter()
        .filter(|x| x.name.as_ref().map(|n| n.as_ref()) != Some("sep"))
        .map(|x| val_to_str(&x.value))
        .collect();
    // cat() does NOT auto-newline (R behavior). The sink's contract
    // is one logical "output chunk" per call — if the assembled text
    // has no trailing newline, the sink may or may not add one
    // depending on impl. StdoutSink adds one for line-buffered I/O.
    e.emit_output(&parts.join(&sep));
    Ok(RVal::Null)
}
/// `clear()` / `cls()` — clear the console. Routes through the
/// frontend-installed clear hook (GUI empties its ConsoleBuffer; CLI
/// emits an ANSI clear-screen). Returns NULL invisibly.
pub(crate) fn bi_clear(_e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_types::out::request_clear();
    Ok(RVal::Null)
}

pub(crate) fn bi_typeof(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rstr(gv(a,0).type_name())) }
pub(crate) fn bi_class(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Read the class attribute (so a Date reports "Date", not "numeric");
    // class_names() falls back to the implicit type when none is set.
    let names = class_names(&gv(a,0));
    Ok(RVal::Character(names.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()))
}
pub(crate) fn bi_is_na(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R: is.na(NaN) is TRUE, and the predicate is elementwise on every
    // atomic type (not just doubles).
    match &gv(a,0) {
        RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(match x { None => true, Some(d) => d.is_nan() })).collect(), Attrs::default())),
        RVal::Integer(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_none())).collect(), Attrs::default())),
        RVal::Logical(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_none())).collect(), Attrs::default())),
        RVal::Character(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_none())).collect(), Attrs::default())),
        _ => Ok(rbool(false)),
    }
}
pub(crate) fn bi_is_nan(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R: is.nan(NA) is FALSE (NA is not NaN); FALSE elementwise for
    // non-double types.
    match &gv(a,0) {
        RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(matches!(x, Some(d) if d.is_nan()))).collect(), Attrs::default())),
        other => Ok(RVal::Logical(vec![Some(false); rval_length(other).max(1)].into(), Attrs::default())),
    }
}
pub(crate) fn bi_is_infinite(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(matches!(x, Some(d) if d.is_infinite()))).collect(), Attrs::default())),
        other => Ok(RVal::Logical(vec![Some(false); rval_length(other).max(1)].into(), Attrs::default())),
    }
}
pub(crate) fn bi_is_finite(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R: NA/NaN/Inf are all not finite; integers/logicals are finite unless NA.
    match &gv(a,0) {
        RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(matches!(x, Some(d) if d.is_finite()))).collect(), Attrs::default())),
        RVal::Integer(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_some())).collect(), Attrs::default())),
        RVal::Logical(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_some())).collect(), Attrs::default())),
        other => Ok(RVal::Logical(vec![Some(false); rval_length(other).max(1)].into(), Attrs::default())),
    }
}
pub(crate) fn bi_seq(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let from = e.scalar_f64(&gv(a,0))?.unwrap_or(1.0);
    let to = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0);
    // `length.out =` wins when supplied: exactly n evenly spaced points from
    // `from` to `to` (R semantics). Previously ignored, so `seq(0, 2*pi,
    // length.out=100)` fell back to by=1 and returned only 7 points.
    let lengthout = gn(a,"length.out").or_else(|| gn(a,"length_out"))
        .and_then(|v| e.scalar_f64(&v).ok().flatten()).map(|n| n as usize);
    if let Some(n) = lengthout {
        let r: Vec<Option<f64>> = match n {
            0 => Vec::new(),
            1 => vec![Some(from)],
            _ => { let step = (to - from) / (n as f64 - 1.0);
                   (0..n).map(|i| Some(from + step * i as f64)).collect() }
        };
        return Ok(RVal::Numeric(r.into(), Attrs::default()));
    }
    let by = gn(a,"by").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(if from<=to {1.0} else {-1.0});
    let mut r = Vec::new(); let mut c = from;
    if by>0.0 { while c<=to+1e-10 { r.push(Some(c)); c+=by; } } else if by<0.0 { while c>=to-1e-10 { r.push(Some(c)); c+=by; } }
    Ok(RVal::Numeric(r.into(), Attrs::default()))
}
pub(crate) fn bi_rep(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    // `times = ` (default 1) and `each = ` (default 1). Both supported,
    // matching R semantics: `rep(c("A","B"), each=3)` → A A A B B B,
    // `rep(c("A","B"), times=3)` → A B A B A B.
    // Critical: arg 1 may be a NAMED arg (`each = 3`), not a positional
    // `times`. Filter on `name.is_none()` before falling back, otherwise
    // `rep(c("A","B","C"), each=3)` reads `times=3` AND `each=3`, giving
    // 27 entries instead of 9.
    let times = gn(a, "times")
        .or_else(|| a.get(1).filter(|p| p.name.is_none()).map(|p| p.value.clone()))
        .and_then(|v| e.scalar_f64(&v).ok().flatten())
        .unwrap_or(1.0) as usize;
    let each = gn(a, "each").and_then(|v| e.scalar_f64(&v).ok().flatten())
        .unwrap_or(1.0) as usize;
    // `length.out =` recycles/truncates the result to exactly that length
    // (R: when given, it wins over `times`).
    let lengthout = gn(a, "length.out").or_else(|| gn(a, "length_out"))
        .and_then(|v| e.scalar_f64(&v).ok().flatten()).map(|n| n as usize);
    fn expand<T: Clone>(v: &[T], each: usize, times: usize, lengthout: Option<usize>) -> Vec<T> {
        let per_pass: Vec<T> = v.iter().flat_map(|x| std::iter::repeat(x.clone()).take(each)).collect();
        let full: Vec<T> = per_pass.iter().cycle().take(per_pass.len() * times).cloned().collect();
        match lengthout {
            Some(n) if !full.is_empty() => full.iter().cycle().take(n).cloned().collect(),
            Some(_) => Vec::new(),
            None => full,
        }
    }
    match &v {
        RVal::Numeric(vs, _)   => Ok(RVal::Numeric(expand(vs, each, times, lengthout).into(), Attrs::default())),
        RVal::Integer(vs, _)   => Ok(RVal::Integer(expand(vs, each, times, lengthout).into(), Attrs::default())).into(),
        RVal::Character(vs, _) => Ok(RVal::Character(expand(vs, each, times, lengthout), Attrs::default())).into(),
        RVal::Logical(vs, _)   => Ok(RVal::Logical(expand(vs, each, times, lengthout).into(), Attrs::default())).into(),
        _ => err!(Runtime, "rep() not supported for {}", v.type_name()).into(),
    }
}
// Phase R: 8 reduction builtins now live in r2-stats. r2-engine adapts
// the pure `(&[EvalArg]) -> Result<RVal, R2Err>` signature to the local
// `BuiltinFn` shape (which carries `&mut Engine` and `&EnvRef`).

pub(crate) fn bi_which(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Logical(v,_) => Ok(RVal::Integer(v.iter().enumerate().filter_map(|(i,x)| if *x==Some(true) { Some(Some((i+1) as i32)) } else { None }).collect(), Attrs::default())), _ => err!(Type, "which requires logical") } }
// Phase K.2: map-kernel dispatch — Rayon decision lives below this layer.
pub(crate) fn bi_abs(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Abs, &v).into(), Attrs::default()))
}
pub(crate) fn bi_sqrt(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sqrt, &v).into(), Attrs::default()))
}
// R rounds half to EVEN (IEC 60559 banker's rounding): round(2.5)=2, round(3.5)=4.
pub(crate) fn bi_round(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let d = if a.len() > 1 { e.scalar_f64(&gv(a,1))?.unwrap_or(0.0) } else { 0.0 } as i32; let f = 10f64.powi(d); Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| (n*f).round_ties_even()/f)).collect(), Attrs::default())) }
pub(crate) fn bi_sort(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let mut n: Vec<f64> = v.into_iter().filter_map(|x| x).collect(); n.sort_by(|a,b| a.partial_cmp(b).unwrap()); Ok(rnums(&n)) }
pub(crate) fn bi_rev(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) {
    RVal::Numeric(v,_)   => Ok(RVal::Numeric(v.iter().rev().cloned().collect(), Attrs::default())),
    RVal::Integer(v,_)   => Ok(RVal::Integer(v.iter().rev().cloned().collect(), Attrs::default())),
    RVal::Character(v,_) => Ok(RVal::Character(v.iter().rev().cloned().collect(), Attrs::default())),
    RVal::Logical(v,_)   => Ok(RVal::Logical(v.iter().rev().cloned().collect(), Attrs::default())),
    RVal::List(items)    => Ok(RVal::List(items.iter().rev().cloned().collect())),
    other => err!(Runtime, "rev() not supported for {}", other.type_name()),
} }
// ── Tier-1 base-R usability builtins ───────────────────────────────

/// Format an f64 the way R prints atomic numbers (integers without a
/// trailing `.0`).
pub(crate) fn fmt_f64(n: f64) -> String {
    if n.is_finite() && n == n.trunc() && n.abs() < 1e15 { format!("{}", n as i64) } else { format!("{}", n) }
}

/// The class vector of a value (explicit `class` attr, else implicit type).
pub(crate) fn class_names(v: &RVal) -> Vec<String> {
    match v {
        RVal::Numeric(_, at) | RVal::Integer(_, at)
        | RVal::Character(_, at) | RVal::Logical(_, at) =>
            at.class.as_ref().map(|c| vec![c.to_string()])
                .unwrap_or_else(|| vec![v.type_name().to_string()]),
        RVal::TypeInstance(i) => vec![i.type_name.to_string()],
        _ => vec![v.type_name().to_string()],
    }
}

/// Coerce any value to a character vector, element-wise.
pub(crate) fn str_vec(v: &RVal) -> Vec<Option<Arc<str>>> {
    match v {
        RVal::Character(cv, _) => cv.clone(),
        RVal::Numeric(nv, _) => nv.iter().copied().map(|x| x.map(|n| Arc::from(fmt_f64(n).as_str()))).collect(),
        RVal::Integer(iv, _) => iv.iter().copied().map(|x| x.map(|n| Arc::from(n.to_string().as_str()))).collect(),
        RVal::Logical(lv, _) => lv.iter().copied().map(|x| x.map(|b| Arc::from(if b { "TRUE" } else { "FALSE" }))).collect(),
        RVal::Factor(f) => f.codes.iter().map(|&c| c.and_then(|i| f.levels.get(i as usize).cloned())).collect(),
        other => vec![Some(Arc::from(val_to_str(other).as_str()))],
    }
}

pub(crate) fn bi_seq_len(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = e.scalar_f64(&gv(a,0))?.unwrap_or(0.0) as i64;
    Ok(rints(&(1..=n.max(0)).map(|i| i as i32).collect::<Vec<_>>()))
}
pub(crate) fn bi_seq_along(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let n = rval_length(&gv(a,0)) as i32;
    Ok(rints(&(1..=n).collect::<Vec<_>>()))
}
pub(crate) fn bi_invisible(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Value passthrough. (Top-level auto-print suppression is a REPL concern.)
    Ok(gv(a,0))
}
pub(crate) fn signif_one(x: f64, d: i32) -> f64 {
    if x == 0.0 || !x.is_finite() { return x; }
    let mag = x.abs().log10().floor() as i32;
    let f = 10f64.powi((d - 1 - mag).clamp(-300, 300));
    (x * f).round() / f
}
pub(crate) fn bi_signif(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let xs = e.as_reals(&gv(a,0))?;
    let d = e.scalar_f64(&gv(a,1))?.unwrap_or(6.0) as i32;
    let out: Vec<Option<f64>> = xs.into_iter().map(|o| o.map(|x| signif_one(x, d.max(1)))).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
pub(crate) fn bi_inherits(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    let cls = class_names(&v);
    let what = str_vec(&gv(a,1));
    let hit = what.iter().flatten().any(|w| cls.iter().any(|c| c.as_str() == w.as_ref()));
    Ok(rbool(hit))
}
pub(crate) fn bi_set_names(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut v = gv(a,0);
    let names: Vec<Arc<str>> = str_vec(&gv(a,1)).into_iter().map(|o| o.unwrap_or_else(|| Arc::from(""))).collect();
    match &mut v {
        RVal::Numeric(_, at) | RVal::Integer(_, at)
        | RVal::Character(_, at) | RVal::Logical(_, at) => at.names = Some(names),
        RVal::List(items) => for (i, (n, _)) in items.iter_mut().enumerate() { *n = names.get(i).cloned(); },
        _ => {}
    }
    Ok(v)
}
pub(crate) fn p_extreme(e: &mut Engine, a: &[EvalArg], want_max: bool) -> Result<RVal, R2Err> {
    let mut vecs = Vec::new();
    for arg in a {
        if arg.name.as_deref() == Some("na.rm") { continue; }
        vecs.push(e.as_reals(&arg.value)?);
    }
    let n = vecs.iter().map(|v| v.len()).max().unwrap_or(0);
    let out: Vec<Option<f64>> = (0..n).map(|i| {
        let mut acc: Option<f64> = None;
        for v in &vecs {
            if v.is_empty() { continue; }
            if let Some(x) = v[i % v.len()] {
                acc = Some(match acc { Some(a2) => if want_max { a2.max(x) } else { a2.min(x) }, None => x });
            }
        }
        acc
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
pub(crate) fn bi_pmin(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { p_extreme(e, a, false) }
pub(crate) fn bi_pmax(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { p_extreme(e, a, true) }

#[derive(Clone, Copy)] pub(crate) enum SetOp { Union, Intersect, Diff }
pub(crate) fn set_op(e: &mut Engine, a: &[EvalArg], op: SetOp) -> Result<RVal, R2Err> {
    let x = gv(a,0); let y = gv(a,1);
    let chr = matches!(x, RVal::Character(..)|RVal::Factor(..)) || matches!(y, RVal::Character(..)|RVal::Factor(..));
    if chr {
        let xs: Vec<Arc<str>> = str_vec(&x).into_iter().flatten().collect();
        let ys: Vec<Arc<str>> = str_vec(&y).into_iter().flatten().collect();
        let mut out: Vec<Arc<str>> = Vec::new();
        match op {
            SetOp::Union => { for v in xs.iter().chain(ys.iter()) { if !out.contains(v) { out.push(v.clone()); } } }
            SetOp::Intersect => { for v in &xs { if ys.contains(v) && !out.contains(v) { out.push(v.clone()); } } }
            SetOp::Diff => { for v in &xs { if !ys.contains(v) && !out.contains(v) { out.push(v.clone()); } } }
        }
        Ok(RVal::Character(out.into_iter().map(Some).collect(), Attrs::default()))
    } else {
        let xs: Vec<f64> = e.as_reals(&x)?.into_iter().flatten().collect();
        let ys: Vec<f64> = e.as_reals(&y)?.into_iter().flatten().collect();
        let mut out: Vec<f64> = Vec::new();
        match op {
            SetOp::Union => { for &v in xs.iter().chain(ys.iter()) { if !out.contains(&v) { out.push(v); } } }
            SetOp::Intersect => { for &v in &xs { if ys.contains(&v) && !out.contains(&v) { out.push(v); } } }
            SetOp::Diff => { for &v in &xs { if !ys.contains(&v) && !out.contains(&v) { out.push(v); } } }
        }
        Ok(rnums(&out))
    }
}
pub(crate) fn bi_union(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { set_op(e, a, SetOp::Union) }
pub(crate) fn bi_intersect(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { set_op(e, a, SetOp::Intersect) }
pub(crate) fn bi_setdiff(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { set_op(e, a, SetOp::Diff) }

/// `x %in% y` — element-wise membership test, returns a logical vector.
pub(crate) fn bi_in(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = gv(a,0); let y = gv(a,1);
    let chr = matches!(x, RVal::Character(..)|RVal::Factor(..)) || matches!(y, RVal::Character(..)|RVal::Factor(..));
    if chr {
        let xs = str_vec(&x);
        let ys: Vec<Arc<str>> = str_vec(&y).into_iter().flatten().collect();
        let out: Vec<Option<bool>> = xs.into_iter().map(|o| Some(o.map(|s| ys.contains(&s)).unwrap_or(false))).collect();
        Ok(RVal::Logical(out.into(), Attrs::default()))
    } else {
        let xs = e.as_reals(&x)?;
        let ys: Vec<f64> = e.as_reals(&y)?.into_iter().flatten().collect();
        let out: Vec<Option<bool>> = xs.into_iter().map(|o| Some(o.map(|v| ys.contains(&v)).unwrap_or(false))).collect();
        Ok(RVal::Logical(out.into(), Attrs::default()))
    }
}

pub(crate) fn bi_append(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = gv(a,0); let vals = gv(a,1);
    let after = gn(a,"after").or_else(|| if a.len() > 2 && a[2].name.is_none() { Some(gv(a,2)) } else { None })
        .and_then(|v| e.scalar_f64(&v).ok().flatten());
    let chr = matches!(x, RVal::Character(..)) || matches!(vals, RVal::Character(..));
    if chr {
        let xs = str_vec(&x); let vs = str_vec(&vals);
        let pos = after.map(|f| f as usize).unwrap_or(xs.len()).min(xs.len());
        let mut out = xs[..pos].to_vec(); out.extend(vs); out.extend_from_slice(&xs[pos..]);
        Ok(RVal::Character(out, Attrs::default()))
    } else {
        let xs = e.as_reals(&x)?; let vs = e.as_reals(&vals)?;
        let pos = after.map(|f| f as usize).unwrap_or(xs.len()).min(xs.len());
        let mut out = xs[..pos].to_vec(); out.extend(vs); out.extend_from_slice(&xs[pos..]);
        Ok(RVal::Numeric(out.into(), Attrs::default()))
    }
}

pub(crate) fn collect_leaves<'a>(v: &'a RVal, out: &mut Vec<&'a RVal>) {
    if let RVal::List(items) = v { for (_, x) in items { collect_leaves(x, out); } } else { out.push(v); }
}
pub(crate) fn bi_unlist(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    let mut leaves = Vec::new(); collect_leaves(&v, &mut leaves);
    let has_chr = leaves.iter().any(|l| matches!(l, RVal::Character(..)|RVal::Factor(..)));
    if has_chr {
        let mut out: Vec<Option<Arc<str>>> = Vec::new();
        for l in &leaves { out.extend(str_vec(l)); }
        Ok(RVal::Character(out, Attrs::default()))
    } else {
        let mut out: Vec<Option<f64>> = Vec::new();
        for l in &leaves { out.extend(e.as_reals(l)?); }
        Ok(RVal::Numeric(out.into(), Attrs::default()))
    }
}

pub(crate) fn bi_cut(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let xs = e.as_reals(&gv(a,0))?;
    let mut br: Vec<f64> = e.as_reals(&gv(a,1))?.into_iter().flatten().collect();
    if br.len() < 2 { return err!(Runtime, "cut(): 'breaks' needs at least 2 values"); }
    br.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let labels: Vec<Arc<str>> = (0..br.len()-1)
        .map(|i| Arc::from(format!("({},{}]", fmt_f64(br[i]), fmt_f64(br[i+1])).as_str()))
        .collect();
    let out: Vec<Option<Arc<str>>> = xs.into_iter().map(|o| o.and_then(|x| {
        (0..br.len()-1).find(|&i| x > br[i] && x <= br[i+1]).map(|i| labels[i].clone())
    })).collect();
    Ok(RVal::Character(out, Attrs::default()))
}

// ── Tier-1 functional / control builtins ───────────────────────────

/// Break a value into its elements as length-1 RVals (list items, or
/// scalars of an atomic vector).
pub(crate) fn elements(v: &RVal) -> Vec<RVal> {
    match v {
        RVal::List(items) => items.iter().map(|(_, x)| x.clone()).collect(),
        RVal::Numeric(nv, _) => nv.iter().copied().map(|o| RVal::Numeric(vec![o].into(), Attrs::default())).collect(),
        RVal::Integer(iv, _) => iv.iter().copied().map(|o| RVal::Integer(vec![o].into(), Attrs::default())).collect(),
        RVal::Character(cv, _) => cv.iter().cloned().map(|o| RVal::Character(vec![o], Attrs::default())).collect(),
        RVal::Logical(lv, _) => lv.iter().copied().map(|o| RVal::Logical(vec![o].into(), Attrs::default())).collect(),
        other => vec![other.clone()],
    }
}
pub(crate) fn is_truthy(e: &mut Engine, v: &RVal) -> bool {
    match v {
        RVal::Logical(lv, _) => lv.first().and_then(|o| *o).unwrap_or(false),
        other => e.scalar_f64(other).ok().flatten().map(|x| x != 0.0).unwrap_or(false),
    }
}

pub(crate) fn bi_reduce(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let xs = elements(&gv(a,1));
    let init = gn(a,"init").or_else(|| if a.len() > 2 && a[2].name.is_none() { Some(gv(a,2)) } else { None });
    let (mut acc, start) = match init {
        Some(v) => (v, 0),
        None => { if xs.is_empty() { return Ok(RVal::Null); } (xs[0].clone(), 1) }
    };
    for x in &xs[start..] {
        acc = e.call_fn(&f, &[EvalArg { name: None, value: acc.clone() },
                              EvalArg { name: None, value: x.clone() }], env)?;
    }
    Ok(acc)
}
pub(crate) fn bi_filter_fp(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let x = gv(a,1);
    let els = elements(&x);
    let mut keep = Vec::new();
    for el in &els {
        let r = e.call_fn(&f, &[EvalArg { name: None, value: el.clone() }], env)?;
        if is_truthy(e, &r) { keep.push(el.clone()); }
    }
    match x {
        RVal::List(_) => Ok(RVal::List(keep.into_iter().map(|v| (None, v)).collect())),
        _ => { let mut out = Vec::new(); for k in &keep { out.extend(e.as_reals(k)?); } Ok(RVal::Numeric(out.into(), Attrs::default())) }
    }
}
pub(crate) fn bi_map(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let vecs: Vec<Vec<RVal>> = a[1..].iter().filter(|arg| arg.name.is_none())
        .map(|arg| elements(&arg.value)).collect();
    let n = vecs.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let call_args: Vec<EvalArg> = vecs.iter()
            .filter(|v| !v.is_empty())
            .map(|v| EvalArg { name: None, value: v[i % v.len()].clone() })
            .collect();
        out.push((None, e.call_fn(&f, &call_args, env)?));
    }
    Ok(RVal::List(out))
}
pub(crate) fn bi_split(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let xr = e.as_reals(&gv(a,0))?;
    let keys = str_vec(&gv(a,1));
    if keys.is_empty() { return err!(Runtime, "split(): empty grouping factor"); }
    let mut groups: Vec<(Arc<str>, Vec<Option<f64>>)> = Vec::new();
    for (i, xv) in xr.iter().enumerate() {
        let k = keys[i % keys.len()].clone().unwrap_or_else(|| Arc::from("NA"));
        if let Some(g) = groups.iter_mut().find(|(gk, _)| *gk == k) { g.1.push(*xv); }
        else { groups.push((k, vec![*xv])); }
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(RVal::List(groups.into_iter()
        .map(|(k, v)| (Some(k), RVal::Numeric(v.into(), Attrs::default()))).collect()))
}
pub(crate) fn bi_stopifnot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    for arg in a {
        let ok = match &arg.value {
            RVal::Logical(lv, _) => !lv.is_empty() && lv.iter().all(|o| *o == Some(true)),
            other => e.as_reals(other).map(|r| !r.is_empty()
                && r.iter().all(|o| matches!(o, Some(x) if *x != 0.0))).unwrap_or(false),
        };
        if !ok {
            let label = arg.name.as_deref().unwrap_or("a condition");
            return err!(Runtime, "{} is not TRUE", label);
        }
    }
    Ok(RVal::Null)
}
pub(crate) fn bi_outer(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let xs: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().flatten().collect();
    let ys: Vec<f64> = e.as_reals(&gv(a,1))?.into_iter().flatten().collect();
    let fun = gn(a,"FUN").or_else(|| if a.len() > 2 && a[2].name.is_none() { Some(gv(a,2)) } else { None });
    let (m, n) = (xs.len(), ys.len());
    let op: Option<char> = match &fun {
        Some(RVal::Character(cv, _)) => cv.first().and_then(|o| o.as_ref()).and_then(|s| s.chars().next()),
        None => Some('*'),
        _ => None,
    };
    let mut data = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let val = match op {
                Some('*') => xs[i] * ys[j],
                Some('+') => xs[i] + ys[j],
                Some('-') => xs[i] - ys[j],
                Some('/') => xs[i] / ys[j],
                Some('^') => xs[i].powf(ys[j]),
                _ => {
                    // FUN is a function value — apply it element-wise.
                    let r = e.call_fn(fun.as_ref().unwrap(),
                        &[EvalArg { name: None, value: rnum(xs[i]) },
                          EvalArg { name: None, value: rnum(ys[j]) }], env)?;
                    e.scalar_f64(&r)?.unwrap_or(f64::NAN)
                }
            };
            data[j * m + i] = val;
        }
    }
    Ok(RVal::Matrix(Matrix::new(data, m, n)))
}

// ── Tier-3 argument / grouping helpers ─────────────────────────────

pub(crate) fn bi_match_arg(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let arg = gv(a,0);
    let choices: Vec<Arc<str>> = str_vec(&gv(a,1)).into_iter().flatten().collect();
    let supplied = str_vec(&arg);
    // No explicit value (NULL) or the whole `choices` vector passed → default.
    if matches!(arg, RVal::Null) || supplied.len() != 1 {
        return Ok(choices.first().map(|s| rstr(s)).unwrap_or(RVal::Null));
    }
    let t = supplied[0].clone().unwrap_or_else(|| Arc::from(""));
    if let Some(c) = choices.iter().find(|c| c.as_ref() == t.as_ref()) { return Ok(rstr(c)); }
    let pref: Vec<&Arc<str>> = choices.iter().filter(|c| c.starts_with(t.as_ref())).collect();
    if pref.len() == 1 { return Ok(rstr(pref[0])); }
    err!(Runtime, "'arg' should be one of the supplied choices")
}
pub(crate) fn bi_ave(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().flatten().collect();
    let n = x.len();
    let pos: Vec<&EvalArg> = a.iter().filter(|e| e.name.is_none()).collect();
    let g: Vec<Arc<str>> = if pos.len() >= 2 {
        str_vec(&pos[1].value).into_iter().map(|o| o.unwrap_or_else(|| Arc::from("NA"))).collect()
    } else { vec![Arc::from(""); n.max(1)] };
    let fun = gn(a, "FUN");
    let mut groups: HashMap<Arc<str>, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let k = g[i % g.len().max(1)].clone();
        groups.entry(k).or_default().push(i);
    }
    let mut out = vec![f64::NAN; n];
    for idxs in groups.values() {
        let vals: Vec<f64> = idxs.iter().map(|&i| x[i]).collect();
        let stat = match &fun {
            Some(f) => e.call_fn(f, &[EvalArg { name: None, value: rnums(&vals) }], env)?
                .scalar_f64().ok().flatten().unwrap_or(f64::NAN),
            None => vals.iter().sum::<f64>() / (vals.len().max(1) as f64),
        };
        for &i in idxs { out[i] = stat; }
    }
    Ok(RVal::Numeric(out.into_iter().map(Some).collect::<Vec<_>>().into(), Attrs::default()))
}
pub(crate) fn bi_nargs(_: &mut Engine, _a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let key: Arc<str> = Arc::from(".nargs");
    Ok(env.lookup(&key).unwrap_or_else(|| rint(0)))
}

// ── Tier-2 1-D numerical methods (function args) ───────────────────

pub(crate) fn interval_bounds(e: &mut Engine, a: &[EvalArg]) -> (f64, f64) {
    if let Some(iv) = gn(a, "interval") {
        if let Ok(r) = iv.as_reals() {
            let v: Vec<f64> = r.into_iter().flatten().collect();
            if v.len() >= 2 { return (v[0], v[1]); }
        }
    }
    let pos: Vec<&EvalArg> = a.iter().filter(|x| x.name.is_none()).collect();
    let lo = gn(a,"lower").and_then(|v| e.scalar_f64(&v).ok().flatten())
        .or_else(|| pos.get(1).and_then(|x| e.scalar_f64(&x.value).ok().flatten())).unwrap_or(0.0);
    let hi = gn(a,"upper").and_then(|v| e.scalar_f64(&v).ok().flatten())
        .or_else(|| pos.get(2).and_then(|x| e.scalar_f64(&x.value).ok().flatten())).unwrap_or(1.0);
    (lo, hi)
}
pub(crate) fn eval_fn1(e: &mut Engine, f: &RVal, x: f64, env: &EnvRef) -> f64 {
    e.call_fn(f, &[EvalArg { name: None, value: rnum(x) }], env)
        .ok().and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(f64::NAN)
}
pub(crate) fn bi_uniroot(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let (mut lo, mut hi) = interval_bounds(e, a);
    let mut flo = eval_fn1(e, &f, lo, env);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        let fm = eval_fn1(e, &f, m, env);
        if flo * fm <= 0.0 { hi = m; } else { lo = m; flo = fm; }
    }
    let root = 0.5 * (lo + hi);
    let froot = eval_fn1(e, &f, root, env);
    Ok(RVal::List(vec![(Some(Arc::from("root")), rnum(root)), (Some(Arc::from("f.root")), rnum(froot))]))
}
pub(crate) fn bi_integrate(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let (lo, hi) = interval_bounds(e, a);
    let nseg = 1000usize;
    let h = (hi - lo) / nseg as f64;
    let mut s = 0.0;
    for i in 0..=nseg {
        let x = lo + h * i as f64;
        let fx = eval_fn1(e, &f, x, env);
        let w = if i == 0 || i == nseg { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * fx;
    }
    Ok(RVal::List(vec![(Some(Arc::from("value")), rnum(s * h / 3.0))]))
}
pub(crate) fn bi_optimize(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let f = gv(a,0);
    let (mut lo, mut hi) = interval_bounds(e, a);
    let maximum = match gn(a,"maximum") {
        Some(RVal::Logical(l, _)) => matches!(l.as_vec().first(), Some(Some(true))),
        Some(other) => e.scalar_f64(&other).ok().flatten().map(|x| x != 0.0).unwrap_or(false),
        None => false,
    };
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let mut fc = eval_fn1(e, &f, c, env);
    let mut fd = eval_fn1(e, &f, d, env);
    for _ in 0..200 {
        let c_better = if maximum { fc > fd } else { fc < fd };
        if c_better { hi = d; d = c; fd = fc; c = hi - gr * (hi - lo); fc = eval_fn1(e, &f, c, env); }
        else { lo = c; c = d; fc = fd; d = lo + gr * (hi - lo); fd = eval_fn1(e, &f, d, env); }
    }
    let xstar = 0.5 * (lo + hi);
    let obj = eval_fn1(e, &f, xstar, env);
    let key = if maximum { "maximum" } else { "minimum" };
    Ok(RVal::List(vec![(Some(Arc::from(key)), rnum(xstar)), (Some(Arc::from("objective")), rnum(obj))]))
}

// ── Tier-3 string / I/O ────────────────────────────────────────────

pub(crate) fn bi_substring(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = str_vec(&gv(a,0));
    let first: Vec<f64> = e.as_reals(&gv(a,1)).unwrap_or_default().into_iter().flatten().collect();
    let last: Vec<f64> = e.as_reals(&gv(a,2)).unwrap_or_default().into_iter().flatten().collect();
    let out: Vec<Option<Arc<str>>> = (0..x.len()).map(|i| {
        x[i].as_ref().map(|s| {
            let chars: Vec<char> = s.chars().collect();
            let f = if first.is_empty() { 1 } else { first[i % first.len()].max(1.0) as usize };
            let l = if last.is_empty() { chars.len() } else { (last[i % last.len()] as usize).min(chars.len()) };
            let fi = f.saturating_sub(1);
            let sub: String = if fi < chars.len() && l >= f { chars[fi..l].iter().collect() } else { String::new() };
            Arc::from(sub.as_str())
        })
    }).collect();
    Ok(RVal::Character(out, Attrs::default()))
}
pub(crate) fn bi_read_lines(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a,0));
    let content = std::fs::read_to_string(&path)
        .map_err(|err| R2Err { msg: format!("readLines: cannot open '{}': {}", path, err), kind: ErrKind::Runtime })?;
    let mut lines: Vec<Option<Arc<str>>> = content.lines().map(|l| Some(Arc::from(l))).collect();
    if let Some(n) = e.scalar_f64(&gv(a,1)).ok().flatten() {
        if n >= 0.0 { lines.truncate(n as usize); }
    }
    Ok(RVal::Character(lines, Attrs::default()))
}
pub(crate) fn bi_write_lines(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let text = str_vec(&gv(a,0));
    let body: String = text.iter().map(|o| o.as_deref().unwrap_or("NA")).collect::<Vec<_>>().join("\n");
    let con = gn(a,"con").or_else(|| if a.len() > 1 && a[1].name.is_none() { Some(gv(a,1)) } else { None });
    match con {
        Some(c) if !matches!(c, RVal::Null) => {
            let path = val_to_str(&c);
            std::fs::write(&path, format!("{}\n", body))
                .map_err(|err| R2Err { msg: format!("writeLines: cannot write '{}': {}", path, err), kind: ErrKind::Runtime })?;
        }
        _ => { e.emit_output(&format!("{}\n", body)); }
    }
    Ok(RVal::Null)
}

