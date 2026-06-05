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

use rayon::prelude::*;
use r2_stats::htest::{fmt_pval, signif_stars};
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
fn quoted_vec(strs: &[String]) -> String {
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
pub(crate) fn bi_class(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::TypeInstance(i) => Ok(rstr(&i.type_name)), v => Ok(rstr(v.type_name())) } }
pub(crate) fn bi_is_na(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_none())).collect(), Attrs::default())), _ => Ok(rbool(false)) } }
pub(crate) fn bi_seq(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let from = e.scalar_f64(&gv(a,0))?.unwrap_or(1.0); let to = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0); let by = gn(a,"by").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(if from<=to {1.0} else {-1.0}); let mut r = Vec::new(); let mut c = from; if by>0.0 { while c<=to+1e-10 { r.push(Some(c)); c+=by; } } else if by<0.0 { while c>=to-1e-10 { r.push(Some(c)); c+=by; } } Ok(RVal::Numeric(r.into(), Attrs::default())) }
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
    fn expand<T: Clone>(v: &[T], each: usize, times: usize) -> Vec<T> {
        let per_pass: Vec<T> = v.iter().flat_map(|x| std::iter::repeat(x.clone()).take(each)).collect();
        per_pass.iter().cycle().take(per_pass.len() * times).cloned().collect()
    }
    match &v {
        RVal::Numeric(vs, _)   => Ok(RVal::Numeric(expand(vs, each, times).into(), Attrs::default())),
        RVal::Integer(vs, _)   => Ok(RVal::Integer(expand(vs, each, times).into(), Attrs::default())).into(),
        RVal::Character(vs, _) => Ok(RVal::Character(expand(vs, each, times), Attrs::default())).into(),
        RVal::Logical(vs, _)   => Ok(RVal::Logical(expand(vs, each, times).into(), Attrs::default())).into(),
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
pub(crate) fn bi_round(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let d = e.scalar_f64(&gv(a,1))?.unwrap_or(0.0) as i32; let f = 10f64.powi(d); Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| (n*f).round()/f)).collect(), Attrs::default())) }
pub(crate) fn bi_sort(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let mut n: Vec<f64> = v.into_iter().filter_map(|x| x).collect(); n.sort_by(|a,b| a.partial_cmp(b).unwrap()); Ok(rnums(&n)) }
pub(crate) fn bi_rev(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Numeric(v,_) => Ok(RVal::Numeric(v.iter().rev().cloned().collect(), Attrs::default())), _ => err!(Runtime, "rev() works with numeric, integer, or character vectors") } }
pub(crate) fn bi_is_num(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Numeric(..)|RVal::Integer(..)))) }
pub(crate) fn bi_is_chr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Character(..)))) }
pub(crate) fn bi_is_lgl(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Logical(..)))) }
pub(crate) fn bi_as_num(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(RVal::Numeric(e.as_reals(&gv(a,0))?.into(), Attrs::default())) }
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
pub(crate) fn bi_as_int(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; Ok(RVal::Integer(v.into_iter().map(|x| x.map(|n| n as i32)).collect(), Attrs::default())) }
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
fn make_family(name: &'static str, link: &'static str) -> RVal {
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
    let mut levels: Vec<Arc<str>> = Vec::new();
    let codes: Vec<Option<u32>> = strs.iter().map(|x| x.as_ref().map(|s| {
        let idx = levels.iter().position(|l: &Arc<str>| l == s).unwrap_or_else(|| {
            levels.push(s.clone()); levels.len() - 1
        });
        idx as u32
    })).collect();
    Ok(RVal::Factor(Factor { codes, levels, ordered: false }))
}
pub(crate) fn bi_names(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::DataFrame(df) => Ok(RVal::Character(df.columns.iter().map(|(n,_)| Some(n.clone())).collect(), Attrs::default())), _ => Ok(RVal::Null) } }
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
