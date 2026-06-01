//! DATA MANIPULATION + NA HANDLING + APPLY FAMILY + MORE MATH —
//! a large cohesive block extracted from lib.rs (engine-split,
//! opus-4.8 session, content-anchored).
//!
//! Covers: rbind/cbind/merge/subset/transform/within, na.omit/
//! complete.cases/is.null/ifelse, apply/tapply/aggregate/do.call,
//! log/exp/ceiling/floor/cumsum/cumprod/diff/range/median/quantile.
//!
//! Module-private helpers (`coerce_to_columns`, `all_dataframes`,
//! `to_string_vec`) are used only within this file. The Oracle
//! parallel/serial dispatch is reached via the full `r2_oracle::`
//! path (no import needed).

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::all)]

use std::sync::Arc;

use rayon::prelude::*;
use r2_types::*;

use crate::{gv, gn, pure_apply, Engine};
use crate::err;

// Helper: coerce an RVal to a column of f64 (for matrix-style cbind/rbind).
// Returns (data, nrows). Matrix input contributes ncol columns of nrow rows.
fn coerce_to_columns(v: &RVal) -> Result<(Vec<f64>, usize, usize), R2Err> {
    match v {
        RVal::Matrix(m) => Ok((m.data.clone(), m.nrow, m.ncol)),
        RVal::Numeric(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Integer(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|i| i as f64).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Logical(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 }).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        _ => err!(Type, "cbind/rbind: cannot coerce {} to numeric matrix", v.type_name()),
    }
}

fn all_dataframes(a: &[EvalArg]) -> bool {
    !a.is_empty() && a.iter().all(|x| matches!(x.value, RVal::DataFrame(_)))
}

// Phase R.2: bi_rbind moved to r2-data::bind. Engine adapter only.
pub(crate) fn bi_rbind(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::bind::bi_rbind(a);
    #[allow(unreachable_code)]
    {
    if a.is_empty() { return err!(Runtime, "rbind: needs at least one argument"); }

    // DataFrame path: all args are data.frames → stack rows
    if all_dataframes(a) {
        let mut iter = a.iter();
        let first = match &iter.next().unwrap().value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
        let ncol = first.ncol();
        let mut columns: Vec<(Arc<str>, RVal)> = first.columns.clone();
        for arg in iter {
            let df = match &arg.value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
            if df.ncol() != ncol { return err!(Runtime, "rbind: column count mismatch ({} vs {})", ncol, df.ncol()); }
            for (i, (name, col2)) in df.columns.iter().enumerate() {
                let (cur_name, cur_col) = columns[i].clone();
                let merged = match (&cur_col, col2) {
                    (RVal::Numeric(v1,_), RVal::Numeric(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Numeric(v.into(), Attrs::default()) }
                    (RVal::Integer(v1,_), RVal::Integer(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Integer(v.into(), Attrs::default()) }
                    (RVal::Character(v1,_), RVal::Character(v2,_)) => { let mut v = v1.clone(); v.extend(v2.clone()); RVal::Character(v, Attrs::default()) }
                    (RVal::Logical(v1,_), RVal::Logical(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Logical(v.into(), Attrs::default()) }
                    _ => return err!(Type, "rbind: incompatible column types at '{}'", name),
                };
                columns[i] = (cur_name, merged);
            }
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path: stack matrices/vectors as rows.
    // A vector v of length k becomes a 1-row, k-column matrix.
    // A matrix contributes its rows as-is.
    let mut blocks: Vec<(Vec<f64>, usize, usize)> = Vec::with_capacity(a.len());
    for arg in a {
        let (data, nrow, ncol) = match &arg.value {
            RVal::Matrix(m) => (m.data.clone(), m.nrow, m.ncol),
            other => {
                let (d, n, _) = coerce_to_columns(other)?;
                // Vector → 1 row, n columns
                (d, 1, n)
            }
        };
        blocks.push((data, nrow, ncol));
    }
    let ncol = blocks[0].2;
    if !blocks.iter().all(|(_, _, c)| *c == ncol) {
        return err!(Runtime, "rbind: column count mismatch across inputs");
    }
    let total_rows: usize = blocks.iter().map(|(_, r, _)| *r).sum();
    // Build column-major output: for each column j, append rows from each block in order.
    let mut data = vec![0.0; total_rows * ncol];
    for j in 0..ncol {
        let mut row_offset = 0;
        for (b_data, b_nrow, _) in &blocks {
            for i in 0..*b_nrow {
                data[j * total_rows + row_offset + i] = b_data[j * b_nrow + i];
            }
            row_offset += b_nrow;
        }
    }
    Ok(RVal::Matrix(Matrix::new(data, total_rows, ncol)))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_cbind moved to r2-data::bind. Engine adapter only.
pub(crate) fn bi_cbind(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::bind::bi_cbind(a);
    #[allow(unreachable_code)]
    {
    if a.is_empty() { return err!(Runtime, "cbind: needs at least one argument"); }

    // DataFrame path: all args are data.frames → side-by-side columns
    if all_dataframes(a) {
        let mut iter = a.iter();
        let first = match &iter.next().unwrap().value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
        let nrow = first.nrow();
        let mut columns: Vec<(Arc<str>, RVal)> = first.columns;
        for arg in iter {
            let df = match &arg.value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
            if df.nrow() != nrow { return err!(Runtime, "cbind: row count mismatch ({} vs {})", nrow, df.nrow()); }
            columns.extend(df.columns);
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path: collect each input as one or more columns of f64.
    // Matrix → its columns; vector (Numeric/Integer/Logical) → one column.
    // Track preserved column names where available.
    let mut blocks: Vec<(Vec<f64>, usize, usize, Option<Vec<Arc<str>>>)> = Vec::with_capacity(a.len());
    let mut any_names = false;
    for arg in a {
        let (data, nrow, ncol) = coerce_to_columns(&arg.value)?;
        let names: Option<Vec<Arc<str>>> = match &arg.value {
            RVal::Matrix(m) => m.col_names.clone(),
            _ => arg.name.as_ref().map(|n| vec![n.clone()]),
        };
        if names.is_some() { any_names = true; }
        blocks.push((data, nrow, ncol, names));
    }
    let nrow = blocks[0].1;
    if !blocks.iter().all(|(_, r, _, _)| *r == nrow) {
        return err!(Runtime, "cbind: row count mismatch across inputs");
    }
    let total_cols: usize = blocks.iter().map(|(_, _, c, _)| *c).sum();
    let mut data = Vec::with_capacity(nrow * total_cols);
    let mut col_names: Vec<Arc<str>> = Vec::with_capacity(total_cols);
    for (b_data, _, b_ncol, b_names) in &blocks {
        data.extend_from_slice(b_data);
        match b_names {
            Some(ns) if ns.len() == *b_ncol => col_names.extend(ns.iter().cloned()),
            _ => for j in 0..*b_ncol { col_names.push(Arc::from(format!("V{}", col_names.len() + 1).as_str())); }
        }
    }
    let mut m = Matrix::new(data, nrow, total_cols);
    if any_names { m.col_names = Some(col_names); }
    Ok(RVal::Matrix(m))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}


fn to_string_vec(col: &RVal) -> Vec<String> {
    match col {
        RVal::Numeric(v,_) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Integer(v,_) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Character(v,_) => v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect(),
        RVal::Logical(v,_) => v.iter().map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect(),
        _ => Vec::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// NA HANDLING: na.omit, complete.cases, is.null, ifelse
// ═══════════════════════════════════════════════════════════════════════



pub(crate) fn bi_is_null(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a,0), RVal::Null)))
}

pub(crate) fn bi_ifelse(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // ifelse(test, yes, no) — vectorized conditional
    let test = e.as_logicals(&gv(a,0))?;
    let yes_val = gv(a,1);
    let no_val = gv(a,2);
    let yes = e.as_reals(&yes_val)?;
    let no = e.as_reals(&no_val)?;
    let result: Vec<Real> = test.iter().enumerate().map(|(i, t)| {
        match t {
            Some(true) => yes.get(i % yes.len()).copied().unwrap_or(None),
            Some(false) => no.get(i % no.len()).copied().unwrap_or(None),
            None => None,
        }
    }).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

// ═══════════════════════════════════════════════════════════════════════
// APPLY FAMILY: apply, tapply, aggregate, do.call
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_apply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_apply(e, a, env);
    #[allow(unreachable_code)] {
    // [DEAD: original body kept for safe rollback — Phase R.2 step 6]
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Type, "apply needs data.frame or matrix") };
    let margin = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0) as i32;
    let func = gv(a,2);

    // Detect pure-builtin fast path.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    if margin == 2 {
        // Per-column inputs.
        let inputs: Vec<RVal> = df.columns.iter().map(|(_, col)| col.clone()).collect();
        let results: Vec<RVal> = match &pure_name {
            Some(fname) => {
                let go_par = r2_oracle::should_parallelize(
                    r2_oracle::Op::PerElementMap,
                    r2_oracle::Shape::n(inputs.len() * 100),
                );
                if go_par {
                    inputs.par_iter().map(|c| pure_apply(fname, c).unwrap_or(Ok(RVal::Null)))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut r = Vec::with_capacity(inputs.len());
                    for c in &inputs { r.push(pure_apply(fname, c).unwrap_or(Ok(RVal::Null))?); }
                    r
                }
            }
            None => {
                let mut r = Vec::with_capacity(inputs.len());
                for c in &inputs {
                    let args = vec![EvalArg { name: None, value: c.clone() }];
                    r.push(e.call_fn(&func, &args, env)?);
                }
                r
            }
        };
        // Simplify to numeric if every result is a scalar Numeric.
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r {
                RVal::Numeric(v,_) if v.len() == 1 => nums.push(v[0]),
                _ => { all_scalar = false; break; }
            }
        }
        if all_scalar {
            let mut attrs = Attrs::default();
            attrs.names = Some(df.columns.iter().map(|(n,_)| n.clone()).collect());
            Ok(RVal::Numeric(nums.into(), attrs))
        } else {
            Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect()))
        }
    } else {
        // Per-row inputs (margin == 1). Pre-extract every row to a Numeric.
        let nrow = df.nrow();
        let rows: Vec<RVal> = (0..nrow).map(|r| {
            let row: Vec<Real> = df.columns.iter().filter_map(|(_, col)| {
                match col {
                    RVal::Numeric(v,_) => v.get(r).copied(),
                    RVal::Integer(v,_) => v.get(r).map(|x| x.map(|n| n as f64)),
                    _ => None,
                }
            }).collect();
            RVal::Numeric(row.into(), Attrs::default())
        }).collect();
        let results: Vec<RVal> = match &pure_name {
            Some(fname) => {
                let go_par = r2_oracle::should_parallelize(
                    r2_oracle::Op::PerElementMap,
                    r2_oracle::Shape::n(rows.len() * 100),
                );
                if go_par {
                    rows.par_iter().map(|r| pure_apply(fname, r).unwrap_or(Ok(RVal::Null)))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut out = Vec::with_capacity(rows.len());
                    for r in &rows { out.push(pure_apply(fname, r).unwrap_or(Ok(RVal::Null))?); }
                    out
                }
            }
            None => {
                let mut out = Vec::with_capacity(rows.len());
                for r in rows {
                    let args = vec![EvalArg { name: None, value: r }];
                    out.push(e.call_fn(&func, &args, env)?);
                }
                out
            }
        };
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r { RVal::Numeric(v,_) if v.len() == 1 => nums.push(v[0]), _ => { all_scalar = false; break; } }
        }
        if all_scalar { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
        else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
    }
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

pub(crate) fn bi_tapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_tapply(e, a, env);
    #[allow(unreachable_code)] {
    let x = e.as_reals(&gv(a,0))?;
    let index = to_string_vec(&gv(a,1));
    let func = gv(a,2);

    // Group values by index
    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in index.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k,_)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    // Phase D: parallel fast path when FUN is a pure builtin.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();

    let computed: Vec<RVal> = match &pure_name {
        Some(fname) => {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(group_inputs.len() * 100),
            );
            if go_par {
                group_inputs.par_iter().map(|input| pure_apply(fname, input).unwrap_or(Ok(RVal::Null)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(group_inputs.len());
                for input in &group_inputs { r.push(pure_apply(fname, input).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        }
        None => {
            let mut r = Vec::with_capacity(group_inputs.len());
            for input in group_inputs {
                let args = vec![EvalArg { name: None, value: input }];
                r.push(e.call_fn(&func, &args, env)?);
            }
            r
        }
    };

    let results: Vec<(Option<Arc<str>>, RVal)> = groups.iter().zip(computed.into_iter())
        .map(|((key, _), result)| (Some(Arc::from(key.as_str())), result))
        .collect();
    Ok(RVal::List(results))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

pub(crate) fn bi_aggregate(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_aggregate(e, a, env);
    #[allow(unreachable_code)] {
    let x = e.as_reals(&gv(a,0))?;
    let by = to_string_vec(&gn(a, "by").unwrap_or(gv(a, 1)));
    let func = gn(a, "FUN").unwrap_or(gv(a, 2));

    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in by.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k,_)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    // Phase D: parallel fast path when FUN is a pure builtin.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();
    let computed: Vec<RVal> = match &pure_name {
        Some(fname) => {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(group_inputs.len() * 100),
            );
            if go_par {
                group_inputs.par_iter().map(|input| pure_apply(fname, input).unwrap_or(Ok(RVal::Null)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(group_inputs.len());
                for input in &group_inputs { r.push(pure_apply(fname, input).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        }
        None => {
            let mut r = Vec::with_capacity(group_inputs.len());
            for input in group_inputs {
                let args = vec![EvalArg { name: None, value: input }];
                r.push(e.call_fn(&func, &args, env)?);
            }
            r
        }
    };

    let mut group_names: Vec<Character> = Vec::with_capacity(groups.len());
    let mut agg_values: Vec<Real> = Vec::with_capacity(groups.len());
    for ((key, _), result) in groups.iter().zip(computed.into_iter()) {
        group_names.push(Some(Arc::from(key.as_str())));
        if let Ok(v) = e.scalar_f64(&result) { agg_values.push(v); } else { agg_values.push(None); }
    }

    Ok(RVal::DataFrame(DataFrame {
        columns: vec![
            (Arc::from("Group"), RVal::Character(group_names, Attrs::default())),
            (Arc::from("Value"), RVal::Numeric(agg_values.into(), Attrs::default())),
        ],
        row_names: None,
    }))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}


// ═══════════════════════════════════════════════════════════════════════
// MORE MATH: log, exp, ceiling, floor, cumsum, cumprod, diff, range, median, quantile
// ═══════════════════════════════════════════════════════════════════════

// Phase K.2: log dispatches to specialized kernel ops when base matches a
// well-known constant (e, 2, 10) for max efficiency. Other bases route
// through Ln + a scalar-divide step (still kernel-dispatched).
pub(crate) fn bi_log(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    let base = gn(a,"base").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(std::f64::consts::E);
    let result = if (base - std::f64::consts::E).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Ln, &v)
    } else if (base - 2.0).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Log2, &v)
    } else if (base - 10.0).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Log10, &v)
    } else {
        // Arbitrary base: Ln then divide. Two passes; specialized base
        // ops above are the common case.
        let lns = r2_kernel::map(r2_kernel::MapOp::Ln, &v);
        let lb = base.ln();
        lns.into_iter().map(|x| x.map(|n| n / lb)).collect()
    };
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}
pub(crate) fn bi_exp(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Exp, &v).into(), Attrs::default()))
}

// Phase R.M.1 — trig and transcendental builtins. Each is a thin
// wrapper that coerces the first argument to Real and routes through
// the kernel's element-wise dispatcher (so it picks up SIMD/parallel
// variants automatically based on size).
pub(crate) fn bi_sin   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sin, &v).into(), Attrs::default()))
}
pub(crate) fn bi_cos   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Cos, &v).into(), Attrs::default()))
}
pub(crate) fn bi_tan   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Tan, &v).into(), Attrs::default()))
}
pub(crate) fn bi_asin  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Asin, &v).into(), Attrs::default()))
}
pub(crate) fn bi_acos  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Acos, &v).into(), Attrs::default()))
}
pub(crate) fn bi_atan  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Atan, &v).into(), Attrs::default()))
}
pub(crate) fn bi_atan2 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Two-argument atan2(y, x). Element-wise pairing; recycle shorter side
    // to match R's vectorization rule.
    let y = e.as_reals(&gv(a,0))?;
    let x = e.as_reals(&gv(a,1))?;
    let n = y.len().max(x.len());
    if y.is_empty() || x.is_empty() {
        return Ok(RVal::Numeric(Vec::<Option<f64>>::new().into(), Attrs::default()));
    }
    let out: Vec<Option<f64>> = (0..n).map(|i| {
        let yi = y[i % y.len()]; let xi = x[i % x.len()];
        match (yi, xi) { (Some(a), Some(b)) => Some(a.atan2(b)), _ => None }
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
pub(crate) fn bi_sinh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sinh, &v).into(), Attrs::default()))
}
pub(crate) fn bi_cosh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Cosh, &v).into(), Attrs::default()))
}
pub(crate) fn bi_tanh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Tanh, &v).into(), Attrs::default()))
}
pub(crate) fn bi_sign  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sign, &v).into(), Attrs::default()))
}
pub(crate) fn bi_trunc (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Trunc, &v).into(), Attrs::default()))
}
pub(crate) fn bi_expm1 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Expm1, &v).into(), Attrs::default()))
}
pub(crate) fn bi_log1p (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log1p, &v).into(), Attrs::default()))
}
pub(crate) fn bi_log2  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log2, &v).into(), Attrs::default()))
}
pub(crate) fn bi_log10 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log10, &v).into(), Attrs::default()))
}
pub(crate) fn bi_ceiling(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| n.ceil())).collect(), Attrs::default()))
}
pub(crate) fn bi_floor(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| n.floor())).collect(), Attrs::default()))
}
// Phase K.9 — rolling/window reductions
pub(crate) fn bi_median(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { return r2_stats::bi_median(a);
    #[allow(unreachable_code)]
    let mut v: Vec<f64> = vec![];
    let n = v.len();

    // Phase D: Oracle picks between two correct algorithms.
    //   - Serial (small n)   → quickselect via `select_nth_unstable_by` — O(n).
    //   - Parallel (large n) → Rayon `par_sort_by` then index — uses all cores.
    // Quickselect alone outperforms `sort` for medians, but doesn't parallelize.
    // Sort-based path is the one that benefits from Rayon.
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    let go_par = r2_oracle::should_parallelize(
        r2_oracle::Op::Reduction,
        r2_oracle::Shape::n(n),
    );
    let m = if go_par {
        v.par_sort_by(cmp);
        if n % 2 == 0 { (v[n/2 - 1] + v[n/2]) / 2.0 } else { v[n/2] }
    } else if n % 2 == 0 {
        // Need both middle elements. Quickselect the upper, then max of the lower half.
        let upper_idx = n / 2;
        let (_lower, upper, _) = v.select_nth_unstable_by(upper_idx, cmp);
        let upper_val = *upper;
        let lower_val = _lower.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (lower_val + upper_val) / 2.0
    } else {
        let mid = n / 2;
        let (_, m, _) = v.select_nth_unstable_by(mid, cmp);
        *m
    };
    Ok(rnum(m))
}
