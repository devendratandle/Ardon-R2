//! Apply family — Phase R.2 step 6 (Option 2: full migration via EngineCtx).
//!
//! `lapply`, `sapply`, `apply`, `tapply`, `aggregate`, `mapply`, `vapply`.
//!
//! These functions iterate over a list/vector/data.frame and invoke a
//! function per item. The inner function may be:
//!   - A pure builtin (`sum`, `mean`, ...) — fast path via `pure_apply`,
//!     no engine needed; parallelism via `kernel::par_for`.
//!   - Anything else (Closure, non-pure builtin) — falls back to
//!     `EngineCtx::ctx_call_fn` to evaluate per item.
//!
//! Phase R.2 step 6 design: domain crate takes `&mut dyn EngineCtx`
//! instead of `&mut Engine`. r2-engine implements `EngineCtx` for
//! `Engine`; r2-data has no engine dependency.
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only
//!   §4.7 Backwards-compatible — engine adapters delegate
//!   §4.9 Parallelism via `r2_kernel::par_for`, not `par_iter`

use r2_oracle::Op;
use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
fn first_arg(args: &[EvalArg]) -> RVal { args.first().map(|a| a.value.clone()).unwrap_or(RVal::Null) }
#[inline]
fn nth_arg(args: &[EvalArg], i: usize) -> RVal { args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null) }

// ── pure_apply allowlist ────────────────────────────────────────────
//
// Pure-function dispatch for builtins safe to call from threads.
// To extend: add a match arm. Avoid anything that needs engine state.

pub fn pure_apply(name: &str, arg: &RVal) -> Option<Result<RVal, R2Err>> {
    let coerce_reals = |v: &RVal| -> Option<Vec<Real>> {
        match v {
            RVal::Numeric(vs, _) => Some(vs.as_vec().clone()),
            RVal::Integer(vs, _) => Some(vs.iter().map(|x| x.map(|n| n as f64)).collect()),
            RVal::Logical(vs, _) => Some(vs.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect()),
            RVal::Matrix(m) => Some(m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect()),
            _ => None,
        }
    };
    match name {
        "sum"  => { let v = coerce_reals(arg)?; let s: Real = v.iter().try_fold(0.0f64, |acc, x| x.map(|n| acc + n));
                    Some(Ok(RVal::Numeric(vec![s].into(), Attrs::default()))) }
        "mean" => { let v = coerce_reals(arg)?; let n = v.len() as f64;
                    let s: Real = v.iter().try_fold(0.0f64, |acc, x| x.map(|val| acc + val));
                    Some(Ok(RVal::Numeric(vec![s.map(|t| t / n)].into(), Attrs::default()))) }
        "sd"   => { let v = coerce_reals(arg)?; let nums: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                    let n = nums.len(); if n < 2 { return Some(Ok(RVal::Numeric(vec![None].into(), Attrs::default()))); }
                    let mean = nums.iter().sum::<f64>() / n as f64;
                    let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                    Some(Ok(RVal::Numeric(vec![Some(var.sqrt())].into(), Attrs::default()))) }
        "var"  => { let v = coerce_reals(arg)?; let nums: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                    let n = nums.len(); if n < 2 { return Some(Ok(RVal::Numeric(vec![None].into(), Attrs::default()))); }
                    let mean = nums.iter().sum::<f64>() / n as f64;
                    let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                    Some(Ok(RVal::Numeric(vec![Some(var)].into(), Attrs::default()))) }
        "min"  => { let v = coerce_reals(arg)?;
                    let m = v.iter().filter_map(|x| *x).fold(f64::INFINITY, f64::min);
                    Some(Ok(RVal::Numeric(vec![Some(m)].into(), Attrs::default()))) }
        "max"  => { let v = coerce_reals(arg)?;
                    let m = v.iter().filter_map(|x| *x).fold(f64::NEG_INFINITY, f64::max);
                    Some(Ok(RVal::Numeric(vec![Some(m)].into(), Attrs::default()))) }
        "prod" => { let v = coerce_reals(arg)?;
                    let p: Real = v.iter().try_fold(1.0f64, |acc, x| x.map(|n| acc * n));
                    Some(Ok(RVal::Numeric(vec![p].into(), Attrs::default()))) }
        "length" => {
            let n = match arg {
                RVal::Numeric(v, _) => v.len(), RVal::Integer(v, _) => v.len(),
                RVal::Character(v, _) => v.len(), RVal::Logical(v, _) => v.len(),
                RVal::List(v) => v.len(), RVal::Matrix(m) => m.data.len(),
                RVal::Null => 0, _ => 1,
            };
            Some(Ok(RVal::Integer(vec![Some(n as i32)].into(), Attrs::default())))
        }
        "sqrt" | "abs" | "exp" | "log" | "log2" | "log10" => {
            let v = coerce_reals(arg)?;
            let f: fn(f64) -> f64 = match name {
                "sqrt" => f64::sqrt, "abs" => f64::abs, "exp" => f64::exp,
                "log" => f64::ln, "log2" => f64::log2, "log10" => f64::log10,
                _ => unreachable!(),
            };
            Some(Ok(RVal::Numeric(v.iter().map(|x| x.map(f)).collect(), Attrs::default())))
        }
        _ => None,
    }
}

// ── Helper: run apply over `items` choosing fast/serial/closure path ─

fn map_items<C: EngineCtx + ?Sized>(
    ctx: &mut C, func: &RVal, items: &[RVal], env: &EnvRef,
) -> Result<Vec<RVal>, R2Err> {
    // Fast path: pure builtin, parallel via kernel::par_for.
    if let RVal::BuiltinFn(fname) = func {
        if !items.is_empty() && pure_apply(fname, &items[0]).is_some() {
            // Aggregate-work dispatch: when items vary in size (e.g. a
            // `list(big_vec, small_vec, big_vec)`), the parallel
            // crossover should be based on TOTAL work across components,
            // not on `items.len()` which would mis-classify a 3-component
            // list of 1M-element vectors as "too small for parallel".
            //
            // For all-scalar items (the common vector-mapped case),
            // total_work == items.len() so behavior is unchanged.
            let component_lens: Vec<usize> = items.iter().map(item_work_units).collect();
            let total_work: usize = component_lens.iter().sum();
            let is_heterogeneous = component_lens.iter().any(|&n| n > 1);
            let dispatch_op = if is_heterogeneous { r2_oracle::Op::ListMap } else { Op::PerElementMap };
            let dispatch_shape = r2_oracle::Shape::n(total_work);
            let fname_owned = fname.to_string();
            let items_owned = items.to_vec();
            // Use par_for's threshold logic via Oracle: ask whether the
            // aggregate work crosses the parallel threshold for the
            // appropriate op, then dispatch by index.
            let go_parallel = matches!(
                r2_oracle::dispatch(dispatch_op, dispatch_shape),
                r2_oracle::Backend::Rayon
            );
            let results: Vec<Result<RVal, R2Err>> = if go_parallel {
                // Use the bypass-Oracle Rayon helper since we've already
                // made the dispatch decision above with the right Op +
                // aggregate work.
                r2_kernel::par_for_rayon(items.len(), move |i| {
                    pure_apply(&fname_owned, &items_owned[i]).unwrap_or(Ok(RVal::Null))
                })
            } else {
                (0..items.len()).map(|i| {
                    pure_apply(&fname_owned, &items_owned[i]).unwrap_or(Ok(RVal::Null))
                }).collect()
            };
            return results.into_iter().collect();
        }
    }
    // Closure / non-pure builtin: sequential engine callback.
    // (Parallel closure dispatch would require Send+Sync on the engine
    // context, which we don't have. Pure builtins above are the parallel
    // path; closures stay sequential.)
    let mut r = Vec::with_capacity(items.len());
    for item in items {
        let call_args = vec![EvalArg { name: None, value: item.clone() }];
        r.push(ctx.ctx_call_fn(func, &call_args, env)?);
    }
    Ok(r)
}

/// Per-item work units for ListMap dispatch — captures the natural
/// "size" of each RVal variant so Oracle compares apples to apples.
fn item_work_units(v: &RVal) -> usize {
    match v {
        RVal::Numeric(r, _)   => r.len_fast(),
        RVal::Integer(r, _)   => r.len(),
        RVal::Logical(r, _)   => r.len(),
        RVal::Character(r, _) => r.len(),
        RVal::Raw(r, _)       => r.len(),
        RVal::List(items)     => items.len(),
        RVal::Matrix(m)       => m.nrow.saturating_mul(m.ncol),
        RVal::DataFrame(df)   => df.nrow().saturating_mul(df.ncol()),
        _ => 1,
    }
}

// ── lapply ──────────────────────────────────────────────────────────

pub fn bi_lapply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let func = nth_arg(a, 1);
    let items = x.to_items()?;
    let results = map_items(ctx, &func, &items, env)?;
    Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect()))
}

// ── sapply ──────────────────────────────────────────────────────────

pub fn bi_sapply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let func = nth_arg(a, 1);
    let items = x.to_items()?;
    let results = map_items(ctx, &func, &items, env)?;
    // Simplify to numeric vector if all results are scalar Numeric.
    let mut nums = Vec::new();
    let mut all_num = true;
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            _ => { all_num = false; break; }
        }
    }
    if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
    else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
}

// ── vapply (type-strict sapply) ─────────────────────────────────────

pub fn bi_vapply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let func = nth_arg(a, 1);
    // a[2] is FUN.VALUE — accepted but not enforced in current impl.
    let items = x.to_items()?;
    let results = map_items(ctx, &func, &items, env)?;
    let mut nums = Vec::with_capacity(results.len());
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            other => return Err(R2Err {
                msg: format!("vapply: FUN returned non-scalar of type '{}'", other.type_name()),
                kind: ErrKind::Type,
            }),
        }
    }
    Ok(RVal::Numeric(nums.into(), Attrs::default()))
}

// ── mapply (multivariate apply) ─────────────────────────────────────

pub fn bi_mapply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    if a.len() < 2 {
        return Err(R2Err { msg: "mapply: needs FUN + at least one input".into(), kind: ErrKind::Runtime });
    }
    let func = first_arg(a);
    let inputs: Vec<Vec<RVal>> = (1..a.len())
        .map(|i| nth_arg(a, i).to_items().unwrap_or_default())
        .collect();
    let max_len = inputs.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 { return Ok(RVal::List(vec![])); }

    // Single-input pure-builtin fast path
    if inputs.len() == 1 {
        if let RVal::BuiltinFn(fname) = &func {
            if pure_apply(fname, &inputs[0][0]).is_some() {
                let results = map_items(ctx, &func, &inputs[0], env)?;
                let mut nums = Vec::new(); let mut all_num = true;
                for r in &results {
                    match r {
                        RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
                        _ => { all_num = false; break; }
                    }
                }
                return if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
                else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) };
            }
        }
    }

    // General serial path: zip with R-style recycling.
    let mut results = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let call_args: Vec<EvalArg> = inputs.iter().map(|input| {
            let idx = if input.is_empty() { 0 } else { i % input.len() };
            EvalArg { name: None, value: input.get(idx).cloned().unwrap_or(RVal::Null) }
        }).collect();
        results.push(ctx.ctx_call_fn(&func, &call_args, env)?);
    }
    let mut nums = Vec::new(); let mut all_num = true;
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            _ => { all_num = false; break; }
        }
    }
    if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
    else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
}

// ── apply (matrix margin = 1 row / 2 col) ───────────────────────────

pub fn bi_apply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let arg0 = first_arg(a);
    let df = match arg0 {
        RVal::DataFrame(df) => df,
        // A matrix is handled as a data.frame of its columns, so the existing
        // per-row (margin=1) / per-column (margin=2) logic applies unchanged.
        RVal::Matrix(m) => {
            let columns: Vec<(Arc<str>, RVal)> = (0..m.ncol).map(|j| {
                let col: Vec<Real> = (0..m.nrow)
                    .map(|i| { let v = m.data[j * m.nrow + i]; if v.is_nan() { None } else { Some(v) } })
                    .collect();
                (Arc::from(format!("V{}", j + 1).as_str()), RVal::Numeric(col.into(), Attrs::default()))
            }).collect();
            DataFrame { columns, row_names: None }
        }
        _ => return Err(R2Err { msg: "apply needs data.frame or matrix".into(), kind: ErrKind::Type }),
    };
    let margin = nth_arg(a, 1).scalar_f64()?.unwrap_or(1.0) as i32;
    let func = nth_arg(a, 2);

    if margin == 2 {
        // Per-column inputs
        let inputs: Vec<RVal> = df.columns.iter().map(|(_, col)| col.clone()).collect();
        let results = map_items(ctx, &func, &inputs, env)?;
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r {
                RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
                _ => { all_scalar = false; break; }
            }
        }
        if all_scalar {
            let mut attrs = Attrs::default();
            attrs.names = Some(df.columns.iter().map(|(n, _)| n.clone()).collect());
            Ok(RVal::Numeric(nums.into(), attrs))
        } else {
            Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect()))
        }
    } else {
        // Per-row inputs
        let nrow = df.nrow();
        let rows: Vec<RVal> = (0..nrow).map(|r| {
            let row: Vec<Real> = df.columns.iter().filter_map(|(_, col)| {
                match col {
                    RVal::Numeric(v, _) => v.get(r).copied(),
                    RVal::Integer(v, _) => v.get(r).map(|x| x.map(|n| n as f64)),
                    _ => None,
                }
            }).collect();
            RVal::Numeric(row.into(), Attrs::default())
        }).collect();
        let results = map_items(ctx, &func, &rows, env)?;
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r { RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]), _ => { all_scalar = false; break; } }
        }
        if all_scalar { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
        else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
    }
}

// ── tapply (split-by-index then apply) ──────────────────────────────

fn to_string_vec(v: &RVal) -> Vec<String> {
    match v {
        RVal::Character(vs, _) => vs.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
        RVal::Numeric(vs, _) => vs.iter().map(|x| x.map(|n| format!("{}", n)).unwrap_or_default()).collect(),
        RVal::Integer(vs, _) => vs.iter().map(|x| x.map(|n| format!("{}", n)).unwrap_or_default()).collect(),
        // Factor index is the COMMON tapply/aggregate case (group by a factor
        // column); map codes → level labels. Without this, the `_` below
        // returned empty and `tapply(x, factor, FUN)` produced nothing.
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string())).unwrap_or_default()).collect(),
        RVal::Logical(vs, _) => vs.iter()
            .map(|x| x.map(|b| if b { "TRUE" } else { "FALSE" }.to_string()).unwrap_or_default()).collect(),
        _ => vec![],
    }
}

pub fn bi_tapply<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = nth_arg(a, 0).as_reals()?;
    let index = to_string_vec(&nth_arg(a, 1));
    let func = nth_arg(a, 2);

    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in index.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k, _)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();
    let computed = map_items(ctx, &func, &group_inputs, env)?;

    let results: Vec<(Option<Arc<str>>, RVal)> = groups.iter().zip(computed.into_iter())
        .map(|((key, _), result)| (Some(Arc::from(key.as_str())), result))
        .collect();
    Ok(RVal::List(results))
}

// ── aggregate ───────────────────────────────────────────────────────

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|ea| ea.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|ea| ea.value.clone())
}

pub fn bi_aggregate<C: EngineCtx + ?Sized>(ctx: &mut C, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = nth_arg(a, 0).as_reals()?;
    let by = to_string_vec(&arg_named(a, "by").unwrap_or_else(|| nth_arg(a, 1)));
    let func = arg_named(a, "FUN").unwrap_or_else(|| nth_arg(a, 2));

    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in by.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k, _)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();
    let computed = map_items(ctx, &func, &group_inputs, env)?;

    let mut group_names: Vec<Character> = Vec::with_capacity(groups.len());
    let mut agg_values: Vec<Real> = Vec::with_capacity(groups.len());
    for ((key, _), result) in groups.iter().zip(computed.into_iter()) {
        group_names.push(Some(Arc::from(key.as_str())));
        if let Ok(v) = result.scalar_f64() { agg_values.push(v); } else { agg_values.push(None); }
    }
    let _ = HashMap::<u8, u8>::new(); // suppress unused-import if HashMap not used
    Ok(RVal::DataFrame(DataFrame {
        columns: vec![
            (Arc::from("Group"), RVal::Character(group_names, Attrs::default())),
            (Arc::from("Value"), RVal::Numeric(agg_values.into(), Attrs::default())),
        ],
        row_names: None,
    }))
}

/// `do.call(func, list_of_args)` — Phase R.7.
/// Calls `func` with the entries of `list_of_args` as arguments.
pub fn bi_do_call<C: EngineCtx + ?Sized>(
    ctx: &mut C, a: &[EvalArg], env: &EnvRef,
) -> Result<RVal, R2Err> {
    let func = a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let arg_list = match a.get(1).map(|x| &x.value) {
        Some(RVal::List(items)) => items.clone(),
        _ => return Err(R2Err {
            msg: "do.call needs list of arguments".into(),
            kind: ErrKind::Type,
        }),
    };
    let ea: Vec<EvalArg> = arg_list.into_iter()
        .map(|(name, val)| EvalArg { name, value: val })
        .collect();
    ctx.ctx_call_fn(&func, &ea, env)
}
