//! AST-based reduction & vector-output kernels (Phase J.4 bricks 2-4): multi-reduction scalar kernels, vector-local fusion, CSE + wave fusion (scalar & SIMD), and vector-returning map kernels. Emits Cranelift directly from the R2 AST.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_types::BinOp;
use std::collections::HashMap;
use crate::*;

// ── Phase J.4 brick 2 — multi-reduction scalar kernels ───────────────
//
// Compiles a scalar-returning function whose body *combines several whole-
// vector reductions* — the shape the single-reduction paths can't express:
//   function(x, y) sum(x*y) / sum(x*x)                 (regression coefficient)
//   function(x)    { m <- mean(x); sum((x-m)*(x-m)) }  (centred sum / variance)
//   function(x, y) { mx<-mean(x); my<-mean(y); sum((x-mx)*(y-my)) } (covariance)
//
// Each `sum`/`prod`/`mean` emits its own fused loop (no intermediate vector is
// ever materialised); scalar locals thread between the loops as plain SSA
// values. Reuses the `Vector1/2ToScalar` ABI (`(ptr[,ptr],len)->f64`) and the
// existing engine dispatch — no new ABI surface.

struct Kctx<'a> {
    x_ptr: Value,
    y_ptr: Option<Value>,
    len: Value,   // i64 element count
    len_f: Value, // f64 element count (for mean / length)
    names: &'a [std::sync::Arc<str>],
    math: &'a HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef>,
}

/// Emit a pure element-wise expression (inside a reduction loop). `env` maps the
/// vector param names → the loaded element value, and any scalar locals → their
/// (loop-invariant) values. No reductions here — those live at the scalar level.
fn emit_elem(bcx: &mut FunctionBuilder, k: &Kctx, e: &r2_types::Expr, env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => env.get(s).copied()
            .ok_or_else(|| JitError::Unsupported(format!("element expr references non-vector `{}`", s))),
        Unary { op, expr } => {
            let v = emit_elem(bcx, k, expr, env)?;
            Ok(match op {
                r2_types::UnOp::Neg => bcx.ins().fneg(v),
                r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => { let z = bcx.ins().f64const(0.0); cmp_to_f64(bcx, v, z, FloatCC::Equal) }
            })
        }
        Binary { op, lhs, rhs } => {
            let a = emit_elem(bcx, k, lhs, env)?;
            let b = emit_elem(bcx, k, rhs, env)?;
            emit_binop(bcx, k, *op, a, b)
        }
        Call { func, args } => {
            let fname = match func.as_ref() { Symbol(s) => s.clone(), _ => return Err(JitError::Unsupported("call target".into())) };
            let mut av = Vec::with_capacity(args.len());
            for a in args { av.push(emit_elem(bcx, k, &a.value, env)?); }
            emit_math_call(bcx, k, fname.as_ref(), &av)
        }
        If { cond, then, else_ } => {
            let e2 = else_.as_ref().ok_or_else(|| JitError::Unsupported("if without else in element expr".into()))?;
            let c = emit_elem(bcx, k, cond, env)?;
            let t = emit_elem(bcx, k, then, env)?;
            let f = emit_elem(bcx, k, e2, env)?;
            let z = bcx.ins().f64const(0.0);
            let nz = bcx.ins().fcmp(FloatCC::NotEqual, c, z);
            Ok(bcx.ins().select(nz, t, f))
        }
        _ => Err(JitError::Unsupported("unsupported node in element expr".into())),
    }
}

fn emit_binop(bcx: &mut FunctionBuilder, k: &Kctx, op: BinOp, a: Value, b: Value) -> JitResult<Value> {
    Ok(match op {
        BinOp::Add => bcx.ins().fadd(a, b),
        BinOp::Sub => bcx.ins().fsub(a, b),
        BinOp::Mul => bcx.ins().fmul(a, b),
        BinOp::Div => bcx.ins().fdiv(a, b),
        BinOp::Lt => cmp_to_f64(bcx, a, b, FloatCC::LessThan),
        BinOp::Gt => cmp_to_f64(bcx, a, b, FloatCC::GreaterThan),
        BinOp::Le => cmp_to_f64(bcx, a, b, FloatCC::LessThanOrEqual),
        BinOp::Ge => cmp_to_f64(bcx, a, b, FloatCC::GreaterThanOrEqual),
        BinOp::Eq => cmp_to_f64(bcx, a, b, FloatCC::Equal),
        BinOp::Ne => cmp_to_f64(bcx, a, b, FloatCC::NotEqual),
        BinOp::Pow => {
            let f = k.math.get("^").ok_or_else(|| JitError::CraneliftError("pow extern missing".into()))?;
            let call = bcx.ins().call(*f, &[a, b]);
            bcx.inst_results(call)[0]
        }
        other => return Err(JitError::Unsupported(format!("binop {:?}", other))),
    })
}

fn emit_math_call(bcx: &mut FunctionBuilder, k: &Kctx, name: &str, av: &[Value]) -> JitResult<Value> {
    match (name, av.len()) {
        ("sqrt", 1) => return Ok(bcx.ins().sqrt(av[0])),
        ("abs", 1) => return Ok(bcx.ins().fabs(av[0])),
        ("floor", 1) => return Ok(bcx.ins().floor(av[0])),
        ("ceil", 1) => return Ok(bcx.ins().ceil(av[0])),
        ("trunc", 1) => return Ok(bcx.ins().trunc(av[0])),
        ("round", 1) => return Ok(bcx.ins().nearest(av[0])),
        ("min", 2) => return Ok(bcx.ins().fmin(av[0], av[1])),
        ("max", 2) => return Ok(bcx.ins().fmax(av[0], av[1])),
        _ => {}
    }
    let me = find_math_extern(name).ok_or_else(|| JitError::Unsupported(format!("call `{}` unsupported", name)))?;
    if me.arity != av.len() { return Err(JitError::Unsupported(format!("`{}` arity", name))); }
    let f = k.math.get(me.r_name).ok_or_else(|| JitError::CraneliftError(format!("extern `{}` missing", name)))?;
    let call = bcx.ins().call(*f, av);
    Ok(bcx.inst_results(call)[0])
}

/// Emit a fused reduction loop: `acc = identity; for i in 0..len { acc = acc
/// <op> elem(i) }`. `scalar_env` provides loop-invariant locals visible in the
/// element expression. Returns the accumulator value.
fn emit_reduction(bcx: &mut FunctionBuilder, k: &Kctx, elem: &r2_types::Expr, op: FusedReduceOp, scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    let header = bcx.create_block();
    let body = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    bcx.append_block_param(header, types::F64);
    bcx.append_block_param(body, types::I64);
    bcx.append_block_param(body, types::F64);
    bcx.append_block_param(exit, types::F64);

    let identity = match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 };
    let zero_i = bcx.ins().iconst(types::I64, 0);
    let id_v = bcx.ins().f64const(identity);
    bcx.ins().jump(header, &[zero_i, id_v]);

    bcx.switch_to_block(header);
    let i_h = bcx.block_params(header)[0];
    let acc_h = bcx.block_params(header)[1];
    let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, k.len);
    bcx.ins().brif(lt, body, &[i_h, acc_h], exit, &[acc_h]);

    bcx.switch_to_block(body);
    let i_b = bcx.block_params(body)[0];
    let acc_b = bcx.block_params(body)[1];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i_b, eight);
    let mf = MemFlags::trusted();
    let mut env = scalar_env.clone();
    let xa = bcx.ins().iadd(k.x_ptr, off);
    let xv = bcx.ins().load(types::F64, mf, xa, 0);
    env.insert(k.names[0].clone(), xv);
    if let Some(yp) = k.y_ptr {
        let ya = bcx.ins().iadd(yp, off);
        let yv = bcx.ins().load(types::F64, mf, ya, 0);
        if k.names.len() > 1 { env.insert(k.names[1].clone(), yv); }
    }
    let e = emit_elem(bcx, k, elem, &env)?;
    let new_acc = match op { FusedReduceOp::Sum => bcx.ins().fadd(acc_b, e), FusedReduceOp::Prod => bcx.ins().fmul(acc_b, e) };
    let one = bcx.ins().iconst(types::I64, 1);
    let i_next = bcx.ins().iadd(i_b, one);
    bcx.ins().jump(header, &[i_next, new_acc]);

    bcx.switch_to_block(exit);
    Ok(bcx.block_params(exit)[0])
}

/// Is `e` computable with F64X2 SIMD (arithmetic + native-instruction math
/// only — no extern transcendentals, comparisons, or branches)?
fn elem_simd_ok(e: &r2_types::Expr) -> bool {
    use r2_types::Expr::*;
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | Symbol(_) => true,
        Unary { op, expr } => matches!(op, r2_types::UnOp::Neg | r2_types::UnOp::Pos) && elem_simd_ok(expr),
        Binary { op, lhs, rhs } => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
            && elem_simd_ok(lhs) && elem_simd_ok(rhs),
        Call { func, args } => matches!(func.as_ref(), Symbol(s)
            if matches!(s.as_ref(), "sqrt" | "abs" | "floor" | "ceil" | "trunc" | "round" | "min" | "max"))
            && args.iter().all(|a| elem_simd_ok(&a.value)),
        _ => false,
    }
}

/// Emit an element expression as an `F64X2` SIMD value. `env` maps each vector
/// param name → the 2-lane loaded value and each scalar local → its splatted
/// value. Pre-gated by `elem_simd_ok`, so every node here is representable.
fn emit_elem_simd(bcx: &mut FunctionBuilder, e: &r2_types::Expr, env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    let splat = |bcx: &mut FunctionBuilder, s: Value| bcx.ins().splat(types::F64X2, s);
    match e {
        NumLit(x) => { let s = bcx.ins().f64const(*x); Ok(splat(bcx, s)) }
        IntLit(x) => { let s = bcx.ins().f64const(*x as f64); Ok(splat(bcx, s)) }
        BoolLit(b) => { let s = bcx.ins().f64const(if *b { 1.0 } else { 0.0 }); Ok(splat(bcx, s)) }
        Symbol(s) => env.get(s).copied().ok_or_else(|| JitError::Unsupported(format!("simd elem `{}`", s))),
        Unary { op, expr } => {
            let v = emit_elem_simd(bcx, expr, env)?;
            Ok(match op { r2_types::UnOp::Neg => bcx.ins().fneg(v), _ => v })
        }
        Binary { op, lhs, rhs } => {
            let a = emit_elem_simd(bcx, lhs, env)?;
            let b = emit_elem_simd(bcx, rhs, env)?;
            Ok(match op { BinOp::Add => bcx.ins().fadd(a, b), BinOp::Sub => bcx.ins().fsub(a, b),
                BinOp::Mul => bcx.ins().fmul(a, b), BinOp::Div => bcx.ins().fdiv(a, b),
                _ => return Err(JitError::Unsupported("simd binop".into())) })
        }
        Call { func, args } => {
            let name = match func.as_ref() { Symbol(s) => s.clone(), _ => return Err(JitError::Unsupported("simd call".into())) };
            let mut av = Vec::with_capacity(args.len());
            for a in args { av.push(emit_elem_simd(bcx, &a.value, env)?); }
            Ok(match (name.as_ref(), av.len()) {
                ("sqrt", 1) => bcx.ins().sqrt(av[0]), ("abs", 1) => bcx.ins().fabs(av[0]),
                ("floor", 1) => bcx.ins().floor(av[0]), ("ceil", 1) => bcx.ins().ceil(av[0]),
                ("trunc", 1) => bcx.ins().trunc(av[0]), ("round", 1) => bcx.ins().nearest(av[0]),
                ("min", 2) => bcx.ins().fmin(av[0], av[1]), ("max", 2) => bcx.ins().fmax(av[0], av[1]),
                _ => return Err(JitError::Unsupported("simd math".into())),
            })
        }
        _ => Err(JitError::Unsupported("simd elem node".into())),
    }
}

/// SIMD (F64X2) version of `emit_reduction_wave`: a 2-elements-per-iteration
/// main loop with vector accumulators, a horizontal reduce, then a scalar tail
/// for the odd element. Returns `None` if any element expression isn't SIMD-able
/// (caller falls back to the scalar wave) — nothing is emitted in that case.
fn emit_reduction_wave_simd(bcx: &mut FunctionBuilder, k: &Kctx, reds: &[(&r2_types::Expr, FusedReduceOp)], scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Option<Vec<Value>>> {
    if !reds.iter().all(|(e, _)| elem_simd_ok(e)) { return Ok(None); }
    let n = reds.len();

    let simd_hdr = bcx.create_block();
    let simd_body = bcx.create_block();
    let simd_exit = bcx.create_block();
    let rem_hdr = bcx.create_block();
    let rem_body = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(simd_hdr, types::I64);
    for _ in 0..n { bcx.append_block_param(simd_hdr, types::F64X2); }
    bcx.append_block_param(simd_exit, types::I64);
    for _ in 0..n { bcx.append_block_param(simd_exit, types::F64X2); }
    bcx.append_block_param(rem_hdr, types::I64);
    for _ in 0..n { bcx.append_block_param(rem_hdr, types::F64); }
    for _ in 0..n { bcx.append_block_param(exit, types::F64); }

    // simd_end = len rounded down to even.
    let simd_end = bcx.ins().band_imm(k.len, -2);
    let mut init = Vec::with_capacity(n + 1);
    init.push(bcx.ins().iconst(types::I64, 0));
    for (_, op) in reds {
        let s = bcx.ins().f64const(match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 });
        init.push(bcx.ins().splat(types::F64X2, s));
    }
    bcx.ins().jump(simd_hdr, &init);

    // SIMD header: while i < simd_end.
    bcx.switch_to_block(simd_hdr);
    let hp: Vec<Value> = bcx.block_params(simd_hdr).to_vec();
    let cond = bcx.ins().icmp(IntCC::SignedLessThan, hp[0], simd_end);
    // simd_body has no params (it reads the header's param Values directly via
    // dominance); simd_exit takes (i, accs...).
    bcx.ins().brif(cond, simd_body, &[], simd_exit, &hp);

    // SIMD body: load 2 lanes, accumulate.
    bcx.switch_to_block(simd_body);
    let bp: Vec<Value> = bcx.block_params(simd_hdr).to_vec(); // same values (single pred)
    let i_b = bp[0];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i_b, eight);
    let mf = MemFlags::trusted();
    let mut env: HashMap<std::sync::Arc<str>, Value> = HashMap::new();
    for (nm, v) in scalar_env { let sp = bcx.ins().splat(types::F64X2, *v); env.insert(nm.clone(), sp); }
    let xa = bcx.ins().iadd(k.x_ptr, off);
    env.insert(k.names[0].clone(), bcx.ins().load(types::F64X2, mf, xa, 0));
    if let Some(yp) = k.y_ptr {
        let ya = bcx.ins().iadd(yp, off);
        let yv = bcx.ins().load(types::F64X2, mf, ya, 0);
        if k.names.len() > 1 { env.insert(k.names[1].clone(), yv); }
    }
    let mut nxt = Vec::with_capacity(n + 1);
    let two = bcx.ins().iconst(types::I64, 2);
    nxt.push(bcx.ins().iadd(i_b, two));
    for (j, (elem, op)) in reds.iter().enumerate() {
        let e = emit_elem_simd(bcx, elem, &env)?;
        let acc = bp[1 + j];
        nxt.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(acc, e), FusedReduceOp::Prod => bcx.ins().fmul(acc, e) });
    }
    bcx.ins().jump(simd_hdr, &nxt);

    // SIMD exit: horizontal-reduce each vector accumulator to a scalar.
    bcx.switch_to_block(simd_exit);
    let ep: Vec<Value> = bcx.block_params(simd_exit).to_vec();
    let i_rem = ep[0];
    let mut scalars = Vec::with_capacity(n + 1);
    scalars.push(i_rem);
    for (j, (_, op)) in reds.iter().enumerate() {
        let v = ep[1 + j];
        let l0 = bcx.ins().extractlane(v, 0);
        let l1 = bcx.ins().extractlane(v, 1);
        scalars.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(l0, l1), FusedReduceOp::Prod => bcx.ins().fmul(l0, l1) });
    }
    bcx.ins().jump(rem_hdr, &scalars);

    // Remainder header: while i < len (0 or 1 iterations).
    bcx.switch_to_block(rem_hdr);
    let rp: Vec<Value> = bcx.block_params(rem_hdr).to_vec();
    let rcond = bcx.ins().icmp(IntCC::SignedLessThan, rp[0], k.len);
    // rem_body has no params (reads rem_hdr's directly); exit takes (accs...).
    bcx.ins().brif(rcond, rem_body, &[], exit, &rp[1..]);

    // Remainder body: scalar accumulate one element.
    bcx.switch_to_block(rem_body);
    let rbp: Vec<Value> = bcx.block_params(rem_hdr).to_vec();
    let i_r = rbp[0];
    let eight_r = bcx.ins().iconst(types::I64, 8); // fresh: simd_body's `eight` doesn't dominate here
    let off_r = bcx.ins().imul(i_r, eight_r);
    let mut env_r: HashMap<std::sync::Arc<str>, Value> = scalar_env.clone();
    let xar = bcx.ins().iadd(k.x_ptr, off_r);
    env_r.insert(k.names[0].clone(), bcx.ins().load(types::F64, mf, xar, 0));
    if let Some(yp) = k.y_ptr {
        let yar = bcx.ins().iadd(yp, off_r);
        let yvr = bcx.ins().load(types::F64, mf, yar, 0);
        if k.names.len() > 1 { env_r.insert(k.names[1].clone(), yvr); }
    }
    let mut nxt_r = Vec::with_capacity(n + 1);
    let one = bcx.ins().iconst(types::I64, 1);
    nxt_r.push(bcx.ins().iadd(i_r, one));
    for (j, (elem, op)) in reds.iter().enumerate() {
        let e = emit_elem(bcx, k, elem, &env_r)?;
        let acc = rbp[1 + j];
        nxt_r.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(acc, e), FusedReduceOp::Prod => bcx.ins().fmul(acc, e) });
    }
    bcx.ins().jump(rem_hdr, &nxt_r);

    bcx.switch_to_block(exit);
    Ok(Some(bcx.block_params(exit).to_vec()))
}

/// Phase J.4 brick 4 — emit a *single* loop that computes several reductions at
/// once (one accumulator each), loading each vector element just once. The
/// reductions' element expressions typically share sub-terms (e.g. `x[i]-mx`);
/// Cranelift's value numbering folds those to one computation. Matches what a
/// hand-written native `cor`/`cov` does: one pass for `sxy`,`sxx`,`syy`.
fn emit_reduction_wave(bcx: &mut FunctionBuilder, k: &Kctx, reds: &[(&r2_types::Expr, FusedReduceOp)], scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Vec<Value>> {
    let n = reds.len();
    let header = bcx.create_block();
    let body = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    for _ in 0..n { bcx.append_block_param(header, types::F64); }
    bcx.append_block_param(body, types::I64);
    for _ in 0..n { bcx.append_block_param(body, types::F64); }
    for _ in 0..n { bcx.append_block_param(exit, types::F64); }

    let mut init_args = Vec::with_capacity(n + 1);
    init_args.push(bcx.ins().iconst(types::I64, 0));
    for (_, op) in reds { init_args.push(bcx.ins().f64const(match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 })); }
    bcx.ins().jump(header, &init_args);

    bcx.switch_to_block(header);
    let hp: Vec<Value> = bcx.block_params(header).to_vec();
    let i_h = hp[0];
    let lt = bcx.ins().icmp(IntCC::SignedLessThan, i_h, k.len);
    let then_args: Vec<Value> = hp.clone();          // (i, accs...)
    let else_args: Vec<Value> = hp[1..].to_vec();    // (accs...)
    bcx.ins().brif(lt, body, &then_args, exit, &else_args);

    bcx.switch_to_block(body);
    let bp: Vec<Value> = bcx.block_params(body).to_vec();
    let i_b = bp[0];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i_b, eight);
    let mf = MemFlags::trusted();
    let mut env = scalar_env.clone();
    let xa = bcx.ins().iadd(k.x_ptr, off);
    let xv = bcx.ins().load(types::F64, mf, xa, 0);
    env.insert(k.names[0].clone(), xv);
    if let Some(yp) = k.y_ptr {
        let ya = bcx.ins().iadd(yp, off);
        let yv = bcx.ins().load(types::F64, mf, ya, 0);
        if k.names.len() > 1 { env.insert(k.names[1].clone(), yv); }
    }
    let mut next = Vec::with_capacity(n + 1);
    let one = bcx.ins().iconst(types::I64, 1);
    next.push(bcx.ins().iadd(i_b, one));
    for (j, (elem, op)) in reds.iter().enumerate() {
        let e = emit_elem(bcx, k, elem, &env)?;
        let acc = bp[1 + j];
        next.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(acc, e), FusedReduceOp::Prod => bcx.ins().fmul(acc, e) });
    }
    bcx.ins().jump(header, &next);

    bcx.switch_to_block(exit);
    Ok(bcx.block_params(exit).to_vec())
}

/// Emit a scalar expression whose leaves are reductions, scalar locals,
/// literals, and scalar math — combined by arithmetic.
fn emit_scalar(bcx: &mut FunctionBuilder, k: &Kctx, e: &r2_types::Expr, scalar_env: &HashMap<std::sync::Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    let is_vec = |s: &str| k.names.iter().any(|n| n.as_ref() == s);
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => {
            if is_vec(s) { return Err(JitError::Unsupported(format!("bare vector `{}` used as scalar", s))); }
            scalar_env.get(s).copied().ok_or_else(|| JitError::Unsupported(format!("unknown scalar `{}`", s)))
        }
        Unary { op, expr } => {
            let v = emit_scalar(bcx, k, expr, scalar_env)?;
            Ok(match op { r2_types::UnOp::Neg => bcx.ins().fneg(v), r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => { let z = bcx.ins().f64const(0.0); cmp_to_f64(bcx, v, z, FloatCC::Equal) } })
        }
        Binary { op, lhs, rhs } => {
            let a = emit_scalar(bcx, k, lhs, scalar_env)?;
            let b = emit_scalar(bcx, k, rhs, scalar_env)?;
            emit_binop(bcx, k, *op, a, b)
        }
        Call { func, args } => {
            let fname = match func.as_ref() { Symbol(s) => s.clone(), _ => return Err(JitError::Unsupported("call target".into())) };
            match (fname.as_ref(), args.len()) {
                ("sum", 1) => emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Sum, scalar_env),
                ("prod", 1) => emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Prod, scalar_env),
                ("mean", 1) => {
                    let s = emit_reduction(bcx, k, &args[0].value, FusedReduceOp::Sum, scalar_env)?;
                    Ok(bcx.ins().fdiv(s, k.len_f))
                }
                ("length", 1) => match &args[0].value {
                    Symbol(v) if is_vec(v) => Ok(k.len_f),
                    _ => Err(JitError::Unsupported("length of non-vector".into())),
                },
                _ => {
                    let mut av = Vec::with_capacity(args.len());
                    for a in args { av.push(emit_scalar(bcx, k, &a.value, scalar_env)?); }
                    emit_math_call(bcx, k, fname.as_ref(), &av)
                }
            }
        }
        _ => Err(JitError::Unsupported("unsupported node in scalar body".into())),
    }
}

/// Does `e` reference any symbol in `names`? Used to decide whether a reduction
/// can join the current fused wave (it cannot if its element expr depends on a
/// scalar local being produced by another reduction in the same wave).
fn expr_refs_any(e: &r2_types::Expr, names: &std::collections::HashSet<std::sync::Arc<str>>) -> bool {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => names.contains(s),
        Unary { expr, .. } => expr_refs_any(expr, names),
        Binary { lhs, rhs, .. } => expr_refs_any(lhs, names) || expr_refs_any(rhs, names),
        Call { func, args } => expr_refs_any(func, names) || args.iter().any(|a| expr_refs_any(&a.value, names)),
        If { cond, then, else_ } => expr_refs_any(cond, names) || expr_refs_any(then, names)
            || else_.as_ref().map_or(false, |x| expr_refs_any(x, names)),
        _ => false,
    }
}

/// Emit the pending reduction batch as one fused loop and bind each result
/// (dividing by `len` for `mean`) into `scalar_env`; then clear the batch.
fn flush_wave(
    bcx: &mut FunctionBuilder, k: &Kctx,
    batch: &mut Vec<(std::sync::Arc<str>, &r2_types::Expr, FusedReduceOp, bool)>,
    batch_names: &mut std::collections::HashSet<std::sync::Arc<str>>,
    scalar_env: &mut HashMap<std::sync::Arc<str>, Value>,
) -> JitResult<()> {
    if batch.is_empty() { return Ok(()); }
    let reds: Vec<(&r2_types::Expr, FusedReduceOp)> = batch.iter().map(|(_, e, op, _)| (*e, *op)).collect();
    // Prefer the F64X2 SIMD wave; fall back to the scalar wave if any element
    // expression isn't SIMD-representable (transcendentals, branches, …).
    let vals = match emit_reduction_wave_simd(bcx, k, &reds, scalar_env)? {
        Some(v) => v,
        None => emit_reduction_wave(bcx, k, &reds, scalar_env)?,
    };
    for ((name, _, _, is_mean), v) in batch.iter().zip(vals.into_iter()) {
        let bound = if *is_mean { bcx.ins().fdiv(v, k.len_f) } else { v };
        scalar_env.insert(name.clone(), bound);
    }
    batch.clear();
    batch_names.clear();
    Ok(())
}

/// Process a kernel body's leading `local <- <scalar>` assignments into
/// `scalar_env`, batching independent reductions into fused waves. Shared by the
/// scalar-return and vector-return kernels.
/// First symbol in `e` that is neither defined, a vector param, nor the loop
/// var — used to guarantee a loop-carried scalar's 0.0-init is never observed
/// (read-before-define inside a compiled loop must reject → interpreter, which
/// reports R's "object not found" instead of silently computing with 0).
fn first_undefined_symbol<'e>(
    e: &'e r2_types::Expr,
    defined: &std::collections::HashSet<String>,
    k: &Kctx,
    var: &str,
) -> Option<&'e str> {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => {
            let name = s.as_ref();
            if defined.contains(name) || var == name
                || k.names.iter().any(|n| n.as_ref() == name) { None } else { Some(name) }
        }
        Unary { expr, .. } => first_undefined_symbol(expr, defined, k, var),
        Binary { lhs, rhs, .. } => first_undefined_symbol(lhs, defined, k, var)
            .or_else(|| first_undefined_symbol(rhs, defined, k, var)),
        // Function-position symbols (sum/mean/sqrt/…) are not variables.
        Call { args, .. } => args.iter().find_map(|a| first_undefined_symbol(&a.value, defined, k, var)),
        If { cond, then, else_ } => first_undefined_symbol(cond, defined, k, var)
            .or_else(|| first_undefined_symbol(then, defined, k, var))
            .or_else(|| else_.as_ref().and_then(|x| first_undefined_symbol(x, defined, k, var))),
        _ => None,
    }
}

/// Phase J.4 (whole-function slice 1) — compile a counted loop whose body is a
/// sequence of scalar assignments that may embed whole-vector reductions:
///
///   for (it in 1:K) { g <- mean(x - b); b <- b + 0.2*g }
///
/// The iterative-algorithm shape (gradient descent, Newton, fixed-point, EM
/// updates). Loop-carried scalars ride as header block params (SSA phis); each
/// iteration re-runs the body's fused reduction waves against the carried
/// values. `1:K` matches R in BOTH directions (step ±1, so `1:0` iterates 1,0).
fn emit_scalar_loop(
    bcx: &mut FunctionBuilder,
    k: &Kctx,
    var: &std::sync::Arc<str>,
    iter: &r2_types::Expr,
    body: &r2_types::Expr,
    scalar_env: &mut HashMap<std::sync::Arc<str>, Value>,
) -> JitResult<()> {
    use r2_types::Expr::*;
    let hi_expr = match iter {
        Binary { op: BinOp::Colon, lhs, rhs }
            if matches!(lhs.as_ref(), NumLit(x) if *x == 1.0)
                || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
        _ => return Err(JitError::Unsupported("iterative kernel: loop must be `for (v in 1:K)`".into())),
    };
    let body_stmts: Vec<r2_types::Expr> = match body {
        Block(b) => b.clone(),
        other => vec![other.clone()],
    };

    // Carried scalars = every Assign target in the body, first-assign order.
    let mut carried: Vec<std::sync::Arc<str>> = Vec::new();
    for bs in &body_stmts {
        match bs {
            Assign { target, .. } => match target.as_ref() {
                Symbol(n) => if !carried.iter().any(|c| c.as_ref() == n.as_ref()) { carried.push(n.clone()); },
                _ => return Err(JitError::Unsupported("iterative kernel: non-symbol assign in loop body".into())),
            },
            _ => return Err(JitError::Unsupported("iterative kernel: loop body must be assignments".into())),
        }
    }
    if carried.is_empty() { return Err(JitError::Unsupported("iterative kernel: empty loop body".into())); }

    // Safety: no name may be read before it is defined (pre-loop or earlier in
    // the body) — otherwise the 0.0 phi-init would be observable where R errors.
    {
        let mut defined: std::collections::HashSet<String> =
            scalar_env.keys().map(|s| s.to_string()).collect();
        for bs in &body_stmts {
            if let Assign { target, value, .. } = bs {
                if let Some(bad) = first_undefined_symbol(value, &defined, k, var.as_ref()) {
                    return Err(JitError::Unsupported(format!("iterative kernel: `{}` read before defined", bad)));
                }
                if let Symbol(n) = target.as_ref() { defined.insert(n.to_string()); }
            }
        }
    }

    // Loop bound (loop-invariant, evaluated once) + R-faithful ±1 step.
    let hi_f = emit_scalar(bcx, k, hi_expr, scalar_env)?;
    let hi_i = bcx.ins().fcvt_to_sint(types::I64, hi_f);
    let one_i = bcx.ins().iconst(types::I64, 1);
    let neg1 = bcx.ins().iconst(types::I64, -1);
    let asc = bcx.ins().icmp(IntCC::SignedGreaterThanOrEqual, hi_i, one_i);
    let step = bcx.ins().select(asc, one_i, neg1);

    let header = bcx.create_block();
    let bodyb = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    for _ in &carried { bcx.append_block_param(header, types::F64); }
    for _ in &carried { bcx.append_block_param(exit, types::F64); }

    let mut init_args: Vec<Value> = vec![one_i];
    let zero = bcx.ins().f64const(0.0);
    for c in &carried {
        init_args.push(scalar_env.get(c).copied().unwrap_or(zero));
    }
    bcx.ins().jump(header, &init_args);

    // Header: continue while (i - hi) * step <= 0 (runs 1..=hi both directions).
    bcx.switch_to_block(header);
    let hp: Vec<Value> = bcx.block_params(header).to_vec();
    let i_h = hp[0];
    let diff = bcx.ins().isub(i_h, hi_i);
    let prod = bcx.ins().imul(diff, step);
    let cond = bcx.ins().icmp_imm(IntCC::SignedLessThan, prod, 1);
    bcx.ins().brif(cond, bodyb, &[], exit, &hp[1..]);

    // Body: bind carried + induction var, run the statements (waves + scalars),
    // loop back with the updated carried values.
    bcx.switch_to_block(bodyb);
    let mut env2 = scalar_env.clone();
    for (j, c) in carried.iter().enumerate() { env2.insert(c.clone(), hp[1 + j]); }
    let i_f = bcx.ins().fcvt_from_sint(types::F64, i_h);
    env2.insert(var.clone(), i_f);
    emit_kernel_prologue(bcx, k, &body_stmts, &mut env2)?;
    let mut next: Vec<Value> = vec![bcx.ins().iadd(i_h, step)];
    for c in &carried {
        next.push(*env2.get(c).ok_or_else(|| JitError::Unsupported("iterative kernel: carried scalar lost".into()))?);
    }
    bcx.ins().jump(header, &next);

    // Exit: carried values become the post-loop scalars; `var` ends at hi (R).
    bcx.switch_to_block(exit);
    let xp: Vec<Value> = bcx.block_params(exit).to_vec();
    for (j, c) in carried.iter().enumerate() { scalar_env.insert(c.clone(), xp[j]); }
    scalar_env.insert(var.clone(), hi_f);
    Ok(())
}

fn emit_kernel_prologue(bcx: &mut FunctionBuilder, k: &Kctx, init: &[r2_types::Expr], scalar_env: &mut HashMap<std::sync::Arc<str>, Value>) -> JitResult<()> {
    let mut batch: Vec<(std::sync::Arc<str>, &r2_types::Expr, FusedReduceOp, bool)> = Vec::new();
    let mut batch_names: std::collections::HashSet<std::sync::Arc<str>> = std::collections::HashSet::new();
    for st in init {
        // J.4 iterative kernels — a counted `for` carrying scalar state across
        // per-iteration reduction waves (gradient descent / Newton / EM shape).
        if let r2_types::Expr::For { var, iter, body } = st {
            flush_wave(bcx, k, &mut batch, &mut batch_names, scalar_env)?;
            emit_scalar_loop(bcx, k, var, iter, body, scalar_env)?;
            continue;
        }
        let (nm, value) = match st {
            r2_types::Expr::Assign { target, value, .. } => match target.as_ref() {
                r2_types::Expr::Symbol(n) => (n.clone(), value.as_ref()),
                _ => return Err(JitError::Unsupported("non-symbol assign in kernel".into())),
            },
            _ => return Err(JitError::Unsupported("non-assign statement before final expr".into())),
        };
        let red: Option<(&r2_types::Expr, FusedReduceOp, bool)> = match value {
            r2_types::Expr::Call { func, args } if args.len() == 1 => match func.as_ref() {
                r2_types::Expr::Symbol(s) if s.as_ref() == "sum" => Some((&args[0].value, FusedReduceOp::Sum, false)),
                r2_types::Expr::Symbol(s) if s.as_ref() == "prod" => Some((&args[0].value, FusedReduceOp::Prod, false)),
                r2_types::Expr::Symbol(s) if s.as_ref() == "mean" => Some((&args[0].value, FusedReduceOp::Sum, true)),
                _ => None,
            },
            _ => None,
        };
        match red {
            Some((elem, op, is_mean)) => {
                if expr_refs_any(elem, &batch_names) { flush_wave(bcx, k, &mut batch, &mut batch_names, scalar_env)?; }
                batch_names.insert(nm.clone());
                batch.push((nm, elem, op, is_mean));
            }
            None => {
                flush_wave(bcx, k, &mut batch, &mut batch_names, scalar_env)?;
                let v = emit_scalar(bcx, k, value, scalar_env)?;
                scalar_env.insert(nm, v);
            }
        }
    }
    flush_wave(bcx, k, &mut batch, &mut batch_names, scalar_env)
}

/// Phase J.4 (matrix/vector-lowering step 1) — compile a **vector-returning**
/// kernel `function(x[,y]) <element-expr>` where the element expr may embed
/// whole-vector reductions of the inputs. Reductions are computed first (fused
/// waves → scalar locals), then a single map pass writes the element expression
/// to the output buffer. This is the centring/standardise/normalise class:
///   d(x) = x - mean(x);  (x-mean(x))/sd;  x/sum(x)
/// The `pow`-free element map is SIMD-vectorised (F64X2 + scalar tail); the ABI
/// is `VectorMap` / `VectorBinaryMap` (`(in.., out, len)`), so the existing
/// engine dispatch applies unchanged.
pub(crate) fn compile_reduction_map_kernel(body: &r2_types::Expr, param_names: &[std::sync::Arc<str>]) -> JitResult<CompiledFn> {
    let n_vec = param_names.len();
    if !(1..=2).contains(&n_vec) { return Err(JitError::Unsupported("reduction-map needs 1-2 params".into())); }

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    // Signature: (in_ptr.. , out_ptr, len), all i64. No return.
    let mut sig = module.make_signature();
    for _ in 0..n_vec { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64)); // out_ptr
    sig.params.push(AbiParam::new(types::I64)); // len

    let func_id = module.declare_function("__jit_reduction_map", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let math: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter().map(|(kk, id)| (*kk, module.declare_func_in_func(*id, &mut bcx.func))).collect();

        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let ps: Vec<Value> = bcx.block_params(entry).to_vec();
        let x_ptr = ps[0];
        let y_ptr = if n_vec == 2 { Some(ps[1]) } else { None };
        let out_ptr = ps[n_vec];
        let len = ps[n_vec + 1];
        let len_f = bcx.ins().fcvt_from_sint(types::F64, len);
        let k = Kctx { x_ptr, y_ptr, len, len_f, names: param_names, math: &math };

        let (last, init): (&r2_types::Expr, &[r2_types::Expr]) = match body {
            r2_types::Expr::Block(stmts) if !stmts.is_empty() => { let (l, i) = stmts.split_last().unwrap(); (l, i) }
            other => (other, &[]),
        };
        let mut scalar_env: HashMap<std::sync::Arc<str>, Value> = HashMap::new();
        emit_kernel_prologue(&mut bcx, &k, init, &mut scalar_env)?;

        // Map pass: out[i] = last(x[i][,y[i]], scalar_locals). SIMD 2-wide + tail.
        let simd_ok = elem_simd_ok(last);
        let simd_end = bcx.ins().band_imm(len, -2);
        let eight = bcx.ins().iconst(types::I64, 8);
        let mf = MemFlags::trusted();

        let shdr = bcx.create_block(); let sbody = bcx.create_block();
        let rhdr = bcx.create_block(); let rbody = bcx.create_block();
        let exit = bcx.create_block();
        bcx.append_block_param(shdr, types::I64);
        bcx.append_block_param(rhdr, types::I64);
        let zero = bcx.ins().iconst(types::I64, 0);
        // SIMD loop only if the element expr is SIMD-clean; else start the scalar loop at 0.
        if simd_ok { bcx.ins().jump(shdr, &[zero]); } else { bcx.ins().jump(rhdr, &[zero]); }

        // SIMD header/body.
        bcx.switch_to_block(shdr);
        let i_sh = bcx.block_params(shdr)[0];
        let scond = bcx.ins().icmp(IntCC::SignedLessThan, i_sh, simd_end);
        bcx.ins().brif(scond, sbody, &[], rhdr, &[i_sh]);
        bcx.switch_to_block(sbody);
        let i_sb = bcx.block_params(shdr)[0];
        let off_s = bcx.ins().imul(i_sb, eight);
        let mut envs: HashMap<std::sync::Arc<str>, Value> = HashMap::new();
        for (nm, v) in &scalar_env { let sp = bcx.ins().splat(types::F64X2, *v); envs.insert(nm.clone(), sp); }
        let xas = bcx.ins().iadd(x_ptr, off_s);
        envs.insert(param_names[0].clone(), bcx.ins().load(types::F64X2, mf, xas, 0));
        if let Some(yp) = y_ptr { let yas = bcx.ins().iadd(yp, off_s); let yv = bcx.ins().load(types::F64X2, mf, yas, 0); if n_vec > 1 { envs.insert(param_names[1].clone(), yv); } }
        let rv_s = emit_elem_simd(&mut bcx, last, &envs)?;
        let oas = bcx.ins().iadd(out_ptr, off_s);
        bcx.ins().store(mf, rv_s, oas, 0);
        let two = bcx.ins().iconst(types::I64, 2);
        let ins = bcx.ins().iadd(i_sb, two);
        bcx.ins().jump(shdr, &[ins]);

        // Scalar remainder header/body.
        bcx.switch_to_block(rhdr);
        let i_rh = bcx.block_params(rhdr)[0];
        let rcond = bcx.ins().icmp(IntCC::SignedLessThan, i_rh, len);
        bcx.ins().brif(rcond, rbody, &[], exit, &[]);
        bcx.switch_to_block(rbody);
        let i_rb = bcx.block_params(rhdr)[0];
        let eight_r = bcx.ins().iconst(types::I64, 8);
        let off_r = bcx.ins().imul(i_rb, eight_r);
        let mut envr = scalar_env.clone();
        let xar = bcx.ins().iadd(x_ptr, off_r);
        envr.insert(param_names[0].clone(), bcx.ins().load(types::F64, mf, xar, 0));
        if let Some(yp) = y_ptr { let yar = bcx.ins().iadd(yp, off_r); let yv = bcx.ins().load(types::F64, mf, yar, 0); if n_vec > 1 { envr.insert(param_names[1].clone(), yv); } }
        let rv_r = emit_elem(&mut bcx, &k, last, &envr)?;
        let oar = bcx.ins().iadd(out_ptr, off_r);
        bcx.ins().store(mf, rv_r, oar, 0);
        let one = bcx.ins().iconst(types::I64, 1);
        let inr = bcx.ins().iadd(i_rb, one);
        bcx.ins().jump(rhdr, &[inr]);

        bcx.switch_to_block(exit);
        bcx.ins().return_(&[]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(func_id, &mut ctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    let kind = if n_vec == 1 { r2_types::JitKind::VectorMap } else { r2_types::JitKind::VectorBinaryMap };
    Ok(CompiledFn { ptr, arity: n_vec, kind, _module: module })
}

/// Compile a multi-reduction scalar kernel over 1–2 vector params. `body` is a
/// scalar expression, or a `Block` of `local <- <scalar>` assignments followed
/// by a final scalar expression.
pub(crate) fn compile_reduction_kernel(body: &r2_types::Expr, param_names: &[std::sync::Arc<str>]) -> JitResult<CompiledFn> {
    let n_vec = param_names.len();
    if !(1..=2).contains(&n_vec) { return Err(JitError::Unsupported("reduction kernel needs 1-2 params".into())); }

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let math_ids = declare_math_imports(&mut module)?;

    let mut sig = module.make_signature();
    for _ in 0..n_vec { sig.params.push(AbiParam::new(types::I64)); }
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::F64));

    let func_id = module.declare_function("__jit_reduction_kernel", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let math: HashMap<&'static str, cranelift::prelude::codegen::ir::FuncRef> =
            math_ids.iter().map(|(kk, id)| (*kk, module.declare_func_in_func(*id, &mut bcx.func))).collect();

        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let ps: Vec<Value> = bcx.block_params(entry).to_vec();
        let x_ptr = ps[0];
        let y_ptr = if n_vec == 2 { Some(ps[1]) } else { None };
        let len = ps[n_vec];
        let len_f = bcx.ins().fcvt_from_sint(types::F64, len);
        let k = Kctx { x_ptr, y_ptr, len, len_f, names: param_names, math: &math };

        let mut scalar_env: HashMap<std::sync::Arc<str>, Value> = HashMap::new();
        let result = match body {
            r2_types::Expr::Block(stmts) => {
                if stmts.is_empty() { return Err(JitError::Unsupported("empty body".into())); }
                let (last, init) = stmts.split_last().unwrap();
                emit_kernel_prologue(&mut bcx, &k, init, &mut scalar_env)?;
                emit_scalar(&mut bcx, &k, last, &scalar_env)?
            }
            other => emit_scalar(&mut bcx, &k, other, &scalar_env)?,
        };
        bcx.ins().return_(&[result]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(func_id, &mut ctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    let kind = if n_vec == 1 { r2_types::JitKind::Vector1ToScalar } else { r2_types::JitKind::Vector2ToScalar };
    Ok(CompiledFn { ptr, arity: n_vec, kind, _module: module })
}
