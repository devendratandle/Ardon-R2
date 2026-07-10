//! Phase J.4 (final piece) — MATRIX-STATE iterative kernels.
//!
//! Compiles a whole function of the multi-parameter linear-model shape:
//!
//!   function(X, y) {                       # X: n×p matrix, y: n-vector
//!     b <- rep(0, ncol(X))                 # p-vector state
//!     for (it in 1:K) {
//!       r <- y - X %*% b                   # n-vector
//!       g <- t(X) %*% r                    # p-vector
//!       b <- b + (s/nrow(X)) * g           # p-vector update
//!     }
//!     b                                    # p-vector out
//!   }
//!
//! into one native unit. `X %*% v` / `t(X) %*% v` call the SHARED matvec
//! externs (`r2_kern_matvec`/`r2_kern_tmatvec`) — matrix math is never
//! re-emitted as bespoke JIT code. Vector state lives in scratch buffers of
//! TWO length classes (N = nrow, P = ncol); element-wise updates and
//! reductions loop over the correct class. Anything outside the supported
//! statement set → `Err` → interpreter (never a guess).
//!
//! ABI (`JitKind::MatVecIterOut`): `(m_ptr, nrow, ncol, v_ptr, out_ptr)`,
//! out is a p-vector the engine allocates. Same fall-back-safety rules as the
//! scalar/vector kernels: read-before-define declines; `1:K` matches R in
//! both directions.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_types::BinOp;
use std::collections::HashMap;
use std::sync::Arc;
use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MClass { N, P }

struct MCtx {
    m_ptr: Value,
    n: Value,
    p: Value,
    n_f: Value,
    p_f: Value,
    y_ptr: Value,
    mat: Arc<str>,
    yname: Arc<str>,
    matvec_ref: cranelift::prelude::codegen::ir::FuncRef,
    tmatvec_ref: cranelift::prelude::codegen::ir::FuncRef,
    bufs: std::cell::RefCell<HashMap<Arc<str>, (Value, MClass)>>,
}

impl MCtx {
    fn len(&self, c: MClass) -> Value { match c { MClass::N => self.n, MClass::P => self.p } }
    fn len_f(&self, c: MClass) -> Value { match c { MClass::N => self.n_f, MClass::P => self.p_f } }
}

/// Length class of an element-wise expression: None = pure scalar. `Err` when
/// classes mix or the matrix appears element-wise.
fn class_of(e: &r2_types::Expr, ctx: &MCtx, scalars: &HashMap<Arc<str>, Value>) -> JitResult<Option<MClass>> {
    use r2_types::Expr::*;
    let combine = |a: Option<MClass>, b: Option<MClass>| -> JitResult<Option<MClass>> {
        match (a, b) {
            (None, x) | (x, None) => Ok(x),
            (Some(x), Some(y)) if x == y => Ok(Some(x)),
            _ => Err(JitError::Unsupported("matrix kernel: mixed N/P vector lengths".into())),
        }
    };
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) => Ok(None),
        Symbol(s) => {
            if s.as_ref() == ctx.mat.as_ref() {
                return Err(JitError::Unsupported("matrix kernel: matrix used element-wise".into()));
            }
            if s.as_ref() == ctx.yname.as_ref() { return Ok(Some(MClass::N)); }
            if let Some((_, c)) = ctx.bufs.borrow().get(s.as_ref()) { return Ok(Some(*c)); }
            if scalars.contains_key(s.as_ref()) { return Ok(None); }
            Err(JitError::Unsupported(format!("matrix kernel: `{}` read before defined", s)))
        }
        Unary { expr, .. } => class_of(expr, ctx, scalars),
        Binary { op: BinOp::MatMul, .. } => Err(JitError::Unsupported("matrix kernel: %*% nested in expression".into())),
        Binary { lhs, rhs, .. } => combine(class_of(lhs, ctx, scalars)?, class_of(rhs, ctx, scalars)?),
        If { cond, then, else_ } => {
            let c = combine(class_of(cond, ctx, scalars)?, class_of(then, ctx, scalars)?)?;
            match else_ { Some(x) => combine(c, class_of(x, ctx, scalars)?), None => Ok(c) }
        }
        Call { func, args } => {
            let name = match func.as_ref() { Symbol(s) => s.as_ref(), _ => return Err(JitError::Unsupported("matrix kernel: call target".into())) };
            match name {
                // Reductions & size queries collapse to scalar.
                "sum" | "prod" | "mean" | "length" | "nrow" | "ncol" => Ok(None),
                "sqrt" | "abs" => class_of(&args[0].value, ctx, scalars),
                _ => Err(JitError::Unsupported(format!("matrix kernel: call `{}`", name))),
            }
        }
        _ => Err(JitError::Unsupported("matrix kernel: unsupported node".into())),
    }
}

/// Element-position value: `elem_env` holds the current element of each
/// same-class buffer (and `y` in N-loops); `scalars` are loop-invariant.
fn melem(bcx: &mut FunctionBuilder, ctx: &MCtx, e: &r2_types::Expr, scalars: &HashMap<Arc<str>, Value>, elem_env: &HashMap<Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => elem_env.get(s.as_ref()).or_else(|| scalars.get(s.as_ref())).copied()
            .ok_or_else(|| JitError::Unsupported(format!("matrix kernel: unknown `{}`", s))),
        Unary { op, expr } => {
            let v = melem(bcx, ctx, expr, scalars, elem_env)?;
            Ok(match op { r2_types::UnOp::Neg => bcx.ins().fneg(v), r2_types::UnOp::Pos => v,
                r2_types::UnOp::Not => { let z = bcx.ins().f64const(0.0); let c = bcx.ins().fcmp(FloatCC::Equal, v, z);
                    let one = bcx.ins().f64const(1.0); let zz = bcx.ins().f64const(0.0); bcx.ins().select(c, one, zz) } })
        }
        Binary { op, lhs, rhs } => {
            let a = melem(bcx, ctx, lhs, scalars, elem_env)?;
            let b = melem(bcx, ctx, rhs, scalars, elem_env)?;
            let cmp = |bcx: &mut FunctionBuilder, cc: FloatCC, a: Value, b: Value| {
                let c = bcx.ins().fcmp(cc, a, b);
                let one = bcx.ins().f64const(1.0); let z = bcx.ins().f64const(0.0);
                bcx.ins().select(c, one, z)
            };
            Ok(match op {
                BinOp::Add => bcx.ins().fadd(a, b), BinOp::Sub => bcx.ins().fsub(a, b),
                BinOp::Mul => bcx.ins().fmul(a, b), BinOp::Div => bcx.ins().fdiv(a, b),
                BinOp::Lt => cmp(bcx, FloatCC::LessThan, a, b), BinOp::Gt => cmp(bcx, FloatCC::GreaterThan, a, b),
                BinOp::Le => cmp(bcx, FloatCC::LessThanOrEqual, a, b), BinOp::Ge => cmp(bcx, FloatCC::GreaterThanOrEqual, a, b),
                BinOp::Eq => cmp(bcx, FloatCC::Equal, a, b), BinOp::Ne => cmp(bcx, FloatCC::NotEqual, a, b),
                other => return Err(JitError::Unsupported(format!("matrix kernel: binop {:?}", other))),
            })
        }
        If { cond, then, else_ } => {
            let e2 = else_.as_ref().ok_or_else(|| JitError::Unsupported("matrix kernel: if without else".into()))?;
            let c = melem(bcx, ctx, cond, scalars, elem_env)?;
            let t = melem(bcx, ctx, then, scalars, elem_env)?;
            let f = melem(bcx, ctx, e2, scalars, elem_env)?;
            let z = bcx.ins().f64const(0.0);
            let nz = bcx.ins().fcmp(FloatCC::NotEqual, c, z);
            Ok(bcx.ins().select(nz, t, f))
        }
        Call { func, args } => {
            let name = match func.as_ref() { Symbol(s) => s.as_ref(), _ => return Err(JitError::Unsupported("matrix kernel: call".into())) };
            // Size queries are loop-invariant scalars — resolve without
            // evaluating the argument (which may be the matrix itself).
            match name {
                "nrow" => return Ok(ctx.n_f),
                "ncol" => return Ok(ctx.p_f),
                "length" => {
                    if let Symbol(s) = &args[0].value {
                        if s.as_ref() == ctx.yname.as_ref() { return Ok(ctx.n_f); }
                        if let Some((_, c)) = ctx.bufs.borrow().get(s.as_ref()) { return Ok(ctx.len_f(*c)); }
                    }
                    return Err(JitError::Unsupported("matrix kernel: length()".into()));
                }
                _ => {}
            }
            let a0 = melem(bcx, ctx, &args[0].value, scalars, elem_env)?;
            Ok(match name {
                "sqrt" => bcx.ins().sqrt(a0),
                "abs" => bcx.ins().fabs(a0),
                _ => return Err(JitError::Unsupported(format!("matrix kernel: elem call `{}`", name))),
            })
        }
        _ => Err(JitError::Unsupported("matrix kernel: elem node".into())),
    }
}

/// Names of same-class buffers (plus `y` for N) referenced by `e`.
fn elem_refs(e: &r2_types::Expr, ctx: &MCtx, class: MClass, out: &mut Vec<Arc<str>>) {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => {
            let is_y = s.as_ref() == ctx.yname.as_ref() && class == MClass::N;
            let is_buf = ctx.bufs.borrow().get(s.as_ref()).map_or(false, |(_, c)| *c == class);
            if (is_y || is_buf) && !out.iter().any(|n| n.as_ref() == s.as_ref()) { out.push(s.clone()); }
        }
        Unary { expr, .. } => elem_refs(expr, ctx, class, out),
        Binary { lhs, rhs, .. } => { elem_refs(lhs, ctx, class, out); elem_refs(rhs, ctx, class, out); }
        If { cond, then, else_ } => { elem_refs(cond, ctx, class, out); elem_refs(then, ctx, class, out);
            if let Some(x) = else_ { elem_refs(x, ctx, class, out); } }
        Call { args, .. } => for a in args { elem_refs(&a.value, ctx, class, out); },
        _ => {}
    }
}

/// One element loop over `class`: either store `elem` into `dst` (map) or
/// accumulate (reduction). Returns the accumulator for reductions.
fn mloop(
    bcx: &mut FunctionBuilder, ctx: &MCtx, class: MClass,
    elem: &r2_types::Expr, scalars: &HashMap<Arc<str>, Value>,
    dst: Option<Value>, red: Option<FusedReduceOp>,
) -> JitResult<Option<Value>> {
    let len = ctx.len(class);
    let header = bcx.create_block();
    let bodyb = bcx.create_block();
    let exit = bcx.create_block();
    bcx.append_block_param(header, types::I64);
    if red.is_some() { bcx.append_block_param(header, types::F64); bcx.append_block_param(exit, types::F64); }

    let zero = bcx.ins().iconst(types::I64, 0);
    let mut init = vec![zero];
    if let Some(op) = red {
        init.push(bcx.ins().f64const(match op { FusedReduceOp::Sum => 0.0, FusedReduceOp::Prod => 1.0 }));
    }
    bcx.ins().jump(header, &init);

    bcx.switch_to_block(header);
    let hp: Vec<Value> = bcx.block_params(header).to_vec();
    let cond = bcx.ins().icmp(IntCC::SignedLessThan, hp[0], len);
    bcx.ins().brif(cond, bodyb, &[], exit, &hp[1..]);

    bcx.switch_to_block(bodyb);
    let i = hp[0];
    let eight = bcx.ins().iconst(types::I64, 8);
    let off = bcx.ins().imul(i, eight);
    let mf = MemFlags::trusted();
    let mut elem_env: HashMap<Arc<str>, Value> = HashMap::new();
    let mut refs = Vec::new();
    elem_refs(elem, ctx, class, &mut refs);
    for nm in &refs {
        let base = if nm.as_ref() == ctx.yname.as_ref() { ctx.y_ptr }
                   else { ctx.bufs.borrow().get(nm.as_ref()).unwrap().0 };
        let addr = bcx.ins().iadd(base, off);
        elem_env.insert(nm.clone(), bcx.ins().load(types::F64, mf, addr, 0));
    }
    let v = melem(bcx, ctx, elem, scalars, &elem_env)?;
    let mut next = Vec::new();
    let one = bcx.ins().iconst(types::I64, 1);
    next.push(bcx.ins().iadd(i, one));
    if let Some(op) = red {
        let acc = hp[1];
        next.push(match op { FusedReduceOp::Sum => bcx.ins().fadd(acc, v), FusedReduceOp::Prod => bcx.ins().fmul(acc, v) });
    } else if let Some(d) = dst {
        let addr = bcx.ins().iadd(d, off);
        bcx.ins().store(mf, v, addr, 0);
    }
    bcx.ins().jump(header, &next);

    bcx.switch_to_block(exit);
    if red.is_some() { Ok(Some(bcx.block_params(exit)[0])) } else { Ok(None) }
}

/// Scalar expression: literals, scalar locals, arithmetic, if/else, sqrt/abs,
/// `nrow(X)`/`ncol(X)`/`length(v)`, and reductions over element expressions.
fn mscalar(bcx: &mut FunctionBuilder, ctx: &MCtx, e: &r2_types::Expr, scalars: &HashMap<Arc<str>, Value>) -> JitResult<Value> {
    use r2_types::Expr::*;
    match e {
        NumLit(x) => Ok(bcx.ins().f64const(*x)),
        IntLit(x) => Ok(bcx.ins().f64const(*x as f64)),
        BoolLit(b) => Ok(bcx.ins().f64const(if *b { 1.0 } else { 0.0 })),
        Symbol(s) => scalars.get(s.as_ref()).copied()
            .ok_or_else(|| JitError::Unsupported(format!("matrix kernel: scalar `{}` unknown", s))),
        Unary { op, expr } => {
            let v = mscalar(bcx, ctx, expr, scalars)?;
            Ok(match op { r2_types::UnOp::Neg => bcx.ins().fneg(v), _ => v })
        }
        Binary { op, lhs, rhs } => {
            let a = mscalar(bcx, ctx, lhs, scalars)?;
            let b = mscalar(bcx, ctx, rhs, scalars)?;
            let cmp = |bcx: &mut FunctionBuilder, cc: FloatCC, a: Value, b: Value| {
                let c = bcx.ins().fcmp(cc, a, b);
                let one = bcx.ins().f64const(1.0); let z = bcx.ins().f64const(0.0);
                bcx.ins().select(c, one, z)
            };
            Ok(match op {
                BinOp::Add => bcx.ins().fadd(a, b), BinOp::Sub => bcx.ins().fsub(a, b),
                BinOp::Mul => bcx.ins().fmul(a, b), BinOp::Div => bcx.ins().fdiv(a, b),
                BinOp::Lt => cmp(bcx, FloatCC::LessThan, a, b), BinOp::Gt => cmp(bcx, FloatCC::GreaterThan, a, b),
                BinOp::Le => cmp(bcx, FloatCC::LessThanOrEqual, a, b), BinOp::Ge => cmp(bcx, FloatCC::GreaterThanOrEqual, a, b),
                BinOp::Eq => cmp(bcx, FloatCC::Equal, a, b), BinOp::Ne => cmp(bcx, FloatCC::NotEqual, a, b),
                other => return Err(JitError::Unsupported(format!("matrix kernel: scalar binop {:?}", other))),
            })
        }
        If { cond, then, else_ } => {
            let e2 = else_.as_ref().ok_or_else(|| JitError::Unsupported("matrix kernel: if without else".into()))?;
            let c = mscalar(bcx, ctx, cond, scalars)?;
            let t = mscalar(bcx, ctx, then, scalars)?;
            let f = mscalar(bcx, ctx, e2, scalars)?;
            let z = bcx.ins().f64const(0.0);
            let nz = bcx.ins().fcmp(FloatCC::NotEqual, c, z);
            Ok(bcx.ins().select(nz, t, f))
        }
        Call { func, args } => {
            let name = match func.as_ref() { Symbol(s) => s.as_ref(), _ => return Err(JitError::Unsupported("matrix kernel: call".into())) };
            match name {
                "nrow" => Ok(ctx.n_f),
                "ncol" => Ok(ctx.p_f),
                "length" => {
                    let cls = class_of(&args[0].value, ctx, scalars)?
                        .ok_or_else(|| JitError::Unsupported("matrix kernel: length of scalar".into()))?;
                    Ok(ctx.len_f(cls))
                }
                "sum" | "prod" | "mean" => {
                    let inner = &args[0].value;
                    let cls = class_of(inner, ctx, scalars)?
                        .ok_or_else(|| JitError::Unsupported("matrix kernel: reduction of scalar".into()))?;
                    let op = if name == "prod" { FusedReduceOp::Prod } else { FusedReduceOp::Sum };
                    let acc = mloop(bcx, ctx, cls, inner, scalars, None, Some(op))?.unwrap();
                    if name == "mean" { Ok(bcx.ins().fdiv(acc, ctx.len_f(cls))) } else { Ok(acc) }
                }
                "sqrt" => { let v = mscalar(bcx, ctx, &args[0].value, scalars)?; Ok(bcx.ins().sqrt(v)) }
                "abs" => { let v = mscalar(bcx, ctx, &args[0].value, scalars)?; Ok(bcx.ins().fabs(v)) }
                _ => Err(JitError::Unsupported(format!("matrix kernel: scalar call `{}`", name))),
            }
        }
        _ => Err(JitError::Unsupported("matrix kernel: scalar node".into())),
    }
}

/// Emit one statement. Returns Ok(()) or Err → the whole compile falls back.
fn emit_mstmt(bcx: &mut FunctionBuilder, ctx: &MCtx, st: &r2_types::Expr, scalars: &mut HashMap<Arc<str>, Value>) -> JitResult<()> {
    use r2_types::Expr::*;
    match st {
        Assign { target, value, .. } => {
            let nm = match target.as_ref() { Symbol(n) => n.clone(),
                _ => return Err(JitError::Unsupported("matrix kernel: non-symbol assign".into())) };
            // (a) matvec: nm <- X %*% v  |  nm <- t(X) %*% v
            if let Binary { op: BinOp::MatMul, lhs, rhs } = value.as_ref() {
                let (transposed, m_ok) = match lhs.as_ref() {
                    Symbol(s) if s.as_ref() == ctx.mat.as_ref() => (false, true),
                    Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "t")
                        && args.len() == 1
                        && matches!(&args[0].value, Symbol(s) if s.as_ref() == ctx.mat.as_ref()) => (true, true),
                    _ => (false, false),
                };
                if !m_ok { return Err(JitError::Unsupported("matrix kernel: %*% lhs must be X or t(X)".into())); }
                let src_name = match rhs.as_ref() { Symbol(s) => s.clone(),
                    _ => return Err(JitError::Unsupported("matrix kernel: %*% rhs must be a named vector".into())) };
                let (want_src, dst_class) = if transposed { (MClass::N, MClass::P) } else { (MClass::P, MClass::N) };
                let src_ptr = if src_name.as_ref() == ctx.yname.as_ref() {
                    if want_src != MClass::N { return Err(JitError::Unsupported("matrix kernel: X %*% y shape".into())); }
                    ctx.y_ptr
                } else {
                    let b = ctx.bufs.borrow();
                    let (ptr, c) = *b.get(src_name.as_ref())
                        .ok_or_else(|| JitError::Unsupported(format!("matrix kernel: `{}` read before defined", src_name)))?;
                    if c != want_src { return Err(JitError::Unsupported("matrix kernel: %*% operand length".into())); }
                    ptr
                };
                let (dst_ptr, dc) = *ctx.bufs.borrow().get(nm.as_ref())
                    .ok_or_else(|| JitError::Unsupported("matrix kernel: matvec target unbuffered".into()))?;
                if dc != dst_class { return Err(JitError::Unsupported("matrix kernel: matvec target class".into())); }
                let fref = if transposed { ctx.tmatvec_ref } else { ctx.matvec_ref };
                bcx.ins().call(fref, &[ctx.m_ptr, ctx.n, ctx.p, src_ptr, dst_ptr]);
                return Ok(());
            }
            // (b) rep(s, ncol(X)) / rep(s, nrow(X)) → fill.
            if let Call { func, args } = value.as_ref() {
                if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "rep") && args.len() == 2 {
                    let fill = mscalar(bcx, ctx, &args[0].value, scalars)?;
                    let cls = match &args[1].value {
                        Call { func: f2, args: a2 } if a2.len() == 1
                            && matches!(&a2[0].value, Symbol(s) if s.as_ref() == ctx.mat.as_ref()) =>
                            match f2.as_ref() {
                                Symbol(n2) if n2.as_ref() == "ncol" => MClass::P,
                                Symbol(n2) if n2.as_ref() == "nrow" => MClass::N,
                                _ => return Err(JitError::Unsupported("matrix kernel: rep size".into())),
                            },
                        _ => return Err(JitError::Unsupported("matrix kernel: rep size must be nrow/ncol(X)".into())),
                    };
                    let (ptr, dc) = *ctx.bufs.borrow().get(nm.as_ref())
                        .ok_or_else(|| JitError::Unsupported("matrix kernel: rep target unbuffered".into()))?;
                    if dc != cls { return Err(JitError::Unsupported("matrix kernel: rep class".into())); }
                    // fill: reuse mloop with a scalar-const elem (a Symbol bound in scalars).
                    let mut sc = scalars.clone();
                    sc.insert(Arc::from(".__fill"), fill);
                    mloop(bcx, ctx, cls, &Symbol(Arc::from(".__fill")), &sc, Some(ptr), None)?;
                    return Ok(());
                }
            }
            // (c) element-wise vector update or (d) scalar.
            match class_of(value, ctx, scalars)? {
                Some(cls) => {
                    let (ptr, dc) = *ctx.bufs.borrow().get(nm.as_ref())
                        .ok_or_else(|| JitError::Unsupported("matrix kernel: vector target unbuffered".into()))?;
                    if dc != cls { return Err(JitError::Unsupported("matrix kernel: assign class mismatch".into())); }
                    mloop(bcx, ctx, cls, value, scalars, Some(ptr), None)?;
                }
                None => {
                    let v = mscalar(bcx, ctx, value, scalars)?;
                    scalars.insert(nm, v);
                }
            }
            Ok(())
        }
        For { var, iter, body } => {
            let hi_expr = match iter.as_ref() {
                Binary { op: BinOp::Colon, lhs, rhs }
                    if matches!(lhs.as_ref(), NumLit(x) if *x == 1.0) || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
                _ => return Err(JitError::Unsupported("matrix kernel: loop must be `1:K`".into())),
            };
            let body_stmts: Vec<r2_types::Expr> = match body.as_ref() {
                Block(b) => b.clone(),
                s => vec![s.clone()],
            };
            // Scalar-carried = assign targets that are NOT buffers.
            let mut carried: Vec<Arc<str>> = Vec::new();
            for bs in &body_stmts {
                if let Assign { target, .. } = bs {
                    if let Symbol(n) = target.as_ref() {
                        if !ctx.bufs.borrow().contains_key(n.as_ref())
                            && !carried.iter().any(|c| c.as_ref() == n.as_ref()) {
                            carried.push(n.clone());
                        }
                    }
                }
            }
            let hi_f = mscalar(bcx, ctx, hi_expr, scalars)?;
            let hi_i = bcx.ins().fcvt_to_sint(types::I64, hi_f);
            let one = bcx.ins().iconst(types::I64, 1);
            let neg1 = bcx.ins().iconst(types::I64, -1);
            let asc = bcx.ins().icmp(IntCC::SignedGreaterThanOrEqual, hi_i, one);
            let step = bcx.ins().select(asc, one, neg1);

            let header = bcx.create_block();
            let bodyb = bcx.create_block();
            let exit = bcx.create_block();
            bcx.append_block_param(header, types::I64);
            for _ in &carried { bcx.append_block_param(header, types::F64); }
            for _ in &carried { bcx.append_block_param(exit, types::F64); }

            let zero = bcx.ins().f64const(0.0);
            let mut init = vec![one];
            for c in &carried { init.push(scalars.get(c.as_ref()).copied().unwrap_or(zero)); }
            bcx.ins().jump(header, &init);

            bcx.switch_to_block(header);
            let hp: Vec<Value> = bcx.block_params(header).to_vec();
            let diff = bcx.ins().isub(hp[0], hi_i);
            let prod = bcx.ins().imul(diff, step);
            let cond = bcx.ins().icmp_imm(IntCC::SignedLessThan, prod, 1);
            bcx.ins().brif(cond, bodyb, &[], exit, &hp[1..]);

            bcx.switch_to_block(bodyb);
            let mut env2 = scalars.clone();
            for (j, c) in carried.iter().enumerate() { env2.insert(c.clone(), hp[1 + j]); }
            let iv = bcx.ins().fcvt_from_sint(types::F64, hp[0]);
            env2.insert(var.clone(), iv);
            for bs in &body_stmts { emit_mstmt(bcx, ctx, bs, &mut env2)?; }
            let mut next = vec![bcx.ins().iadd(hp[0], step)];
            for c in &carried {
                next.push(*env2.get(c.as_ref()).ok_or_else(|| JitError::Unsupported("matrix kernel: carried lost".into()))?);
            }
            bcx.ins().jump(header, &next);

            bcx.switch_to_block(exit);
            let xp: Vec<Value> = bcx.block_params(exit).to_vec();
            for (j, c) in carried.iter().enumerate() { scalars.insert(c.clone(), xp[j]); }
            scalars.insert(var.clone(), hi_f);
            Ok(())
        }
        _ => Err(JitError::Unsupported("matrix kernel: unsupported statement".into())),
    }
}

/// Pre-scan (program order, into loop bodies): classify every buffered name.
fn prescan(stmts: &[r2_types::Expr], ctx_mat: &str, ctx_y: &str, found: &mut Vec<(Arc<str>, MClass)>) -> JitResult<()> {
    use r2_types::Expr::*;
    // Local class query against what's found so far (y = N, buffers by class).
    fn cls_q(e: &r2_types::Expr, mat: &str, y: &str, found: &[(Arc<str>, MClass)]) -> Option<MClass> {
        match e {
            Symbol(s) if s.as_ref() == y => Some(MClass::N),
            Symbol(s) => found.iter().find(|(n, _)| n.as_ref() == s.as_ref()).map(|(_, c)| *c),
            Unary { expr, .. } => cls_q(expr, mat, y, found),
            Binary { op: BinOp::MatMul, .. } => None,
            Binary { lhs, rhs, .. } => cls_q(lhs, mat, y, found).or_else(|| cls_q(rhs, mat, y, found)),
            If { cond, then, else_ } => cls_q(cond, mat, y, found).or_else(|| cls_q(then, mat, y, found))
                .or_else(|| else_.as_ref().and_then(|x| cls_q(x, mat, y, found))),
            Call { func, args } => {
                if matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean" | "length" | "nrow" | "ncol" | "rep")) { return None; }
                args.iter().find_map(|a| cls_q(&a.value, mat, y, found))
            }
            _ => None,
        }
    }
    for st in stmts {
        match st {
            Assign { target, value, .. } => {
                let nm = match target.as_ref() { Symbol(n) => n.clone(), _ => continue };
                let class: Option<MClass> = match value.as_ref() {
                    Binary { op: BinOp::MatMul, lhs, .. } => Some(match lhs.as_ref() {
                        Call { .. } => MClass::P,   // t(X) %*% v → p
                        _ => MClass::N,             // X %*% v → n
                    }),
                    Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "rep") && args.len() == 2 =>
                        match &args[1].value {
                            Call { func: f2, .. } if matches!(f2.as_ref(), Symbol(n2) if n2.as_ref() == "ncol") => Some(MClass::P),
                            Call { func: f2, .. } if matches!(f2.as_ref(), Symbol(n2) if n2.as_ref() == "nrow") => Some(MClass::N),
                            _ => None,
                        },
                    other => cls_q(other, ctx_mat, ctx_y, found),
                };
                if let Some(c) = class {
                    if let Some((_, prev)) = found.iter().find(|(n, _)| n.as_ref() == nm.as_ref()) {
                        if *prev != c { return Err(JitError::Unsupported("matrix kernel: name changes length class".into())); }
                    } else {
                        found.push((nm, c));
                    }
                }
            }
            For { body, .. } | While { body, .. } => {
                let inner: Vec<r2_types::Expr> = match body.as_ref() { Block(b) => b.clone(), s => vec![s.clone()] };
                prescan(&inner, ctx_mat, ctx_y, found)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Compile `function(X, y) { ... ; <p-vector> }` (X = `mat_name`, y = `vec_name`).
pub(crate) fn compile_matvec_kernel(body: &r2_types::Expr, mat_name: &Arc<str>, vec_name: &Arc<str>) -> JitResult<CompiledFn> {
    let stmts: Vec<r2_types::Expr> = match body {
        r2_types::Expr::Block(b) if b.len() >= 2 => b.clone(),
        _ => return Err(JitError::Unsupported("matrix kernel: body must be a block".into())),
    };
    let (last, init) = stmts.split_last().unwrap();

    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::CraneliftError(format!("JITBuilder: {:?}", e)))?;
    jit_builder.symbol("__r2_scratch_alloc", r2_scratch_alloc as *const u8);
    jit_builder.symbol("__r2_scratch_free", r2_scratch_free as *const u8);
    jit_builder.symbol("__r2_kern_matvec", r2_kern_matvec as *const u8);
    jit_builder.symbol("__r2_kern_tmatvec", r2_kern_tmatvec as *const u8);
    let mut module = JITModule::new(jit_builder);

    let mut asig = module.make_signature();
    asig.params.push(AbiParam::new(types::I64));
    asig.returns.push(AbiParam::new(types::I64));
    let alloc_id = module.declare_function("__r2_scratch_alloc", Linkage::Import, &asig)
        .map_err(|e| JitError::CraneliftError(format!("{:?}", e)))?;
    let mut fsig = module.make_signature();
    fsig.params.push(AbiParam::new(types::I64));
    fsig.params.push(AbiParam::new(types::I64));
    let free_id = module.declare_function("__r2_scratch_free", Linkage::Import, &fsig)
        .map_err(|e| JitError::CraneliftError(format!("{:?}", e)))?;
    let mut msig = module.make_signature();
    for _ in 0..5 { msig.params.push(AbiParam::new(types::I64)); }
    let mv_id = module.declare_function("__r2_kern_matvec", Linkage::Import, &msig)
        .map_err(|e| JitError::CraneliftError(format!("{:?}", e)))?;
    let tmv_id = module.declare_function("__r2_kern_tmatvec", Linkage::Import, &msig)
        .map_err(|e| JitError::CraneliftError(format!("{:?}", e)))?;

    // Kernel signature: (m_ptr, nrow, ncol, v_ptr, out_ptr), all i64.
    let mut sig = module.make_signature();
    for _ in 0..5 { sig.params.push(AbiParam::new(types::I64)); }
    let func_id = module.declare_function("__jit_matvec_kernel", Linkage::Export, &sig)
        .map_err(|e| JitError::CraneliftError(format!("declare: {:?}", e)))?;
    let mut fctx = module.make_context();
    fctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut fctx.func, &mut fbctx);
        let aref = module.declare_func_in_func(alloc_id, &mut bcx.func);
        let fref = module.declare_func_in_func(free_id, &mut bcx.func);
        let mvref = module.declare_func_in_func(mv_id, &mut bcx.func);
        let tmvref = module.declare_func_in_func(tmv_id, &mut bcx.func);

        let entry = bcx.create_block();
        bcx.append_block_params_for_function_params(entry);
        bcx.switch_to_block(entry);
        let ps: Vec<Value> = bcx.block_params(entry).to_vec();
        let (m_ptr, n, p, y_ptr, out_ptr) = (ps[0], ps[1], ps[2], ps[3], ps[4]);
        let n_f = bcx.ins().fcvt_from_sint(types::F64, n);
        let p_f = bcx.ins().fcvt_from_sint(types::F64, p);
        let ctx = MCtx {
            m_ptr, n, p, n_f, p_f, y_ptr,
            mat: mat_name.clone(), yname: vec_name.clone(),
            matvec_ref: mvref, tmatvec_ref: tmvref,
            bufs: std::cell::RefCell::new(HashMap::new()),
        };

        // Buffers: pre-scan classes, allocate once at entry.
        let mut found: Vec<(Arc<str>, MClass)> = Vec::new();
        prescan(&stmts, mat_name.as_ref(), vec_name.as_ref(), &mut found)?;
        let eight = bcx.ins().iconst(types::I64, 8);
        for (nm, cls) in &found {
            let bytes = bcx.ins().imul(ctx.len(*cls), eight);
            let call = bcx.ins().call(aref, &[bytes]);
            let ptr = bcx.inst_results(call)[0];
            ctx.bufs.borrow_mut().insert(nm.clone(), (ptr, *cls));
        }

        // NOTE on safety: `class_of`/`mscalar`/`melem` error on any name that is
        // neither a param, a buffer, nor an already-defined scalar — but a
        // buffer READ before its first assignment would see uninitialised
        // memory. Guard: zero-fill every buffer at entry is NOT enough to match
        // R (R errors); instead reject bodies where a buffer is read before its
        // defining statement, which the class checks above cannot see. Enforced
        // here syntactically, mirroring the scalar kernels.
        {
            let mut defined: std::collections::HashSet<String> = std::collections::HashSet::new();
            defined.insert(vec_name.to_string());
            fn check(stmts: &[r2_types::Expr], defined: &mut std::collections::HashSet<String>, mat: &str) -> JitResult<()> {
                use r2_types::Expr::*;
                fn reads_ok(e: &r2_types::Expr, defined: &std::collections::HashSet<String>, mat: &str, lv: Option<&str>) -> bool {
                    match e {
                        Symbol(s) => defined.contains(s.as_ref()) || s.as_ref() == mat || lv == Some(s.as_ref()),
                        NumLit(_) | IntLit(_) | BoolLit(_) => true,
                        Unary { expr, .. } => reads_ok(expr, defined, mat, lv),
                        Binary { lhs, rhs, .. } => reads_ok(lhs, defined, mat, lv) && reads_ok(rhs, defined, mat, lv),
                        If { cond, then, else_ } => reads_ok(cond, defined, mat, lv) && reads_ok(then, defined, mat, lv)
                            && else_.as_ref().map_or(true, |x| reads_ok(x, defined, mat, lv)),
                        Call { args, .. } => args.iter().all(|a| reads_ok(&a.value, defined, mat, lv)),
                        _ => false,
                    }
                }
                for st in stmts {
                    match st {
                        Assign { target, value, .. } => {
                            if !reads_ok(value, defined, mat, None) {
                                return Err(JitError::Unsupported("matrix kernel: read before define".into()));
                            }
                            if let Symbol(n) = target.as_ref() { defined.insert(n.to_string()); }
                        }
                        For { var, iter, body } => {
                            if !reads_ok(iter, defined, mat, None) {
                                return Err(JitError::Unsupported("matrix kernel: loop bound read".into()));
                            }
                            let inner: Vec<r2_types::Expr> = match body.as_ref() { Block(b) => b.clone(), s => vec![s.clone()] };
                            // Within the body, the loop var is defined.
                            let mut d2 = defined.clone();
                            d2.insert(var.to_string());
                            check(&inner, &mut d2, mat)?;
                            // Names the body defines persist after the loop.
                            for bs in &inner { if let Assign { target, .. } = bs { if let Symbol(n) = target.as_ref() { defined.insert(n.to_string()); } } }
                        }
                        _ => return Err(JitError::Unsupported("matrix kernel: statement kind".into())),
                    }
                }
                Ok(())
            }
            let mut all = init.to_vec();
            all.push(r2_types::Expr::Assign {
                target: Box::new(r2_types::Expr::Symbol(Arc::from(".__out"))),
                value: Box::new(last.clone()), superassign: false });
            check(&all, &mut defined, mat_name.as_ref())?;
        }

        let mut scalars: HashMap<Arc<str>, Value> = HashMap::new();
        for st in init { emit_mstmt(&mut bcx, &ctx, st, &mut scalars)?; }

        // Final: a p-class element expression written to out.
        match class_of(last, &ctx, &scalars)? {
            Some(MClass::P) => { mloop(&mut bcx, &ctx, MClass::P, last, &scalars, Some(out_ptr), None)?; }
            _ => return Err(JitError::Unsupported("matrix kernel: result must be a p-vector".into())),
        }

        // Free buffers.
        let ptrs: Vec<(Value, MClass)> = ctx.bufs.borrow().values().copied().collect();
        let eight_e = bcx.ins().iconst(types::I64, 8);
        for (ptr, cls) in ptrs {
            let bytes = bcx.ins().imul(ctx.len(cls), eight_e);
            bcx.ins().call(fref, &[ptr, bytes]);
        }
        bcx.ins().return_(&[]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }
    module.define_function(func_id, &mut fctx).map_err(|e| JitError::CraneliftError(format!("define: {:?}", e)))?;
    module.clear_context(&mut fctx);
    module.finalize_definitions().map_err(|e| JitError::CraneliftError(format!("finalize: {:?}", e)))?;
    let ptr = module.get_finalized_function(func_id);
    Ok(CompiledFn { ptr, arity: 2, kind: r2_types::JitKind::MatVecIterOut, _module: module })
}
