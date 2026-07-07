//! Closure -> JIT entry point (Phase C.2) + scalar helpers.

use r2_types::infer::IrElem;
use crate::*;

// ── Closure → JIT (Phase C.2 entry-point for the engine) ─────────────

/// Attempt to JIT-compile a Closure into a scalar `(f64, ...) -> f64`
/// specialization. Returns `None` if the closure has zero or more than
/// two parameters with default expressions, or its body contains
/// constructs the JIT does not yet support.
///
/// On success, the engine should cache the returned handle keyed by
/// `Arc::as_ptr(&closure.body)` so re-calls reuse the compiled code.
/// Extract a single f64 scalar from an `RVal` if it's a numeric scalar
/// (Real / Int / Bool of length 1, non-NA). Used by the closure-capture
/// inference path to detect "bakeable" free-variable references.
fn scalar_f64_of(v: &r2_types::RVal) -> Option<f64> {
    match v {
        r2_types::RVal::Numeric(r, _) if r.len() == 1 => r[0],
        r2_types::RVal::Integer(r, _) if r.len() == 1 => r[0].map(|n| n as f64),
        r2_types::RVal::Logical(r, _) if r.len() == 1 => r[0].map(|b| if b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Compile-time constant: is the Cranelift JIT functional on this target?
///
/// `cranelift-jit` 0.105 only implements PLT relocation on `x86_64`. On
/// aarch64 (Apple Silicon, ARM Linux, etc.) `JITModule::new()` panics
/// when it encounters any function that needs a PLT entry. We gate the
/// public entry point on this constant so the engine cleanly falls back
/// to the interpreter on unsupported targets, without ever touching
/// Cranelift's PLT path. Lifting this gate is a v0.2.0 task that
/// involves upgrading Cranelift to a version with aarch64 PLT support.
pub const JIT_SUPPORTED: bool = cfg!(target_arch = "x86_64");

/// Allowlist: is every node in `e` a construct that `r2_ir`'s lowering
/// represents *faithfully*? The lowering's catch-all arm silently turns
/// unhandled expressions (notably `for`, `repeat`, `match`, `tryCatch`,
/// `break`/`next`) into a no-op `Null` const — which would make the JIT
/// emit code that quietly skips them and returns a wrong scalar (e.g. a
/// `for`-loop accumulator returning its init value). Rather than denylist
/// the silently-dropped constructs (fragile as the AST grows), we only
/// admit bodies built entirely from the faithfully-lowered set; anything
/// else returns `false` and `try_compile_closure` falls back to the
/// interpreter, which handles every construct correctly.
///
/// Note: `Call`/`Index`/`StrLit` etc. that the scalar codegen can't emit
/// still fail *loudly* (the compile returns `Err` and we fall back) — they
/// don't need gating here. This gate exists only for the constructs that
/// would otherwise compile to silently-wrong code.
pub(crate) fn body_is_jit_lowerable(e: &r2_types::Expr) -> bool {
    use r2_types::Expr::*;
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | NaLit | NullLit | Symbol(_) => true,
        Unary { expr, .. } => body_is_jit_lowerable(expr),
        Binary { lhs, rhs, .. } => body_is_jit_lowerable(lhs) && body_is_jit_lowerable(rhs),
        Assign { value, .. } => body_is_jit_lowerable(value),
        Call { func, args } => {
            body_is_jit_lowerable(func)
                && args.iter().all(|a| body_is_jit_lowerable(&a.value))
        }
        If { cond, then, else_ } => {
            body_is_jit_lowerable(cond)
                && body_is_jit_lowerable(then)
                && else_.as_ref().map_or(true, |e| body_is_jit_lowerable(e))
        }
        While { cond, body } => body_is_jit_lowerable(cond) && body_is_jit_lowerable(body),
        // Phase J.1: counted `for(v in a:b)` only — the IR lowers exactly
        // this form (with loop-carried phis); other iterables fall back.
        For { iter, body, .. } => matches!(iter.as_ref(), Binary { op: r2_types::BinOp::Colon, .. })
            && body_is_jit_lowerable(iter) && body_is_jit_lowerable(body),
        Block(stmts) => stmts.iter().all(body_is_jit_lowerable),
        Return(v) => body_is_jit_lowerable(v),
        Pipe { lhs, rhs } => body_is_jit_lowerable(lhs) && body_is_jit_lowerable(rhs),
        // For / Repeat / Match / TryCatch / Break / Next / FuncDef / Lambda /
        // Index / DblIndex / Dollar / Namespace / StrLit / FStringLit / Dots /
        // TypeDef / MethodDef — not faithfully lowered (or not scalar-numeric):
        // reject so the engine uses the interpreter.
        _ => false,
    }
}

/// Replace every `v[i]` (`Index{Symbol(v), Symbol(i)}`) with `Symbol(v)` so a
/// per-iteration contribution becomes a map body over the element. Recurses
/// through arithmetic / unary / calls (the map-body-eligible shapes).
fn subst_vi(e: &r2_types::Expr, v: &str, i: &str) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Index { object, indices } => {
            if indices.len() == 1 {
                if let (Symbol(o), Some(Symbol(ix))) = (object.as_ref(), &indices[0]) {
                    if o.as_ref() == v && ix.as_ref() == i { return Symbol(o.clone()); }
                }
            }
            Index { object: Box::new(subst_vi(object, v, i)), indices: indices.clone() }
        }
        Binary { op, lhs, rhs } => Binary { op: *op, lhs: Box::new(subst_vi(lhs, v, i)), rhs: Box::new(subst_vi(rhs, v, i)) },
        Unary { op, expr } => Unary { op: *op, expr: Box::new(subst_vi(expr, v, i)) },
        Call { func, args } => Call {
            func: func.clone(),
            args: args.iter().map(|a| r2_types::CallArg { name: a.name.clone(), value: subst_vi(&a.value, v, i) }).collect(),
        },
        other => other.clone(),
    }
}

/// Does `e` mention the bare symbol `name` anywhere?
fn mentions(e: &r2_types::Expr, name: &str) -> bool {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => s.as_ref() == name,
        Binary { lhs, rhs, .. } => mentions(lhs, name) || mentions(rhs, name),
        Unary { expr, .. } => mentions(expr, name),
        Call { func, args } => mentions(func, name) || args.iter().any(|a| mentions(&a.value, name)),
        Index { object, indices } => mentions(object, name) || indices.iter().flatten().any(|x| mentions(x, name)),
        _ => false,
    }
}

/// Does `e` contain the exact indexed load `v[i]`?
fn has_index_vi(e: &r2_types::Expr, v: &str, i: &str) -> bool {
    use r2_types::Expr::*;
    match e {
        Index { object, indices } => {
            (indices.len() == 1 && matches!(object.as_ref(), Symbol(o) if o.as_ref()==v)
                && matches!(&indices[0], Some(Symbol(ix)) if ix.as_ref()==i))
                || has_index_vi(object, v, i)
                || indices.iter().flatten().any(|x| has_index_vi(x, v, i))
        }
        Binary { lhs, rhs, .. } => has_index_vi(lhs, v, i) || has_index_vi(rhs, v, i),
        Unary { expr, .. } => has_index_vi(expr, v, i),
        Call { args, .. } => args.iter().any(|a| has_index_vi(&a.value, v, i)),
        _ => false,
    }
}

/// Phase J.2 — recognize an index-loop fold over a vector param:
/// `function(v){ [n <- length(v);] s <- init; for(i in 1:<len>) s <- s <+/*> f(v[i]); s }`.
/// Returns the per-element map body (with `v[i]` → `v`) and the reduce op, or
/// `None` if the body isn't exactly this shape (→ fall back to interpreter).
pub(crate) fn recognize_index_reduction(body: &r2_types::Expr, v: &str) -> Option<(r2_types::Expr, FusedReduceOp)> {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return None };
    let mut len_var: Option<String> = None;
    let mut acc: Option<String> = None;
    let mut init: Option<f64> = None;
    let mut for_stmt: Option<&r2_types::Expr> = None;
    let mut trailing: Option<String> = None;
    for st in stmts {
        match st {
            Assign { target, value, .. } => {
                let nm = match target.as_ref() { Symbol(n) => n.to_string(), _ => return None };
                match value.as_ref() {
                    Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref()=="length")
                        && args.len()==1 && matches!(&args[0].value, Symbol(a) if a.as_ref()==v) => { len_var = Some(nm); }
                    NumLit(x) => { acc = Some(nm); init = Some(*x); }
                    IntLit(x) => { acc = Some(nm); init = Some(*x as f64); }
                    _ => return None,
                }
            }
            For { .. } => for_stmt = Some(st),
            Symbol(nm) => trailing = Some(nm.to_string()),
            _ => return None,
        }
    }
    let acc = acc?; let init = init?;
    if trailing.as_deref() != Some(acc.as_str()) { return None; }
    let (ivar, iter, fbody) = match for_stmt? { For { var, iter, body } => (var, iter, body), _ => return None };
    let len_expr = match iter.as_ref() {
        Binary { op: r2_types::BinOp::Colon, lhs, rhs }
            if matches!(lhs.as_ref(), NumLit(x) if *x==1.0) || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
        _ => return None,
    };
    let len_ok = match len_expr {
        Symbol(nm) => len_var.as_deref() == Some(nm.as_ref()),
        Call { func, args } => matches!(func.as_ref(), Symbol(f) if f.as_ref()=="length")
            && args.len()==1 && matches!(&args[0].value, Symbol(a) if a.as_ref()==v),
        _ => false,
    };
    if !len_ok { return None; }
    let (t, val) = match fbody.as_ref() { Assign { target, value, .. } => (target, value), _ => return None };
    if !matches!(t.as_ref(), Symbol(nm) if nm.as_ref()==acc) { return None; }
    let (op, contrib) = match val.as_ref() {
        Binary { op, lhs, rhs } => {
            if matches!(lhs.as_ref(), Symbol(nm) if nm.as_ref()==acc) { (*op, rhs.as_ref()) }
            else if matches!(rhs.as_ref(), Symbol(nm) if nm.as_ref()==acc) { (*op, lhs.as_ref()) }
            else { return None; }
        }
        _ => return None,
    };
    let reduce_op = match op {
        r2_types::BinOp::Add if init == 0.0 => FusedReduceOp::Sum,
        r2_types::BinOp::Mul if init == 1.0 => FusedReduceOp::Prod,
        _ => return None,
    };
    if !has_index_vi(contrib, v, ivar.as_ref()) { return None; } // must actually fold v
    let mapped = subst_vi(contrib, v, ivar.as_ref());
    // Reject if the element body still references the loop index or accumulator.
    if mentions(&mapped, ivar.as_ref()) || mentions(&mapped, &acc) { return None; }
    Some((mapped, reduce_op))
}

/// Phase J.2 brick 2 — recognize an index-loop *map* over a vector param:
/// `function(x){ [n<-length(x);] y <- <alloc>; for(i in 1:len) y[i] <- f(x[i]); y }`.
/// Returns the per-element map body (`x[i]` → `x`), or `None` (→ fall back).
pub(crate) fn recognize_index_map(body: &r2_types::Expr, x: &str) -> Option<r2_types::Expr> {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return None };
    let mut len_var: Option<String> = None;
    let mut out_var: Option<String> = None;
    let mut for_stmt: Option<&r2_types::Expr> = None;
    let mut trailing: Option<String> = None;
    for st in stmts {
        match st {
            Assign { target, value, .. } => {
                let nm = match target.as_ref() { Symbol(n) => n.to_string(), _ => return None };
                match value.as_ref() {
                    Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref()=="length")
                        && args.len()==1 && matches!(&args[0].value, Symbol(a) if a.as_ref()==x) => { len_var = Some(nm); }
                    _ => { if out_var.is_some() { return None; } out_var = Some(nm); } // exactly one output alloc
                }
            }
            For { .. } => for_stmt = Some(st),
            Symbol(nm) => trailing = Some(nm.to_string()),
            _ => return None,
        }
    }
    let out = out_var?;
    if trailing.as_deref() != Some(out.as_str()) { return None; }
    let (ivar, iter, fbody) = match for_stmt? { For { var, iter, body } => (var, iter, body), _ => return None };
    let len_expr = match iter.as_ref() {
        Binary { op: r2_types::BinOp::Colon, lhs, rhs }
            if matches!(lhs.as_ref(), NumLit(v) if *v==1.0) || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
        _ => return None,
    };
    let len_ok = match len_expr {
        Symbol(nm) => len_var.as_deref() == Some(nm.as_ref()),
        Call { func, args } => matches!(func.as_ref(), Symbol(f) if f.as_ref()=="length")
            && args.len()==1 && matches!(&args[0].value, Symbol(a) if a.as_ref()==x),
        _ => false,
    };
    if !len_ok { return None; }
    // Loop body must be `y[i] <- f(x[i])`.
    let (t, val) = match fbody.as_ref() { Assign { target, value, .. } => (target, value), _ => return None };
    match t.as_ref() {
        Index { object, indices } if indices.len()==1
            && matches!(object.as_ref(), Symbol(o) if o.as_ref()==out)
            && matches!(&indices[0], Some(Symbol(ix)) if ix.as_ref()==ivar.as_ref()) => {}
        _ => return None,
    }
    if !has_index_vi(val, x, ivar.as_ref()) { return None; }
    let mapped = subst_vi(val, x, ivar.as_ref());
    if mentions(&mapped, ivar.as_ref()) || mentions(&mapped, &out) { return None; }
    if let Some(lv) = &len_var { if mentions(&mapped, lv) { return None; } }
    Some(mapped)
}

/// Phase J.3 — is `e` built entirely from constructs the indexed-load codegen
/// lowers faithfully? Like `body_is_jit_lowerable`, but additionally admits
/// `v[ivar]` where `v` is one of the vector params `vecs` and the index is
/// *exactly* the loop induction variable `ivar` (guaranteeing an in-bounds
/// load over `1:length(v)` — no bounds check needed). Indexed *stores* (an
/// `Index` assignment target) are rejected: this brick is load-only, scalar
/// return.
fn body_is_indexed_lowerable(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>], ivar: &str) -> bool {
    use r2_types::Expr::*;
    let is_vec = |o: &r2_types::Expr| matches!(o, Symbol(s) if vecs.iter().any(|v| v.as_ref() == s.as_ref()));
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | NaLit | NullLit | Symbol(_) => true,
        Unary { expr, .. } => body_is_indexed_lowerable(expr, vecs, ivar),
        Binary { lhs, rhs, .. } => body_is_indexed_lowerable(lhs, vecs, ivar) && body_is_indexed_lowerable(rhs, vecs, ivar),
        // Assign only to a scalar symbol (accumulator/temp); no indexed stores.
        Assign { target, value, .. } => matches!(target.as_ref(), Symbol(_))
            && body_is_indexed_lowerable(value, vecs, ivar),
        Call { func, args } => body_is_indexed_lowerable(func, vecs, ivar)
            && args.iter().all(|a| body_is_indexed_lowerable(&a.value, vecs, ivar)),
        // Require an explicit `else`: an if-without-else that assigns the
        // accumulator lowers its missing branch to Null(=0.0), which would
        // silently zero the accumulator when the condition is false. The
        // interpreter handles such bodies correctly, so decline (→ fallback).
        If { cond, then, else_ } => else_.is_some()
            && body_is_indexed_lowerable(cond, vecs, ivar)
            && body_is_indexed_lowerable(then, vecs, ivar)
            && else_.as_ref().map_or(true, |e| body_is_indexed_lowerable(e, vecs, ivar)),
        While { cond, body } => body_is_indexed_lowerable(cond, vecs, ivar) && body_is_indexed_lowerable(body, vecs, ivar),
        For { iter, body, .. } => matches!(iter.as_ref(), Binary { op: r2_types::BinOp::Colon, .. })
            && body_is_indexed_lowerable(iter, vecs, ivar) && body_is_indexed_lowerable(body, vecs, ivar),
        Block(stmts) => stmts.iter().all(|s| body_is_indexed_lowerable(s, vecs, ivar)),
        Return(v) => body_is_indexed_lowerable(v, vecs, ivar),
        Pipe { lhs, rhs } => body_is_indexed_lowerable(lhs, vecs, ivar) && body_is_indexed_lowerable(rhs, vecs, ivar),
        // The one new admission: v[ivar] on a vector param.
        Index { object, indices } => indices.len() == 1
            && is_vec(object.as_ref())
            && matches!(&indices[0], Some(Symbol(ix)) if ix.as_ref() == ivar),
        _ => false,
    }
}

/// Replace every `length(v)` (v ∈ `vecs`) with `Symbol(newsym)`. Used to turn
/// the loop bound / any `length` use into a reference to the fused `len` param.
fn rewrite_length(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>], newsym: &std::sync::Arc<str>) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "length")
            && args.len() == 1
            && matches!(&args[0].value, Symbol(a) if vecs.iter().any(|v| v.as_ref() == a.as_ref())) => {
            Symbol(newsym.clone())
        }
        Binary { op, lhs, rhs } => Binary { op: *op,
            lhs: Box::new(rewrite_length(lhs, vecs, newsym)), rhs: Box::new(rewrite_length(rhs, vecs, newsym)) },
        Unary { op, expr } => Unary { op: *op, expr: Box::new(rewrite_length(expr, vecs, newsym)) },
        Assign { target, value, superassign } => Assign { target: target.clone(),
            value: Box::new(rewrite_length(value, vecs, newsym)), superassign: *superassign },
        Call { func, args } => Call { func: func.clone(),
            args: args.iter().map(|a| r2_types::CallArg { name: a.name.clone(), value: rewrite_length(&a.value, vecs, newsym) }).collect() },
        If { cond, then, else_ } => If { cond: Box::new(rewrite_length(cond, vecs, newsym)),
            then: Box::new(rewrite_length(then, vecs, newsym)),
            else_: else_.as_ref().map(|e| Box::new(rewrite_length(e, vecs, newsym))) },
        While { cond, body } => While { cond: Box::new(rewrite_length(cond, vecs, newsym)), body: Box::new(rewrite_length(body, vecs, newsym)) },
        For { var, iter, body } => For { var: var.clone(),
            iter: Box::new(rewrite_length(iter, vecs, newsym)), body: Box::new(rewrite_length(body, vecs, newsym)) },
        Block(stmts) => Block(stmts.iter().map(|s| rewrite_length(s, vecs, newsym)).collect()),
        Return(v) => Return(Box::new(rewrite_length(v, vecs, newsym))),
        Index { object, indices } => Index {
            object: Box::new(rewrite_length(object, vecs, newsym)),
            indices: indices.iter().map(|ix| ix.as_ref().map(|x| rewrite_length(x, vecs, newsym))).collect() },
        other => other.clone(),
    }
}

/// Walk `e`, collecting a reference to every `For` node. Used to require the
/// body contains exactly one counted loop (so the induction var is unambiguous).
fn collect_fors<'a>(e: &'a r2_types::Expr, out: &mut Vec<&'a r2_types::Expr>) {
    use r2_types::Expr::*;
    match e {
        For { body, iter, .. } => { out.push(e); collect_fors(iter, out); collect_fors(body, out); }
        Binary { lhs, rhs, .. } => { collect_fors(lhs, out); collect_fors(rhs, out); }
        Unary { expr, .. } => collect_fors(expr, out),
        Assign { value, .. } => collect_fors(value, out),
        Call { args, .. } => for a in args { collect_fors(&a.value, out); },
        If { cond, then, else_ } => { collect_fors(cond, out); collect_fors(then, out); if let Some(x) = else_ { collect_fors(x, out); } }
        While { cond, body } => { collect_fors(cond, out); collect_fors(body, out); }
        Block(s) => for x in s { collect_fors(x, out); },
        Return(v) => collect_fors(v, out),
        Pipe { lhs, rhs } => { collect_fors(lhs, out); collect_fors(rhs, out); }
        _ => {}
    }
}

/// Does `e` contain any `v[..]` index on one of `vecs`?
fn has_any_vec_index(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>]) -> bool {
    use r2_types::Expr::*;
    match e {
        Index { object, indices } => matches!(object.as_ref(), Symbol(o) if vecs.iter().any(|v| v.as_ref()==o.as_ref()))
            || has_any_vec_index(object, vecs) || indices.iter().flatten().any(|x| has_any_vec_index(x, vecs)),
        Binary { lhs, rhs, .. } => has_any_vec_index(lhs, vecs) || has_any_vec_index(rhs, vecs),
        Unary { expr, .. } => has_any_vec_index(expr, vecs),
        Assign { value, .. } => has_any_vec_index(value, vecs),
        Call { args, .. } => args.iter().any(|a| has_any_vec_index(&a.value, vecs)),
        If { cond, then, else_ } => has_any_vec_index(cond, vecs) || has_any_vec_index(then, vecs) || else_.as_ref().map_or(false,|e| has_any_vec_index(e, vecs)),
        While { cond, body } => has_any_vec_index(cond, vecs) || has_any_vec_index(body, vecs),
        For { body, .. } => has_any_vec_index(body, vecs),
        Block(s) => s.iter().any(|x| has_any_vec_index(x, vecs)),
        Return(v) => has_any_vec_index(v, vecs),
        Pipe { lhs, rhs } => has_any_vec_index(lhs, vecs) || has_any_vec_index(rhs, vecs),
        _ => false,
    }
}

/// Phase J.3 — recognize a general scalar-returning loop with real indexed
/// loads over the (1 or 2) vector params:
///   `function(x[, w]) { <scalar inits>; for(i in 1:length(x)) <body with x[i]/w[i]>; result }`
/// Unlike the fold/map recognizers this admits arbitrary indexed-lowerable loop
/// bodies (multi-statement, conditionals, scalar recurrences) as long as every
/// `v[i]` uses the bare loop var (in-bounds) and the loop bound is `length` of a
/// param. Returns the length-rewritten body + the vector param names in order.
pub(crate) fn recognize_indexed_scalar_loop(
    body: &r2_types::Expr,
    params: &[std::sync::Arc<str>],
) -> Option<(r2_types::Expr, Vec<std::sync::Arc<str>>)> {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return None };
    // Must end in a bare symbol (the scalar result) so the compiled function
    // returns a value, not the loop's NULL.
    match stmts.last() { Some(Symbol(_)) => {}, _ => return None }

    let vecs: Vec<std::sync::Arc<str>> = params.to_vec();
    if !has_any_vec_index(body, &vecs) { return None; } // must actually index a vector

    // Exactly one counted loop → unambiguous induction variable.
    let mut fors = Vec::new();
    collect_fors(body, &mut fors);
    if fors.len() != 1 { return None; }
    let (ivar, iter) = match fors[0] { For { var, iter, .. } => (var.clone(), iter), _ => return None };

    // Loop must be `1:<len>` where <len> is `length(vecparam)` or a symbol
    // assigned `length(vecparam)` among the leading statements.
    let len_expr = match iter.as_ref() {
        Binary { op: r2_types::BinOp::Colon, lhs, rhs }
            if matches!(lhs.as_ref(), NumLit(x) if *x == 1.0) || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
        _ => return None,
    };
    let is_len_of_vec = |e: &r2_types::Expr| matches!(e, Call { func, args }
        if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "length")
            && args.len() == 1
            && matches!(&args[0].value, Symbol(a) if vecs.iter().any(|v| v.as_ref() == a.as_ref())));
    let len_ok = match len_expr {
        e if is_len_of_vec(e) => true,
        Symbol(nm) => stmts.iter().any(|s| matches!(s,
            Assign { target, value, .. }
                if matches!(target.as_ref(), Symbol(t) if t.as_ref() == nm.as_ref())
                    && is_len_of_vec(value))),
        _ => false,
    };
    if !len_ok { return None; }

    // Rewrite length(vec) → a synthetic scalar `len` param, then validate the
    // whole body is faithfully lowerable with in-bounds `v[ivar]` loads only.
    let len_sym: std::sync::Arc<str> = std::sync::Arc::from(".__ixloop_n");
    let rewritten = rewrite_length(body, &vecs, &len_sym);
    if !body_is_indexed_lowerable(&rewritten, &vecs, ivar.as_ref()) { return None; }
    Some((rewritten, vecs))
}

/// Phase J.3 — is `e` a faithfully-lowerable indexed-**store** loop body?
/// Admits: reads `in_vec[ivar]`, stores `out[ivar] <- value` (both with the
/// bare loop var → in-bounds), scalar-symbol temporaries, arithmetic, math
/// calls, and `if`/`else` *as a value*. Rejects reads of `out` (no recurrence),
/// nested loops, and `if` without `else`.
fn store_body_ok(e: &r2_types::Expr, in_vecs: &[std::sync::Arc<str>], out: &str, ivar: &str) -> bool {
    use r2_types::Expr::*;
    let is_bare_index = |o: &r2_types::Expr, indices: &[Option<r2_types::Expr>], name: &str| {
        indices.len() == 1
            && matches!(o, Symbol(s) if s.as_ref() == name)
            && matches!(&indices[0], Some(Symbol(ix)) if ix.as_ref() == ivar)
    };
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | NaLit | NullLit | Symbol(_) => true,
        Unary { expr, .. } => store_body_ok(expr, in_vecs, out, ivar),
        Binary { lhs, rhs, .. } => store_body_ok(lhs, in_vecs, out, ivar) && store_body_ok(rhs, in_vecs, out, ivar),
        Assign { target, value, .. } => {
            let target_ok = match target.as_ref() {
                Symbol(_) => true, // scalar temp
                Index { object, indices } => is_bare_index(object, indices, out), // out[i] store
                _ => false,
            };
            target_ok && store_body_ok(value, in_vecs, out, ivar)
        }
        Call { func, args } => store_body_ok(func, in_vecs, out, ivar)
            && args.iter().all(|a| store_body_ok(&a.value, in_vecs, out, ivar)),
        If { cond, then, else_ } => else_.is_some()
            && store_body_ok(cond, in_vecs, out, ivar)
            && store_body_ok(then, in_vecs, out, ivar)
            && else_.as_ref().map_or(true, |x| store_body_ok(x, in_vecs, out, ivar)),
        Block(stmts) => stmts.iter().all(|s| store_body_ok(s, in_vecs, out, ivar)),
        // A read: only an input vector at the bare loop var (never `out`).
        Index { object, indices } => in_vecs.iter().any(|v| is_bare_index(object, indices, v)),
        _ => false, // no For/While/Repeat/Match/etc. inside the loop body
    }
}

/// Does `e` contain a store `out[ivar] <- …`?
fn has_store_to(e: &r2_types::Expr, out: &str, ivar: &str) -> bool {
    use r2_types::Expr::*;
    match e {
        Assign { target, value, .. } => {
            let hit = matches!(target.as_ref(), Index { object, indices }
                if indices.len() == 1
                    && matches!(object.as_ref(), Symbol(o) if o.as_ref() == out)
                    && matches!(&indices[0], Some(Symbol(ix)) if ix.as_ref() == ivar));
            hit || has_store_to(value, out, ivar)
        }
        Binary { lhs, rhs, .. } => has_store_to(lhs, out, ivar) || has_store_to(rhs, out, ivar),
        Unary { expr, .. } => has_store_to(expr, out, ivar),
        Call { args, .. } => args.iter().any(|a| has_store_to(&a.value, out, ivar)),
        If { cond, then, else_ } => has_store_to(cond, out, ivar) || has_store_to(then, out, ivar)
            || else_.as_ref().map_or(false, |x| has_store_to(x, out, ivar)),
        Block(s) => s.iter().any(|x| has_store_to(x, out, ivar)),
        _ => false,
    }
}

/// Phase J.3 — recognize a general indexed-**store** map over 1-2 input vectors:
///   `function(x[, w]) { [n <- length(x);] y <- <alloc>; for(i in 1:length(x)) <body storing y[i]>; y }`
/// The loop body may be multi-statement with scalar temporaries and reads of
/// `x[i]`/`w[i]` (bare loop var). Returns (rewritten IR body = the loop only, with
/// `length(v)`→ the len param), the input vector names, and the output var name.
pub(crate) fn recognize_indexed_store_map(
    body: &r2_types::Expr,
    params: &[std::sync::Arc<str>],
) -> Option<(r2_types::Expr, Vec<std::sync::Arc<str>>, std::sync::Arc<str>)> {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return None };
    let out = match stmts.last() { Some(Symbol(o)) => o.clone(), _ => return None };
    let in_vecs: Vec<std::sync::Arc<str>> = params.to_vec();
    if in_vecs.iter().any(|v| v.as_ref() == out.as_ref()) { return None; } // out must be a fresh local

    let is_len_of_vec = |e: &r2_types::Expr| matches!(e, Call { func, args }
        if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "length")
            && args.len() == 1
            && matches!(&args[0].value, Symbol(a) if in_vecs.iter().any(|v| v.as_ref() == a.as_ref())));

    // Partition the top-level statements: length-aliases, the single output
    // alloc, the (single) For, and the trailing `out`. Anything else → bail.
    let mut len_aliases: Vec<std::sync::Arc<str>> = Vec::new();
    let mut saw_alloc = false;
    let mut for_stmt: Option<&r2_types::Expr> = None;
    for (k, st) in stmts.iter().enumerate() {
        if k == stmts.len() - 1 { break; } // trailing Symbol(out), already captured
        match st {
            Assign { target, value, .. } => {
                let nm = match target.as_ref() { Symbol(n) => n.clone(), _ => return None };
                if is_len_of_vec(value) { len_aliases.push(nm); }
                else if nm.as_ref() == out.as_ref() { saw_alloc = true; }
                else { return None; } // unexpected pre-loop statement
            }
            For { .. } => { if for_stmt.is_some() { return None; } for_stmt = Some(st); }
            _ => return None,
        }
    }
    if !saw_alloc { return None; }
    let (ivar, iter, fbody) = match for_stmt? { For { var, iter, body } => (var.clone(), iter, body), _ => return None };

    // Loop must be `1:<len>` with <len> = length(invec) or a length-alias.
    let len_expr = match iter.as_ref() {
        Binary { op: r2_types::BinOp::Colon, lhs, rhs }
            if matches!(lhs.as_ref(), NumLit(x) if *x == 1.0) || matches!(lhs.as_ref(), IntLit(1)) => rhs.as_ref(),
        _ => return None,
    };
    let len_ok = match len_expr {
        e if is_len_of_vec(e) => true,
        Symbol(nm) => len_aliases.iter().any(|a| a.as_ref() == nm.as_ref()),
        _ => false,
    };
    if !len_ok { return None; }

    // Validate the loop body and require it actually stores out[ivar].
    if !store_body_ok(fbody, &in_vecs, out.as_ref(), ivar.as_ref()) { return None; }
    if !has_store_to(fbody, out.as_ref(), ivar.as_ref()) { return None; }

    // Rewritten IR body = the pre-loop length-aliases + the For (drop alloc &
    // trailing return); length(vec) → the synthetic len param.
    let len_sym: std::sync::Arc<str> = std::sync::Arc::from(".__ixloop_n");
    let mut kept: Vec<r2_types::Expr> = Vec::new();
    for (k, st) in stmts.iter().enumerate() {
        if k == stmts.len() - 1 { break; }
        match st {
            Assign { target, value, .. }
                if matches!(target.as_ref(), Symbol(n) if n.as_ref() == out.as_ref()) && !is_len_of_vec(value) => {}
            other => kept.push(rewrite_length(other, &in_vecs, &len_sym)),
        }
    }
    Some((Block(kept), in_vecs, out))
}

/// J.5 groundwork / `explain()` — the FIRST construct in `e` that keeps it out
/// of the JIT, as a human-readable reason, or `None` if fully lowerable. Mirrors
/// `body_is_jit_lowerable` but reports *why* instead of a bare bool.
pub fn jit_reject_reason(e: &r2_types::Expr) -> Option<String> {
    use r2_types::Expr::*;
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | NaLit | NullLit | Symbol(_) => None,
        Unary { expr, .. } => jit_reject_reason(expr),
        Binary { lhs, rhs, .. } => jit_reject_reason(lhs).or_else(|| jit_reject_reason(rhs)),
        Assign { value, .. } => jit_reject_reason(value),
        Call { func, args } => jit_reject_reason(func).or_else(|| args.iter().find_map(|a| jit_reject_reason(&a.value))),
        If { cond, then, else_ } => jit_reject_reason(cond)
            .or_else(|| jit_reject_reason(then))
            .or_else(|| else_.as_ref().and_then(|e| jit_reject_reason(e))),
        While { cond, body } => jit_reject_reason(cond).or_else(|| jit_reject_reason(body)),
        Block(stmts) => stmts.iter().find_map(jit_reject_reason),
        Return(v) => jit_reject_reason(v),
        Pipe { lhs, rhs } => jit_reject_reason(lhs).or_else(|| jit_reject_reason(rhs)),
        For { iter, body, .. } => {
            if !matches!(iter.as_ref(), Binary { op: r2_types::BinOp::Colon, .. }) {
                return Some("for-loop over a non-range iterable (only counted `for(v in a:b)` JITs)".into());
            }
            jit_reject_reason(iter).or_else(|| jit_reject_reason(body))
        }
        Index { .. } | DblIndex { .. } => Some("vector/list/matrix indexing (`v[i]` / `x[[i]]`) — needs J.2/J.3".into()),
        Dollar { .. } => Some("`$` field access — needs J.3".into()),
        Repeat { .. } => Some("`repeat` loop".into()),
        FuncDef { .. } | Lambda { .. } => Some("a nested function definition".into()),
        Match { .. } => Some("`match`/`switch`".into()),
        TryCatch { .. } => Some("`tryCatch`".into()),
        Break | Next => Some("`break`/`next`".into()),
        StrLit(_) | FStringLit(_) => Some("string values (JIT is numeric)".into()),
        _ => Some("an unsupported construct".into()),
    }
}

/// `explain(f)` backend — report whether closure `f` JIT-compiles (and to which
/// specialization), or the first reason it falls back to the interpreter.
pub fn explain_closure(cl: &r2_types::Closure) -> String {
    if let Some(h) = try_compile_closure(cl) {
        return format!("JIT-compiled → {:?} (native)", h.kind());
    }
    if cl.params.len() > 3 {
        return "interpreter — more than 3 parameters".into();
    }
    if cl.params.iter().any(|p| p.default.is_some() || p.dots) {
        return "interpreter — parameters with defaults or `...`".into();
    }
    let body = &*cl.body;
    if cl.params.len() == 1 {
        if recognize_index_reduction(body, &cl.params[0].name).is_some()
            || recognize_index_map(body, &cl.params[0].name).is_some() {
            return "interpreter — recognized as an index fold/map but that codegen path declined (report this)".into();
        }
    }
    match jit_reject_reason(body) {
        Some(reason) => format!("interpreter — blocked by {reason}"),
        None => "interpreter — body is lowerable but codegen is unsupported here (report this)".into(),
    }
}

/// Phase J.4 — is `e` a *pure single-expression* tree safe to inline by
/// substitution (no local bindings to alpha-rename, no control-flow that binds
/// its own variables)? Literals, symbols, arithmetic, calls, `if`/`else`, and
/// indexing only.
fn is_pure_inlinable(e: &r2_types::Expr) -> bool {
    use r2_types::Expr::*;
    match e {
        NumLit(_) | IntLit(_) | BoolLit(_) | NaLit | NullLit | Symbol(_) => true,
        Unary { expr, .. } => is_pure_inlinable(expr),
        Binary { lhs, rhs, .. } => is_pure_inlinable(lhs) && is_pure_inlinable(rhs),
        Call { func, args } => is_pure_inlinable(func) && args.iter().all(|a| is_pure_inlinable(&a.value)),
        If { cond, then, else_ } => is_pure_inlinable(cond) && is_pure_inlinable(then)
            && else_.as_ref().map_or(true, |x| is_pure_inlinable(x)),
        Index { object, indices } => is_pure_inlinable(object) && indices.iter().flatten().all(is_pure_inlinable),
        Pipe { lhs, rhs } => is_pure_inlinable(lhs) && is_pure_inlinable(rhs),
        _ => false,
    }
}

/// Substitute bare symbols with argument expressions (used to inline a callee
/// body once its params are bound to the caller's argument expressions). Only
/// walks the pure-inlinable node set (guaranteed by `is_pure_inlinable`).
fn substitute_symbols(e: &r2_types::Expr, subst: &std::collections::HashMap<std::sync::Arc<str>, r2_types::Expr>) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => subst.get(s).cloned().unwrap_or_else(|| e.clone()),
        Unary { op, expr } => Unary { op: *op, expr: Box::new(substitute_symbols(expr, subst)) },
        Binary { op, lhs, rhs } => Binary { op: *op, lhs: Box::new(substitute_symbols(lhs, subst)), rhs: Box::new(substitute_symbols(rhs, subst)) },
        Call { func, args } => Call { func: Box::new(substitute_symbols(func, subst)),
            args: args.iter().map(|a| r2_types::CallArg { name: a.name.clone(), value: substitute_symbols(&a.value, subst) }).collect() },
        If { cond, then, else_ } => If { cond: Box::new(substitute_symbols(cond, subst)),
            then: Box::new(substitute_symbols(then, subst)),
            else_: else_.as_ref().map(|x| Box::new(substitute_symbols(x, subst))) },
        Index { object, indices } => Index { object: Box::new(substitute_symbols(object, subst)),
            indices: indices.iter().map(|ix| ix.as_ref().map(|x| substitute_symbols(x, subst))).collect() },
        Pipe { lhs, rhs } => Pipe { lhs: Box::new(substitute_symbols(lhs, subst)), rhs: Box::new(substitute_symbols(rhs, subst)) },
        other => other.clone(),
    }
}

/// Phase J.4 — inline calls to pure JIT-lowerable user closures found in `env`.
/// `function(a,b) sq(a) + sq(b)` with `sq <- function(x) x*x` becomes
/// `a*a + b*b`, so the composed function JITs as one unit instead of bailing on
/// the user-function `Call`. Depth-bounded so (mutual) recursion terminates with
/// a residual `Call` that safely falls back to the interpreter.
fn inline_user_calls(e: &r2_types::Expr, env: &r2_types::EnvRef, depth: u32) -> r2_types::Expr {
    use r2_types::Expr::*;
    if depth == 0 { return e.clone(); }
    match e {
        Call { func, args } => {
            let new_args: Vec<r2_types::CallArg> = args.iter()
                .map(|a| r2_types::CallArg { name: a.name.clone(), value: inline_user_calls(&a.value, env, depth) })
                .collect();
            if let Symbol(f) = func.as_ref() {
                if new_args.iter().all(|a| a.name.is_none()) {
                    if let Some(r2_types::RVal::Closure(cl2)) = env.lookup(f) {
                        if cl2.params.len() == new_args.len()
                            && cl2.params.iter().all(|p| p.default.is_none() && !p.dots)
                            && is_pure_inlinable(&cl2.body)
                        {
                            let mut subst = std::collections::HashMap::new();
                            for (p, a) in cl2.params.iter().zip(new_args.iter()) {
                                subst.insert(p.name.clone(), a.value.clone());
                            }
                            let body_sub = substitute_symbols(&cl2.body, &subst);
                            return inline_user_calls(&body_sub, env, depth - 1);
                        }
                    }
                }
            }
            Call { func: func.clone(), args: new_args }
        }
        Unary { op, expr } => Unary { op: *op, expr: Box::new(inline_user_calls(expr, env, depth)) },
        Binary { op, lhs, rhs } => Binary { op: *op, lhs: Box::new(inline_user_calls(lhs, env, depth)), rhs: Box::new(inline_user_calls(rhs, env, depth)) },
        If { cond, then, else_ } => If { cond: Box::new(inline_user_calls(cond, env, depth)),
            then: Box::new(inline_user_calls(then, env, depth)),
            else_: else_.as_ref().map(|x| Box::new(inline_user_calls(x, env, depth))) },
        While { cond, body } => While { cond: Box::new(inline_user_calls(cond, env, depth)), body: Box::new(inline_user_calls(body, env, depth)) },
        For { var, iter, body } => For { var: var.clone(), iter: Box::new(inline_user_calls(iter, env, depth)), body: Box::new(inline_user_calls(body, env, depth)) },
        Block(s) => Block(s.iter().map(|x| inline_user_calls(x, env, depth)).collect()),
        Assign { target, value, superassign } => Assign { target: target.clone(), value: Box::new(inline_user_calls(value, env, depth)), superassign: *superassign },
        Return(v) => Return(Box::new(inline_user_calls(v, env, depth))),
        Pipe { lhs, rhs } => Pipe { lhs: Box::new(inline_user_calls(lhs, env, depth)), rhs: Box::new(inline_user_calls(rhs, env, depth)) },
        Index { object, indices } => Index { object: Box::new(inline_user_calls(object, env, depth)),
            indices: indices.iter().map(|ix| ix.as_ref().map(|x| inline_user_calls(x, env, depth))).collect() },
        other => other.clone(),
    }
}

pub fn try_compile_closure(cl: &r2_types::Closure) -> Option<std::sync::Arc<dyn r2_types::JitHandle>> {
    // Phase R.M — gate the JIT on supported architectures. On aarch64 the
    // engine falls back to the interpreter; statistical outputs are
    // bit-identical, only wall-clock performance differs.
    if !JIT_SUPPORTED { return None; }

    // Filter out anything we definitely can't handle.
    // Phase C.5 admits 3-param closures for the ternary vector-map path.
    if cl.params.len() > 3 { return None; }
    if cl.params.iter().any(|p| p.default.is_some() || p.dots) { return None; }

    // ── Phase B.1: closure capture inference via partial evaluation ─
    //
    // Free variables in the body are resolved against `cl.env` at JIT
    // compile time. Numeric scalars get substituted as `Expr::NumLit`
    // constants directly in the body AST before IR lowering. The closure
    // becomes self-contained from the JIT's perspective — no new ABI
    // surface, no per-call capture passing.
    //
    // Limitations: only numeric scalars (Real/Int/Bool of length 1) get
    // baked in. Vector-valued captures and other types fall through —
    // the body still references them and the lowering rejects (closure
    // stays interpreter-only).
    //
    // **Correctness window**: this assumes captured values are stable
    // for the lifetime of the closure. R semantics agree — captures
    // are by-value at creation time. If R2 ever adds reactive/observable
    // values, this substitution will need to be invalidated on capture
    // mutation; we'll cross that bridge when it appears.
    // Phase J.4 brick 1 — inline calls to pure JIT-lowerable user helpers first,
    // so a function composed of small numeric helpers compiles as one unit. A
    // body with no such calls is returned structurally unchanged (a clone), so
    // existing paths are unaffected. Depth-bounded → recursion falls back safely.
    let inlined_body = inline_user_calls(cl.body.as_ref(), &cl.env, 8);

    let param_names: Vec<std::sync::Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
    let free_vars = r2_ir::collect_free_vars(&inlined_body, &param_names);
    let body_expr: r2_types::Expr;
    let body_ref: &r2_types::Expr;
    if !free_vars.is_empty() {
        let mut subs: std::collections::HashMap<std::sync::Arc<str>, f64> =
            std::collections::HashMap::new();
        for name in &free_vars {
            if let Some(val) = cl.env.lookup(name) {
                if let Some(scalar) = scalar_f64_of(&val) {
                    subs.insert(name.clone(), scalar);
                }
            }
        }
        if !subs.is_empty() {
            body_expr = r2_ir::substitute_constants(&inlined_body, &subs);
            body_ref = &body_expr;
        } else {
            body_ref = &inlined_body;
        }
    } else {
        body_ref = &inlined_body;
    }

    // Phase J.2 — index-loop fold over a vector param:
    //   function(v){ [n<-length(v);] s<-init; for(i in 1:len) s<-s <+/*> f(v[i]); s }
    // Recognized as a map-reduce over v (v[i] → element), reusing the tested
    // fused map-reduce codegen — no new indexed-load codegen. Runs BEFORE the
    // allowlist gate, which would otherwise reject the `Index` in the body.
    if cl.params.len() == 1 {
        if let Some((mapped, reduce_op)) = recognize_index_reduction(body_ref, &cl.params[0].name) {
            if body_is_jit_lowerable(&mapped) {
                let params = vec![(cl.params[0].name.clone(), r2_types::infer::IrType::scalar(IrElem::Real))];
                let mut inner_ir = r2_ir::lower_function("__map_reduce_inner__", params, &mapped);
                inner_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
                if let Ok(c) = JitCompiler::compile_vector_map_reduce(&inner_ir, reduce_op) {
                    return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                }
            }
        }
        // Brick 2: index-loop map `for(i in 1:len) y[i] <- f(x[i]); y` → VectorMap.
        if let Some(mapped) = recognize_index_map(body_ref, &cl.params[0].name) {
            if body_is_jit_lowerable(&mapped) {
                let params = vec![(cl.params[0].name.clone(), r2_types::infer::IrType::scalar(IrElem::Real))];
                let mut inner_ir = r2_ir::lower_function("__index_map_inner__", params, &mapped);
                inner_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
                if let Ok(c) = JitCompiler::compile_vector_map_generic(&inner_ir) {
                    return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                }
            }
        }
    }

    // Phase J.3 — general scalar-returning loop with real indexed loads over
    // 1-2 vector params (multi-statement folds, conditionals, scalar
    // recurrences reading x[i]/w[i]). Compiles the *actual* loop via `Load`
    // codegen — not a recognised map/reduce shape. Runs before the allowlist
    // gate (which rejects `Index`), and after the specialised fold/map
    // recognisers so those keep precedence for the shapes they cover.
    if cl.params.len() == 1 || cl.params.len() == 2 {
        let pnames: Vec<std::sync::Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
        if let Some((rewritten, vecs)) = recognize_indexed_scalar_loop(body_ref, &pnames) {
            let mut params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = vecs.iter()
                .map(|v| (v.clone(), r2_types::infer::IrType::vector(IrElem::Real, None)))
                .collect();
            params.push((std::sync::Arc::from(".__ixloop_n"), r2_types::infer::IrType::scalar(IrElem::Real)));
            let mut ir = r2_ir::lower_function("__indexed_scalar_loop__", params, &rewritten);
            ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
            if let Ok(c) = JitCompiler::compile_indexed_reduction(&ir) {
                return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
            }
        }
    }

    // Phase J.3 — general indexed-STORE map (1-2 input vectors → 1 output),
    // e.g. two-input `for(i in 1:length(x)) y[i] <- x[i]+w[i]; y` or a
    // multi-statement store body. Runs after the simpler `recognize_index_map`
    // (which handles the single-input `y[i] <- f(x[i])` shape via VectorMap).
    if cl.params.len() == 1 || cl.params.len() == 2 {
        let pnames: Vec<std::sync::Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
        if let Some((rewritten, in_vecs, out)) = recognize_indexed_store_map(body_ref, &pnames) {
            // Params: input vectors, then the output vector, then the len scalar.
            let mut params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = in_vecs.iter()
                .map(|v| (v.clone(), r2_types::infer::IrType::vector(IrElem::Real, None)))
                .collect();
            params.push((out, r2_types::infer::IrType::vector(IrElem::Real, None)));
            params.push((std::sync::Arc::from(".__ixloop_n"), r2_types::infer::IrType::scalar(IrElem::Real)));
            let mut ir = r2_ir::lower_function("__indexed_store_map__", params, &rewritten);
            ir.return_type = r2_types::infer::IrType::null();
            if let Ok(c) = JitCompiler::compile_indexed_store_map(&ir) {
                return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
            }
        }
    }

    // Eligibility gate: bail (→ interpreter) if the body contains any
    // construct the IR lowering silently drops (for/repeat/match/...).
    // Without this, e.g. `function(n){ s<-0; for(k in 1:n) s<-s+k; s }`
    // would JIT-compile with the loop elided and return `s`'s init value.
    if !body_is_jit_lowerable(body_ref) { return None; }

    // Phase C.3 — vector reduction pattern: `function(v) sum(v)` etc.
    if cl.params.len() == 1 {
        if let r2_types::Expr::Call { func, args } = body_ref {
            if let r2_types::Expr::Symbol(fname) = func.as_ref() {
                let supported = matches!(fname.as_ref(), "sum" | "mean" | "length" | "prod");
                if supported && args.len() == 1 {
                    if let r2_types::Expr::Symbol(arg_sym) = &args[0].value {
                        if arg_sym == &cl.params[0].name {
                            if let Ok(c) = JitCompiler::compile_vector_reduction(fname.as_ref()) {
                                return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                            }
                        }
                    }
                    // ── Phase C.9 — fused map-reduce ──
                    // Body is `sum(inner_expr)` / `prod(inner_expr)` where
                    // `inner_expr` is a function of the closure param.
                    // Compile a fused loop: load x[i], apply inner_expr,
                    // accumulate. No intermediate vector allocated.
                    if matches!(fname.as_ref(), "sum" | "prod") {
                        let reduce_op = match fname.as_ref() {
                            "sum"  => FusedReduceOp::Sum,
                            "prod" => FusedReduceOp::Prod,
                            _ => unreachable!(),
                        };
                        let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> =
                            cl.params.iter()
                                .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
                                .collect();
                        let mut inner_ir = r2_ir::lower_function(
                            "__map_reduce_inner__",
                            params,
                            &args[0].value,
                        );
                        inner_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
                        if let Ok(c) = JitCompiler::compile_vector_map_reduce(&inner_ir, reduce_op) {
                            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                        }
                    }
                }
            }
        }
    }

    // Phase J.2 — binary map-reduce: function(x, w) sum(f(x,w)) / prod(...).
    // e.g. dot product `function(x, w) sum(x*w)`. The inner `f` is a per-element
    // function of both vectors → fused (a[i], b[i]) → accumulate loop.
    if cl.params.len() == 2 {
        if let r2_types::Expr::Call { func, args } = body_ref {
            if let r2_types::Expr::Symbol(fname) = func.as_ref() {
                if matches!(fname.as_ref(), "sum" | "prod") && args.len() == 1
                    && body_is_jit_lowerable(&args[0].value) {
                    let reduce_op = if fname.as_ref() == "sum" { FusedReduceOp::Sum } else { FusedReduceOp::Prod };
                    let params = vec![
                        (cl.params[0].name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)),
                        (cl.params[1].name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)),
                    ];
                    let mut inner_ir = r2_ir::lower_function("__binary_map_reduce_inner__", params, &args[0].value);
                    inner_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
                    if let Ok(c) = JitCompiler::compile_vector_binary_map_reduce(&inner_ir, reduce_op) {
                        return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                    }
                }
            }
        }
    }

    // Phase C.4-full — vector ⊗ vector element-wise: function(a, b) a OP b
    if cl.params.len() == 2 {
        if let r2_types::Expr::Binary { op, lhs, rhs } = body_ref {
            if let (r2_types::Expr::Symbol(ls), r2_types::Expr::Symbol(rs)) = (lhs.as_ref(), rhs.as_ref()) {
                if ls == &cl.params[0].name && rs == &cl.params[1].name {
                    if let Ok(c) = JitCompiler::compile_vector_binary_op(*op) {
                        return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                    }
                }
            }
        }
    }

    // Phase C.7 — generic 2-param vector map for any body that lowers to
    // arithmetic + math Calls + branches. Catches `function(a, b) sqrt(a*a + b*b)`,
    // `function(x, y) if (x > y) x else y`, etc. Tried before the
    // simpler `function(a, b) a OP b` path falls through to the scalar fallback.
    if cl.params.len() == 2 {
        let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = cl.params.iter()
            .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
            .collect();
        let mut body_ir = r2_ir::lower_function("__vec_binary_body__", params, body_ref);
        body_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
        if let Ok(c) = JitCompiler::compile_vector_binary_map_generic(&body_ir) {
            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
        }
    }

    // Phase C.4 — element-wise vector map with scalar literal:
    //   function(v) v OP literal     OR     function(v) literal OP v   (commutative ops)
    if cl.params.len() == 1 {
        if let r2_types::Expr::Binary { op, lhs, rhs } = body_ref {
            let pname = &cl.params[0].name;
            let pat = match (lhs.as_ref(), rhs.as_ref()) {
                (r2_types::Expr::Symbol(s), r2_types::Expr::NumLit(k)) if s == pname => Some((*op, *k)),
                (r2_types::Expr::NumLit(k), r2_types::Expr::Symbol(s)) if s == pname
                    && matches!(op, r2_types::BinOp::Add | r2_types::BinOp::Mul)
                    => Some((*op, *k)),
                _ => None,
            };
            if let Some((op, k)) = pat {
                if let Ok(c) = JitCompiler::compile_vector_map_scalar_op(op, k) {
                    return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
                }
            }
        }
    }

    // Phase C.8 — SIMD f64x2 1-param vector map. Tried before the
    // generic scalar path; if the body is SIMD-clean it produces a
    // tight 2-elements-per-iter loop with native SSE2/NEON instructions.
    // Falls through (Err) to the scalar generic path if not clean.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if cl.params.len() == 1 {
        let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = cl.params.iter()
            .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
            .collect();
        let mut body_ir = r2_ir::lower_function("__vec_simd_body__", params, body_ref);
        body_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
        if let Ok(c) = JitCompiler::compile_vector_simd_map_f64x2(&body_ir) {
            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
        }
    }

    // Phase C.4-full part 2 — generic 1-param vector map for any pure
    // arithmetic body (composed expressions, e.g. `(v+1)*2`, `v*v - 1`).
    if cl.params.len() == 1 {
        let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = cl.params.iter()
            .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
            .collect();
        let mut body_ir = r2_ir::lower_function("__vec_body__", params, body_ref);
        body_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
        if let Ok(c) = JitCompiler::compile_vector_map_generic(&body_ir) {
            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
        }
    }

    // Phase C.5 — generic 3-param branchy ternary vector map.
    // Targets `function(c, a, b) if (c > 0) a else b` and similar shapes
    // where three same-length vectors map to one output via a multi-block body.
    if cl.params.len() == 3 {
        let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = cl.params.iter()
            .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
            .collect();
        let mut body_ir = r2_ir::lower_function("__vec_ternary_body__", params, body_ref);
        body_ir.return_type = r2_types::infer::IrType::scalar(IrElem::Real);
        if let Ok(c) = JitCompiler::compile_vector_ternary_map_generic(&body_ir) {
            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
        }
    }

    // Phase J.4 brick 2 — multi-reduction scalar kernel. Combinations of
    // whole-vector reductions the single-reduction paths can't express:
    // `sum(x*y)/sum(x*x)` (regression coef), `{ m<-mean(x); sum((x-m)^2) }`
    // (variance), covariance, etc. Attempted only when the body actually
    // mentions a reduction (so pure vector maps skip it), after all the
    // single-reduction / vector-map paths, before the scalar fallback.
    if (cl.params.len() == 1 || cl.params.len() == 2) && mentions_reduction(body_ref) {
        let pnames: Vec<std::sync::Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
        // Brick 3: fuse vector-valued intermediates + hoist reductions to scalar
        // locals, giving the canonical Block form the kernel codegen consumes.
        let kbody = normalize_reduction_kernel(body_ref, &pnames);
        if let Ok(c) = JitCompiler::compile_reduction_kernel(&kbody, &pnames) {
            return Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>);
        }
    }

    // Phase C.2 — scalar specialization fallback.
    let params: Vec<(std::sync::Arc<str>, r2_types::infer::IrType)> = cl.params.iter()
        .map(|p| (p.name.clone(), r2_types::infer::IrType::scalar(IrElem::Real)))
        .collect();
    let mut func = r2_ir::lower_function("__jit__", params, body_ref);
    func.return_type = r2_types::infer::IrType::scalar(IrElem::Real);

    match JitCompiler::compile(&func) {
        Ok(c) => Some(std::sync::Arc::new(c) as std::sync::Arc<dyn r2_types::JitHandle>),
        Err(_) => None,
    }
}

/// Phase J.4 brick 3 — inline vector-valued locals into a reduction-kernel body
/// by substitution, so composed formulas like `{ e <- pred-obs; sqrt(mean(e*e)) }`
/// or `{ d <- x-mean(x); sum(d*d) }` compile with the intermediate vector *fused*
/// away (no buffer allocated). A leading `local <- rhs` is a **vector** local iff
/// `rhs` has no top-level reduction and references a vector (param or earlier
/// vector-local); such statements are dropped and their definition substituted
/// into later statements. Scalar locals (those whose rhs reduces to a scalar) are
/// kept. Non-Block bodies pass through unchanged.
/// Phase J.4 brick 3 — hoist every reduction sub-expression (`sum`/`prod`/
/// `mean`/`length`) to a fresh scalar local, so what remains inside each fused
/// loop is a pure element expression over vector params + (now hoisted) scalar
/// locals. `sum((x-mean(x))^2)` → `__hr0 <- mean(x); __hr1 <- sum((x-__hr0)^2); __hr1`.
/// Nested reductions are hoisted innermost-first. Emits assignments into `stmts`.
fn hoist_reductions(e: &r2_types::Expr, stmts: &mut Vec<r2_types::Expr>, ctr: &mut u32) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Call { func, args } => {
            let is_red = matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean" | "length"));
            let new_args: Vec<r2_types::CallArg> = args.iter()
                .map(|a| r2_types::CallArg { name: a.name.clone(), value: hoist_reductions(&a.value, stmts, ctr) })
                .collect();
            let call = Call { func: func.clone(), args: new_args };
            if is_red {
                let name: std::sync::Arc<str> = std::sync::Arc::from(format!(".__hr{}", *ctr));
                *ctr += 1;
                stmts.push(Assign { target: Box::new(Symbol(name.clone())), value: Box::new(call), superassign: false });
                Symbol(name)
            } else {
                call
            }
        }
        Unary { op, expr } => Unary { op: *op, expr: Box::new(hoist_reductions(expr, stmts, ctr)) },
        Binary { op, lhs, rhs } => Binary { op: *op,
            lhs: Box::new(hoist_reductions(lhs, stmts, ctr)), rhs: Box::new(hoist_reductions(rhs, stmts, ctr)) },
        If { cond, then, else_ } => If { cond: Box::new(hoist_reductions(cond, stmts, ctr)),
            then: Box::new(hoist_reductions(then, stmts, ctr)),
            else_: else_.as_ref().map(|x| Box::new(hoist_reductions(x, stmts, ctr))) },
        other => other.clone(),
    }
}

/// Normalize a reduction-kernel body: fuse vector locals, then hoist all
/// reductions to scalar locals → a `Block` of `local <- <reduction|scalar>`
/// followed by a final scalar expression (the shape `compile_reduction_kernel`
/// consumes). Non-Block scalar bodies are handled too.
fn normalize_reduction_kernel(body: &r2_types::Expr, vec_params: &[std::sync::Arc<str>]) -> r2_types::Expr {
    use r2_types::Expr::*;
    let inlined = inline_vector_locals(body, vec_params);
    let mut out: Vec<r2_types::Expr> = Vec::new();
    let mut ctr = 0u32;
    let final_expr = match &inlined {
        Block(ss) => {
            let (last, init) = match ss.split_last() { Some(x) => x, None => return inlined };
            for st in init {
                if let Assign { target, value, superassign } = st {
                    let v = hoist_reductions(value, &mut out, &mut ctr);
                    out.push(Assign { target: target.clone(), value: Box::new(v), superassign: *superassign });
                } else {
                    out.push(st.clone());
                }
            }
            hoist_reductions(last, &mut out, &mut ctr)
        }
        other => hoist_reductions(other, &mut out, &mut ctr),
    };
    out.push(final_expr);
    Block(out)
}

/// Does `e` reference a vector name in a *vector position* — i.e. bare, or under
/// element-wise ops, but NOT enclosed in a reduction (`sum`/`prod`/`mean`/
/// `length`, which collapse a vector to a scalar)? Determines whether a local's
/// rhs evaluates to a vector (fuse it) or a scalar (keep it).
fn refs_vector_bare(e: &r2_types::Expr, vec_names: &[std::sync::Arc<str>]) -> bool {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => vec_names.iter().any(|v| v.as_ref() == s.as_ref()),
        Unary { expr, .. } => refs_vector_bare(expr, vec_names),
        Binary { lhs, rhs, .. } => refs_vector_bare(lhs, vec_names) || refs_vector_bare(rhs, vec_names),
        If { cond, then, else_ } => refs_vector_bare(cond, vec_names) || refs_vector_bare(then, vec_names)
            || else_.as_ref().map_or(false, |x| refs_vector_bare(x, vec_names)),
        Call { func, args } => {
            // A reduction collapses its argument to a scalar → not a vector position.
            if matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean" | "length")) {
                return false;
            }
            args.iter().any(|a| refs_vector_bare(&a.value, vec_names))
        }
        _ => false,
    }
}

fn inline_vector_locals(body: &r2_types::Expr, vec_params: &[std::sync::Arc<str>]) -> r2_types::Expr {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return body.clone() };
    if stmts.len() < 2 { return body.clone(); }
    let (last, init) = stmts.split_last().unwrap();

    let mut vecdefs: std::collections::HashMap<std::sync::Arc<str>, r2_types::Expr> = std::collections::HashMap::new();
    let mut vec_names: Vec<std::sync::Arc<str>> = vec_params.to_vec();
    let mut out: Vec<r2_types::Expr> = Vec::new();

    for st in init {
        if let Assign { target, value, superassign } = st {
            if let Symbol(nm) = target.as_ref() {
                // Inline already-known vector-locals into this rhs first.
                let rhs = substitute_symbols(value, &vecdefs);
                if refs_vector_bare(&rhs, &vec_names) {
                    // Vector local: record its (fully-inlined) definition, drop the stmt.
                    vecdefs.insert(nm.clone(), rhs);
                    vec_names.push(nm.clone());
                    continue;
                }
                // Scalar local: keep, with vector-locals substituted in.
                out.push(Assign { target: target.clone(), value: Box::new(rhs), superassign: *superassign });
                continue;
            }
        }
        out.push(st.clone());
    }
    out.push(substitute_symbols(last, &vecdefs));
    Block(out)
}

/// Does `e` contain a `sum`/`prod`/`mean` reduction call? Cheap gate so the
/// multi-reduction kernel is only attempted on plausibly-scalar bodies.
fn mentions_reduction(e: &r2_types::Expr) -> bool {
    use r2_types::Expr::*;
    match e {
        Call { func, args } => matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean"))
            || args.iter().any(|a| mentions_reduction(&a.value)),
        Binary { lhs, rhs, .. } => mentions_reduction(lhs) || mentions_reduction(rhs),
        Unary { expr, .. } => mentions_reduction(expr),
        Assign { value, .. } => mentions_reduction(value),
        If { cond, then, else_ } => mentions_reduction(cond) || mentions_reduction(then)
            || else_.as_ref().map_or(false, |x| mentions_reduction(x)),
        Block(s) => s.iter().any(mentions_reduction),
        Return(v) => mentions_reduction(v),
        _ => false,
    }
}

