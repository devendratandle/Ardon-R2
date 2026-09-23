//! Loop-pattern recognition for the JIT: the allowlist of constructs the
//! IR lowers faithfully, and the recognizers that turn R's indexed loops
//! (`for (i in 1:n) s <- s + f(v[i])`, `y[i] <- f(x[i])`, …) into the
//! map / reduce / indexed-kernel shapes the code generators compile.

use crate::*;

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
pub(crate) fn subst_vi(e: &r2_types::Expr, v: &str, i: &str) -> r2_types::Expr {
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
pub(crate) fn mentions(e: &r2_types::Expr, name: &str) -> bool {
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
pub(crate) fn has_index_vi(e: &r2_types::Expr, v: &str, i: &str) -> bool {
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
pub(crate) fn body_is_indexed_lowerable(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>], ivar: &str) -> bool {
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
pub(crate) fn rewrite_length(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>], newsym: &std::sync::Arc<str>) -> r2_types::Expr {
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
pub(crate) fn collect_fors<'a>(e: &'a r2_types::Expr, out: &mut Vec<&'a r2_types::Expr>) {
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
pub(crate) fn has_any_vec_index(e: &r2_types::Expr, vecs: &[std::sync::Arc<str>]) -> bool {
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
pub(crate) fn store_body_ok(e: &r2_types::Expr, in_vecs: &[std::sync::Arc<str>], out: &str, ivar: &str) -> bool {
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
pub(crate) fn has_store_to(e: &r2_types::Expr, out: &str, ivar: &str) -> bool {
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
