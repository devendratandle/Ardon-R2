//! Inlining user helper functions into a closure body before it is
//! compiled (Phases J.4 / J.5b), so a function composed of small numeric
//! helpers JITs as one unit instead of bailing on the user-function call.

/// Phase J.4 — is `e` a *pure single-expression* tree safe to inline by
/// substitution (no local bindings to alpha-rename, no control-flow that binds
/// its own variables)? Literals, symbols, arithmetic, calls, `if`/`else`, and
/// indexing only.
pub(crate) fn is_pure_inlinable(e: &r2_types::Expr) -> bool {
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
pub(crate) fn substitute_symbols(e: &r2_types::Expr, subst: &std::collections::HashMap<std::sync::Arc<str>, r2_types::Expr>) -> r2_types::Expr {
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
pub(crate) fn inline_user_calls(e: &r2_types::Expr, env: &r2_types::EnvRef, depth: u32) -> r2_types::Expr {
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

/// Phase J.5b — inline user helpers whose bodies are BLOCKS of local
/// assignments ending in an expression: the standard addon-library style
///   vsd <- function(x) { m <- mean(x); s <- sqrt(...); if (s < eps) 1 else s }
/// `inline_user_calls` only handles single-expression helpers (no locals
/// to rename); this pass alpha-renames the helper's locals with a unique
/// prefix, hoists the renamed assignments BEFORE the statement containing
/// the call, and replaces the call with the substituted final expression.
/// Guards: unnamed pure-inlinable args, params without defaults/dots,
/// every non-final statement a single-assignment `Symbol <- pure-expr`
/// (each local assigned once), pure-inlinable final expr.
pub(crate) fn inline_block_helpers(e: &r2_types::Expr, env: &r2_types::EnvRef, ctr: &mut u32, depth: u32) -> r2_types::Expr {
    use r2_types::Expr::*;
    fn expand(e: &r2_types::Expr, env: &r2_types::EnvRef, pre: &mut Vec<r2_types::Expr>, ctr: &mut u32, depth: u32) -> r2_types::Expr {
        if depth == 0 { return e.clone(); }
        match e {
            Call { func, args } => {
                let new_args: Vec<r2_types::CallArg> = args.iter()
                    .map(|a| r2_types::CallArg { name: a.name.clone(), value: expand(&a.value, env, pre, ctr, depth) })
                    .collect();
                if let Symbol(f) = func.as_ref() {
                    if new_args.iter().all(|a| a.name.is_none() && is_pure_inlinable(&a.value)) {
                        if let Some(r2_types::RVal::Closure(cl2)) = env.lookup(f) {
                            if cl2.params.len() == new_args.len()
                                && cl2.params.iter().all(|p| p.default.is_none() && !p.dots)
                            {
                                if let Block(inner) = cl2.body.as_ref() {
                                    if let Some(expanded) = splice_block(inner, &cl2.params, &new_args, env, pre, ctr, depth) {
                                        return expanded;
                                    }
                                }
                            }
                        }
                    }
                }
                Call { func: func.clone(), args: new_args }
            }
            Unary { op, expr } => Unary { op: *op, expr: Box::new(expand(expr, env, pre, ctr, depth)) },
            Binary { op, lhs, rhs } => Binary { op: *op,
                lhs: Box::new(expand(lhs, env, pre, ctr, depth)), rhs: Box::new(expand(rhs, env, pre, ctr, depth)) },
            If { cond, then, else_ } => If { cond: Box::new(expand(cond, env, pre, ctr, depth)),
                then: Box::new(expand(then, env, pre, ctr, depth)),
                else_: else_.as_ref().map(|x| Box::new(expand(x, env, pre, ctr, depth))) },
            other => other.clone(),
        }
    }
    /// Try to splice a helper body `{a <- ..; b <- ..; final}` at a call
    /// site: renamed assigns are pushed to `pre`, the substituted final
    /// expression is returned. None when the body doesn't fit the shape.
    fn splice_block(inner: &[r2_types::Expr], params: &[r2_types::Param], args: &[r2_types::CallArg],
                    env: &r2_types::EnvRef, pre: &mut Vec<r2_types::Expr>, ctr: &mut u32, depth: u32) -> Option<r2_types::Expr> {
        if inner.len() < 2 { return None; }
        let (last, assigns) = inner.split_last()?;
        // shape check first (no side effects until certain)
        let mut locals: std::collections::HashSet<std::sync::Arc<str>> = Default::default();
        for st in assigns {
            match st {
                Assign { target, value, superassign: false } => match target.as_ref() {
                    Symbol(n) if !locals.contains(n.as_ref()) && is_pure_inlinable(value) => { locals.insert(n.clone()); }
                    _ => return None,
                },
                _ => return None,
            }
        }
        if !is_pure_inlinable(last) { return None; }
        let tag = *ctr; *ctr += 1;
        let mut subst: std::collections::HashMap<std::sync::Arc<str>, r2_types::Expr> = Default::default();
        for (p, a) in params.iter().zip(args.iter()) { subst.insert(p.name.clone(), a.value.clone()); }
        for st in assigns {
            if let Assign { target, value, .. } = st {
                if let Symbol(n) = target.as_ref() {
                    let renamed: std::sync::Arc<str> = std::sync::Arc::from(format!(".__ib{}_{}", tag, n));
                    // Substitute params + earlier locals, then keep expanding
                    // nested helpers inside the hoisted value too (both the
                    // single-expression and block-helper kinds).
                    let v = inline_user_calls(&substitute_symbols(value, &subst), env, 4);
                    let v = expand(&v, env, pre, ctr, depth - 1);
                    pre.push(Assign { target: Box::new(Symbol(renamed.clone())), value: Box::new(v), superassign: false });
                    subst.insert(n.clone(), Symbol(renamed));
                }
            }
        }
        let fin = inline_user_calls(&substitute_symbols(last, &subst), env, 4);
        Some(expand(&fin, env, pre, ctr, depth - 1))
    }
    match e {
        Block(stmts) => {
            let mut out: Vec<r2_types::Expr> = Vec::with_capacity(stmts.len());
            for st in stmts {
                let mut pre: Vec<r2_types::Expr> = Vec::new();
                let new_st = match st {
                    Assign { target, value, superassign } => {
                        let v = expand(value, env, &mut pre, ctr, depth);
                        Assign { target: target.clone(), value: Box::new(v), superassign: *superassign }
                    }
                    For { var, iter, body } => For { var: var.clone(), iter: iter.clone(),
                        body: Box::new(inline_block_helpers(body, env, ctr, depth)) },
                    While { cond, body } => While { cond: cond.clone(),
                        body: Box::new(inline_block_helpers(body, env, ctr, depth)) },
                    other => expand(other, env, &mut pre, ctr, depth),
                };
                out.extend(pre);
                out.push(new_st);
            }
            Block(out)
        }
        // Single-expression body: hoisted assigns turn it into a Block.
        other => {
            let mut pre: Vec<r2_types::Expr> = Vec::new();
            let fin = expand(other, env, &mut pre, ctr, depth);
            if pre.is_empty() { fin } else { pre.push(fin); Block(pre) }
        }
    }
}
