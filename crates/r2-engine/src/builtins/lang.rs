//! Phase L.1 — first-class language objects.
//!
//! `eval` / `parse` / `deparse` / `call` / `as.call`. Together with the
//! `quote()` NSE intercept in the evaluator and `RVal::Lang(Arc<Expr>)`,
//! these give R's "code is data" round-trip:
//!     deparse(quote(x + 1))        # "x + 1"
//!     eval(parse(text = "1 + 1"))  # 2
//!
//! `quote` itself is NOT here — it must see its argument UNEVALUATED, so it
//! is handled as a special form in `Engine::eval_in` (like `with`/`curve`).

use std::sync::Arc;

use r2_types::*;

use crate::{gv, Engine};
use crate::err;

/// `eval(expr, envir)` — run a quoted expression. A plain value evaluates to
/// itself (R semantics). A list of language objects (from `parse()` of a
/// multi-statement string) runs each in order and returns the last result.
/// The `envir=` argument is not yet honoured — evaluation uses the calling
/// environment (Phase L.1).
pub(crate) fn bi_eval(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::Lang(expr) => e.eval_in(&expr, env),
        RVal::List(items) => {
            let mut last = RVal::Null;
            for (_, v) in items {
                last = match v {
                    RVal::Lang(expr) => e.eval_in(&expr, env)?,
                    other => other,
                };
            }
            Ok(last)
        }
        other => Ok(other),
    }
}

/// `deparse(x)` — turn a language object back into source text. For a
/// non-language value, falls back to its printed form (good enough for the
/// common `deparse(substitute(x))`-style uses; full value deparsing —
/// `c(1, 2, 3)` etc. — is deferred).
pub(crate) fn bi_deparse(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let s = match gv(a, 0) {
        RVal::Lang(expr) => deparse(&expr),
        other => format!("{}", other),
    };
    Ok(RVal::Character(vec![Some(Arc::from(s.as_str()))], Attrs::default()))
}

/// `parse(text=)` — parse R source into language object(s). A single
/// expression returns a `Lang`; multiple statements return a list of them
/// (R's "expression" vector), which `eval()` runs in sequence.
pub(crate) fn bi_parse(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let src = a.iter().find(|x| x.name.as_deref() == Some("text")).map(|x| &x.value)
        .or_else(|| a.iter().find(|x| x.name.is_none()).map(|x| &x.value));
    let text = match src {
        Some(RVal::Character(v, _)) =>
            v.iter().filter_map(|s| s.as_ref().map(|x| x.to_string())).collect::<Vec<_>>().join("\n"),
        Some(other) => return err!(Runtime, "parse(): text= must be character, got '{}'", other.type_name()),
        None => return err!(Runtime, "parse(): need a text= argument"),
    };
    let exprs = r2_parser::Parser::parse(&text)
        .map_err(|pe| R2Err { msg: format!("parse error: {}", pe), kind: ErrKind::Runtime })?;
    match exprs.len() {
        0 => Ok(RVal::Null),
        1 => Ok(RVal::Lang(Arc::new(exprs.into_iter().next().unwrap()))),
        _ => Ok(RVal::List(exprs.into_iter().map(|e| (None, RVal::Lang(Arc::new(e)))).collect())),
    }
}

// ── Phase L.3 — NSE: substitute / match.call / sys.call ───────────────

/// Does an expression tree use NSE (substitute/match.call/sys.call)? Used
/// once per unique closure body (then cached) to decide whether calling
/// that closure needs an NSE frame. Does NOT descend into nested function
/// definitions — an inner function gets its own gate when it is called.
pub(crate) fn expr_uses_nse(e: &Expr) -> bool {
    fn is_nse(s: &str) -> bool { matches!(s, "substitute" | "match.call" | "sys.call" | "bquote") }
    match e {
        Expr::Call { func, args } => {
            if let Expr::Symbol(s) = func.as_ref() { if is_nse(s) { return true; } }
            expr_uses_nse(func) || args.iter().any(|a| expr_uses_nse(&a.value))
        }
        Expr::Unary { expr, .. } => expr_uses_nse(expr),
        Expr::Binary { lhs, rhs, .. } => expr_uses_nse(lhs) || expr_uses_nse(rhs),
        Expr::Assign { target, value, .. } => expr_uses_nse(target) || expr_uses_nse(value),
        Expr::Index { object, indices } =>
            expr_uses_nse(object) || indices.iter().flatten().any(expr_uses_nse),
        Expr::DblIndex { object, index } => expr_uses_nse(object) || expr_uses_nse(index),
        Expr::Dollar { object, .. } => expr_uses_nse(object),
        Expr::Pipe { lhs, rhs } => expr_uses_nse(lhs) || expr_uses_nse(rhs),
        Expr::If { cond, then, else_ } =>
            expr_uses_nse(cond) || expr_uses_nse(then) || else_.as_deref().is_some_and(expr_uses_nse),
        Expr::For { iter, body, .. } => expr_uses_nse(iter) || expr_uses_nse(body),
        Expr::While { cond, body } => expr_uses_nse(cond) || expr_uses_nse(body),
        Expr::Match { expr, arms } =>
            expr_uses_nse(expr) || arms.iter().any(|a|
                a.patterns.iter().any(expr_uses_nse) || expr_uses_nse(&a.body)),
        Expr::Block(stmts) => stmts.iter().any(expr_uses_nse),
        Expr::Return(e) => expr_uses_nse(e),
        Expr::TryCatch { body, catch, .. } => expr_uses_nse(body) || expr_uses_nse(catch),
        // Inner functions are scoped separately — do not descend.
        _ => false,
    }
}

/// R-style matching of a call's arguments to a function's formals. Returns
/// `(param_name, unevaluated_arg_expr)` for each supplied argument: named
/// args bind by name first, then positional args fill remaining formals in
/// order. `...` formals are skipped (Phase L.3 limitation).
pub(crate) fn match_call_args(call: &Expr, params: &[Param]) -> Vec<(Arc<str>, Expr)> {
    let args: &[CallArg] = match call { Expr::Call { args, .. } => args, _ => return Vec::new() };
    let mut filled: Vec<Option<Expr>> = vec![None; params.len()];
    let mut used = vec![false; args.len()];
    // 1. named args → param of the same name
    for (ai, a) in args.iter().enumerate() {
        if let Some(nm) = &a.name {
            if let Some(pi) = params.iter().position(|p| !p.dots && p.name == *nm) {
                filled[pi] = Some(a.value.clone());
                used[ai] = true;
            }
        }
    }
    // 2. positional args → next unfilled non-dots formal
    let mut pi = 0usize;
    for (ai, a) in args.iter().enumerate() {
        if used[ai] || a.name.is_some() { continue; }
        while pi < params.len() && (params[pi].dots || filled[pi].is_some()) { pi += 1; }
        if pi < params.len() { filled[pi] = Some(a.value.clone()); pi += 1; }
    }
    params.iter().zip(filled).filter_map(|(p, f)| f.map(|e| (p.name.clone(), e))).collect()
}

/// Replace every symbol in `e` that names a mapped parameter with that
/// parameter's expression. Used by `substitute()`.
pub(crate) fn substitute_expr(e: &Expr, map: &[(Arc<str>, Expr)]) -> Expr {
    match e {
        Expr::Symbol(s) => {
            for (name, repl) in map { if name == s { return repl.clone(); } }
            e.clone()
        }
        Expr::Binary { op, lhs, rhs } => Expr::Binary {
            op: *op, lhs: Box::new(substitute_expr(lhs, map)), rhs: Box::new(substitute_expr(rhs, map)) },
        Expr::Unary { op, expr } => Expr::Unary {
            op: *op, expr: Box::new(substitute_expr(expr, map)) },
        Expr::Call { func, args } => Expr::Call {
            func: Box::new(substitute_expr(func, map)),
            args: args.iter().map(|a| CallArg { name: a.name.clone(), value: substitute_expr(&a.value, map) }).collect() },
        Expr::Index { object, indices } => Expr::Index {
            object: Box::new(substitute_expr(object, map)),
            indices: indices.iter().map(|i| i.as_ref().map(|e| substitute_expr(e, map))).collect() },
        Expr::DblIndex { object, index } => Expr::DblIndex {
            object: Box::new(substitute_expr(object, map)), index: Box::new(substitute_expr(index, map)) },
        Expr::Dollar { object, field } => Expr::Dollar {
            object: Box::new(substitute_expr(object, map)), field: field.clone() },
        Expr::Pipe { lhs, rhs } => Expr::Pipe {
            lhs: Box::new(substitute_expr(lhs, map)), rhs: Box::new(substitute_expr(rhs, map)) },
        other => other.clone(),
    }
}

/// `match.call()` — the current call with arguments matched to formal names:
/// `f(1, 2)` inside `function(x, y)` → `f(x = 1, y = 2)`.
pub(crate) fn bi_match_call(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match e.current_nse_frame() {
        Some((call, params)) => {
            let matched = match_call_args(&call, &params);
            let func = match call.as_ref() {
                Expr::Call { func, .. } => func.clone(),
                _ => Box::new(Expr::NullLit),
            };
            let cargs: Vec<CallArg> = matched.into_iter()
                .map(|(n, v)| CallArg { name: Some(n), value: v }).collect();
            Ok(RVal::Lang(Arc::new(Expr::Call { func, args: cargs })))
        }
        None => err!(Runtime, "match.call() used outside a function"),
    }
}

/// `sys.call()` — the current call exactly as written (no formal matching).
pub(crate) fn bi_sys_call(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match e.current_nse_frame() {
        Some((call, _)) => Ok(RVal::Lang(call)),
        None => Ok(RVal::Null),
    }
}

/// `body(f)` — the body of a user-defined function, as a language object.
/// Built-in primitives have no inspectable body → NULL (R's behaviour).
pub(crate) fn bi_body(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::Closure(cl) => Ok(RVal::Lang(cl.body.clone())),
        RVal::BuiltinFn(_) => Ok(RVal::Null),
        other => err!(Runtime, "body(): not a function (got '{}')", other.type_name()),
    }
}

/// `formals(f)` — the formal arguments as a named list: each element is the
/// argument's default (a language object) or NULL when it has none. `...`
/// is named "...". Primitives → NULL.
pub(crate) fn bi_formals(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::Closure(cl) => {
            let items = cl.params.iter().map(|p| {
                let name = if p.dots { Arc::from("...") } else { p.name.clone() };
                let val = p.default.as_ref()
                    .map(|d| RVal::Lang(Arc::new((**d).clone())))
                    .unwrap_or(RVal::Null);
                (Some(name), val)
            }).collect();
            Ok(RVal::List(items))
        }
        RVal::BuiltinFn(_) => Ok(RVal::Null),
        other => err!(Runtime, "formals(): not a function (got '{}')", other.type_name()),
    }
}

/// `args(f)` — a function with the same formals but a NULL body, i.e. the
/// signature alone (R's behaviour). Non-functions pass through unchanged.
pub(crate) fn bi_args(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::Closure(cl) => Ok(RVal::Closure(Closure {
            params: cl.params.clone(),
            body: Arc::new(Expr::NullLit),
            env: cl.env.clone(),
        })),
        other => Ok(other),
    }
}

/// Convert an evaluated value back into an AST node for call construction.
/// Phase L.1 supports quoted expressions and scalar literals; richer values
/// are deferred (they need a literal-RVal AST node). Also used by `bquote`'s
/// `.()` splice.
pub(crate) fn value_to_expr(v: &RVal) -> Result<Expr, R2Err> {
    match v {
        RVal::Lang(e) => Ok((**e).clone()),
        RVal::Null => Ok(Expr::NullLit),
        RVal::Character(cv, _) if cv.len() == 1 =>
            Ok(cv[0].as_ref().map(|s| Expr::StrLit(s.to_string())).unwrap_or(Expr::NaLit)),
        other => match other.scalar_f64() {
            Ok(Some(n)) => Ok(Expr::NumLit(n)),
            _ => err!(Runtime,
                "call/as.call: only scalar or quoted args supported in Phase L.1 (got '{}')",
                other.type_name()),
        },
    }
}

/// `call(name, ...)` — build an unevaluated call to `name` with the given
/// arguments. `call("sum", 1, 2)` → the language object `sum(1, 2)`.
pub(crate) fn bi_call(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let fname = match gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|s| s.clone()),
        _ => None,
    }.ok_or_else(|| R2Err {
        msg: "call(): first argument must be a function-name string".into(),
        kind: ErrKind::Runtime,
    })?;
    let mut cargs = Vec::with_capacity(a.len().saturating_sub(1));
    for arg in &a[1..] {
        cargs.push(CallArg { name: arg.name.clone(), value: value_to_expr(&arg.value)? });
    }
    let call = Expr::Call { func: Box::new(Expr::Symbol(fname)), args: cargs };
    Ok(RVal::Lang(Arc::new(call)))
}

/// `as.call(list)` — turn a list into a call: the first element is the
/// function (a name string or a quoted symbol), the rest are arguments.
pub(crate) fn bi_as_call(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match gv(a, 0) {
        RVal::List(items) if !items.is_empty() => {
            let func = match &items[0].1 {
                RVal::Character(v, _) => Expr::Symbol(
                    v.first().and_then(|s| s.clone()).ok_or_else(|| R2Err {
                        msg: "as.call(): NA function name".into(), kind: ErrKind::Runtime })?),
                other => value_to_expr(other)?,
            };
            let mut cargs = Vec::with_capacity(items.len() - 1);
            for (nm, v) in &items[1..] {
                cargs.push(CallArg { name: nm.clone(), value: value_to_expr(v)? });
            }
            Ok(RVal::Lang(Arc::new(Expr::Call { func: Box::new(func), args: cargs })))
        }
        _ => err!(Runtime, "as.call(): argument must be a non-empty list"),
    }
}
