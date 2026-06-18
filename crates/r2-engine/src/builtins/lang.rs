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

/// Convert an evaluated value back into an AST node for call construction.
/// Phase L.1 supports quoted expressions and scalar literals; richer values
/// are deferred (they need a literal-RVal AST node).
fn value_to_expr(v: &RVal) -> Result<Expr, R2Err> {
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
