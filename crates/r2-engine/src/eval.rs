//! The tree-walking evaluator — `eval_in` (the main Expr match, incl. NSE
//! special forms), `call_fn` (closure/builtin dispatch + JIT call path),
//! and the call-resolution helpers. Extracted from lib.rs as a second
//! `impl Engine` block; identical compiled output, just navigable.

#![allow(clippy::all)]
use std::sync::Arc;
use std::collections::HashMap;
use r2_types::*;
use crate::{Engine, NseFrame, val_to_str, env_insert, body_defines_closure};
use crate::builtins;
use crate::formula::{fmt_expr, split_error_term, split_random_effects};
use crate::na_bitmap::{combine_binary_output, combine_ternary_output, combine_unary_output};
use crate::err;

impl Engine {
    pub fn eval_in(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        // Phase R.M.2 — check the global interrupt flag at the top of every
        // expression evaluation. This is the cheapest universal interruption
        // point in the engine: an atomic-load per Expr is below 1ns on any
        // modern CPU, and it catches everything from runaway loops to deep
        // recursion to long Sys.sleep calls. The REPL's SIGINT handler sets
        // the flag; we raise Interrupt here, which unwinds cleanly to the
        // top-level driver.
        if r2_types::is_interrupted() {
            return Err(R2Err {
                msg: "interrupted".into(),
                kind: ErrKind::Interrupt,
            });
        }

        match expr {
            Expr::NumLit(n) => Ok(rnum(*n)), Expr::IntLit(n) => Ok(rint(*n)),
            Expr::StrLit(s) => Ok(rstr(s)), Expr::BoolLit(b) => Ok(rbool(*b)),
            Expr::NaLit => Ok(rna()), Expr::NullLit => Ok(RVal::Null),
            Expr::FStringLit(parts) => { let mut r = String::new(); for p in parts { match p { FStringPart::Literal(s) => r.push_str(s), FStringPart::Expr(e) => { let v = self.eval_in(e, env)?; r.push_str(&val_to_str(&v)); } } } Ok(rstr(&r)) }
            Expr::Symbol(name) => {
                // 1. Check local scope stack (function-local variables)
                for scope in self.local_scopes.iter().rev() {
                    if let Some(val) = scope.get(name.as_ref()) { return Ok(val.clone()); }
                }
                // 2. Check env chain (parameters, closures)
                if let Some(val) = env.lookup(name) { Ok(val.clone()) }
                // 3. Check global env (top-level assignments, datasets)
                else if let Some(val) = self.global_env.lookup(name) { Ok(val.clone()) }
                // 4. Check builtins
                else if self.registry.resolve(name.as_ref()).is_some() { Ok(RVal::BuiltinFn(name.clone())) }
                // ..1, ..2, … — the N-th element of the captured `...`.
                else if let Some(v) = name.strip_prefix("..").and_then(|s| s.parse::<usize>().ok())
                    .and_then(|n| match self.lookup_dots(env) { Some(RVal::List(d)) => d.get(n.wrapping_sub(1)).map(|(_, v)| v.clone()), _ => None }) { Ok(v) }
                // T / F default to TRUE / FALSE when not bound (R semantics).
                else if name.as_ref() == "T" { Ok(rbool(true)) }
                else if name.as_ref() == "F" { Ok(rbool(false)) }
                else { err!(Runtime, "object '{}' not found", name) }
            }
            Expr::Assign { target, value, superassign } => {
                let val = self.eval_in(value, env)?;
                match target.as_ref() {
                    Expr::Symbol(name) => {
                        if matches!(name.as_ref(), "TRUE"|"FALSE") { return err!(Runtime, "cannot assign to the reserved constant '{}'", name); }
                        if *superassign { self.super_assign(name.clone(), val.clone()); }
                        else { self.scope_insert(name.clone(), val.clone()); }
                        Ok(val)
                    }
                    // Literals are not valid targets (e.g. `1 <- x`).
                    Expr::NumLit(_) | Expr::IntLit(_) | Expr::StrLit(_) | Expr::BoolLit(_)
                    | Expr::NullLit | Expr::NaLit =>
                        err!(Runtime, "cannot assign to a literal value"),
                    Expr::Index { object, indices } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            if indices.len() == 1 {
                                if let Some(idx_expr) = &indices[0] {
                                    let idx = self.eval_in(idx_expr, env)?;
                                    self.assign_index(&mut obj, &idx, &val)?;
                                }
                            } else if indices.len() == 2 {
                                // Matrix `m[i, j] <- v` / `m[i, ] <- v` / `m[, j] <- v`.
                                // An empty subscript (None) selects the whole axis.
                                let ri = match &indices[0] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                                let ci = match &indices[1] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                                self.assign_matrix_index(&mut obj, ri.as_ref(), ci.as_ref(), &val)?;
                            }
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid subscript assignment target") }
                    }
                    Expr::DblIndex { object, index } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            let idx = self.eval_in(index, env)?;
                            self.assign_dbl_index(&mut obj, &idx, &val)?;
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid [[ ]] assignment target") }
                    }
                    Expr::Dollar { object, field } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            self.assign_dollar(&mut obj, field, &val)?;
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid $ assignment target") }
                    }
                    // Replacement function: `fname(obj, ...) <- value`
                    // desugars to `obj <- \`fname<-\`(obj, ..., value=value)`.
                    // Enables names(x)<-, colnames(df)<-, rownames(df)<-, etc.
                    Expr::Call { func, args } => {
                        if let (Expr::Symbol(fname), Some(first)) = (func.as_ref(), args.first()) {
                            let setter = format!("{}<-", fname);
                            if let Some((f, _)) = self.registry.resolve(&setter) {
                                let obj_val = self.eval_in(&first.value, env)?;
                                let mut ea = vec![EvalArg { name: None, value: obj_val }];
                                for extra in &args[1..] {
                                    ea.push(EvalArg { name: extra.name.clone(), value: self.eval_in(&extra.value, env)? });
                                }
                                ea.push(EvalArg { name: Some(Arc::from("value")), value: val.clone() });
                                let new_obj = f(self, &ea, env)?;
                                if let Expr::Symbol(objname) = &first.value {
                                    self.scope_insert(objname.clone(), new_obj);
                                    return Ok(val);
                                }
                                return err!(Runtime, "replacement target must be a variable");
                            }
                            return err!(Runtime, "could not find function \"{}\"", setter);
                        }
                        err!(Runtime, "invalid assignment target")
                    }
                    _ => err!(Runtime, "invalid assignment target"),
                }
            }
            Expr::Block(stmts) => { let mut r = RVal::Null; for s in stmts { r = self.eval_in(s, env)?; } Ok(r) }
            Expr::Binary { op, lhs, rhs } => {
                if *op == BinOp::Colon { let l = self.eval_in(lhs, env)?; let r = self.eval_in(rhs, env)?; return self.seq_colon(&l, &r); }
                if *op == BinOp::Tilde {
                    // Formula: y ~ x evaluates both sides, stores as formula-list
                    // lhs can be NULL for one-sided formulas (~x)
                    let l = self.eval_in(lhs, env)?;
                    let r = self.eval_in(rhs, env)?;
                    return Ok(RVal::List(vec![
                        (Some(Arc::from("~lhs")), l),
                        (Some(Arc::from("~rhs")), r),
                        (Some(Arc::from("~class")), rstr("formula")),
                    ]));
                }
                // Phase 1 fusion: collapse a left-leaning vector⊗scalar
                // arithmetic chain (e.g. `v*2+1`, `(v+1)*2`) into ONE pass
                // instead of one allocation + pass per operator. Safe: only
                // when the base is a Symbol (side-effect-free lookup) and the
                // other operands are numeric literals.
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Mod) {
                    if let Some(fused) = self.try_fuse_scalar_chain(*op, lhs, rhs, env)? {
                        return Ok(fused);
                    }
                }
                let l = self.eval_in(lhs, env)?; let r = self.eval_in(rhs, env)?; self.binary_op(*op, &l, &r)
            }
            Expr::Unary { op, expr: e } => { let v = self.eval_in(e, env)?; self.unary_op(*op, &v) }
            Expr::Call { func, args } => {
                // NSE: library(stats), detach(stats), require(stats) accept bare symbols
                // Convert bare symbol to string without evaluating it
                if let Expr::Symbol(fname) = func.as_ref() {
                    if matches!(fname.as_ref(), "library" | "detach" | "require" | "data" | "help") {
                        let f = self.eval_in(func, env)?;
                        let mut ea = Vec::new();
                        for (i, a) in args.iter().enumerate() {
                            if i == 0 {
                                // First arg: if bare symbol, convert to string (NSE)
                                match &a.value {
                                    Expr::Symbol(sym) => {
                                        // Check if it's actually a variable holding a string
                                        if let Some(val) = env.lookup(sym) {
                                            ea.push(EvalArg { name: a.name.clone(), value: val.clone() });
                                        } else {
                                            // Bare symbol → treat as package name string
                                            ea.push(EvalArg { name: a.name.clone(), value: rstr(sym) });
                                        }
                                    }
                                    _ => ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }),
                                }
                            } else {
                                ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                            }
                        }
                        return self.call_fn(&f, &ea, env);
                    }

                    // NSE for `subset(df, cond)` and `transform(df, name = expr)`:
                    // arg 2+ expressions evaluate in a scope where df's columns
                    // are bound as variables. Without this, `subset(df, x > 2)`
                    // resolves `x` against the global env.
                    if matches!(fname.as_ref(), "subset" | "transform") {
                        if args.len() >= 2 {
                            // Evaluate first arg = data frame.
                            let df_val = self.eval_in(&args[0].value, env)?;
                            if let RVal::DataFrame(df) = &df_val {
                                // Build child env that shadows globals with df columns.
                                let child = Arc::new(Env {
                                    name: Some(Arc::from(".subset.env")),
                                    parent: Some(env.clone()),
                                    bindings: df.columns.iter()
                                        .map(|(n, v)| (n.clone(), v.clone())).collect(),
                                    locked: false,
                                });
                                let f = self.eval_in(func, env)?;
                                let mut ea = vec![EvalArg { name: None, value: df_val.clone() }];
                                for a in args.iter().skip(1) {
                                    let val = self.eval_in(&a.value, &child)?;
                                    ea.push(EvalArg { name: a.name.clone(), value: val });
                                }
                                return self.call_fn(&f, &ea, env);
                            }
                        }
                    }

                    // quote(e) — return the argument UNEVALUATED as a language
                    // object (RVal::Lang). The keystone of Phase L.1; like
                    // curve/with, it must intercept before args are evaluated.
                    if fname.as_ref() == "quote" {
                        return Ok(args.first()
                            .map(|a| RVal::Lang(Arc::new(a.value.clone())))
                            .unwrap_or(RVal::Null));
                    }

                    // rm(x, y, ...) / rm("x") / rm(list=c("a","b")) — NSE:
                    // take the UNEVALUATED symbol/string names and delete the
                    // bindings (was single-character-string only).
                    if fname.as_ref() == "rm" {
                        let mut to_remove: Vec<Arc<str>> = Vec::new();
                        for arg in args {
                            if arg.name.as_deref() == Some("list") {
                                if let Ok(RVal::Character(v, _)) = self.eval_in(&arg.value, env) {
                                    for x in v.into_iter().flatten() { to_remove.push(x); }
                                }
                                continue;
                            }
                            match &arg.value {
                                Expr::Symbol(s) => to_remove.push(s.clone()),
                                Expr::StrLit(s) => to_remove.push(Arc::from(s.as_str())),
                                _ => {}
                            }
                        }
                        for nm in &to_remove {
                            for scope in self.local_scopes.iter_mut() { scope.remove(nm.as_ref()); }
                            let g = Arc::make_mut(&mut self.global_env);
                            g.bindings.remove(nm.as_ref());
                        }
                        return Ok(RVal::Null);
                    }

                    // substitute(e) — return e with the current function's
                    // parameter symbols replaced by the UNEVALUATED expressions
                    // the caller passed (R's promise stand-in). Outside a
                    // function, returns e unchanged. (Phase L.3.)
                    if fname.as_ref() == "substitute" {
                        let arg = args.first().map(|a| a.value.clone()).unwrap_or(Expr::NullLit);
                        let result = if let Some(frame) = self.nse_stack.last() {
                            let map = builtins::lang::match_call_args(&frame.call, &frame.params);
                            builtins::lang::substitute_expr(&arg, &map)
                        } else {
                            arg
                        };
                        return Ok(RVal::Lang(Arc::new(result)));
                    }

                    // bquote(e) — quote e, but splice in any `.(x)` evaluated
                    // in the current environment (quasiquotation). (Phase L.3.)
                    if fname.as_ref() == "bquote" {
                        let arg = args.first().map(|a| a.value.clone()).unwrap_or(Expr::NullLit);
                        let result = self.bquote_walk(&arg, env)?;
                        return Ok(RVal::Lang(Arc::new(result)));
                    }

                    // NSE for `data.frame(y, x1, x2)` — bare-symbol args become
                    // column names. R does this by inspecting the unevaluated
                    // call; we replicate by lifting `Expr::Symbol` arg names
                    // into the EvalArg `name` slot when no explicit `name =`
                    // is given. Without this, `data.frame(y, x1, x2)` would
                    // produce columns V1/V2/V3 and `df[, c("x1","x2")]` would
                    // find nothing.
                    if fname.as_ref() == "data.frame" {
                        let f = self.eval_in(func, env)?;
                        let mut ea = Vec::with_capacity(args.len());
                        for a in args {
                            let val = self.eval_in(&a.value, env)?;
                            let name = a.name.clone().or_else(|| match &a.value {
                                Expr::Symbol(s) => Some(s.clone()),
                                _ => None,
                            });
                            ea.push(EvalArg { name, value: val });
                        }
                        return self.call_fn(&f, &ea, env);
                    }

                    // `tryCatch(expr, error=function(e){...}, finally={...})`
                    // — the function form (distinct from the try/catch
                    // syntax). Eval expr; on a non-control error, call the
                    // `error=` handler with the message; always run `finally=`.
                    if fname.as_ref() == "tryCatch" {
                        let expr = match args.iter().find(|a| a.name.is_none()) {
                            Some(e) => &e.value,
                            None => return Ok(RVal::Null),
                        };
                        let finally = args.iter().find(|a| a.name.as_deref() == Some("finally")).map(|a| a.value.clone());
                        let out = match self.eval_in(expr, env) {
                            Ok(v) => Ok(v),
                            Err(e) if matches!(e.kind, ErrKind::CtrlReturn(_) | ErrKind::CtrlBreak | ErrKind::CtrlNext | ErrKind::Interrupt) => Err(e),
                            Err(e) => {
                                if let Some(h) = args.iter().find(|a| a.name.as_deref() == Some("error")) {
                                    let handler = self.eval_in(&h.value, env)?;
                                    self.call_fn(&handler, &[EvalArg { name: None, value: rstr(&e.msg) }], env)
                                } else { Err(e) }
                            }
                        };
                        if let Some(fe) = finally { let _ = self.eval_in(&fe, env); }
                        return out;
                    }

                    // NSE for `with(data, expr)`: evaluate `expr` in a scope
                    // where `data`'s columns / list elements are bound.
                    if fname.as_ref() == "with" && args.len() >= 2 {
                        let data = self.eval_in(&args[0].value, env)?;
                        let child = Arc::new(Env {
                            name: Some(Arc::from(".with.env")),
                            parent: Some(env.clone()),
                            bindings: match &data {
                                RVal::DataFrame(df) =>
                                    df.columns.iter().map(|(n, v)| (n.clone(), v.clone())).collect(),
                                RVal::List(items) => items.iter()
                                    .filter_map(|(n, v)| n.clone().map(|nm| (nm, v.clone()))).collect(),
                                _ => Default::default(),
                            },
                            locked: false,
                        });
                        return self.eval_in(&args[1].value, &child);
                    }

                    // `switch(EXPR, ...)`: EXPR selects which branch expr to
                    // evaluate — by name (character) or position (numeric).
                    // The branch exprs are taken UNEVALUATED; only the chosen
                    // one runs.
                    if fname.as_ref() == "switch" && !args.is_empty() {
                        let sel = self.eval_in(&args[0].value, env)?;
                        let branches = &args[1..];
                        match &sel {
                            RVal::Character(cv, _) => {
                                if let Some(Some(key)) = cv.first() {
                                    if let Some(b) = branches.iter().find(|b| b.name.as_deref() == Some(key.as_ref())) {
                                        return self.eval_in(&b.value, env);
                                    }
                                    if let Some(d) = branches.iter().find(|b| b.name.is_none()) {
                                        return self.eval_in(&d.value, env);
                                    }
                                }
                                return Ok(RVal::Null);
                            }
                            _ => {
                                if let Some(n) = sel.scalar_f64().ok().flatten() {
                                    let i = n as usize;
                                    if i >= 1 && i <= branches.len() {
                                        return self.eval_in(&branches[i - 1].value, env);
                                    }
                                }
                                return Ok(RVal::Null);
                            }
                        }
                    }

                    // NSE for `curve(expr, from, to, n=101, add=FALSE, ...)`:
                    // the first arg is taken UNEVALUATED and evaluated with
                    // `x` bound to a sequence (R's vectorized model), so
                    // `curve(x^2, 0, 10)` works without lazy promises. A bare
                    // function (`curve(sin, ...)`) is applied to x instead.
                    if fname.as_ref() == "curve" && !args.is_empty() {
                        let named = |nm: &str| args.iter().find(|a| a.name.as_deref() == Some(nm));
                        let positional: Vec<&Expr> =
                            args[1..].iter().filter(|a| a.name.is_none()).map(|a| &a.value).collect();
                        let from_e = named("from").map(|a| &a.value).or_else(|| positional.first().copied());
                        let to_e   = named("to").map(|a| &a.value).or_else(|| positional.get(1).copied());
                        let n_e    = named("n").map(|a| &a.value).or_else(|| positional.get(2).copied());
                        let from = match from_e {
                            Some(e) => self.eval_in(e, env)?.scalar_f64().ok().flatten().unwrap_or(0.0),
                            None => 0.0,
                        };
                        let to = match to_e {
                            Some(e) => self.eval_in(e, env)?.scalar_f64().ok().flatten().unwrap_or(1.0),
                            None => 1.0,
                        };
                        let n = (match n_e {
                            Some(e) => self.eval_in(e, env)?.scalar_f64().ok().flatten().unwrap_or(101.0),
                            None => 101.0,
                        } as usize).max(2);
                        let xs: Vec<f64> = (0..n)
                            .map(|i| from + (to - from) * (i as f64) / ((n - 1) as f64))
                            .collect();
                        let xs_val = RVal::Numeric(
                            xs.iter().map(|v| Some(*v)).collect::<Vec<_>>().into(), Attrs::default());
                        let child = Arc::new(Env {
                            name: Some(Arc::from(".curve.env")),
                            parent: Some(env.clone()),
                            bindings: std::iter::once((Arc::from("x"), xs_val.clone())).collect(),
                            locked: false,
                        });
                        let mut y_val = self.eval_in(&args[0].value, &child)?;
                        if matches!(y_val, RVal::Closure(_)) {
                            y_val = self.call_fn(&y_val, &[EvalArg { name: None, value: xs_val.clone() }], &child)?;
                        }
                        let add = match named("add") {
                            Some(a) => match self.eval_in(&a.value, env)? {
                                RVal::Logical(v, _) => v.first().and_then(|x| *x).unwrap_or(false),
                                other => other.scalar_f64().ok().flatten().map(|x| x != 0.0).unwrap_or(false),
                            },
                            None => false,
                        };
                        let mut ea = vec![
                            EvalArg { name: None, value: xs_val },
                            EvalArg { name: None, value: y_val },
                        ];
                        for nm in ["col", "lwd", "lty", "main", "xlab", "ylab", "ylim", "xlim"] {
                            if let Some(arg) = named(nm) {
                                let v = self.eval_in(&arg.value, env)?;
                                ea.push(EvalArg { name: Some(Arc::from(nm)), value: v });
                            }
                        }
                        return if add {
                            r2_graphics::overlays::bi_lines(&ea)
                        } else {
                            ea.push(EvalArg { name: Some(Arc::from("type")), value: rstr("l") });
                            r2_graphics::plots::bi_plot(&ea)
                        };
                    }

                    // NSE for formula-based functions: lm(y ~ x, data = df)
                    // When first arg is a tilde expr and data= is provided,
                    // resolve bare symbol names as columns in the data frame
                    if matches!(fname.as_ref(), "lm" | "glm" | "t.test" | "rpart" | "rf" | "gbm" | "cv" | "aov" | "manova" | "lmer" | "aggregate" | "boxplot") {
                        if let Some(first_arg) = args.first() {
                            if let Expr::Binary { op: BinOp::Tilde, lhs, rhs } = &first_arg.value {
                                // Find the data frame: `data=` by name, else
                                // R's positional convention — the first
                                // UNNAMED argument after the formula (so
                                // `lm(y ~ x, df)` works like `lm(y ~ x, data=df)`).
                                let data_arg = args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some("data"))
                                    .or_else(|| args.iter().skip(1).find(|a| a.name.is_none()));
                                if let Some(data_a) = data_arg {
                                    let data_val = self.eval_in(&data_a.value, env)?;
                                    if let RVal::DataFrame(ref df) = data_val {
                                        // aggregate(value ~ group, data = df, FUN = ...)
                                        // The formula is purely an input adapter: resolve the
                                        // response column and grouping column from the frame,
                                        // then hand them to aggregate's existing
                                        // (x, by =, FUN =) core unchanged — so the split-apply
                                        // math is identical to the non-formula call.
                                        // Phase 1: a single response and a single grouping
                                        // factor (cbind() / a + b land in Phase 2).
                                        if fname.as_ref() == "aggregate" {
                                            // Phase 2: cbind(y1,y2) ~ g1 + g2 — any number of
                                            // response columns and grouping factors. The formula
                                            // is purely an input adapter (formula_frame); the
                                            // split-apply math (FUN per group) is the same as the
                                            // single-variable case.
                                            let (responses, groups) = self.formula_frame(lhs, rhs, df, env)?;
                                            if groups.is_empty() {
                                                return Err(R2Err { msg: "aggregate(): formula needs at least one grouping factor on the RHS".into(), kind: ErrKind::Runtime });
                                            }
                                            if responses.is_empty() {
                                                return Err(R2Err { msg: "aggregate(): formula needs at least one response on the LHS".into(), kind: ErrKind::Runtime });
                                            }
                                            // Resolve FUN: named FUN=, else first positional arg
                                            // after the formula (skipping data=).
                                            let fun_expr = args.iter().find(|a| a.name.as_deref() == Some("FUN"))
                                                .or_else(|| args.iter().skip(1).find(|a| a.name.is_none()))
                                                .map(|a| &a.value);
                                            let f = match fun_expr {
                                                Some(e) => self.eval_in(e, env)?,
                                                None => return Err(R2Err { msg: "aggregate(): FUN is required".into(), kind: ErrKind::Runtime }),
                                            };
                                            // Element-wise labels for each grouping factor.
                                            let col_labels = |c: &RVal| -> Vec<String> {
                                                match c {
                                                    RVal::Numeric(v, _) => v.iter().map(|x| x.map(|n| fmt_num(n)).unwrap_or_else(|| "NA".into())).collect(),
                                                    RVal::Integer(v, _) => v.iter().map(|x| x.map(|n| n.to_string()).unwrap_or_else(|| "NA".into())).collect(),
                                                    RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into())).collect(),
                                                    RVal::Logical(v, _) => v.iter().map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect(),
                                                    _ => Vec::new(),
                                                }
                                            };
                                            let group_labels: Vec<Vec<String>> = groups.iter().map(|(_, c)| col_labels(c)).collect();
                                            let nrow = group_labels.first().map(|v| v.len()).unwrap_or(0);
                                            // Distinct group combinations (composite key per row).
                                            let mut combos: Vec<Vec<String>> = Vec::new();
                                            let mut row_combo: Vec<usize> = Vec::with_capacity(nrow);
                                            for r in 0..nrow {
                                                let key: Vec<String> = group_labels.iter()
                                                    .map(|g| g.get(r).cloned().unwrap_or_default()).collect();
                                                match combos.iter().position(|c| *c == key) {
                                                    Some(p) => row_combo.push(p),
                                                    None => { combos.push(key); row_combo.push(combos.len() - 1); }
                                                }
                                            }
                                            // Sort combos lexicographically by label tuple (R orders
                                            // aggregate output by grouping levels).
                                            let mut order: Vec<usize> = (0..combos.len()).collect();
                                            order.sort_by(|&i, &j| combos[i].cmp(&combos[j]));
                                            let mut rows_per_combo: Vec<Vec<usize>> = vec![Vec::new(); combos.len()];
                                            for r in 0..nrow { rows_per_combo[row_combo[r]].push(r); }
                                            // Build output: one column per grouping factor, then
                                            // one column per response (real source names).
                                            let mut out_cols: Vec<(Arc<str>, RVal)> = Vec::new();
                                            for (gi, (gname, _)) in groups.iter().enumerate() {
                                                let col: Vec<Character> = order.iter()
                                                    .map(|&ci| Some(Arc::from(combos[ci][gi].as_str()))).collect();
                                                out_cols.push((gname.clone(), RVal::Character(col, Attrs::default())));
                                            }
                                            for (rname, rcol) in &responses {
                                                let vals = self.as_reals(rcol)?;
                                                let mut agg: Vec<Real> = Vec::with_capacity(order.len());
                                                for &ci in &order {
                                                    let gv: Vec<Real> = rows_per_combo[ci].iter()
                                                        .map(|&r| vals.get(r).copied().unwrap_or(None)).collect();
                                                    let res = self.call_fn(&f, &[EvalArg { name: None, value: RVal::Numeric(gv.into(), Attrs::default()) }], env)?;
                                                    agg.push(res.scalar_f64().unwrap_or(None));
                                                }
                                                out_cols.push((rname.clone(), RVal::Numeric(agg.into(), Attrs::default())));
                                            }
                                            return Ok(RVal::DataFrame(DataFrame { columns: out_cols, row_names: None }));
                                        }
                                        // boxplot(y ~ g, data = df): split the response by the
                                        // grouping column and hand one named vector per group to
                                        // the graphics boxplot (its multi-group form).
                                        if fname.as_ref() == "boxplot" {
                                            let (responses, groups) = self.formula_frame(lhs, rhs, df, env)?;
                                            if responses.len() == 1 && groups.len() == 1 {
                                                let y = self.as_reals(&responses[0].1)?;
                                                let glabels: Vec<String> = match &groups[0].1 {
                                                    RVal::Factor(f) => f.codes.iter().map(|c|
                                                        c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string())).unwrap_or_default()).collect(),
                                                    RVal::Character(v, _) => v.iter().map(|x|
                                                        x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
                                                    other => self.as_reals(other)?.iter().map(|x|
                                                        x.map(fmt_num).unwrap_or_default()).collect(),
                                                };
                                                let mut levels: Vec<String> = Vec::new();
                                                for l in &glabels { if !levels.contains(l) { levels.push(l.clone()); } }
                                                levels.sort();
                                                let mut ea: Vec<EvalArg> = Vec::new();
                                                for lvl in &levels {
                                                    let vals: Vec<Real> = y.iter().zip(glabels.iter())
                                                        .filter(|(_, gl)| *gl == lvl).map(|(yi, _)| *yi).collect();
                                                    ea.push(EvalArg { name: Some(Arc::from(lvl.as_str())), value: RVal::Numeric(vals.into(), Attrs::default()) });
                                                }
                                                // Carry styling args (main/col/…), skip formula + data.
                                                for arg in args.iter().skip(1) {
                                                    if arg.name.as_deref() == Some("data") { continue; }
                                                    if arg.name.is_some() {
                                                        ea.push(EvalArg { name: arg.name.clone(), value: self.eval_in(&arg.value, env)? });
                                                    }
                                                }
                                                return self.call_fn(&RVal::BuiltinFn(Arc::from("boxplot")), &ea, env);
                                            }
                                        }

                                        // Check for dot (.) on RHS — means "all other columns"
                                        let is_dot_rhs = matches!(rhs.as_ref(), Expr::Symbol(s) if s.as_ref() == ".");

                                        if is_dot_rhs {
                                            // y ~ . means y = lhs column, x = all other columns
                                            let lhs_name = match lhs.as_ref() {
                                                Expr::Symbol(s) => s.clone(),
                                                _ => return err!(Runtime, "formula LHS must be a column name"),
                                            };
                                            // Extract y
                                            let y_col = df.get_col(&lhs_name).ok_or(R2Err{msg:format!("column '{}' not found", lhs_name),kind:ErrKind::Runtime})?;

                                            // Build x matrix from all OTHER numeric columns
                                            let nrow = df.nrow();
                                            let mut x_data = Vec::new();
                                            let mut x_names = Vec::new();
                                            let mut ncol = 0;
                                            for (cn, cv) in &df.columns {
                                                if cn.as_ref() == lhs_name.as_ref() { continue; }
                                                if let Ok(vals) = self.as_reals(cv) {
                                                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                                                    if nums.len() == nrow { x_data.extend(nums); x_names.push(cn.clone()); ncol += 1; }
                                                }
                                            }
                                            let mut mat = Matrix::new(x_data, nrow, ncol);
                                            mat.col_names = Some(x_names.clone());
                                            let x_mat = RVal::Matrix(mat);

                                            // For lm/glm: use formula path
                                            if matches!(fname.as_ref(), "lm" | "glm") {
                                                let formula = RVal::List(vec![
                                                    (Some(Arc::from("~lhs")), y_col.clone()),
                                                    (Some(Arc::from("~rhs")), x_mat),
                                                    (Some(Arc::from("~class")), rstr("formula")),
                                                ]);
                                                let f = self.eval_in(func, env)?;
                                                let mut ea = vec![EvalArg { name: None, value: formula }];
                                                for a in args.iter().skip(1) { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); }
                                                // Capture original call for `$call` field.
                                                ea.push(EvalArg { name: Some(Arc::from("_call")), value: rstr(&fmt_expr(&Expr::Call { func: func.clone(), args: args.to_vec() })) });
                                                return self.call_fn(&f, &ea, env);
                                            }
                                            // For ML functions: pass (x_matrix, y_vector, ...other args)
                                            let f = self.eval_in(func, env)?;
                                            let mut ea = vec![
                                                EvalArg { name: None, value: x_mat },
                                                EvalArg { name: None, value: y_col.clone() },
                                            ];
                                            for a in args.iter().skip(1) {
                                                if a.name.as_ref().map(|n| n.as_ref()) != Some("data") {
                                                    ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                                                }
                                            }
                                            return self.call_fn(&f, &ea, env);
                                        } else {
                                            // Named columns: resolve normally.
                                            // Phase R.S.1 — split out any Error(...) stratum first so
                                            // it does not get treated as a regular predictor. The
                                            // resulting `rhs_fixed` is the predictor expression with
                                            // the Error term removed; `error_stratum_expr` (if any)
                                            // is what was inside Error(...).
                                            // Phase R.S.3 — also split out (1|group) random-effect
                                            // specs after the Error split, so lmer-style formulas
                                            // like y ~ x + (1|subject) work cleanly.
                                            let (rhs_no_err, error_stratum_expr) = split_error_term(rhs);
                                            let (rhs_fixed, random_grouping_exprs) = split_random_effects(&rhs_no_err);
                                            let lhs_val = self.resolve_formula_term(lhs, df, env)?;
                                            let rhs_val = if matches!(rhs_fixed, Expr::NullLit) {
                                                RVal::Null
                                            } else {
                                                self.resolve_formula_term(&rhs_fixed, df, env)?
                                            };
                                            let mut formula_items = vec![
                                                (Some(Arc::from("~lhs")), lhs_val),
                                                (Some(Arc::from("~rhs")), rhs_val),
                                                (Some(Arc::from("~class")), rstr("formula")),
                                            ];
                                            if let Some(stratum_expr) = error_stratum_expr {
                                                let stratum_val = self.resolve_formula_term(&stratum_expr, df, env)?;
                                                formula_items.push((Some(Arc::from("~error")), stratum_val));
                                            }
                                            for group_expr in &random_grouping_exprs {
                                                let group_val = self.resolve_formula_term(group_expr, df, env)?;
                                                formula_items.push((Some(Arc::from("~random_intercept")), group_val));
                                            }
                                            let formula = RVal::List(formula_items);
                                            let f = self.eval_in(func, env)?;
                                            let mut ea = vec![EvalArg { name: None, value: formula }];
                                            for a in args.iter().skip(1) {
                                                ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                                            }
                                            // Capture original call for `$call` field on the
                                            // fitted-model TypeInstance (lm/glm/aov use it).
                                            ea.push(EvalArg { name: Some(Arc::from("_call")), value: rstr(&fmt_expr(&Expr::Call { func: func.clone(), args: args.to_vec() })) });
                                            return self.call_fn(&f, &ea, env);
                                        }
                                    }
                                }
                                // No data= arg: evaluate formula normally
                            }
                        }
                    }
                    // NSE for system.time: time the expression evaluation.
                    // The inner expression's value is intentionally discarded so it
                    // doesn't get auto-printed by the REPL (matches R's invisible()).
                    if matches!(fname.as_ref(), "system.time") {
                        if let Some(first_arg) = args.first() {
                            let start = std::time::Instant::now();
                            let _ = self.eval_in(&first_arg.value, env)?;
                            let elapsed = start.elapsed();
                            soutln!("   user  system elapsed");
                            soutln!("  {:.3}   0.000   {:.3}", elapsed.as_secs_f64(), elapsed.as_secs_f64());
                            return Ok(RVal::Null);
                        }
                    }
                }
                // Normal call: evaluate all arguments. `...` in the argument
                // list splices the caller's captured dots into this call.
                // Function-position lookup (R keeps fn/var namespaces apart):
                // `c <- c(1,2); c(3,4)` still calls the builtin `c`.
                let f = match self.resolve_call_target(func, env) {
                    Ok(f) => f,
                    Err(not_found) => {
                        // Method-dispatch fallback: `m(obj, …)` where `obj` is a
                        // typed instance and `method m(x: Type) …` is defined.
                        if let Expr::Symbol(name) = func.as_ref() {
                            let has_method = self.methods.keys().any(|(mn, _)| mn.as_ref() == name.as_ref());
                            if has_method && !args.is_empty() {
                                let mut ea = Vec::new();
                                for a in args {
                                    if matches!(a.value, Expr::Dots) {
                                        if let Some(RVal::List(dots)) = self.lookup_dots(env) {
                                            for (nm, val) in dots { ea.push(EvalArg { name: nm, value: val }); }
                                        }
                                    } else {
                                        ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                                    }
                                }
                                if let Some(cl) = self.method_as_closure(name.as_ref(), &ea) {
                                    return self.call_fn(&RVal::Closure(cl), &ea, env);
                                }
                            }
                        }
                        return Err(not_found);
                    }
                };
                // NSE frame (Phase L.3): push the UNEVALUATED call only when
                // the target closure actually uses substitute/match.call/
                // sys.call. Gated, so normal closure calls clone nothing.
                let pushed_nse = match &f {
                    RVal::Closure(cl) if self.closure_uses_nse(&cl.body) => {
                        self.nse_stack.push(NseFrame {
                            call: Arc::new(Expr::Call { func: func.clone(), args: args.to_vec() }),
                            params: cl.params.clone(),
                        });
                        true
                    }
                    _ => false,
                };
                let mut ea = Vec::new();
                for a in args {
                    if matches!(a.value, Expr::Dots) {
                        if let Some(RVal::List(dots)) = self.lookup_dots(env) {
                            for (nm, val) in dots { ea.push(EvalArg { name: nm, value: val }); }
                        }
                    } else {
                        match self.eval_in(&a.value, env) {
                            Ok(v) => ea.push(EvalArg { name: a.name.clone(), value: v }),
                            Err(e) => { if pushed_nse { self.nse_stack.pop(); } return Err(e); }
                        }
                    }
                }
                let result = self.call_fn(&f, &ea, env);
                if pushed_nse { self.nse_stack.pop(); }
                result
            }
            Expr::Pipe { lhs, rhs } => {
                let lv = self.eval_in(lhs, env)?;
                match rhs.as_ref() {
                    Expr::Call { func, args } => { let f = self.resolve_call_target(func, env)?; let mut ea = vec![EvalArg { name: None, value: lv }]; for a in args { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); } self.call_fn(&f, &ea, env) }
                    _ => err!(Runtime, "|> rhs must be a function call"),
                }
            }
            Expr::Index { object, indices } => { let obj = self.eval_in(object, env)?; let mut ei = Vec::new(); for i in indices { ei.push(match i { Some(e) => Some(self.eval_in(e, env)?), None => None }); } self.index_obj(&obj, &ei) }
            Expr::DblIndex { object, index } => { let obj = self.eval_in(object, env)?; let idx = self.eval_in(index, env)?; self.dbl_index(&obj, &idx) }
            Expr::Dollar { object, field } => { let obj = self.eval_in(object, env)?; self.dollar(&obj, field) }
            Expr::Namespace { pkg, name } => {
                // pkg::func() — direct namespace access, bypasses search order
                if self.registry.resolve_in_package(pkg, name).is_some() {
                    // Encode as "pkg::name" so call_fn knows to resolve in specific package
                    Ok(RVal::BuiltinFn(Arc::from(format!("{}::{}", pkg, name).as_str())))
                } else {
                    // Package might not be loaded — try loading namespace only
                    err!(Runtime, "'{}' not found in package '{}' (is it loaded?)", name, pkg)
                }
            }
            Expr::If { cond, then, else_ } => { let c = self.eval_in(cond, env)?; if self.truthy(&c)? { self.eval_in(then, env) } else if let Some(e) = else_ { self.eval_in(e, env) } else { Ok(RVal::Null) } }
            Expr::For { var, iter, body } => {
                // Phase R.T.4-fix — top-level for-loops must re-snapshot env
                // from `self.global_env` each iteration, because subscript
                // assignments (`x[i] <- ...`) write through `env_insert` which
                // replaces the Arc; the body's captured env would otherwise
                // see the pre-loop value of every variable on each iteration.
                // Inside function bodies, writes go to `local_scopes`, which
                // Symbol-lookup checks first, so the original env still works.
                let iv = self.eval_in(iter, env)?;
                let at_top_level = self.local_scopes.is_empty();
                'each_item: for item in self.to_items(&iv)? {
                    self.scope_insert(var.clone(), item);
                    if at_top_level {
                        // Re-snapshot global_env before EACH statement. A single
                        // per-iteration snapshot made a variable that is both
                        // ASSIGNED and READ in the same iteration read its stale
                        // (previous-iteration) value, because a top-level
                        // assignment replaces the env's Arc and the snapshot
                        // kept pointing at the old one. Cloning per statement
                        // makes prior assignments in this iteration visible.
                        let stmts: &[Expr] = match body.as_ref() {
                            Expr::Block(s) => s,
                            single => std::slice::from_ref(single),
                        };
                        for stmt in stmts {
                            let genv = self.global_env.clone();
                            match self.eval_in(stmt, &genv) {
                                Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break 'each_item,
                                Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue 'each_item,
                                Err(e) => return Err(e),
                                _ => {}
                            }
                        }
                    } else {
                        match self.eval_in(body, env) {
                            Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                            Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                            Err(e) => return Err(e),
                            _ => {}
                        }
                    }
                }
                Ok(RVal::Null)
            }
            Expr::While { cond, body } => {
                // Same top-level re-snapshot rule as For: re-clone global_env
                // before the condition AND before each body statement, so an
                // assigned-then-read variable in one iteration isn't stale.
                let at_top_level = self.local_scopes.is_empty();
                loop {
                    if at_top_level {
                        let cenv = self.global_env.clone();
                        let c = self.eval_in(cond, &cenv)?;
                        if !self.truthy(&c)? { break; }
                        let stmts: &[Expr] = match body.as_ref() {
                            Expr::Block(s) => s,
                            single => std::slice::from_ref(single),
                        };
                        let mut do_break = false;
                        for stmt in stmts {
                            let genv = self.global_env.clone();
                            match self.eval_in(stmt, &genv) {
                                Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => { do_break = true; break; }
                                Err(R2Err { kind: ErrKind::CtrlNext, .. }) => break, // skip rest, recheck cond
                                Err(e) => return Err(e),
                                _ => {}
                            }
                        }
                        if do_break { break; }
                    } else {
                        let c = self.eval_in(cond, env)?;
                        if !self.truthy(&c)? { break; }
                        match self.eval_in(body, env) {
                            Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                            Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                            Err(e) => return Err(e),
                            _ => {}
                        }
                    }
                }
                Ok(RVal::Null)
            }
            Expr::Repeat { body } => {
                // `repeat { ... }` — loop forever until `break` (R semantics).
                // Same top-level per-statement re-snapshot as For/While.
                let at_top_level = self.local_scopes.is_empty();
                loop {
                    if at_top_level {
                        let stmts: &[Expr] = match body.as_ref() {
                            Expr::Block(s) => s,
                            single => std::slice::from_ref(single),
                        };
                        let mut do_break = false;
                        for stmt in stmts {
                            let genv = self.global_env.clone();
                            match self.eval_in(stmt, &genv) {
                                Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => { do_break = true; break; }
                                Err(R2Err { kind: ErrKind::CtrlNext, .. }) => break,
                                Err(e) => return Err(e),
                                _ => {}
                            }
                        }
                        if do_break { break; }
                    } else {
                        match self.eval_in(body, env) {
                            Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                            Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                            Err(e) => return Err(e),
                            _ => {}
                        }
                    }
                }
                Ok(RVal::Null)
            }
            Expr::Match { expr: e, arms } => { let val = self.eval_in(e, env)?; for arm in arms { for pat in &arm.patterns { let pv = self.eval_in(pat, env)?; if self.vals_eq(&val, &pv) { return self.eval_in(&arm.body, env); } } } err!(Runtime, "no matching pattern") }
            Expr::FuncDef { params, body } | Expr::Lambda { params, body } => Ok(RVal::Closure(Closure { params: params.clone(), body: Arc::new((**body).clone()), env: env.clone() })),
            Expr::TypeDef { name, fields, parent } => { let td = TypeDef { name: name.clone(), fields: fields.clone(), parent: parent.clone() }; self.types.insert(name.clone(), td.clone()); env_insert(&mut self.global_env, name.clone(), RVal::TypeDef(td.clone())); Ok(RVal::TypeDef(td)) }
            Expr::MethodDef(m) => { self.methods.insert((m.name.clone(), m.type_name.clone()), m.clone()); Ok(RVal::Null) }
            Expr::TryCatch { body, var, catch } => { match self.eval_in(body, env) { Ok(v) => Ok(v), Err(e) => { self.scope_insert(var.clone(), rstr(&e.msg)); self.eval_in(catch, env) } } }
            Expr::Return(v) => { let val = self.eval_in(v, env)?; Err(R2Err { msg: String::new(), kind: ErrKind::CtrlReturn(Box::new(val)) }) }
            Expr::Break => Err(R2Err { msg: String::new(), kind: ErrKind::CtrlBreak }),
            Expr::Next => Err(R2Err { msg: String::new(), kind: ErrKind::CtrlNext }),
            Expr::Dots => Ok(self.lookup_dots(env).unwrap_or(RVal::Null)),
        }
    }

    /// `bquote` walker: quote `e`, but evaluate any `.(x)` sub-expression in
    /// `env` and splice its value back in as a literal. (Phase L.3.)
    fn bquote_walk(&mut self, e: &Expr, env: &EnvRef) -> Result<Expr, R2Err> {
        // .(x) — the unquote escape.
        if let Expr::Call { func, args } = e {
            if matches!(func.as_ref(), Expr::Symbol(s) if s.as_ref() == "." ) && args.len() == 1 {
                let v = self.eval_in(&args[0].value, env)?;
                return builtins::lang::value_to_expr(&v);
            }
        }
        Ok(match e {
            Expr::Binary { op, lhs, rhs } =>
                Expr::Binary { op: *op, lhs: Box::new(self.bquote_walk(lhs, env)?), rhs: Box::new(self.bquote_walk(rhs, env)?) },
            Expr::Unary { op, expr } =>
                Expr::Unary { op: *op, expr: Box::new(self.bquote_walk(expr, env)?) },
            Expr::Call { func, args } => {
                let func = Box::new(self.bquote_walk(func, env)?);
                let mut nargs = Vec::with_capacity(args.len());
                for a in args { nargs.push(r2_types::CallArg { name: a.name.clone(), value: self.bquote_walk(&a.value, env)? }); }
                Expr::Call { func, args: nargs }
            }
            Expr::Index { object, indices } => {
                let object = Box::new(self.bquote_walk(object, env)?);
                let mut idx = Vec::with_capacity(indices.len());
                for i in indices { idx.push(match i { Some(e) => Some(self.bquote_walk(e, env)?), None => None }); }
                Expr::Index { object, indices: idx }
            }
            Expr::DblIndex { object, index } =>
                Expr::DblIndex { object: Box::new(self.bquote_walk(object, env)?), index: Box::new(self.bquote_walk(index, env)?) },
            Expr::Dollar { object, field } =>
                Expr::Dollar { object: Box::new(self.bquote_walk(object, env)?), field: field.clone() },
            Expr::Pipe { lhs, rhs } =>
                Expr::Pipe { lhs: Box::new(self.bquote_walk(lhs, env)?), rhs: Box::new(self.bquote_walk(rhs, env)?) },
            other => other.clone(),
        })
    }

    /// The top NSE frame's (call, params), cloned out — for `match.call()`
    /// / `sys.call()`. `None` when not inside an NSE-using function.
    pub(crate) fn current_nse_frame(&self) -> Option<(Arc<Expr>, Vec<r2_types::Param>)> {
        self.nse_stack.last().map(|f| (f.call.clone(), f.params.clone()))
    }

    /// Resolve the target of a call. R keeps function and variable lookup
    /// separate: in call position `name(...)`, a non-function binding named
    /// `name` is SKIPPED in favour of a function of that name. So
    /// `c <- c(1,2); c(3,4)` still calls the builtin `c`. For a Symbol we
    /// prefer the first *callable* binding (closure/builtin) up the scope
    /// chain, then the builtin registry; otherwise fall back to normal eval
    /// (which yields the proper "not callable"/"not found" error).
    fn resolve_call_target(&mut self, func: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        if let Expr::Symbol(name) = func {
            for scope in self.local_scopes.iter().rev() {
                if let Some(v) = scope.get(name.as_ref()) {
                    if matches!(v, RVal::Closure(_) | RVal::BuiltinFn(_)) { return Ok(v.clone()); }
                }
            }
            if let Some(v) = env.lookup(name) {
                if matches!(v, RVal::Closure(_) | RVal::BuiltinFn(_)) { return Ok(v.clone()); }
            }
            if let Some(v) = self.global_env.lookup(name) {
                if matches!(v, RVal::Closure(_) | RVal::BuiltinFn(_)) { return Ok(v.clone()); }
            }
            if self.registry.resolve(name.as_ref()).is_some() {
                return Ok(RVal::BuiltinFn(name.clone()));
            }
        }
        self.eval_in(func, env)
    }

    /// Method dispatch: if `name` is a `method name(x: Type) …` defined for
    /// the first argument's type (or an ancestor type via `extends`), build
    /// a synthetic closure (param = the object, then the method's extra
    /// params) so the normal closure-call machinery binds and runs the body.
    /// Returns `None` if the first arg isn't a typed instance or no method
    /// matches — the caller then falls back to its "object not found" error.
    fn method_as_closure(&self, name: &str, ea: &[EvalArg]) -> Option<Closure> {
        let inst = match ea.first().map(|a| &a.value) {
            Some(RVal::TypeInstance(i)) => i,
            _ => return None,
        };
        let mut tname = Some(inst.type_name.clone());
        while let Some(t) = tname {
            if let Some(m) = self.methods.get(&(Arc::from(name), t.clone())) {
                let mut params = vec![Param { name: m.param_name.clone(), default: None, dots: false }];
                params.extend(m.extra_params.iter().cloned());
                return Some(Closure {
                    params,
                    body: Arc::new((*m.body).clone()),
                    env: self.global_env.clone(),
                });
            }
            tname = self.types.get(&t).and_then(|td| td.parent.clone());
        }
        None
    }

    /// Does this closure body use NSE (substitute/match.call/sys.call)?
    /// Cached by the body's Arc pointer — computed once per unique body,
    /// then a single HashMap lookup per closure call. Builtin calls never
    /// reach here, so ordinary arithmetic/aggregation pays nothing.
    fn closure_uses_nse(&mut self, body: &Arc<Expr>) -> bool {
        let key = Arc::as_ptr(body) as usize;
        if let Some((arc, flag)) = self.nse_cache.get(&key) {
            if Arc::ptr_eq(arc, body) { return *flag; }
        }
        let flag = builtins::lang::expr_uses_nse(body);
        self.nse_cache.insert(key, (body.clone(), flag));
        flag
    }

    pub(crate) fn call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        match func {
            RVal::BuiltinFn(name) => {
                // Check for pkg::func namespaced call
                if let Some(sep) = name.find("::") {
                    let pkg = &name[..sep];
                    let fname = &name[sep+2..];
                    if let Some(f) = self.registry.resolve_in_package(pkg, fname) {
                        return f(self, args, env);
                    } else {
                        return err!(Runtime, "'{}' not found in package '{}'", fname, pkg);
                    }
                }
                // Normal resolution through search order
                if let Some((f, _pkg)) = self.registry.resolve(name.as_ref()) { f(self, args, env) }
                else { err!(Runtime, "unknown function '{}'", name) }
            }
            RVal::Closure(cl) => {
                // Recursion depth limit
                if self.local_scopes.len() >= 500 {
                    return err!(Runtime, "recursion depth limit exceeded (max 500). Use iteration instead.");
                }

                // ── JIT fast path (Phases C.2 + C.3) ────────────────────────
                if self.jit_enabled
                   && cl.params.len() == args.len()
                   && cl.params.iter().all(|p| !p.dots && p.default.is_none())
                {
                    // Resolve cache: try compile if not yet attempted.
                    let key = Arc::as_ptr(&cl.body) as usize;
                    let handle = match self.jit_cache.get(&key) {
                        // Only a true hit if the retained Arc is the SAME body
                        // (guards against a recycled pointer).
                        Some((body, slot)) if Arc::ptr_eq(body, &cl.body) => slot.clone(),
                        _ => {
                            // Never JIT a function whose body defines/returns a
                            // closure — the IR has no representation for it and
                            // would mis-compile it to a numeric (silent wrong
                            // result). Such functions aren't numeric hot loops
                            // anyway. Computed once per body, then cached.
                            let h = if body_defines_closure(&cl.body) { None } else { r2_jit::try_compile_closure(cl) };
                            self.jit_cache.insert(key, (cl.body.clone(), h.clone()));
                            h
                        }
                    };
                    if let Some(h) = handle {
                        // ── JIT NA-aware zero-copy bridge (Phase F.3 unlock) ──
                        //
                        // Pre-F.3, every JIT call did:
                        //   1. allocate Vec<f64> from Vec<Option<f64>>, encoding None → NaN
                        //   2. run Cranelift loop on raw f64
                        //   3. allocate Vec<Option<f64>> from output, decoding NaN → None
                        // Two O(n) allocation passes and per-element branches both ways.
                        //
                        // Now: RVal::Numeric is Reals which caches an Arc<ColumnarF64>.
                        // - `col.values()` returns &[f64] — dense, contiguous, SIMD-friendly,
                        //   zero alloc (just a slice into existing buffer).
                        // - Cranelift loop still operates on raw f64 (NaN propagates correctly).
                        // - On the way out, we reconstruct Vec<Option<f64>> respecting the
                        //   INPUT bitmap rather than scanning the output for NaN: NA structure
                        //   is preserved exactly, not approximated via NaN encoding.
                        //
                        // Win: 1 alloc round-trip instead of 2, SIMD-friendly dense input,
                        // and structurally-correct NA semantics (NaN ≠ NA distinction kept).
                        match h.kind() {
                            r2_types::JitKind::Vector1ToScalar => {
                                if args.len() == 1 {
                                    if let RVal::Numeric(v, _) = &args[0].value {
                                        // Zero-copy: grab the cached columnar's dense f64 slice.
                                        // Reads None as NaN in the values buffer (already that way
                                        // by ColumnarF64::from_option_slice), so Cranelift's NaN
                                        // arithmetic propagates correctly through the reduction.
                                        // Empty → interpreter: the indexed-loop kernel uses R's
                                        // `1:length` loop, and `1:0` would step out of bounds.
                                        if !v.is_empty() {
                                        let col = v.columnar();
                                        let values = col.values();
                                        let out = unsafe { h.try_call_vec1(values.as_ptr(), values.len() as i64) };
                                        if let Some(val) = out {
                                            return Ok(RVal::Numeric(vec![Some(val)].into(), Attrs::default()));
                                        }
                                        }
                                    } else if let RVal::Matrix(m) = &args[0].value {
                                        // J.3 — a matrix's column-major buffer IS a dense f64
                                        // vector for whole-array reductions (sum(m), sum(m*m), …).
                                        // NaN stands in for NA and propagates correctly.
                                        if !m.data.is_empty() {
                                            let out = unsafe { h.try_call_vec1(m.data.as_ptr(), m.data.len() as i64) };
                                            if let Some(val) = out {
                                                return Ok(RVal::Numeric(vec![Some(val)].into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::Vector2ToScalar => {
                                // Phase J.2 — fused binary map-reduce (e.g. sum(x*w)).
                                if args.len() == 2 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _)) = (&args[0].value, &args[1].value) {
                                        if a.len() == b.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let out = unsafe { h.try_call_vec2(a_vals.as_ptr(), b_vals.as_ptr(), a.len() as i64) };
                                            if let Some(val) = out {
                                                return Ok(RVal::Numeric(vec![Some(val)].into(), Attrs::default()));
                                            }
                                        }
                                    } else if let (RVal::Matrix(a), RVal::Matrix(b)) = (&args[0].value, &args[1].value) {
                                        // J.3 — two same-shaped matrices' flat buffers (e.g. the
                                        // Frobenius inner product sum(A*B) over equal-dim matrices).
                                        if a.data.len() == b.data.len() && !a.data.is_empty() {
                                            let out = unsafe { h.try_call_vec2(a.data.as_ptr(), b.data.as_ptr(), a.data.len() as i64) };
                                            if let Some(val) = out {
                                                return Ok(RVal::Numeric(vec![Some(val)].into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorBinaryMap => {
                                // Two equal-length vectors → output preserves AND-of-bitmaps.
                                if args.len() == 2 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _)) = (&args[0].value, &args[1].value) {
                                        if a.len() == b.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; a.len()];
                                            let ok = unsafe { h.try_call_vec_binary(a_vals.as_ptr(), b_vals.as_ptr(), out_buf.as_mut_ptr(), a.len() as i64) };
                                            if ok {
                                                let a_bits = a_col.valid_bits();
                                                let b_bits = b_col.valid_bits();
                                                let result = combine_binary_output(&out_buf, a_bits, b_bits);
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    } else if let (RVal::Matrix(a), RVal::Matrix(b)) = (&args[0].value, &args[1].value) {
                                        // J.3 — element-wise op over two same-shaped matrices
                                        // (e.g. A + B, A * B). Result keeps A's dim + dimnames.
                                        if a.data.len() == b.data.len() && a.nrow == b.nrow && !a.data.is_empty() {
                                            let mut out_buf: Vec<f64> = vec![0.0; a.data.len()];
                                            let ok = unsafe { h.try_call_vec_binary(a.data.as_ptr(), b.data.as_ptr(), out_buf.as_mut_ptr(), a.data.len() as i64) };
                                            if ok {
                                                return Ok(RVal::Matrix(r2_types::Matrix {
                                                    data: out_buf, nrow: a.nrow, ncol: a.ncol,
                                                    col_names: a.col_names.clone(), row_names: a.row_names.clone(),
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorTernaryMap => {
                                // Three equal-length numeric vectors → vector.
                                // Output bitmap = AND of all three input bitmaps.
                                if args.len() == 3 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _), RVal::Numeric(c, _)) =
                                        (&args[0].value, &args[1].value, &args[2].value)
                                    {
                                        if a.len() == b.len() && b.len() == c.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let c_col = c.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let c_vals = c_col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; a.len()];
                                            let ok = unsafe { h.try_call_vec_ternary(a_vals.as_ptr(), b_vals.as_ptr(), c_vals.as_ptr(), out_buf.as_mut_ptr(), a.len() as i64) };
                                            if ok {
                                                let result = combine_ternary_output(&out_buf, a_col.valid_bits(), b_col.valid_bits(), c_col.valid_bits());
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorMap => {
                                // Element-wise vector → vector. Output bitmap = input bitmap.
                                if args.len() == 1 {
                                    if let RVal::Numeric(v, _) = &args[0].value {
                                        let col = v.columnar();
                                        let values = col.values();
                                        let mut out_buf: Vec<f64> = vec![0.0; values.len()];
                                        let ok = unsafe { h.try_call_vec_map(values.as_ptr(), out_buf.as_mut_ptr(), values.len() as i64) };
                                        if ok {
                                            let bits = col.valid_bits();
                                            let result = combine_unary_output(&out_buf, bits);
                                            return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                        }
                                    } else if let RVal::Matrix(m) = &args[0].value {
                                        // J.3 — element-wise map over a matrix (e.g. sqrt(m), m*m).
                                        // R keeps dim + dimnames; NaN carries NA through the buffer.
                                        if !m.data.is_empty() {
                                            let mut out_buf: Vec<f64> = vec![0.0; m.data.len()];
                                            let ok = unsafe { h.try_call_vec_map(m.data.as_ptr(), out_buf.as_mut_ptr(), m.data.len() as i64) };
                                            if ok {
                                                return Ok(RVal::Matrix(r2_types::Matrix {
                                                    data: out_buf, nrow: m.nrow, ncol: m.ncol,
                                                    col_names: m.col_names.clone(), row_names: m.row_names.clone(),
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::IndexedStoreMap1 => {
                                // J.3 imperative store map: one input vector →
                                // output vector (engine allocates the buffer).
                                if args.len() == 1 {
                                    if let RVal::Numeric(v, _) = &args[0].value {
                                        if !v.is_empty() {
                                            let col = v.columnar();
                                            let values = col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; values.len()];
                                            let ok = unsafe { h.try_call_ixstore1(values.as_ptr(), out_buf.as_mut_ptr(), values.len() as i64) };
                                            if ok {
                                                let result = combine_unary_output(&out_buf, col.valid_bits());
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::IndexedStoreMap2 => {
                                if args.len() == 2 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _)) = (&args[0].value, &args[1].value) {
                                        if a.len() == b.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; a.len()];
                                            let ok = unsafe { h.try_call_ixstore2(a_vals.as_ptr(), b_vals.as_ptr(), out_buf.as_mut_ptr(), a.len() as i64) };
                                            if ok {
                                                let result = combine_binary_output(&out_buf, a_col.valid_bits(), b_col.valid_bits());
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::Scalar => {
                                let mut farg: Vec<f64> = Vec::with_capacity(args.len());
                                let mut all_scalar = true;
                                for ea in args {
                                    match &ea.value {
                                        RVal::Numeric(v, _) if v.len() == 1 => match v[0] { Some(x) => farg.push(x), None => { all_scalar = false; break; } },
                                        RVal::Integer(v, _) if v.len() == 1 => match v[0] { Some(x) => farg.push(x as f64), None => { all_scalar = false; break; } },
                                        RVal::Logical(v, _) if v.len() == 1 => match v[0] { Some(b) => farg.push(if b { 1.0 } else { 0.0 }), None => { all_scalar = false; break; } },
                                        _ => { all_scalar = false; break; }
                                    }
                                }
                                if all_scalar {
                                    if let Some(out) = h.try_call_real(&farg) {
                                        return Ok(RVal::Numeric(vec![Some(out)].into(), Attrs::default()));
                                    }
                                }
                            }
                        }
                    }
                }
                // ── Fallback: tree-walking interpreter (existing path) ──────
                let mut ce = Env::new_child(cl.env.clone(), None);
                let m = Arc::make_mut(&mut ce);
                // R-style argument matching with `...`: named args bind to
                // formals by name; positional args fill the formals before
                // `...`; everything left over is collected into `...` (bound
                // as a List of (name, value) under the key "...").
                let dots_pos = cl.params.iter().position(|p| p.dots);
                let mut used = vec![false; args.len()];
                for p in cl.params.iter().filter(|p| !p.dots) {
                    if let Some(j) = (0..args.len()).find(|&j| !used[j] && args[j].name.as_deref() == Some(p.name.as_ref())) {
                        used[j] = true;
                        m.bindings.insert(p.name.clone(), args[j].value.clone());
                    }
                }
                let before = dots_pos.unwrap_or(cl.params.len());
                let mut pos = 0usize;
                for (i, p) in cl.params.iter().enumerate() {
                    if p.dots || i >= before || m.bindings.contains_key(p.name.as_ref()) { continue; }
                    while pos < args.len() && (used[pos] || args[pos].name.is_some()) { pos += 1; }
                    if pos < args.len() { used[pos] = true; m.bindings.insert(p.name.clone(), args[pos].value.clone()); pos += 1; }
                }
                if dots_pos.is_some() {
                    let dots: Vec<(Option<Arc<str>>, RVal)> = (0..args.len())
                        .filter(|&j| !used[j])
                        .map(|j| (args[j].name.clone(), args[j].value.clone()))
                        .collect();
                    m.bindings.insert(Arc::from("..."), RVal::List(dots));
                }
                // Record the call's argument count for nargs().
                m.bindings.insert(Arc::from(".nargs"), RVal::Integer(vec![Some(args.len() as i32)].into(), Attrs::default()));
                for p in cl.params.iter().filter(|p| !p.dots) {
                    if !m.bindings.contains_key(p.name.as_ref()) {
                        let v = p.default.as_ref().and_then(|d| self.eval_in(d, env).ok()).unwrap_or(RVal::Null);
                        m.bindings.insert(p.name.clone(), v);
                    }
                }
                let func_env = Arc::new(m.clone());
                self.local_scopes.push(HashMap::new());
                let result = match self.eval_in(&cl.body, &func_env) { Err(R2Err { kind: ErrKind::CtrlReturn(v), .. }) => Ok(*v), r => r };
                self.local_scopes.pop();
                result
            }
            RVal::TypeDef(td) => {
                // Gather fields parent-first along the `extends` chain so an
                // inheriting type's constructor accepts and stores inherited
                // fields (not just its own). Positional args fill them in
                // declaration order: ancestors first, then this type's fields.
                let mut chain: Vec<Arc<str>> = Vec::new();
                let mut cur = td.parent.clone();
                while let Some(p) = cur {
                    match self.types.get(&p) {
                        Some(ptd) => { chain.push(p.clone()); cur = ptd.parent.clone(); }
                        None => break,
                    }
                }
                let mut ordered: Vec<(Arc<str>, Option<RVal>)> = Vec::new();
                for anc in chain.iter().rev() {
                    if let Some(ptd) = self.types.get(anc) {
                        for fd in &ptd.fields { ordered.push((fd.name.clone(), fd.default.clone())); }
                    }
                }
                for fd in &td.fields { ordered.push((fd.name.clone(), fd.default.clone())); }
                let mut fields = HashMap::new();
                for (i, (fname, fdefault)) in ordered.iter().enumerate() {
                    let v = self.get_arg(args, i, fname).or_else(|| fdefault.clone()).unwrap_or(RVal::Null);
                    fields.insert(fname.clone(), v);
                }
                Ok(RVal::TypeInstance(TypeInstance { type_name: td.name.clone(), fields }))
            }
            _ => err!(Runtime, "not callable as a function. Check spelling or use help() to find the right function name"),
        }
    }
    fn get_arg(&self, args: &[EvalArg], pos: usize, name: &str) -> Option<RVal> {
        args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone()).or_else(|| args.get(pos).map(|a| a.value.clone()))
    }
    /// Resolve the captured `...` (a List of (name,value)) in the current
    /// scope, if any. Used to expand `...` / `..N` in function bodies.
    fn lookup_dots(&self, env: &EnvRef) -> Option<RVal> {
        for scope in self.local_scopes.iter().rev() {
            if let Some(v) = scope.get("...") { return Some(v.clone()); }
        }
        let key: Arc<str> = Arc::from("...");
        env.lookup(&key).cloned()
    }

}
