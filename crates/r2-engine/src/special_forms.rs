//! R's special forms — calls that see their arguments UNEVALUATED (NSE),
//! so they must be intercepted before the ordinary call path evaluates
//! them: `quote`, `substitute`, `bquote`, `rm`, `missing`, `local`,
//! `with`, `switch`, `tryCatch`, `curve`, `system.time`, the bare-symbol
//! package calls (`library(stats)`), the data-frame scopes of `subset` /
//! `transform`, column naming in `data.frame(y, x1)`, and the formula
//! interface of the model functions (`formula_call.rs`).

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::builtins;
use crate::Engine;

impl Engine {
    /// Evaluate `fname(args)` as a special form. `Ok(None)` means it is not
    /// one — or its conditions do not hold (`missing` of a non-symbol,
    /// `subset` of a non-data-frame, a formula call without `data=`) — and
    /// the call goes down the ordinary path.
    pub(crate) fn eval_special_form(&mut self, fname: &str, func: &Expr, args: &[CallArg],
                                    env: &EnvRef) -> Result<Option<RVal>, R2Err> {
        let v = match fname {
            "library" | "detach" | "require" | "data" | "help" =>
                self.nse_package_call(func, args, env)?,
            "subset" | "transform" => return self.nse_in_frame(func, args, env),
            // quote(e) — return the argument UNEVALUATED as a language
            // object (RVal::Lang). The keystone of Phase L.1.
            "quote" => args.first()
                .map(|a| RVal::Lang(Arc::new(a.value.clone())))
                .unwrap_or(RVal::Null),
            "rm" => self.nse_rm(args, env),
            "substitute" => self.nse_substitute(args),
            // bquote(e) — quote e, but splice in any `.(x)` evaluated in
            // the current environment (quasiquotation). (Phase L.3.)
            "bquote" => {
                let arg = args.first().map(|a| a.value.clone()).unwrap_or(Expr::NullLit);
                RVal::Lang(Arc::new(self.bquote_walk(&arg, env)?))
            }
            "data.frame" => self.nse_data_frame(func, args, env)?,
            "tryCatch" => self.nse_try_catch(args, env)?,
            "missing" if args.len() == 1 => match &args[0].value {
                Expr::Symbol(pn) => self.nse_missing(pn, env),
                _ => return Ok(None),
            },
            "local" if args.len() == 1 && self.registry.resolve("local").is_none() =>
                self.nse_local(&args[0].value, env)?,
            "with" if args.len() >= 2 => self.nse_with(args, env)?,
            "switch" if !args.is_empty() => self.nse_switch(args, env)?,
            "curve" if !args.is_empty() => self.nse_curve(args, env)?,
            "lm" | "glm" | "t.test" | "rpart" | "rf" | "gbm" | "cv" | "aov" | "manova"
                | "lmer" | "aggregate" | "boxplot" => return self.formula_call(fname, func, args, env),
            "system.time" if !args.is_empty() => self.nse_system_time(&args[0].value, env)?,
            // table(x) prints `x` as its header line (R's deparse.level = 1):
            // a bare variable passes its name on as `dnn`.
            "table" if args.len() == 1 && args[0].name.is_none() && matches!(args[0].value, Expr::Symbol(_)) => {
                let Expr::Symbol(name) = &args[0].value else { return Ok(None) };
                let ea = vec![
                    EvalArg { name: None, value: self.eval_in(&args[0].value, env)? },
                    EvalArg { name: Some(Arc::from("dnn")), value: RVal::Character(vec![Some(name.clone())], Attrs::default()) },
                ];
                self.call_fn(&RVal::BuiltinFn(Arc::from("table")), &ea, env)?
            }
            _ => return Ok(None),
        };
        Ok(Some(v))
    }

    /// library(stats), detach(stats), require(stats) accept bare symbols:
    /// a first argument that is a bare symbol naming no variable is the
    /// package name as a string, not evaluated.
    fn nse_package_call(&mut self, func: &Expr, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let f = self.eval_in(func, env)?;
        let mut ea = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let value = match &a.value {
                // A variable holding the name wins over the bare symbol.
                Expr::Symbol(sym) if i == 0 => env.lookup(sym).unwrap_or_else(|| rstr(sym)),
                e => self.eval_in(e, env)?,
            };
            ea.push(EvalArg { name: a.name.clone(), value });
        }
        self.call_fn(&f, &ea, env)
    }

    /// `subset(df, cond)` and `transform(df, name = expr)`: arg 2+
    /// expressions evaluate in a scope where df's columns are bound as
    /// variables. Without this, `subset(df, x > 2)` resolves `x` against
    /// the global env.
    fn nse_in_frame(&mut self, func: &Expr, args: &[CallArg], env: &EnvRef) -> Result<Option<RVal>, R2Err> {
        if args.len() < 2 { return Ok(None); }
        let df_val = self.eval_in(&args[0].value, env)?;
        let RVal::DataFrame(df) = &df_val else { return Ok(None) };
        // Child env that shadows globals with df columns.
        let child = Arc::new(Env {
            name: Some(Arc::from(".subset.env")),
            parent: Some(env.clone()),
            bindings: std::sync::RwLock::new(df.columns.iter()
                .map(|(n, v)| (n.clone(), v.clone())).collect()),
            locked: false,
        });
        let f = self.eval_in(func, env)?;
        let mut ea = vec![EvalArg { name: None, value: df_val.clone() }];
        for a in args.iter().skip(1) {
            let val = self.eval_in(&a.value, &child)?;
            ea.push(EvalArg { name: a.name.clone(), value: val });
        }
        self.call_fn(&f, &ea, env).map(Some)
    }

    /// rm(x, y, ...) / rm("x") / rm(list=c("a","b")): take the UNEVALUATED
    /// symbol/string names and delete the bindings.
    fn nse_rm(&mut self, args: &[CallArg], env: &EnvRef) -> RVal {
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
            for frame in &self.frames { frame.remove(nm.as_ref()); }
            self.global_env.remove(nm.as_ref());
        }
        RVal::Null
    }

    /// substitute(e) — e with the current function's parameter symbols
    /// replaced by the UNEVALUATED expressions the caller passed (R's
    /// promise stand-in). Outside a function, e unchanged. (Phase L.3.)
    fn nse_substitute(&self, args: &[CallArg]) -> RVal {
        let arg = args.first().map(|a| a.value.clone()).unwrap_or(Expr::NullLit);
        let result = if let Some(frame) = self.nse_stack.last() {
            let map = builtins::lang::match_call_args(&frame.call, &frame.params);
            builtins::lang::substitute_expr(&arg, &map)
        } else {
            arg
        };
        RVal::Lang(Arc::new(result))
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

    /// `data.frame(y, x1, x2)` — bare-symbol args become column names. R
    /// does this by inspecting the unevaluated call; we lift `Expr::Symbol`
    /// arg names into the EvalArg `name` slot when no explicit `name =` is
    /// given. Without this, `data.frame(y, x1, x2)` would produce columns
    /// V1/V2/V3 and `df[, c("x1","x2")]` would find nothing.
    fn nse_data_frame(&mut self, func: &Expr, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
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
        self.call_fn(&f, &ea, env)
    }

    /// `tryCatch(expr, error=function(e){...}, finally={...})` — the
    /// function form (distinct from the try/catch syntax). Eval expr; on a
    /// non-control error, call the `error=` handler with the message;
    /// always run `finally=`.
    fn nse_try_catch(&mut self, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
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
        out
    }

    /// `missing(arg)` — TRUE if the named parameter was not supplied in the
    /// current call (defaulted or absent).
    fn nse_missing(&self, pn: &str, env: &EnvRef) -> RVal {
        let miss = env.lookup(".missing")
            .or_else(|| self.frames.last().and_then(|f| f.lookup(".missing")));
        let hit = matches!(&miss, Some(RVal::Character(v, _))
            if v.iter().any(|n| n.as_ref().map(|s| s.as_ref()) == Some(pn)));
        rbool(hit)
    }

    /// `local(expr)` — evaluate in a FRESH child env of the current scope,
    /// so its assignments don't leak out but its closures capture the local
    /// frame (R semantics).
    fn nse_local(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        let child = Env::new_child(env.clone(), Some(".local.env"));
        self.frames.push(child.clone());
        let r = self.eval_in(expr, &child);
        self.frames.pop();
        r
    }

    /// `with(data, expr)` — evaluate `expr` in a scope where `data`'s
    /// columns / list elements are bound.
    fn nse_with(&mut self, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let data = self.eval_in(&args[0].value, env)?;
        let child = Arc::new(Env {
            name: Some(Arc::from(".with.env")),
            parent: Some(env.clone()),
            bindings: std::sync::RwLock::new(match &data {
                RVal::DataFrame(df) =>
                    df.columns.iter().map(|(n, v)| (n.clone(), v.clone())).collect(),
                RVal::List(items) => items.iter()
                    .filter_map(|(n, v)| n.clone().map(|nm| (nm, v.clone()))).collect(),
                _ => Default::default(),
            }),
            locked: false,
        });
        self.eval_in(&args[1].value, &child)
    }

    /// `switch(EXPR, ...)`: EXPR selects which branch expr to evaluate — by
    /// name (character) or position (numeric). The branch exprs are taken
    /// UNEVALUATED; only the chosen one runs.
    fn nse_switch(&mut self, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
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
                Ok(RVal::Null)
            }
            _ => {
                if let Some(n) = sel.scalar_f64().ok().flatten() {
                    let i = n as usize;
                    if i >= 1 && i <= branches.len() {
                        return self.eval_in(&branches[i - 1].value, env);
                    }
                }
                Ok(RVal::Null)
            }
        }
    }

    /// `curve(expr, from, to, n=101, add=FALSE, ...)`: the first arg is
    /// taken UNEVALUATED and evaluated with `x` bound to a sequence (R's
    /// vectorized model), so `curve(x^2, 0, 10)` works without lazy
    /// promises. A bare function (`curve(sin, ...)`) is applied to x.
    fn nse_curve(&mut self, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
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
            bindings: std::sync::RwLock::new(std::iter::once((Arc::from("x"), xs_val.clone())).collect()),
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
        if add {
            r2_graphics::overlays::bi_lines(&ea)
        } else {
            ea.push(EvalArg { name: Some(Arc::from("type")), value: rstr("l") });
            r2_graphics::plots::bi_plot(&ea)
        }
    }

    /// system.time(expr): time the evaluation. The value is intentionally
    /// discarded so the REPL doesn't auto-print it (R's invisible()).
    fn nse_system_time(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        let start = std::time::Instant::now();
        let _ = self.eval_in(expr, env)?;
        let elapsed = start.elapsed();
        soutln!("   user  system elapsed");
        soutln!("  {:.3}   0.000   {:.3}", elapsed.as_secs_f64(), elapsed.as_secs_f64());
        Ok(RVal::Null)
    }
}
