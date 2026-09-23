//! The tree-walking evaluator — `eval_in` (the main Expr match), loops,
//! the ordinary call path (`eval_call`: evaluate the arguments, resolve
//! the target) and R's visibility rules. Elsewhere: assignment
//! (`assign.rs`), the special forms that see their arguments unevaluated
//! (`special_forms.rs`, `formula_call.rs`), and calling a function value
//! (`call.rs`, with its compiled fast path in `jit_call.rs`).

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::{Engine, NseFrame, val_to_str, env_insert};
use crate::builtins;
use crate::err;

impl Engine {
    /// Evaluate `expr` and record R's visibility of the result in
    /// `self.visible`: assignments and loops are invisible; a call is
    /// whatever `call_fn` decided (a builtin by name, a user function by
    /// its body's last expression); blocks, `if`, `return` and friends
    /// carry the visibility of what they evaluated; any other value is
    /// visible.
    pub fn eval_in(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        let r = self.eval_node(expr, env)?;
        match expr {
            Expr::Assign { .. } | Expr::For { .. } | Expr::While { .. } | Expr::Repeat { .. }
                | Expr::TypeDef { .. } | Expr::MethodDef(_) => self.visible = false,
            Expr::Call { func, .. } => {
                // calls that never reach call_fn (the NSE forms handled in
                // eval_node) are still invisible if R's function is
                if let Expr::Symbol(s) = func.as_ref() {
                    if is_invisible_builtin(s) { self.visible = false; }
                }
            }
            Expr::Block(_) | Expr::If { .. } | Expr::Match { .. } | Expr::Return(_)
                | Expr::TryCatch { .. } | Expr::Pipe { .. } | Expr::Break | Expr::Next => {}
            _ => self.visible = true,
        }
        Ok(r)
    }

    fn eval_node(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
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
                // 1. Current call frame — locals + args live in the frame env,
                //    which is the head of the lexical chain.
                // 2. Env chain (enclosing closures → global root).
                if let Some(val) = env.lookup(name) { Ok(val) }
                // 3. Global env (for envs not rooted there, e.g. package envs)
                else if let Some(val) = self.global_env.lookup(name) { Ok(val) }
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
            Expr::Assign { target, value, superassign } => self.eval_assign(target, value, *superassign, env),
            Expr::Block(stmts) => { let mut r = RVal::Null; for s in stmts { r = self.eval_in(s, env)?; } Ok(r) }
            Expr::Binary { op, lhs, rhs } => {
                if *op == BinOp::Colon { let l = self.eval_in(lhs, env)?; let r = self.eval_in(rhs, env)?; return self.seq_colon(&l, &r); }
                if *op == BinOp::Tilde {
                    // Formula: y ~ x resolves both sides against the calling
                    // scope and stores a formula-list. lhs can be NULL for
                    // one-sided formulas (~x).
                    //
                    // The RHS goes through the formula-term resolver, NOT
                    // plain evaluation: `+` is a term separator in a
                    // formula, and evaluating `x1 + x2` arithmetically made
                    // `lm(y ~ x1 + x2)` (no data=) silently fit ONE
                    // predictor — the elementwise sum — with `x2` gone
                    // from the output. Same resolver the data= path uses,
                    // with no frame to shadow the scope.
                    let no_frame = DataFrame { columns: Vec::new(), row_names: None };
                    let l = self.resolve_formula_term(lhs, &no_frame, env)?;
                    let r = self.resolve_formula_term(rhs, &no_frame, env)?;
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
                if let Expr::Symbol(fname) = func.as_ref() {
                    if let Some(v) = self.eval_special_form(fname, func, args, env)? {
                        return Ok(v);
                    }
                }
                self.eval_call(func, args, env)
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
            Expr::If { cond, then, else_ } => { let c = self.eval_in(cond, env)?; if self.truthy(&c)? { self.eval_in(then, env) } else if let Some(e) = else_ { self.eval_in(e, env) } else { self.visible = false; Ok(RVal::Null) } }
            Expr::For { var, iter, body } => self.eval_for(var, iter, body, env),
            Expr::While { cond, body } => {
                loop {
                    let c = self.eval_in(cond, env)?;
                    if !self.truthy(&c)? { break; }
                    match self.eval_in(body, env) {
                        Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                        Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                        Err(e) => return Err(e),
                        _ => {}
                    }
                }
                Ok(RVal::Null)
            }
            Expr::Repeat { body } => {
                // `repeat { ... }` — loop forever until `break` (R semantics).
                loop {
                    match self.eval_in(body, env) {
                        Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                        Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                        Err(e) => return Err(e),
                        _ => {}
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

    /// `for (var in iter) body`.
    fn eval_for(&mut self, var: &Arc<str>, iter: &Expr, body: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        // Environments are interior-mutable: assignments write into
        // the live env, so the body always sees prior writes through
        // the same Arc — no re-snapshot machinery needed.
        //
        // `for (i in a:b)` never builds the range: `1:1e9` as a
        // vector is 4 GB, and the old path then boxed every element
        // into a `Vec<RVal>` — tens of GB for a loop whose body
        // sees one integer at a time. That is what R's ALTREP
        // avoids, and it is also why Esc could not interrupt such a
        // loop: the flag is polled per expression, and no
        // expression boundary is reached while a billion elements
        // are being allocated. Any other iterable is walked one
        // element at a time, again without a Vec of all of them.
        let run_body = |this: &mut Self, item: RVal| -> Result<bool, R2Err> {
            this.scope_insert(var.clone(), item);
            match this.eval_in(body, env) {
                Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => Ok(false),
                Err(R2Err { kind: ErrKind::CtrlNext, .. }) => Ok(true),
                Err(e) => Err(e),
                _ => Ok(true),
            }
        };
        if let Expr::Binary { op: BinOp::Colon, lhs, rhs } = iter {
            let l = self.eval_in(lhs, env)?;
            let r = self.eval_in(rhs, env)?;
            let na = || R2Err { msg: "NA in seq".into(), kind: ErrKind::Runtime };
            let from = self.scalar_f64(&l)?.ok_or_else(na)? as i64;
            let to = self.scalar_f64(&r)?.ok_or_else(na)? as i64;
            let step: i64 = if from <= to { 1 } else { -1 };
            let mut i = from;
            loop {
                if !run_body(self, RVal::Integer(vec![Some(i as i32)].into(), Attrs::default()))? { break; }
                if i == to { break; }
                i += step;
            }
            return Ok(RVal::Null);
        }
        let iv = self.eval_in(iter, env)?;
        let n = match &iv {
            RVal::Integer(v, _) => v.len(),
            RVal::Numeric(v, _) => v.len(),
            RVal::Character(v, _) => v.len(),
            RVal::List(v) => v.len(),
            RVal::DataFrame(df) => df.columns.len(),
            other => return err!(Runtime, "cannot iterate over {}", other.type_name()),
        };
        for k in 0..n {
            let item = match &iv {
                RVal::Integer(v, _) => RVal::Integer(vec![v[k]].into(), Attrs::default()),
                RVal::Numeric(v, _) => RVal::Numeric(vec![v[k]].into(), Attrs::default()),
                RVal::Character(v, _) => RVal::Character(vec![v[k].clone()], Attrs::default()),
                RVal::List(v) => v[k].1.clone(),
                RVal::DataFrame(df) => df.columns[k].1.clone(),
                _ => unreachable!(),
            };
            if !run_body(self, item)? { break; }
        }
        Ok(RVal::Null)
    }

    /// An ordinary call: evaluate all arguments, then dispatch. `...` in
    /// the argument list splices the caller's captured dots into this call.
    /// Function-position lookup (R keeps fn/var namespaces apart):
    /// `c <- c(1,2); c(3,4)` still calls the builtin `c`.
    fn eval_call(&mut self, func: &Expr, args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let f = match self.resolve_call_target(func, env) {
            Ok(f) => f,
            Err(not_found) => {
                // Method-dispatch fallback: `m(obj, …)` where `obj` is a
                // typed instance and `method m(x: Type) …` is defined.
                if let Expr::Symbol(name) = func {
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
        // NSE frame (Phase L.3): push the UNEVALUATED call only when the
        // target closure actually uses substitute/match.call/sys.call.
        // Gated, so normal closure calls clone nothing.
        let pushed_nse = match &f {
            RVal::Closure(cl) if self.closure_uses_nse(&cl.body) => {
                self.nse_stack.push(NseFrame {
                    call: Arc::new(Expr::Call { func: Box::new(func.clone()), args: args.to_vec() }),
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
            // Walk the env chain FRAME BY FRAME, keeping only a callable
            // binding at each level (R skips non-function bindings in call
            // position: `c <- c(1,2); c(3,4)` still calls the builtin).
            let mut cur = Some(env.clone());
            while let Some(e) = cur {
                if let Some(v) = e.bindings.read().unwrap().get(name.as_ref()) {
                    if matches!(v, RVal::Closure(_) | RVal::BuiltinFn(_)) { return Ok(v.clone()); }
                }
                cur = e.parent.clone();
            }
            if let Some(v) = self.global_env.lookup(name) {
                if matches!(v, RVal::Closure(_) | RVal::BuiltinFn(_)) { return Ok(v); }
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

    /// Resolve the captured `...` (a List of (name,value)) in the current
    /// scope, if any. Used to expand `...` / `..N` in function bodies.
    fn lookup_dots(&self, env: &EnvRef) -> Option<RVal> {
        // `...` is bound in the call frame, which heads the env chain.
        env.lookup("...").or_else(|| self.frames.last().and_then(|f| f.lookup("...")))
    }

}

// ── visibility ───────────────────────────────────────────────────────

impl Engine {
    /// A builtin just returned: its value is visible unless R's function
    /// returns invisibly — except the functions that evaluate user code and
    /// return what it returned (`eval`, `tryCatch`, `switch`, …), which
    /// keep the visibility that code left, as they do in R.
    pub(crate) fn after_builtin(&mut self, name: &str) {
        if !is_passthrough_builtin(name) {
            self.visible = !is_invisible_builtin(name);
        }
    }
}

/// The hypothesis tests: their report is kept on the result they return
/// (see `call_fn`) and shown when that result is printed — not when the
/// test runs. So `res <- chisq.test(m)` prints nothing, `res` prints the
/// report, and `chisq.test(m)$p.value` prints just the number.
pub(crate) fn is_htest_builtin(name: &str) -> bool {
    matches!(name,
        "t.test" | "chisq.test" | "wilcox.test" | "var.test" | "ks.test" |
        "fisher.test" | "cor.test" | "prop.test" | "binom.test" |
        "oneway.test" | "kruskal.test" | "shapiro.test" | "bartlett.test" |
        "poisson.test" | "hotelling.test")
}

/// Functions whose value R returns invisibly — their result is not
/// auto-printed at the top level, and a user function ending in one of
/// them returns invisibly too (`f <- function(v) print(v); f(1)` prints
/// once). `anova`, `aov`, `manova`, `summary` and `str` are here because
/// R2's versions still print their report as they run; they leave this
/// list when they keep it on their result, as the hypothesis tests do.
pub(crate) fn is_invisible_builtin(name: &str) -> bool {
    matches!(name,
        "invisible" | "print" | "cat" | "message" | "warning" | "writeLines" |
        "library" | "require" | "detach" | "data" | "help" |
        "set.seed" | "Sys.sleep" | "Sys.setenv" | "setwd" |
        "assign" | "rm" | "remove" | "stopifnot" |
        "write.csv" | "write.table" | "saveRDS" | "save" | "sink" |
        "install.packages" | "uninstall" | "q" | "quit" |
        "clear" | "cls" | "clr" |
        "plot" | "hist" | "boxplot" | "barplot" | "lines" | "points" | "abline" | "legend" | "text" |
        "save.plot" | "dev.view" | "dev.off" |
        "system.time" |
        "anova" | "aov" | "manova" |
        "summary" | "str")
}

/// Builtins that evaluate user code and return its value: they keep the
/// visibility that code left (`tryCatch(invisible(1))` is invisible).
fn is_passthrough_builtin(name: &str) -> bool {
    matches!(name,
        "eval" | "evalq" | "local" | "with" | "switch" | "do.call" | "Recall" |
        "tryCatch" | "try" | "withCallingHandlers" | "suppressWarnings" | "suppressMessages")
}

/// Whether a function body's value is visible, read from its last
/// expression without running it — for the JIT, which returns the value
/// but not the visibility the interpreter would have left.
pub(crate) fn tail_is_visible(body: &Expr) -> bool {
    match body {
        Expr::Block(stmts) => stmts.last().map(tail_is_visible).unwrap_or(false),
        Expr::Assign { .. } | Expr::For { .. } | Expr::While { .. } | Expr::Repeat { .. } => false,
        Expr::Return(v) => tail_is_visible(v),
        Expr::Call { func, .. } => !matches!(func.as_ref(), Expr::Symbol(s) if is_invisible_builtin(s)),
        _ => true,
    }
}
