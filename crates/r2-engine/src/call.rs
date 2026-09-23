//! Calling a function value: a builtin (through the capability policy
//! and the package search order), a closure (compiled when the JIT can
//! take it — `jit_call.rs` — else interpreted in a fresh call frame), or
//! a type's constructor.

#![allow(clippy::all)]
use std::collections::HashMap;
use std::sync::Arc;
use r2_types::*;
use crate::Engine;
use crate::err;
use crate::eval::is_htest_builtin;

impl Engine {
    pub(crate) fn call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        match func {
            RVal::BuiltinFn(name) => self.call_builtin(name, args, env),
            RVal::Closure(cl) => self.call_closure(cl, args),
            RVal::TypeDef(td) => Ok(self.construct_type(td, args)),
            _ => err!(Runtime, "not callable as a function. Check spelling or use help() to find the right function name"),
        }
    }

    fn call_builtin(&mut self, name: &Arc<str>, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        // Capability policy — the SINGLE security choke point for
        // server-hosted (LLM-agent) sessions. Interactive engines run
        // allow-all, so this is a no-op there.
        let bare = name.rsplit("::").next().unwrap_or(name.as_ref());
        if let Some(reason) = self.policy.deny_reason(bare) {
            return err!(Runtime, "{}", reason);
        }
        // pkg::func — resolve in that package only.
        if let Some(sep) = name.find("::") {
            let pkg = &name[..sep];
            let fname = &name[sep+2..];
            let Some(f) = self.registry.resolve_in_package(pkg, fname) else {
                return err!(Runtime, "'{}' not found in package '{}'", fname, pkg);
            };
            let r = f(self, args, env);
            self.after_builtin(fname);
            return r;
        }
        // Normal resolution through search order
        let Some((f, _pkg)) = self.registry.resolve(name.as_ref()) else {
            return err!(Runtime, "unknown function '{}'", name);
        };
        let r = if is_htest_builtin(bare) {
            // R's design: the test RETURNS its result and prints nothing;
            // the report appears when the result is printed. R2's tests
            // write their report as they run, so it is captured here and
            // kept on the result.
            let (r, report) = r2_types::out::capture(|| f(self, args, env));
            match r {
                Ok(RVal::TypeInstance(mut inst)) if !report.is_empty() => {
                    inst.fields.insert(Arc::from(".report"),
                        RVal::Character(vec![Some(Arc::from(report.as_str()))], Attrs::default()));
                    Ok(RVal::TypeInstance(inst))
                }
                // anything else: the text goes out as it always did
                other => { r2_types::out::rout(&report); other }
            }
        } else {
            f(self, args, env)
        };
        self.after_builtin(bare);
        r
    }

    fn call_closure(&mut self, cl: &Closure, args: &[EvalArg]) -> Result<RVal, R2Err> {
        if self.frames.len() >= 500 {
            return err!(Runtime, "recursion depth limit exceeded (max 500). Use iteration instead.");
        }
        if let Some(v) = self.try_jit_call(cl, args) {
            return Ok(v);
        }
        // Tree-walking interpreter.
        let func_env = bind_args(cl, args);
        // R semantics: defaults evaluate lazily IN THE FUNCTION'S OWN
        // environment, so `function(x, y = x * 2)` sees the bound `x`, and
        // later defaults can use earlier ones (`m = n+1, p = m*2` — params
        // are processed in declaration order). Record which params were
        // NOT supplied first, for `missing()`.
        let unsupplied: Vec<Option<Arc<str>>> = cl.params.iter()
            .filter(|p| !p.dots && !func_env.contains(p.name.as_ref()))
            .map(|p| Some(p.name.clone())).collect();
        if !unsupplied.is_empty() {
            func_env.set(Arc::from(".missing"), RVal::Character(unsupplied, Attrs::default()));
        }
        self.frames.push(func_env.clone());
        for p in cl.params.iter().filter(|p| !p.dots) {
            if !func_env.contains(p.name.as_ref()) {
                let v = p.default.as_ref().and_then(|d| self.eval_in(d, &func_env).ok()).unwrap_or(RVal::Null);
                func_env.set(p.name.clone(), v);
            }
        }
        let result = match self.eval_in(&cl.body, &func_env) {
            Err(R2Err { kind: ErrKind::CtrlReturn(v), .. }) => Ok(*v),
            r => r,
        };
        self.frames.pop();
        result
    }

    /// A type's constructor. Fields are gathered parent-first along the
    /// `extends` chain so an inheriting type accepts and stores inherited
    /// fields (not just its own). Positional args fill them in declaration
    /// order: ancestors first, then this type's fields.
    fn construct_type(&self, td: &TypeDef, args: &[EvalArg]) -> RVal {
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
            let v = get_arg(args, i, fname).or_else(|| fdefault.clone()).unwrap_or(RVal::Null);
            fields.insert(fname.clone(), v);
        }
        RVal::TypeInstance(TypeInstance { type_name: td.name.clone(), fields })
    }
}

/// A new call frame with the arguments bound. The frame is a LIVE child of
/// the closure's captured env: closures defined in the body capture it by
/// reference, and `<<-` from inner calls mutates it in place.
///
/// R-style matching with `...`: named args bind to formals by name;
/// positional args fill the formals before `...`; everything left over is
/// collected into `...` (bound as a List of (name, value) under "...").
fn bind_args(cl: &Closure, args: &[EvalArg]) -> EnvRef {
    let func_env = Env::new_child(cl.env.clone(), None);
    {
        let mut m = func_env.bindings.write().unwrap();
        let dots_pos = cl.params.iter().position(|p| p.dots);
        let mut used = vec![false; args.len()];
        for p in cl.params.iter().filter(|p| !p.dots) {
            if let Some(j) = (0..args.len()).find(|&j| !used[j] && args[j].name.as_deref() == Some(p.name.as_ref())) {
                used[j] = true;
                m.insert(p.name.clone(), args[j].value.clone());
            }
        }
        let before = dots_pos.unwrap_or(cl.params.len());
        let mut pos = 0usize;
        for (i, p) in cl.params.iter().enumerate() {
            if p.dots || i >= before || m.contains_key(p.name.as_ref()) { continue; }
            while pos < args.len() && (used[pos] || args[pos].name.is_some()) { pos += 1; }
            if pos < args.len() { used[pos] = true; m.insert(p.name.clone(), args[pos].value.clone()); pos += 1; }
        }
        if dots_pos.is_some() {
            let dots: Vec<(Option<Arc<str>>, RVal)> = (0..args.len())
                .filter(|&j| !used[j])
                .map(|j| (args[j].name.clone(), args[j].value.clone()))
                .collect();
            m.insert(Arc::from("..."), RVal::List(dots));
        }
        // Record the call's argument count for nargs().
        m.insert(Arc::from(".nargs"), RVal::Integer(vec![Some(args.len() as i32)].into(), Attrs::default()));
    }
    func_env
}

/// The argument named `name`, else the one at position `pos`.
fn get_arg(args: &[EvalArg], pos: usize, name: &str) -> Option<RVal> {
    args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone())
        .or_else(|| args.get(pos).map(|a| a.value.clone()))
}
