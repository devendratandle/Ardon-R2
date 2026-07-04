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
    let param_names: Vec<std::sync::Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
    let free_vars = r2_ir::collect_free_vars(cl.body.as_ref(), &param_names);
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
            body_expr = r2_ir::substitute_constants(cl.body.as_ref(), &subs);
            body_ref = &body_expr;
        } else {
            body_ref = cl.body.as_ref();
        }
    } else {
        body_ref = cl.body.as_ref();
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

