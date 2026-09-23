//! Closure -> JIT entry point (Phase C.2): prepare the body, then try the
//! code generators in precedence order until one compiles it. The body
//! preparation passes live in `inline.rs` and `rewrite.rs`, the loop
//! recognizers in `recognize.rs`.

use std::sync::Arc;
use r2_types::infer::{IrElem, IrType};
use r2_types::{Closure, Expr};
use crate::*;

type Handle = Arc<dyn r2_types::JitHandle>;
/// One way of compiling a (prepared) closure body; `None` when the body is
/// not its shape or its compile fails.
type Strategy = fn(&Closure, &Expr) -> Option<Handle>;

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

/// Tried BEFORE the lowerability gate, which would reject their `x[i]`
/// indexing: the indexed-loop shapes.
const INDEXED: &[Strategy] = &[
    index_fold_or_map,
    indexed_scalar_loop,
    indexed_store_map,
];

/// Tried after the gate, in precedence order: specialised shapes first,
/// then generic maps, then the multi-reduction kernels, then the scalar
/// specialization that takes anything lowerable.
const GENERAL: &[Strategy] = &[
    whole_vector_reduction,
    binary_map_reduce,
    vector_binary_op,
    binary_map_generic,
    map_with_literal,
    simd_map,
    map_generic,
    ternary_map,
    matvec_kernel,
    multi_reduction_kernel,
    scalar_fallback,
];

/// Attempt to JIT-compile a Closure. Returns `None` if the closure has more
/// than three parameters, any parameter with a default or `...`, or a body
/// with constructs the JIT does not yet support — the interpreter runs it.
///
/// On success, the engine should cache the returned handle keyed by
/// `Arc::as_ptr(&closure.body)` so re-calls reuse the compiled code.
pub fn try_compile_closure(cl: &Closure) -> Option<Handle> {
    // Phase R.M — gate the JIT on supported architectures. On aarch64 the
    // engine falls back to the interpreter; statistical outputs are
    // bit-identical, only wall-clock performance differs.
    if !JIT_SUPPORTED { return None; }
    // Phase C.5 admits 3-param closures for the ternary vector-map path.
    if cl.params.len() > 3 { return None; }
    if cl.params.iter().any(|p| p.default.is_some() || p.dots) { return None; }

    let body = prepare_body(cl);
    if let Some(h) = INDEXED.iter().find_map(|s| s(cl, &body)) {
        return Some(h);
    }
    // Eligibility gate: bail (→ interpreter) if the body contains any
    // construct the IR lowering silently drops (for/repeat/match/...).
    // Without this, e.g. `function(n){ s<-0; for(k in 1:n) s<-s+k; s }`
    // would JIT-compile with the loop elided and return `s`'s init value.
    if !body_is_jit_lowerable(&body) { return None; }
    GENERAL.iter().find_map(|s| s(cl, &body))
}

/// The body every strategy sees:
///
/// 1. Phase J.4 / J.5b — inline calls to pure JIT-lowerable user helpers,
///    so a function composed of small numeric helpers compiles as one
///    unit. A body with no such calls comes back structurally unchanged.
///    Depth-bounded → recursion falls back safely.
/// 2. Phase J.5 — normalize indexed accumulation loops into vector
///    reductions, so every later pass sees the one canonical spelling and
///    reuses the same compiled waves.
/// 3. Phase B.1 — closure capture inference by partial evaluation: free
///    variables that are numeric scalars (Real/Int/Bool of length 1) in
///    `cl.env` are baked in as constants, so the closure is self-contained
///    from the JIT's perspective — no new ABI surface, no per-call capture
///    passing. Vector-valued and other captures stay as references and the
///    lowering rejects them (the closure stays interpreter-only).
///    **Correctness window**: this assumes captured values are stable for
///    the lifetime of the closure. R semantics agree — captures are
///    by-value at creation time. If R2 ever adds reactive values, this
///    substitution must be invalidated on capture mutation.
fn prepare_body(cl: &Closure) -> Expr {
    let mut ib_ctr = 0u32;
    let inlined = vectorize_indexed_loops(
        &inline_block_helpers(&inline_user_calls(cl.body.as_ref(), &cl.env, 8), &cl.env, &mut ib_ctr, 4));
    if std::env::var("R2_JIT_DEBUG").is_ok() {
        eprintln!("[J5] normalized body: {:?}", inlined);
    }
    let param_names: Vec<Arc<str>> = cl.params.iter().map(|p| p.name.clone()).collect();
    let mut subs: std::collections::HashMap<Arc<str>, f64> = std::collections::HashMap::new();
    for name in &r2_ir::collect_free_vars(&inlined, &param_names) {
        if let Some(scalar) = cl.env.lookup(name).as_ref().and_then(scalar_f64_of) {
            subs.insert(name.clone(), scalar);
        }
    }
    if subs.is_empty() { inlined } else { r2_ir::substitute_constants(&inlined, &subs) }
}

/// A single f64 from an `RVal` that is a numeric scalar (Real / Int / Bool
/// of length 1, non-NA) — the "bakeable" captures of `prepare_body`.
fn scalar_f64_of(v: &r2_types::RVal) -> Option<f64> {
    match v {
        r2_types::RVal::Numeric(r, _) if r.len() == 1 => r[0],
        r2_types::RVal::Integer(r, _) if r.len() == 1 => r[0].map(|n| n as f64),
        r2_types::RVal::Logical(r, _) if r.len() == 1 => r[0].map(|b| if b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn handle(r: JitResult<CompiledFn>) -> Option<Handle> {
    r.ok().map(|c| Arc::new(c) as Handle)
}

fn param_names(cl: &Closure) -> Vec<Arc<str>> {
    cl.params.iter().map(|p| p.name.clone()).collect()
}

/// Lower `body` as a function of the closure's parameters, each a scalar
/// real, returning a scalar real — the per-element form the map and
/// map-reduce generators take.
fn lower_scalar(name: &str, cl: &Closure, body: &Expr) -> r2_ir::IrFunc {
    let params = cl.params.iter().map(|p| (p.name.clone(), IrType::scalar(IrElem::Real))).collect();
    let mut ir = r2_ir::lower_function(name, params, body);
    ir.return_type = IrType::scalar(IrElem::Real);
    ir
}

/// Indexed-kernel parameters: the vectors, then `extra` (an output
/// vector), then the loop-length scalar.
fn indexed_params(vecs: &[Arc<str>], extra: Option<Arc<str>>) -> Vec<(Arc<str>, IrType)> {
    let mut params: Vec<(Arc<str>, IrType)> = vecs.iter()
        .map(|v| (v.clone(), IrType::vector(IrElem::Real, None)))
        .collect();
    if let Some(out) = extra { params.push((out, IrType::vector(IrElem::Real, None))); }
    params.push((Arc::from(".__ixloop_n"), IrType::scalar(IrElem::Real)));
    params
}

// ── Indexed loops (before the gate) ──────────────────────────────────

/// Phase J.2 — index-loop fold over a vector param:
///   function(v){ [n<-length(v);] s<-init; for(i in 1:len) s<-s <+/*> f(v[i]); s }
/// recognized as a map-reduce over v (v[i] → element), reusing the fused
/// map-reduce codegen. Brick 2: the index-loop map
/// `for(i in 1:len) y[i] <- f(x[i]); y` → VectorMap.
fn index_fold_or_map(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 1 { return None; }
    let v = &cl.params[0].name;
    if let Some((mapped, reduce_op)) = recognize_index_reduction(body, v) {
        if body_is_jit_lowerable(&mapped) {
            let ir = lower_scalar("__map_reduce_inner__", cl, &mapped);
            if let Some(h) = handle(JitCompiler::compile_vector_map_reduce(&ir, reduce_op)) { return Some(h); }
        }
    }
    let mapped = recognize_index_map(body, v)?;
    if !body_is_jit_lowerable(&mapped) { return None; }
    handle(JitCompiler::compile_vector_map_generic(&lower_scalar("__index_map_inner__", cl, &mapped)))
}

/// Phase J.3 — general scalar-returning loop with real indexed loads over
/// 1-2 vector params (multi-statement folds, conditionals, scalar
/// recurrences reading x[i]/w[i]). Compiles the *actual* loop via `Load`
/// codegen; after the specialised fold/map recognisers so those keep
/// precedence for the shapes they cover.
fn indexed_scalar_loop(cl: &Closure, body: &Expr) -> Option<Handle> {
    if !(1..=2).contains(&cl.params.len()) { return None; }
    let (rewritten, vecs) = recognize_indexed_scalar_loop(body, &param_names(cl))?;
    let mut ir = r2_ir::lower_function("__indexed_scalar_loop__", indexed_params(&vecs, None), &rewritten);
    ir.return_type = IrType::scalar(IrElem::Real);
    handle(JitCompiler::compile_indexed_reduction(&ir))
}

/// Phase J.3 — general indexed-STORE map (1-2 input vectors → 1 output),
/// e.g. two-input `for(i in 1:length(x)) y[i] <- x[i]+w[i]; y` or a
/// multi-statement store body. After `recognize_index_map`, which handles
/// the single-input `y[i] <- f(x[i])` shape via VectorMap.
fn indexed_store_map(cl: &Closure, body: &Expr) -> Option<Handle> {
    if !(1..=2).contains(&cl.params.len()) { return None; }
    let (rewritten, in_vecs, out) = recognize_indexed_store_map(body, &param_names(cl))?;
    let mut ir = r2_ir::lower_function("__indexed_store_map__", indexed_params(&in_vecs, Some(out)), &rewritten);
    ir.return_type = IrType::null();
    handle(JitCompiler::compile_indexed_store_map(&ir))
}

// ── Whole-vector shapes (after the gate) ─────────────────────────────

/// Phase C.3 — `function(v) sum(v)` (also mean/length/prod), and Phase C.9
/// — the fused map-reduce `sum(f(v))` / `prod(f(v))`: load v[i], apply f,
/// accumulate. No intermediate vector allocated.
fn whole_vector_reduction(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 1 { return None; }
    let Expr::Call { func, args } = body else { return None };
    let Expr::Symbol(fname) = func.as_ref() else { return None };
    if !matches!(fname.as_ref(), "sum" | "mean" | "length" | "prod") || args.len() != 1 { return None; }
    if let Expr::Symbol(arg) = &args[0].value {
        if arg == &cl.params[0].name {
            if let Some(h) = handle(JitCompiler::compile_vector_reduction(fname.as_ref())) { return Some(h); }
        }
    }
    let reduce_op = match fname.as_ref() {
        "sum"  => FusedReduceOp::Sum,
        "prod" => FusedReduceOp::Prod,
        _ => return None,
    };
    let ir = lower_scalar("__map_reduce_inner__", cl, &args[0].value);
    handle(JitCompiler::compile_vector_map_reduce(&ir, reduce_op))
}

/// Phase J.2 — binary map-reduce `function(x, w) sum(f(x, w))` / prod, e.g.
/// the dot product `sum(x*w)`: fused (a[i], b[i]) → accumulate loop.
fn binary_map_reduce(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 2 { return None; }
    let Expr::Call { func, args } = body else { return None };
    let Expr::Symbol(fname) = func.as_ref() else { return None };
    if !matches!(fname.as_ref(), "sum" | "prod") || args.len() != 1
        || !body_is_jit_lowerable(&args[0].value) { return None; }
    let reduce_op = if fname.as_ref() == "sum" { FusedReduceOp::Sum } else { FusedReduceOp::Prod };
    let ir = lower_scalar("__binary_map_reduce_inner__", cl, &args[0].value);
    handle(JitCompiler::compile_vector_binary_map_reduce(&ir, reduce_op))
}

/// Phase C.4-full — vector ⊗ vector element-wise: `function(a, b) a OP b`.
fn vector_binary_op(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 2 { return None; }
    let Expr::Binary { op, lhs, rhs } = body else { return None };
    let (Expr::Symbol(ls), Expr::Symbol(rs)) = (lhs.as_ref(), rhs.as_ref()) else { return None };
    if ls != &cl.params[0].name || rs != &cl.params[1].name { return None; }
    handle(JitCompiler::compile_vector_binary_op(*op))
}

/// Phase C.7 — generic 2-param vector map for any body that lowers to
/// arithmetic + math Calls + branches: `function(a, b) sqrt(a*a + b*b)`,
/// `function(x, y) if (x > y) x else y`, etc.
fn binary_map_generic(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 2 { return None; }
    handle(JitCompiler::compile_vector_binary_map_generic(&lower_scalar("__vec_binary_body__", cl, body)))
}

/// Phase C.4 — element-wise vector map with a scalar literal:
/// `function(v) v OP k`, or `k OP v` for the commutative ops.
fn map_with_literal(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 1 { return None; }
    let Expr::Binary { op, lhs, rhs } = body else { return None };
    let pname = &cl.params[0].name;
    let k = match (lhs.as_ref(), rhs.as_ref()) {
        (Expr::Symbol(s), Expr::NumLit(k)) if s == pname => *k,
        (Expr::NumLit(k), Expr::Symbol(s)) if s == pname
            && matches!(op, r2_types::BinOp::Add | r2_types::BinOp::Mul) => *k,
        _ => return None,
    };
    handle(JitCompiler::compile_vector_map_scalar_op(*op, k))
}

/// Phase C.8 — SIMD f64x2 1-param vector map: a SIMD-clean body becomes a
/// tight 2-elements-per-iteration loop with native SSE2/NEON instructions;
/// anything else falls through to the scalar generic map.
fn simd_map(cl: &Closure, body: &Expr) -> Option<Handle> {
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) || cl.params.len() != 1 { return None; }
    handle(JitCompiler::compile_vector_simd_map_f64x2(&lower_scalar("__vec_simd_body__", cl, body)))
}

/// Phase C.4-full part 2 — generic 1-param vector map for any pure
/// arithmetic body (composed expressions, e.g. `(v+1)*2`, `v*v - 1`).
fn map_generic(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 1 { return None; }
    handle(JitCompiler::compile_vector_map_generic(&lower_scalar("__vec_body__", cl, body)))
}

/// Phase C.5 — generic 3-param branchy ternary vector map:
/// `function(c, a, b) if (c > 0) a else b` and similar multi-block bodies.
fn ternary_map(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 3 { return None; }
    handle(JitCompiler::compile_vector_ternary_map_generic(&lower_scalar("__vec_ternary_body__", cl, body)))
}

/// Phase J.4 matrix state — `function(X, y)` iterative kernels using
/// `X %*% v` / `t(X) %*% v` (multi-parameter GD / IRLS-core). The matrix
/// param is the one used as a `%*%` left operand (bare or `t(...)`); the
/// ABI fixes it as param 0. The engine takes this handle only when arg0 is
/// actually a Matrix with matching dims, so a mis-typed call falls back to
/// the interpreter.
fn matvec_kernel(cl: &Closure, body: &Expr) -> Option<Handle> {
    if cl.params.len() != 2 { return None; }
    let (p0, p1) = (&cl.params[0].name, &cl.params[1].name);
    let mat = matmul_matrix_param(body, p0, p1)?;
    if mat.as_ref() != p0.as_ref() { return None; }
    let (kbody0, _kept) = normalize_reduction_kernel(body, &[p1.clone()]);
    let mut mvctr = 0u32;
    let kbody = hoist_matmuls(&kbody0, &mut mvctr);
    match JitCompiler::compile_matvec_kernel(&kbody, &mat, p1) {
        Ok(c) => Some(Arc::new(c) as Handle),
        Err(e) => {
            if std::env::var("R2_JIT_DEBUG").is_ok() { eprintln!("[matvec-kernel] {:?}", e); }
            None
        }
    }
}

/// Phase J.4 brick 2 — multi-reduction kernel: combinations of whole-vector
/// reductions the single-reduction paths can't express — `sum(x*y)/sum(x*x)`
/// (regression coef), `{ m<-mean(x); sum((x-m)^2) }` (variance),
/// covariance. Only when the body mentions a reduction, so pure maps skip
/// it. Brick 3 fuses vector-valued intermediates and hoists reductions to
/// scalar locals; a vector-valued final expression (`x - mean(x)`, or a
/// KEPT loop-carried vector) takes the vector-output kernel.
fn multi_reduction_kernel(cl: &Closure, body: &Expr) -> Option<Handle> {
    if !(1..=2).contains(&cl.params.len()) || !mentions_reduction(body) { return None; }
    let pnames = param_names(cl);
    let (kbody, kept) = normalize_reduction_kernel(body, &pnames);
    let all_vec_names: Vec<Arc<str>> = pnames.iter().chain(kept.iter()).cloned().collect();
    let final_is_vec = match &kbody {
        Expr::Block(s) => s.last().map_or(false, |e| refs_vector_bare(e, &all_vec_names)),
        other => refs_vector_bare(other, &all_vec_names),
    };
    if final_is_vec {
        handle(JitCompiler::compile_reduction_map_kernel(&kbody, &pnames))
    } else {
        handle(JitCompiler::compile_reduction_kernel(&kbody, &pnames))
    }
}

/// Phase C.2 — the scalar specialization `(f64, ...) -> f64`.
fn scalar_fallback(cl: &Closure, body: &Expr) -> Option<Handle> {
    handle(JitCompiler::compile(&lower_scalar("__jit__", cl, body)))
}

// ── Diagnostics ──────────────────────────────────────────────────────

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
