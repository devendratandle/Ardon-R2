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

