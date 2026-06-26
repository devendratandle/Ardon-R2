//! Rust externs the JIT calls + the math-extern registry.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use r2_types::infer::IrElem;
use std::collections::HashMap;
use crate::*;

// ── Rust externs the JIT calls ───────────────────────────────────────

pub(crate) extern "C" fn r2_extern_sum(ptr: *const f64, len: i64) -> f64 {
    if ptr.is_null() || len < 0 { return 0.0; }
    let s = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.iter().sum()
}

pub(crate) extern "C" fn r2_extern_mean(ptr: *const f64, len: i64) -> f64 {
    if ptr.is_null() || len <= 0 { return f64::NAN; }
    let s = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.iter().sum::<f64>() / len as f64
}

pub(crate) extern "C" fn r2_extern_length(_ptr: *const f64, len: i64) -> f64 {
    len as f64
}

pub(crate) extern "C" fn r2_extern_prod(ptr: *const f64, len: i64) -> f64 {
    if ptr.is_null() || len < 0 { return 1.0; }
    let s = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    s.iter().product()
}

// ════════════════════════════════════════════════════════════════════
// Scalar math externs — JIT-callable from `IrInst::Call`.
//
// Each `r2_math_*` wrapper is `extern "C"` so it has a stable ABI we can
// register as a Cranelift symbol and emit a direct `call` instruction
// to. The wrappers delegate to Rust stdlib methods (`f64::sqrt` etc.)
// which on x86_64 lower to the SSE math instructions or libm calls.
//
// This is the "broaden JIT coverage to bytecode-class workloads" piece:
// any user function whose body is pure scalar arithmetic + comparisons
// + these math calls now lowers fully to native machine code — no
// bytecode VM layer, no per-call interpreter checkpoint.
// ════════════════════════════════════════════════════════════════════

pub(crate) extern "C" fn r2_math_sqrt(x: f64)  -> f64 { x.sqrt() }
pub(crate) extern "C" fn r2_math_abs(x: f64)   -> f64 { x.abs() }
pub(crate) extern "C" fn r2_math_exp(x: f64)   -> f64 { x.exp() }
pub(crate) extern "C" fn r2_math_ln(x: f64)    -> f64 { x.ln() }
pub(crate) extern "C" fn r2_math_log2(x: f64)  -> f64 { x.log2() }
pub(crate) extern "C" fn r2_math_log10(x: f64) -> f64 { x.log10() }
pub(crate) extern "C" fn r2_math_sin(x: f64)   -> f64 { x.sin() }
pub(crate) extern "C" fn r2_math_cos(x: f64)   -> f64 { x.cos() }
pub(crate) extern "C" fn r2_math_tan(x: f64)   -> f64 { x.tan() }
pub(crate) extern "C" fn r2_math_asin(x: f64)  -> f64 { x.asin() }
pub(crate) extern "C" fn r2_math_acos(x: f64)  -> f64 { x.acos() }
pub(crate) extern "C" fn r2_math_atan(x: f64)  -> f64 { x.atan() }
pub(crate) extern "C" fn r2_math_sinh(x: f64)  -> f64 { x.sinh() }
pub(crate) extern "C" fn r2_math_cosh(x: f64)  -> f64 { x.cosh() }
pub(crate) extern "C" fn r2_math_tanh(x: f64)  -> f64 { x.tanh() }
pub(crate) extern "C" fn r2_math_floor(x: f64) -> f64 { x.floor() }
pub(crate) extern "C" fn r2_math_ceil(x: f64)  -> f64 { x.ceil() }
pub(crate) extern "C" fn r2_math_round(x: f64) -> f64 { x.round() }
pub(crate) extern "C" fn r2_math_trunc(x: f64) -> f64 { x.trunc() }
pub(crate) extern "C" fn r2_math_sign(x: f64)  -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}
pub(crate) extern "C" fn r2_math_pow(x: f64, y: f64) -> f64 { x.powf(y) }
pub(crate) extern "C" fn r2_math_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
pub(crate) extern "C" fn r2_math_min2(a: f64, b: f64) -> f64 { a.min(b) }
pub(crate) extern "C" fn r2_math_max2(a: f64, b: f64) -> f64 { a.max(b) }

/// Math-extern entry — table of `(R-level name, C symbol, Rust wrapper, arity)`.
/// Used by `register_math_symbols()` to install pointers on a `JITBuilder`
/// and by `lower_inst` to look up the right declaration when emitting a Call.
pub(crate) struct MathExtern {
    /// The name as it appears in R user code (e.g. `"sqrt"`).
    pub(crate) r_name: &'static str,
    /// The Cranelift symbol name (stable across compilations).
    pub(crate) c_name: &'static str,
    /// Raw function pointer, cast to `*const u8` for `JITBuilder::symbol`.
    pub(crate) ptr: *const u8,
    /// Number of f64 parameters.
    pub(crate) arity: usize,
}

// SAFETY: each `ptr` is a `*const u8` cast of a real `extern "C" fn(f64,...) -> f64`.
// We only ever transmute back to that exact signature via the Cranelift
// declaration, and the wrappers themselves are panic-free pure functions.
unsafe impl Send for MathExtern {}
unsafe impl Sync for MathExtern {}

pub(crate) static MATH_EXTERNS: &[MathExtern] = &[
    // Unary (arity = 1)
    MathExtern { r_name: "sqrt",  c_name: "__r2_math_sqrt",  ptr: r2_math_sqrt  as *const u8, arity: 1 },
    MathExtern { r_name: "abs",   c_name: "__r2_math_abs",   ptr: r2_math_abs   as *const u8, arity: 1 },
    MathExtern { r_name: "exp",   c_name: "__r2_math_exp",   ptr: r2_math_exp   as *const u8, arity: 1 },
    MathExtern { r_name: "log",   c_name: "__r2_math_ln",    ptr: r2_math_ln    as *const u8, arity: 1 },
    MathExtern { r_name: "log2",  c_name: "__r2_math_log2",  ptr: r2_math_log2  as *const u8, arity: 1 },
    MathExtern { r_name: "log10", c_name: "__r2_math_log10", ptr: r2_math_log10 as *const u8, arity: 1 },
    MathExtern { r_name: "sin",   c_name: "__r2_math_sin",   ptr: r2_math_sin   as *const u8, arity: 1 },
    MathExtern { r_name: "cos",   c_name: "__r2_math_cos",   ptr: r2_math_cos   as *const u8, arity: 1 },
    MathExtern { r_name: "tan",   c_name: "__r2_math_tan",   ptr: r2_math_tan   as *const u8, arity: 1 },
    MathExtern { r_name: "asin",  c_name: "__r2_math_asin",  ptr: r2_math_asin  as *const u8, arity: 1 },
    MathExtern { r_name: "acos",  c_name: "__r2_math_acos",  ptr: r2_math_acos  as *const u8, arity: 1 },
    MathExtern { r_name: "atan",  c_name: "__r2_math_atan",  ptr: r2_math_atan  as *const u8, arity: 1 },
    MathExtern { r_name: "sinh",  c_name: "__r2_math_sinh",  ptr: r2_math_sinh  as *const u8, arity: 1 },
    MathExtern { r_name: "cosh",  c_name: "__r2_math_cosh",  ptr: r2_math_cosh  as *const u8, arity: 1 },
    MathExtern { r_name: "tanh",  c_name: "__r2_math_tanh",  ptr: r2_math_tanh  as *const u8, arity: 1 },
    MathExtern { r_name: "floor", c_name: "__r2_math_floor", ptr: r2_math_floor as *const u8, arity: 1 },
    MathExtern { r_name: "ceil",  c_name: "__r2_math_ceil",  ptr: r2_math_ceil  as *const u8, arity: 1 },
    MathExtern { r_name: "round", c_name: "__r2_math_round", ptr: r2_math_round as *const u8, arity: 1 },
    MathExtern { r_name: "trunc", c_name: "__r2_math_trunc", ptr: r2_math_trunc as *const u8, arity: 1 },
    MathExtern { r_name: "sign",  c_name: "__r2_math_sign",  ptr: r2_math_sign  as *const u8, arity: 1 },
    // Binary (arity = 2)
    MathExtern { r_name: "^",     c_name: "__r2_math_pow",   ptr: r2_math_pow   as *const u8, arity: 2 },
    MathExtern { r_name: "atan2", c_name: "__r2_math_atan2", ptr: r2_math_atan2 as *const u8, arity: 2 },
    MathExtern { r_name: "min",   c_name: "__r2_math_min2",  ptr: r2_math_min2  as *const u8, arity: 2 },
    MathExtern { r_name: "max",   c_name: "__r2_math_max2",  ptr: r2_math_max2  as *const u8, arity: 2 },
];

/// Look up a math extern by R-level name.
pub(crate) fn find_math_extern(name: &str) -> Option<&'static MathExtern> {
    MATH_EXTERNS.iter().find(|e| e.r_name == name)
}

/// Register all math externs as symbols on a `JITBuilder` so Cranelift
/// can resolve calls to them. Call this on every `JITBuilder` before
/// constructing the `JITModule`.
pub(crate) fn register_math_symbols(jit_builder: &mut JITBuilder) {
    for e in MATH_EXTERNS {
        jit_builder.symbol(e.c_name, e.ptr);
    }
}

/// Declare math-extern imports on a module and return a name-to-FuncId
/// map. Each Cranelift module that may emit Call instructions calls
/// this once and threads the resulting map through to `lower_inst`.
pub(crate) fn declare_math_imports(
    module: &mut JITModule,
) -> JitResult<HashMap<&'static str, cranelift_module::FuncId>> {
    let mut map = HashMap::new();
    for e in MATH_EXTERNS {
        let mut sig = module.make_signature();
        for _ in 0..e.arity { sig.params.push(AbiParam::new(types::F64)); }
        sig.returns.push(AbiParam::new(types::F64));
        let id = module
            .declare_function(e.c_name, Linkage::Import, &sig)
            .map_err(|err| JitError::CraneliftError(format!("declare math extern {}: {:?}", e.c_name, err)))?;
        map.insert(e.r_name, id);
    }
    Ok(map)
}

pub(crate) fn is_scalar_numeric(e: &IrElem) -> bool {
    matches!(e, IrElem::Real | IrElem::Int | IrElem::Bool)
}
