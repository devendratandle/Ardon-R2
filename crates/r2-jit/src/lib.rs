//! R2 JIT — Cranelift backend.
//!
//! Per docs/ARCHITECTURE.md §5 Phase C and C.1:
//!   - Phase C   spine        : 0-arg scalar arithmetic returning f64. ✅ done.
//!   - Phase C.1 (this file) : function params, multi-block control flow,
//!                              Phi codegen via Cranelift block parameters.
//!
//! Phase C.1 supported subset (all scalar f64 internally):
//!   - N parameters of scalar Real / Int / Bool (all marshaled as f64)
//!   - `IrInst::Const`  for Real / Int / Bool
//!   - `IrInst::Binary` for Add / Sub / Mul / Div
//!     plus comparisons (Lt, Gt, Le, Ge, Eq, Ne) returning f64 1.0/0.0
//!   - `IrInst::Phi` lowered via Cranelift block parameters
//!   - Terminators: Return, Jump, Branch
//!
//! Out of scope (Phase C.2+):
//!   - `IrInst::Call` / `IrInst::Intrinsic` (need symbol table)
//!   - Vector / Matrix codegen (need ARROW ABI from Phase F)
//!   - Engine integration / Closure caching
//!   - Proper Bool ABI (i8 vs f64)
//!
//! Locked decisions: §4.1, §4.5, §4.7, §4.8 honoured (see ARCHITECTURE.md).
//!
//! ## Module layout
//!
//! Split by compilation stage. `error` (JitError/JitResult) and `handle`
//! (CompiledFn + ABI shims) are the leaf types. `compiler` holds the public
//! `JitCompiler` entry points; `codegen` the vectorized SIMD/map/reduce
//! builders; `lower` the IR→Cranelift body lowering; `externs` the
//! `extern "C"` math wrappers + registry; `closure` the engine-facing
//! `try_compile_closure`. Everything is re-exported flat so the only
//! externally-used path, `r2_jit::try_compile_closure`, is unchanged.

mod error;
mod handle;
mod compiler;
mod codegen;
mod externs;
mod closure;
mod lower;

pub use error::*;
pub use handle::*;
pub use compiler::*;
pub use closure::*;
pub(crate) use codegen::*;
pub(crate) use externs::*;
pub(crate) use lower::*;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
