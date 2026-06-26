//! R2 compute kernels — Phase K.
//!
//! Per docs/ARCHITECTURE.md §4.9 + §5 Phase K:
//!   - Parallelism (Rayon, future GPU/Cloud) lives BELOW this layer.
//!   - Builtins call kernel functions only; they don't see backends.
//!   - Each kernel has serial + Rayon impls today; new backends are
//!     additive (just another `impl ReduceBackend for GpuBackend { ... }`).
//!
//! Phase K spine: reduction kernel only. Element-wise (`map`), binary
//! (`a OP b`), and scan kernels arrive in K.1, K.2, K.3.
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only (Rayon qualifies).
//!   §4.7 Backwards-compatible — additive crate, no breaking changes.
//!   §4.9 Rayon stays below this layer.
//!
//! ## Module layout
//!
//! One module per kernel family. Each owns its `Op` enum, backend trait,
//! the two `impl <Family>Backend for {Serial,Rayon}Backend` blocks, any
//! private `apply_*`/scan helpers, and the public Oracle-dispatched entry
//! point. Everything is re-exported flat from the crate root, so callers
//! still write `r2_kernel::reduce`, `r2_kernel::ReduceOp`, etc. The shared
//! `SerialBackend`/`RayonBackend` marker structs live in `reduce` (the
//! first family) and are re-exported alongside the rest.

mod reduce;
mod map;
mod binary;
mod ternary;
mod strided;
mod par;
mod scan;
mod select;
mod rolling;
mod agg;
mod distance;

pub use reduce::*;
pub use map::*;
pub use binary::*;
pub use ternary::*;
pub use strided::*;
pub use par::*;
pub use scan::*;
pub use select::*;
pub use rolling::*;
pub use agg::*;
pub use distance::*;

#[cfg(test)]
mod tests;
