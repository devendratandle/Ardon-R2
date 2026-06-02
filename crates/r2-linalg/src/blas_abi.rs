//! Stable C-ABI surface for the BLAS kernels.
//!
//! Rust has no stable ABI, so a runtime-loadable / hot-swappable BLAS
//! must cross the boundary as plain C: flat `f64` buffers + integer
//! dimensions, exactly like Fortran/reference BLAS and R's `Rblas`.
//! These `#[no_mangle] extern "C"` symbols are what `r2_linalg.dll`
//! (and any drop-in CPU-optimized variant — `r2_linalg_avx2.dll`,
//! an OpenBLAS shim, etc.) export, and what [`crate::blas_dispatch`]
//! looks up at runtime.
//!
//! Contract for `r2_dgemm` (column-major, no transpose):
//!   C(m×n) = alpha · A(m×k) · B(k×n) + beta · C
//!   a points to m*k f64, b to k*n f64, c to m*n f64 (read+write).
//!   Returns 0 on success, non-zero on a (defensive) error.
//!
//! Safety: callers must pass valid, correctly-sized, non-overlapping
//! buffers. The dispatcher upholds this; external callers must too.

use crate::level3;

/// C-ABI signature shared by the export here and the symbol the
/// dispatcher loads from an optional optimized variant DLL.
pub type DgemmCFn = unsafe extern "C" fn(
    m: usize, n: usize, k: usize,
    alpha: f64,
    a: *const f64, b: *const f64,
    beta: f64,
    c: *mut f64,
) -> i32;

/// Reference `dgemm` exported with a stable C ABI. This is the symbol
/// `r2_linalg.dll` ships; an optimized variant exposes the same name.
///
/// # Safety
/// `a`/`b` must be valid for reads of `m*k` / `k*n` `f64`; `c` valid
/// for read+write of `m*n` `f64`. Buffers must not overlap.
#[no_mangle]
pub unsafe extern "C" fn r2_dgemm(
    m: usize, n: usize, k: usize,
    alpha: f64,
    a: *const f64, b: *const f64,
    beta: f64,
    c: *mut f64,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() { return 1; }
    let a = std::slice::from_raw_parts(a, m * k);
    let b = std::slice::from_raw_parts(b, k * n);
    let c = std::slice::from_raw_parts_mut(c, m * n);
    match level3::dgemm(m, n, k, alpha, a, b, beta, c) {
        Ok(()) => 0,
        Err(_) => 2,
    }
}
