//! Runtime BLAS dispatch — load an optimized variant DLL if present,
//! otherwise use the built-in reference kernels. Mirrors R's swappable
//! `Rblas`: the reference BLAS is always available; a faster one
//! (CPU-tuned, OpenBLAS shim, …) can be dropped in without rebuilding.
//!
//! Selection: set the `R2_BLAS` environment variable to the path of a
//! shared library that exports the C-ABI `r2_dgemm` symbol (see
//! [`crate::blas_abi`]). If unset, unreadable, or missing the symbol,
//! the built-in static `dgemm` is used. The decision is made once and
//! cached for the process lifetime.

use std::sync::OnceLock;

use crate::blas_abi::DgemmCFn;
use crate::level3;
use crate::LinalgError;

static EXTERNAL_DGEMM: OnceLock<Option<DgemmCFn>> = OnceLock::new();

/// Resolve (once) the external `r2_dgemm` from `$R2_BLAS`, if any.
/// The loaded library is intentionally leaked (`mem::forget`) so the
/// function pointer stays valid for the whole process — standard for
/// a load-once plugin.
fn external_dgemm() -> Option<DgemmCFn> {
    *EXTERNAL_DGEMM.get_or_init(|| {
        let path = std::env::var_os("R2_BLAS")?;
        // SAFETY: loading an arbitrary user-pointed shared library is
        // inherently unsafe; this is opt-in via an env var, mirroring
        // how R lets a user swap Rblas. We only call the resolved
        // symbol with valid, correctly-sized buffers (see dgemm).
        unsafe {
            let lib = libloading::Library::new(&path).ok()?;
            let sym: libloading::Symbol<DgemmCFn> = lib.get(b"r2_dgemm\0").ok()?;
            let func = *sym;
            std::mem::forget(lib); // keep loaded for process lifetime
            eprintln!("[r2-linalg] using external BLAS: {}", path.to_string_lossy());
            Some(func)
        }
    })
}

/// `dgemm` with runtime dispatch: external optimized variant if
/// `$R2_BLAS` provides one, else the built-in reference kernel.
/// Same signature and semantics as [`crate::level3::dgemm`].
pub fn dgemm_dispatch(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64],
) -> Result<(), LinalgError> {
    // Shape checks up front so both paths share identical validation
    // and the external call gets correctly-sized buffers.
    if a.len() != m * k { return Err(LinalgError::InvalidShape(format!("A: {}x{}", m, k))); }
    if b.len() != k * n { return Err(LinalgError::InvalidShape(format!("B: {}x{}", k, n))); }
    if c.len() != m * n { return Err(LinalgError::InvalidShape(format!("C: {}x{}", m, n))); }

    if let Some(ext) = external_dgemm() {
        // SAFETY: buffers validated above; pointers are non-null and
        // correctly sized; the C contract matches r2_dgemm.
        let rc = unsafe {
            ext(m, n, k, alpha, a.as_ptr(), b.as_ptr(), beta, c.as_mut_ptr())
        };
        if rc == 0 {
            return Ok(());
        }
        // External call failed — fall back to the reference kernel
        // rather than propagate an opaque code.
    }
    level3::dgemm(m, n, k, alpha, a, b, beta, c)
}
