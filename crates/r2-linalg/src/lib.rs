//! R2 Linear Algebra Kernel
//! ========================
//! BLAS-style numerical operations in pure Rust.
//! Column-major matrix storage (Fortran convention, same as R).
//!
//! Design principles:
//!   - Pure Rust throughout: the kernels use only std. The crate depends on
//!     `rayon` (Level-3 parallel dispatch) and `r2-oracle` (the serial/parallel
//!     crossover decision). No FFI.
//!   - Cache-friendly blocked algorithms for matrix multiply
//!   - Explicit SIMD-friendly loops (compiler auto-vectorizes with -O3)
//!   - Fallible operations return Result, not panic
//!   - All matrices are f64 dense
//!
//! Modules:
//!   - level1: vector-vector operations (dot, axpy, norm)
//!   - level2: matrix-vector operations (gemv, trsv)
//!   - level3: matrix-matrix operations (gemm, syrk)
//!   - decomp: LU, Cholesky, QR factorizations
//!   - solve:  linear system solvers

pub mod gemm;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod decomp;
pub mod eigen;
pub mod symeig;
mod simd;
pub mod solve;

pub use level1::*;
pub use level2::*;
pub use level3::*;
pub use decomp::*;
pub use eigen::*;
pub use symeig::*;
pub use solve::*;

/// Errors from linear algebra operations
#[derive(Debug, Clone)]
pub enum LinalgError {
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
    Singular,
    NotSquare,
    NotPositiveDefinite,
    InvalidShape(String),
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LinalgError::DimensionMismatch { expected, got } =>
                write!(f, "dimension mismatch: expected {:?}, got {:?}", expected, got),
            LinalgError::Singular => write!(f, "matrix is singular (not invertible)"),
            LinalgError::NotSquare => write!(f, "matrix must be square"),
            LinalgError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            LinalgError::InvalidShape(s) => write!(f, "invalid shape: {}", s),
        }
    }
}

pub type LinalgResult<T> = Result<T, LinalgError>;
