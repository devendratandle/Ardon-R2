//! `Matrix` — 2D numeric array with linear-algebra operations.

use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════
// MATRIX — 2D numeric array with linear algebra operations
//
// This is the base type that ML libraries build on.
// Stored column-major (like R/Fortran) for BLAS compatibility.
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,     // column-major storage, no NA (use NaN for missing)
    pub nrow: usize,
    pub ncol: usize,
    pub col_names: Option<Vec<Arc<str>>>,
    pub row_names: Option<Vec<Arc<str>>>,
}

impl Matrix {
    pub fn new(data: Vec<f64>, nrow: usize, ncol: usize) -> Self {
        assert_eq!(data.len(), nrow * ncol, "data length must equal nrow * ncol");
        Matrix { data, nrow, ncol, col_names: None, row_names: None }
    }

    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Matrix::new(vec![0.0; nrow * ncol], nrow, ncol)
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    /// Get element at (row, col) — 0-based internal
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.nrow + row]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[col * self.nrow + row] = val;
    }

    /// Get a column as a slice
    pub fn col_slice(&self, col: usize) -> &[f64] {
        let start = col * self.nrow;
        &self.data[start..start + self.nrow]
    }

    /// Transpose — uses r2-linalg kernel
    pub fn transpose(&self) -> Matrix {
        let mut result = vec![0.0; self.nrow * self.ncol];
        r2_linalg::dtranspose(self.nrow, self.ncol, &self.data, &mut result).unwrap();
        Matrix::new(result, self.ncol, self.nrow)
    }

    /// Matrix multiply: self (m x k) * other (k x n) -> (m x n)
    /// Uses r2-linalg dgemm kernel (cache-blocked, SIMD-friendly)
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.ncol != other.nrow {
            return Err(format!("incompatible dimensions: {}x{} * {}x{}", self.nrow, self.ncol, other.nrow, other.ncol));
        }
        let mut c = vec![0.0; self.nrow * other.ncol];
        // Runtime-dispatched: uses an optimized BLAS variant DLL when
        // R2_BLAS points to one, else the built-in reference kernel.
        r2_linalg::dgemm_dispatch(self.nrow, other.ncol, self.ncol, 1.0, &self.data, &other.data, 0.0, &mut c)
            .map_err(|e| e.to_string())?;
        Ok(Matrix::new(c, self.nrow, other.ncol))
    }

    /// Element-wise operation
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Matrix {
        Matrix::new(self.data.iter().map(|x| f(*x)).collect(), self.nrow, self.ncol)
    }

    /// Element-wise binary operation
    pub fn zip_with(&self, other: &Matrix, f: impl Fn(f64, f64) -> f64) -> Result<Matrix, String> {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return Err("matrix dimensions must match".into());
        }
        Ok(Matrix::new(
            self.data.iter().zip(other.data.iter()).map(|(a, b)| f(*a, *b)).collect(),
            self.nrow, self.ncol,
        ))
    }

    /// Scalar multiplication — uses r2-linalg dscal kernel
    pub fn scale(&self, s: f64) -> Matrix {
        let mut data = self.data.clone();
        r2_linalg::dscal(s, &mut data);
        Matrix::new(data, self.nrow, self.ncol)
    }

    /// Add matrices
    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Subtract matrices
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, String> {
        self.zip_with(other, |a, b| a - b)
    }

    /// Column means
    pub fn col_means(&self) -> Vec<f64> {
        (0..self.ncol).map(|c| {
            self.col_slice(c).iter().sum::<f64>() / self.nrow as f64
        }).collect()
    }

    /// Column sums
    pub fn col_sums(&self) -> Vec<f64> {
        (0..self.ncol).map(|c| self.col_slice(c).iter().sum()).collect()
    }

    /// Row sums
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.nrow).map(|r| {
            (0..self.ncol).map(|c| self.get(r, c)).sum()
        }).collect()
    }

    /// Dot product of two column vectors — uses r2-linalg ddot kernel
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        r2_linalg::ddot(a, b)
    }

    /// Frobenius norm — uses r2-linalg dnrm2 kernel
    pub fn norm(&self) -> f64 {
        r2_linalg::dnrm2(&self.data)
    }

    /// Convert to vector of rows (for iteration)
    pub fn rows(&self) -> Vec<Vec<f64>> {
        (0..self.nrow).map(|r| {
            (0..self.ncol).map(|c| self.get(r, c)).collect()
        }).collect()
    }

    /// Solve Ax = b using r2-linalg dgesv (LU with partial pivoting)
    /// MUCH faster and more stable than the old Gaussian elimination
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, String> {
        if self.nrow != self.ncol { return Err("matrix must be square".into()); }
        if self.nrow != b.len() { return Err("dimensions don't match".into()); }
        let mut a = self.data.clone();
        let mut x = b.to_vec();
        r2_linalg::dgesv(self.nrow, &mut a, &mut x).map_err(|e| e.to_string())?;
        Ok(x)
    }

    /// Compute X^T * X — uses r2-linalg dcrossprod (avoids explicit transpose)
    pub fn crossprod(&self) -> Matrix {
        let mut c = vec![0.0; self.ncol * self.ncol];
        r2_linalg::dcrossprod(self.nrow, self.ncol, &self.data, &mut c).unwrap();
        Matrix::new(c, self.ncol, self.ncol)
    }

    /// Compute X^T * y where y is a vector — uses r2-linalg dgemv_t
    pub fn crossprod_vec(&self, y: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.ncol];
        r2_linalg::dgemv_t(self.nrow, self.ncol, 1.0, &self.data, y, 0.0, &mut result).unwrap();
        result
    }
}

