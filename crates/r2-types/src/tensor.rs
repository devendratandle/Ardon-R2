//! `Tensor` — N-dimensional numeric array for ML.

use crate::*;

// ═══════════════════════════════════════════════════════════════════════
// TENSOR — N-dimensional numeric array for ML
//
// This is in BASE so ML addon libraries can build on it.
// The user rarely creates tensors directly; data.frame → tensor
// conversion is automatic in ML pipelines.
//
// Storage: contiguous f64, row-major (C-order) for ML compatibility.
// GPU backing is a future extension (same API, different storage).
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: TensorDType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDType {
    Float64,
    Float32,
    Int32,
    Bool,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "data length must match shape product");
        let strides = Self::compute_strides(&shape);
        Tensor { data, shape, strides, dtype: TensorDType::Float64 }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Tensor::new(vec![0.0; total], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Tensor::new(vec![1.0; total], shape)
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Tensor::new(data, vec![len])
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.data.len() }

    /// Get element by multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat]
    }

    /// Set element by multi-dimensional index
    pub fn set(&mut self, indices: &[usize], val: f64) {
        let flat: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat] = val;
    }

    /// Reshape (same data, different shape)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, String> {
        let new_total: usize = new_shape.iter().product();
        if new_total != self.numel() {
            return Err(format!("cannot reshape {} elements into shape {:?}", self.numel(), new_shape));
        }
        Ok(Tensor::new(self.data.clone(), new_shape))
    }

    /// Flatten to 1D
    pub fn flatten(&self) -> Tensor {
        Tensor::new(self.data.clone(), vec![self.numel()])
    }

    /// Element-wise operation
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor::new(self.data.iter().map(|x| f(*x)).collect(), self.shape.clone())
    }

    /// Element-wise binary (shapes must match or broadcast)
    pub fn zip_with(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, String> {
        if self.shape != other.shape {
            // Simple scalar broadcast
            if other.numel() == 1 {
                let s = other.data[0];
                return Ok(self.map(|x| f(x, s)));
            }
            if self.numel() == 1 {
                let s = self.data[0];
                return Ok(other.map(|x| f(s, x)));
            }
            return Err(format!("shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }
        Ok(Tensor::new(
            self.data.iter().zip(other.data.iter()).map(|(a, b)| f(*a, *b)).collect(),
            self.shape.clone(),
        ))
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a + b) }
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a - b) }
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a * b) }
    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a / b) }

    pub fn scale(&self, s: f64) -> Tensor { self.map(|x| x * s) }
    pub fn sum(&self) -> f64 { self.data.iter().sum() }
    pub fn mean(&self) -> f64 { self.sum() / self.numel() as f64 }

    /// Common activation functions (ML foundation)
    pub fn relu(&self) -> Tensor { self.map(|x| if x > 0.0 { x } else { 0.0 }) }
    pub fn sigmoid(&self) -> Tensor { self.map(|x| 1.0 / (1.0 + (-x).exp())) }
    pub fn tanh_act(&self) -> Tensor { self.map(|x| x.tanh()) }
    pub fn softmax(&self) -> Tensor {
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self.data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        Tensor::new(exps.iter().map(|x| x / sum).collect(), self.shape.clone())
    }
    pub fn log_softmax(&self) -> Tensor {
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = self.data.iter().map(|x| x - max_val).collect();
        let log_sum_exp = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
        Tensor::new(shifted.iter().map(|x| x - log_sum_exp).collect(), self.shape.clone())
    }

    /// 2D matrix multiply for tensors (last two dims)
    pub fn matmul_2d(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("matmul_2d requires 2D tensors".into());
        }
        let m = self.shape[0];
        let k = self.shape[1];
        if k != other.shape[0] { return Err("inner dimensions don't match".into()); }
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    result[i * n + j] += self.get(&[i, p]) * other.get(&[p, j]);
                }
            }
        }
        Ok(Tensor::new(result, vec![m, n]))
    }

    /// Convert from Matrix (column-major → row-major)
    pub fn from_matrix(m: &Matrix) -> Tensor {
        let mut data = vec![0.0; m.nrow * m.ncol];
        for r in 0..m.nrow {
            for c in 0..m.ncol {
                data[r * m.ncol + c] = m.get(r, c);
            }
        }
        Tensor::new(data, vec![m.nrow, m.ncol])
    }

    /// Convert to Matrix (row-major → column-major)
    pub fn to_matrix(&self) -> Result<Matrix, String> {
        if self.ndim() != 2 { return Err("only 2D tensors can convert to Matrix".into()); }
        let nrow = self.shape[0];
        let ncol = self.shape[1];
        let mut data = vec![0.0; nrow * ncol];
        for r in 0..nrow {
            for c in 0..ncol {
                data[c * nrow + r] = self.get(&[r, c]); // column-major
            }
        }
        Ok(Matrix::new(data, nrow, ncol))
    }

    /// Convert DataFrame numeric columns to Tensor (for ML pipelines)
    pub fn from_dataframe(df: &DataFrame, columns: &[&str]) -> Result<Tensor, String> {
        let nrow = df.nrow();
        let ncol = columns.len();
        let mut data = vec![0.0; nrow * ncol];
        for (c, col_name) in columns.iter().enumerate() {
            let col = df.get_col(col_name).ok_or(format!("column '{}' not found", col_name))?;
            match col {
                RVal::Numeric(v, _) => {
                    for (r, val) in v.iter().enumerate() {
                        data[r * ncol + c] = val.unwrap_or(f64::NAN);
                    }
                }
                RVal::Integer(v, _) => {
                    for (r, val) in v.iter().enumerate() {
                        data[r * ncol + c] = val.map(|n| n as f64).unwrap_or(f64::NAN);
                    }
                }
                _ => return Err(format!("column '{}' is not numeric", col_name)),
            }
        }
        Ok(Tensor::new(data, vec![nrow, ncol]))
    }
}

