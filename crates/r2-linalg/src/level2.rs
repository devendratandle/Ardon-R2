//! Level 2 BLAS: matrix-vector operations
//! All matrices in column-major layout: A[i,j] = a[i + j*m]
//! Time complexity: O(mn)

use crate::LinalgError;

/// General matrix-vector multiply: y = α·A·x + β·y
/// A is m×n column-major
pub fn dgemv(m: usize, n: usize, alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape(format!("A length {} != {}x{}", a.len(), m, n))); }
    if x.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (x.len(), 1) }); }
    if y.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (y.len(), 1) }); }

    // y = β·y first
    if beta == 0.0 {
        for yi in y.iter_mut() { *yi = 0.0; }
    } else if beta != 1.0 {
        for yi in y.iter_mut() { *yi *= beta; }
    }
    if alpha == 0.0 { return Ok(()); }

    // y += α·A·x   (iterate over columns for cache efficiency — column-major)
    for j in 0..n {
        let axj = alpha * x[j];
        if axj == 0.0 { continue; }
        let col_start = j * m;
        for i in 0..m {
            y[i] += axj * a[col_start + i];
        }
    }
    Ok(())
}

/// Transposed matrix-vector multiply: y = α·Aᵀ·x + β·y
/// A is m×n column-major, so Aᵀ is n×m
pub fn dgemv_t(m: usize, n: usize, alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape(format!("A length mismatch"))); }
    if x.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (x.len(), 1) }); }
    if y.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (y.len(), 1) }); }

    if beta == 0.0 { for yi in y.iter_mut() { *yi = 0.0; } }
    else if beta != 1.0 { for yi in y.iter_mut() { *yi *= beta; } }
    if alpha == 0.0 { return Ok(()); }

    // Each y[j] = α · (column j of A) · x
    for j in 0..n {
        let col_start = j * m;
        let mut dot = 0.0;
        for i in 0..m { dot += a[col_start + i] * x[i]; }
        y[j] += alpha * dot;
    }
    Ok(())
}

/// Triangular solve: solve L·x = b (lower triangular, forward substitution)
/// L is n×n column-major, unit_diag means diagonal is implicitly 1
pub fn dtrsv_lower(n: usize, l: &[f64], b: &mut [f64], unit_diag: bool) -> Result<(), LinalgError> {
    if l.len() != n * n { return Err(LinalgError::InvalidShape("L not square".into())); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i { sum -= l[j * n + i] * b[j]; }
        if unit_diag { b[i] = sum; }
        else {
            let diag = l[i * n + i];
            if diag.abs() < 1e-15 { return Err(LinalgError::Singular); }
            b[i] = sum / diag;
        }
    }
    Ok(())
}

/// Triangular solve: solve U·x = b (upper triangular, back substitution)
pub fn dtrsv_upper(n: usize, u: &[f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if u.len() != n * n { return Err(LinalgError::InvalidShape("U not square".into())); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum -= u[j * n + i] * b[j]; }
        let diag = u[i * n + i];
        if diag.abs() < 1e-15 { return Err(LinalgError::Singular); }
        b[i] = sum / diag;
    }
    Ok(())
}

/// General triangular solve `op(A)·x = b`, in place on `b`. `A` is n×n
/// column-major; `upper` selects the upper (else lower) triangle; `transpose`
/// solves `Aᵀ·x = b`. Covers the four BLAS `dtrsv` uplo×trans cases. Used by
/// R's `backsolve`/`forwardsolve` (with their `upper.tri` / `transpose` flags).
pub fn dtrsv_gen(n: usize, a: &[f64], b: &mut [f64], upper: bool, transpose: bool) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::InvalidShape("A not square".into())); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }
    // `back` = iterate rows high→low (back substitution). Determined by whether
    // the *effective* operator op(A) is upper (back) or lower (forward):
    //   upper & !trans → upper (back);   lower & !trans → lower (forward)
    //   upper &  trans → lower (forward); lower &  trans → upper (back)
    let back = upper ^ transpose;
    // Off-diagonal element A_effective[i][k]: for op = A it's A[i][k]=a[k*n+i];
    // for op = Aᵀ it's A[k][i]=a[i*n+k]. Diagonal is a[i*n+i] either way.
    let rows: Vec<usize> = if back { (0..n).rev().collect() } else { (0..n).collect() };
    for &i in &rows {
        let mut sum = b[i];
        // Sum over already-solved unknowns (k>i if back, k<i if forward).
        let ks: Vec<usize> = if back { ((i + 1)..n).collect() } else { (0..i).collect() };
        for &k in &ks {
            let aik = if transpose { a[i * n + k] } else { a[k * n + i] };
            sum -= aik * b[k];
        }
        let diag = a[i * n + i];
        if diag.abs() < 1e-300 { return Err(LinalgError::Singular); }
        b[i] = sum / diag;
    }
    Ok(())
}

/// Rank-1 update: A = α·x·yᵀ + A
pub fn dger(m: usize, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("A shape".into())); }
    if x.len() != m || y.len() != n { return Err(LinalgError::DimensionMismatch { expected: (m, n), got: (x.len(), y.len()) }); }

    for j in 0..n {
        let ay = alpha * y[j];
        if ay == 0.0 { continue; }
        let col_start = j * m;
        for i in 0..m { a[col_start + i] += ay * x[i]; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgemv() {
        // A = [[1, 3], [2, 4]] column-major
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let mut y = vec![0.0, 0.0];
        dgemv(2, 2, 1.0, &a, &x, 0.0, &mut y).unwrap();
        // y[0] = 1*5 + 3*6 = 23
        // y[1] = 2*5 + 4*6 = 34
        assert_eq!(y, vec![23.0, 34.0]);
    }

    #[test]
    fn test_dtrsv_lower() {
        // L = [[2, 0], [1, 3]] column-major
        let l = vec![2.0, 1.0, 0.0, 3.0];
        let mut b = vec![4.0, 11.0];
        dtrsv_lower(2, &l, &mut b, false).unwrap();
        // x[0] = 4/2 = 2
        // x[1] = (11 - 1*2) / 3 = 3
        assert_eq!(b, vec![2.0, 3.0]);
    }

    #[test]
    fn test_dtrsv_gen_all_cases() {
        // Genuine upper-triangular U = [[2,1,4],[0,3,5],[0,0,6]] (column-major).
        let u = vec![2.0, 0.0, 0.0,  1.0, 3.0, 0.0,  4.0, 5.0, 6.0];
        // upper, no-trans: U y = [5,8,6] → [0,1,1].
        let mut b = vec![5.0, 8.0, 6.0];
        dtrsv_gen(3, &u, &mut b, true, false).unwrap();
        assert!((b[0]).abs() < 1e-12 && (b[1] - 1.0).abs() < 1e-12 && (b[2] - 1.0).abs() < 1e-12, "{:?}", b);
        // upper, trans: Uᵀ y = [5,8,6] → matches R (2.5, 1.8333.., -2.1944..).
        let mut b = vec![5.0, 8.0, 6.0];
        dtrsv_gen(3, &u, &mut b, true, true).unwrap();
        assert!((b[0] - 2.5).abs() < 1e-9 && (b[1] - 1.833333333).abs() < 1e-6 && (b[2] + 2.194444444).abs() < 1e-6, "{:?}", b);
    }
}
