//! Linear system solvers
//! Built on top of decompositions in `decomp.rs`.

use crate::{dgetrf, dpotrf, dtrsv_lower, dtrsv_upper, LinalgError};

/// Solve A·x = b using LU with partial pivoting
/// A is n×n, b is the right-hand side (consumed and replaced with solution)
pub fn dgesv(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    // Factorize A = P·L·U
    let ipiv = dgetrf(n, a)?;

    // Apply permutation to b
    let mut b_perm = vec![0.0; n];
    for i in 0..n { b_perm[i] = b[ipiv[i]]; }
    b.copy_from_slice(&b_perm);

    // Solve L·y = P·b (unit lower triangular, forward sub)
    dtrsv_lower(n, a, b, true)?;
    // Solve U·x = y (upper triangular, back sub)
    dtrsv_upper(n, a, b)?;
    Ok(())
}

/// Solve A·x = b where A is symmetric positive-definite (via Cholesky)
/// Much faster than dgesv when applicable (X'X in least squares)
pub fn dposv(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    // Factorize A = L·Lᵀ
    dpotrf(n, a)?;
    // Solve L·y = b
    dtrsv_lower(n, a, b, false)?;
    // Solve Lᵀ·x = y — need upper triangular solve using Lᵀ
    // Since A now stores L (lower), we do back-substitution using L transposed
    // L is in column-major lower triangle of a
    dtrsv_lt(n, a, b)?;
    Ok(())
}

/// Solve Lᵀ·x = b (upper triangular back-sub using lower-stored L)
fn dtrsv_lt(n: usize, l: &[f64], b: &mut [f64]) -> Result<(), LinalgError> {
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum -= l[i * n + j] * b[j]; }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 { return Err(LinalgError::Singular); }
        b[i] = sum / diag;
    }
    Ok(())
}

/// Invert a general matrix via LU decomposition
/// Returns A^(-1) in a new vector
pub fn dgetri(n: usize, a: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }

    let mut inv = vec![0.0; n * n];
    let mut lu = a.to_vec();
    let ipiv = dgetrf(n, &mut lu)?;

    // Solve A·X = I column by column
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        // Apply permutation
        let mut ep = vec![0.0; n];
        for i in 0..n { ep[i] = e[ipiv[i]]; }
        // Solve L·y = P·e then U·x = y
        dtrsv_lower(n, &lu, &mut ep, true)?;
        dtrsv_upper(n, &lu, &mut ep)?;
        // Store column
        for i in 0..n { inv[col * n + i] = ep[i]; }
    }
    Ok(inv)
}

/// Least squares: minimize ||A·x - b||² using normal equations
/// For overdetermined systems (m > n)
/// Uses Cholesky on X'X (fast but less accurate than QR for ill-conditioned)
pub fn dlsq_normal(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("A".into())); }
    if b.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (b.len(), 1) }); }

    // Compute X'X (n×n) and X'b (n)
    let mut xtx = vec![0.0; n * n];
    crate::dcrossprod(m, n, a, &mut xtx)?;

    let mut xtb = vec![0.0; n];
    crate::dgemv_t(m, n, 1.0, a, b, 0.0, &mut xtb)?;

    // Solve (X'X) β = X'b
    dposv(n, &mut xtx, &mut xtb)?;
    Ok(xtb)
}

/// Optimized least squares for tall-skinny systems (m >> n, n small)
/// Fuses X'X and X'y into a single pass over X for better cache use
pub fn dlsq_fused(m: usize, n: usize, x: &[f64], y: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if x.len() != m * n { return Err(LinalgError::InvalidShape("X".into())); }
    if y.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (y.len(), 1) }); }

    // Compute X'X and X'y simultaneously (one pass over X)
    let mut xtx = vec![0.0; n * n];
    let mut xty = vec![0.0; n];

    // Process 4 rows at a time for better vectorization
    let m4 = m - (m % 4);
    let mut r = 0;
    while r < m4 {
        let y0 = y[r]; let y1 = y[r+1]; let y2 = y[r+2]; let y3 = y[r+3];
        for j in 0..n {
            let xj0 = x[j * m + r]; let xj1 = x[j * m + r + 1];
            let xj2 = x[j * m + r + 2]; let xj3 = x[j * m + r + 3];
            xty[j] += xj0 * y0 + xj1 * y1 + xj2 * y2 + xj3 * y3;
            for i in j..n {
                let xi0 = x[i * m + r]; let xi1 = x[i * m + r + 1];
                let xi2 = x[i * m + r + 2]; let xi3 = x[i * m + r + 3];
                let dot = xi0 * xj0 + xi1 * xj1 + xi2 * xj2 + xi3 * xj3;
                xtx[j * n + i] += dot;
                if i != j { xtx[i * n + j] += dot; }
            }
        }
        r += 4;
    }
    // Handle remaining rows
    while r < m {
        let yr = y[r];
        for j in 0..n {
            let xjr = x[j * m + r];
            xty[j] += xjr * yr;
            for i in j..n {
                let dot = x[i * m + r] * xjr;
                xtx[j * n + i] += dot;
                if i != j { xtx[i * n + j] += dot; }
            }
        }
        r += 1;
    }

    // Solve via Cholesky
    dposv(n, &mut xtx, &mut xty)?;
    Ok(xty)
}

/// Determinant of an n×n (column-major) matrix via LU with partial
/// pivoting (`dgetrf`): det = sign(P) · ∏ U[i,i]. A singular matrix
/// (zero pivot) returns 0.0.
pub fn ddet(n: usize, a: &[f64]) -> Result<f64, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if n == 0 { return Ok(1.0); }
    let mut lu = a.to_vec();
    let ipiv = match crate::dgetrf(n, &mut lu) {
        Ok(p) => p,
        Err(LinalgError::Singular) => return Ok(0.0),
        Err(e) => return Err(e),
    };
    // Product of U's diagonal.
    let mut det = 1.0_f64;
    for i in 0..n { det *= lu[i * n + i]; }
    // Permutation parity via cycle decomposition: an even-length cycle
    // contributes an odd number of transpositions (flip the sign).
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            let mut j = i;
            let mut clen = 0usize;
            while !visited[j] { visited[j] = true; j = ipiv[j]; clen += 1; }
            if clen % 2 == 0 { det = -det; }
        }
    }
    Ok(det)
}

/// Least squares via Householder QR. Solves min‖Xβ − y‖₂ for the
/// m×n (m ≥ n) column-major X **without forming XᵀX**, so the condition
/// number is not squared — numerically stable for near-collinear
/// predictors where the normal-equations path (`dlsq_fused`) loses
/// accuracy. This is why R's `lm` uses QR.
///
/// X = QR (Householder); β solves Rβ = Qᵀy by back-substitution.
pub fn dlsq_qr(m: usize, n: usize, x: &[f64], y: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if x.len() != m * n { return Err(LinalgError::InvalidShape("lstsq: X shape".into())); }
    if y.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (y.len(), 1) }); }
    if m < n { return Err(LinalgError::InvalidShape("lstsq: need m >= n".into())); }

    // QR factorize a copy of X. After dgeqrf the upper triangle of `qr`
    // holds R; below the diagonal holds the Householder vectors v_k
    // (with v_k[k] = 1 implicit); `tau` holds the reflection scalars.
    let mut qr = x.to_vec();
    let tau = crate::dgeqrf(m, n, &mut qr)?;

    // Apply Qᵀ to y in place:  y ← y − τ_k · v_k · (v_kᵀ y),  k = 0..n.
    let mut b = y.to_vec();
    for k in 0..n {
        let mut vb = b[k]; // v_k[k] = 1
        for i in (k + 1)..m { vb += qr[k * m + i] * b[i]; }
        vb *= tau[k];
        b[k] -= vb;
        for i in (k + 1)..m { b[i] -= vb * qr[k * m + i]; }
    }

    // Back-substitution Rβ = (Qᵀy)[0..n].  R[i,j] = qr[j*m + i] for i ≤ j.
    let mut beta = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n { s -= qr[j * m + i] * beta[j]; }
        let rii = qr[i * m + i];
        if rii.abs() < 1e-300 { return Err(LinalgError::Singular); }
        beta[i] = s / rii;
    }
    Ok(beta)
}

/// Direct 2×2 solve (avoids all overhead for simple regression)
#[inline]
pub fn solve_2x2(a: &[f64; 4], b: &[f64; 2]) -> Result<[f64; 2], LinalgError> {
    // a is column-major 2×2: a[0]=a11, a[1]=a21, a[2]=a12, a[3]=a22
    let det = a[0] * a[3] - a[1] * a[2];
    if det.abs() < 1e-15 { return Err(LinalgError::Singular); }
    let inv_det = 1.0 / det;
    Ok([
        inv_det * (a[3] * b[0] - a[2] * b[1]),
        inv_det * (a[0] * b[1] - a[1] * b[0]),
    ])
}

/// Direct 3×3 solve via Cramer's rule (for 2-predictor regression with intercept)
#[inline]
pub fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Result<[f64; 3], LinalgError> {
    // Column-major 3×3
    let det = a[0]*(a[4]*a[8]-a[5]*a[7]) - a[3]*(a[1]*a[8]-a[2]*a[7]) + a[6]*(a[1]*a[5]-a[2]*a[4]);
    if det.abs() < 1e-15 { return Err(LinalgError::Singular); }
    let inv = 1.0 / det;
    Ok([
        inv * (b[0]*(a[4]*a[8]-a[5]*a[7]) - a[3]*(b[1]*a[8]-a[2]*b[2]) + a[6]*(b[1]*a[5]-a[2]*b[2])),
        inv * (a[0]*(b[1]*a[8]-b[2]*a[7]) - b[0]*(a[1]*a[8]-a[2]*a[7]) + a[6]*(a[1]*b[2]-a[2]*b[1])),
        inv * (a[0]*(a[4]*b[2]-a[5]*b[1]) - a[3]*(a[1]*b[2]-a[2]*b[1]) + b[0]*(a[1]*a[5]-a[2]*a[4])),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgesv() {
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0]; // column-major
        let mut b = vec![5.0, 10.0];
        dgesv(2, &mut a, &mut b).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dlsq_normal() {
        // Simple regression: y = 2x + 1 with noise
        // X = [[1, 1], [1, 2], [1, 3], [1, 4]] column-major
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![3.0, 5.0, 7.0, 9.0]; // exactly 2x + 1
        let beta = dlsq_normal(4, 2, &x, &y).unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-8); // intercept
        assert!((beta[1] - 2.0).abs() < 1e-8); // slope
    }

    #[test]
    fn test_dlsq_qr_matches_known_fit() {
        // Same exact y = 2x + 1 system; QR must recover [1, 2].
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let beta = dlsq_qr(4, 2, &x, &y).unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-10, "intercept {}", beta[0]);
        assert!((beta[1] - 2.0).abs() < 1e-10, "slope {}", beta[1]);
    }

    #[test]
    fn test_dlsq_qr_full_rank_ill_conditioned() {
        // Columns [1, x, x + δx²]: full rank (the x² term keeps col2
        // independent of span{1, x}) but ill-conditioned for small δ —
        // col2 ≈ col1. The data is exactly consistent, so the FIT must
        // be recovered near machine precision regardless of the
        // (large, ill-determined) individual coefficients. This is the
        // regime where normal equations (cond²) degrade and QR doesn't.
        let m = 6;
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let delta = 1e-3;
        let col2: Vec<f64> = xs.iter().map(|v| v + delta * v * v).collect();
        // column-major: col0 = 1s, col1 = x, col2 = x + δx²
        let mut x = Vec::with_capacity(m * 3);
        x.extend(std::iter::repeat(1.0).take(m));
        x.extend(xs.iter());
        x.extend(col2.iter());
        let y: Vec<f64> = (0..m).map(|i| 1.0 + 2.0 * xs[i] + 3.0 * col2[i]).collect();
        let beta = dlsq_qr(m, 3, &x, &y).unwrap();
        let mut max_resid = 0.0_f64;
        for i in 0..m {
            let pred = beta[0] + beta[1] * x[m + i] + beta[2] * x[2 * m + i];
            max_resid = max_resid.max((pred - y[i]).abs());
        }
        assert!(max_resid < 1e-8, "QR fit residual too large: {}", max_resid);
    }
}
