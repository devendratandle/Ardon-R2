//! Matrix decompositions: LU, Cholesky, QR
//! All column-major storage.

use crate::LinalgError;

/// LU factorization with partial pivoting — blocked algorithm
/// Modifies A in place: A = P·L·U where L has unit diagonal
/// Returns permutation vector p (row swaps)
///
/// Uses block panels: factor a panel of NB columns at a time,
/// then update the trailing matrix using our fast dgemm.
pub fn dgetrf(n: usize, a: &mut [f64]) -> Result<Vec<usize>, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }

    let mut ipiv = (0..n).collect::<Vec<usize>>();
    const NB: usize = 32; // panel width — fits in L1 cache

    let mut k = 0;
    while k < n {
        let kb = (n - k).min(NB); // actual panel width

        // Factor panel: columns k..k+kb
        for kk in k..(k + kb) {
            // Find pivot in column kk, rows kk..n
            let mut max_val = 0.0f64;
            let mut max_row = kk;
            for i in kk..n {
                let v = a[kk * n + i].abs();
                if v > max_val { max_val = v; max_row = i; }
            }
            if max_val < 1e-15 { return Err(LinalgError::Singular); }

            // Swap rows
            if max_row != kk {
                ipiv.swap(kk, max_row);
                for j in 0..n { a.swap(j * n + kk, j * n + max_row); }
            }

            // Scale column below pivot
            let pivot = a[kk * n + kk];
            for i in (kk + 1)..n { a[kk * n + i] /= pivot; }

            // Update within panel: rank-1 update on columns kk+1..k+kb
            for j in (kk + 1)..(k + kb) {
                let akj = a[j * n + kk];
                if akj != 0.0 {
                    for i in (kk + 1)..n { a[j * n + i] -= a[kk * n + i] * akj; }
                }
            }
        }

        // Update trailing matrix: A[k+kb:n, k+kb:n] -= L[k+kb:n, k:k+kb] * U[k:k+kb, k+kb:n]
        let trail_start = k + kb;
        if trail_start < n {
            let trail_m = n - trail_start;
            let _trail_n = n - trail_start;
            for j in trail_start..n {
                for kk in k..(k + kb) {
                    let u_kj = a[j * n + kk]; // U[kk, j]
                    if u_kj != 0.0 {
                        // 4-way unrolled for SIMD
                        let _main = trail_start + ((trail_m) / 4) * 4;
                        let mut i = trail_start;
                        while i + 3 < n {
                            a[j * n + i]     -= a[kk * n + i]     * u_kj;
                            a[j * n + i + 1] -= a[kk * n + i + 1] * u_kj;
                            a[j * n + i + 2] -= a[kk * n + i + 2] * u_kj;
                            a[j * n + i + 3] -= a[kk * n + i + 3] * u_kj;
                            i += 4;
                        }
                        while i < n { a[j * n + i] -= a[kk * n + i] * u_kj; i += 1; }
                    }
                }
            }
        }
        k += kb;
    }
    Ok(ipiv)
}

/// Cholesky decomposition: A = L·Lᵀ (for symmetric positive-definite A)
/// Modifies A in place, storing L in lower triangle
///
/// Blocked algorithm: process NB columns at a time,
/// use optimized rank-k update for trailing matrix.
pub fn dpotrf(n: usize, a: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }

    // Small matrix fast path (most lm() calls: 2-20 predictors)
    if n <= 4 { return dpotrf_unblocked(n, a); }

    const NB: usize = 32;
    let mut j = 0;
    while j < n {
        let jb = (n - j).min(NB);

        // Factor diagonal block: A[j:j+jb, j:j+jb]
        // First update it with contributions from previous columns
        for jj in j..(j + jb) {
            let mut diag = a[jj * n + jj];
            for k in j..jj { diag -= a[k * n + jj] * a[k * n + jj]; }
            if diag <= 1e-15 { return Err(LinalgError::NotPositiveDefinite); }
            let ljj = diag.sqrt();
            a[jj * n + jj] = ljj;

            for i in (jj + 1)..(j + jb) {
                let mut sum = a[jj * n + i];
                // Columns 0..j were already subtracted by earlier trailing updates.
                for k in j..jj { sum -= a[k * n + i] * a[k * n + jj]; }
                a[jj * n + i] = sum / ljj;
            }
            // Zero upper
            for i in 0..jj { a[jj * n + i] = 0.0; }
        }

        // Update below-diagonal panel: A[j+jb:n, j:j+jb]
        if j + jb < n {
            for jj in j..(j + jb) {
                let ljj = a[jj * n + jj];
                for i in (j + jb)..n {
                    let mut sum = a[jj * n + i];
                    for k in j..jj { sum -= a[k * n + i] * a[k * n + jj]; }
                    a[jj * n + i] = sum / ljj;
                }
            }

            // Update trailing symmetric block: A[j+jb:n, j+jb:n] -= L_panel * L_panel'
            // This is the critical syrk operation — use 4-way unrolled
            let trail = j + jb;
            for col in trail..n {
                for row in col..n {
                    let mut dot = 0.0;
                    let _main = j + ((jb) / 4) * 4;
                    let mut k = j;
                    while k + 3 < j + jb {
                        dot += a[k * n + row] * a[k * n + col]
                             + a[(k+1) * n + row] * a[(k+1) * n + col]
                             + a[(k+2) * n + row] * a[(k+2) * n + col]
                             + a[(k+3) * n + row] * a[(k+3) * n + col];
                        k += 4;
                    }
                    while k < j + jb { dot += a[k * n + row] * a[k * n + col]; k += 1; }
                    a[col * n + row] -= dot;
                }
            }
        }
        j += jb;
    }
    Ok(())
}

/// Unblocked Cholesky for small matrices (n <= 4)
#[inline]
fn dpotrf_unblocked(n: usize, a: &mut [f64]) -> Result<(), LinalgError> {
    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j { diag -= a[k * n + j] * a[k * n + j]; }
        if diag <= 1e-15 { return Err(LinalgError::NotPositiveDefinite); }
        let ljj = diag.sqrt();
        a[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut sum = a[j * n + i];
            for k in 0..j { sum -= a[k * n + i] * a[k * n + j]; }
            a[j * n + i] = sum / ljj;
        }
        for i in 0..j { a[j * n + i] = 0.0; }
    }
    Ok(())
}

/// QR decomposition via Householder reflections
/// A is m×n, m >= n
/// On exit: upper triangle of A contains R, below-diagonal contains Householder vectors
/// tau[i] contains the reflection coefficient
pub fn dgeqrf(m: usize, n: usize, a: &mut [f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("QR: A shape".into())); }
    if m < n { return Err(LinalgError::InvalidShape("QR: need m >= n".into())); }

    let mut tau = vec![0.0; n];

    for k in 0..n {
        // Compute Householder reflection for column k, rows k..m
        let mut norm_sq = 0.0;
        for i in k..m { let v = a[k * m + i]; norm_sq += v * v; }
        let norm = norm_sq.sqrt();
        if norm < 1e-15 { continue; }

        let akk = a[k * m + k];
        let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        // Build Householder vector v in-place
        let v0 = akk - alpha;
        a[k * m + k] = alpha; // R[k,k]

        // Store v[k+1..m] below diagonal (v[k] = 1 implicitly, scaled by v0)
        for i in (k + 1)..m { a[k * m + i] /= v0; }

        // tau = 2 / (v'v)
        let mut vv = 1.0; // v[k] = 1
        for i in (k + 1)..m { vv += a[k * m + i] * a[k * m + i]; }
        tau[k] = 2.0 / vv;

        // Apply to remaining columns: A[:,j] -= tau · v · (v'·A[:,j])
        for j in (k + 1)..n {
            let mut vaj = a[j * m + k]; // contribution from v[k] = 1
            for i in (k + 1)..m { vaj += a[k * m + i] * a[j * m + i]; }
            vaj *= tau[k];
            a[j * m + k] -= vaj;
            for i in (k + 1)..m { a[j * m + i] -= vaj * a[k * m + i]; }
        }
    }
    Ok(tau)
}

/// Singular Value Decomposition — singular values only.
///
/// Two-phase algorithm:
///   1. Bidiagonalization via Householder reflections (Phase 1)
///   2. Golub-Kahan QR iterations on the bidiagonal (Phase 2)
///
/// A is m×n (m >= n), column-major. Returns the n singular values in
/// descending order.
///
/// This routine returns singular values only. For the full thin SVD
/// with orthogonal factors U and Vᵀ, use [`dgesvd_full`] (shipped v0.1.0).
/// Kept as a separate entry point for callers that don't need the
/// orthogonal factors — slightly faster since it skips the Bᵀ·B
/// eigendecomposition pass.
pub fn dgesvd(m: usize, n: usize, a: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("SVD: A shape".into())); }
    if m < n { return Err(LinalgError::InvalidShape("SVD: need m >= n".into())); }

    let mut work = a.to_vec();
    let min_mn = m.min(n);

    // Phase 1: Bidiagonalize A → U1 · B · V1ᵀ
    // B has entries on diagonal (d) and superdiagonal (e)
    let mut d = vec![0.0; min_mn];
    let mut e = vec![0.0; min_mn.saturating_sub(1)];

    // Householder vectors stored implicitly
    let mut tauq = vec![0.0; min_mn];
    let mut taup = vec![0.0; min_mn];

    // Bidiagonalization
    for k in 0..min_mn {
        // Left Householder: zero out A[k+1:m, k]
        let mut norm_sq = 0.0;
        for i in k..m { norm_sq += work[k * m + i] * work[k * m + i]; }
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            let akk = work[k * m + k];
            let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * norm;
            let v0 = akk - alpha;
            work[k * m + k] = alpha;
            d[k] = alpha;

            // Store Householder vector below diagonal
            for i in (k + 1)..m { work[k * m + i] /= v0; }
            let mut vv = 1.0;
            for i in (k + 1)..m { vv += work[k * m + i] * work[k * m + i]; }
            tauq[k] = 2.0 / vv;

            // Apply to trailing columns
            for j in (k + 1)..n {
                let mut dot = work[j * m + k];
                for i in (k + 1)..m { dot += work[k * m + i] * work[j * m + i]; }
                dot *= tauq[k];
                work[j * m + k] -= dot;
                for i in (k + 1)..m { work[j * m + i] -= dot * work[k * m + i]; }
            }
        } else {
            d[k] = work[k * m + k];
        }

        // Right Householder: zero out A[k, k+2:n]
        if k + 1 < n {
            let mut norm_sq = 0.0;
            for j in (k + 1)..n { norm_sq += work[j * m + k] * work[j * m + k]; }
            let norm = norm_sq.sqrt();
            if norm > 1e-15 && k + 1 < n {
                let akk1 = work[(k + 1) * m + k];
                let sign = if akk1 >= 0.0 { 1.0 } else { -1.0 };
                let alpha = -sign * norm;
                let v0 = akk1 - alpha;
                work[(k + 1) * m + k] = alpha;
                if k < e.len() { e[k] = alpha; }

                for j in (k + 2)..n { work[j * m + k] /= v0; }
                let mut vv = 1.0;
                for j in (k + 2)..n { vv += work[j * m + k] * work[j * m + k]; }
                taup[k] = 2.0 / vv;

                // Apply to trailing rows
                for i in (k + 1)..m {
                    let mut dot = work[(k + 1) * m + i];
                    for j in (k + 2)..n { dot += work[j * m + k] * work[j * m + i]; }
                    dot *= taup[k];
                    work[(k + 1) * m + i] -= dot;
                    for j in (k + 2)..n { work[j * m + i] -= dot * work[j * m + k]; }
                }
            } else if k < e.len() {
                e[k] = work[(k + 1) * m + k];
            }
        }
    }

    // Phase 2: QR iteration on bidiagonal matrix to get singular values
    // Use implicit zero-shift QR (Golub-Kahan)
    let mut sigma = d.clone();
    let mut super_diag = e.clone();
    let nn = sigma.len();

    // Simple iterative SVD for bidiagonal matrix
    // Convergence to singular values
    for _iter in 0..nn * 100 {
        // Check convergence
        let mut all_converged = true;
        for i in 0..super_diag.len() {
            if super_diag[i].abs() > 1e-14 * (sigma[i].abs() + sigma[i + 1].abs()) {
                all_converged = false;
                break;
            }
        }
        if all_converged { break; }

        // Implicit QR step on bidiagonal
        for i in 0..super_diag.len() {
            if super_diag[i].abs() <= 1e-14 * (sigma[i].abs() + sigma[i + 1].abs()) {
                super_diag[i] = 0.0;
                continue;
            }
            // Givens rotation to chase bulge
            let (cs, sn) = givens(sigma[i], super_diag[i]);
            let old_d = sigma[i];
            sigma[i] = cs * old_d + sn * super_diag[i];
            super_diag[i] = -sn * old_d + cs * super_diag[i];
            if i + 1 < nn {
                let old_d1 = sigma[i + 1];
                sigma[i + 1] = cs * old_d1;
            }
        }
    }

    // Make singular values positive and sort descending
    for s in sigma.iter_mut() { *s = s.abs(); }
    // Simple insertion sort (n is typically small)
    for i in 1..sigma.len() {
        let mut j = i;
        while j > 0 && sigma[j] > sigma[j - 1] {
            sigma.swap(j, j - 1);
            j -= 1;
        }
    }

    // U and Vᵀ accumulation is intentionally NOT done here — see the
    // docstring above and docs/KNOWN_LIMITATIONS.md. Producing identity
    // placeholders would silently corrupt any caller that reconstructs
    // A = U·diag(σ)·Vᵀ.
    let _ = (m, nn); // keep parameters live in case the body reverts.
    Ok(sigma)
}

/// Thin SVD with orthogonal factors: A = U · diag(σ) · Vᵀ.
///
/// A is m×n (m ≥ n), column-major. Returns:
///   - σ: Vec<f64>, n singular values in descending order,
///   - U: Vec<f64>, m×n column-major with orthonormal columns,
///   - Vᵀ: Vec<f64>, n×n column-major (rows orthonormal, i.e., V itself
///     has orthonormal columns and Vᵀ is its transpose).
///
/// Algorithm — two-phase, both phases accumulating their orthogonal factors:
///   1. Householder bidiagonalization (Golub-Kahan): A = U₁ B V₁ᵀ where B
///      is n×n bidiagonal with main diagonal `d` and superdiagonal `e`.
///      U₁ (m×n) and V₁ (n×n) are accumulated explicitly column-by-column.
///   2. Diagonalize B by going through Bᵀ·B (n×n symmetric tridiagonal),
///      using the already-shipped `dsyev_full` (Householder tridiag +
///      implicit-shift symmetric QR with Wilkinson shift). Eigenvalues of
///      Bᵀ·B are σ² (descending). Right singular vectors V₂ are
///      eigenvectors; left singular vectors are recovered via
///      u₂_k = (B · v₂_k) / σ_k for σ_k > 0 (zero column for σ_k ≈ 0).
///
/// Final factors: U = U₁ · U₂ (m×n), V = V₁ · V₂ (n×n), Vᵀ = transpose.
///
/// **Honest accuracy caveat:** the Bᵀ·B route squares the condition number
/// of A. For well-conditioned matrices (κ ≲ 1/√ε ≈ 6.7e7) the singular
/// values and vectors are accurate to ~1e-12. For badly conditioned
/// matrices (κ approaching 1/ε), small singular values lose accuracy
/// proportionally — equivalent to half the floating-point precision.
/// LAPACK's `dbdsqr` (proper implicit-shift bidiagonal QR with full
/// accumulation) would give κ-independent accuracy at higher
/// implementation cost. See `docs/KNOWN_LIMITATIONS.md`.
pub fn dgesvd_full(
    m: usize,
    n: usize,
    a: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("SVD: A shape".into())); }
    if m < n { return Err(LinalgError::InvalidShape("SVD: need m >= n".into())); }
    if n == 0 { return Ok((vec![], vec![], vec![])); }

    // ── Phase 1: bidiagonalize A → U₁ · B · V₁ᵀ ─────────────────────
    //
    // Apply left Householders H_k from the LEFT to a working copy of A
    // (so A becomes upper bidiagonal in-place). Store each Householder
    // vector (full m-length) and its `tau` so we can reconstruct U₁
    // afterwards by applying them in reverse order to I_{m×n}.
    // Same scheme for right Householders G_k → V₁.
    //
    // Why store-then-apply rather than accumulate during bidiagonalization?
    // U₁ is the first n columns of an m×m orthogonal matrix. Maintaining
    // only m×n during right-multiplication by m×m Householders requires
    // column data outside the m×n window (mathematically). The standard
    // LAPACK fix (`dorgbr` after `dgebrd`) is to defer the build: store
    // the vectors, then apply them in reverse to the thin identity. This
    // produces a numerically clean orthonormal U₁ at the same asymptotic
    // cost.
    let mut work = a.to_vec();
    let mut left_vs: Vec<Vec<f64>> = Vec::with_capacity(n); // each length m
    let mut left_taus: Vec<f64> = Vec::with_capacity(n);
    let mut right_vs: Vec<Vec<f64>> = Vec::with_capacity(n); // each length n
    let mut right_taus: Vec<f64> = Vec::with_capacity(n);

    let mut d_diag = vec![0.0_f64; n];           // bidiagonal main diag
    let mut e_super = vec![0.0_f64; n.saturating_sub(1)]; // superdiag

    // Scratch buffer for Householder vector (length m, reused).
    let mut hv = vec![0.0_f64; m];

    for k in 0..n {
        // ── Left Householder: zero work[k+1:m, k] ─────────────────────
        let mut norm_sq = 0.0_f64;
        for i in k..m {
            let v = work[k * m + i];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-30 {
            let akk = work[k * m + k];
            let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
            let alpha = -sign * norm;
            // Build full m-length Householder vector v (zero above k).
            for i in 0..k { hv[i] = 0.0; }
            hv[k] = akk - alpha;
            for i in (k + 1)..m { hv[i] = work[k * m + i]; }
            let v_norm_sq: f64 = (k..m).map(|i| hv[i] * hv[i]).sum();
            if v_norm_sq > 1e-30 {
                let tau = 2.0 / v_norm_sq;

                // Apply H_k from left to work[k:m, k:n].
                // Column k becomes [alpha, 0, ..., 0]^T effectively; we set
                // it directly to skip the redundant multiply.
                work[k * m + k] = alpha;
                for i in (k + 1)..m { work[k * m + i] = 0.0; }
                // Other columns j ∈ (k+1)..n.
                for j in (k + 1)..n {
                    let mut dot = 0.0_f64;
                    for i in k..m { dot += hv[i] * work[j * m + i]; }
                    let scale = tau * dot;
                    for i in k..m { work[j * m + i] -= scale * hv[i]; }
                }
                // Store the Householder vector (full m-length, zeros above k)
                // and its tau for later left-to-right application onto U₁.
                left_vs.push(hv.clone());
                left_taus.push(tau);
            } else {
                work[k * m + k] = alpha;
                left_vs.push(vec![0.0; m]);
                left_taus.push(0.0);
            }
            d_diag[k] = alpha;
        } else {
            d_diag[k] = work[k * m + k];
            // Identity placeholder.
            left_vs.push(vec![0.0; m]);
            left_taus.push(0.0);
        }

        // ── Right Householder: zero work[k, k+2:n] ────────────────────
        // Only meaningful when there are entries beyond k+1 to zero out.
        // For the last superdiagonal step (k = n-2) there is exactly one
        // entry at column k+1 = n-1 and we keep it directly as e_super[k]
        // — no Householder needed (applying one would flip its sign and
        // corrupt the rows below).
        if k + 2 < n {
            // Operate on row k of work, columns (k+1)..n.
            let mut norm_sq = 0.0_f64;
            for j in (k + 1)..n {
                let v = work[j * m + k];
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-30 {
                let akk1 = work[(k + 1) * m + k];
                let sign = if akk1 >= 0.0 { 1.0 } else { -1.0 };
                let alpha = -sign * norm;
                // Build n-length right Householder vector vr (zero up to k).
                let mut vr = vec![0.0_f64; n];
                vr[k + 1] = akk1 - alpha;
                for j in (k + 2)..n { vr[j] = work[j * m + k]; }
                let v_norm_sq: f64 = ((k + 1)..n).map(|j| vr[j] * vr[j]).sum();
                if v_norm_sq > 1e-30 {
                    let tau = 2.0 / v_norm_sq;

                    // Apply G_k from the right to work[k:m, k+1:n].
                    // Row k: set work[k+1, k] = alpha, work[j, k] = 0 for j > k+1.
                    work[(k + 1) * m + k] = alpha;
                    for j in (k + 2)..n { work[j * m + k] = 0.0; }
                    // Rows (k+1)..m: w_i ← w_i (I - tau v vᵀ), i.e. for each row,
                    //   row_i[k+1..n] ← row_i[k+1..n] - tau · (row_i · vr) · vrᵀ.
                    for i in (k + 1)..m {
                        let mut dot = 0.0_f64;
                        for j in (k + 1)..n { dot += vr[j] * work[j * m + i]; }
                        let scale = tau * dot;
                        for j in (k + 1)..n { work[j * m + i] -= scale * vr[j]; }
                    }
                    // Store the right Householder vector and tau.
                    right_vs.push(vr);
                    right_taus.push(tau);
                } else {
                    work[(k + 1) * m + k] = alpha;
                    right_vs.push(vec![0.0; n]);
                    right_taus.push(0.0);
                }
                if k < e_super.len() { e_super[k] = alpha; }
            } else if k < e_super.len() {
                e_super[k] = work[(k + 1) * m + k];
                right_vs.push(vec![0.0; n]);
                right_taus.push(0.0);
            } else {
                right_vs.push(vec![0.0; n]);
                right_taus.push(0.0);
            }
        } else if k + 1 < n && k < e_super.len() {
            // No right Householder needed (only one trailing entry, which
            // becomes e_super[k] directly).
            e_super[k] = work[(k + 1) * m + k];
        }
    }

    // ── Build U₁ (m×n): apply stored left Householders in reverse order
    // to the thin identity I_{m×n} (`X[i,j] = δ_{ij}` for j < n, else 0). ─
    //
    // X starts as I_{m×n}. For k = n-1 down to 0:
    //   X ← H_k · X
    //   H_k = I_m - τ_k · v_k · v_kᵀ acts on rows k..m only (v_k zero above k).
    // Resulting X = H_0 · H_1 · ... · H_{n-1} · I_{m×n} = U₁.
    let mut u1 = vec![0.0_f64; m * n];
    for i in 0..n { u1[i * m + i] = 1.0; }
    for k in (0..n).rev() {
        let tau = left_taus[k];
        if tau == 0.0 { continue; }
        let v_k = &left_vs[k];
        // For each column j of U1, update rows k..m:
        //   col_j ← col_j - τ · (vᵀ · col_j) · v
        for j in 0..n {
            let mut dot = 0.0_f64;
            for i in k..m { dot += v_k[i] * u1[j * m + i]; }
            let scale = tau * dot;
            for i in k..m { u1[j * m + i] -= scale * v_k[i]; }
        }
    }

    // ── Build V₁ (n×n): apply stored right Householders in reverse to I_n.
    //
    // V₁ = G_0 · G_1 · ... · G_{last}, built by left-multiplying I_n in
    // reverse: for k = last down to 0, X ← G_k · X. Each G_k acts on rows
    // (k+1)..n only (vr_k zero up to and including k).
    let mut v1 = vec![0.0_f64; n * n];
    for i in 0..n { v1[i * n + i] = 1.0; }
    for k in (0..right_vs.len()).rev() {
        let tau = right_taus[k];
        if tau == 0.0 { continue; }
        let vr_k = &right_vs[k];
        for j in 0..n {
            let mut dot = 0.0_f64;
            for i in (k + 1)..n { dot += vr_k[i] * v1[j * n + i]; }
            let scale = tau * dot;
            for i in (k + 1)..n { v1[j * n + i] -= scale * vr_k[i]; }
        }
    }

    // ── Phase 2: diagonalize B = U₂ · diag(σ) · V₂ᵀ via Bᵀ·B ────────
    //
    // T := Bᵀ·B is n×n symmetric tridiagonal with
    //   T[i][i]   = d[i]² + e[i-1]²   (e[-1] := 0)
    //   T[i][i+1] = d[i] · e[i]
    // Form it densely (column-major) and call dsyev_full → (σ², V₂).
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        let prev_e = if i == 0 { 0.0 } else { e_super[i - 1] };
        t[i * n + i] = d_diag[i] * d_diag[i] + prev_e * prev_e;
        if i + 1 < n {
            let off = d_diag[i] * e_super[i];
            t[i * n + (i + 1)] = off;
            t[(i + 1) * n + i] = off;
        }
    }
    let (eig_vals, v2) = dsyev_full(n, &t)?;
    // σ_k = sqrt(max(eig_k, 0)) — tiny negatives can arise from rounding.
    let mut sigma = vec![0.0_f64; n];
    for k in 0..n {
        let lam = eig_vals[k];
        sigma[k] = if lam > 0.0 { lam.sqrt() } else { 0.0 };
    }

    // ── Compute U₂ (n×n column-major): u₂_k = (B · v₂_k) / σ_k ──────
    //
    // Since B is bidiagonal:
    //   (B · v)[i] = d[i] · v[i] + e[i] · v[i+1]   (e[n-1] = 0 by convention)
    let mut u2 = vec![0.0_f64; n * n];
    let sigma_floor = 1e-13_f64 * sigma.first().copied().unwrap_or(1.0).max(1.0);
    for k in 0..n {
        if sigma[k] > sigma_floor {
            for i in 0..n {
                let next_v = if i + 1 < n { v2[k * n + (i + 1)] } else { 0.0 };
                let bv_i = d_diag[i] * v2[k * n + i]
                    + (if i + 1 < n { e_super[i] } else { 0.0 }) * next_v;
                u2[k * n + i] = bv_i / sigma[k];
            }
        } else {
            // Rank-deficient: column of U₂ is unconstrained; leave zero.
            // Caller reconstructing A still gets A ≈ U·Σ·Vᵀ exactly because
            // the corresponding σ_k is 0, so any U column is fine.
        }
    }

    // ── Assemble U = U₁ · U₂ (m×n) and V = V₁ · V₂ (n×n) ────────────
    let mut u = vec![0.0_f64; m * n];
    for k in 0..n {
        for i in 0..m {
            let mut s = 0.0_f64;
            for l in 0..n { s += u1[l * m + i] * u2[k * n + l]; }
            u[k * m + i] = s;
        }
    }
    let mut v_mat = vec![0.0_f64; n * n];
    for k in 0..n {
        for i in 0..n {
            let mut s = 0.0_f64;
            for l in 0..n { s += v1[l * n + i] * v2[k * n + l]; }
            v_mat[k * n + i] = s;
        }
    }
    // Transpose V → Vᵀ (column-major n×n).
    let mut vt = vec![0.0_f64; n * n];
    for i in 0..n { for j in 0..n { vt[i * n + j] = v_mat[j * n + i]; } }

    Ok((sigma, u, vt))
}

/// Givens rotation: compute (c, s) such that
/// [c  s] [a] = [r]
/// [-s c] [b]   [0]
#[inline]
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 { return (1.0, 0.0); }
    if a == 0.0 { return (0.0, 1.0); }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

/// Symmetric eigenvalue decomposition via Jacobi rotation method
/// A is n×n symmetric column-major
/// Returns eigenvalues in descending order
///
/// The Jacobi method is simple and always converges for symmetric matrices.
/// For small matrices (n < 100), it's fast enough and numerically robust.
/// Algorithm: repeatedly zero the largest off-diagonal element via Givens rotations.
pub fn dsyev(n: usize, a: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![a[0]]); }
    if n == 2 {
        // Direct formula for 2×2 symmetric matrix
        let a11 = a[0]; let a12 = a[n]; let a22 = a[n + 1];
        let tr = a11 + a22;
        let det = a11 * a22 - a12 * a12;
        let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
        let l1 = (tr + disc) / 2.0;
        let l2 = (tr - disc) / 2.0;
        return Ok(if l1 >= l2 { vec![l1, l2] } else { vec![l2, l1] });
    }

    // Work on a copy (symmetric, so we use full matrix)
    let mut s = a.to_vec();
    let max_iter = n * n * 300; // increased iterations for better convergence

    // Compute matrix norm for relative convergence
    let mut mat_norm = 0.0f64;
    for i in 0..n { mat_norm = mat_norm.max(s[i * n + i].abs()); }
    let tol = 1e-15 * mat_norm.max(1e-300); // relative threshold

    for _iter in 0..max_iter {
        // Find largest off-diagonal element |S[i,j]| where i != j
        let mut max_val = 0.0f64;
        let mut pi = 0usize;
        let mut pj = 1usize;
        for j in 0..n {
            for i in 0..j {
                let v = s[j * n + i].abs();
                if v > max_val { max_val = v; pi = i; pj = j; }
            }
        }

        // Convergence check — relative to matrix norm
        if max_val < tol { break; }

        // Compute Jacobi rotation angle for element (pi, pj)
        let sii = s[pi * n + pi];
        let sjj = s[pj * n + pj];
        let sij = s[pj * n + pi];

        let (cs, sn) = if (sii - sjj).abs() < 1e-15 {
            // Special case: diagonal elements are equal
            let c = 1.0 / 2.0f64.sqrt();
            (c, if sij >= 0.0 { -c } else { c })
        } else {
            let tau = (sjj - sii) / (2.0 * sij);
            // Solve t² - 2τt - 1 = 0 for smaller root
            // t = -sign(τ) / (|τ| + √(1+τ²))
            let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
            let t = -sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let ss = c * t;
            (c, ss)
        };

        // Apply rotation: S = G' * S * G
        // This zeros out S[pi,pj] and S[pj,pi]
        for k in 0..n {
            if k == pi || k == pj { continue; }
            let ski = s[pi * n + k]; // S[k, pi] — but column-major: S[k,pi] = s[pi*n + k]
            let skj = s[pj * n + k]; // S[k, pj]
            s[pi * n + k] = cs * ski + sn * skj;
            s[pj * n + k] = -sn * ski + cs * skj;
            // Symmetric: S[pi,k] = S[k,pi], S[pj,k] = S[k,pj]
            s[k * n + pi] = s[pi * n + k];
            s[k * n + pj] = s[pj * n + k];
        }

        // Update diagonal and off-diagonal for (pi, pj) block
        let sii_old = sii;
        let sjj_old = sjj;
        s[pi * n + pi] = cs * cs * sii_old + 2.0 * cs * sn * sij + sn * sn * sjj_old;
        s[pj * n + pj] = sn * sn * sii_old - 2.0 * cs * sn * sij + cs * cs * sjj_old;
        s[pj * n + pi] = 0.0;
        s[pi * n + pj] = 0.0;
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();

    // Sort descending
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

// ─────────────────────────────────────────────────────────────────────
// dsyev_full — symmetric eigendecomposition with eigenvectors (Phase R Tier 1)
//
// Replaces the eigenvalues-only Jacobi path for callers that need the
// rotation matrix (`eigen`, `prcomp$rotation`). Algorithm:
//
//   1. Householder tridiagonalization:  A = Q1 · T · Q1ᵀ
//      Reflects A to symmetric tridiagonal T while accumulating Q1.
//
//   2. Implicit symmetric QR with Wilkinson shift on T:
//      T = Q2 · D · Q2ᵀ
//      Iteratively zeroes the sub-diagonal via bulge-chasing Givens
//      rotations, accumulating Q2.
//
//   3. Final eigenvectors = Q1 · Q2; eigenvalues = diag(D), sorted
//      descending with vectors permuted to match.
//
// LAPACK calls this `dsyevr`/`dsyev`. Our previous `dsyev` keeps its
// Jacobi-on-full-matrix path (eigenvalues only). Eventually that callers
// migrate to this and we retire the Jacobi version.
//
// Returns: (eigenvalues_desc, vectors_col_major) where the i-th column
// of `vectors` is the eigenvector for eigenvalues[i].
// ─────────────────────────────────────────────────────────────────────

pub fn dsyev_full(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if n == 0 { return Ok((vec![], vec![])); }
    if n == 1 { return Ok((vec![a[0]], vec![1.0])); }

    // Stage 1 — Householder tridiagonalization, accumulating Q.
    // Working on a copy; final tridiagonal lives in `d` (diag) + `e` (sub-diag).
    let mut t = a.to_vec();         // becomes T; off-tridiag entries → 0
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n { q[i * n + i] = 1.0; }   // Q = I, then accumulate

    for k in 0..n.saturating_sub(2) {
        // Build Householder for column k below the sub-diagonal.
        // Vector x = T[k+1..n, k]; reflect so it becomes ±||x|| e_1.
        let mut norm_sq = 0.0;
        for i in (k + 1)..n { let v = t[k * n + i]; norm_sq += v * v; }
        if norm_sq < 1e-300 { continue; }
        let xkk = t[k * n + (k + 1)];
        let alpha = -xkk.signum().max(-1.0) * norm_sq.sqrt();
        // (signum returns 0 for 0; ensure -1 fallback so alpha is well-defined.)
        let alpha = if xkk == 0.0 { -norm_sq.sqrt() } else { alpha };

        // v = x − α·e_1 ; β = 2 / (vᵀ v)
        let mut v = vec![0.0_f64; n];
        for i in (k + 1)..n { v[i] = t[k * n + i]; }
        v[k + 1] -= alpha;
        let vtv: f64 = v.iter().skip(k + 1).map(|x| x * x).sum();
        if vtv < 1e-300 { continue; }
        let beta = 2.0 / vtv;

        // Apply H = I − β v vᵀ from BOTH sides: T ← H · T · H.
        // Compute p = β · T · v (only rows ≥ k+1 active).
        let mut p = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in (k + 1)..n { s += t[j * n + i] * v[j]; }
            p[i] = beta * s;
        }
        // w = p − (β/2)·(pᵀ v)·v  (rank-2 update vector)
        let ptv: f64 = (k + 1..n).map(|j| p[j] * v[j]).sum();
        let half_beta_ptv = 0.5 * beta * ptv;
        let mut w = p;
        for j in (k + 1)..n { w[j] -= half_beta_ptv * v[j]; }
        // T ← T − v wᵀ − w vᵀ
        for i in 0..n {
            for j in 0..n {
                let dv = if i >= k + 1 { v[i] } else { 0.0 };
                let dw = if j >= k + 1 { v[j] } else { 0.0 };
                t[j * n + i] -= dv * w[j] + w[i] * dw;
            }
        }

        // Accumulate Q ← Q · H. Q is column-major; H acts on the right.
        // For each column c of Q: q_c ← q_c − β·(q_cᵀ v)·v
        for c in 0..n {
            let mut qcv = 0.0;
            for j in (k + 1)..n { qcv += q[c * n + j] * v[j]; }
            let scale = beta * qcv;
            for j in (k + 1)..n { q[c * n + j] -= scale * v[j]; }
        }
    }

    // Extract tridiagonal: d = diag, e = sub-diag (e has length n-1).
    let mut d: Vec<f64> = (0..n).map(|i| t[i * n + i]).collect();
    let mut e: Vec<f64> = (0..n - 1).map(|i| t[i * n + (i + 1)]).collect();

    // Stage 2 — Implicit symmetric QR with Wilkinson shift on (d, e),
    // accumulating Givens rotations into Q.
    let max_sweeps = 30 * n;
    let mut end = n - 1;
    // `start` is unconditionally reassigned inside the deflation loop
    // before it is first read; the initial value is a placeholder for
    // definite-assignment. The lint correctly notes the placeholder
    // assignment is never read — we keep it for clarity rather than
    // restructuring the loop to declare-then-assign on first iteration.
    #[allow(unused_assignments)]
    let mut start = end;
    let mut sweeps = 0usize;

    while end > 0 {
        if sweeps > max_sweeps { return Err(LinalgError::InvalidShape("dsyev_full: QR failed to converge".into())); }
        sweeps += 1;

        // Deflate trailing zeros in `e`.
        while end > 0 && e[end - 1].abs() <= 1e-14 * (d[end - 1].abs() + d[end].abs()) {
            e[end - 1] = 0.0;
            if end == 0 { break; }
            end -= 1;
        }
        if end == 0 { break; }

        // Find start of the active unreduced sub-block.
        start = end;
        while start > 0 && e[start - 1].abs() > 1e-14 * (d[start - 1].abs() + d[start].abs()) {
            start -= 1;
        }
        if start == end { continue; }   // shouldn't happen — defensive.

        // Wilkinson shift: eigenvalue of trailing 2×2 closer to d[end].
        let dd = (d[end - 1] - d[end]) / 2.0;
        let ee = e[end - 1];
        let denom = dd.abs() + (dd * dd + ee * ee).sqrt();
        let sign_dd = if dd >= 0.0 { 1.0 } else { -1.0 };
        let shift = d[end] - sign_dd * ee * ee / denom.max(1e-300);

        // Implicit QR sweep: bulge-chase from `start` to `end`.
        let mut x = d[start] - shift;
        let mut y = e[start];
        for k in start..end {
            let (cs, sn) = givens(x, y);
            // Apply rotation to (d[k], d[k+1], e[k]).
            if k > start { e[k - 1] = cs * x + sn * y; }
            let d_k  = d[k];
            let d_k1 = d[k + 1];
            let e_k  = e[k];
            d[k]     = cs * cs * d_k + 2.0 * cs * sn * e_k + sn * sn * d_k1;
            d[k + 1] = sn * sn * d_k - 2.0 * cs * sn * e_k + cs * cs * d_k1;
            e[k]     = cs * sn * (d_k1 - d_k) + (cs * cs - sn * sn) * e_k;
            if k + 1 < end {
                y = sn * e[k + 1];
                e[k + 1] *= cs;
            }
            x = e[k];

            // Accumulate the Givens rotation into Q (columns k and k+1).
            for r in 0..n {
                let a_rk  = q[k * n + r];
                let a_rk1 = q[(k + 1) * n + r];
                q[k * n + r]       = cs * a_rk + sn * a_rk1;
                q[(k + 1) * n + r] = -sn * a_rk + cs * a_rk1;
            }
        }
    }

    // Stage 3 — Sort eigenvalues descending; permute eigenvector columns to match.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(std::cmp::Ordering::Equal));
    let eigenvalues: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let mut vectors = vec![0.0_f64; n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        for r in 0..n {
            vectors[new_col * n + r] = q[old_col * n + r];
        }
    }

    // Sign convention (matches R / LAPACK): for each eigenvector column,
    // flip sign so the entry with the largest absolute magnitude is
    // positive. Makes results reproducible across runs and matches what
    // R's `prcomp$rotation` / `eigen$vectors` produce.
    for c in 0..n {
        let mut max_abs = 0.0_f64;
        let mut max_val = 0.0_f64;
        for r in 0..n {
            let v = vectors[c * n + r];
            if v.abs() > max_abs { max_abs = v.abs(); max_val = v; }
        }
        if max_val < 0.0 {
            for r in 0..n { vectors[c * n + r] = -vectors[c * n + r]; }
        }
    }

    Ok((eigenvalues, vectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpotrf() {
        // A = [[4, 12], [12, 37]] column-major  — positive definite
        let mut a = vec![4.0, 12.0, 12.0, 37.0];
        dpotrf(2, &mut a).unwrap();
        // L should be [[2, 0], [6, 1]] column-major → [2, 6, 0, 1]
        assert!((a[0] - 2.0).abs() < 1e-10);
        assert!((a[1] - 6.0).abs() < 1e-10);
        assert!((a[3] - 1.0).abs() < 1e-10);
    }

    /// The BLOCKED path (n > NB = 32), which `test_dpotrf` never reached.
    /// It once subtracted the earlier blocks' columns twice — the trailing
    /// update had already removed them — and left the in-block columns out
    /// of the diagonal. At every size past one block, L·Lᵀ must give A
    /// back and L must equal the unblocked factor.
    #[test]
    fn dpotrf_blocked_reconstructs_a() {
        for &n in &[33usize, 50, 64, 100] {
            // SPD: A = BᵀB + n·I, column-major
            let b: Vec<f64> = (0..n * n).map(|i| ((i * 7 + 3) % 17) as f64 * 0.1 - 0.8).collect();
            let mut a0 = vec![0.0; n * n];
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for p in 0..n { s += b[i * n + p] * b[j * n + p]; }
                    a0[j * n + i] = s + if i == j { n as f64 } else { 0.0 };
                }
            }
            let mut l = a0.clone();
            dpotrf(n, &mut l).unwrap();
            let mut lu = a0.clone();
            dpotrf_unblocked(n, &mut lu).unwrap();
            let scale = a0.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for j in 0..n {
                for i in 0..n {
                    // upper triangle of L is zeroed
                    if i < j { assert_eq!(l[j * n + i], 0.0, "n={n}: upper ({i},{j}) not zero"); }
                    // (L·Lᵀ)[i][j] = Σ_k L[i][k]·L[j][k]
                    let mut s = 0.0;
                    for k in 0..=i.min(j) { s += l[k * n + i] * l[k * n + j]; }
                    assert!((s - a0[j * n + i]).abs() <= 1e-10 * scale,
                        "n={n}: (L·Lᵀ)[{i}][{j}] = {s}, A = {}", a0[j * n + i]);
                    assert!((l[j * n + i] - lu[j * n + i]).abs() <= 1e-10 * scale,
                        "n={n}: blocked L[{i}][{j}] differs from unblocked");
                }
            }
        }
    }

    #[test]
    fn test_dgetrf() {
        // A = [[4, 3], [6, 3]] column-major = [4, 6, 3, 3]
        let mut a = vec![4.0, 6.0, 3.0, 3.0];
        let _p = dgetrf(2, &mut a).unwrap();
        // Factorization should succeed
        assert!(a[0].abs() > 0.0);
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }

    /// Reconstruct A from (eigenvalues, vectors) and check A ≈ Q D Qᵀ.
    fn reconstruct(n: usize, eig: &[f64], q: &[f64]) -> Vec<f64> {
        // Q is column-major n×n.  R[i,j] = Σ_k Q[i,k] · eig[k] · Q[j,k]
        let mut r = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n { s += q[k * n + i] * eig[k] * q[k * n + j]; }
                r[j * n + i] = s;   // column-major
            }
        }
        r
    }

    #[test]
    fn dsyev_full_diagonal_matrix() {
        // diag(3, 7) — trivial. Eigenvalues = {7, 3}, vectors = ±e_i.
        let a = vec![3.0, 0.0, 0.0, 7.0];
        let (eig, q) = dsyev_full(2, &a).unwrap();
        assert!(approx(eig[0], 7.0, 1e-12));
        assert!(approx(eig[1], 3.0, 1e-12));
        let r = reconstruct(2, &eig, &q);
        for k in 0..4 { assert!(approx(r[k], a[k], 1e-12), "reconstruction mismatch at {}", k); }
    }

    #[test]
    fn dsyev_full_two_by_two_known_closed_form() {
        // A = [[2, 1], [1, 2]] → eigenvalues 3 and 1, vectors (1,1)/√2 and (-1,1)/√2.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (eig, q) = dsyev_full(2, &a).unwrap();
        assert!(approx(eig[0], 3.0, 1e-10));
        assert!(approx(eig[1], 1.0, 1e-10));
        let r = reconstruct(2, &eig, &q);
        for k in 0..4 { assert!(approx(r[k], a[k], 1e-10)); }
    }

    #[test]
    fn dsyev_full_3x3_reconstructs() {
        // Symmetric 3×3 with known but non-trivial eigenstructure.
        // A = [[4,1,2],[1,3,0],[2,0,5]] column-major.
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let (eig, q) = dsyev_full(3, &a).unwrap();
        // Eigenvalues sorted descending.
        assert!(eig[0] > eig[1] && eig[1] > eig[2]);
        // Reconstruction within tolerance — the real correctness test.
        let r = reconstruct(3, &eig, &q);
        for k in 0..9 { assert!(approx(r[k], a[k], 1e-8), "k={} got {} want {}", k, r[k], a[k]); }
    }

    #[test]
    fn dsyev_full_eigenvectors_are_orthonormal() {
        // Q must satisfy Qᵀ Q ≈ I.
        let a = vec![5.0, 2.0, 1.0,  2.0, 6.0, 3.0,  1.0, 3.0, 7.0];
        let (_, q) = dsyev_full(3, &a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 { dot += q[i * 3 + k] * q[j * 3 + k]; }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(dot, expected, 1e-10),
                    "Qᵀ Q[{},{}] = {}, expected {}", i, j, dot, expected);
            }
        }
    }

    /// Check that A ≈ U · diag(σ) · Vᵀ on a small well-conditioned 3×2.
    #[test]
    fn test_dgesvd_full_reconstructs_3x2() {
        // A (column-major 3×2): col0 = [1, 2, 3], col1 = [4, 5, 6]
        let a = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let m = 3; let n = 2;
        let (sigma, u, vt) = dgesvd_full(m, n, &a).expect("svd ok");
        // Sigma values for this matrix are ≈ (9.5080, 0.7729).
        assert!(sigma.len() == 2);
        assert!(sigma[0] >= sigma[1]);
        assert!((sigma[0] - 9.5080).abs() < 1e-3);
        assert!((sigma[1] - 0.7729).abs() < 1e-3);

        // Reconstruction: A_reconstructed[i, j] = Σ_k U[i, k] · σ_k · Vᵀ[k, j]
        //   U is m×n column-major, Vᵀ is n×n column-major (so Vᵀ[k, j] = vt[j*n + k]).
        let mut a_rec = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for k in 0..n { s += u[k * m + i] * sigma[k] * vt[j * n + k]; }
                a_rec[j * m + i] = s;
            }
        }
        for (orig, got) in a.iter().zip(a_rec.iter()) {
            assert!((orig - got).abs() < 1e-9, "reconstruction: orig {} got {}", orig, got);
        }

        // Orthogonality: Uᵀ · U ≈ I_n.
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot = 0.0;
                for i in 0..m { dot += u[c1 * m + i] * u[c2 * m + i]; }
                let want = if c1 == c2 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "U col dot[{},{}]={}, want {}", c1, c2, dot, want);
            }
        }
        // Orthogonality: Vᵀ rows orthonormal ⇔ Vᵀ · V ≈ I where V = (Vᵀ)ᵀ.
        // Equivalently: Σ_k Vᵀ[k, i] · Vᵀ[k, j] = δ_{ij}.
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n { dot += vt[i * n + k] * vt[j * n + k]; }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "V row dot[{},{}]={}, want {}", i, j, dot, want);
            }
        }
    }

    /// Reconstruction + orthogonality on a non-trivial 4×3.
    #[test]
    fn test_dgesvd_full_reconstructs_4x3() {
        // Column-major: col0=[1,2,3,4], col1=[5,6,7,8], col2=[9,10,11,13].
        // Last entry perturbed so the matrix isn't rank-deficient.
        let a = vec![1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 10.0, 11.0, 13.0];
        let m = 4; let n = 3;
        let (sigma, u, vt) = dgesvd_full(m, n, &a).expect("svd ok");
        assert!(sigma[0] >= sigma[1] && sigma[1] >= sigma[2]);
        assert!(sigma[2] > 1e-6, "smallest σ unexpectedly tiny: {}", sigma[2]);

        // A_rec[i,j] = Σ_k U[i,k] · σ_k · Vᵀ[k, j]
        for j in 0..n {
            for i in 0..m {
                let mut s = 0.0;
                for k in 0..n { s += u[k * m + i] * sigma[k] * vt[j * n + k]; }
                let orig = a[j * m + i];
                assert!((orig - s).abs() < 1e-8, "rec[{},{}]: orig={} got={}", i, j, orig, s);
            }
        }
        // Uᵀ U ≈ I_n
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot = 0.0;
                for i in 0..m { dot += u[c1 * m + i] * u[c2 * m + i]; }
                let want = if c1 == c2 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9);
            }
        }
        // Vᵀ Vᵀᵀ ≈ I_n  (rows of Vᵀ orthonormal)
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n { dot += vt[i * n + k] * vt[j * n + k]; }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9);
            }
        }
    }

    /// Diagonal 4×3 with known σ.
    #[test]
    fn test_dgesvd_full_diagonal() {
        // A column-major 4×3 with diag entries 5, 3, 1 in the top 3×3.
        let mut a = vec![0.0; 12];
        a[0 * 4 + 0] = 5.0;
        a[1 * 4 + 1] = 3.0;
        a[2 * 4 + 2] = 1.0;
        let (sigma, _u, _vt) = dgesvd_full(4, 3, &a).expect("svd ok");
        assert!((sigma[0] - 5.0).abs() < 1e-10);
        assert!((sigma[1] - 3.0).abs() < 1e-10);
        assert!((sigma[2] - 1.0).abs() < 1e-10);
    }
}
