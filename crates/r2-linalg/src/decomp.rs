//! Matrix decompositions: LU, Cholesky, QR
//! All column-major storage.

use crate::LinalgError;

/// Width at which the recursive LU stops splitting and factors unblocked.
/// Measured on Intel i5-12500, 6 threads (2026-09-25, `--example
/// lapack_cases`, getrf ms at n = 500 / 1000 / 2000): 16 → 5.0 / 19.8 /
/// 113, 32 → 4.4 / 20.2 / 105, 64 → 4.2 / 17.6 / 96.
const LU_BASE: usize = 64;

/// Panel width of the blocked Cholesky, and columns per task of its
/// trailing update. Same machine and harness, potrf ms at n = 1000 / 2000
/// (NB, CW): (128, 32) 19.3 / 108, (128, 64) 15.9 / 108, (128, 128) 15.7 /
/// 84, (192, 64) 18.1 / 96, (256, 128) 19.9 / 96, (96, 64) 16.5 / 92.
const CHOL_NB: usize = 128;
const CHOL_CW: usize = 128;
/// Rows per task of the Cholesky panel solve (a chunk is RB x NB doubles,
/// 128 KB — L2-resident).
const CHOL_RB: usize = 128;

/// `A[r0..r1, c0..c1] -= L·U` on a column-major n x n `a` — the O(n³)
/// updates of the recursive LU and its triangular solves, on the packed
/// GEMM kernel.
///
/// `l` is the m2 x kb block (m2 = r1 − r0) and `u` the kb x n2 block
/// (n2 = c1 − c0), both column-major and contiguous: the caller copies
/// them out of `a`, which costs O(n·kb) against the product's O(n²·kb).
/// The product lands in a scratch buffer kept per thread (a fresh one is a
/// fresh mapping from the OS — ~2,000 page faults at 8 MB) and is
/// subtracted back one column per task.
fn sub_product(n: usize, a: &mut [f64], r0: usize, r1: usize, c0: usize, c1: usize,
               l: &[f64], u: &[f64], kb: usize) {
    use crate::gemm::{f64 as g, Trans};
    use rayon::prelude::*;
    thread_local! {
        static SBUF: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let (m2, n2) = (r1 - r0, c1 - c0);
    if m2 == 0 || n2 == 0 || kb == 0 { return; }
    let parallel = r2_oracle::should_parallelize(r2_oracle::Op::MatMul, r2_oracle::Shape::nmk(m2, n2, kb));
    // Column-major T = L·U is row-major Tᵀ = Uᵀ·Lᵀ, and u / l read
    // row-major ARE Uᵀ (n2 x kb) and Lᵀ (kb x m2) — see level3::dgemm.
    // Taken out of the cell, not borrowed across the (forking) GEMM.
    let mut t = SBUF.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if t.len() < m2 * n2 { t.resize(m2 * n2, 0.0); }
    g::gemm_assign_into(u, Trans::No, l, Trans::No, n2, kb, m2, &mut t[..m2 * n2], parallel);
    let tr = &t;
    let body = |(jj, col): (usize, &mut [f64])| {
        for (x, d) in col[r0..r1].iter_mut().zip(&tr[jj * m2..(jj + 1) * m2]) { *x -= d; }
    };
    if parallel { a[c0 * n..c1 * n].par_chunks_mut(n).enumerate().for_each(body); }
    else { a[c0 * n..c1 * n].chunks_mut(n).enumerate().for_each(body); }
    SBUF.with(|b| { let mut s = b.borrow_mut(); if s.capacity() < t.capacity() { *s = t; } });
}

/// In-place unit-lower triangular solve `A[r0..r0+h, c1..c2] ←
/// L⁻¹·A[r0..r0+h, c1..c2]`, L = the unit-lower part of A[r0..r0+h,
/// r0..r0+h] (LU's L₁₁, so U₁₂ = L₁₁⁻¹·A₁₂). Recursive, as LAPACK's
/// `dtrsm` is blocked: solve the top half, one GEMM brings the bottom
/// half up to date, solve the bottom half — so almost all of its
/// O(h²·(c2−c1)) work is GEMM. Narrow blocks are solved directly, one
/// column per task (columns are independent).
fn trsm_unit_lower(n: usize, a: &mut [f64], r0: usize, h: usize, c1: usize, c2: usize) {
    if h <= TRSM_BASE {
        use rayon::prelude::*;
        let (left, right) = a.split_at_mut(c1 * n);
        let lcols = &left[r0 * n..];
        let r1 = r0 + h;
        let solve = |col: &mut [f64]| {
            for p in 0..h {
                let u = col[r0 + p];
                if u != 0.0 {
                    let lc = &lcols[p * n..p * n + n];
                    for i in (r0 + p + 1)..r1 { col[i] -= lc[i] * u; }
                }
            }
        };
        let right = &mut right[..(c2 - c1) * n];
        if (c2 - c1) * h * h >= 1 << 16 { right.par_chunks_mut(n).for_each(solve); }
        else { right.chunks_mut(n).for_each(solve); }
        return;
    }
    let h1 = h / 2;
    let h2 = h - h1;
    trsm_unit_lower(n, a, r0, h1, c1, c2);
    // A[r0+h1..r0+h, c1..c2] −= L[r0+h1..r0+h, r0..r0+h1] · X[r0..r0+h1, c1..c2]
    let nc = c2 - c1;
    let mut l = vec![0.0; h2 * h1];
    for p in 0..h1 { l[p * h2..(p + 1) * h2].copy_from_slice(&a[(r0 + p) * n + r0 + h1..(r0 + p) * n + r0 + h]); }
    let mut u = vec![0.0; h1 * nc];
    for jj in 0..nc { u[jj * h1..(jj + 1) * h1].copy_from_slice(&a[(c1 + jj) * n + r0..(c1 + jj) * n + r0 + h1]); }
    sub_product(n, a, r0 + h1, r0 + h, c1, c2, &l, &u, h1);
    trsm_unit_lower(n, a, r0 + h1, h2, c1, c2);
}

/// Rows below which [`trsm_unit_lower`] stops recursing.
const TRSM_BASE: usize = 64;

/// LU factorization with partial pivoting — recursive, GEMM-based.
/// Modifies A in place: A = P·L·U where L has unit diagonal.
/// Returns the permutation: row `i` of P·A is row `ipiv[i]` of A.
///
/// LAPACK `dgetrf2`'s recursion ([`lu_rec`]): factor the left half of the
/// columns, bring the right half up to date with a triangular solve and one
/// GEMM, recurse into the right half. Almost all the O(n³) work — including
/// what a fixed-width panel would do one rank-1 update at a time — lands
/// in the packed GEMM kernel.
pub fn dgetrf(n: usize, a: &mut [f64]) -> Result<Vec<usize>, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    let mut ipiv = (0..n).collect::<Vec<usize>>();
    lu_rec(n, a, &mut ipiv, 0, n)?;
    Ok(ipiv)
}

/// Factor columns `c..c+w` of the column-major n x n `a`, rows `c..n`,
/// assuming every column left of `c` is already factored and every column
/// in `c..c+w` already carries their updates. Row swaps are applied across
/// the whole row, so columns right of `c+w` are permuted too and only need
/// their numerical update from the caller.
fn lu_rec(n: usize, a: &mut [f64], ipiv: &mut [usize], c: usize, w: usize) -> Result<(), LinalgError> {
    if w <= LU_BASE {
        // Unblocked base case on a narrow panel.
        for kk in c..c + w {
            let mut max_val = 0.0f64;
            let mut max_row = kk;
            for i in kk..n {
                let v = a[kk * n + i].abs();
                if v > max_val { max_val = v; max_row = i; }
            }
            if max_val < 1e-15 { return Err(LinalgError::Singular); }
            if max_row != kk {
                ipiv.swap(kk, max_row);
                for j in 0..n { a.swap(j * n + kk, j * n + max_row); }
            }
            let pivot = a[kk * n + kk];
            for i in (kk + 1)..n { a[kk * n + i] /= pivot; }
            for j in (kk + 1)..(c + w) {
                let akj = a[j * n + kk];
                if akj != 0.0 {
                    for i in (kk + 1)..n { a[j * n + i] -= a[kk * n + i] * akj; }
                }
            }
        }
        return Ok(());
    }
    let h = w / 2;
    lu_rec(n, a, ipiv, c, h)?;
    // Right half, columns c+h..c+w: U₁₂ = L₁₁⁻¹·A₁₂ on rows c..c+h
    // (unit-lower forward substitution, columns independent), then
    // A₂₂ −= L₂₁·U₁₂ on rows c+h..n.
    let (r1, c1, c2) = (c + h, c + h, c + w);
    trsm_unit_lower(n, a, c, h, c1, c2);
    if r1 < n {
        let (m2, n2) = (n - r1, c2 - c1);
        let mut l = vec![0.0; m2 * h];
        for p in 0..h { l[p * m2..(p + 1) * m2].copy_from_slice(&a[(c + p) * n + r1..(c + p + 1) * n]); }
        let mut u = vec![0.0; h * n2];
        for jj in 0..n2 { u[jj * h..(jj + 1) * h].copy_from_slice(&a[(c1 + jj) * n + c..(c1 + jj) * n + r1]); }
        sub_product(n, a, r1, n, c1, c2, &l, &u, h);
    }
    lu_rec(n, a, ipiv, c1, c2 - c1)
}

/// Cholesky decomposition: A = L·Lᵀ (for symmetric positive-definite A)
/// Modifies A in place, storing L in lower triangle
///
/// Blocked algorithm, `CHOL_NB` columns at a time: factor the diagonal
/// block on a contiguous copy, solve the panel below it in parallel
/// `CHOL_RB`-row chunks, then the trailing update A₂₂ −= L₂₁·L₂₁ᵀ (lower
/// triangle only) in parallel `CHOL_CW`-column chunks on the packed GEMM
/// kernel. Every step reads memory contiguously; scratch is reused.
pub fn dpotrf(n: usize, a: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }

    // Small matrix fast path (most lm() calls: 2-20 predictors)
    if n <= 4 { return dpotrf_unblocked(n, a); }

    let nb = CHOL_NB;
    let mut j = 0;
    // Scratch reused across panels: fresh multi-MB buffers per panel are
    // fresh pages from the OS, a page fault each.
    let (mut dblk, mut pb, mut lt) = (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new());
    while j < n {
        let jb = (n - j).min(nb);

        // Factor the diagonal block A[j:j+jb, j:j+jb] on a contiguous copy
        // (jb x jb column-major, cache-resident), left-looking: column c
        // subtracts L[c][k]·column k for each earlier k as a contiguous
        // axpy. (Columns 0..j were already removed by the trailing
        // updates.) In place in A, every step strode n elements.
        dblk.clear();
        dblk.resize(jb * jb, 0.0);
        for q in 0..jb { dblk[q * jb..(q + 1) * jb].copy_from_slice(&a[(j + q) * n + j..(j + q) * n + j + jb]); }
        for c in 0..jb {
            let (lo, hi) = dblk.split_at_mut(c * jb);
            let col = &mut hi[..jb];
            for k in 0..c {
                let f = lo[k * jb + c];
                if f != 0.0 {
                    for (d, s) in col[c..].iter_mut().zip(&lo[k * jb + c..k * jb + jb]) { *d -= s * f; }
                }
            }
            let diag = col[c];
            if diag <= 1e-15 { return Err(LinalgError::NotPositiveDefinite); }
            let ljj = diag.sqrt();
            col[c] = ljj;
            for d in col[c + 1..].iter_mut() { *d /= ljj; }
        }
        for q in 0..jb {
            let col = &mut a[(j + q) * n..(j + q + 1) * n];
            for x in col[..j + q].iter_mut() { *x = 0.0; }   // upper triangle
            col[j + q..j + jb].copy_from_slice(&dblk[q * jb + q..(q + 1) * jb]);
        }

        // Panel below the diagonal block: L₂₁ = A₂₁·L₁₁⁻ᵀ. Rows are
        // independent, so the panel is cut into CHOL_RB-row chunks, each
        // stored column-major and contiguous; a chunk is solved column-
        // oriented — column q −= L[q][p]·column p for p < q, then ÷ L[q][q],
        // every step a contiguous axpy — and chunks run in parallel. The
        // same pass writes the chunk's rows row-major into `lt`, the
        // L₂₁ᵀ operand of the trailing GEMM, while it is in cache.
        // (Row-by-row substitution spent its time in 128 short dot calls
        // per row, and element-wise transposes strode 1 KB per element.)
        if j + jb < n {
            use rayon::prelude::*;
            let trail = j + jb;
            let m2 = n - trail;
            let nch = m2.div_ceil(CHOL_RB);
            pb.clear();
            pb.resize(nch * CHOL_RB * jb, 0.0);
            for q in 0..jb {
                let col = &a[(j + q) * n + trail..(j + q + 1) * n];
                for (k, seg) in col.chunks(CHOL_RB).enumerate() {
                    let o = k * CHOL_RB * jb + q * CHOL_RB;
                    pb[o..o + seg.len()].copy_from_slice(seg);
                }
            }
            lt.clear();
            lt.resize(m2 * jb, 0.0);
            let d = &dblk;
            pb.par_chunks_mut(CHOL_RB * jb).zip(lt.par_chunks_mut(CHOL_RB * jb)).enumerate()
                .for_each(|(k, (ch, rows))| {
                    let len = CHOL_RB.min(m2 - k * CHOL_RB);
                    for q in 0..jb {
                        let (lo, hi) = ch.split_at_mut(q * CHOL_RB);
                        let seg = &mut hi[..len];
                        for p in 0..q {
                            let f = d[p * jb + q];   // L[q][p]
                            if f != 0.0 {
                                for (x, s) in seg.iter_mut().zip(&lo[p * CHOL_RB..p * CHOL_RB + len]) { *x -= s * f; }
                            }
                        }
                        let ljj = d[q * jb + q];
                        for x in seg.iter_mut() { *x /= ljj; }
                    }
                    for r in 0..len { for q in 0..jb { rows[r * jb + q] = ch[q * CHOL_RB + r]; } }
                });
            for q in 0..jb {
                let col = &mut a[(j + q) * n + trail..(j + q + 1) * n];
                for (k, seg) in col.chunks_mut(CHOL_RB).enumerate() {
                    let o = k * CHOL_RB * jb + q * CHOL_RB;
                    seg.copy_from_slice(&pb[o..o + seg.len()]);
                }
            }

            // Trailing update A₂₂ −= L₂₁·L₂₁ᵀ, lower triangle only, in
            // column chunks of CHOL_CW: a chunk's columns c..c+cw need rows
            // c..n alone, one small serial GEMM computes exactly those, and
            // the result is subtracted straight into A while it is in
            // cache. Chunks in parallel. (A whole-matrix product halved
            // nothing — it computed the full square — and moved the
            // trailing matrix through RAM several times per panel: 103 ms
            // of 141 at n = 2000 on this single-channel machine.)
            //
            // Row-major view: rows c..c+cw of `lt` are L₂₁[c..c+cw, :]
            // (cw x jb); rows c.. of `lt` read transposed are L₂₁[c.., :]ᵀ.
            // Their product is Tᵀ (cw x rows), T = L₂₁[c.., :]·L₂₁[c..c+cw, :]ᵀ.
            use crate::gemm::{f64 as g, Trans};
            // Per-thread product scratch, reused across chunks and panels
            // (a fresh ~2 MB Vec per chunk is ~500 page faults). TAKEN out
            // of the cell for the call, not borrowed across it: the GEMM's
            // packing can fork, a waiting thread may steal another chunk,
            // and that chunk then finds the cell empty and allocates.
            thread_local! {
                static TBUF: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            let ltr = &lt;
            a[trail * n..].par_chunks_mut(CHOL_CW * n).enumerate().for_each(|(ci, block)| {
                let c = ci * CHOL_CW;
                let cw = block.len() / n;
                let rows = m2 - c;
                let mut t = TBUF.with(|tb| std::mem::take(&mut *tb.borrow_mut()));
                if t.len() < cw * rows { t.resize(cw * rows, 0.0); }
                g::gemm_assign_into(&ltr[c * jb..(c + cw) * jb], Trans::No, &ltr[c * jb..], Trans::Yes,
                                    cw, jb, rows, &mut t[..cw * rows], false);
                for jj in 0..cw {
                    let col = &mut block[jj * n + trail + c + jj..(jj + 1) * n];
                    for (x, d) in col.iter_mut().zip(&t[jj * rows + jj..(jj + 1) * rows]) { *x -= d; }
                }
                TBUF.with(|tb| { let mut b = tb.borrow_mut(); if b.capacity() < t.capacity() { *b = t; } });
            });
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

/// QR decomposition via Householder reflections — blocked (LAPACK `dgeqrf`).
/// A is m×n, m >= n, column-major.
/// On exit: upper triangle of A contains R, below-diagonal contains the
/// Householder vectors (v[k] = 1 implicit), `tau[k]` their coefficients —
/// H_k = I − τ_k·v_k·v_kᵀ and Q = H_0·H_1···H_{n−1}.
///
/// Panels of `QR_NB` columns are factored one reflector at a time
/// ([`qr_panel`]); the panel's reflectors are then combined into one block
/// reflector H = I − V·T·Vᵀ (`dlarft`) and applied to everything right of
/// the panel as three GEMMs (`dlarfb`): W = Vᵀ·C, W = Tᵀ·W, C −= V·W.
/// Unblocked, each reflector swept the whole trailing matrix on one core
/// (225 ms at n = 1000 against MKL's 9 ms); blocked, the O(m·n²) work is
/// GEMM. Same factor, same storage — `dlsq_qr` and `lm` read it unchanged.
pub fn dgeqrf(m: usize, n: usize, a: &mut [f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("QR: A shape".into())); }
    if m < n { return Err(LinalgError::InvalidShape("QR: need m >= n".into())); }

    let mut tau = vec![0.0; n];
    let mut k = 0;
    while k < n {
        let kb = QR_NB.min(n - k);
        qr_panel(m, a, &mut tau, k, k + kb);
        if k + kb < n { apply_block_reflector(m, a, &tau, k, kb); }
        k += kb;
    }
    Ok(tau)
}

/// Panel width of the blocked QR. Measured on an Intel i5-12500 (single-
/// channel DDR4), geqrf ms at n = 500 / 1000 / 2000 with 32-column apply
/// chunks: 64 → 10.2 / 42.7 / 246, 96 → 12.7 / 52 / 301, 128 → 15.4 / 65 /
/// 322. Chunks of 16 / 64 columns at width 64: 10.7 / 45.7 / 240 and
/// 10.2 / 38.4 / 320.
const QR_NB: usize = 64;

/// Unblocked Householder QR of columns `k0..k1` (rows k..m for column k),
/// updating only columns inside the panel.
fn qr_panel(m: usize, a: &mut [f64], tau: &mut [f64], k0: usize, k1: usize) {
    for k in k0..k1 {
        // Compute Householder reflection for column k, rows k..m
        let mut norm_sq = 0.0;
        for i in k..m { let v = a[k * m + i]; norm_sq += v * v; }
        let norm = norm_sq.sqrt();
        if norm < 1e-15 { tau[k] = 0.0; continue; }

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

        // Apply to the remaining PANEL columns: A[:,j] -= tau · v · (v'·A[:,j]).
        // Columns are independent: in parallel once the panel is tall
        // enough to pay for the fork (single-threaded, the panels were a
        // quarter of the factorisation: 15 of ~64 ms at n = 1000).
        let (lo, hi) = a.split_at_mut((k + 1) * m);
        let v = &lo[k * m + k + 1..k * m + m];
        let tk = tau[k];
        let app = |col: &mut [f64]| {
            let mut vaj = col[k]; // contribution from v[k] = 1
            for (x, vi) in col[k + 1..m].iter().zip(v) { vaj += vi * x; }
            vaj *= tk;
            col[k] -= vaj;
            for (x, vi) in col[k + 1..m].iter_mut().zip(v) { *x -= vaj * vi; }
        };
        let cols = &mut hi[..(k1 - k - 1) * m];
        if (m - k) * (k1 - k - 1) >= 1 << 14 {
            use rayon::prelude::*;
            cols.par_chunks_mut(m).for_each(app);
        } else {
            cols.chunks_mut(m).for_each(app);
        }
    }
}

/// Apply Hᵀ = (H_k0···H_{k0+kb−1})ᵀ, the panel's reflectors, to columns
/// k0+kb..n of `a` (rows k0..m) as one block reflector: with
/// H = I − V·T·Vᵀ (V the unit-lower mk x kb reflectors, T upper
/// triangular, `dlarft` forward/columnwise), Hᵀ·C = C − V·(Tᵀ·(Vᵀ·C)).
fn apply_block_reflector(m: usize, a: &mut [f64], tau: &[f64], k0: usize, kb: usize) {
    use rayon::prelude::*;
    let mk = m - k0;
    let c0 = k0 + kb;
    // V (mk x kb, column-major): unit diagonal, zeros above.
    let mut v = vec![0.0; mk * kb];
    for p in 0..kb {
        let col = &mut v[p * mk..(p + 1) * mk];
        col[p] = 1.0;
        col[p + 1..].copy_from_slice(&a[(k0 + p) * m + k0 + p + 1..(k0 + p + 1) * m]);
    }
    // T (kb x kb upper, column-major): T[p][p] = τ_p,
    // T[0..p, p] = −τ_p · T[0..p, 0..p] · (V[:, 0..p]ᵀ · v_p).
    let mut t = vec![0.0; kb * kb];
    for p in 0..kb {
        let tp = tau[k0 + p];
        t[p * kb + p] = tp;
        if p == 0 || tp == 0.0 { continue; }
        let vp = &v[p * mk..(p + 1) * mk];
        let z: Vec<f64> = (0..p).map(|q| {
            let vq = &v[q * mk..(q + 1) * mk];
            vq[p..].iter().zip(&vp[p..]).map(|(x, y)| x * y).sum::<f64>()
        }).collect();
        for r in 0..p {
            let mut s = 0.0;
            for q in r..p { s += t[q * kb + r] * z[q]; }
            t[p * kb + r] = -tp * s;
        }
    }
    // Applied in column chunks that stay in cache: copy a chunk out,
    // W = Vᵀ·C, W ← Tᵀ·W, C −= V·W on small serial GEMMs, copy back —
    // chunks in parallel. One pass over the trailing matrix per panel
    // instead of the ~10 a whole-matrix Vᵀ·C / V·W with full-size
    // temporaries costs; on a memory-bound machine that is the difference
    // (n = 2000: 521 ms of block-reflector work with whole-matrix GEMMs).
    //
    // Row-major view (see level3::dgemm): a column-major chunk C (mk x cw)
    // is row-major Cᵀ; V's buffer is row-major Vᵀ (kb x mk); T's is Tᵀ.
    //   Wᵀ  (cw x kb) = Cᵀ·V      = gemm(C, No,  V, Yes)
    //   W2ᵀ (cw x kb) = Wᵀ·T      = gemm(Wᵀ, No, T, Yes)
    //   Cᵀ −= W2ᵀ·Vᵀ              = gemm_into(−W2ᵀ, No, V, No)
    use crate::gemm::{f64 as g, Trans};
    // Chunk width: 64 columns while a chunk (mk x 64 doubles) stays within
    // 512 KB of L2, else 32. Measured (i5-12500, geqrf ms at n = 500 /
    // 1000 / 2000): 32 → 10.5 / 37-40 / 206-211, 64 → 10.0 / 33 / 256-259,
    // 128 → 10.7 / 40-43 / 286-290.
    let cw_cols = if mk * 64 * 8 <= 1 << 19 { 64 } else { 32 };
    let (vr, tr) = (&v, &t);
    a[c0 * m..].par_chunks_mut(cw_cols * m).for_each(|block| {
        let cw = block.len() / m;
        let mut c = vec![0.0; mk * cw];
        for j in 0..cw { c[j * mk..(j + 1) * mk].copy_from_slice(&block[j * m + k0..(j + 1) * m]); }
        let w = g::gemm(&c, Trans::No, vr, Trans::Yes, cw, mk, kb, false);
        let mut w2 = g::gemm(&w, Trans::No, tr, Trans::Yes, cw, kb, kb, false);
        for x in w2.iter_mut() { *x = -*x; }
        g::gemm_into(&w2, Trans::No, vr, Trans::No, cw, kb, mk, &mut c, false);
        for j in 0..cw { block[j * m + k0..(j + 1) * m].copy_from_slice(&c[j * mk..(j + 1) * mk]); }
    });
}

/// Singular Value Decomposition — singular values only.
///
/// A is m×n (m >= n), column-major. Returns the n singular values in
/// descending order — exactly [`dgesvd_full`]'s σ.
///
/// This had its own Golub-Kahan phase 2, and it was wrong at every size:
/// each "Givens" step rotated one side of the bidiagonal and dropped the
/// fill-in, which is not an orthogonal transform, so it converged to
/// numbers that were not the singular values (11x4: 2.217 against the
/// true 2.408; Σσ² ≠ ‖A‖²_F). Until a proper implicit-shift bidiagonal QR
/// (LAPACK `dbdsqr`) is written, the values come from the full routine,
/// which `crates/r2-linalg/tests/decomp_large.rs` checks against A itself.
pub fn dgesvd(m: usize, n: usize, a: &[f64]) -> Result<Vec<f64>, LinalgError> {
    svd_impl(m, n, a, false).map(|(sigma, _, _)| sigma)
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
    svd_impl(m, n, a, true)
}

/// The SVD, with (`vectors`) or without U and Vᵀ. Without them no
/// reflector is stored or applied, and the singular values come straight
/// from the tridiagonal BᵀB: O(m·n²) for the bidiagonalisation, O(n²)
/// after it. Without vectors, U and Vᵀ come back empty.
fn svd_impl(m: usize, n: usize, a: &[f64], vectors: bool)
    -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    use rayon::prelude::*;
    if a.len() != m * n { return Err(LinalgError::InvalidShape("SVD: A shape".into())); }
    if m < n { return Err(LinalgError::InvalidShape("SVD: need m >= n".into())); }
    if n == 0 { return Ok((vec![], vec![], vec![])); }
    // Work above which a column / row sweep forks.
    const PAR: usize = 1 << 15;

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
                // Other columns j ∈ (k+1)..n, each independent.
                let hvk = &hv[k..m];
                let app = |col: &mut [f64]| {
                    let c = &mut col[k..m];
                    let scale = tau * crate::simd::dot(c, hvk);
                    crate::simd::axpy(c, -scale, hvk);
                };
                let cols = &mut work[(k + 1) * m..];
                if (n - k - 1) * (m - k) >= PAR { cols.par_chunks_mut(m).for_each(app); }
                else { cols.chunks_mut(m).for_each(app); }
                // Store the Householder vector (full m-length, zeros above k)
                // and its tau for later left-to-right application onto U₁.
                left_vs.push(if vectors { hv.clone() } else { Vec::new() });
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
                    // Written row by row this strides n·m apart per step; done
                    // column-contiguous in two passes instead: dots[i] =
                    // Σ_j vr[j]·work[i, j] (an axpy per column, rows split
                    // across threads), then column j −= τ·vr[j]·dots.
                    let r0 = k + 1;
                    let mut dots = vec![0.0_f64; m - r0];
                    {
                        let w = &work;
                        let vrr = &vr;
                        let fill = |(ci, ch): (usize, &mut [f64])| {
                            let i0 = r0 + ci * 256;
                            for j in (k + 1)..n {
                                let vj = vrr[j];
                                if vj == 0.0 { continue; }
                                let src = &w[j * m + i0..j * m + i0 + ch.len()];
                                crate::simd::axpy(ch, vj, src);
                            }
                        };
                        if (m - r0) * (n - k - 1) >= PAR { dots.par_chunks_mut(256).enumerate().for_each(fill); }
                        else { dots.chunks_mut(256).enumerate().for_each(fill); }
                    }
                    let vrr = &vr;
                    let upd = |(jj, col): (usize, &mut [f64])| {
                        let s = tau * vrr[k + 1 + jj];
                        if s == 0.0 { return; }
                        crate::simd::axpy(&mut col[r0..m], -s, &dots);
                    };
                    let cols = &mut work[(k + 1) * m..n * m];
                    if (m - r0) * (n - k - 1) >= PAR { cols.par_chunks_mut(m).enumerate().for_each(upd); }
                    else { cols.chunks_mut(m).enumerate().for_each(upd); }
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
    // Columns j < k are still e_j there — zero on rows k.. — so only
    // columns k.. change, each independently.
    let mut u1 = Vec::new();
    if vectors {
        u1 = vec![0.0_f64; m * n];
        for i in 0..n { u1[i * m + i] = 1.0; }
        for k in (0..n).rev() {
            let tau = left_taus[k];
            if tau == 0.0 { continue; }
            let vk = &left_vs[k][k..m];
            let app = |col: &mut [f64]| {
                let c = &mut col[k..m];
                let dot = crate::simd::dot(c, vk);
                let scale = tau * dot;
                crate::simd::axpy(c, -scale, vk);
            };
            let cols = &mut u1[k * m..];
            if (n - k) * (m - k) >= PAR { cols.par_chunks_mut(m).for_each(app); }
            else { cols.chunks_mut(m).for_each(app); }
        }
    }

    // ── Build V₁ (n×n): apply stored right Householders in reverse to I_n.
    //
    // V₁ = G_0 · G_1 · ... · G_{last}, built by left-multiplying I_n in
    // reverse: for k = last down to 0, X ← G_k · X. Each G_k acts on rows
    // (k+1)..n only (vr_k zero up to and including k).
    let mut v1 = Vec::new();
    if vectors {
        v1 = vec![0.0_f64; n * n];
        for i in 0..n { v1[i * n + i] = 1.0; }
        for k in (0..right_vs.len()).rev() {
            let tau = right_taus[k];
            if tau == 0.0 { continue; }
            let vk = &right_vs[k][k + 1..n];
            let app = |col: &mut [f64]| {
                let c = &mut col[k + 1..n];
                let dot = crate::simd::dot(c, vk);
                let scale = tau * dot;
                crate::simd::axpy(c, -scale, vk);
            };
            let cols = &mut v1[(k + 1) * n..];
            if (n - k - 1) * (n - k - 1) >= PAR { cols.par_chunks_mut(n).for_each(app); }
            else { cols.chunks_mut(n).for_each(app); }
        }
    }

    // ── Phase 2: diagonalize B = U₂ · diag(σ) · V₂ᵀ via Bᵀ·B ────────
    //
    // T := Bᵀ·B is n×n symmetric tridiagonal with
    //   T[i][i]   = d[i]² + e[i-1]²   (e[-1] := 0)
    //   T[i][i+1] = d[i] · e[i]
    // Already tridiagonal: straight to the QR iteration (densifying it
    // would pay an O(n³) tridiagonalisation of a tridiagonal matrix).
    let tdiag: Vec<f64> = (0..n).map(|i| {
        let prev_e = if i == 0 { 0.0 } else { e_super[i - 1] };
        d_diag[i] * d_diag[i] + prev_e * prev_e
    }).collect();
    let toff: Vec<f64> = (0..n - 1).map(|i| d_diag[i] * e_super[i]).collect();
    let basis = if vectors {
        let mut id = vec![0.0_f64; n * n];
        for i in 0..n { id[i * n + i] = 1.0; }
        Some((id, n))
    } else { None };
    let (eig_vals, v2) = crate::symeig::tridiag_eigen(tdiag, toff, basis)?;
    // σ_k = sqrt(max(eig_k, 0)) — tiny negatives can arise from rounding.
    let mut sigma = vec![0.0_f64; n];
    for k in 0..n {
        let lam = eig_vals[k];
        sigma[k] = if lam > 0.0 { lam.sqrt() } else { 0.0 };
    }
    if !vectors { return Ok((sigma, vec![], vec![])); }
    let v2 = v2.expect("vectors requested");

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
    // Both on the packed GEMM kernel (these were strided triple loops).
    let mut u = vec![0.0_f64; m * n];
    crate::dgemm(m, n, n, 1.0, &u1, &u2, 0.0, &mut u)?;
    let mut v_mat = vec![0.0_f64; n * n];
    crate::dgemm(n, n, n, 1.0, &v1, &v2, 0.0, &mut v_mat)?;
    // Transpose V → Vᵀ (column-major n×n).
    let mut vt = vec![0.0_f64; n * n];
    for i in 0..n { for j in 0..n { vt[i * n + j] = v_mat[j * n + i]; } }

    Ok((sigma, u, vt))
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

// dsyev_full lives in symeig.rs.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symeig::dsyev_full;

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

    /// The blocked path (every n > 4), which `test_dpotrf` (2x2) never
    /// reached. It once left the in-block columns out of the diagonal —
    /// wrong from 5x5, one block — and, past one NB = 32 block, subtracted
    /// the earlier blocks' columns twice (the trailing update had already
    /// removed them). At one block, at its edge and at several: L·Lᵀ must
    /// give A back and L must equal the unblocked factor.
    #[test]
    fn dpotrf_blocked_reconstructs_a() {
        for &n in &[5usize, 6, 16, 32, 33, 50, 64, 100, 200] {
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
