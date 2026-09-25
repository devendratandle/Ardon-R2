//! Symmetric eigendecomposition with eigenvectors — `dsyev_full`.
//!
//! The same three stages as LAPACK's `dsyev`, each written so the work
//! walks memory contiguously and spreads across cores:
//!
//! 1. **Householder tridiagonalisation** (`dsytd2`): A = Q₁·T·Q₁ᵀ. Each
//!    step touches only the shrinking trailing block, forms p = β·A₂₂·v by
//!    contiguous column dot products (A₂₂ is symmetric, so a column is a
//!    row), and applies the rank-2 update column by column.
//! 2. **Q₁ from the stored reflectors** (`dorgtr`), applied backwards to
//!    the identity after the reduction — each reflector touches only the
//!    trailing columns, each column independently.
//! 3. **Implicit symmetric QR with a Wilkinson shift** on T. Its Givens
//!    rotations are recorded and applied to the eigenvectors in batches
//!    spanning many sweeps — rows of Q are independent across every sweep
//!    — to blocks of RB rows, column-major within a block, so each rotation
//!    is a SIMD update of two contiguous segments. Per element, the
//!    arithmetic and its order are exactly those of applying each rotation
//!    as it is made.
//!
//! The previous version did the same arithmetic over the whole n x n
//! matrix at every step, striding across rows, on one core: 3.2 s at
//! n = 1000 against R's reference LAPACK at 0.6 s. Correctness is pinned
//! by `tests/decomp_large.rs` (A·v = λ·v, Σλ = trace, agreement with the
//! Jacobi eigenvalues) from 4x4 to 200x200.

use crate::LinalgError;
use rayon::prelude::*;

/// Work (element operations) above which a stage forks.
const PAR_MIN: usize = 1 << 15;

/// Rows per block of the eigenvector matrix during the QR iteration.
/// Within a block Q is column-major, so one rotation updates two
/// contiguous RB-long column segments — SIMD across rows — and a block
/// (RB·n doubles, 128 KB at n = 1000) stays in L2 while a whole batch of
/// rotations passes over it.
const RB: usize = 16;

/// Rotations recorded before they are applied to Q. Rows of Q are
/// independent across ALL sweeps, so rotations are batched across sweeps:
/// one fork per batch instead of one per sweep, and the batch (24 bytes a
/// rotation, ~780 KB) is shared from cache by every block.
const ROT_BATCH: usize = 1 << 15;

/// Apply rotations, in order, to one RB-row block of Q (column-major
/// within the block): columns k and k+1 of every row.
fn rotate_block(block: &mut [f64], rots: &[(usize, f64, f64)]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        // SAFETY: avx2() confirmed AVX2 on this CPU.
        return unsafe { rotate_block_avx2(block, rots) };
    }
    rotate_block_impl(block, rots)
}

/// The same loop compiled 4 doubles wide. The workspace builds for
/// baseline x86-64 (SSE2), so without this the rotations ran 2 wide. No
/// FMA: a·x + b·y keeps its two roundings, so results are bit-identical
/// to the baseline build.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rotate_block_avx2(block: &mut [f64], rots: &[(usize, f64, f64)]) {
    rotate_block_impl(block, rots)
}

#[inline(always)]
fn rotate_block_impl(block: &mut [f64], rots: &[(usize, f64, f64)]) {
    for &(k, cs, sn) in rots {
        let (lo, hi) = block.split_at_mut((k + 1) * RB);
        let a = &mut lo[k * RB..];
        let b = &mut hi[..RB];
        for (x, y) in a.iter_mut().zip(b.iter_mut()) {
            let (xa, yb) = (*x, *y);
            *x = cs * xa + sn * yb;
            *y = -sn * xa + cs * yb;
        }
    }
}

fn flush(qb: &mut [f64], n: usize, rots: &mut Vec<(usize, f64, f64)>) {
    if rots.is_empty() { return; }
    let rs = &rots[..];
    if rs.len() * n >= PAR_MIN { qb.par_chunks_mut(RB * n).for_each(|b| rotate_block(b, rs)); }
    else { qb.chunks_mut(RB * n).for_each(|b| rotate_block(b, rs)); }
    rots.clear();
}

fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 { return (1.0, 0.0); }
    if a == 0.0 { return (0.0, 1.0); }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r)
}

use crate::simd::{axpy, dot, rank2};

/// Symmetric eigendecomposition of the column-major n x n `a`.
/// Returns (eigenvalues descending, eigenvectors column-major n x n with
/// column k the unit eigenvector of eigenvalue k, largest-magnitude entry
/// positive — R's / LAPACK's sign convention).
pub fn dsyev_full(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if n == 0 { return Ok((vec![], vec![])); }
    if n == 1 { return Ok((vec![a[0]], vec![1.0])); }

    // ── Stage 1: tridiagonalise, keeping each reflector ─────────────────
    let mut t = a.to_vec();
    let mut refl: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n.saturating_sub(2));
    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;                       // trailing size, rows k+1..n
        let x = &t[k * n + k + 1..k * n + n];    // column k below the diagonal
        let norm_sq: f64 = x.iter().map(|v| v * v).sum();
        if norm_sq < 1e-300 { refl.push((Vec::new(), 0.0)); continue; }
        let xkk = x[0];
        let alpha = if xkk == 0.0 { -norm_sq.sqrt() } else { -xkk.signum() * norm_sq.sqrt() };
        let mut v = x.to_vec();
        v[0] -= alpha;
        let vtv: f64 = v.iter().map(|x| x * x).sum();
        if vtv < 1e-300 { refl.push((Vec::new(), 0.0)); continue; }
        let beta = 2.0 / vtv;

        // p = β·A₂₂·v; A₂₂ symmetric, so p[i] = β·(column i of A₂₂)·v.
        let base = (k + 1) * n + k + 1;
        let col = |i: usize| &t[base + i * n..base + i * n + m];
        let p: Vec<f64> = if m * m >= PAR_MIN {
            (0..m).into_par_iter().map(|i| beta * dot(col(i), &v)).collect()
        } else {
            (0..m).map(|i| beta * dot(col(i), &v)).collect()
        };
        // w = p − (β/2)(pᵀv)·v
        let half = 0.5 * beta * dot(&p, &v);
        let w: Vec<f64> = p.iter().zip(&v).map(|(pi, vi)| pi - half * vi).collect();
        // A₂₂ −= v·wᵀ + w·vᵀ, one column at a time.
        let upd = |(j, c): (usize, &mut [f64])| {
            let c = &mut c[k + 1..n];
            let (wj, vj) = (w[j], v[j]);
            rank2(c, &v, &w, wj, vj);
        };
        let trail = &mut t[(k + 1) * n..];
        if m * m >= PAR_MIN { trail.par_chunks_mut(n).enumerate().for_each(upd); }
        else { trail.chunks_mut(n).enumerate().for_each(upd); }
        // Column / row k become (…, α, 0, …, 0).
        t[k * n + k + 1] = alpha;
        t[(k + 1) * n + k] = alpha;
        for i in k + 2..n { t[k * n + i] = 0.0; t[i * n + k] = 0.0; }
        refl.push((v, beta));
    }
    let d: Vec<f64> = (0..n).map(|i| t[i * n + i]).collect();
    let e: Vec<f64> = (0..n - 1).map(|i| t[i * n + i + 1]).collect();
    drop(t);

    // ── Stage 2: Q₁ = H₀·H₁···H_{n−3}, built backwards from I ───────────
    // When H_k is applied, Q is still the identity outside rows/columns
    // k+1.., so only those columns change: Q[k+1.., c] −= β·(vᵀ·Q[k+1.., c])·v.
    let mut q = vec![0.0f64; n * n];
    for i in 0..n { q[i * n + i] = 1.0; }
    for k in (0..refl.len()).rev() {
        let (v, beta) = &refl[k];
        if *beta == 0.0 { continue; }
        let m = n - k - 1;
        let app = |c: &mut [f64]| {
            let seg = &mut c[k + 1..n];
            let s = beta * dot(seg, v);
            axpy(seg, -s, v);
        };
        let cols = &mut q[(k + 1) * n..];
        if m * m >= PAR_MIN { cols.par_chunks_mut(n).for_each(app); }
        else { cols.chunks_mut(n).for_each(app); }
    }
    drop(refl);

    // Stages 3-4 — shared with the SVD, which arrives already tridiagonal.
    tridiag_eigen(d, e, Some((q, n)))
        .map(|(vals, vecs)| (vals, vecs.expect("vectors requested")))
}

/// Eigen-decomposition of the symmetric TRIDIAGONAL matrix with diagonal
/// `d` (length n) and off-diagonal `e` (length n−1) — stage 3 of
/// [`dsyev_full`], exposed for the SVD, whose BᵀB is tridiagonal to begin
/// with (densifying it would cost an O(n³) tridiagonalisation of a matrix
/// that already is one).
///
/// `basis`: `None` for eigenvalues only (no rotation is recorded or
/// applied — O(n²) overall); `Some((q, n))` to rotate the column-major
/// n x n `q` (the identity, or Q₁ from a tridiagonalisation) into the
/// eigenvectors. Returns eigenvalues descending and, with a basis, the
/// eigenvectors column-major with the largest-magnitude entry of each
/// positive (R's / LAPACK's sign convention).
pub(crate) fn tridiag_eigen(mut d: Vec<f64>, mut e: Vec<f64>, basis: Option<(Vec<f64>, usize)>)
    -> Result<(Vec<f64>, Option<Vec<f64>>), LinalgError> {
    let n = d.len();
    if n == 0 { return Ok((vec![], basis.map(|_| vec![]))); }
    if n == 1 { return Ok((d, basis.map(|(q, _)| q))); }
    let want = basis.is_some();

    // Blocked copy: block b holds rows b·RB.. of Q, column-major within the
    // block (zero rows pad the last block).
    let nb = n.div_ceil(RB);
    let mut qb = Vec::new();
    if let Some((q, _)) = basis {
        qb = vec![0.0f64; nb * RB * n];
        for c in 0..n {
            for r in 0..n { qb[(r / RB) * RB * n + c * RB + r % RB] = q[c * n + r]; }
        }
    }

    // Implicit symmetric QR with Wilkinson shift on (d, e).
    let max_sweeps = 30 * n;
    let mut end = n - 1;
    let mut sweeps = 0usize;
    let mut rots: Vec<(usize, f64, f64)> = Vec::with_capacity(if want { ROT_BATCH + n } else { 0 });
    while end > 0 {
        if sweeps > max_sweeps { return Err(LinalgError::InvalidShape("dsyev_full: QR failed to converge".into())); }
        sweeps += 1;
        // Deflate converged trailing entries.
        while end > 0 && e[end - 1].abs() <= 1e-14 * (d[end - 1].abs() + d[end].abs()) {
            e[end - 1] = 0.0;
            end -= 1;
        }
        if end == 0 { break; }
        // Start of the active unreduced block.
        let mut start = end;
        while start > 0 && e[start - 1].abs() > 1e-14 * (d[start - 1].abs() + d[start].abs()) {
            start -= 1;
        }
        // Wilkinson shift: eigenvalue of the trailing 2x2 closer to d[end].
        let dd = (d[end - 1] - d[end]) / 2.0;
        let ee = e[end - 1];
        let denom = dd.abs() + (dd * dd + ee * ee).sqrt();
        let sign_dd = if dd >= 0.0 { 1.0 } else { -1.0 };
        let shift = d[end] - sign_dd * ee * ee / denom.max(1e-300);

        // Bulge-chase from start to end, recording each rotation.
        let mut x = d[start] - shift;
        let mut y = e[start];
        for k in start..end {
            let (cs, sn) = givens(x, y);
            if k > start { e[k - 1] = cs * x + sn * y; }
            let (d_k, d_k1, e_k) = (d[k], d[k + 1], e[k]);
            d[k]     = cs * cs * d_k + 2.0 * cs * sn * e_k + sn * sn * d_k1;
            d[k + 1] = sn * sn * d_k - 2.0 * cs * sn * e_k + cs * cs * d_k1;
            e[k]     = cs * sn * (d_k1 - d_k) + (cs * cs - sn * sn) * e_k;
            if k + 1 < end {
                y = sn * e[k + 1];
                e[k + 1] *= cs;
            }
            x = e[k];
            if want { rots.push((k, cs, sn)); }
        }
        if rots.len() >= ROT_BATCH { flush(&mut qb, n, &mut rots); }
    }
    flush(&mut qb, n, &mut rots);

    // Sort descending; eigenvector columns follow.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(std::cmp::Ordering::Equal));
    let eigenvalues: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    if !want { return Ok((eigenvalues, None)); }
    let mut vectors = vec![0.0f64; n * n];
    for (new_col, &old_col) in idx.iter().enumerate() {
        let dst = &mut vectors[new_col * n..(new_col + 1) * n];
        for (r, x) in dst.iter_mut().enumerate() { *x = qb[(r / RB) * RB * n + old_col * RB + r % RB]; }
    }
    // Sign convention (R / LAPACK): largest-magnitude entry positive.
    for c in vectors.chunks_mut(n) {
        let mut max_abs = 0.0f64;
        let mut max_val = 0.0f64;
        for &v in c.iter() { if v.abs() > max_abs { max_abs = v.abs(); max_val = v; } }
        if max_val < 0.0 { for v in c.iter_mut() { *v = -*v; } }
    }
    Ok((eigenvalues, Some(vectors)))
}
