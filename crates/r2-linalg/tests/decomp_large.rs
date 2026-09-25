//! Every decomposition at sizes past its blocking and fast paths, checked
//! through what it PROMISES rather than its storage layout: a solve has a
//! small residual, an inverse gives the identity, an eigenpair satisfies
//! A·v = λ·v, a factorisation gives its input back.
//!
//! Written after the blocked Cholesky was found wrong for every n > 4
//! while its only unit test was 2x2 — the fast path. The same sweep then
//! found three more: blocked LU wrong for n > 32 (no triangular solve for
//! U₁₂), symmetric eigenvectors wrong for n >= 4 (Householders accumulated
//! in reverse order), and the values-only SVD wrong at every size. Sizes:
//! 4 and 5 (just past the small fast paths), 33 (one past an NB = 32
//! block), 50, 100 and 200 (several blocks).

use r2_linalg::*;

const SIZES: [usize; 6] = [4, 5, 33, 50, 100, 200];

/// Deterministic pseudo-random data in [-1, 1), column-major m x n
/// (xorshift64*: no short period, so no accidental rank deficiency).
fn mat(m: usize, n: usize, seed: usize) -> Vec<f64> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (seed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    (0..m * n).map(|_| {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        r as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }).collect()
}

/// y = A·x, A column-major m x n.
fn matvec(m: usize, n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for j in 0..n { for i in 0..m { y[i] += a[j * m + i] * x[j]; } }
    y
}

/// C = A·B, column-major, A m x k, B k x n (the definition, not dgemm).
fn matmul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n { for p in 0..k { let bv = b[j * k + p]; for i in 0..m { c[j * m + i] += a[p * m + i] * bv; } } }
    c
}

fn max_abs(v: &[f64]) -> f64 { v.iter().fold(0.0, |m, x| m.max(x.abs())) }

/// A general matrix made diagonally dominant enough to be comfortably
/// invertible.
fn general(n: usize, seed: usize) -> Vec<f64> {
    let mut a = mat(n, n, seed);
    for i in 0..n { a[i * n + i] += n as f64 * 0.5; }
    a
}

/// Symmetric positive definite: BᵀB + n·I.
fn spd(n: usize, seed: usize) -> Vec<f64> {
    let b = mat(n, n, seed);
    let mut a = vec![0.0; n * n];
    for j in 0..n { for i in 0..n {
        let mut s = 0.0;
        for p in 0..n { s += b[i * n + p] * b[j * n + p]; }
        a[j * n + i] = s + if i == j { n as f64 } else { 0.0 };
    } }
    a
}

#[test]
fn lu_solve_residual() {
    for &n in &SIZES {
        let a0 = general(n, 1);
        let b0 = mat(n, 1, 2);
        let (mut a, mut x) = (a0.clone(), b0.clone());
        dgesv(n, &mut a, &mut x).unwrap();
        let r: Vec<f64> = matvec(n, n, &a0, &x).iter().zip(&b0).map(|(y, b)| y - b).collect();
        assert!(max_abs(&r) < 1e-9, "dgesv n={n}: residual {}", max_abs(&r));
    }
}

#[test]
fn inverse_gives_identity() {
    for &n in &SIZES {
        let a = general(n, 3);
        let inv = dgetri(n, &a).unwrap();
        let p = matmul(n, n, n, &a, &inv);
        let mut err = 0.0f64;
        for j in 0..n { for i in 0..n { err = err.max((p[j * n + i] - if i == j { 1.0 } else { 0.0 }).abs()); } }
        assert!(err < 1e-9, "dgetri n={n}: |A·A⁻¹ - I| = {err}");
    }
}

#[test]
fn cholesky_solve_residual() {
    for &n in &SIZES {
        let a0 = spd(n, 4);
        let b0 = mat(n, 1, 5);
        let (mut a, mut x) = (a0.clone(), b0.clone());
        dposv(n, &mut a, &mut x).unwrap();
        let r: Vec<f64> = matvec(n, n, &a0, &x).iter().zip(&b0).map(|(y, b)| y - b).collect();
        let scale = max_abs(&a0);
        assert!(max_abs(&r) < 1e-10 * scale, "dposv n={n}: residual {}", max_abs(&r));
    }
}

/// QR least squares: at the minimiser the residual is orthogonal to every
/// column of X, i.e. Xᵀ(Xβ − y) = 0 — whatever the storage of Q and R.
/// Cross-checked against the normal-equations (Cholesky) solution.
#[test]
fn qr_least_squares_normal_equations() {
    for &n in &SIZES {
        let m = 3 * n;
        let x = mat(m, n, 6);
        let y = mat(m, 1, 7);
        let beta = dlsq_qr(m, n, &x, &y).unwrap();
        let r: Vec<f64> = matvec(m, n, &x, &beta).iter().zip(&y).map(|(p, t)| p - t).collect();
        let mut g = vec![0.0; n];
        for j in 0..n { for i in 0..m { g[j] += x[j * m + i] * r[i]; } }
        assert!(max_abs(&g) < 1e-9, "dlsq_qr n={n}: |Xᵀr| = {}", max_abs(&g));
        let beta_ne = dlsq_normal(m, n, &x, &y).unwrap();
        let d: Vec<f64> = beta.iter().zip(&beta_ne).map(|(a, b)| a - b).collect();
        assert!(max_abs(&d) < 1e-7, "n={n}: QR and normal-equation β differ by {}", max_abs(&d));
    }
}

/// det of a (pivot-scrambled) triangular matrix is the product of its
/// diagonal; row-swapping it flips the sign.
#[test]
fn determinant_of_triangular() {
    for &n in &SIZES {
        let mut t = vec![0.0; n * n];
        let mut want = 1.0f64;
        for j in 0..n {
            for i in 0..=j { t[j * n + i] = ((i * 31 + j * 17) % 13) as f64 / 13.0 - 0.5; }
            let d = 1.0 + ((j * 7) % 5) as f64 * 0.01;
            t[j * n + j] = d;
            want *= d;
        }
        let got = ddet(n, &t).unwrap();
        assert!((got - want).abs() <= 1e-9 * want.abs(), "ddet n={n}: {got} vs {want}");
        // swap rows 0 and 1: determinant changes sign
        let mut s = t.clone();
        for j in 0..n { s.swap(j * n, j * n + 1); }
        let got = ddet(n, &s).unwrap();
        assert!((got + want).abs() <= 1e-9 * want.abs(), "ddet swapped n={n}: {got} vs {}", -want);
    }
}

#[test]
fn symmetric_eigen_pairs() {
    for &n in &SIZES {
        let b = mat(n, n, 8);
        let mut a = vec![0.0; n * n];
        for j in 0..n { for i in 0..n { a[j * n + i] = 0.5 * (b[j * n + i] + b[i * n + j]); } }
        let (vals, vecs) = dsyev_full(n, &a).unwrap();
        assert_eq!(vals.len(), n);
        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        let sum: f64 = vals.iter().sum();
        assert!((trace - sum).abs() < 1e-9 * n as f64, "dsyev_full n={n}: Σλ {sum} vs trace {trace}");
        assert!(vals.windows(2).all(|w| w[0] >= w[1] - 1e-12), "dsyev_full n={n}: not descending");
        let scale = vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for k in 0..n {
            let v = &vecs[k * n..(k + 1) * n];
            let av = matvec(n, n, &a, v);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let err = av.iter().zip(v).map(|(p, q)| (p - vals[k] * q).abs()).fold(0.0, f64::max);
            assert!((norm - 1.0).abs() < 1e-9, "dsyev_full n={n}: |v{k}| = {norm}");
            assert!(err < 1e-9 * scale, "dsyev_full n={n}: |Av - λv| = {err} for pair {k}");
        }
        // the values-only routine must agree
        let only = dsyev(n, &a).unwrap();
        let d = only.iter().zip(&vals).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(d < 1e-8 * scale, "dsyev vs dsyev_full n={n}: {d}");
    }
}

#[test]
fn svd_reconstructs_a() {
    for &n in &SIZES {
        let m = n + 17;
        let a = mat(m, n, 9);
        let (s, u, vt) = dgesvd_full(m, n, &a).unwrap();
        assert_eq!((s.len(), u.len(), vt.len()), (n, m * n, n * n));
        assert!(s.windows(2).all(|w| w[0] >= w[1] - 1e-12), "dgesvd_full n={n}: not descending");
        // U·diag(σ)·Vᵀ = A
        let mut us = u.clone();
        for k in 0..n { for i in 0..m { us[k * m + i] *= s[k]; } }
        let back = matmul(m, n, n, &us, &vt);
        let err = back.iter().zip(&a).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "dgesvd_full {m}x{n}: |UΣVᵀ - A| = {err}");
        // Σσ² = ‖A‖_F²
        let fro: f64 = a.iter().map(|x| x * x).sum();
        let ss: f64 = s.iter().map(|x| x * x).sum();
        assert!((fro - ss).abs() < 1e-9 * fro, "svd n={n}: Σσ² {ss} vs ‖A‖² {fro}");
        // the values-only routine must agree
        let only = dgesvd(m, n, &a).unwrap();
        let d = only.iter().zip(&s).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(d < 1e-9 * s[0], "dgesvd vs dgesvd_full n={n}: {d}");
    }
}
