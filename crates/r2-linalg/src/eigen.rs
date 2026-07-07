//! Non-symmetric real eigenvalue problem (`dgeev`, values only).
//!
//! R's `eigen()` on a **non-symmetric** matrix needs the general real
//! eigenvalue algorithm — the symmetric solver (`dsyev`) gives wrong answers
//! there. This module provides the classic three-stage path for a real square
//! matrix `A`:
//!
//!   1. `elmhes` — reduce `A` to upper Hessenberg form by stabilized Gaussian
//!      elimination (elementary similarity transforms).
//!   2. `hqr`    — the Francis double-shift QR iteration on the Hessenberg
//!      matrix, deflating 1×1 (real eigenvalue) and 2×2 (complex-conjugate
//!      pair) blocks until the real Schur form is reached.
//!
//! Eigenvalues are returned as parallel `(re, im)` vectors. A real matrix can
//! have genuinely complex eigenvalues (they come in conjugate pairs); R2 has no
//! complex type yet, so the caller decides how to surface a non-zero `im`.
//!
//! The algorithm is the standard EISPACK `elmhes`/`hqr` (as popularised by
//! *Numerical Recipes*), transcribed here in 1-based indexing to match the
//! reference and keep the index arithmetic trustworthy. Storage at the API
//! boundary is column-major (Fortran / R convention) like the rest of r2-linalg.

use crate::LinalgError;

/// 1-based dense matrix view over a `(n+1)×(n+1)` backing buffer (row 0 / col 0
/// unused). Keeps the transcription of the reference algorithm literal.
struct M1 {
    n: usize,
    data: Vec<f64>, // (n+1)*(n+1)
}
impl M1 {
    #[inline] fn get(&self, i: usize, j: usize) -> f64 { self.data[i * (self.n + 1) + j] }
    #[inline] fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * (self.n + 1) + j] = v; }
    #[inline] fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * (self.n + 1) + j] += v; }
    #[inline] fn sub(&mut self, i: usize, j: usize, v: f64) { self.data[i * (self.n + 1) + j] -= v; }
}

#[inline] fn sign(a: f64, b: f64) -> f64 { if b >= 0.0 { a.abs() } else { -a.abs() } }

/// Reduce a real matrix to upper Hessenberg form (EISPACK `elmhes`). Only the
/// Hessenberg part is needed for eigenvalues, so the accumulated multipliers
/// left below the subdiagonal are zeroed afterwards.
fn elmhes(a: &mut M1) {
    let n = a.n;
    if n < 3 { return; }
    for m in 2..n {
        // Find the pivot of maximum magnitude in column m-1, rows m..=n.
        let mut x = 0.0f64;
        let mut i_piv = m;
        for i in m..=n {
            if a.get(i, m - 1).abs() > x.abs() { x = a.get(i, m - 1); i_piv = i; }
        }
        if i_piv != m {
            // Interchange rows and columns.
            for j in (m - 1)..=n { let t = a.get(i_piv, j); a.set(i_piv, j, a.get(m, j)); a.set(m, j, t); }
            for i in 1..=n { let t = a.get(i, i_piv); a.set(i, i_piv, a.get(i, m)); a.set(i, m, t); }
        }
        if x != 0.0 {
            for i in (m + 1)..=n {
                let mut y = a.get(i, m - 1);
                if y != 0.0 {
                    y /= x;
                    a.set(i, m - 1, y);
                    for j in m..=n { a.sub(i, j, y * a.get(m, j)); }
                    for j in 1..=n { a.add(j, m, y * a.get(j, i)); }
                }
            }
        }
    }
    // Clear the strictly-below-subdiagonal entries (multipliers) → true Hessenberg.
    for i in 3..=n {
        for j in 1..=(i - 2) { a.set(i, j, 0.0); }
    }
}

/// Francis double-shift QR on an upper Hessenberg matrix (EISPACK `hqr`).
/// Fills `wr`/`wi` (1-based, length n+1) with the eigenvalues' real / imaginary
/// parts. Returns `Singular` if the iteration fails to converge.
fn hqr(a: &mut M1, wr: &mut [f64], wi: &mut [f64]) -> Result<(), LinalgError> {
    let n = a.n;
    let mut anorm = 0.0f64;
    for i in 1..=n {
        for j in i.saturating_sub(1).max(1)..=n { anorm += a.get(i, j).abs(); }
    }
    let mut nn = n;
    let mut t = 0.0f64;
    while nn >= 1 {
        let mut its = 0;
        loop {
            // Look for a single small sub-diagonal element to split the matrix.
            let mut l = nn;
            while l >= 2 {
                let mut s = a.get(l - 1, l - 1).abs() + a.get(l, l).abs();
                if s == 0.0 { s = anorm; }
                if a.get(l, l - 1).abs() + s == s { a.set(l, l - 1, 0.0); break; }
                l -= 1;
            }
            let mut x = a.get(nn, nn);
            if l == nn {
                // One real root found → deflate and re-enter the outer loop.
                wr[nn] = x + t; wi[nn] = 0.0;
                nn -= 1;
                break;
            } else {
                let mut y = a.get(nn - 1, nn - 1);
                let mut w = a.get(nn, nn - 1) * a.get(nn - 1, nn);
                if l == nn - 1 {
                    // Two roots found.
                    let p = 0.5 * (y - x);
                    let q = p * p + w;
                    let mut z = q.abs().sqrt();
                    x += t;
                    if q >= 0.0 {
                        // Real pair.
                        z = p + sign(z, p);
                        wr[nn - 1] = x + z; wr[nn] = wr[nn - 1];
                        if z != 0.0 { wr[nn] = x - w / z; }
                        wi[nn - 1] = 0.0; wi[nn] = 0.0;
                    } else {
                        // Complex conjugate pair.
                        wr[nn - 1] = x + p; wr[nn] = x + p;
                        wi[nn - 1] = -z; wi[nn] = z;
                    }
                    nn -= 2;
                    break;
                } else {
                    // No roots found; continue the QR sweep.
                    if its == 30 { return Err(LinalgError::Singular); }
                    if its == 10 || its == 20 {
                        // Exceptional shift.
                        t += x;
                        for i in 1..=nn { a.sub(i, i, x); }
                        let s = a.get(nn, nn - 1).abs() + a.get(nn - 1, nn - 2).abs();
                        y = 0.75 * s; x = y;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    // Find two consecutive small sub-diagonal elements.
                    let mut m = nn - 2;
                    let (mut p, mut q, mut r);
                    loop {
                        let z = a.get(m, m);
                        let rr = x - z;
                        let ss = y - z;
                        p = (rr * ss - w) / a.get(m + 1, m) + a.get(m, m + 1);
                        q = a.get(m + 1, m + 1) - z - rr - ss;
                        r = a.get(m + 2, m + 1);
                        let s = p.abs() + q.abs() + r.abs();
                        p /= s; q /= s; r /= s;
                        if m == l { break; }
                        let u = a.get(m, m - 1).abs() * (q.abs() + r.abs());
                        let v = p.abs() * (a.get(m - 1, m - 1).abs() + z.abs() + a.get(m + 1, m + 1).abs());
                        if u + v == v { break; }
                        m -= 1;
                    }
                    for i in (m + 2)..=nn {
                        a.set(i, i - 2, 0.0);
                        if i != m + 2 { a.set(i, i - 3, 0.0); }
                    }
                    // Double QR step on rows l..nn, columns m..nn.
                    let mut k = m;
                    while k <= nn - 1 {
                        if k != m {
                            p = a.get(k, k - 1);
                            q = a.get(k + 1, k - 1);
                            r = 0.0;
                            if k != nn - 1 { r = a.get(k + 2, k - 1); }
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 { p /= x; q /= x; r /= x; }
                        }
                        let s = sign((p * p + q * q + r * r).sqrt(), p);
                        if s != 0.0 {
                            if k == m {
                                if l != m { a.set(k, k - 1, -a.get(k, k - 1)); }
                            } else {
                                a.set(k, k - 1, -s * x);
                            }
                            p += s;
                            x = p / s; y = q / s;
                            let z = r / s;
                            let qq = q / p;
                            let rr = r / p;
                            // Row modification.
                            for j in k..=nn {
                                let mut pp = a.get(k, j) + qq * a.get(k + 1, j);
                                if k != nn - 1 {
                                    pp += rr * a.get(k + 2, j);
                                    a.sub(k + 2, j, pp * z);
                                }
                                a.sub(k + 1, j, pp * y);
                                a.sub(k, j, pp * x);
                            }
                            // Column modification.
                            let mmin = if nn < k + 3 { nn } else { k + 3 };
                            for i in l..=mmin {
                                let mut pp = x * a.get(i, k) + y * a.get(i, k + 1);
                                if k != nn - 1 {
                                    pp += z * a.get(i, k + 2);
                                    a.sub(i, k + 2, pp * rr);
                                }
                                a.sub(i, k + 1, pp * qq);
                                a.sub(i, k, pp);
                            }
                        }
                        k += 1;
                    }
                    // A QR sweep did not deflate — iterate again (its accumulates).
                }
            }
        }
    }
    Ok(())
}

/// Eigenvalues of a real, possibly non-symmetric `n×n` matrix `a` (column-major).
/// Returns `(re, im)` parallel vectors of length `n`. Complex eigenvalues appear
/// as conjugate pairs (`im` non-zero). Order follows the QR deflation (not sorted).
pub fn dgeev_values(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if n == 0 { return Ok((Vec::new(), Vec::new())); }
    if n == 1 { return Ok((vec![a[0]], vec![0.0])); }

    // Column-major input → 1-based row-major working matrix.
    let mut wm = M1 { n, data: vec![0.0; (n + 1) * (n + 1)] };
    for j in 0..n {
        for i in 0..n {
            wm.set(i + 1, j + 1, a[j * n + i]);
        }
    }

    elmhes(&mut wm);
    let mut wr = vec![0.0f64; n + 1];
    let mut wi = vec![0.0f64; n + 1];
    hqr(&mut wm, &mut wr, &mut wi)?;

    Ok((wr[1..=n].to_vec(), wi[1..=n].to_vec()))
}

/// Eigenvector for a **real** eigenvalue `lambda` of the real matrix `a`
/// (column-major, `n×n`), by inverse iteration. Returns a unit-norm vector, or
/// `None` if the iteration fails (e.g. defective/degenerate case). The shift is
/// perturbed slightly off `lambda` so `A - shift·I` stays non-singular for the
/// LU solve while still amplifying the eigen-direction.
pub fn dgeev_real_vector(n: usize, a: &[f64], lambda: f64) -> Option<Vec<f64>> {
    if n == 0 || a.len() != n * n { return None; }
    let shift = lambda + 1e-8 * (lambda.abs() + 1.0);
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..4 {
        // Fresh M = A - shift·I each iteration (dgesv overwrites its inputs).
        let mut m = a.to_vec();
        for i in 0..n { m[i * n + i] -= shift; }
        let mut rhs = v.clone();
        if crate::solve::dgesv(n, &mut m, &mut rhs).is_err() { return None; }
        let norm = rhs.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 { return None; }
        for x in &mut rhs { *x /= norm; }
        v = rhs;
    }
    // Sign convention: make the largest-magnitude component positive (stable,
    // matches the sign most references print; eigenvectors are ± anyway).
    let piv = (0..n).max_by(|&i, &j| v[i].abs().partial_cmp(&v[j].abs()).unwrap())?;
    if v[piv] < 0.0 { for x in &mut v { *x = -*x; } }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_by_re(mut re: Vec<f64>, mut im: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        // Sort eigenvalue pairs by (re, im) so comparisons are order-independent.
        let mut idx: Vec<usize> = (0..re.len()).collect();
        idx.sort_by(|&i, &j| re[i].partial_cmp(&re[j]).unwrap().then(im[i].partial_cmp(&im[j]).unwrap()));
        let re2 = idx.iter().map(|&i| re[i]).collect();
        let im2 = idx.iter().map(|&i| im[i]).collect();
        re = re2; im = im2; (re, im)
    }

    #[test]
    fn upper_triangular_eigenvalues_are_the_diagonal() {
        // Column-major 3×3 upper triangular: diag 2,5,7 (non-symmetric).
        // A = [[2,1,4],[0,5,6],[0,0,7]]  (row-major); column-major flat:
        let a = vec![2.0, 0.0, 0.0,   1.0, 5.0, 0.0,   4.0, 6.0, 7.0];
        let (re, im) = dgeev_values(3, &a).unwrap();
        let (re, _im) = sorted_by_re(re, im);
        assert!((re[0] - 2.0).abs() < 1e-9);
        assert!((re[1] - 5.0).abs() < 1e-9);
        assert!((re[2] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn rotation_has_pure_imaginary_pair() {
        // [[0,-1],[1,0]] → eigenvalues ±i. Column-major: [0,1,-1,0].
        let a = vec![0.0, 1.0, -1.0, 0.0];
        let (re, im) = dgeev_values(2, &a).unwrap();
        assert!(re[0].abs() < 1e-9 && re[1].abs() < 1e-9, "re = {:?}", re);
        let mut ims = im.clone(); ims.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((ims[0] + 1.0).abs() < 1e-9 && (ims[1] - 1.0).abs() < 1e-9, "im = {:?}", im);
    }

    #[test]
    fn matches_r_mixed_spectrum_4x4() {
        // R: matrix(c(4,1,2,0, 3,5,1,2, 0,1,6,3, 1,0,2,7), 4,4, byrow=TRUE)
        // eigen()$values = 9.712272, 5.791871, 3.247928 ± 1.219032i
        // Column-major flat of that matrix:
        let a = vec![4.0,3.0,0.0,1.0,  1.0,5.0,1.0,0.0,  2.0,1.0,6.0,2.0,  0.0,2.0,3.0,7.0];
        let (re, im) = dgeev_values(4, &a).unwrap();
        // Collect as sorted (re desc) pairs.
        let mut pairs: Vec<(f64, f64)> = re.iter().zip(im.iter()).map(|(&r, &i)| (r, i)).collect();
        pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.partial_cmp(&y.1).unwrap()));
        assert!((pairs[0].0 - 9.712272).abs() < 1e-5 && pairs[0].1.abs() < 1e-6, "{:?}", pairs);
        assert!((pairs[1].0 - 5.791871).abs() < 1e-5 && pairs[1].1.abs() < 1e-6, "{:?}", pairs);
        // The complex conjugate pair (order within equal re: -im then +im).
        assert!((pairs[2].0 - 3.247928).abs() < 1e-5 && (pairs[2].1 + 1.219032).abs() < 1e-5, "{:?}", pairs);
        assert!((pairs[3].0 - 3.247928).abs() < 1e-5 && (pairs[3].1 - 1.219032).abs() < 1e-5, "{:?}", pairs);
    }

    #[test]
    fn matches_r_complex_pair_3x3() {
        // R: matrix(c(2,-1,0, 1,2,-1, 0,1,2), 3,3, byrow=TRUE) → 2±1.414214i, 2
        let b = vec![2.0,1.0,0.0,  -1.0,2.0,1.0,  0.0,-1.0,2.0];
        let (re, im) = dgeev_values(3, &b).unwrap();
        // All real parts are 2.
        assert!(re.iter().all(|r| (r - 2.0).abs() < 1e-6), "re = {:?}", re);
        let mut ims: Vec<f64> = im.clone(); ims.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ims[0] + 1.414214).abs() < 1e-5 && ims[1].abs() < 1e-6 && (ims[2] - 1.414214).abs() < 1e-5, "im = {:?}", im);
    }

    #[test]
    fn nonsymmetric_real_spectrum() {
        // Companion-like matrix with known real eigenvalues 1, 2, 3.
        // Build A = diag(1,2,3) conjugated by a shear so it's non-symmetric but
        // keeps the spectrum. Use [[1,2,0],[0,2,3],[0,0,3]] (upper-tri, diag).
        let a = vec![1.0, 0.0, 0.0,   2.0, 2.0, 0.0,   0.0, 3.0, 3.0];
        let (re, im) = dgeev_values(3, &a).unwrap();
        let (re, _) = sorted_by_re(re, im.clone());
        assert!(im.iter().all(|x| x.abs() < 1e-9));
        assert!((re[0] - 1.0).abs() < 1e-9 && (re[1] - 2.0).abs() < 1e-9 && (re[2] - 3.0).abs() < 1e-9, "re = {:?}", re);
    }
}
