//! LINPACK's Householder QR in the form R's `qr()` exposes: `dqrdc2` and the
//! operations of `dqrsl`.
//!
//! This is not `dgeqrf`, and the two cannot share an object. R documents `$qr`
//! and `$qraux` as this compact form, and `dqrdc2`'s limited pivoting is what
//! decides `$rank` and `$pivot`: a column whose norm has fallen below `tol` times
//! its original norm moves to the far right, and the others keep their order.
//!
//! All storage is column-major. Each reflector H_j has the vector
//! `(qraux[j], qr[j+1.., j])`; the diagonal of `qr` holds R's diagonal.

use crate::level1::{daxpy, ddot, dnrm2, dscal};

/// The columns' rank and original positions (1-based, as R reports them), with
/// `qr` overwritten by the compact form.
pub struct Decomposition {
    pub rank: usize,
    pub qraux: Vec<f64>,
    pub pivot: Vec<i32>,
}

/// `dqrdc2`: factor the n×p `x` in place.
pub fn dqrdc2(x: &mut [f64], n: usize, p: usize, tol: f64) -> Decomposition {
    assert_eq!(x.len(), n * p, "dqrdc2: x must be n by p");
    let mut pivot: Vec<i32> = (1..=p as i32).collect();
    let mut qraux: Vec<f64> = (0..p).map(|j| dnrm2(&x[j * n..(j + 1) * n])).collect();
    let mut original: Vec<f64> = qraux.iter().map(|&v| if v == 0.0 { 1.0 } else { v }).collect();

    // Columns from `kept` on have been moved aside as negligible.
    let mut kept = p;
    for l in 0..n.min(p) {
        while l < kept && qraux[l] < original[l] * tol {
            x[l * n..p * n].rotate_left(n);
            pivot[l..p].rotate_left(1);
            qraux[l..p].rotate_left(1);
            original[l..p].rotate_left(1);
            kept -= 1;
        }
        if l + 1 == n { break; }

        let (done, rest) = x.split_at_mut((l + 1) * n);
        let v = &mut done[l * n + l..];
        let mut norm = dnrm2(v);
        if norm == 0.0 { continue; }
        if v[0] != 0.0 { norm = norm.copysign(v[0]); }
        dscal(1.0 / norm, v);
        v[0] += 1.0;
        for j in (l + 1)..p {
            let c = &mut rest[(j - l - 1) * n + l..(j - l) * n];
            let t = -ddot(v, c) / v[0];
            daxpy(t, v, c);
            if qraux[j] != 0.0 {
                let ratio = (c[0].abs() / qraux[j]).powi(2);
                let left = (1.0 - ratio).max(0.0);
                // Downdating the norm loses all accuracy near zero: recompute.
                qraux[j] = if left < 1e-6 { dnrm2(&c[1..]) } else { qraux[j] * left.sqrt() };
            }
        }
        qraux[l] = v[0];
        v[0] = -norm;
    }
    Decomposition { rank: kept.min(n), qraux, pivot }
}

/// The same factorisation by the blocked, parallel QR (`dgeqrf`), when it is
/// sure to be the one `dqrdc2` would give: a tall matrix (n >= p) on which
/// `dqrdc2` would move no column, because every diagonal of R keeps well over
/// `tol` of its column's norm. `None` otherwise, and the caller runs `dqrdc2`.
///
/// The two agree on the sign of R, and their reflectors differ only in
/// scaling: LINPACK's vector u with u[0] = qraux is LAPACK's v (v[0] = 1) times
/// tau, so `qraux = tau` and the stored part is `tau * v`. A square matrix's
/// last column has no reflector in LINPACK, where `dgeqrf` reflects it to the
/// sign of its norm; that one is undone.
pub fn dqrdc2_blocked(x: &mut [f64], n: usize, p: usize, tol: f64) -> Option<Decomposition> {
    if n < p || p == 0 { return None; }
    let norms: Vec<f64> = (0..p).map(|j| dnrm2(&x[j * n..(j + 1) * n])).collect();
    let mut a = x.to_vec();
    let tau = crate::dgeqrf(n, p, &mut a).ok()?;
    // dqrdc2 judges a column by a downdated estimate of its norm; the margin
    // keeps clear of the cases where estimate and exact value could disagree.
    const MARGIN: f64 = 100.0;
    for j in 0..p {
        let d = a[j * n + j].abs();
        if norms[j] == 0.0 || d < MARGIN * tol * norms[j] || !d.is_finite() { return None; }
        // dgeqrf skips a column whose remaining norm is below 1e-15 in absolute
        // terms, which is not a rank decision on a finely scaled matrix.
        if tau[j] == 0.0 && !(n == p && j + 1 == p) { return None; }
    }
    let mut qraux = tau;
    for j in 0..p {
        let t = qraux[j];
        for v in &mut a[j * n + j + 1..(j + 1) * n] { *v *= t; }
    }
    if n == p {
        let last = (p - 1) * n + p - 1;
        if qraux[p - 1] != 0.0 { a[last] = -a[last]; }
        qraux[p - 1] = a[last].abs();
    }
    x.copy_from_slice(&a);
    Some(Decomposition { rank: p, qraux, pivot: (1..=p as i32).collect() })
}

/// Apply the reflector H_j to `y`.
#[inline]
fn reflect(qr: &[f64], n: usize, j: usize, aux: f64, y: &mut [f64]) {
    if aux == 0.0 { return; }
    let u = &qr[j * n + j + 1..(j + 1) * n];
    let (head, tail) = y[j..].split_at_mut(1);
    let t = -(aux * head[0] + ddot(u, tail)) / aux;
    head[0] += t * aux;
    daxpy(t, u, tail);
}

/// `y <- Q'y`, using the first `k` reflectors. The last row has none.
pub fn qty(qr: &[f64], n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    for j in 0..k.min(n.saturating_sub(1)) { reflect(qr, n, j, qraux[j], y); }
}

/// `y <- Qy`, using the first `k` reflectors.
pub fn qy(qr: &[f64], n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    for j in (0..k.min(n.saturating_sub(1))).rev() { reflect(qr, n, j, qraux[j], y); }
}

/// The least-squares coefficients of `y` on the first `k` pivoted columns.
/// `Err(j)` names the 1-based column with a zero on R's diagonal.
pub fn coef(qr: &[f64], n: usize, k: usize, qraux: &[f64], y: &[f64]) -> Result<Vec<f64>, usize> {
    let mut b = y.to_vec();
    qty(qr, n, k, qraux, &mut b);
    b.truncate(k);
    for j in (0..k).rev() {
        let d = qr[j * n + j];
        if d == 0.0 { return Err(j + 1); }
        b[j] /= d;
        let t = -b[j];
        daxpy(t, &qr[j * n..j * n + j], &mut b[..j]);
    }
    Ok(b)
}

/// `y <- y - fitted` for the first `k` columns.
pub fn resid(qr: &[f64], n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    qty(qr, n, k, qraux, y);
    y[..k.min(n)].fill(0.0);
    qy(qr, n, k, qraux, y);
}

/// `y <- fitted` on the first `k` columns.
pub fn fitted(qr: &[f64], n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    qty(qr, n, k, qraux, y);
    if k < n { y[k..].fill(0.0); }
    qy(qr, n, k, qraux, y);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked result R gives for `qr(cbind(1, c(1, 2, 4, 8)))`.
    #[test]
    fn the_compact_form_is_r_s() {
        let mut x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 8.0];
        let d = dqrdc2(&mut x, 4, 2, 1e-7);
        assert_eq!(d.rank, 2);
        assert_eq!(d.pivot, vec![1, 2]);
        assert!((x[0] + 2.0).abs() < 1e-14 && (x[1] - 0.5).abs() < 1e-14 && (x[4] + 7.5).abs() < 1e-14);
        assert!((d.qraux[0] - 1.5).abs() < 1e-14);
    }

    #[test]
    fn a_collinear_column_moves_right_and_the_rest_keep_their_order() {
        let mut x = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
        let d = dqrdc2(&mut x, 3, 3, 1e-7);
        assert_eq!(d.rank, 2);
        assert_eq!(d.pivot, vec![1, 3, 2]);
    }

    #[test]
    fn q_then_q_transpose_is_the_identity() {
        let a: Vec<f64> = (0..20).map(|i| ((i * i + 1) as f64).sin()).collect();
        let mut x = a.clone();
        let d = dqrdc2(&mut x, 5, 4, 1e-7);
        let y0 = vec![1.0, -2.0, 3.0, 0.5, 7.0];
        let mut y = y0.clone();
        qty(&x, 5, d.rank, &d.qraux, &mut y);
        qy(&x, 5, d.rank, &d.qraux, &mut y);
        for (u, v) in y.iter().zip(&y0) { assert!((u - v).abs() < 1e-13); }
    }

    #[test]
    fn fitted_plus_residual_is_the_response() {
        let a: Vec<f64> = (0..12).map(|i| ((3 * i + 1) as f64).cos()).collect();
        let mut x = a.clone();
        let d = dqrdc2(&mut x, 6, 2, 1e-7);
        let y0 = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let (mut f, mut r) = (y0.clone(), y0.clone());
        fitted(&x, 6, d.rank, &d.qraux, &mut f);
        resid(&x, 6, d.rank, &d.qraux, &mut r);
        for i in 0..6 { assert!((f[i] + r[i] - y0[i]).abs() < 1e-12); }
        // The residual is orthogonal to every column.
        for j in 0..2 { assert!(ddot(&a[j * 6..(j + 1) * 6], &r).abs() < 1e-11); }
    }

    #[test]
    fn a_single_row_has_no_reflectors() {
        let mut x = vec![2.0, 3.0];
        let d = dqrdc2(&mut x, 1, 2, 1e-7);
        assert_eq!(d.rank, 1);
        assert_eq!(coef(&x, 1, 1, &d.qraux, &[4.0]), Ok(vec![2.0]));
    }

    /// A full-rank n x p matrix: uniform values from a linear congruential generator.
    fn wave(n: usize, p: usize) -> Vec<f64> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        (0..n * p).map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }).collect()
    }

    #[test]
    fn the_blocked_route_gives_dqrdc2_s_object() {
        for (n, p) in [(300, 120), (130, 130), (70, 1), (200, 199)] {
            let a = wave(n, p);
            let (mut x, mut y) = (a.clone(), a.clone());
            let d = dqrdc2(&mut x, n, p, 1e-7);
            let b = dqrdc2_blocked(&mut y, n, p, 1e-7).expect("full rank");
            assert_eq!((b.rank, &b.pivot), (d.rank, &d.pivot), "{n}x{p}");
            let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (i, (u, v)) in x.iter().zip(&y).enumerate() {
                assert!((u - v).abs() <= 1e-10 * scale, "{n}x{p} element {i}: {u} vs {v}");
            }
            for (u, v) in d.qraux.iter().zip(&b.qraux) { assert!((u - v).abs() <= 1e-10, "{n}x{p} qraux {u} vs {v}"); }
        }
    }

    #[test]
    fn the_blocked_route_declines_what_dqrdc2_would_pivot() {
        let mut a = wave(50, 4);
        for i in 0..50 { a[2 * 50 + i] = a[i] + a[50 + i]; }
        assert!(dqrdc2_blocked(&mut a.clone(), 50, 4, 1e-7).is_none(), "collinear");
        assert!(dqrdc2_blocked(&mut wave(4, 6), 4, 6, 1e-7).is_none(), "wide");
        let tiny: Vec<f64> = wave(40, 3).iter().map(|v| v * 1e-20).collect();
        let mut t = tiny.clone();
        if let Some(b) = dqrdc2_blocked(&mut t, 40, 3, 1e-7) {
            let mut u = tiny.clone();
            let d = dqrdc2(&mut u, 40, 3, 1e-7);
            assert_eq!(b.rank, d.rank);
            for (x, y) in t.iter().zip(&u) { assert!((x - y).abs() <= 1e-10 * 1e-20); }
        }
    }

    #[test]
    fn rank_zero_for_a_zero_matrix() {
        let mut x = vec![0.0; 6];
        let d = dqrdc2(&mut x, 3, 2, 1e-7);
        assert_eq!(d.rank, 0);
    }
}
