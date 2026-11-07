//! Blocked reductions to condensed form, and the blocked product of their reflectors:
//! LAPACK's `dsytrd`, `dgebrd` and `dorgqr`.
//!
//! The unblocked reductions touch the whole trailing matrix once per column, which
//! is memory traffic the cores share. Here a panel of 32 columns keeps its updates as
//! low-rank factors, and the trailing matrix gets them all at once, on the packed
//! GEMM kernel. What cannot be blocked is one matrix-vector product per column with
//! the unreduced trailing matrix, half the flops, so these reductions gain at most
//! about 2x. Forming Q has no such part.

use rayon::prelude::*;

use crate::simd::{axpy, dot};
use crate::LinalgError;

const PANEL: usize = 32;
/// Rows x columns above which a matrix-vector product forks.
const PAR: usize = 1 << 15;

/// What to divide a matrix by so that squares of its entries stay in range: its
/// largest entry when that is extreme, otherwise 1, which changes no bit.
pub(crate) fn safe_scale(a: &[f64]) -> f64 {
    let top = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if top > 0.0 && top.is_finite() && !(1e-100..=1e100).contains(&top) { top } else { 1.0 }
}

/// A Householder reflector for `x`, in place: `H x = (alpha, 0…0)`, `H = I − τ v vᵀ`,
/// `v` left in `x`. An `x` already of that form gets `τ = 0`.
fn reflector(x: &mut [f64]) -> (f64, f64) {
    if x.is_empty() { return (0.0, 0.0); }
    let tail: f64 = x[1..].iter().map(|v| v * v).sum();
    if tail == 0.0 {
        let alpha = x[0];
        x.fill(0.0);
        return (alpha, 0.0);
    }
    // the sum again, scaled, if its squares left the safe range
    let top = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let norm = if tail + x[0] * x[0] > 1e-280 && tail < 1e280 { (x[0] * x[0] + tail).sqrt() }
        else { top * x.iter().map(|v| (v / top) * (v / top)).sum::<f64>().sqrt() };
    let alpha = -norm.copysign(x[0]);
    x[0] -= alpha;
    // τ = 2/vᵀv, with vᵀv = 2·norm·(norm + |x₀|) known exactly: no second sum
    let tau = 1.0 / (alpha * -x[0]);
    (alpha, tau)
}

/// `y[i] = column(i)ᵀ · x` for each column of `a` (rows `r0..`), forking when large.
fn columns_dot(a: &[f64], lda: usize, r0: usize, x: &[f64], y: &mut [f64]) {
    let work = a.len() / lda.max(1) * x.len();
    let one = |(c, out): (usize, &mut f64)| { *out = dot(&a[c * lda + r0..c * lda + r0 + x.len()], x); };
    if work >= PAR { y.par_iter_mut().enumerate().for_each(one); }
    else { y.iter_mut().enumerate().for_each(one); }
}

/// `y = A·x` for `A = a[r0.., ..]` of `y.len()` rows, one `axpy` per column, the rows
/// split across threads.
fn matvec(a: &[f64], lda: usize, r0: usize, x: &[f64], y: &mut [f64]) {
    let rows = y.len();
    let fill = |(ci, ch): (usize, &mut [f64])| {
        ch.fill(0.0);
        let i0 = r0 + ci * 256;
        for (j, &xj) in x.iter().enumerate() {
            if xj != 0.0 { axpy(ch, xj, &a[j * lda + i0..j * lda + i0 + ch.len()]); }
        }
    };
    if rows * x.len() >= PAR { y.par_chunks_mut(256).enumerate().for_each(fill); }
    else { y.chunks_mut(256).enumerate().for_each(fill); }
}

/// `y −= A·(Bᵀ·v)`, for the first `cols` columns of `a`, `b` from row `at` down.
fn minus_low_rank(y: &mut [f64], a: &[f64], b: &[f64], ld: usize, at: usize, cols: usize, v: &[f64]) {
    for p in 0..cols {
        let s = dot(&b[p * ld + at..p * ld + at + v.len()], v);
        if s != 0.0 { axpy(y, -s, &a[p * ld + at..p * ld + at + y.len()]); }
    }
}

/// Columns per parallel chunk of an update: what stays in L2, as `apply_block_reflector`
/// measured it.
fn chunk_columns(rows: usize) -> usize { if rows * 64 * 8 <= 1 << 19 { 64 } else { 32 } }

/// `C −= A·Bᵀ`, with `a` and `b` of `depth` columns taken from row `ar` and `br`: small
/// serial products, a chunk of columns to a thread, so one pass goes over `C` per panel.
#[allow(clippy::too_many_arguments)]
fn trailing_update(c: &mut [f64], rows: usize, cols: usize, a: &[f64], lda: usize, ar: usize,
                   b: &[f64], ldb: usize, br: usize, depth: usize) {
    use crate::gemm::{f64 as g, Trans};
    if rows == 0 || cols == 0 || depth == 0 { return; }
    let mut at = vec![0.0f64; rows * depth];
    for p in 0..depth { at[p * rows..(p + 1) * rows].copy_from_slice(&a[p * lda + ar..p * lda + ar + rows]); }
    let width = chunk_columns(rows);
    let at = &at;
    // Column-major C is row-major Cᵀ, and a's buffer is row-major Aᵀ: Cᵀ −= B·Aᵀ.
    c.par_chunks_mut(width * rows).enumerate().for_each(|(ci, chunk)| {
        let cw = chunk.len() / rows;
        let mut bn = vec![0.0f64; cw * depth];
        for j in 0..cw {
            for p in 0..depth { bn[j * depth + p] = -b[p * ldb + br + ci * width + j]; }
        }
        g::gemm_into(&bn, Trans::No, at, Trans::No, cw, depth, rows, chunk, false);
    });
}

/// `dst` of `rows x cols` from the trailing part of `src`, its own buffer reused.
fn trailing_copy(dst: &mut Vec<f64>, src: &[f64], lda: usize, skip: usize, rows: usize, cols: usize) {
    dst.resize(rows * cols, 0.0);
    dst.par_chunks_mut(rows).enumerate().for_each(|(c, col)| {
        col.copy_from_slice(&src[(skip + c) * lda + skip..(skip + c) * lda + skip + rows]);
    });
}

/// A reduction's output: the condensed matrix and, if asked for, its reflectors.
pub(crate) struct Reflectors { pub vs: Vec<Vec<f64>>, pub taus: Vec<f64> }

/// Reduce the symmetric `n x n` `a` (both triangles present, column-major) to a
/// tridiagonal `(d, e)`; the reflectors act on rows `1 + i..`.
pub(crate) fn tridiagonalize(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>, Reflectors), LinalgError> {
    let mut d = Vec::with_capacity(n);
    let mut e = Vec::with_capacity(n.saturating_sub(1));
    let mut refl = Reflectors { vs: Vec::with_capacity(n), taus: Vec::with_capacity(n) };
    let mut t = a.to_vec();
    let mut spare = Vec::new();
    let mut m = n;
    while m > 0 {
        let b = PANEL.min(m);
        let mut v = vec![0.0f64; m * b];
        let mut w = vec![0.0f64; m * b];
        for j in 0..b {
            // Column j with the panel's updates so far.
            let mut col = t[j * m + j..(j + 1) * m].to_vec();
            for p in 0..j {
                axpy(&mut col, -w[p * m + j], &v[p * m + j..(p + 1) * m]);
                axpy(&mut col, -v[p * m + j], &w[p * m + j..(p + 1) * m]);
            }
            d.push(col[0]);
            if j + 1 == m { break; }
            let x = &mut col[1..];
            let (alpha, tau) = reflector(x);
            e.push(alpha);
            refl.vs.push(x.to_vec());
            refl.taus.push(tau);
            let rows = m - j - 1;
            v[j * m + j + 1..(j + 1) * m].copy_from_slice(x);
            if tau == 0.0 { continue; }

            // w = τ(A·v − V(Wᵀv) − W(Vᵀv)), then w −= (τ/2)(wᵀv)·v.
            let vj = x.to_vec();
            let mut y = vec![0.0f64; rows];
            columns_dot(&t[(j + 1) * m..], m, j + 1, &vj, &mut y);
            minus_low_rank(&mut y, &v, &w, m, j + 1, j, &vj);
            minus_low_rank(&mut y, &w, &v, m, j + 1, j, &vj);
            for u in y.iter_mut() { *u *= tau; }
            let half = 0.5 * tau * dot(&y, &vj);
            axpy(&mut y, -half, &vj);
            w[j * m + j + 1..(j + 1) * m].copy_from_slice(&y);
        }
        if b >= m { break; }
        // The rest of the matrix, shrunk to what is left.
        let rest = m - b;
        trailing_copy(&mut spare, &t, m, b, rest, rest);
        trailing_update(&mut spare, rest, rest, &v, m, b, &w, m, b, b);
        trailing_update(&mut spare, rest, rest, &w, m, b, &v, m, b, b);
        std::mem::swap(&mut t, &mut spare);
        m = rest;
    }
    Ok((d, e, refl))
}

/// Reduce the `m x n` `a` (`m >= n`) to an upper bidiagonal `(d, e)`: left reflector
/// `i` acts on rows `i..`, right reflector `i` on columns `i + 1..`.
pub(crate) fn bidiagonalize(a: &[f64], m0: usize, n0: usize, keep: bool)
    -> Result<(Vec<f64>, Vec<f64>, Reflectors, Reflectors), LinalgError>
{
    let mut d = Vec::with_capacity(n0);
    let mut e = Vec::with_capacity(n0.saturating_sub(1));
    let mut left = Reflectors { vs: Vec::new(), taus: Vec::new() };
    let mut right = Reflectors { vs: Vec::new(), taus: Vec::new() };
    let mut t = a.to_vec();
    let mut spare = Vec::new();
    let (mut m, mut n) = (m0, n0);
    while n > 0 {
        let b = PANEL.min(n);
        // A − U Yᵀ − X Vᵀ is the matrix so far, each factor from row/column 0 of `t`.
        let mut u = vec![0.0f64; m * b];
        let mut x = vec![0.0f64; m * b];
        let mut v = vec![0.0f64; n * b];
        let mut y = vec![0.0f64; n * b];
        for j in 0..b {
            // Column j with the panel's updates, and its reflector.
            let mut col = t[j * m + j..(j + 1) * m].to_vec();
            for p in 0..j {
                axpy(&mut col, -y[p * n + j], &u[p * m + j..(p + 1) * m]);
                axpy(&mut col, -v[p * n + j], &x[p * m + j..(p + 1) * m]);
            }
            let (alpha, tauq) = reflector(&mut col);
            d.push(alpha);
            if keep { left.vs.push(col.clone()); left.taus.push(tauq); }
            u[j * m + j..(j + 1) * m].copy_from_slice(&col);
            let cols = n - j - 1;
            if cols == 0 { continue; }

            // y_j = τ(Aᵀu − Y(Uᵀu) − V(Xᵀu)), on columns j + 1 onward.
            let mut yj = vec![0.0f64; cols];
            if tauq != 0.0 {
                columns_dot(&t[(j + 1) * m..], m, j, &col, &mut yj);
                for p in 0..j {
                    let s = dot(&u[p * m + j..(p + 1) * m], &col);
                    if s != 0.0 { axpy(&mut yj, -s, &y[p * n + j + 1..(p + 1) * n]); }
                    let s = dot(&x[p * m + j..(p + 1) * m], &col);
                    if s != 0.0 { axpy(&mut yj, -s, &v[p * n + j + 1..(p + 1) * n]); }
                }
                for s in yj.iter_mut() { *s *= tauq; }
            }
            y[j * n + j + 1..(j + 1) * n].copy_from_slice(&yj);

            // Row j with the panel's updates, and its reflector.
            let mut row: Vec<f64> = (0..cols).map(|c| t[(j + 1 + c) * m + j]).collect();
            for p in 0..=j {
                let s = u[p * m + j];
                if s != 0.0 { axpy(&mut row, -s, &y[p * n + j + 1..(p + 1) * n]); }
            }
            for p in 0..j {
                let s = x[p * m + j];
                if s != 0.0 { axpy(&mut row, -s, &v[p * n + j + 1..(p + 1) * n]); }
            }
            let (alpha, taup) = reflector(&mut row);
            e.push(alpha);
            if keep { right.vs.push(row.clone()); right.taus.push(taup); }
            v[j * n + j + 1..(j + 1) * n].copy_from_slice(&row);
            let rows = m - j - 1;
            if rows == 0 || taup == 0.0 { continue; }

            // x_j = τ(A·v − U(Yᵀv) − X(Vᵀv)), on rows j + 1 downward.
            let mut xj = vec![0.0f64; rows];
            matvec(&t[(j + 1) * m..], m, j + 1, &row, &mut xj);
            for p in 0..=j {
                let s = dot(&y[p * n + j + 1..(p + 1) * n], &row);
                if s != 0.0 { axpy(&mut xj, -s, &u[p * m + j + 1..(p + 1) * m]); }
            }
            for p in 0..j {
                let s = dot(&v[p * n + j + 1..(p + 1) * n], &row);
                if s != 0.0 { axpy(&mut xj, -s, &x[p * m + j + 1..(p + 1) * m]); }
            }
            for s in xj.iter_mut() { *s *= taup; }
            x[j * m + j + 1..(j + 1) * m].copy_from_slice(&xj);
        }
        if b >= n { break; }
        let (rm, rn) = (m - b, n - b);
        trailing_copy(&mut spare, &t, m, b, rm, rn);
        trailing_update(&mut spare, rm, rn, &u, m, b, &y, n, b, b);
        trailing_update(&mut spare, rm, rn, &x, m, b, &v, n, b, b);
        std::mem::swap(&mut t, &mut spare);
        m = rm;
        n = rn;
    }
    Ok((d, e, left, right))
}

/// The first `n` columns of `H₀ H₁ …`, `m` rows, where reflector `i` acts on rows
/// `start + i..`. Each block is applied as one `I − V T Vᵀ` (compact WY, `dlarft`).
pub(crate) fn form_q(m: usize, n: usize, start: usize, r: &Reflectors) -> Result<Vec<f64>, LinalgError> {
    let mut q = vec![0.0f64; m * n];
    for i in 0..n.min(m) { q[i * m + i] = 1.0; }
    let count = r.taus.len();
    let mut k1 = count;
    while k1 > 0 {
        let k0 = k1.saturating_sub(PANEL);
        let b = k1 - k0;
        let r0 = start + k0;
        let rows = m - r0;
        if r0 >= n { k1 = k0; continue; }
        // V, zeros above each reflector's first row.
        let mut v = vec![0.0f64; rows * b];
        for p in 0..b {
            let vec_p = &r.vs[k0 + p];
            v[p * rows + p..p * rows + p + vec_p.len()].copy_from_slice(vec_p);
        }
        // T, upper triangular: T[..p, p] = −τ T[..p, ..p] (V[..p]ᵀ v_p).
        let mut t = vec![0.0f64; b * b];
        for p in 0..b {
            let tau = r.taus[k0 + p];
            t[p * b + p] = tau;
            if tau == 0.0 { continue; }
            let vp = &v[p * rows + p..(p + 1) * rows];
            let w: Vec<f64> = (0..p).map(|c| dot(&v[c * rows + p..(c + 1) * rows], vp)).collect();
            for row in 0..p {
                let s: f64 = (row..p).map(|c| t[c * b + row] * w[c]).sum();
                t[p * b + row] = -tau * s;
            }
        }
        // Q[r0.., r0..] ← (I − V T Vᵀ) Q[r0.., r0..], a chunk of columns to a thread.
        // Row-major view: a column-major chunk C is Cᵀ, V's buffer is Vᵀ, T's is Tᵀ:
        //   Wᵀ = Cᵀ·V,   W₂ᵀ = Wᵀ·Tᵀ,   Cᵀ −= W₂ᵀ·Vᵀ.
        use crate::gemm::{f64 as g, Trans};
        let width = chunk_columns(rows);
        let (vr, tr) = (&v, &t);
        q[r0 * m..].par_chunks_mut(width * m).for_each(|block| {
            let cw = block.len() / m;
            let mut c = vec![0.0f64; rows * cw];
            for j in 0..cw { c[j * rows..(j + 1) * rows].copy_from_slice(&block[j * m + r0..(j + 1) * m]); }
            let w = g::gemm(&c, Trans::No, vr, Trans::Yes, cw, rows, b, false);
            let mut w2 = g::gemm(&w, Trans::No, tr, Trans::No, cw, b, b, false);
            for x in w2.iter_mut() { *x = -*x; }
            g::gemm_into(&w2, Trans::No, vr, Trans::No, cw, b, rows, &mut c, false);
            for j in 0..cw { block[j * m + r0..(j + 1) * m].copy_from_slice(&c[j * rows..(j + 1) * rows]); }
        });
        k1 = k0;
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
        (0..n).map(|_| {
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }).collect()
    }

    fn product(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        crate::dgemm(m, n, k, 1.0, a, b, 0.0, &mut c).unwrap();
        c
    }

    fn transpose(m: usize, n: usize, a: &[f64]) -> Vec<f64> {
        let mut t = vec![0.0; m * n];
        for j in 0..n { for i in 0..m { t[i * n + j] = a[j * m + i]; } }
        t
    }

    fn assert_orthonormal(m: usize, n: usize, q: &[f64], name: &str) {
        let g = product(n, m, n, &transpose(m, n, q), q);
        for j in 0..n { for i in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((g[j * n + i] - want).abs() < 1e-13, "{name}: QᵀQ[{i},{j}] = {}", g[j * n + i]);
        } }
    }

    #[test]
    fn tridiagonalize_is_a_similarity() {
        for n in [1usize, 2, 3, 31, 32, 33, 64, 65, 150] {
            let g = noise(n * n, n as u64);
            let mut a = vec![0.0; n * n];
            for j in 0..n { for i in 0..n { a[j * n + i] = g[j * n + i] + g[i * n + j]; } }
            let (d, e, r) = tridiagonalize(&a, n).unwrap();
            assert_eq!((d.len(), e.len()), (n, n - 1));
            let q = form_q(n, n, 1, &r).unwrap();
            assert_orthonormal(n, n, &q, "tridiagonal Q");
            let mut t = vec![0.0; n * n];
            for i in 0..n { t[i * n + i] = d[i]; }
            for i in 0..n - 1 { t[i * n + i + 1] = e[i]; t[(i + 1) * n + i] = e[i]; }
            let back = product(n, n, n, &product(n, n, n, &q, &t), &transpose(n, n, &q));
            let err = back.iter().zip(&a).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12 * n as f64, "n={n}: |QTQᵀ − A| = {err}");
        }
    }

    #[test]
    fn bidiagonalize_is_an_orthogonal_equivalence() {
        for (m, n) in [(1usize, 1usize), (2, 2), (5, 3), (40, 31), (33, 33), (64, 32), (150, 65), (120, 120)] {
            let a = noise(m * n, (m * 1000 + n) as u64);
            let (d, e, l, r) = bidiagonalize(&a, m, n, true).unwrap();
            assert_eq!((d.len(), e.len()), (n, n - 1));
            let u = form_q(m, n, 0, &l).unwrap();
            let v = form_q(n, n, 1, &r).unwrap();
            assert_orthonormal(m, n, &u, "U");
            assert_orthonormal(n, n, &v, "V");
            let mut b = vec![0.0; n * n];
            for i in 0..n { b[i * n + i] = d[i]; }
            for i in 0..n - 1 { b[(i + 1) * n + i] = e[i]; }
            let back = product(m, n, n, &product(m, n, n, &u, &b), &transpose(n, n, &v));
            let err = back.iter().zip(&a).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(err < 1e-12 * n as f64, "{m}x{n}: |UBVᵀ − A| = {err}");
        }
    }

    #[test]
    fn zero_and_rank_one_matrices_reduce() {
        let (d, e, _, _) = bidiagonalize(&vec![0.0; 80 * 40], 80, 40, true).unwrap();
        assert!(d.iter().chain(&e).all(|x| *x == 0.0));
        let (d, _, _) = tridiagonalize(&vec![1.0; 2500], 50).unwrap();
        assert!((d.iter().sum::<f64>() - 50.0).abs() < 1e-11);
    }
}
