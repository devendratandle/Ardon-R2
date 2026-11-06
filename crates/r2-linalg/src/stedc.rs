//! Eigenvalues and vectors of a symmetric tridiagonal matrix by divide and conquer:
//! LAPACK's `dstedc` (Cuppen; Gu and Eisenstat).
//!
//! Split `T` in two with a rank-one tear, solve each half, then solve the rank-one
//! update of the joined diagonal: `D + ρ z zᵀ`. Its eigenvalues are the roots of the
//! secular equation, one per gap, and its eigenvectors are explicit. Nearly all the
//! flops are one matrix product per merge, on the packed GEMM kernel, and deflation
//! usually removes much of even that, where the QR iteration's rotations are
//! `O(n³)` work no kernel can pack.
//!
//! Orthogonality comes from Gu and Eisenstat: `z` is recomputed from the computed
//! roots, so the vectors are the exact eigenvectors of a matrix next to the given
//! one, and each root keeps its distance to its nearest pole instead of its absolute
//! value, so nothing subtracts two nearly equal numbers.

use rayon::prelude::*;

use crate::symeig::tridiag_eigen;
use crate::LinalgError;

const EPS: f64 = f64::EPSILON / 2.0;
/// Below this a block goes to the QR iteration.
const LEAF: usize = 32;

/// One root of the secular equation: `λ = d[origin] + mu`.
#[derive(Clone, Copy)]
struct Root { origin: usize, mu: f64 }

/// `f(μ) = 1 + ρ Σ z²ᵢ / (δᵢ − μ)` split at the pole `split` (ψ: below it, φ: from it
/// up), with derivatives and the bound on `f`'s rounding.
struct Secular { f: f64, psi: f64, dpsi: f64, phi: f64, dphi: f64, err: f64 }

fn secular(delta: &[f64], z2: &[f64], rho: f64, split: usize, mu: f64) -> Secular {
    let (mut psi, mut dpsi, mut phi, mut dphi) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..split {
        let t = 1.0 / (delta[i] - mu);
        let v = z2[i] * t;
        psi += v;
        dpsi += v * t;
    }
    for i in split..delta.len() {
        let t = 1.0 / (delta[i] - mu);
        let v = z2[i] * t;
        phi += v;
        dphi += v * t;
    }
    let (psi, dpsi, phi, dphi) = (rho * psi, rho * dpsi, rho * phi, rho * dphi);
    let f = 1.0 + psi + phi;
    Secular { f, psi, dpsi, phi, dphi, err: 8.0 * (psi.abs() + phi.abs()) + 1.0 + mu.abs() * (dpsi + dphi) }
}

/// The `j`-th root of `1 + ρ Σ z²ᵢ/(dᵢ − λ) = 0`, `d` strictly ascending and `ρ > 0`.
/// Each iteration fits the two nearest poles to `f`'s value and slope (Bunch, Nielsen
/// and Sorensen) and takes the fit's root, inside a bracket that always holds the
/// answer; a fit that leaves the bracket gives way to bisection.
fn solve_root(d: &[f64], z2: &[f64], rho: f64, norm2: f64, j: usize, delta: &mut [f64]) -> Root {
    let k = d.len();
    let last = j + 1 == k;
    let gap = if last { rho * norm2 } else { d[j + 1] - d[j] };

    // Which pole the root is nearer decides the origin: the distance to it is what
    // can be computed to full relative accuracy.
    let mid = 0.5 * gap;
    for (x, di) in delta.iter_mut().zip(d) { *x = di - d[j]; }
    let at_mid = secular(delta, z2, rho, j + 1, mid);
    let origin = if last || at_mid.f >= 0.0 { j } else { j + 1 };
    if origin != j { for (x, di) in delta.iter_mut().zip(d) { *x = di - d[origin]; } }
    let (mut lo, mut hi) = if last {
        (if at_mid.f >= 0.0 { 0.0 } else { mid }, if at_mid.f >= 0.0 { mid } else { gap })
    } else if origin == j { (0.0, mid) } else { (-mid, 0.0) };

    // The two poles fitted exactly: the interval's ends, or the top two for the last.
    let (a_i, b_i) = if last { (k.saturating_sub(2), k - 1) } else { (j, j + 1) };
    let split = if last { k - 1 } else { j + 1 };
    let (a, b) = (delta[a_i], delta[b_i]);

    let mut mu = 0.5 * (lo + hi);
    for _ in 0..120 {
        let s = secular(delta, z2, rho, split, mu);
        if s.f.abs() <= EPS * s.err { break; }
        if s.f > 0.0 { hi = mu } else { lo = mu }
        if hi - lo <= 2.0 * f64::EPSILON * lo.abs().max(hi.abs()) { break; }

        // c + p/(a − x) + q/(b − x) with f's value and slope at μ, ψ to a, φ to b.
        let p = (a - mu) * (a - mu) * s.dpsi;
        let q = (b - mu) * (b - mu) * s.dphi;
        let c = 1.0 + (s.psi - p / (a - mu)) + (s.phi - q / (b - mu));
        let qa = c;
        let qb = -(c * (a + b) + p + q);
        let qc = c * a * b + p * b + q * a;
        let next = if k == 1 {
            f64::NAN
        } else if qa == 0.0 {
            -qc / qb
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 { f64::NAN } else {
                let t = -0.5 * (qb + disc.sqrt().copysign(qb));
                let (x1, x2) = (t / qa, if t != 0.0 { qc / t } else { f64::NAN });
                if x1 > lo && x1 < hi { x1 } else { x2 }
            }
        };
        mu = if next > lo && next < hi {
            next
        } else if lo > 0.0 && hi / lo > 16.0 {
            (lo * hi).sqrt()            // across many scales, bisect the exponent
        } else if hi < 0.0 && lo / hi > 16.0 {
            -(lo * hi).sqrt()
        } else {
            0.5 * (lo + hi)
        };
        if mu == 0.0 { mu = if lo == 0.0 { hi * 1e-3 } else { lo * 1e-3 }; }
    }
    Root { origin, mu }
}

/// Solve `D + ρ z zᵀ` for the merged block and rotate `q` into its eigenvectors:
/// `(values ascending, q)`. `n1` is the first block's size.
fn merge(mut d: Vec<f64>, mut z: Vec<f64>, rho: f64, mut q: Vec<f64>, n: usize, n1: usize)
    -> Result<(Vec<f64>, Vec<f64>), LinalgError>
{
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let dmax = d.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let zmax = z.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let tol = 8.0 * EPS * dmax.max(zmax);

    // Which rows each column can be non-zero in: 1 the first block's, 3 the
    // second's, 2 both, after a deflating rotation mixed them.
    let mut kind: Vec<u8> = (0..n).map(|i| if i < n1 { 1 } else { 3 }).collect();
    let mut keep: Vec<usize> = Vec::with_capacity(n);
    let mut prev: Option<usize> = None;
    if rho * zmax > tol {
        for &nj in &perm {
            if rho * z[nj].abs() <= tol { continue; }
            let Some(pj) = prev else { prev = Some(nj); continue };
            let (mut s, mut c) = (z[pj], z[nj]);
            let tau = c.hypot(s);
            let t = d[nj] - d[pj];
            c /= tau;
            s = -s / tau;
            if (t * c * s).abs() <= tol {
                // Nearly equal values: a rotation zeroes one z, and that one deflates.
                z[nj] = tau;
                z[pj] = 0.0;
                if kind[pj] != kind[nj] { kind[pj] = 2; kind[nj] = 2; }
                let (lo, hi) = q.split_at_mut(pj.max(nj) * n);
                let (x, y) = if pj < nj {
                    (&mut lo[pj * n..(pj + 1) * n], &mut hi[..n])
                } else {
                    (&mut hi[..n], &mut lo[nj * n..(nj + 1) * n])
                };
                for (a, b) in x.iter_mut().zip(y.iter_mut()) {
                    let (xa, yb) = (*a, *b);
                    *a = c * xa + s * yb;
                    *b = c * yb - s * xa;
                }
                let (dp, dn) = (d[pj], d[nj]);
                d[pj] = dp * c * c + dn * s * s;
                d[nj] = dp * s * s + dn * c * c;
            } else {
                keep.push(pj);
            }
            prev = Some(nj);
        }
        if let Some(pj) = prev { keep.push(pj); }
    }

    let k = keep.len();
    if k == 0 { return Ok(sorted_columns(d, q, n)); }
    let ds: Vec<f64> = keep.iter().map(|&i| d[i]).collect();
    let zs: Vec<f64> = keep.iter().map(|&i| z[i]).collect();
    let z2: Vec<f64> = zs.iter().map(|x| x * x).collect();
    let norm2: f64 = z2.iter().sum();

    // The roots, and for each, its distance from every pole: `dist[j][i] = dᵢ − λⱼ`.
    let mut dist = vec![0.0f64; k * k];
    let roots: Vec<Root> = dist.par_chunks_mut(k).enumerate()
        .map(|(j, col)| {
            let root = solve_root(&ds, &z2, rho, norm2, j, col);
            for (x, di) in col.iter_mut().zip(&ds) { *x = (di - ds[root.origin]) - root.mu; }
            root
        }).collect();

    // Gu and Eisenstat: the z for which these roots are exact.
    let mut zhat: Vec<f64> = (0..k).into_par_iter().map(|i| {
        let mut w = dist[i * k + i];
        for j in 0..k {
            if j != i { w *= dist[j * k + i] / (ds[i] - ds[j]); }
        }
        (-w).max(0.0).sqrt().copysign(zs[i])
    }).collect();
    if zhat.iter().all(|x| *x == 0.0) { zhat.copy_from_slice(&zs); }

    // The eigenvectors of the rank-one update, by rows of the kept columns' kinds.
    let mut vecs = vec![0.0f64; k * k];
    vecs.par_chunks_mut(k).enumerate().for_each(|(j, v)| {
        let col = &dist[j * k..(j + 1) * k];
        let mut norm = 0.0;
        for i in 0..k { let x = zhat[i] / col[i]; v[i] = x; norm += x * x; }
        let norm = norm.sqrt();
        for x in v.iter_mut() { *x /= norm; }
    });

    // Q_new = Q_kept · vecs. The first block's rows see kinds 1 and 2, the second's
    // 2 and 3, so each product leaves out the columns that are zero there.
    let mut out = vec![0.0f64; n * k];
    for (rows, skip) in [(0..n1, 3u8), (n1..n, 1u8)] {
        let picked: Vec<usize> = (0..k).filter(|&c| kind[keep[c]] != skip).collect();
        let nr = rows.len();
        if nr == 0 || picked.is_empty() { continue; }
        let mut a = vec![0.0f64; nr * picked.len()];
        let mut b = vec![0.0f64; picked.len() * k];
        for (p, &c) in picked.iter().enumerate() {
            a[p * nr..(p + 1) * nr].copy_from_slice(&q[keep[c] * n + rows.start..keep[c] * n + rows.end]);
            for j in 0..k { b[j * picked.len() + p] = vecs[j * k + c]; }
        }
        let mut c = vec![0.0f64; nr * k];
        crate::dgemm(nr, k, picked.len(), 1.0, &a, &b, 0.0, &mut c)?;
        for j in 0..k { out[j * n + rows.start..j * n + rows.end].copy_from_slice(&c[j * nr..(j + 1) * nr]); }
    }

    // The deflated columns stand as they are.
    let mut kept = vec![false; n];
    for &i in &keep { kept[i] = true; }
    let mut values = Vec::with_capacity(n);
    let mut merged = vec![0.0f64; n * n];
    for (j, root) in roots.iter().enumerate() {
        values.push(ds[root.origin] + root.mu);
        merged[j * n..(j + 1) * n].copy_from_slice(&out[j * n..(j + 1) * n]);
    }
    let mut at = k;
    for i in 0..n {
        if kept[i] { continue; }
        values.push(d[i]);
        merged[at * n..(at + 1) * n].copy_from_slice(&q[i * n..(i + 1) * n]);
        at += 1;
    }
    Ok(sorted_columns(values, merged, n))
}

/// Values ascending, with their columns.
fn sorted_columns(values: Vec<f64>, q: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let mut out = vec![0.0f64; n * n];
    for (new, &old) in order.iter().enumerate() {
        out[new * n..(new + 1) * n].copy_from_slice(&q[old * n..(old + 1) * n]);
    }
    (order.iter().map(|&i| values[i]).collect(), out)
}

/// `(values ascending, eigenvectors column-major)` of the tridiagonal `d`, `e`.
fn recurse(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    if n <= LEAF {
        let mut id = vec![0.0f64; n * n];
        for i in 0..n { id[i * n + i] = 1.0; }
        let (vals, vecs) = tridiag_eigen(d.to_vec(), e.to_vec(), Some((id, n)))?;
        let vecs = vecs.expect("vectors requested");
        let order: Vec<usize> = (0..n).rev().collect();
        let mut q = vec![0.0f64; n * n];
        for (new, &old) in order.iter().enumerate() {
            q[new * n..(new + 1) * n].copy_from_slice(&vecs[old * n..(old + 1) * n]);
        }
        return Ok((order.iter().map(|&i| vals[i]).collect(), q));
    }
    // T = diag(T₁', T₂') + |β| v vᵀ, with v = (0…0, 1, sign β, 0…0) across the tear.
    let n1 = n / 2;
    let n2 = n - n1;
    let beta = e[n1 - 1];
    let (mut d1, mut d2) = (d[..n1].to_vec(), d[n1..].to_vec());
    d1[n1 - 1] -= beta.abs();
    d2[0] -= beta.abs();
    let ((v1, q1), (v2, q2)) = (recurse(&d1, &e[..n1 - 1])?, recurse(&d2, &e[n1..])?);

    let mut q = vec![0.0f64; n * n];
    for c in 0..n1 { q[c * n..c * n + n1].copy_from_slice(&q1[c * n1..(c + 1) * n1]); }
    for c in 0..n2 { q[(n1 + c) * n + n1..(n1 + c + 1) * n].copy_from_slice(&q2[c * n2..(c + 1) * n2]); }
    let mut z = Vec::with_capacity(n);
    let root2 = std::f64::consts::FRAC_1_SQRT_2;
    for c in 0..n1 { z.push(q1[c * n1 + n1 - 1] * root2); }
    let sign = 1.0f64.copysign(beta);
    for c in 0..n2 { z.push(q2[c * n2] * root2 * sign); }
    let mut vals = v1;
    vals.extend(v2);
    merge(vals, z, 2.0 * beta.abs(), q, n, n1)
}

/// Eigenvalues descending and eigenvectors of the tridiagonal `d`, `e`, the order
/// [`tridiag_eigen`] gives.
pub(crate) fn solve(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    // Scaled to order one, so that no squared quantity can overflow.
    let scale = d.iter().chain(e).fold(0.0f64, |a, x| a.max(x.abs()));
    let s = if scale > 0.0 { scale } else { 1.0 };
    let dn: Vec<f64> = d.iter().map(|x| x / s).collect();
    let en: Vec<f64> = e.iter().map(|x| x / s).collect();
    let (vals, q) = recurse(&dn, &en)?;
    let mut flipped = vec![0.0f64; n * n];
    for c in 0..n { flipped[c * n..(c + 1) * n].copy_from_slice(&q[(n - 1 - c) * n..(n - c) * n]); }
    Ok((vals.iter().rev().map(|x| x * s).collect(), flipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residuals(d: &[f64], e: &[f64], vals: &[f64], q: &[f64]) -> (f64, f64) {
        let n = d.len();
        let mut res = 0.0f64;
        for c in 0..n {
            for r in 0..n {
                let mut t = d[r] * q[c * n + r];
                if r > 0 { t += e[r - 1] * q[c * n + r - 1]; }
                if r + 1 < n { t += e[r] * q[c * n + r + 1]; }
                res = res.max((t - vals[c] * q[c * n + r]).abs());
            }
        }
        let mut orth = 0.0f64;
        for a in 0..n {
            for b in 0..=a {
                let dot: f64 = (0..n).map(|r| q[a * n + r] * q[b * n + r]).sum();
                orth = orth.max((dot - if a == b { 1.0 } else { 0.0 }).abs());
            }
        }
        (res, orth)
    }

    fn check(name: &str, d: &[f64], e: &[f64]) {
        let (vals, q) = solve(d, e).unwrap();
        let scale = d.iter().chain(e).fold(0.0f64, |a, x| a.max(x.abs())).max(1e-300);
        let (res, orth) = residuals(d, e, &vals, &q);
        let n = d.len() as f64;
        assert!(res <= 8.0 * n * f64::EPSILON * scale, "{name}: residual {res}, scale {scale}");
        assert!(orth <= 8.0 * n * f64::EPSILON, "{name}: orthogonality {orth}");
        assert!(vals.windows(2).all(|w| w[0] >= w[1]), "{name}: not descending");
        let down = |v: &[f64]| v.iter().map(|x| x / scale).collect::<Vec<f64>>();
        let (qr, _) = tridiag_eigen(down(d), down(e), None).unwrap();
        for (a, b) in vals.iter().zip(&qr) {
            assert!((a / scale - b).abs() <= 8.0 * n * f64::EPSILON, "{name}: {a} against QR's {b}");
        }
    }

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
        (0..n).map(|_| {
            x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn hard_tridiagonals() {
        for n in [33usize, 64, 65, 200, 513] {
            check("random", &noise(n, 1), &noise(n - 1, 2));
            check("laplacian", &vec![2.0; n], &vec![-1.0; n - 1]);
            check("identity", &vec![1.0; n], &vec![0.0; n - 1]);
            check("zero", &vec![0.0; n], &vec![0.0; n - 1]);
            check("constant off", &vec![0.0; n], &vec![1.0; n - 1]);
            // Wilkinson: pairs of eigenvalues agree to many digits
            let w: Vec<f64> = (0..n).map(|i| (i as f64 - (n as f64 - 1.0) / 2.0).abs()).collect();
            check("wilkinson", &w, &vec![1.0; n - 1]);
            // graded, and one huge off-diagonal among tiny ones
            let g: Vec<f64> = (0..n).map(|i| 10f64.powf(-(i as f64) * 16.0 / n as f64)).collect();
            check("graded", &g, &g[1..]);
            let mut spike = vec![1e-10; n - 1];
            spike[n / 2 - 1] = 1e10;
            check("spike at the tear", &noise(n, 3), &spike);
            let negative: Vec<f64> = noise(n - 1, 4).iter().map(|x| -x.abs()).collect();
            check("negative off", &noise(n, 5), &negative);
            // glued: blocks with nearly equal spectra, split by tiny entries
            let glued: Vec<f64> = (0..n - 1).map(|i| if i % 40 == 39 { 1e-14 } else { 1.0 }).collect();
            check("glued", &vec![1.0; n], &glued);
        }
    }

    #[test]
    fn scale_does_not_matter() {
        let (d, e) = (noise(100, 7), noise(99, 8));
        for s in [1e-200, 1e200] {
            let (ds, es): (Vec<f64>, Vec<f64>) = (d.iter().map(|x| x * s).collect(), e.iter().map(|x| x * s).collect());
            check("scaled", &ds, &es);
        }
    }
}
