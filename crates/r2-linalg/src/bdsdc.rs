//! Singular vectors of an upper bidiagonal matrix by divide and conquer, the role of
//! LAPACK's `dbdsdc` (Gu and Eisenstat), as `stedc` is `dstedc`.
//!
//! Rotations from the right clear the column above the tear, which leaves two square
//! bidiagonals and the row between them:
//!
//! ```text
//!     B G = [ B₁  0  0  ]      diag(U₁, 1, U₂)ᵀ B G diag(V₁, 1, V₂)  is, with the
//!           [ aᵀ  b  c· ]      torn row and column moved first, the arrow matrix
//!           [ 0   0  B₂ ]      M = e₁ zᵀ + diag(d),  d₁ = 0
//! ```
//!
//! Then `MᵀM = D² + z zᵀ`, whose values are the roots of the same secular equation
//! `stedc` solves, and the vectors of `M` are explicit. The roots are solved in `σ²`
//! with each pole's distance computed as `(dᵢ − dₒ)(dᵢ + dₒ)`, so none is lost in a
//! subtraction of squares. `U` and `V` then take one matrix product each.
//!
//! This gives the singular values to absolute accuracy, as LAPACK's does; `bdsqr`
//! keeps its better relative accuracy for the values alone.

use rayon::prelude::*;

use crate::bdsqr;
use crate::stedc::solve_root;
use crate::LinalgError;

const EPS: f64 = f64::EPSILON / 2.0;
const LEAF: usize = 32;

/// `c·x + s·y` and `c·y − s·x` on two columns of a column-major n x n matrix.
fn rotate_columns(q: &mut [f64], n: usize, a: usize, b: usize, c: f64, s: f64) {
    debug_assert_ne!(a, b);
    let (lo, hi) = q.split_at_mut(a.max(b) * n);
    let (x, y) = if a < b {
        (&mut lo[a * n..(a + 1) * n], &mut hi[..n])
    } else {
        (&mut hi[..n], &mut lo[b * n..(b + 1) * n])
    };
    for (p, r) in x.iter_mut().zip(y.iter_mut()) {
        let (xa, yb) = (*p, *r);
        *p = c * xa + s * yb;
        *r = c * yb - s * xa;
    }
}

/// The SVD of `M = e₀ zᵀ + diag(d)`, `d[0] = 0`, applied to the bases `u` and `v`:
/// `(values, u·U_M, v·V_M)`, unsorted.
fn merge(mut d: Vec<f64>, mut z: Vec<f64>, mut u: Vec<f64>, mut v: Vec<f64>, n: usize)
    -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError>
{
    let mut perm: Vec<usize> = (1..n).collect();
    perm.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let dmax = d.iter().fold(0.0f64, |a, x| a.max(*x));
    let zmax = z.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let tol = 8.0 * EPS * dmax.max(zmax);
    if tol == 0.0 { return Ok((d, u, v)); }      // M is zero
    if z[0].abs() <= tol { z[0] = tol.copysign(z[0]); }

    let mut keep = vec![0usize];
    let mut prev: Option<usize> = None;
    for &i in &perm {
        if z[i].abs() <= tol {
            continue;            // the column stands: σ = dᵢ, with eᵢ on both sides
        }
        if d[i] <= tol {
            // Row i is zero. A rotation of V's columns folds zᵢ into z₀, and σ = 0.
            let tau = z[0].hypot(z[i]);
            let (c, s) = (z[0] / tau, z[i] / tau);
            rotate_columns(&mut v, n, 0, i, c, s);
            z[0] = tau;
            z[i] = 0.0;
            d[i] = 0.0;
            continue;
        }
        if let Some(p) = prev {
            if d[i] - d[p] <= tol {
                // Nearly equal: one rotation on both sides, and p's z is gone.
                let tau = z[i].hypot(z[p]);
                let (c, s) = (z[i] / tau, -z[p] / tau);
                rotate_columns(&mut u, n, p, i, c, s);
                rotate_columns(&mut v, n, p, i, c, s);
                z[i] = tau;
                z[p] = 0.0;
                prev = Some(i);
                continue;
            }
            keep.push(p);
        }
        prev = Some(i);
    }
    if let Some(p) = prev { keep.push(p); }

    let k = keep.len();
    let ds: Vec<f64> = keep.iter().map(|&i| d[i]).collect();
    let zs: Vec<f64> = keep.iter().map(|&i| z[i]).collect();
    let z2: Vec<f64> = zs.iter().map(|x| x * x).collect();
    let norm2: f64 = z2.iter().sum();
    let pole = |i: usize, o: usize| (ds[i] - ds[o]) * (ds[i] + ds[o]);

    // dist[j][i] = dᵢ² − σⱼ², to full relative accuracy.
    let mut dist = vec![0.0f64; k * k];
    let roots: Vec<_> = dist.par_chunks_mut(k).enumerate().map(|(j, col)| {
        let root = solve_root(&pole, &z2, 1.0, norm2, j, col);
        for (i, x) in col.iter_mut().enumerate() { *x = pole(i, root.origin) - root.mu; }
        root
    }).collect();
    let sigmas: Vec<f64> = roots.iter().map(|r| (ds[r.origin] * ds[r.origin] + r.mu).sqrt()).collect();

    // Gu and Eisenstat: the z for which these σ are exact.
    let zhat: Vec<f64> = (0..k).into_par_iter().map(|i| {
        let mut w = dist[i * k + i];
        for j in 0..k {
            if j != i { w *= dist[j * k + i] / pole(i, j); }
        }
        (-w).max(0.0).sqrt().copysign(zs[i])
    }).collect();

    // v ∝ ẑᵢ/(dᵢ² − σ²); u ∝ (−1, dᵢẑᵢ/(dᵢ² − σ²)…).
    let mut um = vec![0.0f64; k * k];
    let mut vm = vec![0.0f64; k * k];
    um.par_chunks_mut(k).zip(vm.par_chunks_mut(k)).enumerate().for_each(|(j, (ucol, vcol))| {
        let col = &dist[j * k..(j + 1) * k];
        let (mut nu, mut nv) = (1.0f64, 0.0f64);
        ucol[0] = -1.0;
        vcol[0] = zhat[0] / col[0];
        nv += vcol[0] * vcol[0];
        for i in 1..k {
            let x = zhat[i] / col[i];
            vcol[i] = x;
            nv += x * x;
            let y = ds[i] * x;
            ucol[i] = y;
            nu += y * y;
        }
        let (nu, nv) = (nu.sqrt(), nv.sqrt());
        for x in ucol.iter_mut() { *x /= nu; }
        for x in vcol.iter_mut() { *x /= nv; }
    });

    // Bases of the kept columns times these, and the deflated columns as they are.
    let times = |q: &[f64], m: &[f64]| -> Result<Vec<f64>, LinalgError> {
        let mut kept = vec![0.0f64; n * k];
        for (c, &i) in keep.iter().enumerate() { kept[c * n..(c + 1) * n].copy_from_slice(&q[i * n..(i + 1) * n]); }
        let mut out = vec![0.0f64; n * k];
        crate::dgemm(n, k, k, 1.0, &kept, m, 0.0, &mut out)?;
        Ok(out)
    };
    let (mut new_u, mut new_v) = (times(&u, &um)?, times(&v, &vm)?);
    let mut is_kept = vec![false; n];
    for &i in &keep { is_kept[i] = true; }
    let mut values = sigmas;
    new_u.reserve(n * (n - k));
    new_v.reserve(n * (n - k));
    for i in 0..n {
        if is_kept[i] { continue; }
        values.push(d[i]);
        new_u.extend_from_slice(&u[i * n..(i + 1) * n]);
        new_v.extend_from_slice(&v[i * n..(i + 1) * n]);
    }
    Ok((values, new_u, new_v))
}

fn recurse(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    if n <= LEAF { return bdsqr::by_qr(d.to_vec(), e.to_vec()); }
    let n1 = (n - 1) / 2;
    let n2 = n - 1 - n1;

    // The first half and the torn column make a matrix one column wider than tall.
    // Rotations of each column with the torn one clear it, chasing e[n1 − 1] up, and
    // leave the first half square and the torn row with entries a under it.
    let mut d1 = d[..n1].to_vec();
    let mut e1 = e[..n1 - 1].to_vec();
    let mut a = vec![0.0f64; n1];
    let mut b = d[n1];
    let mut rots = Vec::with_capacity(n1);
    let mut bulge = e[n1 - 1];
    for i in (0..n1).rev() {
        let r = d1[i].hypot(bulge);
        let (cg, sg) = if r == 0.0 { (1.0, 0.0) } else { (d1[i] / r, bulge / r) };
        d1[i] = r;
        a[i] = sg * b;
        b *= cg;
        if i > 0 {
            bulge = -sg * e1[i - 1];
            e1[i - 1] *= cg;
        }
        rots.push((i, cg, sg));
    }
    let (s1, u1, v1) = recurse(&d1, &e1)?;
    let (s2, u2, v2) = recurse(&d[n1 + 1..], &e[n1 + 1..])?;
    let c = e[n1];

    // Bases: column 0 is the torn row and column, then each half's columns.
    let (mut u, mut v) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
    u[n1] = 1.0;
    v[n1] = 1.0;
    for p in 0..n1 {
        u[(1 + p) * n..(1 + p) * n + n1].copy_from_slice(&u1[p * n1..(p + 1) * n1]);
        v[(1 + p) * n..(1 + p) * n + n1].copy_from_slice(&v1[p * n1..(p + 1) * n1]);
    }
    for p in 0..n2 {
        let col = (1 + n1 + p) * n + n1 + 1;
        u[col..col + n2].copy_from_slice(&u2[p * n2..(p + 1) * n2]);
        v[col..col + n2].copy_from_slice(&v2[p * n2..(p + 1) * n2]);
    }
    // V_B = G₁ G₂ … · V: the last rotation found acts first, on rows i and n1.
    v.par_chunks_mut(n).for_each(|col| {
        for &(i, cg, sg) in rots.iter().rev() {
            let (x, y) = (col[i], col[n1]);
            col[i] = cg * x - sg * y;
            col[n1] = sg * x + cg * y;
        }
    });

    let mut dm = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    dm.push(0.0);
    z.push(b);
    for p in 0..n1 {
        dm.push(s1[p]);
        z.push(crate::simd::dot(&a, &v1[p * n1..(p + 1) * n1]));
    }
    for p in 0..n2 { dm.push(s2[p]); z.push(c * v2[p * n2]); }
    merge(dm, z, u, v, n)
}

/// `(σ, U, V)` of the bidiagonal, σ descending and non-negative.
pub(crate) fn solve(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    let scale = d.iter().chain(e).fold(0.0f64, |a, x| a.max(x.abs()));
    let s = if scale > 0.0 { scale } else { 1.0 };
    let dn: Vec<f64> = d.iter().map(|x| x / s).collect();
    let en: Vec<f64> = e.iter().map(|x| x / s).collect();
    let (vals, u, v) = recurse(&dn, &en)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]));
    let pick = |q: &[f64]| {
        let mut out = vec![0.0f64; n * n];
        for (new, &old) in order.iter().enumerate() {
            out[new * n..(new + 1) * n].copy_from_slice(&q[old * n..(old + 1) * n]);
        }
        out
    };
    Ok((order.iter().map(|&i| vals[i] * s).collect(), pick(&u), pick(&v)))
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

    fn check(name: &str, d: &[f64], e: &[f64]) {
        let n = d.len();
        let (s, u, v) = solve(d, e).unwrap();
        let scale = d.iter().chain(e).fold(0.0f64, |a, x| a.max(x.abs())).max(f64::MIN_POSITIVE);
        let limit = 16.0 * (n as f64) * f64::EPSILON;
        // B·vⱼ = σⱼ·uⱼ, which is cheaper than forming the whole product
        for j in 0..n {
            for r in 0..n {
                let mut t = d[r] * v[j * n + r];
                if r + 1 < n { t += e[r] * v[j * n + r + 1]; }
                let got = (t - s[j] * u[j * n + r]).abs() / scale;
                assert!(got <= limit, "{name} n={n}: residual {got} at row {r}, column {j}");
            }
        }
        for (side, q) in [("U", &u), ("V", &v)] {
            for a in 0..n {
                for b in 0..=a {
                    let dot: f64 = (0..n).map(|r| q[a * n + r] * q[b * n + r]).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() <= limit, "{name} n={n}: {side} columns {a}, {b}: {dot}");
                }
            }
        }
        assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|x| *x >= 0.0), "{name}: order");
        let qr = bdsqr::values(d.to_vec(), e.to_vec()).unwrap();
        for (a, b) in s.iter().zip(&qr) {
            assert!((a - b).abs() <= limit * scale, "{name} n={n}: {a} against QR's {b}");
        }
    }

    #[test]
    fn hard_bidiagonals() {
        for n in [33usize, 64, 65, 130, 257, 400] {
            check("random", &noise(n, 1), &noise(n - 1, 2));
            check("ones", &vec![1.0; n], &vec![1.0; n - 1]);
            check("identity", &vec![1.0; n], &vec![0.0; n - 1]);
            check("zero", &vec![0.0; n], &vec![0.0; n - 1]);
            check("diagonal zero", &vec![0.0; n], &vec![1.0; n - 1]);
            let g: Vec<f64> = (0..n).map(|i| 10f64.powf(-(i as f64) * 16.0 / n as f64)).collect();
            check("graded", &g, &g[1..]);
            let up: Vec<f64> = g.iter().rev().copied().collect();
            check("graded up", &up, &up[..n - 1]);
            let mut holes = noise(n, 3);
            for i in (0..n).step_by(7) { holes[i] = 0.0; }
            check("zeros on the diagonal", &holes, &noise(n - 1, 4));
            let cut: Vec<f64> = noise(n - 1, 5).iter().enumerate().map(|(i, x)| if i % 31 == 30 { 0.0 } else { *x }).collect();
            check("split", &noise(n, 6), &cut);
            let tear = (n - 1) / 2;
            let mut spike = vec![1e-12; n - 1];
            spike[tear] = 1.0;
            spike[tear - 1] = 1.0;
            check("spike at the tear", &vec![1e-12; n], &spike);
        }
    }
}
