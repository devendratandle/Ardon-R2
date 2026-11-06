//! Singular values and vectors of an upper bidiagonal matrix: LAPACK's `dbdsqr`.
//!
//! Implicit QR on the bidiagonal itself, with the zero-shift sweeps of Demmel and
//! Kahan (1990), which compute every singular value to high RELATIVE accuracy.
//! Going through `BᵀB` squares the condition number, so a singular vector of a small
//! value carried the rounding of the largest. Here each rotation is orthogonal, so
//! `U` and `V` stay orthonormal to rounding whatever the values are, zero included.
//!
//! The rotations go through the same blocked, batched application as the symmetric
//! QR iteration (`symeig::Rotations`).

use crate::symeig::Rotations;
use crate::LinalgError;

/// Unit roundoff, LAPACK's `dlamch('E')`.
const EPS: f64 = f64::EPSILON / 2.0;
const UNFL: f64 = f64::MIN_POSITIVE;
/// From this size the vectors come from `bdsdc`.
const DC_MIN: usize = 64;

/// `(c, s, r)` with `[c s; -s c] [f; g] = [r; 0]`, `c >= 0` (`dlartg`).
#[inline]
fn lartg(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 { return (1.0, 0.0, f); }
    if f == 0.0 { return (0.0, 1.0f64.copysign(g), g.abs()); }
    let (fa, ga) = (f.abs(), g.abs());
    // inside this range the squares neither overflow nor lose their digits
    const LO: f64 = 1.5e-154;
    const HI: f64 = 4.7e153;
    let d = if fa > LO && fa < HI && ga > LO && ga < HI { (f * f + g * g).sqrt() } else { f.hypot(g) };
    let r = d.copysign(f);
    (fa / d, g / r, r)
}

/// Singular values of `[f g; 0 h]`, smaller first (`dlas2`).
fn las2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let (fhmn, fhmx) = (fa.min(ha), fa.max(ha));
    if fhmn == 0.0 {
        let large = fhmx.max(ga);
        let max = if fhmx == 0.0 { ga } else { large * (1.0 + (fhmx.min(ga) / large).powi(2)).sqrt() };
        return (0.0, max);
    }
    let as_ = 1.0 + fhmn / fhmx;
    let at = (fhmx - fhmn) / fhmx;
    if ga < fhmx {
        let au = (ga / fhmx).powi(2);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        return (fhmn * c, fhmx / c);
    }
    let au = fhmx / ga;
    if au == 0.0 { return (fhmn * fhmx / ga, ga); }
    let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
    (2.0 * (fhmn * c * au), ga / (c + c))
}

/// One 2 x 2 bidiagonal block, solved exactly.
struct Svd2 { ssmin: f64, ssmax: f64, snr: f64, csr: f64, snl: f64, csl: f64 }

/// SVD of `[f g; 0 h]` (`dlasv2`): accurate to a few ulps even when `f` or `h`
/// is far below `g`.
fn lasv2(f: f64, g: f64, h: f64) -> Svd2 {
    let sign = |a: f64, b: f64| a.abs().copysign(b);
    let (mut ft, mut fa, mut ht, mut ha) = (f, f.abs(), h, h.abs());
    let swap = ha > fa;
    let mut pmax = if swap { 3 } else { 1 };
    if swap { std::mem::swap(&mut ft, &mut ht); std::mem::swap(&mut fa, &mut ha); }
    let (gt, ga) = (g, g.abs());
    let (ssmin, ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        (ssmin, ssmax, clt, crt, slt, srt) = (ha, fa, 1.0, 1.0, 0.0, 0.0);
    } else if ga > fa && fa / ga < EPS {
        // g is so large that the singular values are g and a small remainder
        pmax = 2;
        ssmax = ga;
        ssmin = if ha > 1.0 { fa / (ga / ha) } else { fa / ga * ha };
        (clt, slt, srt, crt) = (1.0, ht / gt, 1.0, ft / gt);
    } else {
        if ga > fa { pmax = 2; }
        let d = fa - ha;
        let l = if d == fa { 1.0 } else { d / fa };
        let m = gt / ft;
        let t = 2.0 - l;
        let (mm, tt) = (m * m, t * t);
        let s = (tt + mm).sqrt();
        let r = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
        let a = 0.5 * (s + r);
        ssmin = ha / a;
        ssmax = fa * a;
        let t = if mm == 0.0 {
            if l == 0.0 { sign(2.0, ft) * sign(1.0, gt) } else { gt / sign(d, ft) + m / t }
        } else {
            (m / (s + t) + m / (r + l)) * (1.0 + a)
        };
        let l = (t * t + 4.0).sqrt();
        crt = 2.0 / l;
        srt = t / l;
        clt = (crt + srt * m) / a;
        slt = (ht / ft) * srt / a;
    }
    let (csl, snl, csr, snr) = if swap { (srt, crt, slt, clt) } else { (clt, slt, crt, srt) };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    Svd2 {
        ssmin: sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h)),
        ssmax: sign(ssmax, tsign),
        snr, csr, snl, csl,
    }
}

/// Where the rotations of a sweep go; `None` for values only.
struct Sides { u: Rotations, v: Rotations }

/// Diagonalise the upper bidiagonal `d`, `e` (`e.len() + 1 == d.len()`). Returns the
/// singular values, unsorted and non-negative, in `d` as given back, with `d` and
/// `e` overwritten.
fn iterate(d: &mut [f64], e: &mut [f64], mut sides: Option<&mut Sides>) -> Result<(), LinalgError> {
    let n = d.len();
    if n < 2 { return Ok(()); }
    let tolmul = (EPS.powf(-0.125)).clamp(10.0, 100.0);
    let tol = tolmul * EPS;

    // A lower bound on the smallest singular value, and from it the absolute
    // threshold below which an entry is negligible.
    let mut mu = d[0].abs();
    let mut sminoa = mu;
    for i in 1..n {
        if sminoa == 0.0 { break; }
        mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
        sminoa = sminoa.min(mu);
    }
    let sminoa = sminoa / (n as f64).sqrt();
    let maxit = 6.0 * (n as f64) * (n as f64);
    let thresh = (tol * sminoa).max(maxit * UNFL);

    let mut iters = 0.0f64;
    let mut m = n - 1;          // the last row of the active block
    let (mut old_ll, mut old_m) = (usize::MAX, usize::MAX);
    let mut idir = 0;
    'blocks: while m > 0 {
        if iters > maxit {
            return Err(LinalgError::InvalidShape("SVD: bidiagonal QR failed to converge".into()));
        }

        // Find the block: the diagonal block ending at m with no negligible e.
        let mut smax = d[m].abs();
        let mut ll = m;
        loop {
            let i = ll - 1;
            if e[i].abs() <= thresh {
                e[i] = 0.0;
                if ll == m { m -= 1; continue 'blocks; }
                break;
            }
            smax = smax.max(d[i].abs()).max(e[i].abs());
            ll = i;
            if ll == 0 { break; }
        }

        if ll + 1 == m {
            // 2 x 2: exactly
            let s = lasv2(d[m - 1], e[m - 1], d[m]);
            d[m - 1] = s.ssmax;
            e[m - 1] = 0.0;
            d[m] = s.ssmin;
            if let Some(q) = sides.as_deref_mut() {
                q.v.push(m - 1, s.csr, s.snr);
                q.u.push(m - 1, s.csl, s.snl);
            }
            if m < 2 { break; }
            m -= 2;
            continue;
        }

        // A new block picks the direction that chases towards its smaller end.
        if ll != old_ll || m != old_m {
            idir = if d[ll].abs() >= d[m].abs() { 1 } else { 2 };
        }

        // Convergence, by the criterion that keeps relative accuracy; `smin` is a
        // lower bound on the block's smallest singular value.
        let mut smin;
        if idir == 1 {
            if e[m - 1].abs() <= tol * d[m].abs() { e[m - 1] = 0.0; continue; }
            let mut mu = d[ll].abs();
            smin = mu;
            for i in ll..m {
                if e[i].abs() <= tol * mu { e[i] = 0.0; continue 'blocks; }
                mu = d[i + 1].abs() * (mu / (mu + e[i].abs()));
                smin = smin.min(mu);
            }
        } else {
            if e[ll].abs() <= tol * d[ll].abs() { e[ll] = 0.0; continue; }
            let mut mu = d[m].abs();
            smin = mu;
            for i in (ll..m).rev() {
                if e[i].abs() <= tol * mu { e[i] = 0.0; continue 'blocks; }
                mu = d[i].abs() * (mu / (mu + e[i].abs()));
                smin = smin.min(mu);
            }
        }
        old_ll = ll;
        old_m = m;

        // The shift, or zero when a shift would cost relative accuracy.
        let mut shift = 0.0;
        if (n as f64) * tol * (smin / smax) > EPS.max(0.01 * tol) {
            let sll;
            if idir == 1 {
                sll = d[ll].abs();
                shift = las2(d[m - 1], e[m - 1], d[m]).0;
            } else {
                sll = d[m].abs();
                shift = las2(d[ll], e[ll], d[ll + 1]).0;
            }
            if sll > 0.0 && (shift / sll).powi(2) < EPS { shift = 0.0; }
        }
        iters += (m - ll) as f64;

        match (idir, shift == 0.0) {
            (1, true) => {
                let (mut cs, mut oldcs, mut oldsn) = (1.0, 1.0, 0.0);
                for i in ll..m {
                    let (c, s, r) = lartg(d[i] * cs, e[i]);
                    cs = c;
                    if i > ll { e[i - 1] = oldsn * r; }
                    let (c2, s2, di) = lartg(oldcs * r, d[i + 1] * s);
                    (oldcs, oldsn) = (c2, s2);
                    d[i] = di;
                    if let Some(q) = sides.as_deref_mut() { q.v.push(i, cs, s); q.u.push(i, oldcs, oldsn); }
                }
                let h = d[m] * cs;
                d[m] = h * oldcs;
                e[m - 1] = h * oldsn;
                if e[m - 1].abs() <= thresh { e[m - 1] = 0.0; }
            }
            (2, true) => {
                let (mut cs, mut oldcs, mut oldsn) = (1.0, 1.0, 0.0);
                for i in (ll + 1..=m).rev() {
                    let (c, s, r) = lartg(d[i] * cs, e[i - 1]);
                    cs = c;
                    if i < m { e[i] = oldsn * r; }
                    let (c2, s2, di) = lartg(oldcs * r, d[i - 1] * s);
                    (oldcs, oldsn) = (c2, s2);
                    d[i] = di;
                    if let Some(q) = sides.as_deref_mut() { q.u.push(i - 1, cs, -s); q.v.push(i - 1, oldcs, -oldsn); }
                }
                let h = d[ll] * cs;
                d[ll] = h * oldcs;
                e[ll] = h * oldsn;
                if e[ll].abs() <= thresh { e[ll] = 0.0; }
            }
            (1, false) => {
                let mut f = (d[ll].abs() - shift) * (1.0f64.copysign(d[ll]) + shift / d[ll]);
                let mut g = e[ll];
                for i in ll..m {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i > ll { e[i - 1] = r; }
                    f = cosr * d[i] + sinr * e[i];
                    e[i] = cosr * e[i] - sinr * d[i];
                    g = sinr * d[i + 1];
                    d[i + 1] *= cosr;
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if i + 1 < m {
                        g = sinl * e[i + 1];
                        e[i + 1] *= cosl;
                    }
                    if let Some(q) = sides.as_deref_mut() { q.v.push(i, cosr, sinr); q.u.push(i, cosl, sinl); }
                }
                e[m - 1] = f;
                if e[m - 1].abs() <= thresh { e[m - 1] = 0.0; }
            }
            _ => {
                let mut f = (d[m].abs() - shift) * (1.0f64.copysign(d[m]) + shift / d[m]);
                let mut g = e[m - 1];
                for i in (ll + 1..=m).rev() {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i < m { e[i] = r; }
                    f = cosr * d[i] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                    g = sinr * d[i - 1];
                    d[i - 1] *= cosr;
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if i > ll + 1 {
                        g = sinl * e[i - 2];
                        e[i - 2] *= cosl;
                    }
                    if let Some(q) = sides.as_deref_mut() { q.u.push(i - 1, cosr, -sinr); q.v.push(i - 1, cosl, -sinl); }
                }
                e[ll] = f;
                if e[ll].abs() <= thresh { e[ll] = 0.0; }
            }
        }
    }
    Ok(())
}

/// Singular values of the upper bidiagonal `d`, `e`, descending.
pub(crate) fn values(mut d: Vec<f64>, mut e: Vec<f64>) -> Result<Vec<f64>, LinalgError> {
    iterate(&mut d, &mut e, None)?;
    for x in &mut d { *x = x.abs(); }
    d.sort_by(|a, b| b.total_cmp(a));
    Ok(d)
}

/// The SVD `B = U diag(σ) Vᵀ` of the upper bidiagonal `d`, `e`: σ descending, `U` and
/// `V` column-major n x n, and the largest-magnitude entry of each column of `V`
/// positive, the sign this library has always given.
pub(crate) fn full(d: Vec<f64>, e: Vec<f64>) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    let (sigma, mut u, mut v) = if n >= DC_MIN {
        // Divide and conquer for the vectors; these sweeps for the values, whose relative
        // accuracy it does not have. The two agree to rounding of the largest.
        let (_, u, v) = crate::bdsdc::solve(&d, &e)?;
        (values(d, e)?, u, v)
    } else {
        by_qr(d, e)?
    };
    for k in 0..n {
        let vk = &mut v[k * n..(k + 1) * n];
        let largest = vk.iter().copied().fold(0.0f64, |a, x| if x.abs() > a.abs() { x } else { a });
        if largest < 0.0 {
            for x in vk.iter_mut() { *x = -*x; }
            for x in &mut u[k * n..(k + 1) * n] { *x = -*x; }
        }
    }
    Ok((sigma, u, v))
}

/// [`full`] by the QR sweeps alone, whatever the size and with no sign convention:
/// σ descending, and `u_k σ_k v_kᵀ` summing to the bidiagonal.
pub(crate) fn by_qr(mut d: Vec<f64>, mut e: Vec<f64>) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinalgError> {
    let n = d.len();
    let mut sides = Sides { u: Rotations::identity(n), v: Rotations::identity(n) };
    iterate(&mut d, &mut e, Some(&mut sides))?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].abs().total_cmp(&d[i].abs()));
    let u = sides.u.into_columns(&order);
    let mut v = sides.v.into_columns(&order);
    let mut sigma = Vec::with_capacity(n);
    for (k, &old) in order.iter().enumerate() {
        sigma.push(d[old].abs());
        if d[old] < 0.0 { for x in &mut v[k * n..(k + 1) * n] { *x = -*x; } }
    }
    Ok((sigma, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `U diag(σ) Vᵀ` against the bidiagonal, and both bases against the identity.
    fn check(d: &[f64], e: &[f64], tol: f64) -> Vec<f64> {
        let n = d.len();
        let (s, u, v) = full(d.to_vec(), e.to_vec()).unwrap();
        let scale = d.iter().chain(e).fold(0.0f64, |a, x| a.max(x.abs())).max(f64::MIN_POSITIVE);
        for r in 0..n {
            for c in 0..n {
                let b = if r == c { d[r] } else if c == r + 1 { e[r] } else { 0.0 };
                let got: f64 = (0..n).map(|k| u[k * n + r] * s[k] * v[k * n + c]).sum();
                assert!((got - b).abs() <= tol * scale, "B[{r},{c}]: {got} against {b}");
            }
        }
        for q in [&u, &v] {
            for a in 0..n {
                for b in 0..n {
                    let dot: f64 = (0..n).map(|r| q[a * n + r] * q[b * n + r]).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() <= tol, "columns {a}, {b}: {dot}");
                }
            }
        }
        assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|x| *x >= 0.0), "{s:?}");
        let only = values(d.to_vec(), e.to_vec()).unwrap();
        assert_eq!(only, s, "values alone must be the same sweeps");
        s
    }

    #[test]
    fn one_and_two_rows() {
        assert_eq!(check(&[-3.0], &[], 1e-15), [3.0]);
        let s = check(&[3.0, 2.0], &[4.0], 1e-15);
        // AᵀA has trace 29 and determinant 36
        assert!((s[0] * s[0] + s[1] * s[1] - 29.0).abs() < 1e-13);
        assert!((s[0] * s[1] - 6.0).abs() < 1e-13);
    }

    #[test]
    fn signs_zeros_and_splits() {
        check(&[1.0, -2.0, 3.0, -4.0, 5.0], &[1.0, -1.0, 1.0, -1.0], 1e-14);
        check(&[0.0, 0.0, 0.0], &[0.0, 0.0], 1e-15);
        check(&[1.0, 0.0, 1.0, 0.0], &[1.0, 1.0, 1.0], 1e-14);
        check(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 1.0], 1e-14);
        let s = check(&[5.0, 4.0, 3.0], &[0.0, 0.0], 1e-15);
        assert_eq!(s, [5.0, 4.0, 3.0]);
    }

    /// Both directions and both kinds of sweep: graded each way, and uniform.
    #[test]
    fn graded_either_way_and_uniform() {
        for n in [3usize, 4, 7, 30, 100] {
            let down: Vec<f64> = (0..n).map(|i| 10f64.powf(-(i as f64) * 12.0 / n as f64)).collect();
            let up: Vec<f64> = down.iter().rev().copied().collect();
            let e: Vec<f64> = (0..n - 1).map(|i| 0.5 * down[i + 1]).collect();
            check(&down, &e, 1e-13);
            let e: Vec<f64> = (0..n - 1).map(|i| 0.5 * up[i]).collect();
            check(&up, &e, 1e-13);
            check(&vec![1.0; n], &vec![1.0; n - 1], 1e-12);
        }
    }

    /// The point of the routine: a matrix with known tiny singular values keeps
    /// their RELATIVE accuracy. d = 1, e = 0 scaled so det is exact: the smallest
    /// singular value of this block is a product over the largest ones.
    #[test]
    fn a_tiny_singular_value_keeps_its_digits() {
        let (d, e) = ([1.0, 1e-20, 1.0], [1.0, 1.0]);
        let s = check(&d, &e, 1e-15);
        // the product of the singular values is |det| = 1e-20, to full relative accuracy
        let det: f64 = s.iter().product();
        assert!((det - 1e-20).abs() <= 1e-34, "{det}");
    }

    #[test]
    fn lasv2_reconstructs_extreme_blocks() {
        for &(f, g, h) in &[(1.0, 1e20, 1.0), (1e-20, 1.0, 1e-20), (2.0, 0.0, 3.0), (0.0, 1.0, 1.0),
                            (1.0, 1.0, 0.0), (-1.0, 2.0, -3.0), (3.0, -2.0, 1.0), (1.0, 1.0, 1.0)] {
            let s = lasv2(f, g, h);
            // [csl snl; -snl csl] [f g; 0 h] [csr -snr; snr csr] = diag(ssmax, ssmin)
            let a11 = s.csl * (f * s.csr + g * s.snr) + s.snl * h * s.snr;
            let a22 = -s.snl * (-f * s.snr + g * s.csr) + s.csl * h * s.csr;
            let big = f.abs().max(g.abs()).max(h.abs());
            assert!((a11 - s.ssmax).abs() <= 2e-15 * big, "{f} {g} {h}: {a11} {}", s.ssmax);
            assert!((a22 - s.ssmin).abs() <= 2e-15 * big, "{f} {g} {h}: {a22} {}", s.ssmin);
            let (lo, hi) = las2(f, g, h);
            assert!((lo - s.ssmin.abs()).abs() <= 8e-16 * big && (hi - s.ssmax.abs()).abs() <= 8e-16 * big);
        }
    }
}
