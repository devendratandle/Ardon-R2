//! What `hist()` computes, apart from what it draws — and the two base
//! functions it is built on, `pretty()` and `nclass.Sturges()`.
//!
//! Each is GNU R's algorithm line for line, so a histogram here has R's
//! breaks and R's counts: `R_pretty` (src/appl/pretty.c, R 4.x),
//! `pretty.default`'s sequence, `hist.default`'s break rule and fuzzed
//! bin edges, and `C_BinCount`. `hist()` used to make 10 equal-width bins
//! over the data range and return NULL, so `hist(x, plot = FALSE)$counts`
//! — the way R code reads a histogram — could not work, and the bars it
//! drew were not R's bars.

use crate::{gn, gv};
use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal, TypeInstance};
use std::collections::HashMap;
use std::sync::Arc;

fn err(msg: String) -> R2Err { R2Err { msg, kind: ErrKind::Runtime } }

/// `R_pretty` with `return_bounds = TRUE`: the cell unit, the covering
/// bounds `(lo, up)` and the number of intervals. `h` = `high.u.bias`,
/// `h5` = `u5.bias`; `f.min` is R's default `2^-20`, no eps correction.
fn r_pretty(lo: f64, up: f64, ndiv: usize, min_n: usize, shrink_sml: f64, h: f64, h5: f64)
    -> (f64, f64, f64, usize)
{
    const ROUNDING_EPS: f64 = 1e-10;
    const F_MIN: f64 = 1.0 / 1048576.0;
    let dx = up - lo;
    let (mut cell, i_small);
    if dx == 0.0 && up == 0.0 {
        cell = 1.0;
        i_small = true;
    } else {
        cell = lo.abs().max(up.abs());
        let mut u = 1.0 + if h5 >= 1.5 * h + 0.5 { 1.0 / (1.0 + h) } else { 1.5 / (1.0 + h5) };
        u *= ndiv.max(1) as f64 * f64::EPSILON;
        i_small = dx < cell * u * 3.0;
    }
    if i_small {
        if cell > 10.0 { cell = 9.0 + cell / 10.0; }
        cell *= shrink_sml;
        if min_n > 1 { cell /= min_n as f64; }
    } else {
        cell = dx;
        if ndiv > 1 { cell /= ndiv as f64; }
    }
    let subsmall = (F_MIN * f64::MIN_POSITIVE).max(f64::MIN_POSITIVE);
    if cell < subsmall { cell = subsmall; }
    else if cell > f64::MAX / 1.25 { cell = f64::MAX / 1.25; }

    let base = 10f64.powf(cell.log10().floor());
    let mut unit = base;
    let ns2 = 2.0 * base;
    if ns2 - cell < h * (cell - unit) {
        unit = ns2;
        let ns5 = 5.0 * base;
        if ns5 - cell < h5 * (cell - unit) {
            unit = ns5;
            let ns10 = 10.0 * base;
            if ns10 - cell < h * (cell - unit) { unit = ns10; }
        }
    }
    let mut ns = (lo / unit + ROUNDING_EPS).floor();
    let mut nu = (up / unit - ROUNDING_EPS).ceil();
    while ns * unit > lo + ROUNDING_EPS * unit { ns -= 1.0; }
    if !(ns * unit).is_finite() { ns += 1.0; }
    while nu * unit < up - ROUNDING_EPS * unit { nu += 1.0; }
    if !(nu * unit).is_finite() { nu -= 1.0; }

    let mut k = (0.5 + nu - ns) as i64;
    let ndiv_out;
    if k < min_n as i64 {
        k = min_n as i64 - k;
        if ns >= 0.0 {
            nu += (k / 2) as f64;
            ns -= (k / 2 + k % 2) as f64;
        } else {
            ns -= (k / 2) as f64;
            nu += (k / 2 + k % 2) as f64;
        }
        ndiv_out = min_n;
    } else {
        ndiv_out = k as usize;
    }
    // return_bounds: the bounds cover the original range
    let (mut l, mut u) = (lo, up);
    if ns * unit < l { l = ns * unit; }
    if nu * unit > u { u = nu * unit; }
    (unit, l, u, ndiv_out)
}

/// R's `seq(from, to, length.out = n)`: both ends exact, the interior
/// `from + i * by` (seq.default's `c(from, from + seq_len(n1 - 1) * by, to)`).
fn seq_length_out(from: f64, to: f64, n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n];
    if n > 0 { s[0] = from; }
    if n > 1 { s[n - 1] = to; }
    if n > 2 {
        let by = (to - from) / (n - 1) as f64;
        for (i, si) in s.iter_mut().enumerate().take(n - 1).skip(1) {
            *si = from + i as f64 * by;
        }
    }
    s
}

/// `pretty.default(x, n, min.n, shrink.sml, high.u.bias, u5.bias)` over
/// the finite values of `x`.
pub fn pretty(x: &[f64], n: usize, min_n: usize, shrink_sml: f64, h: f64, h5: f64) -> Vec<f64> {
    let fin: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.is_empty() { return Vec::new(); }
    let lo = fin.iter().copied().fold(f64::INFINITY, f64::min);
    let up = fin.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (_, l, u, nd) = r_pretty(lo, up, n, min_n, shrink_sml, h, h5);
    let mut s = seq_length_out(l, u, nd + 1);
    if nd > 0 {
        // zap the near-zeros seq()'s rounding leaves
        let delta = (u - l) / nd as f64;
        for v in s.iter_mut() { if v.abs() < 1e-14 * delta { *v = 0.0; } }
    }
    s
}

/// `nclass.Sturges(x)`: `ceiling(log2(length(x)) + 1)`.
pub fn nclass_sturges(n: usize) -> usize {
    ((n as f64).log2() + 1.0).ceil() as usize
}

/// A histogram as R's `hist.default(x, breaks, right = TRUE,
/// include.lowest = TRUE)` computes it.
pub struct Hist {
    pub breaks: Vec<f64>,
    pub counts: Vec<f64>,
    pub density: Vec<f64>,
    pub mids: Vec<f64>,
    pub equidist: bool,
}

/// Breaks from `breaks =`: absent or `"Sturges"` → Sturges' count; one
/// number → that many cells, both through `pretty(range(x), n, min.n = 1)`;
/// several numbers → those breaks, sorted.
fn breaks_from(x: &[f64], spec: Option<&RVal>) -> Result<Vec<f64>, R2Err> {
    let nclass = match spec {
        None => nclass_sturges(x.len()),
        Some(RVal::Character(v, _)) => match v.first().cloned().flatten().as_deref() {
            Some(s) if s.eq_ignore_ascii_case("sturges") => nclass_sturges(x.len()),
            Some(s) => return Err(err(format!(
                "hist: breaks = \"{s}\" is not supported yet (\"Sturges\", a number of cells, or a vector of breaks)"))),
            None => nclass_sturges(x.len()),
        },
        Some(v) => {
            let b: Vec<f64> = v.as_reals()?.into_iter().flatten().collect();
            if b.len() > 1 {
                let mut b = b;
                b.sort_by(f64::total_cmp);
                return Ok(b);
            }
            match b.first() {
                Some(&n) if n.is_finite() && n >= 1.0 => n as usize,
                _ => return Err(err("hist: invalid number of 'breaks'".into())),
            }
        }
    };
    let (lo, up) = (x.iter().copied().fold(f64::INFINITY, f64::min),
                    x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    Ok(pretty(&[lo, up], nclass, 1, 0.75, 1.5, 0.5 + 1.5 * 1.5))
}

/// `hist.default`'s computation: breaks, fuzzed edges, `C_BinCount`.
pub fn compute(x: &[f64], spec: Option<&RVal>) -> Result<Hist, R2Err> {
    let x: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if x.is_empty() { return Err(err("hist: 'x' has no finite values".into())); }
    let breaks = breaks_from(&x, spec)?;
    let nb = breaks.len();
    if nb < 2 { return Err(err("hist: invalid number of 'breaks'".into())); }
    let h: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
    if h.iter().any(|&d| d < 0.0) || h.iter().all(|&d| d == 0.0) {
        return Err(err("hist: 'breaks' are not strictly increasing".into()));
    }
    let hmin = h.iter().copied().fold(f64::INFINITY, f64::min);
    let hmax = h.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let hmean = h.iter().sum::<f64>() / h.len() as f64;
    let equidist = hmax - hmin < 1e-7 * hmean;
    if x.iter().any(|&v| v < breaks[0] || v > breaks[nb - 1]) {
        return Err(err("hist: some 'x' not counted; maybe 'breaks' do not span range of 'x'".into()));
    }
    // diddle: 1e-7 of the median width (or, with few breaks, the range or
    // the smallest positive width); right = TRUE, include.lowest = TRUE
    let diddle = 1e-7 * if nb > 5 {
        let mut s = h.clone();
        s.sort_by(f64::total_cmp);
        let m = s.len();
        if m % 2 == 1 { s[m / 2] } else { (s[m / 2 - 1] + s[m / 2]) / 2.0 }
    } else if nb <= 3 {
        let (lo, up) = (x.iter().copied().fold(f64::INFINITY, f64::min),
                        x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        up - lo
    } else {
        h.iter().copied().filter(|&d| d > 0.0).fold(f64::INFINITY, f64::min)
    };
    let fuzzy: Vec<f64> = breaks.iter().enumerate()
        .map(|(i, &b)| if i == 0 { b - diddle } else { b + diddle }).collect();
    // C_BinCount(x, fuzzy, right = TRUE, include.border = TRUE)
    let mut counts = vec![0.0; nb - 1];
    for &xi in &x {
        let (mut lo, mut hi) = (0usize, nb - 1);
        if fuzzy[lo] <= xi && xi <= fuzzy[hi] {
            while hi - lo >= 2 {
                let mid = (hi + lo) / 2;
                if xi > fuzzy[mid] { lo = mid; } else { hi = mid; }
            }
            counts[lo] += 1.0;
        }
    }
    let n = x.len() as f64;
    let density = counts.iter().zip(&h).map(|(c, w)| c / (n * w)).collect();
    let mids = breaks.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    Ok(Hist { breaks, counts, density, mids, equidist })
}

fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|&x| Some(x)).collect::<Vec<_>>().into(), Attrs::default())
}

/// The object `hist()` returns: class `"histogram"`, R's fields.
pub fn to_rval(h: &Hist, xname: &str) -> RVal {
    let mut f = HashMap::new();
    f.insert(Arc::from("breaks"), rnums(&h.breaks));
    f.insert(Arc::from("counts"), rnums(&h.counts));
    f.insert(Arc::from("density"), rnums(&h.density));
    f.insert(Arc::from("mids"), rnums(&h.mids));
    f.insert(Arc::from("xname"), RVal::Character(vec![Some(Arc::from(xname))].into(), Attrs::default()));
    f.insert(Arc::from("equidist"), RVal::Logical(vec![Some(h.equidist)].into(), Attrs::default()));
    RVal::TypeInstance(TypeInstance { type_name: Arc::from("histogram"), fields: f })
}

/// `pretty(x, n = 5, min.n = n %/% 3, shrink.sml = 0.75, high.u.bias = 1.5,
/// u5.bias = .5 + 1.5 * high.u.bias)`.
pub fn bi_pretty(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().flatten().collect();
    let num = |k: &str, pos: usize| gn(a, k).or_else(|| a.get(pos).filter(|e| e.name.is_none()).map(|e| e.value.clone()))
        .and_then(|v| v.scalar_f64().ok().flatten());
    let n = num("n", 1).unwrap_or(5.0);
    if !(n >= 0.0) { return Err(err("pretty: invalid 'n' argument".into())); }
    let n = n as usize;
    let min_n = gn(a, "min.n").and_then(|v| v.scalar_f64().ok().flatten()).map(|v| v as usize).unwrap_or(n / 3);
    let shrink = gn(a, "shrink.sml").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.75);
    let h = gn(a, "high.u.bias").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.5);
    let h5 = gn(a, "u5.bias").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.5 + 1.5 * h);
    Ok(rnums(&pretty(&x, n, min_n, shrink, h, h5)))
}

/// `nclass.Sturges(x)`.
pub fn bi_nclass_sturges(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = gv(a, 0).as_reals()?.len();
    Ok(RVal::Integer(vec![Some(nclass_sturges(n) as i32)].into(), Attrs::default()))
}


#[cfg(test)]
mod tests {
    use super::*;

    // Expected values are GNU R 4.5.3's.
    #[test]
    fn pretty_matches_r() {
        assert_eq!(pretty(&[61.0, 95.5], 5, 1, 0.75, 1.5, 2.75), vec![60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]);
        assert_eq!(pretty(&[0.0, 1.0], 5, 1, 0.75, 1.5, 2.75), vec![0.0, 0.2, 0.4, 0.60000000000000009, 0.8, 1.0]);
        assert_eq!(pretty(&[-3.7, 12.2], 5, 1, 0.75, 1.5, 2.75), vec![-5.0, 0.0, 5.0, 10.0, 15.0]);
        assert_eq!(pretty(&[0.001234, 0.0457], 7, 2, 0.75, 1.5, 2.75), vec![0.0, 0.0050000000000000001, 0.01, 0.014999999999999999,
                   0.02, 0.025000000000000001, 0.029999999999999999, 0.035000000000000003, 0.040000000000000001,
                   0.044999999999999998, 0.050000000000000003]);
        assert_eq!(pretty(&[1e6, 1e6 + 3.0], 5, 1, 0.75, 1.5, 2.75), vec![1e6, 1000000.5, 1000001.0, 1000001.5, 1000002.0, 1000002.5, 1000003.0]);
        assert_eq!(pretty(&[5.0, 5.0], 5, 1, 0.75, 1.5, 2.75), vec![0.0, 5.0]);
        assert_eq!(pretty(&[-1e-3, 2e-3], 3, 1, 0.75, 1.5, 2.75), vec![-0.001, 0.0, 0.001, 0.002]);
    }

    #[test]
    fn hist_counts_match_r() {
        let x = [81.5, 77.0, 92.25, 68.0, 88.0, 95.5, 73.0, 61.0, 84.0, 79.5, 90.0, 70.25];
        let h = compute(&x, None).unwrap();
        assert_eq!(h.breaks, vec![60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]);
        assert_eq!(h.counts, vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]);
        // a value ON a break goes to the bin on its left; the lowest
        // break is closed
        let h = compute(&[0.0, 1.0, 1.0, 2.0], Some(&rnums(&[0.0, 1.0, 2.0]))).unwrap();
        assert_eq!(h.counts, vec![3.0, 1.0]);
    }
}
