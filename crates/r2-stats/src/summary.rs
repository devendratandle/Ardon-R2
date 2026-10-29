//! Summary statistics — Phase R.9.
//!
//! `cov`, `quantile`, `range`, `which.min`, `which.max`, `cumsum`,
//! `cumprod`, `cummax`, `cummin`, `diff`. Pure builtins.

use r2_types::{Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, Real};

/// `cor(X)` for a matrix — the p×p Pearson correlation matrix (column-major).
fn cor_matrix(m: &Matrix) -> RVal {
    // Column-pair Pearson via the shared centred-moment primitive — the
    // same numerics as cor(x, y) and cor.test, by construction.
    let (n, p) = (m.nrow, m.ncol);
    let mut out = vec![0.0f64; p * p];
    for i in 0..p {
        for j in i..p {
            let ci = &m.data[i * n..i * n + n];
            let cj = &m.data[j * n..j * n + n];
            let (_, _, _, sxx, syy, sxy) = crate::moments::centred2_dense(ci, cj);
            let r = crate::moments::pearson_from(sxx, syy, sxy);
            out[j * p + i] = r; out[i * p + j] = r;
        }
    }
    RVal::Matrix(Matrix::new(out, p, p))
}

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_deref() == Some(name)).map(|x| x.value.clone())
}

/// Ranks with ties averaged — R's `rank(x, ties.method = "average")`.
fn avg_ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] { j += 1; }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j { r[idx[k]] = avg; }
        i = j + 1;
    }
    r
}

/// Kendall's tau-b, what R's `cor(method = "kendall")` returns:
/// `(C - D) / sqrt((n0 - n1)(n0 - n2))` with ties in x and y counted
/// apart. O(n^2), as R's own implementation is.
fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let (mut s, mut tx, mut ty) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = (x[i] - x[j]).signum() * if x[i] == x[j] { 0.0 } else { 1.0 };
            let b = (y[i] - y[j]).signum() * if y[i] == y[j] { 0.0 } else { 1.0 };
            s += a * b;
            tx += a * a;
            ty += b * b;
        }
    }
    s / (tx * ty).sqrt()
}

#[inline]
fn rnum(n: f64) -> RVal { RVal::Numeric(vec![Some(n)].into(), Attrs::default()) }

#[inline]
fn rna() -> RVal { RVal::Numeric(vec![None].into(), Attrs::default()) }

#[inline]
fn rint(n: i32) -> RVal { RVal::Integer(vec![Some(n)].into(), Attrs::default()) }

// ─────────────────────────────────────────────────────────────────────
// Pairwise / windowed
// ─────────────────────────────────────────────────────────────────────

pub fn bi_cor(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // cor(X) on a matrix with no second arg → p×p correlation matrix.
    if let RVal::Matrix(m) = first(a) {
        if matches!(nth(a, 1), RVal::Null) { return Ok(cor_matrix(&m)); }
    }
    // One implementation of the centred-moment math (crate::moments) is
    // shared by cor/cov/cor-matrix/cor.test — accuracy consistency rule.
    let x = first(a).as_reals()?;
    let y = nth(a, 1).as_reals()?;
    if x.len() != y.len() { return Err(R2Err { msg: "incompatible dimensions".into(), kind: ErrKind::Runtime }); }
    // `method =` used to be ignored: every call was Pearson's r.
    let method = match named(a, "method") {
        Some(RVal::Character(v, _)) => v.first().cloned().flatten().map(|s| s.to_string()).unwrap_or_default(),
        _ => "pearson".to_string(),
    };
    if method != "pearson" {
        let (xs, ys): (Vec<f64>, Vec<f64>) = x.iter().zip(&y)
            .filter_map(|(a, b)| Some(((*a)?, (*b)?))).unzip();
        // NaN (0/0) is not NA but still has no rank: NA, as R gives
        if xs.len() < 2 || xs.iter().chain(&ys).any(|v| v.is_nan()) { return Ok(rna()); }
        let r = match method.as_str() {
            "spearman" => {
                // Pearson on the average-tie ranks, as R's cor does
                let (rx, ry) = (avg_ranks(&xs), avg_ranks(&ys));
                let (_, _, _, sxx, syy, sxy) = crate::moments::centred2_dense(&rx, &ry);
                crate::moments::pearson_from(sxx, syy, sxy)
            }
            "kendall" => kendall_tau_b(&xs, &ys),
            m => return Err(R2Err {
                msg: format!("cor: method must be \"pearson\", \"kendall\" or \"spearman\", not \"{m}\""),
                kind: ErrKind::Runtime,
            }),
        };
        return Ok(if r.is_nan() { rna() } else { rnum(r) });
    }
    match crate::moments::centred2_pairwise(&x, &y) {
        Some((_, _, _, sxx, syy, sxy)) => {
            let r = crate::moments::pearson_from(sxx, syy, sxy);
            if r.is_nan() { Ok(rna()) } else { Ok(rnum(r)) }
        }
        None => Ok(rna()),
    }
}

pub fn bi_cov(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let y = nth(a, 1).as_reals()?;
    if x.len() != y.len() { return Err(R2Err { msg: "incompatible dimensions".into(), kind: ErrKind::Runtime }); }
    match crate::moments::centred2_pairwise(&x, &y) {
        Some((n, _, _, _, _, sxy)) => Ok(rnum(crate::moments::cov_from(n, sxy))),
        None => Ok(rna()),
    }
}

pub fn bi_diff(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    if v.len() < 2 { return Ok(RVal::Numeric(vec![].into(), Attrs::default())); }
    let result: Vec<Real> = v.windows(2)
        .map(|w| match (w[0], w[1]) { (Some(a), Some(b)) => Some(b - a), _ => None })
        .collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Cumulative reductions
// ─────────────────────────────────────────────────────────────────────

// Phase K.7: cumulative reductions now go through r2-kernel's `scan`
// dispatcher (Oracle picks Serial vs Rayon). Bonus correctness fix:
// the previous cummax/cummin implementations didn't propagate NA past
// the first None — `.map(|n| { max = max.max(n); max })` runs the
// closure even on Some after a None has set the running max. The
// kernel path uses an explicit `hit_na` flag and emits None for all
// positions after the first None, matching R semantics.

pub fn bi_cumsum(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    Ok(RVal::Numeric(r2_kernel::scan(r2_kernel::ScanOp::Cumsum, &v).into(), Attrs::default()))
}

pub fn bi_cumprod(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    Ok(RVal::Numeric(r2_kernel::scan(r2_kernel::ScanOp::Cumprod, &v).into(), Attrs::default()))
}

pub fn bi_cummax(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    Ok(RVal::Numeric(r2_kernel::scan(r2_kernel::ScanOp::Cummax, &v).into(), Attrs::default()))
}

pub fn bi_cummin(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    Ok(RVal::Numeric(r2_kernel::scan(r2_kernel::ScanOp::Cummin, &v).into(), Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Rolling window reductions (Phase K.9 — exposed as user-facing builtins)
// ─────────────────────────────────────────────────────────────────────
//
// `rollsum(x, k)`, `rollmean(x, k)`, `rollmax(x, k)`, `rollmin(x, k)`,
// `rollsd(x, k)` — fixed-width sliding-window reductions. Output is
// shorter than input by `k-1` (right-aligned, no padding). Matches
// `zoo::rollapply(..., align="right")` shape.

fn arg_window(a: &[EvalArg]) -> Result<usize, R2Err> {
    let v = a.get(1).map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let n = v.scalar_f64()?.unwrap_or(0.0);
    if n < 1.0 {
        return Err(R2Err {
            msg: format!("rolling: window must be >= 1, got {}", n),
            kind: ErrKind::Runtime,
        });
    }
    Ok(n as usize)
}

pub fn bi_rollsum(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let w = arg_window(a)?;
    Ok(RVal::Numeric(r2_kernel::rolling(r2_kernel::RollingOp::Sum, &v, w).into(), Attrs::default()))
}

pub fn bi_rollmean(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let w = arg_window(a)?;
    Ok(RVal::Numeric(r2_kernel::rolling(r2_kernel::RollingOp::Mean, &v, w).into(), Attrs::default()))
}

pub fn bi_rollmax(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let w = arg_window(a)?;
    Ok(RVal::Numeric(r2_kernel::rolling(r2_kernel::RollingOp::Max, &v, w).into(), Attrs::default()))
}

pub fn bi_rollmin(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let w = arg_window(a)?;
    Ok(RVal::Numeric(r2_kernel::rolling(r2_kernel::RollingOp::Min, &v, w).into(), Attrs::default()))
}

pub fn bi_rollsd(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let w = arg_window(a)?;
    Ok(RVal::Numeric(r2_kernel::rolling(r2_kernel::RollingOp::Sd, &v, w).into(), Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Quantiles + range / which-extrema
// ─────────────────────────────────────────────────────────────────────

pub fn bi_quantile(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // Phase memory.scratch: use the per-thread pool for the sorted
    // workspace. Quantile is a sort-then-index pattern, classic
    // "materialise and discard" use case.
    let input = first(a).as_reals()?;
    let needed = input.len();
    let mut v = r2_memory::scratch_acquire(needed);
    for x in input.iter() { if let Some(n) = x { v.push(*n); } }
    if v.is_empty() {
        r2_memory::scratch_release(v);
        return Ok(rna());
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let probs: Vec<f64> = if a.len() > 1 {
        nth(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect()
    } else {
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    };
    let n = v.len();
    // `type =` 1..9, R's `quantile.default` line for line (Hyndman & Fan
    // 1996). It used to be ignored: every type returned type 7.
    let qtype = named(a, "type").and_then(|t| t.as_reals().ok())
        .and_then(|t| t.first().copied().flatten()).unwrap_or(7.0) as i64;
    if !(1..=9).contains(&qtype) {
        r2_memory::scratch_release(v);
        return Err(R2Err { msg: "quantile: 'type' must be 1, 2, ..., 9".into(), kind: ErrKind::Runtime });
    }
    let fuzz = 4.0 * f64::EPSILON;
    let result: Vec<Real> = probs.iter().map(|&p| {
        if qtype == 7 {
            // R's own type-7 branch: x[lo], moved toward x[hi] only when
            // they differ (so a tie returns the value exactly).
            let index = (n - 1) as f64 * p;
            let (lo, hi) = (index.floor() as usize, index.ceil() as usize);
            let (lo, hi) = (lo.min(n - 1), hi.min(n - 1));
            let mut q = v[lo];
            if index > lo as f64 && v[hi] != q {
                let h = index - lo as f64;
                q = (1.0 - h) * q + h * v[hi];
            }
            return Some(q);
        }
        let nf = n as f64;
        let (j, h) = if qtype <= 3 {
            let nppm = if qtype == 3 { nf * p - 0.5 } else { nf * p };
            let j = (nppm + fuzz).floor();
            let h = match qtype {
                1 => if nppm > j { 1.0 } else { 0.0 },
                2 => ((if nppm > j { 1.0 } else { 0.0 }) + 1.0) / 2.0,
                _ => if nppm != j || (j as i64).rem_euclid(2) == 1 { 1.0 } else { 0.0 },
            };
            (j, h)
        } else {
            let (aa, bb) = match qtype { 4 => (0.0, 1.0), 5 => (0.5, 0.5), 6 => (0.0, 0.0),
                                         8 => (1.0 / 3.0, 1.0 / 3.0), _ => (3.0 / 8.0, 3.0 / 8.0) };
            let nppm = aa + p * (nf + 1.0 - aa - bb);
            let j = (nppm + fuzz).floor();
            let h = nppm - j;
            (j, if h.abs() < fuzz { 0.0 } else { h })
        };
        // R pads x as c(x[1], x[1], x, x[n], x[n]) and reads x[j+2] and
        // x[j+3] (1-based): padded position k is v[k - 3], clamped.
        let at = |k: f64| v[((k as i64) - 3).clamp(0, n as i64 - 1) as usize];
        let (xl, xh) = (at(j + 2.0), at(j + 3.0));
        let mut q = xl;
        if h == 1.0 { q = xh; }
        else if 0.0 < h && h < 1.0 && xl != xh { q = (1.0 - h) * xl + h * xh; }
        Some(q)
    }).collect();
    r2_memory::scratch_release(v);
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_range(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    let min = v.iter().filter_map(|x| *x).fold(f64::INFINITY, f64::min);
    let max = v.iter().filter_map(|x| *x).fold(f64::NEG_INFINITY, f64::max);
    Ok(RVal::Numeric(vec![Some(min), Some(max)].into(), Attrs::default()))
}

// Phase K.8: which.min / which.max now route through r2-kernel's
// `which_min` / `which_max`. The previous implementations didn't
// propagate NA — they silently skipped Nones, which can give wrong
// answers when the actual extremum is NA. R's behavior is to return
// `integer(0)` when all-NA; we return the smallest valid index as a
// pragmatic 1-based answer (caller can check for non-empty input first).
//
// Note: R uses 1-based indexing; the kernel returns 0-based, so we add 1.
pub fn bi_which_min(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    match r2_kernel::which_min(&v) {
        Some(idx) => Ok(rint((idx + 1) as i32)),
        None => Ok(RVal::Integer(Vec::<r2_types::Integer>::new().into(), Attrs::default())),
    }
}

pub fn bi_which_max(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = first(a).as_reals()?;
    match r2_kernel::which_max(&v) {
        Some(idx) => Ok(rint((idx + 1) as i32)),
        None => Ok(RVal::Integer(Vec::<r2_types::Integer>::new().into(), Attrs::default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        use r2_types::Reals;
        // Columnar-first: dense, no NAs by construction.
        RVal::Numeric(Reals::from_dense_f64(v.to_vec()), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn cov_perfect_correlation_equals_var() {
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let r = bi_cov(&[evarg(x.clone()), evarg(x)]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert!((v[0].unwrap() - 2.5).abs() < 1e-12),
            _ => panic!(),
        }
    }

    #[test]
    fn diff_pairs() {
        let r = bi_diff(&[evarg(nums(&[1.0, 3.0, 6.0, 10.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![2.0, 3.0, 4.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn cumsum_accumulates() {
        let r = bi_cumsum(&[evarg(nums(&[1.0, 2.0, 3.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1.0, 3.0, 6.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn cummax_is_running_max() {
        let r = bi_cummax(&[evarg(nums(&[3.0, 1.0, 4.0, 1.0, 5.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![3.0, 3.0, 4.0, 4.0, 5.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn quantile_default_returns_5_breakpoints() {
        let r = bi_quantile(&[evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn range_returns_min_max() {
        let r = bi_range(&[evarg(nums(&[3.0, 1.0, 4.0, 1.0, 5.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1.0, 5.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn which_min_max_are_1_based() {
        assert!(matches!(
            bi_which_min(&[evarg(nums(&[3.0, 1.0, 4.0]))]).unwrap(),
            RVal::Integer(v, _) if v[0] == Some(2)
        ));
        assert!(matches!(
            bi_which_max(&[evarg(nums(&[3.0, 1.0, 4.0]))]).unwrap(),
            RVal::Integer(v, _) if v[0] == Some(3)
        ));
    }
}
