//! Summary statistics — Phase R.9.
//!
//! `cov`, `quantile`, `range`, `which.min`, `which.max`, `cumsum`,
//! `cumprod`, `cummax`, `cummin`, `diff`. Pure builtins.

use r2_types::{Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, Real};

/// `cor(X)` for a matrix — the p×p Pearson correlation matrix (column-major).
fn cor_matrix(m: &Matrix) -> RVal {
    let (n, p) = (m.nrow, m.ncol);
    let nf = n as f64;
    let mut means = vec![0.0f64; p];
    let mut sds = vec![0.0f64; p];
    for j in 0..p {
        let col = &m.data[j * n..j * n + n];
        let mean = col.iter().sum::<f64>() / nf;
        let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
        means[j] = mean; sds[j] = var.sqrt();
    }
    let mut out = vec![0.0f64; p * p];
    for i in 0..p {
        for j in i..p {
            let ci = &m.data[i * n..i * n + n];
            let cj = &m.data[j * n..j * n + n];
            let cov = ci.iter().zip(cj).map(|(x, y)| (x - means[i]) * (y - means[j])).sum::<f64>() / (nf - 1.0);
            let r = if sds[i] == 0.0 || sds[j] == 0.0 { f64::NAN } else { cov / (sds[i] * sds[j]) };
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
    let x = first(a).as_reals()?;
    let y = nth(a, 1).as_reals()?;
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter_map(|(a, b)| match (a, b) { (Some(a), Some(b)) => Some((*a, *b)), _ => None })
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 { return Ok(rna()); }
    let mx = pairs.iter().map(|(a, _)| a).sum::<f64>() / n;
    let my = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;
    let cov = pairs.iter().map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n - 1.0);
    let sx = (pairs.iter().map(|(a, _)| (a - mx).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let sy = (pairs.iter().map(|(_, b)| (b - my).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if sx == 0.0 || sy == 0.0 { Ok(rna()) } else { Ok(rnum(cov / (sx * sy))) }
}

pub fn bi_cov(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let y = nth(a, 1).as_reals()?;
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter_map(|(a, b)| match (a, b) { (Some(a), Some(b)) => Some((*a, *b)), _ => None })
        .collect();
    let n = pairs.len();
    if n < 2 { return Ok(rna()); }
    let nf = n as f64;
    let (sx, sy) = pairs.iter()
        .fold((0.0_f64, 0.0_f64), |(ax, ay), (bx, by)| (ax + bx, ay + by));
    let mx = sx / nf;
    let my = sy / nf;
    let cov = pairs.iter().map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (nf - 1.0);
    Ok(rnum(cov))
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
    let result: Vec<Real> = probs.iter().map(|p| {
        let idx = p * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        Some(v[lo.min(n - 1)] * (1.0 - frac) + v[hi.min(n - 1)] * frac)
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
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
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
