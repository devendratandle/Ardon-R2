//! Multivariate hypothesis tests — Phase R.S.2.
//!
//! Hotelling's T² in three flavors:
//!   - One-sample: tests `H₀: μ = μ₀` for a p-dimensional mean vector
//!   - Two-sample: tests if two p-dimensional group means differ
//!   - Paired (repeated measures): multivariate generalization of the
//!     paired t-test; tests if mean difference vector = 0
//!
//! Math (n = sample size, p = dimension):
//!
//! - One-sample: `T² = n · (x̄ - μ₀)ᵀ S⁻¹ (x̄ - μ₀)`.
//!   Under H₀: `F = T² · (n-p) / (p·(n-1))` distributed as `F(p, n-p)`.
//! - Two-sample (sizes n₁, n₂) with pooled covariance S_p:
//!   `T² = (n₁·n₂)/(n₁+n₂) · (x̄ - ȳ)ᵀ S_p⁻¹ (x̄ - ȳ)`.
//!   `F = T² · (n₁+n₂-p-1) / (p·(n₁+n₂-2))` distributed as `F(p, n₁+n₂-p-1)`.
//! - Paired (repeated measures): one-sample T² on per-subject
//!   difference vectors against μ₀ = 0.

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::{fmt_num, Attrs, ErrKind, EvalArg, R2Err, RVal, TypeInstance};
#[cfg(test)]
use r2_types::Matrix;

use crate::{fmt_pval, signif_stars};

// ── Small helpers ────────────────────────────────────────────────────

#[inline]
fn gv(a: &[EvalArg], i: usize) -> RVal {
    a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null)
}
#[inline]
fn gn(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter()
        .find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|x| x.value.clone())
}
#[inline]
fn rnum(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }
#[inline]
fn rstr(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }

/// Coerce an RVal into a row-major (n × p) f64 matrix. Accepts
/// `RVal::Matrix` directly; for `RVal::Numeric` or a `List` of column
/// vectors it builds the matrix column-by-column. Returns `(n, p, data)`
/// in row-major layout.
fn as_matrix(v: &RVal) -> Result<(usize, usize, Vec<f64>), R2Err> {
    match v {
        RVal::Matrix(m) => {
            // Matrix stores column-major; convert to row-major for our math.
            let n = m.nrow;
            let p = m.ncol;
            let mut data = vec![0.0; n * p];
            for i in 0..n {
                for j in 0..p {
                    data[i * p + j] = m.get(i, j);
                }
            }
            Ok((n, p, data))
        }
        RVal::List(cols) => {
            if cols.is_empty() {
                return Err(R2Err { msg: "hotelling: empty list".into(), kind: ErrKind::Runtime });
            }
            // Each list element should be a numeric column.
            let n = match &cols[0].1 {
                RVal::Numeric(v, _) => v.len(),
                RVal::Integer(v, _) => v.len(),
                _ => return Err(R2Err {
                    msg: "hotelling: list elements must be numeric vectors".into(),
                    kind: ErrKind::Runtime,
                }),
            };
            let p = cols.len();
            let mut data = vec![0.0; n * p];
            for (j, (_, col)) in cols.iter().enumerate() {
                let vals: Vec<Option<f64>> = col.as_reals()?;
                if vals.len() != n {
                    return Err(R2Err {
                        msg: format!("hotelling: column {} has length {} but expected {}", j+1, vals.len(), n),
                        kind: ErrKind::Runtime,
                    });
                }
                for (i, val) in vals.iter().enumerate() {
                    data[i * p + j] = val.unwrap_or(f64::NAN);
                }
            }
            Ok((n, p, data))
        }
        RVal::Numeric(_, _) | RVal::Integer(_, _) => {
            // 1-D — treat as a column vector (n × 1).
            let vals = v.as_reals()?;
            let n = vals.len();
            let data: Vec<f64> = vals.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            Ok((n, 1, data))
        }
        _ => Err(R2Err {
            msg: format!("hotelling: cannot interpret {} as an n × p matrix", v.type_name()),
            kind: ErrKind::Runtime,
        }),
    }
}

/// Column means of a row-major (n × p) matrix.
fn col_means(data: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut means = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            means[j] += data[i * p + j];
        }
    }
    let nf = n as f64;
    for j in 0..p { means[j] /= nf; }
    means
}

/// Sample covariance matrix S (p × p) using (n - 1) denominator.
/// Returned column-major (matches r2_linalg conventions for dgetri).
fn sample_cov(data: &[f64], n: usize, p: usize) -> Vec<f64> {
    let means = col_means(data, n, p);
    let mut s = vec![0.0; p * p];
    for i in 0..n {
        for a in 0..p {
            let da = data[i * p + a] - means[a];
            for b in 0..p {
                let db = data[i * p + b] - means[b];
                // Column-major (col, row): s[b * p + a]
                s[b * p + a] += da * db;
            }
        }
    }
    let denom = (n as f64 - 1.0).max(1.0);
    for x in &mut s { *x /= denom; }
    s
}

/// Compute xᵀ M⁻¹ x where x has length p and M is p × p (column-major).
/// Uses dgetri to invert M, then accumulates the quadratic form.
fn quad_form_inv(x: &[f64], m: &[f64], p: usize) -> Result<f64, R2Err> {
    let m_inv = r2_linalg::dgetri(p, m).map_err(|e| R2Err {
        msg: format!("hotelling: covariance matrix is singular ({})", e),
        kind: ErrKind::Runtime,
    })?;
    let mut q = 0.0;
    for a in 0..p {
        for b in 0..p {
            // m_inv is column-major: m_inv[b * p + a] = element (a, b).
            q += x[a] * m_inv[b * p + a] * x[b];
        }
    }
    Ok(q)
}

/// Exact F upper-tail `P(F > f)` via the regularized incomplete beta
/// (`htest::f_sf`). Replaces the former Wilson-Hilferty approximation,
/// so manova / Hotelling p-values now match R to ~1e-9 even at small
/// df. Non-integer df (from the Rao-style approximations) are fine —
/// the incomplete beta takes real parameters.
fn f_to_pvalue(f: f64, df1: f64, df2: f64) -> f64 {
    crate::htest::f_sf(f, df1, df2)
}

// ── Hotelling T² — one-sample ────────────────────────────────────────

/// `hotelling.test(X)` or `hotelling.test(X, mu = c(...))`. Tests
/// `H₀: μ = μ₀` for the p-dimensional column mean vector of X.
///
/// X must be n × p with n > p; otherwise the sample covariance is
/// singular and the test is undefined.
pub fn bi_hotelling_one(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (n, p, x) = as_matrix(&gv(a, 0))?;
    if n <= p {
        return Err(R2Err {
            msg: format!("hotelling.test: need n > p (got n = {}, p = {}). The sample covariance is singular otherwise.", n, p),
            kind: ErrKind::Runtime,
        });
    }

    // μ₀ — defaults to zero vector if omitted.
    let mu: Vec<f64> = match gn(a, "mu") {
        Some(v) => {
            let m = v.as_reals()?.into_iter().map(|x| x.unwrap_or(0.0)).collect::<Vec<_>>();
            if m.len() != p {
                return Err(R2Err {
                    msg: format!("hotelling.test: mu must have length p = {} (got {})", p, m.len()),
                    kind: ErrKind::Runtime,
                });
            }
            m
        }
        None => vec![0.0; p],
    };

    let means = col_means(&x, n, p);
    let diff: Vec<f64> = means.iter().zip(&mu).map(|(m, u)| m - u).collect();
    let cov = sample_cov(&x, n, p);
    let q = quad_form_inv(&diff, &cov, p)?;
    let t2 = (n as f64) * q;

    let df1 = p as f64;
    let df2 = (n - p) as f64;
    let f_stat = t2 * df2 / (df1 * (n as f64 - 1.0));
    let p_value = f_to_pvalue(f_stat, df1, df2);

    soutln!("\n  Hotelling's one-sample T² test\n");
    soutln!("data:  n = {}, p = {}", n, p);
    soutln!("T² = {}, F = {}, df1 = {}, df2 = {}, p-value = {}",
        fmt_num(t2), fmt_num(f_stat), df1 as i32, df2 as i32, fmt_pval(p_value));
    soutln!("alternative hypothesis: true mean vector is not equal to mu");
    sout!("sample means: ");
    for (j, m) in means.iter().enumerate() {
        if j > 0 { sout!(", "); }
        sout!("{}", fmt_num(*m));
    }
    soutln!();

    let mut fields = HashMap::new();
    fields.insert(Arc::from("method"), rstr("Hotelling one-sample T²"));
    fields.insert(Arc::from("statistic.t2"), rnum(t2));
    fields.insert(Arc::from("statistic.f"), rnum(f_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("df1"), rnum(df1));
    fields.insert(Arc::from("df2"), rnum(df2));
    fields.insert(Arc::from("n"), rnum(n as f64));
    fields.insert(Arc::from("p.dim"), rnum(p as f64));
    let _ = signif_stars(p_value);
    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("htest"),
        fields,
    }))
}

// ── Hotelling T² — two-sample ────────────────────────────────────────

/// `hotelling.test(X, Y)`. Tests if two p-dimensional group means differ.
/// Uses pooled covariance assuming equal covariance matrices.
pub fn bi_hotelling_two(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (n1, p1, x) = as_matrix(&gv(a, 0))?;
    let (n2, p2, y) = as_matrix(&gv(a, 1))?;
    if p1 != p2 {
        return Err(R2Err {
            msg: format!("hotelling.test (two-sample): X and Y must have same number of columns (got {} and {})", p1, p2),
            kind: ErrKind::Runtime,
        });
    }
    let p = p1;
    if n1 + n2 <= p + 1 {
        return Err(R2Err {
            msg: format!("hotelling.test (two-sample): need n1 + n2 > p + 1 (got n1+n2 = {}, p = {})", n1 + n2, p),
            kind: ErrKind::Runtime,
        });
    }

    let mx = col_means(&x, n1, p);
    let my = col_means(&y, n2, p);
    let diff: Vec<f64> = mx.iter().zip(&my).map(|(a, b)| a - b).collect();

    // Pooled covariance: S_p = ((n1-1)*S_x + (n2-1)*S_y) / (n1+n2-2)
    let sx = sample_cov(&x, n1, p);
    let sy = sample_cov(&y, n2, p);
    let n1f = n1 as f64; let n2f = n2 as f64;
    let denom = n1f + n2f - 2.0;
    let mut s_pool = vec![0.0; p * p];
    for i in 0..(p * p) {
        s_pool[i] = ((n1f - 1.0) * sx[i] + (n2f - 1.0) * sy[i]) / denom;
    }

    let q = quad_form_inv(&diff, &s_pool, p)?;
    let t2 = (n1f * n2f) / (n1f + n2f) * q;

    let df1 = p as f64;
    let df2 = (n1 + n2 - p - 1) as f64;
    let f_stat = t2 * df2 / (df1 * (n1f + n2f - 2.0));
    let p_value = f_to_pvalue(f_stat, df1, df2);

    soutln!("\n  Hotelling's two-sample T² test\n");
    soutln!("data:  n1 = {}, n2 = {}, p = {}", n1, n2, p);
    soutln!("T² = {}, F = {}, df1 = {}, df2 = {}, p-value = {}",
        fmt_num(t2), fmt_num(f_stat), df1 as i32, df2 as i32, fmt_pval(p_value));
    soutln!("alternative hypothesis: true mean vectors of groups 1 and 2 differ");

    let mut fields = HashMap::new();
    fields.insert(Arc::from("method"), rstr("Hotelling two-sample T²"));
    fields.insert(Arc::from("statistic.t2"), rnum(t2));
    fields.insert(Arc::from("statistic.f"), rnum(f_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("df1"), rnum(df1));
    fields.insert(Arc::from("df2"), rnum(df2));
    fields.insert(Arc::from("n1"), rnum(n1f));
    fields.insert(Arc::from("n2"), rnum(n2f));
    fields.insert(Arc::from("p.dim"), rnum(p as f64));
    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("htest"),
        fields,
    }))
}

// ── Hotelling T² — paired / repeated-measures ────────────────────────

/// `hotelling.test(X, Y, paired = TRUE)`. Multivariate paired test:
/// computes within-subject difference matrix D = X - Y (row-wise) and
/// runs the one-sample T² of D against μ₀ = 0.
///
/// X and Y must be n × p with the same dimensions; rows are paired by
/// position (e.g., row i of X and row i of Y are the i-th subject's
/// "before" and "after" measurements).
pub fn bi_hotelling_paired(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (n1, p1, x) = as_matrix(&gv(a, 0))?;
    let (n2, p2, y) = as_matrix(&gv(a, 1))?;
    if n1 != n2 || p1 != p2 {
        return Err(R2Err {
            msg: format!("hotelling.test (paired): X and Y must have identical shape (got {}×{} vs {}×{})", n1, p1, n2, p2),
            kind: ErrKind::Runtime,
        });
    }
    let n = n1; let p = p1;
    if n <= p {
        return Err(R2Err {
            msg: format!("hotelling.test (paired): need n > p (got n = {}, p = {})", n, p),
            kind: ErrKind::Runtime,
        });
    }

    // Difference matrix D (row-major).
    let mut d = vec![0.0; n * p];
    for i in 0..n {
        for j in 0..p {
            d[i * p + j] = x[i * p + j] - y[i * p + j];
        }
    }

    let means = col_means(&d, n, p);
    let cov = sample_cov(&d, n, p);
    let q = quad_form_inv(&means, &cov, p)?;
    let t2 = (n as f64) * q;

    let df1 = p as f64;
    let df2 = (n - p) as f64;
    let f_stat = t2 * df2 / (df1 * (n as f64 - 1.0));
    let p_value = f_to_pvalue(f_stat, df1, df2);

    soutln!("\n  Hotelling's paired T² test (multivariate)\n");
    soutln!("data:  n = {} subjects, p = {} measurements per subject", n, p);
    soutln!("T² = {}, F = {}, df1 = {}, df2 = {}, p-value = {}",
        fmt_num(t2), fmt_num(f_stat), df1 as i32, df2 as i32, fmt_pval(p_value));
    soutln!("alternative hypothesis: true mean difference vector is not equal to 0");
    sout!("mean difference vector: ");
    for (j, m) in means.iter().enumerate() {
        if j > 0 { sout!(", "); }
        sout!("{}", fmt_num(*m));
    }
    soutln!();

    let mut fields = HashMap::new();
    fields.insert(Arc::from("method"), rstr("Hotelling paired T² (multivariate)"));
    fields.insert(Arc::from("statistic.t2"), rnum(t2));
    fields.insert(Arc::from("statistic.f"), rnum(f_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("df1"), rnum(df1));
    fields.insert(Arc::from("df2"), rnum(df2));
    fields.insert(Arc::from("n"), rnum(n as f64));
    fields.insert(Arc::from("p.dim"), rnum(p as f64));
    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("htest"),
        fields,
    }))
}

// MANOVA moved to multivariate/manova.rs.
mod manova;
pub use manova::*;

// ── Dispatcher: `hotelling.test(...)` resolves to one of three flavors ──

/// `hotelling.test(X)` → one-sample.
/// `hotelling.test(X, Y)` → two-sample (independent groups).
/// `hotelling.test(X, Y, paired = TRUE)` → paired/repeated-measures.
pub fn bi_hotelling_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let paired = gn(a, "paired")
        .and_then(|v| v.as_logicals().ok())
        .and_then(|v| v.first().copied().flatten())
        .unwrap_or(false);

    let two_sample = a.len() >= 2 && a[1].name.is_none();

    if two_sample {
        if paired {
            bi_hotelling_paired(a)
        } else {
            bi_hotelling_two(a)
        }
    } else {
        bi_hotelling_one(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(data: Vec<f64>, n: usize, p: usize) -> RVal {
        // Build a Matrix (column-major storage).
        let mut col_major = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                col_major[j * n + i] = data[i * p + j];
            }
        }
        RVal::Matrix(Matrix::new(col_major, n, p))
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }
    fn named(name: &str, v: RVal) -> EvalArg {
        EvalArg { name: Some(Arc::from(name)), value: v }
    }

    #[test]
    fn hotelling_one_sample_diagonal_cov() {
        // Classic test: 5 observations, 2 dimensions, mean far from origin.
        // Data with means (10, 20) and small variance — should reject H₀: μ = 0.
        let x = matrix(
            vec![
                10.0, 20.0,
                10.5, 20.5,
                 9.5, 19.5,
                10.2, 20.3,
                 9.8, 19.7,
            ],
            5, 2,
        );
        let r = bi_hotelling_test(&[evarg(x)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").unwrap().scalar_f64().unwrap().unwrap();
                let t2 = inst.fields.get("statistic.t2").unwrap().scalar_f64().unwrap().unwrap();
                assert!(p < 0.001, "p-value too high: {}", p);
                assert!(t2 > 100.0, "T² should be large: {}", t2);
            }
            _ => panic!("must return TypeInstance"),
        }
    }

    #[test]
    fn hotelling_one_sample_at_mu_should_not_reject() {
        // Same data, but test against the correct μ ≈ (10, 20) — should NOT reject.
        let x = matrix(
            vec![
                10.0, 20.0,
                10.5, 20.5,
                 9.5, 19.5,
                10.2, 20.3,
                 9.8, 19.7,
            ],
            5, 2,
        );
        let mu = RVal::Numeric(vec![Some(10.0), Some(20.0)].into(), Attrs::default());
        let r = bi_hotelling_test(&[evarg(x), named("mu", mu)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").unwrap().scalar_f64().unwrap().unwrap();
                assert!(p > 0.05, "p-value at correct μ should be high: got {}", p);
            }
            _ => panic!("must return TypeInstance"),
        }
    }

    #[test]
    fn hotelling_two_sample_clearly_different_groups() {
        // Two groups with clearly different bivariate means.
        let x = matrix(
            vec![
                 1.0,  1.0,
                 1.5,  1.2,
                 0.8,  1.1,
                 1.2,  0.9,
                 1.3,  1.4,
            ], 5, 2);
        let y = matrix(
            vec![
                 5.0,  6.0,
                 5.5,  6.2,
                 4.8,  6.1,
                 5.2,  5.9,
                 5.3,  6.4,
            ], 5, 2);
        let r = bi_hotelling_test(&[evarg(x), evarg(y)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").unwrap().scalar_f64().unwrap().unwrap();
                assert!(p < 0.001, "two clearly-different groups should reject: p = {}", p);
            }
            _ => panic!("must return TypeInstance"),
        }
    }

    #[test]
    fn hotelling_paired_zero_difference_should_not_reject() {
        // X and Y identical — mean difference vector = 0, should not reject.
        let x = matrix(
            vec![
                 1.0,  2.0,
                 1.5,  2.5,
                 0.8,  1.9,
                 1.2,  2.1,
                 1.3,  2.2,
            ], 5, 2);
        let y = x.clone();
        let true_val = RVal::Logical(vec![Some(true)].into(), Attrs::default());
        let r = bi_hotelling_test(&[evarg(x), evarg(y), named("paired", true_val)]);
        // Identical X and Y → zero covariance → singular → expected error.
        assert!(r.is_err(), "identical X,Y should produce singular covariance error");
    }

    #[test]
    fn hotelling_paired_consistent_shift_should_reject() {
        // Y = X + (3, 5) for every row — consistent shift, should reject H₀.
        let x = matrix(
            vec![
                 1.0,  2.0,
                 1.5,  2.5,
                 0.8,  1.9,
                 1.2,  2.1,
                 1.3,  2.2,
            ], 5, 2);
        let y = matrix(
            vec![
                 4.05,  7.10,
                 4.45,  7.55,
                 3.85,  6.85,
                 4.20,  7.20,
                 4.35,  7.15,
            ], 5, 2);
        let true_val = RVal::Logical(vec![Some(true)].into(), Attrs::default());
        let r = bi_hotelling_test(&[evarg(x), evarg(y), named("paired", true_val)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let p = inst.fields.get("p.value").unwrap().scalar_f64().unwrap().unwrap();
                assert!(p < 0.01, "consistent shift should reject: p = {}", p);
            }
            _ => panic!("must return TypeInstance"),
        }
    }
}
