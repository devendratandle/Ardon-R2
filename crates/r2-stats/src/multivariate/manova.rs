//! MANOVA — Multivariate Analysis of Variance.
//!
//! Split out of `multivariate.rs` (was: small helpers + Hotelling T²
//! family + MANOVA + dispatcher). Shared helpers (`as_matrix`,
//! `col_means`, `f_to_pvalue`, `gv`, `rnum`, `rstr`) are reached via
//! `super::` — a child module sees its parent's items.

use std::collections::HashMap;
use std::sync::Arc;
use r2_types::*;
use super::{as_matrix, col_means, f_to_pvalue, gv, rnum, rstr};
use crate::htest::{fmt_pval, signif_stars};

// ── MANOVA — Multivariate Analysis of Variance ───────────────────────
//
// Tests whether mean *vectors* differ across groups. Given:
//   Y : n × p multivariate response
//   g : n-length grouping factor with k levels
//
// Compute the between-group and within-group sum-of-squares-and-cross-
// products matrices:
//   E = within-group SSCP (sum over groups of within-group deviations)
//   H = between-group SSCP (group means vs grand mean, weighted by size)
//   T = total SSCP        (E + H)
//
// Form the matrix M = E⁻¹H and extract its eigenvalues λ₁ ≥ ... ≥ λₛ
// where s = min(p, k-1). Four classical statistics:
//
//   Wilks' Λ       = ∏ 1/(1+λᵢ)              (small Λ → reject)
//   Pillai's V     = ∑ λᵢ/(1+λᵢ)
//   Hotelling-Law  = ∑ λᵢ
//   Roy's largest  = λ₁
//
// Each maps to an F-statistic via standard formulas. We report all four
// alongside their F-approximations.

/// Group rows of `data` (n × p, row-major) by `group_labels` and return
/// (groups: Vec<String>, group_indices: Vec<Vec<usize>>, sizes: Vec<usize>).
pub(super) fn group_rows(group_labels: &[String]) -> (Vec<String>, Vec<Vec<usize>>, Vec<usize>) {
    let mut groups: Vec<String> = Vec::new();
    let mut indices: Vec<Vec<usize>> = Vec::new();
    for (i, g) in group_labels.iter().enumerate() {
        if let Some(pos) = groups.iter().position(|x| x == g) {
            indices[pos].push(i);
        } else {
            groups.push(g.clone());
            indices.push(vec![i]);
        }
    }
    let sizes = indices.iter().map(|v| v.len()).collect();
    (groups, indices, sizes)
}

/// Compute mean vector for a subset of rows of a row-major (n × p) matrix.
pub(super) fn row_subset_mean(data: &[f64], p: usize, rows: &[usize]) -> Vec<f64> {
    let mut m = vec![0.0; p];
    for &i in rows {
        for j in 0..p {
            m[j] += data[i * p + j];
        }
    }
    let nf = rows.len() as f64;
    for j in 0..p { m[j] /= nf; }
    m
}

/// Compute E (within-group SSCP) and H (between-group SSCP) p×p matrices,
/// column-major to match r2_linalg conventions.
pub(super) fn compute_e_h(
    data: &[f64], n: usize, p: usize,
    group_indices: &[Vec<usize>],
) -> (Vec<f64>, Vec<f64>) {
    // Grand mean.
    let grand = col_means(data, n, p);

    // Group means.
    let group_means: Vec<Vec<f64>> = group_indices
        .iter()
        .map(|rows| row_subset_mean(data, p, rows))
        .collect();

    // E: within-group SSCP — sum over groups of Σ (y_ij - ȳ_g)(y_ij - ȳ_g)ᵀ
    let mut e = vec![0.0; p * p];
    for (g, rows) in group_indices.iter().enumerate() {
        let gm = &group_means[g];
        for &i in rows {
            for a in 0..p {
                let da = data[i * p + a] - gm[a];
                for b in 0..p {
                    let db = data[i * p + b] - gm[b];
                    e[b * p + a] += da * db; // column-major
                }
            }
        }
    }

    // H: between-group SSCP — Σ n_g · (ȳ_g - ȳ..)(ȳ_g - ȳ..)ᵀ
    let mut h = vec![0.0; p * p];
    for (g, rows) in group_indices.iter().enumerate() {
        let n_g = rows.len() as f64;
        let gm = &group_means[g];
        for a in 0..p {
            let da = gm[a] - grand[a];
            for b in 0..p {
                let db = gm[b] - grand[b];
                h[b * p + a] += n_g * da * db;
            }
        }
    }

    (e, h)
}

/// `manova(formula, data)`. The formula's LHS must be a matrix of
/// multivariate responses (typically via `cbind(y1, y2, ...)`); the
/// RHS is the grouping factor.
///
/// Returns a TypeInstance with the four standard test statistics
/// (Wilks' Lambda, Pillai's trace, Hotelling-Lawley trace, Roy's largest
/// root) and the F-approximation + p-value for Wilks' Lambda (the
/// most commonly reported). All eigenvalues are also returned as a
/// vector for users who want to compute their own statistics.
pub fn bi_manova(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);

    // Extract LHS, RHS from formula list.
    let items: Vec<(Option<Arc<str>>, RVal)> = match &first {
        RVal::List(v) => v.clone(),
        _ => return Err(R2Err {
            msg: "manova: first argument must be a formula 'cbind(y1, y2, ...) ~ group'".into(),
            kind: ErrKind::Runtime,
        }),
    };
    let lhs = items.iter()
        .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
        .map(|(_, v)| v.clone())
        .unwrap_or(RVal::Null);
    let rhs = items.iter()
        .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
        .map(|(_, v)| v.clone())
        .unwrap_or(RVal::Null);

    // LHS: build the n × p multivariate response matrix.
    // Engine wraps single columns in `List([(name, col)])`; manova needs
    // a Matrix on the LHS, so accept Matrix directly or unwrap List.
    let lhs_inner = match &lhs {
        RVal::List(items) if !items.is_empty() => items[0].1.clone(),
        other => other.clone(),
    };
    let (n, p, y) = as_matrix(&lhs_inner)?;
    if p < 2 {
        return Err(R2Err {
            msg: format!("manova: LHS must be a multivariate response (≥ 2 columns); got p = {}. Use cbind(y1, y2, ...) on the LHS.", p),
            kind: ErrKind::Runtime,
        });
    }

    // RHS: extract grouping labels.
    let rhs_inner = match &rhs {
        RVal::List(items) => items.iter()
            .find(|(name, _)| !name.as_ref().map(|s| s.starts_with('~')).unwrap_or(true))
            .map(|(_, v)| match v {
                RVal::List(inner) if !inner.is_empty() => inner[0].1.clone(),
                v => v.clone(),
            })
            .unwrap_or(RVal::Null),
        v => v.clone(),
    };
    let group_labels: Vec<String> = match &rhs_inner {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into()))
            .collect(),
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))
                .unwrap_or("NA".into()))
            .collect(),
        _ => {
            let nums = rhs_inner.as_reals()?;
            nums.iter().map(|x| x.map(fmt_num).unwrap_or("NA".into())).collect()
        }
    };
    if group_labels.len() != n {
        return Err(R2Err {
            msg: format!("manova: response has {} rows but grouping factor has {}", n, group_labels.len()),
            kind: ErrKind::Runtime,
        });
    }

    let (groups, group_indices, sizes) = group_rows(&group_labels);
    let k = groups.len();
    if k < 2 {
        return Err(R2Err {
            msg: format!("manova: need ≥ 2 groups (got {})", k),
            kind: ErrKind::Runtime,
        });
    }
    if n - k < p {
        return Err(R2Err {
            msg: format!("manova: insufficient residual degrees of freedom (n - k = {} < p = {}). E will be singular.", n - k, p),
            kind: ErrKind::Runtime,
        });
    }

    let (e, h) = compute_e_h(&y, n, p, &group_indices);

    // Compute M = E⁻¹ H, then its eigenvalues. Use dgetri to invert E
    // (E is symmetric positive-definite when n - k > p) and dgemm to
    // form the product. Eigenvalues via dsyev — actually M = E⁻¹H is
    // not symmetric in general, but its eigenvalues are real. We use
    // a workaround: the eigenvalues of E⁻¹H equal the generalized
    // eigenvalues of (H, E), which are real and computable via the
    // symmetric eigenvalue routine after a Cholesky-style transform.
    //
    // Cheap robust path: solve E·X = H column-by-column via dgesv, then
    // compute eigenvalues of X directly. For small p (typical: 2-10),
    // we use the simpler dgetri approach and a power-iteration-like
    // method... actually the simplest correct approach: form E⁻¹H, then
    // since trace(E⁻¹H) and det(E⁻¹H) are sufficient for the four
    // statistics via Newton's identities (for small p), we extract
    // eigenvalues via a small QR-iteration. r2-linalg's dsyev is for
    // symmetric matrices only.
    //
    // Pragmatic choice for v0.2.0: assume p ≤ 4 (covers the vast
    // majority of real-world MANOVA cases) and compute the eigenvalues
    // of E⁻¹H via the characteristic polynomial. For larger p, fall
    // through to a numerical QR iteration.
    // Phase R.S.2 — proper LAPACK-style eigenvalue computation.
    //
    // The eigenvalues of E⁻¹H equal the eigenvalues of the symmetric
    // matrix B = L⁻¹ H L⁻ᵀ where L is the Cholesky factor of E.
    // Using dsyev_full on B gives machine-precision eigenvalues,
    // dramatically more accurate than running QR iteration on the
    // non-symmetric E⁻¹H.
    //
    // Steps:
    //   1. Cholesky-factor E = L Lᵀ (dpotrf, in-place lower triangle)
    //   2. Solve L·X = H column-by-column (forward substitution) → X = L⁻¹H
    //   3. Solve L·Yᵀ = Xᵀ (forward substitution on transposed rows)
    //      Equivalently: Bᵢⱼ = (L⁻¹ H L⁻ᵀ)ᵢⱼ — symmetric
    //   4. Symmetric eigendecomposition via dsyev_full
    let mut l = e.clone();
    r2_linalg::dpotrf(p, &mut l).map_err(|e| R2Err {
        msg: format!("manova: within-group SSCP E is not positive definite ({:?}) — likely n - k < p or perfectly collinear columns", e),
        kind: ErrKind::Runtime,
    })?;

    // L is in lower triangle of `l` (column-major), upper zeroed by dpotrf.
    // Solve L · X = H, column by column, via forward substitution. The
    // result X overwrites a working buffer.
    let mut x = vec![0.0; p * p];
    for col in 0..p {
        // For column `col` of H, solve L · x = h.
        for i in 0..p {
            let mut sum = h[col * p + i];
            for k in 0..i {
                sum -= l[k * p + i] * x[col * p + k];
            }
            x[col * p + i] = sum / l[i * p + i];
        }
    }

    // Compute B = X · L⁻ᵀ where X = L⁻¹ H. Using the identity:
    //   B[i, j] = (row i of X) · (col j of L⁻ᵀ)
    //   (row i of B)ᵀ = L⁻¹ × (row i of X)ᵀ
    // So for each row i of X, solve L · v = (row i of X)ᵀ via forward
    // substitution; v becomes row i of B. Hand-verified for p=2:
    //   E=[4,2;2,5], H=[3,1;1,4] → B=[[0.75,-0.125],[-0.125,0.9375]],
    //   trace(B)=1.6875, det(B)=0.6875 — matches eigenvalues of E⁻¹H.
    let mut b = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            // X[i, j] in column-major = x[j*p+i]
            let mut sum = x[j * p + i];
            for k in 0..j {
                // L[j, k] = l[k*p+j], B[i, k] = b[k*p+i]
                sum -= l[k * p + j] * b[k * p + i];
            }
            // B[i, j] = b[j*p+i]
            b[j * p + i] = sum / l[j * p + j];
        }
    }

    // Symmetrize B (it's theoretically symmetric, but small numerical
    // asymmetry can creep in; explicit symmetrization stabilizes dsyev).
    for col in 0..p {
        for row in (col + 1)..p {
            let avg = (b[col * p + row] + b[row * p + col]) * 0.5;
            b[col * p + row] = avg;
            b[row * p + col] = avg;
        }
    }

    let mut eigenvalues = r2_linalg::dsyev(p, &b).map_err(|e| R2Err {
        msg: format!("manova: symmetric eigenvalue solver failed ({:?})", e),
        kind: ErrKind::Runtime,
    })?;
    // Sort eigenvalues descending — dsyev's order is not guaranteed across
    // matrix sizes, and the MANOVA statistics take the s = min(p, k-1)
    // *largest* eigenvalues.
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let s = eigenvalues.len().min(p.min(k - 1));
    let lam: Vec<f64> = eigenvalues.into_iter().take(s).collect();

    // Four classical test statistics.
    let wilks: f64 = lam.iter().fold(1.0, |acc, &li| acc / (1.0 + li));
    let pillai: f64 = lam.iter().map(|&li| li / (1.0 + li)).sum();
    let hotelling_lawley: f64 = lam.iter().sum();
    let roy_lambda: f64 = lam.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // ── F-approximations via Rao's formulas (Rencher 2002, §4.10) ──
    //
    // For all four statistics:
    //   v_h = k - 1   (hypothesis df)
    //   v_e = n - k   (error df)
    //   s   = min(p, v_h)
    //   M   = (|v_h - p| - 1) / 2
    //   N   = (v_e - p - 1) / 2
    //   q   = v_h (kept separate from p to match Rao's notation)

    let q = (k - 1) as f64;
    let pf = p as f64;
    let nf = n as f64;
    let v_h = q;
    let v_e = nf - k as f64;
    let s_count = (pf as i64).min(v_h as i64) as f64;
    let m_rao_pillai = ((v_h - pf).abs() - 1.0) / 2.0;
    let n_rao_pillai = (v_e - pf - 1.0) / 2.0;

    // Wilks' Lambda — Rao's F:
    //   s_w = √((p²q² - 4)/(p² + q² - 5)) — degenerate when p² + q² ≤ 5
    //   df1 = pq
    //   df2 = m_w · s_w - pq/2 + 1   where m_w = v_e - 1 - (p - q + 1)/2
    let denom_w = pf * pf + q * q - 5.0;
    let s_w = if denom_w <= 0.0 {
        1.0
    } else {
        ((pf * pf * q * q - 4.0) / denom_w).sqrt()
    };
    let m_w = v_e - 1.0 - (pf - q + 1.0) / 2.0;
    let df1_wilks = pf * q;
    let df2_wilks = m_w * s_w - pf * q / 2.0 + 1.0;
    let wilks_root = wilks.powf(1.0 / s_w);
    let f_wilks = ((1.0 - wilks_root) / wilks_root) * (df2_wilks / df1_wilks);
    let p_wilks = f_to_pvalue(f_wilks, df1_wilks, df2_wilks);

    // Pillai's trace V:
    //   F  = ((2N + s + 1)/(2M + s + 1)) · (V/(s - V))
    //   df = (s(2M + s + 1), s(2N + s + 1))
    let df1_pillai = s_count * (2.0 * m_rao_pillai + s_count + 1.0);
    let df2_pillai = s_count * (2.0 * n_rao_pillai + s_count + 1.0);
    let f_pillai = if s_count - pillai > 1e-12 {
        ((2.0 * n_rao_pillai + s_count + 1.0) / (2.0 * m_rao_pillai + s_count + 1.0))
            * (pillai / (s_count - pillai))
    } else {
        f64::INFINITY
    };
    let p_pillai = f_to_pvalue(f_pillai, df1_pillai, df2_pillai);

    // Hotelling-Lawley trace U:
    //   F  = (2(sN + 1) · U) / (s²(2M + s + 1))
    //   df = (s(2M + s + 1), 2(sN + 1))
    let df1_hl = s_count * (2.0 * m_rao_pillai + s_count + 1.0);
    let df2_hl = 2.0 * (s_count * n_rao_pillai + 1.0);
    let f_hl = if df2_hl > 0.0 && s_count > 0.0 {
        2.0 * (s_count * n_rao_pillai + 1.0) * hotelling_lawley
            / (s_count * s_count * (2.0 * m_rao_pillai + s_count + 1.0))
    } else {
        f64::INFINITY
    };
    let p_hl = f_to_pvalue(f_hl, df1_hl, df2_hl);

    // Roy's largest root — upper bound F approximation (Rencher §4.10.5):
    //   F   = θ · (v_e - d + s) / d   where d = max(p, v_h)
    //   df  = (d, v_e - d + s)
    // This is the standard but conservative; p-value is an *upper bound*.
    let d = pf.max(v_h);
    let df1_roy = d;
    let df2_roy = v_e - d + s_count;
    let f_roy = if df2_roy > 0.0 {
        roy_lambda * df2_roy / d
    } else {
        f64::INFINITY
    };
    let p_roy = f_to_pvalue(f_roy, df1_roy, df2_roy);

    // ── Print R-style multi-statistic table ─────────────────────────
    soutln!("\nMANOVA test summary\n");
    soutln!("{:<18} {:>10} {:>10} {:>5} {:>6} {:>10}",
        "Statistic", "value", "approx F", "num", "den", "Pr(>F)");
    soutln!("{:<18} {:>10} {:>10} {:>5} {:>6} {:>10} {}",
        "Pillai's trace",
        fmt_num(pillai), fmt_num(f_pillai),
        df1_pillai as i32, df2_pillai as i32,
        fmt_pval(p_pillai), signif_stars(p_pillai));
    soutln!("{:<18} {:>10} {:>10} {:>5} {:>6} {:>10} {}",
        "Wilks' Lambda",
        fmt_num(wilks), fmt_num(f_wilks),
        df1_wilks as i32, df2_wilks as i32,
        fmt_pval(p_wilks), signif_stars(p_wilks));
    soutln!("{:<18} {:>10} {:>10} {:>5} {:>6} {:>10} {}",
        "Hotelling-Lawley",
        fmt_num(hotelling_lawley), fmt_num(f_hl),
        df1_hl as i32, df2_hl as i32,
        fmt_pval(p_hl), signif_stars(p_hl));
    soutln!("{:<18} {:>10} {:>10} {:>5} {:>6} {:>10} {}",
        "Roy's largest root",
        fmt_num(roy_lambda), fmt_num(f_roy),
        df1_roy as i32, df2_roy as i32,
        fmt_pval(p_roy), signif_stars(p_roy));
    soutln!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
    soutln!();
    soutln!("n = {}, groups = {}, p = {}, error df = {}",
        n, k, p, v_e as i32);

    // Eigenvalues of E⁻¹H (useful for understanding which dimensions
    // drive the result).
    sout!("eigenvalues of E⁻¹H: ");
    for (i, l) in lam.iter().enumerate() {
        if i > 0 { sout!(", "); }
        sout!("{}", fmt_num(*l));
    }
    soutln!();

    // ── Interpretation guidance — what to focus on for this design ──
    //
    // The four statistics agree when groups are well-separated and
    // assumptions hold. When they disagree, the choice matters:
    //   - Pillai is the most robust under heteroscedasticity and
    //     non-normality. Prefer for small samples or when MV-normality
    //     is doubtful.
    //   - Wilks is the maximum-likelihood test; preferred when MV-normal
    //     and equal covariances hold. Most commonly reported.
    //   - Hotelling-Lawley has slightly more power than Wilks when
    //     assumptions hold.
    //   - Roy's largest root has maximum power when the effect is
    //     concentrated along a single dimension; p-value is an *upper
    //     bound* (i.e., possibly anti-conservative).
    soutln!();
    soutln!("Interpretation:");
    let preferred = if v_e < 2.0 * pf {
        "Pillai's trace (small n relative to p — most robust)"
    } else if (s_count as i64) == 1 {
        "Wilks' Lambda or Pillai's trace (s = 1; statistics are algebraically equivalent here)"
    } else if roy_lambda > 2.0 * (hotelling_lawley - roy_lambda).max(1e-12) {
        "Roy's largest root (one eigenvalue dominates — effect concentrated along one dimension)"
    } else {
        "Pillai's trace or Wilks' Lambda (effect is diffuse across dimensions)"
    };
    soutln!("  Suggested primary statistic: {}.", preferred);
    if (p_pillai > 0.05) != (p_wilks > 0.05) || (p_pillai > 0.05) != (p_hl > 0.05) {
        soutln!("  CAUTION: the four statistics disagree on significance at α=0.05.");
        soutln!("           Heteroscedasticity or non-MV-normality likely. Trust Pillai.");
    } else if p_wilks < 0.001 {
        soutln!("  All four statistics agree: highly significant multivariate effect.");
    } else if p_wilks < 0.05 {
        soutln!("  All four statistics agree: significant multivariate effect.");
    } else {
        soutln!("  All four statistics agree: no significant multivariate effect detected.");
    }
    let _ = sizes;

    let mut fields = HashMap::new();
    fields.insert(Arc::from("method"), rstr("MANOVA"));

    // Statistic values
    fields.insert(Arc::from("pillai.trace"), rnum(pillai));
    fields.insert(Arc::from("wilks.lambda"), rnum(wilks));
    fields.insert(Arc::from("hotelling.lawley"), rnum(hotelling_lawley));
    fields.insert(Arc::from("roy.largest"), rnum(roy_lambda));

    // F-approximations + dfs + p-values for each statistic
    fields.insert(Arc::from("f.pillai"), rnum(f_pillai));
    fields.insert(Arc::from("df1.pillai"), rnum(df1_pillai));
    fields.insert(Arc::from("df2.pillai"), rnum(df2_pillai));
    fields.insert(Arc::from("p.pillai"), rnum(p_pillai));

    fields.insert(Arc::from("f.wilks"), rnum(f_wilks));
    fields.insert(Arc::from("df1.wilks"), rnum(df1_wilks));
    fields.insert(Arc::from("df2.wilks"), rnum(df2_wilks));
    fields.insert(Arc::from("p.wilks"), rnum(p_wilks));
    // Backwards-compatible: `p.value` defaults to Wilks.
    fields.insert(Arc::from("p.value"), rnum(p_wilks));

    fields.insert(Arc::from("f.hotelling.lawley"), rnum(f_hl));
    fields.insert(Arc::from("df1.hotelling.lawley"), rnum(df1_hl));
    fields.insert(Arc::from("df2.hotelling.lawley"), rnum(df2_hl));
    fields.insert(Arc::from("p.hotelling.lawley"), rnum(p_hl));

    fields.insert(Arc::from("f.roy"), rnum(f_roy));
    fields.insert(Arc::from("df1.roy"), rnum(df1_roy));
    fields.insert(Arc::from("df2.roy"), rnum(df2_roy));
    fields.insert(Arc::from("p.roy"), rnum(p_roy));

    // Design info
    fields.insert(Arc::from("n"), rnum(nf));
    fields.insert(Arc::from("p.dim"), rnum(pf));
    fields.insert(Arc::from("k.groups"), rnum(k as f64));
    fields.insert(Arc::from("eigenvalues"), RVal::Numeric(
        lam.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("manova"),
        fields,
    }))
}
