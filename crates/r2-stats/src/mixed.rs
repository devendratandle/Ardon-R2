//! Linear mixed-effects models — Phase R.S.3.
//!
//! `lmer(y ~ x + (1|group), data=df)` fits a one-way random-intercept
//! mixed model `y_ij = X_ij β + u_i + ε_ij` where `u_i ~ N(0, σ²_u)`
//! is the group-i random intercept and `ε_ij ~ N(0, σ²_ε)` is
//! residual error.
//!
//! Estimation uses profiled REML with `θ = σ²_u / σ²_ε`. The marginal
//! per-group covariance is `V_i = σ²_ε (I + θ·J)` where J is the
//! all-ones matrix. We profile β and σ²_ε out of the REML
//! log-likelihood, leaving a 1-D optimization over θ ≥ 0.
//!
//! V_i⁻¹ has a closed form `V_i⁻¹ z = (1/σ²_ε)·[z - α_i · sum(z) · 1]`
//! with `α_i = θ / (1 + n_i·θ)`. This lets us compute β̂, σ²_ε, and
//! the REML profile log-likelihood in O(N·p) per θ — no matrix
//! inversions inside the optimizer loop.
//!
//! Random-slope and crossed/nested random effects are R.S.4 work.

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::{fmt_num, Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, TypeInstance};

use crate::{dist::phi, fmt_pval, signif_stars};

// ── Helpers ──────────────────────────────────────────────────────────

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

/// Unwrap a column wrapped as `List([(name, value)])` (the engine's
/// resolve_formula_term output) into the bare value.
fn unwrap_col(v: &RVal) -> RVal {
    match v {
        RVal::List(items) if !items.is_empty() => items[0].1.clone(),
        other => other.clone(),
    }
}

/// Coerce a column to a vector of string group labels.
fn labels_from(v: &RVal) -> Result<Vec<String>, R2Err> {
    let col = unwrap_col(v);
    Ok(match &col {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into()))
            .collect(),
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))
                .unwrap_or_else(|| "NA".into()))
            .collect(),
        _ => col.as_reals()?.iter().map(|x| x.map(fmt_num).unwrap_or_else(|| "NA".into())).collect(),
    })
}

/// Build the (n × p) row-major fixed-effect design matrix from the
/// formula's `~rhs` value, prepending an intercept column.
fn design_matrix(rhs: &RVal, n: usize) -> Result<(Vec<f64>, usize, Vec<String>), R2Err> {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();

    // Always include an intercept column.
    cols.push(vec![1.0; n]);
    names.push("(Intercept)".into());

    match rhs {
        RVal::Null => {}
        RVal::List(items) => {
            for (cname, cval) in items {
                if cname.as_ref().map(|s| s.starts_with('~')).unwrap_or(false) { continue; }
                let (col_name, col_data) = match cval {
                    RVal::List(inner) if !inner.is_empty() => {
                        let n0 = inner[0].0.as_ref().map(|s| s.to_string())
                            .unwrap_or_else(|| cname.as_ref().map(|s| s.to_string())
                                .unwrap_or_else(|| format!("x{}", cols.len())));
                        (n0, inner[0].1.clone())
                    }
                    _ => {
                        let n0 = cname.as_ref().map(|s| s.to_string())
                            .unwrap_or_else(|| format!("x{}", cols.len()));
                        (n0, cval.clone())
                    }
                };
                let vals: Vec<f64> = col_data.as_reals()?.into_iter()
                    .map(|x| x.unwrap_or(f64::NAN)).collect();
                if vals.len() != n {
                    return Err(R2Err {
                        msg: format!("lmer: predictor '{}' has length {} but response has {}", col_name, vals.len(), n),
                        kind: ErrKind::Runtime,
                    });
                }
                cols.push(vals);
                names.push(col_name);
            }
        }
        RVal::Matrix(m) => {
            for c in 0..m.ncol {
                let vals: Vec<f64> = (0..m.nrow).map(|r| m.get(r, c)).collect();
                cols.push(vals);
                let name = m.col_names.as_ref().and_then(|cn| cn.get(c))
                    .map(|s| s.to_string()).unwrap_or_else(|| format!("x{}", cols.len() - 1));
                names.push(name);
            }
        }
        _ => {
            let vals: Vec<f64> = rhs.as_reals()?.into_iter()
                .map(|x| x.unwrap_or(f64::NAN)).collect();
            cols.push(vals);
            names.push("x1".into());
        }
    }

    let p = cols.len();
    let mut data = vec![0.0; n * p];
    for j in 0..p {
        for i in 0..n {
            data[i * p + j] = cols[j][i];
        }
    }
    Ok((data, p, names))
}

// ── Core REML profile likelihood ─────────────────────────────────────

/// State per random-effect group used inside the optimizer.
struct GroupInfo {
    indices: Vec<usize>,
    n_i: usize,
}

/// Compute the REML profile log-likelihood at θ = σ²_u / σ²_ε.
/// Returns (log_lik, β̂, σ²_ε, X'V⁻¹X). Used inside the optimizer.
fn reml_profile(
    y: &[f64],
    x: &[f64],           // row-major (n × p)
    n: usize,
    p: usize,
    groups: &[GroupInfo],
    theta: f64,
) -> Option<(f64, Vec<f64>, f64, Vec<f64>)> {
    // α_i = θ / (1 + n_i · θ) — the "shrinkage" coefficient per group.
    let alpha: Vec<f64> = groups.iter()
        .map(|g| theta / (1.0 + g.n_i as f64 * theta))
        .collect();

    // Compute the V⁻¹-transformed versions of y and X.
    // V_i⁻¹ z (per group i) = z - α_i · sum_i(z) · 1
    // We just need X'V⁻¹X, X'V⁻¹y, y'V⁻¹y, log|V|. All are O(N·p²).

    // For each group i, precompute sum of y and sum of each X col.
    let mut sum_y_g = vec![0.0; groups.len()];
    let mut sum_x_g = vec![0.0; groups.len() * p];
    for (gi, g) in groups.iter().enumerate() {
        let mut sy = 0.0;
        let mut sx = vec![0.0; p];
        for &i in &g.indices {
            sy += y[i];
            for j in 0..p { sx[j] += x[i * p + j]; }
        }
        sum_y_g[gi] = sy;
        for j in 0..p { sum_x_g[gi * p + j] = sx[j]; }
    }

    // X'V⁻¹X = X'X - Σ_g α_g · (sum_X_g)(sum_X_g)'  (p × p, column-major for dpotrf)
    let mut xtvix = vec![0.0; p * p];
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a];
            for b in 0..p {
                xtvix[b * p + a] += xa * x[i * p + b];
            }
        }
    }
    for (gi, &a_g) in alpha.iter().enumerate() {
        for a in 0..p {
            let sxa = sum_x_g[gi * p + a];
            for b in 0..p {
                let sxb = sum_x_g[gi * p + b];
                xtvix[b * p + a] -= a_g * sxa * sxb;
            }
        }
    }

    // X'V⁻¹y = X'y - Σ_g α_g · sum_X_g · sum_y_g
    let mut xtviy = vec![0.0; p];
    for i in 0..n {
        for j in 0..p { xtviy[j] += x[i * p + j] * y[i]; }
    }
    for (gi, &a_g) in alpha.iter().enumerate() {
        let sy = sum_y_g[gi];
        for j in 0..p { xtviy[j] -= a_g * sum_x_g[gi * p + j] * sy; }
    }

    // y'V⁻¹y = y'y - Σ_g α_g · sum_y_g²
    let mut ytviy = 0.0;
    for &yi in y { ytviy += yi * yi; }
    for (gi, &a_g) in alpha.iter().enumerate() {
        ytviy -= a_g * sum_y_g[gi] * sum_y_g[gi];
    }

    // Solve X'V⁻¹X β = X'V⁻¹y. Use Cholesky for symmetric PD.
    let mut chol = xtvix.clone();
    if r2_linalg::dpotrf(p, &mut chol).is_err() {
        return None;
    }
    // Forward solve L w = X'V⁻¹y
    let mut w = vec![0.0; p];
    for i in 0..p {
        let mut s = xtviy[i];
        for k in 0..i { s -= chol[k * p + i] * w[k]; }
        w[i] = s / chol[i * p + i];
    }
    // Back solve L' β = w
    let mut beta = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = w[i];
        for k in (i + 1)..p { s -= chol[i * p + k] * beta[k]; }
        beta[i] = s / chol[i * p + i];
    }

    // Residual SS: y'V⁻¹y - β'X'V⁻¹y
    let mut beta_dot = 0.0;
    for j in 0..p { beta_dot += beta[j] * xtviy[j]; }
    let rss = ytviy - beta_dot;
    let dfe = (n - p) as f64;
    if dfe <= 0.0 || rss <= 0.0 { return None; }
    let sigma2 = rss / dfe;

    // log|V| / σ²_ε is constant in σ²_ε; for the profile log-lik:
    //   log|V_total / σ²_ε| = Σ_i log(1 + n_i · θ)
    //   log|X'V⁻¹X / σ²_ε^?| handled via Cholesky diagonal
    // -2·log L_REML(θ) = (n-p) log(σ²_ε) + Σ log(1+n_i θ) + 2 Σ log L_ii + const
    let log_det_v_over_s = groups.iter()
        .map(|g| (1.0 + g.n_i as f64 * theta).ln())
        .sum::<f64>();
    let log_det_xvx = 2.0 * (0..p).map(|i| chol[i * p + i].ln()).sum::<f64>();

    // REML log-likelihood — matches lme4's `REML criterion at convergence`.
    // Derivation: at the σ²_ε MLE we have (y-Xβ̂)'V⁻¹(y-Xβ̂)/σ²_ε = (n-p).
    // -2·log L_REML = (n-p)·[log(2π) + log(σ²) + 1]
    //                 + log|V/σ²_ε| + log|X'V⁻¹X|
    // The `+1·(n-p)` term is what was missing earlier; lme4's
    // `getME(model, "REML")` uses exactly this convention.
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    let m2_loglik = dfe * (log2pi + sigma2.ln() + 1.0)
        + log_det_v_over_s
        + log_det_xvx;
    let loglik = -0.5 * m2_loglik;
    Some((loglik, beta, sigma2, xtvix))
}

/// Brent's method maximizing `reml_profile` over θ ∈ [θ_lo, θ_hi]
/// (works in log-space for stability across orders of magnitude).
fn optimize_reml(
    y: &[f64], x: &[f64], n: usize, p: usize, groups: &[GroupInfo],
) -> Result<(f64, f64, Vec<f64>, f64, Vec<f64>), R2Err> {
    // 1-D golden-section search over log10(θ) ∈ [-6, 6]
    let f = |log_theta: f64| -> f64 {
        let theta = 10f64.powf(log_theta);
        match reml_profile(y, x, n, p, groups, theta) {
            Some((ll, _, _, _)) => -ll, // minimize -loglik
            None => 1e18,
        }
    };

    let mut a = -6.0f64;
    let mut b = 6.0f64;
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // ≈ 0.618
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..80 {
        if fc < fd {
            b = d; d = c; fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
        if (b - a).abs() < 1e-8 { break; }
    }
    let log_theta = 0.5 * (a + b);
    let theta = 10f64.powf(log_theta);

    let (loglik, beta, sigma2, xtvix) = reml_profile(y, x, n, p, groups, theta)
        .ok_or_else(|| R2Err {
            msg: "lmer: REML profile failed to evaluate at optimal θ".into(),
            kind: ErrKind::Runtime,
        })?;
    Ok((theta, loglik, beta, sigma2, xtvix))
}

// ── BLUPs (random-intercept estimates) ───────────────────────────────

/// Best linear unbiased predictors of u_i for the random-intercept model:
///   û_i = (n_i · θ / (1 + n_i · θ)) · (ȳ_i - X̄_i · β̂)
fn compute_blups(
    y: &[f64], x: &[f64], _n: usize, p: usize,
    groups: &[GroupInfo], theta: f64, beta: &[f64],
) -> Vec<f64> {
    groups.iter().map(|g| {
        let n_i = g.n_i as f64;
        let mut sum_y = 0.0;
        let mut sum_xb = 0.0;
        for &i in &g.indices {
            sum_y += y[i];
            let mut xb = 0.0;
            for j in 0..p { xb += x[i * p + j] * beta[j]; }
            sum_xb += xb;
        }
        let mean_resid = (sum_y - sum_xb) / n_i;
        (n_i * theta) / (1.0 + n_i * theta) * mean_resid
    }).collect()
}

// ── `bi_lmer` — public builtin ───────────────────────────────────────

pub fn bi_lmer(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);
    let items: Vec<(Option<Arc<str>>, RVal)> = match &first {
        RVal::List(v) => v.clone(),
        _ => return Err(R2Err {
            msg: "lmer: first argument must be a formula like 'y ~ x + (1|group)'".into(),
            kind: ErrKind::Runtime,
        }),
    };
    let lhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
        .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
    let rhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
        .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);

    let ranef_specs: Vec<&RVal> = items.iter()
        .filter(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~random_intercept"))
        .map(|(_, v)| v).collect();

    if ranef_specs.is_empty() {
        return Err(R2Err {
            msg: "lmer: formula must include at least one random-effect term, e.g. (1|group)".into(),
            kind: ErrKind::Runtime,
        });
    }
    if ranef_specs.len() > 1 {
        return Err(R2Err {
            msg: "lmer: multiple random-effect terms (crossed/nested) are R.S.4 work — v0.2.0 supports a single (1|group) only".into(),
            kind: ErrKind::Runtime,
        });
    }

    // y vector.
    let y_raw = unwrap_col(&lhs);
    let y: Vec<f64> = y_raw.as_reals()?.into_iter().filter_map(|x| x).collect();
    let n = y.len();
    if n < 4 {
        return Err(R2Err {
            msg: format!("lmer: need at least 4 observations (got {})", n),
            kind: ErrKind::Runtime,
        });
    }

    // Fixed-effect design (always includes intercept).
    let (x_data, p, x_names) = design_matrix(&rhs, n)?;

    // Random-effect grouping.
    let group_labels = labels_from(ranef_specs[0])?;
    if group_labels.len() != n {
        return Err(R2Err {
            msg: format!("lmer: response has length {} but grouping factor has {}", n, group_labels.len()),
            kind: ErrKind::Runtime,
        });
    }

    let mut group_names: Vec<String> = Vec::new();
    let mut group_indices: Vec<Vec<usize>> = Vec::new();
    for (i, g) in group_labels.iter().enumerate() {
        if let Some(pos) = group_names.iter().position(|x| x == g) {
            group_indices[pos].push(i);
        } else {
            group_names.push(g.clone());
            group_indices.push(vec![i]);
        }
    }
    let n_groups = group_names.len();
    if n_groups < 2 {
        return Err(R2Err {
            msg: format!("lmer: need at least 2 levels of the grouping factor (got {})", n_groups),
            kind: ErrKind::Runtime,
        });
    }
    let groups: Vec<GroupInfo> = group_indices.iter()
        .map(|idx| GroupInfo { indices: idx.clone(), n_i: idx.len() })
        .collect();

    // ── REML optimization ──────────────────────────────────────────
    let (theta, loglik, beta, sigma2, xtvix) = optimize_reml(&y, &x_data, n, p, &groups)?;
    let sigma2_u = theta * sigma2;
    let sigma_e = sigma2.sqrt();
    let sigma_u = sigma2_u.sqrt();

    // Std errors of β̂: σ²_ε · (X'V⁻¹X)⁻¹ via dgetri on the column-major xtvix.
    let xtvix_inv = r2_linalg::dgetri(p, &xtvix).map_err(|e| R2Err {
        msg: format!("lmer: X'V⁻¹X singular ({})", e),
        kind: ErrKind::Runtime,
    })?;
    let mut se = vec![0.0; p];
    let mut t_values = vec![0.0; p];
    let mut p_values = vec![1.0; p];
    let dfe = (n - p) as f64;
    for j in 0..p {
        let var_j = xtvix_inv[j * p + j] * sigma2;
        se[j] = if var_j > 0.0 { var_j.sqrt() } else { 0.0 };
        t_values[j] = if se[j] > 1e-15 { beta[j] / se[j] } else { 0.0 };
        // Two-tailed p-value via normal approximation (lme4's default for large n).
        let _ = dfe;
        p_values[j] = 2.0 * (1.0 - phi(t_values[j].abs()));
    }

    let blups = compute_blups(&y, &x_data, n, p, &groups, theta, &beta);

    // AIC / BIC. Parameters: p fixed + 2 variance components.
    let k_params = (p + 2) as f64;
    let aic = -2.0 * loglik + 2.0 * k_params;
    let bic = -2.0 * loglik + k_params * (n as f64).ln();
    let reml_criterion = -2.0 * loglik;

    // Compute fitted values, raw residuals, and scaled residuals for the
    // summary() expansion. Scaled = residual / σ_ε (lme4 convention).
    let mut fitted = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    let mut scaled_residuals = vec![0.0; n];
    for i in 0..n {
        let group_label = &group_labels[i];
        let gi = group_names.iter().position(|gn| gn == group_label).unwrap_or(0);
        let mut xb = 0.0;
        for j in 0..p { xb += x_data[i * p + j] * beta[j]; }
        let fi = xb + blups[gi];
        fitted[i] = fi;
        residuals[i] = y[i] - fi;
        scaled_residuals[i] = if sigma_e > 1e-15 { residuals[i] / sigma_e } else { 0.0 };
    }

    // Correlation matrix of the fixed-effect estimates (used by summary()).
    let mut corr_fixed = vec![0.0; p * p];
    for a_idx in 0..p {
        for b_idx in 0..p {
            let cov_ab = xtvix_inv[b_idx * p + a_idx] * sigma2;
            let denom = se[a_idx] * se[b_idx];
            corr_fixed[b_idx * p + a_idx] = if denom > 1e-15 { cov_ab / denom } else { 0.0 };
        }
    }

    // Group column name (used by both compact and summary displays).
    let group_col_name = match ranef_specs[0] {
        RVal::List(items) => items.iter()
            .find(|(n, _)| !n.as_ref().map(|s| s.starts_with('~')).unwrap_or(true))
            .and_then(|(n, _)| n.as_ref().map(|s| s.to_string()))
            .unwrap_or_else(|| "group".into()),
        _ => "group".into(),
    };

    // Compact print (matches R's default `print(m)`). The verbose summary
    // — scaled residuals, variance column, t-values, correlation matrix
    // — is produced by summary() via r2-engine's split-handler dispatch.
    print_lmer_compact(
        &group_col_name, sigma_u, sigma_e, n, n_groups,
        &x_names, &beta, reml_criterion,
    );

    // ── Build TypeInstance ─────────────────────────────────────────
    let mut fields = HashMap::new();
    fields.insert(Arc::from("method"), rstr("lmer (random-intercept REML)"));

    // Fixed effects as named numeric vector.
    let mut beta_attrs = Attrs::default();
    beta_attrs.names = Some(x_names.iter().map(|s| Arc::from(s.as_str())).collect());
    fields.insert(Arc::from("fixef"), RVal::Numeric(
        beta.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        beta_attrs,
    ));

    // BLUPs as named numeric vector keyed by group level.
    let mut blup_attrs = Attrs::default();
    blup_attrs.names = Some(group_names.iter().map(|s| Arc::from(s.as_str())).collect());
    fields.insert(Arc::from("ranef"), RVal::Numeric(
        blups.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        blup_attrs,
    ));

    fields.insert(Arc::from("sigma"), rnum(sigma_e));
    fields.insert(Arc::from("sigma.u"), rnum(sigma_u));
    fields.insert(Arc::from("sigma2"), rnum(sigma2));
    fields.insert(Arc::from("sigma2.u"), rnum(sigma2_u));
    fields.insert(Arc::from("theta"), rnum(theta));
    fields.insert(Arc::from("loglik"), rnum(loglik));
    fields.insert(Arc::from("aic"), rnum(aic));
    fields.insert(Arc::from("bic"), rnum(bic));
    fields.insert(Arc::from("reml.criterion"), rnum(reml_criterion));
    fields.insert(Arc::from("group.name"), rstr(&group_col_name));

    // Fitted, residuals, scaled residuals for summary()'s scaled-residuals block.
    fields.insert(Arc::from("fitted.values"), RVal::Numeric(
        fitted.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    fields.insert(Arc::from("residuals"), RVal::Numeric(
        residuals.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    fields.insert(Arc::from("scaled.residuals"), RVal::Numeric(
        scaled_residuals.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));

    // Correlation of fixed effects (stored as a matrix for summary()).
    fields.insert(Arc::from("corr.fixed"), RVal::Matrix(
        Matrix::new(corr_fixed, p, p)
    ));

    fields.insert(Arc::from("std.errors"), RVal::Numeric(
        se.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    fields.insert(Arc::from("t.values"), RVal::Numeric(
        t_values.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    fields.insert(Arc::from("p.values"), RVal::Numeric(
        p_values.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(),
        Attrs::default(),
    ));
    fields.insert(Arc::from("n"), rnum(n as f64));
    fields.insert(Arc::from("n.groups"), rnum(n_groups as f64));
    fields.insert(Arc::from("df.residual"), rnum(dfe));
    if let Some(call_str) = gn(a, "_call") {
        fields.insert(Arc::from("call"), call_str);
    }
    let _ = Matrix::new(vec![], 0, 0); // suppress unused-import warning

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("lmer"),
        fields,
    }))
}

// ── Display helpers — compact (used by lmer()) and full (used by summary()) ──

/// Compact print matching R's default `print(lmerMod)`: formula, REML
/// criterion, random effects (Std.Dev. only), fixed-effect coefficients
/// as a single row. No t-values, no scaled residuals, no correlation
/// matrix — those land in summary().
fn print_lmer_compact(
    group_col_name: &str,
    sigma_u: f64, sigma_e: f64,
    n: usize, n_groups: usize,
    x_names: &[String], beta: &[f64],
    reml_criterion: f64,
) {
    soutln!();
    soutln!("Linear mixed model fit by REML");
    soutln!("REML criterion at convergence: {}", fmt_num(reml_criterion));
    soutln!();
    soutln!("Random effects:");
    soutln!(" Groups   Name        Std.Dev.");
    soutln!(" {:<8} (Intercept) {:>8}", group_col_name, fmt_num(sigma_u));
    soutln!(" Residual             {:>8}", fmt_num(sigma_e));
    soutln!("Number of obs: {}, groups: {}, {}", n, group_col_name, n_groups);
    soutln!();
    soutln!("Fixed Effects:");
    // Print names on one line, values on the next, R-style.
    for nm in x_names { sout!("{:>12} ", nm); }
    soutln!();
    for b in beta { sout!("{:>12} ", fmt_num(*b)); }
    soutln!();
}

/// Verbose summary matching R's `summary(lmerMod)`: adds scaled residuals
/// quantiles, variance column on random effects, std.error + t-value on
/// fixed effects, and the correlation-of-fixed-effects block. Called by
/// r2-engine's `bi_summary` when it sees a TypeInstance of type "lmer".
pub fn format_lmer_summary(inst: &TypeInstance) -> Result<(), R2Err> {
    let f = |key: &str| inst.fields.get(key);
    let f_num = |key: &str| f(key).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(f64::NAN);
    let f_vec = |key: &str| -> Vec<f64> {
        f(key).and_then(|v| v.as_reals().ok())
            .map(|v| v.into_iter().map(|x| x.unwrap_or(f64::NAN)).collect())
            .unwrap_or_default()
    };

    let reml = f_num("reml.criterion");
    let sigma_u = f_num("sigma.u");
    let sigma2_u = f_num("sigma2.u");
    let sigma_e = f_num("sigma");
    let sigma2_e = f_num("sigma2");
    let n = f_num("n") as usize;
    let n_groups = f_num("n.groups") as usize;
    let group_col_name = f("group.name").map(|v| match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or("group".into()),
        _ => "group".into(),
    }).unwrap_or_else(|| "group".into());

    let beta = f_vec("fixef");
    let se   = f_vec("std.errors");
    let tv   = f_vec("t.values");
    let pv   = f_vec("p.values");
    let scaled = f_vec("scaled.residuals");

    let fixef_names: Vec<String> = match f("fixef") {
        Some(RVal::Numeric(_, at)) => at.names.as_ref()
            .map(|n| n.iter().map(|s| s.to_string()).collect())
            .unwrap_or_else(|| (0..beta.len()).map(|i| format!("x{}", i)).collect()),
        _ => (0..beta.len()).map(|i| format!("x{}", i)).collect(),
    };

    soutln!();
    soutln!("Linear mixed model fit by REML  ['lmerMod']");
    if let Some(call) = f("call") {
        if let RVal::Character(c, _) = call {
            if let Some(Some(s)) = c.first() {
                soutln!("Formula: {}", s);
            }
        }
    }
    soutln!();
    soutln!("REML criterion at convergence: {}", fmt_num(reml));
    soutln!();

    // Scaled residuals quantiles.
    if !scaled.is_empty() {
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = sorted.len();
        let q = |p: f64| {
            let idx = (p * (m - 1) as f64).round() as usize;
            sorted[idx.min(m - 1)]
        };
        soutln!("Scaled residuals:");
        soutln!("    {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Min", "1Q", "Median", "3Q", "Max");
        soutln!("    {:>8} {:>8} {:>8} {:>8} {:>8}",
            fmt_num(sorted[0]), fmt_num(q(0.25)), fmt_num(q(0.5)),
            fmt_num(q(0.75)), fmt_num(sorted[m - 1]));
        soutln!();
    }

    // Random effects with Variance and Std.Dev.
    soutln!("Random effects:");
    soutln!(" Groups   Name        Variance  Std.Dev.");
    soutln!(" {:<8} (Intercept) {:>8}  {:>8}",
        group_col_name, fmt_num(sigma2_u), fmt_num(sigma_u));
    soutln!(" Residual             {:>8}  {:>8}", fmt_num(sigma2_e), fmt_num(sigma_e));
    soutln!("Number of obs: {}, groups: {}, {}", n, group_col_name, n_groups);
    soutln!();

    // Fixed effects with Estimate / Std.Error / t value / Pr(>|t|).
    soutln!("Fixed effects:");
    soutln!("  {:<14} {:>10} {:>10} {:>10} {:>10}",
        "", "Estimate", "Std.Error", "t value", "Pr(>|t|)");
    for j in 0..beta.len() {
        let stars = if j < pv.len() { signif_stars(pv[j]) } else { "" };
        soutln!("  {:<14} {:>10} {:>10} {:>10} {:>10} {}",
            fixef_names[j],
            fmt_num(beta[j]),
            if j < se.len() { fmt_num(se[j]) } else { "".into() },
            if j < tv.len() { fmt_num(tv[j]) } else { "".into() },
            if j < pv.len() { fmt_pval(pv[j]) } else { "".into() },
            stars);
    }
    soutln!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
    soutln!();

    // Correlation of fixed effects.
    if let Some(RVal::Matrix(corr_mat)) = f("corr.fixed") {
        let p = corr_mat.nrow;
        if p > 1 {
            soutln!("Correlation of Fixed Effects:");
            // Header row.
            sout!("        ");
            for j in 0..p.saturating_sub(1) {
                let nm = fixef_names.get(j).map(|s| s.as_str()).unwrap_or("");
                let short: String = nm.chars().take(6).collect();
                sout!("{:>7} ", short);
            }
            soutln!();
            // Each subsequent row shows correlations with earlier coefficients.
            for i in 1..p {
                let nm = fixef_names.get(i).map(|s| s.as_str()).unwrap_or("");
                let short: String = nm.chars().take(6).collect();
                sout!("{:<7} ", short);
                for j in 0..i {
                    sout!("{:>7} ", fmt_num(corr_mat.get(i, j)));
                }
                soutln!();
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default())
    }
    fn chars(v: &[&str]) -> RVal {
        RVal::Character(v.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }
    fn formula(y: RVal, x: RVal, group: RVal) -> RVal {
        RVal::List(vec![
            (Some(Arc::from("~lhs")), y),
            (Some(Arc::from("~rhs")), RVal::List(vec![(Some(Arc::from("x")), x)])),
            (Some(Arc::from("~class")), RVal::Character(vec![Some(Arc::from("formula"))], Attrs::default())),
            (Some(Arc::from("~random_intercept")), RVal::List(vec![(Some(Arc::from("g")), group)])),
        ])
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn lmer_recovers_zero_random_effect_when_groups_are_homogeneous() {
        // y = 1 + 2x + noise, but no group effect — θ should be near 0.
        let y = nums(&[
            3.1, 5.0, 7.1, 9.0, 11.1,
            3.0, 5.1, 7.0, 9.1, 11.0,
        ]);
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let g = chars(&["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]);
        let r = bi_lmer(&[evarg(formula(y, x, g))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let theta = inst.fields.get("theta").unwrap().scalar_f64().unwrap().unwrap();
                let sigma_u = inst.fields.get("sigma.u").unwrap().scalar_f64().unwrap().unwrap();
                assert!(theta < 0.5, "θ should be small for homogeneous groups: got {}", theta);
                assert!(sigma_u < 0.5, "σ_u should be small: got {}", sigma_u);
            }
            _ => panic!("must return TypeInstance"),
        }
    }

    #[test]
    fn lmer_recovers_strong_random_effect_when_groups_differ() {
        // y = 1 + 2x + group_effect + noise; group A intercept is much
        // larger than group B intercept — θ should be large.
        let y = nums(&[
            10.1, 12.0, 14.1, 16.0, 18.1,    // group A: high intercept
            3.0,  5.1,  7.0,  9.1,  11.0,    // group B: low intercept
        ]);
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let g = chars(&["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]);
        let r = bi_lmer(&[evarg(formula(y, x, g))]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                let theta = inst.fields.get("theta").unwrap().scalar_f64().unwrap().unwrap();
                let sigma_u = inst.fields.get("sigma.u").unwrap().scalar_f64().unwrap().unwrap();
                assert!(theta > 1.0, "θ should be large for distinct groups: got {}", theta);
                assert!(sigma_u > 1.0, "σ_u should be large: got {}", sigma_u);
            }
            _ => panic!("must return TypeInstance"),
        }
    }

    #[test]
    fn lmer_errors_on_no_random_effect() {
        let y = nums(&[1.0, 2.0, 3.0, 4.0]);
        let x = nums(&[1.0, 2.0, 3.0, 4.0]);
        let formula_no_re = RVal::List(vec![
            (Some(Arc::from("~lhs")), y),
            (Some(Arc::from("~rhs")), RVal::List(vec![(Some(Arc::from("x")), x)])),
            (Some(Arc::from("~class")), RVal::Character(vec![Some(Arc::from("formula"))], Attrs::default())),
        ]);
        let r = bi_lmer(&[evarg(formula_no_re)]);
        assert!(r.is_err(), "lmer without (1|g) should error");
    }
}
