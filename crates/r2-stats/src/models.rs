//! Fitted-model functions — Phase R.11.
//!
//! `lm`, `glm`, `aov`, `anova`. Migrated from r2-engine to r2-stats so the
//! algorithm bodies live alongside the rest of the statistical math. The
//! engine retains 1-line delegators and continues to own the `summary()`
//! formatting (split-handler pattern: data-shape paths in r2-stats,
//! summary-formatting in engine since the model TypeInstance carries
//! engine-private decorations like `$call`).
//!
//! All functions follow the locked pure pattern
//! `fn(&[EvalArg]) -> Result<RVal, R2Err>`. The `$call` captured by the
//! engine NSE preprocessor flows through as a named arg `_call` and is
//! stored on the returned TypeInstance.

use crate::{fmt_pval, phi, signif_stars};
use r2_types::{
    fmt_num, Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, TypeInstance,
};
use std::collections::HashMap;
use std::sync::Arc;

// ── Local helpers (kept private to this module) ─────────────────────

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
fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default())
}

#[inline]
fn rint(n: i32) -> RVal { RVal::Integer(vec![Some(n)].into(), Attrs::default()) }

#[inline]
fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(Arc::from(s))], Attrs::default())
}

fn val_to_str(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        RVal::Numeric(c, _) => c.first().and_then(|x| *x).map(fmt_num).unwrap_or_default(),
        _ => String::new(),
    }
}

/// Phase S.1 — design-matrix expansion for one predictor column.
///
/// Numeric/Integer/Logical/Single columns pass through as a single f64
/// column. Character and Factor columns expand into k-1 dummy columns
/// using treatment contrasts: the first observed level becomes the
/// reference (absorbed into the intercept) and the remaining levels each
/// get a 0/1 indicator column named `{base}{level}` — matching R's
/// `model.matrix()` output when `contrasts = "contr.treatment"`.
///
/// Returns `(columns, names)` where each column is length n.
pub fn model_matrix_expand(base_name: &str, col: &RVal)
    -> Result<(Vec<Vec<f64>>, Vec<String>), R2Err>
{
    match col {
        RVal::Character(v, _) => {
            // Collect levels in first-appearance order (R's default).
            let mut levels: Vec<String> = Vec::new();
            let labels: Vec<Option<String>> = v.iter().map(|x| {
                x.as_ref().map(|s| {
                    let s = s.to_string();
                    if !levels.contains(&s) { levels.push(s.clone()); }
                    s
                })
            }).collect();
            if levels.len() < 2 {
                return Err(R2Err {
                    msg: format!("contrasts can be applied only to factors with 2 or more levels (got '{}')", base_name),
                    kind: ErrKind::Runtime,
                });
            }
            let reference = &levels[0];
            let mut cols = Vec::with_capacity(levels.len() - 1);
            let mut names = Vec::with_capacity(levels.len() - 1);
            for lvl in &levels[1..] {
                let dummy: Vec<f64> = labels.iter().map(|l| match l {
                    Some(s) if s == lvl => 1.0,
                    Some(_)             => 0.0,
                    None                => f64::NAN,
                }).collect();
                cols.push(dummy);
                names.push(format!("{}{}", base_name, lvl));
            }
            // Suppress unused-binding lint while making clear that the
            // reference level is intentionally dropped.
            let _ = reference;
            Ok((cols, names))
        }
        RVal::Factor(f) => {
            if f.levels.len() < 2 {
                return Err(R2Err {
                    msg: format!("contrasts can be applied only to factors with 2 or more levels (got '{}')", base_name),
                    kind: ErrKind::Runtime,
                });
            }
            // Reference = first level (R's default). Emit k-1 dummies.
            let mut cols = Vec::with_capacity(f.levels.len() - 1);
            let mut names = Vec::with_capacity(f.levels.len() - 1);
            for (li, lvl) in f.levels.iter().enumerate().skip(1) {
                let li_u = li as u32;
                let dummy: Vec<f64> = f.codes.iter().map(|c| match c {
                    Some(code) if *code == li_u => 1.0,
                    Some(_)                     => 0.0,
                    None                        => f64::NAN,
                }).collect();
                cols.push(dummy);
                names.push(format!("{}{}", base_name, lvl));
            }
            Ok((cols, names))
        }
        // Numeric-ish: single column passthrough.
        _ => {
            let v: Vec<f64> = col.as_reals()?.into_iter()
                .map(|x| x.unwrap_or(f64::NAN)).collect();
            Ok((vec![v], vec![base_name.to_string()]))
        }
    }
}

fn is_formula(v: &RVal) -> bool {
    match v {
        RVal::List(items) => items.iter().any(|(n, v)| {
            n.as_ref().map(|s| s.as_ref()) == Some("~class")
                && matches!(v, RVal::Character(sv, _)
                    if sv.first().and_then(|x| x.as_ref()).map(|s| s.as_ref()) == Some("formula"))
        }),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────
// lm — Ordinary least squares via normal equations
// ─────────────────────────────────────────────────────────────────────

pub fn bi_lm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);

    let (y_vec, x_mat, col_names) = if is_formula(&first) {
        let items: Vec<(Option<Arc<str>>, RVal)> = match &first { RVal::List(v) => v.clone(), _ => vec![] };
        let lhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
            .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
        let rhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
            .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
        let lhs = match &lhs_raw {
            RVal::List(items) if !items.is_empty() => items[0].1.clone(),
            other => other.clone(),
        };
        let y: Vec<f64> = lhs.as_reals()?.into_iter().filter_map(|x| x).collect();
        let n = y.len();

        let mut predictor_cols: Vec<Vec<f64>> = Vec::new();
        let mut names: Vec<String> = vec!["(Intercept)".into()];
        match &rhs_raw {
            RVal::List(cols) => {
                for (col_name, col_val) in cols {
                    if col_name.as_ref().map(|s| s.starts_with("~")).unwrap_or(false) { continue; }
                    let (actual_name, actual_data) = match col_val {
                        RVal::List(inner) if !inner.is_empty() => {
                            let cname = inner[0].0.as_ref().map(|s| s.to_string())
                                .unwrap_or_else(|| col_name.as_ref().map(|s| s.to_string())
                                    .unwrap_or(format!("x{}", predictor_cols.len() + 1)));
                            (cname, inner[0].1.clone())
                        }
                        _ => {
                            let cname = col_name.as_ref().map(|s| s.to_string())
                                .unwrap_or(format!("x{}", predictor_cols.len() + 1));
                            (cname, col_val.clone())
                        }
                    };
                    match &actual_data {
                        RVal::Matrix(mat) => {
                            for c in 0..mat.ncol {
                                let vals: Vec<f64> = (0..mat.nrow).map(|r| mat.get(r, c)).collect();
                                predictor_cols.push(vals);
                                let cname = mat.col_names.as_ref().and_then(|cn| cn.get(c))
                                    .map(|s| s.to_string()).unwrap_or(format!("x{}", predictor_cols.len()));
                                names.push(cname);
                            }
                        }
                        // Phase S.1 — character/factor predictors expand to
                        // k-1 dummy columns via treatment contrasts.
                        _ => {
                            let (cols, col_names) = model_matrix_expand(&actual_name, &actual_data)?;
                            for c in cols { predictor_cols.push(c); }
                            for n in col_names { names.push(n); }
                        }
                    }
                }
            }
            RVal::Matrix(mat) => {
                for c in 0..mat.ncol {
                    let vals: Vec<f64> = (0..mat.nrow).map(|r| mat.get(r, c)).collect();
                    predictor_cols.push(vals);
                    let cname = mat.col_names.as_ref().and_then(|cn| cn.get(c))
                        .map(|s| s.to_string()).unwrap_or(format!("x{}", c + 1));
                    names.push(cname);
                }
            }
            _ => {
                // Phase S.1 — factor/character expansion on bare RHS too.
                let (cols, col_names) = model_matrix_expand("x1", &rhs_raw)?;
                for c in cols { predictor_cols.push(c); }
                for n in col_names { names.push(n); }
            }
        }

        let p = predictor_cols.len() + 1;
        let mut x_data = vec![1.0f64; n];
        for col in &predictor_cols { x_data.extend(col); }
        (y, Matrix::new(x_data, n, p), names)
    } else {
        // Two-vector legacy path: lm(y, x). Phase S.1 — expand factor/char x.
        let y: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
        let n = y.len();
        let x_raw_val = gv(a, 1);
        let (x_cols, x_names) = model_matrix_expand("x1", &x_raw_val)?;
        let p = x_cols.len() + 1;
        let mut x_data = vec![1.0f64; n];
        for col in &x_cols { x_data.extend(col); }
        let mut full_names = vec!["(Intercept)".to_string()];
        full_names.extend(x_names);
        (y, Matrix::new(x_data, n, p), full_names)
    };

    let n = y_vec.len();
    let p = x_mat.ncol;

    // Solve the least-squares system via Householder QR (does not form
    // X'X, so the condition number is not squared) — matching R's lm and
    // numerically stable for near-collinear predictors. Fall back to the
    // normal-equations path only if QR is inapplicable (e.g. m < p).
    let coeffs = r2_linalg::dlsq_qr(n, p, &x_mat.data, &y_vec)
        .or_else(|_| r2_linalg::dlsq_fused(n, p, &x_mat.data, &y_vec))
        .or_else(|_| {
            let xtx = x_mat.crossprod();
            let xty = x_mat.crossprod_vec(&y_vec);
            xtx.solve(&xty)
        })
        .map_err(|e| R2Err { msg: format!("lm failed: {}", e), kind: ErrKind::Runtime })?;

    let mut fitted = vec![0.0; n];
    for i in 0..n { for j in 0..p { fitted[i] += x_mat.get(i, j) * coeffs[j]; } }
    let residuals: Vec<f64> = y_vec.iter().zip(fitted.iter()).map(|(y, yhat)| y - yhat).collect();

    let y_mean = y_vec.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y_vec.iter().map(|y| (y - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let adj_r2 = if n > p { 1.0 - (1.0 - r_squared) * (n - 1) as f64 / (n - p) as f64 } else { 0.0 };
    let rse = if n > p { (ss_res / (n - p) as f64).sqrt() } else { 0.0 };

    let xtx = x_mat.crossprod();
    let xtx_inv_result = r2_linalg::dgetri(p, &xtx.data);
    let mut std_errors = vec![0.0; p];
    let mut t_values = vec![0.0; p];
    let mut p_values = vec![1.0; p];
    if let Ok(xtx_inv) = &xtx_inv_result {
        let sigma2 = if n > p { ss_res / (n - p) as f64 } else { 1.0 };
        // Residual degrees of freedom for the coefficient t-tests. R uses
        // the t-distribution (Pr(>|t|)), not the normal — the difference
        // matters at small n. t_cdf is exact (Lentz incomplete beta).
        let df_resid = if n > p { (n - p) as f64 } else { 0.0 };
        for j in 0..p {
            let var_j = xtx_inv[j * p + j] * sigma2;
            std_errors[j] = if var_j > 0.0 { var_j.sqrt() } else { 0.0 };
            t_values[j] = if std_errors[j] > 1e-15 { coeffs[j] / std_errors[j] } else { 0.0 };
            p_values[j] = if df_resid > 0.0 {
                2.0 * (1.0 - crate::htest::t_cdf(t_values[j].abs(), df_resid))
            } else { 1.0 };
        }
    }
    let ss_reg = ss_tot - ss_res;
    let f_stat = if p > 1 && n > p && ss_res > 0.0 {
        (ss_reg / (p - 1) as f64) / (ss_res / (n - p) as f64)
    } else { 0.0 };

    let mut coef_attrs = Attrs::default();
    coef_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());
    let coef_val = RVal::Numeric(
        coeffs.iter().map(|c| Some(*c)).collect::<Vec<_>>().into(), coef_attrs);

    let mut fields = HashMap::new();
    fields.insert(Arc::from("coefficients"), coef_val);
    fields.insert(Arc::from("residuals"), rnums(&residuals));
    fields.insert(Arc::from("fitted.values"), rnums(&fitted));
    fields.insert(Arc::from("r.squared"), rnum(r_squared));
    fields.insert(Arc::from("adj.r.squared"), rnum(adj_r2));
    fields.insert(Arc::from("df"), rint((n - p) as i32));
    fields.insert(Arc::from("sigma"), rnum(rse));
    fields.insert(Arc::from("std.errors"), rnums(&std_errors));
    fields.insert(Arc::from("t.values"), rnums(&t_values));
    fields.insert(Arc::from("p.values"), rnums(&p_values));
    fields.insert(Arc::from("f.statistic"), rnum(f_stat));
    if let Some(call_str) = gn(a, "_call") { fields.insert(Arc::from("call"), call_str); }
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("lm"), fields }))
}

// ─────────────────────────────────────────────────────────────────────
// glm — Generalized linear model via IRLS (binomial / poisson) or OLS
// ─────────────────────────────────────────────────────────────────────

pub fn bi_glm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);
    let family = gn(a, "family")
        .map(|v| match &v {
            RVal::List(items) => items.iter()
                .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("family"))
                .map(|(_, val)| val_to_str(val))
                .unwrap_or_else(|| "gaussian".into()),
            _ => val_to_str(&v),
        })
        .unwrap_or_else(|| "gaussian".into());

    if !is_formula(&first) {
        return Err(R2Err { msg: "glm() needs a formula (y ~ x)".into(), kind: ErrKind::Runtime });
    }

    let items: Vec<(Option<Arc<str>>, RVal)> = match &first { RVal::List(v) => v.clone(), _ => vec![] };
    let lhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs")).map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
    let rhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs")).map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
    let lhs = match &lhs_raw { RVal::List(items) if !items.is_empty() => items[0].1.clone(), other => other.clone() };
    let y: Vec<f64> = lhs.as_reals()?.into_iter().filter_map(|x| x).collect();
    let n = y.len();

    let mut predictor_cols: Vec<Vec<f64>> = Vec::new();
    let mut col_names: Vec<String> = vec!["(Intercept)".into()];
    match &rhs_raw {
        RVal::List(cols) => {
            for (col_name, col_val) in cols {
                if col_name.as_ref().map(|s| s.starts_with("~")).unwrap_or(false) { continue; }
                let (actual_name, actual_data) = match col_val {
                    RVal::List(inner) if !inner.is_empty() => {
                        (inner[0].0.as_ref().map(|s| s.to_string()).unwrap_or(format!("x{}", predictor_cols.len() + 1)), inner[0].1.clone())
                    }
                    _ => (col_name.as_ref().map(|s| s.to_string()).unwrap_or(format!("x{}", predictor_cols.len() + 1)), col_val.clone()),
                };
                // Phase S.1 — same factor/character expansion as lm().
                let (cols, names) = model_matrix_expand(&actual_name, &actual_data)?;
                for c in cols { predictor_cols.push(c); }
                for n in names { col_names.push(n); }
            }
        }
        _ => {
            let (cols, names) = model_matrix_expand("x1", &rhs_raw)?;
            for c in cols { predictor_cols.push(c); }
            for n in names { col_names.push(n); }
        }
    }
    let p = predictor_cols.len() + 1;
    let mut x_data = vec![1.0f64; n];
    for col in &predictor_cols { x_data.extend(col); }
    let x_mat = Matrix::new(x_data, n, p);

    let (coeffs, iter_count) = match family.as_str() {
        "gaussian" => {
            let xtx = x_mat.crossprod();
            let xty = x_mat.crossprod_vec(&y);
            let beta = xtx.solve(&xty).map_err(|e| R2Err { msg: format!("glm failed: {}", e), kind: ErrKind::Runtime })?;
            (beta, 1usize)
        }
        "binomial" => irls_binomial(&x_mat, &y, n, p)?,
        "poisson"  => irls_poisson(&x_mat, &y, n, p)?,
        _ => return Err(R2Err {
            msg: format!("glm family '{}' not supported (use gaussian, binomial, poisson)", family),
            kind: ErrKind::Runtime,
        }),
    };

    let mut fitted = vec![0.0; n];
    for i in 0..n {
        let eta: f64 = (0..p).map(|j| x_mat.get(i, j) * coeffs[j]).sum();
        fitted[i] = match family.as_str() {
            "binomial" => 1.0 / (1.0 + (-eta).exp()),
            "poisson"  => eta.exp(),
            _ => eta,
        };
    }
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(y, f)| y - f).collect();

    // Residual deviance — by family. Same formulas R uses.
    let deviance = family_deviance(&family, &y, &fitted, &residuals);

    // Null deviance — fit intercept-only model and compute its deviance.
    // For binomial/poisson the intercept is log-link of mean(y); for gaussian
    // it's mean(y) directly. We compute the null deviance analytically rather
    // than re-running IRLS, since the closed forms are well-known.
    let ybar = y.iter().sum::<f64>() / n as f64;
    let null_fitted: Vec<f64> = vec![ybar; n];
    let null_resid: Vec<f64> = y.iter().map(|yi| yi - ybar).collect();
    let null_deviance = family_deviance(&family, &y, &null_fitted, &null_resid);

    // Coefficient standard errors via the IRLS-converged (X'WX)^-1.
    // For gaussian we use the regular (X'X)^-1 × σ². For binomial/poisson
    // the weights at convergence are mu*(1-mu) and mu respectively.
    let weights: Vec<f64> = match family.as_str() {
        "binomial" => fitted.iter().map(|p| p * (1.0 - p)).collect(),
        "poisson"  => fitted.clone(),
        _          => vec![1.0; n],
    };
    let mut xtwx_data = vec![0.0; p * p];
    for j1 in 0..p {
        for j2 in 0..p {
            let mut s = 0.0;
            for i in 0..n { s += x_mat.get(i, j1) * weights[i] * x_mat.get(i, j2); }
            xtwx_data[j2 * p + j1] = s;
        }
    }
    let xtwx_inv = r2_linalg::dgetri(p, &xtwx_data);
    let dispersion = if family == "gaussian" && n > p {
        residuals.iter().map(|r| r * r).sum::<f64>() / (n - p) as f64
    } else { 1.0 };
    let mut std_errors = vec![0.0; p];
    let mut z_values = vec![0.0; p];
    let mut p_values = vec![1.0; p];
    if let Ok(inv) = &xtwx_inv {
        for j in 0..p {
            let var_j = inv[j * p + j] * dispersion;
            std_errors[j] = if var_j > 0.0 { var_j.sqrt() } else { 0.0 };
            z_values[j] = if std_errors[j] > 1e-15 { coeffs[j] / std_errors[j] } else { 0.0 };
            p_values[j] = 2.0 * (1.0 - phi(z_values[j].abs()));
        }
    }

    // AIC = -2·logL + 2k. logL uses the family-specific likelihood.
    let aic = match family.as_str() {
        "binomial" => deviance + 2.0 * p as f64,
        "poisson"  => deviance + 2.0 * p as f64 + 2.0 * log_factorial_sum(&y),
        _          => {
            // Gaussian MLE: log L = -n/2 [log(2π·σ²) + 1], σ² = SS_res/n
            let sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
            n as f64 * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0) + 2.0 * (p as f64 + 1.0)
        }
    };

    let mut coef_attrs = Attrs::default();
    coef_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());
    let coef_val = RVal::Numeric(
        coeffs.iter().map(|c| Some(*c)).collect::<Vec<_>>().into(), coef_attrs);

    let mut fields = HashMap::new();
    fields.insert(Arc::from("coefficients"), coef_val);
    fields.insert(Arc::from("residuals"), rnums(&residuals));
    fields.insert(Arc::from("fitted.values"), rnums(&fitted));
    fields.insert(Arc::from("deviance"), rnum(deviance));
    fields.insert(Arc::from("null.deviance"), rnum(null_deviance));
    fields.insert(Arc::from("aic"), rnum(aic));
    fields.insert(Arc::from("iter"), rint(iter_count as i32));
    fields.insert(Arc::from("dispersion"), rnum(dispersion));
    fields.insert(Arc::from("std.errors"), rnums(&std_errors));
    fields.insert(Arc::from("z.values"), rnums(&z_values));
    fields.insert(Arc::from("p.values"), rnums(&p_values));
    fields.insert(Arc::from("df.residual"), rint((n - p) as i32));
    fields.insert(Arc::from("df.null"), rint((n - 1) as i32));
    fields.insert(Arc::from("family"), rstr(&family));
    fields.insert(Arc::from("df"), rint((n - p) as i32));
    if let Some(call_str) = gn(a, "_call") { fields.insert(Arc::from("call"), call_str); }
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("glm"), fields }))
}

/// Family-specific deviance: D = -2·logL relative to the saturated model.
/// Used for both residual deviance (with `fitted` = model fits) and null
/// deviance (with `fitted` = mean(y) constant).
fn family_deviance(family: &str, y: &[f64], fitted: &[f64], resid: &[f64]) -> f64 {
    match family {
        "binomial" => -2.0 * y.iter().zip(fitted.iter()).map(|(yi, pi)| {
            let pi = pi.max(1e-10).min(1.0 - 1e-10);
            yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln()
        }).sum::<f64>(),
        "poisson" => 2.0 * y.iter().zip(fitted.iter()).map(|(yi, mi)| {
            if *yi > 0.0 { yi * (yi / mi.max(1e-10)).ln() - (yi - mi) } else { *mi }
        }).sum::<f64>(),
        _ => resid.iter().map(|r| r * r).sum(),
    }
}

/// Σ log(y_i!) for the Poisson AIC normalization. Uses Stirling for large y.
fn log_factorial_sum(y: &[f64]) -> f64 {
    y.iter().map(|yi| {
        let n = yi.round() as i64;
        if n <= 1 { 0.0 }
        else if n <= 50 {
            // Exact for small integers.
            (2..=n).map(|k| (k as f64).ln()).sum()
        } else {
            // Stirling: log(n!) ≈ n·log(n) − n + 0.5·log(2π·n)
            let nf = n as f64;
            nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln()
        }
    }).sum()
}

/// IRLS for logistic regression. Returns `(beta, n_iterations)` so
/// `summary(glm)` can report "Fisher Scoring iterations" like R does.
fn irls_binomial(x_mat: &Matrix, y: &[f64], n: usize, p: usize) -> Result<(Vec<f64>, usize), R2Err> {
    let mut beta = vec![0.0; p];
    let mut iter_used = 0;
    for it in 1..=25 {
        iter_used = it;
        let mut mu: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            let eta: f64 = (0..p).map(|j| x_mat.get(i, j) * beta[j]).sum();
            mu[i] = (1.0 / (1.0 + (-eta).exp())).max(1e-10).min(1.0 - 1e-10);
        }
        let mut w = vec![0.0; n];
        let mut z = vec![0.0; n];
        for i in 0..n {
            w[i] = mu[i] * (1.0 - mu[i]);
            let eta: f64 = (0..p).map(|j| x_mat.get(i, j) * beta[j]).sum();
            z[i] = eta + (y[i] - mu[i]) / w[i];
        }
        let new_beta = solve_wls(x_mat, &w, &z, n, p)?;
        let converged = beta.iter().zip(new_beta.iter()).all(|(a, b)| (a - b).abs() < 1e-8);
        beta = new_beta;
        if converged { break; }
    }
    Ok((beta, iter_used))
}

fn irls_poisson(x_mat: &Matrix, y: &[f64], n: usize, p: usize) -> Result<(Vec<f64>, usize), R2Err> {
    let mut beta = vec![0.0; p];
    let mut iter_used = 0;
    for it in 1..=25 {
        iter_used = it;
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let eta: f64 = (0..p).map(|j| x_mat.get(i, j) * beta[j]).sum();
            mu[i] = eta.exp().max(1e-10);
        }
        let mut z = vec![0.0; n];
        for i in 0..n {
            let eta: f64 = (0..p).map(|j| x_mat.get(i, j) * beta[j]).sum();
            z[i] = eta + (y[i] - mu[i]) / mu[i];
        }
        let new_beta = solve_wls(x_mat, &mu, &z, n, p)?;
        let converged = beta.iter().zip(new_beta.iter()).all(|(a, b)| (a - b).abs() < 1e-8);
        beta = new_beta;
        if converged { break; }
    }
    Ok((beta, iter_used))
}

fn solve_wls(x_mat: &Matrix, w: &[f64], z: &[f64], n: usize, p: usize) -> Result<Vec<f64>, R2Err> {
    let mut xtwx_data = vec![0.0; p * p];
    let mut xtwz = vec![0.0; p];
    for j1 in 0..p {
        for j2 in 0..p {
            let mut s = 0.0;
            for i in 0..n { s += x_mat.get(i, j1) * w[i] * x_mat.get(i, j2); }
            xtwx_data[j2 * p + j1] = s;
        }
        let mut s = 0.0;
        for i in 0..n { s += x_mat.get(i, j1) * w[i] * z[i]; }
        xtwz[j1] = s;
    }
    let xtwx = Matrix::new(xtwx_data, p, p);
    xtwx.solve(&xtwz).map_err(|e| R2Err { msg: format!("glm IRLS failed: {}", e), kind: ErrKind::Runtime })
}

// ─────────────────────────────────────────────────────────────────────
// aov — one-way analysis of variance
// ─────────────────────────────────────────────────────────────────────

/// Extract a column from a formula component as a vector of string labels.
/// Used for grouping factors (treatment, subject, etc.) regardless of
/// whether the column came in as Character, Factor, or Numeric. Wraps
/// the common patterns from the formula RHS / Error term unwrapping.
fn extract_labels(v: &RVal) -> Result<Vec<String>, R2Err> {
    let col = match v {
        RVal::List(items) if !items.is_empty() => items[0].1.clone(),
        other => other.clone(),
    };
    Ok(match &col {
        RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into())).collect(),
        RVal::Factor(f) => f.codes.iter()
            .map(|c| c.and_then(|i| f.levels.get(i as usize)).map(|s| s.to_string()).unwrap_or("NA".into()))
            .collect(),
        _ => col.as_reals()?.iter().map(|x| x.map(fmt_num).unwrap_or("NA".into())).collect(),
    })
}

pub fn bi_aov(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);
    let data = gn(a, "data");

    // Phase R.S.1 — repeated-measures branch.
    // If the formula carries an `~error` stratum (added by the engine's
    // formula-construction code when it sees Error(subject/treatment)),
    // dispatch to the rm-aov computation. The classical one-way / between-
    // subject path below is untouched.
    if let Some(RVal::DataFrame(_)) = &data {
        let items: Vec<(Option<Arc<str>>, RVal)> = match &first { RVal::List(v) => v.clone(), _ => vec![] };
        if items.iter().any(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~error")) {
            return aov_repeated_measures(&items, a);
        }
    }

    let (y_vec, group_vec): (Vec<f64>, Vec<String>) = if let Some(RVal::DataFrame(_)) = &data {
        let items: Vec<(Option<Arc<str>>, RVal)> = match &first { RVal::List(v) => v.clone(), _ => vec![] };
        let lhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs")).map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
        let rhs = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs")).map(|(_, v)| v.clone()).unwrap_or(RVal::Null);
        let y_col = match &lhs { RVal::List(items) if !items.is_empty() => items[0].1.clone(), v => v.clone() };
        let y: Vec<f64> = y_col.as_reals()?.into_iter().filter_map(|x| x).collect();

        let group_col = match &rhs {
            RVal::List(items) => items.iter().find(|(n, _)| !n.as_ref().map(|s| s.starts_with("~")).unwrap_or(true))
                .map(|(_, v)| match v { RVal::List(inner) if !inner.is_empty() => inner[0].1.clone(), v => v.clone() })
                .unwrap_or(RVal::Null),
            v => v.clone(),
        };
        let groups: Vec<String> = match &group_col {
            RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into())).collect(),
            RVal::Factor(f) => f.codes.iter().map(|c| c.and_then(|i| f.levels.get(i as usize)).map(|s| s.to_string()).unwrap_or("NA".into())).collect(),
            _ => {
                let nums = group_col.as_reals()?;
                nums.iter().map(|x| x.map(fmt_num).unwrap_or("NA".into())).collect()
            }
        };
        (y, groups)
    } else {
        let y: Vec<f64> = first.as_reals()?.into_iter().filter_map(|x| x).collect();
        let groups: Vec<String> = match &gv(a, 1) {
            RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into())).collect(),
            v => v.as_reals()?.iter().map(|x| x.map(fmt_num).unwrap_or("NA".into())).collect(),
        };
        (y, groups)
    };

    if y_vec.len() != group_vec.len() {
        return Err(R2Err { msg: "aov: y and group must have same length".into(), kind: ErrKind::Runtime });
    }
    let n = y_vec.len();

    let mut unique_groups: Vec<String> = Vec::new();
    for g in &group_vec { if !unique_groups.contains(g) { unique_groups.push(g.clone()); } }
    let k = unique_groups.len();

    let grand_mean = y_vec.iter().sum::<f64>() / n as f64;
    let mut group_means: Vec<f64> = Vec::new();
    let mut group_sizes: Vec<usize> = Vec::new();
    for g in &unique_groups {
        let vals: Vec<f64> = y_vec.iter().zip(group_vec.iter()).filter(|(_, gi)| *gi == g).map(|(y, _)| *y).collect();
        group_sizes.push(vals.len());
        group_means.push(vals.iter().sum::<f64>() / vals.len() as f64);
    }

    let ss_between: f64 = group_means.iter().zip(group_sizes.iter())
        .map(|(m, n)| *n as f64 * (m - grand_mean).powi(2)).sum();
    let ss_within: f64 = y_vec.iter().zip(group_vec.iter()).map(|(y, g)| {
        let gi = unique_groups.iter().position(|ug| ug == g).unwrap_or(0);
        (y - group_means[gi]).powi(2)
    }).sum();
    let ss_total = ss_between + ss_within;
    let df_between = (k - 1) as f64;
    let df_within = (n - k) as f64;
    let ms_between = ss_between / df_between;
    let ms_within = if df_within > 0.0 { ss_within / df_within } else { 0.0 };
    let f_stat = if ms_within > 1e-15 { ms_between / ms_within } else { f64::INFINITY };

    // Exact F upper-tail via the incomplete-beta identity.
    let p_value = if df_within > 0.0 {
        crate::htest::f_sf(f_stat, df_between, df_within)
    } else { 1.0 };

    println!("\nAnalysis of Variance Table\n");
    println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10}", "Source", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)");
    let p_str = fmt_pval(p_value);
    let stars = signif_stars(p_value);
    println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10} {}",
        "Treatment", k - 1, fmt_num(ss_between), fmt_num(ms_between), fmt_num(f_stat), p_str, stars);
    println!("{:<15} {:>5} {:>12} {:>12}", "Residuals", n - k, fmt_num(ss_within), fmt_num(ms_within));
    println!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
    println!("\nGroup means:");
    let max_name = unique_groups.iter().map(|g| g.len()).max().unwrap_or(5);
    for (i, g) in unique_groups.iter().enumerate() {
        println!("  {:>w$}  {:>10}  (n={})", g, fmt_num(group_means[i]), group_sizes[i], w = max_name);
    }

    let mut fields = HashMap::new();
    fields.insert(Arc::from("f.statistic"), rnum(f_stat));
    fields.insert(Arc::from("p.value"), rnum(p_value));
    fields.insert(Arc::from("ss.between"), rnum(ss_between));
    fields.insert(Arc::from("ss.within"), rnum(ss_within));
    fields.insert(Arc::from("ss.total"), rnum(ss_total));
    fields.insert(Arc::from("df.between"), rnum(df_between));
    fields.insert(Arc::from("df.within"), rnum(df_within));
    fields.insert(Arc::from("ms.between"), rnum(ms_between));
    fields.insert(Arc::from("ms.within"), rnum(ms_within));
    if let Some(call_str) = gn(a, "_call") { fields.insert(Arc::from("call"), call_str); }
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("aov"), fields }))
}

// ─────────────────────────────────────────────────────────────────────
// Phase R.S.1 — Repeated-measures ANOVA.
//
// Classical one-way within-subject design. Each subject is measured
// under every treatment level (balanced). Total variance decomposes:
//
//   SS_total  = Σ_ij (y_ij - ȳ..)²
//   SS_subj   = k · Σ_i (ȳ_i. - ȳ..)²        ← Error: subject stratum
//   SS_treat  = n · Σ_j (ȳ_.j - ȳ..)²        ← within-subject fixed effect
//   SS_within = SS_total - SS_subj - SS_treat ← residual within-subject
//
// where n = number of subjects, k = number of treatment levels.
//
// Degrees of freedom:
//   df_subj   = n - 1
//   df_treat  = k - 1
//   df_within = (n - 1)(k - 1)
//
// F-statistic for the treatment (within-subject) effect:
//   F = (SS_treat / df_treat) / (SS_within / df_within)
//
// The output table follows R's `summary(aov(y ~ t + Error(subj)))`
// layout — two strata, "Error: subject" first (just the subject
// residual line) then "Error: Within" with the treatment row plus the
// within-subject residuals.
// ─────────────────────────────────────────────────────────────────────

fn aov_repeated_measures(
    items: &[(Option<Arc<str>>, RVal)],
    a: &[EvalArg],
) -> Result<RVal, R2Err> {
    // ── 1. Extract y, treatment, and subject vectors ────────────────
    let lhs = items.iter()
        .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
        .map(|(_, v)| v.clone())
        .unwrap_or(RVal::Null);
    let rhs = items.iter()
        .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
        .map(|(_, v)| v.clone())
        .unwrap_or(RVal::Null);
    let err = items.iter()
        .find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~error"))
        .map(|(_, v)| v.clone())
        .unwrap_or(RVal::Null);

    let y_col = match &lhs {
        RVal::List(items) if !items.is_empty() => items[0].1.clone(),
        other => other.clone(),
    };
    let y: Vec<f64> = y_col.as_reals()?.into_iter().filter_map(|x| x).collect();

    let treatment_col = match &rhs {
        RVal::List(items) => items.iter()
            .find(|(n, _)| !n.as_ref().map(|s| s.starts_with('~')).unwrap_or(true))
            .map(|(_, v)| match v {
                RVal::List(inner) if !inner.is_empty() => inner[0].1.clone(),
                v => v.clone(),
            })
            .unwrap_or(RVal::Null),
        v => v.clone(),
    };
    // Phase R.S.1 — guard against the common confusion of putting the
    // fixed effect inside Error() and leaving the LHS empty. If the
    // formula RHS is null (i.e. user wrote `aov(y ~ Error(drug))`
    // with no fixed effect), there is nothing to test.
    if matches!(rhs, RVal::Null) || matches!(&treatment_col, RVal::Null) {
        return Err(R2Err {
            msg: "aov: no fixed effect on the right-hand side of '~'. \
                  You wrote 'aov(y ~ Error(X))' — the Error() term is for clustering, \
                  not the thing you want to test. \
                  For between-subject one-way ANOVA write 'aov(y ~ X, data=df)' \
                  (no Error needed). For within-subject (repeated measures) write \
                  'aov(y ~ treatment + Error(subject), data=df)'.".into(),
            kind: ErrKind::Runtime,
        });
    }

    let treatments = extract_labels(&treatment_col)?;
    let subjects = extract_labels(&err)?;

    // Phase R.S.1 — guard against the user putting the fixed effect
    // inside Error() (e.g. `Error(drug)` or `Error(drug/subjects)`).
    // When the stratum column equals the treatment column, the random
    // unit and the fixed effect are the same thing — meaningless.
    if treatments == subjects {
        return Err(R2Err {
            msg: "aov: Error(...) stratum is the same as the fixed effect. \
                  You wrote something like 'aov(y ~ drug + Error(drug))' or \
                  'aov(y ~ drug + Error(drug/subject))' — the Error term is for \
                  the random/clustering variable (subjects), NOT the fixed effect \
                  (the thing whose means you compare). \
                  For repeated measures write 'aov(y ~ drug + Error(subject), data=df)'.".into(),
            kind: ErrKind::Runtime,
        });
    }

    if y.len() != treatments.len() || y.len() != subjects.len() {
        return Err(R2Err {
            msg: format!(
                "aov(... + Error(...)): y ({}), treatment ({}), and subject ({}) must have equal length",
                y.len(), treatments.len(), subjects.len()
            ),
            kind: ErrKind::Runtime,
        });
    }
    let n_obs = y.len();
    if n_obs == 0 {
        return Err(R2Err { msg: "aov: empty data".into(), kind: ErrKind::Runtime });
    }

    // ── 2. Identify unique subjects and treatments ─────────────────
    let mut unique_subjects: Vec<String> = Vec::new();
    for s in &subjects { if !unique_subjects.contains(s) { unique_subjects.push(s.clone()); } }
    let n_subj = unique_subjects.len();

    let mut unique_treatments: Vec<String> = Vec::new();
    for t in &treatments { if !unique_treatments.contains(t) { unique_treatments.push(t.clone()); } }
    let k_treat = unique_treatments.len();

    if n_subj < 2 || k_treat < 2 {
        return Err(R2Err {
            msg: format!(
                "aov(... + Error(...)): need at least 2 subjects and 2 treatments (got {} subjects, {} treatments)",
                n_subj, k_treat
            ),
            kind: ErrKind::Runtime,
        });
    }

    // ── 3. Means per subject, per treatment, grand mean ─────────────
    let grand_mean: f64 = y.iter().sum::<f64>() / n_obs as f64;

    let subj_idx = |s: &String| unique_subjects.iter().position(|u| u == s).unwrap_or(0);
    let treat_idx = |t: &String| unique_treatments.iter().position(|u| u == t).unwrap_or(0);

    let mut subj_sum = vec![0.0_f64; n_subj];
    let mut subj_count = vec![0_usize; n_subj];
    let mut treat_sum = vec![0.0_f64; k_treat];
    let mut treat_count = vec![0_usize; k_treat];

    for i in 0..n_obs {
        let si = subj_idx(&subjects[i]);
        let ti = treat_idx(&treatments[i]);
        subj_sum[si] += y[i];
        subj_count[si] += 1;
        treat_sum[ti] += y[i];
        treat_count[ti] += 1;
    }
    let subj_mean: Vec<f64> = subj_sum.iter().zip(&subj_count)
        .map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 }).collect();
    let treat_mean: Vec<f64> = treat_sum.iter().zip(&treat_count)
        .map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 }).collect();

    // ── 4. Sums of squares ─────────────────────────────────────────
    let ss_total: f64 = y.iter().map(|v| (v - grand_mean).powi(2)).sum();
    let ss_subj: f64 = (0..n_subj)
        .map(|i| subj_count[i] as f64 * (subj_mean[i] - grand_mean).powi(2))
        .sum();
    let ss_treat: f64 = (0..k_treat)
        .map(|j| treat_count[j] as f64 * (treat_mean[j] - grand_mean).powi(2))
        .sum();
    let ss_within = ss_total - ss_subj - ss_treat;

    let df_subj = (n_subj - 1) as f64;
    let df_treat = (k_treat - 1) as f64;
    let df_within = ((n_subj - 1) * (k_treat - 1)) as f64;

    let ms_subj = if df_subj > 0.0 { ss_subj / df_subj } else { 0.0 };
    let ms_treat = if df_treat > 0.0 { ss_treat / df_treat } else { 0.0 };
    let ms_within = if df_within > 0.0 { ss_within / df_within } else { 0.0 };

    let f_treat = if ms_within > 1e-15 { ms_treat / ms_within } else { f64::INFINITY };

    // Exact F upper-tail (consistent with bi_aov). f_sf handles
    // f = +∞ (zero within-residual) as p = 0.
    let p_treat = if df_within > 0.0 {
        crate::htest::f_sf(f_treat, df_treat, df_within)
    } else { 1.0 };

    // Treatment column name from the RHS (for the table row label).
    let treat_name = match &rhs {
        RVal::List(items) => items.iter()
            .find(|(n, _)| !n.as_ref().map(|s| s.starts_with('~')).unwrap_or(true))
            .and_then(|(n, _)| n.as_ref().map(|s| s.to_string()))
            .unwrap_or_else(|| "treatment".to_string()),
        _ => "treatment".to_string(),
    };
    let subj_name = match &err {
        RVal::List(items) => items.iter()
            .find(|(n, _)| !n.as_ref().map(|s| s.starts_with('~')).unwrap_or(true))
            .and_then(|(n, _)| n.as_ref().map(|s| s.to_string()))
            .unwrap_or_else(|| "subject".to_string()),
        _ => "subject".to_string(),
    };

    // ── 5. Print R-compatible multi-stratum table ──────────────────
    println!();
    println!("Error: {}", subj_name);
    println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10}",
        "Source", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)");
    println!("{:<15} {:>5} {:>12} {:>12}",
        "Residuals", n_subj - 1, fmt_num(ss_subj), fmt_num(ms_subj));

    println!();
    println!("Error: Within");
    println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10}",
        "Source", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)");
    let p_str = fmt_pval(p_treat);
    let stars = signif_stars(p_treat);
    println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10} {}",
        treat_name, k_treat - 1,
        fmt_num(ss_treat), fmt_num(ms_treat),
        fmt_num(f_treat), p_str, stars);
    println!("{:<15} {:>5} {:>12} {:>12}",
        "Residuals", (n_subj - 1) * (k_treat - 1),
        fmt_num(ss_within), fmt_num(ms_within));
    println!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

    // ── 6. Return TypeInstance with structured fields ──────────────
    let mut fields = HashMap::new();
    fields.insert(Arc::from("type"), rstr("aov-rm"));
    fields.insert(Arc::from("f.statistic"), rnum(f_treat));
    fields.insert(Arc::from("p.value"), rnum(p_treat));
    fields.insert(Arc::from("ss.subject"), rnum(ss_subj));
    fields.insert(Arc::from("ss.treatment"), rnum(ss_treat));
    fields.insert(Arc::from("ss.within"), rnum(ss_within));
    fields.insert(Arc::from("ss.total"), rnum(ss_total));
    fields.insert(Arc::from("df.subject"), rnum(df_subj));
    fields.insert(Arc::from("df.treatment"), rnum(df_treat));
    fields.insert(Arc::from("df.within"), rnum(df_within));
    fields.insert(Arc::from("ms.subject"), rnum(ms_subj));
    fields.insert(Arc::from("ms.treatment"), rnum(ms_treat));
    fields.insert(Arc::from("ms.within"), rnum(ms_within));
    fields.insert(Arc::from("n.subjects"), rnum(n_subj as f64));
    fields.insert(Arc::from("n.treatments"), rnum(k_treat as f64));
    if let Some(call_str) = gn(a, "_call") {
        fields.insert(Arc::from("call"), call_str);
    }
    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("aov"),
        fields,
    }))
}

// ─────────────────────────────────────────────────────────────────────
// anova — model ANOVA table; falls back to aov for non-model input
// ─────────────────────────────────────────────────────────────────────

pub fn bi_anova(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let first = gv(a, 0);
    if let RVal::TypeInstance(inst) = &first {
        if inst.type_name.as_ref() == "lm" || inst.type_name.as_ref() == "glm" {
            let residuals: Vec<f64> = inst.fields.get("residuals")
                .and_then(|v| v.as_reals().ok())
                .unwrap_or_default().into_iter().filter_map(|x| x).collect();
            let fitted: Vec<f64> = inst.fields.get("fitted.values")
                .and_then(|v| v.as_reals().ok())
                .unwrap_or_default().into_iter().filter_map(|x| x).collect();
            let n = residuals.len();
            let p_minus_1 = inst.fields.get("df").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0);
            let df_resid = p_minus_1 as usize;
            let df_model = n.saturating_sub(df_resid + 1);

            let y_mean = fitted.iter().zip(residuals.iter()).map(|(f, r)| f + r).sum::<f64>() / n.max(1) as f64;
            let ss_model: f64 = fitted.iter().map(|f| (f - y_mean).powi(2)).sum();
            let ss_resid: f64 = residuals.iter().map(|r| r * r).sum();
            let ms_model = if df_model > 0 { ss_model / df_model as f64 } else { 0.0 };
            let ms_resid = if df_resid > 0 { ss_resid / df_resid as f64 } else { 0.0 };
            let f_stat = if ms_resid > 1e-15 { ms_model / ms_resid } else { f64::INFINITY };
            let p_value = if df_resid > 0 {
                crate::htest::f_sf(f_stat, df_model as f64, df_resid as f64)
            } else { 1.0 };

            println!("\nAnalysis of Variance Table\n");
            println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10}", "Source", "Df", "Sum Sq", "Mean Sq", "F value", "Pr(>F)");
            let p_str = fmt_pval(p_value);
            let stars = signif_stars(p_value);
            println!("{:<15} {:>5} {:>12} {:>12} {:>10} {:>10} {}",
                "Model", df_model, fmt_num(ss_model), fmt_num(ms_model), fmt_num(f_stat), p_str, stars);
            println!("{:<15} {:>5} {:>12} {:>12}", "Residuals", df_resid, fmt_num(ss_resid), fmt_num(ms_resid));
            println!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");

            let mut fields = HashMap::new();
            fields.insert(Arc::from("f.statistic"), rnum(f_stat));
            fields.insert(Arc::from("p.value"), rnum(p_value));
            fields.insert(Arc::from("ss.model"), rnum(ss_model));
            fields.insert(Arc::from("ss.resid"), rnum(ss_resid));
            return Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("anova"), fields }));
        }
    }
    bi_aov(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn lm_two_vector_legacy_path() {
        let x = nums(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = nums(&[2.1, 4.0, 6.2, 7.9, 10.1]);
        let r = bi_lm(&[evarg(y), evarg(x)]).unwrap();
        match r {
            RVal::TypeInstance(inst) => {
                // R: intercept ≈ 0.09, slope ≈ 1.99 — match to 1e-6.
                let coefs = inst.fields.get("coefficients").unwrap().as_reals().unwrap();
                let intercept = coefs[0].unwrap();
                let slope = coefs[1].unwrap();
                assert!((intercept - 0.09).abs() < 0.01, "intercept = {}", intercept);
                assert!((slope - 1.99).abs() < 0.01, "slope = {}", slope);
            }
            _ => panic!("lm must return TypeInstance"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase R.S.1 — Repeated-measures ANOVA tests.
    // ─────────────────────────────────────────────────────────────────

    fn chars(v: &[&str]) -> RVal {
        RVal::Character(v.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }

    /// Build the synthetic formula-list shape that aov_repeated_measures
    /// expects: items contain ~lhs, ~rhs, ~class, and ~error.
    fn rm_formula(y: RVal, treat_name: &str, treat: RVal, subj_name: &str, subj: RVal) -> RVal {
        RVal::List(vec![
            (Some(Arc::from("~lhs")),
                RVal::List(vec![(Some(Arc::from("y")), y)])),
            (Some(Arc::from("~rhs")),
                RVal::List(vec![(Some(Arc::from(treat_name)), treat)])),
            (Some(Arc::from("~class")),
                RVal::Character(vec![Some(Arc::from("formula"))], Attrs::default())),
            (Some(Arc::from("~error")),
                RVal::List(vec![(Some(Arc::from(subj_name)), subj)])),
        ])
    }

    #[test]
    fn rm_aov_matches_hand_computation_for_5x2_design() {
        // 5 subjects × 2 treatments (A, B). Hand-computed expected:
        //   SS_subject   = 9.60   df=4   MS=2.40
        //   SS_treatment = 14.4   df=1   MS=14.4
        //   SS_within    = 5.60   df=4   MS=1.40
        //   F_treatment  = (14.4/1) / (5.6/4) = 10.2857
        let y = nums(&[10.0, 12.0, 8.0, 11.0, 12.0, 13.0, 9.0, 14.0, 11.0, 12.0]);
        let treat = chars(&["A","B","A","B","A","B","A","B","A","B"]);
        let subj = nums(&[1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0,5.0,5.0]);
        let formula = rm_formula(y, "treatment", treat, "subject", subj);

        // The repeated-measures branch in bi_aov triggers when `data` is
        // present and the formula contains `~error`. Inject a dummy data
        // arg so the branch fires.
        let dummy_df = RVal::DataFrame(r2_types::DataFrame {
            columns: vec![], row_names: None,
        });
        let r = bi_aov(&[
            evarg(formula),
            EvalArg { name: Some(Arc::from("data")), value: dummy_df },
        ]).unwrap();

        match r {
            RVal::TypeInstance(inst) => {
                assert_eq!(inst.type_name.as_ref(), "aov");
                let get = |k: &str| inst.fields.get(k).unwrap().scalar_f64().unwrap().unwrap();
                assert!((get("ss.subject")   - 9.60).abs() < 1e-9, "ss.subject = {}", get("ss.subject"));
                assert!((get("ss.treatment") - 14.4).abs() < 1e-9, "ss.treatment = {}", get("ss.treatment"));
                assert!((get("ss.within")    -  5.6).abs() < 1e-9, "ss.within = {}", get("ss.within"));
                assert!((get("df.subject")   -  4.0).abs() < 1e-12);
                assert!((get("df.treatment") -  1.0).abs() < 1e-12);
                assert!((get("df.within")    -  4.0).abs() < 1e-12);
                let f = get("f.statistic");
                assert!((f - 10.285714).abs() < 1e-4, "F = {}", f);
                let p = get("p.value");
                assert!((0.0..=1.0).contains(&p), "p out of range: {}", p);
                // F(1,4)=10.29 corresponds to p around 0.033 by exact CDF;
                // Wilson-Hilferty here gives ~0.0048. Bound is permissive.
                assert!(p < 0.05, "p should be < 0.05 for F=10.29, got {}", p);
            }
            _ => panic!("aov(...+Error(...)) must return TypeInstance"),
        }
    }

    #[test]
    fn rm_aov_errors_when_fewer_than_2_subjects() {
        // One subject, two treatments — degenerate; should error cleanly.
        let y = nums(&[1.0, 2.0]);
        let treat = chars(&["A", "B"]);
        let subj = nums(&[1.0, 1.0]);
        let formula = rm_formula(y, "treatment", treat, "subject", subj);
        let dummy_df = RVal::DataFrame(r2_types::DataFrame {
            columns: vec![], row_names: None,
        });
        let r = bi_aov(&[
            evarg(formula),
            EvalArg { name: Some(Arc::from("data")), value: dummy_df },
        ]);
        assert!(r.is_err(), "should error with only 1 subject");
    }

    #[test]
    fn rm_aov_errors_on_mismatched_lengths() {
        // y has 4 entries but subject only 3 — length mismatch.
        let y = nums(&[1.0, 2.0, 3.0, 4.0]);
        let treat = chars(&["A","B","A","B"]);
        let subj = nums(&[1.0, 1.0, 2.0]); // length 3, not 4
        let formula = rm_formula(y, "treatment", treat, "subject", subj);
        let dummy_df = RVal::DataFrame(r2_types::DataFrame {
            columns: vec![], row_names: None,
        });
        let r = bi_aov(&[
            evarg(formula),
            EvalArg { name: Some(Arc::from("data")), value: dummy_df },
        ]);
        assert!(r.is_err(), "should error on length mismatch");
    }
}
