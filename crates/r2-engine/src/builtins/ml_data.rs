//! ML FOUNDATION + DATA HANDLING — extracted from lib.rs
//! (engine-split, opus-4.8 session, content-anchored).
//!
//! Covers: svd() adapters, the CART decision-tree engine
//! (build_tree + gini + mse_impurity), read.csv parsing
//! (bi_read_csv_v2 + parse_csv_line), and the dplyr-style data
//! verbs (filter/select/mutate/arrange/regex helpers).
//!
//! Tree + CSV helpers are module-private. `r2_ml::tree::TreeNode`
//! is imported inline where used.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::all)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;

// ── svd() — Singular Value Decomposition ─────────────────────────────

// Phase R.4: bi_svd moved to r2-linalg::ops. Returns full thin SVD
// (`$d`, `$u`, `$v`) via `dgesvd_full` (shipped v0.1.0).
pub(crate) fn bi_svd(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_svd(a)
}

// ── eigen() — Eigenvalue decomposition ───────────────────────────────

// Phase R.4: bi_eigen moved to r2-linalg::ops.
pub(crate) fn bi_eigen(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_eigen(a)
}

// ── prcomp() — Principal Component Analysis ──────────────────────────

// Phase R.1 step 4: bi_prcomp moved to r2-ml::dispatch.

// ── kmeans() — K-means clustering ────────────────────────────────────

// Phase R.1 step 4: bi_kmeans moved to r2-ml::dispatch. Per-point
// centroid assignment uses kernel::par_for(Op::PerPointDistance, ...).

// ── knn() — K-nearest neighbors classification ──────────────────────

// Phase R.1 step 4: bi_knn moved to r2-ml::dispatch.

// ── naive.bayes() — Naive Bayes classifier ──────────────────────────

// Phase R.1 step 4: bi_naive_bayes moved to r2-ml::dispatch.

// ── scale() — center and scale matrix columns ───────────────────────

pub(crate) fn bi_scale(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mat = match &gv(a,0) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, "scale() needs matrix") };
    let center = gn(a,"center").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let do_scale = gn(a,"scale").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let (m, n) = (mat.nrow, mat.ncol);
    let mut x = mat.data.clone();
    let means = mat.col_means();
    for c in 0..n {
        let col_start = c * m;
        let mean = if center { means[c] } else { 0.0 };
        let mut ss = 0.0;
        for r in 0..m { ss += (x[col_start + r] - mean).powi(2); }
        let sd = if do_scale { (ss / (m - 1).max(1) as f64).sqrt().max(1e-15) } else { 1.0 };
        for r in 0..m {
            if center { x[col_start + r] -= mean; }
            if do_scale { x[col_start + r] /= sd; }
        }
    }
    Ok(RVal::Matrix(Matrix::new(x, m, n)))
}

// ═══════════════════════════════════════════════════════════════════════
// Decision Tree (CART — Classification and Regression Tree)
// ═══════════════════════════════════════════════════════════════════════

// Phase R.1 step 1: TreeNode struct extracted to r2-ml::tree. The engine
// keeps wrapper definitions of `build_tree` / `tree_predict_one` /
// `count_splits` / `serialize_tree` that delegate to the r2-ml versions —
// this preserves callsite signatures while the actual algorithms live in
// the domain crate.
use r2_ml::tree::TreeNode;

fn build_tree(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{ r2_ml::tree::build_tree(x, y, m, n, row_mask, max_depth, min_samples, depth, is_classification) }

#[allow(dead_code)]
fn __build_tree_old(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{
    let active: Vec<usize> = row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
    let count = active.len();

    // Compute prediction: mean for regression, majority vote for classification
    let prediction = if is_classification {
        let mut votes: HashMap<i64, usize> = HashMap::new();
        for &i in &active { *votes.entry(y[i] as i64).or_insert(0) += 1; }
        votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0)
    } else {
        active.iter().map(|&i| y[i]).sum::<f64>() / count.max(1) as f64
    };

    // Leaf conditions
    if count <= min_samples || depth >= max_depth {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Check if all y values are same
    let all_same = active.windows(2).all(|w| (y[w[0]] - y[w[1]]).abs() < 1e-10);
    if all_same {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Find best split
    let mut best_gain = 0.0f64;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    let parent_impurity = if is_classification { gini(&active, y) } else { mse_impurity(&active, y) };

    for feat in 0..n {
        // Get sorted indices for this feature
        let mut indexed: Vec<(f64, usize)> = active.iter().map(|&i| (x[feat * m + i], i)).collect();
        indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if is_classification {
            // Incremental gini: scan sorted data, maintain left/right class counts
            // Find unique classes (small integers)
            let mut max_class = 0i64;
            for &(_, idx) in &indexed { max_class = max_class.max(y[idx] as i64); }
            let nc = (max_class + 1) as usize;
            if nc > 1000 { continue; } // safety: too many classes

            let mut right_counts = vec![0usize; nc];
            for &(_, idx) in &indexed {
                let c = y[idx] as usize;
                if c < nc { right_counts[c] += 1; }
            }
            let mut left_counts = vec![0usize; nc];
            let mut left_n = 0usize;
            let mut right_n = count;

            // Limit candidate splits to ~32 evenly spaced
            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let c = y[indexed[i].1] as usize;
                if c < nc { left_counts[c] += 1; right_counts[c] -= 1; }
                left_n += 1;
                right_n -= 1;

                // Only evaluate at step boundaries or when value changes
                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }

                last_split = i;
                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;

                // Compute gini from counts directly (no allocation)
                let left_gini = 1.0 - left_counts.iter().map(|&c| { let p = c as f64 / left_n as f64; p * p }).sum::<f64>();
                let right_gini = 1.0 - right_counts.iter().map(|&c| { let p = c as f64 / right_n as f64; p * p }).sum::<f64>();
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        } else {
            // Regression: incremental MSE using running sums
            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let total_sum: f64 = indexed.iter().map(|&(_, idx)| y[idx]).sum();
            let total_sq: f64 = indexed.iter().map(|&(_, idx)| y[idx] * y[idx]).sum();
            let mut left_n = 0usize;

            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let yi = y[indexed[i].1];
                left_sum += yi;
                left_sq += yi * yi;
                left_n += 1;
                let right_n = count - left_n;

                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }
                last_split = i;

                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;
                let right_sum = total_sum - left_sum;

                let left_mse = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_mse = (total_sq - left_sq) / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: parent_impurity };
    }

    // Split
    let mut left_mask = vec![false; m];
    let mut right_mask = vec![false; m];
    for &i in &active {
        if x[best_feature * m + i] <= best_threshold { left_mask[i] = true; }
        else { right_mask[i] = true; }
    }

    let left = build_tree(x, y, m, n, &left_mask, max_depth, min_samples, depth + 1, is_classification);
    let right = build_tree(x, y, m, n, &right_mask, max_depth, min_samples, depth + 1, is_classification);

    TreeNode {
        is_leaf: false, prediction, feature: best_feature, threshold: best_threshold,
        left: Some(Box::new(left)), right: Some(Box::new(right)),
        n_samples: count, impurity: parent_impurity,
    }
}

fn gini(indices: &[usize], y: &[f64]) -> f64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &i in indices { *counts.entry(y[i] as i64).or_insert(0) += 1; }
    let n = indices.len() as f64;
    1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

fn mse_impurity(indices: &[usize], y: &[f64]) -> f64 {
    let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len().max(1) as f64;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len().max(1) as f64
}

// ── rpart() — Decision tree interface ────────────────────────────────

// Phase R.1 step 4: bi_rpart moved to r2-ml::dispatch. The 1-line adapter
// here exists only to satisfy r2-engine's `BuiltinFn` signature, which
// carries `&mut Engine` and `&EnvRef` for stateful builtins. Pure ML
// builtins ignore those — the adapter is FFI glue, not bloat.

// ── rf() — Random Forest ─────────────────────────────────────────────

// Phase R.1 step 4: bi_rf moved to r2-ml::dispatch. Uses kernel::par_for
// instead of par_iter — Rayon stays below the kernel layer (§4.9).

// ═══════════════════════════════════════════════════════════════════════
// PHASE: DATA HANDLING — filter, select, mutate, arrange, regex, etc.
// ═══════════════════════════════════════════════════════════════════════

// ── sub() / regexpr basics ───────────────────────────────────────────




// ── duplicated() / distinct values ───────────────────────────────────


// ── order() — return indices that would sort the vector ──────────────


// ── rank() — ranks of values ─────────────────────────────────────────


// ── cummax, cummin ───────────────────────────────────────────────────

pub(crate) fn bi_cummax(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummax(a) }

pub(crate) fn bi_cummin(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummin(a) }

// ── which() improvements — named results ────────────────────────────

// (which already exists, but let's add which.min/max for data.frame columns)

// ── Improved read.csv — handles quotes, various delimiters, type inference ──

pub(crate) fn bi_read_csv_v2(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a,0) {
        RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).ok_or(R2Err{msg:"NA path".into(),kind:ErrKind::Runtime})?,
        _ => return err!(Runtime, "read.csv needs path"),
    };
    let header = gn(a,"header").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let sep = gn(a,"sep").and_then(|v| match v { RVal::Character(s,_) => s[0].as_ref().map(|s| s.to_string()), _ => None }).unwrap_or(",".into());
    let na_strings = vec!["NA", "na", "N/A", "n/a", "", ".", "NULL", "null", "None", "none"];

    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot read '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let mut lines = content.lines();

    // Parse header
    let col_names: Vec<String> = if header {
        lines.next().map(|l| parse_csv_line(l, &sep)).unwrap_or_default()
    } else { Vec::new() };

    // Read all rows
    let mut raw_rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        raw_rows.push(parse_csv_line(line, &sep));
    }

    if raw_rows.is_empty() { return err!(Runtime, "empty CSV file"); }

    let ncol = col_names.len().max(raw_rows.iter().map(|r| r.len()).max().unwrap_or(0));
    let nrow = raw_rows.len();

    // Build columns with type inference
    let mut columns = Vec::new();
    for c in 0..ncol {
        let name = if c < col_names.len() { Arc::from(col_names[c].as_str()) } else { Arc::from(format!("V{}", c+1).as_str()) };

        let col_vals: Vec<String> = raw_rows.iter().map(|r| r.get(c).cloned().unwrap_or_default()).collect();

        // Type inference: try integer → numeric → character
        let all_int = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<i32>().is_ok());
        let all_num = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<f64>().is_ok());
        let has_num = col_vals.iter().any(|s| s.parse::<f64>().is_ok());

        if all_int && has_num {
            let vals: Vec<Integer> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Integer(vals.into(), Attrs::default())));
        } else if all_num && has_num {
            let vals: Vec<Real> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Numeric(vals.into(), Attrs::default())));
        } else {
            let vals: Vec<Character> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { Some(Arc::from(s.as_str())) }
            }).collect();
            columns.push((name, RVal::Character(vals, Attrs::default())));
        }
    }

    println!("Read {} rows × {} columns from '{}'", nrow, ncol, path);
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// Parse a CSV line handling quoted fields
fn parse_csv_line(line: &str, sep: &str) -> Vec<String> {
    let sep_char = sep.chars().next().unwrap_or(',');
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"'); // escaped quote
                    chars.next();
                } else {
                    in_quotes = false; // end quote
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == sep_char {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// ── DataFrame pipe-friendly operations: filter, select, mutate, arrange ──

// Phase R.2: bi_filter moved to r2-data::dplyr.
pub(crate) fn bi_filter(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_filter(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "filter() needs data.frame") };
    let e = _e;
    let mask = e.as_logicals(&gv(a,1))?;
    let keep: Vec<usize> = mask.iter().enumerate().filter(|(_, m)| **m == Some(true)).map(|(i, _)| i).collect();
    let nrow = df.nrow();

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(keep.iter().map(|&r| if r < v.len() { v[r].clone() } else { None }).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_select moved to r2-data::dplyr.
pub(crate) fn bi_select(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_select(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "select() needs data.frame") };

    // Collect column names from remaining args
    let mut col_names: Vec<String> = Vec::new();
    for i in 1..10 {
        match &gv(a, i) {
            RVal::Character(v, _) => {
                for c in v { if let Some(s) = c { col_names.push(s.to_string()); } }
            }
            RVal::Null => break,
            _ => break,
        }
    }

    if col_names.is_empty() { return Ok(RVal::DataFrame(df)); }

    let columns: Vec<(Arc<str>, RVal)> = col_names.iter().filter_map(|name| {
        df.columns.iter().find(|(n, _)| n.as_ref() == name.as_str()).cloned()
    }).collect();

    if columns.is_empty() { return err!(Runtime, "select: no matching columns found"); }
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_arrange moved to r2-data::dplyr.
pub(crate) fn bi_arrange(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_arrange(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "arrange() needs data.frame") };
    let e = _e;
    let sort_vals = e.as_reals(&gv(a,1))?;
    let decreasing = gn(a,"decreasing").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(false);

    let nrow = df.nrow();
    let mut indices: Vec<usize> = (0..nrow).collect();
    indices.sort_by(|&a, &b| {
        let va = sort_vals.get(a).and_then(|x| *x).unwrap_or(f64::NAN);
        let vb = sort_vals.get(b).and_then(|x| *x).unwrap_or(f64::NAN);
        if decreasing { vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal) }
        else { va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal) }
    });

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(indices.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// ── Sys.getenv() — read environment variable ─────────────────────────

pub(crate) fn bi_sys_getenv(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a,0));
    let val = std::env::var(&name).unwrap_or_default();
    Ok(rstr(&val))
}

// ── file.exists() — check if file exists ─────────────────────────────


// ── list.files() — list files in directory ───────────────────────────


// end of file
