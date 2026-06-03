//! Tree-based model builtins — rpart (CART), rf (random forest),
//! gbm (gradient boosting). Split out of `dispatch.rs`; all three
//! build on `crate::tree`. Shared arg helpers (`gn`,
//! `rval_to_string`) come from the parent via `super::`.

use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use crate::data::extract_ml_data;
use crate::tree::{build_tree, tree_predict_one, serialize_tree, print_tree, count_splits, parallel_random, next_random, SEED_STATE, TreeNode};
use super::{gn, rval_to_string};

/// rpart(x, y) or rpart(y ~ ., data = df) — fits a CART decision tree.
pub fn bi_rpart(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;
    let max_depth = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let min_samples = gn(a, "min_samples").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let user_type = gn(a, "type").map(|v| rval_to_string(&v));

    let (m, n) = (mat.nrow, mat.ncol);
    if y.len() != m {
        return Err(R2Err { msg: "rpart: y length must match x rows".into(), kind: ErrKind::Runtime });
    }

    // Auto-detect: if y has > 10 unique values, regression; else classification.
    let tree_type = user_type.unwrap_or_else(|| {
        let mut uniq: Vec<i64> = y.iter().map(|v| (*v * 1000.0) as i64).collect();
        uniq.sort(); uniq.dedup();
        if uniq.len() > 10 { "regression".into() } else { "classification".into() }
    });
    let is_class = tree_type != "regression";
    let mask = vec![true; m];
    let tree = build_tree(&mat.data, &y, m, n, &mask, max_depth, min_samples, 0, is_class);

    soutln!("\nDecision tree ({}, depth={}):", tree_type, max_depth);
    print_tree(&tree, 0, &col_names);

    // Training accuracy / MSE
    let preds: Vec<f64> = (0..m).map(|i| tree_predict_one(&tree, &mat.data, m, i)).collect();
    if is_class {
        let correct = preds.iter().zip(y.iter())
            .filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        soutln!("\nTraining accuracy: {}/{} ({}%)", correct, m,
            fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        let mse: f64 = preds.iter().zip(y.iter())
            .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        soutln!("\nTraining MSE: {}", fmt_num(mse));
    }

    // Build the rpart TypeInstance with serialized tree fields
    let pred_vals: Vec<Real> = preds.iter().map(|p| Some(*p)).collect();
    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("type"), rstr(&tree_type));
    fields.insert(Arc::from("max_depth"), rnum(max_depth as f64));

    let mut feat_list = Vec::new();
    let mut thresh_list = Vec::new();
    let mut pred_list = Vec::new();
    let mut leaf_list = Vec::new();
    let mut left_list = Vec::new();
    let mut right_list = Vec::new();
    serialize_tree(&tree, &mut feat_list, &mut thresh_list, &mut pred_list,
        &mut leaf_list, &mut left_list, &mut right_list);
    fields.insert(Arc::from("_tree_feat"),
        rnums(&feat_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_thresh"), rnums(&thresh_list));
    fields.insert(Arc::from("_tree_pred"), rnums(&pred_list));
    fields.insert(Arc::from("_tree_leaf"),
        rnums(&leaf_list.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_left"),
        rnums(&left_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_right"),
        rnums(&right_list.iter().map(|&x| x as f64).collect::<Vec<_>>()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("rpart"),
        fields,
    }))
}

/// Random Forest fit. Trees are built independently — parallel-for via the
/// kernel (`r2_kernel::par_for(Op::TreeBuild, ...)`). No `par_iter` here;
/// Rayon stays below the kernel layer per §4.9.
pub fn bi_rf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;
    let ntrees     = gn(a, "ntrees").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;
    let max_depth  = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(10.0) as usize;
    let tree_type  = gn(a, "type").map(|v| rval_to_string(&v)).unwrap_or("classification".into());
    let is_class   = tree_type != "regression";
    let (m, n)     = (mat.nrow, mat.ncol);

    soutln!("Random Forest: {} trees, max_depth={}, features={}", ntrees, max_depth, n);

    // Heap-allocated locals captured by the per-tree closure. Cloned once so
    // the closure is `Sync` (required by kernel::par_for's parallel path).
    let mat_data = mat.data.clone();
    let y_clone  = y.clone();

    let build_one = move |t: usize| -> (crate::tree::TreeNode, Vec<f64>) {
        let mut seed = SEED_STATE.load(Ordering::Relaxed).wrapping_add(t as u64 * 999983);
        let mut mask = vec![false; m];
        for _ in 0..m {
            let idx = (parallel_random(&mut seed) * m as f64) as usize % m;
            mask[idx] = true;
        }
        let tree = build_tree(&mat_data, &y_clone, m, n, &mask, max_depth, 2, 0, is_class);
        let preds: Vec<f64> = (0..m).map(|i| tree_predict_one(&tree, &mat_data, m, i)).collect();
        (tree, preds)
    };

    // Parallelism decision lives in the kernel — `bi_rf` no longer imports
    // Rayon or calls Oracle directly.
    let tree_results: Vec<(crate::tree::TreeNode, Vec<f64>)> =
        r2_kernel::par_for(r2_oracle::Op::TreeBuild, ntrees, build_one);

    // Aggregate (sequential — fast)
    let mut all_preds = vec![vec![0.0; m]; ntrees];
    let mut importance = vec![0usize; n];
    for (t, (tree, preds)) in tree_results.iter().enumerate() {
        all_preds[t] = preds.clone();
        count_splits(tree, &mut importance);
    }

    let mut final_preds = vec![0.0; m];
    if is_class {
        for i in 0..m {
            let mut votes: HashMap<i64, usize> = HashMap::new();
            for t in 0..ntrees {
                *votes.entry(all_preds[t][i] as i64).or_insert(0) += 1;
            }
            final_preds[i] = votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0);
        }
        let correct = final_preds.iter().zip(y.iter())
            .filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        soutln!("Training accuracy: {}/{} ({}%)", correct, m,
            fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        for i in 0..m {
            final_preds[i] = all_preds.iter().map(|p| p[i]).sum::<f64>() / ntrees as f64;
        }
        let mse: f64 = final_preds.iter().zip(y.iter())
            .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        soutln!("Training MSE: {}", fmt_num(mse));
    }

    let pred_vals: Vec<Real> = final_preds.iter().map(|p| Some(*p)).collect();
    let total_splits: usize = importance.iter().sum();
    if total_splits > 0 {
        soutln!("\nFeature importance:");
        let mut imp_idx: Vec<(usize, f64)> = importance.iter().enumerate()
            .map(|(i, &c)| (i, c as f64 / total_splits as f64 * 100.0)).collect();
        imp_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, pct) in imp_idx.iter().take(n.min(10)) {
            if *pct > 0.0 {
                let label = col_names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                soutln!("  {}: {}%", label, fmt_num(*pct));
            }
        }
    }
    let imp_vals: Vec<Real> = importance.iter()
        .map(|&c| Some(c as f64 / total_splits.max(1) as f64)).collect();
    let mut imp_attrs = Attrs::default();
    imp_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());
    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("importance"), RVal::Numeric(imp_vals.into(), imp_attrs));
    fields.insert(Arc::from("ntrees"), rnum(ntrees as f64));
    fields.insert(Arc::from("type"), rstr(&tree_type));
    fields.insert(Arc::from("xnames"),
        RVal::Character(col_names.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("rf"),
        fields,
    }))
}

/// Gradient Boosted Trees fit. The boosting loop is sequential (each
/// iteration depends on previous predictions). Per-iteration row work
/// (residuals, prediction updates, loss reduction) is parallelized via
/// `kernel::par_for(Op::PerElementMap, m, ...)` — no Rayon import.
pub fn bi_gbm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (y, mat, col_names) = extract_ml_data(a)?;

    let ntrees       = gn(a, "ntrees").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;
    let lr           = gn(a, "learning_rate").or(gn(a, "eta")).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.1);
    let max_depth    = gn(a, "max_depth").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(3.0) as usize;
    let min_samples  = gn(a, "min_samples").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;
    let subsample    = gn(a, "subsample").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.8);
    let loss_type    = gn(a, "loss").map(|v| rval_to_string(&v)).unwrap_or("squared".into());

    let (m, n) = (mat.nrow, mat.ncol);
    let is_classification = loss_type == "logistic" || loss_type == "bernoulli";

    // Initialize predictions
    let mut f_vals = vec![0.0; m];
    if is_classification {
        let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
        let neg = (m as f64 - pos).max(1.0);
        let init = (pos / neg).max(1e-10).ln();
        for v in f_vals.iter_mut() { *v = init; }
    } else {
        let mean = y.iter().sum::<f64>() / m as f64;
        for v in f_vals.iter_mut() { *v = mean; }
    }

    let mut trees: Vec<TreeNode> = Vec::new();
    let mut train_losses: Vec<f64> = Vec::new();

    soutln!("Gradient Boosted Trees: {} trees, lr={}, depth={}, loss={}",
        ntrees, lr, max_depth, loss_type);

    for t in 0..ntrees {
        // Compute pseudo-residuals (negative gradient) — kernel-dispatched.
        let f_snap = f_vals.clone();
        let y_snap = y.clone();
        let loss = loss_type.clone();
        let residuals: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            match loss.as_str() {
                "logistic" | "bernoulli" => {
                    let p = 1.0 / (1.0 + (-f_snap[i]).exp());
                    y_snap[i] - p
                }
                "huber" => {
                    let r = y_snap[i] - f_snap[i];
                    let delta = 1.0;
                    if r.abs() <= delta { r } else { delta * r.signum() }
                }
                // "squared" | "ls" | _
                _ => y_snap[i] - f_snap[i],
            }
        });

        // Stochastic subsample (sequential — RNG-bound)
        let mut mask = vec![false; m];
        let n_sample = (m as f64 * subsample) as usize;
        for _ in 0..n_sample {
            let idx = (next_random() * m as f64) as usize % m;
            mask[idx] = true;
        }

        let tree = build_tree(&mat.data, &residuals, m, n, &mask, max_depth, min_samples, 0, false);

        // Update predictions: F += lr * tree(x). Per-row, kernel-dispatched.
        let mat_snap = mat.data.clone();
        let tree_snap = tree.clone();
        let updates: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            lr * tree_predict_one(&tree_snap, &mat_snap, m, i)
        });
        for i in 0..m { f_vals[i] += updates[i]; }

        trees.push(tree);

        // Loss reduction — collect per-element contributions, then sum.
        let f_snap2 = f_vals.clone();
        let y_snap2 = y.clone();
        let loss2 = loss_type.clone();
        let contribs: Vec<f64> = r2_kernel::par_for(r2_oracle::Op::PerElementMap, m, move |i| {
            match loss2.as_str() {
                "logistic" | "bernoulli" => {
                    let p = (1.0 / (1.0 + (-f_snap2[i]).exp())).max(1e-15).min(1.0 - 1e-15);
                    -(y_snap2[i] * p.ln() + (1.0 - y_snap2[i]) * (1.0 - p).ln())
                }
                _ => (y_snap2[i] - f_snap2[i]).powi(2),
            }
        });
        let loss = contribs.iter().sum::<f64>() / m as f64;
        train_losses.push(loss);

        if (t + 1) % 25 == 0 || t == ntrees - 1 {
            sout!("\r  Iter {}/{}: loss = {}", t + 1, ntrees, fmt_num(loss));
        }
    }
    soutln!();

    // Final predictions
    let final_preds: Vec<f64> = if is_classification {
        f_vals.iter().map(|f| if 1.0 / (1.0 + (-f).exp()) > 0.5 { 1.0 } else { 0.0 }).collect()
    } else {
        f_vals.clone()
    };

    if is_classification {
        let correct = final_preds.iter().zip(y.iter()).filter(|(p, y)| (**p as i64) == (**y as i64)).count();
        soutln!("Training accuracy: {}/{} ({}%)", correct, m, fmt_num(correct as f64 / m as f64 * 100.0));
    } else {
        let mse = final_preds.iter().zip(y.iter()).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / m as f64;
        let y_mean = y.iter().sum::<f64>() / m as f64;
        let ss_tot = y.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
        let r2 = 1.0 - (mse * m as f64) / ss_tot;
        soutln!("Training MSE: {},  R²: {}", fmt_num(mse), fmt_num(r2));
    }

    // Feature importance
    let mut importance = vec![0usize; n];
    for tree in &trees { count_splits(tree, &mut importance); }
    let total_splits: usize = importance.iter().sum();
    if total_splits > 0 {
        soutln!("\nFeature importance:");
        let mut imp_idx: Vec<(usize, f64)> = importance.iter().enumerate()
            .map(|(i, &c)| (i, c as f64 / total_splits as f64 * 100.0)).collect();
        imp_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, pct) in imp_idx.iter().take(n.min(10)) {
            if *pct > 0.0 {
                let label = col_names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                soutln!("  {}: {}%", label, fmt_num(*pct));
            }
        }
    }

    // Serialize all trees for predict()
    let mut all_feat = Vec::new();
    let mut all_thresh = Vec::new();
    let mut all_pred = Vec::new();
    let mut all_leaf = Vec::new();
    let mut all_left = Vec::new();
    let mut all_right = Vec::new();
    let mut tree_offsets = Vec::new();
    for tree in &trees {
        tree_offsets.push(all_feat.len() as f64);
        serialize_tree(tree, &mut all_feat, &mut all_thresh, &mut all_pred, &mut all_leaf, &mut all_left, &mut all_right);
    }

    let pred_vals: Vec<Real> = final_preds.iter().map(|p| Some(*p)).collect();
    let imp_vals: Vec<Real> = importance.iter().map(|&c| Some(c as f64 / total_splits.max(1) as f64)).collect();
    let mut imp_attrs = Attrs::default();
    imp_attrs.names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());

    let mut fields = HashMap::new();
    fields.insert(Arc::from("predictions"), RVal::Numeric(pred_vals.into(), Attrs::default()));
    fields.insert(Arc::from("f.values"), rnums(&f_vals));
    fields.insert(Arc::from("ntrees"), rnum(ntrees as f64));
    fields.insert(Arc::from("learning_rate"), rnum(lr));
    fields.insert(Arc::from("loss"), rstr(&loss_type));
    fields.insert(Arc::from("train.loss"), rnums(&train_losses));
    fields.insert(Arc::from("importance"), RVal::Numeric(imp_vals.into(), imp_attrs));
    fields.insert(Arc::from("xnames"),
        RVal::Character(col_names.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()));
    fields.insert(Arc::from("_tree_offsets"), rnums(&tree_offsets));
    fields.insert(Arc::from("_tree_feat"), rnums(&all_feat.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_thresh"), rnums(&all_thresh));
    fields.insert(Arc::from("_tree_pred"), rnums(&all_pred));
    fields.insert(Arc::from("_tree_leaf"),
        rnums(&all_leaf.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_left"),
        rnums(&all_left.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    fields.insert(Arc::from("_tree_right"),
        rnums(&all_right.iter().map(|&x| x as f64).collect::<Vec<_>>()));

    Ok(RVal::TypeInstance(TypeInstance {
        type_name: Arc::from("gbm"),
        fields,
    }))
}
