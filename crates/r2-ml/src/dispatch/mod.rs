//! ML builtin dispatch — Phase R.1 step 4.
//!
//! Each `bi_*` here is a registry entry point: takes `&[EvalArg]`, returns
//! `Result<RVal, R2Err>`. No engine reference. r2-engine wraps these with
//! a 1-line adapter that satisfies its `BuiltinFn` signature (which carries
//! `&mut Engine` and `&EnvRef` for stateful builtins like `lm`).
//!
//! The adapter is FFI glue, not bloat. The function below is the actual
//! definition of `rpart` — language dispatch happens here.

use crate::tree::{build_tree, tree_predict_one, next_random};
use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

/// Inline single-value-to-string converter (was `val_to_str` in engine).
fn rval_to_string(v: &RVal) -> String {
    match v {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into()))
            .collect::<Vec<_>>().join(" "),
        RVal::Numeric(v, _) => v.iter()
            .map(|x| x.map(fmt_num).unwrap_or("NA".into()))
            .collect::<Vec<_>>().join(" "),
        _ => format!("<{}>", v.type_name()),
    }
}

// Tree models (rpart/rf/gbm) moved to dispatch/trees.rs.
mod trees;
pub use trees::*;

/// K-means clustering. Per-point centroid assignment is parallelized via
/// `kernel::par_for(Op::PerPointDistance, m, ...)`. Centroid recompute and
/// convergence check stay sequential — they're cheap O(m).
pub fn bi_kmeans(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut data = Vec::new();
            let mut names: Vec<Arc<str>> = Vec::new();
            let mut ncol = 0;
            for (n, col) in &df.columns {
                if let Ok(vals) = col.as_reals() {
                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                    if nums.len() == nrow { data.extend(nums); names.push(n.clone()); ncol += 1; }
                }
            }
            let mut m = Matrix::new(data, nrow, ncol);
            m.col_names = Some(names);
            m
        }
        _ => return Err(R2Err { msg: "kmeans() needs a matrix or data.frame".into(), kind: ErrKind::Runtime }),
    };
    let col_names: Vec<String> = match &mat.col_names {
        Some(ns) if ns.len() == mat.ncol => ns.iter().map(|s| s.to_string()).collect(),
        _ => (0..mat.ncol).map(|i| format!("X{}", i + 1)).collect(),
    };

    let k_arg = gn(a, "centers").or_else(|| Some(a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null)));
    let k = k_arg.and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(2.0) as usize;
    let max_iter = gn(a, "iter.max").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(100.0) as usize;

    let (m, n) = (mat.nrow, mat.ncol);
    if k == 0 || k > m {
        return Err(R2Err { msg: "kmeans: k must be between 1 and nrow".into(), kind: ErrKind::Runtime });
    }

    // Initialize centroids with **evenly-spaced** rows rather than the
    // first k. Previously rows 0..k were used; when those rows happened
    // to be in the same true cluster (e.g. the canonical 6-point demo
    // `(1,1) (1.2,0.8) (0.8,1.1) (5,5) (5.2,4.8) (4.9,5.1)`), every
    // point assigned to cluster 0 on iteration 1, matched the initial
    // all-zero `cluster` vector, the loop broke before sizes/centroids
    // were recomputed, and the printout showed `sizes 0, 0`.
    let mut centroids = vec![0.0; k * n];
    for c in 0..k {
        let row = (c * m / k).min(m - 1);
        for j in 0..n { centroids[c * n + j] = mat.get(row, j); }
    }

    // Initialise `cluster` to a value the first iteration cannot reach
    // (usize::MAX) so the convergence check never fires on a trivial
    // first-iteration match that wasn't a real fixed point.
    let mut cluster = vec![usize::MAX; m];
    let mut sizes = vec![0usize; k];
    let nrow = mat.nrow;

    for _iter in 0..max_iter {
        let old_cluster = cluster.clone();

        // Per-point centroid assignment — kernel-dispatched.
        let mat_data = mat.data.clone();
        let centroids_snap = centroids.clone();
        let new_assignments: Vec<usize> = r2_kernel::par_for(
            r2_oracle::Op::PerPointDistance, m, move |i| {
                let mut best_c = 0;
                let mut best_dist = f64::INFINITY;
                for c in 0..k {
                    let mut dist = 0.0;
                    for j in 0..n {
                        let d = mat_data[j * nrow + i] - centroids_snap[c * n + j];
                        dist += d * d;
                    }
                    if dist < best_dist { best_dist = dist; best_c = c; }
                }
                best_c
            }
        );
        for i in 0..m { cluster[i] = new_assignments[i]; }

        let converged = cluster == old_cluster;

        // Recompute centroids + sizes (sequential — O(m), cheap).
        // Done unconditionally so the final `sizes` and `centroids`
        // always reflect the converged assignment, even when the loop
        // breaks on the very first iteration.
        for v in centroids.iter_mut() { *v = 0.0; }
        for s in sizes.iter_mut() { *s = 0; }
        for i in 0..m {
            let c = cluster[i];
            sizes[c] += 1;
            for j in 0..n { centroids[c * n + j] += mat.get(i, j); }
        }
        for c in 0..k {
            if sizes[c] > 0 {
                for j in 0..n { centroids[c * n + j] /= sizes[c] as f64; }
            }
        }

        if converged { break; }
    }

    // Within-cluster + total sum of squares
    let mut withinss = vec![0.0; k];
    let mut totss = 0.0;
    let global_mean: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| mat.get(i, j)).sum::<f64>() / m as f64).collect();
    for i in 0..m {
        let c = cluster[i];
        for j in 0..n {
            let d = mat.get(i, j) - centroids[c * n + j];
            withinss[c] += d * d;
            let dg = mat.get(i, j) - global_mean[j];
            totss += dg * dg;
        }
    }
    let total_withinss: f64 = withinss.iter().sum();
    let betweenss = totss - total_withinss;

    println!("K-means clustering with {} clusters of sizes {}", k,
        sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "));
    println!("\nCluster means:");
    print!("     ");
    for j in 0..n { print!("  {:>10}", col_names.get(j).map(|s| s.as_str()).unwrap_or("?")); }
    println!();
    for c in 0..k {
        print!("  [{}]", c + 1);
        for j in 0..n { print!("  {:>10}", fmt_num(centroids[c * n + j])); }
        println!();
    }
    println!("\nWithin cluster sum of squares by cluster:");
    for w in &withinss { print!("  {}", fmt_num(*w)); }
    println!("\n(between_SS / total_SS = {}%)", fmt_num(betweenss / totss * 100.0));

    let cluster_vals: Vec<Integer> = cluster.iter().map(|c| Some((*c + 1) as i32)).collect();
    let mut centers_mat = Matrix::new(centroids, k, n);
    centers_mat.col_names = Some(col_names.iter().map(|s| Arc::from(s.as_str())).collect());

    let mut fields = HashMap::new();
    fields.insert(Arc::from("cluster"), RVal::Integer(cluster_vals.into(), Attrs::default()));
    fields.insert(Arc::from("centers"), RVal::Matrix(centers_mat));
    fields.insert(Arc::from("withinss"), rnums(&withinss));
    fields.insert(Arc::from("tot.withinss"), rnum(total_withinss));
    fields.insert(Arc::from("betweenss"), rnum(betweenss));
    fields.insert(Arc::from("totss"), rnum(totss));
    fields.insert(Arc::from("size"),
        RVal::Integer(sizes.iter().map(|s| Some(*s as i32)).collect(), Attrs::default()));

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("kmeans"), fields }))
}

/// k-fold cross-validation. Folds run in parallel via
/// `kernel::par_for(Op::KFoldCV, k, ...)`. Each fold trains a model on
/// k-1 partitions, evaluates on the held-out fold, returns MSE.
pub fn bi_cv(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "cv: x must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let y_arg = a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null);
    let y: Vec<f64> = y_arg.as_reals()?.into_iter().filter_map(|x| x).collect();
    let model_type = gn(a, "model").map(|v| rval_to_string(&v)).unwrap_or("lm".into());
    let k = gn(a, "k").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(5.0) as usize;

    let (m, n) = (mat.nrow, mat.ncol);
    if y.len() != m {
        return Err(R2Err { msg: "cv: y length must match x rows".into(), kind: ErrKind::Runtime });
    }
    if k < 2 || k > m {
        return Err(R2Err { msg: "cv: k must be between 2 and nrow".into(), kind: ErrKind::Runtime });
    }

    println!("{}-fold cross-validation (model: {})", k, model_type);
    let fold_size = m / k;

    // Per-fold worker — pure function over its inputs, no shared mutable state.
    // Captured by clone for Send+Sync compliance in the parallel path.
    let mat_data = mat.data.clone();
    let mat_nrow = mat.nrow;
    let y_clone = y.clone();
    let model = model_type.clone();

    let run_fold = move |fold: usize| -> Result<f64, R2Err> {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { m } else { test_start + fold_size };

        // Split into train/test (row-major intermediate)
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        let mut train_rows = 0;
        let mut test_rows = 0;

        for i in 0..m {
            // mat is column-major: mat.get(i,j) = mat_data[j*nrow + i]
            if i >= test_start && i < test_end {
                for c in 0..n { test_x.push(mat_data[c * mat_nrow + i]); }
                test_y.push(y_clone[i]);
                test_rows += 1;
            } else {
                for c in 0..n { train_x.push(mat_data[c * mat_nrow + i]); }
                train_y.push(y_clone[i]);
                train_rows += 1;
            }
        }

        // Rebuild matrices column-major
        let mut train_cm = vec![0.0; train_rows * n];
        let mut test_cm = vec![0.0; test_rows * n];
        for c in 0..n {
            for r in 0..train_rows { train_cm[c * train_rows + r] = train_x[r * n + c]; }
            for r in 0..test_rows { test_cm[c * test_rows + r] = test_x[r * n + c]; }
        }

        let metric = match model.as_str() {
            "lm" => {
                let mut x_int = vec![1.0; train_rows];
                x_int.extend(&train_cm);
                let p = n + 1;
                let coeffs = r2_linalg::dlsq_fused(train_rows, p, &x_int, &train_y)
                    .map_err(|e| R2Err {
                        msg: format!("cv lm failed: {}", e),
                        kind: ErrKind::Runtime,
                    })?;
                let mut preds = vec![0.0; test_rows];
                for i in 0..test_rows {
                    preds[i] = coeffs[0];
                    for j in 0..n { preds[i] += coeffs[j + 1] * test_cm[j * test_rows + i]; }
                }
                preds.iter().zip(test_y.iter())
                    .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / test_rows as f64
            }
            "rf" => {
                let train_mat_data = train_cm.clone();
                let mut all_preds = vec![vec![0.0; test_rows]; 50];
                for t in 0..50 {
                    let mut bmask = vec![false; train_rows];
                    for _ in 0..train_rows {
                        bmask[(next_random() * train_rows as f64) as usize % train_rows] = true;
                    }
                    let tree = build_tree(&train_mat_data, &train_y, train_rows, n, &bmask, 5, 2, 0, false);
                    for i in 0..test_rows {
                        all_preds[t][i] = tree_predict_one(&tree, &test_cm, test_rows, i);
                    }
                }
                let mut preds = vec![0.0; test_rows];
                for i in 0..test_rows {
                    preds[i] = all_preds.iter().map(|p| p[i]).sum::<f64>() / 50.0;
                }
                preds.iter().zip(test_y.iter())
                    .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / test_rows as f64
            }
            other => {
                return Err(R2Err {
                    msg: format!("cv: model '{}' not supported (use lm, rf)", other),
                    kind: ErrKind::Runtime,
                });
            }
        };

        println!("  Fold {}: MSE = {}", fold + 1, fmt_num(metric));
        Ok(metric)
    };

    // Kernel-dispatched parallel folds. Each fold returns Result; we
    // collect, then surface the first error if any.
    let fold_results: Vec<Result<f64, R2Err>> =
        r2_kernel::par_for(r2_oracle::Op::KFoldCV, k, run_fold);

    let mut fold_metrics = Vec::with_capacity(k);
    for r in fold_results {
        fold_metrics.push(r?);
    }

    let avg = fold_metrics.iter().sum::<f64>() / k as f64;
    let sd = (fold_metrics.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / (k - 1) as f64).sqrt();
    println!("\nAverage MSE: {} (±{})", fmt_num(avg), fmt_num(sd));

    let mut fields = HashMap::new();
    fields.insert(Arc::from("mse"), rnums(&fold_metrics));
    fields.insert(Arc::from("mean.mse"), rnum(avg));
    fields.insert(Arc::from("sd.mse"), rnum(sd));
    fields.insert(Arc::from("k"), rnum(k as f64));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("cv"), fields }))
}

/// k-nearest-neighbours classification. No parallelism — direct migration.
/// (Future K.5 could par_for over test points; not now.)
pub fn bi_knn(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let train = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "knn: train must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let arg1 = a.get(1).map(|e| &e.value).unwrap_or(&RVal::Null);
    let test = match arg1 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "knn: test must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let labels: Vec<f64> = a.get(2).map(|e| e.value.clone()).unwrap_or(RVal::Null)
        .as_reals()?.into_iter().filter_map(|x| x).collect();
    let k = gn(a, "k").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(3.0) as usize;

    let (n_train, p) = (train.nrow, train.ncol);
    let n_test = test.nrow;
    if labels.len() != n_train {
        return Err(R2Err { msg: "knn: labels length must match train rows".into(), kind: ErrKind::Runtime });
    }

    let mut predictions = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let mut dists: Vec<(f64, usize)> = (0..n_train).map(|j| {
            let mut d = 0.0;
            for col in 0..p { let diff = test.get(i, col) - train.get(j, col); d += diff * diff; }
            (d, j)
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut votes: HashMap<i64, usize> = HashMap::new();
        for idx in 0..k.min(n_train) {
            let label = labels[dists[idx].1] as i64;
            *votes.entry(label).or_insert(0) += 1;
        }
        let best = votes.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l).unwrap_or(0);
        predictions.push(Some(best as f64));
    }

    println!("KNN: classified {} points using k={}", n_test, k);
    Ok(RVal::Numeric(predictions.into(), Attrs::default()))
}

/// Gaussian Naive Bayes classifier. No parallelism — direct migration.
pub fn bi_naive_bayes(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "naive.bayes: x must be matrix".into(), kind: ErrKind::Runtime }),
    };
    let labels: Vec<f64> = a.get(1).map(|e| e.value.clone()).unwrap_or(RVal::Null)
        .as_reals()?.into_iter().filter_map(|x| x).collect();

    let (m, n) = (mat.nrow, mat.ncol);
    if labels.len() != m {
        return Err(R2Err { msg: "naive.bayes: labels length must match rows".into(), kind: ErrKind::Runtime });
    }

    let mut classes: Vec<f64> = labels.clone();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let k = classes.len();
    let mut class_means = vec![0.0; k * n];
    let mut class_vars = vec![0.0; k * n];
    let mut class_counts = vec![0usize; k];
    let mut class_priors = vec![0.0; k];

    for i in 0..m {
        let ci = classes.iter().position(|&c| (c - labels[i]).abs() < 1e-10).unwrap_or(0);
        class_counts[ci] += 1;
        for j in 0..n { class_means[ci * n + j] += mat.get(i, j); }
    }
    for ci in 0..k {
        class_priors[ci] = class_counts[ci] as f64 / m as f64;
        for j in 0..n { class_means[ci * n + j] /= class_counts[ci] as f64; }
    }
    for i in 0..m {
        let ci = classes.iter().position(|&c| (c - labels[i]).abs() < 1e-10).unwrap_or(0);
        for j in 0..n {
            let d = mat.get(i, j) - class_means[ci * n + j];
            class_vars[ci * n + j] += d * d;
        }
    }
    for ci in 0..k {
        for j in 0..n { class_vars[ci * n + j] /= (class_counts[ci] - 1).max(1) as f64; }
    }

    println!("Naive Bayes classifier: {} classes, {} features", k, n);
    for ci in 0..k {
        println!("  Class {}: prior={}, n={}", classes[ci], fmt_num(class_priors[ci]), class_counts[ci]);
    }

    let mut fields = HashMap::new();
    fields.insert(Arc::from("classes"), rnums(&classes));
    fields.insert(Arc::from("priors"), rnums(&class_priors));
    fields.insert(Arc::from("means"), RVal::Matrix(Matrix::new(class_means, k, n)));
    fields.insert(Arc::from("vars"), RVal::Matrix(Matrix::new(class_vars, k, n)));

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("naive.bayes"), fields }))
}

/// Principal Component Analysis. No parallelism — uses r2-linalg's
/// symmetric-eigendecomposition kernel for the heavy math.
pub fn bi_prcomp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg0 = a.first().map(|e| &e.value).unwrap_or(&RVal::Null);
    let mat = match arg0 {
        RVal::Matrix(m) => m.clone(),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut data = Vec::new();
            let mut col_names: Vec<Arc<str>> = Vec::new();
            let mut ncol_num = 0;
            for (name, col) in &df.columns {
                if let Ok(vals) = col.as_reals() {
                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                    if nums.len() == nrow {
                        data.extend(nums);
                        col_names.push(name.clone());
                        ncol_num += 1;
                    }
                }
            }
            let mut m = Matrix::new(data, nrow, ncol_num);
            m.col_names = Some(col_names);
            m
        }
        _ => return Err(R2Err { msg: "prcomp() needs a matrix or data.frame".into(), kind: ErrKind::Runtime }),
    };
    let _col_labels: Vec<String> = match &mat.col_names {
        Some(ns) if ns.len() == mat.ncol => ns.iter().map(|s| s.to_string()).collect(),
        _ => (0..mat.ncol).map(|i| format!("X{}", i + 1)).collect(),
    };

    let center = gn(a, "center").and_then(|v| v.as_logicals().ok())
        .map(|v| v[0] == Some(true)).unwrap_or(true);
    let scale = gn(a, "scale.").or_else(|| gn(a, "scale"))
        .and_then(|v| v.as_logicals().ok()).map(|v| v[0] == Some(true)).unwrap_or(false);

    let (m, n) = (mat.nrow, mat.ncol);

    let mut x = mat.data.clone();
    let means = mat.col_means();
    let mut sds = vec![1.0; n];
    if scale {
        for c in 0..n {
            let col_start = c * m;
            let mean = means[c];
            let mut ss = 0.0;
            for r in 0..m { ss += (x[col_start + r] - mean).powi(2); }
            sds[c] = (ss / (m - 1) as f64).sqrt();
            if sds[c] < 1e-15 { sds[c] = 1.0; }
        }
    }
    if center {
        for c in 0..n {
            let col_start = c * m;
            for r in 0..m {
                x[col_start + r] -= means[c];
                if scale { x[col_start + r] /= sds[c]; }
            }
        }
    }

    // Covariance via Kahan summation
    let mut cov_data = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            let mut sum = 0.0;
            let mut comp = 0.0;
            for r in 0..m {
                let prod = x[i * m + r] * x[j * m + r];
                let y_k = prod - comp;
                let t = sum + y_k;
                comp = (t - sum) - y_k;
                sum = t;
            }
            let cov = sum / (m - 1) as f64;
            cov_data[j * n + i] = cov;
            cov_data[i * n + j] = cov;
        }
    }

    // Tier 1: eigenvectors of the covariance matrix ARE the rotation matrix.
    // Switch to dsyev_full so `$rotation` is now populated.
    let (eigenvalues, rotation_data) = r2_linalg::dsyev_full(n, &cov_data)
        .map_err(|e| R2Err { msg: format!("PCA failed: {}", e), kind: ErrKind::Runtime })?;

    let eigenvalues: Vec<f64> = eigenvalues.iter()
        .map(|v| if *v < 0.0 && v.abs() < 1e-10 { 0.0 } else { *v }).collect();
    let sdev: Vec<f64> = eigenvalues.iter().map(|v| v.max(0.0).sqrt()).collect();
    let total_var: f64 = eigenvalues.iter().filter(|v| **v > 0.0).sum();
    let prop_var: Vec<f64> = eigenvalues.iter()
        .map(|v| if total_var > 0.0 { v.max(0.0) / total_var } else { 0.0 }).collect();

    // No auto-print: R's `p <- prcomp(X)` is invisible. Users call
    // `summary(p)` for the formatted importance-of-components table,
    // or access `p$sdev` / `p$rotation` directly.
    let _ = &sdev;
    let _ = &prop_var;

    let mut rotation_mat = Matrix::new(rotation_data, n, n);
    rotation_mat.col_names = Some(
        (0..n).map(|i| Arc::from(format!("PC{}", i + 1).as_str())).collect()
    );

    let mut fields = HashMap::new();
    fields.insert(Arc::from("sdev"), rnums(&sdev));
    fields.insert(Arc::from("eigenvalues"), rnums(&eigenvalues));
    fields.insert(Arc::from("prop.variance"), rnums(&prop_var));
    fields.insert(Arc::from("center"), rnums(&means));
    fields.insert(Arc::from("rotation"), RVal::Matrix(rotation_mat));
    if scale { fields.insert(Arc::from("scale"), rnums(&sds)); }

    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("prcomp"), fields }))
}
