//! Tree-based ML kernels — Phase R.1 step 1.
//!
//! Pure CART-style decision-tree primitives extracted from r2-engine.
//! No engine reference, no RVal, no global state — these are algorithmic
//! building blocks that any consumer (r2-engine builtins, addon
//! packages, the future general-purpose Rust ML library) can use.
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only (none beyond std).
//!   §4.9 No parallelism in this module — callers (r2-ml::bi_* wrappers,
//!        future kernels) own the par/serial choice.
//!
//! Public surface:
//!   - `TreeNode`              — recursive CART node
//!   - `build_tree(...)`       — fit a single tree (regression or classification)
//!   - `tree_predict_one(...)` — predict one row through a tree
//!   - `count_splits(...)`     — accumulate per-feature split frequencies
//!   - `serialize_tree(...)`   — flatten a tree for storage / RVal output
//!   - `parallel_random(seed)` — thread-local LCG step for ensembles

use std::collections::HashMap;
use r2_kernel::par_for;
use r2_oracle::Op;

// Phase R.12: RNG home consolidated in r2_stats::rng. Re-exports below
// keep all existing `r2_ml::tree::{SEED_STATE, next_random, current_seed,
// set_seed}` call sites working without source changes.
pub use r2_stats::rng::{SEED_STATE, current_seed, set_seed, next_random};

#[derive(Debug, Clone)]
pub struct TreeNode {
    pub is_leaf: bool,
    pub prediction: f64,    // class (classification) or mean (regression)
    pub feature: usize,     // split feature index
    pub threshold: f64,     // split threshold
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
    pub n_samples: usize,
    pub impurity: f64,
}

// ── Impurity measures ────────────────────────────────────────────────

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

// ── Tree construction ────────────────────────────────────────────────

/// Recursively fit a CART tree.
/// `x` is column-major: feature `f` row `i` lives at `x[f * m + i]`.
/// `row_mask[i] == true` means row `i` participates in this subtree.
pub fn build_tree(
    x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool,
) -> TreeNode {
    let active: Vec<usize> = row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
    let count = active.len();

    let prediction = if is_classification {
        let mut votes: HashMap<i64, usize> = HashMap::new();
        for &i in &active { *votes.entry(y[i] as i64).or_insert(0) += 1; }
        votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0)
    } else {
        active.iter().map(|&i| y[i]).sum::<f64>() / count.max(1) as f64
    };

    if count <= min_samples || depth >= max_depth {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }
    let all_same = active.windows(2).all(|w| (y[w[0]] - y[w[1]]).abs() < 1e-10);
    if all_same {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    let parent_impurity = if is_classification { gini(&active, y) } else { mse_impurity(&active, y) };

    // Per-feature best-split closure. Pure function of `feat` + shared
    // read-only state (x, y, active) — perfect for parallel evaluation
    // across the feature axis. Each call returns the locally-best
    // (gain, threshold) for that feature; we reduce after the fan-out.
    //
    // Oracle dispatches Serial vs Rayon based on n_features × n_samples;
    // for shallow trees with few features this stays serial automatically.
    let eval_feature = |feat: usize| -> (f64, f64) {
        let mut indexed: Vec<(f64, usize)> = active.iter()
            .map(|&i| (x[feat * m + i], i)).collect();
        indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal));
        let mut local_gain = 0.0f64;
        let mut local_thresh = 0.0f64;

        if is_classification {
            let mut max_class = 0i64;
            for &(_, idx) in &indexed { max_class = max_class.max(y[idx] as i64); }
            let nc = (max_class + 1) as usize;
            if nc > 1000 { return (local_gain, local_thresh); }
            let mut right_counts = vec![0usize; nc];
            for &(_, idx) in &indexed {
                let c = y[idx] as usize;
                if c < nc { right_counts[c] += 1; }
            }
            let mut left_counts = vec![0usize; nc];
            let mut left_n = 0usize;
            let mut right_n = count;
            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;
            for i in 0..indexed.len() - 1 {
                let c = y[indexed[i].1] as usize;
                if c < nc { left_counts[c] += 1; right_counts[c] -= 1; }
                left_n += 1;
                right_n -= 1;
                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }
                last_split = i;
                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;
                let left_gini  = 1.0 - left_counts.iter().map(|&c| { let p = c as f64 / left_n as f64; p * p }).sum::<f64>();
                let right_gini = 1.0 - right_counts.iter().map(|&c| { let p = c as f64 / right_n as f64; p * p }).sum::<f64>();
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / count as f64;
                let gain = parent_impurity - weighted;
                if gain > local_gain { local_gain = gain; local_thresh = threshold; }
            }
        } else {
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
                let left_mse  = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_mse = (total_sq - left_sq) / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse) / count as f64;
                let gain = parent_impurity - weighted;
                if gain > local_gain { local_gain = gain; local_thresh = threshold; }
            }
        }
        (local_gain, local_thresh)
    };

    // Parallel fan-out across features. Oracle's threshold is keyed on
    // n × count (features × active samples) so small problems still run
    // serial without thread-pool overhead.
    let per_feat = par_for(Op::PerElementMap, n, eval_feature);
    let (mut best_gain, mut best_feature, mut best_threshold) = (0.0f64, 0usize, 0.0f64);
    for (feat, (g, t)) in per_feat.into_iter().enumerate() {
        if g > best_gain { best_gain = g; best_feature = feat; best_threshold = t; }
    }

    if best_gain <= 0.0 {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: parent_impurity };
    }

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

// ── Prediction ───────────────────────────────────────────────────────

pub fn tree_predict_one(node: &TreeNode, x: &[f64], m: usize, row: usize) -> f64 {
    if node.is_leaf { return node.prediction; }
    let val = x[node.feature * m + row];
    if val <= node.threshold {
        node.left.as_ref().map(|n| tree_predict_one(n, x, m, row)).unwrap_or(node.prediction)
    } else {
        node.right.as_ref().map(|n| tree_predict_one(n, x, m, row)).unwrap_or(node.prediction)
    }
}

// ── Split-frequency importance ───────────────────────────────────────

pub fn count_splits(node: &TreeNode, importance: &mut Vec<usize>) {
    if node.is_leaf { return; }
    if node.feature < importance.len() { importance[node.feature] += 1; }
    if let Some(ref l) = node.left { count_splits(l, importance); }
    if let Some(ref r) = node.right { count_splits(r, importance); }
}

// ── Tree serialization (for RVal storage) ────────────────────────────

pub fn serialize_tree(
    node: &TreeNode,
    feat: &mut Vec<usize>, thresh: &mut Vec<f64>, pred: &mut Vec<f64>,
    leaf: &mut Vec<bool>, left: &mut Vec<usize>, right: &mut Vec<usize>,
) {
    let idx = feat.len();
    feat.push(node.feature);
    thresh.push(node.threshold);
    pred.push(node.prediction);
    leaf.push(node.is_leaf);
    left.push(0); right.push(0);
    if !node.is_leaf {
        if let Some(ref l) = node.left {
            left[idx] = feat.len();
            serialize_tree(l, feat, thresh, pred, leaf, left, right);
        }
        if let Some(ref r) = node.right {
            right[idx] = feat.len();
            serialize_tree(r, feat, thresh, pred, leaf, left, right);
        }
    }
}

// Phase R.12: parallel_random moved to r2_stats::rng. Re-exported here
// for backward-compatible call sites in this crate (rf bootstrap, gbm
// subsampling) and in r2_ml::dispatch.
pub use r2_stats::rng::parallel_random;

/// R-style indented tree printer. Prints to stdout; used by `bi_rpart`.
pub fn print_tree(node: &TreeNode, depth: usize, col_names: &[String]) {
    let indent = "  ".repeat(depth);
    if node.is_leaf {
        soutln!("{}* predict: {} (n={})", indent, r2_types::fmt_num(node.prediction), node.n_samples);
    } else {
        let fname = col_names.get(node.feature).map(|s| s.as_str()).unwrap_or("?");
        soutln!("{}{} <= {} (n={})", indent, fname, r2_types::fmt_num(node.threshold), node.n_samples);
        if let Some(ref left) = node.left { print_tree(left, depth + 1, col_names); }
        soutln!("{}{} > {}", indent, fname, r2_types::fmt_num(node.threshold));
        if let Some(ref right) = node.right { print_tree(right, depth + 1, col_names); }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tiny_regression_tree() {
        // y = 2x for x in [0..10]; tree should split somewhere reasonable.
        let m = 10;
        let n = 1;
        let x: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..m).map(|i| 2.0 * i as f64).collect();
        let mask = vec![true; m];
        let tree = build_tree(&x, &y, m, n, &mask, 3, 1, 0, false);
        // Leaf prediction at root sample mean
        let pred_mid = tree_predict_one(&tree, &x, m, 5);
        assert!(pred_mid > 0.0 && pred_mid < 20.0);
    }

    #[test]
    fn count_splits_counts_correctly() {
        let m = 8;
        let n = 1;
        let x: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..m).map(|i| if i < 4 { 0.0 } else { 1.0 }).collect();
        let mask = vec![true; m];
        let tree = build_tree(&x, &y, m, n, &mask, 3, 1, 0, true);
        let mut imp = vec![0; n];
        count_splits(&tree, &mut imp);
        assert!(imp[0] >= 1, "expected at least one split on feature 0");
    }

    #[test]
    fn parallel_random_in_unit_range() {
        let mut s = 42u64;
        for _ in 0..1000 {
            let r = parallel_random(&mut s);
            assert!(r >= 0.0 && r < 1.0, "rng out of range: {}", r);
        }
    }

    #[test]
    fn serialize_round_trip_lengths() {
        let m = 6;
        let x: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mask = vec![true; m];
        let tree = build_tree(&x, &y, m, 1, &mask, 2, 1, 0, true);
        let mut feat = vec![]; let mut thresh = vec![]; let mut pred = vec![];
        let mut leaf = vec![]; let mut left = vec![]; let mut right = vec![];
        serialize_tree(&tree, &mut feat, &mut thresh, &mut pred, &mut leaf, &mut left, &mut right);
        assert_eq!(feat.len(), thresh.len());
        assert_eq!(feat.len(), pred.len());
        assert_eq!(feat.len(), leaf.len());
        assert!(feat.len() >= 1);
    }
}
