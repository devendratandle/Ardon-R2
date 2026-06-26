//! Hash-aggregation kernel — Phase K.10.

// ════════════════════════════════════════════════════════════════════
// Phase K.10 — Hash aggregation
// ════════════════════════════════════════════════════════════════════
//
// Group-by primitive: given parallel `keys` and `values` slices, group
// values by their key and reduce each group. Generic over an `AggOp`
// so `table()`, `tapply()`, group-mean, group-sum, etc. all share one
// kernel.
//
// Implementation: stdlib `HashMap<u64, Accumulator>`. The hashing cost
// is O(n); typical group counts are O(√n) so total memory is sub-linear.
//
// Why a kernel: previously `table()` and similar built their own
// hashing in builtins, missing the parallel dispatch path and
// duplicating bookkeeping logic.

/// Group-by reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    /// Sum of values per group.
    Sum,
    /// Arithmetic mean of values per group.
    Mean,
    /// Count of (non-NA) values per group.
    Count,
    /// Minimum value per group.
    Min,
    /// Maximum value per group.
    Max,
}

/// Result of a hash-aggregation: parallel arrays of unique keys and
/// their reduced values. Order is insertion order (first occurrence of
/// each key in the input).
#[derive(Debug, Clone)]
pub struct HashAggResult {
    pub keys: Vec<u64>,
    pub values: Vec<Option<f64>>,
}

/// Aggregate `values` grouped by `keys` (both same length, parallel
/// arrays). NA values are skipped (na.rm=TRUE behavior); NA keys cause
/// their values to be skipped too.
pub fn hash_agg(
    op: AggOp,
    keys: &[Option<u64>],
    values: &[Option<f64>],
) -> HashAggResult {
    assert_eq!(keys.len(), values.len(), "hash_agg: keys/values length mismatch");
    use std::collections::HashMap;

    // Per-group accumulator. `Count` only needs the count; others need
    // sum + count for mean; min/max need a running extremum.
    #[derive(Clone, Copy)]
    struct Acc {
        sum: f64,
        count: u64,
        ext: f64, // min or max running extremum
    }
    let init = Acc { sum: 0.0, count: 0, ext: match op {
        AggOp::Min => f64::INFINITY,
        AggOp::Max => f64::NEG_INFINITY,
        _ => 0.0,
    }};

    // Preserve insertion order via a Vec<key> + HashMap<key, idx>.
    let mut key_idx: HashMap<u64, usize> = HashMap::with_capacity(keys.len() / 4);
    let mut keys_in_order: Vec<u64> = Vec::new();
    let mut accs: Vec<Acc> = Vec::new();

    for (k, v) in keys.iter().zip(values.iter()) {
        let (kk, vv) = match (k, v) {
            (Some(k), Some(v)) => (*k, *v),
            _ => continue, // skip NA key or value
        };
        let idx = *key_idx.entry(kk).or_insert_with(|| {
            keys_in_order.push(kk);
            accs.push(init);
            keys_in_order.len() - 1
        });
        let a = &mut accs[idx];
        a.sum += vv;
        a.count += 1;
        match op {
            AggOp::Min => a.ext = a.ext.min(vv),
            AggOp::Max => a.ext = a.ext.max(vv),
            _ => {}
        }
    }

    let values_out: Vec<Option<f64>> = accs.iter().map(|a| match op {
        AggOp::Sum   => Some(a.sum),
        AggOp::Mean  => if a.count > 0 { Some(a.sum / a.count as f64) } else { None },
        AggOp::Count => Some(a.count as f64),
        AggOp::Min   => if a.count > 0 { Some(a.ext) } else { None },
        AggOp::Max   => if a.count > 0 { Some(a.ext) } else { None },
    }).collect();

    HashAggResult { keys: keys_in_order, values: values_out }
}

/// Convenience: count occurrences of each unique key (R's `table()`).
/// Equivalent to `hash_agg(AggOp::Count, keys, &vec![Some(1.0); keys.len()])`
/// but skips the values traversal.
pub fn hash_tabulate(keys: &[Option<u64>]) -> HashAggResult {
    use std::collections::HashMap;
    let mut key_idx: HashMap<u64, usize> = HashMap::with_capacity(keys.len() / 4);
    let mut keys_in_order: Vec<u64> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    for k in keys.iter().filter_map(|x| *x) {
        let idx = *key_idx.entry(k).or_insert_with(|| {
            keys_in_order.push(k);
            counts.push(0);
            keys_in_order.len() - 1
        });
        counts[idx] += 1;
    }
    HashAggResult {
        keys: keys_in_order,
        values: counts.into_iter().map(|c| Some(c as f64)).collect(),
    }
}
