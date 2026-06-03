//! `summary()` for data-shaped inputs — Phase R.2 step 5.
//!
//! Two paths handled here (the data-domain ones):
//!   - `summary(df)`        → R-style per-column statistics (mean, median,
//!                            quartiles, min/max for numerics; top counts
//!                            for character columns)
//!   - `summary(numeric)`   → six-number summary of a single vector
//!
//! Model-instance summaries (`summary(lm_fit)`, `summary(rpart_fit)` etc.)
//! stay in r2-engine because they depend on engine-private helpers and
//! per-model field formatting.
//!
//! Public API: `try_summary(arg) -> Option<()>` — returns `Some(())` when
//! this crate handled the input (data-shaped), `None` to signal "not my
//! kind", letting the engine fall through to model-specific code.
//!
//! This split-handler pattern preserves Phase R's locked decisions:
//!   §4.5 No engine-only deps in r2-data ✓
//!   §4.7 Backwards-compatible — same printed output, just different file ✓
//!   §4.9 No Rayon import here either; uses kernel::par_for via Oracle ✓

use r2_kernel::par_for;
use r2_oracle::Op;
use r2_types::*;
use std::sync::Arc;

/// If `arg` is a data-shaped input (DataFrame or numeric vector), prints
/// the summary and returns `Some(())`. Otherwise returns `None`.
pub fn try_summary(arg: &RVal) -> Option<()> {
    match arg {
        RVal::DataFrame(df) => { summary_dataframe(df); Some(()) }
        RVal::Numeric(v, _) => { summary_numeric(v); Some(()) }
        _ => None,
    }
}

// ── DataFrame path ──────────────────────────────────────────────────

enum ColData {
    Numeric(Vec<f64>),
    Char(Vec<Option<Arc<str>>>),
    AllNA,
    Other(&'static str),
}

fn summary_dataframe(df: &DataFrame) {
    // Stage 1 (sequential): pre-extract per-column data using RVal methods.
    let mut headers: Vec<String> = Vec::with_capacity(df.columns.len());
    let mut prepped: Vec<ColData> = Vec::with_capacity(df.columns.len());
    for (name, col) in &df.columns {
        headers.push(format!("{:^18}", name));
        let item = match col {
            RVal::Numeric(_, _) | RVal::Integer(_, _) => {
                let n: Vec<f64> = col.as_reals().unwrap_or_default()
                    .into_iter().filter_map(|x| x).collect();
                if n.is_empty() { ColData::AllNA } else { ColData::Numeric(n) }
            }
            RVal::Character(vals, _) => ColData::Char(vals.clone()),
            other => ColData::Other(other.type_name()),
        };
        prepped.push(item);
    }

    // Stage 2 (parallel via kernel::par_for): per-column compute.
    let prepped_arc = std::sync::Arc::new(prepped);
    let prepped_for_closure = prepped_arc.clone();
    let n_cols = prepped_arc.len();
    let col_summaries: Vec<Vec<String>> = par_for(Op::PerElementMap, n_cols, move |i| {
        compute_one(&prepped_for_closure[i])
    });

    // Print headers + 6-row summary block.
    for h in &headers { sout!("{}", h); }
    soutln!();
    for row in 0..6 {
        for (ci, _) in headers.iter().enumerate() {
            let s = col_summaries.get(ci).and_then(|c| c.get(row))
                .map(|s| s.as_str()).unwrap_or("");
            sout!("{:<18}", s);
        }
        soutln!();
    }
}

fn compute_one(item: &ColData) -> Vec<String> {
    let fs = |v: f64| -> String {
        if (v - v.round()).abs() < 1e-10 { format!("{}", v.round() as i64) }
        else {
            let s = format!("{:.4}", v);
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    };
    match item {
        ColData::Numeric(data) => {
            let mut n = data.clone();
            n.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = n.len();
            let mean = n.iter().sum::<f64>() / len as f64;
            let median = if len % 2 == 0 { (n[len/2 - 1] + n[len/2]) / 2.0 } else { n[len/2] };
            vec![
                format!(" Min.   :{:>8}", fs(n[0])),
                format!(" 1st Qu.:{:>8}", fs(n[len/4])),
                format!(" Median :{:>8}", fs(median)),
                format!(" Mean   :{:>8}", fs(mean)),
                format!(" 3rd Qu.:{:>8}", fs(n[3 * len / 4])),
                format!(" Max.   :{:>8}", fs(n[len - 1])),
            ]
        }
        ColData::Char(vals) => {
            let mut counts: Vec<(String, usize)> = Vec::new();
            for x in vals {
                if let Some(s) = x {
                    if let Some(entry) = counts.iter_mut().find(|(k, _)| k == s.as_ref()) {
                        entry.1 += 1;
                    } else {
                        counts.push((s.to_string(), 1));
                    }
                }
            }
            counts.sort_by(|a, b| b.1.cmp(&a.1));
            let mut lines: Vec<String> = counts.iter().take(6)
                .map(|(k, v)| format!(" {}:{}", k, v)).collect();
            while lines.len() < 6 { lines.push(String::new()); }
            lines
        }
        ColData::AllNA => vec!["all NA".into(); 6],
        ColData::Other(t) => vec![format!(" {}", t); 6],
    }
}

// ── Numeric vector path ─────────────────────────────────────────────

fn summary_numeric(v: &[Real]) {
    let mut n: Vec<f64> = v.iter().filter_map(|x| *x).collect();
    if n.is_empty() { soutln!("No data"); return; }
    n.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = n.len();
    let mean = n.iter().sum::<f64>() / len as f64;
    let median = if len % 2 == 0 { (n[len/2 - 1] + n[len/2]) / 2.0 } else { n[len/2] };
    soutln!("   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.");
    soutln!("{:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        fmt_num(n[0]), fmt_num(n[len/4]), fmt_num(median),
        fmt_num(mean), fmt_num(n[3 * len / 4]), fmt_num(n[len - 1]));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_handles_dataframe() {
        let df = DataFrame {
            columns: vec![
                (Arc::from("x"), RVal::Numeric(vec![Some(1.0), Some(2.0), Some(3.0)].into(), Attrs::default())),
            ],
            row_names: None,
        };
        let r = try_summary(&RVal::DataFrame(df));
        assert!(r.is_some(), "DataFrame should be handled");
    }

    #[test]
    fn summary_handles_numeric() {
        let v: Vec<Real> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let r = try_summary(&RVal::Numeric(v.into(), Attrs::default()));
        assert!(r.is_some());
    }

    #[test]
    fn summary_passes_through_typeinstance() {
        // A model object: not a data shape — should return None so engine handles it.
        let inst = TypeInstance {
            type_name: Arc::from("rpart"),
            fields: std::collections::HashMap::new(),
        };
        let r = try_summary(&RVal::TypeInstance(inst));
        assert!(r.is_none(), "TypeInstance must NOT be handled by data-domain summary");
    }
}
