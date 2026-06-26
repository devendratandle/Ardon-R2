//! `xts` irregular time series — Phase R.T.3.

use super::*;
use r2_types::{RVal, Attrs, EvalArg, R2Err, ErrKind};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.3 — xts irregular time series
//
// Storage matches R's xts package philosophy: a numeric matrix (column-major
// vector + dim attr) with `class = "xts"` and three custom attributes:
//
//   * `index`        — numeric vector of n seconds-since-epoch (POSIXct)
//                      or days-since-epoch (Date), one per row
//   * `index.class`  — "Date" or "POSIXct"
//   * `col.names`    — optional Vec<Arc<str>> stored via Attrs.names
//                      (we use Attrs.names because a matrix-with-dim
//                      doesn't otherwise carry column labels)
//
// The vector's underlying values flow through normal numeric arithmetic.
// All xts machinery (subset, merge, na.locf, first/last) operates on the
// index-aware envelope and reconstructs the result with a fresh xts attrs
// bundle.
// ═══════════════════════════════════════════════════════════════════════

pub fn xts_attrs(nrow: usize, ncol: usize, index: Vec<f64>, index_class: &str, col_names: Option<Vec<Arc<str>>>) -> Attrs {
    let mut a = Attrs::default();
    a.class = Some(Arc::from("xts"));
    a.dim = Some(vec![nrow, ncol]);
    a.custom.insert(Arc::from("index"), RVal::Numeric(
        index.into_iter().map(Some).collect::<Vec<_>>().into(), Attrs::default()));
    a.custom.insert(Arc::from("index.class"), RVal::Character(
        vec![Some(Arc::from(index_class))], Attrs::default()));
    if let Some(cn) = col_names {
        a.names = Some(cn);
    }
    a
}

pub fn get_xts(v: &RVal) -> Result<(&[Option<f64>], usize, usize, Vec<f64>, String, Option<Vec<Arc<str>>>), R2Err> {
    match v {
        RVal::Numeric(xs, attrs) if attrs.class.as_deref() == Some("xts") => {
            let dim = attrs.dim.as_ref().ok_or_else(|| R2Err { msg: "xts: missing dim".into(), kind: ErrKind::Type })?;
            if dim.len() != 2 { return Err(R2Err { msg: "xts: dim must be 2-D".into(), kind: ErrKind::Type }); }
            let (nrow, ncol) = (dim[0], dim[1]);
            let idx = attrs.custom.get(&Arc::from("index"))
                .ok_or_else(|| R2Err { msg: "xts: missing index attr".into(), kind: ErrKind::Type })?;
            let idx_class = attrs.custom.get(&Arc::from("index.class"));
            let cls = match idx_class {
                Some(RVal::Character(cs, _)) => cs.first().and_then(|x| x.as_ref().map(|s| s.to_string())).unwrap_or_else(|| "POSIXct".into()),
                _ => "POSIXct".into(),
            };
            let idx_vec: Vec<f64> = match idx {
                RVal::Numeric(xs, _) => xs.iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
                _ => return Err(R2Err { msg: "xts: index must be numeric".into(), kind: ErrKind::Type }),
            };
            Ok((xs, nrow, ncol, idx_vec, cls, attrs.names.clone()))
        }
        _ => Err(R2Err { msg: "not an xts object".into(), kind: ErrKind::Type }),
    }
}

/// `xts(data, order.by = ...)` — data is a numeric vector or matrix
/// (numeric with dim attr); `order.by` is a Date or POSIXct vector.
pub fn bi_xts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let data = gv(a, 0);
    let order_by = named(a, "order.by").ok_or_else(|| R2Err {
        msg: "xts(): 'order.by' is required (use a Date or POSIXct vector)".into(),
        kind: ErrKind::Runtime,
    })?;

    let (idx, idx_class) = match order_by {
        RVal::Numeric(xs, attrs) => {
            let cls = match attrs.class.as_deref() {
                Some("Date") => "Date",
                Some("POSIXct") => "POSIXct",
                _ => "POSIXct",
            };
            let v: Vec<f64> = xs.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            (v, cls)
        }
        _ => return Err(R2Err { msg: "xts(): order.by must be Date or POSIXct".into(), kind: ErrKind::Type }),
    };

    let (vals, nrow, ncol, col_names) = match data {
        RVal::Numeric(xs, attrs) => {
            let v: Vec<Option<f64>> = xs.iter().copied().collect();
            match attrs.dim.as_ref() {
                Some(d) if d.len() == 2 => (v, d[0], d[1], attrs.names.clone()),
                _ => {
                    let n = xs.len();
                    (v, n, 1, None)
                }
            }
        }
        RVal::Matrix(m) => {
            let v: Vec<Option<f64>> = m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect();
            (v, m.nrow, m.ncol, m.col_names.clone())
        }
        RVal::Integer(xs, _) => {
            let v: Vec<Option<f64>> = xs.iter().map(|x| x.map(|n| n as f64)).collect();
            let n = v.len();
            (v, n, 1, None)
        }
        other => return Err(R2Err { msg: format!("xts(): data must be numeric (got '{}')", other.type_name()), kind: ErrKind::Type }),
    };

    if idx.len() != nrow {
        return Err(R2Err {
            msg: format!("xts(): length of order.by ({}) must equal nrow of data ({})", idx.len(), nrow),
            kind: ErrKind::Runtime,
        });
    }

    // Sort by index ascending — xts requires this.
    let mut order: Vec<usize> = (0..nrow).collect();
    order.sort_by(|&i, &j| idx[i].partial_cmp(&idx[j]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_idx: Vec<f64> = order.iter().map(|&i| idx[i]).collect();
    let mut sorted_vals: Vec<Option<f64>> = Vec::with_capacity(vals.len());
    for c in 0..ncol {
        for &r in &order {
            sorted_vals.push(vals[c * nrow + r]);
        }
    }

    Ok(RVal::Numeric(sorted_vals.into(), xts_attrs(nrow, ncol, sorted_idx, idx_class, col_names)))
}

/// `index(x)` — returns the time index as a Date or POSIXct vector.
pub fn bi_index(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (_, _, _, idx, cls, _) = get_xts(v)?;
    let mut attrs = Attrs::default();
    attrs.class = Some(Arc::from(cls.as_str()));
    Ok(RVal::Numeric(idx.into_iter().map(Some).collect::<Vec<_>>().into(), attrs))
}

/// `coredata(x)` — returns the data without the index, as a plain matrix.
pub fn bi_coredata(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (xs, nrow, ncol, _, _, col_names) = get_xts(v)?;
    let mut attrs = Attrs::default();
    attrs.dim = Some(vec![nrow, ncol]);
    attrs.names = col_names;
    Ok(RVal::Numeric(xs.iter().copied().collect::<Vec<_>>().into(), attrs))
}

pub fn bi_is_xts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let is = matches!(gv(a, 0), RVal::Numeric(_, attrs) if attrs.class.as_deref() == Some("xts"));
    Ok(RVal::Logical(vec![Some(is)].into(), Attrs::default()))
}

/// `xts.subset(x, "2024-01/2024-03")` — date-string range subset.
/// Accepts: "YYYY", "YYYY-MM", "YYYY-MM-DD" (point or open-ended range),
/// or any of those separated by "/" (closed range).
/// "/2024-03" means "from start through 2024-03". "2024-01/" means "from 2024-01 onward".
pub fn bi_xts_subset(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (xs, nrow, ncol, idx, idx_class, col_names) = get_xts(v)?;
    let range = named(a, "range")
        .or_else(|| Some(gv(a, 1)))
        .and_then(|v| if let RVal::Character(cs, _) = v { cs.first().and_then(|x| x.as_ref().map(|s| s.to_string())) } else { None })
        .ok_or_else(|| R2Err { msg: "xts.subset(): need a date-range character string".into(), kind: ErrKind::Runtime })?;

    let (lo, hi) = parse_range(&range, &idx_class)?;
    let mut keep: Vec<usize> = Vec::new();
    for (i, &t) in idx.iter().enumerate() {
        if t >= lo && t <= hi { keep.push(i); }
    }
    if keep.is_empty() {
        // Return an empty xts with the same shape (0 rows, ncol cols).
        return Ok(RVal::Numeric(Vec::<Option<f64>>::new().into(),
            xts_attrs(0, ncol, vec![], &idx_class, col_names)));
    }
    let new_n = keep.len();
    let new_idx: Vec<f64> = keep.iter().map(|&i| idx[i]).collect();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(new_n * ncol);
    for c in 0..ncol {
        for &r in &keep {
            out.push(xs[c * nrow + r]);
        }
    }
    Ok(RVal::Numeric(out.into(), xts_attrs(new_n, ncol, new_idx, &idx_class, col_names)))
}

/// Parse "2024-01", "2024-01-15", or "lo/hi" forms into a numeric index range.
fn parse_range(range: &str, idx_class: &str) -> Result<(f64, f64), R2Err> {
    let (lo_str, hi_str) = match range.find('/') {
        Some(i) => (&range[..i], &range[i+1..]),
        None    => (range, range),
    };
    let lo = endpoint_low(lo_str, idx_class)?;
    let hi = endpoint_high(hi_str, idx_class)?;
    if lo > hi {
        return Err(R2Err { msg: format!("xts.subset(): low > high in range '{}'", range), kind: ErrKind::Runtime });
    }
    Ok((lo, hi))
}

fn endpoint_low(s: &str, idx_class: &str) -> Result<f64, R2Err> {
    if s.is_empty() { return Ok(f64::NEG_INFINITY); }
    let scale = if idx_class == "Date" { 1.0 } else { 86_400.0 };
    let (y, m, d) = parse_partial(s)?;
    let day = days_from_civil(y, m.unwrap_or(1), d.unwrap_or(1));
    Ok(day as f64 * scale)
}

fn endpoint_high(s: &str, idx_class: &str) -> Result<f64, R2Err> {
    if s.is_empty() { return Ok(f64::INFINITY); }
    let _scale = if idx_class == "Date" { 1.0 } else { 86_400.0 };
    let (y, m, d) = parse_partial(s)?;
    let (hy, hm, hd) = match (m, d) {
        (None, _) => (y + 1, 1u32, 1u32),               // YYYY → exclusive end at next year
        (Some(mm), None) => {
            if mm == 12 { (y + 1, 1u32, 1u32) } else { (y, mm + 1, 1u32) }
        }
        (Some(mm), Some(dd)) => (y, mm, dd + 1),         // YYYY-MM-DD → exclusive end next day
    };
    // Subtract 1 second (or 1 day for Date) to make it inclusive.
    let end_day = days_from_civil(hy, hm, hd);
    if idx_class == "Date" {
        Ok(end_day as f64 - 1.0)
    } else {
        Ok(end_day as f64 * 86_400.0 - 1.0)
    }
}

fn parse_partial(s: &str) -> Result<(i64, Option<u32>, Option<u32>), R2Err> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.is_empty() || parts.iter().any(|p| p.is_empty()) {
        return Err(R2Err { msg: format!("xts.subset(): bad date '{}'", s), kind: ErrKind::Runtime });
    }
    let y: i64 = parts[0].parse().map_err(|_| R2Err { msg: format!("xts.subset(): bad year in '{}'", s), kind: ErrKind::Runtime })?;
    let m = if parts.len() >= 2 {
        Some(parts[1].parse().map_err(|_| R2Err { msg: format!("xts.subset(): bad month in '{}'", s), kind: ErrKind::Runtime })?)
    } else { None };
    let d = if parts.len() >= 3 {
        Some(parts[2].parse().map_err(|_| R2Err { msg: format!("xts.subset(): bad day in '{}'", s), kind: ErrKind::Runtime })?)
    } else { None };
    Ok((y, m, d))
}

/// `first(x, n=6)` — first n rows. `last(x, n=6)` — last n rows.
pub fn bi_first(a: &[EvalArg]) -> Result<RVal, R2Err> { first_or_last(a, true) }
pub fn bi_last(a: &[EvalArg])  -> Result<RVal, R2Err> { first_or_last(a, false) }

fn first_or_last(a: &[EvalArg], first: bool) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (xs, nrow, ncol, idx, idx_class, col_names) = get_xts(v)?;
    let n: usize = named(a, "n").or_else(|| Some(gv(a, 1)))
        .and_then(|v| match v {
            RVal::Numeric(vs, _) => vs.first().and_then(|x| *x).map(|x| x as usize),
            RVal::Integer(vs, _) => vs.first().and_then(|x| *x).map(|x| x as usize),
            _ => None,
        }).unwrap_or(6).min(nrow);
    let rows: Vec<usize> = if first { (0..n).collect() } else { (nrow - n..nrow).collect() };
    let new_idx: Vec<f64> = rows.iter().map(|&i| idx[i]).collect();
    let mut out: Vec<Option<f64>> = Vec::with_capacity(n * ncol);
    for c in 0..ncol {
        for &r in &rows {
            out.push(xs[c * nrow + r]);
        }
    }
    Ok(RVal::Numeric(out.into(), xts_attrs(n, ncol, new_idx, &idx_class, col_names)))
}

/// `na.locf(x)` — last observation carried forward (xts and numeric vectors).
pub fn bi_na_locf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    if let RVal::Numeric(_, attrs) = v {
        if attrs.class.as_deref() == Some("xts") {
            let (xs, nrow, ncol, idx, idx_class, col_names) = get_xts(v)?;
            let mut out: Vec<Option<f64>> = xs.iter().copied().collect();
            for c in 0..ncol {
                let mut last: Option<f64> = None;
                for r in 0..nrow {
                    let pos = c * nrow + r;
                    match out[pos] {
                        Some(v) => last = Some(v),
                        None => if let Some(lv) = last { out[pos] = Some(lv); },
                    }
                }
            }
            return Ok(RVal::Numeric(out.into(), xts_attrs(nrow, ncol, idx, &idx_class, col_names)));
        }
    }
    // Fallback for plain numeric vectors.
    match v {
        RVal::Numeric(xs, attrs) => {
            let mut out: Vec<Option<f64>> = xs.iter().copied().collect();
            let mut last: Option<f64> = None;
            for x in out.iter_mut() {
                match *x { Some(v) => last = Some(v), None => if let Some(lv) = last { *x = Some(lv); } }
            }
            Ok(RVal::Numeric(out.into(), attrs.clone()))
        }
        other => Err(R2Err { msg: format!("na.locf(): cannot handle '{}'", other.type_name()), kind: ErrKind::Type }),
    }
}

/// `merge.xts(a, b, ...)` — outer join by time index, NA-fills gaps.
/// All inputs must be xts with the same index.class.
pub fn bi_merge_xts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    if a.len() < 2 {
        return Err(R2Err { msg: "merge.xts(): need at least two xts objects".into(), kind: ErrKind::Runtime });
    }
    // Collect all xts inputs.
    let mut series: Vec<(Vec<Option<f64>>, usize, usize, Vec<f64>, String, Option<Vec<Arc<str>>>)> = Vec::new();
    for arg in a {
        let (xs, nrow, ncol, idx, cls, col_names) = get_xts(&arg.value)?;
        series.push((xs.iter().copied().collect(), nrow, ncol, idx, cls, col_names));
    }
    let idx_class = series[0].4.clone();
    for s in &series {
        if s.4 != idx_class {
            return Err(R2Err { msg: format!("merge.xts(): mixed index classes '{}' and '{}'", idx_class, s.4), kind: ErrKind::Runtime });
        }
    }
    // Union of all timestamps, sorted ascending.
    let mut all_idx: Vec<f64> = series.iter().flat_map(|s| s.3.iter().copied()).collect();
    all_idx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_idx.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let n = all_idx.len();

    let total_cols: usize = series.iter().map(|s| s.2).sum();
    let mut out: Vec<Option<f64>> = vec![None; n * total_cols];
    let mut all_names: Vec<Arc<str>> = Vec::with_capacity(total_cols);
    let mut col_off = 0;
    for (k, s) in series.iter().enumerate() {
        // For each row in s, find its position in all_idx (binary search).
        for r in 0..s.1 {
            let t = s.3[r];
            let pos = all_idx.partition_point(|&x| x < t - 1e-9);
            if pos < n && (all_idx[pos] - t).abs() < 1e-9 {
                for c in 0..s.2 {
                    out[(col_off + c) * n + pos] = s.0[c * s.1 + r];
                }
            }
        }
        // Column names: use provided or `x_k.col_j`.
        for c in 0..s.2 {
            let name = s.5.as_ref().and_then(|cn| cn.get(c)).cloned()
                .unwrap_or_else(|| Arc::from(format!("x{}.{}", k + 1, c + 1).as_str()));
            all_names.push(name);
        }
        col_off += s.2;
    }
    Ok(RVal::Numeric(out.into(), xts_attrs(n, total_cols, all_idx, &idx_class, Some(all_names))))
}

/// R-style print for xts: each row prefixed by its formatted timestamp.
pub fn format_xts(xs: &[Option<f64>], nrow: usize, ncol: usize, idx: &[f64], idx_class: &str, col_names: Option<&[Arc<str>]>) -> String {
    let mut out = String::new();
    if nrow == 0 {
        out.push_str("(empty xts object)\n");
        return out;
    }
    // Format every cell.
    let cell_strs: Vec<String> = xs.iter()
        .map(|x| match x { Some(v) => fmt_compact(*v), None => "NA".into() })
        .collect();
    // Format every timestamp.
    let ts_strs: Vec<String> = idx.iter().map(|&t| {
        if idx_class == "Date" { format_date(t, "%Y-%m-%d") } else { format_posixct(t, "%Y-%m-%d %H:%M:%S") }
    }).collect();

    // Compute widths.
    let ts_w = ts_strs.iter().map(|s| s.len()).max().unwrap_or(10);
    let mut col_w: Vec<usize> = (0..ncol).map(|c| {
        let h = col_names.and_then(|cn| cn.get(c)).map(|s| s.len()).unwrap_or(0);
        let v = (0..nrow).map(|r| cell_strs[c * nrow + r].len()).max().unwrap_or(1);
        h.max(v).max(4)
    }).collect();
    for w in col_w.iter_mut() { *w += 1; }

    // Header.
    out.push_str(&format!("{:>w$}", "", w = ts_w));
    for c in 0..ncol {
        let h = col_names.and_then(|cn| cn.get(c)).map(|s| s.to_string()).unwrap_or_else(|| format!("[,{}]", c + 1));
        out.push_str(&format!(" {:>w$}", h, w = col_w[c]));
    }
    out.push('\n');

    let max_rows = nrow.min(20);
    for r in 0..max_rows {
        out.push_str(&format!("{:>w$}", ts_strs[r], w = ts_w));
        for c in 0..ncol {
            out.push_str(&format!(" {:>w$}", cell_strs[c * nrow + r], w = col_w[c]));
        }
        out.push('\n');
    }
    if nrow > max_rows {
        out.push_str(&format!("... ({} more rows)\n", nrow - max_rows));
    }
    out
}

