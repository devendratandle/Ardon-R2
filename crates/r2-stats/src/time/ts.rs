//! Classic `ts()` regular time series — Phase R.T.2.

use super::*;
use r2_types::{RVal, Attrs, EvalArg, R2Err, ErrKind};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.2 — ts() regular time series
//
// Storage: numeric vector with `class = "ts"` and a custom attr
// `tsp = c(start, end, frequency)`, exactly matching R. Time values are
// encoded as `year + (period-1)/frequency`, so:
//
//   * monthly:   1960.0     = Jan 1960, 1960.0833... = Feb 1960
//   * quarterly: 1960.0     = Q1   1960, 1960.25     = Q2   1960
//   * annual:    1960.0     = 1960
//
// `start` and `end` are either a single number or `c(year, period)`.
// ═══════════════════════════════════════════════════════════════════════

/// Convert R's "compact" start spec into a single numeric time value.
/// `c(1960, 3)` with freq=12 → 1960 + 2/12 = 1960.1666...
/// Single number → returned as-is.
fn spec_to_time(spec: &RVal, freq: f64) -> Result<f64, R2Err> {
    match spec {
        RVal::Numeric(v, _) => {
            let xs: Vec<f64> = v.iter().filter_map(|x| *x).collect();
            match xs.len() {
                1 => Ok(xs[0]),
                2 => Ok(xs[0] + (xs[1] - 1.0) / freq),
                _ => Err(R2Err { msg: "ts() start/end must be a single number or c(year, period)".into(), kind: ErrKind::Runtime }),
            }
        }
        RVal::Integer(v, _) => {
            let xs: Vec<f64> = v.iter().filter_map(|x| x.map(|n| n as f64)).collect();
            match xs.len() {
                1 => Ok(xs[0]),
                2 => Ok(xs[0] + (xs[1] - 1.0) / freq),
                _ => Err(R2Err { msg: "ts() start/end must be a single number or c(year, period)".into(), kind: ErrKind::Runtime }),
            }
        }
        _ => Err(R2Err { msg: "ts() start/end must be numeric".into(), kind: ErrKind::Type }),
    }
}

pub fn ts_attrs(start: f64, end: f64, freq: f64) -> Attrs {
    let mut a = Attrs::default();
    a.class = Some(Arc::from("ts"));
    a.custom.insert(Arc::from("tsp"), RVal::Numeric(
        vec![Some(start), Some(end), Some(freq)].into(), Attrs::default()));
    a
}

pub fn get_tsp(v: &RVal) -> Result<(f64, f64, f64), R2Err> {
    let attrs = match v {
        RVal::Numeric(_, a) => a,
        _ => return Err(R2Err { msg: "not a ts object".into(), kind: ErrKind::Type }),
    };
    let tsp = attrs.custom.get(&Arc::from("tsp"))
        .ok_or_else(|| R2Err { msg: "not a ts object (no tsp attr)".into(), kind: ErrKind::Type })?;
    if let RVal::Numeric(xs, _) = tsp {
        if xs.len() == 3 {
            return Ok((xs[0].unwrap_or(f64::NAN), xs[1].unwrap_or(f64::NAN), xs[2].unwrap_or(f64::NAN)));
        }
    }
    Err(R2Err { msg: "ts object has malformed tsp".into(), kind: ErrKind::Type })
}

/// `ts(data, start = 1, end = NULL, frequency = 1, deltat = NULL)`
pub fn bi_ts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let data = gv(a, 0);
    let xs: Vec<Option<f64>> = match data {
        RVal::Numeric(v, _) => v.iter().copied().collect(),
        RVal::Integer(v, _) => v.iter().map(|x| x.map(|n| n as f64)).collect(),
        RVal::Logical(v, _) => v.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect(),
        other => return Err(R2Err { msg: format!("ts(): need numeric data, got '{}'", other.type_name()), kind: ErrKind::Type }),
    };
    let n = xs.len();
    if n == 0 {
        return Err(R2Err { msg: "ts(): data must be non-empty".into(), kind: ErrKind::Runtime });
    }

    // frequency / deltat
    let freq = match named(a, "frequency") {
        Some(RVal::Numeric(v, _)) => v.first().and_then(|x| *x).unwrap_or(1.0),
        Some(RVal::Integer(v, _)) => v.first().and_then(|x| x.map(|n| n as f64)).unwrap_or(1.0),
        _ => match named(a, "deltat") {
            Some(RVal::Numeric(v, _)) => 1.0 / v.first().and_then(|x| *x).unwrap_or(1.0),
            _ => 1.0,
        },
    };
    if freq <= 0.0 || !freq.is_finite() {
        return Err(R2Err { msg: format!("ts(): frequency must be a positive finite number (got {})", freq), kind: ErrKind::Runtime });
    }

    // start
    let start = match named(a, "start") {
        Some(s) => spec_to_time(s, freq)?,
        None => 1.0,
    };

    // end: either supplied, or inferred from n and start.
    let end = match named(a, "end") {
        Some(e) => spec_to_time(e, freq)?,
        None => start + (n as f64 - 1.0) / freq,
    };

    Ok(RVal::Numeric(xs.into(), ts_attrs(start, end, freq)))
}

pub fn bi_tsp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (s, e, f) = get_tsp(gv(a, 0))?;
    Ok(RVal::Numeric(vec![Some(s), Some(e), Some(f)].into(), Attrs::default()))
}

pub fn bi_start(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (s, _, f) = get_tsp(gv(a, 0))?;
    Ok(time_pair(s, f))
}
pub fn bi_end(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (_, e, f) = get_tsp(gv(a, 0))?;
    Ok(time_pair(e, f))
}
pub fn bi_frequency(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (_, _, f) = get_tsp(gv(a, 0))?;
    Ok(RVal::Numeric(vec![Some(f)].into(), Attrs::default()))
}
pub fn bi_deltat(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (_, _, f) = get_tsp(gv(a, 0))?;
    Ok(RVal::Numeric(vec![Some(1.0 / f)].into(), Attrs::default()))
}

/// Return c(year, period). Period is 1-based and wraps at frequency.
fn time_pair(t: f64, freq: f64) -> RVal {
    if freq == 1.0 {
        return RVal::Numeric(vec![Some(t.floor()), Some(1.0)].into(), Attrs::default());
    }
    let year = t.floor();
    let frac = (t - year) * freq;
    let period = (frac.round() as i64 + 1) as f64; // 0-indexed → 1-indexed
    RVal::Numeric(vec![Some(year), Some(period)].into(), Attrs::default())
}

/// `time(x)` — numeric vector of time points for each observation.
pub fn bi_time(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (start, _end, freq) = get_tsp(v)?;
    let n = match v { RVal::Numeric(xs, _) => xs.len(), _ => 0 };
    let dt = 1.0 / freq;
    let out: Vec<Option<f64>> = (0..n).map(|i| Some(start + i as f64 * dt)).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}

/// `cycle(x)` — 1-based period within the cycle (Jan=1..Dec=12 for monthly).
pub fn bi_cycle(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (start, _end, freq) = get_tsp(v)?;
    let n = match v { RVal::Numeric(xs, _) => xs.len(), _ => 0 };
    let f = freq.round() as i64;
    if f <= 0 {
        return Err(R2Err { msg: "cycle(): frequency must be a positive integer".into(), kind: ErrKind::Runtime });
    }
    // Start period index (0-based).
    let start_period = ((start - start.floor()) * freq).round() as i64;
    let out: Vec<Option<i32>> = (0..n)
        .map(|i| Some((((start_period + i as i64) % f) + 1) as i32))
        .collect();
    Ok(RVal::Integer(out.into(), Attrs::default()))
}

/// `window(x, start = NULL, end = NULL)` — extract a contiguous sub-series.
pub fn bi_window(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (ts_start, ts_end, freq) = get_tsp(v)?;
    let n = match v { RVal::Numeric(xs, _) => xs.len(), _ => return Err(R2Err { msg: "window(): need ts object".into(), kind: ErrKind::Type }) };

    let new_start = match named(a, "start") {
        Some(s) => spec_to_time(s, freq)?,
        None => ts_start,
    };
    let new_end = match named(a, "end") {
        Some(e) => spec_to_time(e, freq)?,
        None => ts_end,
    };
    if new_start < ts_start - 1e-9 || new_end > ts_end + 1e-9 || new_start > new_end {
        return Err(R2Err {
            msg: format!("window(): [{}, {}] is outside the series range [{}, {}]", new_start, new_end, ts_start, ts_end),
            kind: ErrKind::Runtime,
        });
    }
    let dt = 1.0 / freq;
    let i0 = ((new_start - ts_start) / dt).round() as usize;
    let i1 = ((new_end   - ts_start) / dt).round() as usize;
    if i1 >= n {
        return Err(R2Err { msg: "window(): index out of bounds".into(), kind: ErrKind::Runtime });
    }
    let xs = if let RVal::Numeric(xs, _) = v { xs } else { unreachable!() };
    let slice: Vec<Option<f64>> = xs[i0..=i1].iter().copied().collect();
    let actual_start = ts_start + i0 as f64 * dt;
    let actual_end   = ts_start + i1 as f64 * dt;
    Ok(RVal::Numeric(slice.into(), ts_attrs(actual_start, actual_end, freq)))
}

pub fn bi_is_ts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let is = matches!(gv(a, 0), RVal::Numeric(_, attrs) if attrs.class.as_deref() == Some("ts"));
    Ok(RVal::Logical(vec![Some(is)].into(), Attrs::default()))
}

/// Format a ts object the way R's print.ts does — a labeled matrix when
/// frequency is monthly/quarterly, otherwise a simple time-tagged vector.
pub fn format_ts(xs: &[Option<f64>], start: f64, _end: f64, freq: f64) -> String {
    let f_int = freq.round() as usize;
    let n = xs.len();
    let mut out = String::new();
    let start_year = start.floor() as i64;
    let start_period = ((start - start.floor()) * freq).round() as usize; // 0-based

    if f_int == 12 || f_int == 4 {
        // Labeled matrix form.
        let headers: Vec<String> = if f_int == 12 {
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                .iter().map(|s| s.to_string()).collect()
        } else {
            (1..=4).map(|i| format!("Qtr{}", i)).collect()
        };

        let strs: Vec<String> = xs.iter()
            .map(|x| match x { Some(v) => fmt_compact(*v), None => "NA".into() })
            .collect();
        let cell_w = strs.iter().map(|s| s.len()).max().unwrap_or(1)
            .max(headers.iter().map(|s| s.len()).max().unwrap_or(1));

        // Header row.
        out.push_str(&format!("{:>5}", ""));
        for h in &headers { out.push_str(&format!(" {:>w$}", h, w = cell_w)); }
        out.push('\n');

        // Rows.
        let mut idx = 0;
        let mut year = start_year;
        let mut col = start_period;
        // Leading row may be partial.
        loop {
            if idx >= n { break; }
            out.push_str(&format!("{:>5}", year));
            for c in 0..f_int {
                if c < col || idx >= n {
                    out.push_str(&format!(" {:>w$}", "", w = cell_w));
                } else {
                    out.push_str(&format!(" {:>w$}", strs[idx], w = cell_w));
                    idx += 1;
                }
            }
            out.push('\n');
            col = 0;
            year += 1;
        }
    } else {
        // Simple form: just print values prefixed by [1], like a vector.
        out.push_str("[1] ");
        for (i, x) in xs.iter().enumerate() {
            if i > 0 { out.push(' '); }
            out.push_str(&match x { Some(v) => fmt_compact(*v), None => "NA".into() });
        }
        out.push('\n');
    }
    out
}

