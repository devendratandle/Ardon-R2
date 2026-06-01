//! Date / POSIXct support — Phase R.T.1.
//!
//! Storage mirrors R exactly:
//!
//! * `Date`    — `RVal::Numeric` of days since 1970-01-01, `class = "Date"`.
//! * `POSIXct` — `RVal::Numeric` of seconds since 1970-01-01 UTC,
//!               `class = "POSIXct"` (R uses `c("POSIXct","POSIXt")`; we
//!               store the leaf class only because `Attrs.class` is scalar).
//!
//! All arithmetic flows through the existing numeric machinery — `Date + n`
//! just adds `n` days because the underlying f64 already encodes days.
//! `Date - Date` returns a numeric `difftime` (we attach `class = "difftime"`
//! and the units in `Attrs.custom`).
//!
//! Civil ↔ day-count conversion uses Howard Hinnant's `days_from_civil`
//! algorithm (proleptic Gregorian, exact, no leap-second handling — same
//! semantics as R).

use r2_types::{RVal, Attrs, EvalArg, R2Err, ErrKind, Character};
use std::sync::Arc;

// ── Day ↔ (y,m,d) conversion (Hinnant) ────────────────────────────────

/// Days from civil date. Returns days since 1970-01-01.
/// `y` is full year (e.g. 2024), `m` is 1..=12, `d` is 1..=31.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u64;
    let m_u = m as i64;
    let doy = ((153 * (if m_u > 2 { m_u - 3 } else { m_u + 9 }) + 2) / 5 + d as i64 - 1) as u64;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

/// Inverse of `days_from_civil`. Returns (year, month 1..=12, day 1..=31).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ── strftime-style parsing & formatting (subset) ──────────────────────

/// Supported tokens: %Y %m %d %H %M %S %F (=%Y-%m-%d) %T (=%H:%M:%S).
/// %e and %k accept space-padded; everything else is literal.
pub fn parse_datetime(s: &str, fmt: &str) -> Option<(i64, u32, u32, u32, u32, u32)> {
    let bytes = s.as_bytes();
    let fbytes = fmt.as_bytes();
    let mut i = 0;
    let mut j = 0;
    let (mut y, mut mo, mut d, mut h, mut mi, mut se): (i64, u32, u32, u32, u32, u32)
        = (1970, 1, 1, 0, 0, 0);

    while j < fbytes.len() {
        if fbytes[j] == b'%' && j + 1 < fbytes.len() {
            // Expand %F and %T shortcuts inline.
            let token = fbytes[j + 1];
            let (start, end) = (i, bytes.len());
            match token {
                b'Y' => {
                    let (val, n) = read_int(&bytes[start..end], 4)?;
                    y = val as i64; i += n;
                }
                b'm' | b'd' | b'H' | b'M' | b'S' => {
                    let (val, n) = read_int(&bytes[start..end], 2)?;
                    match token {
                        b'm' => mo = val,
                        b'd' => d  = val,
                        b'H' => h  = val,
                        b'M' => mi = val,
                        b'S' => se = val,
                        _ => unreachable!(),
                    }
                    i += n;
                }
                b'F' => {
                    // %F == %Y-%m-%d
                    let sub = std::str::from_utf8(&bytes[start..end]).ok()?;
                    let (yy, mm, dd, _, _, _) = parse_datetime(sub, "%Y-%m-%d")?;
                    y = yy; mo = mm; d = dd;
                    i += 10; // YYYY-MM-DD
                }
                b'T' => {
                    let sub = std::str::from_utf8(&bytes[start..end]).ok()?;
                    let (_, _, _, hh, mm, ss) = parse_datetime(sub, "1970-01-01 %H:%M:%S")?;
                    h = hh; mi = mm; se = ss;
                    i += 8;
                }
                _ => return None,
            }
            j += 2;
        } else {
            if i >= bytes.len() || bytes[i] != fbytes[j] { return None; }
            i += 1; j += 1;
        }
    }
    Some((y, mo, d, h, mi, se))
}

fn read_int(b: &[u8], max: usize) -> Option<(u32, usize)> {
    let mut n = 0usize;
    let mut v: u32 = 0;
    while n < max && n < b.len() && b[n].is_ascii_digit() {
        v = v * 10 + (b[n] - b'0') as u32;
        n += 1;
    }
    if n == 0 { None } else { Some((v, n)) }
}

pub fn format_date(days: f64, fmt: &str) -> String {
    if days.is_nan() { return "NA".into(); }
    let (y, m, d) = civil_from_days(days.floor() as i64);
    format_civil(y, m, d, 0, 0, 0, fmt)
}

pub fn format_posixct(secs: f64, fmt: &str) -> String {
    if secs.is_nan() { return "NA".into(); }
    let total = secs.floor() as i64;
    let days = total.div_euclid(86_400);
    let rem  = total.rem_euclid(86_400) as u32;
    let (y, m, d) = civil_from_days(days);
    let h  = rem / 3600;
    let mi = (rem % 3600) / 60;
    let se = rem % 60;
    format_civil(y, m, d, h, mi, se, fmt)
}

fn format_civil(y: i64, m: u32, d: u32, h: u32, mi: u32, se: u32, fmt: &str) -> String {
    let mut out = String::with_capacity(fmt.len() + 4);
    let bytes = fmt.as_bytes();
    let mut j = 0;
    while j < bytes.len() {
        if bytes[j] == b'%' && j + 1 < bytes.len() {
            match bytes[j + 1] {
                b'Y' => out.push_str(&format!("{:04}", y)),
                b'm' => out.push_str(&format!("{:02}", m)),
                b'd' => out.push_str(&format!("{:02}", d)),
                b'H' => out.push_str(&format!("{:02}", h)),
                b'M' => out.push_str(&format!("{:02}", mi)),
                b'S' => out.push_str(&format!("{:02}", se)),
                b'F' => out.push_str(&format!("{:04}-{:02}-{:02}", y, m, d)),
                b'T' => out.push_str(&format!("{:02}:{:02}:{:02}", h, mi, se)),
                b'%' => out.push('%'),
                other => { out.push('%'); out.push(other as char); }
            }
            j += 2;
        } else {
            out.push(bytes[j] as char);
            j += 1;
        }
    }
    out
}

// ── Builtins ──────────────────────────────────────────────────────────

fn gv(a: &[EvalArg], i: usize) -> &RVal {
    static NIL: RVal = RVal::Null;
    a.get(i).map(|x| &x.value).unwrap_or(&NIL)
}

fn named<'a>(a: &'a [EvalArg], key: &str) -> Option<&'a RVal> {
    a.iter().find(|x| x.name.as_deref() == Some(key)).map(|x| &x.value)
}

fn as_str(v: &RVal) -> Option<String> {
    match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref().map(|s| s.to_string())),
        _ => None,
    }
}

/// `as.Date(x, format = "%Y-%m-%d")`
pub fn bi_as_date(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let fmt = named(a, "format").and_then(as_str).unwrap_or_else(|| "%Y-%m-%d".into());
    match gv(a, 0) {
        RVal::Character(v, _) => {
            let mut out = Vec::with_capacity(v.len());
            for s in v {
                match s.as_ref() {
                    None => out.push(None),
                    Some(s) => {
                        let parsed = parse_datetime(s, &fmt);
                        match parsed {
                            Some((y, m, d, _, _, _)) => out.push(Some(days_from_civil(y, m, d) as f64)),
                            None => return Err(R2Err {
                                msg: format!("character string '{}' is not in standard format '{}'", s, fmt),
                                kind: ErrKind::Runtime,
                            }),
                        }
                    }
                }
            }
            Ok(RVal::Numeric(out.into(), Attrs { class: Some(Arc::from("Date")), ..Default::default() }))
        }
        RVal::Numeric(v, _) => {
            // Treat as already days-since-epoch.
            Ok(RVal::Numeric(v.clone(), Attrs { class: Some(Arc::from("Date")), ..Default::default() }))
        }
        other => Err(R2Err {
            msg: format!("as.Date(): cannot coerce object of type '{}'", other.type_name()),
            kind: ErrKind::Type,
        }),
    }
}

/// `as.POSIXct(x, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")`
pub fn bi_as_posixct(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let fmt = named(a, "format").and_then(as_str).unwrap_or_else(|| "%Y-%m-%d %H:%M:%S".into());
    match gv(a, 0) {
        RVal::Character(v, _) => {
            let mut out = Vec::with_capacity(v.len());
            for s in v {
                match s.as_ref() {
                    None => out.push(None),
                    Some(s) => match parse_datetime(s, &fmt) {
                        Some((y, m, d, h, mi, se)) => {
                            let days = days_from_civil(y, m, d);
                            let secs = days * 86_400 + (h as i64) * 3600 + (mi as i64) * 60 + se as i64;
                            out.push(Some(secs as f64));
                        }
                        None => return Err(R2Err {
                            msg: format!("character string '{}' is not in standard format '{}'", s, fmt),
                            kind: ErrKind::Runtime,
                        }),
                    },
                }
            }
            Ok(RVal::Numeric(out.into(), Attrs { class: Some(Arc::from("POSIXct")), ..Default::default() }))
        }
        RVal::Numeric(v, _) => {
            Ok(RVal::Numeric(v.clone(), Attrs { class: Some(Arc::from("POSIXct")), ..Default::default() }))
        }
        other => Err(R2Err {
            msg: format!("as.POSIXct(): cannot coerce object of type '{}'", other.type_name()),
            kind: ErrKind::Type,
        }),
    }
}

/// `format(x, format = ...)` — dispatched by class. The engine routes Date /
/// POSIXct values here; everything else falls through to the default formatter.
pub fn bi_format_time(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let cls = match v {
        RVal::Numeric(_, attrs) => attrs.class.as_deref(),
        _ => None,
    };
    let fmt = named(a, "format").and_then(as_str);
    match (v, cls) {
        (RVal::Numeric(days, _), Some("Date")) => {
            let fmt = fmt.unwrap_or_else(|| "%Y-%m-%d".into());
            let out: Vec<Character> = days.iter()
                .map(|x| x.map(|d| Arc::from(format_date(d, &fmt).as_str())))
                .collect();
            Ok(RVal::Character(out, Attrs::default()))
        }
        (RVal::Numeric(secs, _), Some("POSIXct")) => {
            let fmt = fmt.unwrap_or_else(|| "%Y-%m-%d %H:%M:%S".into());
            let out: Vec<Character> = secs.iter()
                .map(|x| x.map(|s| Arc::from(format_posixct(s, &fmt).as_str())))
                .collect();
            Ok(RVal::Character(out, Attrs::default()))
        }
        _ => Err(R2Err {
            msg: "format(): object has no Date/POSIXct class".into(),
            kind: ErrKind::Type,
        }),
    }
}

pub fn bi_sys_date(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400) as f64;
    Ok(RVal::Numeric(vec![Some(days)].into(),
        Attrs { class: Some(Arc::from("Date")), ..Default::default() }))
}

pub fn bi_sys_time(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Ok(RVal::Numeric(vec![Some(secs)].into(),
        Attrs { class: Some(Arc::from("POSIXct")), ..Default::default() }))
}

/// `difftime(t1, t2, units = "days")` — returns t1 − t2 in the given units.
/// Accepts Date or POSIXct on either side; mixed inputs are coerced to seconds.
pub fn bi_difftime(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let units = named(a, "units").and_then(as_str).unwrap_or_else(|| "days".into());

    let v1 = to_seconds(gv(a, 0))?;
    let v2 = to_seconds(gv(a, 1))?;
    if v1.len() != v2.len() && v1.len() != 1 && v2.len() != 1 {
        return Err(R2Err { msg: format!("difftime(): length mismatch ({} vs {})", v1.len(), v2.len()), kind: ErrKind::Runtime });
    }
    let n = v1.len().max(v2.len());
    let divisor = match units.as_str() {
        "secs"    => 1.0,
        "mins"    => 60.0,
        "hours"   => 3600.0,
        "days"    => 86_400.0,
        "weeks"   => 7.0 * 86_400.0,
        other => return Err(R2Err {
            msg: format!("difftime(): unknown units '{}'. Use secs/mins/hours/days/weeks.", other),
            kind: ErrKind::Runtime,
        }),
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = v1[i.min(v1.len() - 1)];
        let b = v2[i.min(v2.len() - 1)];
        out.push(match (a, b) {
            (Some(x), Some(y)) => Some((x - y) / divisor),
            _ => None,
        });
    }
    let mut attrs = Attrs::default();
    attrs.class = Some(Arc::from("difftime"));
    attrs.custom.insert(Arc::from("units"), RVal::Character(
        vec![Some(Arc::from(units.as_str()))], Attrs::default()));
    Ok(RVal::Numeric(out.into(), attrs))
}

fn to_seconds(v: &RVal) -> Result<Vec<Option<f64>>, R2Err> {
    match v {
        RVal::Numeric(xs, attrs) => {
            let scale = match attrs.class.as_deref() {
                Some("Date")    => 86_400.0,
                Some("POSIXct") => 1.0,
                Some(other) => return Err(R2Err {
                    msg: format!("difftime(): unsupported class '{}'", other),
                    kind: ErrKind::Type,
                }),
                None => 86_400.0, // treat bare numerics as days
            };
            Ok(xs.iter().map(|x| x.map(|y| y * scale)).collect())
        }
        other => Err(R2Err {
            msg: format!("difftime(): cannot coerce '{}'", other.type_name()),
            kind: ErrKind::Type,
        }),
    }
}

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

fn ts_attrs(start: f64, end: f64, freq: f64) -> Attrs {
    let mut a = Attrs::default();
    a.class = Some(Arc::from("ts"));
    a.custom.insert(Arc::from("tsp"), RVal::Numeric(
        vec![Some(start), Some(end), Some(freq)].into(), Attrs::default()));
    a
}

fn get_tsp(v: &RVal) -> Result<(f64, f64, f64), R2Err> {
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

fn fmt_compact(x: f64) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        let s = format!("{:.4}", x);
        // Trim trailing zeros but keep at least one decimal.
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed.contains('.') { trimmed.to_string() } else { format!("{}.0", trimmed) }
    }
}

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

fn xts_attrs(nrow: usize, ncol: usize, index: Vec<f64>, index_class: &str, col_names: Option<Vec<Arc<str>>>) -> Attrs {
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

fn get_xts(v: &RVal) -> Result<(&[Option<f64>], usize, usize, Vec<f64>, String, Option<Vec<Arc<str>>>), R2Err> {
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

// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.4 — TS analytics (acf, pacf, decompose, lag, diff, etc.)
//
// Statistical conventions match base R / `stats::acf`:
//   * c(k) = (1/N) * Σ_{t=1..N-k} (x_t − x̄)(x_{t+k} − x̄)   — sample autocovariance
//   * r(k) = c(k) / c(0)                                    — autocorrelation
//   * default lag.max = floor(10 * log10(N))
//
// PACF uses Durbin–Levinson recursion. decompose() uses classical
// seasonal decomposition: centered MA → detrend → average within season
// → recenter seasonal to mean zero → random = x − trend − seasonal
// (additive) or x / (trend·seasonal) (multiplicative).
// ═══════════════════════════════════════════════════════════════════════

fn extract_numeric(v: &RVal) -> Result<Vec<f64>, R2Err> {
    match v {
        RVal::Numeric(xs, _) => Ok(xs.iter().map(|x| x.unwrap_or(f64::NAN)).collect()),
        RVal::Integer(xs, _) => Ok(xs.iter().map(|x| x.map(|n| n as f64).unwrap_or(f64::NAN)).collect()),
        other => Err(R2Err { msg: format!("need numeric vector, got '{}'", other.type_name()), kind: ErrKind::Type }),
    }
}

/// Sample autocovariances c(0..=lag_max).
fn autocov(x: &[f64], lag_max: usize) -> Vec<f64> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let centered: Vec<f64> = x.iter().map(|v| v - mean).collect();
    (0..=lag_max).map(|k| {
        let mut s = 0.0;
        for t in 0..(x.len() - k) { s += centered[t] * centered[t + k]; }
        s / n
    }).collect()
}

/// `acf(x, lag.max = NULL, type = "correlation", plot = FALSE)`
pub fn bi_acf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = extract_numeric(gv(a, 0))?;
    let n = x.len();
    if n < 2 { return Err(R2Err { msg: "acf(): need at least 2 observations".into(), kind: ErrKind::Runtime }); }
    let default_max = ((10.0 * (n as f64).log10()).floor() as usize).min(n - 1);
    let lag_max = named(a, "lag.max")
        .and_then(|v| extract_numeric(v).ok().and_then(|xs| xs.first().copied()))
        .map(|x| (x as usize).min(n - 1))
        .unwrap_or(default_max);
    let kind = named(a, "type").and_then(as_str).unwrap_or_else(|| "correlation".into());

    let cov = autocov(&x, lag_max);
    let out: Vec<Option<f64>> = match kind.as_str() {
        "covariance" => cov.iter().map(|&v| Some(v)).collect(),
        "correlation" => cov.iter().map(|&v| Some(v / cov[0])).collect(),
        other => return Err(R2Err { msg: format!("acf(): unknown type '{}'. Use 'correlation' or 'covariance'.", other), kind: ErrKind::Runtime }),
    };
    let lags: Vec<Option<f64>> = (0..=lag_max).map(|k| Some(k as f64)).collect();
    Ok(RVal::List(vec![
        (Some(Arc::from("acf")),    RVal::Numeric(out.into(),  Attrs::default())),
        (Some(Arc::from("lag")),    RVal::Numeric(lags.into(), Attrs::default())),
        (Some(Arc::from("n.used")), RVal::Numeric(vec![Some(n as f64)].into(), Attrs::default())),
        (Some(Arc::from("type")),   RVal::Character(vec![Some(Arc::from(kind.as_str()))], Attrs::default())),
    ]))
}

/// `pacf(x, lag.max = NULL)` via Durbin–Levinson recursion.
pub fn bi_pacf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = extract_numeric(gv(a, 0))?;
    let n = x.len();
    if n < 2 { return Err(R2Err { msg: "pacf(): need at least 2 observations".into(), kind: ErrKind::Runtime }); }
    let default_max = ((10.0 * (n as f64).log10()).floor() as usize).min(n - 1);
    let lag_max = named(a, "lag.max")
        .and_then(|v| extract_numeric(v).ok().and_then(|xs| xs.first().copied()))
        .map(|x| (x as usize).min(n - 1))
        .unwrap_or(default_max).max(1);

    let cov = autocov(&x, lag_max);
    let r: Vec<f64> = cov.iter().map(|c| c / cov[0]).collect();

    // Durbin–Levinson: phi[k,k] is the PACF at lag k.
    let mut phi: Vec<Vec<f64>> = vec![vec![0.0; lag_max + 1]; lag_max + 1];
    let mut v = vec![0.0; lag_max + 1];
    v[0] = 1.0; // normalized
    phi[1][1] = r[1];
    v[1] = v[0] * (1.0 - phi[1][1].powi(2));
    let mut pacf_vals: Vec<f64> = vec![phi[1][1]];

    for k in 2..=lag_max {
        let mut num = r[k];
        for j in 1..k {
            num -= phi[k-1][j] * r[k-j];
        }
        let pkk = if v[k-1].abs() < 1e-12 { 0.0 } else { num / v[k-1] };
        phi[k][k] = pkk;
        for j in 1..k {
            phi[k][j] = phi[k-1][j] - pkk * phi[k-1][k-j];
        }
        v[k] = v[k-1] * (1.0 - pkk.powi(2));
        pacf_vals.push(pkk);
    }
    let pacf_opt: Vec<Option<f64>> = pacf_vals.into_iter().map(Some).collect();
    let lags: Vec<Option<f64>> = (1..=lag_max).map(|k| Some(k as f64)).collect();
    Ok(RVal::List(vec![
        (Some(Arc::from("acf")),    RVal::Numeric(pacf_opt.into(), Attrs::default())),
        (Some(Arc::from("lag")),    RVal::Numeric(lags.into(), Attrs::default())),
        (Some(Arc::from("n.used")), RVal::Numeric(vec![Some(n as f64)].into(), Attrs::default())),
        (Some(Arc::from("type")),   RVal::Character(vec![Some(Arc::from("partial"))], Attrs::default())),
    ]))
}

/// Centered moving average of period `f`.  For even `f` (e.g. 12 monthly),
/// uses a 2x12 MA: average of two adjacent 12-MAs.
fn centered_ma(x: &[f64], f: usize) -> Vec<Option<f64>> {
    let n = x.len();
    let mut out = vec![None; n];
    if f < 2 || f > n { return out; }
    if f % 2 == 1 {
        let half = f / 2;
        for i in half..n - half {
            let s: f64 = x[i - half..=i + half].iter().sum();
            out[i] = Some(s / f as f64);
        }
    } else {
        let half = f / 2;
        for i in half..n - half {
            // 2x f MA: mean of MA(i-1) and MA(i), each of width f, where MA(i)
            // uses x[i-half+1..=i+half], length f.
            let s1: f64 = x[i - half..i + half].iter().sum();
            let s2: f64 = x[i - half + 1..=i + half].iter().sum();
            out[i] = Some(0.5 * (s1 + s2) / f as f64);
        }
    }
    out
}

/// `decompose(x, type = "additive")` — classical seasonal decomposition.
/// Requires `x` to be a ts object with frequency > 1.
pub fn bi_decompose(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (_start, _end, freq) = get_tsp(v)?;
    let f = freq.round() as usize;
    if f < 2 {
        return Err(R2Err { msg: "decompose(): time series has no or 1 periods per cycle (frequency must be > 1)".into(), kind: ErrKind::Runtime });
    }
    let x: Vec<f64> = match v {
        RVal::Numeric(xs, _) => xs.iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
        _ => return Err(R2Err { msg: "decompose(): need ts object".into(), kind: ErrKind::Type }),
    };
    let n = x.len();
    if n < 2 * f {
        return Err(R2Err { msg: format!("decompose(): need at least {} observations (got {})", 2 * f, n), kind: ErrKind::Runtime });
    }
    let kind = named(a, "type").and_then(as_str).unwrap_or_else(|| "additive".into());
    let multiplicative = kind == "multiplicative";

    let trend = centered_ma(&x, f);

    // Detrend.
    let detrend: Vec<Option<f64>> = (0..n).map(|i| match (trend[i], x[i]) {
        (Some(t), v) if !v.is_nan() => Some(if multiplicative { v / t } else { v - t }),
        _ => None,
    }).collect();

    // Average detrended values within each season position.
    let mut figure = vec![0.0; f];
    let mut counts = vec![0usize; f];
    for (i, d) in detrend.iter().enumerate() {
        if let Some(val) = d {
            let pos = i % f;
            figure[pos] += val;
            counts[pos] += 1;
        }
    }
    for p in 0..f {
        if counts[p] > 0 { figure[p] /= counts[p] as f64; }
    }
    // Recenter the seasonal figure so it sums to zero (additive) or
    // averages to 1 (multiplicative).
    if multiplicative {
        let mean_fig: f64 = figure.iter().sum::<f64>() / f as f64;
        for v in figure.iter_mut() { *v /= mean_fig; }
    } else {
        let mean_fig: f64 = figure.iter().sum::<f64>() / f as f64;
        for v in figure.iter_mut() { *v -= mean_fig; }
    }
    let seasonal: Vec<Option<f64>> = (0..n).map(|i| Some(figure[i % f])).collect();

    // Random = x − trend − seasonal (additive) or x / (trend·seasonal) (mult).
    let random: Vec<Option<f64>> = (0..n).map(|i| match (trend[i], seasonal[i]) {
        (Some(t), Some(s)) => Some(if multiplicative { x[i] / (t * s) } else { x[i] - t - s }),
        _ => None,
    }).collect();

    let attrs_ts = |start: f64, end: f64, freq: f64| ts_attrs(start, end, freq);
    let (s, e, fr) = get_tsp(v)?;
    let to_ts = |xs: Vec<Option<f64>>| RVal::Numeric(xs.into(), attrs_ts(s, e, fr));

    let observed = match v { RVal::Numeric(xs, _) => RVal::Numeric(xs.clone(), attrs_ts(s, e, fr)), _ => unreachable!() };

    Ok(RVal::List(vec![
        (Some(Arc::from("x")),        observed),
        (Some(Arc::from("seasonal")), to_ts(seasonal)),
        (Some(Arc::from("trend")),    to_ts(trend)),
        (Some(Arc::from("random")),   to_ts(random)),
        (Some(Arc::from("figure")),   RVal::Numeric(figure.into_iter().map(Some).collect::<Vec<_>>().into(), Attrs::default())),
        (Some(Arc::from("type")),     RVal::Character(vec![Some(Arc::from(kind.as_str()))], Attrs::default())),
    ]))
}

/// `is.regular(x)` — TRUE for ts objects (always regular) and xts where
/// all consecutive gaps are equal within 1e-9.
pub fn bi_is_regular(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let regular = match v {
        RVal::Numeric(_, attrs) => match attrs.class.as_deref() {
            Some("ts") => true,
            Some("xts") => {
                if let Some(RVal::Numeric(idx, _)) = attrs.custom.get(&Arc::from("index")) {
                    if idx.len() < 2 { true } else {
                        let gaps: Vec<f64> = idx.windows(2)
                            .filter_map(|w| match (w[0], w[1]) { (Some(a), Some(b)) => Some(b - a), _ => None })
                            .collect();
                        if gaps.is_empty() { false } else {
                            let g0 = gaps[0];
                            gaps.iter().all(|g| (g - g0).abs() < 1e-9)
                        }
                    }
                } else { false }
            }
            _ => false,
        },
        _ => false,
    };
    Ok(RVal::Logical(vec![Some(regular)].into(), Attrs::default()))
}

/// `periodicity(x)` — classify the median gap of an xts index.
pub fn bi_periodicity(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (_, _, _, idx, idx_class, _) = get_xts(v)?;
    if idx.len() < 2 {
        return Err(R2Err { msg: "periodicity(): need at least 2 observations".into(), kind: ErrKind::Runtime });
    }
    let mut gaps: Vec<f64> = idx.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = gaps[gaps.len() / 2];
    let median_secs = if idx_class == "Date" { median * 86_400.0 } else { median };
    let label = match median_secs {
        x if x < 60.0          => "seconds",
        x if x < 3600.0        => "minutes",
        x if x < 86_400.0      => "hours",
        x if x < 7.0 * 86_400.0 => "daily",
        x if x < 30.0 * 86_400.0 => "weekly",
        x if x < 92.0 * 86_400.0 => "monthly",
        x if x < 366.0 * 86_400.0 => "quarterly",
        _                      => "yearly",
    };
    Ok(RVal::List(vec![
        (Some(Arc::from("scale")),   RVal::Character(vec![Some(Arc::from(label))], Attrs::default())),
        (Some(Arc::from("frequency")), RVal::Numeric(vec![Some(median)].into(), Attrs::default())),
        (Some(Arc::from("units")),   RVal::Character(vec![Some(Arc::from(if idx_class == "Date" { "days" } else { "secs" }))], Attrs::default())),
    ]))
}

/// `lag(x, k = 1)` — for a ts object, shift the time origin backwards by
/// k periods (R's lag.ts behavior). For a plain numeric vector, prepend
/// k NAs (i.e. behave like dplyr::lag).
pub fn bi_lag(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let k: i32 = named(a, "k").or_else(|| Some(gv(a, 1)))
        .and_then(|v| match v {
            RVal::Numeric(vs, _) => vs.first().and_then(|x| *x).map(|x| x as i32),
            RVal::Integer(vs, _) => vs.first().and_then(|x| *x),
            _ => None,
        }).unwrap_or(1);

    if let RVal::Numeric(xs, attrs) = v {
        if attrs.class.as_deref() == Some("ts") {
            let (s, e, f) = get_tsp(v)?;
            let dt = 1.0 / f;
            return Ok(RVal::Numeric(xs.clone(), ts_attrs(s - k as f64 * dt, e - k as f64 * dt, f)));
        }
    }
    // Plain vector: shift by prepending NAs (k > 0) or appending (k < 0).
    let xs = extract_numeric(v)?;
    let n = xs.len();
    let mut out: Vec<Option<f64>> = vec![None; n];
    if k >= 0 {
        let k = (k as usize).min(n);
        for i in k..n { out[i] = Some(xs[i - k]); }
    } else {
        let k = ((-k) as usize).min(n);
        for i in 0..n - k { out[i] = Some(xs[i + k]); }
    }
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}

/// `diff_ts(x, lag = 1, differences = 1)` — differences of a numeric vector
/// or ts object. Named `diff_ts` to avoid clashing with the existing diff().
pub fn bi_diff_ts(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let lag: usize = named(a, "lag").and_then(|v| extract_numeric(v).ok().and_then(|xs| xs.first().copied()))
        .map(|x| x as usize).unwrap_or(1).max(1);
    let differences: usize = named(a, "differences").and_then(|v| extract_numeric(v).ok().and_then(|xs| xs.first().copied()))
        .map(|x| x as usize).unwrap_or(1).max(1);

    let is_ts = matches!(v, RVal::Numeric(_, attrs) if attrs.class.as_deref() == Some("ts"));
    let mut xs = extract_numeric(v)?;
    for _ in 0..differences {
        if xs.len() <= lag { return Err(R2Err { msg: "diff_ts(): not enough observations".into(), kind: ErrKind::Runtime }); }
        let mut next = Vec::with_capacity(xs.len() - lag);
        for i in lag..xs.len() { next.push(xs[i] - xs[i - lag]); }
        xs = next;
    }
    let out: Vec<Option<f64>> = xs.iter().map(|&x| if x.is_nan() { None } else { Some(x) }).collect();
    if is_ts {
        let (s, e, f) = get_tsp(v)?;
        let total_dropped = lag * differences;
        let dt = 1.0 / f;
        return Ok(RVal::Numeric(out.into(), ts_attrs(s + total_dropped as f64 * dt, e, f)));
    }
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}

// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.5 — period aggregation (aggregate.ts, apply.*, to.*)
//
// Splits an xts/ts into period-length chunks (week/month/quarter/year)
// and applies a function (mean/sum/last/etc.) per chunk. Period
// boundaries follow R's xts: weeks end Sunday, months end last day of
// month, quarters end Mar/Jun/Sep/Dec, years end Dec 31.
// ═══════════════════════════════════════════════════════════════════════

/// Map a Date (days-since-epoch) or POSIXct (secs-since-epoch) value to
/// the bucket key for a given period. Two values in the same bucket
/// share the same key; we just use the END date of the bucket as the
/// canonical key (matches R's xts convention).
fn period_key(t: f64, idx_class: &str, period: &str) -> f64 {
    let days = if idx_class == "Date" {
        t.floor() as i64
    } else {
        (t / 86_400.0).floor() as i64
    };
    let (y, m, d) = civil_from_days(days);
    let key_days = match period {
        "daily"     => days,
        "weekly"    => {
            // Sunday-ending week: 1970-01-04 was a Sunday → days % 7 == 3 is Sun.
            let off = (days + 4).rem_euclid(7); // days since most recent Sat
            days + (6 - off)                    // next Sat? actually let's use Sun.
        }
        "monthly"   => {
            let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
            days_from_civil(ny, nm, 1) - 1
        }
        "quarterly" => {
            let q_end_m = ((m - 1) / 3 + 1) * 3;
            let (ny, nm) = if q_end_m == 12 { (y + 1, 1) } else { (y, q_end_m + 1) };
            days_from_civil(ny, nm, 1) - 1
        }
        "yearly"    => days_from_civil(y + 1, 1, 1) - 1,
        _           => days,
    };
    let _ = d; // not used directly
    if idx_class == "Date" { key_days as f64 } else { (key_days as f64) * 86_400.0 + 86_399.0 }
}

fn apply_fn(values: &[f64], fname: &str) -> f64 {
    match fname {
        "mean"  => values.iter().sum::<f64>() / values.len() as f64,
        "sum"   => values.iter().sum(),
        "first" => values[0],
        "last"  => *values.last().unwrap(),
        "min"   => values.iter().cloned().fold(f64::INFINITY, f64::min),
        "max"   => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        _       => f64::NAN,
    }
}

/// Common engine: collect rows into buckets, apply FUN, return a new xts.
fn aggregate_by_period(v: &RVal, period: &str, fname: &str) -> Result<RVal, R2Err> {
    let (xs, nrow, ncol, idx, idx_class, col_names) = get_xts(v)?;
    if nrow == 0 {
        return Ok(RVal::Numeric(Vec::<Option<f64>>::new().into(),
            xts_attrs(0, ncol, vec![], &idx_class, col_names)));
    }
    let keys: Vec<f64> = idx.iter().map(|&t| period_key(t, &idx_class, period)).collect();
    let mut bucket_keys: Vec<f64> = keys.clone();
    bucket_keys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bucket_keys.dedup_by(|a, b| (*a - *b).abs() < 0.5);

    let mut out: Vec<Option<f64>> = Vec::with_capacity(bucket_keys.len() * ncol);
    for c in 0..ncol {
        for &k in &bucket_keys {
            let vals: Vec<f64> = (0..nrow)
                .filter(|&r| (keys[r] - k).abs() < 0.5)
                .filter_map(|r| xs[c * nrow + r])
                .collect();
            out.push(if vals.is_empty() { None } else { Some(apply_fn(&vals, fname)) });
        }
    }
    let new_n = bucket_keys.len();
    Ok(RVal::Numeric(out.into(), xts_attrs(new_n, ncol, bucket_keys, &idx_class, col_names)))
}

fn read_fun(a: &[EvalArg]) -> String {
    named(a, "FUN").and_then(|v| match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref().map(|s| s.to_string())),
        RVal::BuiltinFn(n) => Some(n.to_string()),
        _ => None,
    }).unwrap_or_else(|| "mean".into())
}

pub fn bi_to_daily(a: &[EvalArg])     -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "daily",     &read_fun(a)) }
pub fn bi_to_weekly(a: &[EvalArg])    -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "weekly",    &read_fun(a)) }
pub fn bi_to_monthly(a: &[EvalArg])   -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "monthly",   &read_fun(a)) }
pub fn bi_to_quarterly(a: &[EvalArg]) -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "quarterly", &read_fun(a)) }
pub fn bi_to_yearly(a: &[EvalArg])    -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "yearly",    &read_fun(a)) }
pub fn bi_apply_daily(a: &[EvalArg])    -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "daily",     &read_fun(a)) }
pub fn bi_apply_weekly(a: &[EvalArg])   -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "weekly",    &read_fun(a)) }
pub fn bi_apply_monthly(a: &[EvalArg])  -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "monthly",   &read_fun(a)) }
pub fn bi_apply_quarterly(a: &[EvalArg])-> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "quarterly", &read_fun(a)) }
pub fn bi_apply_yearly(a: &[EvalArg])   -> Result<RVal, R2Err> { aggregate_by_period(gv(a, 0), "yearly",    &read_fun(a)) }

// Hindu calendar (tithi/hnc.date/saka-era) moved to time/hindu.rs.
mod hindu;
pub use hindu::*;


// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_day_zero() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn known_dates_round_trip() {
        for &(y, m, d) in &[(2000, 1, 1), (2024, 2, 29), (1999, 12, 31), (1900, 3, 1)] {
            let z = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(z), (y, m, d), "round-trip failed for {}-{}-{}", y, m, d);
        }
    }

    #[test]
    fn parse_iso_date() {
        let p = parse_datetime("2024-03-15", "%Y-%m-%d").unwrap();
        assert_eq!((p.0, p.1, p.2), (2024, 3, 15));
    }

    #[test]
    fn format_round_trip() {
        let days = days_from_civil(2024, 3, 15) as f64;
        assert_eq!(format_date(days, "%Y-%m-%d"), "2024-03-15");
        assert_eq!(format_date(days, "%d/%m/%Y"), "15/03/2024");
    }

    #[test]
    fn ts_start_end_inference() {
        // ts(1:24, start=c(1960,1), frequency=12) → ends at Dec 1961
        let data = RVal::Numeric((1..=24).map(|i| Some(i as f64)).collect::<Vec<_>>().into(), Attrs::default());
        let args = vec![
            EvalArg { name: None, value: data },
            EvalArg { name: Some(Arc::from("start")), value: RVal::Numeric(vec![Some(1960.0), Some(1.0)].into(), Attrs::default()) },
            EvalArg { name: Some(Arc::from("frequency")), value: RVal::Numeric(vec![Some(12.0)].into(), Attrs::default()) },
        ];
        let v = bi_ts(&args).unwrap();
        let (s, e, f) = get_tsp(&v).unwrap();
        assert!((s - 1960.0).abs() < 1e-9);
        assert!((f - 12.0).abs() < 1e-9);
        // End = 1960 + 23/12 ≈ 1961.9166...
        assert!((e - (1960.0 + 23.0/12.0)).abs() < 1e-9);
    }

    #[test]
    fn ts_window_extracts_subrange() {
        let data = RVal::Numeric((1..=24).map(|i| Some(i as f64)).collect::<Vec<_>>().into(), Attrs::default());
        let v = bi_ts(&vec![
            EvalArg { name: None, value: data },
            EvalArg { name: Some(Arc::from("start")), value: RVal::Numeric(vec![Some(1960.0), Some(1.0)].into(), Attrs::default()) },
            EvalArg { name: Some(Arc::from("frequency")), value: RVal::Numeric(vec![Some(12.0)].into(), Attrs::default()) },
        ]).unwrap();
        // window(x, start=c(1960,6), end=c(1960,12)) → 7 obs (Jun..Dec 1960)
        let w = bi_window(&vec![
            EvalArg { name: None, value: v },
            EvalArg { name: Some(Arc::from("start")), value: RVal::Numeric(vec![Some(1960.0), Some(6.0)].into(), Attrs::default()) },
            EvalArg { name: Some(Arc::from("end")),   value: RVal::Numeric(vec![Some(1960.0), Some(12.0)].into(), Attrs::default()) },
        ]).unwrap();
        if let RVal::Numeric(xs, _) = &w {
            assert_eq!(xs.len(), 7);
            assert_eq!(xs[0], Some(6.0));
            assert_eq!(xs[6], Some(12.0));
        } else { panic!("window did not return numeric"); }
    }

    #[test]
    fn posixct_seconds_round_trip() {
        // 2024-03-15 12:34:56 UTC
        let days = days_from_civil(2024, 3, 15);
        let secs = days * 86400 + 12 * 3600 + 34 * 60 + 56;
        assert_eq!(format_posixct(secs as f64, "%Y-%m-%d %H:%M:%S"), "2024-03-15 12:34:56");
    }
}
