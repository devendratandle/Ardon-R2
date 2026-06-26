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


// Hindu calendar (tithi/hnc.date/saka-era) moved to time/hindu.rs.
mod hindu;
pub use hindu::*;

// Time-series families split out by domain. mod.rs keeps the Date/POSIXct
// core + the shared calendar/arg primitives (days_from_civil, parse_datetime,
// gv/named/as_str, …) that these child modules reach via `use super::*`.
mod ts;
mod xts;
mod tsanalysis;
pub use ts::*;
pub use xts::*;
pub use tsanalysis::*;


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
