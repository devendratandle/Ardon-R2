//! What R's operators carry besides the values: names and `dim` from the
//! operands, and the arithmetic of dates, date-times and time differences
//! (`Date + 1` is a date, `Date - Date` a `difftime` in days). The values
//! themselves come from `ops.rs`.

use std::sync::Arc;

use r2_types::*;

use crate::{err, Engine};

impl Engine {
    pub(crate) fn binary_op(&mut self, op: BinOp, lhs: &RVal, rhs: &RVal) -> Result<RVal, R2Err> {
        if let Some(r) = time_arith(self, op, lhs, rhs)? { return Ok(r); }
        let out = self.binary_values(op, lhs, rhs)?;
        Ok(carry_attrs(op, lhs, rhs, out))
    }

    pub(crate) fn unary_op(&self, op: UnOp, v: &RVal) -> Result<RVal, R2Err> {
        if let (UnOp::Neg, RVal::Matrix(m)) = (op, v) {
            let mut out = m.clone();
            for x in &mut out.data { *x = -*x; }
            return Ok(RVal::Matrix(out));
        }
        let out = self.unary_values(op, v)?;
        // `-x` keeps every attribute, `!x` its names and dim.
        let keep = match v {
            RVal::Matrix(m) => Attrs { dim: Some(vec![m.nrow, m.ncol]), ..Default::default() },
            RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) => match op {
                UnOp::Neg => at.clone(),
                _ => Attrs { names: at.names.clone(), dim: at.dim.clone(), ..Default::default() },
            },
            _ => return Ok(out),
        };
        Ok(with_attrs(out, keep))
    }
}

fn attrs_of(v: &RVal) -> Option<&Attrs> {
    match v {
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => Some(at),
        _ => None,
    }
}

fn is_plain(at: &Attrs) -> bool {
    at.names.is_none() && at.dim.is_none() && at.class.is_none() && at.custom.is_empty()
}

/// R takes the result's attributes from the operands as long as the result,
/// the first operand's winning. A comparison or a logical operator keeps only
/// names and dim; arithmetic keeps everything, a class included.
fn carry_attrs(op: BinOp, lhs: &RVal, rhs: &RVal, out: RVal) -> RVal {
    if matches!(out, RVal::Matrix(_)) { return out; }
    let n = rval_length(&out);
    let structural = matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
        | BinOp::And | BinOp::Or);
    let mut keep = Attrs::default();
    for side in [rhs, lhs] {
        if rval_length(side) != n { continue; }
        match side {
            RVal::Matrix(m) => keep.dim = Some(vec![m.nrow, m.ncol]),
            other => if let Some(at) = attrs_of(other) {
                if is_plain(at) { continue; }
                if at.names.is_some() { keep.names = at.names.clone(); }
                if at.dim.is_some() { keep.dim = at.dim.clone(); }
                if !structural {
                    if at.class.is_some() { keep.set_classes(at.classes()); }
                    for (k, v) in &at.custom { keep.custom.insert(k.clone(), v.clone()); }
                }
            },
        }
    }
    if is_plain(&keep) { return out; }
    with_attrs(out, keep)
}

/// `out` with `keep`'s attributes. A number with two dimensions and nothing
/// else is a matrix, so it becomes one.
fn with_attrs(out: RVal, keep: Attrs) -> RVal {
    match out {
        RVal::Numeric(v, _) => {
            if let (Some(&[r, c]), None, true) = (keep.dim.as_deref(), &keep.class, keep.custom.is_empty()) {
                if r * c == v.len() && keep.names.is_none() {
                    return RVal::Matrix(Matrix::new(v.iter().map(|x| x.unwrap_or(f64::NAN)).collect(), r, c));
                }
            }
            RVal::Numeric(v, keep)
        }
        RVal::Integer(v, _) => RVal::Integer(v, keep),
        RVal::Logical(v, _) => RVal::Logical(v, keep),
        RVal::Character(v, _) => RVal::Character(v, keep),
        other => other,
    }
}

// ── dates, date-times and time differences ─────────────────────────────

#[derive(Clone, PartialEq)]
enum Time { Date, Posix, Diff(f64) }

/// Seconds in each of `difftime`'s units.
pub(crate) fn unit_secs(u: &str) -> Option<f64> {
    Some(match u { "secs" => 1.0, "mins" => 60.0, "hours" => 3600.0, "days" => 86400.0, "weeks" => 604800.0, _ => return None })
}

fn time_of(v: &RVal) -> Option<Time> {
    let at = attrs_of(v)?;
    match at.class.as_deref()? {
        "Date" => Some(Time::Date),
        "POSIXct" => Some(Time::Posix),
        "difftime" => {
            let u = match at.custom.get("units") { Some(RVal::Character(u, _)) => u.first().cloned().flatten(), _ => None };
            Some(Time::Diff(u.as_deref().and_then(unit_secs).unwrap_or(1.0)))
        }
        _ => None,
    }
}

fn op_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/", BinOp::Pow => "^",
        BinOp::Mod => "%%", BinOp::IntDiv => "%/%", _ => "?",
    }
}

/// A `difftime` of `secs` seconds each, in R's automatic units: the smallest
/// difference decides between seconds, minutes, hours and days.
pub(crate) fn difftime_auto(secs: Vec<Option<f64>>) -> RVal {
    let least = secs.iter().flatten().filter(|x| x.is_finite()).map(|x| x.abs()).fold(f64::INFINITY, f64::min);
    let units = if !least.is_finite() || least < 60.0 { "secs" } else if least < 3600.0 { "mins" } else if least < 86400.0 { "hours" } else { "days" };
    difftime_in(secs, units)
}

pub(crate) fn difftime_in(secs: Vec<Option<f64>>, units: &str) -> RVal {
    let per = unit_secs(units).unwrap_or(1.0);
    let mut at = Attrs { class: Some(Arc::from("difftime")), ..Default::default() };
    at.custom.insert(Arc::from("units"), RVal::Character(vec![Some(Arc::from(units))], Attrs::default()));
    RVal::Numeric(secs.into_iter().map(|x| x.map(|s| s / per)).collect(), at)
}

fn recycle(e: &Engine, l: &RVal, r: &RVal, f: impl Fn(f64, f64) -> f64) -> Result<Vec<Option<f64>>, R2Err> {
    let (a, b) = (e.as_reals(l)?, e.as_reals(r)?);
    if a.is_empty() || b.is_empty() { return Ok(Vec::new()); }
    let n = a.len().max(b.len());
    Ok((0..n).map(|i| match (a[i % a.len()], b[i % b.len()]) { (Some(x), Some(y)) => Some(f(x, y)), _ => None }).collect())
}

/// The result keeps the first operand's attributes when it is the time value
/// (`d + 1` keeps the date's names and time zone).
fn timed(values: Vec<Option<f64>>, like: &RVal) -> RVal {
    let at = attrs_of(like).cloned().unwrap_or_default();
    RVal::Numeric(values.into(), at)
}

fn time_arith(e: &Engine, op: BinOp, l: &RVal, r: &RVal) -> Result<Option<RVal>, R2Err> {
    let (tl, tr) = (time_of(l), time_of(r));
    if tl.is_none() && tr.is_none() { return Ok(None); }
    if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
        | BinOp::And | BinOp::Or | BinOp::AndShort | BinOp::OrShort | BinOp::Colon | BinOp::Tilde | BinOp::MatMul) {
        return Ok(None);
    }
    use Time::*;
    let add = |x: f64, y: f64| x + y;
    let sub = |x: f64, y: f64| x - y;
    Ok(Some(match (op, &tl, &tr) {
        // a date moves by days, a date-time by seconds; a difftime converts
        (BinOp::Add, Some(Date), None) => timed(recycle(e, l, r, add)?, l),
        (BinOp::Add, None, Some(Date)) => timed(recycle(e, l, r, add)?, r),
        (BinOp::Sub, Some(Date), None) => timed(recycle(e, l, r, sub)?, l),
        (BinOp::Add | BinOp::Sub, Some(Date), Some(Diff(s))) => {
            let days = s / 86400.0;
            timed(recycle(e, l, r, |x, y| if op == BinOp::Add { x + y * days } else { x - y * days })?, l)
        }
        (BinOp::Add, Some(Diff(s)), Some(Date)) => { let days = s / 86400.0; timed(recycle(e, l, r, |x, y| x * days + y)?, r) }
        (BinOp::Sub, Some(Date), Some(Date)) => difftime_in(recycle(e, l, r, |x, y| (x - y) * 86400.0)?, "days"),
        (BinOp::Add, Some(Posix), None) => timed(recycle(e, l, r, add)?, l),
        (BinOp::Add, None, Some(Posix)) => timed(recycle(e, l, r, add)?, r),
        (BinOp::Sub, Some(Posix), None) => timed(recycle(e, l, r, sub)?, l),
        (BinOp::Add | BinOp::Sub, Some(Posix), Some(Diff(s))) => {
            let s = *s;
            timed(recycle(e, l, r, |x, y| if op == BinOp::Add { x + y * s } else { x - y * s })?, l)
        }
        (BinOp::Add, Some(Diff(s)), Some(Posix)) => { let s = *s; timed(recycle(e, l, r, |x, y| x * s + y)?, r) }
        (BinOp::Sub, Some(Posix), Some(Posix)) => difftime_auto(recycle(e, l, r, sub)?),
        // a time difference keeps its units through + - * / by numbers
        (BinOp::Add | BinOp::Sub, Some(Diff(_)), Some(Diff(s2))) => {
            let s2 = *s2;
            let per = if let Some(Diff(s1)) = &tl { *s1 } else { 1.0 };
            timed(recycle(e, l, r, |x, y| if op == BinOp::Add { x + y * s2 / per } else { x - y * s2 / per })?, l)
        }
        (BinOp::Div, Some(Diff(s1)), Some(Diff(s2))) => {
            let ratio = s1 / s2;
            RVal::Numeric(recycle(e, l, r, |x, y| x / y * ratio)?.into(), Attrs::default())
        }
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Some(Diff(_)), None) => {
            timed(recycle(e, l, r, |x, y| match op { BinOp::Add => x + y, BinOp::Sub => x - y, BinOp::Mul => x * y, _ => x / y })?, l)
        }
        (BinOp::Add | BinOp::Mul, None, Some(Diff(_))) => {
            timed(recycle(e, l, r, |x, y| if op == BinOp::Add { x + y } else { x * y })?, r)
        }
        (_, Some(Date), _) | (_, _, Some(Date)) => return err!(Runtime, "{} not defined for \"Date\" objects", op_name(op)),
        (_, Some(Posix), _) | (_, _, Some(Posix)) => return err!(Runtime, "'{}' not defined for \"POSIXt\" objects", op_name(op)),
        _ => return err!(Runtime, "'{}' not defined for \"difftime\" objects", op_name(op)),
    }))
}

/// How a date, a date-time or a time difference prints (installed into
/// r2-types' printing by `Engine::new`). A date-time shows its clock only when
/// some element is not at midnight, and always its time zone.
pub(crate) fn time_text(v: &RVal) -> Option<ClassText> {
    let RVal::Numeric(xs, at) = v else { return None };
    match at.class.as_deref()? {
        "Date" => Some(ClassText::Elements(xs.iter().map(|x| match x {
            Some(d) => format!("\"{}\"", r2_stats::time::format_date(*d, "%Y-%m-%d")),
            None => "NA".into(),
        }).collect())),
        "POSIXct" => {
            let tz = match at.custom.get("tzone") { Some(RVal::Character(t, _)) => t.first().cloned().flatten(), _ => None };
            let tz = tz.as_deref().filter(|t| !t.is_empty()).unwrap_or("UTC").to_string();
            let clock = xs.iter().flatten().any(|s| s.rem_euclid(86400.0) != 0.0);
            let fmt = if clock { "%Y-%m-%d %H:%M:%S" } else { "%Y-%m-%d" };
            Some(ClassText::Elements(xs.iter().map(|x| match x {
                Some(s) => format!("\"{} {}\"", r2_stats::time::format_posixct(*s, fmt), tz),
                None => "NA".into(),
            }).collect()))
        }
        "difftime" => {
            let units = match at.custom.get("units") { Some(RVal::Character(u, _)) => u.first().cloned().flatten(), _ => None };
            let units = units.as_deref().unwrap_or("days").to_string();
            let plain = RVal::Numeric(xs.clone(), Attrs { names: at.names.clone(), ..Default::default() });
            Some(ClassText::Text(if xs.len() == 1 && at.names.is_none() {
                format!("Time difference of {} {}", format_reals(xs, print_digits(), 0)[0].trim(), units)
            } else {
                format!("Time differences in {}\n{}", units, plain)
            }))
        }
        _ => None,
    }
}

/// `out` with `test`'s names and dim, as `ifelse` gives its result.
pub(crate) fn shaped_like(test: &RVal, out: RVal) -> RVal {
    if rval_length(test) != rval_length(&out) { return out; }
    let keep = match test {
        RVal::Matrix(m) => Attrs { dim: Some(vec![m.nrow, m.ncol]), ..Default::default() },
        other => match attrs_of(other) {
            Some(at) => Attrs { names: at.names.clone(), dim: at.dim.clone(), ..Default::default() },
            None => return out,
        },
    };
    if is_plain(&keep) { return out; }
    with_attrs(out, keep)
}

/// The arguments with a matrix-shaped first one (a logical matrix, say) made
/// a `Matrix`, for the functions that take only that.
pub(crate) fn matrix_first(a: &[EvalArg]) -> Vec<EvalArg> {
    let mut out = a.to_vec();
    if let Some(first) = out.first_mut() {
        if let Some(at) = attrs_of(&first.value) {
            if let Some(&[r, c]) = at.dim.as_deref() {
                if let Ok(v) = first.value.as_reals() {
                    if v.len() == r * c {
                        first.value = RVal::Matrix(Matrix::new(v.iter().map(|x| x.unwrap_or(f64::NAN)).collect(), r, c));
                    }
                }
            }
        }
    }
    out
}

/// `seq(from, to, by, length.out)` from a date. `by` is a number of days or
/// "day", "week", "month", "quarter", "year", optionally with a count
/// ("2 weeks"); months and years step the calendar, keeping the day.
pub(crate) fn seq_dates(e: &Engine, a: &[EvalArg]) -> Result<RVal, R2Err> {
    let arg = |name: &str, pos: usize| a.iter().find(|x| x.name.as_deref() == Some(name)).map(|x| x.value.clone())
        .or_else(|| a.iter().filter(|x| x.name.is_none()).nth(pos).map(|x| x.value.clone()));
    let from = match arg("from", 0) { Some(v) => e.scalar_f64(&v)?.unwrap_or(f64::NAN), None => f64::NAN };
    let to = match arg("to", 1) { Some(v) => e.scalar_f64(&v)?, None => None };
    let length = a.iter().find(|x| matches!(x.name.as_deref(), Some("length.out" | "length"))).map(|x| e.scalar_f64(&x.value)).transpose()?.flatten();
    let (count, unit) = match arg("by", 2) {
        Some(RVal::Character(s, _)) => {
            let text = s.first().cloned().flatten().unwrap_or_else(|| Arc::from("day"));
            let mut parts = text.split_whitespace().rev();
            let unit = parts.next().unwrap_or("day").trim_end_matches('s').to_string();
            let count: f64 = parts.next().and_then(|n| n.parse().ok()).unwrap_or(1.0);
            (count, unit)
        }
        Some(v) => (e.scalar_f64(&v)?.unwrap_or(1.0), "day".to_string()),
        None => match (to, length) {
            (Some(t), Some(n)) if n > 1.0 => ((t - from) / (n - 1.0), "day".to_string()),
            _ => (1.0, "day".to_string()),
        },
    };
    let step = |k: f64| -> f64 {
        match unit.as_str() {
            "week" => from + 7.0 * count * k,
            "month" | "quarter" | "year" => {
                let months = count * k * match unit.as_str() { "quarter" => 3.0, "year" => 12.0, _ => 1.0 };
                add_months(from, months as i64)
            }
            _ => from + count * k,
        }
    };
    if !matches!(unit.as_str(), "day" | "week" | "month" | "quarter" | "year") {
        return err!(Runtime, "invalid string for 'by'");
    }
    let mut out = Vec::new();
    match (to, length) {
        (_, Some(n)) => for k in 0..n.max(0.0) as usize { out.push(Some(step(k as f64))); },
        (Some(t), None) => {
            if count == 0.0 { return err!(Runtime, "invalid '(to - from)/by' in seq(.)"); }
            let mut k = 0.0;
            loop {
                let d = step(k);
                if (count > 0.0 && d > t + 1e-9) || (count < 0.0 && d < t - 1e-9) || k > 1e7 { break; }
                out.push(Some(d));
                k += 1.0;
            }
        }
        (None, None) => return err!(Runtime, "exactly two of 'to', 'by' and 'length.out' / 'along.with' must be specified"),
    }
    Ok(RVal::Numeric(out.into(), Attrs { class: Some(Arc::from("Date")), ..Default::default() }))
}

/// `days` (since 1970-01-01) moved by whole calendar months. A day past the
/// end of the new month overflows into the next, as R's seq.Date does.
fn add_months(days: f64, months: i64) -> f64 {
    let (y, m, d) = r2_stats::time::civil_from_days(days.floor() as i64);
    let total = y * 12 + (m as i64 - 1) + months;
    let (ny, nm) = (total.div_euclid(12), (total.rem_euclid(12) + 1) as u32);
    (r2_stats::time::days_from_civil(ny, nm, 1) + d as i64 - 1) as f64 + (days - days.floor())
}
