//! TS analytics (acf/pacf/decompose/lag/diff) + period aggregation
//! (aggregate.ts / apply.* / to.*) — Phase R.T.4 & R.T.5.

use super::*;
use r2_types::{RVal, Attrs, EvalArg, R2Err, ErrKind};
use std::sync::Arc;

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
