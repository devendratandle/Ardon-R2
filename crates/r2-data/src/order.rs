//! `sort`, `order`, `rank`, `unique`, `duplicated`.
//!
//! Vector-shaped utilities for sorting and de-duplication. The orderings
//! and equality they use live in `r2_types::sort`, shared with factor
//! levels and the grouping functions.

use r2_types::{duplicated_mask, elem_keys, order_indices, take, Attrs, ElemKey, ErrKind, EvalArg, Logical, NaLast, R2Err, RVal, Real, SortCol};

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

fn flag(a: &[EvalArg], name: &str) -> bool {
    arg_named(a, name).and_then(|v| v.as_logicals().ok())
        .map_or(false, |v| v.first().copied().flatten() == Some(true))
}

/// `na.last = TRUE / FALSE / NA` (`default` when not given).
fn na_last(a: &[EvalArg], default: NaLast) -> NaLast {
    match arg_named(a, "na.last").and_then(|v| v.as_logicals().ok()).and_then(|v| v.first().copied()) {
        Some(Some(true)) => NaLast::Last,
        Some(Some(false)) => NaLast::First,
        Some(None) => NaLast::Remove,
        None => default,
    }
}

fn type_err(what: &str, v: &RVal) -> R2Err {
    R2Err { msg: format!("{}() not supported for {}", what, v.type_name()), kind: ErrKind::Type }
}

fn keys_of(what: &str, v: &RVal) -> Result<Vec<ElemKey>, R2Err> {
    elem_keys(v).ok_or_else(|| type_err(what, v))
}

pub fn bi_duplicated(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let keys = keys_of("duplicated", &first_arg(a))?;
    Ok(RVal::Logical(duplicated_mask(&keys).into_iter().map(Some).collect::<Vec<Logical>>().into(), Attrs::default()))
}

/// `unique(x)`: first occurrences, in order, keeping the type (and a
/// factor's levels). NA is kept once, like any other value.
pub fn bi_unique(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let keys = keys_of("unique", &x)?;
    let keep: Vec<usize> = duplicated_mask(&keys).iter().enumerate().filter(|(_, d)| !**d).map(|(i, _)| i).collect();
    Ok(match take(&x, &keep) {
        // R's unique drops names.
        RVal::Numeric(v, _) => RVal::Numeric(v, Attrs::default()),
        RVal::Integer(v, _) => RVal::Integer(v, Attrs::default()),
        RVal::Logical(v, _) => RVal::Logical(v, Attrs::default()),
        RVal::Character(v, _) => RVal::Character(v, Attrs::default()),
        other => other,
    })
}

/// `sort(x, decreasing = FALSE, na.last = NA)`: type-preserving; NA and
/// NaN are dropped unless `na.last` says where to put them.
pub fn bi_sort(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let col = SortCol::new(&x).ok_or_else(|| type_err("sort", &x))?;
    let idx = order_indices(&[col], flag(a, "decreasing"), na_last(a, NaLast::Remove));
    Ok(take(&x, &idx))
}

/// `order(..., decreasing = FALSE, na.last = TRUE)`: every unnamed argument
/// is a key, later keys break ties; the sort is stable.
pub fn bi_order(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let vals: Vec<RVal> = a.iter().filter(|x| x.name.is_none()).map(|x| x.value.clone()).collect();
    let keys = vals.iter().map(|v| SortCol::new(v).ok_or_else(|| type_err("order", v)))
        .collect::<Result<Vec<_>, _>>()?;
    if keys.windows(2).any(|w| w[0].len() != w[1].len()) {
        return Err(R2Err { msg: "order(): argument lengths differ".into(), kind: ErrKind::Runtime });
    }
    let idx = order_indices(&keys, flag(a, "decreasing"), na_last(a, NaLast::Last));
    Ok(RVal::Integer(idx.iter().map(|i| Some((*i + 1) as i32)).collect::<Vec<_>>().into(), Attrs::default()))
}

/// `rank(x, na.last = TRUE, ties.method = "average")`; also "first",
/// "min" and "max". NAs take the last ranks in order of appearance
/// (`na.last = "keep"` leaves them NA).
pub fn bi_rank(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first_arg(a);
    let col = SortCol::new(&x).ok_or_else(|| type_err("rank", &x))?;
    let method = match arg_named(a, "ties.method") {
        Some(RVal::Character(v, _)) => v.first().cloned().flatten().map(|s| s.to_string()).unwrap_or_default(),
        _ => "average".into(),
    };
    let keep_na = matches!(arg_named(a, "na.last"), Some(RVal::Character(v, _)) if v.first().cloned().flatten().as_deref() == Some("keep"));
    let idx = order_indices(std::slice::from_ref(&col), false, NaLast::Last);
    let mut ranks: Vec<Real> = vec![None; col.len()];
    let mut start = 0;
    while start < idx.len() {
        // A run of tied (non-missing) elements shares positions start..end.
        let mut end = start + 1;
        if !col.is_na(idx[start]) {
            while end < idx.len() && !col.is_na(idx[end]) && col.cmp(idx[start], idx[end]).is_eq() { end += 1; }
        }
        for (k, &i) in idx[start..end].iter().enumerate() {
            let r = match method.as_str() {
                "first" => (start + k + 1) as f64,
                "min" => (start + 1) as f64,
                "max" => end as f64,
                _ => (start + 1 + end) as f64 / 2.0,
            };
            ranks[i] = if keep_na && col.is_na(i) { None } else { Some(r) };
        }
        start = end;
    }
    if matches!(method.as_str(), "first" | "min" | "max") {
        return Ok(RVal::Integer(ranks.iter().map(|r| r.map(|v| v as i32)).collect::<Vec<_>>().into(), Attrs::default()));
    }
    Ok(RVal::Numeric(ranks.into(), Attrs::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn duplicated_marks_repeats() {
        let r = bi_duplicated(&[evarg(nums(&[1.0, 2.0, 1.0, 3.0, 2.0]))]).unwrap();
        match r {
            RVal::Logical(v, _) => {
                let got: Vec<bool> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![false, false, true, false, true]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn unique_keeps_first_occurrences() {
        let r = bi_unique(&[evarg(nums(&[1.0, 2.0, 1.0, 3.0, 2.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn order_returns_sort_indices_1based() {
        let r = bi_order(&[evarg(nums(&[3.0, 1.0, 2.0]))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![2, 3, 1]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rank_assigns_ascending_ranks() {
        let r = bi_rank(&[evarg(nums(&[30.0, 10.0, 20.0]))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![3.0, 1.0, 2.0]);
            }
            _ => panic!(),
        }
    }
}
