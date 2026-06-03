//! NA handling + merge. Phase R.7.
//!
//! `na.omit`, `complete.cases`, `merge`. The `merge` function is a
//! single-shared-column inner-join with a `.y` suffix for collisions
//! on non-key columns. R's full `merge()` semantics (multi-key joins,
//! all.x/all.y, by.x/by.y) tracked as a v0.2.0 item in KNOWN_LIMITATIONS.

use r2_types::{Attrs, Character, DataFrame, ErrKind, EvalArg, Logical, R2Err, RVal, Real};
use std::sync::Arc;

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth_arg(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

fn val_to_str(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => String::new(),
    }
}

fn to_string_vec(col: &RVal) -> Vec<String> {
    match col {
        RVal::Numeric(v, _) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Integer(v, _) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Character(v, _) => v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect(),
        RVal::Logical(v, _) => v.iter().map(|x| match x {
            Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into()
        }).collect(),
        _ => Vec::new(),
    }
}

fn filter_col_by_mask(col: &RVal, keep: &[bool]) -> RVal {
    match col {
        RVal::Numeric(v, _) => RVal::Numeric(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Integer(v, _) => RVal::Integer(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Character(v, _) => RVal::Character(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(x.clone()) } else { None }).collect(),
            Attrs::default(),
        ),
        RVal::Logical(v, _) => RVal::Logical(
            v.iter().zip(keep).filter_map(|(x, k)| if *k { Some(*x) } else { None }).collect(),
            Attrs::default(),
        ),
        _ => col.clone(),
    }
}

pub fn bi_na_omit(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::Numeric(v, _) => Ok(RVal::Numeric(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::Integer(v, _) => Ok(RVal::Integer(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::Character(v, _) => Ok(RVal::Character(
            v.iter().filter(|x| x.is_some()).cloned().collect(),
            Attrs::default(),
        )),
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let keep: Vec<bool> = (0..nrow).map(|r| {
                df.columns.iter().all(|(_, col)| match col {
                    RVal::Numeric(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Integer(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Character(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    _ => true,
                })
            }).collect();
            let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
                (name.clone(), filter_col_by_mask(col, &keep))
            }).collect();
            let removed = keep.iter().filter(|x| !**x).count();
            if removed > 0 { soutln!("Removed {} rows with NA values", removed); }
            Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
        }
        _ => Ok(first_arg(a)),
    }
}

pub fn bi_complete_cases(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let result: Vec<Logical> = (0..nrow).map(|r| {
                Some(df.columns.iter().all(|(_, col)| match col {
                    RVal::Numeric(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Integer(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    RVal::Character(v, _) => v.get(r).map(|x| x.is_some()).unwrap_or(false),
                    _ => true,
                }))
            }).collect();
            Ok(RVal::Logical(result.into(), Attrs::default()))
        }
        _ => Err(R2Err { msg: "complete.cases needs data.frame".into(), kind: ErrKind::Runtime }),
    }
}

pub fn bi_merge(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df1 = match &first_arg(a) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "merge needs data.frame".into(), kind: ErrKind::Type }),
    };
    let df2 = match &nth_arg(a, 1) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "merge needs data.frame".into(), kind: ErrKind::Type }),
    };
    let by_col = arg_named(a, "by").map(|v| val_to_str(&v)).unwrap_or_else(|| {
        for (n1, _) in &df1.columns {
            for (n2, _) in &df2.columns {
                if n1 == n2 { return n1.to_string(); }
            }
        }
        String::new()
    });
    if by_col.is_empty() {
        return Err(R2Err { msg: "merge: no common column found, specify by=".into(), kind: ErrKind::Runtime });
    }
    let key1 = df1.get_col(&by_col).ok_or_else(|| R2Err {
        msg: format!("'{}' not in first df", by_col), kind: ErrKind::Runtime,
    })?;
    let key2 = df2.get_col(&by_col).ok_or_else(|| R2Err {
        msg: format!("'{}' not in second df", by_col), kind: ErrKind::Runtime,
    })?;
    let k1 = to_string_vec(key1);
    let k2 = to_string_vec(key2);

    let mut match_pairs: Vec<(usize, usize)> = Vec::new();
    for (i, v1) in k1.iter().enumerate() {
        for (j, v2) in k2.iter().enumerate() {
            if v1 == v2 { match_pairs.push((i, j)); }
        }
    }

    let mut columns: Vec<(Arc<str>, Vec<String>)> = Vec::new();
    columns.push((Arc::from(by_col.as_str()), match_pairs.iter().map(|(i, _)| k1[*i].clone()).collect()));
    for (name, col) in &df1.columns {
        if name.as_ref() == by_col { continue; }
        let sv = to_string_vec(col);
        columns.push((name.clone(), match_pairs.iter().map(|(i, _)| sv[*i].clone()).collect()));
    }
    for (name, col) in &df2.columns {
        if name.as_ref() == by_col { continue; }
        let sv = to_string_vec(col);
        let final_name = if df1.columns.iter().any(|(n, _)| n == name) {
            Arc::from(format!("{}.y", name).as_str())
        } else {
            name.clone()
        };
        columns.push((final_name, match_pairs.iter().map(|(_, j)| sv[*j].clone()).collect()));
    }

    let typed_cols: Vec<(Arc<str>, RVal)> = columns.into_iter().map(|(name, vals)| {
        let all_num = vals.iter().all(|s| s.is_empty() || s == "NA" || s.parse::<f64>().is_ok());
        if all_num && !vals.is_empty() {
            let nums: Vec<Real> = vals.iter().map(|s| if s == "NA" { None } else { s.parse().ok() }).collect();
            (name, RVal::Numeric(nums.into(), Attrs::default()))
        } else {
            let strs: Vec<Character> = vals.iter()
                .map(|s| if s == "NA" { None } else { Some(Arc::from(s.as_str())) })
                .collect();
            (name, RVal::Character(strs, Attrs::default()))
        }
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns: typed_cols, row_names: None }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn na_omit_drops_na_from_numeric() {
        let v = RVal::Numeric(vec![Some(1.0), None, Some(2.0), None].into(), Attrs::default());
        let r = bi_na_omit(&[evarg(v)]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert_eq!(v.len(), 2),
            _ => panic!(),
        }
    }

    #[test]
    fn complete_cases_marks_clean_rows() {
        let df = DataFrame {
            columns: vec![
                (Arc::from("x"), RVal::Numeric(vec![Some(1.0), None, Some(3.0)].into(), Attrs::default())),
                (Arc::from("y"), RVal::Numeric(vec![Some(2.0), Some(5.0), None].into(), Attrs::default())).into(),
            ],
            row_names: None,
        };
        let r = bi_complete_cases(&[evarg(RVal::DataFrame(df))]).unwrap();
        match r {
            RVal::Logical(v, _) => {
                let got: Vec<bool> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![true, false, false]);
            }
            _ => panic!(),
        }
    }
}
