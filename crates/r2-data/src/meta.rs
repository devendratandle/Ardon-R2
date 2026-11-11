//! DataFrame meta + slicing. Phase R.7.
//!
//! `nrow`, `ncol`, `dim`, `colnames`, `rownames`, `is.data.frame`,
//! `as.data.frame`, `head`, `tail`. Pure functions over `RVal`.

use r2_types::{Attrs, DataFrame, ErrKind, EvalArg, Matrix, R2Err, RVal, Real};
use std::sync::Arc;

#[inline]
fn first_arg(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn nth_arg(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

#[inline]
fn rint(n: i32) -> RVal {
    RVal::Integer(vec![Some(n)].into(), Attrs::default())
}

#[inline]
fn rbool(b: bool) -> RVal {
    RVal::Logical(vec![Some(b)].into(), Attrs::default())
}

pub fn bi_nrow(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => Ok(rint(df.nrow() as i32)),
        RVal::Matrix(m) => Ok(rint(m.nrow as i32)),
        _ => Ok(RVal::Null),
    }
}

pub fn bi_ncol(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => Ok(rint(df.ncol() as i32)),
        RVal::Matrix(m) => Ok(rint(m.ncol as i32)),
        _ => Ok(RVal::Null),
    }
}

pub fn bi_dim(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => Ok(RVal::Integer(
            vec![Some(df.nrow() as i32), Some(df.ncol() as i32)].into(),
            Attrs::default().into(),
        )),
        RVal::Matrix(m) => Ok(RVal::Integer(
            vec![Some(m.nrow as i32), Some(m.ncol as i32)].into(),
            Attrs::default().into(),
        )),
        RVal::Numeric(_, at) | RVal::Integer(_, at) | RVal::Logical(_, at) | RVal::Character(_, at) => Ok(match &at.dim {
            Some(d) => RVal::Integer(d.iter().map(|&k| Some(k as i32)).collect(), Attrs::default()),
            None => RVal::Null,
        }),
        _ => Ok(RVal::Null),
    }
}

pub fn bi_colnames(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => Ok(RVal::Character(
            df.columns.iter().map(|(n, _)| Some(n.clone())).collect(),
            Attrs::default(),
        )),
        RVal::Matrix(m) => Ok(match &m.col_names {
            Some(cn) => RVal::Character(cn.iter().map(|n| Some(n.clone())).collect(), Attrs::default()),
            None => RVal::Null,
        }),
        _ => Ok(RVal::Null),
    }
}

pub fn bi_rownames(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => match &df.row_names {
            Some(rn) => Ok(RVal::Character(rn.iter().map(|n| Some(n.clone())).collect(), Attrs::default())),
            None => Ok(RVal::Character(
                (1..=df.nrow()).map(|i| Some(Arc::from(format!("{}", i).as_str()))).collect(),
                Attrs::default(),
            )),
        },
        // R returns NULL for an unnamed matrix (no "1","2",… synthesis,
        // unlike data frames).
        RVal::Matrix(m) => Ok(match &m.row_names {
            Some(rn) => RVal::Character(rn.iter().map(|n| Some(n.clone())).collect(), Attrs::default()),
            None => RVal::Null,
        }),
        // a table result (anova) keeps its row names with it
        RVal::TypeInstance(inst) => Ok(inst.fields.get(".rownames").cloned().unwrap_or(RVal::Null)),
        _ => Ok(RVal::Null),
    }
}

pub fn bi_is_data_frame(a: &[EvalArg]) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(first_arg(a), RVal::DataFrame(_))))
}

pub fn bi_as_data_frame(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &first_arg(a) {
        RVal::DataFrame(df) => Ok(RVal::DataFrame(df.clone())),
        RVal::Matrix(m) => {
            let mut columns = Vec::new();
            for c in 0..m.ncol {
                let col: Vec<Real> = (0..m.nrow).map(|r| Some(m.get(r, c))).collect();
                let name = m.col_names.as_ref().and_then(|cn| cn.get(c)).cloned()
                    .unwrap_or_else(|| Arc::from(format!("V{}", c + 1).as_str()));
                columns.push((name, RVal::Numeric(col.into(), Attrs::default())));
            }
            Ok(RVal::DataFrame(DataFrame { columns, row_names: m.row_names.clone() }))
        }
        RVal::List(items, _) => {
            let columns: Vec<(Arc<str>, RVal)> = items.iter().enumerate().map(|(i, (n, v))| {
                let name = n.as_ref().cloned()
                    .unwrap_or_else(|| Arc::from(format!("V{}", i + 1).as_str()));
                (name, v.clone())
            }).collect();
            Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
        }
        other => Err(R2Err {
            msg: format!("cannot coerce {} to data.frame", other.type_name()),
            kind: ErrKind::Runtime,
        }),
    }
}

pub fn bi_head(a: &[EvalArg]) -> Result<RVal, R2Err> { head_tail(a, false) }

pub fn bi_tail(a: &[EvalArg]) -> Result<RVal, R2Err> { head_tail(a, true) }

/// R's `head(x, n)` / `tail(x, n)`: the first / last `n` elements of a
/// vector or factor, or rows of a data frame or matrix. Every column is
/// cut (factor columns included), row names and dimnames are kept — `tail`
/// labels unnamed rows with their original numbers, as R does — and a
/// negative `n` means all but `|n|`.
fn head_tail(a: &[EvalArg], tail: bool) -> Result<RVal, R2Err> {
    let o = first_arg(a);
    let n = arg_named(a, "n").or(Some(nth_arg(a, 1)))
        .and_then(|v| v.scalar_f64().ok().flatten())
        .unwrap_or(6.0) as i64;
    let pick = |len: usize| -> Vec<usize> {
        let k = if n >= 0 { (n as usize).min(len) } else { len.saturating_sub(n.unsigned_abs() as usize) };
        if tail { (len - k..len).collect() } else { (0..k).collect() }
    };
    Ok(match &o {
        RVal::DataFrame(df) => {
            let idx = pick(df.nrow());
            let row_names = match &df.row_names {
                Some(rn) => Some(idx.iter().map(|&i| rn[i].clone()).collect()),
                None if tail => Some(idx.iter().map(|&i| Arc::from((i + 1).to_string().as_str())).collect()),
                None => None,
            };
            let columns = df.columns.iter().map(|(name, col)| (name.clone(), cut(col, &idx))).collect();
            RVal::DataFrame(DataFrame { columns, row_names })
        }
        RVal::Matrix(m) => {
            let idx = pick(m.nrow);
            let mut data = Vec::with_capacity(idx.len() * m.ncol);
            for c in 0..m.ncol { for &r in &idx { data.push(m.get(r, c)); } }
            let mut out = Matrix::new(data, idx.len(), m.ncol);
            out.col_names = m.col_names.clone();
            out.row_names = m.row_names.as_ref().map(|rn| idx.iter().map(|&i| rn[i].clone()).collect());
            RVal::Matrix(out)
        }
        other => cut(other, &pick(r2_types::rval_length(other))),
    })
}

/// `x[idx]` for a column or vector, keeping its class (a `Date` stays a
/// `Date`) and names. A time series stays one, its `tsp` moved to the
/// window kept.
fn cut(v: &RVal, idx: &[usize]) -> RVal {
    let keep = |at: &Attrs| {
        let mut out = Attrs::default();
        out.names = at.names.as_ref().map(|ns| idx.iter().map(|&i| ns[i].clone()).collect());
        out.set_classes(at.classes());
        if let (Some(RVal::Numeric(t, _)), Some(&first), Some(&last)) = (at.custom.get("tsp"), idx.first(), idx.last()) {
            if let (Some(start), Some(freq)) = (t.get(0).copied().flatten(), t.get(2).copied().flatten()) {
                let tsp = vec![Some(start + first as f64 / freq), Some(start + last as f64 / freq), Some(freq)];
                out.custom.insert(Arc::from("tsp"), RVal::Numeric(tsp.into(), Attrs::default()));
            }
        }
        out
    };
    match v {
        RVal::Numeric(x, at) => RVal::Numeric(idx.iter().map(|&i| x[i]).collect::<Vec<Real>>().into(), keep(at)),
        RVal::Integer(x, at) => RVal::Integer(idx.iter().map(|&i| x[i]).collect::<Vec<Option<i32>>>().into(), keep(at)),
        RVal::Logical(x, at) => RVal::Logical(idx.iter().map(|&i| x[i]).collect::<Vec<Option<bool>>>().into(), keep(at)),
        RVal::Character(x, at) => RVal::Character(idx.iter().map(|&i| x[i].clone()).collect(), keep(at)),
        RVal::Factor(_) => r2_types::take(v, idx),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn dim_returns_nrow_ncol_for_dataframe() {
        let df = DataFrame {
            columns: vec![
                (Arc::from("a"), RVal::Numeric(vec![Some(1.0), Some(2.0), Some(3.0)].into(), Attrs::default())),
                (Arc::from("b"), RVal::Numeric(vec![Some(4.0), Some(5.0), Some(6.0)].into(), Attrs::default())).into(),
            ],
            row_names: None,
        };
        let r = bi_dim(&[evarg(RVal::DataFrame(df))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![3, 2]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn head_takes_first_n_of_numeric() {
        let v = RVal::Numeric(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)].into(), Attrs::default());
        let r = bi_head(&[evarg(v), evarg(RVal::Numeric(vec![Some(2.0)].into(), Attrs::default()))]).unwrap();
        match r {
            RVal::Numeric(v, _) => assert_eq!(v.len(), 2),
            _ => panic!(),
        }
    }

    #[test]
    fn matrix_dimnames_getters_match_r() {
        use r2_types::Matrix;
        // R: colnames/rownames of an unnamed matrix are NULL.
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(matches!(bi_colnames(&[evarg(RVal::Matrix(m.clone()))]).unwrap(), RVal::Null));
        assert!(matches!(bi_rownames(&[evarg(RVal::Matrix(m.clone()))]).unwrap(), RVal::Null));
        // After setting, the getters return the character vector.
        let mut named = m;
        named.col_names = Some(vec![Arc::from("a"), Arc::from("b")]);
        named.row_names = Some(vec![Arc::from("r1"), Arc::from("r2")]);
        let cn = bi_colnames(&[evarg(RVal::Matrix(named.clone()))]).unwrap();
        match cn {
            RVal::Character(v, _) => {
                let got: Vec<String> = v.iter().filter_map(|x| x.clone().map(|s| s.to_string())).collect();
                assert_eq!(got, vec!["a", "b"]);
            }
            other => panic!("colnames returned {:?}", other),
        }
        let rn = bi_rownames(&[evarg(RVal::Matrix(named))]).unwrap();
        match rn {
            RVal::Character(v, _) => {
                let got: Vec<String> = v.iter().filter_map(|x| x.clone().map(|s| s.to_string())).collect();
                assert_eq!(got, vec!["r1", "r2"]);
            }
            other => panic!("rownames returned {:?}", other),
        }
    }

    #[test]
    fn is_data_frame_distinguishes() {
        assert!(matches!(bi_is_data_frame(&[evarg(RVal::Null)]).unwrap(), RVal::Logical(v, _) if v[0] == Some(false)));
        let df = RVal::DataFrame(DataFrame { columns: vec![], row_names: None });
        assert!(matches!(bi_is_data_frame(&[evarg(df)]).unwrap(), RVal::Logical(v, _) if v[0] == Some(true)));
    }
}
