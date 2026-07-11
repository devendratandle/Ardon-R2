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
        RVal::List(items) => {
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

pub fn bi_head(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let o = first_arg(a);
    let n = arg_named(a, "n").or(Some(nth_arg(a, 1)))
        .and_then(|v| v.scalar_f64().ok().flatten())
        .unwrap_or(6.0) as usize;
    match &o {
        RVal::Numeric(v, at) => Ok(RVal::Numeric(v.iter().take(n).cloned().collect(), at.clone())),
        RVal::Integer(v, at) => Ok(RVal::Integer(v.iter().take(n).cloned().collect(), at.clone())),
        RVal::Character(v, at) => Ok(RVal::Character(v.iter().take(n).cloned().collect(), at.clone())),
        RVal::Logical(v, at) => Ok(RVal::Logical(v.iter().take(n).cloned().collect(), at.clone())),
        RVal::DataFrame(df) => {
            let nr = n.min(df.nrow());
            let cols: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
                let sub = match col {
                    RVal::Numeric(v, _) => RVal::Numeric(v.iter().take(nr).cloned().collect(), Attrs::default()),
                    RVal::Integer(v, _) => RVal::Integer(v.iter().take(nr).cloned().collect(), Attrs::default()),
                    RVal::Character(v, _) => RVal::Character(v.iter().take(nr).cloned().collect(), Attrs::default()),
                    RVal::Logical(v, _) => RVal::Logical(v.iter().take(nr).cloned().collect(), Attrs::default()),
                    _ => col.clone(),
                };
                (name.clone(), sub)
            }).collect();
            Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: None }))
        }
        RVal::Matrix(m) => {
            let nr = n.min(m.nrow);
            let mut data = Vec::with_capacity(nr * m.ncol);
            for c in 0..m.ncol {
                for r in 0..nr { data.push(m.get(r, c)); }
            }
            Ok(RVal::Matrix(Matrix::new(data, nr, m.ncol)))
        }
        _ => Ok(o),
    }
}

pub fn bi_tail(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let o = first_arg(a);
    let n = nth_arg(a, 1).scalar_f64()?.unwrap_or(6.0) as usize;
    match &o {
        RVal::Numeric(v, _) => {
            let skip = v.len().saturating_sub(n);
            Ok(RVal::Numeric(v.iter().skip(skip).cloned().collect(), Attrs::default()))
        }
        _ => Ok(o),
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
