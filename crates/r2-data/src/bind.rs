//! cbind / rbind — Phase R.2 step 1.
//!
//! Both behave dual-mode:
//!   - All inputs are `data.frame` → returns a `data.frame` (column-typed,
//!     preserves names and types per column).
//!   - Otherwise → returns a `Matrix` (numeric, contiguous f64 buffer).
//!
//! Keeps r2-engine's existing semantics bit-for-bit; only the location of
//! the code changes.

use r2_types::*;
use std::sync::Arc;

/// Coerce an `RVal` to (data, nrows, ncols) in column-major f64 form.
/// Used by both cbind (vector→1col) and rbind (vector→1row).
pub(crate) fn coerce_to_columns(v: &RVal) -> Result<(Vec<f64>, usize, usize), R2Err> {
    match v {
        RVal::Matrix(m) => Ok((m.data.clone(), m.nrow, m.ncol)),
        RVal::Numeric(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Integer(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|i| i as f64).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Logical(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 }).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        _ => Err(R2Err {
            msg: format!("cbind/rbind: cannot coerce {} to numeric matrix", v.type_name()),
            kind: ErrKind::Type,
        }),
    }
}

fn all_dataframes(a: &[EvalArg]) -> bool {
    !a.is_empty() && a.iter().all(|x| matches!(x.value, RVal::DataFrame(_)))
}

/// rbind — stack rows. data.frame path stacks per-column with type checks;
/// matrix path stacks blocks (vectors become 1-row 1-col-per-element matrices).
pub fn bi_rbind(a: &[EvalArg]) -> Result<RVal, R2Err> {
    if a.is_empty() {
        return Err(R2Err { msg: "rbind: needs at least one argument".into(), kind: ErrKind::Runtime });
    }

    if all_dataframes(a) {
        let mut iter = a.iter();
        let first = match &iter.next().unwrap().value {
            RVal::DataFrame(df) => df.clone(),
            _ => unreachable!(),
        };
        let ncol = first.ncol();
        let mut columns: Vec<(Arc<str>, RVal)> = first.columns.clone();
        for arg in iter {
            let df = match &arg.value {
                RVal::DataFrame(df) => df.clone(),
                _ => unreachable!(),
            };
            if df.ncol() != ncol {
                return Err(R2Err {
                    msg: format!("rbind: column count mismatch ({} vs {})", ncol, df.ncol()),
                    kind: ErrKind::Runtime,
                });
            }
            for (i, (name, col2)) in df.columns.iter().enumerate() {
                let (cur_name, cur_col) = columns[i].clone();
                let merged = match (&cur_col, col2) {
                    (RVal::Numeric(v1, _),   RVal::Numeric(v2, _))   => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Numeric(v.into(), Attrs::default()) }
                    (RVal::Integer(v1, _),   RVal::Integer(v2, _))   => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Integer(v.into(), Attrs::default()) }
                    (RVal::Character(v1, _), RVal::Character(v2, _)) => { let mut v = v1.clone(); v.extend(v2.clone());  RVal::Character(v, Attrs::default()) }
                    (RVal::Logical(v1, _),   RVal::Logical(v2, _))   => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Logical(v.into(), Attrs::default()) }
                    _ => return Err(R2Err {
                        msg: format!("rbind: incompatible column types at '{}'", name),
                        kind: ErrKind::Type,
                    }),
                };
                columns[i] = (cur_name, merged);
            }
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path
    let mut blocks: Vec<(Vec<f64>, usize, usize)> = Vec::with_capacity(a.len());
    for arg in a {
        let (data, nrow, ncol) = match &arg.value {
            RVal::Matrix(m) => (m.data.clone(), m.nrow, m.ncol),
            other => {
                let (d, n, _) = coerce_to_columns(other)?;
                (d, 1, n)
            }
        };
        blocks.push((data, nrow, ncol));
    }
    let ncol = blocks[0].2;
    if !blocks.iter().all(|(_, _, c)| *c == ncol) {
        return Err(R2Err { msg: "rbind: column count mismatch across inputs".into(), kind: ErrKind::Runtime });
    }
    let total_rows: usize = blocks.iter().map(|(_, r, _)| *r).sum();
    let mut data = vec![0.0; total_rows * ncol];
    for j in 0..ncol {
        let mut row_offset = 0;
        for (b_data, b_nrow, _) in &blocks {
            for i in 0..*b_nrow {
                data[j * total_rows + row_offset + i] = b_data[j * b_nrow + i];
            }
            row_offset += b_nrow;
        }
    }
    Ok(RVal::Matrix(Matrix::new(data, total_rows, ncol)))
}

/// cbind — append columns. data.frame path concatenates columns;
/// matrix path stacks blocks side-by-side, preserving column names.
pub fn bi_cbind(a: &[EvalArg]) -> Result<RVal, R2Err> {
    if a.is_empty() {
        return Err(R2Err { msg: "cbind: needs at least one argument".into(), kind: ErrKind::Runtime });
    }

    // data.frame path: if ANY argument is a data.frame, the result is a
    // data.frame — non-df args (vectors) are appended as new columns
    // (named by their argument name, else V<n>). R's `cbind(df, x=...)`.
    if a.iter().any(|x| matches!(&x.value, RVal::DataFrame(_))) {
        let nrow = a.iter().find_map(|x| match &x.value {
            RVal::DataFrame(d) => Some(d.nrow()), _ => None,
        }).unwrap_or(0);
        let mut columns: Vec<(Arc<str>, RVal)> = Vec::new();
        for arg in a {
            match &arg.value {
                RVal::DataFrame(df) => {
                    if df.nrow() != nrow {
                        return Err(R2Err {
                            msg: format!("cbind: row count mismatch ({} vs {})", nrow, df.nrow()),
                            kind: ErrKind::Runtime,
                        });
                    }
                    columns.extend(df.columns.clone());
                }
                other => {
                    let name = arg.name.clone()
                        .unwrap_or_else(|| Arc::from(format!("V{}", columns.len() + 1).as_str()));
                    columns.push((name, other.clone()));
                }
            }
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path
    let mut blocks: Vec<(Vec<f64>, usize, usize, Option<Vec<Arc<str>>>)> = Vec::with_capacity(a.len());
    let mut any_names = false;
    for arg in a {
        let (data, nrow, ncol) = coerce_to_columns(&arg.value)?;
        let names: Option<Vec<Arc<str>>> = match &arg.value {
            RVal::Matrix(m) => m.col_names.clone(),
            _ => arg.name.as_ref().map(|n| vec![n.clone()]),
        };
        if names.is_some() { any_names = true; }
        blocks.push((data, nrow, ncol, names));
    }
    let nrow = blocks[0].1;
    if !blocks.iter().all(|(_, r, _, _)| *r == nrow) {
        return Err(R2Err { msg: "cbind: row count mismatch across inputs".into(), kind: ErrKind::Runtime });
    }
    let total_cols: usize = blocks.iter().map(|(_, _, c, _)| *c).sum();
    let mut data = Vec::with_capacity(nrow * total_cols);
    let mut col_names: Vec<Arc<str>> = Vec::with_capacity(total_cols);
    for (b_data, _, b_ncol, b_names) in &blocks {
        data.extend_from_slice(b_data);
        match b_names {
            Some(ns) if ns.len() == *b_ncol => col_names.extend(ns.iter().cloned()),
            _ => for _ in 0..*b_ncol {
                col_names.push(Arc::from(format!("V{}", col_names.len() + 1).as_str()));
            }
        }
    }
    let mut m = Matrix::new(data, nrow, total_cols);
    if any_names { m.col_names = Some(col_names); }
    Ok(RVal::Matrix(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnum_args(vs: &[f64]) -> Vec<EvalArg> {
        vec![EvalArg {
            name: None,
            value: RVal::Numeric(vs.iter().map(|x| Some(*x)).collect(), Attrs::default()),
        }]
    }

    #[test]
    fn cbind_two_vectors_makes_2col_matrix() {
        let a = vec![
            EvalArg { name: None, value: RVal::Numeric(vec![Some(1.0), Some(2.0), Some(3.0)].into(), Attrs::default()) },
            EvalArg { name: None, value: RVal::Numeric(vec![Some(4.0), Some(5.0), Some(6.0)].into(), Attrs::default()) }.into(),
        ];
        let r = bi_cbind(&a).expect("ok");
        match r {
            RVal::Matrix(m) => {
                assert_eq!(m.nrow, 3);
                assert_eq!(m.ncol, 2);
                assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected Matrix, got {:?}", other.type_name()),
        }
    }

    #[test]
    fn rbind_row_count_mismatch_errs() {
        let a = vec![
            EvalArg { name: None, value: RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), Attrs::default()) },
            EvalArg { name: None, value: RVal::Numeric(vec![Some(3.0), Some(4.0), Some(5.0)].into(), Attrs::default()) }.into(),
        ];
        let r = bi_rbind(&a);
        assert!(r.is_err(), "expected error on mismatched column count");
    }

    #[test]
    fn cbind_single_arg_round_trip() {
        let r = bi_cbind(&rnum_args(&[1.0, 2.0, 3.0])).expect("ok");
        match r {
            RVal::Matrix(m) => assert_eq!(m.ncol, 1),
            other => panic!("expected Matrix, got {:?}", other.type_name()),
        }
    }
}
