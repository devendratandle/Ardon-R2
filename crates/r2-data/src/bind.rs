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

/// The extent `cbind` and `rbind` share. Matrices fix it and must agree on it;
/// with none, the longest vector does.
fn shared_extent(a: &[EvalArg], side: &str, of: fn(&Matrix) -> usize) -> Result<usize, R2Err> {
    let (mut fixed, mut longest) = (None, 0);
    for (i, arg) in a.iter().enumerate() {
        match &arg.value {
            RVal::Null => {}
            RVal::Matrix(m) => match fixed {
                None => fixed = Some(of(m)),
                Some(e) if e != of(m) => return Err(R2Err {
                    msg: format!("number of {side} of matrices must match (see arg {})", i + 1),
                    kind: ErrKind::Runtime,
                }),
                _ => {}
            },
            other => longest = longest.max(coerce_to_columns(other)?.1),
        }
    }
    Ok(fixed.unwrap_or(longest))
}

/// A vector recycled or cut to `extent`, warning as R does when it does not
/// divide evenly. An empty vector drops out unless everything is empty.
fn recycled(v: &RVal, extent: usize, call: &str, side: &str, arg: usize) -> Result<Option<Vec<f64>>, R2Err> {
    let (d, n, _) = coerce_to_columns(v)?;
    if n == 0 { return Ok(if extent == 0 { Some(d) } else { None }); }
    if extent % n != 0 {
        r2_types::out::rerr(&format!("Warning: {call}: number of {side} of result is not a multiple of vector length (arg {})
", arg + 1));
    }
    Ok(Some((0..extent).map(|i| d[i % n]).collect()))
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
    let ncol = shared_extent(a, "columns", |m| m.ncol)?;
    let mut blocks: Vec<(Vec<f64>, usize, usize)> = Vec::with_capacity(a.len());
    for (i, arg) in a.iter().enumerate() {
        match &arg.value {
            RVal::Null => {}
            RVal::Matrix(m) => blocks.push((m.data.clone(), m.nrow, m.ncol)),
            other => if let Some(d) = recycled(other, ncol, "rbind", "columns", i)? { blocks.push((d, 1, ncol)); },
        }
    }
    if blocks.is_empty() { return Ok(RVal::Null); }
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
    let nrow = shared_extent(a, "rows", |m| m.nrow)?;
    let mut blocks: Vec<(Vec<f64>, usize, usize, Option<Vec<Arc<str>>>)> = Vec::with_capacity(a.len());
    let mut any_names = false;
    for (i, arg) in a.iter().enumerate() {
        let names: Option<Vec<Arc<str>>> = match &arg.value {
            RVal::Matrix(m) => m.col_names.clone(),
            _ => arg.name.as_ref().map(|n| vec![n.clone()]),
        };
        let (data, ncol) = match &arg.value {
            RVal::Null => continue,
            RVal::Matrix(m) => (m.data.clone(), m.ncol),
            other => match recycled(other, nrow, "cbind", "rows", i)? { Some(d) => (d, 1), None => continue },
        };
        if names.is_some() { any_names = true; }
        blocks.push((data, nrow, ncol, names));
    }
    if blocks.is_empty() { return Ok(RVal::Null); }
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
    fn rbind_matrices_must_agree_and_vectors_recycle() {
        let matrix = |ncol| EvalArg { name: None, value: RVal::Matrix(Matrix::new(vec![1.0; 2 * ncol], 2, ncol)) };
        let e = bi_rbind(&[matrix(2), matrix(3)]).unwrap_err();
        assert!(e.msg.contains("number of columns of matrices must match (see arg 2)"), "{}", e.msg);

        let vector = |v: &[f64]| EvalArg { name: None, value: RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) };
        match bi_rbind(&[vector(&[1.0, 2.0]), vector(&[3.0, 4.0, 5.0])]).unwrap() {
            RVal::Matrix(m) => {
                assert_eq!((m.nrow, m.ncol), (2, 3));
                assert_eq!(m.data, vec![1.0, 3.0, 2.0, 4.0, 1.0, 5.0]);
            }
            other => panic!("expected a matrix, got {}", other.type_name()),
        }
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
