//! ML data preprocessing — Phase R.1 step 3.
//!
//! `extract_ml_data` resolves either:
//!   - a formula+data combination (`y ~ x1 + x2`, `data = df`), or
//!   - a matrix+vector pair (`x_matrix, y_vector`)
//! into a uniform `(y: Vec<f64>, x: Matrix, col_names: Vec<String>)` shape
//! that all ML builtins consume.
//!
//! Pure: no Engine reference. Uses `RVal::as_reals()` (Phase R.1 step 2)
//! for type coercion. The formula representation here is the R2-engine
//! internal one (a `RVal::List` with `~lhs`/`~rhs`/`~class` entries) —
//! the engine's NSE preprocessor builds that shape before calling.

use r2_types::*;

#[inline]
fn gv(args: &[EvalArg], i: usize) -> RVal {
    args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter()
        .find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

pub fn extract_ml_data(a: &[EvalArg]) -> Result<(Vec<f64>, Matrix, Vec<String>), R2Err> {
    let first = gv(a, 0);

    // Check if first arg is a formula (List with ~class = "formula").
    let is_formula = match &first {
        RVal::List(items, _) => items.iter().any(|(n, v)| {
            n.as_ref().map(|s| s.as_ref()) == Some("~class")
                && matches!(v, RVal::Character(sv, _)
                    if sv.first().and_then(|x| x.as_ref()).map(|s| s.as_ref()) == Some("formula"))
        }),
        _ => false,
    };

    if is_formula {
        let data = gn(a, "data").ok_or(R2Err {
            msg: "formula requires data= argument".into(),
            kind: ErrKind::Runtime,
        })?;
        let df = match &data {
            RVal::DataFrame(df) => df.clone(),
            _ => return Err(R2Err { msg: "data must be data.frame".into(), kind: ErrKind::Runtime }),
        };

        let items: Vec<(Option<std::sync::Arc<str>>, RVal)> = match &first {
            RVal::List(v, _) => v.clone(),
            _ => vec![],
        };
        let lhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~lhs"))
            .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);

        // y from LHS
        let y_col = match &lhs_raw {
            RVal::List(items, _) if !items.is_empty() => items[0].1.clone(),
            other => other.clone(),
        };
        let y: Vec<f64> = match &y_col {
            RVal::Character(v, _) => {
                let mut levels: Vec<String> = Vec::new();
                v.iter().map(|x| {
                    let s = x.as_ref().map(|s| s.to_string()).unwrap_or("NA".into());
                    if let Some(pos) = levels.iter().position(|l| l == &s) { (pos + 1) as f64 }
                    else { levels.push(s); levels.len() as f64 }
                }).collect()
            }
            RVal::Factor(f) => {
                f.codes.iter().map(|c| c.map(|i| (i + 1) as f64).unwrap_or(f64::NAN)).collect()
            }
            _ => y_col.as_reals()?.into_iter().filter_map(|x| x).collect(),
        };

        // RHS — explicit columns or `.` (all-other-numeric)
        let rhs_raw = items.iter().find(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some("~rhs"))
            .map(|(_, v)| v.clone()).unwrap_or(RVal::Null);

        let mut x_data: Vec<f64> = Vec::new();
        let mut col_names: Vec<String> = Vec::new();
        let nrow = df.nrow();

        let has_rhs_data = match &rhs_raw {
            RVal::List(items, _) => items.iter()
                .any(|(n, _)| !n.as_ref().map(|s| s.starts_with("~")).unwrap_or(true)),
            RVal::Null => false,
            _ => true,
        };

        if !has_rhs_data {
            let y_name = match &lhs_raw {
                RVal::List(items, _) if !items.is_empty() => items[0].0.as_ref().map(|s| s.to_string()),
                _ => None,
            };
            for (name, col) in &df.columns {
                if y_name.as_ref().map(|yn| yn == name.as_ref()).unwrap_or(false) { continue; }
                if let Ok(vals) = col.as_reals() {
                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                    if nums.len() == nrow { x_data.extend(&nums); col_names.push(name.to_string()); }
                }
            }
        } else {
            match &rhs_raw {
                RVal::List(items, _) => {
                    for (name, val) in items {
                        if name.as_ref().map(|s| s.starts_with("~")).unwrap_or(false) { continue; }
                        let actual = match val { RVal::List(inner, _) if !inner.is_empty() => &inner[0].1, v => v };
                        if let Ok(vals) = actual.as_reals() {
                            let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                            if nums.len() == nrow {
                                x_data.extend(&nums);
                                let cname = match val {
                                    RVal::List(inner, _) if !inner.is_empty() => inner[0].0.as_ref().map(|s| s.to_string()),
                                    _ => name.as_ref().map(|s| s.to_string()),
                                };
                                col_names.push(cname.unwrap_or(format!("X{}", col_names.len() + 1)));
                            }
                        }
                    }
                }
                _ => {
                    if let Ok(vals) = rhs_raw.as_reals() {
                        let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                        x_data.extend(&nums);
                        col_names.push("X1".into());
                    }
                }
            }
        }

        let ncol = col_names.len();
        if ncol == 0 {
            return Err(R2Err { msg: "no numeric predictor columns found".into(), kind: ErrKind::Runtime });
        }
        Ok((y, Matrix::new(x_data, nrow, ncol), col_names))
    } else {
        // Matrix + vector path
        let mat = match &first {
            RVal::Matrix(m) => m.clone(),
            _ => return Err(R2Err {
                msg: "first argument must be matrix or formula".into(),
                kind: ErrKind::Runtime,
            }),
        };
        let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
        let col_names: Vec<String> = match &mat.col_names {
            Some(names) if names.len() == mat.ncol => names.iter().map(|s| s.to_string()).collect(),
            _ => (0..mat.ncol).map(|i| format!("X{}", i + 1)).collect(),
        };
        Ok((y, mat, col_names))
    }
}
