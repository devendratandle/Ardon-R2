//! Element assignment — `x[i]<-`, `x[[i]]<-`, `x$f<-` (companion to indexing.rs).

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::{Engine, val_to_str};
use crate::err;

impl Engine {
    /// `target <- value` (and `<<-` when `superassign`): a variable, an
    /// element (`x[i]`, `m[i, j]`, `x[[i]]`, `x$f`) or a replacement call
    /// (`names(x) <- v`). An assignment evaluates to the assigned value.
    pub(crate) fn eval_assign(&mut self, target: &Expr, value: &Expr, superassign: bool,
                              env: &EnvRef) -> Result<RVal, R2Err> {
        let val = self.eval_in(value, env)?;
        match target {
            Expr::Symbol(name) => {
                if matches!(name.as_ref(), "TRUE"|"FALSE") { return err!(Runtime, "cannot assign to the reserved constant '{}'", name); }
                if superassign { self.super_assign(name.clone(), val.clone()); }
                else { self.scope_insert(name.clone(), val.clone()); }
                Ok(val)
            }
            // Literals are not valid targets (e.g. `1 <- x`).
            Expr::NumLit(_) | Expr::IntLit(_) | Expr::StrLit(_) | Expr::BoolLit(_)
            | Expr::NullLit | Expr::NaLit =>
                err!(Runtime, "cannot assign to a literal value"),
            Expr::Index { object, indices } => {
                if let Expr::Symbol(name) = object.as_ref() {
                    let mut obj = self.eval_in(object, env)?;
                    if indices.len() == 1 {
                        if let Some(idx_expr) = &indices[0] {
                            let idx = self.eval_in(idx_expr, env)?;
                            self.assign_index(&mut obj, &idx, &val)?;
                        }
                    } else if indices.len() == 2 {
                        // Matrix `m[i, j] <- v` / `m[i, ] <- v` / `m[, j] <- v`.
                        // An empty subscript (None) selects the whole axis.
                        let ri = match &indices[0] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                        let ci = match &indices[1] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                        self.assign_matrix_index(&mut obj, ri.as_ref(), ci.as_ref(), &val)?;
                    }
                    self.scope_insert(name.clone(), obj.clone());
                    Ok(val)
                } else { err!(Runtime, "invalid subscript assignment target") }
            }
            Expr::DblIndex { object, index } => {
                if let Expr::Symbol(name) = object.as_ref() {
                    let mut obj = self.eval_in(object, env)?;
                    let idx = self.eval_in(index, env)?;
                    self.assign_dbl_index(&mut obj, &idx, &val)?;
                    self.scope_insert(name.clone(), obj.clone());
                    Ok(val)
                } else { err!(Runtime, "invalid [[ ]] assignment target") }
            }
            Expr::Dollar { object, field } => {
                if let Expr::Symbol(name) = object.as_ref() {
                    let mut obj = self.eval_in(object, env)?;
                    self.assign_dollar(&mut obj, field, &val)?;
                    self.scope_insert(name.clone(), obj.clone());
                    Ok(val)
                } else { err!(Runtime, "invalid $ assignment target") }
            }
            // Replacement function: `fname(obj, ...) <- value`
            // desugars to `obj <- \`fname<-\`(obj, ..., value=value)`.
            // Enables names(x)<-, colnames(df)<-, rownames(df)<-, etc.
            Expr::Call { func, args } => {
                if let (Expr::Symbol(fname), Some(first)) = (func.as_ref(), args.first()) {
                    let setter = format!("{}<-", fname);
                    if let Some((f, _)) = self.registry.resolve(&setter) {
                        let obj_val = self.eval_in(&first.value, env)?;
                        let mut ea = vec![EvalArg { name: None, value: obj_val }];
                        for extra in &args[1..] {
                            ea.push(EvalArg { name: extra.name.clone(), value: self.eval_in(&extra.value, env)? });
                        }
                        ea.push(EvalArg { name: Some(Arc::from("value")), value: val.clone() });
                        let new_obj = f(self, &ea, env)?;
                        if let Expr::Symbol(objname) = &first.value {
                            self.scope_insert(objname.clone(), new_obj);
                            return Ok(val);
                        }
                        return err!(Runtime, "replacement target must be a variable");
                    }
                    return err!(Runtime, "could not find function \"{}\"", setter);
                }
                err!(Runtime, "invalid assignment target")
            }
            _ => err!(Runtime, "invalid assignment target"),
        }
    }

    pub(crate) fn assign_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        let positions = self.as_reals(idx)?;
        match obj {
            RVal::Numeric(v, _) => {
                let new_vals = self.as_reals(val)?;
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid (1-based indexing)"); }
                        // Extend vector if needed
                        // Reals: DerefMut to &mut [Real] doesn't allow push.
                        // Move out, push, move back via .into() reconstruction.
                        let mut tmp: Vec<Real> = std::mem::take(&mut *v).into_inner();
                        while tmp.len() < i { tmp.push(None); }
                        tmp[i - 1] = new_vals.get(pi % new_vals.len()).copied().unwrap_or(None);
                        *v = tmp.into();
                    }
                }
                Ok(())
            }
            RVal::Character(v, _) => {
                let new_val = match val { RVal::Character(sv, _) => sv.clone(), _ => vec![Some(Arc::from(val_to_str(val).as_str()))] };
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                        while v.len() < i { v.push(None); }
                        v[i - 1] = new_val.get(pi % new_val.len()).cloned().unwrap_or(None);
                    }
                }
                Ok(())
            }
            RVal::Integer(v, _) => {
                let new_vals = self.as_reals(val)?;
                // Ints/Logicals share the F.3 pattern: DerefMut gives a
                // slice not a Vec, so push/extend need a take→push→put-back.
                let mut tmp: Vec<Integer> = std::mem::take(&mut *v).into_inner();
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                        while tmp.len() < i { tmp.push(None); }
                        tmp[i - 1] = new_vals.get(pi % new_vals.len()).copied().unwrap_or(None).map(|n| n as i32);
                    }
                }
                *v = tmp.into();
                Ok(())
            }
            _ => err!(Runtime, "cannot assign by index to {}", obj.type_name()),
        }
    }

    /// `m[rows, cols] <- v` — matrix element/row/column assignment. An
    /// empty subscript (`None`) selects the whole axis. Values recycle
    /// column-major over the selected submatrix (R semantics).
    pub(crate) fn assign_matrix_index(&mut self, obj: &mut RVal, row: Option<&RVal>, col: Option<&RVal>, val: &RVal) -> Result<(), R2Err> {
        let new_vals = self.as_reals(val)?;
        let sel = |axis: Option<&RVal>, n: usize, this: &Self| -> Result<Vec<usize>, R2Err> {
            match axis {
                None => Ok((0..n).collect()),
                Some(RVal::Logical(mask, _)) => Ok(mask.iter().enumerate()
                    .filter_map(|(i, b)| if *b == Some(true) { Some(i) } else { None }).collect()),
                Some(idx) => this.resolve_subscript(idx, n),
            }
        };
        if let RVal::Matrix(m) = obj {
            let keep_rows = sel(row, m.nrow, self)?;
            let keep_cols = sel(col, m.ncol, self)?;
            if new_vals.is_empty() { return err!(Runtime, "replacement has length zero"); }
            let mut k = 0usize;
            for &j in &keep_cols {
                for &i in &keep_rows {
                    let v = new_vals[k % new_vals.len()];
                    m.data[j * m.nrow + i] = v.unwrap_or(f64::NAN);
                    k += 1;
                }
            }
            Ok(())
        } else {
            err!(Runtime, "two-subscript assignment not supported for {}", obj.type_name())
        }
    }

    pub(crate) fn assign_dbl_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::List(items) => {
                let i = self.scalar_f64(idx)?.unwrap_or(1.0) as usize;
                if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                while items.len() < i { items.push((None, RVal::Null)); }
                items[i - 1].1 = val.clone();
                Ok(())
            }
            _ => self.assign_index(obj, idx, val),
        }
    }

    pub(crate) fn assign_dollar(&mut self, obj: &mut RVal, field: &str, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::DataFrame(df) => {
                // Find existing column or add new
                if let Some(pos) = df.columns.iter().position(|(n, _)| n.as_ref() == field) {
                    df.columns[pos].1 = val.clone();
                } else {
                    df.columns.push((Arc::from(field), val.clone()));
                }
                Ok(())
            }
            RVal::List(items) => {
                let field_arc = Arc::from(field);
                if let Some(pos) = items.iter().position(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some(field)) {
                    items[pos].1 = val.clone();
                } else {
                    items.push((Some(field_arc), val.clone()));
                }
                Ok(())
            }
            RVal::TypeInstance(inst) => {
                inst.fields.insert(Arc::from(field), val.clone());
                Ok(())
            }
            _ => err!(Runtime, "$ assignment not supported for {}", obj.type_name()),
        }
    }
}
