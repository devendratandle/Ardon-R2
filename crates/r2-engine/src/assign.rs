//! Element assignment — `x[i]<-`, `x[[i]]<-`, `x$f<-` (companion to indexing.rs).

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::{Engine, val_to_str};
use crate::err;

impl Engine {
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
