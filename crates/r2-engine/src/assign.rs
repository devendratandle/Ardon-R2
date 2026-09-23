//! Element assignment — `x[i]<-`, `x[[i]]<-`, `x$f<-` (companion to indexing.rs).

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::Engine;
use crate::err;

impl Engine {
    /// `target <- value` (and `<<-` when `superassign`): a variable, an
    /// element (`x[i]`, `m[i, j]`, `x[[i]]`, `x$f`) or a replacement call
    /// (`names(x) <- v`). An assignment evaluates to the assigned value.
    pub(crate) fn eval_assign(&mut self, target: &Expr, value: &Expr, superassign: bool,
                              env: &EnvRef) -> Result<RVal, R2Err> {
        let val = self.eval_in(value, env)?;
        self.assign_to(target, val.clone(), superassign, env)?;
        Ok(val)
    }

    /// Store `val` into `target`. An element target reads its object,
    /// replaces the element and stores the object back into its own target
    /// the same way, so nested forms work: `df$a[2] <- v` stores `df$a`
    /// with element 2 replaced back into `df`, and `names(x)[2] <- "b"`
    /// goes through `names<-`.
    fn assign_to(&mut self, target: &Expr, val: RVal, superassign: bool, env: &EnvRef) -> Result<(), R2Err> {
        match target {
            Expr::Symbol(name) => {
                if matches!(name.as_ref(), "TRUE"|"FALSE") { return err!(Runtime, "cannot assign to the reserved constant '{}'", name); }
                if superassign { self.super_assign(name.clone(), val); }
                else { self.scope_insert(name.clone(), val); }
                Ok(())
            }
            // Literals are not valid targets (e.g. `1 <- x`).
            Expr::NumLit(_) | Expr::IntLit(_) | Expr::StrLit(_) | Expr::BoolLit(_)
            | Expr::NullLit | Expr::NaLit =>
                err!(Runtime, "cannot assign to a literal value"),
            Expr::Index { object, indices } => {
                let mut obj = self.eval_in(object, env)?;
                match indices.len() {
                    1 => {
                        // `x[] <- v` fills every element.
                        let idx = match &indices[0] {
                            Some(e) => self.eval_in(e, env)?,
                            None => RVal::Logical(vec![Some(true)].into(), Attrs::default()),
                        };
                        self.assign_index(&mut obj, &idx, &val)?;
                    }
                    2 => {
                        // `m[i, j] <- v` / `df[i, j] <- v`; an empty subscript
                        // (None) selects the whole axis.
                        let ri = match &indices[0] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                        let ci = match &indices[1] { Some(e) => Some(self.eval_in(e, env)?), None => None };
                        self.assign_matrix_index(&mut obj, ri.as_ref(), ci.as_ref(), &val)?;
                    }
                    _ => return err!(Runtime, "only one- and two-subscript assignment is supported"),
                }
                self.assign_to(object, obj, superassign, env)
            }
            Expr::DblIndex { object, index } => {
                let mut obj = self.eval_in(object, env)?;
                let idx = self.eval_in(index, env)?;
                self.assign_dbl_index(&mut obj, &idx, &val)?;
                self.assign_to(object, obj, superassign, env)
            }
            Expr::Dollar { object, field } => {
                let mut obj = self.eval_in(object, env)?;
                self.assign_dollar(&mut obj, field, &val)?;
                self.assign_to(object, obj, superassign, env)
            }
            // Replacement function: `fname(obj, ...) <- value` calls the
            // setter `fname<-`(obj, ..., value = value) — names(x)<-,
            // colnames(df)<-, rownames(df)<-, levels(f)<-, ...
            Expr::Call { func, args } => {
                if let (Expr::Symbol(fname), Some(first)) = (func.as_ref(), args.first()) {
                    let setter = format!("{}<-", fname);
                    if let Some((f, _)) = self.registry.resolve(&setter) {
                        let obj_val = self.eval_in(&first.value, env)?;
                        let mut ea = vec![EvalArg { name: None, value: obj_val }];
                        for extra in &args[1..] {
                            ea.push(EvalArg { name: extra.name.clone(), value: self.eval_in(&extra.value, env)? });
                        }
                        ea.push(EvalArg { name: Some(Arc::from("value")), value: val });
                        let new_obj = f(self, &ea, env)?;
                        return self.assign_to(&first.value, new_obj, superassign, env);
                    }
                    return err!(Runtime, "could not find function \"{}\"", setter);
                }
                err!(Runtime, "invalid assignment target")
            }
            _ => err!(Runtime, "invalid assignment target"),
        }
    }

    /// `obj[idx] <- val`. Vectors, factors and lists follow
    /// `r2_types::replace_elems`; a matrix replaces within its cells; a
    /// data frame replaces (or adds) whole columns.
    pub(crate) fn assign_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::Matrix(m) => {
                let cells = RVal::Numeric(m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect::<Vec<Real>>().into(), Attrs::default());
                let RVal::Numeric(new, _) = replace_elems(&cells, idx, val)? else {
                    return err!(Runtime, "a numeric matrix can only take numbers");
                };
                if new.len() != m.data.len() { return err!(Index, "subscript out of bounds"); }
                m.data = new.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
                Ok(())
            }
            RVal::DataFrame(df) => {
                let names: Vec<Arc<str>> = df.columns.iter().map(|(n, _)| n.clone()).collect();
                let (pos, new_names) = assign_positions(idx, names.len(), Some(&names))?;
                let nrow = df.nrow();
                for (k, &p) in pos.iter().enumerate() {
                    let col = match val {
                        RVal::DataFrame(src) => src.columns[k % src.columns.len().max(1)].1.clone(),
                        RVal::List(items) => items[k % items.len().max(1)].1.clone(),
                        other => recycle(other, nrow),
                    };
                    if p < df.columns.len() { df.columns[p].1 = col; }
                    else if let Some((_, nm)) = new_names.iter().find(|(q, _)| *q == p) { df.columns.push((nm.clone(), col)); }
                    else { df.columns.push((Arc::from(format!("V{}", p + 1).as_str()), col)); }
                }
                Ok(())
            }
            _ => { *obj = replace_elems(obj, idx, val)?; Ok(()) }
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
        } else if let RVal::DataFrame(df) = obj {
            // `df[i, j] <- v`: each selected column takes its share of `v`
            // (column by column, recycled), rows as in `x[i] <- v`.
            let nrow = df.nrow();
            let names: Vec<Arc<str>> = df.columns.iter().map(|(n, _)| n.clone()).collect();
            let (cols, new_cols) = match col {
                Some(c) => assign_positions(c, names.len(), Some(&names))?,
                None => ((0..names.len()).collect(), Vec::new()),
            };
            let rows: Vec<usize> = match row {
                Some(r) => assign_positions(r, nrow, None)?.0,
                None => (0..nrow).collect(),
            };
            let row_idx = RVal::Numeric(rows.iter().map(|r| Some((*r + 1) as f64)).collect::<Vec<Real>>().into(), Attrs::default());
            let vlen = rval_length(val);
            if vlen == 0 { return err!(Runtime, "replacement has length zero"); }
            for (k, &p) in cols.iter().enumerate() {
                let share: Vec<usize> = (0..rows.len()).map(|r| (k * rows.len() + r) % vlen).collect();
                let chunk = match val {
                    RVal::DataFrame(src) => src.columns[k % src.columns.len().max(1)].1.clone(),
                    other => take(other, &share),
                };
                let slot = if p < df.columns.len() { p } else {
                    let nm = new_cols.iter().find(|(q, _)| *q == p).map(|(_, n)| n.clone())
                        .unwrap_or_else(|| Arc::from(format!("V{}", p + 1).as_str()));
                    df.columns.push((nm, RVal::Logical(vec![None; nrow].into(), Attrs::default())));
                    df.columns.len() - 1
                };
                df.columns[slot].1 = replace_elems(&df.columns[slot].1, &row_idx, &chunk)?;
            }
            Ok(())
        } else {
            err!(Runtime, "two-subscript assignment not supported for {}", obj.type_name())
        }
    }

    /// `obj[[idx]] <- val`: one element. On a list or data frame a name
    /// selects (or adds) the element and `NULL` removes it.
    pub(crate) fn assign_dbl_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        match (&*obj, idx) {
            (RVal::List(_) | RVal::DataFrame(_), RVal::Character(k, _)) => {
                let Some(field) = k.first().cloned().flatten() else { return err!(Runtime, "[[ ]] with a missing name"); };
                self.assign_dollar(obj, &field, val)
            }
            (RVal::List(items), _) => {
                let i = self.scalar_f64(idx)?.unwrap_or(1.0) as usize;
                if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                let mut items = items.clone();
                if matches!(val, RVal::Null) {
                    if i <= items.len() { items.remove(i - 1); }
                } else {
                    while items.len() < i { items.push((None, RVal::Null)); }
                    items[i - 1].1 = val.clone();
                }
                *obj = RVal::List(items);
                Ok(())
            }
            (RVal::DataFrame(df), _) => {
                let i = self.scalar_f64(idx)?.unwrap_or(1.0) as usize;
                match df.columns.get(i.wrapping_sub(1)) {
                    Some((name, _)) => { let name = name.clone(); self.assign_dollar(obj, &name, val) }
                    None => err!(Index, "subscript out of bounds"),
                }
            }
            _ => {
                if rval_length(val) != 1 { return err!(Runtime, "more elements supplied than there are to replace"); }
                self.assign_index(obj, idx, val)
            }
        }
    }

    pub(crate) fn assign_dollar(&mut self, obj: &mut RVal, field: &str, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::DataFrame(df) => {
                // Replace or add a column (a scalar recycles to every row);
                // `NULL` removes it.
                let pos = df.columns.iter().position(|(n, _)| n.as_ref() == field);
                if matches!(val, RVal::Null) {
                    if let Some(p) = pos { df.columns.remove(p); }
                    return Ok(());
                }
                let col = if df.columns.is_empty() { val.clone() } else { recycle(val, df.nrow()) };
                match pos {
                    Some(p) => df.columns[p].1 = col,
                    None => df.columns.push((Arc::from(field), col)),
                }
                Ok(())
            }
            RVal::List(items) => {
                let pos = items.iter().position(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some(field));
                match (pos, val) {
                    (Some(p), RVal::Null) => { items.remove(p); }
                    (None, RVal::Null) => {}
                    (Some(p), _) => items[p].1 = val.clone(),
                    (None, _) => items.push((Some(Arc::from(field)), val.clone())),
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

/// A length-1 atomic value repeated to `n` elements (R recycles a scalar
/// assigned to a data-frame column); anything else unchanged.
fn recycle(v: &RVal, n: usize) -> RVal {
    if n > 1 && rval_length(v) == 1 && matches!(v, RVal::Numeric(..) | RVal::Integer(..) | RVal::Logical(..) | RVal::Character(..) | RVal::Factor(_)) {
        take(v, &vec![0; n])
    } else { v.clone() }
}
