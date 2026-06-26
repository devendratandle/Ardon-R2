//! Subscripting / extraction methods — `[`, `[[`, `$`, matrix and
//! data.frame indexing, positive/negative subscript resolution. Extracted
//! from lib.rs as a second `impl Engine` block.

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::Engine;
use crate::err;

impl Engine {
    pub(crate) fn index_obj(&self, obj: &RVal, idx: &[Option<RVal>]) -> Result<RVal, R2Err> {
        if idx.len()==1 {
            if let Some(i) = &idx[0] {
                // 1D indexing of a Matrix → column-major linear access, returning a Numeric vector
                if let RVal::Matrix(m) = obj {
                    let pos = self.as_reals(i)?;
                    let total = m.nrow * m.ncol;
                    let mut out = Vec::with_capacity(pos.len());
                    for p in &pos {
                        match p {
                            Some(k) => {
                                let k = *k as usize;
                                if k == 0 || k > total {
                                    if self.mode == ErrorMode::Strict { return err!(Index, "index {} out of bounds (matrix has {} elements)", k, total); }
                                    out.push(None);
                                } else {
                                    let v = m.data[k - 1];
                                    out.push(if v.is_nan() { None } else { Some(v) });
                                }
                            }
                            None => out.push(None),
                        }
                    }
                    return Ok(RVal::Numeric(out.into(), Attrs::default()));
                }
                // Single-bracket on a data.frame selects COLUMNS (by name,
                // position, or logical mask) and returns a data.frame — R's
                // `df[cols]` semantics, distinct from `df[[col]]`.
                if let RVal::DataFrame(df) = obj {
                    return self.df_select_cols(df, i);
                }
                return self.index_1d(obj, i);
            }
        }
        if idx.len()==2 {
            if let RVal::DataFrame(df) = obj { return self.index_df(df, &idx[0], &idx[1]); }
            if let RVal::Matrix(m) = obj { return self.index_matrix(m, &idx[0], &idx[1]); }
        }
        err!(Runtime, "invalid indexing")
    }
    /// `df[cols]` — select columns by name / position / logical mask,
    /// returning a data.frame.
    fn df_select_cols(&self, df: &DataFrame, idx: &RVal) -> Result<RVal, R2Err> {
        let cols: Vec<(Arc<str>, RVal)> = match idx {
            RVal::Character(names, _) => {
                let mut out = Vec::new();
                for nm in names.iter().flatten() {
                    match df.get_col(nm.as_ref()) {
                        Some(v) => out.push((nm.clone(), v.clone())),
                        None => return err!(Runtime, "undefined columns selected: '{}'", nm),
                    }
                }
                out
            }
            RVal::Logical(mask, _) => df.columns.iter().enumerate()
                .filter(|(i, _)| mask.get(*i).and_then(|m| *m) == Some(true))
                .map(|(_, c)| c.clone()).collect(),
            _ => {
                let pos = self.as_reals(idx)?;
                pos.iter().filter_map(|p| p.and_then(|v| {
                    let i = v as usize;
                    if i >= 1 { df.columns.get(i - 1).cloned() } else { None }
                })).collect()
            }
        };
        Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: df.row_names.clone() }))
    }
    /// `x[[i]]` — extract a single list/data.frame element (by name or
    /// position) or a single atomic element.
    pub(crate) fn dbl_index(&self, obj: &RVal, idx: &RVal) -> Result<RVal, R2Err> {
        match obj {
            RVal::List(items) => {
                if let RVal::Character(cv, _) = idx {
                    if let Some(Some(name)) = cv.first() {
                        for (n, v) in items { if n.as_deref() == Some(name.as_ref()) { return Ok(v.clone()); } }
                        return err!(Runtime, "subscript out of bounds: '{}'", name);
                    }
                }
                let i = self.as_reals(idx)?.first().copied().flatten().unwrap_or(0.0) as usize;
                items.get(i.wrapping_sub(1)).map(|(_, v)| v.clone())
                    .ok_or_else(|| R2Err { msg: "subscript out of bounds".into(), kind: ErrKind::Runtime })
            }
            RVal::DataFrame(df) => {
                if let RVal::Character(cv, _) = idx {
                    if let Some(Some(name)) = cv.first() {
                        return df.get_col(name.as_ref()).cloned()
                            .ok_or_else(|| R2Err { msg: format!("undefined column '{}'", name), kind: ErrKind::Runtime });
                    }
                }
                let i = self.as_reals(idx)?.first().copied().flatten().unwrap_or(0.0) as usize;
                df.columns.get(i.wrapping_sub(1)).map(|(_, v)| v.clone())
                    .ok_or_else(|| R2Err { msg: "subscript out of bounds".into(), kind: ErrKind::Runtime })
            }
            _ => {
                let i = self.as_reals(idx)?.first().copied().flatten().unwrap_or(0.0);
                self.index_obj(obj, &[Some(rnum(i))])
            }
        }
    }
    /// Resolve a numeric subscript to 0-based kept positions. Supports R's
    /// NEGATIVE (exclusion) indexing: all-negative → keep everything except
    /// those positions; otherwise positive 1-based selection.
    fn resolve_subscript(&self, idx: &RVal, n: usize) -> Result<Vec<usize>, R2Err> {
        let pos = self.as_reals(idx)?;
        let any_neg = pos.iter().any(|p| matches!(p, Some(v) if *v < 0.0));
        Ok(if any_neg {
            let excl: std::collections::HashSet<usize> = pos.iter()
                .filter_map(|p| p.and_then(|v| { let k = (-v) as usize; if k >= 1 && k <= n { Some(k - 1) } else { None } }))
                .collect();
            (0..n).filter(|i| !excl.contains(i)).collect()
        } else {
            pos.iter().filter_map(|p| p.and_then(|v| { let i = v as usize; if i >= 1 && i <= n { Some(i - 1) } else { None } })).collect()
        })
    }

    fn index_matrix(&self, m: &Matrix, row: &Option<RVal>, col: &Option<RVal>) -> Result<RVal, R2Err> {
        // Resolve rows
        let keep_rows: Vec<usize> = match row {
            None => (0..m.nrow).collect(),
            Some(RVal::Logical(mask, _)) => mask.iter().enumerate().filter_map(|(i, b)| if *b == Some(true) { Some(i) } else { None }).collect(),
            Some(idx) => self.resolve_subscript(idx, m.nrow)?,
        };
        // Resolve columns
        let keep_cols: Vec<usize> = match col {
            None => (0..m.ncol).collect(),
            Some(RVal::Logical(mask, _)) => mask.iter().enumerate().filter_map(|(j, b)| if *b == Some(true) { Some(j) } else { None }).collect(),
            Some(idx) => self.resolve_subscript(idx, m.ncol)?,
        };
        // Single element → scalar Numeric
        if keep_rows.len() == 1 && keep_cols.len() == 1 {
            let v = m.data[keep_cols[0] * m.nrow + keep_rows[0]];
            return Ok(RVal::Numeric(vec![if v.is_nan() { None } else { Some(v) }].into(), Attrs::default()));
        }
        // Single column or single row → drop to vector (R's default `drop=TRUE`)
        if keep_cols.len() == 1 {
            let j = keep_cols[0];
            let out: Vec<Real> = keep_rows.iter().map(|&i| {
                let v = m.data[j * m.nrow + i];
                if v.is_nan() { None } else { Some(v) }
            }).collect();
            return Ok(RVal::Numeric(out.into(), Attrs::default()));
        }
        if keep_rows.len() == 1 {
            let i = keep_rows[0];
            let out: Vec<Real> = keep_cols.iter().map(|&j| {
                let v = m.data[j * m.nrow + i];
                if v.is_nan() { None } else { Some(v) }
            }).collect();
            return Ok(RVal::Numeric(out.into(), Attrs::default()));
        }
        // General submatrix → Matrix (column-major)
        let mut data = Vec::with_capacity(keep_rows.len() * keep_cols.len());
        for &j in &keep_cols {
            for &i in &keep_rows {
                data.push(m.data[j * m.nrow + i]);
            }
        }
        let mut out = Matrix::new(data, keep_rows.len(), keep_cols.len());
        if let Some(cn) = &m.col_names {
            out.col_names = Some(keep_cols.iter().map(|&j| cn[j].clone()).collect());
        }
        if let Some(rn) = &m.row_names {
            out.row_names = Some(keep_rows.iter().map(|&i| rn[i].clone()).collect());
        }
        Ok(RVal::Matrix(out))
    }
    fn index_1d(&self, obj: &RVal, idx: &RVal) -> Result<RVal, R2Err> { match idx { RVal::Logical(mask,_) => self.logical_sub(obj,mask), RVal::Factor(f) => { let pos: Vec<Real> = f.codes.iter().map(|&c| c.map(|i| i as f64 + 1.0)).collect(); self.pos_sub(obj, &pos) } _ => { let pos = self.as_reals(idx)?; if pos.iter().any(|p| matches!(p, Some(v) if *v < 0.0)) { let keep = self.resolve_subscript(idx, r2_types::rval_length(obj))?; let pos1: Vec<Real> = keep.iter().map(|&i| Some((i+1) as f64)).collect(); self.pos_sub(obj, &pos1) } else { self.pos_sub(obj,&pos) } } } }
    fn pos_sub(&self, obj: &RVal, pos: &[Real]) -> Result<RVal, R2Err> { match obj { RVal::Numeric(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { if self.mode==ErrorMode::Strict { return err!(Index,"index {} out of bounds (len {})",i,v.len()); } r.push(None); } else { r.push(v[i-1]); } } None => r.push(None), } } Ok(RVal::Numeric(r.into(), Attrs::default())) } RVal::Character(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { r.push(None); } else { r.push(v[i-1].clone()); } } None => r.push(None), } } Ok(RVal::Character(r, Attrs::default())) } RVal::Integer(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { r.push(None); } else { r.push(v[i-1]); } } None => r.push(None), } } Ok(RVal::Integer(r.into(), Attrs::default())) } RVal::Logical(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { r.push(None); } else { r.push(v[i-1]); } } None => r.push(None), } } Ok(RVal::Logical(r.into(), Attrs::default())) } RVal::Factor(fc) => { let mut codes = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>fc.codes.len() { codes.push(None); } else { codes.push(fc.codes[i-1]); } } None => codes.push(None), } } let mut nf = fc.clone(); nf.codes = codes; Ok(RVal::Factor(nf)) } RVal::Single(..)|RVal::Raw(..)|RVal::List(..)|RVal::DataFrame(..)|RVal::Matrix(..)|RVal::Tensor(..)|RVal::Formula(..)|RVal::Closure(..)|RVal::BuiltinFn(..)|RVal::Lang(..)|RVal::TypeDef(..)|RVal::TypeInstance(..)|RVal::Null|RVal::Env(..) => err!(Index,"cannot subset {}",obj.type_name()), } }
    fn logical_sub(&self, obj: &RVal, mask: &[Logical]) -> Result<RVal, R2Err> { match obj { RVal::Numeric(v,_) => Ok(RVal::Numeric(v.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(val,m)| if *m==Some(true) { Some(*val) } else { None }).collect(), Attrs::default())), RVal::Integer(v,_) => Ok(RVal::Integer(v.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(val,m)| if *m==Some(true) { Some(*val) } else { None }).collect(), Attrs::default())), RVal::Character(v,_) => Ok(RVal::Character(v.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(val,m)| if *m==Some(true) { Some(val.clone()) } else { None }).collect(), Attrs::default())), RVal::Logical(v,_) => Ok(RVal::Logical(v.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(val,m)| if *m==Some(true) { Some(*val) } else { None }).collect(), Attrs::default())), RVal::Factor(fc) => { let mut nf = fc.clone(); nf.codes = fc.codes.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(c,m)| if *m==Some(true) { Some(*c) } else { None }).collect(); Ok(RVal::Factor(nf)) }, RVal::Single(..)|RVal::Raw(..)|RVal::List(..)|RVal::DataFrame(..)|RVal::Matrix(..)|RVal::Tensor(..)|RVal::Formula(..)|RVal::Closure(..)|RVal::BuiltinFn(..)|RVal::Lang(..)|RVal::TypeDef(..)|RVal::TypeInstance(..)|RVal::Null|RVal::Env(..) => err!(Index,"logical subset not impl for {}",obj.type_name()) } }
    fn index_df(&self, df: &DataFrame, row: &Option<RVal>, col: &Option<RVal>) -> Result<RVal, R2Err> {
        // Determine which rows to keep
        let nrow = df.nrow();
        let keep_rows: Vec<usize> = match row {
            None => (0..nrow).collect(), // all rows
            Some(RVal::Logical(mask, _)) => {
                mask.iter().enumerate().filter_map(|(i, m)| if *m == Some(true) { Some(i) } else { None }).collect()
            }
            Some(idx) => self.resolve_subscript(idx, nrow)?,
        };

        // Determine which columns to keep
        let ncol = df.ncol();
        let keep_cols: Vec<usize> = match col {
            None => (0..ncol).collect(), // all columns
            Some(RVal::Character(names, _)) => {
                names.iter().filter_map(|n| n.as_ref().and_then(|name| {
                    df.columns.iter().position(|(cn, _)| cn.as_ref() == name.as_ref())
                })).collect()
            }
            Some(idx) => self.resolve_subscript(idx, ncol)?,
        };

        // If single column selected, return as vector
        if keep_cols.len() == 1 && row.is_none() {
            return Ok(df.columns[keep_cols[0]].1.clone());
        }

        // Build new DataFrame
        let new_cols: Vec<(Arc<str>, RVal)> = keep_cols.iter().map(|&ci| {
            let (name, col) = &df.columns[ci];
            let new_col = self.subset_col_by_rows(col, &keep_rows);
            (name.clone(), new_col)
        }).collect();

        Ok(RVal::DataFrame(DataFrame { columns: new_cols, row_names: None }))
    }

    fn subset_col_by_rows(&self, col: &RVal, rows: &[usize]) -> RVal {
        match col {
            RVal::Numeric(v, _) => RVal::Numeric(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(rows.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            // Factor columns must be row-filtered too (else `iris[mask,]`
            // leaves Species at full length and breaks `~ Species` etc.).
            RVal::Factor(f) => {
                let mut nf = f.clone();
                nf.codes = rows.iter().map(|&r| f.codes.get(r).copied().unwrap_or(None)).collect();
                RVal::Factor(nf)
            }
            // Non-atomic / non-column types aren't normal df columns; clone
            // as-is. Listed explicitly (no `_`) so a NEW atomic vector type
            // is a compile error here — the old silent `_ => clone()` is what
            // left factor columns un-filtered (length 100 vs 150).
            RVal::Single(..)|RVal::Raw(..)|RVal::List(..)|RVal::DataFrame(..)|RVal::Matrix(..)
            |RVal::Tensor(..)|RVal::Formula(..)|RVal::Closure(..)|RVal::BuiltinFn(..)|RVal::Lang(..)
            |RVal::TypeDef(..)|RVal::TypeInstance(..)|RVal::Null|RVal::Env(..) => col.clone(),
        }
    }
    pub(crate) fn dollar(&self, obj: &RVal, field: &str) -> Result<RVal, R2Err> { match obj { RVal::DataFrame(df) => df.get_col(field).cloned().ok_or(R2Err{msg:format!("column '{}' not found",field),kind:ErrKind::Runtime}), RVal::List(items) => { for (n,v) in items { if n.as_ref().map(|s| s.as_ref())==Some(field) { return Ok(v.clone()); } } err!(Runtime,"'{}' not in list",field) } RVal::TypeInstance(inst) => inst.fields.get(field).cloned().ok_or(R2Err{msg:format!("field '{}' not found",field),kind:ErrKind::Runtime}), _ => err!(Runtime,"$ applied to {}",obj.type_name()), } }
    // Phase R.1 step 2: coercion methods extracted to RVal methods in
    // r2-types. Engine wrappers retained so existing call sites
    // (`e.as_reals(arg)`, `e.scalar_f64(arg)`) keep working unchanged.
    // New code can call `arg.as_reals()` / `arg.scalar_f64()` directly,
    // bypassing the engine — required by domain crates that don't see
    // the `Engine` type (r2-stats, r2-ml).
}
