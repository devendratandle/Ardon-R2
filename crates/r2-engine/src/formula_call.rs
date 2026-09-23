//! The formula interface of the model and summary functions:
//! `lm(y ~ x, data = df)`, `aggregate(y ~ g, df, FUN)`, `boxplot(y ~ g,
//! df)`, `rf(y ~ ., df)`. The call is intercepted before its arguments
//! are evaluated so the formula's bare names resolve as the data frame's
//! columns; the functions themselves then receive plain values.

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use crate::Engine;
use crate::err;
use crate::formula::{fmt_expr, split_error_term, split_random_effects};

impl Engine {
    /// `Ok(None)` when the first argument is not a formula or no data frame
    /// is given — the formula is then evaluated normally against the
    /// calling scope.
    pub(crate) fn formula_call(&mut self, fname: &str, func: &Expr, args: &[CallArg],
                               env: &EnvRef) -> Result<Option<RVal>, R2Err> {
        let Some(first_arg) = args.first() else { return Ok(None) };
        let Expr::Binary { op: BinOp::Tilde, lhs, rhs } = &first_arg.value else { return Ok(None) };
        // Find the data frame: `data=` by name, else R's positional
        // convention — the first UNNAMED argument after the formula (so
        // `lm(y ~ x, df)` works like `lm(y ~ x, data=df)`).
        let data_arg = args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some("data"))
            .or_else(|| args.iter().skip(1).find(|a| a.name.is_none()));
        let Some(data_a) = data_arg else { return Ok(None) };
        let data_val = self.eval_in(&data_a.value, env)?;
        let RVal::DataFrame(ref df) = data_val else { return Ok(None) };

        if fname == "aggregate" {
            return self.aggregate_formula(lhs, rhs, df, args, env).map(Some);
        }
        if fname == "boxplot" {
            if let Some(v) = self.boxplot_formula(lhs, rhs, df, args, env)? {
                return Ok(Some(v));
            }
        }
        // A dot (.) on the RHS means "all other columns".
        let v = if matches!(rhs.as_ref(), Expr::Symbol(s) if s.as_ref() == ".") {
            self.dot_formula_call(fname, func, lhs, df, args, env)?
        } else {
            self.named_formula_call(func, lhs, rhs, df, args, env)?
        };
        Ok(Some(v))
    }

    /// aggregate(cbind(y1, y2) ~ g1 + g2, data = df, FUN = ...) — any
    /// number of response columns and grouping factors. The formula is
    /// purely an input adapter (formula_frame); the split-apply math (FUN
    /// per group) is the same as the non-formula call.
    fn aggregate_formula(&mut self, lhs: &Expr, rhs: &Expr, df: &DataFrame, args: &[CallArg],
                         env: &EnvRef) -> Result<RVal, R2Err> {
        let (responses, groups) = self.formula_frame(lhs, rhs, df, env)?;
        if groups.is_empty() {
            return Err(R2Err { msg: "aggregate(): formula needs at least one grouping factor on the RHS".into(), kind: ErrKind::Runtime });
        }
        if responses.is_empty() {
            return Err(R2Err { msg: "aggregate(): formula needs at least one response on the LHS".into(), kind: ErrKind::Runtime });
        }
        // Resolve FUN: named FUN=, else first positional arg after the
        // formula (skipping data=).
        let fun_expr = args.iter().find(|a| a.name.as_deref() == Some("FUN"))
            .or_else(|| args.iter().skip(1).find(|a| a.name.is_none()))
            .map(|a| &a.value);
        let f = match fun_expr {
            Some(e) => self.eval_in(e, env)?,
            None => return Err(R2Err { msg: "aggregate(): FUN is required".into(), kind: ErrKind::Runtime }),
        };
        // Element-wise labels for each grouping factor.
        let col_labels = |c: &RVal| -> Vec<String> {
            match c {
                RVal::Numeric(v, _) => v.iter().map(|x| x.map(|n| fmt_num(n)).unwrap_or_else(|| "NA".into())).collect(),
                RVal::Integer(v, _) => v.iter().map(|x| x.map(|n| n.to_string()).unwrap_or_else(|| "NA".into())).collect(),
                RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into())).collect(),
                RVal::Logical(v, _) => v.iter().map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect(),
                _ => Vec::new(),
            }
        };
        let group_labels: Vec<Vec<String>> = groups.iter().map(|(_, c)| col_labels(c)).collect();
        let nrow = group_labels.first().map(|v| v.len()).unwrap_or(0);
        // Distinct group combinations (composite key per row).
        let mut combos: Vec<Vec<String>> = Vec::new();
        let mut row_combo: Vec<usize> = Vec::with_capacity(nrow);
        for r in 0..nrow {
            let key: Vec<String> = group_labels.iter()
                .map(|g| g.get(r).cloned().unwrap_or_default()).collect();
            match combos.iter().position(|c| *c == key) {
                Some(p) => row_combo.push(p),
                None => { combos.push(key); row_combo.push(combos.len() - 1); }
            }
        }
        // Sort combos lexicographically by label tuple (R orders aggregate
        // output by grouping levels).
        let mut order: Vec<usize> = (0..combos.len()).collect();
        order.sort_by(|&i, &j| combos[i].cmp(&combos[j]));
        let mut rows_per_combo: Vec<Vec<usize>> = vec![Vec::new(); combos.len()];
        for r in 0..nrow { rows_per_combo[row_combo[r]].push(r); }
        // Build output: one column per grouping factor, then one column per
        // response (real source names).
        let mut out_cols: Vec<(Arc<str>, RVal)> = Vec::new();
        for (gi, (gname, _)) in groups.iter().enumerate() {
            let col: Vec<Character> = order.iter()
                .map(|&ci| Some(Arc::from(combos[ci][gi].as_str()))).collect();
            out_cols.push((gname.clone(), RVal::Character(col, Attrs::default())));
        }
        for (rname, rcol) in &responses {
            let vals = self.as_reals(rcol)?;
            let mut agg: Vec<Real> = Vec::with_capacity(order.len());
            for &ci in &order {
                let gv: Vec<Real> = rows_per_combo[ci].iter()
                    .map(|&r| vals.get(r).copied().unwrap_or(None)).collect();
                let res = self.call_fn(&f, &[EvalArg { name: None, value: RVal::Numeric(gv.into(), Attrs::default()) }], env)?;
                agg.push(res.scalar_f64().unwrap_or(None));
            }
            out_cols.push((rname.clone(), RVal::Numeric(agg.into(), Attrs::default())));
        }
        Ok(RVal::DataFrame(DataFrame { columns: out_cols, row_names: None }))
    }

    /// boxplot(y ~ g, data = df): split the response by the grouping column
    /// and hand one named vector per group to the graphics boxplot (its
    /// multi-group form). `None` unless there is exactly one response and
    /// one group; other shapes take the general formula path.
    fn boxplot_formula(&mut self, lhs: &Expr, rhs: &Expr, df: &DataFrame, args: &[CallArg],
                       env: &EnvRef) -> Result<Option<RVal>, R2Err> {
        let (responses, groups) = self.formula_frame(lhs, rhs, df, env)?;
        if responses.len() != 1 || groups.len() != 1 { return Ok(None); }
        let y = self.as_reals(&responses[0].1)?;
        let glabels: Vec<String> = match &groups[0].1 {
            RVal::Factor(f) => f.codes.iter().map(|c|
                c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string())).unwrap_or_default()).collect(),
            RVal::Character(v, _) => v.iter().map(|x|
                x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
            other => self.as_reals(other)?.iter().map(|x|
                x.map(fmt_num).unwrap_or_default()).collect(),
        };
        let mut levels: Vec<String> = Vec::new();
        for l in &glabels { if !levels.contains(l) { levels.push(l.clone()); } }
        levels.sort();
        let mut ea: Vec<EvalArg> = Vec::new();
        for lvl in &levels {
            let vals: Vec<Real> = y.iter().zip(glabels.iter())
                .filter(|(_, gl)| *gl == lvl).map(|(yi, _)| *yi).collect();
            ea.push(EvalArg { name: Some(Arc::from(lvl.as_str())), value: RVal::Numeric(vals.into(), Attrs::default()) });
        }
        // Carry styling args (main/col/…), skip formula + data.
        for arg in args.iter().skip(1) {
            if arg.name.as_deref() == Some("data") { continue; }
            if arg.name.is_some() {
                ea.push(EvalArg { name: arg.name.clone(), value: self.eval_in(&arg.value, env)? });
            }
        }
        self.call_fn(&RVal::BuiltinFn(Arc::from("boxplot")), &ea, env).map(Some)
    }

    /// `y ~ .`: y is the LHS column, the predictors every OTHER numeric
    /// column. lm/glm get it as a formula; the ML functions as (x, y).
    fn dot_formula_call(&mut self, fname: &str, func: &Expr, lhs: &Expr, df: &DataFrame,
                        args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let lhs_name = match lhs {
            Expr::Symbol(s) => s.clone(),
            _ => return err!(Runtime, "formula LHS must be a column name"),
        };
        let y_col = df.get_col(&lhs_name).ok_or(R2Err{msg:format!("column '{}' not found", lhs_name),kind:ErrKind::Runtime})?;

        // Build x matrix from all OTHER numeric columns
        let nrow = df.nrow();
        let mut x_data = Vec::new();
        let mut x_names = Vec::new();
        let mut ncol = 0;
        for (cn, cv) in &df.columns {
            if cn.as_ref() == lhs_name.as_ref() { continue; }
            if let Ok(vals) = self.as_reals(cv) {
                let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                if nums.len() == nrow { x_data.extend(nums); x_names.push(cn.clone()); ncol += 1; }
            }
        }
        let mut mat = Matrix::new(x_data, nrow, ncol);
        mat.col_names = Some(x_names.clone());
        let x_mat = RVal::Matrix(mat);

        let f = self.eval_in(func, env)?;
        if matches!(fname, "lm" | "glm") {
            let formula = RVal::List(vec![
                (Some(Arc::from("~lhs")), y_col.clone()),
                (Some(Arc::from("~rhs")), x_mat),
                (Some(Arc::from("~class")), rstr("formula")),
            ]);
            let mut ea = vec![EvalArg { name: None, value: formula }];
            for a in args.iter().skip(1) { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); }
            // Capture original call for `$call` field.
            ea.push(call_text_arg(func, args));
            return self.call_fn(&f, &ea, env);
        }
        // For ML functions: pass (x_matrix, y_vector, ...other args)
        let mut ea = vec![
            EvalArg { name: None, value: x_mat },
            EvalArg { name: None, value: y_col.clone() },
        ];
        for a in args.iter().skip(1) {
            if a.name.as_ref().map(|n| n.as_ref()) != Some("data") {
                ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
            }
        }
        self.call_fn(&f, &ea, env)
    }

    /// Named columns, resolved against the frame. Phase R.S.1 splits out
    /// any Error(...) stratum first so it is not treated as a predictor;
    /// Phase R.S.3 then splits out (1|group) random-effect terms, so
    /// lmer-style formulas like y ~ x + (1|subject) work cleanly.
    fn named_formula_call(&mut self, func: &Expr, lhs: &Expr, rhs: &Expr, df: &DataFrame,
                          args: &[CallArg], env: &EnvRef) -> Result<RVal, R2Err> {
        let (rhs_no_err, error_stratum_expr) = split_error_term(rhs);
        let (rhs_fixed, random_grouping_exprs) = split_random_effects(&rhs_no_err);
        let lhs_val = self.resolve_formula_term(lhs, df, env)?;
        let rhs_val = if matches!(rhs_fixed, Expr::NullLit) {
            RVal::Null
        } else {
            self.resolve_formula_term(&rhs_fixed, df, env)?
        };
        let mut formula_items = vec![
            (Some(Arc::from("~lhs")), lhs_val),
            (Some(Arc::from("~rhs")), rhs_val),
            (Some(Arc::from("~class")), rstr("formula")),
        ];
        if let Some(stratum_expr) = error_stratum_expr {
            let stratum_val = self.resolve_formula_term(&stratum_expr, df, env)?;
            formula_items.push((Some(Arc::from("~error")), stratum_val));
        }
        for group_expr in &random_grouping_exprs {
            let group_val = self.resolve_formula_term(group_expr, df, env)?;
            formula_items.push((Some(Arc::from("~random_intercept")), group_val));
        }
        let formula = RVal::List(formula_items);
        let f = self.eval_in(func, env)?;
        let mut ea = vec![EvalArg { name: None, value: formula }];
        for a in args.iter().skip(1) {
            ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
        }
        // Capture original call for `$call` field on the fitted-model
        // TypeInstance (lm/glm/aov use it).
        ea.push(call_text_arg(func, args));
        self.call_fn(&f, &ea, env)
    }
}

/// The call as written, deparsed, passed as `_call` so a fitted model can
/// report it in `$call`.
fn call_text_arg(func: &Expr, args: &[CallArg]) -> EvalArg {
    let call = Expr::Call { func: Box::new(func.clone()), args: args.to_vec() };
    EvalArg { name: Some(Arc::from("_call")), value: rstr(&fmt_expr(&call)) }
}
