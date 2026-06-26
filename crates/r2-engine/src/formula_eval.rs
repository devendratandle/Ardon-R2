//! Formula resolution — turn `y ~ x` terms into data.frame columns for models.

#![allow(clippy::all)]
use std::sync::Arc;
use r2_types::*;
use std::collections::HashMap;
use crate::Engine;
use crate::formula::fmt_expr;

impl Engine {
    pub(crate) fn resolve_formula_term(&mut self, expr: &Expr, df: &DataFrame, env: &EnvRef) -> Result<RVal, R2Err> {
        match expr {
            Expr::Symbol(name) => {
                // Look up as column name first — preserve the name!
                if let Some(col) = df.get_col(name) {
                    Ok(RVal::List(vec![(Some(name.clone()), col.clone())]))
                } else {
                    self.eval_in(expr, env)
                }
            }
            Expr::Binary { op: BinOp::Add, lhs, rhs } => {
                let l = self.resolve_formula_term(lhs, df, env)?;
                let r = self.resolve_formula_term(rhs, df, env)?;
                let mut cols = Vec::new();
                match l {
                    RVal::List(items) => cols.extend(items),
                    other => cols.push((None, other)),
                }
                match r {
                    RVal::List(items) => cols.extend(items),
                    other => cols.push((None, other)),
                }
                Ok(RVal::List(cols))
            }
            Expr::NullLit => Ok(RVal::Null),
            // Phase S.1 — data-scope fix. For any non-trivial sub-expression
            // (Call, Binary*, Index, etc.) the bare names inside should
            // resolve against the data.frame columns FIRST, then the
            // enclosing env. Real R does this via the formula's environment;
            // we approximate by pushing all df columns into a temporary
            // scope frame for the duration of the eval. Fixes:
            //   lm(Sepal.Width ~ factor(Species), data = iris)
            //   lm(y ~ I(x^2) + log(z), data = df)
            _ => {
                let mut frame: HashMap<Arc<str>, RVal> = HashMap::new();
                for (n, v) in &df.columns {
                    frame.insert(n.clone(), v.clone());
                }
                self.local_scopes.push(frame);
                let result = self.eval_in(expr, env);
                self.local_scopes.pop();
                result
            }
        }
    }

    // ── Phase 2 — structured formula frame ───────────────────────────
    //
    // Splits a formula into one-or-more response columns (handling
    // `cbind(a, b, ...)` on the LHS) and one-or-more grouping terms
    // (handling `a + b` on the RHS), resolving each name against the
    // data frame. This is the "model.frame" input-adapter: it only
    // assembles named columns — it never runs any statistics. Returns
    // (responses, groups) as (name, column) pairs.
    pub(crate) fn formula_frame(
        &mut self, lhs: &Expr, rhs: &Expr, df: &DataFrame, env: &EnvRef,
    ) -> Result<(Vec<(Arc<str>, RVal)>, Vec<(Arc<str>, RVal)>), R2Err> {
        let responses = self.resolve_response_terms(lhs, df, env)?;
        let groups = self.resolve_additive_terms(rhs, df, env)?;
        Ok((responses, groups))
    }

    /// LHS responses: `cbind(y1, y2, ...)` → one entry per argument;
    /// anything else → a single response.
    pub(crate) fn resolve_response_terms(
        &mut self, lhs: &Expr, df: &DataFrame, env: &EnvRef,
    ) -> Result<Vec<(Arc<str>, RVal)>, R2Err> {
        if let Expr::Call { func, args } = lhs {
            if matches!(func.as_ref(), Expr::Symbol(s) if s.as_ref() == "cbind") {
                let mut out = Vec::with_capacity(args.len());
                for a in args {
                    out.push(self.resolve_single_term(&a.value, df, env)?);
                }
                return Ok(out);
            }
        }
        Ok(vec![self.resolve_single_term(lhs, df, env)?])
    }

    /// RHS additive terms: split on `+` recursively into individual
    /// grouping terms.
    pub(crate) fn resolve_additive_terms(
        &mut self, rhs: &Expr, df: &DataFrame, env: &EnvRef,
    ) -> Result<Vec<(Arc<str>, RVal)>, R2Err> {
        if let Expr::Binary { op: BinOp::Add, lhs, rhs } = rhs {
            let mut l = self.resolve_additive_terms(lhs, df, env)?;
            let mut r = self.resolve_additive_terms(rhs, df, env)?;
            l.append(&mut r);
            return Ok(l);
        }
        Ok(vec![self.resolve_single_term(rhs, df, env)?])
    }

    /// Resolve one formula term (a bare column name or an expression
    /// like `factor(x)`) to a (display-name, column) pair. Bare symbols
    /// keep their column name; expressions are deparsed for the name.
    pub(crate) fn resolve_single_term(
        &mut self, expr: &Expr, df: &DataFrame, env: &EnvRef,
    ) -> Result<(Arc<str>, RVal), R2Err> {
        match self.resolve_formula_term(expr, df, env)? {
            RVal::List(mut items) if items.len() == 1 => {
                let (n, col) = items.remove(0);
                let name = n.unwrap_or_else(|| Arc::from(fmt_expr(expr).as_str()));
                Ok((name, col))
            }
            other => Ok((Arc::from(fmt_expr(expr).as_str()), other)),
        }
    }

    // ── Phase 1 fusion — vector⊗scalar arithmetic chains ─────────────
    //
    // `v*2+1`, `(v+1)*2`, `v*a+b+c` … evaluate as one allocation + pass
    // per operator (each binary op materialises an intermediate vector).
    // This collapses a left-leaning chain of (vector OP literal) ops into
    // a SINGLE pass over the base vector. Returns Ok(None) when the shape
    // doesn't qualify (caller falls back to the normal per-op path).
    //
    // Safety/correctness constraints (so falling back can't double-run
    // side effects, and NA semantics are preserved):
    //   * the base operand must be a Symbol (a side-effect-free lookup),
    //   * every other operand must be a numeric literal,
    //   * the base must be a dense (no-NA) numeric vector of length ≥ 64,
    //   * ≥ 2 ops (a single op already has a fast columnar path).
}
