//! Scalar-chain arithmetic fusion: collapse `v*2+1`-style chains into one pass.

#![allow(clippy::all)]
use r2_types::*;
use crate::Engine;
use crate::err;

impl Engine {
    pub(crate) fn try_fuse_scalar_chain(
        &mut self, op: BinOp, lhs: &Expr, rhs: &Expr, env: &EnvRef,
    ) -> Result<Option<RVal>, R2Err> {
        fn lit(e: &Expr) -> Option<f64> {
            match e { Expr::NumLit(n) => Some(*n), Expr::IntLit(i) => Some(*i as f64), _ => None }
        }
        fn is_arith(op: BinOp) -> bool {
            matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Mod)
        }
        // Flatten a left-leaning (Symbol OP lit) OP lit … chain.
        // Returns (base Symbol expr, ops in apply order).
        fn flatten(e: &Expr) -> Option<(&Expr, Vec<(BinOp, f64)>)> {
            if let Expr::Binary { op, lhs, rhs } = e {
                if is_arith(*op) {
                    if let Some(s) = lit(rhs) {
                        if matches!(lhs.as_ref(), Expr::Symbol(_)) {
                            return Some((lhs, vec![(*op, s)]));
                        }
                        if let Some((base, mut ops)) = flatten(lhs) {
                            ops.push((*op, s));
                            return Some((base, ops));
                        }
                    }
                }
            }
            None
        }

        let s_outer = match lit(rhs) { Some(s) => s, None => return Ok(None) };
        let (base_expr, mut ops) = if matches!(lhs, Expr::Symbol(_)) {
            (lhs, Vec::new())
        } else if let Some((b, ops)) = flatten(lhs) {
            (b, ops)
        } else {
            return Ok(None);
        };
        ops.push((op, s_outer));
        if ops.len() < 2 { return Ok(None); } // single op → existing fast path

        // Base is a Symbol → eval is a side-effect-free lookup, so bailing
        // out after this point is safe (the fallback re-looks-up cheaply).
        let base = self.eval_in(base_expr, env)?;
        let a = match &base { RVal::Numeric(a, _) => a, _ => return Ok(None) };
        if a.len() < 64 { return Ok(None); }
        let col = a.columnar();
        if !col.is_dense() { return Ok(None); } // NA present → normal path

        #[inline]
        fn step(op: BinOp, a: f64, b: f64) -> f64 {
            match op {
                BinOp::Add => a + b, BinOp::Sub => a - b, BinOp::Mul => a * b,
                // R: IEEE division (no zero error); %% is floored modulo.
                BinOp::Div => a / b, BinOp::Pow => a.powf(b), BinOp::Mod => a - (a / b).floor() * b,
                _ => a,
            }
        }
        let src = col.values();
        let out: Vec<f64> = src.iter().map(|&x| {
            let mut acc = x;
            for (o, s) in &ops { acc = step(*o, acc, *s); }
            acc
        }).collect();
        Ok(Some(RVal::Numeric(Reals::from_columnar(r2_arrow::ColumnarF64::from_vec(out)), Attrs::default())))
    }

    // ── Subscript assignment helpers ─────────────────────────────────

}
