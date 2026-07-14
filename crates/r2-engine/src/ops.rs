//! Engine operator methods — binary/unary operators, the `:` sequence,
//! and the label vector for string/factor comparison. Extracted from
//! lib.rs as a second `impl Engine` block (same crate, same exe).

use r2_types::*;
use crate::Engine;
use crate::err;

impl Engine {
    fn label_vec(&self, v: &RVal) -> Vec<Option<String>> {
        match v {
            RVal::Character(c, _) => c.iter().map(|x| x.as_ref().map(|s| s.to_string())).collect(),
            RVal::Factor(f) => f.codes.iter()
                .map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string()))).collect(),
            other => match self.as_reals(other) {
                Ok(rs) => (0..rs.len()).map(|i| rs[i].map(r2_types::fmt_num)).collect(),
                Err(_) => Vec::new(),
            },
        }
    }

    pub(crate) fn binary_op(&mut self, op: BinOp, lhs: &RVal, rhs: &RVal) -> Result<RVal, R2Err> {
        // Matrix multiply: %*%
        if op == BinOp::MatMul {
            return match (lhs, rhs) {
                (RVal::Matrix(a), RVal::Matrix(b)) => {
                    a.matmul(b).map(RVal::Matrix).map_err(|e| R2Err{msg:e,kind:ErrKind::Runtime})
                }
                _ => {
                    // Treat numeric vectors as column vectors, or coerce to matrix
                    let lv: Vec<f64> = self.as_reals(lhs)?.into_iter().filter_map(|x| x).collect();
                    let rv: Vec<f64> = self.as_reals(rhs)?.into_iter().filter_map(|x| x).collect();
                    let (lm, rm) = match (lhs, rhs) {
                        (RVal::Matrix(a), _) => (a.clone(), Matrix::new(rv.clone(), rv.len(), 1)),
                        (_, RVal::Matrix(b)) => (Matrix::new(lv.clone(), 1, lv.len()), b.clone()),
                        _ => (Matrix::new(lv.clone(), lv.len(), 1), Matrix::new(rv.clone(), 1, rv.len())),
                    };
                    lm.matmul(&rm).map(RVal::Matrix).map_err(|e| R2Err{msg:e,kind:ErrKind::Runtime})
                }
            };
        }
        // Element-wise arithmetic with a MATRIX operand keeps R's dims:
        // Matrix ⊙ Matrix (conformable), Matrix ⊙ vector/scalar (recycled
        // over the column-major data). Previously matrices were coerced to
        // plain vectors here, silently losing dim — surfaced by the aarch64
        // CI where the interpreter (not the JIT) runs these shapes.
        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow) {
            let mat_dims = match (lhs, rhs) {
                (RVal::Matrix(a), RVal::Matrix(b)) => {
                    if a.nrow != b.nrow || a.ncol != b.ncol {
                        return err!(Runtime, "non-conformable arrays ({}x{} vs {}x{})", a.nrow, a.ncol, b.nrow, b.ncol);
                    }
                    Some((a.nrow, a.ncol))
                }
                (RVal::Matrix(a), _) | (_, RVal::Matrix(a)) => Some((a.nrow, a.ncol)),
                _ => None,
            };
            if let Some((nr, nc)) = mat_dims {
                let to_data = |e: &mut Self, v: &RVal| -> Result<Vec<f64>, R2Err> {
                    Ok(match v {
                        RVal::Matrix(m) => m.data.clone(),
                        other => e.as_reals(other)?.into_iter()
                            .map(|o| o.unwrap_or(f64::NAN)).collect(),
                    })
                };
                let lv = to_data(self, lhs)?;
                let rv = to_data(self, rhs)?;
                let n = nr * nc;
                if lv.is_empty() || rv.is_empty() || lv.len().max(rv.len()) != n {
                    return err!(Runtime, "dims [product {}] do not match the length of object [{}]",
                                n, lv.len().max(rv.len()).max(1));
                }
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let a = lv[i % lv.len()];
                    let b = rv[i % rv.len()];
                    out.push(match op {
                        BinOp::Add => a + b, BinOp::Sub => a - b,
                        BinOp::Mul => a * b, BinOp::Div => a / b,
                        BinOp::Pow => a.powf(b),
                        _ => unreachable!(),
                    });
                }
                let mut m = Matrix::new(out, nr, nc);
                if let RVal::Matrix(src) = lhs { m.col_names = src.col_names.clone(); m.row_names = src.row_names.clone(); }
                else if let RVal::Matrix(src) = rhs { m.col_names = src.col_names.clone(); m.row_names = src.row_names.clone(); }
                return Ok(RVal::Matrix(m));
            }
        }
        // Logical operators — handled before numeric coercion to preserve
        // R's NA semantics (`TRUE & NA = NA`, `FALSE & NA = FALSE`, etc.).
        //
        // BinOp naming note: the lexer maps single `&` → Token::And and
        // double `&&` → Token::AndShort. So:
        //   - `BinOp::And` / `BinOp::Or`           → R's `&` / `|`  (elementwise)
        //   - `BinOp::AndShort` / `BinOp::OrShort` → R's `&&` / `||` (scalar short-circuit)
        if matches!(op, BinOp::AndShort | BinOp::OrShort | BinOp::And | BinOp::Or) {
            let l = self.as_logicals(lhs)?;
            let r = self.as_logicals(rhs)?;
            // Scalar short-circuit forms `&&` / `||`: take first element of each side.
            if matches!(op, BinOp::AndShort | BinOp::OrShort) {
                let a = l.first().copied().flatten();
                let b = r.first().copied().flatten();
                let result = match op {
                    BinOp::AndShort => match (a, b) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None, // any NA with non-FALSE → NA
                    },
                    BinOp::OrShort => match (a, b) {
                        (Some(true), _) | (_, Some(true)) => Some(true),
                        (Some(false), Some(false)) => Some(false),
                        _ => None,
                    },
                    _ => unreachable!(),
                };
                return Ok(RVal::Logical(vec![result].into(), Attrs::default()));
            }
            // Elementwise vector forms `&` and `|`.
            let (ll, rl) = (l.len(), r.len());
            if ll == 0 || rl == 0 {
                return Ok(RVal::Logical(Vec::<Logical>::new().into(), Attrs::default()));
            }
            if ll != rl && ll != 1 && rl != 1 {
                if self.mode == ErrorMode::Strict {
                    return err!(Runtime, "logical vectors length {} vs {} mismatch", ll, rl);
                } else {
                    self.warnings.push(format!("Warning: recycling logical {} and {}", ll, rl));
                }
            }
            let len = ll.max(rl);
            let out: Vec<Logical> = (0..len).map(|i| {
                let a = l[i % ll];
                let b = r[i % rl];
                match op {
                    // R: TRUE & NA = NA; FALSE & NA = FALSE; NA & NA = NA
                    BinOp::And => match (a, b) {
                        (Some(false), _) | (_, Some(false)) => Some(false),
                        (Some(true), Some(true)) => Some(true),
                        _ => None,
                    },
                    // R: TRUE | NA = TRUE; FALSE | NA = NA; NA | NA = NA
                    BinOp::Or => match (a, b) {
                        (Some(true), _) | (_, Some(true)) => Some(true),
                        (Some(false), Some(false)) => Some(false),
                        _ => None,
                    },
                    _ => unreachable!(),
                }
            }).collect();
            return Ok(RVal::Logical(out.into(), Attrs::default()));
        }

        // ── Phase F.7: Single (f32) promotion semantics ────────────────
        //
        // `Single op Single` stays Single (f32). Mixed `Single op anything`
        // promotes to Numeric (f64). This matches NumPy's dtype promotion
        // rules and R's `as.single` discipline.
        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
            if let (RVal::Single(a, _), RVal::Single(b, _)) = (lhs, rhs) {
                if a.len() == b.len() && a.len() >= 1 {
                    let av = a.as_vec();
                    let bv = b.as_vec();
                    let out: Vec<r2_types::Single> = (0..av.len()).map(|i| {
                        match (av[i], bv[i]) {
                            (Some(x), Some(y)) => Some(match op {
                                BinOp::Add => x + y, BinOp::Sub => x - y,
                                BinOp::Mul => x * y, BinOp::Div => x / y,
                                _ => unreachable!(),
                            }),
                            _ => None,
                        }
                    }).collect();
                    return Ok(RVal::Single(Singles::new(out), Attrs::default()));
                }
            }
            // Mixed Single+Numeric (or Single+Integer/Logical): promote
            // by falling through to the existing Numeric path.
            // (`as_reals` already handles Single → Vec<Real> below.)
        }

        // ── Columnar fast path for dense element-wise arithmetic ────────
        //
        // When both sides are `RVal::Numeric` of the same length and the op
        // is a real arithmetic op (Add/Sub/Mul/Div/Pow/Mod), route through
        // `ColumnarF64::binary` which operates on dense `&[f64]` slices via
        // a tight loop — no per-element `Option<f64>` match, no `as_reals`
        // clone, no `i%len` modulo. NA semantics preserved by the
        // columnar kernel: output bitmap = AND of input bitmaps.
        //
        // Threshold: only worth it above ~64 elements. Below that the
        // columnar setup cost dominates and the slow path is faster.
        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Mod) {
            if let (RVal::Numeric(a, _), RVal::Numeric(b, _)) = (lhs, rhs) {
                // Use `len_fast()` (reads whichever form is cached) — calling
                // `.len()` here would Deref → materialise the boxed
                // `Vec<Option<f64>>`, the exact O(n) round-trip this columnar
                // path exists to avoid for `rnorm`/`seq`-style dense inputs.
                // Take the columnar kernel when the repack is free (both
                // operands already columnar — the producer-flip default) or
                // when n is large enough to amortise a repack.
                if a.len_fast() == b.len_fast()
                    && (a.len_fast() >= 64 || (a.len_fast() >= 2 && a.has_columnar() && b.has_columnar())) {
                    use r2_arrow::ArrowBinaryOp;
                    let arrow_op = match op {
                        BinOp::Add => ArrowBinaryOp::Add,
                        BinOp::Sub => ArrowBinaryOp::Sub,
                        BinOp::Mul => ArrowBinaryOp::Mul,
                        BinOp::Div => ArrowBinaryOp::Div,
                        BinOp::Pow => ArrowBinaryOp::Pow,
                        BinOp::Mod => ArrowBinaryOp::Mod,
                        _ => unreachable!(),
                    };
                    let ac = a.columnar();
                    let bc = b.columnar();
                    let result = ac.binary(arrow_op, &bc)
                        .map_err(|e| R2Err { msg: e, kind: ErrKind::Runtime })?;
                    return Ok(RVal::Numeric(Reals::from_columnar(result), Attrs::default()));
                }
                // Scalar-vector recycling: vector OP scalar via binary_scalar.
                // Only safe when the scalar is not NA — propagate-NA path
                // falls back to the slow path below.
                if b.len_fast() == 1 && a.len_fast() >= 64 {
                    if let Some(s) = b[0] {
                        use r2_arrow::ArrowBinaryOp;
                        let arrow_op = match op {
                            BinOp::Add => ArrowBinaryOp::Add,
                            BinOp::Sub => ArrowBinaryOp::Sub,
                            BinOp::Mul => ArrowBinaryOp::Mul,
                            BinOp::Div => ArrowBinaryOp::Div,
                            BinOp::Pow => ArrowBinaryOp::Pow,
                            BinOp::Mod => ArrowBinaryOp::Mod,
                            _ => unreachable!(),
                        };
                        let ac = a.columnar();
                        let result = ac.binary_scalar(arrow_op, s);
                        return Ok(RVal::Numeric(Reals::from_columnar(result), Attrs::default()));
                    }
                }
            }
        }

        // String / factor comparisons (== != < > <= >=) compare LABELS,
        // not numeric codes — so `iris$Species == "setosa"` works. Engaged
        // only when a side is character/factor; pure-numeric comparisons
        // still take the fast numeric path below.
        if matches!(op, BinOp::Eq|BinOp::Ne|BinOp::Lt|BinOp::Gt|BinOp::Le|BinOp::Ge)
            && (matches!(lhs, RVal::Character(..)|RVal::Factor(..))
                || matches!(rhs, RVal::Character(..)|RVal::Factor(..)))
        {
            let ls = self.label_vec(lhs);
            let rs = self.label_vec(rhs);
            let (ll, rl) = (ls.len(), rs.len());
            if ll == 0 || rl == 0 { return Ok(RVal::Logical(Vec::<Logical>::new().into(), Attrs::default())); }
            let len = ll.max(rl);
            let out: Vec<Logical> = (0..len).map(|i| match (&ls[i % ll], &rs[i % rl]) {
                (Some(a), Some(b)) => Some(match op {
                    BinOp::Eq => a == b, BinOp::Ne => a != b,
                    BinOp::Lt => a < b, BinOp::Gt => a > b, BinOp::Le => a <= b, BinOp::Ge => a >= b,
                    _ => false }),
                _ => None,
            }).collect();
            return Ok(RVal::Logical(out.into(), Attrs::default()));
        }

        let l = self.as_reals(lhs)?; let r = self.as_reals(rhs)?;
        let (ll, rl) = (l.len(), r.len());
        // R recycles silently when the longer length is a multiple of the
        // shorter (c(1,2,3,4) + c(10,20) is legal); only a fractional
        // recycle is worth flagging.
        if ll != rl && ll != 1 && rl != 1 && ll.max(rl) % ll.min(rl).max(1) != 0 { if self.mode == ErrorMode::Strict { return err!(Runtime, "vectors length {} vs {} mismatch", ll, rl); } else { self.warnings.push(format!("Warning: recycling {} and {}", ll, rl)); } }
        let len = ll.max(rl);
        match op {
            BinOp::Eq|BinOp::Ne|BinOp::Lt|BinOp::Gt|BinOp::Le|BinOp::Ge => {
                let r: Vec<Logical> = (0..len).map(|i| { let (a,b) = (l[i%ll], r[i%rl]); match (a,b) { (Some(a),Some(b)) => Some(match op { BinOp::Eq => (a-b).abs()<f64::EPSILON, BinOp::Ne => (a-b).abs()>=f64::EPSILON, BinOp::Lt => a<b, BinOp::Gt => a>b, BinOp::Le => a<=b, BinOp::Ge => a>=b, _ => false }), _ => None } }).collect();
                Ok(RVal::Logical(r.into(), Attrs::default()))
            }
            _ => {
                // R semantics: division by zero is IEEE (1/0 = Inf, 0/0 = NaN),
                // never an error. %% is FLOORED modulo (sign of the divisor):
                // -7 %% 3 = 2, matching R, not Rust's truncated `%`.
                let r: Vec<Real> = (0..len).map(|i| { let (a,b) = (l[i%ll], r[i%rl]); match (a,b) { (Some(a),Some(b)) => Some(match op { BinOp::Add => a+b, BinOp::Sub => a-b, BinOp::Mul => a*b, BinOp::Div => a/b, BinOp::Pow => a.powf(b), BinOp::Mod => a - (a/b).floor()*b, BinOp::IntDiv => (a/b).floor(), _ => 0.0 }), _ => None } }).collect(); Ok(RVal::Numeric(r.into(), Attrs::default()))
            }
        }
    }
    pub(crate) fn unary_op(&self, op: UnOp, v: &RVal) -> Result<RVal, R2Err> { match op { UnOp::Neg => { let r = self.as_reals(v)?; Ok(RVal::Numeric(r.into_iter().map(|x| x.map(|n| -n)).collect(), Attrs::default())) } UnOp::Pos => Ok(v.clone()), UnOp::Not => { let r = self.as_logicals(v)?; Ok(RVal::Logical(r.into_iter().map(|x| x.map(|b| !b)).collect(), Attrs::default())) } } }
    // `a:b` — columnar-first: ranges are dense i32 with no NAs, so build
    // ColumnarI32 directly; the boxed Vec<Option<i32>> view materialises
    // only if a caller walks elements.
    pub(crate) fn seq_colon(&self, l: &RVal, r: &RVal) -> Result<RVal, R2Err> { let from = self.scalar_f64(l)?.ok_or(R2Err{msg:"NA in seq".into(),kind:ErrKind::Runtime})? as i32; let to = self.scalar_f64(r)?.ok_or(R2Err{msg:"NA in seq".into(),kind:ErrKind::Runtime})? as i32; let s: Vec<i32> = if from<=to { (from..=to).collect() } else { (to..=from).rev().collect() }; Ok(RVal::Integer(Ints::from_dense_i32(s), Attrs::default())) }
}
