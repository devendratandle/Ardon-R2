//! The JIT fast path of a closure call (Phases C.2 + C.3): look up or
//! compile the body, then run the compiled kernel on the arguments' raw
//! buffers. Every function here returns `None` when the arguments do not
//! fit the kernel's shape, and the interpreter runs the body instead.
//!
//! The NA-aware zero-copy bridge (Phase F.3): `RVal::Numeric` is `Reals`,
//! which caches an `Arc<ColumnarF64>`, so `col.values()` is a dense
//! `&[f64]` with no allocation — NA reads as NaN and propagates through
//! the kernel's arithmetic. On the way out the result's NA structure is
//! rebuilt from the INPUT bitmaps rather than by scanning the output for
//! NaN, so NA is preserved exactly (NaN ≠ NA is kept). One allocation
//! round-trip instead of two. A matrix's column-major buffer (J.3) is the
//! same kind of dense f64 vector; results keep its dim and dimnames.

use std::sync::Arc;
use r2_types::*;
use crate::{Engine, body_defines_closure};
use crate::na_bitmap::{combine_binary_output, combine_ternary_output, combine_unary_output};

impl Engine {
    /// Run `cl` compiled, if it can be. Only a call that binds every
    /// parameter positionally (no `...`, no defaults) is eligible.
    pub(crate) fn try_jit_call(&mut self, cl: &Closure, args: &[EvalArg]) -> Option<RVal> {
        if !self.jit_enabled
            || cl.params.len() != args.len()
            || !cl.params.iter().all(|p| !p.dots && p.default.is_none())
        {
            return None;
        }
        let h = self.jit_handle(cl)?;
        // a compiled body obeys the same rule as an interpreted one:
        // visible unless its last expression is invisible
        self.visible = crate::eval::tail_is_visible(&cl.body);
        run_kernel(h.as_ref(), args)
    }

    /// The compiled body, compiling on first sight. The cache holds the
    /// body's Arc, and only the SAME body is a hit (guards against a
    /// recycled pointer).
    fn jit_handle(&mut self, cl: &Closure) -> Option<Arc<dyn JitHandle>> {
        let key = Arc::as_ptr(&cl.body) as usize;
        match self.jit_cache.get(&key) {
            Some((body, slot)) if Arc::ptr_eq(body, &cl.body) => slot.clone(),
            _ => {
                // Never JIT a function whose body defines/returns a closure
                // — the IR has no representation for it and would
                // mis-compile it to a numeric (silent wrong result). Such
                // functions aren't numeric hot loops anyway. Computed once
                // per body, then cached.
                let h = if body_defines_closure(&cl.body) { None } else { r2_jit::try_compile_closure(cl) };
                self.jit_cache.insert(key, (cl.body.clone(), h.clone()));
                h
            }
        }
    }
}

fn run_kernel(h: &dyn JitHandle, args: &[EvalArg]) -> Option<RVal> {
    let a: Vec<&RVal> = args.iter().map(|e| &e.value).collect();
    match h.kind() {
        JitKind::Vector1ToScalar  => vector1_to_scalar(h, &a),
        JitKind::Vector2ToScalar  => vector2_to_scalar(h, &a),
        JitKind::VectorBinaryMap  => vector_binary_map(h, &a),
        JitKind::VectorTernaryMap => vector_ternary_map(h, &a),
        JitKind::VectorMap        => vector_map(h, &a),
        JitKind::IndexedStoreMap1 => indexed_store1(h, &a),
        JitKind::IndexedStoreMap2 => indexed_store2(h, &a),
        JitKind::MatVecIterOut    => matvec(h, &a),
        JitKind::Scalar           => scalar(h, &a),
    }
}

fn num1(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }

/// A matrix of `like`'s shape and dimnames holding `data`.
fn matrix_like(like: &Matrix, data: Vec<f64>) -> RVal {
    RVal::Matrix(Matrix {
        data, nrow: like.nrow, ncol: like.ncol,
        col_names: like.col_names.clone(), row_names: like.row_names.clone(),
    })
}

/// A reduction over one vector or one matrix (`sum(m)`, `sum(m*m)`, …).
/// Empty → interpreter: the indexed-loop kernel uses R's `1:length` loop,
/// and `1:0` would step out of bounds.
fn vector1_to_scalar(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    match a {
        [RVal::Numeric(v, _)] if !v.is_empty() => h.try_call_vec1(&v.columnar().values()).map(num1),
        [RVal::Matrix(m)] if !m.data.is_empty() => h.try_call_vec1(&m.data).map(num1),
        _ => None,
    }
}

/// Phase J.2 — fused binary map-reduce (e.g. `sum(x*w)`), or over two
/// same-sized matrices (the Frobenius inner product `sum(A*B)`).
fn vector2_to_scalar(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    match a {
        [RVal::Numeric(x, _), RVal::Numeric(y, _)] if x.len() == y.len() && !x.is_empty() =>
            h.try_call_vec2(&x.columnar().values(), &y.columnar().values()).map(num1),
        [RVal::Matrix(x), RVal::Matrix(y)] if x.data.len() == y.data.len() && !x.data.is_empty() =>
            h.try_call_vec2(&x.data, &y.data).map(num1),
        _ => None,
    }
}

/// Two equal-length vectors → a vector whose NA bitmap is the AND of
/// theirs; or element-wise over two same-shaped matrices (`A + B`).
fn vector_binary_map(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    match a {
        [RVal::Numeric(x, _), RVal::Numeric(y, _)] if x.len() == y.len() && !x.is_empty() => {
            let (xc, yc) = (x.columnar(), y.columnar());
            let mut out = vec![0.0; x.len()];
            if !h.try_call_vec_binary(&xc.values(), &yc.values(), &mut out) { return None; }
            let result = combine_binary_output(&out, xc.valid_bits(), yc.valid_bits());
            Some(RVal::Numeric(result.into(), Attrs::default()))
        }
        [RVal::Matrix(x), RVal::Matrix(y)]
            if x.data.len() == y.data.len() && x.nrow == y.nrow && !x.data.is_empty() =>
        {
            let mut out = vec![0.0; x.data.len()];
            if !h.try_call_vec_binary(&x.data, &y.data, &mut out) { return None; }
            Some(matrix_like(x, out))
        }
        _ => None,
    }
}

/// Three equal-length vectors → a vector; NA bitmap = AND of all three.
fn vector_ternary_map(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    let [RVal::Numeric(x, _), RVal::Numeric(y, _), RVal::Numeric(z, _)] = a else { return None };
    if x.len() != y.len() || y.len() != z.len() || x.is_empty() { return None; }
    let (xc, yc, zc) = (x.columnar(), y.columnar(), z.columnar());
    let mut out = vec![0.0; x.len()];
    if !h.try_call_vec_ternary(&xc.values(), &yc.values(), &zc.values(), &mut out) { return None; }
    let result = combine_ternary_output(&out, xc.valid_bits(), yc.valid_bits(), zc.valid_bits());
    Some(RVal::Numeric(result.into(), Attrs::default()))
}

/// Element-wise vector → vector (NA bitmap = the input's), or over a
/// matrix (`sqrt(m)`, `m*m`).
fn vector_map(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    match a {
        [RVal::Numeric(v, _)] => {
            let col = v.columnar();
            let values = col.values();
            let mut out = vec![0.0; values.len()];
            if !h.try_call_vec_map(&values, &mut out) { return None; }
            Some(RVal::Numeric(combine_unary_output(&out, col.valid_bits()).into(), Attrs::default()))
        }
        [RVal::Matrix(m)] if !m.data.is_empty() => {
            let mut out = vec![0.0; m.data.len()];
            if !h.try_call_vec_map(&m.data, &mut out) { return None; }
            Some(matrix_like(m, out))
        }
        _ => None,
    }
}

/// J.3 imperative store map: one input vector → an output vector the
/// engine allocates.
fn indexed_store1(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    let [RVal::Numeric(v, _)] = a else { return None };
    if v.is_empty() { return None; }
    let col = v.columnar();
    let values = col.values();
    let mut out = vec![0.0; values.len()];
    if !h.try_call_ixstore1(&values, &mut out) { return None; }
    Some(RVal::Numeric(combine_unary_output(&out, col.valid_bits()).into(), Attrs::default()))
}

fn indexed_store2(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    let [RVal::Numeric(x, _), RVal::Numeric(y, _)] = a else { return None };
    if x.len() != y.len() || x.is_empty() { return None; }
    let (xc, yc) = (x.columnar(), y.columnar());
    let mut out = vec![0.0; x.len()];
    if !h.try_call_ixstore2(&xc.values(), &yc.values(), &mut out) { return None; }
    let result = combine_binary_output(&out, xc.valid_bits(), yc.valid_bits());
    Some(RVal::Numeric(result.into(), Attrs::default()))
}

/// J.4 matrix state: (n×p matrix, n-vector) → p-vector.
fn matvec(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    let [RVal::Matrix(m), RVal::Numeric(v, _)] = a else { return None };
    if m.nrow == 0 || m.ncol == 0 || v.len() != m.nrow { return None; }
    let mut out = vec![0.0f64; m.ncol];
    if !h.try_call_matvec(&m.data, m.nrow, m.ncol, &v.columnar().values(), &mut out) { return None; }
    Some(RVal::Numeric(Reals::from_dense_f64(out), Attrs::default()))
}

/// Scalars in, scalar out: every argument a length-1 non-NA number or
/// logical.
fn scalar(h: &dyn JitHandle, a: &[&RVal]) -> Option<RVal> {
    let mut farg: Vec<f64> = Vec::with_capacity(a.len());
    for v in a {
        let x = match v {
            RVal::Numeric(v, _) if v.len() == 1 => v[0]?,
            RVal::Integer(v, _) if v.len() == 1 => v[0]? as f64,
            RVal::Logical(v, _) if v.len() == 1 => if v[0]? { 1.0 } else { 0.0 },
            _ => return None,
        };
        farg.push(x);
    }
    h.try_call_real(&farg).map(num1)
}
