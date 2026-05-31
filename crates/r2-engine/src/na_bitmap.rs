//! Output-bitmap combiners for the SIMD / Cranelift JIT arithmetic
//! pipeline. The JIT emits dense `Vec<f64>` results; these helpers
//! fold the per-operand validity bitmaps back in so the final
//! `Vec<Real>` carries `None` for any position where ANY operand was
//! `NA`, or where the computed value is NaN-from-arithmetic.
//!
//! Pure functions, no Engine coupling.

use r2_types::Real;

/// For unary maps: output bitmap = input bitmap. Positions marked
/// invalid in the input emit `None` regardless of the f64 value the
/// JIT loop produced (which would be NaN from NaN-propagation — same
/// result, but going through the bitmap lets us distinguish
/// NaN-from-arithmetic from NA-from-input later).
pub(crate) fn combine_unary_output(values: &[f64], in_bits: Option<&[u8]>) -> Vec<Real> {
    match in_bits {
        // Dense input: output is None only where arithmetic produced
        // NaN (e.g. log of negative). Preserves R semantics.
        None => values.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect(),
        // Sparse input: respect the input bitmap exactly. NaN-from-
        // arithmetic (in a "valid" position) still becomes None.
        Some(bits) => values.iter().enumerate().map(|(i, x)| {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                if x.is_nan() { None } else { Some(*x) }
            } else { None }
        }).collect(),
    }
}

/// Binary output bitmap = AND of input bitmaps. Position is valid
/// iff both inputs were valid at that index.
pub(crate) fn combine_binary_output(
    values: &[f64],
    a_bits: Option<&[u8]>,
    b_bits: Option<&[u8]>,
) -> Vec<Real> {
    let valid_at = |i: usize| -> bool {
        let va = match a_bits { None => true, Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1 };
        let vb = match b_bits { None => true, Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1 };
        va && vb
    };
    values.iter().enumerate().map(|(i, x)| {
        if valid_at(i) {
            if x.is_nan() { None } else { Some(*x) }
        } else { None }
    }).collect()
}

/// Ternary output bitmap = AND of three input bitmaps. Position
/// valid iff all three inputs were valid at that index AND the
/// computed result is not NaN-from-arithmetic.
pub(crate) fn combine_ternary_output(
    values: &[f64],
    a_bits: Option<&[u8]>,
    b_bits: Option<&[u8]>,
    c_bits: Option<&[u8]>,
) -> Vec<Real> {
    let valid_at = |bits: Option<&[u8]>, i: usize| -> bool {
        match bits { None => true, Some(b) => (b[i / 8] >> (i % 8)) & 1 == 1 }
    };
    values.iter().enumerate().map(|(i, x)| {
        if valid_at(a_bits, i) && valid_at(b_bits, i) && valid_at(c_bits, i) {
            if x.is_nan() { None } else { Some(*x) }
        } else { None }
    }).collect()
}
