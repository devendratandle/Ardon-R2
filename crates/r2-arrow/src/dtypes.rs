//! Additional columnar dtypes: i32 / i64 / bool (Phase F.6).
//!
//! Independent siblings of `ColumnarF64`, same validity-bitmap
//! convention. See the section comment below for the design notes.

// ── Additional dtypes: i32 / bool (Phase F.6) ─────────────────────────
//
// Mirrors `ColumnarF64`'s layout for two additional primitives. Bool uses
// a packed-bit values buffer (one bit per element) on top of the same
// validity-bitmap convention, which halves memory vs storing one byte
// per logical and trades a tiny bit of shift/mask cost.
//
// Not included in F.6.v1: `ColumnarI64` (mechanical copy of I32 with
// `i64` — add when an actual i64 hot path materializes) and
// `ColumnarUtf8` (variable-length strings need a separate offsets array
// plus a values byte buffer — its own design pass, deferred).
//
// **What this unlocks:** when a future "F.6 storage migration" pass
// changes `RVal::Integer` / `RVal::Logical` to use these columnar forms
// (same shape as the F.3 change for `RVal::Numeric`), integer/logical
// reductions get the same SIMD-friendly dense path that f64 already has.

/// Packed columnar storage for `Option<i32>`.
#[derive(Debug, Clone)]
pub struct ColumnarI32 {
    values: Vec<i32>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarI32 {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarI32 { values: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a dense `Vec<i32>` (no nulls).
    pub fn from_vec(values: Vec<i32>) -> Self {
        let len = values.len();
        ColumnarI32 { values, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `&[Option<i32>]` with lazy bitmap allocation: the
    /// bitmap is only created when the first `None` is encountered.
    pub fn from_option_slice(opts: &[Option<i32>]) -> Self {
        let len = opts.len();
        let mut values = Vec::with_capacity(len);
        let mut bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, opt) in opts.iter().enumerate() {
            match opt {
                Some(v) => {
                    values.push(*v);
                    if let Some(b) = bits.as_mut() {
                        b[i / 8] |= 1 << (i % 8);
                    }
                }
                None => {
                    values.push(0);
                    if bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        bits = Some(new_bits);
                    }
                    null_count += 1;
                }
            }
        }
        ColumnarI32 { values, valid_bits: bits, len, null_count }
    }

    /// Materialize back into `Vec<Option<i32>>`.
    pub fn to_options(&self) -> Vec<Option<i32>> {
        match &self.valid_bits {
            None => self.values.iter().take(self.len).map(|v| Some(*v)).collect(),
            Some(bits) => (0..self.len).map(|i| {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }).collect(),
        }
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True if zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no nulls — enables the SIMD-friendly dense fast path.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }
    /// Borrow the raw values buffer. Positions marked invalid by
    /// `valid_bits` are arbitrary (typically 0) — consult the bitmap.
    pub fn values(&self) -> &[i32] { &self.values[..self.len] }
    /// Validity bitmap when present (None ⇒ all valid).
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Read one element with bounds check.
    pub fn get(&self, i: usize) -> Option<i32> {
        assert!(i < self.len, "ColumnarI32 index {} out of bounds (len {})", i, self.len);
        match &self.valid_bits {
            None => Some(self.values[i]),
            Some(bits) => if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None },
        }
    }

    /// Sum as i64 (avoids i32 overflow on long columns).
    /// With nulls and `na_rm=false`, returns `None`.
    pub fn sum(&self, na_rm: bool) -> Option<i64> {
        if self.is_dense() {
            return Some(self.values().iter().map(|v| *v as i64).sum());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s: i64 = 0;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i] as i64; }
        }
        Some(s)
    }

    /// Min, NA-aware.
    pub fn min(&self, na_rm: bool) -> Option<i32> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(*self.values().iter().min().unwrap());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i32> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.min(self.values[i])));
            }
        }
        m
    }

    /// Max, NA-aware.
    pub fn max(&self, na_rm: bool) -> Option<i32> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(*self.values().iter().max().unwrap());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i32> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.max(self.values[i])));
            }
        }
        m
    }
}

impl Default for ColumnarI32 {
    fn default() -> Self { ColumnarI32::new() }
}

/// Packed columnar storage for `Option<i64>`. Mechanical mirror of
/// `ColumnarI32` for the 64-bit integer dtype that external columnar
/// files (Arrow `int64`, Parquet `INT64`) carry. `sum` accumulates into
/// `i128` so even a full column of `i64::MAX` cannot overflow.
#[derive(Debug, Clone)]
pub struct ColumnarI64 {
    values: Vec<i64>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarI64 {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarI64 { values: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a dense `Vec<i64>` (no nulls).
    pub fn from_vec(values: Vec<i64>) -> Self {
        let len = values.len();
        ColumnarI64 { values, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `&[Option<i64>]` with lazy bitmap allocation.
    pub fn from_option_slice(opts: &[Option<i64>]) -> Self {
        let len = opts.len();
        let mut values = Vec::with_capacity(len);
        let mut bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, opt) in opts.iter().enumerate() {
            match opt {
                Some(v) => {
                    values.push(*v);
                    if let Some(b) = bits.as_mut() { b[i / 8] |= 1 << (i % 8); }
                }
                None => {
                    values.push(0);
                    if bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        bits = Some(new_bits);
                    }
                    null_count += 1;
                }
            }
        }
        ColumnarI64 { values, valid_bits: bits, len, null_count }
    }

    /// Materialize back into `Vec<Option<i64>>`.
    pub fn to_options(&self) -> Vec<Option<i64>> {
        match &self.valid_bits {
            None => self.values.iter().take(self.len).map(|v| Some(*v)).collect(),
            Some(bits) => (0..self.len).map(|i| {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }).collect(),
        }
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True if zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no nulls — enables the dense fast path.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }
    /// Borrow the raw values buffer (invalid positions are arbitrary).
    pub fn values(&self) -> &[i64] { &self.values[..self.len] }
    /// Validity bitmap when present (None ⇒ all valid).
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Read one element with bounds check.
    pub fn get(&self, i: usize) -> Option<i64> {
        assert!(i < self.len, "ColumnarI64 index {} out of bounds (len {})", i, self.len);
        match &self.valid_bits {
            None => Some(self.values[i]),
            Some(bits) => if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None },
        }
    }

    /// Sum as `i128` (cannot overflow). `na_rm=false` with nulls ⇒ `None`.
    pub fn sum(&self, na_rm: bool) -> Option<i128> {
        if self.is_dense() {
            return Some(self.values().iter().map(|v| *v as i128).sum());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s: i128 = 0;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i] as i128; }
        }
        Some(s)
    }

    /// Min, NA-aware.
    pub fn min(&self, na_rm: bool) -> Option<i64> {
        if self.len == 0 { return None; }
        if self.is_dense() { return Some(*self.values().iter().min().unwrap()); }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i64> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.min(self.values[i])));
            }
        }
        m
    }

    /// Max, NA-aware.
    pub fn max(&self, na_rm: bool) -> Option<i64> {
        if self.len == 0 { return None; }
        if self.is_dense() { return Some(*self.values().iter().max().unwrap()); }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i64> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.max(self.values[i])));
            }
        }
        m
    }
}

impl Default for ColumnarI64 {
    fn default() -> Self { ColumnarI64::new() }
}

/// Packed columnar storage for `Option<bool>`.
///
/// Values use a packed-bit representation (one bit per element) like the
/// validity bitmap — so 1 million bools fit in 125 KB of values + 125 KB
/// of bitmap = 250 KB total, versus 16 MB for `Vec<Option<bool>>` (Rust's
/// `Option<bool>` is 1 byte; plus null bookkeeping).
#[derive(Debug, Clone)]
pub struct ColumnarBool {
    /// Packed bits: bit `i` is value of element `i`.
    /// Position marked invalid in `valid_bits` ⇒ value bit is don't-care.
    value_bits: Vec<u8>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarBool {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarBool { value_bits: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a dense `Vec<bool>` (no nulls).
    pub fn from_vec(values: Vec<bool>) -> Self {
        let len = values.len();
        let mut value_bits = vec![0u8; (len + 7) / 8];
        for (i, &b) in values.iter().enumerate() {
            if b { value_bits[i / 8] |= 1 << (i % 8); }
        }
        ColumnarBool { value_bits, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `&[Option<bool>]`.
    pub fn from_option_slice(opts: &[Option<bool>]) -> Self {
        let len = opts.len();
        let mut value_bits = vec![0u8; (len + 7) / 8];
        let mut valid_bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, opt) in opts.iter().enumerate() {
            match opt {
                Some(b) => {
                    if *b { value_bits[i / 8] |= 1 << (i % 8); }
                    if let Some(bits) = valid_bits.as_mut() {
                        bits[i / 8] |= 1 << (i % 8);
                    }
                }
                None => {
                    if valid_bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        valid_bits = Some(new_bits);
                    }
                    null_count += 1;
                }
            }
        }
        ColumnarBool { value_bits, valid_bits, len, null_count }
    }

    /// Materialize back into `Vec<Option<bool>>`.
    pub fn to_options(&self) -> Vec<Option<bool>> {
        (0..self.len).map(|i| {
            let valid = match &self.valid_bits {
                None => true,
                Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1,
            };
            if valid {
                Some((self.value_bits[i / 8] >> (i % 8)) & 1 == 1)
            } else {
                None
            }
        }).collect()
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True if zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no nulls.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }
    /// Borrow packed value bitmap (LSB-first within each byte).
    pub fn value_bits(&self) -> &[u8] { &self.value_bits }
    /// Validity bitmap when present.
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Read one element.
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.len, "ColumnarBool index {} out of bounds (len {})", i, self.len);
        let valid = match &self.valid_bits {
            None => true,
            Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1,
        };
        if valid { Some((self.value_bits[i / 8] >> (i % 8)) & 1 == 1) } else { None }
    }

    /// Count of TRUE values among valid elements.
    pub fn count_true(&self) -> usize {
        match &self.valid_bits {
            None => {
                // Dense: popcount over value_bits, masking trailing slop bits.
                let mut total = 0usize;
                for b in 0..(self.len / 8) {
                    total += self.value_bits[b].count_ones() as usize;
                }
                let rem = self.len % 8;
                if rem > 0 {
                    let mask = (1u8 << rem) - 1;
                    total += (self.value_bits[self.len / 8] & mask).count_ones() as usize;
                }
                total
            }
            Some(bits) => {
                // Sparse: only count where valid AND set.
                let mut total = 0usize;
                for i in 0..self.len {
                    if (bits[i / 8] >> (i % 8)) & 1 == 1
                        && (self.value_bits[i / 8] >> (i % 8)) & 1 == 1
                    {
                        total += 1;
                    }
                }
                total
            }
        }
    }

    /// Count of FALSE values among valid elements.
    pub fn count_false(&self) -> usize {
        let valid = self.len - self.null_count;
        valid - self.count_true()
    }

    /// `any(x)` — TRUE if at least one TRUE, NA-aware: returns None if
    /// no TRUE and at least one NA (R semantics).
    pub fn any(&self) -> Option<bool> {
        if self.count_true() > 0 { return Some(true); }
        if self.null_count > 0 { return None; }
        Some(false)
    }

    /// `all(x)` — TRUE if every valid element is TRUE, NA-aware.
    pub fn all(&self) -> Option<bool> {
        if self.count_false() > 0 { return Some(false); }
        if self.null_count > 0 { return None; }
        Some(true)
    }
}

impl Default for ColumnarBool {
    fn default() -> Self { ColumnarBool::new() }
}

#[cfg(test)]
mod f6_dtypes_tests {
    use super::*;

    #[test]
    fn i32_dense_roundtrip() {
        let c = ColumnarI32::from_vec(vec![1, 2, 3, -4, 5]);
        assert_eq!(c.len(), 5);
        assert_eq!(c.null_count(), 0);
        assert!(c.is_dense());
        assert_eq!(c.to_options(), vec![Some(1), Some(2), Some(3), Some(-4), Some(5)]);
        assert!(c.valid_bits().is_none(), "dense should not allocate bitmap");
    }

    #[test]
    fn i32_with_nulls() {
        let c = ColumnarI32::from_option_slice(&[Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.to_options(), vec![Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(c.get(0), Some(1));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(3));
    }

    #[test]
    fn i32_reductions() {
        let c = ColumnarI32::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(c.sum(false), Some(31));
        assert_eq!(c.min(false), Some(1));
        assert_eq!(c.max(false), Some(9));
    }

    #[test]
    fn i32_sum_with_na_propagates() {
        let c = ColumnarI32::from_option_slice(&[Some(1), None, Some(3)]);
        // Without na_rm, NA poisons the result.
        assert_eq!(c.sum(false), None);
        // With na_rm, skip nulls.
        assert_eq!(c.sum(true), Some(4));
    }

    #[test]
    fn i64_roundtrip_reductions_and_overflow() {
        let c = ColumnarI64::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(c.len(), 8);
        assert_eq!(c.sum(false), Some(31i128));
        assert_eq!(c.min(false), Some(1));
        assert_eq!(c.max(false), Some(9));
        // i64::MAX summed would overflow i64 but fits the i128 accumulator.
        let big = ColumnarI64::from_vec(vec![i64::MAX, i64::MAX, i64::MAX]);
        assert_eq!(big.sum(false), Some(3i128 * i64::MAX as i128));
        // Nulls: poison without na_rm, skipped with.
        let n = ColumnarI64::from_option_slice(&[Some(10), None, Some(20)]);
        assert_eq!(n.null_count(), 1);
        assert_eq!(n.sum(false), None);
        assert_eq!(n.sum(true), Some(30i128));
        assert_eq!(n.to_options(), vec![Some(10), None, Some(20)]);
    }

    #[test]
    fn i32_sum_avoids_overflow_via_i64() {
        // i32::MAX summed 3 times would overflow i32 but fits i64.
        let c = ColumnarI32::from_vec(vec![i32::MAX, i32::MAX, i32::MAX]);
        let s = c.sum(false).unwrap();
        assert_eq!(s, 3i64 * i32::MAX as i64);
    }

    #[test]
    fn bool_dense_packs_one_bit_per_element() {
        let c = ColumnarBool::from_vec(vec![true, false, true, true, false, true, false, true]);
        // 8 elements packed into 1 byte.
        assert_eq!(c.value_bits().len(), 1);
        assert_eq!(c.count_true(), 5);
        assert_eq!(c.count_false(), 3);
        assert_eq!(c.to_options(),
            vec![Some(true), Some(false), Some(true), Some(true),
                 Some(false), Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn bool_count_true_handles_trailing_partial_byte() {
        // 5 bits — partial last byte. Without masking, padding bits would
        // leak into popcount.
        let c = ColumnarBool::from_vec(vec![true, true, true, true, true]);
        assert_eq!(c.count_true(), 5);
    }

    #[test]
    fn bool_with_nulls() {
        let c = ColumnarBool::from_option_slice(&[Some(true), None, Some(false), Some(true), None]);
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.count_true(), 2);
        assert_eq!(c.count_false(), 1);
        assert_eq!(c.to_options(), vec![Some(true), None, Some(false), Some(true), None]);
    }

    #[test]
    fn bool_any_and_all() {
        // Dense — no NA.
        let all_true = ColumnarBool::from_vec(vec![true, true, true]);
        assert_eq!(all_true.all(), Some(true));
        assert_eq!(all_true.any(), Some(true));

        let mixed = ColumnarBool::from_vec(vec![true, false, true]);
        assert_eq!(mixed.all(), Some(false));
        assert_eq!(mixed.any(), Some(true));

        let all_false = ColumnarBool::from_vec(vec![false, false]);
        assert_eq!(all_false.all(), Some(false));
        assert_eq!(all_false.any(), Some(false));

        // With NA — R semantics: any() returns None when uncertain, all() too.
        let with_na = ColumnarBool::from_option_slice(&[None, Some(false)]);
        // any: no TRUE among valid, but NA exists → unknown.
        assert_eq!(with_na.any(), None);
        let with_na_t = ColumnarBool::from_option_slice(&[None, Some(true)]);
        // any: has a TRUE → definitively true regardless of NA.
        assert_eq!(with_na_t.any(), Some(true));

        let with_na_f = ColumnarBool::from_option_slice(&[None, Some(false)]);
        // all: has a FALSE → definitively false regardless of NA.
        assert_eq!(with_na_f.all(), Some(false));
    }
}
