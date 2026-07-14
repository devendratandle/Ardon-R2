//! NA-aware element type aliases + the columnar storage wrappers
//! (`Reals`/`Singles`/`Ints`/`Logicals`) backing R2 numeric vectors.

use std::sync::Arc;

// ── NA-aware element types ───────────────────────────────────────────

pub type Logical = Option<bool>;
pub type Integer = Option<i32>;
pub type Real = Option<f64>;
pub type Character = Option<Arc<str>>;

// ── Reals — Phase F.3 storage wrapper ────────────────────────────────
//
// Transparent wrapper over `Vec<Real>` that ALSO caches a
// `Arc<ColumnarF64>` for fast repeated `to_columnar()` access. Existing
// pattern-match code that expects `&[Real]` semantics continues to work
// via `Deref` — `v.iter()`, `v.len()`, `v[i]`, `&v[..]` all unchanged.
// Construction sites use `Reals::from(vec)` or `vec.into()`.
//
// Caching: the columnar form is computed lazily on first request and
// shared across clones via `Arc`.
/// `Reals` — dual-storage container for nullable `f64` data.
///
/// **F.3 native-columnar storage (v0.1.0):** `Reals` now holds **either**
/// a `Vec<Option<f64>>` (the legacy "boxed" form, source of truth for the
/// `Deref<Target=[Real]>` API surface), **or** an `Arc<ColumnarF64>` (the
/// native columnar form used by the binary/reduction kernels and the
/// JIT zero-copy bridge), **or both**. Whichever was set at construction
/// time is the canonical one; the other materialises lazily on demand.
///
/// Why this matters: before F.3, every numeric vector built by `rnorm`,
/// `seq`, comparison ops, etc. produced a `Vec<Option<f64>>` first, then
/// paid an O(n) re-pack to `ColumnarF64` on the first `.columnar()` call.
/// Binary fast-path results paid a third O(n) `to_options()` to rebuild
/// the boxed view. That cost dominated `a + b` and `sum(a)` on 1e7
/// vectors. F.3 lets producers that natively yield dense f64 (rnorm,
/// runif, binary kernel outputs) build via `from_columnar(...)` so the
/// boxed `Vec<Option<f64>>` is **never materialised** if no caller asks
/// for `&[Real]`.
///
/// API: `Deref<Target=[Real]>` continues to work — first slice access
/// materialises the `Vec<Option<f64>>` if it wasn't built yet. So legacy
/// callers see no behavior change, only better performance on the paths
/// that stay columnar end-to-end.
#[derive(Debug, Default)]
pub struct Reals {
    data: std::sync::OnceLock<Vec<Real>>,
    columnar: std::sync::OnceLock<std::sync::Arc<r2_arrow::ColumnarF64>>,
}

impl Reals {
    /// Build from a `Vec<Real>` (legacy boxed form). The columnar view
    /// materialises on first `.columnar()` call.
    pub fn new(data: Vec<Real>) -> Self {
        let r = Reals { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.data.set(data);
        r
    }

    /// Build from a pre-computed `ColumnarF64` without materialising the
    /// boxed `Vec<Real>` form. The latter only gets built if a caller
    /// later accesses `&[Real]` via `Deref` / `iter()` / `as_vec()`.
    /// This is the F.3 zero-conversion path used by the engine binary
    /// fast path and any builtin that produces dense f64.
    pub fn from_columnar(col: r2_arrow::ColumnarF64) -> Self {
        let r = Reals { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.columnar.set(std::sync::Arc::new(col));
        r
    }

    /// Build from a dense `Vec<f64>` with no nulls (the common case for
    /// `rnorm`, `runif`, `seq`, etc. that produce no NAs by construction).
    /// Skips the `Option<f64>` allocation entirely — the columnar form is
    /// built as a tight memcpy of the dense `Vec<f64>`. The boxed
    /// `Vec<Real>` view materialises only if a caller asks for `&[Real]`.
    pub fn from_dense_f64(data: Vec<f64>) -> Self {
        Self::from_columnar(r2_arrow::ColumnarF64::from_vec(data))
    }

    /// Consume into a `Vec<Real>`, materialising from columnar if needed.
    pub fn into_inner(mut self) -> Vec<Real> {
        if self.data.get().is_some() {
            self.data.take().unwrap()
        } else if let Some(c) = self.columnar.get() {
            c.to_options()
        } else {
            Vec::new()
        }
    }

    /// Get a reference to the boxed-form `Vec<Real>`, materialising if
    /// only the columnar form is set. O(n) on the first call after
    /// `from_columnar`; O(1) thereafter.
    pub fn as_vec(&self) -> &Vec<Real> {
        self.data.get_or_init(|| {
            match self.columnar.get() {
                Some(c) => c.to_options(),
                None => Vec::new(),
            }
        })
    }

    /// Get the cached `Arc<ColumnarF64>`, materialising from the boxed
    /// form if only that is set. O(n) on first call; O(1) thereafter.
    pub fn columnar(&self) -> std::sync::Arc<r2_arrow::ColumnarF64> {
        self.columnar.get_or_init(|| {
            match self.data.get() {
                Some(v) => std::sync::Arc::new(r2_arrow::ColumnarF64::from_option_slice(v)),
                None => std::sync::Arc::new(r2_arrow::ColumnarF64::from_vec(Vec::new())),
            }
        }).clone()
    }

    /// Length without forcing materialisation of either form — answers
    /// from whichever is already populated.
    pub fn len_fast(&self) -> usize {
        if let Some(v) = self.data.get() { v.len() }
        else if let Some(c) = self.columnar.get() { c.len() }
        else { 0 }
    }

    /// Empty check that doesn't materialise.
    pub fn is_empty_fast(&self) -> bool { self.len_fast() == 0 }

    /// True when the Arrow form is already cached — `.columnar()` is then
    /// O(1), so columnar kernels are profitable at ANY length (no repack).
    pub fn has_columnar(&self) -> bool { self.columnar.get().is_some() }
}

impl Clone for Reals {
    fn clone(&self) -> Self {
        // Preserve whichever forms are already cached. Arc clone is cheap;
        // the data Vec clones in O(n) if it was materialised.
        let r = Reals { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        if let Some(v) = self.data.get()     { let _ = r.data.set(v.clone()); }
        if let Some(c) = self.columnar.get() { let _ = r.columnar.set(c.clone()); }
        r
    }
}

impl PartialEq for Reals {
    fn eq(&self, other: &Self) -> bool {
        // Compare via boxed form for now; future could compare columnar.
        self.as_vec() == other.as_vec()
    }
}

impl std::ops::Deref for Reals {
    type Target = [Real];
    fn deref(&self) -> &[Real] { self.as_vec().as_slice() }
}

impl std::ops::DerefMut for Reals {
    fn deref_mut(&mut self) -> &mut [Real] {
        // Mutation requires the boxed form. Materialise if needed.
        if self.data.get().is_none() {
            let v = match self.columnar.get() {
                Some(c) => c.to_options(),
                None => Vec::new(),
            };
            let _ = self.data.set(v);
        }
        // Mutating invalidates the columnar cache.
        self.columnar = std::sync::OnceLock::new();
        self.data.get_mut().unwrap()
    }
}

impl From<Vec<Real>> for Reals {
    fn from(v: Vec<Real>) -> Self { Reals::new(v) }
}

impl FromIterator<Real> for Reals {
    fn from_iter<I: IntoIterator<Item = Real>>(iter: I) -> Self {
        Reals::new(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a Reals {
    type Item = &'a Real;
    type IntoIter = std::slice::Iter<'a, Real>;
    fn into_iter(self) -> Self::IntoIter { self.as_vec().iter() }
}

// Indexing pass-through.
impl<I: std::slice::SliceIndex<[Real]>> std::ops::Index<I> for Reals {
    type Output = I::Output;
    fn index(&self, idx: I) -> &Self::Output { &self.as_vec()[idx] }
}

// ── Singles — Phase F.7 single-precision storage wrapper ────────────
//
// Mirrors `Reals` but for `f32` payload. Two-storage layout: a boxed
// `Vec<Option<f32>>` and/or an `Arc<ColumnarF32>`. Either can be the
// canonical form; the other materialises lazily.
//
// Promotion semantics: `Singles + Singles → Singles`. Any mixing with
// `Reals` promotes to `Reals` (f64) — see engine `binary_op`. This is
// the same pattern as NumPy's dtype promotion and R's `as.single()`.

/// Single-precision float, possibly null. Equivalent to `Option<f32>`.
pub type Single = Option<f32>;

#[derive(Debug, Default)]
pub struct Singles {
    data: std::sync::OnceLock<Vec<Single>>,
    columnar: std::sync::OnceLock<std::sync::Arc<r2_arrow::ColumnarF32>>,
}

impl Singles {
    pub fn new(data: Vec<Single>) -> Self {
        let r = Singles { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.data.set(data);
        r
    }

    /// Build from a pre-computed ColumnarF32 without materialising the
    /// boxed form.
    pub fn from_columnar(col: r2_arrow::ColumnarF32) -> Self {
        let r = Singles { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.columnar.set(std::sync::Arc::new(col));
        r
    }

    /// Build from a dense `Vec<f32>` with no nulls.
    pub fn from_dense_f32(data: Vec<f32>) -> Self {
        Self::from_columnar(r2_arrow::ColumnarF32::from_vec(data))
    }

    /// Convert from a `Reals` (f64) — lossy narrowing. Use for `as.single()`.
    pub fn from_reals(r: &Reals) -> Self {
        let col_f64 = r.columnar();
        Self::from_columnar(r2_arrow::ColumnarF32::from_f64(&col_f64))
    }

    /// Materialize as `Reals` (f64) — lossless widening. Used for
    /// promotion when mixing Single with Numeric.
    pub fn to_reals(&self) -> Reals {
        let col_f32 = self.columnar();
        Reals::from_columnar(col_f32.to_f64())
    }

    pub fn into_inner(mut self) -> Vec<Single> {
        if self.data.get().is_some() {
            self.data.take().unwrap()
        } else if let Some(c) = self.columnar.get() {
            c.to_options()
        } else {
            Vec::new()
        }
    }

    pub fn as_vec(&self) -> &Vec<Single> {
        self.data.get_or_init(|| {
            match self.columnar.get() {
                Some(c) => c.to_options(),
                None => Vec::new(),
            }
        })
    }

    pub fn columnar(&self) -> std::sync::Arc<r2_arrow::ColumnarF32> {
        self.columnar.get_or_init(|| {
            match self.data.get() {
                Some(v) => std::sync::Arc::new(r2_arrow::ColumnarF32::from_option_slice(v)),
                None => std::sync::Arc::new(r2_arrow::ColumnarF32::from_vec(Vec::new())),
            }
        }).clone()
    }

    pub fn len_fast(&self) -> usize {
        if let Some(v) = self.data.get() { v.len() }
        else if let Some(c) = self.columnar.get() { c.len() }
        else { 0 }
    }

    pub fn is_empty_fast(&self) -> bool { self.len_fast() == 0 }
}

impl Clone for Singles {
    fn clone(&self) -> Self {
        let r = Singles { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        if let Some(v) = self.data.get()     { let _ = r.data.set(v.clone()); }
        if let Some(c) = self.columnar.get() { let _ = r.columnar.set(c.clone()); }
        r
    }
}

impl PartialEq for Singles {
    fn eq(&self, other: &Self) -> bool { self.as_vec() == other.as_vec() }
}

impl std::ops::Deref for Singles {
    type Target = [Single];
    fn deref(&self) -> &[Single] { self.as_vec().as_slice() }
}

impl From<Vec<Single>> for Singles {
    fn from(v: Vec<Single>) -> Self { Singles::new(v) }
}

impl FromIterator<Single> for Singles {
    fn from_iter<I: IntoIterator<Item = Single>>(iter: I) -> Self {
        Singles::new(iter.into_iter().collect())
    }
}

// ── Ints — Phase F.6 storage wrapper ─────────────────────────────────
// Mirrors `Reals`: `Vec<Integer>` + cached `Arc<ColumnarI32>`.
#[derive(Debug, Default)]
pub struct Ints {
    // Dual-form like `Reals`: EITHER view materialises lazily from the
    // other. `from_dense_i32` (e.g. `1:n` ranges) skips the per-element
    // Option boxing entirely until a caller demands `&[Integer]`.
    data: std::sync::OnceLock<Vec<Integer>>,
    columnar: std::sync::OnceLock<std::sync::Arc<r2_arrow::ColumnarI32>>,
}
impl Ints {
    pub fn new(data: Vec<Integer>) -> Self {
        let r = Ints { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.data.set(data);
        r
    }
    /// Dense, no-NA constructor (ranges, indices): columnar only.
    pub fn from_dense_i32(v: Vec<i32>) -> Self {
        let r = Ints { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        let _ = r.columnar.set(std::sync::Arc::new(r2_arrow::ColumnarI32::from_vec(v)));
        r
    }
    pub fn into_inner(mut self) -> Vec<Integer> {
        if self.data.get().is_some() { self.data.take().unwrap() }
        else if let Some(c) = self.columnar.get() { c.to_options() }
        else { Vec::new() }
    }
    pub fn as_vec(&self) -> &Vec<Integer> {
        self.data.get_or_init(|| match self.columnar.get() {
            Some(c) => c.to_options(),
            None => Vec::new(),
        })
    }
    pub fn columnar(&self) -> std::sync::Arc<r2_arrow::ColumnarI32> {
        self.columnar.get_or_init(|| {
            let boxed = self.data.get().map(|v| v.as_slice()).unwrap_or(&[]);
            std::sync::Arc::new(r2_arrow::ColumnarI32::from_option_slice(boxed))
        }).clone()
    }
    /// Length from whichever form is populated — no materialisation.
    pub fn len_fast(&self) -> usize {
        if let Some(v) = self.data.get() { v.len() }
        else if let Some(c) = self.columnar.get() { c.len() }
        else { 0 }
    }
}
impl Clone for Ints {
    fn clone(&self) -> Self {
        let r = Ints { data: std::sync::OnceLock::new(), columnar: std::sync::OnceLock::new() };
        if let Some(v) = self.data.get() { let _ = r.data.set(v.clone()); }
        if let Some(c) = self.columnar.get() { let _ = r.columnar.set(c.clone()); }
        r
    }
}
impl PartialEq for Ints {
    fn eq(&self, other: &Self) -> bool { self.as_vec() == other.as_vec() }
}
impl std::ops::Deref for Ints {
    type Target = [Integer];
    fn deref(&self) -> &[Integer] { self.as_vec() }
}
impl std::ops::DerefMut for Ints {
    fn deref_mut(&mut self) -> &mut [Integer] {
        // Mutation requires the boxed form. Materialise if needed, then
        // invalidate the columnar cache.
        if self.data.get().is_none() {
            let v = match self.columnar.get() {
                Some(c) => c.to_options(),
                None => Vec::new(),
            };
            let _ = self.data.set(v);
        }
        self.columnar = std::sync::OnceLock::new();
        self.data.get_mut().unwrap()
    }
}
impl From<Vec<Integer>> for Ints { fn from(v: Vec<Integer>) -> Self { Ints::new(v) } }
impl FromIterator<Integer> for Ints {
    fn from_iter<I: IntoIterator<Item = Integer>>(iter: I) -> Self { Ints::new(iter.into_iter().collect()) }
}
impl<'a> IntoIterator for &'a Ints {
    type Item = &'a Integer;
    type IntoIter = std::slice::Iter<'a, Integer>;
    fn into_iter(self) -> Self::IntoIter { self.as_vec().iter() }
}
impl<I: std::slice::SliceIndex<[Integer]>> std::ops::Index<I> for Ints {
    type Output = I::Output;
    fn index(&self, idx: I) -> &Self::Output { &self.as_vec()[idx] }
}

// ── Logicals — Phase F.6 storage wrapper ─────────────────────────────
// Mirrors `Reals`: `Vec<Logical>` + cached `Arc<ColumnarBool>`.
#[derive(Debug, Default)]
pub struct Logicals {
    data: Vec<Logical>,
    columnar: std::sync::OnceLock<std::sync::Arc<r2_arrow::ColumnarBool>>,
}
impl Logicals {
    pub fn new(data: Vec<Logical>) -> Self { Logicals { data, columnar: std::sync::OnceLock::new() } }
    pub fn into_inner(self) -> Vec<Logical> { self.data }
    pub fn as_vec(&self) -> &Vec<Logical> { &self.data }
    pub fn columnar(&self) -> std::sync::Arc<r2_arrow::ColumnarBool> {
        self.columnar.get_or_init(|| {
            std::sync::Arc::new(r2_arrow::ColumnarBool::from_option_slice(&self.data))
        }).clone()
    }
}
impl Clone for Logicals {
    fn clone(&self) -> Self {
        let r = Logicals { data: self.data.clone(), columnar: std::sync::OnceLock::new() };
        if let Some(c) = self.columnar.get() { let _ = r.columnar.set(c.clone()); }
        r
    }
}
impl PartialEq for Logicals {
    fn eq(&self, other: &Self) -> bool { self.data == other.data }
}
impl std::ops::Deref for Logicals {
    type Target = [Logical];
    fn deref(&self) -> &[Logical] { &self.data }
}
impl std::ops::DerefMut for Logicals {
    fn deref_mut(&mut self) -> &mut [Logical] {
        self.columnar = std::sync::OnceLock::new();
        &mut self.data
    }
}
impl From<Vec<Logical>> for Logicals { fn from(v: Vec<Logical>) -> Self { Logicals::new(v) } }
impl FromIterator<Logical> for Logicals {
    fn from_iter<I: IntoIterator<Item = Logical>>(iter: I) -> Self { Logicals::new(iter.into_iter().collect()) }
}
impl<'a> IntoIterator for &'a Logicals {
    type Item = &'a Logical;
    type IntoIter = std::slice::Iter<'a, Logical>;
    fn into_iter(self) -> Self::IntoIter { self.as_vec().iter() }
}
impl<I: std::slice::SliceIndex<[Logical]>> std::ops::Index<I> for Logicals {
    type Output = I::Output;
    fn index(&self, idx: I) -> &Self::Output { &self.as_vec()[idx] }
}

// ── Attributes ───────────────────────────────────────────────────────

