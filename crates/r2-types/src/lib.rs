// ═══════════════════════════════════════════════════════════════════════
// r2-types: The R2 Type System
//
// Design principles:
//   - Everything is a vector (scalar = vector of length 1)
//   - NA is compile-time enforced via Option<T>
//   - Text is ALWAYS text (never auto-factor)
//   - TRUE/FALSE/T/F are immutable reserved values
//   - 1-based indexing (user-facing)
//   - One unified `type` system (no S3/S4/R5/R6)
//   - Tensor type in base for ML library support
//   - Matrix type with linear algebra primitives
//   - Expr (AST) and RVal (runtime) are separate
// ═══════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

// Phase A — Type inferencer (annotation-only pass, see docs/ARCHITECTURE.md §5).
pub mod infer;

// ── Error types (Phase R foundation) ────────────────────────────────
//
// `R2Err` and `ErrKind` were previously in r2-engine. Moving them to
// r2-types lets any domain crate (r2-stats, r2-ml, r2-data, ...)
// implement builtins that return R2Err without depending on r2-engine.

#[derive(Debug)]
pub struct R2Err { pub msg: String, pub kind: ErrKind }

#[derive(Debug)]
pub enum ErrKind {
    Runtime,
    Type,
    Index,
    CtrlBreak,
    CtrlNext,
    CtrlReturn(Box<RVal>),
    /// Phase R.M.2 — user interrupt (Ctrl+C). The engine raises this when
    /// it observes the global INTERRUPT flag set by the SIGINT handler;
    /// the REPL catches it, prints a brief notice, and returns to the
    /// prompt instead of exiting. Non-interactive (script) mode treats
    /// it as a normal error and exits with a non-zero status.
    Interrupt,
}

impl PartialEq for ErrKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ErrKind::Runtime, ErrKind::Runtime) => true,
            (ErrKind::Type, ErrKind::Type) => true,
            (ErrKind::Index, ErrKind::Index) => true,
            (ErrKind::CtrlBreak, ErrKind::CtrlBreak) => true,
            (ErrKind::CtrlNext, ErrKind::CtrlNext) => true,
            (ErrKind::CtrlReturn(_), ErrKind::CtrlReturn(_)) => true,
            (ErrKind::Interrupt, ErrKind::Interrupt) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for R2Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error: {}", self.msg)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Phase R.M.2 — global interrupt flag.
//
// Set by the SIGINT (Ctrl+C) handler installed in the REPL binary. Read
// by the engine's evaluation loop at safe interruption points (top of
// each expression, top of each loop iteration, top of each function
// call). When set, the engine returns Err(R2Err { kind: Interrupt, .. }),
// which unwinds the call stack cleanly. The REPL catches it, prints a
// notice, clears the flag, and returns to the prompt — the process
// stays alive. Script mode does not catch it and exits non-zero.
// ─────────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicBool, Ordering};

pub static INTERRUPT: AtomicBool = AtomicBool::new(false);

/// Set by the SIGINT handler. The engine polls `is_interrupted()` at
/// safe points and raises `ErrKind::Interrupt` when it observes `true`.
#[inline]
pub fn request_interrupt() {
    INTERRUPT.store(true, Ordering::Relaxed);
}

/// Check and consume the interrupt flag in one step. Returns `true` if
/// the flag was set (and clears it).
#[inline]
pub fn take_interrupt() -> bool {
    INTERRUPT.swap(false, Ordering::Relaxed)
}

/// Non-consuming check. Used by the engine eval loop to decide whether
/// to raise `ErrKind::Interrupt` without clearing the flag (the REPL
/// driver clears it after catching).
#[inline]
pub fn is_interrupted() -> bool {
    INTERRUPT.load(Ordering::Relaxed)
}

/// Clear the flag without consuming it. Called by the REPL after it has
/// finished handling an Interrupt error, before returning to the prompt.
#[inline]
pub fn clear_interrupt() {
    INTERRUPT.store(false, Ordering::Relaxed);
}

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
    data: Vec<Integer>,
    columnar: std::sync::OnceLock<std::sync::Arc<r2_arrow::ColumnarI32>>,
}
impl Ints {
    pub fn new(data: Vec<Integer>) -> Self { Ints { data, columnar: std::sync::OnceLock::new() } }
    pub fn into_inner(self) -> Vec<Integer> { self.data }
    pub fn as_vec(&self) -> &Vec<Integer> { &self.data }
    pub fn columnar(&self) -> std::sync::Arc<r2_arrow::ColumnarI32> {
        self.columnar.get_or_init(|| {
            std::sync::Arc::new(r2_arrow::ColumnarI32::from_option_slice(&self.data))
        }).clone()
    }
}
impl Clone for Ints {
    fn clone(&self) -> Self {
        let r = Ints { data: self.data.clone(), columnar: std::sync::OnceLock::new() };
        if let Some(c) = self.columnar.get() { let _ = r.columnar.set(c.clone()); }
        r
    }
}
impl PartialEq for Ints {
    fn eq(&self, other: &Self) -> bool { self.data == other.data }
}
impl std::ops::Deref for Ints {
    type Target = [Integer];
    fn deref(&self) -> &[Integer] { &self.data }
}
impl std::ops::DerefMut for Ints {
    fn deref_mut(&mut self) -> &mut [Integer] {
        self.columnar = std::sync::OnceLock::new();
        &mut self.data
    }
}
impl From<Vec<Integer>> for Ints { fn from(v: Vec<Integer>) -> Self { Ints::new(v) } }
impl FromIterator<Integer> for Ints {
    fn from_iter<I: IntoIterator<Item = Integer>>(iter: I) -> Self { Ints::new(iter.into_iter().collect()) }
}
impl<'a> IntoIterator for &'a Ints {
    type Item = &'a Integer;
    type IntoIter = std::slice::Iter<'a, Integer>;
    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}
impl<I: std::slice::SliceIndex<[Integer]>> std::ops::Index<I> for Ints {
    type Output = I::Output;
    fn index(&self, idx: I) -> &Self::Output { &self.data[idx] }
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
    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}
impl<I: std::slice::SliceIndex<[Logical]>> std::ops::Index<I> for Logicals {
    type Output = I::Output;
    fn index(&self, idx: I) -> &Self::Output { &self.data[idx] }
}

// ── Attributes ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Attrs {
    pub names: Option<Vec<Arc<str>>>,
    pub dim: Option<Vec<usize>>,
    pub class: Option<Arc<str>>,
    pub custom: HashMap<Arc<str>, RVal>,
}

// ── Factor (explicit categorical — never auto-created) ───────────────

#[derive(Debug, Clone)]
pub struct Factor {
    pub codes: Vec<Option<u32>>,
    pub levels: Vec<Arc<str>>,
    pub ordered: bool,
}

// ── Formula (first-class for statistical modeling) ───────────────────

#[derive(Debug, Clone)]
pub struct Formula {
    pub lhs: Option<Box<FormulaExpr>>,
    pub rhs: Box<FormulaExpr>,
}

#[derive(Debug, Clone)]
pub enum FormulaExpr {
    Var(Arc<str>),
    Intercept,
    Dot,
    Add(Box<FormulaExpr>, Box<FormulaExpr>),
    Remove(Box<FormulaExpr>, Box<FormulaExpr>),
    Interact(Box<FormulaExpr>, Box<FormulaExpr>),
    Cross(Box<FormulaExpr>, Box<FormulaExpr>),
    Group(Box<FormulaExpr>, Box<FormulaExpr>),
    AsIs(Box<FormulaExpr>),
}

// ── R2 Type Definition (the ONE object system) ───────────────────────

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: Arc<str>,
    pub fields: Vec<FieldDef>,
    pub parent: Option<Arc<str>>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: Arc<str>,
    pub field_type: FieldType,
    pub default: Option<RVal>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    Numeric, Integer, Character, Logical, Any,
    ListOf(Box<FieldType>),
    TypeRef(Arc<str>),
    Tensor,
    Matrix,
}

#[derive(Debug, Clone)]
pub struct TypeInstance {
    pub type_name: Arc<str>,
    pub fields: HashMap<Arc<str>, RVal>,
}

// ── Environment (lexical scope chain) ────────────────────────────────

pub type EnvRef = Arc<Env>;

#[derive(Debug, Clone)]
pub struct Env {
    pub name: Option<Arc<str>>,
    pub bindings: HashMap<Arc<str>, RVal>,
    pub parent: Option<EnvRef>,
    pub locked: bool,
}

impl Env {
    pub fn new_global() -> EnvRef {
        Arc::new(Env { name: Some(Arc::from(".GlobalEnv")), bindings: HashMap::new(), parent: None, locked: false })
    }
    pub fn new_child(parent: EnvRef, name: Option<&str>) -> EnvRef {
        Arc::new(Env { name: name.map(Arc::from), bindings: HashMap::new(), parent: Some(parent), locked: false })
    }
    pub fn lookup(&self, name: &str) -> Option<&RVal> {
        self.bindings.get(name).or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }
}

// ── Closure ──────────────────────────────────────────────────────────

/// Shared, JIT-friendly view of a function body. Using `Arc<Expr>` (rather
/// than `Box<Expr>`) lets the engine use `Arc::as_ptr(&body)` as a stable
/// cache key across Closure clones — needed by the JIT cache (Phase C.2).
#[derive(Debug, Clone)]
pub struct Closure {
    pub params: Vec<Param>,
    pub body: Arc<Expr>,
    pub env: EnvRef,
}

/// What signature a JIT-compiled function was specialized for.
/// (Phase C.2 → Scalar; C.3 → adds Vector1ToScalar; C.4 → adds VectorMap.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitKind {
    /// `(f64, f64, ...) -> f64` — every arg and the return are scalar.
    Scalar,
    /// `(*const f64, i64) -> f64` — one f64 vector in, one scalar out.
    Vector1ToScalar,
    /// `(*const f64, *mut f64, i64) -> ()` — element-wise vector → vector.
    /// Caller pre-allocates the output buffer of the same length.
    VectorMap,
    /// `(*const f64, *const f64, *mut f64, i64) -> ()` — element-wise binary
    /// vector ⊗ vector → vector. Both inputs must be same length.
    VectorBinaryMap,
    /// `(*const f64, *const f64, *const f64, *mut f64, i64) -> ()` —
    /// element-wise ternary map: three same-length input vectors → one output.
    /// Used for branchy closures over three columns, e.g. an `ifelse`-shape
    /// `function(c, a, b) if (c > 0) a else b`. Phase C.5.
    VectorTernaryMap,
}

/// Object-safe handle to a JIT-compiled function. Lives in r2-jit (and any
/// future backends); declared here so `r2-engine` can hold one without a
/// direct dependency on `r2-jit`.
pub trait JitHandle: std::fmt::Debug {
    /// Specialization shape — engine uses this to pick the right call method.
    fn kind(&self) -> JitKind;
    /// Number of formal parameters (in source units, not ABI slots).
    fn arity(&self) -> usize;
    /// Scalar dispatch (Phase C.2). Returns `None` if `kind()` isn't Scalar.
    fn try_call_real(&self, args: &[f64]) -> Option<f64>;
    /// Vector1ToScalar dispatch (Phase C.3). Returns `None` if `kind()` isn't
    /// Vector1ToScalar. SAFETY contract: `ptr` must point to `len` valid f64s.
    /// Default impl returns None so existing impls compile unchanged.
    unsafe fn try_call_vec1(&self, _ptr: *const f64, _len: i64) -> Option<f64> { None }
    /// VectorMap dispatch (Phase C.4). SAFETY: `in_ptr` and `out_ptr` must
    /// each point to `len` valid f64s; out_ptr is written to.
    unsafe fn try_call_vec_map(&self, _in_ptr: *const f64, _out_ptr: *mut f64, _len: i64) -> bool { false }
    /// VectorBinaryMap dispatch (C.4-full). SAFETY: all three pointers must
    /// reference at least `len` valid f64s; out_ptr is written to.
    unsafe fn try_call_vec_binary(&self, _a_ptr: *const f64, _b_ptr: *const f64, _out_ptr: *mut f64, _len: i64) -> bool { false }
    /// VectorTernaryMap dispatch (Phase C.5). SAFETY: all four pointers must
    /// reference at least `len` valid f64s; out_ptr is written to.
    unsafe fn try_call_vec_ternary(
        &self,
        _a_ptr: *const f64,
        _b_ptr: *const f64,
        _c_ptr: *const f64,
        _out_ptr: *mut f64,
        _len: i64,
    ) -> bool { false }
}

/// EngineCtx — Phase R.2 step 6.
///
/// Trait that domain crates use when they need to call back into the
/// language evaluator (e.g., the apply family invoking a user-supplied
/// closure). r2-engine implements this for `Engine`; domain crates
/// program against the trait so they have no engine dependency.
///
/// Locked decision (§4.7): backwards-compatible — existing engine code
/// keeps using `Engine::call_fn` directly. The trait is a parallel
/// surface for crates outside the engine.
pub trait EngineCtx {
    /// Apply a function value to evaluated arguments.
    fn ctx_call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err>;
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Arc<str>,
    pub default: Option<Box<Expr>>,
    pub dots: bool,
}

// ── Method (attached to a type) ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Method {
    pub name: Arc<str>,
    pub type_name: Arc<str>,
    pub param_name: Arc<str>,
    pub extra_params: Vec<Param>,
    pub body: Box<Expr>,
}

// ── DataFrame ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<(Arc<str>, RVal)>,
    pub row_names: Option<Vec<Arc<str>>>,
}

impl DataFrame {
    pub fn nrow(&self) -> usize { self.columns.first().map_or(0, |(_, col)| rval_length(col)) }
    pub fn ncol(&self) -> usize { self.columns.len() }
    pub fn get_col(&self, name: &str) -> Option<&RVal> {
        self.columns.iter().find(|(n, _)| n.as_ref() == name).map(|(_, v)| v)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MATRIX — 2D numeric array with linear algebra operations
//
// This is the base type that ML libraries build on.
// Stored column-major (like R/Fortran) for BLAS compatibility.
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,     // column-major storage, no NA (use NaN for missing)
    pub nrow: usize,
    pub ncol: usize,
    pub col_names: Option<Vec<Arc<str>>>,
    pub row_names: Option<Vec<Arc<str>>>,
}

impl Matrix {
    pub fn new(data: Vec<f64>, nrow: usize, ncol: usize) -> Self {
        assert_eq!(data.len(), nrow * ncol, "data length must equal nrow * ncol");
        Matrix { data, nrow, ncol, col_names: None, row_names: None }
    }

    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Matrix::new(vec![0.0; nrow * ncol], nrow, ncol)
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    /// Get element at (row, col) — 0-based internal
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.nrow + row]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[col * self.nrow + row] = val;
    }

    /// Get a column as a slice
    pub fn col_slice(&self, col: usize) -> &[f64] {
        let start = col * self.nrow;
        &self.data[start..start + self.nrow]
    }

    /// Transpose — uses r2-linalg kernel
    pub fn transpose(&self) -> Matrix {
        let mut result = vec![0.0; self.nrow * self.ncol];
        r2_linalg::dtranspose(self.nrow, self.ncol, &self.data, &mut result).unwrap();
        Matrix::new(result, self.ncol, self.nrow)
    }

    /// Matrix multiply: self (m x k) * other (k x n) -> (m x n)
    /// Uses r2-linalg dgemm kernel (cache-blocked, SIMD-friendly)
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.ncol != other.nrow {
            return Err(format!("incompatible dimensions: {}x{} * {}x{}", self.nrow, self.ncol, other.nrow, other.ncol));
        }
        let mut c = vec![0.0; self.nrow * other.ncol];
        // Runtime-dispatched: uses an optimized BLAS variant DLL when
        // R2_BLAS points to one, else the built-in reference kernel.
        r2_linalg::dgemm_dispatch(self.nrow, other.ncol, self.ncol, 1.0, &self.data, &other.data, 0.0, &mut c)
            .map_err(|e| e.to_string())?;
        Ok(Matrix::new(c, self.nrow, other.ncol))
    }

    /// Element-wise operation
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Matrix {
        Matrix::new(self.data.iter().map(|x| f(*x)).collect(), self.nrow, self.ncol)
    }

    /// Element-wise binary operation
    pub fn zip_with(&self, other: &Matrix, f: impl Fn(f64, f64) -> f64) -> Result<Matrix, String> {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return Err("matrix dimensions must match".into());
        }
        Ok(Matrix::new(
            self.data.iter().zip(other.data.iter()).map(|(a, b)| f(*a, *b)).collect(),
            self.nrow, self.ncol,
        ))
    }

    /// Scalar multiplication — uses r2-linalg dscal kernel
    pub fn scale(&self, s: f64) -> Matrix {
        let mut data = self.data.clone();
        r2_linalg::dscal(s, &mut data);
        Matrix::new(data, self.nrow, self.ncol)
    }

    /// Add matrices
    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Subtract matrices
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, String> {
        self.zip_with(other, |a, b| a - b)
    }

    /// Column means
    pub fn col_means(&self) -> Vec<f64> {
        (0..self.ncol).map(|c| {
            self.col_slice(c).iter().sum::<f64>() / self.nrow as f64
        }).collect()
    }

    /// Column sums
    pub fn col_sums(&self) -> Vec<f64> {
        (0..self.ncol).map(|c| self.col_slice(c).iter().sum()).collect()
    }

    /// Row sums
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.nrow).map(|r| {
            (0..self.ncol).map(|c| self.get(r, c)).sum()
        }).collect()
    }

    /// Dot product of two column vectors — uses r2-linalg ddot kernel
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        r2_linalg::ddot(a, b)
    }

    /// Frobenius norm — uses r2-linalg dnrm2 kernel
    pub fn norm(&self) -> f64 {
        r2_linalg::dnrm2(&self.data)
    }

    /// Convert to vector of rows (for iteration)
    pub fn rows(&self) -> Vec<Vec<f64>> {
        (0..self.nrow).map(|r| {
            (0..self.ncol).map(|c| self.get(r, c)).collect()
        }).collect()
    }

    /// Solve Ax = b using r2-linalg dgesv (LU with partial pivoting)
    /// MUCH faster and more stable than the old Gaussian elimination
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, String> {
        if self.nrow != self.ncol { return Err("matrix must be square".into()); }
        if self.nrow != b.len() { return Err("dimensions don't match".into()); }
        let mut a = self.data.clone();
        let mut x = b.to_vec();
        r2_linalg::dgesv(self.nrow, &mut a, &mut x).map_err(|e| e.to_string())?;
        Ok(x)
    }

    /// Compute X^T * X — uses r2-linalg dcrossprod (avoids explicit transpose)
    pub fn crossprod(&self) -> Matrix {
        let mut c = vec![0.0; self.ncol * self.ncol];
        r2_linalg::dcrossprod(self.nrow, self.ncol, &self.data, &mut c).unwrap();
        Matrix::new(c, self.ncol, self.ncol)
    }

    /// Compute X^T * y where y is a vector — uses r2-linalg dgemv_t
    pub fn crossprod_vec(&self, y: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.ncol];
        r2_linalg::dgemv_t(self.nrow, self.ncol, 1.0, &self.data, y, 0.0, &mut result).unwrap();
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TENSOR — N-dimensional numeric array for ML
//
// This is in BASE so ML addon libraries can build on it.
// The user rarely creates tensors directly; data.frame → tensor
// conversion is automatic in ML pipelines.
//
// Storage: contiguous f64, row-major (C-order) for ML compatibility.
// GPU backing is a future extension (same API, different storage).
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: TensorDType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDType {
    Float64,
    Float32,
    Int32,
    Bool,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "data length must match shape product");
        let strides = Self::compute_strides(&shape);
        Tensor { data, shape, strides, dtype: TensorDType::Float64 }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Tensor::new(vec![0.0; total], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Tensor::new(vec![1.0; total], shape)
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Tensor::new(data, vec![len])
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.data.len() }

    /// Get element by multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat]
    }

    /// Set element by multi-dimensional index
    pub fn set(&mut self, indices: &[usize], val: f64) {
        let flat: usize = indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum();
        self.data[flat] = val;
    }

    /// Reshape (same data, different shape)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, String> {
        let new_total: usize = new_shape.iter().product();
        if new_total != self.numel() {
            return Err(format!("cannot reshape {} elements into shape {:?}", self.numel(), new_shape));
        }
        Ok(Tensor::new(self.data.clone(), new_shape))
    }

    /// Flatten to 1D
    pub fn flatten(&self) -> Tensor {
        Tensor::new(self.data.clone(), vec![self.numel()])
    }

    /// Element-wise operation
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor::new(self.data.iter().map(|x| f(*x)).collect(), self.shape.clone())
    }

    /// Element-wise binary (shapes must match or broadcast)
    pub fn zip_with(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, String> {
        if self.shape != other.shape {
            // Simple scalar broadcast
            if other.numel() == 1 {
                let s = other.data[0];
                return Ok(self.map(|x| f(x, s)));
            }
            if self.numel() == 1 {
                let s = self.data[0];
                return Ok(other.map(|x| f(s, x)));
            }
            return Err(format!("shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }
        Ok(Tensor::new(
            self.data.iter().zip(other.data.iter()).map(|(a, b)| f(*a, *b)).collect(),
            self.shape.clone(),
        ))
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a + b) }
    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a - b) }
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a * b) }
    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> { self.zip_with(other, |a, b| a / b) }

    pub fn scale(&self, s: f64) -> Tensor { self.map(|x| x * s) }
    pub fn sum(&self) -> f64 { self.data.iter().sum() }
    pub fn mean(&self) -> f64 { self.sum() / self.numel() as f64 }

    /// Common activation functions (ML foundation)
    pub fn relu(&self) -> Tensor { self.map(|x| if x > 0.0 { x } else { 0.0 }) }
    pub fn sigmoid(&self) -> Tensor { self.map(|x| 1.0 / (1.0 + (-x).exp())) }
    pub fn tanh_act(&self) -> Tensor { self.map(|x| x.tanh()) }
    pub fn softmax(&self) -> Tensor {
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self.data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        Tensor::new(exps.iter().map(|x| x / sum).collect(), self.shape.clone())
    }
    pub fn log_softmax(&self) -> Tensor {
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = self.data.iter().map(|x| x - max_val).collect();
        let log_sum_exp = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
        Tensor::new(shifted.iter().map(|x| x - log_sum_exp).collect(), self.shape.clone())
    }

    /// 2D matrix multiply for tensors (last two dims)
    pub fn matmul_2d(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("matmul_2d requires 2D tensors".into());
        }
        let m = self.shape[0];
        let k = self.shape[1];
        if k != other.shape[0] { return Err("inner dimensions don't match".into()); }
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    result[i * n + j] += self.get(&[i, p]) * other.get(&[p, j]);
                }
            }
        }
        Ok(Tensor::new(result, vec![m, n]))
    }

    /// Convert from Matrix (column-major → row-major)
    pub fn from_matrix(m: &Matrix) -> Tensor {
        let mut data = vec![0.0; m.nrow * m.ncol];
        for r in 0..m.nrow {
            for c in 0..m.ncol {
                data[r * m.ncol + c] = m.get(r, c);
            }
        }
        Tensor::new(data, vec![m.nrow, m.ncol])
    }

    /// Convert to Matrix (row-major → column-major)
    pub fn to_matrix(&self) -> Result<Matrix, String> {
        if self.ndim() != 2 { return Err("only 2D tensors can convert to Matrix".into()); }
        let nrow = self.shape[0];
        let ncol = self.shape[1];
        let mut data = vec![0.0; nrow * ncol];
        for r in 0..nrow {
            for c in 0..ncol {
                data[c * nrow + r] = self.get(&[r, c]); // column-major
            }
        }
        Ok(Matrix::new(data, nrow, ncol))
    }

    /// Convert DataFrame numeric columns to Tensor (for ML pipelines)
    pub fn from_dataframe(df: &DataFrame, columns: &[&str]) -> Result<Tensor, String> {
        let nrow = df.nrow();
        let ncol = columns.len();
        let mut data = vec![0.0; nrow * ncol];
        for (c, col_name) in columns.iter().enumerate() {
            let col = df.get_col(col_name).ok_or(format!("column '{}' not found", col_name))?;
            match col {
                RVal::Numeric(v, _) => {
                    for (r, val) in v.iter().enumerate() {
                        data[r * ncol + c] = val.unwrap_or(f64::NAN);
                    }
                }
                RVal::Integer(v, _) => {
                    for (r, val) in v.iter().enumerate() {
                        data[r * ncol + c] = val.map(|n| n as f64).unwrap_or(f64::NAN);
                    }
                }
                _ => return Err(format!("column '{}' is not numeric", col_name)),
            }
        }
        Ok(Tensor::new(data, vec![nrow, ncol]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RVal — every R2 runtime value
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum RVal {
    // Atomic vectors
    Numeric(Reals, Attrs),
    /// Single-precision float vector (Phase F.7). Opt-in via `as.single(x)`.
    /// Half the memory of `Numeric`. Arithmetic promotes to `Numeric` (f64)
    /// when mixed; pure Single + Single stays in f32.
    Single(Singles, Attrs),
    Integer(Ints, Attrs),
    Character(Vec<Character>, Attrs),
    Logical(Logicals, Attrs),
    Raw(Vec<u8>, Attrs),

    // Compound
    List(Vec<(Option<Arc<str>>, RVal)>),
    DataFrame(DataFrame),
    Matrix(Matrix),
    Factor(Factor),
    Tensor(Tensor),

    // Language objects
    Formula(Formula),
    Closure(Closure),
    BuiltinFn(Arc<str>),

    // Type system
    TypeDef(TypeDef),
    TypeInstance(TypeInstance),

    // Special
    Null,
    Env(EnvRef),
}

// ═══════════════════════════════════════════════════════════════════════
// Expr — AST nodes (what the parser produces, separate from RVal)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    NumLit(f64),
    IntLit(i32),
    StrLit(String),
    BoolLit(bool),
    FStringLit(Vec<FStringPart>),
    NaLit,
    NullLit,

    // Identifiers
    Symbol(Arc<str>),

    // Operations
    Unary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },

    // Assignment
    Assign { target: Box<Expr>, value: Box<Expr> },

    // Function call
    Call { func: Box<Expr>, args: Vec<CallArg> },

    // Indexing
    Index { object: Box<Expr>, indices: Vec<Option<Expr>> },
    DblIndex { object: Box<Expr>, index: Box<Expr> },
    Dollar { object: Box<Expr>, field: Arc<str> },

    // Namespace
    Namespace { pkg: Arc<str>, name: Arc<str> },

    // Pipe
    Pipe { lhs: Box<Expr>, rhs: Box<Expr> },

    // Control flow
    If { cond: Box<Expr>, then: Box<Expr>, else_: Option<Box<Expr>> },
    For { var: Arc<str>, iter: Box<Expr>, body: Box<Expr> },
    While { cond: Box<Expr>, body: Box<Expr> },
    Match { expr: Box<Expr>, arms: Vec<MatchArm> },
    Block(Vec<Expr>),

    // Functions
    FuncDef { params: Vec<Param>, body: Box<Expr> },
    Lambda { params: Vec<Param>, body: Box<Expr> },
    Return(Box<Expr>),

    // R2 type system
    TypeDef { name: Arc<str>, fields: Vec<FieldDef>, parent: Option<Arc<str>> },
    MethodDef(Method),

    // Try-catch
    TryCatch { body: Box<Expr>, var: Arc<str>, catch: Box<Expr> },

    // Control
    Break,
    Next,
    Dots,
}

#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<Arc<str>>,
    pub value: Expr,
}

/// Evaluated argument — used at runtime after expressions are evaluated to values
#[derive(Debug, Clone)]
pub struct EvalArg {
    pub name: Option<Arc<str>>,
    pub value: RVal,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub patterns: Vec<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum FStringPart {
    Literal(String),
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Pos, Not }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow, Mod, IntDiv,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or, AndShort, OrShort,
    Colon, Tilde, MatMul,
}

// ── Error mode ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorMode {
    Strict,
    Lenient,
}

// ── Helpers ──────────────────────────────────────────────────────────

pub fn rval_length(v: &RVal) -> usize {
    match v {
        RVal::Numeric(v, _) => v.len(),
        RVal::Single(v, _) => v.len(),
        RVal::Integer(v, _) => v.len(),
        RVal::Character(v, _) => v.len(),
        RVal::Logical(v, _) => v.len(),
        RVal::Raw(v, _) => v.len(),
        RVal::List(v) => v.len(),
        RVal::DataFrame(df) => df.nrow(),
        RVal::Factor(f) => f.codes.len(),
        RVal::Matrix(m) => m.nrow * m.ncol,
        RVal::Tensor(t) => t.numel(),
        RVal::Null => 0,
        _ => 1,
    }
}

pub fn rnum(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }
pub fn rint(x: i32) -> RVal { RVal::Integer(vec![Some(x)].into(), Attrs::default()) }
pub fn rstr(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }
pub fn rbool(b: bool) -> RVal { RVal::Logical(vec![Some(b)].into(), Attrs::default()) }
pub fn rna() -> RVal { RVal::Numeric(vec![None].into(), Attrs::default()) }
pub fn rnums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default()) }
pub fn rints(v: &[i32]) -> RVal { RVal::Integer(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }

/// Central numeric formatting: 7 decimal places, scientific for extreme values
pub fn fmt_num(n: f64) -> String {
    if n.is_nan() { return "NaN".into(); }
    if n.is_infinite() { return if n > 0.0 { "Inf".into() } else { "-Inf".into() }; }
    if n == 0.0 { return "0".into(); }
    let abs = n.abs();
    if abs >= 1e15 || (abs < 1e-4 && abs > 0.0) {
        // Scientific notation
        let s = format!("{:.7e}", n);
        if let Some(pos) = s.find('e') {
            let mantissa = s[..pos].trim_end_matches('0').trim_end_matches('.').to_string();
            let exp = &s[pos..];
            format!("{}{}", mantissa, exp)
        } else { s }
    } else if (n - n.round()).abs() < 1e-10 && abs < 1e12 {
        // Integer-valued float: show without decimals.
        // Bug fix: use `n.round()` not `n as i64` — the latter truncates
        // toward zero so 0.9999999999999998 became "0" instead of "1".
        format!("{}", n.round() as i64)
    } else {
        // Fixed notation: smart decimal places based on magnitude
        let decimals = if abs >= 100.0 { 4 } else if abs >= 10.0 { 5 } else if abs >= 1.0 { 6 } else { 7 };
        let s = format!("{:.prec$}", n, prec = decimals);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

// ── Display ──────────────────────────────────────────────────────────

impl fmt::Display for RVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RVal::Null => write!(f, "NULL"),
            RVal::Numeric(v, attrs) => {
                // If vector has names, display R-style: names on top, values below
                if let Some(names) = &attrs.names {
                    if names.len() == v.len() && !names.is_empty() {
                        let strs: Vec<String> = v.iter().map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect();
                        let widths: Vec<usize> = names.iter().zip(strs.iter()).map(|(n, s)| n.len().max(s.len()) + 1).collect();
                        // Names row
                        for (i, name) in names.iter().enumerate() { write!(f, "{:>w$}", name, w = widths[i])?; }
                        writeln!(f)?;
                        // Values row
                        for (i, s) in strs.iter().enumerate() { write!(f, "{:>w$}", s, w = widths[i])?; }
                        return Ok(());
                    }
                }
                write_vec(f, v, |x| match x { Some(n) => fmt_num(*n), None => "NA".into() })
            }
            RVal::Single(v, _) => {
                // Print like Numeric but with `(single)` annotation
                // after the value list. f32-as-displayed loses precision
                // beyond ~7 digits; the `fmt_num` helper handles that.
                write_vec(f, v, |x| match x { Some(n) => fmt_num(*n as f64), None => "NA".into() })
            }
            RVal::Integer(v, attrs) => {
                if let Some(names) = &attrs.names {
                    if names.len() == v.len() && !names.is_empty() {
                        let strs: Vec<String> = v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect();
                        let widths: Vec<usize> = names.iter().zip(strs.iter()).map(|(n, s)| n.len().max(s.len()) + 1).collect();
                        for (i, name) in names.iter().enumerate() { write!(f, "{:>w$}", name, w = widths[i])?; }
                        writeln!(f)?;
                        for (i, s) in strs.iter().enumerate() { write!(f, "{:>w$}", s, w = widths[i])?; }
                        return Ok(());
                    }
                }
                write_vec(f, v, |x| match x { Some(n) => format!("{}", n), None => "NA".into() })
            }
            RVal::Character(v, _) => write_vec(f, v, |x| match x { Some(s) => format!("\"{}\"", s), None => "NA".into() }),
            RVal::Logical(v, _) => write_vec(f, v, |x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }),
            RVal::Tensor(t) => {
                write!(f, "Tensor {:?}", t.shape)?;
                if t.numel() <= 20 {
                    write!(f, " [")?;
                    for (i, v) in t.data.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", fmt_num(*v))?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            RVal::Matrix(m) => {
                let max_rows = m.nrow.min(20);
                let rn_width = format!("[{},]", max_rows).len();
                // Compute column widths
                let mut col_widths: Vec<usize> = Vec::new();
                let mut col_headers: Vec<String> = Vec::new();
                for c in 0..m.ncol {
                    let header = m.col_names.as_ref().and_then(|cn| cn.get(c)).map(|s| s.to_string()).unwrap_or(format!("[,{}]", c + 1));
                    let max_val = (0..max_rows).map(|r| fmt_num(m.get(r, c)).len()).max().unwrap_or(1);
                    col_widths.push(header.len().max(max_val));
                    col_headers.push(header);
                }
                // Header
                write!(f, "{:>w$}", "", w = rn_width)?;
                for (c, h) in col_headers.iter().enumerate() { write!(f, " {:>w$}", h, w = col_widths[c])?; }
                writeln!(f)?;
                // Rows
                for r in 0..max_rows {
                    let rn = m.row_names.as_ref().and_then(|rn| rn.get(r)).map(|s| s.to_string()).unwrap_or(format!("[{},]", r + 1));
                    write!(f, "{:>w$}", rn, w = rn_width)?;
                    for c in 0..m.ncol { write!(f, " {:>w$}", fmt_num(m.get(r, c)), w = col_widths[c])?; }
                    writeln!(f)?;
                }
                if m.nrow > 20 { writeln!(f, "... ({} more rows)", m.nrow - 20)?; }
                Ok(())
            }
            RVal::DataFrame(df) => {
                let nrow = df.nrow().min(20);
                let ncol = df.columns.len();
                let rn_width = format!("{}", nrow).len().max(1);

                // Compute column widths based on content and header
                let mut col_strs: Vec<Vec<String>> = Vec::new();
                let mut col_widths: Vec<usize> = Vec::new();
                let mut is_char: Vec<bool> = Vec::new();
                for (name, col) in &df.columns {
                    let elems: Vec<String> = (0..nrow).map(|r| fmt_elem(col, r)).collect();
                    let max_elem = elems.iter().map(|s| s.len()).max().unwrap_or(0);
                    let w = name.len().max(max_elem);
                    col_widths.push(w);
                    is_char.push(matches!(col, RVal::Character(..) | RVal::Factor(..)));
                    col_strs.push(elems);
                }

                // Header row
                write!(f, "{:>w$}", "", w = rn_width + 1)?;
                for (i, (name, _)) in df.columns.iter().enumerate() {
                    write!(f, " {:>w$}", name, w = col_widths[i])?;
                }
                writeln!(f)?;

                // Data rows
                for r in 0..nrow {
                    write!(f, "{:>w$}", r + 1, w = rn_width)?;
                    for c in 0..ncol {
                        if is_char[c] {
                            write!(f, " {:>w$}", col_strs[c][r], w = col_widths[c])?;
                        } else {
                            write!(f, " {:>w$}", col_strs[c][r], w = col_widths[c])?;
                        }
                    }
                    writeln!(f)?;
                }
                if df.nrow() > 20 { writeln!(f, "... ({} more rows)", df.nrow() - 20)?; }
                Ok(())
            }
            RVal::TypeInstance(inst) => {
                match inst.type_name.as_ref() {
                    "lm" | "glm" => {
                        writeln!(f, "\nCall: {}(formula)\n", inst.type_name)?;
                        writeln!(f, "Coefficients:")?;
                        if let Some(coefs) = inst.fields.get("coefficients") {
                            write!(f, "{}", coefs)?;
                        }
                        Ok(())
                    }
                    "rpart" | "rf" | "kmeans" | "prcomp" | "naive.bayes" | "gbm" | "cv" | "confusion" | "aov" | "anova" | "cor.test" | "shapiro.test" | "wilcox.test" | "fisher.test" | "htest" => {
                        write!(f, "<{} model>", inst.type_name)
                    }
                    _ => {
                        // User-defined types: show fields
                        writeln!(f, "<{}>", inst.type_name)?;
                        for (k, v) in &inst.fields { writeln!(f, "  ${}: {}", k, v)?; }
                        Ok(())
                    }
                }
            }
            RVal::List(items) => {
                for (i, (name, val)) in items.iter().enumerate() {
                    if let Some(n) = name { writeln!(f, "${}", n)?; } else { writeln!(f, "[[{}]]", i + 1)?; }
                    writeln!(f, "{}", val)?;
                }
                Ok(())
            }
            RVal::Factor(fct) => {
                let display_vals: Vec<String> = fct.codes.iter().map(|c| match c {
                    Some(idx) => fct.levels.get(*idx as usize).map(|s| s.to_string()).unwrap_or("NA".into()),
                    None => "NA".into(),
                }).collect();
                write_vec(f, &display_vals, |s| s.clone())?;
                write!(f, "\nLevels: {}", fct.levels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" "))
            }
            RVal::BuiltinFn(name) => {
                let sig = builtin_signature(name);
                write!(f, "{}", sig)
            }
            RVal::Closure(cls) => {
                let params: Vec<String> = cls.params.iter().map(|p| {
                    if p.dots { "...".into() }
                    else if p.default.is_some() { format!("{} = <default>", p.name) }
                    else { p.name.to_string() }
                }).collect();
                write!(f, "function({})\n{{\n    <user-defined>\n}}", params.join(", "))
            }
            _ => write!(f, "<{}>", self.type_name()),
        }
    }
}

fn write_vec<T>(f: &mut fmt::Formatter, v: &[T], fmt_fn: impl Fn(&T) -> String) -> fmt::Result {
    if v.is_empty() { return write!(f, "character(0)"); }
    let strs: Vec<String> = v.iter().map(&fmt_fn).collect();
    let mut pos = 0;
    while pos < strs.len() {
        write!(f, "[{}]", pos + 1)?;
        let mut used = format!("[{}]", pos + 1).len();
        while pos < strs.len() {
            let next = format!(" {}", strs[pos]);
            if used + next.len() > 80 && used > 4 { break; }
            write!(f, "{}", next)?;
            used += next.len();
            pos += 1;
        }
        if pos < strs.len() { writeln!(f)?; }
    }
    Ok(())
}

fn fmt_elem(col: &RVal, row: usize) -> String {
    match col {
        RVal::Numeric(v, _) => v.get(row).map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).unwrap_or_default(),
        RVal::Integer(v, _) => v.get(row).map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).unwrap_or_default(),
        RVal::Character(v, _) => v.get(row).map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).unwrap_or_default(),
        RVal::Logical(v, _) => v.get(row).map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).unwrap_or_default(),
        _ => "?".into(),
    }
}

impl RVal {
    pub fn type_name(&self) -> &'static str {
        match self {
            RVal::Numeric(..) => "numeric", RVal::Single(..) => "single",
            RVal::Integer(..) => "integer",
            RVal::Character(..) => "character", RVal::Logical(..) => "logical",
            RVal::Raw(..) => "raw", RVal::List(..) => "list",
            RVal::DataFrame(..) => "data.frame", RVal::Matrix(..) => "matrix",
            RVal::Factor(..) => "factor", RVal::Tensor(..) => "tensor",
            RVal::Formula(..) => "formula", RVal::Closure(..) => "function",
            RVal::BuiltinFn(..) => "builtin", RVal::TypeDef(..) => "type",
            RVal::TypeInstance(..) => "instance", RVal::Null => "NULL",
            RVal::Env(..) => "environment",
        }
    }

    /// Phase R.1 step 2 — coerce a numeric-ish `RVal` to `Vec<Real>`.
    /// Was a method on Engine (`Engine::as_reals`); moved here because it
    /// doesn't need engine state. Engine still has a thin wrapper for
    /// backward compatibility.
    pub fn as_reals(&self) -> Result<Vec<Real>, R2Err> {
        match self {
            RVal::Numeric(v, _)  => Ok(v.as_vec().clone()),
            // Single promotes to f64 on read (Phase F.7 promotion rule).
            RVal::Single(v, _)   => Ok(v.iter().map(|x| x.map(|n| n as f64)).collect()),
            RVal::Integer(v, _)  => Ok(v.iter().map(|x| x.map(|n| n as f64)).collect()),
            RVal::Logical(v, _)  => Ok(v.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect()),
            RVal::Matrix(m)      => Ok(m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect()),
            _ => Err(R2Err {
                msg: format!("cannot convert {} to numeric. If this is a data.frame column, use df$column_name", self.type_name()),
                kind: ErrKind::Type,
            }),
        }
    }

    /// Coerce to `Vec<Single>` (f32). Lossy narrowing from Numeric.
    pub fn as_singles(&self) -> Result<Vec<Single>, R2Err> {
        match self {
            RVal::Single(v, _)   => Ok(v.as_vec().clone()),
            RVal::Numeric(v, _)  => Ok(v.iter().map(|x| x.map(|n| n as f32)).collect()),
            RVal::Integer(v, _)  => Ok(v.iter().map(|x| x.map(|n| n as f32)).collect()),
            RVal::Logical(v, _)  => Ok(v.iter().map(|x| x.map(|b| if b { 1.0_f32 } else { 0.0 })).collect()),
            _ => Err(R2Err {
                msg: format!("cannot convert {} to single", self.type_name()),
                kind: ErrKind::Type,
            }),
        }
    }

    /// Coerce to a `Vec<Logical>`.
    pub fn as_logicals(&self) -> Result<Vec<Logical>, R2Err> {
        match self {
            RVal::Logical(v, _) => Ok(v.as_vec().clone()),
            RVal::Numeric(v, _) => Ok(v.iter().map(|x| x.map(|n| n != 0.0)).collect()),
            _ => Err(R2Err {
                msg: format!("cannot coerce {} to logical", self.type_name()),
                kind: ErrKind::Type,
            }),
        }
    }

    /// Extract the first numeric scalar (NA-aware).
    pub fn scalar_f64(&self) -> Result<Real, R2Err> {
        Ok(self.as_reals()?.into_iter().next().unwrap_or(None))
    }

    /// Iterate an RVal as a sequence of single-element items. Used by
    /// the apply family. Was an Engine method; moved here as an RVal
    /// method since it doesn't need engine state.
    pub fn to_items(&self) -> Result<Vec<RVal>, R2Err> {
        match self {
            RVal::Integer(v, _) => Ok(v.iter().map(|x| RVal::Integer(vec![*x].into(), Attrs::default())).collect()),
            RVal::Numeric(v, _) => Ok(v.iter().map(|x| RVal::Numeric(vec![*x].into(), Attrs::default())).collect()),
            RVal::Character(v, _) => Ok(v.iter().map(|x| RVal::Character(vec![x.clone()], Attrs::default())).collect()),
            RVal::List(v) => Ok(v.iter().map(|(_, val)| val.clone()).collect()),
            _ => Err(R2Err {
                msg: format!("cannot iterate over {}", self.type_name()),
                kind: ErrKind::Runtime,
            }),
        }
    }

    /// Phase F.1 — produce a columnar `ColumnarF64` view of any numeric-ish
    /// `RVal`. Materializes by converting (Vec<Option<f64>> → ColumnarF64).
    /// Future F.2+ will store the columnar form directly inside RVal,
    /// making this a zero-copy borrow.
    ///
    /// Returns `None` for non-numeric types (caller falls back to existing
    /// `as_reals` path or errors).
    pub fn to_columnar(&self) -> Option<r2_arrow::ColumnarF64> {
        match self {
            // F.3a: borrow the slice — no Vec clone before walking.
            RVal::Numeric(v, _) => Some(r2_arrow::ColumnarF64::from_option_slice(v)),
            RVal::Integer(v, _) => {
                // Integer → Real conversion still requires one allocation.
                let opts: Vec<Option<f64>> = v.iter().map(|x| x.map(|n| n as f64)).collect();
                Some(r2_arrow::ColumnarF64::from_options(opts))
            }
            RVal::Logical(v, _) => {
                let opts: Vec<Option<f64>> = v.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect();
                Some(r2_arrow::ColumnarF64::from_options(opts))
            }
            RVal::Matrix(m) => {
                // Already contiguous f64; treat as dense column. NaN entries
                // are *valid values* in Matrix (same as r2-engine convention).
                Some(r2_arrow::ColumnarF64::from_vec(m.data.clone()))
            }
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2); // col-major: [[1,2],[3,4]]
        let b = Matrix::new(vec![5.0, 7.0, 6.0, 8.0], 2, 2);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.nrow, 2);
        assert_eq!(c.ncol, 2);
    }

    #[test]
    fn test_matrix_solve() {
        // 2x + 3y = 8, x + y = 3 → x=1, y=2
        let a = Matrix::new(vec![2.0, 1.0, 3.0, 1.0], 2, 2);
        let b = vec![8.0, 3.0];
        let x = a.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = m.transpose();
        assert_eq!(t.nrow, 3);
        assert_eq!(t.ncol, 2);
    }

    #[test]
    fn test_tensor_basic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.numel(), 6);
    }

    #[test]
    fn test_tensor_relu() {
        let t = Tensor::new(vec![-1.0, 0.0, 1.0, -2.0, 3.0, -0.5], vec![6]);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_tensor_sigmoid() {
        let t = Tensor::new(vec![0.0], vec![1]);
        let s = t.sigmoid();
        assert!((s.data[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.softmax();
        assert!((s.sum() - 1.0).abs() < 1e-10);
        assert!(s.data[2] > s.data[1] && s.data[1] > s.data[0]);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul_2d(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.get(&[0, 0]), 19.0); // 1*5 + 2*7
    }

    #[test]
    fn test_tensor_from_matrix() {
        let m = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2); // col-major [[1,2],[3,4]]
        let t = Tensor::from_matrix(&m);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 2.0);
        assert_eq!(t.get(&[1, 0]), 3.0);
        assert_eq!(t.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_matrix_crossprod() {
        // X^T * X should be symmetric
        let x = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let xtx = x.crossprod();
        assert_eq!(xtx.nrow, 2);
        assert_eq!(xtx.ncol, 2);
    }
}

/// Return R2-style function signature for built-in functions
pub fn builtin_signature(name: &str) -> String {
    let sig = match name {
        // Statistics
        "lm" => "function(formula, data, subset, weights, na.action,\n    method = \"qr\", model = TRUE, x = FALSE, y = FALSE)\n{\n    # Linear regression via normal equations: beta = (X'X)^-1 X'y\n    # Returns: coefficients, residuals, fitted.values, r.squared,\n    #          adj.r.squared, sigma, std.errors, t.values, p.values\n    .Built-in\n}",
        "glm" => "function(formula, data, family = \"gaussian\", subset, weights)\n{\n    # Generalized linear model (binomial, poisson, gaussian)\n    # Uses IRLS (Iteratively Reweighted Least Squares)\n    # Returns: coefficients, deviance, fitted.values\n    .Built-in\n}",
        "t.test" => "function(x, y = NULL, mu = 0, alternative = \"two.sided\",\n    conf.level = 0.95, paired = FALSE)\n{\n    # Student's t-test (one-sample, two-sample, paired)\n    # Returns: statistic, p.value, conf.int, estimate\n    .Built-in\n}",
        "chisq.test" => "function(x, p = NULL, correct = TRUE)\n{\n    # Chi-squared test\n    # x = vector: goodness-of-fit test\n    # x = matrix: test of independence\n    # correct: Yates' continuity correction for 2x2 tables\n    # Returns: statistic, p.value, parameter (df)\n    .Built-in\n}",
        "aov" => "function(formula, data)\n{\n    # One-way Analysis of Variance\n    # Tests if group means differ significantly\n    # Returns: f.statistic, p.value, ss.between, ss.within\n    .Built-in\n}",
        "anova" => "function(model)\n{\n    # ANOVA table for lm/glm model\n    # Shows: Df, Sum Sq, Mean Sq, F value, Pr(>F)\n    .Built-in\n}",
        "cor.test" => "function(x, y, method = \"pearson\")\n{\n    # Test if correlation is significantly different from zero\n    # Returns: estimate (r), statistic (t), p.value, df\n    .Built-in\n}",
        "shapiro.test" => "function(x)\n{\n    # Shapiro-Wilk test for normality\n    # H0: data is normally distributed\n    # Returns: statistic (W), p.value\n    .Built-in\n}",
        "wilcox.test" => "function(x, y = NULL, mu = 0, alternative = \"two.sided\")\n{\n    # Wilcoxon rank-sum (2-sample) or signed-rank (1-sample) test\n    # Non-parametric alternative to t.test\n    # Returns: statistic, p.value\n    .Built-in\n}",
        "fisher.test" => "function(x)\n{\n    # Fisher's exact test for 2x2 contingency tables\n    # Returns: p.value, estimate (odds ratio)\n    .Built-in\n}",
        // ML
        "rpart" => "function(formula, data, max_depth = 5, min_samples = 5,\n    type = \"auto\")\n{\n    # Decision tree (CART)\n    # Auto-detects classification vs regression\n    # Returns: predictions, tree structure\n    .Built-in\n}",
        "rf" => "function(formula, data, ntrees = 100, max_depth = 10,\n    type = \"classification\")\n{\n    # Random forest (bootstrap aggregation of decision trees)\n    # Returns: predictions, feature importance\n    .Built-in\n}",
        "gbm" => "function(formula, data, ntrees = 100, learning_rate = 0.1,\n    max_depth = 3, subsample = 0.8, loss = \"squared\")\n{\n    # Gradient boosted trees (XGBoost-style)\n    # loss: \"squared\", \"logistic\", \"huber\"\n    # Returns: predictions, importance, train.loss\n    .Built-in\n}",
        "kmeans" => "function(x, centers, iter.max = 100)\n{\n    # K-means clustering\n    # Returns: cluster, centers, withinss, totss, size\n    .Built-in\n}",
        "knn" => "function(train, test, labels, k = 3)\n{\n    # K-nearest neighbors classification\n    # Returns: predicted class labels\n    .Built-in\n}",
        "prcomp" => "function(x, center = TRUE, scale. = FALSE)\n{\n    # Principal Component Analysis\n    # Returns: sdev, eigenvalues, prop.variance\n    .Built-in\n}",
        "naive.bayes" => "function(x, y)\n{\n    # Gaussian Naive Bayes classifier\n    # Returns: classes, priors, means, vars\n    .Built-in\n}",
        // Data
        "read.csv" => "function(file, header = TRUE, sep = \",\")\n{\n    # Read CSV file into data.frame\n    # Handles: quoted fields, NA values, type inference\n    .Built-in\n}",
        "write.csv" => "function(x, file)\n{\n    # Write data.frame to CSV file\n    .Built-in\n}",
        "filter" => "function(df, mask)\n{\n    # Keep rows where mask is TRUE\n    .Built-in\n}",
        "select" => "function(df, ...)\n{\n    # Keep only named columns\n    .Built-in\n}",
        "mutate" => "function(df, ...)\n{\n    # Add or modify columns (named arguments)\n    .Built-in\n}",
        "summary" => "function(object, ...)\n{\n    # Summary statistics — auto-dispatches by class:\n    # numeric:    Min, 1Q, Median, Mean, 3Q, Max\n    # data.frame: per-column summary\n    # lm/glm:     coefficients, std.errors, t, p, R2, F\n    # rpart/rf/gbm: model-specific summary\n    # kmeans:     cluster sizes, within-SS\n    .Built-in\n}",
        "plot" => "function(x, y = NULL, main = \"\", xlab = \"\", ylab = \"\",\n    col = \"steelblue\")\n{\n    # Scatter plot (SVG output)\n    # Auto-dispatches: lm->residuals, gbm->loss curve\n    .Built-in\n}",
        // Core
        "mean" => "function(x, na.rm = FALSE)\n{\n    # Arithmetic mean\n    .Built-in\n}",
        "sd" => "function(x, na.rm = FALSE)\n{\n    # Standard deviation (n-1 denominator)\n    .Built-in\n}",
        "var" => "function(x, na.rm = FALSE)\n{\n    # Variance (n-1 denominator)\n    .Built-in\n}",
        "cor" => "function(x, y)\n{\n    # Pearson correlation coefficient\n    .Built-in\n}",
        "c" => "function(...)\n{\n    # Combine values into a vector\n    .Built-in\n}",
        "print" => "function(x, ...)\n{\n    # Print value to console\n    .Built-in\n}",
        "cat" => "function(..., sep = \" \")\n{\n    # Concatenate and print\n    .Built-in\n}",
        "paste" => "function(..., sep = \" \")\n{\n    # Concatenate strings with separator\n    .Built-in\n}",
        "length" => "function(x)\n{\n    # Length of vector/list\n    .Built-in\n}",
        "head" => "function(x, n = 6)\n{\n    # First n elements/rows\n    .Built-in\n}",
        "tail" => "function(x, n = 6)\n{\n    # Last n elements/rows\n    .Built-in\n}",
        "data.frame" => "function(...)\n{\n    # Create data frame from named vectors\n    .Built-in\n}",
        "matrix" => "function(data, nrow, ncol, byrow = FALSE)\n{\n    # Create matrix (column-major by default)\n    .Built-in\n}",
        _ => return format!("function(...)\n{{\n    .Built-in(\"{}\")\n}}", name),
    };
    sig.to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// List-dispatch metadata — Phase L (auto-parallel over heterogeneous lists)
// ═══════════════════════════════════════════════════════════════════════
//
// A `list(a=..., b=...)` in R is a labeled heterogeneous container. When
// such a list is passed to an apply-family function (`lapply`/`sapply`),
// the natural parallelism axis is **across components** — each
// component's processing is an independent unit. R2's Oracle can pick
// Serial vs Rayon for this fork-join based on the aggregate work of the
// components.
//
// `ListMeta` is a lightweight snapshot of the per-component shape used
// by Oracle. Built on demand (not embedded in `RVal::List`, so legacy
// callers see no change in the type). Computed in one O(n) pass over
// the components — n is typically <10 so this is cheap.

/// Per-component shape information extracted from `RVal::List`.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Component label, if named (`list(a = ...)`); `None` for positional.
    pub name: Option<Arc<str>>,
    /// The component's RVal variant as a stable string tag.
    /// Used by Oracle and apply-family to decide processing strategy.
    pub kind: &'static str,
    /// Length: vector size, df nrow, matrix nrow * ncol, etc.
    /// Sub-list length is 1 by convention (don't recurse — Oracle treats
    /// nested lists as one work unit and re-enters at apply time).
    pub len: usize,
}

/// Aggregate metadata over a list's components. Built via `list_meta()`.
#[derive(Debug, Clone)]
pub struct ListMeta {
    /// Per-component info, same order as the list.
    pub components: Vec<ComponentInfo>,
    /// Sum of component lengths — what Oracle uses for the parallel
    /// threshold check (a list of [1M numeric + 100 char] has aggregate
    /// work dominated by the big numeric component).
    pub total_work: usize,
    /// `Some(kind)` when every component shares the same RVal variant;
    /// `None` otherwise. Lets future passes specialize code (e.g. fuse
    /// per-component math when all numeric) without a runtime check.
    pub homogeneous_kind: Option<&'static str>,
}

/// Build a `ListMeta` snapshot for a list's components.
///
/// O(n_components) — typically n < 10 so the cost is negligible vs the
/// dispatch decision it enables.
pub fn list_meta(items: &[(Option<Arc<str>>, RVal)]) -> ListMeta {
    let mut components = Vec::with_capacity(items.len());
    let mut total_work = 0usize;
    let mut first_kind: Option<&'static str> = None;
    let mut homogeneous = true;
    for (name, val) in items {
        let kind = val.type_name();
        let len = match val {
            RVal::Numeric(v, _)   => v.len_fast(),
            RVal::Integer(v, _)   => v.len(),
            RVal::Logical(v, _)   => v.len(),
            RVal::Character(v, _) => v.len(),
            RVal::Raw(v, _)       => v.len(),
            RVal::List(v)         => v.len(),
            RVal::Matrix(m)       => m.nrow.saturating_mul(m.ncol),
            RVal::DataFrame(df)   => df.nrow().saturating_mul(df.ncol()),
            _ => 1,
        };
        match first_kind {
            None => first_kind = Some(kind),
            Some(k) if k != kind => homogeneous = false,
            _ => {}
        }
        total_work = total_work.saturating_add(len);
        components.push(ComponentInfo { name: name.clone(), kind, len });
    }
    ListMeta {
        components,
        total_work,
        homogeneous_kind: if homogeneous { first_kind } else { None },
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Routed console output
//
// Compute crates (r2-stats, …) print formatted results but cannot reach
// the engine's OutputSink. They emit through this thread-local hook
// instead of writing straight to stdout, so a frontend can capture the
// output:
//   * GUI  — installs a hook forwarding each line to its ConsoleBuffer
//            (otherwise the output is lost: a windowed app has no console).
//   * CLI  — leaves the hook unset → falls back to stdout / stderr.
//
// Output is line-buffered: text is accumulated and only *complete* lines
// (split on '\n') are dispatched, each WITHOUT the trailing newline —
// matching `ConsoleBuffer::push_output` and correctly reassembling
// piecewise `print!` + `println!` sequences (e.g. table rows).
// ═══════════════════════════════════════════════════════════════════════
pub mod out {
    use std::cell::RefCell;

    thread_local! {
        static HOOK: RefCell<Option<Box<dyn FnMut(&str, bool)>>> = RefCell::new(None);
        static LINEBUF: RefCell<String> = RefCell::new(String::new());
    }

    /// Install (or clear, with `None`) the per-thread output hook. The
    /// closure receives one complete line (no trailing newline) and a
    /// flag: `true` = error stream, `false` = standard output.
    pub fn set_output_hook(hook: Option<Box<dyn FnMut(&str, bool)>>) {
        HOOK.with(|h| *h.borrow_mut() = hook);
    }

    fn dispatch(line: &str, is_err: bool) {
        HOOK.with(|h| {
            let mut slot = h.borrow_mut();
            if let Some(f) = slot.as_mut() {
                f(line, is_err);
            } else if is_err {
                eprintln!("{}", line);
            } else {
                println!("{}", line);
            }
        });
    }

    fn write_routed(s: &str, is_err: bool) {
        LINEBUF.with(|lb| {
            let mut buf = lb.borrow_mut();
            buf.push_str(s);
            while let Some(pos) = buf.find('\n') {
                let line: String = buf[..pos].to_string();
                buf.drain(..=pos);
                dispatch(&line, is_err);
            }
        });
    }

    /// Emit standard output through the routed sink (line-buffered).
    pub fn rout(s: &str) { write_routed(s, false); }
    /// Emit error output through the routed sink (line-buffered).
    pub fn rerr(s: &str) { write_routed(s, true); }

    thread_local! {
        static CLEAR_HOOK: RefCell<Option<Box<dyn FnMut()>>> = RefCell::new(None);
    }

    /// Install (or clear) the per-thread "clear console" hook. The GUI
    /// installs one that empties its `ConsoleBuffer`; the CLI leaves it
    /// unset and falls back to an ANSI clear-screen sequence.
    pub fn set_clear_hook(hook: Option<Box<dyn FnMut()>>) {
        CLEAR_HOOK.with(|h| *h.borrow_mut() = hook);
    }

    /// Clear the console — invoked by the `clear()` / `cls()` builtin.
    /// Routes to the installed hook (GUI buffer); otherwise emits the
    /// ANSI "clear screen + scrollback + home" sequence for terminals.
    pub fn request_clear() {
        CLEAR_HOOK.with(|h| {
            let mut slot = h.borrow_mut();
            if let Some(f) = slot.as_mut() {
                f();
            } else {
                // \x1b[2J clear screen, \x1b[3J scrollback, \x1b[H home.
                print!("\x1b[2J\x1b[3J\x1b[H");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        });
    }
}

#[cfg(test)]
mod out_tests {
    use super::out;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn routed_output_is_line_buffered_and_captured() {
        let captured: Rc<RefCell<Vec<(String, bool)>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = captured.clone();
        out::set_output_hook(Some(Box::new(move |line: &str, is_err: bool| {
            sink.borrow_mut().push((line.to_string(), is_err));
        })));

        // println-style: one complete line, trailing newline stripped.
        out::rout("Welch Two Sample t-test\n");
        // print-style fragments: joined until the newline flushes them.
        out::rout("mean of x = 4.86");
        out::rout(", mean of y = 6.06\n");
        // error stream carries the is_err flag.
        out::rerr("a warning\n");

        out::set_output_hook(None);

        let got = captured.borrow();
        assert_eq!(got[0], ("Welch Two Sample t-test".to_string(), false));
        assert_eq!(got[1], ("mean of x = 4.86, mean of y = 6.06".to_string(), false));
        assert_eq!(got[2], ("a warning".to_string(), true));
    }
}
