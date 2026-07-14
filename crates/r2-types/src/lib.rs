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

// ── Module layout ────────────────────────────────────────────────────
//
// The type system is large, so it's split into cohesive modules and
// re-exported flat from the crate root — every `r2_types::X` path (and
// the pervasive `use r2_types::*`) is unchanged. `lib.rs` keeps the
// interlinked core: the central value structs (Attrs/Factor/Env/Closure/
// DataFrame/…), `RVal` and its Display/methods, and the `out` sink.
//
//   error    — R2Err / ErrKind + the global interrupt flag
//   columnar — NA-aware element aliases + Reals/Singles/Ints/Logicals
//   matrix   — Matrix (2D + linear algebra)
//   tensor   — Tensor (N-dimensional, ML)
//   expr     — Expr AST + call/arg/operator enums
mod error;
mod columnar;
mod matrix;
mod tensor;
mod expr;

pub use error::*;
pub use columnar::*;
pub use matrix::*;
pub use tensor::*;
pub use expr::*;

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

/// Environments are R's mutable scope objects: a frame of bindings plus a
/// parent pointer. Interior mutability (RwLock) makes every `EnvRef` a LIVE
/// reference — a closure that captured its defining environment sees (and
/// can mutate, via `<<-`) the same frame on every call. This is what makes
/// stateful closures (counters, accumulators, factories) work like R.
#[derive(Debug)]
pub struct Env {
    pub name: Option<Arc<str>>,
    pub bindings: std::sync::RwLock<HashMap<Arc<str>, RVal>>,
    pub parent: Option<EnvRef>,
    pub locked: bool,
}

impl Clone for Env {
    fn clone(&self) -> Self {
        Env {
            name: self.name.clone(),
            bindings: std::sync::RwLock::new(self.bindings.read().unwrap().clone()),
            parent: self.parent.clone(),
            locked: self.locked,
        }
    }
}

impl Env {
    pub fn new_global() -> EnvRef {
        Arc::new(Env { name: Some(Arc::from(".GlobalEnv")), bindings: std::sync::RwLock::new(HashMap::new()), parent: None, locked: false })
    }
    pub fn new_child(parent: EnvRef, name: Option<&str>) -> EnvRef {
        Arc::new(Env { name: name.map(Arc::from), bindings: std::sync::RwLock::new(HashMap::new()), parent: Some(parent), locked: false })
    }
    /// Walk the scope chain; clone the value out (bindings live behind a lock).
    pub fn lookup(&self, name: &str) -> Option<RVal> {
        if let Some(v) = self.bindings.read().unwrap().get(name) { return Some(v.clone()); }
        self.parent.as_ref().and_then(|p| p.lookup(name))
    }
    /// Bind `name` in THIS frame (R's `<-` inside the owning scope).
    pub fn set(&self, name: Arc<str>, val: RVal) {
        self.bindings.write().unwrap().insert(name, val);
    }
    /// R's `<<-`: rebind where `name` is found in the ENCLOSING chain
    /// (starting at this frame's parent). Returns false if not found —
    /// the caller then binds in the global environment, as R does.
    pub fn set_in_enclosing(&self, name: &Arc<str>, val: &RVal) -> bool {
        let mut cur = self.parent.clone();
        while let Some(e) = cur {
            if e.bindings.read().unwrap().contains_key(name.as_ref()) {
                e.bindings.write().unwrap().insert(name.clone(), val.clone());
                return true;
            }
            cur = e.parent.clone();
        }
        false
    }
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.read().unwrap().contains_key(name)
    }
    pub fn remove(&self, name: &str) -> Option<RVal> {
        self.bindings.write().unwrap().remove(name)
    }
    /// Snapshot of the frame's own names (for ls()/exists()).
    pub fn own_names(&self) -> Vec<Arc<str>> {
        self.bindings.read().unwrap().keys().cloned().collect()
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
    /// `(*const f64, *const f64, i64) -> f64` — TWO same-length f64 vectors in,
    /// one scalar out. Fused binary map-reduce, e.g. `sum(x*w)` (dot product).
    /// Phase J.2.
    Vector2ToScalar,
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
    /// `(*const f64, *mut f64, i64) -> f64` — Phase J.3 indexed-**store** map:
    /// an imperative loop `for(i in 1:length(x)) y[i] <- f(x[i])` with an
    /// arbitrary (multi-statement) body writing `y[i]` via a real `Store`. The
    /// f64 return is a dummy (the loop yields NULL); the out buffer carries the
    /// result. One input vector.
    IndexedStoreMap1,
    /// `(*const f64, *const f64, *mut f64, i64) -> f64` — two-input indexed
    /// store map, e.g. `for(i in 1:length(x)) y[i] <- x[i] + w[i]`. Phase J.3.
    IndexedStoreMap2,
    /// `(m_ptr, nrow, ncol, v_ptr, out_ptr)` — Phase J.4 matrix state: a
    /// column-major n×p matrix + an n-vector in, a p-vector out. The compiled
    /// body is an iterative kernel with `X %*% b` / `t(X) %*% r` statements
    /// (multi-parameter gradient descent / IRLS-core shape).
    MatVecIterOut,
}

/// Object-safe handle to a JIT-compiled function. Lives in r2-jit (and any
/// future backends); declared here so `r2-engine` can hold one without a
/// direct dependency on `r2-jit`.
// `Send + Sync`: a JIT handle wraps immutable, reentrant native code (a raw
// function pointer kept alive by an owned module). Sharing it across threads
// for *calling* is safe — the code is read-only after finalize and has no
// shared mutable state. Required so `Engine` (which caches handles) can be
// used by parallel workers (Phase P). The concrete impl asserts this via
// `unsafe impl Send + Sync` with the same justification.
pub trait JitHandle: std::fmt::Debug + Send + Sync {
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
    /// Vector2ToScalar dispatch (Phase J.2) — fused binary map-reduce.
    /// SAFETY: `a_ptr`/`b_ptr` must each reference `len` valid f64s.
    unsafe fn try_call_vec2(&self, _a_ptr: *const f64, _b_ptr: *const f64, _len: i64) -> Option<f64> { None }
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
    /// IndexedStoreMap1 dispatch (Phase J.3). SAFETY: `in_ptr`/`out_ptr` each
    /// reference at least `len` valid f64s; out_ptr is written to.
    unsafe fn try_call_ixstore1(&self, _in_ptr: *const f64, _out_ptr: *mut f64, _len: i64) -> bool { false }
    /// IndexedStoreMap2 dispatch (Phase J.3). SAFETY: all three pointers each
    /// reference at least `len` valid f64s; out_ptr is written to.
    unsafe fn try_call_ixstore2(&self, _a_ptr: *const f64, _b_ptr: *const f64, _out_ptr: *mut f64, _len: i64) -> bool { false }
    /// MatVecIterOut dispatch (Phase J.4 matrix state). SAFETY: `m_ptr` holds
    /// nrow*ncol f64s (column-major), `v_ptr` nrow f64s, `out_ptr` ncol f64s.
    unsafe fn try_call_matvec(&self, _m_ptr: *const f64, _nrow: i64, _ncol: i64, _v_ptr: *const f64, _out_ptr: *mut f64) -> bool { false }
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
    /// A quoted, unevaluated expression — R's LANGSXP/EXPRSXP. Produced by
    /// `quote()`/`parse()`/`body()`; consumed by `eval()`/`deparse()`.
    /// `Arc<Expr>` is how `Closure` already stores its body, so this is a
    /// cheap, shareable handle onto the AST. (Phase L.1.)
    Lang(Arc<Expr>),

    // Type system
    TypeDef(TypeDef),
    TypeInstance(TypeInstance),

    // Special
    Null,
    Env(EnvRef),
}

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
// Columnar-first: dense f64 slices have no NAs by construction, so build
// the Arrow form directly (tight memcpy); the boxed Vec<Option> view only
// materialises if a caller later asks for &[Real].
pub fn rnums(v: &[f64]) -> RVal { RVal::Numeric(Reals::from_dense_f64(v.to_vec()), Attrs::default()) }
pub fn rints(v: &[i32]) -> RVal { RVal::Integer(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }

/// Deparse an `Expr` back into R source text — the inverse of parsing.
/// Lives here (next to `Expr`) so both `RVal::Lang`'s `Display` and the
/// engine's `deparse()` builtin share one implementation. (Phase L.1;
/// formerly `fmt_expr` in r2-engine/formula.rs.)
pub fn deparse(e: &Expr) -> String {
    match e {
        Expr::Symbol(s) => s.to_string(),
        Expr::NumLit(n) => fmt_num(*n),
        Expr::IntLit(n) => format!("{}", n),
        Expr::StrLit(s) => format!("\"{}\"", s),
        Expr::BoolLit(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        Expr::NaLit => "NA".into(),
        Expr::NullLit => "NULL".into(),
        Expr::Binary { op, lhs, rhs } => {
            let opstr = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
                BinOp::Pow => "^", BinOp::Mod => "%%", BinOp::IntDiv => "%/%",
                BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::Lt => "<", BinOp::Gt => ">",
                BinOp::Le => "<=", BinOp::Ge => ">=",
                BinOp::And => "&", BinOp::Or => "|",
                BinOp::AndShort => "&&", BinOp::OrShort => "||",
                BinOp::Tilde => "~", BinOp::MatMul => "%*%",
                BinOp::Colon => ":",
            };
            format!("{} {} {}", deparse(lhs), opstr, deparse(rhs))
        }
        Expr::Call { func, args } => {
            let fname = deparse(func);
            let parts: Vec<String> = args.iter().map(|a| match &a.name {
                Some(n) => format!("{} = {}", n, deparse(&a.value)),
                None => deparse(&a.value),
            }).collect();
            format!("{}({})", fname, parts.join(", "))
        }
        Expr::Dollar { object, field } => format!("{}${}", deparse(object), field),
        Expr::Index { object, indices } => {
            let parts: Vec<String> = indices.iter().map(|i| match i {
                Some(e) => deparse(e),
                None => String::new(),
            }).collect();
            format!("{}[{}]", deparse(object), parts.join(", "))
        }
        _ => "<expr>".into(),
    }
}

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
        // Fixed notation: 7 SIGNIFICANT digits like R's default (digits=7),
        // so 0.03177295 keeps all 7 (the old per-magnitude table capped
        // sub-1 values at 7 DECIMALS, silently dropping a digit for
        // leading-zero values like 0.03…).
        let decimals = (6 - abs.log10().floor() as i32).clamp(0, 15) as usize;
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
            // A quoted expression prints as its deparsed source (R's behaviour).
            RVal::Lang(e) => write!(f, "{}", deparse(e)),
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
            RVal::Lang(..) => "call",
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
            // `as.numeric(factor)` returns the integer codes (1-based), as in
            // R — needed by manova/aov/etc. to build group design matrices.
            RVal::Factor(f)      => Ok(f.codes.iter().map(|c| c.map(|i| i as f64 + 1.0)).collect()),
            // A list of length-1 numerics flattens to a vector (R's
            // `as.numeric(list(1,2,3))`, and `as.numeric(tapply(...))`).
            RVal::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for (_, v) in items {
                    let r = v.as_reals()?;
                    if r.len() != 1 {
                        return Err(R2Err { msg: "cannot coerce list with non-scalar elements to numeric".into(), kind: ErrKind::Type });
                    }
                    out.push(r[0]);
                }
                Ok(out)
            }
            // Exhaustive on purpose (no `_`): adding a new RVal variant must
            // be a COMPILE error here, not a silent runtime "cannot convert".
            // That is exactly the gap that let factors slip past for so long.
            RVal::Character(..) | RVal::Raw(..) | RVal::DataFrame(..)
            | RVal::Tensor(..) | RVal::Formula(..) | RVal::Closure(..) | RVal::BuiltinFn(..)
            | RVal::Lang(..) | RVal::TypeDef(..) | RVal::TypeInstance(..) | RVal::Null
            | RVal::Env(..) => Err(R2Err {
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
            // A data.frame iterates over its COLUMNS (R's list-like semantics),
            // so lapply/sapply/Map fold over columns, not cells.
            RVal::DataFrame(df) => Ok(df.columns.iter().map(|(_, val)| val.clone()).collect()),
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
