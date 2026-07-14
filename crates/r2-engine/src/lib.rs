// Builtin function names mirror R's exact identifiers (`Sys.time`,
// `colMeans`, `rowSums`, …). The Rust function names follow the R
// names so the registration site is grep-friendly; the snake_case
// lint is silenced crate-wide rather than scattered per item.
#![allow(non_snake_case)]

// R2 Engine — layered namespace resolution for proper function masking
// Both <- and = work for assignment (user's choice)
//
// Resolution order (top wins):
//   1. User-defined functions in global environment
//   2. Last loaded addon package
//   3. ... earlier addon packages ...
//   4. Base libraries (stats, graphics, utils, base)
//   5. CORE primitives (IMMUTABLE — addons CANNOT mask these)
//
// pkg::func() bypasses resolution — goes direct to package namespace.
// detach(pkg) removes layer — everything below is naturally restored.

use r2_types::*;
use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;

pub type BuiltinFn = fn(&mut Engine, &[EvalArg], &EnvRef) -> Result<RVal, R2Err>;

// Builtin shims grouped by domain. See `src/builtins/mod.rs`. The
// `use ::*` line brings each shim into the same scope as a bare
// `bi_plot` / `bi_hist` etc. so the registration tables below don't
// need a `builtins::graphics::` prefix.
// ── Routed output macros ─────────────────────────────────────────────
// Drop-in stdout-macro replacements that send formatted builtin output
// (str, summary, data-frame printing, package + mode messages, …)
// through the GUI/CLI-capturable sink (r2_types::out) instead of the raw
// process console — a windowed GUI has none. Defined before the module
// declarations so the builtin submodules can use them.
macro_rules! soutln {
    () => { $crate::__rout("\n") };
    ($($arg:tt)*) => { $crate::__rout(&format!("{}\n", format_args!($($arg)*))) };
}
macro_rules! sout {
    ($($arg:tt)*) => { $crate::__rout(&format!("{}", format_args!($($arg)*))) };
}
#[allow(unused_macros)]
macro_rules! serrln {
    () => { $crate::__rerr("\n") };
    ($($arg:tt)*) => { $crate::__rerr(&format!("{}\n", format_args!($($arg)*))) };
}
#[doc(hidden)]
pub fn __rout(s: &str) { r2_types::out::rout(s); }
#[doc(hidden)]
#[allow(dead_code)]
pub fn __rerr(s: &str) { r2_types::out::rerr(s); }

mod builtins;
use builtins::core::*;
use builtins::data_apply::*;
use builtins::sys_models::*;
use builtins::ml_data::*;
use builtins::misc::*;
use builtins::data::*;
use builtins::graphics::*;
use builtins::io::*;
use builtins::ml::*;
use builtins::stats::*;
use builtins::strings::*;
use builtins::lang::*;
use builtins::coerce::*;

// PackageLayer / PackageTier / FunctionRegistry moved to `registry.rs`.
mod registry;
mod registry_tables;
mod packages;
mod ops;
mod eval;
mod indexing;
mod assign;
mod fusion;
mod formula_eval;
use packages::*;
pub use registry::{FunctionRegistry, PackageLayer, PackageTier};

// NA-bitmap combiners for the SIMD / JIT pipeline live in their own
// pure module. Re-exported back into lib.rs's namespace via `use`
// so the eval loop call sites are unchanged.
mod na_bitmap;

// Formula-walking helpers (Error(...) splitter for repeated measures,
// (1|group) random-intercept splitter, Expr→source deparser).
mod formula;  // deparse + Error()/(1|g) splitters; used by eval.rs & formula_eval.rs

// ── Engine ───────────────────────────────────────────────────────────

pub struct Engine {
    pub global_env: EnvRef,
    pub mode: ErrorMode,
    pub registry: FunctionRegistry,
    pub lib_paths: Vec<String>,                              // where to find packages on disk
    pub installed: HashMap<String, InstalledPkgInfo>,         // discovered packages
    types: HashMap<Arc<str>, TypeDef>,
    methods: HashMap<(Arc<str>, Arc<str>), Method>,
    pub(crate) warnings: Vec<String>,
    /// Call-frame stack: one live environment per active function call.
    /// Frames are real `Env`s (interior-mutable), so a closure defined in a
    /// frame captures it by reference and `<<-` mutates the ORIGINAL frame —
    /// R's environment semantics. The stack tracks call depth and the
    /// current write target; lookup goes through the env chain.
    frames: Vec<EnvRef>,
    /// JIT cache keyed by closure body's Arc pointer (Phase C.2).
    /// Value is `None` when compilation has been tried and rejected,
    /// `Some(handle)` when a callable specialization exists.
    // Keyed by the closure body's Arc pointer. The value RETAINS a clone of
    // that `Arc<Expr>` so the address cannot be freed and recycled by a
    // different anonymous closure — without this, short-lived anon closures
    // (e.g. repeated `Reduce`/`Map`/`sapply` calls) could collide on a reused
    // pointer and run the WRONG compiled body.
    jit_cache: HashMap<usize, (Arc<Expr>, Option<Arc<dyn JitHandle>>)>,
    /// Master switch — disabled via env `R2_JIT=0`. Default on.
    jit_enabled: bool,
    /// Call frames for NSE (Phase L.3): pushed ONLY when calling a closure
    /// whose body uses substitute/match.call/sys.call (see `nse_cache`).
    /// `substitute`/`match.call`/`sys.call` read the top frame. Empty for
    /// the overwhelmingly common non-NSE call — zero hot-path cost.
    nse_stack: Vec<NseFrame>,
    /// Gate cache: `Arc::as_ptr(&closure.body)` → (retained Arc, uses_nse?).
    /// The Arc is retained to defeat pointer recycling (same fix as
    /// `jit_cache`). Computed once per unique closure body.
    nse_cache: HashMap<usize, (Arc<Expr>, bool)>,
}

/// One NSE call frame (Phase L.3). Captures the UNEVALUATED call so that
/// `substitute`/`match.call`/`sys.call` can recover the caller's
/// expressions — R's eager-model stand-in for promises.
pub(crate) struct NseFrame {
    /// The `Expr::Call` that invoked the closure (func + unevaluated args).
    pub(crate) call: Arc<Expr>,
    /// The called closure's formals — for R-style arg→param matching.
    pub(crate) params: Vec<r2_types::Param>,
}

/// Does this expression define or return a closure anywhere? Used to keep
/// the numeric JIT away from functions that build/return functions (which it
/// cannot represent). Conservative: also true for inline `function(...)`
/// passed to sapply/Reduce/etc. — those aren't numeric hot loops either.
fn body_defines_closure(e: &Expr) -> bool {
    match e {
        Expr::FuncDef { .. } | Expr::Lambda { .. } => true,
        Expr::Block(s) => s.iter().any(body_defines_closure),
        Expr::Binary { lhs, rhs, .. } => body_defines_closure(lhs) || body_defines_closure(rhs),
        Expr::Unary { expr, .. } => body_defines_closure(expr),
        Expr::Assign { target, value, .. } => body_defines_closure(target) || body_defines_closure(value),
        Expr::Call { func, args } => body_defines_closure(func) || args.iter().any(|a| body_defines_closure(&a.value)),
        Expr::If { cond, then, else_ } =>
            body_defines_closure(cond) || body_defines_closure(then) || else_.as_deref().is_some_and(body_defines_closure),
        Expr::For { iter, body, .. } => body_defines_closure(iter) || body_defines_closure(body),
        Expr::While { cond, body } => body_defines_closure(cond) || body_defines_closure(body),
        Expr::Repeat { body } => body_defines_closure(body),
        Expr::Return(e) => body_defines_closure(e),
        Expr::Index { object, indices } => body_defines_closure(object) || indices.iter().flatten().any(body_defines_closure),
        Expr::DblIndex { object, index } => body_defines_closure(object) || body_defines_closure(index),
        Expr::Dollar { object, .. } => body_defines_closure(object),
        Expr::Pipe { lhs, rhs } => body_defines_closure(lhs) || body_defines_closure(rhs),
        _ => false,
    }
}

/// Info about an installed (but not necessarily loaded) package
#[derive(Clone, Debug)]
pub struct InstalledPkgInfo {
    pub name: String,
    pub version: String,
    pub path: String,
    pub exports: Vec<String>,
    pub depends: Vec<String>,
}

// Phase R foundation: error types now live in r2-types (so per-domain
// crates like r2-stats can return R2Err without depending on r2-engine).
pub use r2_types::{R2Err, ErrKind};

#[macro_export] macro_rules! err { ($k:ident, $($a:tt)*) => { Err(R2Err { msg: format!($($a)*), kind: ErrKind::$k }) }; }

pub(crate) fn gv(args: &[EvalArg], i: usize) -> RVal { args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null) }
pub(crate) fn gn(args: &[EvalArg], name: &str) -> Option<RVal> { args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone()) }

/// Bind in-place: `Env` is interior-mutable, so writes are visible through
/// every Arc alias of the environment (closures, the frame stack, `env`
/// parameters held by callers) — no copy-on-write detachment.
pub(crate) fn env_insert(env: &mut EnvRef, name: Arc<str>, val: RVal) {
    env.set(name, val);
}

fn mkpkg(name: &str, tier: PackageTier, fns: Vec<(&str, BuiltinFn)>) -> PackageLayer {
    let exports = fns.iter().map(|(n,_)| n.to_string()).collect();
    let functions = fns.into_iter().map(|(n,f)| (n.to_string(), f)).collect();
    PackageLayer { name: name.to_string(), tier, functions, exports }
}

// Phase R.2 step 6: implement r2-types' `EngineCtx` so domain crates
// (r2-data::apply) can call back into the evaluator without depending
// on r2-engine. The trait method just delegates to the existing
// (private) `Engine::call_fn`.
impl r2_types::EngineCtx for Engine {
    fn ctx_call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        self.call_fn(func, args, env)
    }
}

impl Engine {
    /// Phase P — a per-worker engine for parallel `apply`. Shares the
    /// read-only registry / global env / installed packages (cloned; the
    /// heavy inner data is `Arc`), with FRESH per-eval scratch (scopes,
    /// warnings, an empty JIT cache, NSE stack). Workers never mutate shared
    /// state — closure calls build child envs copy-on-write — so parallel
    /// evaluation is data-race-free for pure map closures.
    pub fn fork_worker(&self) -> Engine {
        Engine {
            global_env: self.global_env.clone(),
            mode: self.mode,
            registry: self.registry.clone(),
            lib_paths: self.lib_paths.clone(),
            installed: self.installed.clone(),
            types: self.types.clone(),
            methods: self.methods.clone(),
            warnings: Vec::new(),
            // Carry the caller's frame stack (Arc clones — frames are shared
            // live envs guarded by RwLock) so a mapped closure still sees
            // enclosing-function locals. Pure map closures only read.
            frames: self.frames.clone(),
            jit_cache: HashMap::new(),
            jit_enabled: self.jit_enabled,
            nse_stack: Vec::new(),
            nse_cache: HashMap::new(),
        }
    }

    /// Install the host's output sink as the single, process-wide
    /// console (R's `R_WriteConsole` model). This is the ONE channel:
    /// engine `print`/`cat`/formatter output AND every compute crate's
    /// `soutln!` (via `r2_types::out`) converge here, so the frontend
    /// wires output exactly once. The CLI leaves it unset → output
    /// falls back to stdout/stderr; the GUI installs a sink backed by
    /// its `ConsoleBuffer`.
    ///
    /// `r2_types::out` is line-buffered and hands the sink complete
    /// lines (no trailing newline) — `StdoutSink` appends one,
    /// `ConsoleBuffer::push_output` treats each as a line.
    pub fn set_output_sink(&mut self, mut sink: Box<dyn r2_console::OutputSink>) {
        r2_types::out::set_output_hook(Some(Box::new(move |line: &str, is_err: bool| {
            if is_err { sink.write_error(line); } else { sink.write_output(line); }
        })));
    }

    /// Emit through the single console channel. `print`/`cat`/etc. call
    /// this; it preserves the historical "one trailing newline" sink
    /// contract, then routes through `r2_types::out` — the same channel
    /// the compute crates use — so there is exactly one output path.
    pub fn emit_output(&mut self, text: &str) {
        if text.ends_with('\n') { r2_types::out::rout(text); }
        else { r2_types::out::rout(&format!("{}\n", text)); }
    }
    pub fn emit_error(&mut self, text: &str) {
        if text.ends_with('\n') { r2_types::out::rerr(text); }
        else { r2_types::out::rerr(&format!("{}\n", text)); }
    }

    /// Opt in to the browser-based plot viewer (interactive CLI only).
    /// By default no auto-view occurs — scripts, the test suite, and the
    /// GUI (own plot window) never spawn a browser. The interactive REPL
    /// calls this so `plot()` opens a live viewer, like RGui opening a
    /// device. Exposed here so `r2-repl` needn't depend on r2-graphics.
    pub fn enable_plot_autoview(&self) {
        r2_graphics::device::enable_autoview();
    }

    pub fn new() -> Self {
        let global = Env::new_global();
        let mut e = Engine {
            global_env: global, mode: ErrorMode::Strict,
            registry: FunctionRegistry::new(),
            lib_paths: {
                let mut paths = vec![];
                // Windows: %USERPROFILE%\.r2\library
                if let Ok(home) = std::env::var("USERPROFILE") {
                    paths.push(format!("{}\\.r2\\library", home));
                }
                // Unix: ~/.r2/library
                if let Ok(home) = std::env::var("HOME") {
                    paths.push(format!("{}/.r2/library", home));
                }
                paths.push("/usr/lib/r2/library".into());
                paths
            },
            installed: HashMap::new(),
            types: HashMap::new(), methods: HashMap::new(), warnings: Vec::new(),
            frames: Vec::new(),
            jit_cache: HashMap::new(),
            jit_enabled: std::env::var("R2_JIT").map(|v| v != "0").unwrap_or(true),
            nse_stack: Vec::new(),
            nse_cache: HashMap::new(),
        };

        // Built-in package layers. The tables live in `registry_tables`
        // (single source of truth — also read by `try_reload_base`).
        e.registry.add_layer(mkpkg("core", PackageTier::Core, registry_tables::core_table()));
        e.registry.add_layer(mkpkg("base", PackageTier::Base, registry_tables::base_table()));
        e.registry.add_layer(mkpkg("stats", PackageTier::Base, registry_tables::stats_table()));
        e.registry.add_layer(mkpkg("graphics", PackageTier::Base, registry_tables::graphics_table()));
        e.registry.add_layer(mkpkg("utils", PackageTier::Base, registry_tables::utils_table()));

        // ── DATASETS ─────────────────────────────────────────────────
        r2_base::register_datasets(&mut e.global_env.bindings.write().unwrap());

        // ── BUILT-IN CONSTANTS (Phase R.M.1) ─────────────────────────
        // R-compatible numeric constants. Users write `pi`, `Inf`, `NaN`
        // and they resolve to these without needing a function call.
        let scalar = |x: f64| RVal::Numeric(vec![Some(x)].into(), Attrs::default());
        e.global_env.set(Arc::from("pi"),  scalar(std::f64::consts::PI));
        e.global_env.set(Arc::from("Inf"), scalar(f64::INFINITY));
        e.global_env.set(Arc::from("NaN"), scalar(f64::NAN));
        e
    }

    /// Load addon package — blocks if it tries to mask core functions
    pub fn load_addon(&mut self, layer: PackageLayer) -> Result<Vec<String>, String> {
        for name in &layer.exports {
            if self.registry.is_core(name) {
                return Err(format!("package '{}' cannot mask core function '{}'", layer.name, name));
            }
        }
        let masks = self.registry.check_masks(&layer.exports);
        let mut warnings = Vec::new();
        for (func, from) in &masks {
            let msg = format!("package '{}' masks '{}' from '{}'", layer.name, func, from);
            warnings.push(msg.clone());
            self.warnings.push(format!("Warning: {}", msg));
        }
        self.registry.add_layer(layer);
        Ok(warnings)
    }

    /// Detach package — lower layers naturally restore for builtins.
    /// For addon packages (R2 scripts), also removes functions from global env.
    pub fn detach_package(&mut self, name: &str) -> Result<Vec<String>, String> {
        // Get exports before removing
        let exports: Vec<String> = self.registry.layers.iter()
            .find(|l| l.name == name)
            .map(|l| l.exports.clone())
            .unwrap_or_default();

        let result = self.registry.remove_layer(name)?;

        // For addon packages: remove their functions + types from global env
        for fname in &exports {
            self.global_env.remove(fname.as_str());
        }

        // Drop any types and methods the package contributed, so a detached
        // package leaves no type/method dispatch behind.
        for ex in &exports {
            self.types.remove(ex.as_str());
        }
        self.methods.retain(|(mname, _), _| !exports.iter().any(|x| x.as_str() == mname.as_ref()));

        Ok(result)
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<RVal, R2Err> { let env = self.global_env.clone(); self.eval_in(expr, &env) }
    /// Enable / disable the Phase C.2 JIT path. Used by benchmarks and
    /// for opting out at runtime; the env var `R2_JIT=0` does the same.
    pub fn set_jit_enabled(&mut self, on: bool) { self.jit_enabled = on; self.jit_cache.clear(); }

    pub fn as_reals(&self, obj: &RVal) -> Result<Vec<Real>, R2Err> { obj.as_reals() }
    pub fn as_logicals(&self, obj: &RVal) -> Result<Vec<Logical>, R2Err> { obj.as_logicals() }
    pub(crate) fn scalar_f64(&self, obj: &RVal) -> Result<Real, R2Err> { obj.scalar_f64() }
    pub(crate) fn truthy(&self, obj: &RVal) -> Result<bool, R2Err> { match obj { RVal::Logical(v,_) => v.first().copied().flatten().ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), RVal::Numeric(v,_) => v.first().copied().flatten().map(|n| n!=0.0).ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), _ => err!(Type,"cannot coerce {} to logical",obj.type_name()) } }
    fn vals_eq(&self, a: &RVal, b: &RVal) -> bool { match (a,b) { (RVal::Numeric(a,_),RVal::Numeric(b,_)) => a==b, (RVal::Character(a,_),RVal::Character(b,_)) => a==b, (RVal::Integer(a,_),RVal::Integer(b,_)) => a==b, _ => false } }
    pub(crate) fn to_items(&self, obj: &RVal) -> Result<Vec<RVal>, R2Err> { match obj { RVal::Integer(v,_) => Ok(v.iter().map(|x| RVal::Integer(vec![*x].into(),Attrs::default())).collect()), RVal::Numeric(v,_) => Ok(v.iter().map(|x| RVal::Numeric(vec![*x].into(),Attrs::default())).collect()), RVal::Character(v,_) => Ok(v.iter().map(|x| RVal::Character(vec![x.clone()],Attrs::default())).collect()), RVal::List(v) => Ok(v.iter().map(|(_,val)| val.clone()).collect()), RVal::DataFrame(df) => Ok(df.columns.iter().map(|(_,val)| val.clone()).collect()), _ => err!(Runtime,"cannot iterate over {}",obj.type_name()) } }
    pub fn drain_warnings(&mut self) -> Vec<String> { std::mem::take(&mut self.warnings) }

    /// Insert into the correct scope: the current call frame (inside a
    /// function) or the global environment (top level). Writes are in-place;
    /// every holder of the env Arc sees them immediately.
    fn scope_insert(&mut self, name: Arc<str>, val: RVal) {
        match self.frames.last() {
            Some(f) => f.set(name, val),
            None => self.global_env.set(name, val),
        }
    }

    /// `<<-` superassignment — R's LEXICAL rule: rebind where the name is
    /// found walking the current frame's ENCLOSING environment chain (the
    /// closure capture chain, not the dynamic call stack); if absent all
    /// the way up, bind in the global environment.
    fn super_assign(&mut self, name: Arc<str>, val: RVal) {
        if let Some(f) = self.frames.last() {
            if f.set_in_enclosing(&name, &val) { return; }
        }
        self.global_env.set(name, val);
    }

}
/// JIT NA-aware output reconstruction helpers (Phase F.3 unlock).
///
/// For unary maps: output bitmap = input bitmap. For positions marked
/// invalid in the input, we emit `None` regardless of the f64 value
/// the Cranelift loop produced (which would be NaN from NaN-propagation
/// — same result, but going through the bitmap is structurally cleaner
/// and lets us distinguish NaN-from-arithmetic from NA-from-input later).
// combine_unary_output / combine_binary_output / combine_ternary_output
// moved to src/na_bitmap.rs.

/// Stringify a parser `Expr` back to source-like text. Used by the
/// lm/glm/aov NSE preprocessor to capture the original call shape as a
/// `$call` field on the fitted-model TypeInstance — so `summary(fit)`
/// can print `Call: lm(formula = y ~ x, data = df)` instead of the
/// generic placeholder `Call: lm(formula)`. Covers symbols, numeric
/// literals, binary/unary operators, function calls, and indexing —
/// the subset needed for typical model formulas.
// ─────────────────────────────────────────────────────────────────────
// Phase R.S.1 — Error(...) term splitter for repeated-measures formulas.
//
// In R's aov() syntax, `y ~ x + Error(subject/treatment)` declares that
// `x` is the fixed effect and `Error(subject/treatment)` defines the
// random-effect stratum for within-subject ANOVA. The Error term must
// be lifted out of the predictor expansion (otherwise it would try to
// resolve "Error" as a builtin and fail) and tagged separately so the
// stats engine can build per-stratum sums of squares later in R.S.1.
//
// `split_error_term` walks the RHS expression tree and returns
// `(fixed_part, optional_stratum_expr)`. The fixed part is the RHS with
// any Error(...) subexpressions removed; the stratum is whatever was
// inside the Error() call. When no Error() is present, the result is
// `(rhs, None)` and behavior is unchanged.
// ─────────────────────────────────────────────────────────────────────

// Error(...) / random-intercept formula splitters moved to src/formula.rs.

// ─────────────────────────────────────────────────────────────────────
// Phase R.S.3 — Random-effect specification splitter for lmer formulas.
//
// `lmer(y ~ x + (1|subject), data=df)` declares a random intercept per
// subject. In R2's parser the `|` is parsed as BinOp::Or, so the inner
// expression `(1|subject)` becomes Binary{Or, NumLit(1), Symbol(subject)}.
//
// For v0.2.0 Tier 1 we support only intercept-only random effects:
// `(1|group)`. Random slopes `(1+x|group)`, crossed effects
// `(1|s) + (1|item)`, and nested `(1|s/cohort)` are R.S.4 work.
//
// `split_random_effects` walks the RHS, lifts `(1|group)` subexpressions
// into a separate list, and returns the fixed-effect remainder.
// ─────────────────────────────────────────────────────────────────────

// is_random_intercept / random_intercept_grouping / split_random_effects /
// fmt_expr moved to src/formula.rs.

pub(crate) fn val_to_str(v: &RVal) -> String { match v { RVal::Numeric(v,_) => v.iter().map(|x| match x {Some(n)=>fmt_num(*n),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Single(v,_) => v.iter().map(|x| match x {Some(n)=>fmt_num(*n as f64),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Integer(v,_) => v.iter().map(|x| match x {Some(n)=>format!("{}",n),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Character(v,_) => v.iter().map(|x| match x {Some(s)=>s.to_string(),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Logical(v,_) => v.iter().map(|x| match x {Some(true)=>"TRUE",Some(false)=>"FALSE",None=>"NA"}).collect::<Vec<_>>().join(" "), RVal::Null => "NULL".into(), _ => format!("<{}>",v.type_name()) } }

// ═══════════════════════════════════════════════════════════════════════
// BUILTINS
// ═══════════════════════════════════════════════════════════════════════

// Phase R.2: bi_c moved to r2-data::concat. Engine adapter only.
// Core builtins (length/print/cat/coercions/glm-family/summary/...)
// moved to builtins/core.rs.

// cov(x, y) — sample covariance with Bessel correction:
//   cov = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)
// Drops NA pairs (matches R's `use = "complete.obs"` default style for now).
// Oracle decides serial vs parallel for the inner reductions.

// ═══════════════════════════════════════════════════════════════════════
// read.csv — parse CSV file into DataFrame
// ═══════════════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════════════
// lm() — linear regression using normal equations: β = (X^T X)^-1 X^T y
// ═══════════════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════════════
// plot() — SVG scatter plot output
// ═══════════════════════════════════════════════════════════════════════

// (bi_plot — model-aware dispatch + r2-graphics delegation — moved
// to src/builtins/graphics.rs.)

// ═══════════════════════════════════════════════════════════════════════
// matrix(), tensor(), t(), crossprod()
// ═══════════════════════════════════════════════════════════════════════

// Phase R.4: matrix/tensor/t/crossprod moved to r2-linalg::ops.
fn bi_matrix(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_base::linalg_ops::bi_matrix(a) }
fn bi_tensor(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_base::linalg_ops::bi_tensor(a) }
fn bi_transpose(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_base::linalg_ops::bi_transpose(a) }
fn bi_crossprod(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_base::linalg_ops::bi_crossprod(a) }

// ═══════════════════════════════════════════════════════════════════════
// String operations
// ═══════════════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════════════
// table() — frequency counts
// ═══════════════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════════════
// sapply / lapply — apply function over vector/list
// ═══════════════════════════════════════════════════════════════════════

// ── Pure-builtin allowlist for parallel apply (Phase D) ──────────────
//
// Each entry is a "pure" implementation: takes a single RVal, returns an
// RVal, no engine access. Safe to call from multiple threads concurrently.
// `bi_lapply` / `bi_sapply` use this fast path when the inner function is
// a `BuiltinFn` whose name appears here. Any other inner function falls
// back to the serial `e.call_fn(...)` path that respects full semantics.
//
// To extend: add a match arm here. Avoid anything that reads engine config,
// looks up other functions, or mutates global state.
pub(crate) fn pure_apply(name: &str, arg: &RVal) -> Option<Result<RVal, R2Err>> {
    let coerce_reals = |v: &RVal| -> Option<Vec<Real>> {
        match v {
            RVal::Numeric(vs, _) => Some(vs.as_vec().clone()),
            RVal::Integer(vs, _) => Some(vs.iter().map(|x| x.map(|n| n as f64)).collect()),
            RVal::Logical(vs, _) => Some(vs.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 })).collect()),
            RVal::Matrix(m) => Some(m.data.iter().map(|x| if x.is_nan() { None } else { Some(*x) }).collect()),
            _ => None,
        }
    };
    match name {
        "sum" => {
            let v = coerce_reals(arg)?;
            let s: Real = v.iter().try_fold(0.0f64, |acc, x| x.map(|n| acc + n));
            Some(Ok(RVal::Numeric(vec![s].into(), Attrs::default())))
        }
        "mean" => {
            let v = coerce_reals(arg)?;
            let n = v.len() as f64;
            let s: Real = v.iter().try_fold(0.0f64, |acc, x| x.map(|val| acc + val));
            Some(Ok(RVal::Numeric(vec![s.map(|t| t / n)].into(), Attrs::default())))
        }
        "sd" => {
            let v = coerce_reals(arg)?;
            let nums: Vec<f64> = v.iter().filter_map(|x| *x).collect();
            let n = nums.len();
            if n < 2 { return Some(Ok(RVal::Numeric(vec![None].into(), Attrs::default()))); }
            let mean = nums.iter().sum::<f64>() / n as f64;
            let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            Some(Ok(RVal::Numeric(vec![Some(var.sqrt())].into(), Attrs::default())))
        }
        "var" => {
            let v = coerce_reals(arg)?;
            let nums: Vec<f64> = v.iter().filter_map(|x| *x).collect();
            let n = nums.len();
            if n < 2 { return Some(Ok(RVal::Numeric(vec![None].into(), Attrs::default()))); }
            let mean = nums.iter().sum::<f64>() / n as f64;
            let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            Some(Ok(RVal::Numeric(vec![Some(var)].into(), Attrs::default())))
        }
        "min" => {
            let v = coerce_reals(arg)?;
            let m = v.iter().filter_map(|x| *x).fold(f64::INFINITY, f64::min);
            Some(Ok(RVal::Numeric(vec![Some(m)].into(), Attrs::default())))
        }
        "max" => {
            let v = coerce_reals(arg)?;
            let m = v.iter().filter_map(|x| *x).fold(f64::NEG_INFINITY, f64::max);
            Some(Ok(RVal::Numeric(vec![Some(m)].into(), Attrs::default())))
        }
        "prod" => {
            let v = coerce_reals(arg)?;
            let p: Real = v.iter().try_fold(1.0f64, |acc, x| x.map(|n| acc * n));
            Some(Ok(RVal::Numeric(vec![p].into(), Attrs::default())))
        }
        "length" => {
            let n = match arg {
                RVal::Numeric(v, _) => v.len(),
                RVal::Integer(v, _) => v.len(),
                RVal::Character(v, _) => v.len(),
                RVal::Logical(v, _) => v.len(),
                RVal::List(v) => v.len(),
                RVal::Matrix(m) => m.data.len(),
                RVal::Null => 0,
                _ => 1,
            };
            Some(Ok(RVal::Integer(vec![Some(n as i32)].into(), Attrs::default())))
        }
        // Element-wise math (returns vector of same length)
        "sqrt" | "abs" | "exp" | "log" | "log2" | "log10" => {
            let v = coerce_reals(arg)?;
            let f: fn(f64) -> f64 = match name {
                "sqrt" => f64::sqrt, "abs" => f64::abs, "exp" => f64::exp,
                "log" => f64::ln, "log2" => f64::log2, "log10" => f64::log10,
                _ => unreachable!(),
            };
            Some(Ok(RVal::Numeric(v.iter().map(|x| x.map(f)).collect(), Attrs::default())))
        }
        _ => None,
    }
}

// Phase R.2 step 6: apply family moved to r2-data::apply via EngineCtx.
fn bi_sapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_sapply(e, a, env);
    #[allow(unreachable_code)] {
    let x = gv(a, 0);
    let func = gv(a, 1);
    let items = e.to_items(&x)?;

    // Phase D: parallel fast path when inner function is a pure builtin.
    let results: Vec<RVal> = if let RVal::BuiltinFn(fname) = &func {
        if !items.is_empty() && pure_apply(fname, &items[0]).is_some() {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(items.len() * 100),
            );
            let fname_owned = fname.to_string();
            if go_par {
                items.par_iter().map(|item| {
                    pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null))
                }).collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(items.len());
                for item in &items { r.push(pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        } else {
            // Fall through to serial engine call below.
            let mut r = Vec::with_capacity(items.len());
            for item in items {
                let call_args = vec![EvalArg { name: None, value: item }];
                r.push(e.call_fn(&func, &call_args, env)?);
            }
            r
        }
    } else {
        let mut r = Vec::with_capacity(items.len());
        for item in items {
            let call_args = vec![EvalArg { name: None, value: item }];
            r.push(e.call_fn(&func, &call_args, env)?);
        }
        r
    };

    // Try to simplify to numeric vector (existing behavior).
    let mut nums = Vec::new();
    let mut all_num = true;
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            _ => { all_num = false; break; }
        }
    }
    if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
    else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

fn bi_lapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_lapply(e, a, env);
    #[allow(unreachable_code)] {
    let x = gv(a, 0);
    let func = gv(a, 1);
    let items = e.to_items(&x)?;

    // Phase D: parallel fast path when inner function is a pure builtin.
    if let RVal::BuiltinFn(fname) = &func {
        if !items.is_empty() && pure_apply(fname, &items[0]).is_some() {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(items.len() * 100),
            );
            let fname_owned = fname.to_string();
            let results: Vec<(Option<Arc<str>>, RVal)> = if go_par {
                items.par_iter()
                    .map(|item| pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null)).map(|v| (None, v)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(items.len());
                for item in &items {
                    r.push((None, pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null))?));
                }
                r
            };
            return Ok(RVal::List(results));
        }
    }

    // Fallback: serial engine call.
    let mut results = Vec::new();
    for item in items {
        let call_args = vec![EvalArg { name: None, value: item }];
        results.push((None, e.call_fn(&func, &call_args, env)?));
    }
    Ok(RVal::List(results))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

fn bi_vapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_vapply(e, a, env);
    #[allow(unreachable_code)] {
    let x = gv(a, 0);
    let func = gv(a, 1);
    // gv(a, 2) is FUN.VALUE — ignored for now; future strict-checking lives here.
    let items = e.to_items(&x)?;

    let results: Vec<RVal> = if let RVal::BuiltinFn(fname) = &func {
        if !items.is_empty() && pure_apply(fname, &items[0]).is_some() {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(items.len() * 100),
            );
            let fname_owned = fname.to_string();
            if go_par {
                items.par_iter().map(|item| pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(items.len());
                for item in &items { r.push(pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        } else {
            let mut r = Vec::with_capacity(items.len());
            for item in items {
                let call_args = vec![EvalArg { name: None, value: item }];
                r.push(e.call_fn(&func, &call_args, env)?);
            }
            r
        }
    } else {
        let mut r = Vec::with_capacity(items.len());
        for item in items {
            let call_args = vec![EvalArg { name: None, value: item }];
            r.push(e.call_fn(&func, &call_args, env)?);
        }
        r
    };

    // vapply must return a vector — error if any result isn't a scalar Numeric.
    let mut nums = Vec::with_capacity(results.len());
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            other => return err!(Type, "vapply: FUN returned non-scalar of type '{}'", other.type_name()),
        }
    }
    Ok(RVal::Numeric(nums.into(), Attrs::default()))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

// mapply moved to r2-data::apply.
// Iterates over multiple lists/vectors in lockstep, calling FUN with one
// element from each. Length is the longest input (R's recycling rule).
// Phase D parallel path: when FUN is a pure-allowlist builtin AND there is
// exactly ONE iteration vector, runs through par_iter. With multiple inputs,
// the pure_apply table doesn't model multi-arg builtins yet, so falls back
// to serial. (Multi-arg pure builtins is a V2 extension.)
fn bi_mapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_mapply(e, a, env);
    #[allow(unreachable_code)] {
    if a.len() < 2 { return err!(Runtime, "mapply: needs FUN + at least one input"); }
    let func = gv(a, 0);
    let inputs: Vec<Vec<RVal>> = (1..a.len())
        .map(|i| e.to_items(&gv(a, i)).unwrap_or_default())
        .collect();
    let max_len = inputs.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 { return Ok(RVal::List(vec![])); }

    // Single-input pure-builtin fast path.
    if inputs.len() == 1 {
        if let RVal::BuiltinFn(fname) = &func {
            if pure_apply(fname, &inputs[0][0]).is_some() {
                let items = &inputs[0];
                let go_par = r2_oracle::should_parallelize(
                    r2_oracle::Op::PerElementMap,
                    r2_oracle::Shape::n(items.len() * 100),
                );
                let fname_owned = fname.to_string();
                let results: Vec<RVal> = if go_par {
                    items.par_iter().map(|item| pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null)))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut r = Vec::with_capacity(items.len());
                    for item in items { r.push(pure_apply(&fname_owned, item).unwrap_or(Ok(RVal::Null))?); }
                    r
                };
                // Simplify like sapply.
                let mut nums = Vec::new(); let mut all_num = true;
                for r in &results {
                    match r {
                        RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
                        _ => { all_num = false; break; }
                    }
                }
                return if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
                else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) };
            }
        }
    }

    // General serial path: zip inputs in lockstep with R's recycling rule.
    let mut results = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let call_args: Vec<EvalArg> = inputs.iter().map(|input| {
            let idx = if input.is_empty() { 0 } else { i % input.len() };
            EvalArg { name: None, value: input.get(idx).cloned().unwrap_or(RVal::Null) }
        }).collect();
        results.push(e.call_fn(&func, &call_args, env)?);
    }
    let mut nums = Vec::new(); let mut all_num = true;
    for r in &results {
        match r {
            RVal::Numeric(v, _) if v.len() == 1 => nums.push(v[0]),
            _ => { all_num = false; break; }
        }
    }
    if all_num { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
    else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

// ═══════════════════════════════════════════════════════════════════════
// Distribution functions — dnorm, pnorm, qnorm, rnorm
// ═══════════════════════════════════════════════════════════════════════





// ═══════════════════════════════════════════════════════════════════════
// hist() — text histogram (+ SVG)
// ═══════════════════════════════════════════════════════════════════════



// ═══════════════════════════════════════════════════════════════════════
// library(), detach(), require() — package loading/unloading
//
// These are CORE builtins — no addon can mask them.
//
// How it works:
//   library("stats")     → re-attaches base package if detached
//   library("mypkg")     → reads from disk, parses .r files, loads functions
//   detach("stats")      → removes from search path, functions gone
//   detach("mypkg")      → same, addon removed
//   detach("core")       → ERROR: cannot detach core
//   require("pkg")       → like library() but returns TRUE/FALSE
//   stats::mean(x)       → works even if stats is detached (direct namespace)
//   installed.packages() → list what's available on disk
//   .libPaths()          → show/set library search paths
// ═══════════════════════════════════════════════════════════════════════

// Package machinery (library/require/detach/installed.packages/.libPaths
// + try_reload_base/try_load_from_disk/load_r2pkg_layout) moved to
// src/packages.rs — see `mod packages;` + `use packages::*;` at the top.

// ═══════════════════════════════════════════════════════════════════════
// DATA MANIPULATION: rbind, cbind, merge, subset, transform, within
// ═══════════════════════════════════════════════════════════════════════

// DATA MANIPULATION + NA + APPLY + MORE MATH moved to builtins/data_apply.rs.

// ═══════════════════════════════════════════════════════════════════════
// MORE DISTRIBUTIONS: pnorm, qnorm, rbinom, rpois, dbinom
// ═══════════════════════════════════════════════════════════════════════





// Error function approximation (Abramowitz & Stegun)
// Phase R.9: erf, phi, qnorm_approx now live in r2_stats::dist.
// Engine uses re-exports below to keep call sites unchanged.

// Phase R.10: signif_stars, fmt_pval moved to r2_stats::tests
// (re-exported at crate root). Engine model summaries (lm, glm) still
// import the same functions via the re-export below.

// Phase R.9: qnorm_approx now lives in r2_stats::dist (re-exported above).

// ═══════════════════════════════════════════════════════════════════════
// source() — run R2 script file
// ═══════════════════════════════════════════════════════════════════════

// source/system.time/t.test/chisq.test/installers/predict/glm/confint
// moved to builtins/sys_models.rs.
// ═══════════════════════════════════════════════════════════════════════
// Graphics additions: lines(), points(), abline(), legend()
// These append to the last SVG plot file
// ═══════════════════════════════════════════════════════════════════════

// (overlay shims bi_lines / bi_points / bi_abline / bi_legend moved
// to src/builtins/graphics.rs. Their pre-Phase-R.3 dead bodies were
// dropped here too — they had been #[cfg(any())] guarded since the
// move to r2-graphics::overlays and were never compiled.)

// (par/dev/colors shims moved to src/builtins/graphics.rs — see
// `mod builtins;` + `use builtins::graphics::*;` at the top of the
// file. Phase: r2-engine modularisation, sprint 1.)

#[cfg(any())]
#[allow(dead_code, unused_variables)]
fn _legacy_bi_legend(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let pos = val_to_str(&gv(a,0));
    let legend_items = gn(a, "legend").unwrap_or(RVal::Null);
    let col = gn(a, "col");

    let labels: Vec<String> = match &legend_items {
        RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
        _ => vec!["Series 1".into()],
    };
    let colors: Vec<String> = match &col {
        Some(RVal::Character(v, _)) => v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("black".into())).collect(),
        _ => vec!["black".into(), "red".into(), "blue".into(), "green".into()],
    };

    let (lx, ly) = match pos.as_str() {
        "topleft" => (70.0, 45.0),
        "topright" => (420.0, 45.0),
        "bottomleft" => (70.0, 330.0),
        "bottomright" => (420.0, 330.0),
        _ => (420.0, 45.0),
    };

    let svg_path = "plot.svg";
    let mut svg = std::fs::read_to_string(svg_path).unwrap_or_default();
    if svg.is_empty() { return err!(Runtime, "no plot open — call plot() first"); }

    let mut elems = format!(r#"<rect x="{:.0}" y="{:.0}" width="140" height="{}" fill="white" stroke="black" stroke-width="0.5"/>"#, lx-5.0, ly-15.0, labels.len() * 20 + 10);
    for (i, label) in labels.iter().enumerate() {
        let c = colors.get(i).map(|s| s.as_str()).unwrap_or("black");
        let yp = ly + i as f64 * 20.0;
        elems.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="12" height="12" fill="{}"/>"#, lx, yp-9.0, c));
        elems.push_str(&format!(r#"<text x="{:.0}" y="{:.0}" font-size="11">{}</text>"#, lx + 18.0, yp, label));
    }

    svg = svg.replace("</svg>", &format!("{}</svg>", elems));
    let _ = std::fs::write(svg_path, &svg);
    soutln!("Legend added to {}", svg_path);
    Ok(RVal::Null)
}

/// `explain(f)` — report how R2 will execute closure `f`: JIT-compiled (and to
/// which native specialization) or interpreter (with the first reason it can't
/// JIT). The developer feedback loop for writing fast R2 libraries. (J.5
/// groundwork / execution explainability.)
pub(crate) fn bi_explain(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> {
    let v = match a.first().map(|p| &p.value) {
        Some(v) => v,
        None => return err!(Runtime, "explain: needs an argument (a function or data)"),
    };
    // A function → JIT execution plan.
    if let RVal::Closure(cl) = v {
        let verdict = r2_jit::explain_closure(cl);
        return Ok(RVal::Character(vec![Some(Arc::from(verdict.as_str()))], Attrs::default()));
    }
    // Data → size + architecture + how the Oracle will dispatch work on it.
    let hw = r2_oracle::hw();
    let (n, shape, mm): (usize, String, Option<(usize, usize)>) = match v {
        RVal::Matrix(m) => (m.nrow * m.ncol, format!("matrix {}x{}", m.nrow, m.ncol), Some((m.nrow, m.ncol))),
        RVal::DataFrame(df) => (df.nrow() * df.columns.len(), format!("data.frame {}x{}", df.nrow(), df.columns.len()), None),
        RVal::Numeric(x, _) => (x.len(), format!("numeric vector, length {}", x.len()), None),
        RVal::Integer(x, _) => (x.len(), format!("integer vector, length {}", x.len()), None),
        RVal::Logical(x, _) => (x.len(), format!("logical vector, length {}", x.len()), None),
        RVal::Character(x, _) => (x.len(), format!("character vector, length {}", x.len()), None),
        RVal::List(x) => (x.len(), format!("list, {} elements", x.len()), None),
        other => (1, other.type_name().to_string(), None),
    };
    let method = |op| if r2_oracle::dispatch(op, r2_oracle::Shape::n(n)) == r2_oracle::Backend::Rayon {
        format!("PARALLEL ({} cores)", hw.cores)
    } else { "serial".to_string() };
    let simd = if hw.has_avx512 { " +AVX-512" } else if hw.has_avx2 { " +AVX2" } else if hw.has_fma { " +FMA" } else { "" };
    let mut s = String::new();
    s.push_str(&format!("data:     {shape}  (~{} KB in memory)\n", (n * 8) / 1024 + 1));
    s.push_str(&format!("hardware: {} / {}, {} cores{simd}\n", hw.arch, hw.os, hw.cores));
    s.push_str(&format!("method — reductions (sum/mean/sd):   {}\n", method(r2_oracle::Op::Reduction)));
    s.push_str(&format!("method — element-wise (a+b, f(x)):   {}\n", method(r2_oracle::Op::PerElementMap)));
    if let Some((nr, nc)) = mm {
        let mmb = if r2_oracle::dispatch(r2_oracle::Op::MatMul, r2_oracle::Shape::nmk(nr, nc, nc)) == r2_oracle::Backend::Rayon {
            format!("PARALLEL ({} cores)", hw.cores)
        } else { "serial".to_string() };
        s.push_str(&format!("method — matrix multiply:            {mmb}\n"));
    }
    s.push_str("(parallel ≈ that many cores faster than serial; the Oracle picks per data size)");
    Ok(RVal::Character(vec![Some(Arc::from(s.as_str()))], Attrs::default()))
}

// ═══════════════════════════════════════════════════════════════════════
// help-block + trailing builtins moved to builtins/misc.rs.

/// Phase P — `mclapply(x, FUN)` / `par.sapply(x, FUN)`: apply `FUN` to each
/// element of `x` **in parallel** across cores, returning a list of results.
/// Each element is evaluated in an isolated per-worker engine (see
/// `fork_worker`). `FUN` must be a pure function (no `<<-` to shared state),
/// matching R's `parallel::` worker isolation. Falls back to serial for a
/// non-closure `FUN` or tiny `x`.
pub(crate) fn bi_mclapply(e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> {
    let x = a.first().map(|p| p.value.clone()).unwrap_or(RVal::Null);
    let f = a.iter().find(|p| p.name.as_deref() == Some("FUN"))
        .map(|p| p.value.clone())
        .unwrap_or_else(|| a.get(1).map(|p| p.value.clone()).unwrap_or(RVal::Null));
    if !matches!(f, RVal::Closure(_)) {
        return err!(Runtime, "mclapply: FUN must be a function");
    }
    // Split x into per-element RVals.
    let elems: Vec<RVal> = match &x {
        RVal::Numeric(v, _)   => v.iter().map(|o| RVal::Numeric(vec![*o].into(), Attrs::default())).collect(),
        RVal::Integer(v, _)   => v.iter().map(|o| RVal::Integer(vec![*o].into(), Attrs::default())).collect(),
        RVal::Character(v, _) => v.iter().map(|o| RVal::Character(vec![o.clone()], Attrs::default())).collect(),
        RVal::Logical(v, _)   => v.iter().map(|o| RVal::Logical(vec![*o].into(), Attrs::default())).collect(),
        RVal::List(items)     => items.iter().map(|(_, val)| val.clone()).collect(),
        _ => return err!(Runtime, "mclapply: x must be a vector or list"),
    };

    // Per-element RNG stream (Phase P): seed derived from the element index
    // and the current global seed, so `sample()`/`rnorm()` inside FUN are
    // independent per task AND reproducible regardless of core count (better
    // than R's default parallel RNG). set.seed() controls the base.
    let base = r2_stats::rng::current_seed();
    let seed_for = |i: usize| base ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64 + 1));
    let run_i = |worker: &mut Engine, i: usize| -> Result<RVal, R2Err> {
        r2_stats::rng::set_worker_seed(Some(seed_for(i)));
        let genv = worker.global_env.clone();
        let out = worker.call_fn(&f, &[EvalArg { name: None, value: elems[i].clone() }], &genv);
        r2_stats::rng::set_worker_seed(None);
        out
    };
    let n_el = elems.len();
    // Serial for trivially small inputs (thread overhead not worth it).
    let results: Vec<Result<RVal, R2Err>> = if n_el < 4 {
        (0..n_el).map(|i| run_i(&mut e.fork_worker(), i)).collect()
    } else {
        let eng: &Engine = &*e;
        // R2's tree-walking interpreter recurses deeply; give workers a big
        // stack (Rayon's default ~2 MB overflows on nested library calls).
        let pool = rayon::ThreadPoolBuilder::new()
            .stack_size(64 * 1024 * 1024)
            .build()
            .map_err(|err| R2Err { msg: format!("mclapply: thread pool: {err}"), kind: ErrKind::Runtime })?;
        pool.install(|| {
            (0..n_el).into_par_iter()
                .map_init(|| eng.fork_worker(), |worker, i| run_i(worker, i))
                .collect()
        })
    };

    let mut out = Vec::with_capacity(results.len());
    for r in results { out.push((None, r?)); }
    Ok(RVal::List(out))
}

/// Phase P — `par.sapply(x, FUN)`: like `mclapply` but simplifies the result
/// (all length-1 numeric → vector; all equal-length numeric → a matrix with
/// one column per element; otherwise the list).
pub(crate) fn bi_par_sapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let items = match bi_mclapply(e, a, env)? {
        RVal::List(it) => it,
        other => return Ok(other),
    };
    if items.is_empty() { return Ok(RVal::List(items)); }
    // Try to read every result as a dense numeric vector of equal length.
    let cols: Option<Vec<Vec<f64>>> = items.iter()
        .map(|(_, v)| v.as_reals().ok().map(|r| r.into_iter().flatten().collect::<Vec<f64>>()))
        .collect();
    if let Some(cols) = cols {
        let len = cols[0].len();
        if len >= 1 && cols.iter().all(|c| c.len() == len) {
            if len == 1 {
                let flat: Vec<Real> = cols.iter().map(|c| Some(c[0])).collect();
                return Ok(RVal::Numeric(flat.into(), Attrs::default()));
            }
            // column-major matrix: len rows × items cols
            let mut data = Vec::with_capacity(len * cols.len());
            for c in &cols { data.extend_from_slice(c); }
            return Ok(RVal::Matrix(r2_types::Matrix::new(data, len, cols.len())));
        }
    }
    Ok(RVal::List(items))
}
