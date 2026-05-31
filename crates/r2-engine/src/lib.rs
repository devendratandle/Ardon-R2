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
mod builtins;
use builtins::data::*;
use builtins::graphics::*;
use builtins::io::*;
use builtins::ml::*;
use builtins::stats::*;
use builtins::strings::*;

// PackageLayer / PackageTier / FunctionRegistry moved to `registry.rs`.
mod registry;
pub use registry::{FunctionRegistry, PackageLayer, PackageTier};

// NA-bitmap combiners for the SIMD / JIT pipeline live in their own
// pure module. Re-exported back into lib.rs's namespace via `use`
// so the eval loop call sites are unchanged.
mod na_bitmap;
use na_bitmap::{combine_binary_output, combine_ternary_output, combine_unary_output};

// Formula-walking helpers (Error(...) splitter for repeated measures,
// (1|group) random-intercept splitter, Expr→source deparser).
mod formula;
use formula::{
    fmt_expr, is_error_call, is_random_intercept, random_intercept_grouping,
    split_error_term, split_random_effects, unwrap_nested_error,
};

// ── Engine ───────────────────────────────────────────────────────────

pub struct Engine {
    pub global_env: EnvRef,
    pub mode: ErrorMode,
    pub registry: FunctionRegistry,
    pub lib_paths: Vec<String>,                              // where to find packages on disk
    pub installed: HashMap<String, InstalledPkgInfo>,         // discovered packages
    types: HashMap<Arc<str>, TypeDef>,
    methods: HashMap<(Arc<str>, Arc<str>), Method>,
    warnings: Vec<String>,
    /// Lightweight local scope stack for function scoping.
    /// Each entry is a simple HashMap — no Arc overhead.
    local_scopes: Vec<HashMap<Arc<str>, RVal>>,
    /// JIT cache keyed by closure body's Arc pointer (Phase C.2).
    /// Value is `None` when compilation has been tried and rejected,
    /// `Some(handle)` when a callable specialization exists.
    jit_cache: HashMap<usize, Option<Arc<dyn JitHandle>>>,
    /// Master switch — disabled via env `R2_JIT=0`. Default on.
    jit_enabled: bool,
    /// Output sink installed by the host (CLI / GUI). All engine-side
    /// `print()`, `cat()`, `message()`, formatter output, etc. write
    /// through this — never directly to stdout. Default is
    /// `r2_console::StdoutSink` so headless / unit-test usage still
    /// shows results in a terminal.
    output_sink: Box<dyn r2_console::OutputSink>,
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

macro_rules! err { ($k:ident, $($a:tt)*) => { Err(R2Err { msg: format!($($a)*), kind: ErrKind::$k }) }; }

fn gv(args: &[EvalArg], i: usize) -> RVal { args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null) }
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> { args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone()) }

/// Helper: mutate an Arc<Env> safely — avoids temporary-dropped-while-borrowed
fn env_insert(env: &mut EnvRef, name: Arc<str>, val: RVal) {
    let mut binding = env.clone();
    let g = Arc::make_mut(&mut binding);
    g.bindings.insert(name, val);
    *env = Arc::new(g.clone());
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
    /// Replace the default `StdoutSink` with a host-provided sink.
    /// The CLI keeps the default; the GUI installs one backed by its
    /// `Arc<Mutex<ConsoleBuffer>>` so `print()` output lands in the
    /// transcript widget.
    pub fn set_output_sink(&mut self, sink: Box<dyn r2_console::OutputSink>) {
        self.output_sink = sink;
    }

    /// Emit a line through the active output sink. Used by builtins
    /// (and gradually-migrated `println!` sites).
    pub fn emit_output(&mut self, text: &str) {
        self.output_sink.write_output(text);
    }
    pub fn emit_error(&mut self, text: &str) {
        self.output_sink.write_error(text);
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
            local_scopes: Vec::new(),
            jit_cache: HashMap::new(),
            jit_enabled: std::env::var("R2_JIT").map(|v| v != "0").unwrap_or(true),
            output_sink: Box::new(r2_console::StdoutSink),
        };

        // ── CORE: immutable, CANNOT be masked or detached ────────────
        e.registry.add_layer(mkpkg("core", PackageTier::Core, vec![
            ("c",bi_c),("length",bi_length),("print",bi_print),("cat",bi_cat),
            ("typeof",bi_typeof),("class",bi_class),("is.na",bi_is_na),
            ("is.numeric",bi_is_num),("is.character",bi_is_chr),("is.logical",bi_is_lgl),
            ("as.numeric",bi_as_num),("as.single",bi_as_single),("is.single",bi_is_single),
            ("as.character",bi_as_chr),("as.integer",bi_as_int),
            ("as.factor",bi_as_factor),("as.logical",bi_as_logical),("as.data.frame",bi_as_data_frame),
            ("is.data.frame",bi_is_data_frame),("is.factor",bi_is_factor),("is.matrix",bi_is_matrix),
            ("list",bi_list),("list.meta",bi_list_meta),
            ("data.frame",bi_df),("matrix",bi_matrix),("tensor",bi_tensor),
            ("strict",bi_strict),("lenient",bi_lenient),
            // library/detach/require are CORE — no addon can override them
            ("library",bi_library),("detach",bi_detach),("require",bi_require),
            ("installed.packages",bi_installed_packages),(".libPaths",bi_lib_paths),
            ("install.from.dir",bi_install_from_dir),("install.from.zip",bi_install_from_zip),
            ("install.from.github",bi_install_from_github),("uninstall",bi_uninstall),
            ("install.packages",bi_install_packages),
        ]));

        // ── BASE: can be masked by addons, can be detached ───────────
        e.registry.add_layer(mkpkg("base", PackageTier::Base, vec![
            ("seq",bi_seq),("rep",bi_rep),("paste",bi_paste),("paste0",bi_paste0),
            ("which",bi_which),("sort",bi_sort),("rev",bi_rev),("unique",bi_unique),
            ("abs",bi_abs),("sqrt",bi_sqrt),("round",bi_round),("max",bi_max),("min",bi_min),
            ("nchar",bi_nchar),("toupper",bi_toupper),("tolower",bi_tolower),
            ("substr",bi_substr),("grep",bi_grep),("gsub",bi_gsub),("strsplit",bi_strsplit),
            ("sub",bi_sub),("grepl",bi_grepl),("regexpr",bi_regexpr),
            ("duplicated",bi_duplicated),("order",bi_order),("rank",bi_rank),
            ("cummax",bi_cummax),("cummin",bi_cummin),
            ("filter",bi_filter),("select",bi_select),("arrange",bi_arrange),("mutate",bi_mutate),
            ("factor",bi_factor),("names",bi_names),("nrow",bi_nrow),("ncol",bi_ncol),
            ("table",bi_table),("sapply",bi_sapply),("lapply",bi_lapply),("mapply",bi_mapply),("vapply",bi_vapply),
            // data manipulation
            ("rbind",bi_rbind),("cbind",bi_cbind),("merge",bi_merge),
            // NA handling
            ("na.omit",bi_na_omit),("complete.cases",bi_complete_cases),
            ("is.null",bi_is_null),("ifelse",bi_ifelse),
            // apply family
            ("apply",bi_apply),("tapply",bi_tapply),("aggregate",bi_aggregate),
            ("do.call",bi_do_call),
            // math
            ("log",bi_log),("exp",bi_exp),("ceiling",bi_ceiling),("floor",bi_floor),
            ("log2",bi_log2),("log10",bi_log10),("log1p",bi_log1p),("expm1",bi_expm1),
            // trigonometry (Phase R.M.1)
            ("sin",bi_sin),("cos",bi_cos),("tan",bi_tan),
            ("asin",bi_asin),("acos",bi_acos),("atan",bi_atan),("atan2",bi_atan2),
            ("sinh",bi_sinh),("cosh",bi_cosh),("tanh",bi_tanh),
            ("sign",bi_sign),("trunc",bi_trunc),
            ("cumsum",bi_cumsum),("cumprod",bi_cumprod),("cummax",bi_cummax),("cummin",bi_cummin),("diff",bi_diff),
            // rolling-window (Phase K.9)
            ("rollsum",bi_rollsum),("rollmean",bi_rollmean),("rollmax",bi_rollmax),("rollmin",bi_rollmin),("rollsd",bi_rollsd),
            // more base
            ("which.min",bi_which_min),("which.max",bi_which_max),("range",bi_range),
            ("prod",bi_prod),("any",bi_any),("all",bi_all),
            ("trimws",bi_trimws),("startsWith",bi_starts_with),("endsWith",bi_ends_with),
            ("sprintf",bi_sprintf),("stop",bi_stop),("warning",bi_warning),("message",bi_message),
            ("ls",bi_ls),("rm",bi_rm),("exists",bi_exists),
            // factor and data inspection
            ("levels",bi_levels),("nlevels",bi_nlevels),
            ("dim",bi_dim),("colnames",bi_colnames),("rownames",bi_rownames),
            ("data",bi_data),
            // row/col operations
            ("rowSums",bi_rowSums),("colSums",bi_colSums),("rowMeans",bi_rowMeans),("colMeans",bi_colMeans),
            ("set.seed",bi_set_seed),("Sys.sleep",bi_Sys_sleep),("readline",bi_readline),
                ("as.Date",bi_as_date),("as.POSIXct",bi_as_posixct),("format.Date",bi_format_time),
                ("format.POSIXct",bi_format_time),("Sys.Date",bi_sys_date),("Sys.time",bi_sys_time),
                ("difftime",bi_difftime),
                ("ts",bi_ts),("tsp",bi_tsp),("start",bi_ts_start),("end",bi_ts_end),
                ("frequency",bi_frequency),("deltat",bi_deltat),("time",bi_time_idx),
                ("cycle",bi_cycle),("window",bi_window),("is.ts",bi_is_ts),
                ("xts",bi_xts),("index",bi_index),("coredata",bi_coredata),("is.xts",bi_is_xts),
                ("xts.subset",bi_xts_subset),("first",bi_first),("last",bi_last),
                ("na.locf",bi_na_locf),("merge.xts",bi_merge_xts),
                ("acf",bi_acf),("pacf",bi_pacf),("decompose",bi_decompose),
                ("is.regular",bi_is_regular),("periodicity",bi_periodicity),
                ("lag",bi_lag),("diff_ts",bi_diff_ts),
                ("to.daily",bi_to_daily),("to.weekly",bi_to_weekly),
                ("to.monthly",bi_to_monthly),("to.quarterly",bi_to_quarterly),
                ("to.yearly",bi_to_yearly),
                ("apply.daily",bi_apply_daily),("apply.weekly",bi_apply_weekly),
                ("apply.monthly",bi_apply_monthly),("apply.quarterly",bi_apply_quarterly),
                ("apply.yearly",bi_apply_yearly),
                ("tithi",bi_tithi),("hindu.date",bi_hindu_date),("hnc.date",bi_hnc_date),
        ]));
        e.registry.add_layer(mkpkg("stats", PackageTier::Base, vec![
            ("sum",bi_sum),("mean",bi_mean),("sd",bi_sd),("var",bi_var),("cor",bi_cor),("cov",bi_cov),
            ("lm",bi_lm),("summary",bi_summary),
            ("rnorm",bi_rnorm),("dnorm",bi_dnorm),("runif",bi_runif),("sample",bi_sample),
            // more distributions
            ("pnorm",bi_pnorm),("qnorm",bi_qnorm),("rbinom",bi_rbinom),("rpois",bi_rpois),
            // more stats
            ("median",bi_median),("quantile",bi_quantile),
            // hypothesis tests
            ("t.test",bi_t_test),("chisq.test",bi_chisq_test),("hotelling.test",bi_hotelling_test),("manova",bi_manova),("lmer",bi_lmer),
            // model accessors
            ("predict",bi_predict),("residuals",bi_residuals),("fitted",bi_fitted),("coef",bi_coef),
            ("glm",bi_glm),("confint",bi_confint),("binomial",bi_binomial),("gaussian",bi_gaussian),("poisson",bi_poisson),("subset",bi_subset),("transform",bi_transform),
            // ML functions
            ("svd",bi_svd),("eigen",bi_eigen),("prcomp",bi_prcomp),
            ("kmeans",bi_kmeans),("knn",bi_knn),("naive.bayes",bi_naive_bayes),("scale",bi_scale),
            ("rpart",bi_rpart),("rf",bi_rf),("gbm",bi_gbm),("cv",bi_cv),("aov",bi_aov),("anova",bi_anova),("cor.test",bi_cor_test),("shapiro.test",bi_shapiro_test),("wilcox.test",bi_wilcox_test),("fisher.test",bi_fisher_test),("weighted.mean",bi_weighted_mean),("IQR",bi_iqr),("confusion.matrix",bi_confusion_matrix),
        ]));
        e.registry.add_layer(mkpkg("graphics", PackageTier::Base, vec![
            ("plot",bi_plot),("hist",bi_hist),("boxplot",bi_boxplot),("barplot",bi_barplot),
            ("save.plot",bi_save_plot),
            ("lines",bi_lines),("points",bi_points),("abline",bi_abline),("legend",bi_legend),
            ("par",bi_par),("dev.off",bi_dev_off),("save_plot",bi_save_plot),("dev.view",bi_dev_view),
            // Session B — multi-device graphics. Each `dev.new()` opens a
            // fresh plot window; `dev.set()` / `dev.list()` / `dev.cur()`
            // navigate the open devices.
            ("dev.new",bi_dev_new),("dev.set",bi_dev_set),("dev.list",bi_dev_list),
            ("dev.cur",bi_dev_cur),
            // R-style color helpers — pure functions, available to
            // any plot call's col= / border= argument.
            ("rgb",bi_rgb),("gray",bi_gray),("grey",bi_gray),("hsv",bi_hsv),
            ("rainbow",bi_rainbow),("heat.colors",bi_heat_colors),
            ("terrain.colors",bi_terrain_colors),("topo.colors",bi_topo_colors),
            ("cm.colors",bi_cm_colors),("adjustcolor",bi_adjustcolor),
        ]));
        e.registry.add_layer(mkpkg("utils", PackageTier::Base, vec![
            ("head",bi_head),("tail",bi_tail),("str",bi_str),
            ("read.csv",bi_read_csv_v2),("write.csv",bi_write_csv),
            ("search",bi_search),("t",bi_transpose),("crossprod",bi_crossprod),
            ("source",bi_source),("system.time",bi_system_time),
            ("read.table",bi_read_table),("write.table",bi_write_table),("read.delim",bi_read_delim),
            ("Sys.time",bi_Sys_time),("help",bi_help),("getwd",bi_getwd),("setwd",bi_setwd),
            ("file.exists",bi_file_exists),("list.files",bi_list_files),("Sys.getenv",bi_sys_getenv),("save",bi_save),("load",bi_load),("version",bi_version),("clear",bi_clear),("cls",bi_clear),(".Internal",bi_internal),
        ]));

        // ── DATASETS ─────────────────────────────────────────────────
        let mut binding = e.global_env.clone();
        let g = Arc::make_mut(&mut binding);
        r2_base::register_datasets(&mut g.bindings);

        // ── BUILT-IN CONSTANTS (Phase R.M.1) ─────────────────────────
        // R-compatible numeric constants. Users write `pi`, `Inf`, `NaN`
        // and they resolve to these without needing a function call.
        let scalar = |x: f64| RVal::Numeric(vec![Some(x)].into(), Attrs::default());
        g.bindings.insert(Arc::from("pi"),  scalar(std::f64::consts::PI));
        g.bindings.insert(Arc::from("Inf"), scalar(f64::INFINITY));
        g.bindings.insert(Arc::from("NaN"), scalar(f64::NAN));

        e.global_env = Arc::new(g.clone());
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

        // For addon packages: remove their functions from global env
        let mut binding = self.global_env.clone();
        let g = Arc::make_mut(&mut binding);
        for fname in &exports {
            g.bindings.remove(fname.as_str());
        }
        self.global_env = Arc::new(g.clone());

        Ok(result)
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<RVal, R2Err> { let env = self.global_env.clone(); self.eval_in(expr, &env) }
    /// Enable / disable the Phase C.2 JIT path. Used by benchmarks and
    /// for opting out at runtime; the env var `R2_JIT=0` does the same.
    pub fn set_jit_enabled(&mut self, on: bool) { self.jit_enabled = on; self.jit_cache.clear(); }

    pub fn eval_in(&mut self, expr: &Expr, env: &EnvRef) -> Result<RVal, R2Err> {
        // Phase R.M.2 — check the global interrupt flag at the top of every
        // expression evaluation. This is the cheapest universal interruption
        // point in the engine: an atomic-load per Expr is below 1ns on any
        // modern CPU, and it catches everything from runaway loops to deep
        // recursion to long Sys.sleep calls. The REPL's SIGINT handler sets
        // the flag; we raise Interrupt here, which unwinds cleanly to the
        // top-level driver.
        if r2_types::is_interrupted() {
            return Err(R2Err {
                msg: "interrupted".into(),
                kind: ErrKind::Interrupt,
            });
        }

        match expr {
            Expr::NumLit(n) => Ok(rnum(*n)), Expr::IntLit(n) => Ok(rint(*n)),
            Expr::StrLit(s) => Ok(rstr(s)), Expr::BoolLit(b) => Ok(rbool(*b)),
            Expr::NaLit => Ok(rna()), Expr::NullLit => Ok(RVal::Null),
            Expr::FStringLit(parts) => { let mut r = String::new(); for p in parts { match p { FStringPart::Literal(s) => r.push_str(s), FStringPart::Expr(e) => { let v = self.eval_in(e, env)?; r.push_str(&val_to_str(&v)); } } } Ok(rstr(&r)) }
            Expr::Symbol(name) => {
                // 1. Check local scope stack (function-local variables)
                for scope in self.local_scopes.iter().rev() {
                    if let Some(val) = scope.get(name.as_ref()) { return Ok(val.clone()); }
                }
                // 2. Check env chain (parameters, closures)
                if let Some(val) = env.lookup(name) { Ok(val.clone()) }
                // 3. Check global env (top-level assignments, datasets)
                else if let Some(val) = self.global_env.lookup(name) { Ok(val.clone()) }
                // 4. Check builtins
                else if self.registry.resolve(name.as_ref()).is_some() { Ok(RVal::BuiltinFn(name.clone())) }
                else { err!(Runtime, "object '{}' not found", name) }
            }
            Expr::Assign { target, value } => {
                let val = self.eval_in(value, env)?;
                match target.as_ref() {
                    Expr::Symbol(name) => {
                        if matches!(name.as_ref(), "TRUE"|"FALSE"|"T"|"F") { return err!(Runtime, "cannot assign to reserved keyword '{}'", name); }
                        self.scope_insert(name.clone(), val.clone());
                        Ok(val)
                    }
                    Expr::Index { object, indices } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            if indices.len() == 1 {
                                if let Some(idx_expr) = &indices[0] {
                                    let idx = self.eval_in(idx_expr, env)?;
                                    self.assign_index(&mut obj, &idx, &val)?;
                                }
                            }
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid subscript assignment target") }
                    }
                    Expr::DblIndex { object, index } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            let idx = self.eval_in(index, env)?;
                            self.assign_dbl_index(&mut obj, &idx, &val)?;
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid [[ ]] assignment target") }
                    }
                    Expr::Dollar { object, field } => {
                        if let Expr::Symbol(name) = object.as_ref() {
                            let mut obj = self.eval_in(object, env)?;
                            self.assign_dollar(&mut obj, field, &val)?;
                            self.scope_insert(name.clone(), obj.clone());
                            Ok(val)
                        } else { err!(Runtime, "invalid $ assignment target") }
                    }
                    _ => err!(Runtime, "invalid assignment target"),
                }
            }
            Expr::Block(stmts) => { let mut r = RVal::Null; for s in stmts { r = self.eval_in(s, env)?; } Ok(r) }
            Expr::Binary { op, lhs, rhs } => {
                if *op == BinOp::Colon { let l = self.eval_in(lhs, env)?; let r = self.eval_in(rhs, env)?; return self.seq_colon(&l, &r); }
                if *op == BinOp::Tilde {
                    // Formula: y ~ x evaluates both sides, stores as formula-list
                    // lhs can be NULL for one-sided formulas (~x)
                    let l = self.eval_in(lhs, env)?;
                    let r = self.eval_in(rhs, env)?;
                    return Ok(RVal::List(vec![
                        (Some(Arc::from("~lhs")), l),
                        (Some(Arc::from("~rhs")), r),
                        (Some(Arc::from("~class")), rstr("formula")),
                    ]));
                }
                let l = self.eval_in(lhs, env)?; let r = self.eval_in(rhs, env)?; self.binary_op(*op, &l, &r)
            }
            Expr::Unary { op, expr: e } => { let v = self.eval_in(e, env)?; self.unary_op(*op, &v) }
            Expr::Call { func, args } => {
                // NSE: library(stats), detach(stats), require(stats) accept bare symbols
                // Convert bare symbol to string without evaluating it
                if let Expr::Symbol(fname) = func.as_ref() {
                    if matches!(fname.as_ref(), "library" | "detach" | "require" | "data" | "help" | "rm") {
                        let f = self.eval_in(func, env)?;
                        let mut ea = Vec::new();
                        for (i, a) in args.iter().enumerate() {
                            if i == 0 {
                                // First arg: if bare symbol, convert to string (NSE)
                                match &a.value {
                                    Expr::Symbol(sym) => {
                                        // Check if it's actually a variable holding a string
                                        if let Some(val) = env.lookup(sym) {
                                            ea.push(EvalArg { name: a.name.clone(), value: val.clone() });
                                        } else {
                                            // Bare symbol → treat as package name string
                                            ea.push(EvalArg { name: a.name.clone(), value: rstr(sym) });
                                        }
                                    }
                                    _ => ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }),
                                }
                            } else {
                                ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                            }
                        }
                        return self.call_fn(&f, &ea, env);
                    }

                    // NSE for `subset(df, cond)` and `transform(df, name = expr)`:
                    // arg 2+ expressions evaluate in a scope where df's columns
                    // are bound as variables. Without this, `subset(df, x > 2)`
                    // resolves `x` against the global env.
                    if matches!(fname.as_ref(), "subset" | "transform") {
                        if args.len() >= 2 {
                            // Evaluate first arg = data frame.
                            let df_val = self.eval_in(&args[0].value, env)?;
                            if let RVal::DataFrame(df) = &df_val {
                                // Build child env that shadows globals with df columns.
                                let child = Arc::new(Env {
                                    name: Some(Arc::from(".subset.env")),
                                    parent: Some(env.clone()),
                                    bindings: df.columns.iter()
                                        .map(|(n, v)| (n.clone(), v.clone())).collect(),
                                    locked: false,
                                });
                                let f = self.eval_in(func, env)?;
                                let mut ea = vec![EvalArg { name: None, value: df_val.clone() }];
                                for a in args.iter().skip(1) {
                                    let val = self.eval_in(&a.value, &child)?;
                                    ea.push(EvalArg { name: a.name.clone(), value: val });
                                }
                                return self.call_fn(&f, &ea, env);
                            }
                        }
                    }

                    // NSE for `data.frame(y, x1, x2)` — bare-symbol args become
                    // column names. R does this by inspecting the unevaluated
                    // call; we replicate by lifting `Expr::Symbol` arg names
                    // into the EvalArg `name` slot when no explicit `name =`
                    // is given. Without this, `data.frame(y, x1, x2)` would
                    // produce columns V1/V2/V3 and `df[, c("x1","x2")]` would
                    // find nothing.
                    if fname.as_ref() == "data.frame" {
                        let f = self.eval_in(func, env)?;
                        let mut ea = Vec::with_capacity(args.len());
                        for a in args {
                            let val = self.eval_in(&a.value, env)?;
                            let name = a.name.clone().or_else(|| match &a.value {
                                Expr::Symbol(s) => Some(s.clone()),
                                _ => None,
                            });
                            ea.push(EvalArg { name, value: val });
                        }
                        return self.call_fn(&f, &ea, env);
                    }

                    // NSE for formula-based functions: lm(y ~ x, data = df)
                    // When first arg is a tilde expr and data= is provided,
                    // resolve bare symbol names as columns in the data frame
                    if matches!(fname.as_ref(), "lm" | "glm" | "t.test" | "rpart" | "rf" | "gbm" | "cv" | "aov" | "manova" | "lmer") {
                        if let Some(first_arg) = args.first() {
                            if let Expr::Binary { op: BinOp::Tilde, lhs, rhs } = &first_arg.value {
                                // Check if data= is provided
                                let data_arg = args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some("data"));
                                if let Some(data_a) = data_arg {
                                    let data_val = self.eval_in(&data_a.value, env)?;
                                    if let RVal::DataFrame(ref df) = data_val {
                                        // Check for dot (.) on RHS — means "all other columns"
                                        let is_dot_rhs = matches!(rhs.as_ref(), Expr::Symbol(s) if s.as_ref() == ".");

                                        if is_dot_rhs {
                                            // y ~ . means y = lhs column, x = all other columns
                                            let lhs_name = match lhs.as_ref() {
                                                Expr::Symbol(s) => s.clone(),
                                                _ => return err!(Runtime, "formula LHS must be a column name"),
                                            };
                                            // Extract y
                                            let y_col = df.get_col(&lhs_name).ok_or(R2Err{msg:format!("column '{}' not found", lhs_name),kind:ErrKind::Runtime})?;

                                            // Build x matrix from all OTHER numeric columns
                                            let nrow = df.nrow();
                                            let mut x_data = Vec::new();
                                            let mut x_names = Vec::new();
                                            let mut ncol = 0;
                                            for (cn, cv) in &df.columns {
                                                if cn.as_ref() == lhs_name.as_ref() { continue; }
                                                if let Ok(vals) = self.as_reals(cv) {
                                                    let nums: Vec<f64> = vals.into_iter().filter_map(|x| x).collect();
                                                    if nums.len() == nrow { x_data.extend(nums); x_names.push(cn.clone()); ncol += 1; }
                                                }
                                            }
                                            let mut mat = Matrix::new(x_data, nrow, ncol);
                                            mat.col_names = Some(x_names.clone());
                                            let x_mat = RVal::Matrix(mat);

                                            // For lm/glm: use formula path
                                            if matches!(fname.as_ref(), "lm" | "glm") {
                                                let formula = RVal::List(vec![
                                                    (Some(Arc::from("~lhs")), y_col.clone()),
                                                    (Some(Arc::from("~rhs")), x_mat),
                                                    (Some(Arc::from("~class")), rstr("formula")),
                                                ]);
                                                let f = self.eval_in(func, env)?;
                                                let mut ea = vec![EvalArg { name: None, value: formula }];
                                                for a in args.iter().skip(1) { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); }
                                                // Capture original call for `$call` field.
                                                ea.push(EvalArg { name: Some(Arc::from("_call")), value: rstr(&fmt_expr(&Expr::Call { func: func.clone(), args: args.to_vec() })) });
                                                return self.call_fn(&f, &ea, env);
                                            }
                                            // For ML functions: pass (x_matrix, y_vector, ...other args)
                                            let f = self.eval_in(func, env)?;
                                            let mut ea = vec![
                                                EvalArg { name: None, value: x_mat },
                                                EvalArg { name: None, value: y_col.clone() },
                                            ];
                                            for a in args.iter().skip(1) {
                                                if a.name.as_ref().map(|n| n.as_ref()) != Some("data") {
                                                    ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                                                }
                                            }
                                            return self.call_fn(&f, &ea, env);
                                        } else {
                                            // Named columns: resolve normally.
                                            // Phase R.S.1 — split out any Error(...) stratum first so
                                            // it does not get treated as a regular predictor. The
                                            // resulting `rhs_fixed` is the predictor expression with
                                            // the Error term removed; `error_stratum_expr` (if any)
                                            // is what was inside Error(...).
                                            // Phase R.S.3 — also split out (1|group) random-effect
                                            // specs after the Error split, so lmer-style formulas
                                            // like y ~ x + (1|subject) work cleanly.
                                            let (rhs_no_err, error_stratum_expr) = split_error_term(rhs);
                                            let (rhs_fixed, random_grouping_exprs) = split_random_effects(&rhs_no_err);
                                            let lhs_val = self.resolve_formula_term(lhs, df, env)?;
                                            let rhs_val = if matches!(rhs_fixed, Expr::NullLit) {
                                                RVal::Null
                                            } else {
                                                self.resolve_formula_term(&rhs_fixed, df, env)?
                                            };
                                            let mut formula_items = vec![
                                                (Some(Arc::from("~lhs")), lhs_val),
                                                (Some(Arc::from("~rhs")), rhs_val),
                                                (Some(Arc::from("~class")), rstr("formula")),
                                            ];
                                            if let Some(stratum_expr) = error_stratum_expr {
                                                let stratum_val = self.resolve_formula_term(&stratum_expr, df, env)?;
                                                formula_items.push((Some(Arc::from("~error")), stratum_val));
                                            }
                                            for group_expr in &random_grouping_exprs {
                                                let group_val = self.resolve_formula_term(group_expr, df, env)?;
                                                formula_items.push((Some(Arc::from("~random_intercept")), group_val));
                                            }
                                            let formula = RVal::List(formula_items);
                                            let f = self.eval_in(func, env)?;
                                            let mut ea = vec![EvalArg { name: None, value: formula }];
                                            for a in args.iter().skip(1) {
                                                ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? });
                                            }
                                            // Capture original call for `$call` field on the
                                            // fitted-model TypeInstance (lm/glm/aov use it).
                                            ea.push(EvalArg { name: Some(Arc::from("_call")), value: rstr(&fmt_expr(&Expr::Call { func: func.clone(), args: args.to_vec() })) });
                                            return self.call_fn(&f, &ea, env);
                                        }
                                    }
                                }
                                // No data= arg: evaluate formula normally
                            }
                        }
                    }
                    // NSE for system.time: time the expression evaluation.
                    // The inner expression's value is intentionally discarded so it
                    // doesn't get auto-printed by the REPL (matches R's invisible()).
                    if matches!(fname.as_ref(), "system.time") {
                        if let Some(first_arg) = args.first() {
                            let start = std::time::Instant::now();
                            let _ = self.eval_in(&first_arg.value, env)?;
                            let elapsed = start.elapsed();
                            println!("   user  system elapsed");
                            println!("  {:.3}   0.000   {:.3}", elapsed.as_secs_f64(), elapsed.as_secs_f64());
                            return Ok(RVal::Null);
                        }
                    }
                }
                // Normal call: evaluate all arguments
                let f = self.eval_in(func, env)?;
                let mut ea = Vec::new(); for a in args { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); }
                self.call_fn(&f, &ea, env)
            }
            Expr::Pipe { lhs, rhs } => {
                let lv = self.eval_in(lhs, env)?;
                match rhs.as_ref() {
                    Expr::Call { func, args } => { let f = self.eval_in(func, env)?; let mut ea = vec![EvalArg { name: None, value: lv }]; for a in args { ea.push(EvalArg { name: a.name.clone(), value: self.eval_in(&a.value, env)? }); } self.call_fn(&f, &ea, env) }
                    _ => err!(Runtime, "|> rhs must be a function call"),
                }
            }
            Expr::Index { object, indices } => { let obj = self.eval_in(object, env)?; let mut ei = Vec::new(); for i in indices { ei.push(match i { Some(e) => Some(self.eval_in(e, env)?), None => None }); } self.index_obj(&obj, &ei) }
            Expr::Dollar { object, field } => { let obj = self.eval_in(object, env)?; self.dollar(&obj, field) }
            Expr::Namespace { pkg, name } => {
                // pkg::func() — direct namespace access, bypasses search order
                if self.registry.resolve_in_package(pkg, name).is_some() {
                    // Encode as "pkg::name" so call_fn knows to resolve in specific package
                    Ok(RVal::BuiltinFn(Arc::from(format!("{}::{}", pkg, name).as_str())))
                } else {
                    // Package might not be loaded — try loading namespace only
                    err!(Runtime, "'{}' not found in package '{}' (is it loaded?)", name, pkg)
                }
            }
            Expr::If { cond, then, else_ } => { let c = self.eval_in(cond, env)?; if self.truthy(&c)? { self.eval_in(then, env) } else if let Some(e) = else_ { self.eval_in(e, env) } else { Ok(RVal::Null) } }
            Expr::For { var, iter, body } => {
                // Phase R.T.4-fix — top-level for-loops must re-snapshot env
                // from `self.global_env` each iteration, because subscript
                // assignments (`x[i] <- ...`) write through `env_insert` which
                // replaces the Arc; the body's captured env would otherwise
                // see the pre-loop value of every variable on each iteration.
                // Inside function bodies, writes go to `local_scopes`, which
                // Symbol-lookup checks first, so the original env still works.
                let iv = self.eval_in(iter, env)?;
                let at_top_level = self.local_scopes.is_empty();
                for item in self.to_items(&iv)? {
                    self.scope_insert(var.clone(), item);
                    let body_env_owned;
                    let body_env: &EnvRef = if at_top_level {
                        body_env_owned = self.global_env.clone();
                        &body_env_owned
                    } else {
                        env
                    };
                    match self.eval_in(body, body_env) {
                        Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                        Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                        Err(e) => return Err(e),
                        _ => {}
                    }
                }
                Ok(RVal::Null)
            }
            Expr::While { cond, body } => {
                // Same top-level re-snapshot rule as For.
                let at_top_level = self.local_scopes.is_empty();
                loop {
                    let cond_env_owned;
                    let cur_env: &EnvRef = if at_top_level {
                        cond_env_owned = self.global_env.clone();
                        &cond_env_owned
                    } else { env };
                    let c = self.eval_in(cond, cur_env)?;
                    if !self.truthy(&c)? { break; }
                    match self.eval_in(body, cur_env) {
                        Err(R2Err { kind: ErrKind::CtrlBreak, .. }) => break,
                        Err(R2Err { kind: ErrKind::CtrlNext, .. }) => continue,
                        Err(e) => return Err(e),
                        _ => {}
                    }
                }
                Ok(RVal::Null)
            }
            Expr::Match { expr: e, arms } => { let val = self.eval_in(e, env)?; for arm in arms { for pat in &arm.patterns { let pv = self.eval_in(pat, env)?; if self.vals_eq(&val, &pv) { return self.eval_in(&arm.body, env); } } } err!(Runtime, "no matching pattern") }
            Expr::FuncDef { params, body } | Expr::Lambda { params, body } => Ok(RVal::Closure(Closure { params: params.clone(), body: Arc::new((**body).clone()), env: env.clone() })),
            Expr::TypeDef { name, fields, parent } => { let td = TypeDef { name: name.clone(), fields: fields.clone(), parent: parent.clone() }; self.types.insert(name.clone(), td.clone()); env_insert(&mut self.global_env, name.clone(), RVal::TypeDef(td.clone())); Ok(RVal::TypeDef(td)) }
            Expr::MethodDef(m) => { self.methods.insert((m.name.clone(), m.type_name.clone()), m.clone()); Ok(RVal::Null) }
            Expr::TryCatch { body, var, catch } => { match self.eval_in(body, env) { Ok(v) => Ok(v), Err(e) => { self.scope_insert(var.clone(), rstr(&e.msg)); self.eval_in(catch, env) } } }
            Expr::Return(v) => { let val = self.eval_in(v, env)?; Err(R2Err { msg: String::new(), kind: ErrKind::CtrlReturn(Box::new(val)) }) }
            Expr::Break => Err(R2Err { msg: String::new(), kind: ErrKind::CtrlBreak }),
            Expr::Next => Err(R2Err { msg: String::new(), kind: ErrKind::CtrlNext }),
            Expr::Dots => Ok(RVal::Null),
            _ => err!(Runtime, "cannot evaluate expression"),
        }
    }

    fn call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
        match func {
            RVal::BuiltinFn(name) => {
                // Check for pkg::func namespaced call
                if let Some(sep) = name.find("::") {
                    let pkg = &name[..sep];
                    let fname = &name[sep+2..];
                    if let Some(f) = self.registry.resolve_in_package(pkg, fname) {
                        return f(self, args, env);
                    } else {
                        return err!(Runtime, "'{}' not found in package '{}'", fname, pkg);
                    }
                }
                // Normal resolution through search order
                if let Some((f, _pkg)) = self.registry.resolve(name.as_ref()) { f(self, args, env) }
                else { err!(Runtime, "unknown function '{}'", name) }
            }
            RVal::Closure(cl) => {
                // Recursion depth limit
                if self.local_scopes.len() >= 500 {
                    return err!(Runtime, "recursion depth limit exceeded (max 500). Use iteration instead.");
                }

                // ── JIT fast path (Phases C.2 + C.3) ────────────────────────
                if self.jit_enabled
                   && cl.params.len() == args.len()
                   && cl.params.iter().all(|p| !p.dots && p.default.is_none())
                {
                    // Resolve cache: try compile if not yet attempted.
                    let key = Arc::as_ptr(&cl.body) as usize;
                    let handle = match self.jit_cache.get(&key) {
                        Some(slot) => slot.clone(),
                        None => {
                            let h = r2_jit::try_compile_closure(cl);
                            self.jit_cache.insert(key, h.clone());
                            h
                        }
                    };
                    if let Some(h) = handle {
                        // ── JIT NA-aware zero-copy bridge (Phase F.3 unlock) ──
                        //
                        // Pre-F.3, every JIT call did:
                        //   1. allocate Vec<f64> from Vec<Option<f64>>, encoding None → NaN
                        //   2. run Cranelift loop on raw f64
                        //   3. allocate Vec<Option<f64>> from output, decoding NaN → None
                        // Two O(n) allocation passes and per-element branches both ways.
                        //
                        // Now: RVal::Numeric is Reals which caches an Arc<ColumnarF64>.
                        // - `col.values()` returns &[f64] — dense, contiguous, SIMD-friendly,
                        //   zero alloc (just a slice into existing buffer).
                        // - Cranelift loop still operates on raw f64 (NaN propagates correctly).
                        // - On the way out, we reconstruct Vec<Option<f64>> respecting the
                        //   INPUT bitmap rather than scanning the output for NaN: NA structure
                        //   is preserved exactly, not approximated via NaN encoding.
                        //
                        // Win: 1 alloc round-trip instead of 2, SIMD-friendly dense input,
                        // and structurally-correct NA semantics (NaN ≠ NA distinction kept).
                        match h.kind() {
                            r2_types::JitKind::Vector1ToScalar => {
                                if args.len() == 1 {
                                    if let RVal::Numeric(v, _) = &args[0].value {
                                        // Zero-copy: grab the cached columnar's dense f64 slice.
                                        // Reads None as NaN in the values buffer (already that way
                                        // by ColumnarF64::from_option_slice), so Cranelift's NaN
                                        // arithmetic propagates correctly through the reduction.
                                        let col = v.columnar();
                                        let values = col.values();
                                        let out = unsafe { h.try_call_vec1(values.as_ptr(), values.len() as i64) };
                                        if let Some(val) = out {
                                            return Ok(RVal::Numeric(vec![Some(val)].into(), Attrs::default()));
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorBinaryMap => {
                                // Two equal-length vectors → output preserves AND-of-bitmaps.
                                if args.len() == 2 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _)) = (&args[0].value, &args[1].value) {
                                        if a.len() == b.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; a.len()];
                                            let ok = unsafe { h.try_call_vec_binary(a_vals.as_ptr(), b_vals.as_ptr(), out_buf.as_mut_ptr(), a.len() as i64) };
                                            if ok {
                                                let a_bits = a_col.valid_bits();
                                                let b_bits = b_col.valid_bits();
                                                let result = combine_binary_output(&out_buf, a_bits, b_bits);
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorTernaryMap => {
                                // Three equal-length numeric vectors → vector.
                                // Output bitmap = AND of all three input bitmaps.
                                if args.len() == 3 {
                                    if let (RVal::Numeric(a, _), RVal::Numeric(b, _), RVal::Numeric(c, _)) =
                                        (&args[0].value, &args[1].value, &args[2].value)
                                    {
                                        if a.len() == b.len() && b.len() == c.len() && !a.is_empty() {
                                            let a_col = a.columnar();
                                            let b_col = b.columnar();
                                            let c_col = c.columnar();
                                            let a_vals = a_col.values();
                                            let b_vals = b_col.values();
                                            let c_vals = c_col.values();
                                            let mut out_buf: Vec<f64> = vec![0.0; a.len()];
                                            let ok = unsafe { h.try_call_vec_ternary(a_vals.as_ptr(), b_vals.as_ptr(), c_vals.as_ptr(), out_buf.as_mut_ptr(), a.len() as i64) };
                                            if ok {
                                                let result = combine_ternary_output(&out_buf, a_col.valid_bits(), b_col.valid_bits(), c_col.valid_bits());
                                                return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                            }
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::VectorMap => {
                                // Element-wise vector → vector. Output bitmap = input bitmap.
                                if args.len() == 1 {
                                    if let RVal::Numeric(v, _) = &args[0].value {
                                        let col = v.columnar();
                                        let values = col.values();
                                        let mut out_buf: Vec<f64> = vec![0.0; values.len()];
                                        let ok = unsafe { h.try_call_vec_map(values.as_ptr(), out_buf.as_mut_ptr(), values.len() as i64) };
                                        if ok {
                                            let bits = col.valid_bits();
                                            let result = combine_unary_output(&out_buf, bits);
                                            return Ok(RVal::Numeric(result.into(), Attrs::default()));
                                        }
                                    }
                                }
                            }
                            r2_types::JitKind::Scalar => {
                                let mut farg: Vec<f64> = Vec::with_capacity(args.len());
                                let mut all_scalar = true;
                                for ea in args {
                                    match &ea.value {
                                        RVal::Numeric(v, _) if v.len() == 1 => match v[0] { Some(x) => farg.push(x), None => { all_scalar = false; break; } },
                                        RVal::Integer(v, _) if v.len() == 1 => match v[0] { Some(x) => farg.push(x as f64), None => { all_scalar = false; break; } },
                                        RVal::Logical(v, _) if v.len() == 1 => match v[0] { Some(b) => farg.push(if b { 1.0 } else { 0.0 }), None => { all_scalar = false; break; } },
                                        _ => { all_scalar = false; break; }
                                    }
                                }
                                if all_scalar {
                                    if let Some(out) = h.try_call_real(&farg) {
                                        return Ok(RVal::Numeric(vec![Some(out)].into(), Attrs::default()));
                                    }
                                }
                            }
                        }
                    }
                }
                // ── Fallback: tree-walking interpreter (existing path) ──────
                let mut ce = Env::new_child(cl.env.clone(), None);
                let m = Arc::make_mut(&mut ce);
                for (i, p) in cl.params.iter().enumerate() { let v = self.get_arg(args, i, &p.name).or_else(|| p.default.as_ref().and_then(|d| self.eval_in(d, env).ok())).unwrap_or(RVal::Null); m.bindings.insert(p.name.clone(), v); }
                let func_env = Arc::new(m.clone());
                self.local_scopes.push(HashMap::new());
                let result = match self.eval_in(&cl.body, &func_env) { Err(R2Err { kind: ErrKind::CtrlReturn(v), .. }) => Ok(*v), r => r };
                self.local_scopes.pop();
                result
            }
            RVal::TypeDef(td) => { let mut fields = HashMap::new(); for (i, fd) in td.fields.iter().enumerate() { let v = self.get_arg(args, i, &fd.name).or_else(|| fd.default.clone()).unwrap_or(RVal::Null); fields.insert(fd.name.clone(), v); } Ok(RVal::TypeInstance(TypeInstance { type_name: td.name.clone(), fields })) }
            _ => err!(Runtime, "not callable as a function. Check spelling or use help() to find the right function name"),
        }
    }
    fn get_arg(&self, args: &[EvalArg], pos: usize, name: &str) -> Option<RVal> {
        args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone()).or_else(|| args.get(pos).map(|a| a.value.clone()))
    }

    fn binary_op(&mut self, op: BinOp, lhs: &RVal, rhs: &RVal) -> Result<RVal, R2Err> {
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
                if a.len() == b.len() && a.len() >= 64 {
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
                    // Preserve strict-mode division-by-zero semantics.
                    if (op == BinOp::Div || op == BinOp::Mod) && self.mode == ErrorMode::Strict {
                        if b.iter().any(|x| *x == Some(0.0)) {
                            return err!(Runtime, "division by zero");
                        }
                    }
                    let ac = a.columnar();
                    let bc = b.columnar();
                    let result = ac.binary(arrow_op, &bc)
                        .map_err(|e| R2Err { msg: e, kind: ErrKind::Runtime })?;
                    return Ok(RVal::Numeric(Reals::from_columnar(result), Attrs::default()));
                }
                // Scalar-vector recycling: vector OP scalar via binary_scalar.
                // Only safe when the scalar is not NA — propagate-NA path
                // falls back to the slow path below.
                if b.len() == 1 && a.len() >= 64 {
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
                        if (op == BinOp::Div || op == BinOp::Mod) && self.mode == ErrorMode::Strict && s == 0.0 {
                            return err!(Runtime, "division by zero");
                        }
                        let ac = a.columnar();
                        let result = ac.binary_scalar(arrow_op, s);
                        return Ok(RVal::Numeric(Reals::from_columnar(result), Attrs::default()));
                    }
                }
            }
        }

        let l = self.as_reals(lhs)?; let r = self.as_reals(rhs)?;
        let (ll, rl) = (l.len(), r.len());
        if ll != rl && ll != 1 && rl != 1 { if self.mode == ErrorMode::Strict { return err!(Runtime, "vectors length {} vs {} mismatch", ll, rl); } else { self.warnings.push(format!("Warning: recycling {} and {}", ll, rl)); } }
        let len = ll.max(rl);
        match op {
            BinOp::Eq|BinOp::Ne|BinOp::Lt|BinOp::Gt|BinOp::Le|BinOp::Ge => {
                let r: Vec<Logical> = (0..len).map(|i| { let (a,b) = (l[i%ll], r[i%rl]); match (a,b) { (Some(a),Some(b)) => Some(match op { BinOp::Eq => (a-b).abs()<f64::EPSILON, BinOp::Ne => (a-b).abs()>=f64::EPSILON, BinOp::Lt => a<b, BinOp::Gt => a>b, BinOp::Le => a<=b, BinOp::Ge => a>=b, _ => false }), _ => None } }).collect();
                Ok(RVal::Logical(r.into(), Attrs::default()))
            }
            _ => {
                // Strict mode: division by zero check before computation
                if (op == BinOp::Div || op == BinOp::Mod || op == BinOp::IntDiv) && self.mode == ErrorMode::Strict {
                    if r.iter().any(|x| *x == Some(0.0)) { return err!(Runtime, "division by zero"); }
                }
                let r: Vec<Real> = (0..len).map(|i| { let (a,b) = (l[i%ll], r[i%rl]); match (a,b) { (Some(a),Some(b)) => Some(match op { BinOp::Add => a+b, BinOp::Sub => a-b, BinOp::Mul => a*b, BinOp::Div => a/b, BinOp::Pow => a.powf(b), BinOp::Mod => a%b, BinOp::IntDiv => (a/b).floor(), _ => 0.0 }), _ => None } }).collect(); Ok(RVal::Numeric(r.into(), Attrs::default()))
            }
        }
    }
    fn unary_op(&self, op: UnOp, v: &RVal) -> Result<RVal, R2Err> { match op { UnOp::Neg => { let r = self.as_reals(v)?; Ok(RVal::Numeric(r.into_iter().map(|x| x.map(|n| -n)).collect(), Attrs::default())) } UnOp::Pos => Ok(v.clone()), UnOp::Not => { let r = self.as_logicals(v)?; Ok(RVal::Logical(r.into_iter().map(|x| x.map(|b| !b)).collect(), Attrs::default())) } } }
    fn seq_colon(&self, l: &RVal, r: &RVal) -> Result<RVal, R2Err> { let from = self.scalar_f64(l)?.ok_or(R2Err{msg:"NA in seq".into(),kind:ErrKind::Runtime})? as i32; let to = self.scalar_f64(r)?.ok_or(R2Err{msg:"NA in seq".into(),kind:ErrKind::Runtime})? as i32; let s: Vec<Integer> = if from<=to { (from..=to).map(Some).collect() } else { (to..=from).rev().map(Some).collect() }; Ok(RVal::Integer(s.into(), Attrs::default())) }
    fn index_obj(&self, obj: &RVal, idx: &[Option<RVal>]) -> Result<RVal, R2Err> {
        if idx.len()==1 {
            if let Some(i) = &idx[0] {
                // 1D indexing of a Matrix → column-major linear access, returning a Numeric vector
                if let RVal::Matrix(m) = obj {
                    let pos = self.as_reals(i)?;
                    let total = m.nrow * m.ncol;
                    let mut out = Vec::with_capacity(pos.len());
                    for p in &pos {
                        match p {
                            Some(k) => {
                                let k = *k as usize;
                                if k == 0 || k > total {
                                    if self.mode == ErrorMode::Strict { return err!(Index, "index {} out of bounds (matrix has {} elements)", k, total); }
                                    out.push(None);
                                } else {
                                    let v = m.data[k - 1];
                                    out.push(if v.is_nan() { None } else { Some(v) });
                                }
                            }
                            None => out.push(None),
                        }
                    }
                    return Ok(RVal::Numeric(out.into(), Attrs::default()));
                }
                return self.index_1d(obj, i);
            }
        }
        if idx.len()==2 {
            if let RVal::DataFrame(df) = obj { return self.index_df(df, &idx[0], &idx[1]); }
            if let RVal::Matrix(m) = obj { return self.index_matrix(m, &idx[0], &idx[1]); }
        }
        err!(Runtime, "invalid indexing")
    }
    fn index_matrix(&self, m: &Matrix, row: &Option<RVal>, col: &Option<RVal>) -> Result<RVal, R2Err> {
        // Resolve rows
        let keep_rows: Vec<usize> = match row {
            None => (0..m.nrow).collect(),
            Some(RVal::Logical(mask, _)) => mask.iter().enumerate().filter_map(|(i, b)| if *b == Some(true) { Some(i) } else { None }).collect(),
            Some(idx) => {
                let pos = self.as_reals(idx)?;
                pos.iter().filter_map(|p| p.map(|v| {
                    let i = v as usize;
                    if i >= 1 && i <= m.nrow { Some(i - 1) } else { None }
                }).flatten()).collect()
            }
        };
        // Resolve columns
        let keep_cols: Vec<usize> = match col {
            None => (0..m.ncol).collect(),
            Some(RVal::Logical(mask, _)) => mask.iter().enumerate().filter_map(|(j, b)| if *b == Some(true) { Some(j) } else { None }).collect(),
            Some(idx) => {
                let pos = self.as_reals(idx)?;
                pos.iter().filter_map(|p| p.map(|v| {
                    let j = v as usize;
                    if j >= 1 && j <= m.ncol { Some(j - 1) } else { None }
                }).flatten()).collect()
            }
        };
        // Single element → scalar Numeric
        if keep_rows.len() == 1 && keep_cols.len() == 1 {
            let v = m.data[keep_cols[0] * m.nrow + keep_rows[0]];
            return Ok(RVal::Numeric(vec![if v.is_nan() { None } else { Some(v) }].into(), Attrs::default()));
        }
        // Single column or single row → drop to vector (R's default `drop=TRUE`)
        if keep_cols.len() == 1 {
            let j = keep_cols[0];
            let out: Vec<Real> = keep_rows.iter().map(|&i| {
                let v = m.data[j * m.nrow + i];
                if v.is_nan() { None } else { Some(v) }
            }).collect();
            return Ok(RVal::Numeric(out.into(), Attrs::default()));
        }
        if keep_rows.len() == 1 {
            let i = keep_rows[0];
            let out: Vec<Real> = keep_cols.iter().map(|&j| {
                let v = m.data[j * m.nrow + i];
                if v.is_nan() { None } else { Some(v) }
            }).collect();
            return Ok(RVal::Numeric(out.into(), Attrs::default()));
        }
        // General submatrix → Matrix (column-major)
        let mut data = Vec::with_capacity(keep_rows.len() * keep_cols.len());
        for &j in &keep_cols {
            for &i in &keep_rows {
                data.push(m.data[j * m.nrow + i]);
            }
        }
        let mut out = Matrix::new(data, keep_rows.len(), keep_cols.len());
        if let Some(cn) = &m.col_names {
            out.col_names = Some(keep_cols.iter().map(|&j| cn[j].clone()).collect());
        }
        if let Some(rn) = &m.row_names {
            out.row_names = Some(keep_rows.iter().map(|&i| rn[i].clone()).collect());
        }
        Ok(RVal::Matrix(out))
    }
    fn index_1d(&self, obj: &RVal, idx: &RVal) -> Result<RVal, R2Err> { match idx { RVal::Logical(mask,_) => self.logical_sub(obj,mask), _ => { let pos = self.as_reals(idx)?; self.pos_sub(obj,&pos) } } }
    fn pos_sub(&self, obj: &RVal, pos: &[Real]) -> Result<RVal, R2Err> { match obj { RVal::Numeric(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { if self.mode==ErrorMode::Strict { return err!(Index,"index {} out of bounds (len {})",i,v.len()); } r.push(None); } else { r.push(v[i-1]); } } None => r.push(None), } } Ok(RVal::Numeric(r.into(), Attrs::default())) } RVal::Character(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { r.push(None); } else { r.push(v[i-1].clone()); } } None => r.push(None), } } Ok(RVal::Character(r, Attrs::default())) } RVal::Integer(v,_) => { let mut r = Vec::new(); for p in pos { match p { Some(i) => { let i = *i as usize; if i==0||i>v.len() { r.push(None); } else { r.push(v[i-1]); } } None => r.push(None), } } Ok(RVal::Integer(r.into(), Attrs::default())) } _ => err!(Index,"cannot subset {}",obj.type_name()), } }
    fn logical_sub(&self, obj: &RVal, mask: &[Logical]) -> Result<RVal, R2Err> { match obj { RVal::Numeric(v,_) => Ok(RVal::Numeric(v.iter().zip(mask.iter().chain(std::iter::repeat(&None))).filter_map(|(val,m)| if *m==Some(true) { Some(*val) } else { None }).collect(), Attrs::default())), _ => err!(Index,"logical subset not impl for {}",obj.type_name()) } }
    fn index_df(&self, df: &DataFrame, row: &Option<RVal>, col: &Option<RVal>) -> Result<RVal, R2Err> {
        // Determine which rows to keep
        let nrow = df.nrow();
        let keep_rows: Vec<usize> = match row {
            None => (0..nrow).collect(), // all rows
            Some(RVal::Logical(mask, _)) => {
                mask.iter().enumerate().filter_map(|(i, m)| if *m == Some(true) { Some(i) } else { None }).collect()
            }
            Some(idx) => {
                let positions = self.as_reals(idx)?;
                positions.iter().filter_map(|p| p.map(|v| {
                    let i = v as usize;
                    if i >= 1 && i <= nrow { Some(i - 1) } else { None }
                }).flatten()).collect()
            }
        };

        // Determine which columns to keep
        let ncol = df.ncol();
        let keep_cols: Vec<usize> = match col {
            None => (0..ncol).collect(), // all columns
            Some(RVal::Character(names, _)) => {
                names.iter().filter_map(|n| n.as_ref().and_then(|name| {
                    df.columns.iter().position(|(cn, _)| cn.as_ref() == name.as_ref())
                })).collect()
            }
            Some(idx) => {
                let positions = self.as_reals(idx)?;
                positions.iter().filter_map(|p| p.map(|v| {
                    let i = v as usize;
                    if i >= 1 && i <= ncol { Some(i - 1) } else { None }
                }).flatten()).collect()
            }
        };

        // If single column selected, return as vector
        if keep_cols.len() == 1 && row.is_none() {
            return Ok(df.columns[keep_cols[0]].1.clone());
        }

        // Build new DataFrame
        let new_cols: Vec<(Arc<str>, RVal)> = keep_cols.iter().map(|&ci| {
            let (name, col) = &df.columns[ci];
            let new_col = self.subset_col_by_rows(col, &keep_rows);
            (name.clone(), new_col)
        }).collect();

        Ok(RVal::DataFrame(DataFrame { columns: new_cols, row_names: None }))
    }

    fn subset_col_by_rows(&self, col: &RVal, rows: &[usize]) -> RVal {
        match col {
            RVal::Numeric(v, _) => RVal::Numeric(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(rows.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(rows.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            _ => col.clone(),
        }
    }
    fn dollar(&self, obj: &RVal, field: &str) -> Result<RVal, R2Err> { match obj { RVal::DataFrame(df) => df.get_col(field).cloned().ok_or(R2Err{msg:format!("column '{}' not found",field),kind:ErrKind::Runtime}), RVal::List(items) => { for (n,v) in items { if n.as_ref().map(|s| s.as_ref())==Some(field) { return Ok(v.clone()); } } err!(Runtime,"'{}' not in list",field) } RVal::TypeInstance(inst) => inst.fields.get(field).cloned().ok_or(R2Err{msg:format!("field '{}' not found",field),kind:ErrKind::Runtime}), _ => err!(Runtime,"$ applied to {}",obj.type_name()), } }
    // Phase R.1 step 2: coercion methods extracted to RVal methods in
    // r2-types. Engine wrappers retained so existing call sites
    // (`e.as_reals(arg)`, `e.scalar_f64(arg)`) keep working unchanged.
    // New code can call `arg.as_reals()` / `arg.scalar_f64()` directly,
    // bypassing the engine — required by domain crates that don't see
    // the `Engine` type (r2-stats, r2-ml).
    pub fn as_reals(&self, obj: &RVal) -> Result<Vec<Real>, R2Err> { obj.as_reals() }
    pub fn as_logicals(&self, obj: &RVal) -> Result<Vec<Logical>, R2Err> { obj.as_logicals() }
    fn scalar_f64(&self, obj: &RVal) -> Result<Real, R2Err> { obj.scalar_f64() }
    fn truthy(&self, obj: &RVal) -> Result<bool, R2Err> { match obj { RVal::Logical(v,_) => v.first().copied().flatten().ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), RVal::Numeric(v,_) => v.first().copied().flatten().map(|n| n!=0.0).ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), _ => err!(Type,"cannot coerce {} to logical",obj.type_name()) } }
    fn vals_eq(&self, a: &RVal, b: &RVal) -> bool { match (a,b) { (RVal::Numeric(a,_),RVal::Numeric(b,_)) => a==b, (RVal::Character(a,_),RVal::Character(b,_)) => a==b, (RVal::Integer(a,_),RVal::Integer(b,_)) => a==b, _ => false } }
    fn to_items(&self, obj: &RVal) -> Result<Vec<RVal>, R2Err> { match obj { RVal::Integer(v,_) => Ok(v.iter().map(|x| RVal::Integer(vec![*x].into(),Attrs::default())).collect()), RVal::Numeric(v,_) => Ok(v.iter().map(|x| RVal::Numeric(vec![*x].into(),Attrs::default())).collect()), RVal::Character(v,_) => Ok(v.iter().map(|x| RVal::Character(vec![x.clone()],Attrs::default())).collect()), RVal::List(v) => Ok(v.iter().map(|(_,val)| val.clone()).collect()), _ => err!(Runtime,"cannot iterate over {}",obj.type_name()) } }
    pub fn drain_warnings(&mut self) -> Vec<String> { std::mem::take(&mut self.warnings) }

    /// Insert into the correct scope: local (inside function) or global (top-level)
    fn scope_insert(&mut self, name: Arc<str>, val: RVal) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, val);
        } else {
            env_insert(&mut self.global_env, name, val);
        }
    }

    /// Resolve a formula term: bare symbol → column in data.frame, else evaluate normally
    fn resolve_formula_term(&mut self, expr: &Expr, df: &DataFrame, env: &EnvRef) -> Result<RVal, R2Err> {
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

    // ── Subscript assignment helpers ─────────────────────────────────

    fn assign_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        let positions = self.as_reals(idx)?;
        match obj {
            RVal::Numeric(v, _) => {
                let new_vals = self.as_reals(val)?;
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid (1-based indexing)"); }
                        // Extend vector if needed
                        // Reals: DerefMut to &mut [Real] doesn't allow push.
                        // Move out, push, move back via .into() reconstruction.
                        let mut tmp: Vec<Real> = std::mem::take(&mut *v).into_inner();
                        while tmp.len() < i { tmp.push(None); }
                        tmp[i - 1] = new_vals.get(pi % new_vals.len()).copied().unwrap_or(None);
                        *v = tmp.into();
                    }
                }
                Ok(())
            }
            RVal::Character(v, _) => {
                let new_val = match val { RVal::Character(sv, _) => sv.clone(), _ => vec![Some(Arc::from(val_to_str(val).as_str()))] };
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                        while v.len() < i { v.push(None); }
                        v[i - 1] = new_val.get(pi % new_val.len()).cloned().unwrap_or(None);
                    }
                }
                Ok(())
            }
            RVal::Integer(v, _) => {
                let new_vals = self.as_reals(val)?;
                // Ints/Logicals share the F.3 pattern: DerefMut gives a
                // slice not a Vec, so push/extend need a take→push→put-back.
                let mut tmp: Vec<Integer> = std::mem::take(&mut *v).into_inner();
                for (pi, pos) in positions.iter().enumerate() {
                    if let Some(p) = pos {
                        let i = *p as usize;
                        if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                        while tmp.len() < i { tmp.push(None); }
                        tmp[i - 1] = new_vals.get(pi % new_vals.len()).copied().unwrap_or(None).map(|n| n as i32);
                    }
                }
                *v = tmp.into();
                Ok(())
            }
            _ => err!(Runtime, "cannot assign by index to {}", obj.type_name()),
        }
    }

    fn assign_dbl_index(&mut self, obj: &mut RVal, idx: &RVal, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::List(items) => {
                let i = self.scalar_f64(idx)?.unwrap_or(1.0) as usize;
                if i == 0 { return err!(Runtime, "index 0 is not valid"); }
                while items.len() < i { items.push((None, RVal::Null)); }
                items[i - 1].1 = val.clone();
                Ok(())
            }
            _ => self.assign_index(obj, idx, val),
        }
    }

    fn assign_dollar(&mut self, obj: &mut RVal, field: &str, val: &RVal) -> Result<(), R2Err> {
        match obj {
            RVal::DataFrame(df) => {
                // Find existing column or add new
                if let Some(pos) = df.columns.iter().position(|(n, _)| n.as_ref() == field) {
                    df.columns[pos].1 = val.clone();
                } else {
                    df.columns.push((Arc::from(field), val.clone()));
                }
                Ok(())
            }
            RVal::List(items) => {
                let field_arc = Arc::from(field);
                if let Some(pos) = items.iter().position(|(n, _)| n.as_ref().map(|s| s.as_ref()) == Some(field)) {
                    items[pos].1 = val.clone();
                } else {
                    items.push((Some(field_arc), val.clone()));
                }
                Ok(())
            }
            RVal::TypeInstance(inst) => {
                inst.fields.insert(Arc::from(field), val.clone());
                Ok(())
            }
            _ => err!(Runtime, "$ assignment not supported for {}", obj.type_name()),
        }
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

fn val_to_str(v: &RVal) -> String { match v { RVal::Numeric(v,_) => v.iter().map(|x| match x {Some(n)=>fmt_num(*n),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Single(v,_) => v.iter().map(|x| match x {Some(n)=>fmt_num(*n as f64),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Integer(v,_) => v.iter().map(|x| match x {Some(n)=>format!("{}",n),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Character(v,_) => v.iter().map(|x| match x {Some(s)=>s.to_string(),None=>"NA".into()}).collect::<Vec<_>>().join(" "), RVal::Logical(v,_) => v.iter().map(|x| match x {Some(true)=>"TRUE",Some(false)=>"FALSE",None=>"NA"}).collect::<Vec<_>>().join(" "), RVal::Null => "NULL".into(), _ => format!("<{}>",v.type_name()) } }

// ═══════════════════════════════════════════════════════════════════════
// BUILTINS
// ═══════════════════════════════════════════════════════════════════════

// Phase R.2: bi_c moved to r2-data::concat. Engine adapter only.
// (bi_c moved to src/builtins/data.rs.)
fn bi_length(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rint(rval_length(&gv(a,0)) as i32)) }
fn bi_print(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    // Phase R.T.1 — class-aware print for Date / POSIXct. R prints these in
    // human form (`"2024-03-15"`) rather than the raw days/seconds f64. We
    // dispatch here instead of inside Display because the formatter lives in
    // r2-stats and we don't want r2-types depending on r2-stats.
    if let RVal::Numeric(xs, attrs) = &v {
        match attrs.class.as_deref() {
            Some("Date") => {
                let strs: Vec<String> = xs.iter()
                    .map(|x| match x { Some(d) => format!("\"{}\"", r2_stats::time::format_date(*d, "%Y-%m-%d")), None => "NA".into() })
                    .collect();
                e.emit_output(&quoted_vec(&strs));
                return Ok(v);
            }
            Some("xts") => {
                let dim = attrs.dim.as_ref();
                let idx_v = attrs.custom.get(&std::sync::Arc::from("index"));
                let cls_v = attrs.custom.get(&std::sync::Arc::from("index.class"));
                if let (Some(d), Some(RVal::Numeric(idx, _)), Some(RVal::Character(cls, _))) = (dim, idx_v, cls_v) {
                    if d.len() == 2 {
                        let nrow = d[0];
                        let ncol = d[1];
                        let xs_vec: Vec<Option<f64>> = xs.iter().copied().collect();
                        let idx_vec: Vec<f64> = idx.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
                        let cls_s = cls.first().and_then(|x| x.as_ref().map(|s| s.to_string())).unwrap_or_else(|| "POSIXct".into());
                        e.emit_output(&r2_stats::time::format_xts(&xs_vec, nrow, ncol, &idx_vec, &cls_s, attrs.names.as_deref()));
                        return Ok(v);
                    }
                }
            }
            Some("ts") => {
                if let Some(RVal::Numeric(tsp, _)) = attrs.custom.get(&std::sync::Arc::from("tsp")) {
                    if tsp.len() == 3 {
                        let s = tsp[0].unwrap_or(f64::NAN);
                        let e2 = tsp[1].unwrap_or(f64::NAN);
                        let f = tsp[2].unwrap_or(1.0);
                        let xs_vec: Vec<Option<f64>> = xs.iter().copied().collect();
                        e.emit_output(&r2_stats::time::format_ts(&xs_vec, s, e2, f));
                        return Ok(v);
                    }
                }
            }
            Some("POSIXct") => {
                let strs: Vec<String> = xs.iter()
                    .map(|x| match x { Some(s) => format!("\"{}\"", r2_stats::time::format_posixct(*s, "%Y-%m-%d %H:%M:%S")), None => "NA".into() })
                    .collect();
                e.emit_output(&quoted_vec(&strs));
                return Ok(v);
            }
            _ => {}
        }
    }
    e.emit_output(&format!("{}", v));
    Ok(v)
}

// R-style `[1] "a" "b"` formatter for vector of already-quoted strings.
fn quoted_vec(strs: &[String]) -> String {
    let mut s = String::from("[1] ");
    for (i, x) in strs.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(x);
    }
    s
}
fn bi_cat(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let sep = gn(a,"sep").map(|v| val_to_str(&v)).unwrap_or(" ".into());
    let parts: Vec<String> = a.iter()
        .filter(|x| x.name.as_ref().map(|n| n.as_ref()) != Some("sep"))
        .map(|x| val_to_str(&x.value))
        .collect();
    // cat() does NOT auto-newline (R behavior). The sink's contract
    // is one logical "output chunk" per call — if the assembled text
    // has no trailing newline, the sink may or may not add one
    // depending on impl. StdoutSink adds one for line-buffered I/O.
    e.emit_output(&parts.join(&sep));
    Ok(RVal::Null)
}
fn bi_typeof(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rstr(gv(a,0).type_name())) }
fn bi_class(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::TypeInstance(i) => Ok(rstr(&i.type_name)), v => Ok(rstr(v.type_name())) } }
fn bi_is_na(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Numeric(v,_) => Ok(RVal::Logical(v.iter().map(|x| Some(x.is_none())).collect(), Attrs::default())), _ => Ok(rbool(false)) } }
fn bi_seq(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let from = e.scalar_f64(&gv(a,0))?.unwrap_or(1.0); let to = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0); let by = gn(a,"by").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(if from<=to {1.0} else {-1.0}); let mut r = Vec::new(); let mut c = from; if by>0.0 { while c<=to+1e-10 { r.push(Some(c)); c+=by; } } else if by<0.0 { while c>=to-1e-10 { r.push(Some(c)); c+=by; } } Ok(RVal::Numeric(r.into(), Attrs::default())) }
fn bi_rep(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    // `times = ` (default 1) and `each = ` (default 1). Both supported,
    // matching R semantics: `rep(c("A","B"), each=3)` → A A A B B B,
    // `rep(c("A","B"), times=3)` → A B A B A B.
    // Critical: arg 1 may be a NAMED arg (`each = 3`), not a positional
    // `times`. Filter on `name.is_none()` before falling back, otherwise
    // `rep(c("A","B","C"), each=3)` reads `times=3` AND `each=3`, giving
    // 27 entries instead of 9.
    let times = gn(a, "times")
        .or_else(|| a.get(1).filter(|p| p.name.is_none()).map(|p| p.value.clone()))
        .and_then(|v| e.scalar_f64(&v).ok().flatten())
        .unwrap_or(1.0) as usize;
    let each = gn(a, "each").and_then(|v| e.scalar_f64(&v).ok().flatten())
        .unwrap_or(1.0) as usize;
    fn expand<T: Clone>(v: &[T], each: usize, times: usize) -> Vec<T> {
        let per_pass: Vec<T> = v.iter().flat_map(|x| std::iter::repeat(x.clone()).take(each)).collect();
        per_pass.iter().cycle().take(per_pass.len() * times).cloned().collect()
    }
    match &v {
        RVal::Numeric(vs, _)   => Ok(RVal::Numeric(expand(vs, each, times).into(), Attrs::default())),
        RVal::Integer(vs, _)   => Ok(RVal::Integer(expand(vs, each, times).into(), Attrs::default())).into(),
        RVal::Character(vs, _) => Ok(RVal::Character(expand(vs, each, times), Attrs::default())).into(),
        RVal::Logical(vs, _)   => Ok(RVal::Logical(expand(vs, each, times).into(), Attrs::default())).into(),
        _ => err!(Runtime, "rep() not supported for {}", v.type_name()).into(),
    }
}
// Phase R: 8 reduction builtins now live in r2-stats. r2-engine adapts
// the pure `(&[EvalArg]) -> Result<RVal, R2Err>` signature to the local
// `BuiltinFn` shape (which carries `&mut Engine` and `&EnvRef`).
// (bi_sum moved to src/builtins/stats.rs.)

// (bi_mean/sd/var moved to src/builtins/stats.rs.)
// (bi_paste moved to src/builtins/strings.rs.)
// (bi_paste0 moved to src/builtins/strings.rs.)
// (bi_head / bi_tail moved to src/builtins/data.rs.)
fn bi_which(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Logical(v,_) => Ok(RVal::Integer(v.iter().enumerate().filter_map(|(i,x)| if *x==Some(true) { Some(Some((i+1) as i32)) } else { None }).collect(), Attrs::default())), _ => err!(Type, "which requires logical") } }
// Phase K.2: map-kernel dispatch — Rayon decision lives below this layer.
fn bi_abs(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Abs, &v).into(), Attrs::default()))
}
fn bi_sqrt(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sqrt, &v).into(), Attrs::default()))
}
fn bi_round(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let d = e.scalar_f64(&gv(a,1))?.unwrap_or(0.0) as i32; let f = 10f64.powi(d); Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| (n*f).round()/f)).collect(), Attrs::default())) }
// (bi_max/bi_min moved to src/builtins/stats.rs.)
fn bi_sort(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; let mut n: Vec<f64> = v.into_iter().filter_map(|x| x).collect(); n.sort_by(|a,b| a.partial_cmp(b).unwrap()); Ok(rnums(&n)) }
fn bi_rev(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::Numeric(v,_) => Ok(RVal::Numeric(v.iter().rev().cloned().collect(), Attrs::default())), _ => err!(Runtime, "rev() works with numeric, integer, or character vectors") } }
// (bi_unique moved to src/builtins/data.rs.)
// (bi_nchar moved to src/builtins/strings.rs.)
fn bi_is_num(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Numeric(..)|RVal::Integer(..)))) }
fn bi_is_chr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Character(..)))) }
fn bi_is_lgl(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(rbool(matches!(gv(a,0), RVal::Logical(..)))) }
fn bi_as_num(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(RVal::Numeric(e.as_reals(&gv(a,0))?.into(), Attrs::default())) }
/// `as.single(x)` — coerce to f32 single-precision storage (Phase F.7).
/// Halves memory footprint vs `as.numeric`; arithmetic with `numeric`
/// promotes back to f64.
fn bi_as_single(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let singles = v.as_singles()?;
    Ok(RVal::Single(Singles::new(singles), Attrs::default()))
}
fn bi_is_single(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a,0), RVal::Single(..))))
}
fn bi_as_chr(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::Character(v, _) => Ok(RVal::Character(v.clone(), Attrs::default())),
        RVal::Numeric(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|n| Arc::from(fmt_num(n).as_str()))).collect(), Attrs::default())),
        RVal::Integer(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|n| Arc::from(format!("{}", n).as_str()))).collect(), Attrs::default())),
        RVal::Logical(v, _) => Ok(RVal::Character(v.iter().map(|x| x.map(|b| Arc::from(if b { "TRUE" } else { "FALSE" }))).collect(), Attrs::default())),
        RVal::Factor(f) => Ok(RVal::Character(f.codes.iter().map(|c| c.and_then(|i| f.levels.get(i as usize).cloned())).collect(), Attrs::default())),
        _ => Ok(rstr(&val_to_str(&gv(a,0)))),
    }
}
fn bi_as_int(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let v = e.as_reals(&gv(a,0))?; Ok(RVal::Integer(v.into_iter().map(|x| x.map(|n| n as i32)).collect(), Attrs::default())) }
fn bi_strict(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { e.mode = ErrorMode::Strict; println!("Mode: strict"); Ok(RVal::Null) }
fn bi_lenient(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { e.mode = ErrorMode::Lenient; println!("Mode: lenient"); Ok(RVal::Null) }
fn bi_df(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { let cols: Vec<(Arc<str>, RVal)> = a.iter().enumerate().map(|(i,arg)| { let n = arg.name.clone().unwrap_or_else(|| Arc::from(format!("V{}",i+1).as_str())); (n, arg.value.clone()) }).collect(); Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: None })) }
fn bi_list(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(RVal::List(a.iter().map(|x| (x.name.clone(), x.value.clone())).collect())) }

/// `list.meta(lst)` — introspect a list's per-component shape.
///
/// Returns a list with three named fields:
///   - `$kinds`: character vector of RVal-variant tags per component
///   - `$lens`: integer vector of component lengths
///   - `$total_work`: integer scalar — aggregate work across components
///   - `$homogeneous`: character scalar (`""` if mixed types) — same kind
///                    everywhere when non-empty
///
/// User code can use this to decide whether/how to parallelize over a
/// list's components, mirroring what the engine's auto-dispatch does.
/// Maps onto `r2_types::list_meta()`.
fn bi_list_meta(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let lst = a.first().map(|x| &x.value).ok_or_else(|| R2Err {
        msg: "list.meta: needs a list argument".into(),
        kind: ErrKind::Runtime,
    })?;
    let items = match lst {
        RVal::List(items) => items.clone(),
        _ => return Err(R2Err {
            msg: format!("list.meta: not a list (got {})", lst.type_name()),
            kind: ErrKind::Type,
        }),
    };
    let meta = r2_types::list_meta(&items);
    let kinds: Vec<Character> = meta.components.iter()
        .map(|c| Some(std::sync::Arc::from(c.kind))).collect();
    let lens: Vec<Integer> = meta.components.iter()
        .map(|c| Some(c.len as i32)).collect();
    let homog = match meta.homogeneous_kind {
        Some(k) => std::sync::Arc::from(k),
        None => std::sync::Arc::from(""),
    };
    let mut fields: HashMap<Arc<str>, RVal> = HashMap::new();
    fields.insert(Arc::from("kinds"),       RVal::Character(kinds, Attrs::default()));
    fields.insert(Arc::from("lens"),        RVal::Integer(lens.into(), Attrs::default()));
    fields.insert(Arc::from("total_work"),  RVal::Integer(vec![Some(meta.total_work as i32)].into(), Attrs::default()));
    fields.insert(Arc::from("homogeneous"), RVal::Character(vec![Some(homog)], Attrs::default()));
    Ok(RVal::List(fields.into_iter().map(|(k, v)| (Some(k), v)).collect()))
}

/// GLM family constructors. R's `glm(..., family = binomial())` calls
/// `binomial()` as a function returning a family descriptor. Engine's
/// `bi_glm` consumes either the descriptor list or the bare string
/// `"binomial"` / `"gaussian"` / `"poisson"`. Returning a tagged list
/// keeps the call path R-compatible.
fn make_family(name: &'static str, link: &'static str) -> RVal {
    RVal::List(vec![
        (Some(Arc::from("family")), rstr(name)),
        (Some(Arc::from("link")), rstr(link)),
        (Some(Arc::from("~class")), rstr("family")),
    ])
}
fn bi_binomial(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("binomial", "logit")) }
fn bi_gaussian(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("gaussian", "identity")) }
fn bi_poisson(_:  &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { Ok(make_family("poisson", "log")) }

/// `subset(df, mask)` — keep rows where `mask` is TRUE.
///
/// NSE form `subset(df, x > 2)` (where `x` resolves against df columns) is
/// supported: the engine pre-processor (see `Expr::Call` dispatch above)
/// evaluates the condition expression in a child env that binds the
/// data-frame's columns as variables, then passes the resulting logical
/// vector to this builtin. Compound conditions like `subset(df, x > 1 & y < 50)`
/// work too. Integration tests live in `tests/nse_subset_transform.rs`.
fn bi_subset(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return err!(Type, "subset() needs a data.frame"),
    };
    let mask: Vec<bool> = e.as_logicals(&gv(a, 1))?
        .into_iter().map(|x| x == Some(true)).collect();
    if mask.len() != df.nrow() {
        return err!(Runtime, "subset: mask length ({}) != nrow ({})", mask.len(), df.nrow());
    }
    fn pick<T: Clone>(v: &[T], m: &[bool]) -> Vec<T> {
        v.iter().zip(m).filter_map(|(x, k)| if *k { Some(x.clone()) } else { None }).collect()
    }
    let cols: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let filtered = match col {
            RVal::Numeric(v, _)   => RVal::Numeric(pick(v, &mask).into(), Attrs::default()),
            RVal::Integer(v, _)   => RVal::Integer(pick(v, &mask).into(), Attrs::default()).into(),
            RVal::Character(v, _) => RVal::Character(pick(v, &mask), Attrs::default()).into(),
            RVal::Logical(v, _)   => RVal::Logical(pick(v, &mask).into(), Attrs::default()).into(),
            _ => col.clone().into(),
        };
        (name.clone(), filtered)
    }).collect();
    Ok(RVal::DataFrame(DataFrame { columns: cols, row_names: None }))
}

/// `transform(df, name = expr)` — append/overwrite named columns.
///
/// NSE form `transform(df, z = x + y)` is supported: the engine
/// pre-processor evaluates each `name = expr` value in a child env binding
/// df columns, so `x` and `y` resolve to the data-frame's columns rather
/// than the global env. Integration tests in `tests/nse_subset_transform.rs`.
fn bi_transform(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return err!(Type, "transform() needs a data.frame"),
    };
    for arg in a.iter().skip(1) {
        let name = match &arg.name {
            Some(n) => n.clone(),
            None => continue, // unnamed extras ignored
        };
        // Replace if column already exists, else append.
        if let Some(pos) = df.columns.iter().position(|(n, _)| n == &name) {
            df.columns[pos] = (name, arg.value.clone());
        } else {
            df.columns.push((name, arg.value.clone()));
        }
    }
    Ok(RVal::DataFrame(df))
}
fn bi_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // R's `factor()` coerces numeric/integer/logical to character first.
    // We do the same — converting to string keys and building the levels
    // in order of first appearance.
    let strs: Vec<Option<Arc<str>>> = match &gv(a, 0) {
        RVal::Character(v, _) => v.clone(),
        RVal::Numeric(v, _) => v.iter()
            .map(|x| x.map(|n| Arc::from(fmt_num(n).as_str()))).collect(),
        RVal::Integer(v, _) => v.iter()
            .map(|x| x.map(|n| Arc::from(format!("{}", n).as_str()))).collect(),
        RVal::Logical(v, _) => v.iter()
            .map(|x| x.map(|b| Arc::from(if b { "TRUE" } else { "FALSE" }))).collect(),
        other => return err!(Type, "factor() not supported for {}", other.type_name()),
    };
    let mut levels: Vec<Arc<str>> = Vec::new();
    let codes: Vec<Option<u32>> = strs.iter().map(|x| x.as_ref().map(|s| {
        let idx = levels.iter().position(|l: &Arc<str>| l == s).unwrap_or_else(|| {
            levels.push(s.clone()); levels.len() - 1
        });
        idx as u32
    })).collect();
    Ok(RVal::Factor(Factor { codes, levels, ordered: false }))
}
fn bi_names(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { match &gv(a,0) { RVal::DataFrame(df) => Ok(RVal::Character(df.columns.iter().map(|(n,_)| Some(n.clone())).collect(), Attrs::default())), _ => Ok(RVal::Null) } }
// (bi_nrow / bi_ncol moved to src/builtins/data.rs.)
fn bi_str(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    match &v {
        RVal::DataFrame(df) => {
            println!("'data.frame':  {} obs. of  {} variables:", df.nrow(), df.ncol());
            for (n, c) in &df.columns {
                let preview = match c {
                    RVal::Numeric(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect();
                        format!(" num  {}", vals.join(" "))
                    }
                    RVal::Integer(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect();
                        format!(" int  {}", vals.join(" "))
                    }
                    RVal::Character(v, _) => {
                        let vals: Vec<String> = v.iter().take(4).map(|x| match x { Some(s) => format!("\"{}\"", s), None => "NA".into() }).collect();
                        format!(" chr  {}", vals.join(" "))
                    }
                    RVal::Logical(v, _) => {
                        let vals: Vec<String> = v.iter().take(6).map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect();
                        format!(" logi {}", vals.join(" "))
                    }
                    RVal::Factor(f) => {
                        let vals: Vec<String> = f.codes.iter().take(6).map(|x| match x { Some(c) => format!("{}", c + 1), None => "NA".into() }).collect();
                        format!(" Factor w/ {} levels {:?}: {}", f.levels.len(), f.levels.iter().take(4).map(|s| s.to_string()).collect::<Vec<_>>(), vals.join(" "))
                    }
                    _ => format!(" {}", c.type_name()),
                };
                println!(" $ {:15}:{}", n, preview);
            }
        }
        RVal::Numeric(v, _) => { let vals: Vec<String> = v.iter().take(10).map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect(); println!(" num [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::Integer(v, _) => { let vals: Vec<String> = v.iter().take(10).map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(); println!(" int [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::Character(v, _) => { let vals: Vec<String> = v.iter().take(5).map(|x| match x { Some(s) => format!("\"{}\"", s), None => "NA".into() }).collect(); println!(" chr [1:{}] {}", v.len(), vals.join(" ")); }
        RVal::List(items) => { println!("List of {}", items.len()); for (i, (n, v)) in items.iter().enumerate().take(10) { let label = n.as_ref().map(|s| format!("${}", s)).unwrap_or(format!("[[{}]]", i+1)); println!(" {} : {} [1:{}]", label, v.type_name(), rval_length(v)); } }
        _ => println!(" {} [1:{}]", v.type_name(), rval_length(&v)),
    }
    Ok(RVal::Null)
}
fn bi_summary(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = gv(a,0);
    // Phase R.2 step 5: data-shaped paths (DataFrame, Numeric) handled by
    // r2-data::summary. Returns Some(()) if handled; falls through here
    // for TypeInstance (model summaries) and other inputs.
    if r2_data::summary::try_summary(&v).is_some() {
        return Ok(RVal::Null);
    }
    match &v {
        RVal::DataFrame(df) => {
            // [DEAD: handled by r2-data::summary::try_summary above. Kept
            // for #[allow(unreachable_code)] body-balance.]
            let mut headers: Vec<String> = Vec::new();
            // Pre-extracted per-column work item.
            enum ColData {
                Numeric(Vec<f64>),                      // pre-filtered, NA-stripped
                Char(Vec<Option<Arc<str>>>),            // raw values for counting
                AllNA,
                Other(&'static str),                    // type name to display
            }
            let mut prepped: Vec<ColData> = Vec::with_capacity(df.columns.len());
            for (name, col) in &df.columns {
                headers.push(format!("{:^18}", name));
                let item = match col {
                    RVal::Numeric(_, _) | RVal::Integer(_, _) => {
                        let n: Vec<f64> = e.as_reals(col).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        if n.is_empty() { ColData::AllNA } else { ColData::Numeric(n) }
                    }
                    RVal::Character(vals, _) => ColData::Char(vals.clone()),
                    other => ColData::Other(other.type_name()),
                };
                prepped.push(item);
            }

            // Stage 2: parallel per-column compute (no engine borrow needed).
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(prepped.len() * 100), // weight columns; threshold avoids parallelizing tiny frames
            );
            let compute_one = |item: &ColData| -> Vec<String> {
                let fs = |v: f64| -> String {
                    if (v - v.round()).abs() < 1e-10 { format!("{}", v as i64) }
                    else { let s = format!("{:.4}", v); s.trim_end_matches('0').trim_end_matches('.').to_string() }
                };
                match item {
                    ColData::Numeric(data) => {
                        let mut n = data.clone();
                        n.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let len = n.len();
                        let mean = n.iter().sum::<f64>() / len as f64;
                        let median = if len % 2 == 0 { (n[len/2-1] + n[len/2]) / 2.0 } else { n[len/2] };
                        vec![
                            format!(" Min.   :{:>8}", fs(n[0])),
                            format!(" 1st Qu.:{:>8}", fs(n[len/4])),
                            format!(" Median :{:>8}", fs(median)),
                            format!(" Mean   :{:>8}", fs(mean)),
                            format!(" 3rd Qu.:{:>8}", fs(n[3*len/4])),
                            format!(" Max.   :{:>8}", fs(n[len-1])),
                        ]
                    }
                    ColData::Char(vals) => {
                        let mut counts: Vec<(String, usize)> = Vec::new();
                        for x in vals {
                            if let Some(s) = x {
                                if let Some(entry) = counts.iter_mut().find(|(k, _)| k == s.as_ref()) { entry.1 += 1; }
                                else { counts.push((s.to_string(), 1)); }
                            }
                        }
                        counts.sort_by(|a, b| b.1.cmp(&a.1));
                        let mut lines: Vec<String> = counts.iter().take(6).map(|(k, v)| format!(" {}:{}", k, v)).collect();
                        while lines.len() < 6 { lines.push(String::new()); }
                        lines
                    }
                    ColData::AllNA => vec!["all NA".into(); 6],
                    ColData::Other(t) => vec![format!(" {}", t); 6],
                }
            };
            let col_summaries: Vec<Vec<String>> = if go_par {
                prepped.par_iter().map(|item| compute_one(item)).collect()
            } else {
                prepped.iter().map(|item| compute_one(item)).collect()
            };

            // Print columns side by side
            for h in &headers { print!("{}", h); }
            println!();
            for row in 0..6 {
                for (ci, _) in headers.iter().enumerate() {
                    let s = col_summaries.get(ci).and_then(|c| c.get(row)).map(|s| s.as_str()).unwrap_or("");
                    print!("{:<18}", s);
                }
                println!();
            }
            Ok(RVal::Null)
        }
        RVal::Numeric(v,_) => {
            let mut n: Vec<f64> = v.iter().filter_map(|x| *x).collect();
            if n.is_empty() { println!("No data"); return Ok(RVal::Null); }
            n.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let len = n.len();
            let mean = n.iter().sum::<f64>() / len as f64;
            let median = if len % 2 == 0 { (n[len/2-1] + n[len/2]) / 2.0 } else { n[len/2] };
            println!("   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.");
            println!("{:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
                fmt_num(n[0]), fmt_num(n[len/4]), fmt_num(median),
                fmt_num(mean), fmt_num(n[3*len/4]), fmt_num(n[len-1]));
            Ok(RVal::Null)
        }
        RVal::TypeInstance(inst) => {
            // Phase R.S.3 — `summary(lmer)` dispatches to the verbose
            // R-style formatter (scaled residuals + variance/Std.Dev. columns
            // + t-values + p-values + correlation matrix).
            if inst.type_name.as_ref() == "lmer" {
                r2_stats::mixed::format_lmer_summary(inst)?;
                return Ok(RVal::Null);
            }
            match inst.type_name.as_ref() {
                "lm" | "glm" => {
                    // Show the captured original call (`lm(y ~ x, data = df)`)
                    // when available; fall back to the generic placeholder
                    // for old-style positional calls without NSE capture.
                    let call = inst.fields.get("call")
                        .map(|v| val_to_str(v))
                        .unwrap_or_else(|| format!("{}(formula)", inst.type_name));
                    println!("\nCall:\n{}", call);
                    // Residuals summary
                    if let Some(res) = inst.fields.get("residuals") {
                        let r: Vec<f64> = e.as_reals(res).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        if !r.is_empty() {
                            let mut sorted = r.clone();
                            sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
                            let n = sorted.len();
                            println!("\nResiduals:");
                            println!("      Min        1Q    Median        3Q       Max");
                            println!("{:>9} {:>9} {:>9} {:>9} {:>9}",
                                fmt_num(sorted[0]), fmt_num(sorted[n/4]),
                                fmt_num(sorted[n/2]), fmt_num(sorted[3*n/4]),
                                fmt_num(sorted[n-1]));
                        }
                    }
                    // Coefficient table with Std.Error, t value, Pr(>|t|)
                    let coefs_val = inst.fields.get("coefficients");
                    let se_val = inst.fields.get("std.errors");
                    let is_glm = inst.type_name.as_ref() == "glm";
                    // glm stores z.values; lm stores t.values. Both use p.values.
                    let stat_val = if is_glm {
                        inst.fields.get("z.values").or_else(|| inst.fields.get("t.values"))
                    } else {
                        inst.fields.get("t.values")
                    };
                    let pv_val = inst.fields.get("p.values");
                    if let Some(cv) = coefs_val {
                        let coeffs: Vec<f64> = e.as_reals(cv).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let se: Vec<f64> = se_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let stat: Vec<f64> = stat_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let pv: Vec<f64> = pv_val.and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                        let names: Vec<String> = match cv {
                            RVal::Numeric(_, at) => at.names.as_ref().map(|n| n.iter().map(|s| s.to_string()).collect()).unwrap_or_else(|| (0..coeffs.len()).map(|i| format!("X{}", i)).collect()),
                            _ => (0..coeffs.len()).map(|i| format!("X{}", i)).collect(),
                        };
                        let (stat_label, pval_label) = if is_glm {
                            ("z value", "Pr(>|z|)")
                        } else {
                            ("t value", "Pr(>|t|)")
                        };
                        println!("\nCoefficients:");
                        println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "", "Estimate", "Std. Error", stat_label, pval_label);
                        for i in 0..coeffs.len() {
                            let s = se.get(i).copied().unwrap_or(0.0);
                            let t = stat.get(i).copied().unwrap_or(0.0);
                            let p = pv.get(i).copied().unwrap_or(1.0);
                            let stars = signif_stars(p);
                            let p_str = fmt_pval(p);
                            println!("{:<15} {:>12} {:>12} {:>10} {:>10} {}",
                                names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                                fmt_num(coeffs[i]), fmt_num(s), fmt_num(t), p_str, stars);
                        }
                        println!("---");
                        println!("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1");
                    }
                    println!();
                    // Residual standard error / R² / F-statistic are LM-specific
                    // (gaussian linear model with closed-form OLS). For GLM the
                    // analogous diagnostics are residual deviance + AIC, printed
                    // in the glm-specific block below.
                    if !is_glm {
                        if let Some(sig) = inst.fields.get("sigma") {
                            let sv = e.scalar_f64(sig).ok().flatten().unwrap_or(0.0);
                            print!("Residual standard error: {}", fmt_num(sv));
                            if let Some(df) = inst.fields.get("df") {
                                let dv = e.scalar_f64(df).ok().flatten().unwrap_or(0.0);
                                print!(" on {} degrees of freedom", dv as i32);
                            }
                            println!();
                        }
                        if let Some(r2) = inst.fields.get("r.squared") {
                            let rv = e.scalar_f64(r2).ok().flatten().unwrap_or(0.0);
                            print!("Multiple R-squared:  {},", fmt_num(rv));
                        }
                        if let Some(ar2) = inst.fields.get("adj.r.squared") {
                            let av = e.scalar_f64(ar2).ok().flatten().unwrap_or(0.0);
                            println!("  Adjusted R-squared:  {}", fmt_num(av));
                        }
                    }
                    if !is_glm { if let Some(fs) = inst.fields.get("f.statistic") {
                        let fv = e.scalar_f64(fs).ok().flatten().unwrap_or(0.0);
                        if let Some(df) = inst.fields.get("df") {
                            let dv = e.scalar_f64(df).ok().flatten().unwrap_or(0.0) as i32;
                            let coefs: Vec<f64> = inst.fields.get("coefficients").and_then(|v| e.as_reals(v).ok()).unwrap_or_default().into_iter().filter_map(|x| x).collect();
                            let p_1 = coefs.len().saturating_sub(1);
                            println!("F-statistic: {} on {} and {} DF", fmt_num(fv), p_1, dv);
                        } else {
                            println!("F-statistic: {}", fmt_num(fv));
                        }
                    } }
                    // GLM-specific diagnostics: Null/Residual deviance + AIC + Fisher iterations.
                    if is_glm {
                        if let Some(d) = inst.fields.get("dispersion") {
                            let dv = e.scalar_f64(d).ok().flatten().unwrap_or(1.0);
                            let fam = inst.fields.get("family").map(|v| val_to_str(v)).unwrap_or_default();
                            println!();
                            println!("(Dispersion parameter for {} family taken to be {})", fam, fmt_num(dv));
                        }
                        if let (Some(nd), Some(dfn)) = (inst.fields.get("null.deviance"), inst.fields.get("df.null")) {
                            let ndv = e.scalar_f64(nd).ok().flatten().unwrap_or(0.0);
                            let dfn = e.scalar_f64(dfn).ok().flatten().unwrap_or(0.0) as i32;
                            println!();
                            println!("    Null deviance: {} on {} degrees of freedom", fmt_num(ndv), dfn);
                        }
                        if let (Some(rd), Some(dfr)) = (inst.fields.get("deviance"), inst.fields.get("df.residual")) {
                            let rdv = e.scalar_f64(rd).ok().flatten().unwrap_or(0.0);
                            let dfr = e.scalar_f64(dfr).ok().flatten().unwrap_or(0.0) as i32;
                            println!("Residual deviance: {} on {} degrees of freedom", fmt_num(rdv), dfr);
                        }
                        if let Some(aic) = inst.fields.get("aic") {
                            let av = e.scalar_f64(aic).ok().flatten().unwrap_or(0.0);
                            println!("AIC: {}", fmt_num(av));
                        }
                        if let Some(it) = inst.fields.get("iter") {
                            let iv = e.scalar_f64(it).ok().flatten().unwrap_or(0.0) as i32;
                            println!();
                            println!("Number of Fisher Scoring iterations: {}", iv);
                        }
                    }
                }
                "rpart" => {
                    println!("\nDecision Tree Summary:");
                    if let Some(tp) = inst.fields.get("type") { println!("Type: {}", tp); }
                    if let Some(md) = inst.fields.get("max_depth") { println!("Max depth: {}", md); }
                    if let Some(pred) = inst.fields.get("predictions") { println!("Training samples: {}", rval_length(pred)); }
                }
                "rf" => {
                    println!("\nRandom Forest Summary:");
                    if let Some(nt) = inst.fields.get("ntrees") { println!("Number of trees: {}", nt); }
                    if let Some(tp) = inst.fields.get("type") { println!("Type: {}", tp); }
                    if let Some(pred) = inst.fields.get("predictions") { println!("Training samples: {}", rval_length(pred)); }
                }
                "gbm" => {
                    println!("\nGradient Boosted Trees Summary:");
                    if let Some(nt) = inst.fields.get("ntrees") { println!("Number of trees: {}", nt); }
                    if let Some(lr) = inst.fields.get("learning_rate") { println!("Learning rate: {}", lr); }
                    if let Some(loss) = inst.fields.get("loss") { println!("Loss function: {}", loss); }
                    if let Some(tl) = inst.fields.get("train.loss") {
                        let losses = e.as_reals(tl).unwrap_or_default();
                        if let Some(last) = losses.last().and_then(|x| *x) { println!("Final training loss: {}", fmt_num(last)); }
                    }
                    if let Some(imp) = inst.fields.get("importance") {
                        println!("Feature importance:");
                        let vals = e.as_reals(imp).unwrap_or_default();
                        let names: Vec<String> = inst.fields.get("xnames")
                            .and_then(|v| if let RVal::Character(cs, _) = v { Some(cs.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()) } else { None })
                            .unwrap_or_else(|| (0..vals.len()).map(|i| format!("X{}", i + 1)).collect());
                        let mut indexed: Vec<(usize, f64)> = vals.iter().enumerate().filter_map(|(i, x)| x.map(|v| (i, v * 100.0))).collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        for (i, pct) in indexed.iter().take(10) {
                            if *pct > 0.0 {
                                let label = names.get(*i).map(|s| s.as_str()).unwrap_or("?");
                                println!("  {}: {}%", label, fmt_num(*pct));
                            }
                        }
                    }
                }
                "kmeans" => {
                    println!("\nK-means Clustering Summary:");
                    if let Some(sz) = inst.fields.get("size") { println!("Cluster sizes: {}", sz); }
                    if let Some(tw) = inst.fields.get("tot.withinss") { println!("Total within-SS: {}", tw); }
                    if let Some(bs) = inst.fields.get("betweenss") { println!("Between-SS: {}", bs); }
                    if let Some(ts) = inst.fields.get("totss") {
                        if let Some(bs) = inst.fields.get("betweenss") {
                            let tot = e.scalar_f64(ts).ok().flatten().unwrap_or(1.0);
                            let bet = e.scalar_f64(bs).ok().flatten().unwrap_or(0.0);
                            println!("Between/Total: {}%", fmt_num(bet / tot * 100.0));
                        }
                    }
                }
                "prcomp" => {
                    println!("\nPCA Summary:");
                    if let Some(sd) = inst.fields.get("sdev") { println!("Standard deviations: {}", sd); }
                    if let Some(pv) = inst.fields.get("prop.variance") { println!("Proportion of variance: {}", pv); }
                }
                "cv" => {
                    println!("\nCross-Validation Summary:");
                    if let Some(k) = inst.fields.get("k") { println!("Folds: {}", k); }
                    if let Some(mm) = inst.fields.get("mean.mse") { println!("Mean MSE: {}", mm); }
                    if let Some(sd) = inst.fields.get("sd.mse") { println!("SD MSE: {}", sd); }
                }
                "confusion" => {
                    println!("\nConfusion Matrix Summary:");
                    if let Some(acc) = inst.fields.get("accuracy") { println!("Accuracy: {}", acc); }
                }
                "aov" | "anova" => {
                    // Already printed by aov()/anova() — just suppress field dump
                    let fv = inst.fields.get("f.statistic").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(0.0);
                    let pv = inst.fields.get("p.value").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(1.0);
                    println!("\nANOVA: F = {}, p-value = {}", fmt_num(fv), fmt_pval(pv));
                }
                "cor.test" | "shapiro.test" | "wilcox.test" | "fisher.test" | "htest" => {
                    // Already printed by test function — show key result
                    if let Some(pv) = inst.fields.get("p.value") {
                        let p = e.scalar_f64(pv).ok().flatten().unwrap_or(1.0);
                        println!("p-value: {}", fmt_pval(p));
                    }
                    if let Some(est) = inst.fields.get("estimate") {
                        let ev = e.scalar_f64(est).ok().flatten().unwrap_or(0.0);
                        println!("estimate: {}", fmt_num(ev));
                    }
                }
                _ => {
                    println!("\n<{}>", inst.type_name);
                    for (k, v) in &inst.fields {
                        if !k.starts_with('_') { println!("  ${}: {}", k, v); }
                    }
                }
            }
            Ok(RVal::Null)
        }
        _ => { println!("{}", v); Ok(RVal::Null) }
    }
}
fn bi_search(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { for p in e.registry.search_path() { println!("{}", p); } Ok(RVal::Null) }
// (bi_cor moved to src/builtins/stats.rs.)

// cov(x, y) — sample covariance with Bessel correction:
//   cov = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n - 1)
// Drops NA pairs (matches R's `use = "complete.obs"` default style for now).
// Oracle decides serial vs parallel for the inner reductions.
// (bi_cov moved to src/builtins/stats.rs.)

// ═══════════════════════════════════════════════════════════════════════
// read.csv — parse CSV file into DataFrame
// ═══════════════════════════════════════════════════════════════════════

// (bi_write_csv moved to src/builtins/io.rs.)

// ═══════════════════════════════════════════════════════════════════════
// lm() — linear regression using normal equations: β = (X^T X)^-1 X^T y
// ═══════════════════════════════════════════════════════════════════════

// (bi_lm moved to src/builtins/stats.rs.)

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

// (bi_toupper/tolower/substr/grep/gsub/strsplit moved to src/builtins/strings.rs.)

// ═══════════════════════════════════════════════════════════════════════
// table() — frequency counts
// ═══════════════════════════════════════════════════════════════════════

// (bi_table moved to src/builtins/data.rs.)

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
fn pure_apply(name: &str, arg: &RVal) -> Option<Result<RVal, R2Err>> {
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

// (bi_rnorm moved to src/builtins/stats.rs.)

// (bi_dnorm moved to src/builtins/stats.rs.)

// (bi_runif moved to src/builtins/stats.rs.)

// (bi_sample moved to src/builtins/stats.rs.)

// ═══════════════════════════════════════════════════════════════════════
// hist() — text histogram (+ SVG)
// ═══════════════════════════════════════════════════════════════════════

// (bi_hist moved to src/builtins/graphics.rs.)


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

fn bi_library(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        // library(stats) without quotes — symbol
        _ => return err!(Runtime, "library() needs a package name (character string)"),
    };

    // 1. Check if already loaded and attached
    let already = e.registry.layers.iter().any(|l| l.name == name);
    if already {
        println!("package '{}' is already loaded", name);
        return Ok(RVal::Null);
    }

    // 2. Try to re-attach a known base package (compiled into binary)
    let base_result = try_reload_base(e, &name);
    if base_result {
        println!("Loading package: '{}'", name);
        // Print masking warnings
        for w in e.drain_warnings() { println!("{}", w); }
        return Ok(RVal::Null);
    }

    // 3. Try to load from disk (addon package)
    let loaded = try_load_from_disk(e, &name)?;
    if loaded {
        println!("Loading package: '{}'", name);
        for w in e.drain_warnings() { println!("{}", w); }
        return Ok(RVal::Null);
    }

    err!(Runtime, "there is no package called '{}'", name)
}

fn bi_require(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    match bi_library(e, a, env) {
        Ok(_) => Ok(rbool(true)),
        Err(e) => {
            println!("Warning: {}", e.msg);
            Ok(rbool(false))
        }
    }
}

fn bi_detach(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "detach() needs a package name"),
    };

    // Strip "package:" prefix if present (R compatibility)
    let name = name.strip_prefix("package:").unwrap_or(&name).to_string();

    match e.detach_package(&name) {
        Ok(restored) => {
            println!("Detached package: '{}'", name);
            if !restored.is_empty() {
                println!("Restored functions: {}", restored.join(", "));
            }
            Ok(RVal::Null)
        }
        Err(msg) => err!(Runtime, "{}", msg),
    }
}

fn bi_installed_packages(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Show base packages (always available)
    println!("{:<20} {:<10} {}", "Package", "Version", "Tier");
    

    // Built-in base packages
    let base_pkgs = vec![
        ("base", "0.1.0", "base"),
        ("stats", "0.1.0", "base"),
        ("graphics", "0.1.0", "base"),
        ("utils", "0.1.0", "base"),
        ("datasets", "0.1.0", "base"),
    ];
    for (name, ver, tier) in &base_pkgs {
        let status = if e.registry.layers.iter().any(|l| l.name == *name) { "loaded" } else { "available" };
        println!("{:<20} {:<10} {} [{}]", name, ver, tier, status);
    }

    // Installed addons from disk
    for (name, info) in &e.installed {
        let status = if e.registry.layers.iter().any(|l| l.name == *name) { "loaded" } else { "installed" };
        println!("{:<20} {:<10} addon [{}]", name, info.version, status);
    }

    // Scan disk for packages not yet discovered
    for lib_path in &e.lib_paths.clone() {
        let path = std::path::Path::new(lib_path);
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let pkg_name = entry.file_name().to_string_lossy().to_string();
                    if !e.installed.contains_key(&pkg_name)
                        && entry.path().join("MANIFEST.toml").exists()
                    {
                        println!("{:<20} {:<10} addon [installed]", pkg_name, "?");
                    }
                }
            }
        }
    }

    Ok(RVal::Null)
}

fn bi_lib_paths(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if a.is_empty() {
        // Get: show current paths
        let paths: Vec<Character> = e.lib_paths.iter().map(|p| Some(Arc::from(p.as_str()))).collect();
        Ok(RVal::Character(paths, Attrs::default()))
    } else {
        // Set: update paths
        match &gv(a, 0) {
            RVal::Character(v, _) => {
                e.lib_paths = v.iter().filter_map(|x| x.as_ref().map(|s| s.to_string())).collect();
                println!("Library paths updated");
                Ok(RVal::Null)
            }
            _ => err!(Runtime, ".libPaths() needs character vector"),
        }
    }
}

// ── Helper: re-attach a base package that was detached ───────────────

fn try_reload_base(e: &mut Engine, name: &str) -> bool {
    match name {
        "base" => {
            e.registry.add_layer(mkpkg("base", PackageTier::Base, vec![
                ("seq",bi_seq),("rep",bi_rep),("paste",bi_paste),("paste0",bi_paste0),
                ("which",bi_which),("sort",bi_sort),("rev",bi_rev),("unique",bi_unique),
                ("abs",bi_abs),("sqrt",bi_sqrt),("round",bi_round),("max",bi_max),("min",bi_min),
                ("nchar",bi_nchar),("toupper",bi_toupper),("tolower",bi_tolower),
                ("substr",bi_substr),("grep",bi_grep),("gsub",bi_gsub),("strsplit",bi_strsplit),
                ("sub",bi_sub),("grepl",bi_grepl),("regexpr",bi_regexpr),
                ("duplicated",bi_duplicated),("order",bi_order),("rank",bi_rank),
                ("cummax",bi_cummax),("cummin",bi_cummin),
                ("filter",bi_filter),("select",bi_select),("arrange",bi_arrange),("mutate",bi_mutate),
                ("factor",bi_factor),("names",bi_names),("nrow",bi_nrow),("ncol",bi_ncol),
                ("table",bi_table),("sapply",bi_sapply),("lapply",bi_lapply),("mapply",bi_mapply),("vapply",bi_vapply),
                ("rbind",bi_rbind),("cbind",bi_cbind),("merge",bi_merge),
                ("na.omit",bi_na_omit),("complete.cases",bi_complete_cases),
                ("is.null",bi_is_null),("ifelse",bi_ifelse),
                ("apply",bi_apply),("tapply",bi_tapply),("aggregate",bi_aggregate),
                ("do.call",bi_do_call),
                ("log",bi_log),("exp",bi_exp),("ceiling",bi_ceiling),("floor",bi_floor),
                ("cumsum",bi_cumsum),("cumprod",bi_cumprod),("cummax",bi_cummax),("cummin",bi_cummin),("diff",bi_diff),
                ("rollsum",bi_rollsum),("rollmean",bi_rollmean),("rollmax",bi_rollmax),("rollmin",bi_rollmin),("rollsd",bi_rollsd),
                ("which.min",bi_which_min),("which.max",bi_which_max),("range",bi_range),
                ("prod",bi_prod),("any",bi_any),("all",bi_all),
                ("trimws",bi_trimws),("startsWith",bi_starts_with),("endsWith",bi_ends_with),
                ("sprintf",bi_sprintf),("stop",bi_stop),("warning",bi_warning),("message",bi_message),
                ("ls",bi_ls),("rm",bi_rm),("exists",bi_exists),
                ("levels",bi_levels),("nlevels",bi_nlevels),
                ("dim",bi_dim),("colnames",bi_colnames),("rownames",bi_rownames),
                ("data",bi_data),
                ("rowSums",bi_rowSums),("colSums",bi_colSums),("rowMeans",bi_rowMeans),("colMeans",bi_colMeans),
                ("set.seed",bi_set_seed),("Sys.sleep",bi_Sys_sleep),("readline",bi_readline),
                ("as.Date",bi_as_date),("as.POSIXct",bi_as_posixct),("format.Date",bi_format_time),
                ("format.POSIXct",bi_format_time),("Sys.Date",bi_sys_date),("Sys.time",bi_sys_time),
                ("difftime",bi_difftime),
                ("ts",bi_ts),("tsp",bi_tsp),("start",bi_ts_start),("end",bi_ts_end),
                ("frequency",bi_frequency),("deltat",bi_deltat),("time",bi_time_idx),
                ("cycle",bi_cycle),("window",bi_window),("is.ts",bi_is_ts),
                ("xts",bi_xts),("index",bi_index),("coredata",bi_coredata),("is.xts",bi_is_xts),
                ("xts.subset",bi_xts_subset),("first",bi_first),("last",bi_last),
                ("na.locf",bi_na_locf),("merge.xts",bi_merge_xts),
                ("acf",bi_acf),("pacf",bi_pacf),("decompose",bi_decompose),
                ("is.regular",bi_is_regular),("periodicity",bi_periodicity),
                ("lag",bi_lag),("diff_ts",bi_diff_ts),
                ("to.daily",bi_to_daily),("to.weekly",bi_to_weekly),
                ("to.monthly",bi_to_monthly),("to.quarterly",bi_to_quarterly),
                ("to.yearly",bi_to_yearly),
                ("apply.daily",bi_apply_daily),("apply.weekly",bi_apply_weekly),
                ("apply.monthly",bi_apply_monthly),("apply.quarterly",bi_apply_quarterly),
                ("apply.yearly",bi_apply_yearly),
                ("tithi",bi_tithi),("hindu.date",bi_hindu_date),("hnc.date",bi_hnc_date),
            ]));
            true
        }
        "stats" => {
            e.registry.add_layer(mkpkg("stats", PackageTier::Base, vec![
                ("sum",bi_sum),("mean",bi_mean),("sd",bi_sd),("var",bi_var),("cor",bi_cor),("cov",bi_cov),
                ("lm",bi_lm),("summary",bi_summary),
                ("rnorm",bi_rnorm),("dnorm",bi_dnorm),("runif",bi_runif),("sample",bi_sample),
                ("pnorm",bi_pnorm),("qnorm",bi_qnorm),("rbinom",bi_rbinom),("rpois",bi_rpois),
                ("median",bi_median),("quantile",bi_quantile),
                ("t.test",bi_t_test),("chisq.test",bi_chisq_test),("hotelling.test",bi_hotelling_test),("manova",bi_manova),("lmer",bi_lmer),
                ("predict",bi_predict),("residuals",bi_residuals),("fitted",bi_fitted),("coef",bi_coef),
                ("glm",bi_glm),("confint",bi_confint),("binomial",bi_binomial),("gaussian",bi_gaussian),("poisson",bi_poisson),("subset",bi_subset),("transform",bi_transform),
                ("svd",bi_svd),("eigen",bi_eigen),("prcomp",bi_prcomp),
                ("kmeans",bi_kmeans),("knn",bi_knn),("naive.bayes",bi_naive_bayes),("scale",bi_scale),
                ("rpart",bi_rpart),("rf",bi_rf),("gbm",bi_gbm),("cv",bi_cv),("aov",bi_aov),("anova",bi_anova),("cor.test",bi_cor_test),("shapiro.test",bi_shapiro_test),("wilcox.test",bi_wilcox_test),("fisher.test",bi_fisher_test),("weighted.mean",bi_weighted_mean),("IQR",bi_iqr),("confusion.matrix",bi_confusion_matrix),
            ]));
            true
        }
        "graphics" => {
            e.registry.add_layer(mkpkg("graphics", PackageTier::Base, vec![
                ("plot",bi_plot),("hist",bi_hist),("boxplot",bi_boxplot),("barplot",bi_barplot),
            ("save.plot",bi_save_plot),
                ("lines",bi_lines),("points",bi_points),("abline",bi_abline),("legend",bi_legend),
                ("par",bi_par),("dev.off",bi_dev_off),("save_plot",bi_save_plot),("dev.view",bi_dev_view),
                ("dev.new",bi_dev_new),("dev.set",bi_dev_set),("dev.list",bi_dev_list),
                ("dev.cur",bi_dev_cur),
                ("rgb",bi_rgb),("gray",bi_gray),("grey",bi_gray),("hsv",bi_hsv),
                ("rainbow",bi_rainbow),("heat.colors",bi_heat_colors),
                ("terrain.colors",bi_terrain_colors),("topo.colors",bi_topo_colors),
                ("cm.colors",bi_cm_colors),("adjustcolor",bi_adjustcolor),
            ]));
            true
        }
        "utils" => {
            e.registry.add_layer(mkpkg("utils", PackageTier::Base, vec![
                ("head",bi_head),("tail",bi_tail),("str",bi_str),
                ("read.csv",bi_read_csv_v2),("write.csv",bi_write_csv),
                ("search",bi_search),("t",bi_transpose),("crossprod",bi_crossprod),
                ("source",bi_source),("system.time",bi_system_time),
                ("read.table",bi_read_table),("write.table",bi_write_table),("read.delim",bi_read_delim),
                ("Sys.time",bi_Sys_time),("help",bi_help),("getwd",bi_getwd),("setwd",bi_setwd),
                ("file.exists",bi_file_exists),("list.files",bi_list_files),("Sys.getenv",bi_sys_getenv),("save",bi_save),("load",bi_load),("version",bi_version),("clear",bi_clear),("cls",bi_clear),(".Internal",bi_internal),
            ]));
            true
        }
        _ => false,
    }
}

// ── Helper: load addon package from disk ──────────────────────────────
//
// Package directory structure on disk:
//   ~/.r2/library/mypkg/
//   ├── MANIFEST.toml    # name, version, exports, depends
//   ├── R2/
//   │   ├── functions.r  # R2 source code defining functions
//   │   └── *.r
//   └── data/            # optional datasets
//
// This reads the .r files, parses them, evaluates them to extract
// function definitions, and registers them as a package layer.

fn try_load_from_disk(e: &mut Engine, name: &str) -> Result<bool, R2Err> {
    // Path A — r2-pkg standard layout: ~/.r2/packages/<name>/R/*.r2
    if let Ok(pkg_root) = r2_pkg::pkg_dir(name) {
        if pkg_root.is_dir() && pkg_root.join("package.r2").exists() {
            return load_r2pkg_layout(e, name, &pkg_root);
        }
    }
    // Path B — legacy layout: <lib_path>/<name>/R2/*.r
    for lib_path in &e.lib_paths.clone() {
        let pkg_dir = std::path::Path::new(lib_path).join(name);
        let r2_dir = pkg_dir.join("R2");
        if !r2_dir.is_dir() { continue; }

        // Read all .r files
        let mut all_source = String::new();
        if let Ok(entries) = std::fs::read_dir(&r2_dir) {
            let mut files: Vec<_> = entries.flatten()
                .filter(|e| e.path().extension().map(|ext| ext == "r").unwrap_or(false))
                .collect();
            files.sort_by_key(|e| e.file_name());
            for entry in files {
                match std::fs::read_to_string(entry.path()) {
                    Ok(content) => { all_source.push_str(&content); all_source.push('\n'); }
                    Err(err) => return err!(Runtime, "cannot read {}: {}", entry.path().display(), err),
                }
            }
        }
        if all_source.is_empty() { return err!(Runtime, "package '{}' has no R2 source files", name); }

        // Parse
        let stmts = r2_parser::Parser::parse(&all_source)
            .map_err(|pe| R2Err { msg: format!("error parsing package '{}': {}", name, pe), kind: ErrKind::Runtime })?;

        // Snapshot: record existing global names BEFORE eval
        let before: Vec<Arc<str>> = e.global_env.bindings.keys().cloned().collect();

        // Evaluate all statements — assignments go directly into global_env
        let env = e.global_env.clone();
        for stmt in &stmts {
            match e.eval_in(stmt, &env) {
                Ok(_) => {}
                Err(err) => {
                    if err.kind != ErrKind::CtrlBreak && err.kind != ErrKind::CtrlNext {
                        eprintln!("Warning in package '{}': {}", name, err.msg);
                    }
                }
            }
        }

        // Diff: find NEW bindings that are closures
        let mut exports = Vec::new();
        for (fname, fval) in &e.global_env.bindings {
            if !before.contains(fname) && matches!(fval, RVal::Closure(_)) {
                if e.registry.is_core(fname) {
                    return err!(Runtime, "package '{}' cannot mask core function '{}'", name, fname);
                }
                exports.push(fname.to_string());
            }
        }

        if exports.is_empty() {
            return err!(Runtime, "package '{}' defines no functions", name);
        }

        // Register layer for search/detach tracking
        let layer = PackageLayer {
            name: name.to_string(),
            tier: PackageTier::Addon,
            functions: HashMap::new(),
            exports: exports.clone(),
        };
        let masks = e.registry.check_masks(&exports);
        for (func, from) in &masks {
            e.warnings.push(format!("Warning: package '{}' masks '{}' from '{}'", name, func, from));
        }
        e.registry.add_layer(layer);

        e.installed.insert(name.to_string(), InstalledPkgInfo {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            path: pkg_dir.to_string_lossy().to_string(),
            exports,
            depends: Vec::new(),
        });

        return Ok(true);
    }
    Ok(false)
}

// Load a package laid out per the r2-pkg convention:
//   <pkg_root>/package.r2      manifest (required)
//   <pkg_root>/R/*.r2          source files, sourced alphabetically
fn load_r2pkg_layout(e: &mut Engine, name: &str, pkg_root: &std::path::Path) -> Result<bool, R2Err> {
    let manifest = r2_pkg::read_manifest(pkg_root)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let files = r2_pkg::package_source_files(name)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;

    let mut all_source = String::new();
    for f in &files {
        match std::fs::read_to_string(f) {
            Ok(c) => { all_source.push_str(&c); all_source.push('\n'); }
            Err(err) => return err!(Runtime, "cannot read {}: {}", f.display(), err),
        }
    }
    // Empty R/ dir is allowed for metadata-only packages, but library() of one
    // would be a no-op — treat as an error so the user knows nothing happened.
    if all_source.trim().is_empty() {
        return err!(Runtime, "package '{}' has no .r2 source files under R/", name);
    }

    let stmts = r2_parser::Parser::parse(&all_source)
        .map_err(|pe| R2Err { msg: format!("error parsing package '{}': {}", name, pe), kind: ErrKind::Runtime })?;

    let before: Vec<Arc<str>> = e.global_env.bindings.keys().cloned().collect();
    let env = e.global_env.clone();
    for stmt in &stmts {
        if let Err(err) = e.eval_in(stmt, &env) {
            if err.kind != ErrKind::CtrlBreak && err.kind != ErrKind::CtrlNext {
                eprintln!("Warning in package '{}': {}", name, err.msg);
            }
        }
    }

    // Determine exports: prefer the manifest's package_exports list if non-empty,
    // otherwise fall back to "every new closure" so authors don't have to maintain
    // the list while iterating.
    let mut exports: Vec<String> = Vec::new();
    if !manifest.exports.is_empty() {
        for ex in &manifest.exports {
            let key: Arc<str> = Arc::from(ex.as_str());
            if matches!(e.global_env.bindings.get(&key), Some(RVal::Closure(_))) {
                exports.push(ex.clone());
            } else {
                eprintln!("Warning: package '{}' exports '{}' but it was not defined in any R/ file", name, ex);
            }
        }
    } else {
        for (fname, fval) in &e.global_env.bindings {
            if !before.contains(fname) && matches!(fval, RVal::Closure(_)) {
                exports.push(fname.to_string());
            }
        }
    }
    if exports.is_empty() {
        return err!(Runtime, "package '{}' defines no exported functions", name);
    }
    for ex in &exports {
        if e.registry.is_core(ex) {
            return err!(Runtime, "package '{}' cannot mask core function '{}'", name, ex);
        }
    }

    let layer = PackageLayer {
        name: name.to_string(),
        tier: PackageTier::Addon,
        functions: HashMap::new(),
        exports: exports.clone(),
    };
    let masks = e.registry.check_masks(&exports);
    for (func, from) in &masks {
        e.warnings.push(format!("Warning: package '{}' masks '{}' from '{}'", name, func, from));
    }
    e.registry.add_layer(layer);
    e.installed.insert(name.to_string(), InstalledPkgInfo {
        name: name.to_string(),
        version: manifest.version.clone(),
        path: pkg_root.to_string_lossy().to_string(),
        exports,
        depends: Vec::new(),
    });
    Ok(true)
}

// ═══════════════════════════════════════════════════════════════════════
// DATA MANIPULATION: rbind, cbind, merge, subset, transform, within
// ═══════════════════════════════════════════════════════════════════════

// Helper: coerce an RVal to a column of f64 (for matrix-style cbind/rbind).
// Returns (data, nrows). Matrix input contributes ncol columns of nrow rows.
fn coerce_to_columns(v: &RVal) -> Result<(Vec<f64>, usize, usize), R2Err> {
    match v {
        RVal::Matrix(m) => Ok((m.data.clone(), m.nrow, m.ncol)),
        RVal::Numeric(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Integer(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|i| i as f64).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        RVal::Logical(vs, _) => {
            let n = vs.len();
            let data: Vec<f64> = vs.iter().map(|x| x.map(|b| if b { 1.0 } else { 0.0 }).unwrap_or(f64::NAN)).collect();
            Ok((data, n, 1))
        }
        _ => err!(Type, "cbind/rbind: cannot coerce {} to numeric matrix", v.type_name()),
    }
}

fn all_dataframes(a: &[EvalArg]) -> bool {
    !a.is_empty() && a.iter().all(|x| matches!(x.value, RVal::DataFrame(_)))
}

// Phase R.2: bi_rbind moved to r2-data::bind. Engine adapter only.
fn bi_rbind(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::bind::bi_rbind(a);
    #[allow(unreachable_code)]
    {
    if a.is_empty() { return err!(Runtime, "rbind: needs at least one argument"); }

    // DataFrame path: all args are data.frames → stack rows
    if all_dataframes(a) {
        let mut iter = a.iter();
        let first = match &iter.next().unwrap().value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
        let ncol = first.ncol();
        let mut columns: Vec<(Arc<str>, RVal)> = first.columns.clone();
        for arg in iter {
            let df = match &arg.value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
            if df.ncol() != ncol { return err!(Runtime, "rbind: column count mismatch ({} vs {})", ncol, df.ncol()); }
            for (i, (name, col2)) in df.columns.iter().enumerate() {
                let (cur_name, cur_col) = columns[i].clone();
                let merged = match (&cur_col, col2) {
                    (RVal::Numeric(v1,_), RVal::Numeric(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Numeric(v.into(), Attrs::default()) }
                    (RVal::Integer(v1,_), RVal::Integer(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Integer(v.into(), Attrs::default()) }
                    (RVal::Character(v1,_), RVal::Character(v2,_)) => { let mut v = v1.clone(); v.extend(v2.clone()); RVal::Character(v, Attrs::default()) }
                    (RVal::Logical(v1,_), RVal::Logical(v2,_)) => { let mut v = v1.as_vec().clone(); v.extend(v2.as_vec()); RVal::Logical(v.into(), Attrs::default()) }
                    _ => return err!(Type, "rbind: incompatible column types at '{}'", name),
                };
                columns[i] = (cur_name, merged);
            }
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path: stack matrices/vectors as rows.
    // A vector v of length k becomes a 1-row, k-column matrix.
    // A matrix contributes its rows as-is.
    let mut blocks: Vec<(Vec<f64>, usize, usize)> = Vec::with_capacity(a.len());
    for arg in a {
        let (data, nrow, ncol) = match &arg.value {
            RVal::Matrix(m) => (m.data.clone(), m.nrow, m.ncol),
            other => {
                let (d, n, _) = coerce_to_columns(other)?;
                // Vector → 1 row, n columns
                (d, 1, n)
            }
        };
        blocks.push((data, nrow, ncol));
    }
    let ncol = blocks[0].2;
    if !blocks.iter().all(|(_, _, c)| *c == ncol) {
        return err!(Runtime, "rbind: column count mismatch across inputs");
    }
    let total_rows: usize = blocks.iter().map(|(_, r, _)| *r).sum();
    // Build column-major output: for each column j, append rows from each block in order.
    let mut data = vec![0.0; total_rows * ncol];
    for j in 0..ncol {
        let mut row_offset = 0;
        for (b_data, b_nrow, _) in &blocks {
            for i in 0..*b_nrow {
                data[j * total_rows + row_offset + i] = b_data[j * b_nrow + i];
            }
            row_offset += b_nrow;
        }
    }
    Ok(RVal::Matrix(Matrix::new(data, total_rows, ncol)))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_cbind moved to r2-data::bind. Engine adapter only.
fn bi_cbind(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::bind::bi_cbind(a);
    #[allow(unreachable_code)]
    {
    if a.is_empty() { return err!(Runtime, "cbind: needs at least one argument"); }

    // DataFrame path: all args are data.frames → side-by-side columns
    if all_dataframes(a) {
        let mut iter = a.iter();
        let first = match &iter.next().unwrap().value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
        let nrow = first.nrow();
        let mut columns: Vec<(Arc<str>, RVal)> = first.columns;
        for arg in iter {
            let df = match &arg.value { RVal::DataFrame(df) => df.clone(), _ => unreachable!() };
            if df.nrow() != nrow { return err!(Runtime, "cbind: row count mismatch ({} vs {})", nrow, df.nrow()); }
            columns.extend(df.columns);
        }
        return Ok(RVal::DataFrame(DataFrame { columns, row_names: None }));
    }

    // Matrix path: collect each input as one or more columns of f64.
    // Matrix → its columns; vector (Numeric/Integer/Logical) → one column.
    // Track preserved column names where available.
    let mut blocks: Vec<(Vec<f64>, usize, usize, Option<Vec<Arc<str>>>)> = Vec::with_capacity(a.len());
    let mut any_names = false;
    for arg in a {
        let (data, nrow, ncol) = coerce_to_columns(&arg.value)?;
        let names: Option<Vec<Arc<str>>> = match &arg.value {
            RVal::Matrix(m) => m.col_names.clone(),
            _ => arg.name.as_ref().map(|n| vec![n.clone()]),
        };
        if names.is_some() { any_names = true; }
        blocks.push((data, nrow, ncol, names));
    }
    let nrow = blocks[0].1;
    if !blocks.iter().all(|(_, r, _, _)| *r == nrow) {
        return err!(Runtime, "cbind: row count mismatch across inputs");
    }
    let total_cols: usize = blocks.iter().map(|(_, _, c, _)| *c).sum();
    let mut data = Vec::with_capacity(nrow * total_cols);
    let mut col_names: Vec<Arc<str>> = Vec::with_capacity(total_cols);
    for (b_data, _, b_ncol, b_names) in &blocks {
        data.extend_from_slice(b_data);
        match b_names {
            Some(ns) if ns.len() == *b_ncol => col_names.extend(ns.iter().cloned()),
            _ => for j in 0..*b_ncol { col_names.push(Arc::from(format!("V{}", col_names.len() + 1).as_str())); }
        }
    }
    let mut m = Matrix::new(data, nrow, total_cols);
    if any_names { m.col_names = Some(col_names); }
    Ok(RVal::Matrix(m))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// (bi_merge moved to src/builtins/data.rs.)

fn to_string_vec(col: &RVal) -> Vec<String> {
    match col {
        RVal::Numeric(v,_) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Integer(v,_) => v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect(),
        RVal::Character(v,_) => v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect(),
        RVal::Logical(v,_) => v.iter().map(|x| match x { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() }).collect(),
        _ => Vec::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// NA HANDLING: na.omit, complete.cases, is.null, ifelse
// ═══════════════════════════════════════════════════════════════════════

// (bi_na_omit moved to src/builtins/data.rs.)

// (bi_complete_cases moved to src/builtins/data.rs.)

fn bi_is_null(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a,0), RVal::Null)))
}

fn bi_ifelse(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // ifelse(test, yes, no) — vectorized conditional
    let test = e.as_logicals(&gv(a,0))?;
    let yes_val = gv(a,1);
    let no_val = gv(a,2);
    let yes = e.as_reals(&yes_val)?;
    let no = e.as_reals(&no_val)?;
    let result: Vec<Real> = test.iter().enumerate().map(|(i, t)| {
        match t {
            Some(true) => yes.get(i % yes.len()).copied().unwrap_or(None),
            Some(false) => no.get(i % no.len()).copied().unwrap_or(None),
            None => None,
        }
    }).collect();
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

// ═══════════════════════════════════════════════════════════════════════
// APPLY FAMILY: apply, tapply, aggregate, do.call
// ═══════════════════════════════════════════════════════════════════════

fn bi_apply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_apply(e, a, env);
    #[allow(unreachable_code)] {
    // [DEAD: original body kept for safe rollback — Phase R.2 step 6]
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Type, "apply needs data.frame or matrix") };
    let margin = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0) as i32;
    let func = gv(a,2);

    // Detect pure-builtin fast path.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    if margin == 2 {
        // Per-column inputs.
        let inputs: Vec<RVal> = df.columns.iter().map(|(_, col)| col.clone()).collect();
        let results: Vec<RVal> = match &pure_name {
            Some(fname) => {
                let go_par = r2_oracle::should_parallelize(
                    r2_oracle::Op::PerElementMap,
                    r2_oracle::Shape::n(inputs.len() * 100),
                );
                if go_par {
                    inputs.par_iter().map(|c| pure_apply(fname, c).unwrap_or(Ok(RVal::Null)))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut r = Vec::with_capacity(inputs.len());
                    for c in &inputs { r.push(pure_apply(fname, c).unwrap_or(Ok(RVal::Null))?); }
                    r
                }
            }
            None => {
                let mut r = Vec::with_capacity(inputs.len());
                for c in &inputs {
                    let args = vec![EvalArg { name: None, value: c.clone() }];
                    r.push(e.call_fn(&func, &args, env)?);
                }
                r
            }
        };
        // Simplify to numeric if every result is a scalar Numeric.
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r {
                RVal::Numeric(v,_) if v.len() == 1 => nums.push(v[0]),
                _ => { all_scalar = false; break; }
            }
        }
        if all_scalar {
            let mut attrs = Attrs::default();
            attrs.names = Some(df.columns.iter().map(|(n,_)| n.clone()).collect());
            Ok(RVal::Numeric(nums.into(), attrs))
        } else {
            Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect()))
        }
    } else {
        // Per-row inputs (margin == 1). Pre-extract every row to a Numeric.
        let nrow = df.nrow();
        let rows: Vec<RVal> = (0..nrow).map(|r| {
            let row: Vec<Real> = df.columns.iter().filter_map(|(_, col)| {
                match col {
                    RVal::Numeric(v,_) => v.get(r).copied(),
                    RVal::Integer(v,_) => v.get(r).map(|x| x.map(|n| n as f64)),
                    _ => None,
                }
            }).collect();
            RVal::Numeric(row.into(), Attrs::default())
        }).collect();
        let results: Vec<RVal> = match &pure_name {
            Some(fname) => {
                let go_par = r2_oracle::should_parallelize(
                    r2_oracle::Op::PerElementMap,
                    r2_oracle::Shape::n(rows.len() * 100),
                );
                if go_par {
                    rows.par_iter().map(|r| pure_apply(fname, r).unwrap_or(Ok(RVal::Null)))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    let mut out = Vec::with_capacity(rows.len());
                    for r in &rows { out.push(pure_apply(fname, r).unwrap_or(Ok(RVal::Null))?); }
                    out
                }
            }
            None => {
                let mut out = Vec::with_capacity(rows.len());
                for r in rows {
                    let args = vec![EvalArg { name: None, value: r }];
                    out.push(e.call_fn(&func, &args, env)?);
                }
                out
            }
        };
        let mut nums = Vec::new();
        let mut all_scalar = true;
        for r in &results {
            match r { RVal::Numeric(v,_) if v.len() == 1 => nums.push(v[0]), _ => { all_scalar = false; break; } }
        }
        if all_scalar { Ok(RVal::Numeric(nums.into(), Attrs::default())) }
        else { Ok(RVal::List(results.into_iter().map(|v| (None, v)).collect())) }
    }
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

fn bi_tapply(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_tapply(e, a, env);
    #[allow(unreachable_code)] {
    let x = e.as_reals(&gv(a,0))?;
    let index = to_string_vec(&gv(a,1));
    let func = gv(a,2);

    // Group values by index
    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in index.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k,_)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    // Phase D: parallel fast path when FUN is a pure builtin.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();

    let computed: Vec<RVal> = match &pure_name {
        Some(fname) => {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(group_inputs.len() * 100),
            );
            if go_par {
                group_inputs.par_iter().map(|input| pure_apply(fname, input).unwrap_or(Ok(RVal::Null)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(group_inputs.len());
                for input in &group_inputs { r.push(pure_apply(fname, input).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        }
        None => {
            let mut r = Vec::with_capacity(group_inputs.len());
            for input in group_inputs {
                let args = vec![EvalArg { name: None, value: input }];
                r.push(e.call_fn(&func, &args, env)?);
            }
            r
        }
    };

    let results: Vec<(Option<Arc<str>>, RVal)> = groups.iter().zip(computed.into_iter())
        .map(|((key, _), result)| (Some(Arc::from(key.as_str())), result))
        .collect();
    Ok(RVal::List(results))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

fn bi_aggregate(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> { return r2_data::apply::bi_aggregate(e, a, env);
    #[allow(unreachable_code)] {
    let x = e.as_reals(&gv(a,0))?;
    let by = to_string_vec(&gn(a, "by").unwrap_or(gv(a, 1)));
    let func = gn(a, "FUN").unwrap_or(gv(a, 2));

    let mut groups: Vec<(String, Vec<Real>)> = Vec::new();
    for (i, key) in by.iter().enumerate() {
        if let Some(grp) = groups.iter_mut().find(|(k,_)| k == key) {
            grp.1.push(x.get(i).copied().unwrap_or(None));
        } else {
            groups.push((key.clone(), vec![x.get(i).copied().unwrap_or(None)]));
        }
    }

    // Phase D: parallel fast path when FUN is a pure builtin.
    let pure_name: Option<String> = if let RVal::BuiltinFn(fname) = &func {
        if pure_apply(fname, &RVal::Numeric(vec![Some(0.0)].into(), Attrs::default())).is_some() {
            Some(fname.to_string())
        } else { None }
    } else { None };

    let group_inputs: Vec<RVal> = groups.iter()
        .map(|(_, vals)| RVal::Numeric(vals.clone().into(), Attrs::default())).collect();
    let computed: Vec<RVal> = match &pure_name {
        Some(fname) => {
            let go_par = r2_oracle::should_parallelize(
                r2_oracle::Op::PerElementMap,
                r2_oracle::Shape::n(group_inputs.len() * 100),
            );
            if go_par {
                group_inputs.par_iter().map(|input| pure_apply(fname, input).unwrap_or(Ok(RVal::Null)))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let mut r = Vec::with_capacity(group_inputs.len());
                for input in &group_inputs { r.push(pure_apply(fname, input).unwrap_or(Ok(RVal::Null))?); }
                r
            }
        }
        None => {
            let mut r = Vec::with_capacity(group_inputs.len());
            for input in group_inputs {
                let args = vec![EvalArg { name: None, value: input }];
                r.push(e.call_fn(&func, &args, env)?);
            }
            r
        }
    };

    let mut group_names: Vec<Character> = Vec::with_capacity(groups.len());
    let mut agg_values: Vec<Real> = Vec::with_capacity(groups.len());
    for ((key, _), result) in groups.iter().zip(computed.into_iter()) {
        group_names.push(Some(Arc::from(key.as_str())));
        if let Ok(v) = e.scalar_f64(&result) { agg_values.push(v); } else { agg_values.push(None); }
    }

    Ok(RVal::DataFrame(DataFrame {
        columns: vec![
            (Arc::from("Group"), RVal::Character(group_names, Attrs::default())),
            (Arc::from("Value"), RVal::Numeric(agg_values.into(), Attrs::default())),
        ],
        row_names: None,
    }))
    } // end #[allow(unreachable_code)] (Phase R.2 step 6)
}

// (bi_do_call moved to src/builtins/data.rs.)

// ═══════════════════════════════════════════════════════════════════════
// MORE MATH: log, exp, ceiling, floor, cumsum, cumprod, diff, range, median, quantile
// ═══════════════════════════════════════════════════════════════════════

// Phase K.2: log dispatches to specialized kernel ops when base matches a
// well-known constant (e, 2, 10) for max efficiency. Other bases route
// through Ln + a scalar-divide step (still kernel-dispatched).
fn bi_log(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    let base = gn(a,"base").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(std::f64::consts::E);
    let result = if (base - std::f64::consts::E).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Ln, &v)
    } else if (base - 2.0).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Log2, &v)
    } else if (base - 10.0).abs() < 1e-12 {
        r2_kernel::map(r2_kernel::MapOp::Log10, &v)
    } else {
        // Arbitrary base: Ln then divide. Two passes; specialized base
        // ops above are the common case.
        let lns = r2_kernel::map(r2_kernel::MapOp::Ln, &v);
        let lb = base.ln();
        lns.into_iter().map(|x| x.map(|n| n / lb)).collect()
    };
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}
fn bi_exp(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Exp, &v).into(), Attrs::default()))
}

// Phase R.M.1 — trig and transcendental builtins. Each is a thin
// wrapper that coerces the first argument to Real and routes through
// the kernel's element-wise dispatcher (so it picks up SIMD/parallel
// variants automatically based on size).
fn bi_sin   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sin, &v).into(), Attrs::default()))
}
fn bi_cos   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Cos, &v).into(), Attrs::default()))
}
fn bi_tan   (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Tan, &v).into(), Attrs::default()))
}
fn bi_asin  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Asin, &v).into(), Attrs::default()))
}
fn bi_acos  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Acos, &v).into(), Attrs::default()))
}
fn bi_atan  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Atan, &v).into(), Attrs::default()))
}
fn bi_atan2 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Two-argument atan2(y, x). Element-wise pairing; recycle shorter side
    // to match R's vectorization rule.
    let y = e.as_reals(&gv(a,0))?;
    let x = e.as_reals(&gv(a,1))?;
    let n = y.len().max(x.len());
    if y.is_empty() || x.is_empty() {
        return Ok(RVal::Numeric(Vec::<Option<f64>>::new().into(), Attrs::default()));
    }
    let out: Vec<Option<f64>> = (0..n).map(|i| {
        let yi = y[i % y.len()]; let xi = x[i % x.len()];
        match (yi, xi) { (Some(a), Some(b)) => Some(a.atan2(b)), _ => None }
    }).collect();
    Ok(RVal::Numeric(out.into(), Attrs::default()))
}
fn bi_sinh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sinh, &v).into(), Attrs::default()))
}
fn bi_cosh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Cosh, &v).into(), Attrs::default()))
}
fn bi_tanh  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Tanh, &v).into(), Attrs::default()))
}
fn bi_sign  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Sign, &v).into(), Attrs::default()))
}
fn bi_trunc (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Trunc, &v).into(), Attrs::default()))
}
fn bi_expm1 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Expm1, &v).into(), Attrs::default()))
}
fn bi_log1p (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log1p, &v).into(), Attrs::default()))
}
fn bi_log2  (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log2, &v).into(), Attrs::default()))
}
fn bi_log10 (e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(r2_kernel::map(r2_kernel::MapOp::Log10, &v).into(), Attrs::default()))
}
fn bi_ceiling(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| n.ceil())).collect(), Attrs::default()))
}
fn bi_floor(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_reals(&gv(a,0))?;
    Ok(RVal::Numeric(v.into_iter().map(|x| x.map(|n| n.floor())).collect(), Attrs::default()))
}
// (bi_cumsum / cumprod / diff moved to src/builtins/stats.rs.)
// Phase K.9 — rolling/window reductions
// (bi_roll* moved to src/builtins/stats.rs.)
fn bi_median(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { return r2_stats::bi_median(a);
    #[allow(unreachable_code)]
    let mut v: Vec<f64> = vec![];
    let n = v.len();

    // Phase D: Oracle picks between two correct algorithms.
    //   - Serial (small n)   → quickselect via `select_nth_unstable_by` — O(n).
    //   - Parallel (large n) → Rayon `par_sort_by` then index — uses all cores.
    // Quickselect alone outperforms `sort` for medians, but doesn't parallelize.
    // Sort-based path is the one that benefits from Rayon.
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    let go_par = r2_oracle::should_parallelize(
        r2_oracle::Op::Reduction,
        r2_oracle::Shape::n(n),
    );
    let m = if go_par {
        v.par_sort_by(cmp);
        if n % 2 == 0 { (v[n/2 - 1] + v[n/2]) / 2.0 } else { v[n/2] }
    } else if n % 2 == 0 {
        // Need both middle elements. Quickselect the upper, then max of the lower half.
        let upper_idx = n / 2;
        let (_lower, upper, _) = v.select_nth_unstable_by(upper_idx, cmp);
        let upper_val = *upper;
        let lower_val = _lower.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (lower_val + upper_val) / 2.0
    } else {
        let mid = n / 2;
        let (_, m, _) = v.select_nth_unstable_by(mid, cmp);
        *m
    };
    Ok(rnum(m))
}
// (bi_quantile moved to src/builtins/stats.rs.)

// ═══════════════════════════════════════════════════════════════════════
// MORE DISTRIBUTIONS: pnorm, qnorm, rbinom, rpois, dbinom
// ═══════════════════════════════════════════════════════════════════════

// (bi_pnorm moved to src/builtins/stats.rs.)

// (bi_qnorm moved to src/builtins/stats.rs.)

// (bi_rbinom moved to src/builtins/stats.rs.)

// (bi_rpois moved to src/builtins/stats.rs.)

// Error function approximation (Abramowitz & Stegun)
// Phase R.9: erf, phi, qnorm_approx now live in r2_stats::dist.
// Engine uses re-exports below to keep call sites unchanged.
use r2_stats::{phi, qnorm_approx};

// Phase R.10: signif_stars, fmt_pval moved to r2_stats::tests
// (re-exported at crate root). Engine model summaries (lm, glm) still
// import the same functions via the re-export below.
use r2_stats::{fmt_pval, signif_stars};

// Phase R.9: qnorm_approx now lives in r2_stats::dist (re-exported above).

// ═══════════════════════════════════════════════════════════════════════
// source() — run R2 script file
// ═══════════════════════════════════════════════════════════════════════

fn bi_source(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).ok_or(R2Err{msg:"NA path".into(),kind:ErrKind::Runtime})?, _ => return err!(Runtime, "source() needs file path") };
    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot read '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let stmts = r2_parser::Parser::parse(&content).map_err(|pe| R2Err{msg:format!("parse error in '{}': {}", path, pe),kind:ErrKind::Runtime})?;
    let mut last = RVal::Null;
    for stmt in &stmts {
        last = e.eval_in(stmt, env)?;
    }
    Ok(last)
}

// ═══════════════════════════════════════════════════════════════════════
// system.time() — measure execution time
// ═══════════════════════════════════════════════════════════════════════

fn bi_system_time(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let func = gv(a,0);
    let start = std::time::Instant::now();
    let _ = e.call_fn(&func, &[], env)?;
    let elapsed = start.elapsed();
    println!("   user  system elapsed");
    println!("  {:.3}   0.000   {:.3}", elapsed.as_secs_f64(), elapsed.as_secs_f64());
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// t.test() — Student's t-test
// ═══════════════════════════════════════════════════════════════════════

// (bi_t_test moved to src/builtins/stats.rs.)

// Phase R.10: t_cdf, incomplete_beta, gamma_approx live in r2_stats.
// All engine call sites migrated; imports retired.

// ═══════════════════════════════════════════════════════════════════════
// chisq.test() — Chi-squared test for independence
// ═══════════════════════════════════════════════════════════════════════

// (bi_chisq_test moved to src/builtins/stats.rs.)

// Phase R.S.2 — Hotelling T² (one-sample / two-sample / paired)
fn bi_hotelling_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::multivariate::bi_hotelling_test(a)
}
// Phase R.S.2 — MANOVA (multivariate analysis of variance)
fn bi_manova(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::multivariate::bi_manova(a)
}
// Phase R.S.3 — linear mixed-effects model (random intercept)
fn bi_lmer(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::mixed::bi_lmer(a)
}

// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.1 — Date / POSIXct builtins (thin engine adapters)
// ═══════════════════════════════════════════════════════════════════════

fn bi_as_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_as_date(a)
}
fn bi_as_posixct(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_as_posixct(a)
}
fn bi_format_time(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_format_time(a)
}
fn bi_sys_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_sys_date(a)
}
fn bi_sys_time(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_sys_time(a)
}
fn bi_difftime(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_difftime(a)
}
// ── Phase R.T.2 — ts() class adapters ─────────────────────────────────
// (bi_ts moved to src/builtins/stats.rs.)
fn bi_tsp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_tsp(a) }
fn bi_ts_start(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_start(a) }
fn bi_ts_end(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_end(a) }
fn bi_frequency(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_frequency(a) }
fn bi_deltat(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_deltat(a) }
fn bi_time_idx(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_time(a) }
fn bi_cycle(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_cycle(a) }
fn bi_window(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_window(a) }
fn bi_is_ts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_ts(a) }
// ── Phase R.T.3 — xts adapters ────────────────────────────────────────
fn bi_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_xts(a) }
fn bi_index(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_index(a) }
fn bi_coredata(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_coredata(a) }
fn bi_is_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_xts(a) }
fn bi_xts_subset(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_xts_subset(a) }
fn bi_first(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_first(a) }
fn bi_last(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_last(a) }
fn bi_na_locf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_na_locf(a) }
fn bi_merge_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_merge_xts(a) }
// ── Phase R.T.4 — TS analytics ─────────────────────────────────────────
fn bi_acf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_acf(a) }
fn bi_pacf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_pacf(a) }
fn bi_decompose(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_decompose(a) }
fn bi_is_regular(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_regular(a) }
fn bi_periodicity(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_periodicity(a) }
fn bi_lag(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_lag(a) }
fn bi_diff_ts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_diff_ts(a) }
// ── Phase R.T.5 — period aggregation ────────────────────────────────
fn bi_to_daily(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_daily(a) }
fn bi_to_weekly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_weekly(a) }
fn bi_to_monthly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_monthly(a) }
fn bi_to_quarterly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_quarterly(a) }
fn bi_to_yearly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_yearly(a) }
fn bi_apply_daily(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_daily(a) }
fn bi_apply_weekly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_weekly(a) }
fn bi_apply_monthly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_monthly(a) }
fn bi_apply_quarterly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_quarterly(a) }
fn bi_apply_yearly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_yearly(a) }
// ── Phase R.T.5b — Hindu calendar ───────────────────────────────────
fn bi_tithi(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_tithi(a) }
fn bi_hindu_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_hindu_date(a) }
fn bi_hnc_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_hnc_date(a) }

// ═══════════════════════════════════════════════════════════════════════
// Phase R.S.4 — Addon installers (r2-pkg backed)
//
// install.from.dir("path/to/pkg")
//   Validates package.r2 manifest and copies into ~/.r2/packages/<name>/
//
// install.from.zip("path/to/pkg.zip")
//   Extracts into a tmp dir then delegates to install_from_dir. Uses the
//   system `unzip` (or `tar -xf` on Windows 10+) — no Rust zip dep.
//
// install.from.github("user/repo" [, ref="main"])
//   Shells out to `git clone --depth 1` into a tmp dir, then installs.
//
// uninstall("name")
//   Removes the package directory; library() of it afterwards will fail.
// ═══════════════════════════════════════════════════════════════════════

fn bi_install_from_dir(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA path".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.dir() needs a path (character string)"),
    };
    let m = r2_pkg::install_from_dir(std::path::Path::new(&path))
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, path);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

fn bi_install_from_zip(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let zip_path = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA path".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.zip() needs a zip path"),
    };
    let zip = std::path::Path::new(&zip_path);
    if !zip.exists() {
        return err!(Runtime, "zip file not found: {}", zip_path);
    }
    // Extract to a tmp directory using the platform's unzip/tar tool.
    let tmp = std::env::temp_dir().join(format!("r2-install-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)
        .map_err(|e| R2Err { msg: format!("cannot create tmp dir: {}", e), kind: ErrKind::Runtime })?;

    // Try `tar -xf` first (available on Windows 10+, macOS, Linux).
    let status = std::process::Command::new("tar")
        .arg("-xf").arg(zip)
        .arg("-C").arg(&tmp)
        .status();
    if !matches!(&status, Ok(s) if s.success()) {
        // Fall back to `unzip`.
        let status2 = std::process::Command::new("unzip")
            .arg("-q").arg(zip).arg("-d").arg(&tmp)
            .status();
        if !matches!(status2, Ok(s) if s.success()) {
            let _ = std::fs::remove_dir_all(&tmp);
            return err!(Runtime, "could not extract '{}' — neither `tar` nor `unzip` succeeded. Install one or extract manually then use install.from.dir().", zip_path);
        }
    }

    // Find the package root inside tmp (top-level dir containing package.r2,
    // OR tmp itself if the zip was extracted flat).
    let src_dir = find_pkg_root(&tmp)
        .ok_or_else(|| R2Err { msg: format!("no package.r2 found inside '{}'", zip_path), kind: ErrKind::Runtime })?;

    let m = r2_pkg::install_from_dir(&src_dir)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let _ = std::fs::remove_dir_all(&tmp);
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, zip_path);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

fn bi_install_from_github(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let spec = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA repo".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.github() needs a repo spec like \"user/repo\""),
    };
    // Optional ref="branch_or_tag" arg.
    let git_ref: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("ref"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let url = if spec.starts_with("http://") || spec.starts_with("https://") || spec.starts_with("git@") {
        spec.clone()
    } else if spec.contains('/') {
        format!("https://github.com/{}.git", spec)
    } else {
        return err!(Runtime, "github spec must be \"user/repo\" or a full git URL; got '{}'", spec);
    };

    let tmp = std::env::temp_dir().join(format!("r2-gh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);

    let mut cmd = std::process::Command::new("git");
    cmd.arg("clone").arg("--depth").arg("1");
    if let Some(r) = &git_ref {
        cmd.arg("--branch").arg(r);
    }
    cmd.arg(&url).arg(&tmp);
    let status = cmd.status()
        .map_err(|e| R2Err { msg: format!("cannot run `git`: {}. Install git and put it on PATH.", e), kind: ErrKind::Runtime })?;
    if !status.success() {
        return err!(Runtime, "git clone failed for '{}'", url);
    }

    // Optional subdir = "r2pkg-foo" arg — for monorepos where many
    // packages live under one repo (e.g. Ardon-R2-libraries).
    let subdir: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("subdir"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let search_root = match &subdir {
        Some(s) => tmp.join(s),
        None    => tmp.clone(),
    };
    if !search_root.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
        return err!(Runtime, "subdir '{}' not found in cloned repo '{}'", subdir.unwrap_or_default(), spec);
    }

    let src_dir = find_pkg_root(&search_root)
        .ok_or_else(|| R2Err { msg: format!("no package.r2 found in cloned repo '{}'{}", spec, subdir.as_ref().map(|s| format!(" subdir '{}'", s)).unwrap_or_default()), kind: ErrKind::Runtime })?;

    let m = r2_pkg::install_from_dir(&src_dir)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let _ = std::fs::remove_dir_all(&tmp);
    let from_str = match subdir { Some(s) => format!("github:{}/{}", spec, s), None => format!("github:{}", spec) };
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, from_str);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

fn bi_uninstall(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "uninstall() needs a package name"),
    };
    r2_pkg::uninstall(&name)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    println!("* removed package '{}'", name);
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// Unified `install.packages()` dispatcher
//
// One entry point that auto-detects the install source from the `path`
// argument. Mirrors R's `install.packages()` for muscle memory, and
// future-compatible with a website-based registry.
//
// Signature:
//   install.packages(name, path = NULL, subdir = NULL, ref = NULL)
//
// `path` dispatch rules (checked in order):
//
//   1. NULL / omitted        → registry lookup (not yet implemented;
//                              returns an informative error pointing
//                              at the future r2-packages.dev endpoint)
//   2. ends with ".zip"      → local zip extract → install
//                              (or download-then-extract if it's a URL)
//   3. "user/repo[/subdir]"  → github shorthand: clone, then install
//                              (a single `/`, no path separator, no
//                              backslash, no dot before the slash)
//   4. starts with "http://" or "https://" or "git@" or ends ".git"
//                            → git clone or URL download
//   5. an existing directory → install.from.dir
//
// `name` is the expected package name; after extracting/cloning we
// verify it matches the manifest's `package_name` and warn on mismatch.
// `subdir` (optional) — for monorepos like Ardon-R2-libraries.
// `ref`    (optional) — branch/tag for GitHub installs.
// ═══════════════════════════════════════════════════════════════════════

fn bi_install_packages(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.packages() needs a package name (character string)"),
    };
    let path: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("path"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let path = match path {
        Some(p) => p,
        None => {
            return err!(Runtime,
                "install.packages('{}') without 'path' would consult the R2 registry, \
                 which is not yet online. Until then, supply path = ... pointing to: \
                 \n  - a local directory:        path = 'path/to/pkg-dir'\
                 \n  - a local zip file:         path = 'path/to/pkg.zip'\
                 \n  - a GitHub repo:            path = 'user/repo'\
                 \n  - a monorepo subdir:        path = 'user/repo', subdir = 'r2pkg-foo'\
                 \n  - a specific branch/tag:    path = 'user/repo', ref = 'v0.1.0'", name);
        }
    };

    // Classify the path. Order matters: zip before github-shorthand
    // (since "foo/bar.zip" contains a slash too).
    let is_url   = path.starts_with("http://") || path.starts_with("https://");
    let is_zip   = path.to_lowercase().ends_with(".zip");
    let is_git   = path.starts_with("git@") || path.ends_with(".git");
    let is_dir   = std::path::Path::new(&path).is_dir();
    // GitHub shorthand: one slash, no backslash, no path-separator
    // characters past the slash, and the part before the slash has no
    // dots (would otherwise look like a domain).
    let is_github_shorthand = {
        let p = path.trim_end_matches('/');
        let slashes: Vec<usize> = p.match_indices('/').map(|(i, _)| i).collect();
        !is_url && !is_git && !is_zip && !is_dir &&
            slashes.len() >= 1 && !p.contains('\\') &&
            !p[..slashes[0]].contains('.')
    };

    // Build a faux EvalArg list to call our existing helpers.
    let make_arg = |val: RVal| EvalArg { name: None, value: val };

    if is_dir {
        let m = r2_pkg::install_from_dir(std::path::Path::new(&path))
            .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
        verify_name(&m.name, &name);
        println!("* installed '{}' (version {}) from local dir {}", m.name, m.version, path);
        return Ok(RVal::Null);
    }

    if is_zip {
        // Existing handler accepts a local path. URL-zip support would
        // need a download step; for now we error if the user passed a URL.
        if is_url {
            return err!(Runtime, "install.packages(): downloading remote .zip is not yet supported. \
                Download manually then pass the local path.");
        }
        let zip_args = vec![make_arg(RVal::Character(vec![Some(std::sync::Arc::from(path.as_str()))], Attrs::default()))];
        return bi_install_from_zip(e, &zip_args, env);
    }

    if is_github_shorthand || is_git || is_url {
        // Reuse install.from.github for the heavy lifting. Pass through
        // subdir and ref if present.
        let mut gh_args = vec![make_arg(RVal::Character(vec![Some(std::sync::Arc::from(path.as_str()))], Attrs::default()))];
        for arg_name in &["ref", "subdir"] {
            if let Some(x) = a.iter().find(|x| x.name.as_deref() == Some(*arg_name)) {
                gh_args.push(EvalArg { name: Some(std::sync::Arc::from(*arg_name)), value: x.value.clone() });
            }
        }
        return bi_install_from_github(e, &gh_args, env);
    }

    err!(Runtime,
        "install.packages(): could not classify path '{}'. Expected one of:\
         \n  - existing local directory\
         \n  - .zip file\
         \n  - GitHub shorthand 'user/repo'\
         \n  - full URL (https://... or git@...)", path)
}

fn verify_name(manifest_name: &str, requested_name: &str) {
    if manifest_name != requested_name {
        eprintln!(
            "Warning: package name mismatch — you asked for '{}' but the manifest \
             says '{}'. Installed under '{}'.",
            requested_name, manifest_name, manifest_name);
    }
}

// Search up to 2 levels deep for a directory containing package.r2.
fn find_pkg_root(start: &std::path::Path) -> Option<std::path::PathBuf> {
    if start.join("package.r2").exists() { return Some(start.to_path_buf()); }
    if let Ok(entries) = std::fs::read_dir(start) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("package.r2").exists() {
                return Some(p);
            }
        }
    }
    None
}

// Phase R.10: chi_sq_cdf and ln_gamma moved to r2_stats. No engine callers remain.

// ═══════════════════════════════════════════════════════════════════════
// predict() and residuals() for lm objects
// ═══════════════════════════════════════════════════════════════════════

fn bi_predict(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let model = gv(a, 0);
    let newdata = gn(a, "newdata").or(Some(gv(a, 1)));

    match &model {
        RVal::TypeInstance(inst) => {
            match inst.type_name.as_ref() {
                "lm" | "glm" => {
                    // If newdata provided, compute X*beta; else return fitted
                    if let Some(RVal::Matrix(xnew)) = &newdata {
                        let coeffs: Vec<f64> = e.as_reals(inst.fields.get("coefficients").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let p = coeffs.len();
                        let n = xnew.nrow;
                        let mut preds = vec![0.0; n];
                        for i in 0..n {
                            preds[i] = coeffs[0]; // intercept
                            for j in 1..p.min(xnew.ncol + 1) {
                                preds[i] += coeffs[j] * xnew.get(i, j - 1);
                            }
                        }
                        // Apply link function for glm
                        if inst.type_name.as_ref() == "glm" {
                            let family = inst.fields.get("family").map(|v| val_to_str(v)).unwrap_or("gaussian".into());
                            for p in preds.iter_mut() {
                                *p = match family.as_str() {
                                    "binomial" => 1.0 / (1.0 + (-*p).exp()),
                                    "poisson" => p.exp(),
                                    _ => *p,
                                };
                            }
                        }
                        Ok(rnums(&preds))
                    } else {
                        inst.fields.get("fitted.values").cloned().ok_or(R2Err{msg:"no fitted values".into(),kind:ErrKind::Runtime})
                    }
                }
                "rpart" => {
                    // Predict using serialized tree
                    if let Some(RVal::Matrix(xnew)) = &newdata {
                        let feat: Vec<f64> = e.as_reals(inst.fields.get("_tree_feat").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let thresh: Vec<f64> = e.as_reals(inst.fields.get("_tree_thresh").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let pred: Vec<f64> = e.as_reals(inst.fields.get("_tree_pred").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let leaf: Vec<f64> = e.as_reals(inst.fields.get("_tree_leaf").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let left: Vec<f64> = e.as_reals(inst.fields.get("_tree_left").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let right: Vec<f64> = e.as_reals(inst.fields.get("_tree_right").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();

                        let mut preds = Vec::new();
                        for i in 0..xnew.nrow {
                            let mut node = 0usize;
                            loop {
                                if node >= leaf.len() || leaf[node] == 1.0 { break; }
                                let f = feat[node] as usize;
                                let t = thresh[node];
                                if xnew.get(i, f) <= t { node = left[node] as usize; }
                                else { node = right[node] as usize; }
                            }
                            preds.push(if node < pred.len() { pred[node] } else { 0.0 });
                        }
                        Ok(rnums(&preds))
                    } else {
                        inst.fields.get("predictions").cloned().ok_or(R2Err{msg:"no predictions".into(),kind:ErrKind::Runtime})
                    }
                }
                "rf" | "kmeans" | "naive.bayes" | "gbm" => {
                    inst.fields.get("predictions").or(inst.fields.get("cluster")).cloned()
                        .ok_or(R2Err{msg:"no predictions".into(),kind:ErrKind::Runtime})
                }
                _ => err!(Runtime, "predict() does not support {} objects", inst.type_name),
            }
        }
        _ => err!(Runtime, "predict() needs a model object (lm, glm, rpart, rf, kmeans)"),
    }
}

fn bi_residuals(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" => {
            inst.fields.get("residuals").cloned().ok_or(R2Err{msg:"no residuals".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "residuals() needs an lm object"),
    }
}

fn bi_fitted(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" => {
            inst.fields.get("fitted.values").cloned().ok_or(R2Err{msg:"no fitted values".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "fitted() needs an lm object"),
    }
}

fn bi_coef(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" || inst.type_name.as_ref() == "glm" => {
            inst.fields.get("coefficients").cloned().ok_or(R2Err{msg:"no coefficients".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "coef() needs an lm or glm object"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// boxplot() — SVG box-and-whisker plot
// ═══════════════════════════════════════════════════════════════════════

// (bi_boxplot moved to src/builtins/graphics.rs. The dead-body
// fallback below was #[cfg(any())] — never compiled — and is
// deleted as part of the migration cleanup.)

#[cfg(any())]
#[allow(dead_code, unused_variables)]
fn _deleted_legacy_bi_boxplot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let title = gn(a,"main").map(|v| val_to_str(&v)).unwrap_or("Boxplot".into());
    let (w, h) = (500.0, 400.0);
    let (ml, mr, mt, mb) = (60.0, 30.0, 30.0, 40.0);
    let pw = w - ml - mr; let ph = h - mt - mb;

    // Collect each argument as a data group
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for (gi, arg) in a.iter().enumerate() {
        if arg.name.as_ref().map(|n| n.as_ref()) == Some("main") { continue; }
        let data: Vec<f64> = e.as_reals(&arg.value)?.into_iter().filter_map(|x| x).collect();
        let name = arg.name.as_ref().map(|n| n.to_string()).unwrap_or(format!("V{}", gi + 1));
        groups.push((name, data));
    }

    if groups.is_empty() { return err!(Runtime, "boxplot needs data"); }

    // Find global range
    let all_min = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::INFINITY, f64::min);
    let all_max = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (all_max - all_min).abs() < 1e-10 { 1.0 } else { all_max - all_min };

    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#, w, h, w, h);
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(r#"<text x="{}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#, w/2.0, title));

    let ng = groups.len() as f64;
    let bw = pw / ng * 0.6;
    let gap = pw / ng;

    for (i, (name, data)) in groups.iter().enumerate() {
        if data.len() < 2 { continue; }
        let mut sorted = data.clone();
        sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q1 = sorted[n / 4]; let median = sorted[n / 2]; let q3 = sorted[3 * n / 4];
        let min_val = sorted[0]; let max_val = sorted[n - 1];
        let iqr = q3 - q1;
        let lower_fence = (q1 - 1.5 * iqr).max(min_val);
        let upper_fence = (q3 + 1.5 * iqr).min(max_val);

        let cx = ml + gap * i as f64 + gap / 2.0;
        let map_y = |v: f64| mt + ph - (v - all_min) / range * ph;

        // Whiskers
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(lower_fence), cx, map_y(q1)));
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(q3), cx, map_y(upper_fence)));
        // Fence caps
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx-bw/4.0, map_y(lower_fence), cx+bw/4.0, map_y(lower_fence)));
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx-bw/4.0, map_y(upper_fence), cx+bw/4.0, map_y(upper_fence)));
        // Box
        let by = map_y(q3); let bh = map_y(q1) - by;
        svg.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="{:.0}" height="{:.0}" fill="{}" stroke="black"/>"#, cx-bw/2.0, by, bw, bh, "#93c5fd"));
        // Median line
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black" stroke-width="2"/>"#, cx-bw/2.0, map_y(median), cx+bw/2.0, map_y(median)));
        // Label
        svg.push_str(&format!(r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10">{}</text>"#, cx, h-mb+15.0, name));
    }
    svg.push_str("</svg>");
    let _ = std::fs::write("boxplot.svg", &svg);
    println!("Boxplot saved to boxplot.svg");
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// barplot() — SVG bar chart
// ═══════════════════════════════════════════════════════════════════════

// (bi_barplot moved to src/builtins/graphics.rs.)

#[cfg(any())]
#[allow(dead_code, unused_variables)]
fn _legacy_bi_barplot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let heights: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|x| x).collect();
    let title = gn(a,"main").map(|v| val_to_str(&v)).unwrap_or("Barplot".into());
    let names = gn(a,"names.arg");

    let labels: Vec<String> = if let Some(RVal::Character(v, _)) = &names {
        v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()
    } else {
        (1..=heights.len()).map(|i| format!("{}", i)).collect()
    };

    let (w, h) = (600.0, 400.0);
    let (ml, mr, mt, mb) = (60.0, 20.0, 30.0, 50.0);
    let pw = w - ml - mr; let ph = h - mt - mb;
    let max_h = heights.iter().cloned().fold(0.0f64, f64::max);
    let bw = pw / heights.len() as f64 * 0.8;
    let gap = pw / heights.len() as f64;

    let colors = vec!["#3b82f6","#ef4444","#22c55e","#f59e0b","#8b5cf6","#ec4899","#06b6d4","#f97316"];

    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#, w, h, w, h);
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(r#"<text x="{}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#, w/2.0, title));

    for (i, &val) in heights.iter().enumerate() {
        let bh = if max_h > 0.0 { val / max_h * ph } else { 0.0 };
        let bx = ml + gap * i as f64 + (gap - bw) / 2.0;
        let by = mt + ph - bh;
        let color = colors[i % colors.len()];
        svg.push_str(&format!(r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#, bx, by, bw, bh, color));
        // Value on top
        svg.push_str(&format!(r#"<text x="{:.0}" y="{:.0}" text-anchor="middle" font-size="10">{:.1}</text>"#, bx+bw/2.0, by-5.0, val));
        // Label below
        let label = labels.get(i).map(|s| s.as_str()).unwrap_or("");
        svg.push_str(&format!(r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10" transform="rotate(-30,{:.0},{})">{}</text>"#, bx+bw/2.0, h-mb+20.0, bx+bw/2.0, h-mb+20.0, label));
    }
    svg.push_str("</svg>");
    let _ = std::fs::write("barplot.svg", &svg);
    println!("Barplot saved to barplot.svg");
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// read.table(), write.table(), read.delim() — more file I/O
// ═══════════════════════════════════════════════════════════════════════

// (bi_read_table moved to src/builtins/io.rs.)

// (bi_read_delim moved to src/builtins/io.rs.)

// (bi_write_table moved to src/builtins/io.rs.)

// ═══════════════════════════════════════════════════════════════════════
// which.min(), which.max(), range(), prod(), any(), all()
// ═══════════════════════════════════════════════════════════════════════

fn bi_which_min(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_which_min(a) }

fn bi_which_max(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_which_max(a) }

fn bi_range(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_range(a) }

// (bi_prod moved to src/builtins/stats.rs.)

fn bi_any(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a,0))?;
    Ok(rbool(v.iter().any(|x| *x == Some(true))))
}

fn bi_all(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a,0))?;
    Ok(rbool(v.iter().all(|x| *x == Some(true))))
}

// ═══════════════════════════════════════════════════════════════════════
// sprintf(), trimws(), startsWith(), endsWith(), nrow/ncol for matrix
// ═══════════════════════════════════════════════════════════════════════

// (bi_sprintf moved to src/builtins/strings.rs.)

// (bi_trimws moved to src/builtins/strings.rs.)

fn bi_starts_with(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = match &gv(a,0) { RVal::Character(v,_) => v.clone(), _ => return err!(Type, "startsWith needs character") };
    let prefix = match &gv(a,1) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Type, "startsWith needs prefix") };
    let result: Vec<Logical> = x.iter().map(|s| s.as_ref().map(|s| s.starts_with(&prefix.as_str()))).collect();
    Ok(RVal::Logical(result.into(), Attrs::default()))
}

fn bi_ends_with(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = match &gv(a,0) { RVal::Character(v,_) => v.clone(), _ => return err!(Type, "endsWith needs character") };
    let suffix = match &gv(a,1) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Type, "endsWith needs suffix") };
    let result: Vec<Logical> = x.iter().map(|s| s.as_ref().map(|s| s.ends_with(&suffix.as_str()))).collect();
    Ok(RVal::Logical(result.into(), Attrs::default()))
}

fn bi_Sys_time(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    Ok(rnum(now))
}

fn bi_stop(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    err!(Runtime, "{}", msg)
}

fn bi_warning(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    e.warnings.push(format!("Warning: {}", msg));
    Ok(RVal::Null)
}

fn bi_message(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    eprintln!("{}", msg);
    Ok(RVal::Null)
}

fn bi_ls(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let names: Vec<Character> = e.global_env.bindings.keys()
        .map(|k| Some(k.clone()))
        .collect();
    Ok(RVal::Character(names, Attrs::default()))
}

fn bi_rm(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Runtime, "rm needs name") };
    let mut binding = e.global_env.clone();
    let g = Arc::make_mut(&mut binding);
    g.bindings.remove(name.as_str());
    e.global_env = Arc::new(g.clone());
    Ok(RVal::Null)
}

fn bi_exists(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Runtime, "exists needs name") };
    Ok(rbool(e.global_env.lookup(&name).is_some() || e.registry.resolve(&name).is_some()))
}

// ═══════════════════════════════════════════════════════════════════════
// glm() — Generalized Linear Model (logistic regression via IRLS)
// ═══════════════════════════════════════════════════════════════════════

fn bi_glm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_glm(a) }

// ═══════════════════════════════════════════════════════════════════════
// confint() — confidence intervals for model coefficients
// ═══════════════════════════════════════════════════════════════════════

fn bi_confint(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let level = gn(a, "level").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(0.95);
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" || inst.type_name.as_ref() == "glm" => {
            let coeffs_val = inst.fields.get("coefficients").ok_or(R2Err{msg:"no coefficients".into(),kind:ErrKind::Runtime})?;
            let coeffs = e.as_reals(coeffs_val)?.into_iter().filter_map(|x| x).collect::<Vec<f64>>();
            let sigma = inst.fields.get("sigma").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(1.0);
            let df = inst.fields.get("df").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(30.0);

            let alpha = 1.0 - level;
            let t_crit = qnorm_approx(1.0 - alpha / 2.0); // approximate

            let names: Vec<String> = match coeffs_val {
                RVal::Numeric(_, attrs) => attrs.names.as_ref().map(|n| n.iter().map(|s| s.to_string()).collect()).unwrap_or_else(|| (0..coeffs.len()).map(|i| format!("x{}", i)).collect()),
                _ => (0..coeffs.len()).map(|i| format!("x{}", i)).collect(),
            };

            // Standard errors (simplified — assumes diagonal of (X'X)^-1 * sigma^2)
            let se = sigma / (df.sqrt());

            println!("{:>15} {:>15} {:>15}", "", fmt_num(alpha/2.0*100.0).to_string() + " %", fmt_num((1.0-alpha/2.0)*100.0).to_string() + " %");
            for (i, name) in names.iter().enumerate() {
                let lo = coeffs[i] - t_crit * se;
                let hi = coeffs[i] + t_crit * se;
                println!("{:>15} {:>15} {:>15}", name, fmt_num(lo), fmt_num(hi));
            }
            Ok(RVal::Null)
        }
        _ => err!(Runtime, "confint() needs lm or glm object"),
    }
}

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
    println!("Legend added to {}", svg_path);
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// help() — basic help system
// ═══════════════════════════════════════════════════════════════════════

fn bi_help(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let topic = val_to_str(&gv(a,0));
    let help_text = match topic.as_str() {
        // Statistics
        "lm" => "lm(formula, data)\n  Linear regression.\n  Example: lm(mpg ~ wt, data = mtcars)\n         lm(mpg ~ ., data = mtcars)  # all predictors\n  Returns: coefficients, residuals, fitted.values, r.squared",
        "glm" => "glm(formula, data, family)\n  Generalized linear model.\n  family: \"gaussian\" (default), \"binomial\" (logistic), \"poisson\"\n  Example: glm(y ~ x, data = df, family = \"binomial\")",
        "t.test" => "t.test(x, y, mu)\n  Student's t-test.\n  One-sample: t.test(x, mu = 0)\n  Two-sample: t.test(x, y)",
        "chisq.test" => "chisq.test(x, p) or chisq.test(matrix)\n  Goodness-of-fit: chisq.test(c(10,20,30), p=c(0.2,0.3,0.5))\n  Independence:    chisq.test(matrix(c(10,20,30,40), nrow=2))\n  Returns: statistic, p.value, parameter (df)",
        "aov" => "aov(y ~ group, data = df)\n  One-way Analysis of Variance.\n  Tests if group means differ significantly.\n  Returns: f.statistic, p.value, ss.between, ss.within\n  Example: aov(Sepal.Length ~ Species, data = iris)",
        "anova" => "anova(model)\n  ANOVA table for lm/glm model.\n  Shows: Source, Df, Sum Sq, Mean Sq, F value, Pr(>F)\n  Example: anova(lm(mpg ~ wt + hp, data = mtcars))",
        "cor.test" => "cor.test(x, y)\n  Test if Pearson correlation is significant.\n  Returns: estimate (r), statistic (t), p.value, df\n  Example: cor.test(iris$Sepal.Length, iris$Petal.Length)",
        "shapiro.test" => "shapiro.test(x)\n  Shapiro-Wilk test for normality.\n  H0: data is normally distributed.\n  Returns: statistic (W), p.value\n  Example: shapiro.test(iris$Sepal.Length)",
        "wilcox.test" => "wilcox.test(x, y) or wilcox.test(x, mu = 0)\n  Wilcoxon rank-sum (2-sample) or signed-rank (1-sample) test.\n  Non-parametric alternative to t.test.\n  Example: wilcox.test(x, y)",
        "fisher.test" => "fisher.test(m)\n  Fisher's exact test for 2x2 contingency tables.\n  m: 2x2 matrix of counts.\n  Returns: p.value, estimate (odds ratio)\n  Example: fisher.test(matrix(c(10,5,3,12), nrow=2))",
        "weighted.mean" => "weighted.mean(x, w)\n  Weighted arithmetic mean.\n  Example: weighted.mean(c(1,2,3), c(0.5, 0.3, 0.2))",
        "IQR" => "IQR(x)\n  Interquartile range (Q3 - Q1).\n  Example: IQR(iris$Sepal.Length)",
        // ML
        "rpart" => "rpart(x, y) or rpart(y ~ ., data = df)\n  Decision tree (CART).\n  Args: max_depth=5, min_samples=5, type=\"auto\"\n  Auto-detects regression vs classification.\n  Example: rpart(Petal.Length ~ ., data = iris)",
        "rf" => "rf(x, y) or rf(y ~ ., data = df)\n  Random forest.\n  Args: ntrees=100, max_depth=10, type=\"classification\"\n  Returns: predictions, feature importance\n  Example: rf(Species ~ ., data = iris, ntrees = 50)",
        "gbm" => "gbm(x, y) or gbm(y ~ ., data = df)\n  Gradient boosted trees (XGBoost-style).\n  Args: ntrees=100, learning_rate=0.1, max_depth=3,\n        subsample=0.8, loss=\"squared\"/\"logistic\"/\"huber\"\n  Returns: predictions, importance, train.loss\n  Example: gbm(mpg ~ ., data = mtcars, ntrees = 100)",
        "kmeans" => "kmeans(x, centers = k)\n  K-means clustering.\n  Args: centers (required), iter.max=100\n  Returns: cluster, centers, withinss, totss\n  Example: kmeans(x, centers = 3)",
        "knn" => "knn(train, test, labels, k = 3)\n  K-nearest neighbors classification.\n  Example: knn(x_train, x_test, y_train, k = 5)",
        "prcomp" => "prcomp(x)\n  Principal Component Analysis.\n  Args: center=TRUE, scale.=FALSE\n  Returns: sdev, eigenvalues, prop.variance\n  Example: prcomp(iris[,1:4])",
        "naive.bayes" => "naive.bayes(x, y)\n  Gaussian Naive Bayes classifier.\n  Returns: classes, priors, means, vars",
        "cv" => "cv(x, y, model = \"lm\", k = 5)\n  K-fold cross-validation.\n  model: \"lm\" or \"rf\"\n  Returns: per-fold MSE, mean, sd\n  Example: cv(x, y, model = \"lm\", k = 10)",
        "confusion.matrix" => "confusion.matrix(predicted, actual)\n  Confusion matrix with precision, recall, F1.\n  Example: confusion.matrix(pred, y)",
        // Graphics
        "plot" => "plot(x, y, main, xlab, ylab, col)\n  Scatter plot (SVG output).\n  Example: plot(x, y, main = \"Title\")",
        "hist" => "hist(x, breaks, main)\n  Histogram (SVG output).\n  Example: hist(rnorm(1000), breaks = 20)",
        "boxplot" => "boxplot(x, y, ..., main)\n  Box-and-whisker plot.\n  Example: boxplot(iris$Sepal.Length)",
        "barplot" => "barplot(heights, names.arg, main)\n  Bar chart.\n  Example: barplot(c(10,20,30))",
        // Data
        "read.csv" => "read.csv(file, header=TRUE, sep=\",\")\n  Read CSV into data.frame. Handles quotes, NA, type inference.\n  Example: df <- read.csv(\"data.csv\")",
        "filter" => "filter(df, mask)\n  Keep rows where mask is TRUE.\n  Example: filter(iris, iris$Sepal.Length > 7)",
        "select" => "select(df, \"col1\", \"col2\")\n  Keep only named columns.\n  Example: select(iris, \"Sepal.Length\", \"Species\")",
        "mutate" => "mutate(df, new_col = values)\n  Add or modify columns.\n  Example: mutate(iris, ratio = iris$Sepal.Length / iris$Sepal.Width)",
        "arrange" => "arrange(df, col_values, decreasing=FALSE)\n  Sort data.frame by values.",
        "save" => "save(file) or save(object, file)\n  Save session or single object.\n  Extensions: .r2s (session), .r2d (data), .r2m (model)\n  Examples:\n    save(\"session.r2s\")       # save all variables\n    save(iris, \"data.r2d\")     # save data object\n    save(model, \"model.r2m\")   # save trained model",
        "load" => "load(file)\n  Load saved session, data, or model.\n  Returns loaded object for .r2d and .r2m files.\n  Examples:\n    load(\"session.r2s\")        # restore all variables\n    d <- load(\"data.r2d\")      # load data\n    m <- load(\"model.r2m\")     # load model",
        // Core
        "c" => "c(...)\n  Combine values into a vector.\n  Example: c(1, 2, 3)",
        "library" => "library(package)\n  Load a package.\n  Example: library(mymath)",
        "data.frame" => "data.frame(...)\n  Create data frame.\n  Example: data.frame(x = 1:5, y = c(\"a\",\"b\",\"c\",\"d\",\"e\"))",
        "matrix" => "matrix(data, nrow, ncol)\n  Create matrix.\n  Example: matrix(1:12, nrow = 3, ncol = 4)",
        "scale" => "scale(x, center=TRUE, scale=TRUE)\n  Center and standardize matrix columns.",
        ".Internal" | "internal" => ".Internal(name, ...)\n  Call Rust primitive from Ardon-R2 script.\n  Available primitives:\n    matmul, crossprod, crossprod_vec, solve, solve_lstsq,\n    inverse, cholesky, eigenvalues, svd,\n    rnorm_vec, pnorm, qnorm\n  Example: beta <- .Internal(\"solve_lstsq\", X, y)",
        "summary" | "str" | "head" | "tail" | "names" | "dim" | "class" => "Data inspection functions:\n  summary(x)  — summary statistics\n  str(x)      — structure\n  head(x, n)  — first n rows\n  tail(x, n)  — last n rows\n  names(x)    — column names\n  dim(x)      — dimensions\n  class(x)    — type/class",
        _ => "Ardon-R2 Help System — Available topics:\n\n  Statistics:  lm, glm, t.test, chisq.test, cor, cor.test\n               aov, anova, shapiro.test, wilcox.test, fisher.test\n               mean, sd, var, median, quantile, IQR, weighted.mean\n  ML:          rpart, rf, gbm, kmeans, knn, prcomp, naive.bayes\n  Evaluation:  cv, confusion.matrix\n  Graphics:    plot, hist, boxplot, barplot\n  Data:        read.csv, filter, select, mutate, arrange\n  Session:     save, load, version\n  Core:        c, library, data.frame, matrix, scale, .Internal\n  Inspection:  summary, str, head, tail, names, dim, class\n\n  Type help(\"topic\") or ?topic for details.",
    };
    println!("\n{}\n", help_text);
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// Sys.getenv(), Sys.setenv(), getwd(), setwd()
// ═══════════════════════════════════════════════════════════════════════

fn bi_getwd(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let cwd = std::env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    Ok(rstr(&cwd))
}

fn bi_setwd(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a,0));
    std::env::set_current_dir(&path).map_err(|e| R2Err{msg:format!("cannot set working directory: {}", e),kind:ErrKind::Runtime})?;
    Ok(rstr(&path))
}

// ═══════════════════════════════════════════════════════════════════════
// as.factor(), data(), as.logical(), nlevels(), levels()
// ═══════════════════════════════════════════════════════════════════════

fn bi_as_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let val = gv(a, 0);
    match &val {
        RVal::Character(v, _) => {
            let mut levels: Vec<Arc<str>> = Vec::new();
            let codes: Vec<Option<u32>> = v.iter().map(|x| x.as_ref().map(|s| {
                let idx = levels.iter().position(|l| l == s).unwrap_or_else(|| { levels.push(s.clone()); levels.len() - 1 });
                idx as u32
            })).collect();
            Ok(RVal::Factor(Factor { codes, levels, ordered: false }))
        }
        RVal::Factor(..) => Ok(val), // already a factor
        RVal::Numeric(v, _) => {
            let mut levels: Vec<Arc<str>> = Vec::new();
            let codes: Vec<Option<u32>> = v.iter().map(|x| x.map(|n| {
                let s = Arc::from(fmt_num(n).as_str());
                let idx = levels.iter().position(|l| *l == s).unwrap_or_else(|| { levels.push(s); levels.len() - 1 });
                idx as u32
            })).collect();
            Ok(RVal::Factor(Factor { codes, levels, ordered: false }))
        }
        _ => err!(Type, "cannot coerce {} to factor", val.type_name()),
    }
}

fn bi_levels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Factor(f) => Ok(RVal::Character(f.levels.iter().map(|l| Some(l.clone())).collect(), Attrs::default())),
        _ => Ok(RVal::Null),
    }
}

fn bi_nlevels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) { RVal::Factor(f) => Ok(rint(f.levels.len() as i32)), _ => Ok(rint(0)) }
}

fn bi_as_logical(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a, 0))?;
    Ok(RVal::Logical(v.into(), Attrs::default()))
}

fn bi_data(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let val = gv(a, 0);
    match &val {
        RVal::Character(v, _) => {
            let name = v[0].as_ref().map(|s| s.to_string()).unwrap_or_default();
            if e.global_env.lookup(&name).is_some() {
                println!("Dataset '{}' is already loaded", name);
            } else {
                println!("Dataset '{}' not found", name);
            }
        }
        RVal::DataFrame(_) => {
            println!("Dataset is already loaded in the environment");
        }
        _ => {
            println!("Available datasets: iris, mtcars, airquality");
        }
    }
    Ok(RVal::Null)
}

// (bi_dim moved to src/builtins/data.rs.)

// (bi_colnames moved to src/builtins/data.rs.)

// (bi_rownames moved to src/builtins/data.rs.)

// (bi_is_data_frame moved to src/builtins/data.rs.)

fn bi_is_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Factor(_))))
}

fn bi_is_matrix(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Matrix(_))))
}

// ═══════════════════════════════════════════════════════════════════════
// set.seed() — reproducible random numbers
// ═══════════════════════════════════════════════════════════════════════

// Phase R.12: RNG primitives consolidated in r2_stats::rng. Engine
// retains a 1-line shim for the `.Internal("rnorm_vec", …)` path which
// R-language helper code still calls. All bi_* RNG builtins delegate
// directly to r2_stats::rng.
fn bi_set_seed(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::rng::bi_set_seed(a) }
fn r2_next_random() -> f64 { r2_stats::rng::next_random() }

// Phase R.1: parallel_random lives in r2_ml::tree. Engine no longer calls
// it directly (r2_ml::dispatch handles all rf/gbm RNG internally).

// ═══════════════════════════════════════════════════════════════════════
// as.data.frame() — convert matrix or list to data.frame
// ═══════════════════════════════════════════════════════════════════════

// (bi_as_data_frame moved to src/builtins/data.rs.)

// ═══════════════════════════════════════════════════════════════════════
// Memory safety: allocation guard (scaffolded, not yet wired into call sites)
// ═══════════════════════════════════════════════════════════════════════
//
// TODO (v0.2.0): call check_alloc() before any large allocation in builtins
// that take user-supplied size arguments (matrix, rep, seq with length.out,
// numeric(n), etc.) to give users a friendly error instead of an OOM-kill.
// Currently scaffolded but unused — the #[allow(dead_code)] survives the
// dead-code lint without removing the design.

#[allow(dead_code)]
const MAX_ALLOC_BYTES: usize = 500_000_000; // 500MB max single allocation

#[allow(dead_code)]
fn check_alloc(elements: usize, elem_size: usize) -> Result<(), R2Err> {
    let bytes = elements * elem_size;
    if bytes > MAX_ALLOC_BYTES {
        return err!(Runtime, "allocation of {} bytes exceeds limit (max {} MB). Use chunked processing for large data.", bytes, MAX_ALLOC_BYTES / 1_000_000);
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// rowSums(), colSums(), rowMeans(), colMeans()
// ═══════════════════════════════════════════════════════════════════════

fn bi_rowSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let mut sums = vec![0.0f64; nrow];
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    for (i, v) in vals.iter().enumerate() { if let Some(n) = v { sums[i] += n; } }
                }
            }
            Ok(rnums(&sums))
        }
        RVal::Matrix(m) => {
            let sums: Vec<f64> = (0..m.nrow).map(|r| (0..m.ncol).map(|c| m.get(r, c)).sum()).collect();
            Ok(rnums(&sums))
        }
        _ => err!(Runtime, "rowSums needs data.frame or matrix"),
    }
}

fn bi_colSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let mut results = Vec::new();
            for (_name, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    let s: f64 = vals.iter().filter_map(|x| *x).sum();
                    results.push(s);
                }
            }
            let mut attrs = Attrs::default();
            attrs.names = Some(df.columns.iter().filter_map(|(n, col)| {
                if e.as_reals(col).is_ok() { Some(n.clone()) } else { None }
            }).collect());
            Ok(RVal::Numeric(results.iter().map(|x| Some(*x)).collect(), attrs))
        }
        RVal::Matrix(m) => {
            let sums: Vec<f64> = (0..m.ncol).map(|c| (0..m.nrow).map(|r| m.get(r, c)).sum()).collect();
            Ok(rnums(&sums))
        }
        _ => err!(Runtime, "colSums needs data.frame or matrix"),
    }
}

fn bi_rowMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow();
            let ncol_num = df.columns.iter().filter(|(_, col)| e.as_reals(col).is_ok()).count();
            let mut sums = vec![0.0f64; nrow];
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    for (i, v) in vals.iter().enumerate() { if let Some(n) = v { sums[i] += n; } }
                }
            }
            Ok(rnums(&sums.iter().map(|s| s / ncol_num as f64).collect::<Vec<_>>()))
        }
        _ => err!(Runtime, "rowMeans needs data.frame or matrix"),
    }
}

fn bi_colMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::DataFrame(df) => {
            let nrow = df.nrow() as f64;
            let mut results = Vec::new();
            for (_, col) in &df.columns {
                if let Ok(vals) = e.as_reals(col) {
                    let s: f64 = vals.iter().filter_map(|x| *x).sum();
                    results.push(s / nrow);
                }
            }
            Ok(rnums(&results))
        }
        _ => err!(Runtime, "colMeans needs data.frame or matrix"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// abs() for vectors — fix to handle negative values in ifelse context
// ═══════════════════════════════════════════════════════════════════════

fn bi_Sys_sleep(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let secs = match &gv(a,0) { RVal::Numeric(v,_) => v[0].unwrap_or(0.0), _ => 0.0 };
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    Ok(RVal::Null)
}

/// `readline(prompt="")` — blocks until the user types a line on stdin
/// and presses Enter. Returns the line as a character scalar (without
/// the trailing newline). The prompt, if provided, is printed first.
/// Used for interactive prompts in scripts ("press Enter to continue",
/// "type a filename:", etc.).
fn bi_readline(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::{BufRead, Write};
    let prompt = gv(a, 0);
    let prompt_str = match &prompt {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        RVal::Null => String::new(),
        other => val_to_str(other),
    };
    if !prompt_str.is_empty() {
        print!("{}", prompt_str);
        let _ = std::io::stdout().flush();
    }
    let mut line = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut line);
    let trimmed = line.trim_end_matches(|c| c == '\n' || c == '\r').to_string();
    Ok(RVal::Character(
        vec![Some(std::sync::Arc::from(trimmed.as_str()))],
        Attrs::default(),
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// PHASE 4: ML FOUNDATION
// ═══════════════════════════════════════════════════════════════════════

// ── svd() — Singular Value Decomposition ─────────────────────────────

// Phase R.4: bi_svd moved to r2-linalg::ops. Returns full thin SVD
// (`$d`, `$u`, `$v`) via `dgesvd_full` (shipped v0.1.0).
fn bi_svd(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_svd(a)
}

// ── eigen() — Eigenvalue decomposition ───────────────────────────────

// Phase R.4: bi_eigen moved to r2-linalg::ops.
fn bi_eigen(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_base::linalg_ops::bi_eigen(a)
}

// ── prcomp() — Principal Component Analysis ──────────────────────────

// Phase R.1 step 4: bi_prcomp moved to r2-ml::dispatch.
// (bi_prcomp moved to src/builtins/ml.rs.)

// ── kmeans() — K-means clustering ────────────────────────────────────

// Phase R.1 step 4: bi_kmeans moved to r2-ml::dispatch. Per-point
// centroid assignment uses kernel::par_for(Op::PerPointDistance, ...).
// (bi_kmeans moved to src/builtins/ml.rs.)

// ── knn() — K-nearest neighbors classification ──────────────────────

// Phase R.1 step 4: bi_knn moved to r2-ml::dispatch.
// (bi_knn moved to src/builtins/ml.rs.)

// ── naive.bayes() — Naive Bayes classifier ──────────────────────────

// Phase R.1 step 4: bi_naive_bayes moved to r2-ml::dispatch.
// (bi_naive_bayes moved to src/builtins/ml.rs.)

// ── scale() — center and scale matrix columns ───────────────────────

fn bi_scale(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mat = match &gv(a,0) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, "scale() needs matrix") };
    let center = gn(a,"center").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let do_scale = gn(a,"scale").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let (m, n) = (mat.nrow, mat.ncol);
    let mut x = mat.data.clone();
    let means = mat.col_means();
    for c in 0..n {
        let col_start = c * m;
        let mean = if center { means[c] } else { 0.0 };
        let mut ss = 0.0;
        for r in 0..m { ss += (x[col_start + r] - mean).powi(2); }
        let sd = if do_scale { (ss / (m - 1).max(1) as f64).sqrt().max(1e-15) } else { 1.0 };
        for r in 0..m {
            if center { x[col_start + r] -= mean; }
            if do_scale { x[col_start + r] /= sd; }
        }
    }
    Ok(RVal::Matrix(Matrix::new(x, m, n)))
}

// ═══════════════════════════════════════════════════════════════════════
// Decision Tree (CART — Classification and Regression Tree)
// ═══════════════════════════════════════════════════════════════════════

// Phase R.1 step 1: TreeNode struct extracted to r2-ml::tree. The engine
// keeps wrapper definitions of `build_tree` / `tree_predict_one` /
// `count_splits` / `serialize_tree` that delegate to the r2-ml versions —
// this preserves callsite signatures while the actual algorithms live in
// the domain crate.
use r2_ml::tree::TreeNode;

fn build_tree(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{ r2_ml::tree::build_tree(x, y, m, n, row_mask, max_depth, min_samples, depth, is_classification) }

#[allow(dead_code)]
fn __build_tree_old(x: &[f64], y: &[f64], m: usize, n: usize, row_mask: &[bool],
    max_depth: usize, min_samples: usize, depth: usize, is_classification: bool) -> TreeNode
{
    let active: Vec<usize> = row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
    let count = active.len();

    // Compute prediction: mean for regression, majority vote for classification
    let prediction = if is_classification {
        let mut votes: HashMap<i64, usize> = HashMap::new();
        for &i in &active { *votes.entry(y[i] as i64).or_insert(0) += 1; }
        votes.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k as f64).unwrap_or(0.0)
    } else {
        active.iter().map(|&i| y[i]).sum::<f64>() / count.max(1) as f64
    };

    // Leaf conditions
    if count <= min_samples || depth >= max_depth {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Check if all y values are same
    let all_same = active.windows(2).all(|w| (y[w[0]] - y[w[1]]).abs() < 1e-10);
    if all_same {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: 0.0 };
    }

    // Find best split
    let mut best_gain = 0.0f64;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    let parent_impurity = if is_classification { gini(&active, y) } else { mse_impurity(&active, y) };

    for feat in 0..n {
        // Get sorted indices for this feature
        let mut indexed: Vec<(f64, usize)> = active.iter().map(|&i| (x[feat * m + i], i)).collect();
        indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if is_classification {
            // Incremental gini: scan sorted data, maintain left/right class counts
            // Find unique classes (small integers)
            let mut max_class = 0i64;
            for &(_, idx) in &indexed { max_class = max_class.max(y[idx] as i64); }
            let nc = (max_class + 1) as usize;
            if nc > 1000 { continue; } // safety: too many classes

            let mut right_counts = vec![0usize; nc];
            for &(_, idx) in &indexed {
                let c = y[idx] as usize;
                if c < nc { right_counts[c] += 1; }
            }
            let mut left_counts = vec![0usize; nc];
            let mut left_n = 0usize;
            let mut right_n = count;

            // Limit candidate splits to ~32 evenly spaced
            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let c = y[indexed[i].1] as usize;
                if c < nc { left_counts[c] += 1; right_counts[c] -= 1; }
                left_n += 1;
                right_n -= 1;

                // Only evaluate at step boundaries or when value changes
                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }

                last_split = i;
                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;

                // Compute gini from counts directly (no allocation)
                let left_gini = 1.0 - left_counts.iter().map(|&c| { let p = c as f64 / left_n as f64; p * p }).sum::<f64>();
                let right_gini = 1.0 - right_counts.iter().map(|&c| { let p = c as f64 / right_n as f64; p * p }).sum::<f64>();
                let weighted = (left_n as f64 * left_gini + right_n as f64 * right_gini) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        } else {
            // Regression: incremental MSE using running sums
            let mut left_sum = 0.0;
            let mut left_sq = 0.0;
            let total_sum: f64 = indexed.iter().map(|&(_, idx)| y[idx]).sum();
            let total_sq: f64 = indexed.iter().map(|&(_, idx)| y[idx] * y[idx]).sum();
            let mut left_n = 0usize;

            let step = (indexed.len() / 32).max(1);
            let mut last_split = 0;

            for i in 0..indexed.len() - 1 {
                let yi = y[indexed[i].1];
                left_sum += yi;
                left_sq += yi * yi;
                left_n += 1;
                let right_n = count - left_n;

                if i - last_split < step && i + 1 < indexed.len() - 1 { continue; }
                if (indexed[i].0 - indexed[i + 1].0).abs() < 1e-10 { continue; }
                last_split = i;

                let threshold = (indexed[i].0 + indexed[i + 1].0) / 2.0;
                let right_sum = total_sum - left_sum;

                let left_mse = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_mse = (total_sq - left_sq) / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let weighted = (left_n as f64 * left_mse + right_n as f64 * right_mse) / count as f64;
                let gain = parent_impurity - weighted;

                if gain > best_gain { best_gain = gain; best_feature = feat; best_threshold = threshold; }
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode { is_leaf: true, prediction, feature: 0, threshold: 0.0, left: None, right: None, n_samples: count, impurity: parent_impurity };
    }

    // Split
    let mut left_mask = vec![false; m];
    let mut right_mask = vec![false; m];
    for &i in &active {
        if x[best_feature * m + i] <= best_threshold { left_mask[i] = true; }
        else { right_mask[i] = true; }
    }

    let left = build_tree(x, y, m, n, &left_mask, max_depth, min_samples, depth + 1, is_classification);
    let right = build_tree(x, y, m, n, &right_mask, max_depth, min_samples, depth + 1, is_classification);

    TreeNode {
        is_leaf: false, prediction, feature: best_feature, threshold: best_threshold,
        left: Some(Box::new(left)), right: Some(Box::new(right)),
        n_samples: count, impurity: parent_impurity,
    }
}

fn gini(indices: &[usize], y: &[f64]) -> f64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &i in indices { *counts.entry(y[i] as i64).or_insert(0) += 1; }
    let n = indices.len() as f64;
    1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
}

fn mse_impurity(indices: &[usize], y: &[f64]) -> f64 {
    let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len().max(1) as f64;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / indices.len().max(1) as f64
}

// ── rpart() — Decision tree interface ────────────────────────────────

// Phase R.1 step 4: bi_rpart moved to r2-ml::dispatch. The 1-line adapter
// here exists only to satisfy r2-engine's `BuiltinFn` signature, which
// carries `&mut Engine` and `&EnvRef` for stateful builtins. Pure ML
// builtins ignore those — the adapter is FFI glue, not bloat.
// (bi_rpart moved to src/builtins/ml.rs.)

// ── rf() — Random Forest ─────────────────────────────────────────────

// Phase R.1 step 4: bi_rf moved to r2-ml::dispatch. Uses kernel::par_for
// instead of par_iter — Rayon stays below the kernel layer (§4.9).
// (bi_rf moved to src/builtins/ml.rs.)

// ═══════════════════════════════════════════════════════════════════════
// PHASE: DATA HANDLING — filter, select, mutate, arrange, regex, etc.
// ═══════════════════════════════════════════════════════════════════════

// ── sub() / regexpr basics ───────────────────────────────────────────

// (bi_sub moved to src/builtins/strings.rs.)

// (bi_grepl moved to src/builtins/strings.rs.)

// (bi_regexpr moved to src/builtins/strings.rs.)

// ── duplicated() / distinct values ───────────────────────────────────

// (bi_duplicated moved to src/builtins/data.rs.)

// ── order() — return indices that would sort the vector ──────────────

// (bi_order moved to src/builtins/data.rs.)

// ── rank() — ranks of values ─────────────────────────────────────────

// (bi_rank moved to src/builtins/data.rs.)

// ── cummax, cummin ───────────────────────────────────────────────────

fn bi_cummax(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummax(a) }

fn bi_cummin(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_cummin(a) }

// ── which() improvements — named results ────────────────────────────

// (which already exists, but let's add which.min/max for data.frame columns)

// ── Improved read.csv — handles quotes, various delimiters, type inference ──

fn bi_read_csv_v2(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a,0) {
        RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).ok_or(R2Err{msg:"NA path".into(),kind:ErrKind::Runtime})?,
        _ => return err!(Runtime, "read.csv needs path"),
    };
    let header = gn(a,"header").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(true);
    let sep = gn(a,"sep").and_then(|v| match v { RVal::Character(s,_) => s[0].as_ref().map(|s| s.to_string()), _ => None }).unwrap_or(",".into());
    let na_strings = vec!["NA", "na", "N/A", "n/a", "", ".", "NULL", "null", "None", "none"];

    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot read '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let mut lines = content.lines();

    // Parse header
    let col_names: Vec<String> = if header {
        lines.next().map(|l| parse_csv_line(l, &sep)).unwrap_or_default()
    } else { Vec::new() };

    // Read all rows
    let mut raw_rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        raw_rows.push(parse_csv_line(line, &sep));
    }

    if raw_rows.is_empty() { return err!(Runtime, "empty CSV file"); }

    let ncol = col_names.len().max(raw_rows.iter().map(|r| r.len()).max().unwrap_or(0));
    let nrow = raw_rows.len();

    // Build columns with type inference
    let mut columns = Vec::new();
    for c in 0..ncol {
        let name = if c < col_names.len() { Arc::from(col_names[c].as_str()) } else { Arc::from(format!("V{}", c+1).as_str()) };

        let col_vals: Vec<String> = raw_rows.iter().map(|r| r.get(c).cloned().unwrap_or_default()).collect();

        // Type inference: try integer → numeric → character
        let all_int = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<i32>().is_ok());
        let all_num = col_vals.iter().all(|s| na_strings.contains(&s.as_str()) || s.parse::<f64>().is_ok());
        let has_num = col_vals.iter().any(|s| s.parse::<f64>().is_ok());

        if all_int && has_num {
            let vals: Vec<Integer> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Integer(vals.into(), Attrs::default())));
        } else if all_num && has_num {
            let vals: Vec<Real> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { s.parse().ok() }
            }).collect();
            columns.push((name, RVal::Numeric(vals.into(), Attrs::default())));
        } else {
            let vals: Vec<Character> = col_vals.iter().map(|s| {
                if na_strings.contains(&s.as_str()) { None } else { Some(Arc::from(s.as_str())) }
            }).collect();
            columns.push((name, RVal::Character(vals, Attrs::default())));
        }
    }

    println!("Read {} rows × {} columns from '{}'", nrow, ncol, path);
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

/// Parse a CSV line handling quoted fields
fn parse_csv_line(line: &str, sep: &str) -> Vec<String> {
    let sep_char = sep.chars().next().unwrap_or(',');
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"'); // escaped quote
                    chars.next();
                } else {
                    in_quotes = false; // end quote
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == sep_char {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// ── DataFrame pipe-friendly operations: filter, select, mutate, arrange ──

// Phase R.2: bi_filter moved to r2-data::dplyr.
fn bi_filter(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_filter(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "filter() needs data.frame") };
    let e = _e;
    let mask = e.as_logicals(&gv(a,1))?;
    let keep: Vec<usize> = mask.iter().enumerate().filter(|(_, m)| **m == Some(true)).map(|(i, _)| i).collect();
    let nrow = df.nrow();

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(keep.iter().map(|&r| if r < v.len() { v[r].clone() } else { None }).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(keep.iter().map(|&r| if r < v.len() { v[r] } else { None }).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_select moved to r2-data::dplyr.
fn bi_select(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_select(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "select() needs data.frame") };

    // Collect column names from remaining args
    let mut col_names: Vec<String> = Vec::new();
    for i in 1..10 {
        match &gv(a, i) {
            RVal::Character(v, _) => {
                for c in v { if let Some(s) = c { col_names.push(s.to_string()); } }
            }
            RVal::Null => break,
            _ => break,
        }
    }

    if col_names.is_empty() { return Ok(RVal::DataFrame(df)); }

    let columns: Vec<(Arc<str>, RVal)> = col_names.iter().filter_map(|name| {
        df.columns.iter().find(|(n, _)| n.as_ref() == name.as_str()).cloned()
    }).collect();

    if columns.is_empty() { return err!(Runtime, "select: no matching columns found"); }
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// Phase R.2: bi_arrange moved to r2-data::dplyr.
fn bi_arrange(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    return r2_data::dplyr::bi_arrange(a);
    #[allow(unreachable_code)]
    {
    let df = match &gv(a,0) { RVal::DataFrame(df) => df.clone(), _ => return err!(Runtime, "arrange() needs data.frame") };
    let e = _e;
    let sort_vals = e.as_reals(&gv(a,1))?;
    let decreasing = gn(a,"decreasing").and_then(|v| e.as_logicals(&v).ok()).map(|v| v[0] == Some(true)).unwrap_or(false);

    let nrow = df.nrow();
    let mut indices: Vec<usize> = (0..nrow).collect();
    indices.sort_by(|&a, &b| {
        let va = sort_vals.get(a).and_then(|x| *x).unwrap_or(f64::NAN);
        let vb = sort_vals.get(b).and_then(|x| *x).unwrap_or(f64::NAN);
        if decreasing { vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal) }
        else { va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal) }
    });

    let columns: Vec<(Arc<str>, RVal)> = df.columns.iter().map(|(name, col)| {
        let new_col = match col {
            RVal::Numeric(v, _) => RVal::Numeric(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Integer(v, _) => RVal::Integer(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Character(v, _) => RVal::Character(indices.iter().map(|&r| v.get(r).cloned().unwrap_or(None)).collect(), Attrs::default()),
            RVal::Logical(v, _) => RVal::Logical(indices.iter().map(|&r| v.get(r).copied().unwrap_or(None)).collect(), Attrs::default()),
            _ => col.clone(),
        };
        (name.clone(), new_col)
    }).collect();

    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
    } // end of #[allow(unreachable_code)] block (Phase R.2)
}

// ── Sys.getenv() — read environment variable ─────────────────────────

fn bi_sys_getenv(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a,0));
    let val = std::env::var(&name).unwrap_or_default();
    Ok(rstr(&val))
}

// ── file.exists() — check if file exists ─────────────────────────────

// (bi_file_exists moved to src/builtins/io.rs.)

// ── list.files() — list files in directory ───────────────────────────

// (bi_list_files moved to src/builtins/io.rs.)

// end of file

// ═══════════════════════════════════════════════════════════════════════
// Gradient Boosted Trees (XGBoost-style)
// ═══════════════════════════════════════════════════════════════════════
//
// Algorithm:
//   1. Initialize F₀ = mean(y) for regression, log(p/(1-p)) for classification
//   2. For each iteration t = 1..T:
//      a. Compute pseudo-residuals: rᵢ = -∂L/∂F(xᵢ)
//      b. Fit a regression tree to pseudo-residuals
//      c. Update: F_t(x) = F_{t-1}(x) + η · tree_t(x)
//   3. Final prediction: F_T(x)

// Phase R.1 step 4: bi_gbm moved to r2-ml::dispatch. Per-iteration row work
// uses kernel::par_for; outer boosting loop stays sequential by algorithm.
// (bi_gbm moved to src/builtins/ml.rs.)

// ═══════════════════════════════════════════════════════════════════════
// save() / load() — Session persistence
// ═══════════════════════════════════════════════════════════════════════

fn bi_save(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // save("file.r2s")          — save all session variables
    // save(object, "file.r2d")  — save single data object
    // save(model, "file.r2m")   — save model object
    let first = gv(a, 0);

    // Check if first arg is a string (session save) or an object (object save)
    let (obj_to_save, path) = match &first {
        RVal::Character(_, _) => {
            // save("session.r2s") — save all variables
            let path = val_to_str(&first);
            (None, path)
        }
        _ => {
            // save(object, "file.r2d") — save single object
            let path = gn(a, "file").or(Some(gv(a, 1))).map(|v| val_to_str(&v))
                .unwrap_or("object.r2d".into());
            (Some(first.clone()), path)
        }
    };

    let mut out = String::new();

    // Header with format version
    out.push_str("#R2 v0.1.1\n");

    if let Some(obj) = obj_to_save {
        // Single object save
        let serialized = serialize_rval(&obj);
        if serialized.is_empty() {
            return err!(Runtime, "cannot serialize {} objects", obj.type_name());
        }
        out.push_str(&format!("_obj={}\n", serialized));
        std::fs::write(&path, &out).map_err(|e| R2Err{msg:format!("cannot save to '{}': {}", path, e),kind:ErrKind::Runtime})?;
        let ext = path.rsplit('.').next().unwrap_or("");
        let kind = match ext { "r2m" => "model", "r2d" => "data", _ => "object" };
        println!("Saved {} ({}) to '{}'", kind, obj.type_name(), path);
    } else {
        // Session save — all variables
        let mut count = 0;
        for (name, val) in &e.global_env.bindings {
            if matches!(name.as_ref(), "iris" | "mtcars" | "airquality") { continue; }
            let serialized = serialize_rval(val);
            if !serialized.is_empty() {
                out.push_str(&format!("{}={}\n", name, serialized));
                count += 1;
            }
        }
        std::fs::write(&path, &out).map_err(|e| R2Err{msg:format!("cannot save to '{}': {}", path, e),kind:ErrKind::Runtime})?;
        println!("Saved {} objects to '{}'", count, path);
    }
    Ok(RVal::Null)
}

fn bi_load(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = gn(a,"file").or(Some(gv(a,0))).map(|v| val_to_str(&v)).unwrap_or("session.r2s".into());
    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot load '{}': {}", path, e),kind:ErrKind::Runtime})?;

    let ext = path.rsplit('.').next().unwrap_or("");
    let mut count = 0;

    for line in content.lines() {
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some(eq_pos) = line.find('=') {
            let name = &line[..eq_pos];
            let val_str = &line[eq_pos+1..];
            if let Some(val) = deserialize_rval(val_str) {
                if name == "_obj" {
                    // Single-object file — return immediately with the value.
                    let kind = match ext { "r2m" => "model", "r2d" => "data", _ => "object" };
                    println!("Loaded {} ({}) from '{}'", kind, val.type_name(), path);
                    return Ok(val);
                }
                env_insert(&mut e.global_env, Arc::from(name), val);
                count += 1;
            }
        }
    }
    println!("Loaded {} objects from '{}'", count, path);
    Ok(RVal::Null)
}

fn serialize_rval(val: &RVal) -> String {
    match val {
        RVal::Numeric(v, _) => {
            let nums: Vec<String> = v.iter().map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect();
            format!("N:{}", nums.join(","))
        }
        RVal::Integer(v, _) => {
            let nums: Vec<String> = v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect();
            format!("I:{}", nums.join(","))
        }
        RVal::Character(v, _) => {
            let strs: Vec<String> = v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect();
            format!("C:{}", strs.join("\t"))
        }
        RVal::Logical(v, _) => {
            let vals: Vec<String> = v.iter().map(|x| match x { Some(true) => "T".into(), Some(false) => "F".into(), None => "NA".into() }).collect();
            format!("L:{}", vals.join(","))
        }
        RVal::DataFrame(df) => {
            // Serialize DataFrame: D:ncol\tcol1_name\ttype:data\tcol2_name\ttype:data...
            let mut parts = vec![format!("{}", df.columns.len())];
            for (name, col) in &df.columns {
                let col_ser = serialize_rval(col);
                parts.push(format!("{}:{}", name, col_ser));
            }
            format!("D:{}", parts.join("\x1f")) // unit separator
        }
        RVal::Matrix(m) => {
            let nums: Vec<String> = m.data.iter().map(|n| fmt_num(*n)).collect();
            format!("M:{}:{}:{}", m.nrow, m.ncol, nums.join(","))
        }
        RVal::TypeInstance(inst) => {
            // Serialize model: T:classname\x1ffield1=ser\x1ffield2=ser...
            let mut parts = vec![inst.type_name.to_string()];
            for (k, v) in &inst.fields {
                let v_ser = serialize_rval(v);
                if !v_ser.is_empty() {
                    parts.push(format!("{}={}", k, v_ser));
                }
            }
            format!("T:{}", parts.join("\x1f"))
        }
        _ => String::new(),
    }
}

fn deserialize_rval(s: &str) -> Option<RVal> {
    if s.len() < 2 { return None; }
    let (typ, data) = (s.as_bytes()[0] as char, &s[2..]);
    match typ {
        'N' => {
            let vals: Vec<Real> = data.split(',').map(|s| if s == "NA" { None } else { s.parse().ok() }).collect();
            Some(RVal::Numeric(vals.into(), Attrs::default()))
        }
        'I' => {
            let vals: Vec<Integer> = data.split(',').map(|s| if s == "NA" { None } else { s.parse().ok() }).collect();
            Some(RVal::Integer(vals.into(), Attrs::default()))
        }
        'C' => {
            let vals: Vec<Character> = data.split('\t').map(|s| if s == "NA" { None } else { Some(Arc::from(s)) }).collect();
            Some(RVal::Character(vals, Attrs::default()))
        }
        'L' => {
            let vals: Vec<Logical> = data.split(',').map(|s| match s { "T" => Some(true), "F" => Some(false), _ => None }).collect();
            Some(RVal::Logical(vals.into(), Attrs::default()))
        }
        'M' => {
            // Matrix: M:nrow:ncol:data
            let parts: Vec<&str> = data.splitn(3, ':').collect();
            if parts.len() != 3 { return None; }
            let nrow: usize = parts[0].parse().ok()?;
            let ncol: usize = parts[1].parse().ok()?;
            let vals: Vec<f64> = parts[2].split(',').filter_map(|s| s.parse().ok()).collect();
            Some(RVal::Matrix(Matrix::new(vals, nrow, ncol)))
        }
        'D' => {
            // DataFrame: D:ncol\x1fcol_name:type:data...
            let parts: Vec<&str> = data.split('\x1f').collect();
            if parts.is_empty() { return None; }
            let mut columns = Vec::new();
            for part in &parts[1..] {
                if let Some(colon) = part.find(':') {
                    let col_name = &part[..colon];
                    let col_data = &part[colon+1..];
                    if let Some(val) = deserialize_rval(col_data) {
                        columns.push((Arc::from(col_name), val));
                    }
                }
            }
            Some(RVal::DataFrame(DataFrame { columns, row_names: None }))
        }
        'T' => {
            // TypeInstance: T:classname\x1ffield=val...
            let parts: Vec<&str> = data.split('\x1f').collect();
            if parts.is_empty() { return None; }
            let type_name = Arc::from(parts[0]);
            let mut fields = HashMap::new();
            for part in &parts[1..] {
                if let Some(eq) = part.find('=') {
                    let key = Arc::from(&part[..eq]);
                    let val_str = &part[eq+1..];
                    if let Some(val) = deserialize_rval(val_str) {
                        fields.insert(key, val);
                    }
                }
            }
            Some(RVal::TypeInstance(TypeInstance { type_name, fields }))
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// cv() — Cross-validation helper
// ═══════════════════════════════════════════════════════════════════════

// Phase R.1 step 4: bi_cv moved to r2-ml::dispatch. Folds run via
// kernel::par_for(Op::KFoldCV, k, ...).
// (bi_cv moved to src/builtins/ml.rs.)

// ═══════════════════════════════════════════════════════════════════════
// confusion.matrix() — for classification evaluation
// ═══════════════════════════════════════════════════════════════════════

fn bi_confusion_matrix(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // confusion.matrix(predicted, actual) or confusion.matrix(model)
    let pred: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|x| x).collect();
    let actual: Vec<f64> = e.as_reals(&gv(a,1))?.into_iter().filter_map(|x| x).collect();

    if pred.len() != actual.len() { return err!(Runtime, "confusion.matrix: lengths must match"); }

    // Find unique classes
    let mut classes: Vec<i64> = Vec::new();
    for v in pred.iter().chain(actual.iter()) {
        let c = *v as i64;
        if !classes.contains(&c) { classes.push(c); }
    }
    classes.sort();
    let k = classes.len();

    // Build confusion matrix
    let mut cm = vec![0i32; k * k];
    for i in 0..pred.len() {
        let pi = classes.iter().position(|&c| c == pred[i] as i64).unwrap_or(0);
        let ai = classes.iter().position(|&c| c == actual[i] as i64).unwrap_or(0);
        cm[ai * k + pi] += 1; // row = actual, col = predicted
    }

    // Print
    println!("\nConfusion Matrix:");
    print!("{:>12}", "Predicted→");
    for c in &classes { print!("{:>8}", c); }
    println!("{:>10}", "Total");
    

    let n = pred.len();
    let mut correct = 0;
    for (ai, ac) in classes.iter().enumerate() {
        print!("Actual {:>4} ", ac);
        let mut row_total = 0;
        for pi in 0..k {
            print!("{:>8}", cm[ai * k + pi]);
            row_total += cm[ai * k + pi];
            if ai == pi { correct += cm[ai * k + pi]; }
        }
        println!("{:>10}", row_total);
    }

    
    let accuracy = correct as f64 / n as f64;
    println!("Accuracy: {}/{} ({}%)", correct, n, fmt_num(accuracy * 100.0));

    // Per-class precision and recall
    println!("\n{:>8} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1");
    for (ci, c) in classes.iter().enumerate() {
        let tp = cm[ci * k + ci] as f64;
        let pred_total: f64 = (0..k).map(|ai| cm[ai * k + ci] as f64).sum();
        let actual_total: f64 = (0..k).map(|pi| cm[ci * k + pi] as f64).sum();
        let precision = if pred_total > 0.0 { tp / pred_total } else { 0.0 };
        let recall = if actual_total > 0.0 { tp / actual_total } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        println!("{:>8} {:>10} {:>10} {:>10}", c, fmt_num(precision), fmt_num(recall), fmt_num(f1));
    }

    let mut fields = HashMap::new();
    fields.insert(Arc::from("accuracy"), rnum(accuracy));
    fields.insert(Arc::from("matrix"), RVal::Matrix(Matrix::new(cm.iter().map(|&x| x as f64).collect(), k, k)));
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("confusion"), fields }))
}

// ═══════════════════════════════════════════════════════════════════════
// mutate() — add/modify DataFrame columns
// ═══════════════════════════════════════════════════════════════════════

// Phase R.2: bi_mutate moved to r2-data::dplyr.
fn bi_mutate(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::dplyr::bi_mutate(a)
}

// ═══════════════════════════════════════════════════════════════════════
// version() — show R2 version info
// ═══════════════════════════════════════════════════════════════════════

fn bi_version(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    println!("\nR2 — Statistical Computing, Reimagined");
    println!("Version: 0.1.1");
    println!("Created by: Devendra Tandale");
    println!("An AI assisted project");
    println!("Platform: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
    println!("Kernel: r2-linalg (pure Rust, no C/C++ dependencies)");
    println!("ML algorithms: 12 built-in");
    println!("Parallel cores: {}", rayon::current_num_threads());
    println!("Functions: 191+");
    println!("Codebase: 9,800+ lines of Rust");
    println!("License: AGPL v3");
    println!();
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// clear() / cls() — clear the terminal screen
// ═══════════════════════════════════════════════════════════════════════

fn bi_clear(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::Write;
    // ANSI escape: \x1b[2J clears the visible region; \x1b[3J clears the scrollback
    // (supported by Windows Terminal, modern conhost, and all *nix terminals).
    // \x1b[H homes the cursor.
    print!("\x1b[3J\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// aov() / anova() — Analysis of Variance
// ═══════════════════════════════════════════════════════════════════════

fn bi_aov(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_aov(a) }

fn bi_anova(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_anova(a) }

// ═══════════════════════════════════════════════════════════════════════
// Additional Statistical Tests
// ═══════════════════════════════════════════════════════════════════════

// ── cor.test() — test if correlation is significant ──────────────────

fn bi_cor_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_cor_test(a) }

// ── shapiro.test() — test for normality ──────────────────────────────

fn bi_shapiro_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_shapiro_test(a) }

// ── wilcox.test() — Wilcoxon rank-sum / signed-rank test ─────────────

fn bi_wilcox_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_wilcox_test(a) }

// ── fisher.test() — Fisher's exact test for 2×2 tables ──────────────

fn bi_fisher_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_fisher_test(a) }

// ── weighted.mean() ──────────────────────────────────────────────────

fn bi_weighted_mean(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|v| v).collect();
    let w: Vec<f64> = gn(a, "w").or(Some(gv(a, 1)))
        .and_then(|v| e.as_reals(&v).ok())
        .unwrap_or(vec![Some(1.0); x.len()])
        .into_iter().filter_map(|v| v).collect();
    let n = x.len().min(w.len());
    let sum_w: f64 = w[..n].iter().sum();
    let wm: f64 = x[..n].iter().zip(w[..n].iter()).map(|(x, w)| x * w).sum::<f64>() / sum_w;
    Ok(rnum(wm))
}

// ── IQR() — interquartile range ──────────────────────────────────────

fn bi_iqr(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let mut x: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|v| v).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len();
    if n < 2 { return err!(Runtime, "IQR needs at least 2 values"); }
    let q1 = x[n / 4];
    let q3 = x[3 * n / 4];
    Ok(rnum(q3 - q1))
}

// ═══════════════════════════════════════════════════════════════════════
// .Internal() — Bridge from R2 scripts to Rust primitives
// ═══════════════════════════════════════════════════════════════════════
//
// This enables R2-language functions to call Rust-implemented math.
// Example: .Internal("solve_lstsq", x_matrix, y_vector)
//
// Users write statistics in R2 syntax.
// Only heavy math runs in Rust via .Internal().

fn bi_internal(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a, 0));

    match name.as_str() {
        // Matrix operations
        "matmul" => {
            let a_mat = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal matmul: need matrix") };
            let b_mat = match &gv(a,2) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal matmul: need matrix") };
            Ok(RVal::Matrix(a_mat.matmul(&b_mat).map_err(|e| R2Err{msg:e,kind:ErrKind::Runtime})?))
        }
        "crossprod" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal crossprod: need matrix") };
            Ok(RVal::Matrix(m.crossprod()))
        }
        "crossprod_vec" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal crossprod_vec: need matrix") };
            let v: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = m.crossprod_vec(&v);
            Ok(rnums(&result))
        }
        // Linear algebra
        "solve" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal solve: need matrix") };
            let b: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = m.solve(&b).map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "solve_lstsq" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal solve_lstsq: need matrix") };
            let y: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = r2_linalg::dlsq_fused(m.nrow, m.ncol, &m.data, &y)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "inverse" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal inverse: need matrix") };
            let result = r2_linalg::dgetri(m.nrow, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(RVal::Matrix(Matrix::new(result, m.nrow, m.ncol)))
        }
        "cholesky" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal cholesky: need matrix") };
            let mut data = m.data.clone();
            r2_linalg::dpotrf(m.nrow, &mut data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(RVal::Matrix(Matrix::new(data, m.nrow, m.ncol)))
        }
        "eigenvalues" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal eigenvalues: need matrix") };
            let result = r2_linalg::dsyev(m.nrow, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "svd" => {
            // Full thin SVD: A = U · diag(d) · Vᵀ.
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal svd: need matrix") };
            let (sigma, u_data, vt_data) = r2_linalg::dgesvd_full(m.nrow, m.ncol, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            let n = m.ncol;
            // Transpose Vᵀ → V (R convention: $v holds V, not Vᵀ).
            let mut v_data = vec![0.0_f64; n * n];
            for i in 0..n { for j in 0..n { v_data[j * n + i] = vt_data[i * n + j]; } }
            let mut fields = HashMap::new();
            fields.insert(Arc::from("d"), rnums(&sigma));
            fields.insert(Arc::from("u"), RVal::Matrix(Matrix::new(u_data, m.nrow, n)));
            fields.insert(Arc::from("v"), RVal::Matrix(Matrix::new(v_data, n, n)));
            Ok(RVal::List(fields.into_iter().map(|(k,v)| (Some(k), v)).collect()))
        }
        // Random numbers
        "rnorm_vec" => {
            let n = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0) as usize;
            let mu = e.scalar_f64(&gv(a,2))?.unwrap_or(0.0);
            let sigma = e.scalar_f64(&gv(a,3))?.unwrap_or(1.0);
            let vals: Vec<Real> = (0..n).map(|_| {
                let u1 = r2_next_random().max(1e-15);
                let u2 = r2_next_random();
                Some(mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos())
            }).collect();
            Ok(RVal::Numeric(vals.into(), Attrs::default()))
        }
        // Phi (normal CDF) for p-values
        "pnorm" => {
            let x = e.scalar_f64(&gv(a,1))?.unwrap_or(0.0);
            Ok(rnum(phi(x)))
        }
        "qnorm" => {
            let p = e.scalar_f64(&gv(a,1))?.unwrap_or(0.5);
            Ok(rnum(qnorm_approx(p)))
        }

        _ => err!(Runtime, ".Internal: unknown function '{}'", name),
    }
}
