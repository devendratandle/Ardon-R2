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
use builtins::core::*;
use builtins::data_apply::*;
use builtins::sys_models::*;
use builtins::ml_data::*;
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
    pub(crate) warnings: Vec<String>,
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

#[macro_export] macro_rules! err { ($k:ident, $($a:tt)*) => { Err(R2Err { msg: format!($($a)*), kind: ErrKind::$k }) }; }

pub(crate) fn gv(args: &[EvalArg], i: usize) -> RVal { args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null) }
pub(crate) fn gn(args: &[EvalArg], name: &str) -> Option<RVal> { args.iter().find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|a| a.value.clone()) }

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

    pub(crate) fn call_fn(&mut self, func: &RVal, args: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
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
    pub(crate) fn scalar_f64(&self, obj: &RVal) -> Result<Real, R2Err> { obj.scalar_f64() }
    pub(crate) fn truthy(&self, obj: &RVal) -> Result<bool, R2Err> { match obj { RVal::Logical(v,_) => v.first().copied().flatten().ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), RVal::Numeric(v,_) => v.first().copied().flatten().map(|n| n!=0.0).ok_or(R2Err{msg:"NA where TRUE/FALSE needed".into(),kind:ErrKind::Runtime}), _ => err!(Type,"cannot coerce {} to logical",obj.type_name()) } }
    fn vals_eq(&self, a: &RVal, b: &RVal) -> bool { match (a,b) { (RVal::Numeric(a,_),RVal::Numeric(b,_)) => a==b, (RVal::Character(a,_),RVal::Character(b,_)) => a==b, (RVal::Integer(a,_),RVal::Integer(b,_)) => a==b, _ => false } }
    pub(crate) fn to_items(&self, obj: &RVal) -> Result<Vec<RVal>, R2Err> { match obj { RVal::Integer(v,_) => Ok(v.iter().map(|x| RVal::Integer(vec![*x].into(),Attrs::default())).collect()), RVal::Numeric(v,_) => Ok(v.iter().map(|x| RVal::Numeric(vec![*x].into(),Attrs::default())).collect()), RVal::Character(v,_) => Ok(v.iter().map(|x| RVal::Character(vec![x.clone()],Attrs::default())).collect()), RVal::List(v) => Ok(v.iter().map(|(_,val)| val.clone()).collect()), _ => err!(Runtime,"cannot iterate over {}",obj.type_name()) } }
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

// DATA MANIPULATION + NA + APPLY + MORE MATH moved to builtins/data_apply.rs.

// ═══════════════════════════════════════════════════════════════════════
// MORE DISTRIBUTIONS: pnorm, qnorm, rbinom, rpois, dbinom
// ═══════════════════════════════════════════════════════════════════════





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

// ML FOUNDATION + DATA HANDLING moved to builtins/ml_data.rs.
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
