//! Single source of truth for the built-in package registration tables.
//!
//! These tables used to be duplicated verbatim: once in `Engine::new()`
//! (startup registration) and once in `try_reload_base()` (re-registering
//! a `base`/`stats`/`graphics`/`utils` layer after `detach()` +
//! `library()`). The two copies had already drifted — the reload copy of
//! `base` was missing ~16 functions (`log2`/`log10`/`log1p`/`expm1`, all
//! the trig functions, `sign`/`trunc`) — so a detach+reload silently
//! dropped them.
//!
//! Both call sites now read from the functions here, so:
//!   * a new built-in is registered in **one** place, and
//!   * startup and reload can never diverge again.
//!
//! `use super::*` pulls the `bi_*` functions (private to the crate root)
//! into scope — a child module may name its parent's private items.

use super::*;

/// CORE tier — immutable, cannot be masked or detached. Not reloadable,
/// so this one is only used by `Engine::new()`.
pub(crate) fn core_table() -> Vec<(&'static str, BuiltinFn)> {
    vec![
        ("c",bi_c),("length",bi_length),("print",bi_print),("cat",bi_cat),
        ("clear",bi_clear),("cls",bi_clear),("clr",bi_clear),
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
    ]
}

/// BASE tier — can be masked by addons, can be detached + reloaded.
pub(crate) fn base_table() -> Vec<(&'static str, BuiltinFn)> {
    vec![
        ("seq",bi_seq),("rep",bi_rep),("paste",bi_paste),("paste0",bi_paste0),
        ("which",bi_which),("sort",bi_sort),("rev",bi_rev),("unique",bi_unique),
        ("seq_len",bi_seq_len),("seq_along",bi_seq_along),("unlist",bi_unlist),("setNames",bi_set_names),("append",bi_append),("pmin",bi_pmin),("pmax",bi_pmax),("setdiff",bi_setdiff),("union",bi_union),("intersect",bi_intersect),("invisible",bi_invisible),("inherits",bi_inherits),("cut",bi_cut),("signif",bi_signif),("Reduce",bi_reduce),("Filter",bi_filter_fp),("Map",bi_map),("split",bi_split),("stopifnot",bi_stopifnot),("outer",bi_outer),("attr",bi_attr),("attributes",bi_attributes),("structure",bi_structure),("format",bi_format),("%in%",bi_in),("factorial",bi_factorial),("choose",bi_choose),("gamma",bi_gamma),("lgamma",bi_lgamma),("beta",bi_beta),("combn",bi_combn),("mad",bi_mad),("fivenum",bi_fivenum),("numeric",bi_numeric),("integer",bi_integer),("character",bi_character),("logical",bi_logical),("as.matrix",bi_as_matrix),("as.vector",bi_as_vector),("as.list",bi_as_list),("is.function",bi_is_function),("is.list",bi_is_list),("is.vector",bi_is_vector),("is.element",bi_in),("substring",bi_substring),("readLines",bi_read_lines),("writeLines",bi_write_lines),("uniroot",bi_uniroot),("integrate",bi_integrate),("optimize",bi_optimize),("match.arg",bi_match_arg),("ave",bi_ave),("nargs",bi_nargs),
        // Phase L.1 — first-class language objects (quote is an NSE special
        // form in eval_in, not registered here).
        ("eval",bi_eval),("parse",bi_parse),("deparse",bi_deparse),("call",bi_call),("as.call",bi_as_call),
        // Phase L.2 — function introspection (read-only).
        ("body",bi_body),("formals",bi_formals),("args",bi_args),
        // Common base predicates/comparison.
        ("isTRUE",bi_is_true),("isFALSE",bi_is_false),("identical",bi_identical),("all.equal",bi_all_equal),("diag",bi_diag),("toString",bi_to_string),
        // Replacement functions: `names(x)<-`, `colnames(x)<-`, `rownames(x)<-`.
        ("names<-",bi_names_set),("colnames<-",bi_colnames_set),("rownames<-",bi_rownames_set),
        ("gregexpr",bi_gregexpr),("regmatches",bi_regmatches),
        // Operators as functions (for Reduce/Map/do.call).
        ("+",bi_op_add),("-",bi_op_sub),("*",bi_op_mul),("/",bi_op_div),("^",bi_op_pow),("%%",bi_op_mod),
        ("==",bi_op_eq),("!=",bi_op_ne),("<",bi_op_lt),(">",bi_op_gt),("<=",bi_op_le),(">=",bi_op_ge),
        // Phase L.3 — NSE call introspection (substitute/bquote are NSE
        // special forms in eval_in, not registered here).
        ("match.call",bi_match_call),("sys.call",bi_sys_call),
        ("abs",bi_abs),("sqrt",bi_sqrt),("round",bi_round),("max",bi_max),("min",bi_min),
        ("nchar",bi_nchar),("toupper",bi_toupper),("tolower",bi_tolower),
        ("substr",bi_substr),("grep",bi_grep),("gsub",bi_gsub),("strsplit",bi_strsplit),
        ("sub",bi_sub),("grepl",bi_grepl),("regexpr",bi_regexpr),
        ("duplicated",bi_duplicated),("order",bi_order),("rank",bi_rank),
        ("cummax",bi_cummax),("cummin",bi_cummin),
        ("filter",bi_filter),("select",bi_select),("arrange",bi_arrange),("mutate",bi_mutate),
        ("factor",bi_factor),("names",bi_names),("nrow",bi_nrow),("ncol",bi_ncol),
        ("table",bi_table),("sapply",bi_sapply),("lapply",bi_lapply),("mapply",bi_mapply),("vapply",bi_vapply),("mclapply",bi_mclapply),("par.lapply",bi_mclapply),
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
        ("format.POSIXct",bi_format_time),("strftime",bi_format_time),("Sys.Date",bi_sys_date),("Sys.time",bi_sys_time),
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
    ]
}

/// STATS tier — distributions, models, ML, linear algebra.
pub(crate) fn stats_table() -> Vec<(&'static str, BuiltinFn)> {
    vec![
        ("sum",bi_sum),("mean",bi_mean),("sd",bi_sd),("var",bi_var),("cor",bi_cor),("cov",bi_cov),
        ("lm",bi_lm),("summary",bi_summary),("plssem",bi_plssem),("csem",bi_plssem),
        ("rnorm",bi_rnorm),("dnorm",bi_dnorm),("runif",bi_runif),("sample",bi_sample),
        ("dexp",bi_dexp),("pexp",bi_pexp),("qexp",bi_qexp),("dbinom",bi_dbinom),("pbinom",bi_pbinom),("dpois",bi_dpois),("ppois",bi_ppois),("dt",bi_dt),("pt",bi_pt),("dchisq",bi_dchisq),("pchisq",bi_pchisq),("pf",bi_pf),("rexp",bi_rexp),("qt",bi_qt),("qchisq",bi_qchisq),("qf",bi_qf),("qbinom",bi_qbinom),("qpois",bi_qpois),("density",bi_density),
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
        ("svd",bi_svd),("eigen",bi_eigen),("prcomp",bi_prcomp),("solve",bi_solve),("det",bi_det),("mmap.write",bi_mmap_write),("mmap.col",bi_mmap_col),("mmap.map",bi_mmap_map),("mmap.csv",bi_mmap_csv),("mmap.lm",bi_mmap_lm),("read.parquet",bi_read_parquet),
        ("kmeans",bi_kmeans),("knn",bi_knn),("naive.bayes",bi_naive_bayes),("scale",bi_scale),
        ("rpart",bi_rpart),("rf",bi_rf),("gbm",bi_gbm),("cv",bi_cv),("aov",bi_aov),("anova",bi_anova),("cor.test",bi_cor_test),("shapiro.test",bi_shapiro_test),("wilcox.test",bi_wilcox_test),("fisher.test",bi_fisher_test),("weighted.mean",bi_weighted_mean),("IQR",bi_iqr),("confusion.matrix",bi_confusion_matrix),
    ]
}

/// GRAPHICS tier — high-level plots, overlays, devices, color helpers.
pub(crate) fn graphics_table() -> Vec<(&'static str, BuiltinFn)> {
    vec![
        ("plot",bi_plot),("hist",bi_hist),("boxplot",bi_boxplot),("barplot",bi_barplot),("pairs",bi_pairs),("pie",bi_pie),("matplot",bi_matplot),("plot.new",bi_plot_new),("plot.window",bi_plot_window),
        ("save.plot",bi_save_plot),
        ("lines",bi_lines),("points",bi_points),("abline",bi_abline),("legend",bi_legend),("text",bi_text),("title",bi_title),("axis",bi_axis),("rect",bi_rect),
        ("par",bi_par),("dev.off",bi_dev_off),("save_plot",bi_save_plot),("dev.view",bi_dev_view),("pdf",bi_pdf),("png",bi_png),("svg",bi_svg),
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
    ]
}

/// UTILS tier — I/O, inspection, session helpers.
pub(crate) fn utils_table() -> Vec<(&'static str, BuiltinFn)> {
    vec![
        ("head",bi_head),("tail",bi_tail),("str",bi_str),
        ("read.csv",bi_read_csv_v2),("write.csv",bi_write_csv),
        ("search",bi_search),("t",bi_transpose),("crossprod",bi_crossprod),
        ("source",bi_source),("system.time",bi_system_time),
        ("read.table",bi_read_table),("write.table",bi_write_table),("read.delim",bi_read_delim),
        ("Sys.time",bi_Sys_time),("help",bi_help),("getwd",bi_getwd),("setwd",bi_setwd),
        ("file.exists",bi_file_exists),("list.files",bi_list_files),("Sys.getenv",bi_sys_getenv),("save",bi_save),("load",bi_load),("version",bi_version),("clear",bi_clear),("cls",bi_clear),(".Internal",bi_internal),
    ]
}
