//! Miscellaneous builtins (trailing lib.rs block) — extracted in the
//! opus-4.8 engine-split session, content-anchored.
//!
//! Covers: help(), Sys.getenv/setenv/getwd/setwd, as.factor/data/
//! as.logical/nlevels/levels, set.seed, as.data.frame, the memory
//! allocation guard, rowSums/colSums/rowMeans/colMeans, GBM, save()/
//! load() session persistence (serialize_rval/deserialize_rval),
//! cv(), confusion.matrix(), mutate(), version(), clear()/cls(),
//! aov()/anova(), additional statistical tests, and the .Internal()
//! bridge.
//!
//! Module-private helpers: r2_next_random, check_alloc,
//! serialize_rval, deserialize_rval.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::all)]
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_stats::dist::{phi, qnorm_approx};
use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;
use crate::env_insert;

// help() — basic help system
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_help(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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
    soutln!("\n{}\n", help_text);
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// Sys.getenv(), Sys.setenv(), getwd(), setwd()
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_getwd(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let cwd = std::env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    Ok(rstr(&cwd))
}

pub(crate) fn bi_setwd(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a,0));
    std::env::set_current_dir(&path).map_err(|e| R2Err{msg:format!("cannot set working directory: {}", e),kind:ErrKind::Runtime})?;
    Ok(rstr(&path))
}

// ═══════════════════════════════════════════════════════════════════════
// as.factor(), data(), as.logical(), nlevels(), levels()
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_as_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_levels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Factor(f) => Ok(RVal::Character(f.levels.iter().map(|l| Some(l.clone())).collect(), Attrs::default())),
        _ => Ok(RVal::Null),
    }
}

pub(crate) fn bi_nlevels(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a, 0) { RVal::Factor(f) => Ok(rint(f.levels.len() as i32)), _ => Ok(rint(0)) }
}

pub(crate) fn bi_as_logical(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a, 0))?;
    Ok(RVal::Logical(v.into(), Attrs::default()))
}

pub(crate) fn bi_data(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let val = gv(a, 0);
    match &val {
        RVal::Character(v, _) => {
            let name = v[0].as_ref().map(|s| s.to_string()).unwrap_or_default();
            if e.global_env.lookup(&name).is_some() {
                soutln!("Dataset '{}' is already loaded", name);
            } else {
                soutln!("Dataset '{}' not found", name);
            }
        }
        RVal::DataFrame(_) => {
            soutln!("Dataset is already loaded in the environment");
        }
        _ => {
            soutln!("Available datasets: iris, mtcars, airquality");
        }
    }
    Ok(RVal::Null)
}





pub(crate) fn bi_is_factor(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Factor(_))))
}

pub(crate) fn bi_is_matrix(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rbool(matches!(gv(a, 0), RVal::Matrix(_))))
}

// ═══════════════════════════════════════════════════════════════════════
// set.seed() — reproducible random numbers
// ═══════════════════════════════════════════════════════════════════════

// Phase R.12: RNG primitives consolidated in r2_stats::rng. Engine
// retains a 1-line shim for the `.Internal("rnorm_vec", …)` path which
// R-language helper code still calls. All bi_* RNG builtins delegate
// directly to r2_stats::rng.
pub(crate) fn bi_set_seed(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::rng::bi_set_seed(a) }
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

pub(crate) fn bi_rowSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_colSums(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_rowMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_colMeans(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_Sys_sleep(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let secs = match &gv(a,0) { RVal::Numeric(v,_) => v[0].unwrap_or(0.0), _ => 0.0 };
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    Ok(RVal::Null)
}

/// `readline(prompt="")` — blocks until the user types a line on stdin
/// and presses Enter. Returns the line as a character scalar (without
/// the trailing newline). The prompt, if provided, is printed first.
/// Used for interactive prompts in scripts ("press Enter to continue",
/// "type a filename:", etc.).
pub(crate) fn bi_readline(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::{BufRead, Write};
    let prompt = gv(a, 0);
    let prompt_str = match &prompt {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        RVal::Null => String::new(),
        other => val_to_str(other),
    };
    if !prompt_str.is_empty() {
        sout!("{}", prompt_str);
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

pub(crate) fn bi_save(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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
        soutln!("Saved {} ({}) to '{}'", kind, obj.type_name(), path);
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
        soutln!("Saved {} objects to '{}'", count, path);
    }
    Ok(RVal::Null)
}

pub(crate) fn bi_load(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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
                    soutln!("Loaded {} ({}) from '{}'", kind, val.type_name(), path);
                    return Ok(val);
                }
                env_insert(&mut e.global_env, Arc::from(name), val);
                count += 1;
            }
        }
    }
    soutln!("Loaded {} objects from '{}'", count, path);
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

pub(crate) fn bi_confusion_matrix(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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
    soutln!("\nConfusion Matrix:");
    sout!("{:>12}", "Predicted→");
    for c in &classes { sout!("{:>8}", c); }
    soutln!("{:>10}", "Total");
    

    let n = pred.len();
    let mut correct = 0;
    for (ai, ac) in classes.iter().enumerate() {
        sout!("Actual {:>4} ", ac);
        let mut row_total = 0;
        for pi in 0..k {
            sout!("{:>8}", cm[ai * k + pi]);
            row_total += cm[ai * k + pi];
            if ai == pi { correct += cm[ai * k + pi]; }
        }
        soutln!("{:>10}", row_total);
    }

    
    let accuracy = correct as f64 / n as f64;
    soutln!("Accuracy: {}/{} ({}%)", correct, n, fmt_num(accuracy * 100.0));

    // Per-class precision and recall
    soutln!("\n{:>8} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1");
    for (ci, c) in classes.iter().enumerate() {
        let tp = cm[ci * k + ci] as f64;
        let pred_total: f64 = (0..k).map(|ai| cm[ai * k + ci] as f64).sum();
        let actual_total: f64 = (0..k).map(|pi| cm[ci * k + pi] as f64).sum();
        let precision = if pred_total > 0.0 { tp / pred_total } else { 0.0 };
        let recall = if actual_total > 0.0 { tp / actual_total } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        soutln!("{:>8} {:>10} {:>10} {:>10}", c, fmt_num(precision), fmt_num(recall), fmt_num(f1));
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
pub(crate) fn bi_mutate(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_data::dplyr::bi_mutate(a)
}

// ═══════════════════════════════════════════════════════════════════════
// version() — show R2 version info
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_version(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    soutln!("\nR2 — Statistical Computing, Reimagined");
    soutln!("Version: 0.1.1");
    soutln!("Created by: Devendra Tandale");
    soutln!("An AI assisted project");
    soutln!("Platform: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
    soutln!("Kernel: r2-linalg (pure Rust, no C/C++ dependencies)");
    soutln!("ML algorithms: 12 built-in");
    soutln!("Parallel cores: {}", rayon::current_num_threads());
    soutln!("Functions: 191+");
    soutln!("Codebase: 9,800+ lines of Rust");
    soutln!("License: AGPL v3");
    soutln!();
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// clear() / cls() — clear the terminal screen
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_clear(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::Write;
    // ANSI escape: \x1b[2J clears the visible region; \x1b[3J clears the scrollback
    // (supported by Windows Terminal, modern conhost, and all *nix terminals).
    // \x1b[H homes the cursor.
    sout!("\x1b[3J\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// aov() / anova() — Analysis of Variance
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_aov(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_aov(a) }

pub(crate) fn bi_anova(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_anova(a) }

// ═══════════════════════════════════════════════════════════════════════
// Additional Statistical Tests
// ═══════════════════════════════════════════════════════════════════════

// ── cor.test() — test if correlation is significant ──────────────────

pub(crate) fn bi_cor_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_cor_test(a) }

// ── shapiro.test() — test for normality ──────────────────────────────

pub(crate) fn bi_shapiro_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_shapiro_test(a) }

// ── wilcox.test() — Wilcoxon rank-sum / signed-rank test ─────────────

pub(crate) fn bi_wilcox_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_wilcox_test(a) }

// ── fisher.test() — Fisher's exact test for 2×2 tables ──────────────

pub(crate) fn bi_fisher_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::htest::bi_fisher_test(a) }

// ── weighted.mean() ──────────────────────────────────────────────────

pub(crate) fn bi_weighted_mean(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_iqr(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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

pub(crate) fn bi_internal(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
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
