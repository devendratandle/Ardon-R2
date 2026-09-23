//! `help()` / `?topic`: answers from FUNCTIONS.md, which is embedded in the
//! binary and parsed on first use, so help cannot drift from the reference.

#![allow(clippy::all)]

use r2_types::*;

use crate::{gv, val_to_str, Engine};

/// The function reference, embedded in the binary.
///
/// `include_str!` rather than a second hand-maintained table: FUNCTIONS.md
/// is the document users already read, so parsing it at first use means
/// `?topic` cannot drift from the reference the way a copy would. It costs
/// 20 KB — 0.14% of the binary — against 401 of 438 builtins having had no
/// help at all, which is the problem actually worth solving.
///
/// Long-form material (architecture, tutorials, the manual) deliberately
/// stays OUT of the binary and ships beside it. What belongs here is the
/// one-line answer a REPL user needs without leaving the prompt, because a
/// binary that has to find a docs folder to answer `?mean` has given up the
/// property that makes it worth shipping as one file.
const FUNCTIONS_MD: &str = include_str!("../../../../FUNCTIONS.md");

/// One parsed reference entry: `name(args)  description`, plus any indented
/// continuation lines beneath it.
struct HelpEntry {
    sig: String,
    desc: String,
    more: Vec<String>,
}

/// Split one reference line into `(name, signature, description)`.
///
/// Hand-scanned rather than regexed: r2-engine has no regex dependency and
/// the shape is fixed. A space must separate the description, which is what
/// stops `TRUE/FALSE/T/F  Logical constants` being read as a function
/// called `TRUE`.
fn parse_ref_line(line: &str) -> Option<(&str, &str, &str)> {
    let b = line.as_bytes();
    if b.is_empty() || !(b[0].is_ascii_alphabetic() || b[0] == b'.') {
        return None;
    }
    let mut i = 0;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_') {
        i += 1;
    }
    let name = &line[..i];
    let rest = &line[i..];
    // The signature runs to the MATCHING paren: `lmer(y ~ x + (1|g), data=)`
    // nests one level, and cutting at the first `)` would lose it.
    let (sig, rest) = if rest.starts_with('(') {
        let mut depth = 0i32;
        let mut close = None;
        for (i, ch) in rest.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => { depth -= 1; if depth == 0 { close = Some(i); break; } }
                _ => {}
            }
        }
        match close {
            Some(p) => (&rest[..p + 1], &rest[p + 1..]),
            None => return None,
        }
    } else {
        ("", rest)
    };
    // A signature that fills its line — `llm.new(dim=128, ...)` — carries
    // its description on the indented line below; accept it with an
    // empty description and let the continuation supply the text.
    if rest.trim().is_empty() {
        return if sig.is_empty() { None } else { Some((name, sig, "")) };
    }
    if !rest.starts_with(' ') {
        return None;
    }
    // `rest` is returned UNTRIMMED: the run of spaces between columns is
    // the only thing that marks one, and trimming here would erase it.
    Some((name, sig, rest))
}

/// Split one reference line into every function it documents.
///
/// The reference is laid out in COLUMNS in places:
///
/// ```text
/// abs(x)      Absolute value          sqrt(x)     Square root
/// ```
///
/// Read one-per-line, that gives `abs` the description "Absolute value
/// sqrt(x) Square root" and never indexes `sqrt` at all — which is how
/// `?sqrt`, `?log`, `?min`, `?sort` and 160 others came to answer nothing
/// while the reference documented them the whole time.
///
/// A column break is a run of two or more spaces followed by something
/// that itself parses as an entry. `factorial(x) x!  (via gamma)` is not
/// one, because `(via gamma)` does not parse — which is the test that
/// keeps ordinary two-space prose from being split.
fn parse_ref_columns(line: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut cur = line;
    loop {
        let Some((name, sig, rest)) = parse_ref_line(cur) else { break };
        // An alias chain: `acf(x) / pacf(x)  Auto- / partial-correlation`,
        // or `dexp(x) / pexp(q) / qexp(p)  Exponential ...` — any length.
        // Every name in the chain gets the description at the end of it.
        let trimmed = rest.trim();
        if trimmed.starts_with("/ ") {
            let mut names = vec![(name.to_string(), sig.to_string())];
            let mut tail = trimmed;
            let mut desc = String::new();
            let mut ok = false;
            while let Some(after) = tail.strip_prefix("/ ") {
                match parse_ref_line(after) {
                    Some((n2, s2, r2)) => {
                        names.push((n2.to_string(), s2.to_string()));
                        tail = r2.trim();
                        if !tail.starts_with("/ ") { desc = tail.to_string(); ok = true; break; }
                    }
                    None => break,
                }
            }
            if ok {
                for (n, sg) in names { out.push((n, sg, desc.clone())); }
                break;
            }
        }
        // Otherwise look for the next column.
        //
        // Two guards, both learned from getting it wrong. Start scanning
        // AFTER the leading run of spaces, or the gap between the
        // signature and its own description reads as a column break and
        // every entry comes out with an empty description. And require the
        // candidate to carry a PARENTHESISED signature: without that,
        // `lgamma(x)   log Γ(x)` splits at "log" and `?log` answers
        // "Γ(x)", while `Absolute value` becomes a function called
        // `Absolute`.
        let b = rest.as_bytes();
        let mut i = 0;
        while i < b.len() && b[i] == b' ' { i += 1; }
        let mut cut = None;
        while i + 1 < b.len() {
            if b[i] == b' ' && b[i + 1] == b' ' {
                let mut j = i;
                while j < b.len() && b[j] == b' ' { j += 1; }
                if j < b.len() {
                    if let Some((_, sig2, _)) = parse_ref_line(&rest[j..]) {
                        if !sig2.is_empty() {
                            cut = Some((i, j));
                            break;
                        }
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        }
        match cut {
            Some((e, st)) => {
                out.push((name.to_string(), sig.to_string(), rest[..e].trim().to_string()));
                cur = &rest[st..];
            }
            None => {
                out.push((name.to_string(), sig.to_string(), rest.trim().to_string()));
                break;
            }
        }
    }
    out
}

/// The reference, parsed once on first use.
fn help_index() -> &'static std::collections::HashMap<String, HelpEntry> {
    use std::sync::OnceLock;
    static IDX: OnceLock<std::collections::HashMap<String, HelpEntry>> = OnceLock::new();
    IDX.get_or_init(|| {
        let mut map: std::collections::HashMap<String, HelpEntry> =
            std::collections::HashMap::new();
        let mut cur: Option<String> = None;
        for line in FUNCTIONS_MD.lines() {
            if line.starts_with('#') || line.starts_with("```") || line.trim().is_empty() {
                cur = None;
                continue;
            }
            let cols = parse_ref_columns(line);
            if !cols.is_empty() {
                for (name, sig, desc) in &cols {
                    map.insert(name.clone(), HelpEntry {
                        sig: sig.clone(), desc: desc.clone(), more: Vec::new(),
                    });
                }
                // Continuation lines belong to the LAST column on the line.
                cur = cols.last().map(|(n, _, _)| n.clone());
            } else if line.starts_with(' ') {
                if let Some(n) = cur.as_ref() {
                    if let Some(e) = map.get_mut(n) {
                        if e.desc.is_empty() { e.desc = line.trim().to_string(); }
                        else { e.more.push(line.trim().to_string()); }
                    }
                }
            }
        }
        map
    })
}

/// Everything `?topic` does when the topic has no hand-written entry.
///
/// Three outcomes in order: the reference entry; a SEARCH across names and
/// descriptions, so both a misspelling and a concept like `?"regression"`
/// land somewhere useful; or the overview. Silence was the old behaviour
/// for 401 of 438 functions.
fn help_fallback(topic: &str) -> Result<RVal, R2Err> {
    let idx = help_index();
    if let Some(e) = idx.get(topic) {
        soutln!("");
        soutln!("{}{}", topic, e.sig);
        soutln!("  {}", e.desc);
        for m in &e.more {
            soutln!("  {m}");
        }
        soutln!("");
        return Ok(RVal::Null);
    }
    if topic.is_empty() {
        soutln!("");
        soutln!("{}", HELP_OVERVIEW);
        soutln!("");
        return Ok(RVal::Null);
    }
    let needle = topic.to_lowercase();
    let mut hits: Vec<(&String, &HelpEntry)> = idx
        .iter()
        .filter(|(n, e)| {
            n.to_lowercase().contains(&needle) || e.desc.to_lowercase().contains(&needle)
        })
        .collect();
    if !hits.is_empty() {
        hits.sort_by(|a, b| {
            let pa = !a.0.to_lowercase().starts_with(&needle);
            let pb = !b.0.to_lowercase().starts_with(&needle);
            pa.cmp(&pb).then(a.0.len().cmp(&b.0.len())).then(a.0.cmp(b.0))
        });
        soutln!("");
        soutln!("No help topic {topic:?}. Closest matches:");
        soutln!("");
        for (n, e) in hits.iter().take(12) {
            soutln!("  {}{}  -  {}", n, e.sig, e.desc);
        }
        if hits.len() > 12 {
            soutln!("");
            soutln!("  ...and {} more.", hits.len() - 12);
        }
        soutln!("");
        return Ok(RVal::Null);
    }
    soutln!("");
    soutln!("No help for {topic:?}. It may be a function with no reference entry");
    soutln!("yet - see FUNCTIONS.md, or the docs/ folder installed beside R2.");
    soutln!("");
    Ok(RVal::Null)
}

/// Shown by `help()` with no argument.
const HELP_OVERVIEW: &str = "Ardon-R2 Help System — Available topics:\n\n  Statistics:  lm, glm, t.test, chisq.test, cor, cor.test\n               aov, anova, shapiro.test, wilcox.test, fisher.test\n               mean, sd, var, median, quantile, IQR, weighted.mean\n  ML:          rpart, rf, gbm, kmeans, knn, prcomp, naive.bayes\n  Evaluation:  cv, confusion.matrix\n  Graphics:    plot, hist, boxplot, barplot\n  Data:        read.csv, filter, select, mutate, arrange\n  Session:     save, load, version\n  Core:        c, library, data.frame, matrix, scale, .Internal\n  Inspection:  summary, str, head, tail, names, dim, class\n\n  Type help(\"topic\") or ?topic for details.";

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
        "is.ordered" => "is.ordered(x)\n  TRUE for an ordered factor (factor(..., ordered = TRUE)).\n  Example: is.ordered(factor(c(\"s\", \"m\"), ordered = TRUE))",
        "library" => "library(package)\n  Load a package.\n  Example: library(mymath)",
        "data.frame" => "data.frame(...)\n  Create data frame.\n  Example: data.frame(x = 1:5, y = c(\"a\",\"b\",\"c\",\"d\",\"e\"))",
        "matrix" => "matrix(data, nrow, ncol)\n  Create matrix.\n  Example: matrix(1:12, nrow = 3, ncol = 4)",
        "scale" => "scale(x, center=TRUE, scale=TRUE)\n  Center and standardize matrix columns.",
        ".Internal" | "internal" => ".Internal(name, ...)\n  Call Rust primitive from Ardon-R2 script.\n  Available primitives:\n    matmul, crossprod, crossprod_vec, solve, solve_lstsq,\n    inverse, cholesky, eigenvalues, svd,\n    rnorm_vec, pnorm, qnorm\n  Example: beta <- .Internal(\"solve_lstsq\", X, y)",
        "summary" | "str" | "head" | "tail" | "names" | "dim" | "class" => "Data inspection functions:\n  summary(x)  — summary statistics\n  str(x)      — structure\n  head(x, n)  — first n rows\n  tail(x, n)  — last n rows\n  names(x)    — column names\n  dim(x)      — dimensions\n  class(x)    — type/class",
        _ => return help_fallback(&topic),
    };
    soutln!("\n{}\n", help_text);
    Ok(RVal::Null)
}
