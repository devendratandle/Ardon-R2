# Changelog

Version-by-version record of what became **available** or got **fixed** for
users. It describes capabilities, not internal implementation — algorithm
choices and refactors live in the code and `docs/ARCHITECTURE.md`.

---

## Unreleased

**New — mutable environments (R's environment semantics).** Environments
are now live, shared objects rather than snapshots. Everything built on
them works as in R: stateful closures (`make_counter()` factories,
accumulators), `<<-` from any nesting depth using R's lexical rule
(rebind in the nearest enclosing frame that has the name, else global),
independent per-call factory environments, and the new `local(expr)`.
As a direct consequence the interpreter got dramatically faster — the old
copy-on-write model cloned the global binding table on every top-level
assignment: top-level loops run ~31× faster, function calls ~18× faster
(release, measured on the same machine).

**New — differential-vs-R correctness harness.** `tests/differential/run.sh`
executes a battery of R scripts under both Ardon-R2 and GNU R and compares
their outputs numerically. Any semantic divergence from R now fails a test
before it can ship. Nine case files cover matrix arithmetic and metadata,
indexing, statistics, lm/glm, data frames, strings, control flow, and
numeric edge semantics.

**Fixed — R-compatibility batch (found by the harness on its first run):**
- `v[["name"]]` extracts from a named vector (previously an error).
- `is.nan()`, `is.infinite()`, `is.finite()` added; `is.na()` is TRUE for
  NaN and works elementwise on every atomic type.
- `summary(fit)` returns the summary object, so `s <- summary(fit);
  s$r.squared` works (printing unchanged).
- `scale(x)` accepts a plain numeric vector (promoted to n×1, as in R).
- Division by zero follows IEEE/R semantics: `1/0` is `Inf`, `0/0` is
  `NaN` — no longer a runtime error.
- `%%` is floored modulo like R (`-7 %% 3` is `2`, was `-1`), across the
  scalar, fused, and columnar paths.
- `round()` rounds half to even (banker's rounding): `round(2.5)` is `2`.
- Recycling is silent when the longer length is a clean multiple of the
  shorter (`c(1,2,3,4) + c(10,20)` works).
- `glm(..., family = binomial)` accepts the bare family function.
- `deviance(model)` added for lm/glm.
- Numbers print with 7 significant digits like R (values below 1 with
  leading zeros previously lost a digit).
- `colnames()` / `rownames()` getters work on matrices (previously NULL
  even when names were set).

## v0.3.7 (July 2026)

**New — the JIT compiles whole iterative algorithms.** Plain R2 source
functions carrying scalar, vector, and matrix state across `for`/`while`
loops compile to one native unit — gradient descent, Newton, EM,
fixed-point, shrinkage, and multi-parameter GD/IRLS with `X %*% b` /
`t(X) %*% r`. Measured vs interpreted (release): matrix-state GD **32×**,
vector-state iteration up to ~64×, scalar-state training loops ~3×.
Anything outside the compiled subset (mis-typed calls, read-before-define,
unsupported shapes) falls back to the interpreter — never a wrong answer.

**New — statistics formulas reach/beat native.** `var`/`cov`/`cor`/`sd`/
z-score/RMSE/R² written as one-line R2 formulas JIT with common-
subexpression elimination, single-pass SIMD wave fusion, and user-helper
inlining; `cor` written in R2 source outruns the native builtin.
`explain(f)` reports whether a function compiled and, if not, exactly why;
`explain(x)` reports data size, architecture, and the serial-vs-parallel
plan.

**New — parallel apply for library code.** `mclapply` / `par.lapply` /
`par.sapply` run pure closures across cores with isolated workers and
per-worker reproducible RNG (~4.5× on 6 cores; the r2sem PLS-SEM source
library runs ~13× faster than R's cSEM on identical estimates). Composes
with the JIT.

**New — linear algebra.** Non-symmetric `eigen()` (complex spectra reported
honestly via `$imaginary`), `backsolve`/`forwardsolve`, `rcond`, `kappa`.
Matrix multiply routes small/thin shapes to a fast path (5×5 `%*%` ~34×
faster; thin regression shapes like `X %*% w` benefit throughout).

**GUI (verified live on screen).** Console selection is a solid blue band
with white inverted text; warm paper background, deeper ink, and scarlet
input for long-session readability; the console snaps back to the prompt
when output arrives (previously the returning prompt could stay hidden
after scrolling up or right); platform-appropriate monospace fonts on
Windows/macOS/Linux with a directory-scan fallback; plot y-axis labels and
frame no longer clip at the default device size.

## v0.3.6 (July 2026)

**Fixed**
- `factor(x, levels = c(...))` now honours the given levels (set and order);
  previously the levels argument was ignored and values were sorted
  alphabetically, producing wrong codes.
- Random generators take positional parameters: `rnorm(n, 10, 3)`,
  `runif(n, 5, 15)`, `rexp`, `rbinom`, `rpois` — these silently dropped the
  parameters before and returned defaults.
- JIT no longer mis-runs a `for`-loop accumulator inside a function
  (`function(n){ s<-0; for(k in 1:n) s<-s+k; s }` now returns the sum, not 0).
- `abline(lm(y ~ x))` draws the actual fitted line (intercept/slope read from
  the model) instead of a 45° `y = x` line.
- `seq(from, to, length.out = n)` returns exactly `n` evenly spaced points;
  previously `length.out` was ignored and it fell back to `by = 1`
  (`seq(0, 2*pi, length.out = 100)` gave 7 points, which broke `matplot` curves).
- `pairs()` draws axis tick labels on the matrix's outer edges (alternating
  sides); `barplot`/`boxplot` no longer overlay numeric ticks on the
  categorical x-axis.

**Graphics**
- Shared graphical parameters now behave like R's `par()` across **every**
  plot type (`plot`/`hist`/`matplot`/`barplot`/`boxplot`/`pairs`): `las`
  (axis-label rotation 0/1/2/3), `col.axis`/`cex.axis`, and per-call `mar=`
  or `par(mar=)` margins — each with a tuned default. Category labels on
  bar/box plots rotate with `las` (horizontal default, vertical on `las=2/3`).

**Performance**
- Element-wise arithmetic (`a + b`, `-`, `*`, `/`) on numeric vectors is now
  zero-copy on the columnar path — at parity with, and in repeated use
  slightly faster than, R. This was R2's one remaining slower-than-R workload.

**Added**
- **Type methods & inheritance.** `method name(x: Type) …` now actually
  dispatches — `name(obj)` runs the method registered for `obj`'s type,
  including methods inherited via `type B extends A`. Inheriting types also
  inherit their parent's fields. (Previously `method` parsed but calling it
  gave "object not found".)
- **Addon packages can export types and methods**, not just functions —
  `library()` of a `type`+`method` package works, and `detach()` cleans them
  up.
- **Date rendering.** `format(d, fmt)` / `strftime` / `as.character()` print
  `Date`/`POSIXct` as calendar strings (not the raw day/second count),
  `c()` keeps the date class, and `class()` reports `"Date"`.
- Linux install guide and one-shot installer for the CLI and GUI
  (`INSTALL_LINUX.md`, `scripts/install-linux.sh`).

---

## v0.3.3 (June 2026)

**Available**
- Metaprogramming: `quote`, `eval`, `parse`, `deparse`, `call`/`as.call`,
  `body`/`formals`/`args`, `substitute`, `match.call`/`sys.call`, `bquote` —
  code is data and back. (Arguments are still eagerly evaluated; no lazy
  promises yet.)
- `isTRUE`/`isFALSE`/`identical`/`all.equal`/`diag`/`toString`; operators
  usable as functions (`Reduce(\`+\`, x)`); `repeat { }` loops.

**Fixed**
- Factors: `factor[i]`, `factor == "level"`, factor columns under
  `df[mask, ]`, `as.numeric(factor)`, and `tapply`/`aggregate` by a factor.
- `sprintf` full `%[flags][width][.precision]` specs; `ifelse` keeps the
  branch type; `pdf()` writes one page per plot.
- Negative (exclusion) indexing on vectors, matrices, and data frames;
  `strsplit()` returns a list; vectorized `paste`/`substr`; `as.numeric("…")`.
- Replacement functions `names<-`/`colnames<-`/`rownames<-`; `rm()` of
  multiple names; `T`/`F` reassignable; variables may be named `c`/`t`/`df`.
- Top-level `for`/`while` no longer read a loop variable one iteration stale.

---

## v0.3.2 (June 2026)

**Available**
- Graphics: `pairs()`, `pie()`, `matplot()`, `curve()`; overlays `text()`,
  `title()`, `axis()`, `rect()`; plot params `col`/`cex`/`pch`/`type`/`lwd`;
  `pdf()`/`png()`/`svg()` file devices.
- ~85 more base-R functions: `seq_len`/`seq_along`/`%in%`/`setdiff`/`union`/
  `intersect`/`unlist`/`split`/`cut`/`pmin`/`pmax`; `Reduce`/`Filter`/`Map`;
  `switch`/`with`/`tryCatch`/`stopifnot`; `attr`/`attributes`/`structure`/
  `format`/`inherits`; the `dexp/pexp/qexp`, `dbinom/…`, `dpois/…`, `dt/…`,
  `dchisq/…`, `pf/qf` distribution families; `uniroot`/`integrate`/`optimize`.

**Fixed**
- `...` (dots) are captured and forwarded into inner calls; variadic
  `sum`/`min`/`max`/`prod`; `[[ ]]` read indexing; data-frame column
  iteration in `lapply`/`sapply`/`for`.

---

## v0.3.1 (June 2026)

**GUI**
- Resolution-adaptive UI (720p → 4K) with edge/corner resize cursors and a
  legible title-bar logo.

---

## v0.3.0 (June 2026)

**Available**
- Out-of-core compute on larger-than-RAM data: `mmap.csv()` streams a CSV to
  per-column sidecars; `mmap.lm()` fits least squares in one streaming pass;
  `mmap.map()` transforms out-of-core; streaming `sd`/`var`/`prod`/`range`
  and approximate `median`/`quantile` over mmapped columns.
- `read.parquet()` — pure-Rust Parquet import.
- Hardware-aware matrix multiply: multi-core + runtime AVX2/AVX-512 — about
  14× faster than default R on `%*%`, results identical.

**Fixed**
- `na.rm = TRUE` honoured across `sum`/`mean`/`min`/`max`/`prod`/`var`/`sd`/
  `median`; `names()` works on lists; `lm`/`glm`/`aov` accept the positional
  `data` argument (`lm(y ~ x, df)`).

---

## v0.2.2 (June 2026)

**Available**
- `solve()` (inverse / linear solve) and `det()` exposed as functions.

**Fixed**
- Statistical output (`t.test`, `chisq.test`, `aov`, `manova`, `wilcox.test`,
  model summaries, ML output) now appears in the desktop GUI console, not
  only the CLI.
- Numerical accuracy now matches R to ~1e-9: `lm` (stable least squares),
  `qnorm`, and every t / F / ANOVA / MANOVA / Hotelling p-value and CI.
- `aggregate()` accepts the formula interface, including multi-term and
  `cbind(...)` multi-response formulas, with real source column names.

---

## v0.2.1 (June 2026)

**Available**
- Runtime-swappable BLAS: matrix multiply can dispatch at runtime to a
  CPU-specialised build of the same pure-Rust kernel via `R2_BLAS`, falling
  back to the built-in kernel. Stays strictly pure-Rust.

---

## v0.2.0 (June 2026)

**Available**
- Native desktop GUI (`R2Gui`): an MDI workspace with floating console and
  graphics windows (drag/resize), replacing the CLI-only build.
- Multivariate statistics: `manova()` (Wilks / Pillai / Hotelling-Lawley /
  Roy), `hotelling.test()` (one-/two-sample/paired), and repeated-measures
  `aov(y ~ x + Error(subject))`.

---

## v0.1.1 (May 2026)

**Available**
- In-memory graphics device with full `par()` (multi-panel `mfrow`/`mfcol`,
  margins, `pch`/`lty`/`lwd`/`col`) and a built-in browser plot viewer
  (`dev.view()`) with a session gallery.
- `.r2d` native binary dataset format; formula data scope + factor expansion
  in `lm()`/`glm()`; opt-in single-precision (f32) storage.
- Wider JIT coverage for user closures (math calls, 2-argument and
  branchy bodies, fused map-reduce).

---

## v0.1.0 (May 2026) — first stable release

**Available**
- A Cranelift JIT that compiles pure-arithmetic user functions to native
  code (scalar, vector maps, reductions, branchy and composed bodies), with
  a central scheduler choosing serial vs. parallel execution.
- Columnar numeric storage with dense fast-path reductions; `cumsum`/
  `cumprod`/`cummax`/`cummin`, `which.max`/`which.min`, rolling
  `sum`/`mean`/`max`/`min`/`sd`, hash-based group aggregation, and
  Euclidean/Manhattan/cosine distances.
- Real `svd()`, `eigen()` (eigenvalues **and** eigenvectors), and QR;
  `prcomp()$rotation` is genuine.
- R-faithful hypothesis tests (Welch t-test, formula and paired forms, exact
  Fisher); RFC-4180 CSV parsing; regular expressions.

---

## v0.0.9 (April 2026) — initial release

**Available**
- Core language: vectors, data frames, formulas, R-style assignment and
  1-based indexing.
- Statistics: `lm`, `glm`, `t.test`, `aov`, `shapiro.test`, `cor.test`.
- Machine learning, all built in: decision tree, random forest, gradient
  boosting, KNN, PCA, K-means, naive Bayes (12 algorithms).
- Math kernel: BLAS-style operations, matrix decompositions, SVD,
  eigenvalues.
- Data handling: CSV read, filter/select/mutate/arrange.
