# Known Limitations

What R2 does **not** do yet (or does differently from R), and **when we plan
to address each**. This lists *open* gaps only — resolved items move to
`CHANGELOG.md`. If something here blocks you, an issue helps us prioritise.

> Recently resolved (see CHANGELOG): full `sprintf` specs, `svd()`/`eigen()`
> eigenvectors, exact t/F/ANOVA/MANOVA p-values, memory-mapped out-of-core
> columns, zero-copy element-wise arithmetic, `factor(levels=)`, positional
> `rnorm`/`runif`/… parameters, **`format()`/`strftime` on `Date`/`POSIXct`**,
> **closure-state `<<-`** (the counter-factory idiom), **the full `merge()`
> join** (composite keys, `all`/`all.x`/`all.y`, `by.x`/`by.y`, `.x`/`.y`
> suffixes, key ordering, type preservation), and **`data.frame()` no longer
> turning `stringsAsFactors` into a column** or leaving a short column
> un-recycled — all verified against this build at v0.4.0 and covered by
> `tests/differential/cases/merge_joins.R`.

## Resolution schedule (priority)

Every row below was re-tested against the v0.4.0 binary before this table
was written. Three entries carrying stale targets turned out to be already
fixed, and one — `merge()` silently ignoring `all.x`/`all.y` — turned out
to be worse than documented and was fixed here rather than scheduled.

| Limitation | Impact | Target |
|---|---|---|
| `acf(x, k)` ignores the lag argument and returns every lag | Low — values are right, the count is not | **next release** |
| `ecdf(x)` is not implemented (the last item of the old `MISSING_FUNCTIONS.md` roadmap; the other 15 shipped) | Low | **next release** |
| Mixed-effects models — `lmer` exists but rejects the `(1\|group)` random-effect term | Medium | **v0.5.0** |
| Addon package system (load R2-script packages; optional-domain feature flags) | Medium — ecosystem | **v0.5.0** |
| `manova()` eigenvalues drift ~1–3% from R (needs a non-symmetric eigensolver) | Medium — accuracy | **v0.5.0** |
| Split-plot ANOVA: `Error(subject/within)` collapses to the outer stratum | Medium | **v0.5.0** |
| Multi-factor `aov(y ~ a + b)` is not implemented — it now stops with an error (it used to fit `a` alone silently); one-way and `+ Error(subject)` work | Medium | **v0.5.0** |
| `anova(lm)` gives one overall "Model" row, not R's sequential (Type I) table per term; the totals agree | Medium — code ported from R reads a different table | **v0.5.0** |
| `hist(breaks = "Scott" / "FD" / a function)` is not supported (Sturges, a number of cells or a vector of breaks are); `hist(x, plot = FALSE)` typed at the console does not print the object (assign it); `freq =` / `right =` / `include.lowest =` are not read | Low | **next release** |
| `quantile()` drops NAs silently (R errors unless `na.rm = TRUE`) and its result is unnamed (R: `"30%"`) | Low | **next release** |
| Parquet corpora must be converted first (`r2-arrow/parquet_io.rs` is unwired from training) | Low — memmap path now ships | **v0.5.0** |
| LLM tokens have no end-of-text marker, so `llm.generate` runs on past a learned answer for the `n` it was asked for | Medium — fine-tuned answers need trimming | **next release** |
| On the GPU (`device = "gpu"`), frozen blocks, LoRA adapters and `accum` stop with an error (CPU only); AdamW, clipping, the schedule, ignored targets and padded batches run there | Medium | **v0.5.0** |
| `device = "gpu"` needs a build with `--features gpu`; the shipped `r2` is CPU only, and nothing picks the device for you | Low | **v0.5.0** |
| `llm.train` holds the tokens of its text in RAM (4 bytes each); only `tinystories_train` streams a corpus to a memory-mapped token file. `Model::load_dir` copies all weights to the heap: a model larger than RAM cannot be served yet, though the file is memory-mapped | Medium — sets the largest model and corpus | **v0.5.0** |
| Checkpoints stall the training thread while they write (124 ms at 7.24M, disk included); no background writer | Low | **v0.5.0** |
| `read.parquet` cannot read zstd-compressed files — the only zstd implementation is C, and R2 ships none. snappy/gzip/lz4/brotli read fine; recompress zstd files with snappy | Low — a `pyarrow` one-liner | by design — needs a pure-Rust zstd in the `parquet` crate |
| `confint(glm)` is the Wald interval (R's `confint.default`), not the profile-likelihood interval R's `confint.glm` computes | Low — they agree away from the boundary | **v0.5.0** |
| `format(Sys.time())` prints UTC where R prints local time; `Sys.time() - t` is numeric, not `difftime` | Low | **next release** |
| Oracle parallelism-threshold auto-calibration (hardware awareness) | Low — perf tuning | **v1.0+** |
| Apple-Silicon JIT (falls back to interpreter — upstream Cranelift aarch64 PLT) | Low — correct, just slower on ARM Mac | upstream |

---

## Language / evaluation

- **Deliberate difference (Windows only):** a string literal that starts
  like a pasted path — `"C:\Users\..."` or `"\\server\share"`, single
  backslashes — is read literally, where R treats `\U`, `\n` as escapes.
  A *relative* path with single backslashes (`"data\new.csv"`) is still
  read the R way (`\n` = newline): write `"data/new.csv"` or
  `"data\\new.csv"`. R2 also keeps unknown escapes such as `"\d"` as
  written where R reports an error.

- **A hypothesis test's report is laid out differently from R's
  `print.htest`**: R2 prints the method, then the statistic line; R adds
  the `data:` line, tab-indents the title, and prints the alternative,
  interval and estimates in a fixed order. The values are the same.
  (When the report appears is now R's: on printing the result, never while
  the test runs.)
- **The printed layout of `summary(fit)` and ANOVA tables differs from
  R's** (column widths, digits, the Residuals block); their values and
  the `coefficients` matrix match. `class()` gives one class where R
  gives two (`summaryDefault` without `table`, `anova` without
  `data.frame`); `inherits()` works.
- **A data frame or matrix prints at most 20 rows** (then `... (n more
  rows)`); R prints them all (up to `max.print`). Values and layout of
  the printed rows match R.
- `sapply` simplifies length-1 results to a vector; equal-length longer
  results stay a list where R builds a matrix.
- **Where R2's distributions differ from R, on purpose.** In extreme cases R is
  wrong and R2 is not, checked against a 340-digit reference or exact fractions:
  `pbeta(0.75, 1e4, 25, log.p = TRUE)` is -Inf in R and -2743.79 here;
  `qbeta(1e-300, 10, 1e5, lower.tail = FALSE)` is 1 in R and 0.00735 here;
  `qbeta(1e-300, 1e4, 25)` is 1.1e-308 in R and 0.92354 here;
  `qf(1e-20, 3, 12)` is 1.9% off in R, and `qf(1e-20, 1, 1)` is 0 in R and
  2.47e-40 here; `qhyper(1e-20, 100, 900, 50, lower.tail = FALSE)` is 24 in R and
  30 here. A quantile below the smallest double is 0 here, where R can return a
  nonzero value that misses its target (`qbeta(0.01, 0.001, 0.01)`, truly about
  e^-4500). In the other direction, `qt` at very large `df` is good to about 1e-12
  here, where R reaches 1e-16.
- **Missing distributions**: the non-central forms (`ncp != 0` stops with an
  error), `ptukey`, the Wilcoxon and signed-rank families, and `rmultinom`.
  R's generators draw different streams from R2's by design, so a seeded script
  gets the right distribution but not R's numbers.
- **`rf` is two functions.** `rf(n, df1, df2)` with single numbers is R's F
  generator; with a formula or vectors it is R2's random forest. A forest needs
  more than one row, so the two never collide.
- Taking a column of a matrix (`m[, 4]`) drops the row names R keeps.
- **`qr()` differs from R in three places.** R2 chooses its own routine, so
  `LAPACK = TRUE` is accepted and changes nothing; R's LAPACK object (full column
  pivoting, `$rank` always `min(dim)`) is never returned. `qr()` of a model fit stops
  with an error, where R returns the fit's `$qr`. At rank 0,
  `qr.fitted` returns 0, where R returns `y` as the fitted values and also as the
  residuals. On an ill-conditioned matrix the last `qraux` is a repeatedly downdated
  norm and differs from R's: for the Hilbert matrix of order 6 the truth is
  3.98624e-07, R has 3.98600e-07 and this has 3.98607e-07. Below the rank, the
  columns moved aside hold rounding noise, in R as here, and the two differ there.
- **S3 dispatch covers the common forms.** `UseMethod`, `NextMethod`, `g.default`,
  methods for builtins (`print.foo`, `format.foo`, `summary.foo`, `length.foo`) and
  for operators (`+.foo`, `Ops.foo` with `.Generic`) work, and auto-print uses
  `print.foo`. Not covered: methods for `[`, `[[` and `$`, the `Math` and `Summary`
  groups, `.Class`, and S4. `NextMethod()` passes the arguments the method was called
  with, not values the method changed. A builtin dispatches only on a class the
  object was given, not on an implicit one (`print.numeric` is not called by
  `print(1)`; `UseMethod` does use implicit classes).
- **Classed results are lists with differences.** `names(fit)` is sorted and lists the
  fields R2 keeps, which are not R's for `lm` or `htest`; `length` counts the same
  fields. `qr` is the exception and has R's four fields in R's order. `class(x) <-`
  stops with an error for a matrix, a data frame or a factor, and for the values
  "matrix", "array", "function", "data.frame" and "factor".
- **Several attributes print sorted by name**, where R prints them in the order they
  were set.
- **A missing name is the text `<NA>`.** `x[5]` and `x["z"]` print as in R, but
  `names(x["z"])` is the string "<NA>" where R has `NA`.
- **A matrix holds doubles.** `is.integer(matrix(1:4, 2))` is FALSE and `m[0]` is
  `numeric(0)`, where R says TRUE and `integer(0)`. A logical or character result
  with two dimensions (`m > 2`) is a vector with `dim`, which prints, subsets and goes
  through `apply`, `which(arr.ind = TRUE)`, `ifelse` and `g[i, j]` as R's does; `t()` of
  one is not covered.
- **Dates print with R's layout but not its locale.** Date-times are UTC unless `tz`
  says otherwise, and `weekdays`/`months` are in English.
- **`density()`'s `y` is the exact kernel sum**, where R bins the data and uses the
  FFT; the two differ in the third or fourth digit. `bw`, `n` and the grid `x` are R's.
  `d$call` is text, where R keeps a language object; it prints the same inside quotes.
- **`wilcox.test` gives R's p-value but not R's warning** "cannot compute exact p-value
  with ties"; with ties it takes the normal approximation, as R does. No `conf.int`.
- **`anova()` and `cor.test()` print in R2's own layout**; their values are R's.
  `anova()` handles one `lm`; two models (`anova(m1, m2)`) and `glm` fits give R2's
  older single test.
- **Points outside `xlim`/`ylim` are drawn outside the frame**, where R clips them.
- `sprintf`: no `*` widths, `%o`, `%a`, or `%5$s` argument positions.
- **String collation follows R on Windows / English locales** (letters
  case- and accent-blind first, then accents, then lowercase first;
  punctuation before digits before letters), for Latin letters. Scripts
  outside Latin-1 sort by code point, and R in the C locale (byte order)
  will differ.
- **No lazy promises.** Arguments are evaluated eagerly, so `substitute()`
  works but the captured expression must still be evaluable, and R's
  skip-unused-argument semantics don't apply.

## Dates & time series

- **`acf(x, k)` ignores `k`.** R returns lags `0..k`; R2 returns every lag
  it can compute regardless of the argument. On `x` of length 10,
  `acf(x, 3)` gives 4 values in R and 10 in R2. The **autocorrelations
  themselves are exact** — the overlapping values match R to every digit
  printed (`1, 0.45, 0.5, -0.033333333`) — so this is an argument-handling
  bug, not a numerical one. Targeted for the next release.

## Statistics

- **Non-central distributions** (`ncp != 0` in the chi-squared, t, F and beta
  functions) are not implemented and say so with an error.
- **`manova()` eigenvalues** of E⁻¹H drift ~1–3% from R's values on some
  designs (R2 routes through a symmetric solver; an exact non-symmetric
  eigensolver would close it). The four test statistics and their ordering
  are correct; the small drift is in the reported eigenvalues. v0.5.0.
- **Split-plot ANOVA.** `aov(y ~ x + Error(subject/within))` collapses to
  the outer (whole-plot) stratum — the one-way repeated-measures case is
  exact, but a full multi-stratum split-plot decomposition is not done. v0.5.0.
- **Mixed-effects models are stubbed, not working.** `lmer` is registered
  as a builtin, but a formula carrying the standard random-effect term is
  rejected: `lmer(y ~ x + (1|g))` errors with "formula must include at
  least one random-effect term". So the function exists and cannot yet be
  called successfully — verified against v0.4.0. v0.5.0.
- **Paired Hotelling T².** R's `Hotelling` package and standard textbooks
  disagree on the paired convention; R2 follows the textbook definition.
  Documented difference, not a bug.

## Linear algebra


## Graphics

Shipped: `plot`/`hist`/`boxplot`/`barplot`/`pairs`/`pie`/`matplot`,
`lines`/`points`/`abline`/`rect`/`text`/`legend`/`title`/`axis`,
`xlim`/`ylim`, the `cex.*`/`font.*`/`col.*`/`las`/`sub` chrome on every
plot type, `rgb`/`hsv`/`adjustcolor`, SVG/PDF devices, the CLI browser
viewer and the GUI window. Not yet:

- `mtext()`; `image()`, `contour()`, `persp()`; `col2rgb()`.
- Log-scale axes (`log = "x"/"y"/"xy"`), `xaxt`/`yaxt = "n"`, `tck`/`tcl`,
  `mgp`.
- R's default axis labels come from `deparse(substitute(x))`; R2's
  builtins do not receive the unevaluated argument, so `plot(x, y)` labels
  the axes `"x"`/`"y"` by convention rather than by deparsing the
  expression passed.
- Per-bar `col=` vectors on `boxplot`/`barplot` fill uniformly.

## Packages / extensibility

- **Addon packages — script packages work (functions, types & methods);
  online registry pending.** `install.packages(name, path=…)` installs
  pure-R2-script packages from a local dir, a `.zip`, or a GitHub
  `user/repo`; `library()`/`require()`/`detach()`/`uninstall()` work; and a
  package may export **functions, types, and methods** (verified end to end).
  **Open:** (1) `install.packages(name)` with no `path` — the online package
  registry isn't live; (2) optional-domain Cargo feature flags for a smaller
  minimal build. v0.5.0. Packages are R2 script, JIT-compiled at load; there
  is no compiled-binary package form and none is planned.

## Platform

- **Apple Silicon JIT.** Ardon-R2 runs fully on M-series Macs, but the
  Cranelift JIT falls back to the interpreter there (upstream Cranelift
  doesn't yet implement aarch64 PLT relocation). Results are identical;
  only the JIT speedup is unavailable until the upstream fix lands.

## Performance tuning

- **Oracle thresholds are fixed, not auto-calibrated.** The serial-vs-
  parallel cut-overs in `r2_oracle` are hand-tuned constants rather than
  measured per-machine. A calibration pass (with CPU-feature/cache
  detection) is bundled with the broader hardware-awareness work. v1.0+.
