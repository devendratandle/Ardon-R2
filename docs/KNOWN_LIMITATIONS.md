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
| `acf(x, k)` ignores the lag argument and returns every lag | Low — values are right, the count is not | **v0.4.1** |
| `ecdf(x)` is not implemented (the last item of the old `MISSING_FUNCTIONS.md` roadmap; the other 15 shipped) | Low | **v0.4.1** |
| Mixed-effects models — `lmer` exists but rejects the `(1\|group)` random-effect term | Medium | **v0.5.0** |
| Addon package system (load R2-script packages; optional-domain feature flags) | Medium — ecosystem | **v0.5.0** |
| `manova()` eigenvalues drift ~1–3% from R (needs a non-symmetric eigensolver) | Medium — accuracy | **v0.5.0** |
| Split-plot ANOVA: `Error(subject/within)` collapses to the outer stratum | Medium | **v0.5.0** |
| Parquet corpora must be converted first (`r2-arrow/parquet_io.rs` is unwired from training) | Low — memmap path now ships | **v0.5.0** |
| `read.parquet` cannot read zstd-compressed files — the only zstd implementation is C, and R2 ships none. snappy/gzip/lz4/brotli read fine; recompress zstd files with snappy | Low — a `pyarrow` one-liner | by design — needs a pure-Rust zstd in the `parquet` crate |
| `confint(glm)` is the Wald interval (R's `confint.default`), not the profile-likelihood interval R's `confint.glm` computes | Low — they agree away from the boundary | **v0.5.0** |
| `format(Sys.time())` prints UTC where R prints local time; `Sys.time() - t` is numeric, not `difftime` | Low | **v0.4.1** |
| Divide-and-conquer SVD/eigensolver (speed on large/wide matrices; `prcomp` on ≳100 features) | Low — perf, not correctness | **v1.0** |
| Oracle parallelism-threshold auto-calibration (hardware awareness) | Low — perf tuning | **v1.0+** |
| Apple-Silicon JIT (falls back to interpreter — upstream Cranelift aarch64 PLT) | Low — correct, just slower on ARM Mac | upstream |

---

## Language / evaluation

- **No visibility flag.** R tracks whether a value is invisible as code
  runs; R2 decides auto-printing by the SHAPE of the top-level statement.
  So `f <- function(v) print(v); f(1)` prints twice (R: once), and a
  function ending in `invisible(x)` still auto-prints. Needs a visibility
  flag in the evaluator, read by both consoles.
- **Hypothesis tests print when called**, not when their result is
  printed: `chisq.test(...)$p.value` shows the whole report before the
  value (R returns an `htest` object that prints only on display). The
  reports are ~50 direct writes across ~15 tests; each needs to be kept
  on the result and printed by its print method.
- **`format()` of a vector formats each element on its own**: R pads a
  vector to common decimals (`format(c(1, 2.5))` is `"1.0" "2.5"`, R2
  gives `"1" "2.5"`), and `nsmall` is treated as an exact decimal count
  rather than R's minimum. `print` of a numeric vector shares the first.
- `sprintf`: no `*` widths, `%o`, `%a`, or `%5$s` argument positions.
- **No lazy promises.** Arguments are evaluated eagerly, so `substitute()`
  works but the captured expression must still be evaluable, and R's
  skip-unused-argument semantics don't apply.

## Dates & time series

- **`acf(x, k)` ignores `k`.** R returns lags `0..k`; R2 returns every lag
  it can compute regardless of the argument. On `x` of length 10,
  `acf(x, 3)` gives 4 values in R and 10 in R2. The **autocorrelations
  themselves are exact** — the overlapping values match R to every digit
  printed (`1, 0.45, 0.5, -0.033333333`) — so this is an argument-handling
  bug, not a numerical one. Targeted for v0.4.1.

## Statistics

- **Non-central chi-squared** (`ncp != 0` in `dchisq`/`pchisq`/`qchisq`) is
  not implemented and says so with an error. Parameter recycling
  (`pchisq(3, c(1, 2, 5))`) is implemented for the chi-squared family;
  the other d/p/q functions still take their parameters as scalars. In
  `pt`, `pf`, `pbinom` and `ppois`, `log.p = TRUE` is the log of the
  computed tail, so a tail below ~1e-308 gives -Inf where R's log-scale
  arithmetic continues (`pnorm` and `pchisq` do continue).
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

- **No divide-and-conquer SVD/eigensolver.** `svd()`/`eigen()` are correct
  and accurate but use QR-iteration; very large or very wide matrices
  (`prcomp` on ≳100 features) are slower than R's LAPACK D&C routines. This
  is a *speed* gap, not an accuracy one. v1.0.

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
