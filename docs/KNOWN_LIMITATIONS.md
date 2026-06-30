# Known Limitations

What R2 does **not** do yet (or does differently from R), and **when we plan
to address each**. This lists *open* gaps only — resolved items move to
`CHANGELOG.md`. If something here blocks you, an issue helps us prioritise.

> Recently resolved (see CHANGELOG): full `sprintf` specs, `svd()`/`eigen()`
> eigenvectors, exact t/F/ANOVA/MANOVA p-values, memory-mapped out-of-core
> columns, zero-copy element-wise arithmetic, `factor(levels=)`, positional
> `rnorm`/`runif`/… parameters.

## Resolution schedule (priority)

| Limitation | Impact | Target |
|---|---|---|
| `format()`/`strftime` don't render `Date`/`POSIXct` (print the raw day/second count) | High — dates look wrong | **v0.3.5** |
| `merge()` — single-key inner join only (no multi-key, no outer joins) | Medium | **v0.3.5** |
| `acf()` lag-count semantics differ slightly from R | Low | **v0.3.5** |
| Mutable environments — `<<-` / closure-state counters only work at top level | High — blocks a common R idiom | **v0.4.0** |
| Addon package system (load R2-script packages; optional-domain feature flags) | Medium — ecosystem | **v0.4.0** |
| Mixed-effects models (`lmer`-style random effects) | Medium | **v0.4.0** |
| `manova()` eigenvalues drift ~1–3% from R (needs a non-symmetric eigensolver) | Medium — accuracy | **v0.4.0** |
| Split-plot ANOVA: `Error(subject/within)` collapses to the outer stratum | Medium | **v0.4.0** |
| Divide-and-conquer SVD/eigensolver (speed on large/wide matrices; `prcomp` on ≳100 features) | Low — perf, not correctness | **v1.0** |
| Dynamic (compiled `.dll`) packages | Low — only if real demand | **v1.0+** |
| Oracle parallelism-threshold auto-calibration (hardware awareness) | Low — perf tuning | **v1.0+** |
| Apple-Silicon JIT (falls back to interpreter — upstream Cranelift aarch64 PLT) | Low — correct, just slower on ARM Mac | upstream |

---

## Language / evaluation

- **Mutable environments.** `x <<- 5` works at top level, but the
  closure-state factory pattern (a returned function mutating a captured
  variable, e.g. a counter) does not — R2 currently snapshots a closure's
  captured environment. Same root as the for-loop captured-environment
  behaviour. Architectural; targeted for v0.4.0.
- **No lazy promises.** Arguments are evaluated eagerly, so `substitute()`
  works but the captured expression must still be evaluable, and R's
  skip-unused-argument semantics don't apply.

## Dates & time series

- **`format(d, fmt)` / `strftime` don't format `Date`/`POSIXct`.** The
  generic `format()` doesn't dispatch to the date formatter, so a `Date`
  prints as its raw day-count (e.g. `19797` instead of `2024-03-15`).
  `as.Date`, date arithmetic, and `difftime` are correct — only the
  string rendering is missing. Targeted for v0.3.5.
- **`acf(x, k)`** returns a slightly different lag count than R; the
  autocovariances themselves are correct. Targeted for v0.3.5.

## Statistics

- **`manova()` eigenvalues** of E⁻¹H drift ~1–3% from R's values on some
  designs (R2 routes through a symmetric solver; an exact non-symmetric
  eigensolver would close it). The four test statistics and their ordering
  are correct; the small drift is in the reported eigenvalues. v0.4.0.
- **Split-plot ANOVA.** `aov(y ~ x + Error(subject/within))` collapses to
  the outer (whole-plot) stratum — the one-way repeated-measures case is
  exact, but a full multi-stratum split-plot decomposition is not done. v0.4.0.
- **Mixed-effects models** (`lmer`-style random effects, REML) are not
  implemented. v0.4.0.
- **Paired Hotelling T².** R's `Hotelling` package and standard textbooks
  disagree on the paired convention; R2 follows the textbook definition.
  Documented difference, not a bug.

## Linear algebra

- **No divide-and-conquer SVD/eigensolver.** `svd()`/`eigen()` are correct
  and accurate but use QR-iteration; very large or very wide matrices
  (`prcomp` on ≳100 features) are slower than R's LAPACK D&C routines. This
  is a *speed* gap, not an accuracy one. v1.0.

## Data manipulation

- **`merge(df1, df2)`** does a single-key inner join (auto-detected or via
  `by=`). Multi-column keys (`by = c("a","b")`) and outer joins
  (`all.x`/`all.y`) are not implemented. v0.3.5.

## Packages / extensibility

- **No addon package system yet.** All functionality is statically linked
  into the one binary. Planned: load pure-R2-script packages from a path,
  and split optional domains behind Cargo feature flags so a minimal build
  is smaller. v0.4.0. Dynamic compiled (`.dll`) packages only if there's
  real demand (v1.0+).

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
