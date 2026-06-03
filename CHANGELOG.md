# Changelog

## Unreleased

### Statistics — exact p-values (Lentz incomplete beta)

`incomplete_beta` was a 1000-panel trapezoidal rule (~1e-3–1e-4 error,
worse near boundaries); the ANOVA tables additionally used a
Wilson-Hilferty F→p approximation (~1e-3, off by up to ~2× at small
df). Both are replaced with **exact** methods:

- **`incomplete_beta`** now uses the Lentz continued fraction (Numerical
  Recipes §6.4) with the symmetry swap, accurate to ~1e-9 across the
  full range — including the `b < 1` corner the `t.test` path exercises
  (the case that defeated the earlier CF attempt).
- **New `f_sf(f, df1, df2)`** — exact F upper-tail via the incomplete
  beta. `aov` (incl. repeated-measures `Error(...)` strata) and `lm`'s
  F-test now use it instead of Wilson-Hilferty.
- A **zero residual** (perfectly-fit / degenerate design) now yields
  **p = 0** rather than the approximation's spurious p = 1.

- **`manova` / `hotelling.test`** now route through `f_sf` as well
  (their `f_to_pvalue` no longer uses Wilson-Hilferty), so small-df
  multivariate cases are exact too (e.g. manova F(2,3)=52.13 →
  p = 0.0047, was 0.0013; Wilks F(2,2)=34.75 → p = 0.028, was 0.0154).
- **`lm` coefficient p-values** now use the **t-distribution**
  (`Pr(>|t|)` via the exact `t_cdf`) instead of the normal
  approximation. This matters at small residual df: for n=10, p=2
  (df=8) the x-coefficient p went from ≈3.9e-4 (normal) to 0.00755
  (R's value) — a ~19× correction. (`glm` keeps the normal/Wald *z*
  test, which is correct for GLMs.)

Effect: **every** F-distribution p-value in R2 — repeated-measures
ANOVA, one-/two-way ANOVA, `lm` F-tests, `manova`, `hotelling.test` —
plus `t.test` p-values/CIs now match R to ~1e-9 (e.g. RM-ANOVA
F(2,6)=12.40 → p = 0.0074, was 0.0052; two-sample t p = 1.6409e-5, was
1.644e-5).

Repeated-measures ANOVA structure (`aov(y ~ x + Error(subject))`, incl.
nested `Error(subject/within)`) was already correct; this fixes its
p-values.

### Engine — formula support for `aggregate()` (multi-term)

`aggregate(...)` now accepts the **formula interface**, including
multi-term formulas:

```r
aggregate(val ~ grp, data = df, FUN = mean)            # single
aggregate(y1 ~ g1 + g2, data = df, FUN = mean)         # multiple groups
aggregate(cbind(y1, y2) ~ g1, data = df, FUN = sum)    # multiple responses
aggregate(cbind(y1, y2) ~ g1 + g2, data = df, FUN = mean)
```

Previously only the positional `aggregate(x, by, FUN)` form existed and
the formula failed with "object '…' not found" — `aggregate` was the
one formula-aware function missing from the engine's centralized
formula handling.

The fix introduces **`formula_frame`**, a small "model.frame"-style
input adapter in the engine that splits a formula into its response
columns (handling `cbind(...)`) and grouping factors (handling
`a + b`), resolving each name against `data=`. It only *assembles named
columns* — the split-apply math (FUN applied per group) is unchanged,
so results match the non-formula form.

- Output columns now use the **real source names** (`g1`, `g2`, `y1`,
  `y2`) and are ordered by grouping level, matching R (previously the
  single-variable path emitted generic `Group`/`Value`).
- Both `FUN = mean` (named) and a positional function argument work.
- Regression tests pin single, multi-group, multi-response, and a
  t.test **formula↔vector equivalence** check.

Note (unchanged behaviour): `t.test`, `lm`, `aov`, one-sample `t.test`,
and `manova(cbind(...) ~ g)` already handled formulas correctly.

### Release automation — macOS & Linux desktop GUI

The `release` workflow now builds and packages the desktop GUI for
**all three** platforms (previously CLI-only on macOS/Linux):

- **Linux**: `R2Gui-linux-x86_64.tar.gz` — `R2Gui` + `r2` binaries, a
  `.desktop` launcher, icon, and run notes. The Ubuntu runner installs
  the winit/wgpu system deps (X11/Wayland + GL headers).
- **macOS**: `R2Gui-macos-arm64.zip` — a proper `Ardon-R2.app` bundle
  (Info.plist + generated `.icns`), zipped with `ditto`. Unsigned for
  now (Gatekeeper override documented in the bundled README-FIRST).
- The GUI jobs always build, package, and upload a workflow artifact;
  they attach to the GitHub Release only on a tag push — so a manual
  `workflow_dispatch` run verifies both platforms compile without
  cutting a release.

No engine/library code changed — packaging and CI only.

## v0.2.1 (released June 2026)

### Runtime-swappable BLAS (DLL dispatch) — all pure-Rust

`r2-linalg` is now the **reference kernel** that can hand off to a
faster build of the *same pure-Rust kernel* at runtime, without
rebuilding the launcher. R2 stays strictly pure-Rust — the variants
are R2's own code compiled with different CPU-SIMD targets, not
external C/Fortran BLAS.

- **Stable C-ABI surface** (`blas_abi.rs`): `r2_dgemm` is exported
  from `r2_linalg.dll` with a plain C signature (flat `f64` buffers +
  integer dims). Rust has no stable ABI, so even Rust→Rust across a
  runtime-loaded `cdylib` must use a C boundary — that's the only
  reason it's C; both sides are Rust.
- **Runtime dispatch** (`blas_dispatch.rs`): set the `R2_BLAS`
  environment variable to a shared library exporting `r2_dgemm` and
  matrix multiply (`%*%`) routes through it; unset/missing/unreadable
  falls back to the built-in pure-Rust kernel. Resolved once, cached
  for the process.
- Lays the groundwork for the planned installer-time CPU dispatch
  (`r2_linalg_avx2.dll` / `_avx512.dll` / `_scalar.dll`).
- Reference kernels stay pure-Rust; only the opt-in dispatch path
  links `libloading`.

Verified: `A %*% B` produces identical results on the static and
DLL-loaded paths.

## v0.2.0 (released June 2026)

### Headline — native RGui-style desktop GUI

A from-scratch GUI framework (`r2-ui`: winit + wgpu + fontdue) replaces
the earlier eframe/egui experiment and powers a new desktop app:

- MDI desktop with floating **R2 Console** and **R2 Graphics**
  sub-windows: drag/resize on all 4 edges + 4 corners, traffic-light
  buttons, maximize/restore, two-level faint-blue active/passive title
  strips, title-bar logo, per-window menu sets, right-click context
  menus, vertical + horizontal scrollbars.
- **Multiple graphics devices**: `dev.new()`, `dev.set()`,
  `dev.list()`, `dev.cur()` — one native sub-window per device.
- Plot text rendered via a **fontdue overlay** (Console-quality
  crispness in the plot pane; SVG text no longer rasterized soft by
  resvg). **Copy plot as image** (clipboard bitmap), **Copy plot SVG**,
  native **Save plot** dialog (SVG/PNG at window resolution).
- **R-style colour helpers**: `rgb`, `gray`/`grey`, `hsv`, `rainbow`,
  `heat.colors`, `terrain.colors`, `topo.colors`, `cm.colors`,
  `adjustcolor`. `col=` / `border=` now honored on `hist`, `boxplot`,
  `barplot`; 4 % axis padding; Arial/crisp-edge SVG chrome.
- Universal paste sanitizer (smart quotes, em-dashes, mixed line
  endings → clean text); multi-line paste flows through the
  ConsoleBuffer continuation logic.

### Engine modularization

`r2-engine/src/lib.rs` reduced from **6,243 → 2,402 LoC**, with builtins
distributed across 12 `builtins/*.rs` modules plus `registry.rs`,
`formula.rs`, `na_bitmap.rs`. `r2-stats/time.rs` split into
`time/{mod,hindu}.rs`; `r2-stats/multivariate.rs` and `r2-ml/dispatch.rs`
also split along clean domain seams. See `docs/BIG_FILE_AUDIT.md`.

### Distribution

`R2-Setup-0.2.0.exe` ships `r2.exe` (CLI) + `R2Gui.exe` (R2-UI desktop),
Start-menu entries for both, GUI launched by default.

### Phase R.S.2 — MANOVA (Multivariate ANOVA)

`manova(formula, data)` now performs full multivariate ANOVA. The
formula's LHS is a multivariate response (typically built via
`cbind(y1, y2, ...)`); the RHS is the grouping factor.

Computes E (within-group SSCP) and H (between-group SSCP), then
extracts eigenvalues of E⁻¹H via QR iteration. Reports all four
classical test statistics in one table:

- **Wilks' Lambda** Λ = ∏ 1/(1+λᵢ)              (primary, F-approximated)
- **Pillai's trace** V = ∑ λᵢ/(1+λᵢ)
- **Hotelling-Lawley** = ∑ λᵢ
- **Roy's largest root** = λ₁

F-approximations are computed via Rao's standard formulas for all
four statistics — not just Wilks. Each statistic now reports its
own value, F-stat, (df₁, df₂), and p-value. Eigenvalues of E⁻¹H
are computed via the Cholesky-symmetrized path (E = LLᵀ then
`dsyev` on the symmetric L⁻¹HL⁻ᵀ) — machine-precision regardless
of p, as opposed to the earlier QR-iteration approach.

The output also includes a **situational-awareness block**:

- The four eigenvalues of E⁻¹H, so the user can see which dimensions
  drive the effect.
- A suggested primary statistic, chosen from the design:
  - Pillai when n is small relative to p (most robust)
  - Wilks or Pillai when s = 1 (algebraically equivalent)
  - Roy when one eigenvalue dominates (effect concentrated in one
    dimension)
  - Pillai or Wilks otherwise (diffuse effect)
- A significance summary, with a CAUTION line when the four tests
  disagree (a real diagnostic for assumption violations).

Hand-verified on iris MANOVA (n=150, k=3, p=4):

| Statistic | R | R2 |
|---|---|---|
| Pillai's trace | 1.192, F=53.5, df=(8,290) | 1.187, F=52.9, df=(8,290) |
| Wilks' Lambda  | 0.0234, F=199.1, df=(8,288) | 0.0239, F=195.6, df=(8,286) |
| Hotelling-Lawley | 32.48, F=580.5, df=(8,286) | 32.06, F=573.1, df=(8,286) |
| Roy's largest root | 32.19, F=1166.7, df=(4,145) | 31.78, F=1152.1, df=(4,145) |

All conclusions identical (p < 2e-16). Differences are well within
F-approximation tolerance.

Sample script: `samples/test_manova_accuracy.r2`.

### Phase R.S.2 — Hotelling T² (one-sample, two-sample, paired)

New `crates/r2-stats/src/multivariate.rs` (~480 LoC) implements
Hotelling's T² in three flavors via a single dispatcher
`hotelling.test(...)`:

- **One-sample**: `hotelling.test(X)` or `hotelling.test(X, mu=c(...))`
  tests H₀: μ = μ₀ for a p-dimensional mean vector.
- **Two-sample**: `hotelling.test(X, Y)` tests if two p-dimensional
  group means differ. Uses pooled covariance assuming equal Σ.
- **Paired (repeated-measures)**: `hotelling.test(X, Y, paired=TRUE)`
  is the multivariate generalization of the paired t-test — runs
  one-sample T² on the per-subject difference matrix D = X − Y
  against μ₀ = 0.

Math is exact to machine precision for T², F, df. P-value uses the
Wilson-Hilferty F→z approximation (same as the existing aov path);
this is accurate for moderate n (≥10) and is within a factor of ~2
of R's exact F-CDF at very small df. Hand-verified test case:
n=4 paired subjects, p=2 measurements, expected T²=318 and F=106 —
R2 produces exactly those values.

Five new unit tests in `multivariate::tests` (`hotelling_one_sample_*`,
`hotelling_two_sample_*`, `hotelling_paired_*`). Sample script at
`samples/test_hotelling_accuracy.r2` for end-to-end accuracy
verification against hand-computed expected values.

### Phase R.S.1 — Repeated-measures aov() with Error(...) syntax

New `aov(y ~ treatment + Error(subject), data=df)` implements
classical one-way within-subject ANOVA:

- New engine helper `split_error_term` lifts `Error(...)` out of the
  predictor expansion and tags the stratum as `~error` in the formula
  list. Nested `Error(subject/treatment)` collapses to outer-stratum
  semantics (the one-way RM case).
- New stats function `aov_repeated_measures` decomposes total
  variance into between-subject (SS_subject), treatment (SS_treatment),
  and within-subject residual (SS_within). F-statistic and p-value for
  the treatment effect via the within stratum. Output matches R's
  `summary(aov(... + Error(...)))` two-stratum layout exactly when
  R uses `factor(subject)`.
- TypeInstance return carries the full numeric panel (ss.subject,
  ss.treatment, ss.within, df.*, ms.*, f.statistic, p.value,
  n.subjects, n.treatments) for programmatic access.

`t.test()` extended with two formula-shaped paired-test shortcuts
R itself does not support:

- `t.test(y ~ treatment + Error(subject), paired=TRUE, data=df)` —
  Error(subject) acts as the `id=` argument for the existing
  paired-by-id pipeline. R rejects this syntax outright.
- `t.test(y ~ Error(subject), paired=TRUE, data=df)` — when no
  treatment grouping is on the RHS, pair observations by subject in
  row-of-appearance order. Each subject must have exactly two
  observations; the first becomes "obs1" and the second "obs2".

Teaching-style errors fire when:

- The formula has only `Error(X)` with no fixed effect
  (`aov(y ~ Error(drug))` — explains the user-of-Error confusion).
- The Error stratum equals the fixed effect
  (`aov(y ~ drug + Error(drug))` or `aov(y ~ drug + Error(drug/subject))` —
  identifies the inversion mistake).
- `t.test(y ~ Error(subject))` without `paired=TRUE`.
- A subject in the pair-by-row-order path has !=2 observations.

Three unit tests added: `rm_aov_matches_hand_computation_for_5x2_design`,
`rm_aov_errors_when_fewer_than_2_subjects`,
`rm_aov_errors_on_mismatched_lengths`. Hand-verified against R 4.5.3
output: F-statistic, all sums of squares, and degrees of freedom match
bit-identically when R uses `factor(subject)`.

---

## v0.1.1 (2026-05-18)

### Phase R.G.2 — Built-in HTTP plot viewer with session gallery (latest)

The graphics device is now paired with a tiny built-in HTTP server
(`crates/r2-graphics/src/server.rs`, ~290 LoC, `std::net` only — zero
new dependencies). Calling **`dev.view()`** starts the server on
`127.0.0.1:8765` (scans 8765–8775 if the first port is in use) and
opens the user's default browser via OS shell-out (`cmd /c start`,
`open`, or `xdg-open` per platform). Subsequent `plot()` / `hist()` /
`boxplot()` / `barplot()` / overlay calls render through the device,
and the browser tab live-updates.

The viewer page is a self-contained two-pane layout — no external
CSS, no CDN, no JS framework:

- **Top pane** — the "current" plot. Auto-refreshes from
  `/current.svg` every 1.5 seconds; users see plots appear as they
  run them in the REPL.
- **Bottom pane** — "Session gallery". A grid of thumbnails for every
  `.svg` file in the working directory, sorted newest first. Rebuilt
  every 2 seconds from a new `/list` JSON endpoint. **Clicking any
  thumbnail pins the top pane to that file**; a "return to live"
  link resumes auto-refresh. This fixes the UX gap where earlier
  plots seemed to "vanish" — they were always saved on disk, but the
  viewer previously only showed `/current.svg`.

Server endpoints:

| Endpoint | Behavior |
|---|---|
| `GET /` | The two-pane HTML page |
| `GET /current.svg` | Most recently modified `.svg` in cwd |
| `GET /list` | JSON `[{name, mtime}]` of every `.svg` in cwd |
| `GET /<name>.svg` | Any `.svg` in cwd by name (path-traversal guarded) |

A new general-purpose builtin **`readline(prompt="")`** was added to
`r2-engine` so scripts can pause for user input. It returns the typed
line as a character scalar without the trailing newline. Used by
`samples/demo_graphics.r` to give the user full control over pacing
and filenames during the interactive demo (every plot waits, prompts
for a save name, and waits again before advancing).

Limitations of v0.1.1:

- The server thread is a daemon. In script mode the process exits
  when the script returns, killing the server. Scripts that use
  `dev.view()` should end with a `readline()` or `Sys.sleep()` block.
  REPL mode keeps the process alive naturally.
- Browser polls every 1.5 s; no WebSocket push. Good enough for
  interactive use, not for animation.
- No native window — uses the user's existing browser. A real native
  GUI window via `winit` + `tiny-skia` is a v0.2.0 candidate.

### Phase R.G — In-memory graphics device + full `par()`

The `r2-graphics` crate has been re-architected around a thread-local,
in-memory `PlotDevice` (`crates/r2-graphics/src/device.rs`). The
previous file-state model (which detected "is a plot open" by reading
`plot.svg` from cwd) was racy under cargo's parallel test execution
and limited `r2-graphics` to a single plot at a time. The new model:

- **`PlotDevice`** holds the accumulated SVG body, the canvas size,
  full `PlotParams`, and a panel cursor for multi-panel layouts.
  Source of truth for the "is a plot open" predicate.
- **`begin_plot()`** is called by every primary plot (`plot`, `hist`,
  `boxplot`, `barplot`) to obtain the rectangle it should draw into,
  honoring `par(mfrow=...)` / `par(mfcol=...)` automatically. Overlays
  (`lines`, `points`, `abline`, `legend`) call `append_svg()` which
  errors cleanly when no plot is open.
- **Auto-flush preserved**: each plot still writes `plot.svg`,
  `hist.svg`, etc. to the working directory after drawing — legacy
  user-facing UX is unchanged.

`par()` now ships as a full builtin with three call shapes:

```r
par()                        # snapshot — returns all params as a named list
par("col")                   # query a single param
par(col="red", lwd=2)        # set; returns previous values for restore
oldpar <- par(mfrow=c(2,2))  # save-and-set idiom; par(oldpar) restores
```

Supported parameters: `mfrow`, `mfcol`, `mar`, `oma`, `cex`, `cex.axis`,
`cex.lab`, `cex.main`, `col`, `bg`, `fg`, `lty`, `lwd`, `pch`, `las`,
`new`. Defaults match CRAN R 4.5.x. Multi-panel `mfrow=c(2,2)` /
`mfcol=c(2,3)` is implemented end-to-end: the panel cursor advances on
each `plot()` call, all four panels render into the same SVG canvas,
and the cursor wraps cleanly when overflowed.

Two device-control builtins added:

- **`dev.off()`** — close the current device and reset to defaults.
- **`save_plot(path)`** — explicitly flush the device's SVG to a file
  (useful when users want a name other than `plot.svg`, or want to
  capture a multi-panel canvas after several `plot()` calls).

The previously-flaky `lines_errs_when_no_plot_open` test is no longer
ignored — it now relies on the in-memory `has_plot` flag instead of
filesystem state and passes deterministically across all platforms
regardless of cargo's test-parallelism order.

### Phase R.M — Cranelift JIT aarch64 gate

`r2-jit` now exposes a compile-time constant `JIT_SUPPORTED` (true on
`x86_64`, false elsewhere) and `try_compile_closure` returns `None`
early on unsupported targets. The engine falls back to the interpreter
on Apple Silicon and ARM Linux without ever touching Cranelift's PLT
path (which panics on aarch64 in `cranelift-jit 0.105`). Statistical
outputs remain bit-identical across architectures; only wall-clock
performance differs. The JIT test module is gated to
`target_arch = "x86_64"` so CI on aarch64 hosts is clean. Lifting the
gate is a v0.2.0 task (upgrade Cranelift to a version with aarch64
PLT support). macOS (Apple Silicon, `macos-latest`) is restored to
the CI matrix as a fully tested platform.

### Phase D.1 — `.r2d` native binary dataset format

New `r2-base/src/r2d.rs` defines a compact little-endian binary format
for built-in data.frames: magic `R2D1`, u16 version, u32 n_cols, u32
n_rows, typed columns (Numeric/Integer/Logical/Character) with validity
bitmaps. Pure-std implementation — no new external crates.

The five inline datasets (`iris`, `mtcars`, `airquality`,
`ToothGrowth`, `faithful`) moved out of hand-coded Rust arrays and
into `crates/r2-base/datasets/*.r2d` files (totalling ~18 KB),
loaded via `include_bytes!` and parsed on first call. `r2-base/src/lib.rs`
dropped from **362 → 157 lines** (–57%). Every canonical-R integrity
test still passes (`iris_column_sums`, `iris_row_spot_check`,
`mtcars_column_sums`).

Future work: extend the loader to recognize R's native `.rda` (gzip+XDR)
header so users can drop CRAN-format saves straight into the global env.

### Phase S.1 — Formula data scope + factor expansion in lm()/glm()

Two related fixes uncovered while running `lm(Sepal.Width ~ Species, data = iris[1:100,])`:

1. **`model_matrix_expand()`** — new helper in `r2-stats/src/models.rs`.
   Character and factor predictor columns now expand into k-1 dummy 0/1
   columns using treatment contrasts (first observed level absorbed into
   the intercept). Dummy column names follow R's convention
   `{base}{level}` (e.g. `Speciesversicolor`). Wired into all three
   `bi_lm` paths (named formula, bare-RHS formula, two-vector legacy)
   and the matching path in `bi_glm`. Previously these died with
   `cannot convert character to numeric`.

2. **Formula data-scope for Call/Index/Binary expressions** —
   `Engine::resolve_formula_term` now pushes the data.frame's columns
   onto the local-scope stack for the duration of non-trivial RHS
   evaluation. Fixes `lm(y ~ factor(x), data = df)`,
   `lm(y ~ log(x) + I(z^2), data = df)`, etc. — previously the bare
   names inside the call expressions failed to resolve against the
   data argument.

Coefficients verified bit-identical to CRAN R 4.5.3 for
`lm(Sepal.Width ~ Species, data = iris[1:100,])`:
intercept = 3.428, Speciesversicolor = -0.658.

### Phase F.7 — Single-precision (f32) opt-in storage

New `as.single(x)` builtin coerces a numeric vector to **f32 storage**
— half the memory of f64 (4 bytes/elem vs 8). The new `RVal::Single`
variant follows NumPy-style dtype promotion: `Single op Single → Single`
(stays f32), any mixing with `Numeric`/`Integer`/`Logical` promotes
back to f64 and produces `Numeric`.

```r
x <- as.single(c(1.5, 2.5, 3.5))    # 12 bytes instead of 24
class(x)                             # "single"
is.single(x)                         # TRUE

x + as.single(c(0.1, 0.2, 0.3))     # Single (stays in f32)
x + c(0.1, 0.2, 0.3)                 # Numeric (promoted to f64)
```

When to use it: large numeric vectors where memory pressure matters
more than the precision loss. Stats workloads requiring full f64
precision (lm coefficients, eigendecomposition, log-likelihoods)
should stay on `numeric` — the f32 conversion would silently degrade
accuracy.

What ships:
- `r2_arrow::ColumnarF32` with full API parallel to `ColumnarF64`
- `r2_types::Singles` dual-storage wrapper (f32 columnar + lazy boxed view)
- `RVal::Single(Singles, Attrs)` variant
- `as.single()` / `is.single()` builtins
- Promotion logic in `binary_op`
- 6 new ColumnarF32 unit tests + working end-to-end demo

### Phase C.9 — Fused map-reduce JIT

`function(x) sum(f(x))` and `function(x) prod(f(x))` shapes now compile
to a **single fused Cranelift loop**: load `x[i]`, apply `f`, accumulate
into a running sum/prod, repeat. No intermediate vector materialised.

**Bench impact** (1e7-element vector, `function(x) sum(sqrt(x*x + 1))`):

| Path | Time | Notes |
|---|---:|---|
| **Fused closure (Phase C.9)** | **0.06s** | Single JIT loop, no intermediates |
| Unfused inline `sum(sqrt(v*v + 1))` | 0.66s | Engine materialises each intermediate |

Same bit-identical result (13,545,708.6748). **11× faster** when written
as a closure. The inline form still hits the slower path because the
engine doesn't do AST-level operator-fusion pre-analysis yet — that's
a separate, bigger architectural piece for a future version.

The fused path supports arbitrary multi-block inner bodies (branchy
code with `if`/`else` via Phase C.5 still works inside `sum(...)`)
and uses the same math-extern lowering as Phase C.6 (so `sin`, `cos`,
`log`, `exp` etc. work inside the fused body).

### Phase M.1 — F64ScratchPool (r2-memory)

Per-thread recyclable `Vec<f64>` pool. Bucket-organised by power-of-two
capacity (16 elements through 256M); per-bucket cap of 4 buffers
prevents unbounded growth.

Public API: `scratch_acquire(min_cap)`, `scratch_release(buf)`,
`with_scratch(min_cap, |buf| ...)`, `scratch_stats()` for diagnostics.

Wired into:
- `r2_kernel::reduce` (ReduceOp::Median path)
- `r2_kernel::reduce_strided` (Median path)
- `r2_kernel::nth_smallest`
- `r2_stats::summary::bi_quantile`

**Honest scoping**: the pool eliminates allocator overhead (~1-2ms per
medium buffer) but doesn't show big single-call wins because the work
done with the buffer (sort, partial sum, etc.) dominates. Real wins
emerge in tight loops doing the same op on the same data size — e.g.
bootstrap resampling, repeated quantile computation, k-means distance
per iteration. Pool stats confirmed: 11 hits + 1 miss over 12 acquires
of the same size.

### Phase B.1 — Closure capture inference

User closures that reference variables from their enclosing scope now
JIT to native code instead of falling through to the tree-walking
interpreter, **as long as the captured values are numeric scalars**.

```r
scale <- 2.5
f <- function(x) x * scale + 1
f(big_vec)   # before v0.1.1: tree-walks each element
             # v0.1.1+:       JITs via VectorMap with 2.5 and 1 baked in
```

Implementation: at JIT-compile time, `r2_ir::collect_free_vars` walks
the body's AST to find free symbols; `r2_ir::substitute_constants`
returns a fresh AST tree where each free symbol bound to a numeric
scalar in the closure's captured env is replaced by an `Expr::NumLit`.
The substituted body then goes through the existing JIT lowering paths
identically to literal-constant code. No new ABI surface, no per-call
capture passing — partial evaluation at compile time.

Non-scalar captures (vectors, lists, closures) still fall through to
interpreter for now; they need either a different ABI surface
(per-call extra-arg passing) or further specialization to be JIT'd.

### Phase K.7–K.11 — Kernel layer expansion

(See v0.1.0 entry below for the kernel-layer narrative; the work
shipped alongside the capture inference in this version.)

Test count: 247 → 278 (+31 across kernel expansion and capture
inference). All passing.

## v0.1.0 (2026-05) — First Stable Release

### Kernel-layer expansion — Phases K.7–K.11 (latest)

Five new kernel-layer op families closing the long-standing
"kernel-shaped-but-builtin-implemented" gap:

- **Phase K.7 — Scan**: `cumsum` / `cumprod` / `cummax` / `cummin` now
  route through `r2_kernel::scan` with Oracle-driven Serial-vs-Rayon
  dispatch. Rayon backend uses two-pass parallel scan (Blelloch). Bonus
  correctness fix on `cummax` / `cummin`: NA now properly propagates
  forward (previous impl silently skipped past the first None).
- **Phase K.8 — Select**: `which_max` / `which_min` (NA-aware index),
  `nth_smallest` (quickselect, O(n) avg), `top_k` / `bottom_k`
  (heap-based, O(n log k)). User-facing builtins `which.max` / `which.min`
  now route through kernel.
- **Phase K.9 — Rolling**: sliding-window `sum` / `mean` / `max` / `min` / `sd`.
  Sum/Mean use incremental update; Max/Min use deque-based O(n) algorithm.
  New user-facing builtins: `rollsum` / `rollmean` / `rollmax` / `rollmin` / `rollsd`.
- **Phase K.10 — Hash aggregation**: `hash_agg(op, keys, values)` for
  group-by reductions in O(n); `hash_tabulate(keys)` for `table()`-style
  counts. Replaces O(n²) linear-scan loops in builtin code.
- **Phase K.11 — Distance kernels**: `Euclidean` / `Manhattan` / `Cosine`
  with `distance` (pair) and `pairwise_distance` (n×n matrix). NA-aware,
  symmetric, parallel via `par_for_rayon` for n≥16. Shared kernel for
  k-means / knn / hierarchical clustering.

Test count: 247 → 276 (+29 kernel tests). All passing.

### M-R2-JIT + L-R2-Dispatch (2026-05-17)

**M-R2-JIT** (math-extern JIT call lowering, Phase C.6): user closures
whose bodies include `sqrt`, `abs`, `exp`, `log`, `sin`, `cos`, etc. now
compile end-to-end to native machine code. No bytecode VM, no per-call
interpreter checkpoint. 24 math functions supported (8 dispatch to native
Cranelift hardware instructions, 16 to Rust-call wrappers — pure Rust
with `extern "C"` ABI for predictable Cranelift dispatch; not OS-level
FFI). Math comparison vs R 4.5.3
default Rblas on 1e6-element vectors: R2 beats R on 3 of 5 idioms,
up to 4.8× faster on multi-call fused bodies (`sin²+cos²`).

**Phase C.7** — extended JIT coverage to 2-arg closures with arbitrary
multi-block bodies. `function(x, y) sqrt(x*x + y*y)`-shape functions
now JIT (previously fell back to interpreter). 2.6× speedup on the
benchmark.

**L-R2-Dispatch** (Phase L.1, list-aware auto-parallel): new
`Op::ListMap` in r2-oracle, new `list_meta()` in r2-types,
`map_items` in apply-family now computes per-item work units rather
than item count. Fixed a real bug: `lapply` on a 3-component list of
1M-element vectors was staying serial because `3 < 50K threshold`;
now sums aggregate work (3M) and parallelizes. New user-facing
`list.meta()` builtin exposes the metadata to R2 scripts.

### Performance — F.3 native-columnar storage

R2 now beats default-Rblas R on linear regression (2× faster) and matrix
multiply (1.4× faster), is within 2× on most other workloads.

- **`Reals` dual-storage**: data can be held as `Vec<Option<f64>>` *or*
  `Arc<ColumnarF64>` (or both), with lazy materialisation either way.
  Producers of dense f64 (`rnorm`, `runif`, the engine binary fast path)
  now never materialise the `Vec<Option<f64>>` if no caller asks for it.
- **Columnar-aware reductions**: `sum` / `mean` / `min` / `max` route
  through `ColumnarF64` native methods on cached `&[f64]` slices —
  closes the 7.5× gap on `sum_mean_1e7` down to 1.6×.
- **Engine binary fast path**: for `Numeric op Numeric` with same length
  ≥ 64, routes through `ColumnarF64::binary` instead of per-element
  `Option<f64>::match` — closes the 30× gap on `vec_add_1e7` down to 4.2×.
- **Dataset integrity guard tests**: 3 new tests in `r2-base` assert
  iris / mtcars column sums + spot rows match canonical R. Caught and
  fixed a previous Petal.Length / Petal.Width transcription error in
  the iris dataset (~30 row positions had been wrong) — that bug had
  been silently corrupting any `cor()`, `kmeans`, `eigen` etc. call
  that touched those columns.

### Other v0.1.0 highlights (pre-F.3)

This version closes every Tier 0–4 roadmap item from the original
dependency map, except items explicitly out of scope (GPU, closure
JIT, full LAPACK `dbdsqr`, autograd) or intentionally bundled into
Phase G hardware-awareness work (Oracle calibration).

### Numerical correctness — major

- **Full thin SVD** with orthogonal factors: `svd(M)` now returns `$d`,
  `$u`, `$v` (R convention). New `r2_linalg::dgesvd_full(m, n, A) →
  (σ, U, Vᵀ)` via Householder bidiagonalization with `dorgbr`-style
  reverse application of stored reflectors, diagonalization via Bᵀ·B
  through the already-shipped `dsyev_full`. Honest κ-dependent accuracy
  note in `KNOWN_LIMITATIONS.md`.
- **`dsyev_full`** for symmetric eigendecomposition: Householder
  tridiagonalization + implicit Wilkinson-shift QR + back-transform.
  `eigen()` now returns real `$vectors`; `prcomp()$rotation` is real.
- **Welch–Satterthwaite df** for two-sample `t.test` (was silently using
  pooled Student df — real statistical bug).
- **Exact hypergeometric `fisher.test`** (was χ² approximation).
- **R-correct NA semantics for `&` / `|`** elementwise on logical
  vectors: previously had no handler in `binary_op` and was silently
  returning all-zero numeric (real lurking bug).
- **glm full diagnostics**: null deviance, AIC, Fisher iterations,
  dispersion, std errors / z values / p values.

### New language capabilities

- **NSE wiring** for `subset(df, cond)` and `transform(df, name = expr)`:
  expressions evaluate in a child env that binds df columns.
- **Formula syntax** `t.test(x ~ y)` with group labels in output.
- **Paired t-test** with Pearson r in output.
- **`id =` named arg** on `t.test` for within-subject auto-pairing.
- **R-style `t.test` output**: data/CI/alternative-hypothesis/sample-estimates layout.
- **`$call` capture** on `lm`/`glm`/`aov` so `summary(fit)` shows the
  actual call instead of `lm(formula)`.

### JIT compiler

- **Branchy multi-block IR support** (Phase C.5) — `function(x) if (x > 0) x else -x` mapped over a vector now JITs end-to-end.
- **`VectorTernaryMap` ABI** for 3-column `ifelse`-shape closures.
- **Zero-copy bridge** between JIT and columnar storage; output reconstruction via input bitmap (NA structure preserved exactly).

### Memory + kernel layer

- **F.3/F.6 columnar storage migration**: `RVal::Numeric/Integer/Logical` now carry `OnceLock<Arc<ColumnarT>>` caches. Packed-bit `ColumnarBool` is ~64× smaller in memory than `Vec<Option<bool>>`.
- **F.4 element-wise columnar binary kernels** + **F.5 mmap-backed `MmapColumnar`** with zero-copy `&[f64]` view.
- **Phase K.5 `TernaryOp::MulAdd`** (uses `f64::mul_add` so single rounded op on FMA-capable hardware).
- **Phase K.6 `reduce_strided`** for zero-copy reductions over non-contiguous matrix rows.

### I/O + strings

- **RFC 4180 CSV state-machine parser**: embedded separators, doubled quotes, multi-line fields, UTF-8 BOM stripping.
- **`regex-lite`** (pure-Rust POSIX-ERE) behind default-on `regex` feature for `grep`/`grepl`/`gsub`/`sub`/`regexpr`; `fixed=TRUE` forces literal.

### Architecture / engine

- **R.11 model split-handler**: `lm`/`glm`/`aov`/`anova` data path lives in `r2_stats::models`; engine retains 1-line delegators + `summary()` formatter.
- **R.12 RNG consolidation**: all six random-variate builtins share `r2_stats::rng::SEED_STATE` so `set.seed()` is genuinely reproducible across the family.
- **Engine line count**: 7,282 → ~4,860 (-33%) from R.11/R.12 migrations.

### Tier 0 bug fixes shipped this version

- `matrix(data, nrow, ncol)` positional args now honoured (was reading nrow/ncol from keyword form only — silently produced wrong shape).
- `kmeans()` initialization now uses evenly-spaced rows + recomputes centroids/sizes before convergence check (was collapsing to single cluster, never recomputing sizes).
- `rep()` works for character/integer/logical and supports `each =`.
- `factor()` accepts numeric/integer/logical (coerces to string).
- `data.frame(y, x1, x2)` — bare-symbol args become column names automatically.
- `binomial()`, `gaussian()`, `poisson()` family constructors for `glm(family = binomial())`.

### Tests

168 → **233 passing**. Build clean.

---

## v0.0.9 (2026-04-26) — Initial Launch Release

### Core Language
- 192 built-in functions
- 9,853 lines of Rust across 11 crates
- Both `<-` and `=` assignment
- 1-based indexing, formula syntax `y ~ x1 + x2`
- Pipe operator `|>`, f-strings `f"hello {name}"`
- Lambda `\(x) x^2`, R2> prompt
- .Internal() bridge — users write functions in R2 syntax

### Statistics
- `lm()` with std.errors, t-values, p-values, F-statistic, significance stars
- `glm()` — binomial, Poisson, Gaussian families
- `t.test()`, `chisq.test()` (with Yates' correction), `cor.test()`
- `aov()`, `anova()` — Analysis of Variance
- `shapiro.test()` — normality test
- `wilcox.test()` — non-parametric test
- `fisher.test()` — exact test for 2x2 tables
- `weighted.mean()`, `IQR()`
- Distribution functions: rnorm, runif, rbinom, rpois, dnorm, pnorm, qnorm

### Machine Learning (12 algorithms, all built-in)
- `rpart()` — decision tree (CART)
- `rf()` — random forest (Rayon parallel, 2.3x faster than R)
- `gbm()` — gradient boosted trees (3 loss functions)
- `kmeans()` — K-means clustering
- `knn()` — K-nearest neighbors
- `naive.bayes()` — Gaussian naive Bayes
- `prcomp()` — PCA
- `svd()`, `eigen()`, `scale()`
- `cv()` — K-fold cross-validation
- `confusion.matrix()` — with precision/recall/F1

### Math Kernel (r2-linalg, pure Rust)
- 1,278 lines, Rust-only dependencies, no C/C++
- BLAS Level 1-3 with 8x4 micro-kernel, cache blocking
- 2.2x faster matrix multiply than R (Windows default BLAS)
- LU, Cholesky, QR, SVD, Eigenvalues (Jacobi)
- Fused least-squares solver, Cramer 2x2/3x3

### Rayon Parallelism
- Random Forest tree building on all CPU cores
- Thread-safe RNG (AtomicU64)
- Automatic core detection, zero configuration

### Data Handling
- `read.csv()` / `write.csv()` — quotes, NA, type inference
- `filter()`, `select()`, `arrange()`, `mutate()`
- `save()` / `load()` — .r2s (session), .r2d (data), .r2m (model)
- 5 built-in datasets: iris, mtcars, airquality, ToothGrowth, faithful

### Graphics
- `plot()` with model auto-dispatch (lm→residuals, gbm→loss curve)
- `hist()`, `boxplot()`, `barplot()` — SVG output

### System
- `library()` / `detach()` — package system
- `help()`, `?topic`, `??topic` — 52 help topics
- `version()` — shows core count, platform, license
- Crash-proof REPL with catch_unwind
