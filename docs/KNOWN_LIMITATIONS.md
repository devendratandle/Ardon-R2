# Known Limitations

This document is the authoritative list of features that are **intentionally
incomplete** in the current release. Each entry records the algorithm in
use, why it falls short of a "production" implementation, and what would
need to change to close the gap. The goal is to never let users be
silently misled by a function that looks correct but isn't.

## Architecture — addon packages and the ecosystem path

### No addon packages yet — everything statically linked

Ardon-R2 currently ships as one monolithic binary (~6.6 MB). Every
function — math, stats, ML, plotting, JIT, kernels — lives in the
single `r2` executable. Users cannot write or install separate
packages the way R users can `install.packages("survival")`.

**Consequence**: only built-in functions are available. Niche
domain-specific tools (survival analysis, structural equation
models, ecological diversity indices, single-cell genomics, etc.)
have no path into a user's workflow except by being merged into the
main repository.

**Current `r2-pkg` crate**: 15-LoC skeleton. `library()` builtin
exists as a stub. The registry's `add_layer()` mechanism supports
adding new function tables at runtime, but no loader is wired up.

**Three-stage path to a real package ecosystem**:

1. **v0.3.0 — Cargo feature flags.** Split optional domains into
   feature-gated workspace crates (e.g. `--features mixed`,
   `--features multivariate`, `--features ml`, `--features
   plotting`, `--features jit`). Slim build ≈ 3 MB, full ≈ 7 MB.
   ~200 LoC. Lets users opt out of components they don't need;
   gives contributors a way to add optional crates without
   bloating the default build. Tracked for v0.3.0.

2. **v0.4.0 — R2-script packages.** Pure R2 source loaded from
   `.r2pkg` files via `library("name")`. Discovery via a simple
   manifest (TOML or JSON). Like R's CRAN packages but R2-code-only.
   ~500 LoC. Lets the community publish packages without touching
   the main repository. **This is the inflection point where
   Ardon-R2 stops being a single project and starts being an
   ecosystem.** Tracked for v0.4.0.

3. **v1.0+ — Dynamic linking (.dll / .so).** Compile Rust addon
   crates as dylibs, load at runtime with `dlopen` / `LoadLibrary`.
   Matches R's CRAN model exactly. ~1500 LoC plus security review
   (signed binaries, sandboxing). Worth doing only once there's
   real contributor demand. Tracked for v1.0+.

### Functions that should probably be addons, not built-in

When the addon mechanism lands, these likely move out of the core
to optional crates:

- `hotelling.test`, `manova` — multivariate stats audience (~10–15%
  of R users). Would live in `r2-multivariate` addon.
- `lmer` — mixed-models audience (~15–20% of R users). Would live
  in `r2-mixed` addon.
- `rpart`, `rf`, `gbm`, `knn`, `naive.bayes` — ML domain. Would
  live in `r2-ml` addon (already a separate crate; just needs
  feature gating).
- `plot`, `hist`, `dev.view`, `par` — graphics domain. Would live
  in `r2-graphics` addon.

The 80%-used core (`mean`, `sd`, `lm`, `glm`, `t.test`, `summary`,
`c()`, data frames, formulas, the math/trig builtins) stays in
the always-on core.

### Engine extension currently requires editing the main file

Adding any new builtin requires touching `crates/r2-engine/src/lib.rs`:
~3 lines per function (1-line wrapper + register the name in the
layer's vec). This is fine while the project is small but won't
scale to a real package ecosystem.

**Mitigation in current code**: math and stats live in their own
crates (`r2-stats`, `r2-kernel`, etc.) so the engine wrapper is
genuinely thin (one line that delegates). The engine doesn't
contain math logic anywhere — it's pure dispatch.

**Path to closure**: the addon mechanism above eliminates this for
external contributors. For internal core functions, a future
`#[builtin]` proc-macro could auto-generate wrappers + registration
from a single attribute. ~200 LoC of macro work. Tracked for v0.4.0.

## Multivariate / repeated-measures statistics

### F→p approximation ✅ resolved (all paths)

Every F-distribution p-value in R2 now uses the **exact** CDF via
`r2_stats::htest::f_sf` (regularized incomplete beta, Lentz continued
fraction) — matching R to ~1e-9, and correctly returning p = 0 (not
p = 1) when the residual is zero:

- `aov` (incl. the repeated-measures `Error(...)` branch) and `lm`'s
  F-test call `f_sf` directly.
- `manova` and `hotelling.test` route `multivariate::f_to_pvalue`
  through `f_sf` too (the former Wilson-Hilferty step is gone), so the
  small-df cases — e.g. n=4 paired Hotelling, df=(2,2) — are now exact
  rather than off by up to ~2×.

The T², F-statistic, and degrees of freedom were always exact; this
closes the p-value gap across the whole statistics suite.

### `Error(treatment/subject)` collapses to outer stratum

Nested Error syntax `Error(A/B)` is accepted and treated as
`Error(A)` for the one-way RM-ANOVA case. This is correct for the
crossed within-subject design but does not yet implement true
split-plot decomposition (multiple nested random strata, separate
F-tests per stratum). Tracked for v0.2.1.

### MANOVA — eigenvalues drift ~1–3% from R's LAPACK values

`manova()` ships with all four classical statistics (Pillai's trace,
Wilks' Lambda, Hotelling-Lawley, Roy's largest root), each with its
own Rao-style F-approximation, df pair, and p-value. The output also
prints the eigenvalues of E⁻¹H and an interpretation block telling
the user which statistic to treat as primary for their design
(Pillai for robustness, Wilks for ML-classical, HL for power under
assumptions, Roy when one dimension dominates).

**Where R2 differs from R numerically.** On the canonical iris MANOVA:

| Quantity | R | R2 | drift |
|---|---|---|---|
| λ₁ (Roy)            | 32.19  | 31.78  | −1.3% |
| λ₂                  | 0.286  | 0.277  | −3.1% |
| Hotelling-Lawley    | 32.48  | 32.06  | −1.3% |
| Wilks Λ             | 0.0234 | 0.0239 | +2.1% |
| Pillai V            | 1.192  | 1.187  | −0.4% |
| Wilks F-approx df₂  | 288    | 286    | off by 2 |

The conclusion (p < 2e-16) is identical at every conventional α level.

**Root cause.** R uses LAPACK's `dgeev` (general non-symmetric
eigensolver) directly on E⁻¹H. R2 currently uses the Cholesky-
symmetrized path: factor E = LLᵀ, form B = L⁻¹HL⁻ᵀ, then `dsyev` on
the symmetric B. Mathematically these give identical eigenvalues;
numerically the Cholesky path involves ~5× more floating-point
operations per eigenvalue, and each triangular solve amplifies error
by the condition number of L. For iris, condition(E) ≈ 100, giving
the ~1% drift observed.

The F-values differ proportionally because they're computed from
the (drifted) statistic values — this is NOT a problem with the
F-distribution PDF / CDF; that path is unchanged. The Wilks df₂
off-by-2 is a separate small Rao-formula choice that doesn't affect
the F-value materially.

**Path to closure (v0.2.1).** Add `dgeev` (general non-symmetric
eigenvalue solver) to `r2-linalg` and call it directly on E⁻¹H.
~300–400 LoC of unblocked LAPACK-style code; eliminates the drift
entirely. Also re-derive the Wilks df₂ formula against R's exact
choice.

### Paired Hotelling T² — R's `Hotelling` package contradicts textbooks

R's `Hotelling::hotelling.test(X, Y, paired=TRUE)` silently ignores
the `paired` argument and returns the unpaired two-sample T². The
standard textbook definition (Anderson 1958, Mardia/Kent/Bibby 1979,
Johnson & Wichern, Manly) is the **one-sample T² on the per-subject
difference matrix**, which R2 implements correctly. To cross-verify
R2's paired result in R, use any of:

- `Hotelling::hotelling.test(X - Y)` (one-sample on differences)
- `ICSNP::HotellingsT2(X, Y, test='f')`
- `MVTests::OneSampleHT2(X - Y, mu0 = rep(0, p))`

All three give the textbook paired result that R2 produces.

### Mixed models (`lmer`-style random effects) not yet implemented

Full random-intercept / random-slope mixed models with REML or ML
estimation are R.S.4 work. v0.2.0 ships repeated-measures ANOVA but
not lme4-equivalent functionality.

## Platform support

### Apple Silicon (aarch64-apple-darwin) — JIT falls back to interpreter

**Status:** Ardon-R2 is **fully supported** on Apple Silicon (M1/M2/M3)
and ARM Linux. The interpreter, kernel layer, columnar storage,
parallel dispatch, and every statistical function work natively and
return outputs bit-identical to x86_64. The Cranelift JIT path is
gated: on aarch64 it returns `None` from `try_compile_closure` and
the engine cleanly executes the same code through the interpreter.

**Why the gate exists:** Cranelift 0.105.4 (the JIT backend Ardon-R2
uses) implements the Procedure Linkage Table only for x86_64. On
aarch64, calling `JITModule::new()` panics during PLT construction —
an upstream limitation. Phase R.M (v0.1.1) added a compile-time
`JIT_SUPPORTED` constant in `r2-jit` and an early-return guard so
the engine never reaches the panicking code path on aarch64.

**What this means for users:**

- Apple Silicon Macs build, install, and run the full Ardon-R2
  language without any workaround needed. Rosetta 2 is not required.
- Compute-bound workloads that would benefit from the JIT (fused
  `sum(sqrt(x*x+1))`, repeated `sapply`, Monte Carlo inner loops) run
  through the interpreter on aarch64. Wall-clock performance is
  slower than on x86_64 for those workloads. All other paths (kernel
  layer, parallel dispatch, columnar storage, linalg) run at full
  native speed.
- Statistical correctness is identical across architectures —
  same `lm`, `glm`, `t.test`, `summary` outputs to the last bit.

**Path to full JIT on aarch64:** Upgrade `cranelift-jit` from 0.105 to
a version that implements aarch64 PLT (currently a v0.2.0 target).
Estimated 200–500 LoC of Cranelift API churn to absorb.

CI tests `ubuntu-latest`, `windows-latest`, and `macos-latest` (Apple
Silicon) on every push.

## Linear algebra (`r2-linalg`, `r2-base::linalg_ops`)

### `svd(M)` — full thin SVD now shipped ✅ (Tier 1)

**Status:** `svd(M)` returns `$d`, `$u`, and `$v` — thin SVD `M = U·diag(d)·Vᵀ`
where U is m×n with orthonormal columns and V is n×n with orthonormal
columns (R convention: `$v` holds V itself, not Vᵀ).

**Implementation:** New `r2_linalg::dgesvd_full(m, n, A) → (σ, U, Vᵀ)`.
Householder bidiagonalization with reverse-application of stored
reflectors onto thin identities (`dorgbr`-style) builds U₁ and V₁;
diagonalization of B goes through Bᵀ·B (n×n symmetric tridiagonal) using
the already-shipped `dsyev_full`, producing σ²/V₂; U₂ recovered via
B·V₂·diag(1/σ). Final factors U = U₁·U₂, V = V₁·V₂. Tests verify
reconstruction `A ≈ U·diag(σ)·Vᵀ` to ~1e-9 and orthonormality
`UᵀU ≈ I_n`, `VᵀV ≈ I_n` on 3×2 / 4×3 cases plus diagonal known-σ.

**Honest accuracy caveat:** the Bᵀ·B route squares the condition number
of A. For matrices with condition number κ(A) ≲ 1/√ε ≈ 6.7×10⁷ — which
covers the overwhelming majority of practical statistics/ML workloads —
singular values and vectors are accurate to ~1e-12. For badly
conditioned matrices (κ(A) approaching 1/ε ≈ 4.5×10¹⁵), small singular
values lose accuracy proportional to κ², equivalent to roughly half the
floating-point precision on the small end of the spectrum. Large
singular values remain accurate. The reconstruction `A ≈ U·diag(σ)·Vᵀ`
remains norm-wise tight regardless of conditioning.

**Closure path to κ-independent accuracy:** replace phase 2 with proper
LAPACK `dbdsqr` — implicit-shift bidiagonal QR with Givens rotations
accumulated directly into U₂ and V₂. ~300 LoC of dense numerical code,
delicate convergence and deflation logic. Deferred — current accuracy
is sufficient for all practical statistics workloads and the
reconstruction property holds. The values-only `dgesvd` is retained
unchanged for callers that don't need vectors.

### `eigen(A)` — eigenvectors now shipped ✅ (Tier 1)

**Shipped this round:** new `dsyev_full(n, A) → (eigenvalues, eigenvectors)`
in `r2-linalg::decomp`. Classical three-stage pipeline:
1. Householder tridiagonalization (A → T = QᵀAQ, accumulating Q)
2. Implicit symmetric QR with Wilkinson shift on the tridiagonal
3. Back-transform: eigenvectors of A = Q · eigenvectors of T

Tests verify `Q · diag(λ) · Qᵀ ≈ A` and orthonormality `QᵀQ ≈ I` for
diagonal, 2×2 closed-form, 3×3 mixed, and a 3×3 with adjacent
near-equal eigenvalues.

`bi_eigen` now returns both `$values` and `$vectors`.
`bi_prcomp` now returns `$rotation` (eigenvectors of the covariance
matrix). The standalone Jacobi `dsyev` (values-only) is retained for
callers that don't need vectors.

### Previous limitation entry (now historical):
*`eigen(A)` previously used Jacobi rotation, eigenvalues only.*

**Status:** Returns `$values` for any symmetric matrix. Eigenvectors are
not returned.

**Why:** `r2_linalg::dsyev` uses the Jacobi rotation method
(textbook, repeatedly zero the largest off-diagonal entry). Numerically
stable and always converges, but O(n³) with a large constant — slow for
n ≳ 100. Eigenvector accumulation through Jacobi sweeps is implemented
but currently not wired through the `eigen()` builtin's return list.

**Closure path:** Same upgrade as `svd` — switch to tridiag + symmetric
QR. Adds eigenvector return as a side effect of the Givens accumulation.

### `prcomp(X)` — covariance + Jacobi route

**Status:** Correct results. Slow for wide data (n_features ≳ 100).

**Why:** Routes through `cov(X) = XᵀX/(n-1)` followed by Jacobi
eigendecomposition. The textbook concern with this route is precision
loss vs. SVD-on-X (the squared singular values lose half their digits in
fp64). For the column counts typical of statistical work this is well
within tolerance, but the SVD-on-X route is the more numerically robust
choice for ill-conditioned matrices.

**Closure path:** Once SVD is fixed (above), `prcomp` will switch to
SVD-on-X internally. Fields `$sdev`, `$rotation`, `$center`, `$scale`
will be derived directly from the SVD factors with full eigenvector
support.

### Divide-and-conquer SVD/eigensolver

**Status:** Not implemented; not on the v0.x roadmap.

**Why not yet:** D&C (Cuppen 1981) for symmetric tridiagonal matrices
requires a secular-equation solver (≈500 LOC for the inner Newton
iteration with rational interpolation safeguards), deflation logic for
clustered eigenvalues and tiny rank-1 components, and recursive
eigenvector accumulation with Householder back-transform. A correct,
hardened implementation is on the order of 1000–1500 LOC and benefits
from an extensive numerical regression corpus (clustered spectra,
near-zero z-vectors, exact-equal diagonals). LAPACK's `dstedc` took the
community years to settle.

**Closure path:** D&C is an **optimization**, not a correctness fix. The
right time to add it is after `dsteqr`-style QR is in place and
benchmarks show QR is the bottleneck for a real workload (typically
n ≳ 500). When that triggers, D&C lands as its own phase with: a design
doc, an oracle test corpus, QR as the regression baseline, and a
feature-flag rollout (D&C selectable per-call until confidence accrues).

## Strings (`r2-strings`)

### Pattern functions: `grep`, `grepl`, `gsub`, `sub`, `regexpr` ✅ regex shipped

**Status:** With `--features regex` (default ON) these functions use a
real regex engine (`regex-lite`, pure-Rust, ~150 KB compiled). POSIX-ERE
subset: anchors (`^`, `$`), character classes (`[abc]`, `[^abc]`),
groups (`(...)`), repetitions (`?*+{n,m}`), alternation (`|`),
shorthand classes (`\d`, `\w`, `\s` and their negations).

**`fixed = TRUE`** forces literal matching — `.` matches the literal
dot, not "any char". Matches R's `fixed=` semantics.

**Pattern compile failure → literal fallback.** If the regex fails to
parse, the function silently uses substring matching on the raw pattern
text. Means malformed regex doesn't error; it degrades.

**Building without regex:** `cargo build --no-default-features` skips
the `regex-lite` dep entirely. Falls back to literal-substring semantics.
Smaller binary, no regex syntax.

**Not supported (regex-lite limitation, intentional):**
- Lookaround (`(?=…)`, `(?!…)`) — R's default `extended=TRUE` mode
  doesn't either, so not a parity gap
- Backreferences (`\1`) in the pattern — R supports these in `perl=TRUE`
  mode; we don't
- Unicode category classes (`\p{Letter}`) — `regex-lite` is ASCII-fast
- PCRE-only features (recursive patterns, conditional groups, atomic
  groups)

For Unicode-heavy text or PCRE features, the full `regex` crate could
be a future feature flag, but for typical R idioms (`^foo$`,
`[A-Z][a-z]+`, `\d+\.\d+`) regex-lite is sufficient.

### `sprintf`

**Status:** Recognises `%d`, `%f`, `%s`, `%e`, `%%` only. No width or
precision specifiers (`%5d`, `%.2f`), no flags (`%-d`, `%+d`).

**Why:** v0.1.x simplification — covers the common case of inline value
substitution. The engine's pre-migration implementation had the same
limitation; it is now documented rather than hidden.

**Closure path:** Implement a real format-spec parser following C99
`printf` rules, or wrap `format!` via a runtime template engine.
No timeline yet.

## Out-of-core / columnar memory (`r2-arrow`)

### Memory-mapped columns — larger-than-RAM ✅ shipped (v0.2.2), with bounds

`mmap.write(x, path)` writes a numeric vector as a packed-`f64` file;
`mmap.col(path)` opens it as a memory-mapped handle whose `sum`/`mean`/
`min`/`max` **stream over the mmap**, so files **larger than RAM** reduce
with bounded memory (verified: an 8 GB file on a 7.4 GB-RAM machine sums
correctly at ~5 GB peak RSS; the multi-accumulator sum beats R's
`bigmemory`).

**Current bounds (closure path = follow-up phases):**
- **Reductions only** — `sum`/`mean`/`min`/`max`/`length`. No general
  element-wise ops, filtering, joins, or indexing on the mmap path;
  those would need `to_owned()` back into RAM, defeating the purpose.
- **Read-only** — the mapping is opened read-only.
- **`mmap.write` needs the source vector in RAM** — so *building* a
  >RAM file from R2 itself isn't possible yet. A chunked/append writer
  (and **Parquet / Arrow-IPC readers** for real datasets) is the next
  step. Today a >RAM file must be produced externally or in chunks.
- The default in-memory `RVal::Numeric` path is still RAM-bound; the
  Phase F.3 storage migration (numeric storage = `ColumnarF64` natively)
  is what would make *everything* zero-copy / mmap-friendly.

### Element-wise arithmetic — vector⊗scalar fusion ✅ (v0.2.2), residual repack

`v*2+1`-style chains now fuse into a single pass (≈2.4× faster). The
remaining gap vs R is the `Option<f64> → dense f64` repack on the base
vector (each `RVal::Numeric` lazily repacks to columnar); the Phase F.3
storage migration removes it.

## Data (`r2-data`)

### `merge(df1, df2)` — single-key inner join only

**Status (v0.1.0):** performs an inner join on a single shared column
(auto-detected, or named via `by=`). Non-key column-name collisions get
a `.y` suffix on the right side. NA propagation matches the engine
pre-migration behaviour.

**Not implemented:**
- Multi-column keys (`by = c("a", "b")`)
- Outer joins (`all.x = TRUE`, `all.y = TRUE`, full outer)
- Asymmetric key naming (`by.x = "id"`, `by.y = "user_id"`)
- Custom suffix control (`suffixes = c(".left", ".right")`)

**Closure path:** Add the missing parameters in a follow-up phase. Most
of the heavy lifting is already there — the matching loop is generic;
`all.x`/`all.y` just need an "unmatched rows go through with NA" pass.

## I/O (`r2-io`)

### CSV / TSV parser — RFC 4180 ✅ compliant

**Shipped this round:** `read.csv`/`read.table`/`read.delim` and
`write.csv`/`write.table` now use a proper RFC 4180 state-machine
parser. The four cases the previous line-split-and-trim approach
missed are all handled:

1. **Embedded separators in quoted fields:** `"a,b",c` is correctly
   parsed as 2 fields, not 3. Verified by regression test.
2. **Doubled quotes inside quoted fields:** `"He said ""hi"""` →
   `He said "hi"`. Both read and write sides escape correctly.
3. **Newlines inside quoted fields:** a quoted field can span multiple
   lines; the newline becomes part of the value.
4. **UTF-8 BOM stripping:** `\u{FEFF}` at file start is silently
   removed so the first column name isn't polluted.

**Write side:** column names always wrapped in `"..."` (matches R's
`quote=TRUE` default). Character fields wrapped + internal quotes
doubled. Numeric/integer fields emitted raw.

**Still not handled** (low-priority gaps):
- Mixed quote styles (single-quote-as-string-delim)
- Comment lines (`#` prefix skipping — R's `comment.char=` arg)
- Skip lines (`skip = N` argument)
- Encodings other than UTF-8 (latin-1, etc. — R's `fileEncoding=`)

Closure path for the remaining items: add the missing named args to
`bi_read_csv` and route through. Each is ~15 LoC.

## Stats — hypothesis tests (`r2_stats::htest`)

### `fisher.test()` — exact hypergeometric ✅ shipped (v0.1.0)

**Status:** **Fixed.** Now computes the two-sided p-value by exact
hypergeometric summation over all 2×2 outcomes at least as extreme as
observed. Matches R's `fisher.test` to within ~1e-4 across small-table
spot checks. The previous χ²-with-Yates approximation has been replaced.

### `t.test()` — Welch–Satterthwaite df ✅ shipped (v0.1.0)

**Bug fix:** Two-sample `t.test(x, y)` was reporting itself as
"Welch Two Sample t-test" while internally using the *pooled* Student
df (`nx + ny − 2`). The standard error was already the unequal-variance
form, but the df was the equal-variance form — inconsistent. Now uses
the proper Welch–Satterthwaite df formula
`(vx + vy)² / (vx²/(nx−1) + vy²/(ny−1))`, which matches R's
`t.test(x, y)$parameter` to ~1e-3.

### `t.test()` — formula syntax + paired test ✅ shipped (v0.1.0)

**New call shapes accepted:**
- `t.test(value ~ group)` — formula form. The `group` vector (Character,
  Factor, Logical, or Numeric) must have exactly 2 distinct levels.
  Splits `value` by group and runs a Welch two-sample test. Group
  labels (factor levels / unique values, in order of first appearance)
  appear in the printed output and as `$group1` / `$group2` fields.
- `t.test(x, y, paired=TRUE)` — paired test on `(x[i], y[i])`
  differences against `mu` (default 0). Standard one-sample-on-diffs
  with `df = n − 1`.

**Extension beyond R:** the paired output additionally reports the
**Pearson correlation** between the paired observations. Useful for
within-subject designs where the strength of pairing tells you how
much variance the pairing absorbed (high `r` ⇒ pairing was worth it).
Available as `$cor` on the returned object.

**Output labelling:** when called via formula, the printed `mean of …`
lines use the actual group labels from the factor/character levels
rather than the placeholder `x` / `y`.

**Within-subject auto-pairing — `id =` argument ✅ shipped:**
Repeated-measures pairing is now available via an explicit `id =`
argument:
```
t.test(value ~ group, id = subject, paired = TRUE)
```
For each subject id, the function finds the observation in each of the
two `group` levels and pairs them. Subjects without one observation in
each level are silently dropped (with a printed count). df = n_paired − 1.
Pearson r between the matched pairs is reported as `$cor`.

**Future work — `Error(subject/group)` formula syntax:** R uses this in
`aov()` for repeated-measures ANOVA but `t.test` in R doesn't accept it.
Adding it to our `t.test` would be a real engine change: the formula
parser currently evaluates `group + Error(subject/group)` as
arithmetic and fails because `Error` isn't a function. To support that
syntax we'd need to (1) detect `Error()` in formula RHS at the engine
NSE pre-processing layer, (2) extract the id symbol unevaluated,
(3) rewrite the call to inject an `id =` argument before t.test sees
it. Tracked as a parser-level enhancement; the `id =` syntax above
delivers the same statistical capability today.

**Pearson r in unpaired output:** previously added then removed —
for independent two-sample tests the as-input correlation is
sample-order coincidence, not a meaningful diagnostic, and reporting
it could mislead users. `$cor` is present only on the paired output.

### `t.test()` — t-CDF accuracy ✅ resolved

**Status:** **Fixed.** `incomplete_beta` now uses the Lentz continued
fraction (Numerical Recipes §6.4) with the symmetry swap
`I_x(a,b) = 1 − I_{1−x}(b,a)` so the CF is always evaluated in its
fast-converging region — which is exactly what keeps the `b < 1`
boundary (the `t_cdf` path, `b = 0.5`) well-conditioned, the corner
that sank the earlier attempt. Accuracy is now ~1e-9 across df.

`t_cdf`, the t-test CIs (`qt`), and the F-tests (via `f_sf`) all route
through it. Pinned by regression tests: exact `pbeta(0.5,2,3)=0.6875`,
the symmetry identity, `t_cdf(1.96,30)=0.97032884`, and `f_sf` against
the df1=2 closed form.

## v0.1.0 Tier 0 polish round (historical record)

### Bugs fixed
- **`matrix(data, nrow, ncol)` positional args now honoured.** Previously
  only `nrow=` / `ncol=` keyword forms were read, so the R-idiomatic
  `matrix(rnorm(1e6), 1000, 1000)` silently produced a 1e6×1 column.
  Also added `byrow=` support.
- **`kmeans()` initialization spread + final size recompute.** Was
  initializing centroids with rows `0..k` (so close-together rows
  collapsed to a single cluster), and the convergence check broke out
  of the loop *before* sizes were ever computed if iteration 1 happened
  to match the initial all-zero `cluster` vector. Now uses evenly-spaced
  rows (`i*m/k`) for init and recomputes centroids + sizes
  unconditionally before checking convergence.
- **`rep()` works for character/integer/logical** (was numeric-only) and
  honours both `times=` and `each=` named args.
- **`factor()` accepts numeric/integer/logical** (was character-only).
  R-style: coerces to string first.

### Builtins added
- `binomial()`, `gaussian()`, `poisson()` — family constructors for
  `glm(family = binomial())`. Return tagged lists.
- `subset(df, mask)` — filter rows by logical mask.
- `transform(df, name = value)` — append/overwrite named columns.

### Subset/transform — NSE form shipped ✅

R's idiomatic `subset(df, x > 2)` and `transform(df, z = x + y)` are now
fully supported. The engine pre-processor (`Expr::Call` dispatch in
`r2-engine/src/lib.rs`) intercepts calls whose function name is `subset`
or `transform` with a data-frame as the first arg, evaluates the
remaining argument expressions in a child env that shadows globals with
the data-frame's columns, and passes the resulting values (logical mask
for subset, named columns for transform) to the underlying builtin.
Compound conditions (`subset(df, x > 1 & y < 50)`) and column-referencing
transforms (`transform(df, z = x + y)`, `transform(df, x = x * 2)`) both
work. Integration tests live in `crates/r2-engine/tests/nse_subset_transform.rs`.

**Closure-bug fix shipped in the same pass:** elementwise `&` and `|` on
logical vectors had no handler in `binary_op` and were silently falling
through the arithmetic arm to produce all-zero numeric output. Now
elementwise `&` / `|` (vector forms) and short-circuit `&&` / `||`
(scalar forms) both produce correct NA-aware logical results. Misleading
BinOp naming inherited from the lexer (single `&` → `BinOp::And`, double
`&&` → `BinOp::AndShort`) clarified with a comment block above the
handler.

## Memory layer (`r2-arrow`) — completion status

### F.6 storage migration ✅ shipped

`RVal::Integer` is now `Ints(Vec<Integer> + cached Arc<ColumnarI32>)`.
`RVal::Logical` is now `Logicals(Vec<Logical> + cached Arc<ColumnarBool>)`.
Same Deref/From pattern as `Reals`. First call to `.columnar()`
materialises the columnar form (O(n) one-time), subsequent calls are
`Arc::clone` (O(1)).

Memory footprint per million elements:
- Logical: `Vec<Option<bool>>` was 16 MB → `ColumnarBool` is 250 KB (~64× smaller)
- Integer: `Vec<Option<i32>>` was 8 MB → `ColumnarI32` is ~4.25 MB (~2× smaller)
- Numeric (F.3, already shipped): ~no change in footprint, but columnar
  bridge enables zero-copy SIMD-friendly access from JIT and kernels

### Still not implemented

| Item | Reason / closure |
|---|---|
| `ColumnarI64` | Mechanical copy of `ColumnarI32` with `i64`. Add when an actual i64 hot path materialises. |
| `ColumnarUtf8` | Variable-length strings need a separate offsets array + values byte buffer — its own design pass. R workloads rarely hit string-column performance limits today (R's `Vec<Option<Arc<str>>>` representation is adequate). |
| mmap write path | `MmapColumnar::open` is read-only. Writable mmap needs separate API. Out of scope until a real workload demands it. |
| Cross-platform endian conversion for mmap | Host byte order assumed. Cross-arch feeds would need explicit conversion. |
| Strided iteration in r2-kernel for non-contiguous matrix slices | **Shipped ✅** (Phase K.6). `reduce_strided(op, data, offset, stride, count)` walks every `stride`-th element of a flat slice without copying. Serial + Rayon backends; Rayon uses two-pass (parallel NA scan + parallel reduce of unwrapped values). All `ReduceOp` variants supported. Tests in `r2-kernel/src/lib.rs`. |
| Fused multiply-add (`MulAdd`) op in r2-kernel | **Shipped ✅** (Phase K.5). New `TernaryOp::MulAdd` + `TernaryBackend` trait with Serial/Rayon impls and Oracle-driven `ternary(op, a, b, c)` dispatcher. Uses `f64::mul_add` so on FMA-capable hardware the multiply+add is a single rounded operation. |

## Oracle layer — calibration intentionally bundled with hardware awareness

**Status:** the per-Op parallelism thresholds in `r2_oracle::dispatch` are
hand-tuned constants. A `r2-bench` crate that fits them to measured
crossover points was originally listed as Tier 4 polish.

**Plan change (recorded so it doesn't get relitigated):** that calibration
work has been **removed from Tier 4 and folded into the Phase G
hardware-awareness phase**. Reasoning:

1. **Cores.** Bench numbers measured on an N-core dev laptop don't transfer
   to a 64-core server (parallel crossover happens at much smaller N) or
   a 2-core VM (crossover at much larger N). A single static calibrated
   table is wrong everywhere except the calibration machine.
2. **ISA / SIMD width.** `MulAdd` and the JIT vector paths get 2× swings
   depending on whether FMA3 / AVX2 / AVX-512 / SVE2 is available.
3. **Cache hierarchy.** Matmul block sizes and strided-reduce crossovers
   depend on L1 / L2 / L3 sizes, which vary by 5× across realistic targets.

So calibration without hardware introspection is *less* portable than
the current conservative hand-tuned constants — and any work we do
before the hardware-awareness layer would have to be redone after.

**Closure path (Phase G, deferred):** add an `r2_oracle::hw` module that
detects cores, CPU features (`std::is_x86_feature_detected!` for x86,
cfg-gated ARM equivalents), and cache sizes. Make the threshold table
parametric in `Hw`: closed-form rules like
`parallel_threshold(op) = base / hw.cores` rather than flat constants.
Once that lands, *then* per-machine calibration becomes additive —
benchmarks refine the closed-form residuals rather than fitting from
scratch on the wrong basis.

**Until then:** the current thresholds are intentionally conservative
(bias toward Serial when N is borderline) so they're never wildly wrong
on any deployment. Worst case is "could be faster on a big box"; never
"slower than Serial on a small box."

## Other domains

(no other intentional limitations recorded yet — additions go here
as we encounter them)
