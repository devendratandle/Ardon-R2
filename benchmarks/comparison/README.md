# R vs R2 Comparison Suite

Paired scripts that run the same nine accuracy checks and eight
performance benchmarks on R and on R2, then print a side-by-side
comparison table.

## Prerequisites

- **CRAN R 4.x+** with `Rscript` on PATH.
- **r2 built in release mode**:
  ```
  cargo build --release
  ```
  This produces `target/release/r2` (or `r2.exe` on Windows).

## Running

### Windows (PowerShell)

```powershell
pwsh ./bench/r_vs_r2/run.ps1
```

### Linux / macOS

```bash
bash bench/r_vs_r2/run.sh
```

Or run the four scripts manually and diff the output yourself:

```bash
Rscript bench/r_vs_r2/accuracy.R       > out_R.txt
target/release/r2 bench/r_vs_r2/accuracy.r2 > out_R2.txt

Rscript bench/r_vs_r2/performance.R    > perf_R.txt
target/release/r2 bench/r_vs_r2/performance.r2 > perf_R2.txt
```

## What's compared

### Accuracy (9 sections)

| # | Test | What's checked |
|---|---|---|
| 1 | Descriptive stats on `iris$Sepal.Length` | mean / median / sd / var / min / max |
| 2 | `lm(mpg ~ wt + hp, data = mtcars)` | intercept, β-wt, β-hp, R², adj-R², F-stat |
| 3 | `glm(am ~ wt + hp, family = binomial, ...)` | intercept, βs, null/residual deviance, AIC |
| 4 | Two-sample Welch `t.test` (setosa vs versicolor petal length) | t, df, p-value |
| 5 | `svd(A)` on a 5×3 matrix | top 3 singular values, reconstruction error ‖A − UΣVᵀ‖ |
| 6 | `eigen(cov(iris[,1:4]))` | top 4 eigenvalues |
| 7 | `aov(Sepal.Length ~ Species, ...)` | F-stat, p-value |
| 8 | `kmeans(iris[,1:4], 3)` | tot.withinss, smallest cluster size, largest cluster size |
| 9 | `cor(...)` Pearson on iris columns | r |

**Expected agreement**: numerical deltas should be < `1e-3` in absolute
terms for most rows. Exceptions and why:

- **t-test p-value**: R2 uses a 1000-panel trapezoidal incomplete-beta
  integration (~1e-4 accuracy); R uses series-expansion. Agreement to
  ~3 decimals.
- **kmeans size_smallest/largest**: R and R2 may converge to different
  local minima from the same seed because the seed-state PRNGs differ.
  Within ±3 is typical; the total within-cluster sum-of-squares agrees
  more tightly.
- **SVD signs**: U and V columns may differ in sign convention. R's
  LAPACK uses one convention; R2's `dgesvd_full` enforces the "largest-
  magnitude entry per column is positive" rule. The singular values
  and the reconstruction error are unaffected.

### Performance (8 sections, wall-clock seconds)

| # | Workload | Dominant cost |
|---|---|---|
| 1 | `a + b` on `n=1e7` numeric vectors | element-wise + sum |
| 2 | `sum(a); mean(a)` on `n=1e7` | reduction |
| 3 | `sort(rnorm(1e6))` | sort |
| 4 | `A %*% B`, 500×500 matrices | BLAS-3 matmul |
| 5 | `lm(y ~ x1+...+x5)` on `n=1e5` | normal equations + summary |
| 6 | `kmeans(M, k=5)` on `1e5 × 10` | iterative reassignment |
| 7 | `sapply(1:30, function(i) mean(...))` on iris | apply-family |
| 8 | `svd(M)` on 200×100 | bidiagonalization + QR |

**Caveat about BLAS**: R's matrix-multiply speed depends entirely on which
BLAS R is linked against. On Windows with the default `Rblas.dll`, R2's
pure-Rust BLAS-3 typically wins by ~2×. Against R linked with OpenBLAS
or Intel MKL, R wins (those are SIMD-optimized + multithreaded by
specialists). The 500×500 size is chosen so the comparison is
representative without being dominated by warm-up overhead.

## Interpreting the output

The driver script prints each key with R's value, R2's value, and the
absolute delta plus relative percentage. Example row:

```
lm.beta_wt                     -3.8778334574          -3.8778334568          5.6e-10 (0.00%)
```

For performance:

```
matmul_500x500                 0.0420                 0.0210                 0.021 (50.00%)
```

The R column shows R's wall-clock seconds; R2's column shows R2's; the
delta column is `|R - R2|` in seconds and the percent change.

## Adding new tests

Add a matched pair of lines to both `accuracy.R` (or `.r2`) and
`performance.R` (or `.r2`), using the `KEY=VALUE` convention. The
driver auto-discovers any new keys.

## Known limitations of the comparison

- R2's RNG is not bit-identical to R's, so any test that depends on
  exact random draws (k-means seed → centers, random forest splits)
  will diverge in run-to-run details. The aggregate metrics (sizes,
  losses) should still agree to a tolerance.
- The driver does no statistical testing of timing differences (no
  multi-run jitter envelope). For publication-quality benchmarks, wrap
  each block in `microbenchmark::microbenchmark(times = 100)` on the R
  side and average over multiple R2 runs.
