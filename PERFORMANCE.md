# Performance — Ardon-R2 vs R

Head-to-head timing and numerical-accuracy comparison of Ardon-R2 against
CRAN R. Reproduce on your own machine: see `bench/r_vs_r2/RUN_THIS.md`.

Numbers below are **R2 v0.3.4 vs CRAN R 4.5.3** (default reference Rblas),
elapsed seconds, warm-cache, one 6-core AVX2 workstation (no AVX-512),
measured **2026-06-27**.

## Standard workloads

| Operation | R | R2 | Ratio | Notes |
|---|---:|---:|---:|---|
| **Matrix multiply** (1024×1024) | 0.650 | **0.030** | 🏆 **R2 22× faster** | Multiversioned AVX2 DGEMM + Oracle multicore |
| **Matrix multiply** (500×500) | 0.050 | **0.013** | 🏆 **R2 3.9× faster** | |
| **sapply iris × 30 reps** | 0.010 | **0.0004** | 🏆 **R2 ~25× faster** | R near timer resolution |
| **Linear model** (lm 1e5 × 2 cols) | 0.030 | **0.013** | 🏆 **R2 2.3× faster** | F.3 columnar + JIT |
| **Sort** (1e6 doubles) | 0.100 | **0.064** | 🏆 **R2 1.6× faster** | |
| **Sum + mean** (1e7) | 0.030 | **0.018** | 🏆 **R2 1.6× faster** | Columnar-native reductions |
| K-means (1e5 × 10, k=5) | 0.440 | 0.427 | tie | |
| SVD (200×100) | ~0.000 | 0.011 | R (timer res.) | Both sub-15 ms |
| Element-wise add (1e7) | 0.040 | 0.145 | R 3.6× | Memory-bandwidth bound; `Vec<Option>↔Columnar` conversion |

**Headline:** R2's strongest wins are matrix multiply (22× on 1024² against
R's reference BLAS), fused math-JIT loops (`sin²+cos²` at 5×), and the apply
family (`sapply` ~25×). R wins on single memory-bandwidth-bound passes
(element-wise add) and sub-15 ms ops that sit at R's timer resolution.
Results below ~15 ms are run-to-run noisy — read them as "comparable," not
precise ratios.

## Accuracy

On deterministic statistical workloads (descriptives, `lm`, `glm`, Welch
`t.test`, `aov`, SVD, eigen, `cor`), R2 v0.3.4 matches R 4.5.3 to **~7
significant figures** — e.g. `lm` β = `37.22727 / −3.877831 / −0.0317729`,
R² `0.8267855`, F `69.21121`; eigenvalues `6, 3, 1` exact; `cor`
`0.8717538`. Run `bench/r_vs_r2/accuracy.{R,r2}` to reproduce.

## Math-JIT comparison (user closures compiled to native)

The **M-R2-JIT** path compiles user functions whose bodies are pure scalar
arithmetic + math calls to native machine code via Cranelift. Common stats
idioms like `f <- function(x) sqrt(x*x + 1)` or
`function(x) sin(x)^2 + cos(x)^2` now bypass the tree-walking interpreter
entirely.

| Closure body | R | R2 | Ratio |
|---|---:|---:|---:|
| `sqrt(x*x + 1)` | 0.006s | 0.014s | R 2.2× (memory-bandwidth bound) |
| `log(exp(x))` | 0.048s | **0.029s** | 🏆 **R2 1.7×** (chained extern calls fuse) |
| `sin(x)² + cos(x)²` | 0.184s | **0.037s** | 🏆 **R2 5.0×** (4 calls + ops in one loop) |
| `sqrt(x² + y²)` | 0.008s | 0.022s | R 2.8× (Phase C.7 closed it from 7.3×) |
| `\|sin(x)\| + \|cos(x)\|` | 0.073s | **0.027s** | 🏆 **R2 2.7×** |

All on 1e6-element vectors, single workstation. R2 wins whenever the
function fuses multiple math operations (the JIT generates one tight loop
with all ops inline); R wins on single-call sqrt where memory bandwidth
dominates and its libm SIMD path has slightly tighter per-call memory
footprint. Reproduce with `pwsh bench\r_vs_r2\run.ps1` and inspect
`math_jit.R` / `math_jit.r2`.

## Reproducibility caveats

- R's matrix-multiply speed depends entirely on which BLAS it's linked
  against. Default CRAN R on Windows ships reference Rblas (the slow netlib
  BLAS). R linked against OpenBLAS or Intel MKL will reverse the matmul
  result. R2's edge holds against the default; tuned BLAS wins.
- Element-wise add (1e7) is the one workload where R2 is still meaningfully
  slower (3.6×). The gap is in the `Vec<Option<f64>> ↔ ColumnarF64` legacy
  conversion; closing it requires further F.3 native-columnar migration of
  the value type. Tracked in `docs/KNOWN_LIMITATIONS.md`.
- Built-in ML (GBM, Random Forest, decision tree, KNN, naive Bayes,
  k-means) is available directly in base R2 — no package install. R needs
  CRAN packages (`gbm`, `randomForest`, `rpart`, `e1071`) for the
  equivalents.

## Run it yourself

```sh
cargo build --release
pwsh bench\r_vs_r2\run.ps1     # Windows — produces a comparison table
bash bench/r_vs_r2/run.sh      # Linux / macOS
```

See `bench/r_vs_r2/README.md` for what's tested and how to interpret the
deltas, and `benchmarks/README.md` for the matmul / crossprod / sum / lm
micro-suite.
