# Performance — Ardon-R2 vs R

Head-to-head timing and numerical-accuracy comparison of Ardon-R2 against
CRAN R. Reproduce on your own machine: see `benchmarks/comparison/RUN_THIS.md`.

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
| Element-wise add (1e7) | 0.041 | **0.032** | 🏆 **R2 1.3× faster** | Zero-copy columnar path (v0.3.4); 10-rep avg |

**Headline:** R2's strongest wins are matrix multiply (22× on 1024² against
R's reference BLAS), fused math-JIT loops (`sin²+cos²` at 5×), and the apply
family (`sapply` ~25×). As of v0.3.4 element-wise arithmetic is also a win
(the columnar path stays zero-copy). R still edges ahead on a few sub-15 ms
ops that sit at its timer resolution. Results below ~15 ms are run-to-run
noisy — read them as "comparable," not precise ratios.

## Accuracy

On deterministic statistical workloads (descriptives, `lm`, `glm`, Welch
`t.test`, `aov`, SVD, eigen, `cor`), R2 v0.3.4 matches R 4.5.3 to **~7
significant figures** — e.g. `lm` β = `37.22727 / −3.877831 / −0.0317729`,
R² `0.8267855`, F `69.21121`; eigenvalues `6, 3, 1` exact; `cor`
`0.8717538`. Run `benchmarks/comparison/accuracy.{R,r2}` to reproduce.

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
| `sqrt(x² + y²)` | 0.008s | 0.022s | R 2.8× |
| `\|sin(x)\| + \|cos(x)\|` | 0.073s | **0.027s** | 🏆 **R2 2.7×** |

All on 1e6-element vectors, single workstation. R2 wins whenever the
function fuses multiple math operations (the JIT generates one tight loop
with all ops inline); R wins on single-call sqrt where memory bandwidth
dominates and its libm SIMD path has slightly tighter per-call memory
footprint. Reproduce with `pwsh benchmarks/comparison/run.ps1` and inspect
`math_jit.R` / `math_jit.r2`.

## Reproducibility caveats

- R's matrix-multiply speed depends entirely on which BLAS it's linked
  against. Default CRAN R on Windows ships reference Rblas (the slow netlib
  BLAS). R linked against OpenBLAS or Intel MKL will reverse the matmul
  result. R2's edge holds against the default; tuned BLAS wins.
- Element-wise add was R2's one meaningful loss (~3.6× in earlier releases),
  caused by a `Vec<Option<f64>> ↔ ColumnarF64` round-trip on the arithmetic
  path. **Fixed in v0.3.4**: the fast-path guard no longer
  materialises the boxed form, and the columnar kernel hoists the op match
  out of the loop so each op auto-vectorises — `a + b` is now zero-copy and
  at parity with (slightly faster than) R.
- Built-in ML (GBM, Random Forest, decision tree, KNN, naive Bayes,
  k-means) is available directly in base R2 — no package install. R needs
  CRAN packages (`gbm`, `randomForest`, `rpart`, `e1071`) for the
  equivalents.

## Run it yourself

```sh
cargo build --release
pwsh benchmarks/comparison/run.ps1     # Windows: produces a comparison table
bash benchmarks/comparison/run.sh      # Linux / macOS
```

See `benchmarks/comparison/README.md` for what's tested and how to interpret the
deltas, and `benchmarks/README.md` for the matmul / crossprod / sum / lm
micro-suite.
