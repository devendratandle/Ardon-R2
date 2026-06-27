# Benchmarks — Ardon-R2 vs R

Reproducible head-to-head of the same operations in R and Ardon-R2.

```sh
Rscript benchmarks/bench_r.R            # R side
r2 benchmarks/bench_r2.R                # R2 side (release build)
```

## Results

6-core AVX2 CPU (no AVX-512); **R 4.5.3 with its default reference BLAS**;
release R2 v0.3.4 (`dgemm` multiversioned AVX2 + Oracle multicore).
Measured 2026-06-27 (elapsed seconds, warm cache).

| Operation | R 4.5.3 | Ardon-R2 v0.3.4 | R2 speedup |
|---|---|---|---|
| `%*%` matmul 1024×1024 | 0.65 s (3.30 GFLOP/s) | **0.030 s** | **22×** |
| `crossprod` 100k×50 | 0.11 s | **0.050 s** | **2.2×** |
| `sum` 5e7 | 0.08 s | **0.044 s** | **1.8×** |
| `lm` 100k×2 | 0.03 s | **0.013 s** | **2.3×** |

**Correctness** (deterministic, fixed inputs — identical in both):
matmul-sum `729`, crossprod-sum `693`, `lm` coef `(2, 3, −1.5)`.

## Honest notes

- The big matmul gap is because **stock R ships a single-threaded
  reference BLAS with no AVX**. Against R built with OpenBLAS/MKL,
  `%*%` would be far closer — that comparison is worth running on a
  machine where R has an optimized BLAS.
- `crossprod` / `sum` are **memory-bandwidth-bound**, so the wins are
  modest regardless of SIMD — expected, not a shortfall.
- R2's edge comes from runtime **AVX2/AVX-512 multiversioning** +
  **Oracle-gated multicore** in one portable binary; results match R
  exactly.
