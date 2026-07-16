# Benchmarks — Ardon-R2 vs R

Reproducible head-to-head of the same operations in R and Ardon-R2.

```sh
Rscript benchmarks/bench_r.R            # R side
r2 benchmarks/bench_r2.R                # R2 side (release build)
```

## Results

The authoritative, current benchmark tables (v0.3.8, split by workload
class, plus CPU-vs-integrated-GPU and accuracy) live in the repo-root
[`PERFORMANCE.md`](../PERFORMANCE.md). The runnable scripts for that report
are in [`v038/`](v038/) (`bench_r2.r2` + `bench_r.R`, identical algorithms
and sizes).
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
