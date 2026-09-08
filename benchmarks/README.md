# Benchmarks — Ardon-R2 vs R

Reproducible head-to-head of the same operations in R and Ardon-R2.

```sh
Rscript benchmarks/bench_r.R            # R side
r2 benchmarks/bench_r2.R                # R2 side (release build)
```

## Results

**Two reports, two different comparisons.** Keeping them apart matters:
they measure different precisions, different kernels, and different
opponents. Neither one's numbers belong in the other.

| Report | Compares | Status |
|---|---|---|
| [`../PERFORMANCE.md`](../PERFORMANCE.md) | **R2 vs CRAN R** — f64 statistics, `%*%`, JIT-compiled user code, plus CPU-vs-GPU and digit-level accuracy | Timings taken at v0.3.8; accuracy re-verified at v0.4.0 (13/13 differential cases) |
| [`llm/REPORT.md`](llm/REPORT.md) | **R2 vs PyTorch and JAX** — f32 LLM training, `sgemm`, attention, tokenizer | Current. Also lists what measured *worse*, so it isn't retried |

Runnable scripts for the R comparison are in [`v038/`](v038/)
(`bench_r2.r2` + `bench_r.R`, identical algorithms and sizes). For the LLM
comparison see section 7 of `llm/REPORT.md` — every figure there names the
command that reproduces it.

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
