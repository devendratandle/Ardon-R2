# PLS-SEM speed comparison — cSEM (R) vs r2sem / plssem (Ardon-R2)

**Date:** 2026-07-07  **Machine:** 6-core, AVX2+FMA, Windows 11
**R:** 4.5.3 (reference BLAS) · **cSEM:** 0.6.1 · **Ardon-R2:** release build

## Setup (identical for every engine)

- Same data: `set.seed(42)`, `n = 1000`, 21 reflective indicators, 3 latent
  constructs (Stress, Anxiety, Depression), exported once to
  `samples/plssem_data.csv` and read by all engines.
- Same model: Anxiety ~ Stress; Depression ~ Stress + Anxiety; each construct
  measured reflectively (`=~`) by 7 indicators.
- Same estimator settings: **Mode A reflective, factorial inner scheme, no
  disattenuation**, 200 bootstrap resamples.

> cSEM's *default* is consistent PLSc (disattenuation, path scheme), which gives
> higher path estimates (0.511 / 0.324 / 0.419). That is a methodological choice,
> not an error — to compare like-for-like it is turned off here so all engines
> compute the *same* quantity.

## Estimates — identical across engines (correctness check)

| Path | cSEM (matched) | r2sem (source lib) | plssem (native) |
|---|---|---|---|
| Anxiety ~ Stress    | 0.4610 | 0.461 | 0.461 |
| Depression ~ Stress | 0.3063 | 0.306 | 0.306 |
| Depression ~ Anxiety| 0.3759 | 0.376 | 0.376 |

Bootstrap SEs also agree (e.g. Anxiety~Stress: cSEM 0.0239 vs R2 0.024). The
match validates the R2 implementations against the reference library.

## Wall-clock (200 bootstrap resamples)

| Engine | Method | Time | vs cSEM (matched) |
|---|---|---:|---:|
| **cSEM** (default) | consistent PLSc, path, single-thread | 12.44 s | — |
| **cSEM** (matched) | plain PLS, factorial, single-thread | **9.86 s** | 1× |
| **r2sem** | R2 *source library*, parallel bootstrap (`par.sapply`) | **~1.3 s** | **≈ 7.6× faster** |
| **plssem** | R2 *native Rust builtin* | **0.104 s** | **≈ 95× faster** |

r2sem run-to-run: 1.10–1.47 s (parallel-scheduling jitter); ~1.3 s typical.

## Reading the result

- **r2sem** is a library written *in the R2 language* (pure `.r2` source, no
  native code) and still runs ~7–8× faster than cSEM on identical inputs and
  identical estimates. The win is **parallelism** — r2sem's bootstrap loop uses
  `par.sapply` across 6 cores (Phase P); cSEM's bootstrap is single-threaded.
- **plssem** is the first-party **native builtin** (Rust): ~95× faster than
  cSEM and ~13× faster than the source library — the expected gap between a
  hand-optimised native kernel and language-level source.
- The two R2 numbers bracket the design: *source libraries* get parallelism for
  free; the *hot 5%* that matters most (`lm`, `plssem`) ships as native kernels.

## Caveats (honest)

- Single machine, single dataset/size; cSEM bootstrap is single-threaded by
  default (a real-world default, not a handicap we imposed). A multi-core cSEM
  run via `future` would narrow the r2sem gap but not the native-builtin gap.
- The recent JIT work (J.2/J.3) does **not** contribute here: r2sem is written
  in vectorised style (inline `sum(x*y)`, matrix ops) already at native kernel
  speed; the JIT accelerates imperative `for`-loop code, which r2sem avoids.

## Reproduce

```
r2 samples/plssem_demo.r2        # writes samples/plssem_data.csv + native plssem
Rscript samples/plssem_compare.R # cSEM default (PLSc)
r2 devlib/test_r2sem.r2          # r2sem source library
```
