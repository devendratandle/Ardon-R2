# Performance — Ardon-R2 vs R

**Sections 1–4 re-timed 2026-09-28 (0.4.1 in development); the GPU
section is from v0.3.8 (2026-07-15); accuracy re-verified at v0.4.0.**

Head-to-head timing and numerical accuracy of Ardon-R2 against CRAN R,
measured on one 6-core AVX2 laptop (AMD Ryzen 5 4500U, no AVX-512, held
2375 MHz on AC power) with an **AMD Radeon integrated GPU**. Elapsed
seconds, best of 3 warm runs, two interleaved passes (R2 then R, twice),
R2 (release, no-LTO CI profile) vs CRAN R 4.5.3 (default reference Rblas).
Ranges are the two passes. Reproduce:
`benchmarks/v038/bench_r2.r2` and `bench_r.R` (identical algorithms and
sizes), GPU: `cargo run -p r2-gpu --release --features gpu --example flops`.

> **What changed since these timings were taken, and why they still stand.**
> v0.3.9/v0.4.0 rewrote the **single-precision** matrix path — a
> blocked/packed `sgemm` with a runtime-dispatched AVX2 micro-kernel — for
> LLM training. R's `%*%` and every statistical routine here run in
> **double** precision through `level3::dgemm`, which that work left alone
> until 2026-09-20, when its AVX2 tier gained the fused multiply-adds it
> had claimed all along (500×500: 16–20 → 19–27 GFLOP/s). So the table
> below is, if anything, conservative for `%*%`; the f32 numbers live in
> `benchmarks/llm/REPORT.md` and are a different measurement of a
> different kernel. Re-run the harness above if you want them re-taken on
> your own machine.

**This file is R-vs-R2.** For R2 against PyTorch and JAX on LLM training,
see **[`benchmarks/llm/REPORT.md`](benchmarks/llm/REPORT.md)** — that is the
single source for those numbers, and none of them appear here.

The comparison is split by **class of workload**, because that is what
decides who wins — R runs C internals natively but user code interpreted,
while R2 JIT-compiles user code to the same native kernels.

## 1. C internals — R2 builtins vs R builtins

Both engines run native machine code here (R's C, R2's Rust). Parity or
better is the realistic ceiling.

| Operation | R | R2 | Result |
|---|---:|---:|---|
| Vector add (1e7) | 0.0265–0.0336 | **0.0231–0.0297** | R2 1.1× |
| Sum (1e7) | 0.0129–0.0139 | **0.0081–0.0091** | R2 1.5–1.6× |
| Sort (1e6) | **0.0650–0.0675** | 0.0700 | R 1.05× (tie) |
| sd (1e6) | **0.0048** | 0.0056–0.0057 | R 1.2× |
| cor (1e6) | **0.0108–0.0111** | 0.0316 | R 2.9× |
| Matrix multiply (500×500) | 0.0490–0.0518 | **0.0043–0.0047** | R2 11× |

(`sort` on 1e6 doubles regressed to 0.45 s with the 2026-09-23 R-order
rewrite of sort/order — every comparison went through the general
NA-aware path — and was restored the same day by a numeric fast path.)

### Linear algebra at n = 1000 (re-timed 2026-09-25)

The same user-level calls in both engines on the same deterministic,
full-rank 1000×1000 matrices (condition number ~1e4), median of 5,
milliseconds, two interleaved passes on this machine (AMD Ryzen 5 4500U,
held 2375 MHz, on AC power, 2026-09-28). R 4.5.3 with its reference
BLAS/LAPACK; R2 release build (no LTO). Reproduce:
`Rscript benchmarks/linalg_vs_r.R` then `r2 benchmarks/linalg_vs_r.R`.

| Call | R | R2 | Result |
|---|---:|---:|---|
| `A %*% A` | 530–540 | **22–26** | R2 20–25× |
| `solve(S)` (inverse) | 700–730 | **118–124** | R2 5.8–6.2× |
| `chol(S)` | 120–130 | **33–34** | R2 3.6–3.9× |
| `eigen(S, symmetric = TRUE)` | 1,060–1,070 | **372–430** | R2 2.5–2.9× |
| `svd(A, nu = 0, nv = 0)` | 870–890 | **402–425** | R2 2.0–2.2× |
| `svd(A)` | 2,610–2,640 | **794–806** | R2 3.2–3.3× |
| `qr(A)` | 450–460 | — | not available in R2 yet |

(An earlier version of this table, 2026-09-25, timed a rank-2 matrix —
`matrix(sin(1:n^2), n)` — which made `qr` and `svd` atypical on both
sides. R2's `svd` also now pays for accurate small singular values and an
orthonormal `u`; see CHANGELOG, 2026-09-28.)

Against R built on an optimized BLAS (OpenBLAS, MKL) these gaps would be
far smaller: R2's own routines are 0.15-0.43× of Intel MKL at this size
(see `CHANGELOG.md`, 2026-09-25).

## 2. User statistical formulas — R2 JIT vs R interpreter

The same one-line formula: R interprets it, R2 compiles it to fused SIMD
kernels.

| Formula | R | R2 | Speed-up |
|---|---:|---:|---|
| Variance `sum((x-mean(x))^2)/(n-1)` (1e6) | 0.0072–0.0076 | **0.0055–0.0063** | 1.2–1.3× |
| Correlation one-liner (1e6) | 0.0239–0.0245 | **0.0096–0.0114** | 2.1–2.5× |

## 3. Textbook loops — the decisive gap

The same `for` loop, character-for-character. R executes it in its
bytecode interpreter; R2 normalizes and JIT-compiles it.

| Loop | R | R2 | Speed-up |
|---|---:|---:|---|
| Variance loop `for(i…) s <- s+(x[i]-mean(x))^2` (1e5) | 26.07–26.60 | **0.0006–0.0007** | **≈ 40,000×** |
| Dot-product loop (1e6) | 0.0499–0.0519 | **0.0083–0.0087** | 6.0× |
| Gradient descent, 500 iters (1e6) | 2.59–2.60 | **0.259–0.270** | 9.6–10× |

The variance loop recomputes `mean(x)` every iteration — R does the literal
O(n²) work; R2 recognizes the pattern, hoists the reduction, and emits one
pass. (Note the 1e5 size: at 1e6 R would take ~40 minutes; R2 stays sub-
millisecond.)

## 4. Addon-library code — composed helpers

Library-style functions built from small helpers with local variables and
guards (the r2sem pattern). R2 inlines and compiles the whole chain native.

| Workload | R | R2 | Result |
|---|---:|---:|---|
| Standardize `(x-mean)/sd` via composed helpers (1e6) | **0.0096–0.0099** | 0.0115–0.0116 | R 1.2× |

(At this vectorized size both are native-fast; the R2 win shows on the
*loop* and *repeated-call* forms above and in the r2sem library, which runs
~13× faster than R's cSEM — see `benchmarks/csem_vs_r2sem.md`.)

## 5. GPU — CPU vs integrated GPU (element maps, f32)

*The LLM training step also runs on the GPU end to end (`R2_GPU=1`) and
is 1.34-1.37x faster than PyTorch-DirectML at 7M parameters, 1.5-2.0x at
40M, on the same integrated GPU — `benchmarks/llm/REPORT.md` §4b. The
table below is the older element-map dispatcher.*

Machine: AMD Radeon integrated GPU (Vulkan). **Integrated GPUs need no
dedicated hardware** — wgpu reaches any Vulkan/DX12/Metal device.

| Size | CPU (elems/s) | GPU (elems/s) | Routed to GPU? | GPU vs CPU accuracy |
|---|---:|---:|:--:|---:|
| small (4 096) | 303 M | 308 M | no (below threshold) | 0.0 (exact) |
| medium (262 144) | 260 M | 0.51 M | yes | 1.79e-7 |
| large (4.19 M) | 55.9 M | (device limit) | — | — |

**Honest reading:** the GPU path is **accurate** (1.79e-7, f32 epsilon) and
**works on integrated hardware**, but is **not yet fast** — each call
re-initializes the device and transfers data, so a single element-map pass
loses to the CPU (and the largest size hit a device limit). This is a
correct *foundation*: the accuracy contract holds, the CPU fallback is
automatic, and the small case correctly stays on CPU by policy. Making the
GPU a *win* needs device/queue caching, resident buffers, and batched
kernels — tracked as follow-up work. Statistics are **never** GPU-routed
(f32 would break R-faithful accuracy); only ML-class element maps are
eligible.

## Accuracy

Every release is gated by a differential harness that runs identical scripts
under R2 and CRAN R and compares numerically (`tests/differential/run.sh`):
**27/27 cases pass, 2026-09-28** (matrix arithmetic, indexing,
statistics, lm/glm, data frames, strings, control flow, numeric edge
semantics, closures/environments, default args, indexed-loop math, matrix
metadata, merge/join semantics, factors, sorting and collation, element
assignment, linear algebra, SVD accuracy, a 555-value grid of distribution
functions with far tails, and whole-output cases for printing).

**Digit-level agreement** on identical fixed input (not RNG — the two
engines' random streams differ by design, so accuracy is measured on the
computation, `benchmarks/v038/accuracy.r2`):

| Quantity | R2 vs R agreement |
|---|---|
| `sqrt(2)`, `qnorm(0.975)`, `sin(1)` | **bit-identical (17 sig figs)** |
| `mean`, `sd`, `median`, `sum`, quantiles | 16–17 sig figs |
| `var`, `cor`, `cov`, `prod`, `lm` coef/R², `exp`, `log` | 15–16 sig figs |
| `pnorm` (normal CDF), `erf`/`erfc` | **15–16 sig figs, incl. extreme tail** |

Everything is full double precision, differing only in the last 1–2 ULP
from summation order. `pnorm` uses **R's exact algorithm** — Cody's SPECFUN
`pnorm_both`, evaluated *directly on the z-score* with R's own normal-CDF
coefficient set (not a generic `erf(x/√2)`, which loses ~1 ULP to the `÷√2`
rounding and mishandles the tail split-exp). The upper tail is returned
directly (`phi_upper`), so right-tail p-values avoid the `1 − Φ`
cancellation. Result: the normal CDF and every p-value built on it (t/z
tests, `lm`, mixed models) agree with R to 15–16 figures **including the
extreme tail** — `pnorm(-8)` is now correct to ~1e-13 relative (it was
~2% wrong when routed through erf). `erf`/`erfc` themselves use Cody's 1969
approximation (~1e-16). `qnorm`, the inverse CDF, is bit-identical
(Wichura's AS241). The t, F, binomial, Poisson and exponential d/p/q
functions agree with R to 1e-13 on 543 of 555 grid values (far tails and
`log.p` included); the known exceptions are listed in
`docs/KNOWN_LIMITATIONS.md` — `qf` is a bisection good to ~1e-11, and the
beta/gamma families are not implemented yet. Accuracy is the release gate,
not an afterthought.

## Build

- Full CLI release (no-LTO CI profile, cold): **5m 44s** on this 6-core box.
- The shipped installer profile (fat LTO) links slower but produces the
  distributed binaries; CI uses the no-LTO profile for realistic JIT
  behaviour without the multi-GB link cost.

## Headline

Ardon-R2 is **at parity with R on most of R's own C internals** (ahead on
`sum` 1.5×, behind on `cor` 2.9× and `sd` 1.2×), **11–25× faster on matrix
multiply and 2–6× on linear algebra** against R's reference BLAS/LAPACK, and
**dramatically faster on the code users and libraries actually write** —
6–10× on real loops, and ~40,000× where R does redundant interpreted work
R2 compiles away. Accuracy matches CRAN R (27/27 differential cases). The GPU dispatcher is a correct, accurate foundation that
brings integrated-GPU capability to any machine; performance tuning of the
GPU path is ongoing.

*All numbers are one machine, one run-set — treat sub-15 ms results as
"comparable," and reproduce with the scripts in `benchmarks/v038/`.*
