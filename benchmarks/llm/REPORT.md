# Ardon-R2 vs PyTorch — LLM model training

**Final report, 2026-09-08.** Supersedes and replaces `RESULTS.md`,
`COMPONENTS.md`, `SPEED_REPORT.md`, `SHAPE_SURVEY.md`, `LMO_PLAN.md`,
`COMPONENT_ASSESSMENT.md` and `docs/PERFORMANCE_VS_PYTORCH.md`, all deleted.

Every number was measured on this machine with both sides run in the **same
window**. Nothing is carried over from an earlier session.

```
CPU      6-core, AVX2+FMA, no AVX-512. ~460 GFLOP/s f32 peak.
torch    2.13.0+cpu, BLAS_INFO=mkl (Intel MKL 2026.1), OpenMP, 6 threads
jax      0.11.1, CPU backend (XLA -> Eigen)
corpus   corpus.txt — 19.4 MB of TinyStories
config   dim 256, 4 layers, vocab 8,000, ffn 768, 30 steps x 32 x 64
```

---

## 1. Result

| phase | R2 | PyTorch | |
|---|---:|---:|---|
| read | 0.02 s | 0.03 s | — |
| learn BPE merges | 1.46 s | 1.70 s | R2 1.16x |
| **tokenize 19.4 MB** | **1.62 s** | 15.58 s | **R2 9.6x faster** |
| **train** | 32.82 s | **18.04 s** | **PyTorch 1.82x faster** |
| **TOTAL** | 35.92 s | **35.35 s** | **PyTorch 1.02x — a dead heat** |

Three interleaved pairs on the training phase: R2/PyTorch = 1.92x, 1.80x,
1.82x. **Training is 1.8x behind. The whole pipeline is level**, because
R2 tokenises 9.6x faster and training is no longer far enough behind for
that to be irrelevant.

**Learning is at parity.** Bits per byte is the comparable metric — the
vocabularies differ (8,000 vs 8,143) and cross-entropy is per token:

| | R2 | PyTorch |
|---|---|---|
| tokens/byte | 0.242 (4.13 bytes/token) | 0.243 (4.12) |
| loss | 9.1682 -> 6.2805 | 9.1396 -> 6.0964 |
| **bits/byte** | **2.1943** | **2.1355** |

2.75% apart, from different init RNGs. Neither ordering is a result.

---

## 2. Correctness

`accuracy_check.py` rebuilds the model in `torch.float64` from the
published equations and compares the loss, the logits and **all 21
parameter gradient blocks**:

```
loss     R2 5.0328483582   ref 5.0328480242   diff 3.3e-07
logits   max abs diff 1.2e-06  (relative 5.7e-07)
median relative error 7.3e-07, worst 1.0e-06     PASS
```

Flat across every block at f32 epsilon — rounding, not a defect. Verified
to have teeth: perturbing one gradient element by 50% makes it fail and
names the right block.

A **uniform** error across all blocks means the harness is wrong; **one
block far above the rest** means that operator is.

---

## 3. Closed

| stage | standing vs the best of PyTorch and JAX | what closed it |
|---|---|---|
| Tokenizer | **R2 9.6x AHEAD** | — |
| Accuracy | **at parity or better** | — |
| Projections / matmul forward | 1.0-1.7x behind MKL, **ahead of Eigen at some shapes** | `r2_linalg::gemm::sgemm` — blocked and packed Goto/BLIS, `MR x NR = 6 x 16`, `KC/MC/NC = 256/96/1024`, AVX2 micro-kernel chosen at runtime. 27-73 -> 118-195 GFLOP/s |
| `grad_A` (NT) | folded into the above | became `sgemm`'s NT case; 676 -> 77 ms on the output-head shape |
| `grad_B` (TN) | folded into the above | was the one gradient never parallelised; then became `sgemm`'s TN case; 465 -> 110 ms |
| Attention forward | 1.5-2.3x behind; **beats torch-explicit and JAX at 4,096 tokens** | `Op::Attention` — one tape node instead of 1,156, no slices, no materialised score matrix; plus a 4-accumulator dot and runtime AVX2 |
| Attention fwd+bwd | 2.1-2.7x behind | as above; backward recomputes probabilities rather than storing a quadratic buffer |
| Embedding | gather, not one-hot matmul | `Op::Embed`; and `requires` guards stopped an 8.4 GFLOP `grad_A` being computed into a leaf that needs no gradient |

Training went from **10.4x behind to 1.8x**. The single largest cause was
that no `target-cpu` is set, so every crate compiled for baseline x86-64 —
SSE2, and **no FMA at all**. AVX2 without packing buys 30%; packing without
AVX2 buys nothing; together they are 5.1x. Each cause hid the other, which
is why "the GEMM is not slow" survived so long as a settled claim.

AVX2 is selected at **runtime**, not by a build flag: the installer ships
one binary, and an illegal instruction on a user's older CPU is not a trade
for throughput. It also beat `-C target-cpu=native`, since only the kernel
needs the wide codegen.

**Pure Rust is not the ceiling.** PyTorch sends `aten::mm` to MKL's
hand-written assembly, which R2 will never ship. But JAX does not — XLA:CPU
sends `dot` to Eigen (`__xla_cpu_runtime_EigenBatchMatMulF32`; no
`__xla_cpu_runtime_OneDnn*` entry points), portable C++ templates with zero
assembly, and Eigen matches or beats MKL here. The gap was structure.

---

## 4. Open

Ranked by what each is worth at the shipping shape.

1. **`grad_B` (TN) is the slowest of the three GEMM cases.** Its `M` is the
   weight's *input* dim, so it has the fewest row-blocks to spread across
   cores. `grad_A` + `grad_B` together are 2.6x behind the best reference,
   weighted by calls per step — the largest remaining item.
2. **Attention is 1.5-2.7x behind `scaled_dot_product_attention`**, which
   blocks over keys and keeps the running softmax in registers. R2 already
   beats torch's explicit form and JAX; only the fused kernel leads.
3. **`level3::dgemm` is 11-28 GFLOP/s**, 5-12% of the f64 ceiling — it did
   not benefit from any of this. `sgemm` is a per-type macro, so f64 is a
   one-line instantiation; a column-major `C = A·B` is the row-major
   `Cᵀ = Bᵀ·Aᵀ`, i.e. the same kernel with operands swapped. That would
   lift `%*%` and every blocked LAPACK algorithm built on it.
4. **The embedding forward is 9-17x behind** but **0.07% of a step** — a
   lead on a tape-wide allocator defect (a fresh 2 MB buffer costs ~370 us
   in page faults against torch's ~19), not a target in itself.
5. **`pipeline_phases` does not auto-join** the way the LMO benchmarks do.
   Pairing its halves by hand across two windows produced a wrong claim
   once already.

---

## 5. Closed by measurement — do not retry

| attempt | result |
|---|---|
| Parallel zeroing of gradient buffers in `backward()` | 187.5 -> 191.5 s. A fork-join per buffer costs more than the memset saves |
| Splitting softmax's fused `exp`+`sum` to vectorise the sum | 31.05 -> 34.77 s. 16.4M elements/step, so the extra pass costs ~64 MB of traffic; the serial add was already hidden behind the `exp` |
| `max4` replacing `fold(NEG_INFINITY, f32::max)` | Worse. `f32::max` lowers to `llvm.maxnum`, which LLVM **can** vectorise as a reduction — unlike `+` |
| Moving `dot4` into a shared crate | Worse. It is called inside `#[target_feature(avx2)]` kernels, and a cross-crate fn is not reliably inlined there; the hottest loop silently lost its wide codegen |
| Querying `rayon::current_num_threads()` per GEMM call | 35.26 -> 36.69 s. Cache it |
| Shrinking the GEMM row-block when B is the large operand | grad_B 148 -> 159 ms. Every row-block re-streams the packed B panel |
| Porting `level3::dgemm`'s packed path to f32 | Rejected before writing: 5-12% of the f64 ceiling, no better than the naive f32 loop at 6-16% of its own |
| Materialising transposes for the gradient cases | 393 ms vs 326 ms |
| Flash blocking with an online softmax | No change — the K/V re-reads it removes were already served from L1. Kept anyway: right structure, and what makes long contexts survivable |
| Sparse embedding gradient (37x on the op) | Worth **0.07%** of a training step. Measure the share before optimising |

Two rules out of these. A dependency chain costs nothing when something
else is already the bottleneck. And "serial float reduction" is not one
problem — `+` cannot be reassociated by the compiler and `max` can.

---

## 6. Reproducing

```
# the headline
cargo run --release -p r2-train --example pipeline_phases
python benchmarks/llm/pipeline_phases.py

# correctness
cargo run --release -p r2-train --example dump_reference
python benchmarks/llm/accuracy_check.py

# per stage — the R2 half writes lmoN_r2.json, the Python half joins it
# and prints one table with a verdict against PyTorch AND JAX
cargo run --release -p r2-train --example lmo1_embedding   && python benchmarks/llm/lmo1_embedding.py
cargo run --release -p r2-train --example lmo14_grad_ab    && python benchmarks/llm/lmo14_grad_ab.py
cargo run --release -p r2-train --example lmo15_attention  && python benchmarks/llm/lmo15_attention.py

# kernel rate on the model's shapes
cargo run --release -p r2-tensor --example gemm_rate
```

All `--release` builds on this machine need
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`,
or fat LTO across ~70 test binaries runs for hours.

**This machine drifts ~10% over hours** — the same unchanged binary
measured 31.05 s and 34.10 s two hours apart. Interleave every A/B in one
window (stash, measure, pop, measure); nothing under ~10% is a result
without it. Only the step total counts: one change here measured 18% better
at op level and 4% worse on the training step, in the same hour.
