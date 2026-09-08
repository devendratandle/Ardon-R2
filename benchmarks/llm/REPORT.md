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
| **tokenize 19.4 MB** | **1.57 s** | 15.63 s | **R2 10.0x faster** |
| **train** | 31.23 s | **17.23 s** | **PyTorch 1.81x faster** |
| **TOTAL** | **34.26 s** | 34.56 s | **a dead heat** |

Five interleaved pairs on the training phase: 1.92x, 1.80x, 1.82x, 1.67x,
1.90x. **Training is 1.8x behind. The whole pipeline is level**, because R2
tokenises 10x faster and training is no longer far enough behind for that
to be irrelevant.

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
| Tokenizer | **R2 10x AHEAD** | — |
| Accuracy | **at parity or better** | — |
| Projections / matmul forward | 1.0-1.7x behind MKL, **ahead of Eigen at some shapes** | `r2_linalg::gemm::sgemm` — blocked and packed Goto/BLIS, `MR x NR = 6 x 16`, `KC/MC/NC = 256/96/1024`, AVX2 micro-kernel chosen at runtime. 27-73 -> 118-195 GFLOP/s |
| `grad_A` (NT) | folded into the above | became `sgemm`'s NT case; 676 -> 77 ms on the output-head shape |
| `grad_B` (TN) | folded into the above | was the one gradient never parallelised; then became `sgemm`'s TN case; 465 -> 110 ms |
| Attention forward | 1.5-2.3x behind; **beats torch-explicit and JAX at 4,096 tokens** | `Op::Attention` — one tape node instead of 1,156, no slices, no materialised score matrix; plus a 4-accumulator dot and runtime AVX2 |
| Attention fwd+bwd | 2.1-2.7x behind | as above; backward recomputes probabilities rather than storing a quadratic buffer |
| `softmax_ce` | 155.6 -> 66.4 ms, 2.3x | fused around the log-sum-exp. `-ln(softmax(x)[t])` is `lse - x[t]`, so the loss needs a per-row max and sum and never the probabilities. It had been materialising the whole `2048 x 8000` probability matrix — 64 MB — to read 2,048 values out of it, and building it a SECOND time in the backward. `lse` is now kept from the forward. Also better conditioned: the old form clamped with `.max(1e-30)`, capping a confidently-wrong prediction at a loss of 69 |
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

## 4. Where a step goes now

`cargo run --release -p r2-train --example step_census`, at the shipping
shape. This is the map for anything done next; every target picked without
it this session was picked wrongly.

| stage | ms/step | share |
|---|---:|---:|
| `sgemm` x3 (forward, grad_A, grad_B) | 500 | **46%** |
| **unattributed** | **359** | **33%** |
| `softmax_ce` fwd+bwd | 66 | 6% |
| attention, 4 layers | 50 | 5% |
| `backward()`'s blanket gradient zeroing | 31 | 3% |
| silu forward x4 | 22 | 2% |
| tape's copy of every parameter | 12 | 1% |
| rmsnorm forward x8 | 6 | 1% |
| optimizer + flatten + writeback | 4 | <1% |

The step's tape holds **71.9M elements across 3,000-odd nodes for a 7.2M
parameter model** — 10x the model, each node owning a value buffer and a
gradient buffer. Allocation churn was measured and is NOT the cost (10 ms);
the buffers are lazily paged, so the fault cost is bundled into whichever
op writes them.

**The unattributed third is elementwise work and its backward** — `add`,
`mul`, `silu` backward (an `exp` per element over 6.3M elements), `rmsnorm`
backward, RoPE, and the memory traffic of writing every intermediate. It
has not been broken down further, and it is the largest unexamined item
after `sgemm`.

---

## 5. Open

Ranked by the census in section 4, not by how interesting they are.

1. **`sgemm` is 46% of the step, and the deficit is PARALLELISM, not the
   kernel.** `--example gemm_scaling` separates the two:

   | block | case | M | 1 thread | 6 threads | scale |
   |---|---|---:|---:|---:|---:|
   | output head | NN | 2048 | 55.9 GF/s | 150.5 | 2.69x |
   | output head | NT | 2048 | 60.6 | 148.0 | 2.44x |
   | output head | **TN** | **256** | **58.4** | **68.3** | **1.17x** |
   | ffn w1/w3 | NN | 2048 | 63.3 | 198.4 | 3.14x |
   | ffn w1/w3 | **TN** | **256** | **62.3** | **85.6** | **1.37x** |
   | ffn w2 | TN | 768 | 62.9 | 167.3 | 2.66x |

   **Single-threaded, all three cases are identical — 56-64 GF/s.** The
   micro-kernel does not care about the transpose. Threaded, they diverge
   entirely by `M`, because parallelism runs over ROW-BLOCKS of C and TN's
   `M` is the weight's input dim (256), not the token count (2048). At
   `MC = 96` that is 3 blocks for 6 cores; `ffn w2`, whose `M` is 768,
   scales 2.66x instead of 1.17x, which is the same explanation confirming
   itself.

   Two things follow. **(a)** TN needs parallelism over a second loop —
   MKL and BLIS use a thread mesh over `jc` x `ic` chosen per shape, while
   R2 threads one loop. Row-major C makes column bands non-contiguous, so
   it needs either an unsafe disjoint split or computing `Cᵀ = Bᵀ·Aᵀ`.
   **(b)** Even the healthy cases scale only 2.4-3.3x on 6 cores, so half
   the machine is unused everywhere — the packed B panel is re-streamed by
   every row-block, and A is re-packed once per column panel.

   At ~60 GF/s single-thread against ~77 GF/s of single-core peak, the
   kernel itself is at 78% — that last stretch is MKL's hand-written
   assembly, software pipelining and explicit prefetch, and it is the part
   that is genuinely hard in safe Rust. **The parallelism is not.**
2. **A third of the step is unattributed** and has never been broken down:
   elementwise ops and their backwards. `silu` backward alone is an `exp`
   per element over 6.3M elements. Nobody has looked, which by this
   session's record makes it the most likely place for a surprise.
3. **Attention is 1.5-2.7x behind `scaled_dot_product_attention`**, which
   blocks over keys and keeps the running softmax in registers — but it is
   only 5% of a step, so closing it entirely buys ~3%. R2 already beats
   torch's explicit form and JAX; only the fused kernel leads.
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

## 6. Closed by measurement — do not retry

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

Fusing `softmax_ce` belongs in this table as much as in the closed one: it
is a real 2.3x on the op and removes 128 MB of per-step allocation, but 89
ms of a 1,065 ms step is 8%, under this machine's drift, and it did **not**
measurably move the training ratio (1.82x before, 1.81x after). Kept for
the allocation and the conditioning, not claimed as a speedup.

Two rules out of these. A dependency chain costs nothing when something
else is already the bottleneck. And "serial float reduction" is not one
problem — `+` cannot be reassociated by the compiler and `max` can.

---

## 7. Reproducing

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
