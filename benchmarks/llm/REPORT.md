# Ardon-R2 vs PyTorch — LLM model training

**Current status, 2026-09-08.** This is the only performance report for LLM
training. Earlier ones were deleted rather than kept: a superseded number is
worse than no number. Everything below describes the code as it stands, not
how it got there.

Both sides were run on this machine in the **same window**, interleaved.
Nothing is carried over from an earlier session.

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
| learn BPE merges | 1.41 s | 1.71 s | R2 1.21x |
| **tokenize 19.4 MB** | **1.54 s** | 17.01 s | **R2 11.0x faster** |
| **train** | 24.52 s | **17.05 s** | **PyTorch 1.44x faster** |
| **TOTAL** | **27.47 s** | 35.75 s | **R2 1.30x faster** |

Two interleaved pairs, both giving 1.44x on training (24.52/17.05 and
24.88/17.22). **Training is 1.4x behind; the whole pipeline is ahead**,
because R2 tokenises about 11x faster.

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

Reading it: a **uniform** error across all blocks means the harness is
wrong; **one block far above the rest** means that operator is.

---

## 3. Standing, stage by stage

Measured against the best of PyTorch and JAX. Each R2 benchmark writes
`lmoN_r2.json`; its Python half joins that and prints one table with a
per-row verdict, so the comparison cannot be quietly skipped.

| stage | R2 vs best reference | share of a step |
|---|---|---:|
| Tokenizer | **R2 ~10x AHEAD** | — |
| Accuracy | **at parity or better** | — |
| `sgemm` — forward, `grad_A`, `grad_B` | 1.0-1.7x behind MKL | **39%** |
| Attention forward | 1.5-2.3x; **beats torch-explicit and JAX at 4,096 tokens** | 5% |
| Attention fwd+bwd | 2.1-2.7x | (within the 5%) |
| `softmax_ce` | fused; not separately compared | 7% |
| Embedding | 1.9-3.8x fwd+bwd | 0.07% |

Only MKL's `sgemm` and `scaled_dot_product_attention` are clearly ahead,
and both are hand-written assembly R2 does not ship.

### `sgemm` rates on the shapes a step runs

`cargo run --release -p r2-tensor --example gemm_rate` — GFLOP/s.

| block | m x k x n | calls/step | NN | NT (`grad_A`) | TN (`grad_B`) |
|---|---|---:|---:|---:|---:|
| output head | 2048x256x8000 | 1 | 149.8 | 139.0 | 180.3 |
| ffn w1/w3 | 2048x256x768 | 8 | 112.7 | 209.3 | 144.1 |
| ffn w2 | 2048x768x256 | 4 | 214.2 | 205.6 | 138.8 |
| q/o proj | 2048x256x256 | 8 | 154.1 | 167.7 | 134.0 |
| k/v proj | 2048x256x128 | 8 | 168.2 | 153.0 | 123.7 |

MKL measures 171-286 on the same shapes; the i-k-j loop this replaced
managed 27-73.

---

## 4. Where a step goes

`cargo run --release -p r2-train --example step_census`. **This is the map
for anything done next.** Every target picked without it was picked wrongly.

Every stage is measured forward AND backward, at the count a step runs it.

| stage | ms/step | share |
|---|---:|---:|
| `sgemm` x3 (forward, `grad_A`, `grad_B`) | 371 | **44%** |
| unattributed | 111 | 13% |
| silu x4 | 68 | 8% |
| `softmax_ce` | 67 | 8% |
| attention, 4 layers | 50 | 6% |
| mul x4 | 36 | 4% |
| rmsnorm x8 | 35 | 4% |
| `backward()`'s blanket gradient zeroing | 27 | 3% |
| add x8 | 23 | 3% |
| RoPE x8 | 22 | 3% |
| tape's copy of every parameter | 12 | 1% |
| tape allocation churn | 10 | 1% |
| embed | 6 | 1% |

The step's tape holds **71.9M elements across ~3,000 nodes for a 7.2M
parameter model** — 10x the model, every node owning a value buffer and a
gradient buffer. Allocation churn was measured and is NOT the cost (10 ms):
the buffers are lazily paged, so the fault cost sits inside whichever op
writes them.

---

## 5. Open

Ranked by the census above, not by how interesting they are.

1. **`silu` is 8% of a step** — its backward computes an `exp` per element
   across 6.3M elements, and the forward another 6.3M. That is the largest
   remaining item after the GEMM, and the same shape of problem RoPE turned
   out to be: transcendental functions in an inner loop where the maths does
   not require them per element.
2. **`sgemm` parallel efficiency is 2.6-3.4x on six cores.** Serially the
   kernel runs 56-67 GFLOP/s against ~77 of single-core peak — 78%, and
   that last stretch is MKL's hand-written assembly, software pipelining
   and explicit prefetch. But the threading leaves half the machine unused
   even where row-blocks are plentiful: every row-block re-streams the
   whole packed B panel, and A is re-packed once per column panel. A thread
   mesh over `jc` x `ic`, as BLIS uses, is the fix — and unlike the
   assembly, it is not a language problem.
3. **Attention is 1.5-2.7x behind `scaled_dot_product_attention`**, which
   blocks over keys and keeps the running softmax in registers. Only 5% of
   a step, so closing it entirely buys about 3%.
4. **`level3::dgemm` is 11-28 GFLOP/s**, 5-12% of the f64 ceiling — it did
   not benefit from any of this work. `sgemm` is a per-type macro, so f64
   is a one-line instantiation; a column-major `C = A·B` is the row-major
   `Cᵀ = Bᵀ·Aᵀ`, i.e. the same kernel with operands swapped. That would
   lift `%*%` and every blocked LAPACK algorithm built on it.
5. **13% of a step is still unattributed** — the memory traffic of writing
   every intermediate value and gradient buffer, spread across ops rather
   than concentrated anywhere.
6. **`pipeline_phases` does not auto-join** the way the LMO benchmarks do.
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
| Shrinking the GEMM row-block simply to get more of them | grad_B 148 -> 159 ms. Every row-block re-streams the packed B panel, and for `grad_B` that panel is the large operand. The transpose path fixed this properly instead |
| Porting `level3::dgemm`'s packed path to f32 | Rejected before writing: 5-12% of the f64 ceiling, no better than the naive f32 loop at 6-16% of its own |
| Materialising transposes to feed a naive kernel | 393 ms vs 326 ms. Materialising only the RESULT of a transposed GEMM, to fix parallelism, is a different thing and does pay — see `gemm.rs` |
| Flash blocking with an online softmax | No measurable change; the K/V re-reads it removes were already served from L1. Kept: right structure, and what makes long contexts survivable |
| Sparse embedding gradient | 37x on the op, **0.07%** of a step. Measure the share before optimising |
| Recomputing RoPE's angles per element | 138.9 ms/step, 14.5%, for two multiplies and two adds per pair. `powf` + `sin_cos` were being evaluated per (row, head, pair) when the angle depends only on (position, pair): 262,144 transcendental pairs where 2,048 are distinct. A precomputed table made it 22.1 ms, 6.3x, bit-identical |
| Fusing `softmax_ce` | 2.3x on the op (155.6 -> 66.4 ms) but no measurable change to the training ratio — 89 ms of a 1,065 ms step is under this machine's drift. Kept for the 128 MB it stops allocating and for better conditioning, not claimed as a speedup |

Two rules out of these. **A dependency chain costs nothing when something
else is already the bottleneck.** And **"serial float reduction" is not one
problem** — `+` cannot be reassociated by the compiler and `max` can.

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

# where a step goes, and the kernel's rate and parallel scaling
cargo run --release -p r2-train  --example step_census
cargo run --release -p r2-tensor --example gemm_rate
cargo run --release -p r2-tensor --example gemm_scaling
```

Every `--release` build on this machine needs
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`,
or fat LTO across ~70 test binaries runs for hours.

**This machine drifts about 10% over hours** — the same unchanged binary
measured 31.05 s and 34.10 s two hours apart. Interleave every A/B in one
window (stash, measure, pop, measure); nothing under ~10% is a result
without it. And only the step total counts: one change measured 18% better
at op level and 4% worse on the training step, in the same hour.
