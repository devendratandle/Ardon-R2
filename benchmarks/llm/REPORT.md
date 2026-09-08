# Ardon-R2 vs PyTorch — LLM model training

**Current status, 2026-09-09.** This is the only performance report for LLM
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
config   dim 256, 4 layers, vocab 8,000, ffn 768 — 7.24M parameters
run      500 steps x 32 x 64 = 1,024,000 tokens (section 1, the headline)
         30 steps x 32 x 64 for the op-level arms (sections 3-4)
```

---

## 1. Result

**R2 trains 1.10x FASTER than PyTorch, and 1.15x faster end to end.**

This is a real training run rather than a step benchmark: **500 Adam steps
on TinyStories, 1,024,000 tokens, a 7.24M-parameter model**, both sides
from the SAME initial weights on the SAME token stream, in one window. The
machine held **2375 MHz before, between and after every run** — verified,
not assumed, so these seconds are representative rather than throttled.

| pair | R2 train | PyTorch train | |
|---|---:|---:|---|
| 1 | **269.72 s** | 297.32 s | **R2 1.10x** |
| 2 | **271.79 s** | 299.07 s | **R2 1.10x** |

| phase | R2 | PyTorch | |
|---|---:|---:|---|
| **tokenize 19.1 MB** | **1.74 s** | 15.56 s | **R2 9.0x faster** |
| **train, 500 steps** | **270 s** | 298 s | **R2 1.10x faster** |
| **TOTAL** | **271.5 s** | 312.9 s | **R2 1.15x faster** |

3,797 vs 3,444 tokens/s; 539 vs 595 ms/step. Held-out loss is **4.0499 on
both sides** and unchanged from before any of this work — the three
optimisations that produced it removed work, they did not approximate
anything.

### How it got here: three pieces of pure waste

`--example step_census` was re-run after the earlier round of work, and the
three largest items after the GEMM turned out not to be arithmetic at all.
None of them is something PyTorch does. Each was measured on its own in
interleaved pairs at 60 steps:

| removed | was | measured |
|---|---:|---:|
| `backward()` memsetting gradient buffers that are already zero | 27.7 ms, 4.8% | **8.0%** |
| Flattening 7.24M params in and out to reach a flat `Adam::step` | 32.3 ms, 5.6% | **9.6%** together |
| Cloning every parameter onto the tape each step | 11.4 ms, 2.0% | (with above) |

Compounded, ~18% off the step — against a starting deficit of 1.03-1.05x,
which is what turned a gap into a lead.

The zeroing is the sharpest example. `Tape::push` allocates every gradient
buffer with `vec![0.0; n]` and `train_step` builds a **fresh tape every
step**, so the blanket reset was writing zeros over zeros — 71.86M
elements, **287 MB of memset**, every step, achieving nothing. Worse than
the write, `fill` TOUCHES every page, forcing resident what the allocator
had handed out as untouched, including buffers for `requires = false`
nodes no backward ever writes.

Note that threading that memset was tried in v0.3.9 and measured WORSE
(187.5 -> 191.5 s). That was a correct measurement of the wrong fix. The
lesson is not about zeroing: **when an optimisation makes something
slower, ask whether the work should exist before making it faster.**

### The pipeline advantage is a SHORT-RUN property

An earlier version of this section led with **"R2 1.71x faster end to
end"**, measured on a 30-step arm. That number was real and it does not
survive at any length anyone would actually train at, so it is corrected
here rather than left standing.

At 30 steps, training is ~21 s against ~3 s of tokenizing, and an 11x
tokenizer win carries the total. At 500 steps tokenizing is **0.6% of R2's
pipeline**, so the 1.15x total above is now carried by the TRAINING ratio
rather than by the tokenizer — which is why it is a durable number where
the old one was not.

**Quote the training ratio, not the pipeline ratio.** Only a run long
enough to be a model rather than a benchmark shows this; no amount of
op-level measurement would have.

### The corpus is never held in memory

Training reads its tokens through a memory map, and tokenizes to that file
in streamed chunks, so nothing ever materialises either the corpus text or
the id stream:

```
memory  token stream mapped: 96 B resident vs 37.7 MB as a Vec<usize>
```

The old path cost roughly **four times the corpus** before a step ran — the
text, the `Vec<u32>` from encoding, and a `Vec<usize>` at eight bytes per
token. At 19.4 MB that was ~76 MB and merely wasteful; at 700 MB it is
~2.7 GB and it is the reason the run does not start at all. Resident cost
is now O(batch x seq).

Tokenizing streams too, in 1 MB chunks, and the chunk boundary is the part
that had to be right: **a chunk must end on a GPT-2 pre-token boundary**,
because merges never cross one. Splitting on newlines looks obviously safe
and is not — measured over 3 MB at 64 KB chunks, the ids diverged from
whole-text encoding (`[1293, 400]` against `[10, 470]` for the same text)
because a newline plus the following space is a single pre-token. Cutting
on `bpe::pretokenize`'s last boundary instead reproduces the whole-corpus
token count exactly, 4,615,591 either way, and
`chunked_encoding_matches_whole_text_on_pretoken_boundaries` pins it.

Tokenizing also happens **once**: a stamp records the corpus path, its
size, the split point and the two tokenizer settings, so a later run over
the same corpus maps the existing files and skips both BPE training and
encoding (3.15 s -> 0 s here).

### Learning is IDENTICAL, not merely comparable

Both sides load the same dumped initial weights and read the same token
ids, so this is a far stronger statement than the bits/byte comparison it
replaces (which was 2.75% apart, from different init RNGs).

| step | R2 | PyTorch | diff |
|---:|---:|---:|---:|
| 1 | 9.1095 | 9.1095 | 0.0000 |
| 100 | 5.3016 | 5.3016 | 0.0000 |
| 200 | 4.3790 | 4.3790 | 0.0000 |
| 300 | 4.2990 | 4.2990 | 0.0000 |
| 400 | 3.8118 | 3.8118 | 0.0000 |
| 500 | 3.6091 | 3.6091 | 0.0000 |

All eleven checkpoints agree to four decimals across 500 optimizer steps.
Held-out loss **4.0499 on both sides** (1.4111 bits/byte, perplexity
57.39), on the tail of the corpus neither trained on. Both models then
generate the *same sentence* from "Once upon a time":

> ", there was a little girl named Lucy. She was three years old and loved
> to play with her friends. One day, she saw a big, old man..."

**The harness gates itself.** Before training, both sides evaluate the
held-out set from those identical weights — one forward pass in two
languages, which must agree to f32 rounding. It reports **2.04e-05** and
aborts the whole comparison if it exceeds 2e-3, because a disagreement
there means the two sides are not running the same model and every row
below it would be meaningless.

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
| output head | 2048x256x8000 | 1 | 199.9 | 257.7 | 221.4 |
| ffn w1/w3 | 2048x256x768 | 8 | 240.3 | 233.2 | 168.7 |
| ffn w2 | 2048x768x256 | 4 | 226.0 | 238.4 | 196.0 |
| q/o proj | 2048x256x256 | 8 | 192.1 | 204.0 | 159.5 |
| k/v proj | 2048x256x128 | 8 | 182.9 | 142.8 | 126.8 |

Weighted by calls per step, **196.7 GFLOP/s**. MKL measures 171-286 on the
same shapes, so several are now inside its range; the i-k-j loop this
replaced managed 27-73.

The micro-kernel is **hand-written with AVX2 intrinsics** — twelve named
`__m256` accumulators, so the 6x16 tile's registers are placed rather than
hoped for. Leaving it to the optimiser cost 20-25%: 56-67 GFLOP/s serial
against 73-84 now. This is what BLIS does for its portable kernels too,
and it stays pure Rust with no C dependency — `unsafe` here buys explicit
registers, not a foreign library.

---

## 4. Where a step goes

`cargo run --release -p r2-train --example step_census`. **This is the map
for anything done next.** Every target picked without it was picked wrongly.

The census warms to the sustained clock before measuring and re-takes the
step total at the end, printing the drift — an earlier version timed the
step at boost clock and the stages under throttle, and the stages summed
to 155%. Shares are only meaningful while that drift is small; this run
reported +3.2%.

| stage | ms/step | share |
|---|---:|---:|
| `sgemm` x3 (forward, `grad_A`, `grad_B`) | 527 | **57%** |
| rmsnorm x8 | 65 | 7% |
| `softmax_ce` | 61 | 7% |
| mul x4 | 50 | 5% |
| attention, 4 layers | 50 | 5% |
| silu x4 | 48 | 5% |
| RoPE x8 | 37 | 4% |
| add x8 | 33 | 4% |
| `backward()`'s blanket gradient zeroing | 31 | 3% |
| tape's copy of every parameter | 20 | 2% |
| tape allocation churn | 19 | 2% |
| embed | 10 | 1% |

The step's tape holds **71.9M elements across ~3,000 nodes for a 7.2M
parameter model** — 10x the model, every node owning a value buffer and a
gradient buffer. Allocation churn was measured and is NOT the cost;
the buffers are lazily paged, so the fault cost sits inside whichever op
writes them.

**`sgemm` is now 57% of a step and nothing else is above 7%.** The
remaining work is one large item and a long thin tail.

---

## 5. Open

Ranked by the census above, not by how interesting they are.

0. **Re-run `--example step_census` — the shares below predate the three
   removals in section 1.** Fixing the top item promotes whatever was
   hiding under it, and a stale census is how the split-K attempt below
   came to be aimed at a problem that had already been fixed.
1. **`sgemm` is 57% of a step**, and after it nothing is above 7%. It
   scales 2.7-3.8x against a measured machine ceiling of 5.53x, so roughly
   another 1.4-2x of parallel efficiency is available — see item 2.
2. **`sgemm` scales 2.7-3.8x against a machine ceiling of 5.53x.**
   `--example scaling_ceiling` measures what this machine can actually give
   a perfectly parallel workload — 5.53x on six cores, 92% efficient — so
   the bar is that, not 6.00x. Threading `pack_b` took scaling from 1.9-2.8
   to 2.7-3.8; the remaining serial fraction is `pack_a` (per row-block, so
   already inside the parallel region but repeated per column panel) and
   the fork-joins themselves. A BLIS-style thread mesh over `jc` x `ic`
   would let threads sharing a B panel avoid re-packing it.
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
   once already. `tinystories_train` (section 1) does auto-join and should
   be preferred for anything end-to-end; `pipeline_phases` survives only
   for its byte-level-vs-BPE arm comparison.
7. **Parquet is still unwired.** `r2-arrow/src/parquet_io.rs` exists and
   nothing in the training path calls it, so a corpus that arrives as
   Parquet has to be converted first. The memmap half of this is now done
   — see below.

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
| Assuming the 6-core scaling ceiling is 6.00x | It is **5.53x** here, measured on register-resident work with no memory traffic (`--example scaling_ceiling`). Judge every parallel speedup against that. The first version of that probe used constant inputs, LLVM folded the whole loop away, and it reported 0.00x — seed through `black_box` |
| Assuming poor GEMM scaling meant a bad thread mesh | It meant a SERIAL `pack_b` outside the parallel region. A ~22% serial fraction predicts 1/(0.22 + 0.78/6) = 2.85x by Amdahl, and 1.9-2.8x was measured. Threading the pack fixed it without touching the mesh — do the arithmetic before building the complicated thing |
| Threading `add` and `mul` | No change (35.8 -> 35.4 ms, 23.3 -> 23.1). Three memory streams per flop makes them bandwidth-bound, and bandwidth does not thread. `silu`, which has a real `exp` per element, went 68 -> 36 ms on the same change — the difference between the two IS the diagnosis |
| Storing `silu`'s sigmoid to skip the backward's `exp` | Not attempted: 6.3 MB per node, ~25 MB a step, to save compute on a machine already bound by memory traffic. Wrong direction |
| Recomputing RoPE's angles per element | 138.9 ms/step, 14.5%, for two multiplies and two adds per pair. `powf` + `sin_cos` were being evaluated per (row, head, pair) when the angle depends only on (position, pair): 262,144 transcendental pairs where 2,048 are distinct. A precomputed table made it 22.1 ms, 6.3x, bit-identical |
| Leaving `exp` to libm | 32.8M scalar `exp` calls a step in `softmax_ce` and 12.6M in `silu`, none of which vectorise, because a call is a call. A hand-written AVX2 Cephes reduction took `softmax_ce` 149.3 -> 61.4 ms and `silu` 59.1 -> 48.2. Watch the underflow: `2^n` written into the exponent field wraps into the sign bit below x = -87.34 and returns **-inf**, which the accuracy test caught before it reached the trainer |
| Fusing `softmax_ce` | 2.3x on the op (155.6 -> 66.4 ms) but no measurable change to the training ratio — 89 ms of a 1,065 ms step is under this machine's drift. Kept for the 128 MB it stops allocating and for better conditioning, not claimed as a speedup |

| **Splitting N instead of M in `sgemm`** (the `jc` half of a BLIS mesh, as Eigen's `parallelize_gemm` does) | Isolated kernel WORSE (q/o proj TN 155 -> 108 GF/s); step total 30.93/35.22/32.46 against 33.48/30.71/31.86, i.e. noise with the pairs split 1-2. **The reason is structural and worth keeping:** the present partitioning is packing-OPTIMAL — B is packed once per `(jc, pc)` and shared, A once per row-block, nothing duplicated. Splitting N makes EVERY worker pack ALL of A; splitting both without sharing duplicates A `n_jc` times and B `n_ic` times. Any fork-join mesh pays redundant packing that exceeds the shared-panel contention it removes. A real BLIS mesh needs a persistent thread TEAM with barriers and shared packed buffers, which rayon's fork-join model does not express |
| **Shrinking the shared B panel** (`R2_GEMM_NC`) so each worker streams less of it | 60 steps: **1024 -> 30.41 s**, 512 -> 32.20, 256 -> 32.42. The shipped default is already the best of the three; a narrower panel re-streams A more often than it saves on B |
| **Split-K in `sgemm`** — parallelising the depth loop, each worker owning a private C accumulator | Four interleaved pairs at 60 steps: 35.23/38.22, 35.38/36.23, 35.61/35.97, 36.15/35.89 s. Mean 2.8%, and pair 1's baseline is a first-run outlier — the other three are +2.4%, +1.0%, **−0.7%**, straddling zero. Under this machine's ~10% bar, so not a result. The premise was wrong too: `mc_blk` already shrinks until there are **6-84 row-blocks**, so `grad_B` was never block-starved, and removing 7 of its 8 fork-joins changed nothing measurable — **the TN shortfall is memory bandwidth, not synchronisation.** Op-level it looked actively bad (q/o proj TN 157→118 GF/s with a serial reduction, 168 with a threaded one) which is the usual isolated-kernel scatter; only the step total settled it |
| Timing PyTorch's tokenizer BEFORE its training loop, in one process | It holds a ~19 MB string and a 4.6M-element id list alive through training, and PyTorch's measured training time moved **20.7%** between two runs fifteen minutes apart (302.85 -> 365.46 s) while R2's moved 5.6%. That asymmetry — one side moving four times as much as the other — is the signature of a perturbation, not of drift. Moved after training; the ratio returned to 1.04x. **A measurement that shares a process with its neighbour must run after it** |
| Reading the pipeline ratio as a property of the two implementations | It is a property of the RUN LENGTH. 1.71x at 30 steps, 1.02x at 500, same code both times — tokenizing is 0.5% of a real training pipeline. Quote the training ratio |

Three rules out of these. **A dependency chain costs nothing when something
else is already the bottleneck.** **"Serial float reduction" is not one
problem** — `+` cannot be reassociated by the compiler and `max` can. And
**a ratio measured on a short run is not the same quantity as the ratio on
a real one**; benchmark at the length the claim is about.

---

## 7. Reproducing

```
# the headline — a real 500-step training run, both sides from the same
# weights on the same tokens. The Python half joins R2's manifest and
# prints the one table, so the halves cannot be paired by hand.
cargo run --release -p r2-train --example tinystories_train
python benchmarks/llm/tinystories_train.py

# byte-level vs BPE arms (does NOT auto-join — read section 5, item 6)
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
