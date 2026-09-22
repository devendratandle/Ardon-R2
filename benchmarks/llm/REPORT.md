# Ardon-R2 vs PyTorch — LLM model training

**Current status, 2026-09-22.** This is the only performance report for LLM
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

**R2 trains 1.5-1.66x FASTER than PyTorch at 7M parameters, 1.40-1.63x at 40M, and 1.25-1.28x faster end to end.**

This is a real training run rather than a step benchmark: **300 Adam steps
on TinyStories, 614,400 tokens, a 7.24M-parameter model**, both sides from
the SAME initial weights on the SAME token stream, in one window
(2026-09-19). The machine held **2375 MHz before, between and after every
run** — verified, not assumed, so these seconds are representative rather
than throttled.

| pair | R2 train | PyTorch train | |
|---|---:|---:|---|
| 1 | **168.80 s** | 196.06 s | **R2 1.16x** |
| 2 | **159.85 s** | 183.77 s | **R2 1.15x** |
| 3 | **147.32 s** | 174.67 s | **R2 1.19x** (cold machine after a reboot; fastest run on both sides) |
| 4 | **140.63 s** | 172.30 s | **R2 1.23x** (with the tape's value-buffer pool, below) |
| 5 | **131.82 s** | 169.83 s | **R2 1.29x** (gradients pooled too, first writer assigns) |
| 6 | **126.71 s** | 174.97 s | **R2 1.38x** (tiled attention; PyTorch's run drifted up in this window) |
| 7 | **38.99 / 38.89 s** (100 steps) | 57.9 / 58.5 s | **R2 1.49-1.50x** (attention re-blocked, below; 100-step pairs, 2026-09-20) |
| 8 | **36.33 / 35.86 s** (100 steps) | 60.4 / 59.4 s | **R2 1.66x / 1.66x** (`sgemm` column-group partition, section below; 2026-09-20) |

| phase | R2 | PyTorch | |
|---|---:|---:|---|
| **tokenize 19.1 MB** | **1.70 s** | 16.90 s | **R2 9.97x faster** |
| **train, 300 steps** | **168.8 s** | 196.1 s | **R2 1.16x faster** |
| **TOTAL** | **170.5 s** | 213.0 s | **R2 1.25x faster** |

3,640-3,844 vs 3,134-3,343 tokens/s; 533-563 vs 613-654 ms/step. Held-out
loss **4.3113 vs 4.3114**, perplexity 74.54 on both, the training loss
identical to four decimals at all eleven checkpoints, and both models
generate the same sentence from the same prompt.

The previous standing figure was 1.10x (two pairs at 500 steps,
2026-09-09: 269.72/297.32 and 271.79/299.07 s). The step between them is
one change — the tape is freed off the training thread — and the two
harnesses below show where it came from.

### Forward, backward, optimizer — and the fourth piece of waste

`--example phase_split` and `benchmarks/llm/phase_split.py` cut one step
into its phases on both sides (same model, same 2,048 tokens/step, two
interleaved pairs, 2026-09-19):

| phase | R2 | PyTorch | |
|---|---:|---:|---|
| forward | 154 / 156 ms | 233 / 228 | **R2 1.48x** |
| backward | 295 / 298 | 415 / 420 | **R2 1.41x** |
| optimizer phase | 44 / 48 | 29 / 30 | PyTorch 1.5x |

R2 leads both compute phases; the optimizer phase was the odd one out, so
it was split again: **Adam itself is 10.7 ms** (PyTorch's is 29 — R2 2.7x
faster) and the other **38 ms was `drop(tape)`** — returning ~3,000 value
and gradient buffers (~287 MB) to the allocator one at a time, 7.7% of a
step, more than Adam and rmsnorm together. No op census had a line for it
because freeing is not an op.

First fix: the tape was dropped on a background thread. A/B on the step
total, three interleaved pairs: 487 -> 455, 510 -> 455, 497 -> 485 ms.
That hid the free; it did not remove it, and the next step still
page-faulted the same ~143 MB of activations back in.

Second fix, **the value-buffer pool** (`BufPool` in r2-autograd): every
forward op writes its whole output, so the previous step's value buffers
are handed to the next tape by exact length and reused without zeroing.
`sgemm_assign_into` writes `C = A·B` into a buffer whose contents are
ignored (first depth slab assigns, later ones accumulate), so the 64 MB
logits buffer needs no memset before the output-head GEMM; a test pins
it bit-identical to `sgemm` across NN/NT/TN, multi-slab K and the
transposed-result path. Gradient buffers are NOT pooled — backward
accumulates into them, and zeroing a reused one is the memset the
`differentiated` fix removed — so they stay calloc'd and are freed off
the training thread.

Measured with `--example phase_split` (`R2_POOL=0/1`): tape drop **43 ->
0.1 ms**, forward **153 -> 146 / 159 -> 134 ms** (the faults). At run
length, two interleaved pairs of 300 steps, R2 alone: **147.84 -> 141.53
s and 150.10 -> 140.63 s** (-4.3%, -6.3%), loss curve identical. Under
this machine's 10% bar as a single pair; taken as real because the
mechanism was measured directly and both pairs agree. Against PyTorch in
the same window: 140.63 vs 172.30 s, **1.23x**.

Third fix, **gradients pooled too — and never zeroed.** `embed_probe`
showed the embedding's scatter-add costs 413 us into a warm table
gradient and 2,126 us into a fresh calloc'd one: the first touch of each
page is a fault plus a kernel memset. Pooling the gradient buffers and
zeroing them in one parallel pass measured NO better (143.1 -> 152.3 s
and 143.4 -> 142.5: DRAM write bandwidth either way, closed below). So
the backward now tracks, per node, whether its gradient has been written
in this pass: the FIRST consumer to contribute ASSIGNS, later ones
accumulate — `sgemm_assign_into` for the GEMM gradients, an FMA with a
zero addend in the `silu` and `softmax_ce` kernels (bit-identical to
accumulating into zero), `copy_from_slice` for the elementwise arms; the
arms that write only part of a buffer (a scatter, a slice, a masked
triangle) zero it on their first write. Nothing is memset, nothing is
faulted. A test runs a transformer-shaped graph three times through one
pool against a fresh tape and requires every value and gradient to be
identical. Two interleaved 300-step pairs, R2 alone: **139.96 -> 131.56
s and 140.15 -> 131.82 s** (-6.0%, -5.9%), loss curve identical. Against
PyTorch in the same window: 131.82 vs 169.83 s, **1.29x**.

The embedding item (LMO-1) is closed by these two: the gather was
already at parity (62 vs 65 us), and its backward's cost was the fault on
a fresh 8 MB table gradient, which no longer exists.

### Attention, tiled the way a GEMM is (LMO-15)

Both directions computed every score as `dot4(q_row, k_row)`: eight FMAs
and a HORIZONTAL REDUCTION per (query, key) pair, and the reduction was
the cost. `scaled_dot_product_attention` forms a tile of scores
lane-parallel — one query element broadcast against a row of keys — and
never reduces. The kernels now pack Kᵀ (and Vᵀ for the backward) once
per sequence and kv-head, compute 4 x 16 score and dP tiles in the
GEMM's register-tile form, and keep the three accumulations (dQ, dK, dV)
in broadcast form — key-outer in the backward, so each dK/dV row is read
and written once per four-query block instead of once per pair. The
online-softmax rescaling went with it (it had measured nothing here; the
score row of a query block is 4 x seq floats).

`lmo15_attention` (+ `.py`), 2,048 tokens, same window:

| | before | after | SDPA |
|---|---:|---:|---:|
| forward | ~2,000 us (1.5x behind) | **1,136 us** | 1,344 — R2 ahead |
| backward | 8,636 us | **6,378 us** | ~4,300 |
| fwd+bwd | 10,889 us (1.9x) | **8,643 us (1.5x)** | 5,619 |

Ahead of SDPA on the forward at 2,048 and 4,096 tokens (0.8x, 0.6x).
Finite-difference gradient test and decomposition test pass. Two
300-step pairs, R2 alone: **131.18 -> 128.05 s and 132.46 -> 126.71 s**
(-2.4%, -4.3%), loss curve identical to four decimals. Against PyTorch
in the same window: 126.71 vs 174.97 s, **1.38x**.

#### The second cut: why the medium model was only 1.05x (2026-09-20)

The medium model (dim 768, seq 256, batch 8; section below) trained at
parity while the small one led by 1.38x, and the reason was found by a
full in-situ census of both sides at that size — R2's forward census (a
timer between tape pushes) and backward census by op, PyTorch's
`torch.profiler` self-time per aten op, forward and backward profiled
separately (`op_profile.py`). Every non-GEMM op in R2 was ahead of its
PyTorch counterpart except one: **attention backward, 286-293 ms a step
against SDPA's 78-90** — 17% of R2's backward. Its cost per token grows
with `seq`; the GEMMs' does not, so at 4x the sequence length it grew
4x relative to everything else. Two structural causes in the kernel:

1. **Parallelism was the batch size.** The work was split by sequence.
   32 sequences on 6 cores is 89% utilisation; 8 sequences is two rounds
   with four cores idle in the second, 67% at best; batch 1 was serial.
   Units are now (sequence, query head) in the forward — 96 at this
   shape — and (sequence, kv head) in the backward, 32, where a unit runs
   its GQA group so dK/dV are complete inside it. A head's columns are
   strided through the tensor, so `head_pieces` collects each unit's row
   pieces as reborrowed slices: no copy, no unsafe.
2. **The accumulation was memory-bound.** For every (query, key) pair the
   backward read-modify-wrote three `hd`-float rows: 384 loads and 384
   stores per 384 FLOPs, on rows that live in L2 at seq 256. Now sixteen
   queries are finished per block; dV_j and dK_j are reduced over the
   block's queries in 64-lane register accumulators and touched in memory
   once per block, dQ_i is reduced over all its keys in registers and
   written once.

And three things found on the way, each measured with the kernel in
isolation (`lmo15_attention`, 8 x 256, 12/4 heads, one thread):

- A runtime tile-row count (`br <= 4`) made LLVM keep the score tile on
  the stack, three memory operations per FMA; the ragged tail now runs a
  `BR = 1` instance of the same constant-row tile.
- Sixteen-lane accumulators are two dependent FMA chains; Zen 2 needs
  eight in flight, so the accumulator is now the widest of 64/32/16/8
  lanes that divides `hd` (a const generic, the same code at every width).
- **The AVX2+FMA kernels contained no FMA instructions.** Rust never
  contracts `a * b + c` — that would change rounding — whatever
  `#[target_feature(enable = "fma")]` says; the emitted assembly had zero
  `vfmadd`. `f32::mul_add` under a `const F: bool` (true on the FMA
  build; plain `a*b + c` on the baseline, so no machine falls into libm's
  software `fma`). Worth checking every hand-written kernel for.

| ms per step, in situ, dim 768 | before | **after** | SDPA |
|---|---:|---:|---:|
| attention forward | 66-75 | **34** | 36 |
| attention backward | 286-293 | **89** | 78 |
| backward, everything but GEMMs | 374-378 | **183** | 377 |
| forward, everything but GEMMs | 139-145 | **100** | 118 |

Isolated, one thread, backward: 123 -> 87 (blocking) -> 78 ms (FMA).
Training pairs, real corpus, same window, 2026-09-20: medium model
**48.29 vs 55.45 s and 43.43 vs 55.20 s (1.15x, 1.27x)** from 1.05x;
small model at 100 steps **38.99 vs 57.9 s and 38.89 vs 58.5 s (1.49x,
1.50x)**. Held-out loss identical to four decimals on every pair (6.1412
and 5.6134), and identical to the runs before the change.

### How the lead scales with model size

The headline is a small-model number; this is the same harness one
size up. Same harness, `R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4
R2_SEQ=256 R2_BATCH=8 R2_STEPS=20` (39.8M parameters, 2,048 tokens/step,
every GEMM 3-9x larger than the 7M model's), one pair, clock 2375 MHz
before and after (2026-09-20):

| | R2 | PyTorch | |
|---|---:|---:|---|
| 20 steps | **58.24 s** (2,912 ms/step) | 61.16 s (3,058) | **1.05x** — parity within noise |
| 20 steps, attention re-blocked | **48.29 s / 43.43 s** | 55.45 / 55.20 s | **1.15x / 1.27x** |
| 20 steps, + `sgemm` column groups | **41.33 s / 34.22 s** | 57.8 / 55.9 s | **1.40x / 1.63x** |
| held-out loss | 6.1412 | 6.1412 | identical, before and after |

The 1.05x was NOT the GEMMs going to parity, which is what the first
reading of it said. The census (previous section) put the GEMMs at
~80% of R2's step here and R2 ahead on them; the missing 25% was one
kernel, attention backward, whose cost per token grows with the
sequence length. With it fixed the medium model is 1.15-1.27x ahead:
GEMMs ahead 1.1-1.2x in situ, everything around them 2.3x ahead
(270 vs 610 ms a step). The remaining lever at scale is the per-core
`sgemm`: both sides sit at 50-70% of the 124 GFLOP/s per-core FMA peak,
the TN (`grad_B`) shapes are the ones where R2 is behind (next section),
and at billions of parameters the GEMM share only grows.

### Where the medium model's step goes (dim 768, in situ, 2026-09-20)

GEMMs, PyTorch profiled inside its steps (`mm_profile.py`) against R2's
in-situ hook (`--example gemm_insitu`), p50 us:

| shape (calls) | R2 | PyTorch | |
|---|---:|---:|---|
| 2048x768x2304 (12) w1/w3 fwd, w2 grad_A | 26,157 | 34,321 | R2 1.3x ahead |
| 2048x2304x768 (12) w2 fwd, w1/w3 grad_A | 25,457 | 32,095 | R2 1.3x ahead |
| 2048x768x768 (16) q/o fwd + grad_A | 8,522 | 11,717 | R2 1.4x ahead |
| 2048x768x8000 (1) head fwd | 94,742 | 129,981 | R2 1.4x ahead |
| 2048x8000x768 (1) head grad_A | 89,624 | 116,052 | R2 1.3x ahead |
| 2048x256x768 (8) k/v grad_A | 2,800 | 4,868 | R2 1.7x ahead |
| **768x2048x2304 (8) w1/w3 grad_B (TN)** | 38,323 | 33,340 | **R2 1.15x behind** |
| **768x2048x768 (8) q/o grad_B (TN)** | 11,146 | 9,702 | **1.15x behind** |
| **768x2048x8000 (1) head grad_B (TN)** | 129,712 | 122,123 | **1.06x behind** |
| **768x2048x256 (8) k/v grad_B (TN)** | 4,351 | 3,749 | **1.16x behind** |
| all GEMMs | **1,669 ms** | 2,034 ms | R2 1.22x ahead |

Every NN and NT shape ahead, every TN (`grad_B = Aᵀ·g`) shape behind:
the first NAMED cause at this size. Closed the same day — next section.

### `grad_B` kernel to kernel, and the partition it needed (2026-09-20)

PyTorch's `MmBackward0` computes `grad_A = grad.mm(B.t())` and `grad_B
= A.t().mm(grad)`: two more `aten::mm` calls on stride-only views, which
MKL takes as its NT and TN cases — operation for operation what R2's tape
does through `sgemm(.., Trans::Yes, ..)`. So the like-for-like test is
the three calls at the same shape, both sides in one window, and
`benchmarks/llm/gemm_cases.py` (+ `--example gemm_cases`) runs exactly
that over 23 shapes: the small and medium models, wider layers, 512 and
8,192 tokens. Ratio > 1 is R2 ahead:

| case | before: min / median / max | after: min / median / max |
|---|---|---|
| NN forward | 0.60 / 1.19 / 1.92 | **0.77 / 1.37 / 2.18** |
| NT `grad_A` | 0.65 / 1.16 / 1.69 | **0.91 / 1.26 / 1.72** |
| TN `grad_B` | **0.50** / 0.87 / 2.11 | **0.65 / 1.32 / 2.32** |

Before: every TN shape with M <= 768 ran at 0.50-0.84x of MKL while the
same kernel at M >= 2048 ran 1.1-1.7x ahead — and the plain NN case at
768x2048x768 was just as slow, so it was the SHAPE (short M, long K),
not the transpose. Load balance was the first theory and a 2-D
partition alone measured nothing. Probes inside the kernel found it:
the row-block partition packs one shared `KC x NC` panel of B per depth
slab that every core then streams from L3 sixteen times per block; with
8 row-blocks per fork the panel's cold fetch is never amortised (the
micro-kernel ran 45% slower per FLOP than on the long-M shape) and there
were three times as many forks.

`gemm_colgroups`: A packed once for the whole depth (in parallel); each
task owns a group of `NR`-wide column strips of C — plus a group of
row-blocks when `n` is short — over every slab, packs its OWN strips of
B slab by slab (kept in that core's L2 for every row block), accumulates
into a private tile, folded into C once. Three forks per call instead
of three per slab. Bit-identical to the serial kernel. Taken whenever A
is not the large operand (<= 32 MB), the depth is >= 3 slabs or `m` is
short, and `n` is wide or `m` is short; the row-block form keeps the
rest (`grad_A` at 8,192 tokens, A = g = 75 MB, and the shallow K = 256
forward of the small model). Two more things on the way: a fresh
multi-MB scratch `Vec` per call was ~1,000 page faults per call (now
per-thread scratch, reused), and the packers were element loops with
two bounds tests each (now fixed-size array moves on the contiguous
side).

Best of two, same window, TN us (old -> new, MKL): 2048x768x768 13.3 ->
**10.2** (7.7-10.6); 768x2304 41.9 -> **25.5** (31.6); 768x8000 145 ->
**84** (109); 1024x4096 86 -> **60** (79); 8192x768x768 52 -> **38** (42).
Below 1.0 after: only the dim-256 shapes with M or N = 256 (1-4 ms calls
where both partitions sit at ~0.7x MKL), 2% of the small model's step.

Training pairs, real corpus, same window: medium **41.33 vs 57.8 s and
34.22 vs 55.9 s (1.40x, 1.63x)**; small at 100 steps **36.33 vs 60.4 and
35.86 vs 59.4 (1.66x, 1.66x)**. Held-out loss identical to four decimals
on every pair.

Where the medium model's step goes NOW (same census, after both fixes;
R2 1,678 ms/step in this window, PyTorch 2,745-3,000 in its stable
windows — its run in this one throttled to 3,400 and is not quoted):

| ms/step, dim 768 | R2 | PyTorch |
|---|---:|---:|
| forward GEMMs | 438 | 648-706 |
| forward, everything else | 97 (attention 30) | 114-128 (SDPA 34-40) |
| backward GEMMs | 876 | 1,272-1,394 |
| backward, everything else | 191 (attention 84) | 377-407 (SDPA 78-91) |
| optimizer | 61 | 162-189 |

Nothing in R2's step is behind its PyTorch counterpart any more; the
GEMMs are 1.5x ahead in situ and everything around them 2-3x.

The FMA audit that the attention finding called for: every
`#[target_feature(fma)]` kernel checked in the emitted assembly. The
f32 kernels (exp, silu, softmax_ce, the sgemm micro-kernel) use explicit
`_mm256_fmadd_ps` and were fine; Adam separates mul and add on purpose
(bit-identity with its scalar reference). **`level3::dgemm` and
`dot4` — f64, R's `%*%` — had the attention bug: plain `c += a*b` under
`target_feature`, zero `vfmadd`.** Same fix (`mul_add` under a
`const F: bool`); 500x500 dgemm 16-20 -> 19-27 GFLOP/s across four
interleaved rounds. It has had none of the sgemm structure and remains
the open f64 item.

Backward census in situ (`R2_TAPE_STATS=1 --example phase_split`):
matmul 82%, attention 12% (the kernel that turned out to be the medium
model's whole deficit — see "The second cut" above; now 5%), rmsnorm 4%
before it was made row-parallel (119 -> 33 ms; the forward was serial
too). Run length, two 20-step pairs: 46.97 -> 46.89 s
and 48.09 -> 46.19 s — under the bar, kept as strictly less work.

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
| `sgemm` — forward, `grad_A`, `grad_B` | **AHEAD of MKL kernel to kernel** on every case from dim 768 up: median NN 1.37x, NT 1.26x, TN 1.32x over 23 shapes (`gemm_cases.py`, 2026-09-20); dim-256 shapes with M or N = 256 ~0.7x | **39-80%** by size |
| Attention forward | **parity with SDPA in situ** (30-34 vs 34-40 ms at dim 768 / seq 256); ahead at >= 2,048 tokens isolated | 4-5% |
| Attention backward | **parity with SDPA in situ** (84-89 vs 78-101 ms) — was 3.3x behind | 5-8% |
| `softmax_ce` | fused; not separately compared | 2-7% |
| Embedding | 1.9-3.8x fwd+bwd | 0.07% |

Nothing R2 ships is clearly behind its reference any more (2026-09-20);
the two that were — MKL's `sgemm` on short-M shapes and SDPA's backward —
were partitioning and memory traffic, not the hand-written assembly R2
does not ship.

### Forward vs backward vs optimizer

`cargo run --release -p r2-train --example phase_split` and
`python benchmarks/llm/phase_split.py` — the same step cut into its three
phases on both sides, same model, same 2,048 tokens/step, two interleaved
pairs in one window (2026-09-19, clock 2375 MHz). ms/step, median of 20.

| phase | R2 | PyTorch 2.13 + MKL | ratio |
|---|---:|---:|---:|
| forward (to loss) | 154 / 156 | 233 / 228 | **R2 1.48x faster** |
| backward (all gradients) | 295 / 298 | 415 / 420 | **R2 1.41x faster** |
| optimizer (Adam) | 44 / 48 | 29 / 30 | PyTorch 1.5x faster |
| whole step | 489 / 519 | 662 / 662 | R2 1.28-1.35x faster |

Each side's phases sum to its whole step (R2 493 vs 489; torch 677 vs
662), so the split is accounting for the step and not for something
beside it. Backward is ~1.9x forward on both sides, as it should be:
two GEMMs per weight against one.

The optimizer is the one phase PyTorch wins. R2's number is not Adam's
arithmetic alone: the phase includes taking the weights back off the
tape and DROPPING the tape (freeing ~71.9M elements of value and gradient
buffers), while PyTorch's graph frees during `backward()`. Adam itself is
a single-threaded AVX2 kernel over 7.24M parameters; PyTorch's
`_foreach` Adam runs multi-threaded. At 9% of a step it is the smallest
phase, but it is a real, unclaimed lever.

The whole-step ratio in this short window (1.28-1.35x) is higher than
the 500-step figure (1.10x). The 500-step number is the one to quote: it
is sustained, on real data, from identical weights, and both sides are
throttled alike. Short windows favour whichever side runs while the
part is cooler. The PHASE ratios are what this table adds.

### `sgemm` rates on the shapes a step runs

**Superseded 2026-09-20** by the kernel-to-kernel sweep against MKL in
section 1 ("`grad_B` kernel to kernel"): that table is R2 and MKL at the
same shape in the same window; this one is R2 alone with hot operands,
kept for the history of the micro-kernel. After the column-group
partition the TN column below reads 260-390 GFLOP/s on the dim-768
shapes.

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
(2026-09-20: the in-situ censuses — `R2_TAPE_STATS=1 --example
phase_split` against `op_profile.py` — supersede this op-level one for
attribution, because they time ops inside a real step at any model size;
the medium-model table in section 1 is the current map. This section is
the small model on 2026-09-09.)

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

## 4b. The GPU step — R2 vs PyTorch-DirectML on the same integrated GPU (2026-09-22)

R2's training step runs on the device end to end (`R2_GPU=1`,
`crates/r2-train/src/gpu_llm.rs`): weights, activations, gradients and
Adam state resident, tokens in, a loss out, one kernel launch per op and
no allocator in the loop — the same model as the tape, written as a
static graph. Every kernel is deterministic (fixed-order sums, no
atomics; split-K and the rmsnorm dW reduction go through partial slabs
summed in order), matches the tape's loss and every gradient block to
f32 rounding (`gpu_loss_and_gradients_match_the_tape_block_by_block`,
max relative error 3-7e-7 per block), and trains TinyStories to the
SAME losses as the CPU trainer — 9.1095 -> 6.3693 at every checkpoint,
held-out 6.2690 on both.

The reference is PyTorch on the same GPU through **torch-directml
0.2.5** (torch 2.4.1 — the version it pins), the PyTorch route to an
AMD/Intel integrated GPU on Windows: `TS_DEVICE=dml python
benchmarks/llm/tinystories_train.py`, same manifest, same tokens, same
initial weights, same validity gate. **It is run at its best, not its
default:** DirectML's backward through `repeat_interleave` (the GQA head
expansion) into SDPA took **3.4 s per layer** here (14 s per step —
`benchmarks/llm/dml_phase_split.py` and the op probes found it: SDPA
alone 10 ms, `repeat_interleave` alone 1 ms, the two together 3,364
ms), and the same expansion written as a `cat` of head slices takes 18
ms, so the script uses that under DML. What remains in its step is its
own: Adam's `lerp` has no DML kernel and falls back to the CPU every
step.

```
GPU      AMD Radeon Vega 6 (Ryzen 5 4500U), Vulkan via wgpu 30; no VRAM,
         shares the CPU's ~20 GB/s DDR4; ~1.1 TFLOP/s f32 peak; 6 CUs;
         32 KB workgroup memory; no subgroups; offers shader-f16
torch    2.4.1+cpu + torch-directml 0.2.5.dev240914, SDPA attention
```

**Small model, 7.24M, 30 steps x 32 x 64, three interleaved pairs, clock
2375 MHz before and after:**

| pair | R2 GPU step | PyTorch-DirectML step | | held-out |
|---|---:|---:|---|---|
| 1 | **370.9 ms** | 506.2 ms | **R2 1.37x** | 6.2690 vs 6.2690 |
| 2 | **363.9 ms** | 487.7 ms | **R2 1.34x** | 6.2690 vs 6.2690 |
| 3 | **362.7 ms** | 488.5 ms | **R2 1.35x** | 6.2690 vs 6.2690 |
| 4 | **337.1 ms** | 513.9 ms | **R2 1.52x** | 6.2690 vs 6.2690 |
| 5 | **336.6 ms** | 518.5 ms | **R2 1.54x** | 6.2690 vs 6.2690 |

Pairs 4-5 are after the tiled attention backward below; 1-3 are kept
because the DirectML side is the same code in both and drifts by 6% on
its own, which is the honest width of this comparison.

**Medium model, 39.8M (dim 768, 4 layers, ffn 2304, 12/4 heads, 8 x
256), 20 steps, two pairs:**

| pair | R2 GPU step | PyTorch-DirectML step | | held-out |
|---|---:|---:|---|---|
| 1 | **1,665 ms** | 3,277 ms | **R2 1.97x** | 6.1603 vs 6.1603 |
| 2 | **1,909 ms** | 2,886 ms | **R2 1.51x** | 6.1603 vs 6.1603 |

**R2's GPU step is 1.34-1.54x faster than PyTorch-DirectML at 7M and
1.5-2.0x at 40M, on the same integrated GPU, learning identically.**

For scale, on this machine the CPU trainer is about level at 7M: 395
ms/step against the GPU's 337 (the R2 CPU step is itself 1.66x
PyTorch+MKL) — a 1.1 TFLOP/s
integrated GPU that shares the CPU's memory bus is a correctness and
portability platform for this code, not a speed platform. The kernels
are the ones that will run on a discrete GPU; their rates here:

| kernel | rate here | note |
|---|---:|---|
| `sgemm` 128x128 tile, 8x8 per thread | 330-425 GFLOP/s | split-K below 24 output tiles (deterministic partials); `gemm_bench` |
| `sgemm` short-output shapes (`grad_B` of a 256-wide projection) | 150-210 | was 53-110 before split-K |
| attention forward (one thread per row) | 20-30 | 2% of the small step |
| attention backward (register-tiled dK+dV, one thread per row dQ) | — | **2x the untiled pair** (6.55 vs 13.35 ms at 32x64, 39.4 vs 77.4 at 8x256); 14% -> 8% of the small step |
| rmsnorm fwd / bwd | ~6.5 GB/s | 8 rows per workgroup, dW reduced in row blocks; was 50 ms a step, now 19 |
| Adam | one launch per block | a few ULP from the CPU (WGSL may contract a*b+c), bit-identical to itself |

Where the GPU step goes at 7M (`R2_GPU_STATS=<step>` drains the queue
after every launch and prints the table): GEMMs ~58% (the 8000-wide head
alone 20%), attention backward 8%, attention forward 2%, rmsnorm 5%, the
rest under 3% each. The step is 157 launches.

### The tiled attention backward (2026-09-22)

`attn_tiled::dkv_wgsl` replaces the two one-thread-per-row kernels that
computed dV and dK separately, each re-reading every staged row once per
FMA and each recomputing `P` from scratch. One workgroup owns 32 keys and
walks the query blocks once: `Sᵀ = K·Qᵀ` and `dPᵀ = V·dOᵀ` are
register-blocked GEMMs over `hd` in 16-deep slabs (4 x 4 per thread, two
loads per four FMAs), and the `32 x hd` dK and dV tiles stay in registers
— 4 keys x `hd/8` columns per thread — for the whole walk, including
across the query heads that share a kv head. Every element is still summed
by one thread in a fixed order, so it is bit-reproducible, and it agrees
with the reference kernels to f32 rounding.

| | tiled | one thread per row | |
|---|---:|---:|---|
| backward, 32 x 64 tokens, 4/2 heads, hd 64 | **6.55 ms** | 13.35 ms | **2.04x** (CPU 7.21) |
| backward, 8 x 256, 12/4 heads, hd 64 | **39.4 ms** | 77.4 ms | **1.96x** (CPU 29.6) |

On the whole step, interleaved pairs (`R2_GPU_ATTN_TILED_BWD` switches
the two inside one process, so both are measured in one window):

| model | tiled | untiled | |
|---|---:|---:|---|
| 7M, four pairs | 334.6 / 339.7 / 382.1 / 357.9 ms | 359.9 / 375.3 / 406.2 / 379.1 | **1.06-1.10x** |
| 40M, two pairs | 1,637.8 / 1,629.0 ms | 1,745.8 / 1,779.9 | **1.07-1.09x** |

Held-out loss unchanged to four decimals on both. dQ is untouched — its
rows belong to another workgroup, so tiling it needs atomics or a second
reduction pass — and it is now the larger half of what attention costs.

### Mixed precision (2026-09-22)

`Precision::Mixed` (`R2_GPU_PREC=mixed`) is the standard recipe: f32
master weights and Adam state, an f16 copy of every GEMM weight cast once
per step, f16 activations and activation gradients, f32 residual stream,
logits, row statistics and weight gradients, f32 accumulation in every
kernel, and dynamic loss scaling. `Tensor` carries its storage type and
each kernel is compiled per type signature.

**It changes nothing on this machine** — 382.7 / 386.3 ms against 375.7 /
392.4 f32, within the noise, held-out 6.2691 vs 6.2690 — and that is the
expected answer: Vega 6 has no f16 arithmetic units, so halving the bytes
buys nothing when the step is bound by launch latency on six CUs, and the
per-step weight cast is new work. It is built and verified for hardware
that does have them.

One thing it found, which would bite silently elsewhere: **this adapter
SATURATES an overflowing f32 -> f16 store to ±65,504 instead of producing
an infinity.** The textbook overflow test for loss scaling looks for
inf/nan and would never fire here; training would drift on quietly
corrupted gradients. `nonfinite_flag` tests magnitude as well, and the
kernel test asserts the saturating case.

Activation checkpointing (`R2_GPU_CKPT=1`) keeps one layer's work set
instead of every layer's — 14·t·d + t·nh floats per layer become t·d —
for +15% step time (438 vs 382 ms), the same bits.

**Measured and closed on this device (do not retry here):** a 64 x 64
GEMM tile (nothing over split-K), a register-prefetching slab loop
(25-30% slower everywhere: the staged registers cost the occupancy the
big tile lives on), the tiled FlashAttention-2 forward (`attn_tiled`,
2.24 vs 2.12 ms and 11.7 vs 9.4 ms against the one-thread-per-row
kernel; kept under test as the shape the backward needs — whose tiling
DID pay, above), and splitting attention's reduction axis into partial
slabs (splits 1/2/4/8/16: 14.8, 16.2, 19.6, 26.1, 40.7 ms —
monotonically worse).

## 5. Open

Ranked by the census above, not by how interesting they are.

0. **Re-run `--example step_census` — the shares below predate the three
   removals in section 1.** Fixing the top item promotes whatever was
   hiding under it, and a stale census is how the split-K attempt below
   came to be aimed at a problem that had already been fixed.
1. **`sgemm` is 57% of a step**, and after it nothing is above 7%.

2. **Where `sgemm` stands against MKL — measured per core and per
   thread count, same shapes, same window (2026-09-19).** This corrects
   two earlier claims: the machine peak is not 460 GFLOP/s, and the
   remaining gap is not per-core.

   `--example fma_power` (throughput-bound FMA, twelve independent
   accumulators, no memory): **1 core 123.7 GFLOP/s, 6 cores 622.1,
   5.03x.** That is the machine: 3.9 GHz boost on one core, 3.2 GHz on
   all six, and power is NOT the ceiling.

   `--example gemm_scaling` and `benchmarks/llm/mkl_scaling.py` (torch
   `set_num_threads` 1 and 6):

   | | 1 core | 6 cores | scale |
   |---|---:|---:|---:|
   | FMA peak | 124 | 622 | 5.03x |
   | **R2 `sgemm`** | 58-87 | 152-243 | 2.4-3.4x |
   | **MKL `sgemm`** | 60-83 | 156-285 | 2.3-4.0x |

   **Per core, R2 and MKL are the same speed** — both at 50-70% of the
   core's FMA peak. The 6x16 micro-kernel issues 8 loads per 12 FMAs (six
   A broadcasts, two B vectors) and the packing and C-update passes sit
   on top; that is the microarchitecture's price for K=256 slabs and MKL
   pays it identically. There is no per-core lever.

   **Across cores, both lose ~40% of the remaining 5x** to the shared
   hierarchy — an L3 that is two 4 MB CCX halves (the packed B panel is
   copied into both) and laptop DRAM — which is what `bandwidth_sweep`
   measured. PyTorch faces the same wall: MKL scales 2.3-4.0x here.

   **R2's actual deficit was confined to short-M (TN) shapes** — and
   CLOSED 2026-09-20 by the column-group partition (section 1, "`grad_B`
   kernel to kernel"): it WAS partitioning, specifically every core
   cold-fetching one shared packed B panel per slab. The paragraphs
   below are the history of finding that.

   The `bandwidth_sweep` table stands as the measurement of the shared
   hierarchy; the conclusions drawn from it that the machine peak was
   460 and that the residual was per-core are withdrawn.

   **2026-09-20: the persistent thread team and the pack-free kernel
   are REMOVED from the tree** (both rows in the closed table; the code
   is in git history). Neither changed the step total; the small-shape
   question was closed by measuring PyTorch in situ instead:

   **CLOSED 2026-09-20 by the measurement that was missing:** PyTorch's
   GEMMs profiled INSIDE its own training steps
   (`benchmarks/llm/mm_profile.py`, `torch.profiler`, shapes recorded),
   against R2's in-situ latencies from the in-situ probe (now `--example gemm_insitu`):

   | shape (calls/step) | PyTorch p50 / mean us | R2 p50 / mean us | |
   |---|---:|---:|---|
   | k/v `grad_B` 256x2048x128 (8) | 755 / 892 | 760-787 / 853-940 | parity |
   | q/o `grad_B` 256x2048x256 (8) | 1,414 / 1,667 | 1,327-1,446 / 1,463-1,900 | parity |
   | k/v forward 2048x256x128 (8) | 791 / 880 | 557-577 / 584-610 | **R2 1.4x ahead** |
   | k/v `grad_A` 2048x128x256 (8) | 625 / 735 | 573-581 / 590-595 | R2 1.2x ahead |
   | q/o fwd + `grad_A` 2048x256x256 (16) | 1,368 / 1,470 | 1,021-1,042 / 1,104-1,144 | R2 1.3x ahead |
   | all GEMMs per step | 364.8 ms | ~250 ms (census) | R2 ~1.45x ahead |

   Every "R2 1.3-2.3x behind on the small shapes" figure above compared
   R2 inside a real step (cold operands) with MKL in an isolated loop
   (hot operands). MKL inside a real step pays the same: its output-head
   `grad_B` runs at 132 GFLOP/s in situ against 156-221 isolated. On
   these shapes R2 is at parity or ahead of what PyTorch actually pays,
   which is why the team, the pack-free kernel and the whole-K slab all
   measured flat or worse — they were solving a gap that was not there.
   Rule, added to the measurement law: **the reference must be measured
   the same way as the subject.** MKL's isolated rates remain the
   per-core ceiling comparison; they are not a per-call comparison.

3. **Attention — CLOSED 2026-09-20** (section 1, "The second cut"):
   parity with SDPA in situ, forward and backward, after (sequence, head)
   work units, register-blocked accumulations and real FMA. At seq 256
   it had been 17% of the backward and 3.3x behind.
4. **`level3::dgemm` is 19-27 GFLOP/s** after its FMA fix (2026-09-20; it
   was 16-20 with none) — still a fraction of the f64 ceiling, because it
   has none of the sgemm structure. `sgemm` is a per-type macro, so f64
   is close to a one-line instantiation; a column-major `C = A·B` is the
   row-major `Cᵀ = Bᵀ·Aᵀ`, i.e. the same kernel with operands swapped.
   That would lift `%*%` and every blocked LAPACK algorithm built on it.
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
| **Fusing GEMMs that share a left operand** — `h·w1` and `h·w3` as one 2048x256x1536 call; `h·wq`, `h·wk`, `h·wv` as one 2048x256x512 (the llama.cpp layout; PyTorch eager runs the separate calls, as R2 does) | `--example fused_shapes`, separate vs fused interleaved, 9 rounds, median ms. FFN: NN 7.14 -> 7.99 (**slower**, 1.12x), NT 7.55 -> 7.07, TN 10.97 -> 9.43 — net **4.6%** on a block that is ~17% of a step. QKV: NN 3.08 -> 2.58, NT 3.22 -> 2.24, TN 3.80 -> 3.05 — net ~22% on a block that is ~5% of a step. Together ~2% of a step BEFORE the copies a fused layout needs (slicing gate/up, or a strided SwiGLU), under this machine's ~10% bar. The FFN forward regression is not panel imbalance: `R2_GEMM_NC=768` (two equal panels) gives the same 1.11x. Wider n does not help NN on this kernel — the shape table already said so (output head NN at n=8000 is below w1/w3 at n=768). Same lesson as the ceiling work: what remains is per-core, not structural |
| **Swapping the micro-kernel loop nest to B-strip-outer / A-strip-inner** (the textbook Goto order, so the 16 KB B strip stays in L1 and the shared 1 MB panel is read from L3 once per block instead of once per A strip) | `gemm_rate` before/after: output head NN **195 -> 152**, NT 206 -> 186; w1/w3 NN 231 -> 210; the rest flat. On Zen 2 the private L2 is 512 KB, so the original order's B re-stream is L2-served and cheap, while the swapped order's C-tile writes (96 rows at stride n, 32 KB apart on the output head) are what thrash. Loop order is tuned to this cache geometry already; reverted |
| **One whole-K slab for small outputs** (k/v and q/o gradients: 256x128 / 256x256 with K=2048), on the theory that eight KC=256 slabs are eight serial packs and eight fork-joins of pure overhead | `gemm_rate`, `R2_GEMM_FULLK` 0/1 interleaved: k/v TN 145 -> 117 / 142 -> 86, q/o TN 155 -> 106 / 166 -> 90 — **worse**. KC is not overhead: it is what keeps the 16 KB B strip and 6 KB A strip in L1 per tile; at kc=2048 they are 128 KB and 48 KB and stream from L2 on every tile. Also learned: sub-millisecond kernels scatter +/-25% between identical runs here (q/o NN 208 then 161, no change), so the small-shape residue cannot be resolved at op level below that. The honest item is a persistent thread team (one fork per call, cooperative packing behind barriers) — the one thing MKL has on these shapes (3.2-4.0x vs 2.4x) and nowhere else — worth ~4% of a step |
| **Pooling gradient buffers and zeroing them in one parallel pass** before backward, to replace calloc's page faults with a memset | Two 300-step pairs against the values-only pool: 143.09 -> 152.30 s and 143.42 -> 142.52 — worse, then flat. A 6-thread memset of ~172 MB costs what the faults cost: this laptop's DRAM write bandwidth. The fix that worked removes the zeroing entirely (first writer assigns) |
| **A persistent GEMM thread team with a SPINNING barrier** (`gemm_team`; code removed 2026-09-20, see git history) | Bit-identical to the serial kernel (test). Median call 25-30% faster than fork-join on the k/v and q/o gradient shapes — but the TAIL: p99 21 ms, max 83 ms, against fork-join's 2-10 ms (the in-situ probe (now `--example gemm_insitu`), per-call latencies inside real training steps). Five workers spinning occupy the cores the descheduled sixth needs, so one OS deschedule costs a scheduler quantum or more. 300 steps: 128.48 -> 132.06 s. **Diagnosed, not abandoned:** the same team with a YIELDING barrier keeps the median gain and has a tighter tail than fork-join (max 2-4 ms; 20-25% less time in these calls per step) and is now the default. At run length that is ~2% of a step and the pairs straddle zero (124.40 -> 127.32, 126.48 -> 124.27): kept for the lower variance, not claimed as a speedup |
| **A pack-free direct kernel for the `grad_B` shapes** (`tn_direct`; code removed 2026-09-20, see git history): the same 6x16 register tile reading `A[t][i..i+6]` and `g[t][j..j+16]` straight from the operands, no packing, no K slabs, one fork | a standalone pack-free kernel (removed with it), isolated, 15 interleaved rounds: **0.70x / 0.74x / 0.69x** of `sgemm`'s time on the k/v, q/o and w1/w3 gradient shapes, results to f32 rounding. Then 300 training steps: **117.80 -> 144.33 s (+22%), 130.33 -> 141.78 s (+9%)**. The isolated benchmark multiplied the same operands 75 times, so they were L2-hot; in a step they are the activation and the upstream gradient, 2-6 MB each and cold, and the direct kernel re-streams every operand strip once per tile — eight to sixteen times from DRAM. Packing is what makes a cold operand cross memory once. The most instructive negative of the small-shape work: an op-level 0.7x that was real and irrelevant |
| **Rewriting the silu backward in ATen's expression order** (`dy*s*(1+x*(1-s))`, plain store) on the theory that PyTorch's is "more vectorised" | `--example silu_forms`, 2048x768, 1 thread, 21 rounds: R2's two-FMA form 1.430 ms, ATen order 1.435 ms — identical; the two agree to 1.1e-7 (< f32 eps). The REAL `aten::silu_backward` on the same array: 2.096 ms on 1 thread (R2 **1.47x faster per core**), 1.262 ms on 6. Both are the same 8-wide AVX2 loop; PyTorch's Sleef `exp` is wider-range and slower than R2's clamped Cephes, which is all silu needs |
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

# the GPU step (section 4b): R2 on the device, then PyTorch-DirectML on
# the same device from the same manifest. torch-directml pins torch 2.4.1,
# so it lives in its own venv (here G:/r2-target/venv-dml).
cargo build --release -p r2-train --features gpu --example tinystories_train
R2_GPU=1 target/release/examples/tinystories_train
TS_DEVICE=dml G:/r2-target/venv-dml/Scripts/python.exe benchmarks/llm/tinystories_train.py
R2_GPU=1 R2_GPU_STATS=4 target/release/examples/tinystories_train    # the GPU step census
G:/r2-target/venv-dml/Scripts/python.exe benchmarks/llm/dml_phase_split.py   # DirectML's phases
cargo run --release -p r2-gpu --features gpu --example gemm_bench      # GPU sgemm vs CPU sgemm per shape
cargo run --release -p r2-gpu --features gpu --example attn_bench      # attention kernels vs the tape's

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

# IN SITU, any model size (the size knobs R2_DIM/LAYERS/FFN/HEADS/KV/SEQ/
# BATCH/STEPS apply to every harness here): both phases by op on both
# sides, and every GEMM call inside real steps by shape
R2_TAPE_STATS=1 R2_GEMM_STATS=1 cargo run --release -p r2-train --example phase_split
python benchmarks/llm/op_profile.py
R2_GEMM_STATS=1 cargo run --release -p r2-train --example gemm_insitu
python benchmarks/llm/mm_profile.py

# KERNEL TO KERNEL: forward / grad_A / grad_B at 23 shapes, R2 and MKL
# per shape in one window (the Python half runs both)
cargo build --release -p r2-linalg --example gemm_cases
python benchmarks/llm/gemm_cases.py
```

Every `--release` build on this machine needs
`CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`,
or fat LTO across ~70 test binaries runs for hours.

**This machine drifts about 10% over hours** — the same unchanged binary
measured 31.05 s and 34.10 s two hours apart. Interleave every A/B in one
window (stash, measure, pop, measure); nothing under ~10% is a result
without it. And only the step total counts: one change measured 18% better
at op level and 4% worse on the training step, in the same hour.
