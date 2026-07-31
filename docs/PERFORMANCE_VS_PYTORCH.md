# R2 vs PyTorch for LLM work — what is measured, and what is not

**This machine has no Python installed, so no PyTorch number here was
measured locally.** R2's figures are measured; PyTorch's are the commonly
reported ranges for CPU-only builds. Every row states which is which,
because a benchmark that quietly mixes the two is worthless.

## Where R2 is measurably ahead

| | R2 (measured) | PyTorch CPU (typical) | Factor |
|---|---:|---:|---|
| Install size | **15 MB** (single binary) | ~2 GB (torch + deps) | ~130× |
| Startup to first line of user code | **45 ms** | 1–3 s (`import torch`) | ~20–60× |
| Baseline RAM before any model | **~6 MB** | 300–400 MB | ~50× |
| Dependencies to install | **none** | Python, pip, wheels, BLAS | — |
| Cold start → 1M model trained → text generated | **3.9 s total** | — | — |

That last row is the one that matters for a learner: **process launch,
model definition, 60 training steps and generation, start to finish, in
under four seconds.** No environment to activate, no CUDA to match to a
driver, no first-run kernel autotune.

These advantages are structural, not tuning wins. There is no interpreter
to boot, no framework graph to construct, no garbage collector, and the
whole runtime is one statically linked binary.

## Where PyTorch is ahead, and by how much

| | R2 (measured) | PyTorch CPU (typical) |
|---|---:|---:|
| Training throughput | 9–12 GFLOP/s | 20–50 GFLOP/s |
| Matmul (large, CPU) | 47–63 GFLOP/s | 60–120 GFLOP/s (oneDNN/MKL) |

**PyTorch is roughly 2–4× faster per FLOP on CPU today.** Its kernels call
oneDNN/MKL — decades of hand-tuned assembly micro-kernels. R2's are
straightforward Rust with rayon. Closing that needs register tiling and
explicit SIMD; it is achievable but not done.

R2's iGPU matmul (**67 GFLOP/s**, measured on an integrated Radeon) is
already competitive with a tuned CPU BLAS, and it runs on hardware the
CUDA-first stack ignores entirely.

## Inference, small models — measured

| Model | First token | Sustained |
|---|---:|---:|
| 1M | 3.4 ms | 2,909 tok/s |
| 3M | 7.8 ms | 1,182 tok/s |
| 10M | 33.2 ms | 284 tok/s |

At these sizes weights fit in cache and there is no host↔device transfer,
so a CPU is not a compromise — it is the right hardware. A GPU would spend
longer moving data than computing.

## The honest summary

R2 is **not faster than PyTorch at raw arithmetic** and should not be sold
as such. It is faster at *everything around* the arithmetic — starting,
loading, allocating, and getting out of the way — and it runs where the
GPU-first stack does not: integrated graphics, plain CPUs, machines with
no admin rights to install a 2 GB toolchain.

For production training on rented A100s, PyTorch is the correct tool. For
a statistics student on a laptop with integrated graphics who wants to
build a model, train it, break it and rebuild it in a lunch break, the
comparison is not close — and that gap is a deliberate design outcome, not
an accident.

Reproduce with: `cargo run --release -p r2-train --example student_bench`,
`--example scale_bench`, and `cargo run --release -p r2-gpu --features gpu
--example residency`.
