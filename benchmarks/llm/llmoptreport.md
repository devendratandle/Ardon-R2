# LLM training optimization report

An operation-by-operation audit of R2's GPU training, against the book's maths, PyTorch,
and the hardware's own limits. PyTorch is a reference. The target is R2's peak on the card.

`REPORT.md` keeps the earlier harness results. Current GPU optimization figures are here.

## What optimization means here

Keep accuracy, reliability, validity and consistency, whether the maths stays or changes,
and deliver faster at minimum cost in every resource: time, device memory, host memory,
copies and power.

## The gate

A change stays only if it passes `benchmarks/llm/gate.py`:

| Test | Check |
|---|---|
| Accuracy | Held-out loss within 0.0002 of PyTorch. A new formula also has its error measured against a higher-precision reference |
| Validity | Kernel tests compare outputs and gradients with the tape. A new formula also gets a finite-difference check |
| Reliability | Every test binary passes, with no panic or kernel fault, at all three sizes |
| Consistency | Final weights identical across rounds and across ranks |

## Rules of evidence

- Only a real step total proves a gain, and under 10% stays unproven.
- A small gain may stay if its row improves in every round, the step gets no worse, and
  the code stays simple.
- Figures from one run are compared within that run. The card's speed changes between runs.

## cuBLAS

NVIDIA's CUDA licence covers cuBLAS and says "You may not reverse engineer, decompile or
disassemble" any portion of the SDK. So R2 does not disassemble it or extract its kernels.
It is studied from outside: timing by shape, layout and operand order, and what NVIDIA's
own profiling interface reports. Open sources are CUTLASS, MAGMA and the published papers.

## The audit sheet

Each operation is filled in this order, and nothing changes until steps 1 to 6 are done.

1. **Book maths:** definition, equivalent forms, stability.
2. **PyTorch's steps:** what it executes, down to kernel, grid, temporaries, allocations.
3. **R2's steps:** the same, line by line.
4. **Memory:** bytes read, written and stored; caches; shared memory.
5. **Gap:** time against PyTorch and the ceiling; whether compute, memory or code limits it.
6. **Options:** every realistic change, with expected saving and risk.
7. **Test:** one change at a time, through the gate.
8. **Verdict:** kept or closed, with figures, so a closed idea is not retried.

## Order of work

| Order | Operation | Share of R2's step at 125M |
|---:|---|---:|
| 1 | GEMM | About 80% |
| 2 | Attention | About 14% |
| 3 | Silu, `mul`, add | About 3% |
| 4 | Rmsnorm, loss, embedding | About 2% |
| 5 | Adam, RoPE | About 2% |
| Separately | Gradient exchange and slow-card waits | Two or more GPUs |

## Baseline: 125M, one T4, Kaggle run 26

GEMM FLOPs are exact. The others are estimates by formula, with `exp` as about 12 FLOPs.

| Operation | GFLOP | R2 ms | PyTorch ms | R2 GFLOP/s | PyTorch GFLOP/s |
|---|---:|---:|---:|---:|---:|
| GEMM forward | 978 | 298 | 322 | 3,280 | 3,040 |
| GEMM backward | 1,956 | 572 | 622 | 3,420 | 3,140 |
| Attention forward | ~41 | 31 | 22 | 1,320 | 1,860 |
| Attention backward | ~100 | 117 | 77 | 850 | 1,300 |
| Rmsnorm | ~0.9 | 16 | 41 | 56 | 22 |
| RoPE | ~0.45 | 5 | 43 | 85 | 10 |
| Silu and `mul` | ~5.3 | 32 | 35 | 170 | 150 |
| Add | ~0.08 | 3.5 | 3.8 | 21 | 20 |
| Softmax loss and embedding | ~0.9 | 3.1 | 3.8 | 290 | 240 |
| Adam | ~1.4 | 14 | 38 | 100 | 37 |

## Results by run

### Run 27 (9b9323e): four-warp attention products, row kernels stopping at the diagonal tile, query tile loaded once, overlapped exchange copies

The gate passes: all tests pass, and losses match at every size. Attention PTX committed in 85ceab8.

| One T4 | Run 25 | Run 27 |
|---|---|---|
| 7.2M | R2 1.17-1.24x | R2 1.16-1.24x |
| 39.8M | R2 1.07-1.17x | R2 1.21-1.32x |
| 125M | R2 1.02-1.07x | R2 1.03-1.06x |

| Two T4s, against DDP | Run 25 | Run 27 |
|---|---|---|
| 7.2M | R2 1.13-1.14x | R2 1.08-1.09x |
| 39.8M | R2 1.06x | R2 1.11-1.12x |
| 125M | PyTorch 1.01-1.02x | Level, R2 1.00-1.01x |

| Row, 125M | Run 26 | Run 27 |
|---|---|---|
| Attention backward, R2 ms per call | 9.71 | 9.39 |
| Attention forward, R2 ms per call | 2.61 | 2.62 |
| Gradient exchange, R2 ms | 164 | 116 |
| Gradient exchange, PyTorch ms | 131 | 130 |

**Verdict:** the overlapped exchange is kept: 29% less at 125M, now ahead of `all_reduce`
there, and still 1.17x behind at 39.8M. The attention changes gave little: backward down
3%, forward unchanged, despite fewer loads, so loads may not be what limits forward.
They stay because the gate passes and the code is no larger, but they are not proven gains.

### Open leads from run 27

- **Attention in the real step:** forward costs 4.3 ms per call in the census against 2.6 ms
  in its row, while GEMMs cost no more. The row's data is smooth and training data is not.
- **Time outside the kernels:** kernels and Adam add up to about 1,170 ms, against a real
  step near 1,260 ms over 30 steps. The first step may account for it.
- **One card slower than the other:** rank 1 waits about 300 ms per step for rank 0 at 125M,
  more than the copies themselves.
