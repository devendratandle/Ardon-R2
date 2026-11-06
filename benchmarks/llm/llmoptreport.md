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

### Run 28 (b9cf375): the card's limits and the leads from run 27

The T4 under load runs at its 70 W power cap, and that decides its clock.

| Card, 125M, 30 steps | R2 clock | R2 step | PyTorch clock | PyTorch step |
|---|---:|---:|---:|---:|
| Card 0, round 1 | 645 MHz, 78 C | 1,218 ms | 630 MHz, 80 C | 1,280 ms |
| Card 1, round 1 | 930 MHz, 52 C | 870 ms | 900 MHz, 63 C | 983 ms |
| Card 0, round 2 | 765 MHz, 67 C | 1,021 ms | 690 MHz, 75 C | 1,190 ms |
| Card 1, round 2 | 855 MHz, 63 C | 947 ms | 780 MHz, 72 C | 1,079 ms |

All readings sit at 65-69 W with throttle reason 0x4, the software power cap. Clock is
lower when the card is hotter, and PyTorch always ran second, on the card R2 had just heated,
so these ratios may favour R2. Run 30 alternates the order.

| Burst limit, card 0 | Measured |
|---|---:|
| Multiply-adds in registers | 6,630-6,970 GFLOP/s |
| Memory read, past the caches | 242 GB/s |
| Memory write | 225 GB/s |
| Memory copy, read plus write | 190 GB/s |
| Tile loads, row storage, lanes across rows | 87-89 GB/s |
| Tile loads, column storage | 254-258 GB/s |
| PCIe, down / up / both at once | 13.1 / 12.3 / 22.3 GB/s |

**Closed:**
- **Smooth data:** attention takes the same time on data with unalike neighbours.
- **First step:** it costs the same as the median step, so setup does not explain the time
  outside the kernels at 125M. The kernel rows run in bursts at a higher clock than a sustained
  step, which may explain it, and run 30 measures sustained rates.

**Open:**
- **Accuracy gate failed at 100 steps:** held-out loss 5.1245 against 5.1256 at 125M, over the
  0.0002 limit. R2's weights repeat and PyTorch's default run does not, so this may be PyTorch's
  own scatter, but one pair cannot show that. Run 30 runs each side twice.
- **Attention in the real step:** forward costs 4.3 ms per call in the census against 2.6 ms in
  its row. Run 29 tests peaked scores.
- **Attention's loads:** `attn_gemm` and `attn_fused` had lanes across rows, the 88 GB/s pattern.
  R2's own `sgemm_tiled` already walks along. Changed for run 29.

### Runs 29 and 30 (9797ac3, a8c2558): loads along the row, sustained power, fair order

Run 29 ran the tests alone: 45 pass with the new attention loads. Run 30 measured.

**The gate fails.** All 45 tests pass and no scoreboard check differs, but at 125M and 100 steps
R2 holds 5.1245 in both runs against PyTorch's 5.1258 and 5.1256. PyTorch's own spread is 0.0002
and the gap is 0.0011-0.0014, so PyTorch's scatter alone does not explain it. The other sizes agree:
6.2250 at 125M and 30 steps, 5.2064 at 39.8M and 100 steps, 4.3113 against 4.3114 at 7.2M and 300.
Whether a change of rounding alone moves the loss this far is not yet known.

**Fair order: R2, PyTorch, PyTorch, R2, on one T4.** The earlier T4 ratios favoured R2.

| Pair | R2 ms | PyTorch ms | Mean of each | Last two runs, equally hot |
|---|---:|---:|---:|---:|
| 7.2M, 300 steps | 27.2, 25.7 | 30.9, 30.2 | R2 1.16x | R2 1.18x |
| 39.8M, 100 steps | 146.1, 148.4 | 157.6, 159.9 | R2 1.08x | R2 1.08x |
| 125M, 30 steps, card 0 | 1,089, 1,147 | 1,197, 1,241 | R2 1.09x | R2 1.08x |
| 125M, 30 steps, card 1 | 845, 900 | 937, 954 | R2 1.08x | R2 1.06x |
| 125M, 100 steps, card 0 | 1,071, 1,188 | 1,235, 1,241 | R2 1.10x | R2 1.04x |

Only 7.2M clears the 10% rule. At 125M a run on the cooler card gains more than R2 leads by.
The same model takes 845 ms on card 1 at 51 C and 1,089 ms on card 0 at 72 C.

**Sustained for 30 s, all under the 70 W cap:**

| Work | Card 0 | Card 1 | Clock | GFLOP/s per W, card 0 |
|---|---:|---:|---:|---:|
| Multiply-adds in registers | 6,605 GFLOP/s | 6,852 GFLOP/s | 1,320-1,350 MHz | 99 |
| cuBLAS 4096x768x3072, constant data | 4,992 GFLOP/s | 5,401 GFLOP/s | 1,035-1,125 MHz | 76 |
| GEMMs in the 125M step, census | about 3,100 GFLOP/s | | 645-735 MHz | about 47 |
| Memory read | 243 GB/s | 243 GB/s | 1,365-1,500 MHz | |
| Memory copy, read plus write | 194 GB/s | 191 GB/s | 1,320-1,515 MHz | |

Against 2 FLOPs per lane per cycle on 2,560 lanes, registers reach 98%, sustained cuBLAS 94%
and the step's GEMMs about 86%. The GEMMs are near what the card's clock allows, and energy sets
that clock. The constant data flatters cuBLAS, so its gap to the step is not yet explained.

**Attention, within the run against PyTorch:**

| Row, 125M | Run 26 | Run 30 |
|---|---:|---:|
| Backward, R2 / PyTorch ms per call | 9.71 / 6.4 | 8.56 / 6.35 |
| Backward, PyTorch ahead by | 1.52x | 1.35x |
| Forward, R2 / PyTorch ms per call | 2.61 / 1.8 | 2.45 / 1.86 |
| Forward, PyTorch ahead by | 1.41x | 1.32x |

**Verdict:** the loads along the row stay, and backward is the only clear gain, at about 12%.

**Closed:**
- **Peaked scores:** ten times the amplitude leaves attention at 8.14 and 2.47 ms for R2 and
  6.32 and 1.83 ms for PyTorch.
- **Time outside the kernels at 125M:** the rows sum to 1,175 ms for R2 and 1,224 ms for PyTorch,
  inside the step times on that card. The step is its operations.
- **Tile loads:** the test's tile is 16 columns wide, so walking along a row gives 64-byte runs at
  101-111 GB/s and by column gives 256-byte runs at 255-259 GB/s. The length of the run matters,
  not row or column as such. Attention's tile at `hd` 64 already has 256-byte runs.

**Open:**
- **Accuracy at 100 steps:** run R2 with tiled products, the same maths with different rounding.
  If its loss moves about as far, the gap is rounding carried through training. If it stays at
  5.1245, R2 and PyTorch differ somewhere real.
- **Forward attention in the step:** 4.2 ms per call in the census against 2.45 ms in its row, from
  one kernel with the same arguments, while backward agrees at 8.8 and 8.56 ms. Real activations or
  operands just written remain as causes.
- **Energy:** cuBLAS sustained on random data, to show what the data costs in clock. Fusing work that
  shares an operand may save energy as well as time, and only a step total at the cap can show it.

### Run 31 (cee1d7d): the loss gap, operand data and clock, attention on real activations

All 45 tests pass and the PTX is current. `gate.py` still fails. Its weight failures come from this
run comparing tiled with cuBLAS on purpose: each path repeats its own weights exactly.

**Loss at 125M, 100 steps:**

| Held-out batches | R2 cuBLAS | R2 tiled | PyTorch, each run |
|---|---:|---:|---:|
| One, run 30 | 5.1245, 5.1245 | | 5.1258, 5.1256 |
| One, run 31 | 5.1245 | 5.1238 | 5.1265, 5.1259 |
| Three | 5.1365 | 5.1358 | 5.1371 |

| Step, one batch | R2 cuBLAS | R2 tiled | PyTorch |
|---:|---:|---:|---:|
| 10 to 60 | Identical at four decimals on all sides | | |
| 70 | 5.8207 | 5.8206 | 5.8210, 5.8208 |
| 80 | 5.2620 | 5.2621 | 5.2632, 5.2624 |
| 90 | 5.2083 | 5.2082 | 5.2084, 5.2086 |
| 100 | 5.1245 | 5.1238 | 5.1265, 5.1259 |

Other rounding moves R2 by 0.0007, and PyTorch differs from itself by up to 0.0009 across four runs,
so the 0.0002 limit cannot hold at this size and length: PyTorch fails it against itself. That does
not clear the gap. R2 stays below every PyTorch run, by 0.0006 on three batches and 0.0011-0.0020 on
one, which more samples must settle. R2 is the lower side, and the gap is under 0.04% of the loss.

**What operand data costs, cuBLAS 4096x768x3072 held for 30 s:**

| Operands | Card 0 | Card 0 clock | Card 1 | Card 1 clock |
|---|---:|---:|---:|---:|
| Constant 0.5 | 4,641-4,689 GFLOP/s | 960 MHz | 4,923-5,401 GFLOP/s | 1,020-1,125 MHz |
| Random | 2,961-3,081 GFLOP/s | 615-630 MHz | 3,125-3,497 GFLOP/s | 645-720 MHz |

Random operands cost about a third of the clock under the same power cap. The step's GEMMs at about
3,100 GFLOP/s and 645-735 MHz match this. **The real step's GEMMs already run at what cuBLAS sustains
on this card with data that is not constant.** cuBLAS holds about 94% of 2 FLOPs per lane per cycle
either way, so the limit is the energy per FLOP on real data.

**R2 tiled products, sustained under the cap:** 1,445-1,458 ms per step at 975-990 MHz, against
cuBLAS at 1,196 ms and 630 MHz in the same set. Tiled runs a third of the clock higher and still costs
about 1.2x the energy per step.

**Closed:**
- **Forward attention on just-written memory:** repeating it on the same operands takes the same
  time, 35.2 and 35.0 ms, then 36.3 and 36.5 ms.
- **Forward attention at 4.2 ms per call:** this was largely heat. This census reads 2.9-3.0 ms and
  its backward 7.0-7.3 ms. Compared with backward, forward still costs about 1.45x what its row shows,
  so the activations themselves remain the lead.

**Open:**
- **Loss:** several runs per side, each with a one-ulp change to the starting weights, then compare
  the two means against their spread.
- **Energy per FLOP:** operands of 0.5 have an empty mantissa. Test sustained cuBLAS on random data
  with 24, 16, 12 and 8 significant bits. Only if the clock rises does training on fewer bits become
  worth testing, and it would have to pass the gate.

### Run 32 (2251683): the loss gap settled, operand bits, `torch.compile`

All 45 tests pass and the PTX is current. `gate.py` still fails, on its 0.0002 limit and on the
weights differing across nudges, which this run does on purpose. Raw logs of runs 26 onward are in
`kaggle-logs/`.

**Loss at 125M, 100 steps, three held-out batches.** Each pair starts from the same weights, with one
moved by one ulp:

| Nudge | R2 | PyTorch | PyTorch minus R2 |
|---:|---:|---:|---:|
| 1 | 5.1362 | 5.1367 | +0.0005 |
| 2 | 5.1364 | 5.1385 | +0.0021 |
| 3 | 5.1364 | 5.1385 | +0.0021 |
| 4 | 5.1372 | 5.1381 | +0.0009 |
| 5 | 5.1379 | 5.1369 | -0.0010 |
| 6 | 5.1370 | 5.1363 | -0.0007 |
| Mean | 5.13685 | 5.13750 | +0.00065 |
| Standard deviation | 0.00064 | 0.00098 | 0.00133 |

**Verdict:** no sign of a real difference. The paired mean is 1.2 standard errors from zero, and its
95% range is -0.0007 to +0.0020. The two sides overlap, and PyTorch is lower in two of six. One ulp in
one weight of 125M moves R2 by up to 0.0017, so the limit of 0.0002 cannot hold at this size and
length. A bias smaller than that range cannot be excluded from six pairs.

**R2 and PyTorch alternating without a pause:** in pairs 2 to 6 the card stays equally hot. R2 takes
1,065.5-1,068.5 ms and PyTorch 1,123.3-1,124.3 ms, so R2 leads 1.05x, and under the 10% rule.

**Fewer significant bits in random operands, cuBLAS sustained:**

| Bits | Card 0, round 1 | Card 0, round 2 | Card 0 clock | Card 1, round 2 |
|---:|---:|---:|---:|---:|
| 24 | 3,350 GFLOP/s | 3,232 GFLOP/s | 675-690 MHz | 3,195 GFLOP/s |
| 16 | 3,406 GFLOP/s | 3,419 GFLOP/s | 705 MHz | 3,262 GFLOP/s |
| 12 | 3,552 GFLOP/s | 3,531 GFLOP/s | 735 MHz | 3,429 GFLOP/s |
| 8 | 3,657 GFLOP/s | 3,668 GFLOP/s | 765 MHz | 3,529 GFLOP/s |

Card 1's first round heated from 55 C to 71 C across the row, which hides the effect, so it is left out.
**Closed:** 16 bits gain about 4%, and 8 bits about 11% while giving up most of the precision. Training
on fewer bits is not worth testing on this card. Empty mantissas do not explain the gain from constant
operands in run 31, so identical data must account for most of it.

**`torch.compile`, in the order R2, plain, compiled, compiled, plain, R2.** Median step after the first, ms:

| Model | R2 | PyTorch plain | PyTorch compiled | R2 over plain | R2 over compiled |
|---|---:|---:|---:|---:|---:|
| 7.2M | 22.4, 23.5 | 27.0, 27.9 | 23.9, 24.0 | 1.20x | 1.04x |
| 39.8M | 130.7, 135.8 | 143.5, 148.7 | 136.5, 137.5 | 1.10x | 1.03x |
| 125M | 977.6, 1,026.0 | 1,070.2, 1,085.5 | 1,039.0, 994.8 | 1.08x | 1.02x |

Compiled PyTorch trains to the same losses at these lengths. Its first step costs 12.7-25.3 s when
compiling and 2.5-4.4 s on the repeat in the same session. Compiled runs sit in the hottest part of the
order, which does not favour them. **Against compiled PyTorch, R2 has no proven per-step lead at any size.**
PyTorch's `fused=True` Adam has not been tested and might narrow it further.

R2's weights repeat exactly across runs here. Compiled PyTorch's do not, and plain PyTorch's differ
except at 7.2M.
