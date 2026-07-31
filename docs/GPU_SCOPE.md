# How much of the CPU stack needs a GPU twin?

**Short answer: about 5%, and that is a design decision rather than a
shortcut.** The CPU stack is ~6,900 lines across kernels, columnar memory
and linear algebra. The GPU needs roughly **10 kernels plus one buffer
layer** — because most of the CPU stack *must not* run on a GPU, and some
of it has no GPU meaning at all.

## What must NEVER move to the GPU

WGSL has no f64. Ardon-R2's entire promise is R-faithful accuracy — the
differential harness pins `pnorm`, `var`, `sd`, `cor`, `lm` and the rest
against CRAN R to 14–16 significant figures. Running those in f32 would
give ~7 digits and silently break the product's defining claim.

So the accuracy contract in `r2-gpu`'s header is a scope decision as much
as a safety one:

| CPU component | Lines | GPU twin? | Why |
|---|---:|---|---|
| `r2-kernel` reduce / scan / agg | ~800 | **Never** | `sum`/`var`/`cor` in f32 lose accuracy to cancellation. These are the harness-verified paths. |
| `r2-stats` (distributions, tests) | large | **Never** | Same reason: f64 is the product. |
| `r2-linalg` (BLAS, LU/QR/SVD) | 2,799 | **Never** | Decompositions are numerically delicate; f32 SVD is not the same answer. |
| `r2-arrow` columnar + mmap | 2,018 | **No** | Solves *larger-than-RAM on CPU*. The GPU analogue is streaming weight tiles to VRAM — a different mechanism, not a port. mmap stays CPU-side for loading. |
| `rayon` parallelism | — | **No** | wgpu's dispatch *is* the parallelism. There is nothing to translate. |
| `r2-jit` (Cranelift) | 2,600 | **No** | CPU codegen. The GPU equivalent is WGSL generation, which `r2-gpu` already does for element-wise ops. |

That is roughly **6,900 lines with no GPU counterpart by design**, plus
the JIT. Anything that would need porting is precisely the code we are
contractually obliged *not* to run in f32.

## What the GPU actually needs

Only the f32 ML path — high arithmetic intensity, where f32 is the
accepted numeric type anyway:

| Kernel | Status |
|---|---|
| matmul (tiled, shared-memory) | ✅ done — 67 GFLOP/s on an integrated Radeon |
| element-wise maps (relu/sigmoid/tanh/exp/scale) | ✅ done |
| weight residency (`upload` / `matmul_resident`) | ✅ done — 2.3× at 2048³ |
| rmsnorm | ▢ |
| softmax (row-wise, max-stable) | ▢ |
| RoPE | ▢ |
| SiLU / SwiGLU | ▢ (composable from element maps + mul) |
| fused attention (scores → mask → softmax → context) | ▢ |
| backward: matmul | ✅ routed (transpose + matmul) |
| backward: rmsnorm / softmax / rope | ▢ |

**That is ~10 kernels, not a second stack.** `r2-tensor/ops.rs` — the
complete transformer op set — is 7 public functions. The GPU mirror of it
is small because a transformer is a small op vocabulary repeated many
times, which is exactly why GPUs suit it.

## The real work is NOT the kernels

Measured evidence (see `crates/r2-gpu/examples/residency.rs`): weight
residency gives **2.3× at 2048³ but only 1.1× at training shapes**,
because what dominates there is uploading activations and downloading
results — traffic residency does not touch.

So the missing piece is **a device-resident activation layer**: values
stay in VRAM between ops so a whole layer runs without a host round trip,
and only the loss comes back. That is one buffer/lifetime abstraction —
call it `DeviceTensor` — plus teaching the tape to hold device handles
instead of `Vec<f32>`.

It is worth being explicit that this is the same work a *discrete* GPU
would need. The current shortfall is not an integrated-GPU limitation;
per-call transfers hurt a PCIe card more, not less.

## Order of work

1. **`DeviceTensor`** — resident buffers with lifetimes. Unlocks
   everything else; without it, extra kernels just add more round trips.
2. **rmsnorm / softmax / rope** — the three ops between the matmuls. Once
   activations are resident these keep the data on-device.
3. **Fused attention** — biggest single win, and it removes the
   slice/concat traffic the fused-batch trainer currently pays on CPU.
4. **Backward for the above** — backward is ~⅔ of training FLOPs; the
   matmul half is already routed.

## The estimate

**~10 kernels and one buffer layer: on the order of 1,500–2,000 lines**,
against ~6,900 lines of CPU stack that stays CPU-only forever. The GPU is
a narrow accelerator for the f32 ML path, not a parallel universe of the
statistical runtime — and keeping it narrow is what protects the accuracy
guarantee that makes Ardon-R2 worth using.
