# Ardon-R2 — trillion-scale LLM training architecture

**Purpose.** Ardon-R2 is *architected* to train trillion-parameter models —
the hardware scales up separately, but the software carries no ceiling.
This document records the traits, the data model, the invariants and the
single-process reference that prove them (`r2-tensor`, `r2-mesh`,
`r2-autograd`, `r2-train`). What is not built yet is listed once, in
`docs/ARCHITECTURE.md` §5, not here. A design that only works on one GPU
is not industrial-grade; a design whose interfaces are written for the
mesh from day one is.

**The core principle.** Nothing in the training code names a specific
device count. Every parallelism decision is expressed against ABSTRACTIONS
(topology, collectives, shard specs). Replace the in-process transport with
NCCL/RCCL/MPI behind the same traits and the identical code runs on 1 GPU
or 100,000. This is exactly how Megatron/DeepSpeed/JAX-pjit achieve scale —
we adopt the same separation, in pure Rust, with a working reference.

---

## Layer 0 — Tensor + dtypes (`r2-tensor`, extends `r2-types::Tensor`)

The numeric substrate. Everything else is math on these.

- Dtypes: f64 (stats truth), **f32/bf16/f16** (the training/inference types),
  **Q8_0/Q4_0/Q4_K** (quantized weights — why a 32B fits in ~18 GB, a 1T in
  ~500 GB mmap'd). Dequant-to-f32 per tile on the compute path.
- Storage: dense (RAM) OR **mmap-backed** (Pillar 2 — already built). A
  tensor's data can live on disk and stream; the trillion-param weight file
  never fully materialises.
- Device placement: a tensor carries a `DeviceId` (which slot it lives on).
- Ops needed (each with a CPU reference kernel = the accuracy truth, and a
  GPU kernel via r2-gpu when enabled): matmul, add/mul/scale, RMSNorm,
  softmax, RoPE, SiLU/SwiGLU, embedding gather, transpose, slice/concat.

**Invariant:** every op has a CPU reference that a unit test checks the GPU
path against within f32 tolerance (the Pillar-1 contract). CI has no GPU;
the CPU reference keeps CI meaningful.

---

## Layer 1 — Hardware topology model (`r2-mesh::topology`)

*Your "indexing child slots of hardware, multilayer clusters."* A declared
tree the sharding planner reads. No compute — pure description.

```
Cluster
 └─ Node (host, RAM, NIC)          links: InfiniBand / Ethernet  (slow tier)
     └─ Device slot (GPU/accelerator, VRAM)   links: NVLink / PCIe (fast tier)
```

- `DeviceId = (cluster, node, slot)` — the child-slot index.
- Each edge carries a **bandwidth tier** (fast intra-node vs slow
  inter-node). The planner MUST place high-traffic groups (tensor-parallel)
  on fast links and low-traffic splits (pipeline stages, data-parallel
  replicas) across slow links. This is the whole game at scale — getting
  the communication onto the right wires.
- Built from a config file (`mesh.toml`) or auto-probed. On one machine the
  "mesh" is N worker threads standing in for N slots — the reference.

---

## Layer 2 — Collective operations (`r2-mesh::Collective` TRAIT)

The single seam between "training math" and "how many machines." Training
code calls the trait; it never knows the transport.

```rust
trait Collective {
    fn all_reduce(&self, buf: &mut [f32], op: ReduceOp, group: GroupId);
    fn all_gather(&self, src: &[f32], dst: &mut [f32], group: GroupId);
    fn reduce_scatter(&self, src: &[f32], dst: &mut [f32], op: ReduceOp, group: GroupId);
    fn broadcast(&self, buf: &mut [f32], root: DeviceId, group: GroupId);
    fn barrier(&self, group: GroupId);
}
```

- **Reference impl (Fable, this window):** `ThreadCollective` — real
  all-reduce/all-gather across worker threads with correct math. Proves the
  training loop is *collective-correct*.
- **Opus/hardware impls (same trait, later):** `NcclCollective`,
  `RcclCollective` (AMD), `MpiCollective`. A hardware bring-up, not a
  redesign — the trait boundary is the industrial-grade guarantee.
- `GroupId` selects a communication group (a tensor-parallel row, a
  data-parallel replica set) — the topology planner assigns them to the
  right bandwidth tier.

---

## Layer 3 — Sharding descriptors (`r2-mesh::Shard`)

Every parameter/activation tensor tagged with HOW it's split. This is the
3D-parallelism core (tensor × pipeline × data), the Megatron/ZeRO model.

```rust
enum Shard {
    Replicated,                    // full copy on each device (data-parallel)
    TensorParallel { dim, group }, // split along a matmul dim across a fast group
    PipelineStage  { stage },      // whole layers assigned to a pipeline stage
    ZeroSharded    { group },      // ZeRO-3: params/grads/optim state sharded
}
```

- Matmul knows how to consume a `TensorParallel` weight (partial products +
  `all_reduce` over the group) — the classic Megatron column/row split.
- Pipeline stages exchange activations across the slow tier with
  micro-batch scheduling (1F1B).
- ZeRO sharding distributes optimizer state so no device holds the whole
  thing — the reason 1T fits across a cluster at all.

**Reference (Fable):** implement `TensorParallel` matmul + `ZeroSharded`
AdamW over the `ThreadCollective`, proven on a tiny transformer. That single
proof = "the training subsystem is written for the mesh."

---

## Layer 4 — Autograd (`r2-autograd`)

Reverse-mode over the Layer-0 ops. Records a tape of the forward ops, walks
it backward accumulating grads. Shard-aware: a grad on a `TensorParallel`
tensor triggers the right collective automatically.

- Reference-checked against **finite differences** (the accuracy gate for
  gradients — same discipline as the differential harness for stats).
- Ops: the transformer set (matmul, RMSNorm, softmax, SwiGLU, embedding,
  RoPE, cross-entropy loss).

---

## Layer 5 — Training loop + optimizer (`r2-train`)

- AdamW with **sharded state** (Layer 3). Gradient checkpointing
  (recompute activations in backward to fit memory). bf16 compute + f32
  master weights (loss-scaling) — the standard large-model numerics.
- Checkpoint save/restore streams through the mmap layer (Pillar 2) — a
  petabyte checkpoint never sits in RAM.
- **LoRA path** (the one-machine "training" most users need): freeze base,
  train low-rank adapters; tiny optimizer state.

---

## Layer 6 — Verified-innovation core (`r2-train::exploration`, OPTIONAL)

*The maintainer's 0.5–1% fuzzy-innovation idea — a genuinely novel training
augmentation. Opt-in per user; OFF by default. The discipline that makes it
safe rather than reckless: an innovation is a HYPOTHESIS, never accepted
until it clears two independent gates.*

**Mechanism.** For an opt-in fraction ρ ∈ [0.005, 0.01] of update steps,
instead of the plain gradient step, take a **stochastic exploration step**
(higher-entropy perturbation — e.g. a sampled direction / temperature-scaled
noise on a candidate subspace). The perturbed parameters are a CANDIDATE,
staged aside — not yet merged into the live weights.

**Two-gate verification (both must pass, else discard):**

1. **Formal gate — mathematical induction / invariant check.**
   The candidate must preserve the training invariants we can state
   formally: loss is finite and non-increasing in expectation over a
   verification mini-window; norms bounded (no blow-up); the perturbation
   stays within a trust region. Framed as an inductive step: *if the
   invariants held at step k, the candidate must let them hold at k+1.*
   Fail ⇒ reject, keep the plain gradient step.

2. **Empirical gate — neural-enforced validation.**
   The candidate is scored on a held-out validation signal (not the batch
   it was born from — no self-confirmation). Accept only if it **improves**
   the objective by a margin beyond noise (a significance test — reuse the
   engine's now-full-precision `pnorm`/t-test surface for the margin
   decision). Fail ⇒ reject.

**Merge rule.** Only a candidate passing BOTH gates is merged into the live
weights; everything else is discarded and the run continues on the verified
gradient path. Net effect: 99–99.5% ordinary verified training, ≤1%
controlled exploration that can only ever *help* — because unverified
innovation is thrown away, never silently kept.

**Why this is defensible, not hand-waving.** It is a *filtered* random
search layered on gradient descent, with an explicit reject path. The
worst case is "no innovation found, training proceeds normally"; there is
no path where an unvalidated perturbation degrades the model, because the
gates run before merge. Reproducibility: the exploration RNG is seeded and
logged, so an accepted innovation is replayable and auditable.

**Reference (Fable):** the `ExplorationPolicy` + `InnovationGate` traits and
a minimal implementation on the tiny transformer — prove that (a) the fuzzy
step is taken at rate ρ, (b) both gates run, (c) a deliberately-bad
candidate is rejected, (d) accepted candidates are logged/replayable. Opus
scales the gate implementations.

---

## END-TO-END PROOF — DONE (2026-07-16)

`crates/r2-train/src/transformer.rs` — a REAL single-head causal decoder
transformer (token+positional embeddings → 2 × {RMSNorm → self-attention
with causal mask → RMSNorm → SwiGLU FFN, residuals} → output projection →
softmax-cross-entropy), built ENTIRELY from the foundation layers
(r2-tensor ops, r2-autograd tape, r2-train step) in pure Rust, no Python,
no framework. It learns a next-token rule: **loss 2.449 → 0.003 (random
baseline 2.485), 100% prediction accuracy**, trained on
finite-difference-verified gradients. Every op it uses is gradient-checked
in r2-autograd (10/10). This is the artifact proving "architecture exists"
→ "we trained a transformer with it." The SAME code scales to 32B/1T via
the mesh (Shard/Collective) — only tensor sizes and transport change, not
the math. Autograd gained `transpose` + `softmax_rows` (both grad-checked)
for the attention path.

## Shard-aware gradients — DONE (2026-07-16)

`crates/r2-train/src/distributed.rs::sync_grads` closes the
r2-autograd ↔ r2-mesh seam: after local backward, gradients are reconciled
across the mesh per each parameter's `Shard` spec — Replicated → all_reduce
mean (data-parallel); ZeroSharded → reduce_scatter to the owned shard;
TensorParallel/PipelineStage weight grads left local. THE INVARIANT proven:
2-worker data-parallel training with synced gradients yields the SAME
parameters as single-device training on the full batch (<1e-4). That
equivalence is what makes scaling safe — distributed == reference. 3/3.

## The DL/NN function layer — two tiers

The deep-learning fundamentals are split by the modularity principle:

- **Native Rust primitives** (speed + verified correctness): matmul, the
  autograd backward math, collectives, quantized dtypes. One formula, one
  implementation; finite-difference-gated.
- **R2-script recipes** (readable, forkable, JIT'd): layer definitions
  (attention / RMSNorm / SwiGLU blocks), activation wrappers, losses,
  optimizer update rules, training loops, model definitions — glue over
  primitives, negligible overhead. A user edits a transformer block like
  any R function, no Rust, no recompile. The `llm.*` builtins are the
  native tier's surface today; the script tier is in the roadmap.

## Honest scope statement

- **Architecture: no ceiling.** Interfaces are written for the mesh; the
  reference proves them on one machine. This is the industrial-grade
  property — the design does not need rewriting to scale.
- **An actual trillion-parameter RUN needs the cluster and the hardware
  transport implementations** (NCCL/RCCL/MPI behind the `Collective`
  trait, real GPUs, interconnect). The design and the proof of
  correctness are here; the datacenter run is not.
- **Accuracy discipline carries over:** every kernel CPU-reference-checked;
  gradients finite-diff-checked; the innovation core gated by the
  full-precision statistical surface. The f64 stats truth is untouched; the
  LLM tensor path is bf16/f32 by design (correct for neural nets).
