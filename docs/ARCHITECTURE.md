# R2 Architecture — Compiler & Runtime Roadmap

> **Read this first** at the start of every session that touches R2's
> compiler, runtime, IR, JIT, scheduler, or memory layout. It exists so
> we never re-derive context from scratch. If you change a locked design
> decision, update this file in the same commit.

---

## 1. Purpose of this document

R2 v0.1.0 ships a working tree-walking interpreter (~10K lines, 192 builtins).
The next several versions transform R2 into a **compiled, scheduled, columnar
runtime** without rewriting the working interpreter. This file is the
single source of truth for that transformation — its layers, their status,
and the rules we have agreed on.

Anything not in this file is open to debate. Anything in this file is
locked unless this file is changed.

---

## 2. Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    R2 LANGUAGE (Frontend)                   │
│   Parser │ AST │ Type Inferencer │ REPL │ Notebook │ Script │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  R2-IR (Typed SSA, columnar)                │
│        Effects · Shapes · Cost annotations · Origins        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ⚡ ORACLE — The Auto-Scheduler ⚡              │
│   Cost Model · Accuracy Model · Placement · Precision       │
│   Decides: serial │ Rayon │ GPU │ Cloud-RAM │ mixed         │
└──┬───────────┬──────────┬──────────┬───────────┬────────────┘
   ▼           ▼          ▼          ▼           ▼
┌──────┐  ┌────────┐  ┌──────┐  ┌────────┐  ┌─────────┐
│ JIT  │  │ Rayon  │  │ GPU  │  │ Cloud  │  │ FFI Hub │
│Crane │  │work-   │  │disp- │  │ RAM    │  │ 200–500 │
│lift  │  │stealing│  │atcher│  │ shards │  │ syscalls│
└──┬───┘  └───┬────┘  └──┬───┘  └───┬────┘  └────┬────┘
   └──────────┴──────────┴──────────┴────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                R2-ARROW — Memory Substrate                  │
│  Columnar buffers · Zero-copy views · Arena+RC hybrid       │
│  NUMA-aware · RDMA-ready · GPU-mappable · mmap-capable      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              MICROKERNELS (3–4× hand-tuned)                 │
│   ① Math kernel (built)   ② BLAS/LAPACK   ③ Stats   ④ Tensor│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│      Hardware: CPU(SIMD) │ GPU │ NIC(RDMA) │ Disk           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Layer status (as of v0.1.0)

| Layer | Component | Status | Where it lives |
|---|---|---|---|
| Frontend | Lexer | ✅ built | `crates/r2-parser/src/lexer.rs` (152 LoC) |
| Frontend | Parser → AST | ✅ built | `crates/r2-parser/src/parser.rs` (333 LoC) |
| Frontend | REPL | ✅ built | `crates/r2-repl/src/main.rs` |
| Frontend | Type Inferencer | ✅ built | `crates/r2-types/src/infer.rs` (Phase A, 9 tests passing) |
| Frontend | Notebook UI | ✗ not started | (out of scope until Phase 4) |
| IR | Typed SSA | ✅ built | `crates/r2-ir/src/lib.rs` (Phase B, 8 tests passing) |
| Oracle | Auto-scheduler | ✅ V1 built | `crates/r2-oracle/` — 5 tests passing. `dispatch(Op, Shape) → Backend{Serial\|Rayon}`. `bi_kmeans` already migrated. `bi_rf`, `bi_gbm`, `bi_cv` still inline; migrate next. V2 adds GPU/Cloud. |
| JIT | Cranelift backend | ◐ scalar + vector reductions + vector maps + composed bodies | `crates/r2-jit/src/lib.rs` + engine call path (Phases C through C.4-full part 2, 16 tests passing). Scalar functions; vector reductions; element-wise `v OP scalar`; element-wise `v OP w`; **generic 1-param composed arithmetic bodies (`(v+1)*2`, `v*v - 1`, etc.) lowered via IR into single fused native loops.** NA propagates via NaN. Next: C.5 matrices, C.6 `.Internal` direct lowering. |
| Rayon | Work-stealing | ✅ Phase D wrapped | Oracle-dispatched: `bi_rf`, `bi_kmeans`, reductions (`sum`/`mean`/`sd`/`var`/`min`/`max`/`prod`/`median`), `bi_cv`, `bi_summary`, **full apply family** (`lapply`/`sapply`/`apply`/`tapply`/`aggregate`) with pure-builtin allowlist (`sum`/`mean`/`sd`/`var`/`min`/`max`/`prod`/`length`/`sqrt`/`abs`/`exp`/`log`/`log2`/`log10`), `bi_gbm` (per-iteration row-work parallel; outer boosting loop sequential by algorithm), **inner tree split-search now parallel across features via `par_for` (v0.0.9 polish — Oracle dispatches per-node based on n_features × active_samples; trees with few features stay serial automatically)**. Closures and non-pure builtins fall back to serial path. `mapply`/`vapply` multi-arg pure-allowlist parallel = V2. |
| GPU | Dispatcher | ✗ not started | — |
| Cloud-RAM | Shards | ✗ not started | — |
| FFI Hub | 200–500 syscalls | ✗ not started | — |
| Memory | R2-ARROW columnar | ◐ F.1+F.2+F.3a+F.4 shipped; F.3 storage migration pending | `crates/r2-arrow/src/lib.rs` (8 reduction + 6 binary tests). **F.4 element-wise kernels now in place**: `binary`, `binary_scalar`, `add`/`sub`/`mul`/`div` shortcut methods. Dense × dense uses tight `for i in 0..n` over `&[f64]` slices for compiler auto-vectorization; sparse path ANDs validity bitmaps so NA propagates correctly. Same semantics as `r2_kernel::binary()` but zero-copy through the columnar representation — ready to be the default arithmetic path once F.3 lands. Previous lib.rs sketch — `ColumnarF64`, null bitmap, dense fast-path reductions. All 7 numeric reductions on columnar view. F.3a: `from_option_slice` + lazy-bitmap. **F.3 attempted but reverted in-session**: changing the `RVal::Numeric` variant to take `Reals` (Vec<Real> + cached `Arc<ColumnarF64>`) requires fixing ~75 construction sites across r2-base/r2-utils/r2-engine; could not complete safely in one session's budget. The `Reals` wrapper type is committed to `r2-types` for the eventual migration. **Verdict**: destructive F.3 needs a dedicated full-budget session — it's the kind of refactor that must finish atomically (every site updated) or the build breaks. F.4: element-wise vector ops on columnar; F.5: mmap-backed columns; F.6: i32/i64/bool/Utf8 dtypes. |
| Microkernel ① | Math kernel | ✅ built | `crates/r2-linalg/` (1,278 LoC, BLAS L1-L3, decompositions) |
| Microkernel ② | BLAS/LAPACK | ◐ partial | LU, Cholesky, QR, SVD, Jacobi eigen — done. Pivoting QR, Lanczos — todo |
| Microkernel ③ | Reduction + Map + Binary + ParFor kernels (Phase K → K.4) | ✅ kernel API complete | `crates/r2-kernel/src/lib.rs` (20 tests). Four op families: `ReduceOp`, `MapOp`, `BinaryOp`, plus generic `par_for(kind, n, f)` — backend-dispatched parallel-for-each. Backends: `SerialBackend`, `RayonBackend`. Dispatchers: `reduce()`, `map()`, `binary()`, `par_for()` — all ask Oracle. **`par_for` lets ML builtins like `bi_rf` use parallelism without importing Rayon — kernel owns the dispatch (§4.9).** |
| Microkernel ③' | Stats domain crate (Phase R.0) | ✅ math + bi_* in r2-stats | `crates/r2-stats/src/lib.rs` (6 tests). Math layer + builtin layer (8 reductions via `reduce_builtin!` macro). `R2Err`/`ErrKind` moved to `r2-types`. `register_builtins() -> Vec<(name, fn)>` pattern locked. r2-engine `bi_sum`/etc are now 1-line adapters. |
| Microkernel ③'' | ML domain crate (Phase R.1 step 4: ✅ all 8 migrated) | ✅ Phase R.1 complete | All 8 ML builtins now live in `r2-ml::dispatch`: **`bi_rpart`**, **`bi_rf`**, **`bi_gbm`**, **`bi_kmeans`**, **`bi_cv`**, **`bi_knn`**, **`bi_naive_bayes`**, **`bi_prcomp`**. Engine `bi_*` are 1-line delegators. Every ML builtin that uses parallelism is on `kernel::par_for` — zero Rayon imports in any ML body. ML domain crate is the second fully-populated domain after r2-stats. |
| Microkernel ④ | Tensor | ✗ stub | `Tensor` type exists in `r2-types`, no ops |

---

## 4. Locked design decisions

These have been agreed; do not relitigate without updating this section.

1. **No restart.** The interpreter stays. New layers are added *alongside*,
   not in place of, the existing tree-walker. The REPL always has a fallback
   tree-walk path.
2. **IR is column-shaped SSA**, not scalar SSA. Values carry shape + element
   type (e.g., `Numeric[150]`, `Matrix[150, 4]`). Scalars are length-1.
3. **`.Internal()` becomes IR intrinsics.** User-defined functions and
   `.Internal()` calls meet at the IR level — no string dispatch at runtime
   for compiled paths.
4. **Compilation is per-Closure, lazily.** A user function compiles on first
   call. The compiled native code is cached on the `Closure` object alongside
   its AST body, which stays for fallback.
5. **AGPL v3 stays.** External dependencies stay Rust-only (no C/C++/Fortran).
   Cranelift, Rayon, `arrow-rs`/`arrow2`, `wgpu` are all permissible — they
   are pure Rust.
6. **Oracle V1 is a threshold dispatcher**, not a calibrated cost model.
   Real cost models, GPU placement, and Cloud sharding are V2+ work.
7. **Backwards compatibility:** existing user scripts must keep working
   through every phase. New features are opt-in or transparent acceleration.
8. **No separate bytecode layer.** R2-IR fills the bytecode role. Cranelift
   consumes R2-IR directly to produce native code (Phase C). If a portable
   fallback interpreter is ever needed, it's a small (~200 LoC) walker over
   `IrInst` — added then, not pre-emptively.
9. **Rayon lives BELOW the kernel layer, never inside builtins.** Phase D
   sprinkled `par_iter` directly into `bi_*` function bodies as an
   expedient; this is architecturally wrong. The correction (Phase K) moves
   parallelism into kernels: `bi_sum` calls `kernel::reduce(buf, Op::Sum)`,
   the kernel's backend dispatcher (Serial / Rayon / future GPU / Cloud)
   chooses how to execute. Builtins must not see Rayon, atomics, or chunking.
   This matches PyTorch ATen, JAX XLA, NumPy/BLAS layering.
10. **Engine restructure (Phase R) is a prerequisite for sustained
    development**, not a polish task. The 8 KLoC `r2-engine/src/lib.rs`
    monolith makes every session token-expensive. Splitting it into per-domain
    crates (`r2-stats`, `r2-ml`, `r2-data`, `r2-graphics`) is now in the
    critical path before Phase F.3 storage migration.

---

## 5. Build order — current state & roadmap

The full phase-by-phase build history (Phase A→F→K→R, completed) used to
live here. It duplicated `CHANGELOG.md` (the authoritative release log),
so it has been archived to `code-history/` (and remains in git history).
This section now carries only the **current state** and the
**not-yet-built phases** (kept until we reach them).

### Current state (v0.2.2)

Shipped layers:

- **Frontend / IR / JIT** — type inferencer, R2-IR (typed SSA), Cranelift
  JIT for user functions (scalar + vector reductions + fused composed
  arithmetic bodies). aarch64 falls back to the interpreter.
- **Oracle V1** — hardware-scaled serial/Rayon dispatcher
  (`dispatch(Op, Shape) → Backend`); `R2_FORCE_SERIAL` knob for A/B.
- **Kernel layer** (`r2-kernel`) — `reduce`/`map`/`binary`/`par_for`,
  Oracle-dispatched; Rayon lives here, never in builtins.
- **Arrow bridge** (`r2-arrow`) — `ColumnarF64` + null bitmap + dense
  reductions; **memory-mapped out-of-core** (`mmap.col` → streaming
  `sum`/`mean`/`min`/`max`, larger-than-RAM); vector⊗scalar chain fusion.
- **Domain crates** — `r2-stats`, `r2-ml`, `r2-data`, `r2-linalg`,
  `r2-graphics`, each exposing `register_builtins()`.
- **Console** — one unified sink (the **r2dterminal**, `r2_types::out`),
  mirroring R's `R_WriteConsole`; the frontend installs the target
  (CLI → stdout, GUI → `ConsoleBuffer`). Graphics is a separate
  lazy device (GUI window / CLI browser / script SVG).

Numerics: exact p-values across the suite (Lentz incomplete beta + AS241
`qnorm`); `lm` via Householder QR; `solve`/`det` exposed.

### Phase F.3 — `RVal::Numeric` storage migration ← NEXT

Make numeric storage natively `ColumnarF64` (drop the `Option<f64>`
re-pack). Removes the residual element-wise repack cost the v0.2.2 fusion
work exposed, makes the engine zero-copy / mmap-friendly, and lets the
out-of-core path extend beyond reductions. Destructive — shipped as its
own release.

### Phase F.4–F.6 — element-wise kernels, mmap dtypes

General element-wise kernels on the columnar substrate; a chunked/append
mmap writer + Parquet / Arrow-IPC readers (so >RAM files can be built
from R2 itself); additional dtypes.

### Phase G — GPU dispatcher (wgpu), FFI hub, cloud RAM

Oracle V2 adds GPU / Cloud backends *below* the kernel layer; builtins
stay unchanged.

### Phase H — Accelerator hub (v0.3.0+)

Pluggable accelerators behind the kernel/Oracle boundary.

> Release history → `CHANGELOG.md`. Archived phase narrative →
> `code-history/`.

---

## 6. File map for fast orientation

| Need to find… | Look here |
|---|---|
| A builtin function | `crates/r2-engine/src/lib.rs` (search `bi_<name>`) |
| Builtin registration | Same file, search `("name",bi_name)` (two registry blocks ~lines 270-300 and ~2000-2160) |
| RVal type | `crates/r2-types/src/lib.rs` (top of file) |
| Closure / user function shape | `crates/r2-types/src/lib.rs` ~line 122 |
| AST node types (`Expr`) | `crates/r2-types/src/lib.rs` |
| Tree-walk evaluator | `crates/r2-engine/src/lib.rs` `fn eval_in` and `fn call_fn` |
| Formula NSE preprocessing | `crates/r2-engine/src/lib.rs` ~line 451 (`if matches!(fname, "lm" \| "rpart" \| ...)`) |
| `.Internal()` intrinsics | `crates/r2-engine/src/lib.rs` `bi_internal` ~line 6247 |
| Math kernel (BLAS-style) | `crates/r2-linalg/src/{level1,level2,level3,decomp,solve}.rs` |
| Matrix struct | `crates/r2-types/src/lib.rs` ~line 173 (column-major, `Vec<f64>`, `col_names`/`row_names`) |
| Embedded datasets | `crates/r2-base/src/lib.rs` (iris, mtcars, airquality, ToothGrowth, faithful) |

---

## 7. Token-efficiency working agreement

To keep design conversations short:

1. **Reference this file by section number** — "see §4.3" beats re-explaining.
2. **Excerpt, don't dump.** Long REPL output → 5 lines + `(rest as before)`.
3. **One layer per session.** No layer-hopping mid-conversation.
4. **Subagents for code surveys.** "Find every place X is used" goes to
   the `Explore` subagent — its summary returns, not raw greps.
5. **Commit after each working feature.** Sessions starting from a clean
   git state need less context to bootstrap.
6. **Update this file when decisions change.** Stale design docs are worse
   than missing ones.

---

## 8. Out-of-scope reminders

These are tempting but explicitly **not** part of the architecture push:

- Rich rpart summary (CP table, surrogate splits) — deferred indefinitely.
- True categorical splits in tree models — deferred.
- Notebook frontend — deferred to Phase 4+.
- Distributed cluster execution — V3.0, do not design now.
- Replacing the parser — it works, leave it.

---

## 9. Open questions (not yet decided)

- Should `RVal::Bytecode` be a new variant or live as an `Option` field on
  `Closure`? (Leaning: field on Closure, keeps `RVal` enum smaller.)
- ~~Should the IR be SSA-with-phi or direct-style with mutable locals?~~
  **Resolved (Phase B):** SSA-with-phi. See `crates/r2-ir/src/lib.rs`.
- ~~ARROW: roll our own buffers or depend on `arrow2`?~~
  **Resolved (Phase F.1-F.2):** rolled our own (`crates/r2-arrow/`) — zero
  deps, columnar-Arrow-shaped, ready for arrow2 interop later if needed.
- **Kernel API granularity (Phase K)**: should kernels operate on
  `&ColumnarF64` directly, or on a more abstract `Buffer<T>` trait that
  could later wrap GPU/Cloud-resident data? Leaning toward `Buffer<T>` so
  GPU backend is an addition, not a rewrite. Resolve when Phase K starts.
- **Phase R crate boundaries**: where do borderline functions live?
  e.g., `summary()` on a data.frame fans out per-column stats — does it
  live in `r2-stats` (it computes statistics) or `r2-data` (it's a frame
  operation)? Likely `r2-data` calling into `r2-stats` kernels. Resolve
  when Phase R starts.

Resolve these in the relevant phase, then move them to §4 (locked decisions).

---

