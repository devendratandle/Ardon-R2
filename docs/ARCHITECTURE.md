# R2 Architecture — Compiler & Runtime Roadmap

> **Read this first** at the start of every session that touches R2's
> compiler, runtime, IR, JIT, scheduler, or memory layout. It exists so
> we never re-derive context from scratch. If you change a locked design
> decision, update this file in the same commit.

---

## 1. Purpose of this document

R2 v0.3.4 ships a working tree-walking interpreter (~48K lines, 400+ builtins)
with a Cranelift JIT for eligible user functions.
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

## 3. Layer status (as of v0.3.4)

| Layer | Component | Status | Where it lives |
|---|---|---|---|
| Frontend | Lexer | ✅ built | `crates/r2-parser/src/lexer.rs` (152 LoC) |
| Frontend | Parser → AST | ✅ built | `crates/r2-parser/src/parser.rs` (333 LoC) |
| Frontend | REPL | ✅ built | `crates/r2-repl/src/main.rs` |
| Frontend | Type Inferencer | ✅ built | `crates/r2-types/src/infer.rs` (Phase A, 9 tests passing) |
| Frontend | Notebook UI | ✗ not started | (out of scope until Phase 4) |
| IR | Typed SSA | ✅ built | `crates/r2-ir/src/lib.rs` (Phase B, 8 tests passing) |
| Oracle | Auto-scheduler | ✅ V1 built | `crates/r2-oracle/`. `dispatch(Op, Shape) → Backend{Serial\|Rayon}`, hardware-scaled thresholds. V2 adds GPU/Cloud tiers. |
| JIT | Cranelift backend | ◐ numeric code → native; matrix-multiply queued | `crates/r2-jit/`. Compiles counted loops, imperative indexed element loops (`x[i]`, `y[i]<-…`), whole-vector/-matrix reductions & maps, and the statistics primitive set to native **checked** code — fused, CSE'd, F64X2-SIMD; guards deopt to the interpreter, NA propagates. Capability detail in the **Phase J** section below. |
| Rayon | Work-stealing | ✅ Oracle-dispatched | Parallelism owned by `r2-kernel` (never imported in builtins). Parallel: the numeric reductions, the apply family (`lapply`/`sapply`/`apply`/`tapply`/`aggregate`) over a pure-builtin allowlist, and the ML builtins (`rf`/`gbm`/`kmeans`/`cv`) via `par_for`. Closures & non-pure builtins fall back to serial. |
| GPU | Dispatcher | ✗ not started | — |
| Cloud-RAM | Shards | ✗ not started | — |
| FFI Hub | 200–500 syscalls | ✗ not started | — |
| Memory | R2-ARROW columnar | ◐ storage + kernels shipped; default-storage migration queued | `crates/r2-arrow/`. `ColumnarF64` with null bitmap; dense fast-path reductions + zero-copy element-wise binary kernels (NA via validity bitmaps); `i32`/`i64`/`bool` dtypes; mmap-backed out-of-core columns. *Remaining:* make columnar the **default** `RVal::Numeric` storage — a large atomic refactor (the `Reals` wrapper is staged in `r2-types`) — plus parquet/Utf8. |
| Microkernel ① | Math kernel | ✅ built | `crates/r2-linalg/` (1,278 LoC, BLAS L1-L3, decompositions) |
| Microkernel ② | BLAS/LAPACK | ◐ partial | LU, Cholesky, QR, SVD, symmetric eigen (`dsyev`), non-symmetric eigen (`dgeev`), triangular solve (`backsolve`/`forwardsolve`), condition number (`rcond`/`kappa`) — done. Pivoting QR, Lanczos, complex type — todo |
| Microkernel ③ | Reduction/Map/Binary/ParFor kernels | ✅ complete | `crates/r2-kernel/`. Four op families (reduce/map/binary + generic `par_for`), each Oracle-dispatched to a Serial or Rayon backend — lets domain crates use parallelism without importing Rayon. |
| Microkernel ③' | Stats domain crate | ✅ complete | `crates/r2-stats/`. Math + builtin layers; the `register_builtins() -> Vec<(name, fn)>` pattern; engine `bi_*` are 1-line adapters. |
| Microkernel ③'' | ML domain crate | ✅ complete | `crates/r2-ml/`. All 8 ML builtins (`rpart`/`rf`/`gbm`/`kmeans`/`cv`/`knn`/`naive.bayes`/`prcomp`); engine `bi_*` delegate; parallelism via `kernel::par_for`. |
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
- **Columnar numeric storage (Phase F.3, shipped)** — `RVal::Numeric`
  holds `Reals`, a lazy dual representation (boxed `Vec<Option<f64>>`
  ⇄ `Arc<ColumnarF64>`, either canonical, the other materialised on
  demand). Dense producers (`rnorm`/`seq`/…) and the fusion fast path use
  `from_dense_f64`/`from_columnar` to stay zero-repack end-to-end. Same
  dual-storage pattern exists for `Singles`/F32, `Ints`/I32,
  `Logicals`/Bool.
- **Domain crates** — `r2-stats`, `r2-ml`, `r2-data`, `r2-linalg`,
  `r2-graphics`, each exposing `register_builtins()`.
- **Console** — one unified sink (the **r2dterminal**, `r2_types::out`),
  mirroring R's `R_WriteConsole`; the frontend installs the target
  (CLI → stdout, GUI → `ConsoleBuffer`). Graphics is a separate
  lazy device (GUI window / CLI browser / script SVG).

Numerics: exact p-values (Lentz incomplete beta + AS241 `qnorm`); `lm` via
Householder QR; `solve`/`det`; **`na.rm=` honored** across all reductions.
**Level-3 BLAS is Oracle-gated multi-core *and* runtime SIMD-
multiversioned** — the `dgemm` kernel is compiled at three tiers
(AVX-512 → AVX2 → SSE2) and dispatched on `hw()` at runtime: ~2.9×/core
from AVX2+FMA, **~7× combined** with cores (6-core AVX2 box), one binary
running on any x86-64 CPU (`R2_SIMD`/`R2_NO_SIMD` knobs). `crossprod` is
multi-core too but memory-bandwidth-bound (small SIMD gain). Reaches users
via `%*%`. GUI caches + pre-warms the SVG font DB (fast first plot).

### Phase F.5 — out-of-core ops beyond reductions (shipped)

`MmapWriter` (chunked >RAM file builder), `MmapColumnar::map_to`
(out-of-core scalar map), streaming `sd`/`var`/`prod`/`range`/`length`
wired into the `bi_*` mmap interception. Surfaced as `mmap.map`.

### Phase F.6 — additional dtypes & on-disk formats (partial)

Shipped: `i64` columnar dtype; **`read.parquet`** (pure-Rust parquet/arrow
crates, row-group streaming). Pending: `Utf8` columnar dtype (native
string columns) and an Arrow-IPC / Feather zero-copy reader.

### Out-of-core compute ← NEXT

Make "assign big data → stat/ML → result" real end-to-end, with bounded RAM:

- **A1 (shipped):** `mmap.csv` — stream a CSV to per-column packed-f64
  sidecars, return a named list of `mmap.col` handles → a >RAM CSV is
  analyzable with the existing out-of-core reductions.
- **A2:** out-of-core `lm` / `cov` via **streaming normal equations**
  (accumulate XᵀX + Xᵀy in one pass — reuses the parallel `crossprod` —
  then solve the small p×p system).
- **A3:** `quantile` / `median` (external sort / t-digest) + element-wise
  and filter/subset over mmap columns.

### Phase G — GPU dispatcher (wgpu), FFI hub, cloud RAM

Oracle V2 adds GPU / Cloud backends *below* the kernel layer; builtins
stay unchanged. Best tackled once the CPU out-of-core/compute path above
is complete.

### Phase H — Accelerator hub (v0.3.0+)

Pluggable accelerators behind the kernel/Oracle boundary.

### Phase J — Type-specializing JIT (the "fastest **and** safest" bet)

**Goal:** make *modular, pure-R2 source libraries* run at near-native speed
without an unsafe escape hatch. R forces a choice — fast (unsafe C
extensions) *or* safe (slow pure-R). R2's JIT compiles R2's **checked**
semantics to native via Cranelift, so speed and safety are not traded off.
Every phase keeps a hard invariant: **guards check assumptions on entry and
deopt to the interpreter on violation — the JIT never emits unsafe code.**

Chosen direction: **expand the JIT; skip a bytecode VM** (tiered model =
AST-interpret cold code → JIT hot code directly). A bytecode tier (~2–3×
baseline uplift, R-proven) is optional and only built if cold-code startup
ever matters. Near-native for *iterative* libraries (bootstrap, MCMC,
optimisation loops) arrives around J.4. Feasible as a long-horizon effort
because Cranelift (the backend — the hard part) already exists.

**Accomplished (J.1–J.4):** pure-R2 numeric code — counted loops, imperative
`x[i]`/`y[i]<-…` element loops, whole-vector/-matrix reductions, and the entire
first-order statistics primitive set — now JIT-compiles to native, checked code.
The headline: a statistic written *once* as an R2 formula (`var`, `cov`, `cor`,
`sd`, z-score, RMSE, R², normalise) reaches or **beats** the hand-written native
builtin, with no unsafe escape hatch. Shared sub-expressions are computed once
and independent reductions run in a single SIMD pass, so `cor` written in R2 out-
runs the C-style native `cor`. (Implementation detail archived in
`code-history/phase-j-jit-detail.md`.)

| Phase | Capability | Status |
|---|---|---|
| **J.1** | Counted `for(v in a:b)` loops — scalar, math-intrinsic, and nested — compile to native. | ✅ shipped |
| **J.2** | In-loop vector element access: index folds and dot products / weighted sums (`sum(x*w)`) become fused native loops. | ✅ shipped |
| **J.3** | Real indexed loads **and** stores + matrix unboxing: imperative `x[i]`/`y[i]<-f(x[i],w[i])` loops (dot products, scalar recurrences, multi-statement folds & maps) and whole-matrix elementwise/reduction ops JIT natively. *Remaining:* 2-D `m[i,j]`, offset indices (`x[i-1]`), list `x[[i]]`. | ✅ shipped (niche remainders) |
| **J.4** | Compose source libraries to native: user-helper inlining, multi-reduction & vector-returning kernels for the statistics primitive set, with common-subexpression elimination, single-pass wave fusion, and F64X2 SIMD. A formula written once reaches/beats the native builtin. *Remaining:* matrix-multiply lowering (`%*%`→the shared `dgemm`, `cor(matrix)`, `scale`) producing matrix outputs on a scratch arena — the r2sem `fit_once` bottleneck, the largest remaining brick. | ◐ scalar/vector-reduction class shipped; matrix lowering queued |
| **J.5** | Profile-driven tiered dispatch + on-stack replacement — automatic, safe hot/cold tiering. | queued (groundwork: `explain(f)`) |
| **J.6** | Trace-based JIT across call boundaries — PyPy-class, the last ~2×. | queued (optional) |

Why native builtins still exist: the standard dynamic-language pattern is
*interpreter/JIT for breadth + native kernels for the hot 5%*. First-party
hot kernels (`lm`, `plssem`, …) stay native builtins; J.4 is what lets
*third-party source* libraries reach the same class without engine edits.

### Phase P — parallel `apply` for library code (the *other* speed lever)

**Accomplished:** `mclapply(x, FUN)` / `par.lapply` / `par.sapply` run an R2
closure over the elements of `x` across cores — the *other* speed lever (the JIT
attacks per-fit compute; this attacks embarrassingly-parallel repetition:
bootstrap, permutation tests, cross-validation, Monte-Carlo). Each element
evaluates in an isolated worker sharing the registry + global env read-only via
`Arc`, so there is no shared mutation and no data-race / silent-wrong-math class;
scope is **pure** map closures (no `<<-` to shared state), matching R's
`parallel::` isolation. **~4.5× on a 6-core interpreted workload**; composes with
the JIT — a `par.sapply` of a JIT-compiled closure runs parallel *and* native.
*Remaining:* chunking, and an opt-in guard against `<<-` in the mapped closure.

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

