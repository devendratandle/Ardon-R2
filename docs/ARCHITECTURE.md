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

## 5. Build order (phases)

Work proceeds in order. Each phase is independently mergeable and shippable.

### Phase A — Type inferencer  *(small, no rebuild risk)*
- Pure annotation pass over AST.
- Output: every AST node tagged with shape + element type (or `Unknown`).
- Single new file `crates/r2-types/src/infer.rs` (~300 LoC).
- Engine ignores it for now — it just runs and validates.

### Phase B — R2-IR types & builder
- New crate `crates/r2-ir/`.
- Defines `IrNode`, `IrType`, `IrShape`, `IrFunc`.
- Builder lowers an annotated AST → IR.
- Still no runtime impact — IR is built then thrown away (validation only).

### Phase C — Cranelift JIT for user-defined functions
- New crate `crates/r2-jit/`.
- IR → Cranelift codegen → native code.
- `Closure` gains an optional compiled-pointer field.
- Compile on first call. **This is the V2.0 “10-20× faster” milestone.**

### Phase D — Rayon expansion ✅ wrapped (will be GUTTED in Phase K)
Status: shipped. 20+ builtins use `par_iter` via Oracle dispatch. This was
the right *behavior* for V0.1.0 but is the wrong *architecture* — Rayon
leaks into builtin bodies. Phase K migrates these calls down into a
kernel layer; the builtins shrink dramatically when they stop owning the
parallelism decision.

### Phase E — Oracle V1 ✅ shipped
`crates/r2-oracle/` — `dispatch(Op, Shape) → Backend{Serial | Rayon}`. Reused
unchanged in Phase K (kernels ask Oracle).

### Phase F.1, F.2, F.3a — ARROW bridge ✅ shipped
`crates/r2-arrow/` — `ColumnarF64`, null bitmap, dense fast-path reductions.
8 reductions go through the bridge. Lazy-bitmap optimization in F.3a.
Storage migration (F.3 destructive) **deferred** — see ordering below.

**v0.2.2 additions on top of the bridge:**
- **Out-of-core**: `MmapColumnar` (mmap'd packed-f64) exposed to R2 as
  `mmap.write` / `mmap.col`; `sum`/`mean`/`min`/`max` stream over the
  mapping → larger-than-RAM reductions with bounded memory.
- **Vector⊗scalar fusion**: left-leaning `(v OP lit) OP lit …` chains
  collapse to one pass in the engine (≈2.4× on `v*2+1`).
- **Unified console sink** (“r2dterminal”, `r2_types::out`): engine +
  every compute crate’s output converge on one frontend-installed sink,
  mirroring R’s `R_WriteConsole`.

### Phase F.3 — destructive storage migration ← NEXT (justified by v0.2.2 perf)
Make `RVal::Numeric` natively `ColumnarF64` (no `Option<f64>` re-pack).
This removes the residual element-wise repack cost the fusion work
exposed, makes the whole engine zero-copy / mmap-friendly, and lets the
out-of-core path extend beyond reductions.

### Phase K — Kernel layer extraction ✅ largely shipped (correction phase)
**Goal:** move all parallelism out of builtins into a backend-dispatched
kernel layer. Lock in the architecture diagram's "Microkernels" tier.
`crates/r2-kernel/` exists (`reduce`/`map`/`binary`/`par_for`, Oracle-
dispatched). Reductions/apply/ML route through it. Remaining: migrate the
last inline reduction sites + columnar `var`/`sd`/`median`.

- New crate `crates/r2-kernel/` (or extend `r2-arrow` / `r2-linalg`).
- Define traits: `Reducer<T>`, `Mapper<T>`, `BinaryOp<T>`.
- Each kernel has serial + Rayon impls. Backend chosen via Oracle.
- Builtins call `kernel::reduce(buf, Op::Sum)` etc. — never `par_iter`
  directly.

### Phase K — Kernel layer extraction ← NEXT (correction phase)
**Goal:** move all parallelism out of builtins into a backend-dispatched
kernel layer. Lock in the architecture diagram's "Microkernels" tier.

- New crate `crates/r2-kernel/` (or extend `r2-arrow` / `r2-linalg`).
- Define traits: `Reducer<T>`, `Mapper<T>`, `BinaryOp<T>`.
- Each kernel has serial + Rayon impls. Backend chosen via Oracle.
- Builtins call `kernel::reduce(buf, Op::Sum)` etc. — never `par_iter`
  directly.
- **Migration target**: `bi_sum`, `bi_mean`, `bi_sd`, `bi_var`, `bi_min`,
  `bi_max`, `bi_prod` first (they're already columnar-aware). After K
  lands, those builtins become 5-10 lines each.
- After K, the apply family's `pure_apply` allowlist routes through
  kernels too; many entries become 1-line wrappers.

### Phase R — Restructure engine into per-domain crates ← in progress
**Goal:** split `r2-engine/src/lib.rs` (~6.8 KLoC) into per-domain crates.
Each consumes the kernel API and exposes `register_builtins()`.

**Phase R foundation ✅ shipped:**
- `R2Err`, `ErrKind` moved from `r2-engine` to `r2-types`. Re-exported
  from r2-engine for compatibility. Domain crates can now return `R2Err`
  without an engine dependency — the cycle is broken.
- The `register_builtins() -> Vec<(name, fn)>` pattern locked in
  (signature: `fn(&[EvalArg]) -> Result<RVal, R2Err>`). r2-engine adapts
  at registration time.

**Phase R.0 — `r2-stats` ✅ done:**
- Math layer: `sum`/`mean`/`min`/`max`/`prod`/`var`/`sd`/`median`/`cov`/
  `cor`/`sqrt`/`abs`/`exp`/`ln`/`add`/`sub`/`mul`/`div`.
- Builtin layer: 8 reductions defined via `reduce_builtin!` macro.
- `register_builtins()` returns the 8 reduction (name, fn) pairs.
- r2-engine `bi_sum`/etc are now 1-line adapters: `r2_stats::bi_sum(a)`.
- Engine LoC for these 8 functions dropped from ~250 to ~8.
- 6 r2-stats tests passing.

**Phase R.1 — `r2-ml` scaffold ◐ established, function moves blocked:**
- Crate created with `register_builtins() -> Vec::new()` for now.
- Empty by design: ML builtins (rf, gbm, kmeans, rpart, knn, naive.bayes,
  cv, prcomp) are tangled with engine-private helpers:
  - `extract_ml_data`, `build_tree`, `tree_predict_one`, `parallel_random`,
    `count_splits`, `serialize_tree`
  - These use `&mut Engine` for type coercion (as_reals, scalar_f64).
- **Sub-plan to actually move ML builtins**:
  1. Extract pure tree helpers (`build_tree`, `tree_predict_one`,
     `count_splits`, `serialize_tree`, `parallel_random`) into
     `r2-ml::tree` — algorithmic, no engine needed. (~200 LoC moved.)
  2. Move `as_reals` / `scalar_f64` from Engine to RVal methods in
     r2-types. (~50 LoC, mechanical.)
  3. Refactor `extract_ml_data` to use the relocated helpers.
  4. Move each `bi_*` ML function one by one.
- Each step is one focused session. Step 1 unblocks the rest.

**Phase R.2 — `r2-data` ✅ complete:**
- ✅ R.2 steps 1-5: `cbind`, `rbind`, `c`, `filter`, `select`, `arrange`,
  `mutate`, **`summary`** (data-shaped paths only — DataFrame and Numeric).
  Per-column fan-out uses `kernel::par_for(PerElementMap, ncols, ...)`.
- ✅ R.2 step 6: full migration of the **apply family** —
  `lapply`, `sapply`, `vapply`, `mapply`, `apply`, `tapply`, `aggregate` —
  to `r2_data::apply`. Engine retains 1-line delegators wrapping the
  original bodies in `#[allow(unreachable_code)]` blocks for safe rollback.
- **Pattern split-handler**: `summary(model)` for lm/glm/rpart/rf/gbm/etc.
  stays in r2-engine (model-aware logic with engine-private helpers like
  `signif_stars`, `fmt_pval`); engine calls `r2_data::summary::try_summary`
  first and falls through if it returns None. Clean separation:
  data-shape summaries in r2-data, model-shape summaries in r2-engine.
- **Pattern EngineCtx trait**: new `r2_types::EngineCtx` trait exposes
  `ctx_call_fn(func, args, env) -> RVal` so r2-data can invoke closure
  callbacks during apply without depending on r2-engine. Engine implements
  the trait (`impl EngineCtx for Engine` delegates to `call_fn`). Migrated
  apply functions take `<C: EngineCtx + ?Sized>` and route closure paths
  through the trait while the `pure_apply` allowlist still dispatches via
  `r2_kernel::par_for` for parallel pure-builtin execution.
- 14 builtins migrated, 15 tests passing across 5 modules
  (bind, concat, dplyr, summary, apply).

**Phase R.3 — `r2-graphics` ✅ complete:**
- ✅ Migrated all 8 graphics builtins to `r2-graphics`:
  primary plots (`plot`, `hist`, `boxplot`, `barplot`) in `plots` module,
  overlays (`lines`, `points`, `abline`, `legend`) that splice into the
  open `plot.svg` in `overlays` module.
- All builtins follow the locked pure pattern
  `fn(&[EvalArg]) -> Result<RVal, R2Err>` — no engine dependency. Args
  coerce via `RVal::as_reals()` / `scalar_f64()` from r2-types.
- **Pattern split-handler** (matches summary): `plot(model)` for
  `lm`/`glm`/`gbm`/`kmeans` keeps its model-aware dispatch in r2-engine
  (it constructs synthetic args from `inst.fields` and recurses). The
  `RVal::TypeInstance` arm runs in engine, then falls through to
  `r2_graphics::plots::bi_plot` for the data path.
- Engine retains 1-line delegators; previous bodies preserved under
  `#[cfg(any())]` `_legacy_*` shadow functions for safe rollback during
  the migration window.
- 5 r2-graphics tests pass (plot file write, hist null, boxplot empty
  error, barplot null, overlay-without-plot error).

After R.3, r2-engine drops further toward ~1500 LoC: parser glue,
registry, JIT integration, environment, REPL bridge. Each future builtin
lands in its domain crate, never in the engine.

**Phase R.4 — `r2-base::linalg_ops` ✅ complete:**
- ✅ Migrated `matrix`, `tensor`, `t`, `crossprod`, `svd`, `eigen` —
  thin wrappers over r2-linalg's BLAS-style kernels (`dgesvd`, `dsyev`)
  and the `Matrix` / `Tensor` constructors in r2-types.
- **Crate placement:** the linalg-domain builtins live in
  `r2-base::linalg_ops` rather than `r2-linalg::ops` because r2-types
  already depends on r2-linalg (Matrix methods call into BLAS-style
  kernels). Putting builtins in r2-linalg would create a cycle. r2-base
  sits one layer up: depends on r2-types and r2-linalg, exposes the
  builtins as the standard pure `fn(&[EvalArg]) -> Result<RVal,R2Err>`.
- **Honesty fix on `bi_svd`** (see `docs/KNOWN_LIMITATIONS.md`): the
  function previously returned identity placeholders for `$u` / `$v`
  while `dgesvd` does not yet accumulate orthogonal factors. The new
  implementation returns `$d` only and prints a one-line note. Silent
  identity-matrix returns are gone.
- 6 builtins migrated, 5 tests passing in r2-base.
- Engine becomes 1-line delegators for all six.
- **Kernel-layer honesty fix:** `r2_linalg::dgesvd` signature changed
  from `Result<(Vec<f64>, Vec<f64>, Vec<f64>), _>` (sigma, U, Vᵀ — the
  last two were identity placeholders) to `Result<Vec<f64>, _>` (sigma
  only). Both callers (the new `bi_svd` builtin and the engine's
  `.Internal svd` glue) updated. No future caller can resurrect the
  silent-failure mode by reading the U/Vᵀ slots — the slots are gone.

**Phase R.5 — finish r2-ml ✅ complete:**
- The 8 ML builtins (`prcomp`, `kmeans`, `knn`, `naive_bayes`, `rpart`,
  `rf`, `gbm`, `cv`) had been migrated to `r2_ml::dispatch::*` in
  Phase R.1 but engine retained their full bodies under
  `#[allow(unreachable_code)]` blocks for safe rollback. R.5 collapses
  every one of those to a true 1-line delegator.
- Orphan helpers retired from engine: `extract_ml_data` (126 lines —
  now lives at `r2_ml::data::extract_ml_data`), and the tree-walk
  helpers `tree_predict_one`, `print_tree`, `serialize_tree`,
  `count_splits` (38 lines combined).
- Net reduction: r2-engine `lib.rs` shrinks from 7282 → 6218 lines
  (~1064 lines / ~14% smaller). All 123 workspace tests still pass.

**Phase R.6 — `r2-strings` ✅ complete:**
- New crate `r2-strings` hosts 14 character-vector builtins:
  `toupper`, `tolower`, `substr`, `grep`, `grepl`, `gsub`, `sub`,
  `regexpr`, `strsplit`, `paste`, `paste0`, `nchar`, `trimws`, `sprintf`.
- Pure pattern: `fn(&[EvalArg]) -> Result<RVal, R2Err>`. Depends only on
  r2-types. No regex dep — matching uses substring `contains`/`find`/
  `replace` (documented in KNOWN_LIMITATIONS).
- **Bug fix at point of migration:** `bi_substr` had an off-by-one in
  the engine: `substr("abcdef", 2, 4)` returned `"bcde"` (4 chars)
  instead of R's `"bcd"` (3 chars). Corrected here; test pins the fix.
- 13 r2-strings unit tests pass. Engine `lib.rs` 6218 → 6126 lines.
  All 136 workspace tests pass.

**Phase R.7 — finish `r2-data` ✅ complete:**
- 18 builtins migrated, organised into four new modules:
  - `r2_data::table` — `table` (frequency counts, prints + returns named Integer)
  - `r2_data::meta` — `nrow`, `ncol`, `dim`, `colnames`, `rownames`,
    `is.data.frame`, `as.data.frame`, `head`, `tail`
  - `r2_data::clean` — `na.omit`, `complete.cases`, `merge`
  - `r2_data::order` — `duplicated`, `unique`, `order`, `rank`
- `do.call` lands in `r2_data::apply` alongside the rest of the apply
  family. Uses the `EngineCtx` trait (same pattern as `lapply`/`sapply`)
  to invoke the target function without depending on r2-engine.
- 17 builtins are pure `fn(&[EvalArg]) -> Result<RVal,R2Err>`; `do.call`
  takes `&mut C: EngineCtx + ?Sized` for the call-back.
- **`merge()` honest scoping (v0.1.0):** single-key inner join with
  `.y` suffix on collisions. Multi-key joins, `all.x`/`all.y` outer
  joins, and `by.x`/`by.y` are tracked as v0.2.0 work in
  KNOWN_LIMITATIONS.
- 10 new r2-data unit tests pass (table char-counts, dim of df, head
  takes-n, is.data.frame discriminator, na.omit drops NA from numeric,
  complete.cases marks clean rows, duplicated/unique/order/rank).
- Engine `lib.rs` 6126 → 5846 lines. All 146 workspace tests pass.

**Phase R.8 — `r2-io` ✅ complete:**
- New crate `r2-io` hosts 7 file/text I/O builtins:
  `read.csv`, `read.table`, `read.delim`, `write.csv`, `write.table`,
  `file.exists`, `list.files`. Pure pattern, depends only on r2-types.
- **CSV parser (v0.1.0):** full RFC 4180 state-machine parser shipped
  in Phase R.14. Handles embedded separators in quoted fields,
  doubled-quote escape `""`, multi-line quoted fields, and UTF-8 BOM
  stripping. Write side escapes quotes/separators/newlines in column
  names and character values. See ARCHITECTURE.md §11 + KNOWN_LIMITATIONS.
- Engine-state I/O kept in r2-engine: `save` and `load` need
  `e.global_env.bindings` for session round-trip and remain there. Also
  retained: the unused `bi_read_csv_v2` (preexisting orphan, not in
  registry).
- 5 r2-io tests pass (read CSV round-trip, write CSV produces quoted
  header, read.table tab default, read.csv `header=FALSE` uses V-names,
  file.exists reports truthfully).
- Engine `lib.rs` 5846 → 5726 lines. All 151 workspace tests pass.

**Phase R.9 — `r2-stats` deepening ✅ complete:**
- Two new modules in r2-stats: `dist` (probability distributions) and
  `summary` (windowed/cumulative/quantile reductions).
- Migrated 14 builtins:
  - `dist`: `dnorm`, `pnorm`, `qnorm` (correlate with `phi`, `qnorm_approx`).
  - `summary`: `cor`, `cov`, `cumsum`, `cumprod`, `cummax`, `cummin`,
    `diff`, `quantile`, `range`, `which.min`, `which.max`.
- **Numerical helpers relocated:** `erf`, `phi`, `qnorm_approx` moved
  from r2-engine into `r2_stats::dist` and re-exported from the crate
  root. Engine now `use`s them rather than redefining locally —
  duplicate definitions retired.
- **R.9 scope decision:** Random-variate generators (`rnorm`, `runif`,
  `sample`, `rbinom`, `rpois`, `set.seed`) are NOT migrated this phase.
  They share an RNG state with `r2_ml::tree::SEED_STATE` and are kept
  in r2-engine pending a separate decision on where the RNG primitive
  should live (candidates: r2-stats hosts the canonical state, r2-utils
  becomes RNG home, or r2-rng new crate). Statistical tests
  (`t.test`, `chisq.test`, `aov`, `fisher.test`) also defer — they
  return `RVal::TypeInstance` and have engine-private formatting
  helpers that need their own split-handler analysis.
- 11 new r2-stats tests pass. Engine `lib.rs` 5726 → 5597 lines.
  All 162 workspace tests pass.

**Phase R.10 — hit-and-get hypothesis tests ✅ complete:**
- New module `r2_stats::htest` hosts 6 hypothesis tests:
  `t.test`, `chisq.test`, `cor.test`, `shapiro.test`, `wilcox.test`,
  `fisher.test`. All migrate as plain pure builtins
  (`fn(&[EvalArg]) -> Result<RVal,R2Err>`).
- **Architectural distinction:** unlike fitted-model functions
  (`lm`/`glm`/`aov`), these tests are *hit-and-get* — the function call
  itself prints the formatted result and returns a small
  `RVal::TypeInstance` for programmatic use. There is no `summary(test)`
  step. So no split-handler pattern is needed; the migration is straight
  data-in / result-out. (Multi-group / model functions like `aov` and
  `anova` legitimately need split-handler treatment because their
  `summary()` differs by class — those stay deferred.)
- **Numerical primitives relocated:** `t_cdf`, `chi_sq_cdf`, `ln_gamma`,
  `gamma_approx`, `incomplete_beta`, `fmt_pval`, `signif_stars` moved
  from r2-engine into `r2_stats::htest` and re-exported at the crate
  root. Engine model summaries (`lm`/`glm` output formatting) keep
  using them via the re-export — duplicate definitions retired.
- **Module-naming note:** the new file is `htest.rs`, not `tests.rs` —
  using `tests` would clash with the conventional `#[cfg(test)] mod
  tests` block already present in r2-stats's lib.rs.
- **Honest scoping:** `fisher.test` uses the chi-squared approximation
  with Yates correction rather than R's exact hypergeometric. Accurate
  for moderate counts; off in the extreme tail. Documented in
  KNOWN_LIMITATIONS.
- 6 new r2-stats htest tests pass (one-sample t-test on centered data,
  cor.test perfect-correlation, chisq.test uniform-obs high-p,
  shapiro.test result shape, fmt_pval scaling, signif_stars thresholds).
- Engine `lib.rs` 5597 → 5176 lines (~421 lines collapsed across the 6
  delegators + 67 lines of relocated helpers). All 168 workspace tests
  pass.

### Phase F.3 — RVal::Numeric storage migration to Arc<ColumnarF64>
After K + R, F.3 becomes mechanical: only the storage layer in `r2-types`
changes; domain crates already speak through kernels and don't touch raw
`Vec<Option<f64>>`. The 130-site refactor scope from the F.3 attempt notes
shrinks dramatically once builtins no longer destructure RVal::Numeric
directly — most accesses go through kernel adapters.

### Phase F.4-F.6 — Element-wise kernels, mmap, dtypes
Element-wise vector kernels (add/sub/mul/div) on columnar buffers. mmap-backed
columns for big files. i32/i64/bool/Utf8 dtypes. Independent of K/R/F.3
ordering after they land.

### Phase R.S.2 — MANOVA ✅ on `dev` (targets v0.2.0)

`manova(cbind(y1, y2, ...) ~ group, data=df)` ships full multivariate
ANOVA. Pipeline:

1. Parse the formula. LHS arrives as a `Matrix` (from `cbind` eval)
   or as a wrapped `List([(name, Matrix)])`; both are accepted.
2. Build group indices from the RHS grouping factor.
3. Compute E (within-group SSCP) and H (between-group SSCP), both
   p × p column-major.
4. Form `M = E⁻¹H` via `r2_linalg::dgetri` + manual GEMM.
5. Extract eigenvalues of M via QR iteration with Gram-Schmidt
   (small-p closed forms for p = 1, 2). Acceptable accuracy for the
   p ≤ 10 typical of practical MANOVA.
6. Compute all four classical statistics from the eigenvalues:
   Wilks' Λ, Pillai's V, Hotelling-Lawley trace, Roy's largest root.
7. F-approximation for Wilks via Rao's formula (R's default).
   Other statistics emit raw values; their F-approximations are
   v0.2.1 polish.

Verified against R's canonical iris example: R2 produces Wilks
Λ ≈ 0.0239 vs R's 0.0234, F ≈ 197.6 vs R's 199.1 — within
numerical tolerance of the QR-iteration eigenvalue routine.

### Phase R.S.2 — Multivariate hypothesis tests (Hotelling T²) ✅ on `dev` (targets v0.2.0)

New `crates/r2-stats/src/multivariate.rs` (~480 LoC, no new deps).
Single dispatcher `hotelling.test(...)` resolves to three flavors:

- One-sample: tests H₀: μ = μ₀ for a p-dimensional mean.
  T² = n · (x̄−μ₀)ᵀ S⁻¹ (x̄−μ₀); F = T²·(n−p)/(p·(n−1)) ~ F(p, n−p).
- Two-sample: pooled-covariance T² for two independent groups.
- Paired: one-sample T² on the per-subject difference matrix.

Math is exact to machine precision for T², F, and df. P-value uses
the Wilson-Hilferty F→z approximation (consistent with the existing
`aov` path) — accurate for moderate n, tightens to R's exact F-CDF
as n grows. Hand-verified test case (n=4 paired, p=2) produces
T²=318, F=106 exactly.

Five unit tests for the three flavors. Accuracy verification script
in `samples/test_hotelling_accuracy.r2` with hand-computed expected
values and equivalent R `Hotelling::hotelling.test()` calls printed
alongside R2's output.

### Phase R.S.1 — Repeated-measures aov() + Error() syntax ✅ on `dev` (targets v0.2.0)

`aov(y ~ x + Error(subject), data=df)` now performs proper one-way
within-subject ANOVA:

- `r2-engine`: new helper `split_error_term` walks the formula RHS,
  lifts `Error(...)` out of the predictor expansion, and tags the
  stratum as `~error` in the formula list. Nested
  `Error(subject/treatment)` collapses to the outermost stratum for
  the one-way RM case.
- `r2-stats/src/models.rs`: new `aov_repeated_measures` decomposes
  total variance into SS_subject (between-subject), SS_treatment
  (within-subject fixed effect), and SS_within (within-subject
  residual). Output matches R's `summary(aov(...+Error(...)))`
  two-stratum layout (Error: subject + Error: Within) — bit-identical
  to R 4.5.3 when R uses `factor(subject)`.

`t.test()` extended with two formula-shaped paired forms R itself
rejects:

- `t.test(y ~ group + Error(subject), paired=TRUE)` routes to the
  existing `pair_by_id` infrastructure, treating Error(subject) as
  an implicit id= argument.
- `t.test(y ~ Error(subject), paired=TRUE)` is the row-order paired
  shortcut — each subject must have exactly 2 observations; the
  first becomes "obs1" and the second "obs2".

Teaching-style errors fire on the common confusion patterns:

- Fixed effect inside Error(): `aov(y ~ Error(drug))`.
- Stratum equals the fixed effect: `aov(y ~ drug + Error(drug))`
  or `aov(y ~ drug + Error(drug/subject))`.
- Paired-formula t.test without `paired=TRUE`.
- Pair-by-row-order with any subject not having exactly 2 obs.

Three unit tests, hand-verified math against independent computation
plus R 4.5.3 reference output.

### Phase R.M — Cranelift JIT aarch64 gate ✅ shipped (v0.1.1)

`r2-jit` now exposes `pub const JIT_SUPPORTED: bool = cfg!(target_arch
= "x86_64")` and `try_compile_closure()` returns `None` immediately on
unsupported targets. The engine cleanly falls back to the interpreter
on aarch64-apple-darwin and aarch64-unknown-linux without ever reaching
`JITModule::new()`, which panics on aarch64 in `cranelift-jit 0.105`
during PLT construction.

The JIT test module is gated by `#[cfg(all(test, target_arch =
"x86_64"))]` so CI on Apple Silicon hosts is clean. Statistical
outputs are bit-identical across architectures — only wall-clock
performance differs on aarch64 for compute-bound workloads. Lifting
the gate is a v0.2.0 task (upgrade `cranelift-jit` to a version with
aarch64 PLT support, absorb the API churn).

### Phase R.G — In-memory `PlotDevice` + full `par()` ✅ shipped (v0.1.1)

`r2-graphics` re-architected around a thread-local `PlotDevice`
(`crates/r2-graphics/src/device.rs`) holding the SVG body, canvas
geometry, full `PlotParams`, and a multi-panel cursor. Replaces the
old file-state model (which detected "is a plot open" by reading
`plot.svg` from cwd) — that model was racy under cargo's parallel
test execution and limited the crate to a single plot at a time.

Three new builtins ship in this phase:

- **`par(...)`** — three call shapes (`par()` snapshot, `par("name")`
  query, `par(name=value, ...)` setter with previous-value return for
  `oldpar <- par(...)` restore). Supports `mfrow`, `mfcol`, `mar`,
  `oma`, `cex`, `cex.axis`, `cex.lab`, `cex.main`, `col`, `bg`, `fg`,
  `lty`, `lwd`, `pch`, `las`, `new`. Defaults match CRAN R 4.5.x.
- **`dev.off()`** — close the current device, reset to defaults.
- **`save_plot(path)`** — explicitly flush the device SVG to a file
  (useful for multi-panel canvases or non-default filenames).

Multi-panel `mfrow=c(2,2)` / `mfcol=c(2,3)` is implemented end-to-end:
`begin_plot()` returns the rectangle for the current panel and advances
the cursor for the next call, with row-major or column-major fill
chosen by `mfrow` vs. `mfcol`. The cursor wraps cleanly when overflowed.

The auto-flush legacy UX is preserved — each `plot()` / `hist()` /
`boxplot()` / `barplot()` still writes its default file (`plot.svg`,
`hist.svg`, etc.) at the end of the call.

The previously-flaky `lines_errs_when_no_plot_open` test is no longer
ignored. It now relies on the device's in-memory `has_plot` flag
instead of filesystem state and passes deterministically on every
platform, independent of cargo's test-parallelism order.

### Phase R.G.2 — Built-in HTTP plot viewer with session gallery ✅ shipped (v0.1.1)

`crates/r2-graphics/src/server.rs` (~290 LoC, `std::net` only — zero
external dependencies). `dev.view()` starts a tiny HTTP server on
`127.0.0.1:8765` (scans `8765..8775` if the first port is in use) and
opens the user's default browser via per-OS shell-out (`cmd /c start`,
`open`, `xdg-open`). The server thread is started lazily on first
call, guarded by `OnceLock` so repeat calls are idempotent.

Endpoints:

| Path | Behavior |
|---|---|
| `GET /` | Self-contained HTML page with two panes (no CSS/JS deps) |
| `GET /current.svg` | Most recently modified `.svg` in cwd — what the live pane polls |
| `GET /list` | JSON `[{name, mtime}]` of every `.svg` in cwd, newest first |
| `GET /<name>.svg` | Any `.svg` in cwd by name (path-traversal guarded) |

The HTML page is a two-pane layout:

- **Current** — auto-refreshing `<img>` polling `/current.svg` every
  1500 ms via a JavaScript probe-then-swap pattern (avoids the
  visible flicker of a full `<img>.src` reset on each tick).
- **Session gallery** — grid of thumbnails for every `.svg` in cwd,
  rebuilt every 2 s from `/list`. Each thumbnail is clickable to pin
  the top pane to that file. A "return to live" link resumes polling.
  Closes the UX gap from R.G where users felt earlier plots had
  "vanished" — they were always on disk, but the viewer previously
  only displayed `/current.svg`.

A general-purpose **`readline(prompt="")`** builtin was added to
`r2-engine` so scripts can pause for stdin input. Returns the typed
line as a character scalar (trimmed of trailing newline). Used by
`samples/demo_graphics.r` for the interactive walk-through.

Limitations of v0.1.1:

- The server thread is a daemon. In script mode the process exits
  when the script returns, killing the server. Scripts that use
  `dev.view()` should end with `readline()` or `Sys.sleep()` to keep
  the process alive while the browser is in use. REPL mode keeps the
  process alive naturally.
- Browser polls every 1.5 s — fine for interactive use, not animation.
- No native window. A real GUI window via `winit` + `tiny-skia` would
  be a separate phase; cost ~1000–2000 LoC. Path 3 ("HTTP server +
  browser") was chosen for v0.1.1 because it gives 80% of the felt
  UX for 10% of the work.

### Phase G — GPU dispatcher *(via wgpu)*, FFI Hub, Cloud RAM
Deferred. Becomes a new Backend impl in the kernel trait — Phase K's
abstraction makes adding GPU surface-only work, no builtin changes.

**Hardware introspection bit shipped early (Phase G partial):** the
`r2_oracle::hw()` module detects cores, FMA/AVX2/AVX-512, arch, OS,
and scales parallel thresholds per-machine. GPU/FFI/Cloud bits all
remain at 0%.

### Phase H — Accelerator Hub (future, v0.3.0+)

A generic trait abstraction for non-CPU compute backends, layered above
Phase G's individual implementations. Pattern: trait `AcceleratorBackend`
lives in a core `r2-accelerator` crate; concrete backends ship as optional
feature-gated workspace members.

- **`r2-accel-wgpu`** — GPU via wgpu. Pure-Rust, cross-vendor. Default
  accelerator target. (This is what Phase G GPU dispatcher actually
  produces; Phase H names the abstraction it implements.)
- **`r2-accel-tpu`** — Google TPU via XLA HLO IR compilation. Requires
  XLA toolchain bindings (non-pure-Rust dep), gated behind
  `--features tpu`. Building R2 without this feature produces a smaller
  binary with no TPU code. TPU-specific: only useful for TPU-shaped
  workloads (huge ML training), not generic stats. Niche.
- **`r2-accel-cuda`** — direct CUDA bindings (skips wgpu). Only worth
  building if a real user needs CUDA-specific features wgpu can't
  expose. Same feature-gating pattern.
- **`r2-accel-distributed`** — cluster / Cloud-RAM shards as an
  accelerator backend (workloads dispatched to remote nodes). Bridges
  to the original Phase G Cloud-RAM concept.

Oracle's `Backend` enum extends with `Accelerator(BackendId)`; per-op
dispatch logic stays in `r2-kernel`, per-backend implementation lives
in the respective feature-gated crate.

**Why Phase H is a separate phase from G:** Phase G is one specific
implementation (GPU via wgpu). Phase H is the *abstraction* that lets
multiple implementations coexist. Building Phase H first would be
premature; building it after Phase G ships wgpu gives us a working
reference implementation to define the trait against.

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

## §10. v0.1.0 Status Summary

**Engine 7282 → ~4860 lines (-33%) across R.4–R.14 migrations + Tier 0–4 polish.**

### Numerical correctness shipped this version

- **Full thin SVD** with orthogonal factors: `dgesvd_full(m, n, A) → (σ, U, Vᵀ)`. Householder bidiagonalization with `dorgbr`-style reverse application of stored reflectors onto thin identities, diagonalization via Bᵀ·B + already-shipped `dsyev_full`. Honest κ-dependent accuracy caveat documented; proper `dbdsqr` deferred.
- **`dsyev_full`** (Householder tridiag + Wilkinson-shift QR with back-transform) — replaces the eigenvalues-only Jacobi `dsyev` for `eigen()` and `prcomp()$rotation`. R/LAPACK sign convention.
- **Welch–Satterthwaite df** for `t.test` two-sample unequal-variance (was pooled Student — silent statistical bug).
- **Exact hypergeometric `fisher.test`** via `lchoose`/`hypergeom_pmf` (was χ² approximation).
- **`subset(df, cond)` / `transform(df, name = expr)` NSE** wired through engine pre-processor with child-env binding of df columns.
- **`&` / `|` elementwise logical ops** with R-correct NA semantics (`TRUE & NA = NA`, `FALSE & NA = FALSE`, `TRUE | NA = TRUE`, `FALSE | NA = NA`). Previously silently returned all-zero numeric — a real lurking bug.
- **glm full diagnostics**: null deviance, AIC (family-specific log-likelihood), Fisher iterations, dispersion, std errors / z values / p values; `summary(glm_fit)` now matches R's coefficient table.

### Architecture changes shipped this version

- **F.3/F.6 storage migration**: `RVal::Numeric` → `Reals(Vec<Real> + OnceLock<Arc<ColumnarF64>>)`; same for `RVal::Integer` → `Ints`, `RVal::Logical` → `Logicals` (packed-bit `ColumnarBool` is ~64× smaller than `Vec<Option<bool>>` in memory).
- **F.4 columnar binary kernels** + **F.5 mmap-backed `MmapColumnar`** with zero-copy `&[f64]` view.
- **JIT NA-aware zero-copy bridge**: engine JIT call sites use `Reals.columnar()` for dense `&[f64]` slice; output reconstruction via input bitmap, not NaN-encoding.
- **JIT branchy code** (Phase C.5): multi-block IR lowered inside per-element row loop; new `VectorTernaryMap` ABI for 3-arg ifelse-shape closures.
- **R.11 model split-handler**: `lm`/`glm`/`aov`/`anova` data path migrated to `r2_stats::models`; engine retains 1-line delegators + the `summary()` formatter.
- **R.12 RNG consolidation**: all six random-variate builtins drive from a single `r2_stats::rng::SEED_STATE` so `set.seed()` is genuinely reproducible across the family.
- **R.13 regex feature**: `regex-lite` (pure-Rust POSIX-ERE) behind default-on `regex` feature for `grep`/`grepl`/`gsub`/`sub`/`regexpr`.
- **R.14 RFC 4180 CSV**: state-machine parser handling embedded separators, doubled quotes, multi-line fields, UTF-8 BOM stripping.
- **Tier 4 kernel polish (Phase K.5/K.6)**: `TernaryOp::MulAdd` (uses `f64::mul_add` on FMA hardware); `reduce_strided` for zero-copy reductions over non-contiguous matrix rows.

### Test count

**168 (baseline) → 233 passing** with clean build. All migrations and new features have direct regression tests.

### Honest scoping that remains

(Full list in `docs/KNOWN_LIMITATIONS.md`. None block v0.1.0 ship.)

- **Lentz CF for incomplete-beta** — current trapezoidal-rule integration is ~1e-4 accurate; matters only for `t.test`/`F.test` in extreme-tail regions.
- **`sprintf` width/precision** — current subset handles `%d %f %s %e %%`.
- **`merge()` multi-key / outer-join** — v0.2.0 work.
- **PNG/PDF graphics backends** — currently SVG only.
- **`Reduce`/`Filter`/`Map`** — apply-family extras for v0.2.0.
- **Oracle calibration** — intentionally bundled into Phase G hardware-awareness work; calibrating on a single dev machine produces non-portable thresholds.

### Out of scope for v0.x

D&C eigensolver, GPU/WGPU backend, full closure JIT, distributed training, autograd. Documented as future or rejected items.

---

## §11. Cumulative phase log (R.4 baseline → v0.1.0)

The following phases shipped between the original Phase R.10 snapshot and v0.1.0. Recorded here so the architecture history is complete in this file.

| Phase | What |
|---|---|
| Phase F.3 — RVal storage migration | `RVal::Numeric(Vec<Real>, Attrs)` → `RVal::Numeric(Reals, Attrs)` where `Reals = Vec<Real> + OnceLock<Arc<ColumnarF64>>`. Same pattern applied to `RVal::Integer` and `RVal::Logical` (Phase F.6 storage). `Reals`/`Ints`/`Logicals` all expose `Deref<Target=[T]>` so the bulk of read-side code compiles unchanged. ~440 + ~80 + ~50 construction sites updated via auto-injection script with manual fixups for the few sites that consume the vec rather than borrow it. |
| Phase F.4 — element-wise columnar binary kernels | `ColumnarF64::binary(op, other)` and `binary_scalar(op, scalar)` with `ArrowBinaryOp { Add, Sub, Mul, Div, Pow, Mod }`. Dense×dense fast path is a tight `for i in 0..n` over `&[f64]`; sparse path ANDs validity bitmaps. |
| Phase F.5 — mmap-backed columnar reader | `MmapColumnar` type behind default-on `mmap` feature flag (`memmap2`, pure-Rust). Reads packed `.f64` files as borrowed `&[f64]` without heap allocation. Reductions on the borrowed slice; bridge to owned `ColumnarF64` via `to_columnar()`. |
| Phase F.6 — additional columnar dtypes | `ColumnarI32` (packed `Vec<i32>` + null bitmap; reductions return `i64` for sum to avoid overflow) and `ColumnarBool` (one-bit-per-value packed values + separate validity bitmap; ~16× memory reduction vs `Vec<Option<bool>>`). `count_true` uses `count_ones()` popcount with trailing-byte masking. `any`/`all` honour R's NA-aware semantics. |
| Phase F.6 storage | `RVal::Integer` → `Ints(Vec<Integer> + Arc<ColumnarI32>)`, `RVal::Logical` → `Logicals(Vec<Logical> + Arc<ColumnarBool>)`. Same Deref/From pattern as Reals. |
| Phase R.11 — model split-handler | `bi_lm` / `bi_glm` / `bi_aov` / `bi_anova` migrated from engine to `r2_stats::models` (~535 LoC moved). Engine retains 1-line delegators + the `summary(model)` formatter (split-handler — formatting stays in engine because TypeInstance fields carry engine-private decorations). |
| Phase R.12 — RNG home | `SEED_STATE` + `next_random` + `parallel_random` consolidated in `r2_stats::rng` (was `r2_ml::tree`). All six random-variate builtins (`rnorm`/`runif`/`sample`/`rbinom`/`rpois`/`set.seed`) migrated; all now drive from the global atomic so `set.seed()` is genuinely reproducible across the whole distribution family (previously each had its own ad-hoc seed init — silent reproducibility bug). |
| Phase R.13 — regex feature flag | `regex-lite` behind default-on `regex` feature in r2-strings. `grep`/`grepl`/`gsub`/`sub`/`regexpr` route through the regex engine; `fixed=TRUE` forces literal; pattern that fails to compile silently falls back to substring. |
| Phase R.14 — RFC 4180 CSV | State-machine parser replacing line-split. Handles embedded separators in quoted fields, doubled-quote escape, multi-line quoted fields, UTF-8 BOM stripping. Write side properly escapes column names and character values. |
| Tier 1 — `dsteqr` algorithm upgrade | New `r2_linalg::dsyev_full(n, A) → (eigenvalues, eigenvectors)`: Householder tridiagonalization + implicit symmetric QR with Wilkinson shift + back-transform. Replaces the eigenvalues-only Jacobi path for `eigen()` and `prcomp()$rotation`. R/LAPACK sign convention applied (largest-magnitude entry per column is positive). |
| JIT NA-aware perf | Engine JIT call sites now use `RVal::Numeric.columnar()` to grab the dense `&[f64]` slice directly — zero-copy bridge. Eliminates one of the two `O(n)` allocation passes per JIT call. Output reconstruction uses input bitmap structurally rather than NaN-encoding (`combine_unary_output` / `combine_binary_output` helpers). |
| `$call` capture | Engine NSE preprocessor stringifies the original `Expr::Call` via `fmt_expr` and injects it as `_call` named arg; `bi_lm` / `bi_glm` / `bi_aov` store it in `$call`. `summary(model)` prints `Call:` block with the actual formula + data instead of generic `lm(formula)`. |
| glm full diagnostics | `bi_glm` extended with null deviance (intercept-only model), AIC (family-specific log-likelihood: Bernoulli / Poisson + Stirling factorial / Gaussian), Fisher Scoring iteration count (returned by IRLS helpers), dispersion parameter, std errors / z values / p values via `(X'WX)^-1 × dispersion`. Engine `summary()` adds a glm-specific branch: dispersion, null/residual deviance, AIC, Fisher iterations; coefficient table headers swap to `z value` / `Pr(>|z|)`. |
| Oracle calibration — plan change recorded | Originally listed as Tier 4 polish (build an `r2-bench` crate that fits per-Op parallelism thresholds from measured crossover points). **Removed from Tier 4 and folded into Phase G hardware-awareness work.** Reason: bench numbers measured on a single dev machine don't transfer across core counts, ISA / SIMD width, or cache hierarchies — so calibration before hardware introspection is *less* portable than the current conservative hand-tuned constants. The Phase G closure is: detect cores/CPU-features/cache via `r2_oracle::hw`, make thresholds parametric in `Hw` via closed-form rules, *then* layer optional per-machine bench refinement on top. Full rationale recorded in KNOWN_LIMITATIONS.md → "Oracle layer — calibration intentionally bundled with hardware awareness". |
| Phase C.8 — SIMD f64x2 vectorized vector map | New `compile_vector_simd_map_f64x2` path: when an IR body is "SIMD-clean" (single block, only `Const`/`Unary`/`Binary` arith + native-instr Calls like `sqrt`/`abs`/`floor`/`ceil`/`trunc`/`round`/`min`/`max`, no branches, no Rust-call externs like sin/cos/log/exp), Cranelift emits a 2-elements-per-iteration loop using `F64X2` SIMD types. Loop body uses `fadd.f64x2`, `fmul.f64x2`, `sqrt.f64x2` etc. — all natively supported on x86_64 (SSE2 mandatory) and aarch64 (NEON). Scalar remainder handles odd-length tails. Cfg-gated to x86_64/aarch64 targets; other archs fall through to the existing scalar generic path. **Honest finding:** the path is correct (tested for both even and odd inputs) but doesn't reduce time on memory-bandwidth-bound workloads (e.g. `sqrt(x*x + 1)` on 1e6 doubles is dominated by 24 MB of memory traffic at ~4 GB/s = 6 ms minimum, vs the actual ~14 ms — SIMD halves compute work but memory traffic stays). Pays off on compute-heavy bodies where multiple math ops chain per memory access. |
| Phase C.7 — generic 2-arg vector map | `compile_vector_binary_map_generic` extends the multi-block IR lowering to two input vectors, mirroring the 1-arg (Phase C.5) and 3-arg paths. Closures like `function(x, y) sqrt(x*x + y*y)` now JIT to a single fused native loop. Closed the 7.3×-slower-than-R gap on `sqrt(x²+y²)` to 2.7×. |
| Push A — native math instructions | `IrInst::Call(sqrt|abs|floor|ceil|trunc|round|min|max)` now emits the corresponding Cranelift instruction (`fsqrt`, `fabs`, `floor`, `ceil`, `trunc`, `nearest`, `fmin`, `fmax`) instead of a `call` to the Rust wrapper. Single hardware instruction, no call-dispatch overhead. Transcendentals (`sin`, `cos`, `exp`, `log`, …) still route through Rust-call wrappers (Cranelift emits `call`, the target is a Rust `extern "C" fn` whose body calls `f64::sin()` etc. — pure Rust, not foreign-function-interface in the OS sense) since x86 has no direct hardware instruction for them. |
| Phase C.6 — M-R2-JIT (math-extern Call lowering) | New `IrInst::Call` arm in `lower_inst` lets Cranelift JIT compile user closures whose bodies include math functions (`sqrt`, `abs`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `trunc`, `sign`, `^`/pow, `atan2`, `min`, `max`). Each math function is registered as an `extern "C" fn(f64,…) -> f64` symbol on the `JITBuilder` and declared as an import on every `JITModule`; `lower_inst` emits a direct Cranelift `call` instruction with the f64 args. `BinOp::Pow` routes through the same `^` extern. Closes the "broad JIT spectrum without bytecode VM" target: any user closure whose body is scalar arithmetic + comparisons + these 24 math functions now compiles end-to-end to native code with no per-call interpreter checkpoint and no intermediate bytecode layer. End-to-end verified: `sin(x)^2 + cos(x)^2` returns exact 1.0 on 1e6-element vector input. |
| Phase F.7 — Single-precision (f32) opt-in storage | New `ColumnarF32` mirrors `ColumnarF64`'s API for f32 payload. New `r2_types::Singles` wrapper (dual-storage `OnceLock<Vec<Option<f32>>>` + `OnceLock<Arc<ColumnarF32>>`, lazy materialisation either direction — same shape as F.3's `Reals`). New `RVal::Single(Singles, Attrs)` variant. New builtins: `as.single(x)` to coerce, `is.single(x)` to test. Engine `binary_op` extended with **NumPy-style dtype promotion**: `Single op Single → Single` (stays in f32), any other mixing promotes both sides to f64 and produces `Numeric`. **Memory savings**: 4 bytes/elem vs 8 — half the footprint. **Precision tradeoff**: ~7-9 decimal digits in storage vs f64's ~15-17. Reductions internally promote to f64 for accumulation accuracy (`ColumnarF32::sum` returns f64). **Scope**: storage-only optimization; computation precision unchanged when it matters. Suitable for memory-bound workloads on large datasets where storage size dominates total RSS. Skip for stats results requiring full f64 precision (regression coefficients, eigendecomposition). 6 new ColumnarF32 unit tests. |
| Phase C.9 — Fused map-reduce JIT | New `compile_vector_map_reduce(body_ir, reduce_op)` and `FusedReduceOp` enum (`Sum`, `Prod`). When `try_compile_closure` sees `function(x) sum(inner_expr)` or `function(x) prod(inner_expr)` where `inner_expr` is a function of the closure param, it lowers the inner expression to IR and compiles a fused Cranelift loop with the shape `(*const f64, i64) -> f64` (Vector1ToScalar). The loop carries the accumulator as a block parameter (Cranelift's Phi-via-block-param pattern), threads it through all IR blocks of the inner body, and combines each iteration's result into the running accumulator before advancing `i`. **No intermediate vector is materialised.** Multi-block inner bodies (with branches via Phase C.5) are supported — the accumulator and loop index thread through Jump/Branch/Phi terminators just like the vector-map paths. **Bench impact**: `function(x) sum(sqrt(x*x + 1))` on 1e7 elements drops from **0.66s (unfused, materialises 8 MB intermediate three times for `x*x`, `+1`, `sqrt`) to 0.06s (fused, single pass, ~6 ns/elem) — 11× faster.** Bit-identical results. Note: only the **closure form** triggers fusion; the inline form `sum(sqrt(v*v + 1))` still evaluates intermediates because the engine doesn't yet do AST-level operator-fusion pre-analysis (separate future work, much bigger lift). For users writing performance-critical code, the discipline is: wrap fusable chains in a `function(x) ...` and call it. |
| Phase M.1 — F64ScratchPool (r2-memory) | `F64ScratchPool` — per-thread recyclable `Vec<f64>` pool, bucket-organised by power-of-two capacity (16 elements through 256M elements), with per-bucket cap of 4 buffers to prevent unbounded growth on workloads churning many distinct sizes. Public API: `scratch_acquire(min_cap)`, `scratch_release(buf)`, `with_scratch(min_cap, fn)`, `scratch_stats()` for diagnostics. Wired into the kernel-layer paths that materialise short-lived numeric buffers — `ReduceOp::Median` (serial and strided), `nth_smallest`, `bi_quantile`. Pool stats verified: 11 hits + 1 miss over 12 acquires of the same size in tight loop. **Honest scoping**: per-call savings are ~1-2ms for medium buffers; the pool's real value is in tight loops doing the same op on the same data size (bootstrap resampling, repeated quantile, k-means distance per iteration). Doesn't help when the allocated buffer becomes long-lived output (e.g. JIT binary op result that moves into a `Reals` — that buffer can't return to pool). Real-world wins emerge in workloads with allocation-bound inner loops, not in single-call benchmarks. |
| Phase B.1 — closure capture inference via partial evaluation | `r2_ir::collect_free_vars(body, params)` walks an `Expr` AST and returns the set of free symbol names — symbols referenced that aren't bound by the closure's params or by `Expr::Assign` along the path. `r2_ir::substitute_constants(body, name→f64 map)` returns a fresh `Expr` tree with each free `Expr::Symbol(name)` replaced by `Expr::NumLit(value)`. `try_compile_closure` in r2-jit now calls these before lowering: for each free var in the body, look up the closure's captured env; if the binding is a numeric scalar (Real/Int/Bool of length 1), bake it in as a compile-time constant. Closures like `scale <- 2.5; f <- function(x) x * scale + 1` previously fell through to interpreter (free var blocked JIT lowering); now `scale` is substituted and the closure JITs end-to-end. No new ABI surface — partial evaluation at compile time, all downstream JIT paths handle the substituted body identically to literal-constants code. Correctness invariant: relies on R's by-value capture semantics, which are stable for the closure's lifetime. |
| Phase K.7 — Scan kernel | New `ScanOp` enum (`Cumsum`, `Cumprod`, `Cummax`, `Cummin`) with Serial + Rayon backends. Rayon uses two-pass Blelloch-style parallel scan: per-chunk reduce → sequential prefix combine → per-chunk re-scan with seed. Threshold 4096 elements; below that, serial. NA propagates forward (any None poisons everything after, matching R's `cumsum(c(1, NA, 3))` → `c(1, NA, NA)`). Migrated `bi_cumsum` / `bi_cumprod` / `bi_cummax` / `bi_cummin` in r2-stats to route through the kernel — bonus correctness fix: previous cummax/cummin builtins didn't propagate NA past the first None. |
| Phase K.8 — Select/find kernel | New `which_max` / `which_min` (NA-aware, returns 0-based first-occurrence index), `nth_smallest` (quickselect via stdlib `select_nth_unstable_by`, O(n) average), `top_k` / `bottom_k` (binary-heap based, O(n log k) regardless of input size). Skip NAs in quickselect-style ops to match R's `na.rm=TRUE` default. Migrated `bi_which_min` / `bi_which_max` to route through the kernel — fixes NA propagation (previous impl silently skipped NAs, could give wrong answer when actual extremum is NA). |
| Phase K.9 — Rolling-window kernel | New `RollingOp` enum (`Sum`, `Mean`, `Max`, `Min`, `Sd`). Output length = `n - w + 1` (right-aligned, no padding, matches `zoo::rollapply(align="right")`). Sum/Mean use incremental sliding-sum update O(n). Max/Min use deque-based O(n) sliding extremum (each element enters/leaves deque at most once). Sd uses two-pass-per-window (simpler than Welford incremental, avoids numerical drift on long windows). NA: any None in the window emits None at that position. New user-facing builtins `rollsum` / `rollmean` / `rollmax` / `rollmin` / `rollsd` in r2-stats. |
| Phase K.10 — Hash aggregation kernel | New `AggOp` enum (`Sum`, `Mean`, `Count`, `Min`, `Max`) with `hash_agg(op, keys, values)` and convenience `hash_tabulate(keys)`. Uses stdlib `HashMap<u64, Acc>` keyed on `u64` hashed keys; preserves insertion order of keys via parallel `Vec<u64>` + index map. Replaces O(n²) linear scans that builtin code was doing for table-style work with O(n) hash. Output is `HashAggResult { keys, values }` — flat parallel arrays. |
| Phase K.11 — Distance kernel | New `DistanceOp` enum (`Euclidean`, `Manhattan`, `Cosine`) with `distance(op, a, b)` for two same-length point vectors and `pairwise_distance(op, data, nrow, ncol)` for an n×n distance matrix from a row-major n×p data matrix. NA-aware (skips positions where either operand is None; matches `na.rm=TRUE`). Diagonal=0 + symmetric (D[i,j]=D[j,i]). Parallel path via `par_for_rayon` over n² indices when `n >= 16` and Oracle picks Rayon. Shared kernel for kmeans/knn/hierarchical clustering distance computations. |
| Phase L.1 — list-aware auto-parallel dispatch | New `r2_oracle::Op::ListMap` (threshold 10,000 aggregate work units, hardware-scaled). New `r2_types::list_meta()` extracts per-component `ComponentInfo` (name, kind, length) + aggregate `total_work` + `homogeneous_kind` flag. The apply-family `map_items` path now computes per-item work units (vs the previous `items.len()`) so `lapply(list(big_vec_a, big_vec_b, big_vec_c), f)` dispatches Rayon based on aggregate work — fixing a real bug where 3-component lists of large vectors were staying serial because `3 < 50K threshold`. New `r2_kernel::par_for_rayon(n, f)` helper for callers that have already made the dispatch decision (avoids the Oracle round-trip). New `list.meta()` user-facing builtin returns the same metadata to R2 scripts. Naming: **L-R2-Dispatch** (parallel to **M-R2-JIT**). |
| Phase G partial — hardware-aware Oracle | New `r2_oracle::hw()` returns a cached `Hw` snapshot with: `cores` (via `std::thread::available_parallelism`), `has_fma`/`has_avx2`/`has_avx512` (via `std::is_x86_feature_detected!`, cfg-gated for x86_64), `ram_mb_hint` (via `R2_RAM_MB` env var; auto-detect deferred to Phase G proper), `arch`/`os` (via `std::env::consts`). Zero new deps. `dispatch(op, shape)` now scales the per-Op parallel threshold by core count: `threshold = base * clamp(8/cores, 0.25, 8.0)` — 1-core VM raises the bar 8×, 64-core server lowers it to 0.25×. `TreeBuild` / `KFoldCV` exempted (always parallel by design). Closes the loop opened in the earlier "Oracle calibration plan change" entry: we now have the hardware-introspection layer that the deferred calibration work depends on, without making any single-machine-calibration commitment. Full `r2-bench` calibration still deferred. |
| F.3 native-columnar storage migration | `Reals` storage upgraded from `(Vec<Real>, OnceLock<Arc<ColumnarF64>>)` to dual-`OnceLock` form where **either** representation can be the canonical one and the other materialises lazily. Producers that natively yield dense `f64` (e.g. `rnorm`, `runif`, the engine binary fast path) now build via `Reals::from_dense_f64(Vec<f64>)` / `Reals::from_columnar(ColumnarF64)` — the boxed `Vec<Option<f64>>` view is **never materialised** if no caller asks for `&[Real]`. Reductions (`sum`/`mean`/`min`/`max`) route through `ColumnarF64`'s native methods on cached `&[f64]` slices, skipping the per-element `Option<f64>` match path. `Deref<Target=[Real]>` preserved for backward compat — legacy callers see no behavior change, only better performance on paths that stay columnar end-to-end. **Bench impact (R2 vs CRAN R 4.5.3, default Rblas, on this machine):** `vec_add_1e7` 30× slower → 4.2× slower; `sum_mean_1e7` 7.5× slower → 1.6× slower; `lm_1e5x5` 1.6× slower → 2× **faster**; `matmul_500x500` already 1.4× faster, unchanged. New 3-test `dataset_integrity` mod in r2-base asserts canonical R checksums on built-in datasets so future transcription errors fail at build time. |
| Tier 4 — kernel polish: MulAdd + strided reduction | New `TernaryOp::MulAdd` (`a*b + c` via `f64::mul_add` — single rounded op on FMA hardware) with `TernaryBackend` trait, Serial/Rayon impls, Oracle-driven `ternary(op, a, b, c)` dispatcher. New `reduce_strided(op, data, offset, stride, count)` for zero-copy reductions over non-contiguous matrix rows (column-major layout: row access has stride = nrow). Rayon strided path is two-pass: parallel NA scan + parallel reduce of unwrapped values. All `ReduceOp` variants supported. 8 new tests pin serial/rayon agreement, NA propagation, and reconstruction against the naive copy-into-Vec path. Eliminates the row-reduction allocation that domain crates were paying for matrix-row reductions. |
| Tier 0 — subset/transform NSE + logical-ops fix | The NSE branches for `subset(df, cond)` and `transform(df, name = expr)` were already wired in the engine pre-processor but had no integration coverage and the docs flagged them as deferred. Wrote `crates/r2-engine/tests/nse_subset_transform.rs` (4 tests covering simple subset, compound `&` subset, transform append, transform overwrite). The compound-condition test surfaced a real engine bug: `&` and `|` on logical vectors had no handler in `binary_op` and were silently falling through the arithmetic arm, returning all-zero numeric output. Added an explicit handler with R-correct NA semantics (`TRUE & NA = NA`, `FALSE & NA = FALSE`, `TRUE \| NA = TRUE`, `FALSE \| NA = NA`), handling both elementwise vector (`&`/`\|`) and scalar short-circuit (`&&`/`\|\|`) forms. Also fixed inverted `BinOp` display strings (lexer maps single `&` → `BinOp::And`, double `&&` → `BinOp::AndShort` — naming is misleading; clarified with code comment). Stale docstrings on `bi_subset`/`bi_transform` and KNOWN_LIMITATIONS section updated. |
| Tier 1 — full thin SVD with U and V | New `r2_linalg::dgesvd_full(m, n, A) → (σ, U, Vᵀ)`. Householder bidiagonalization (Golub-Kahan) stores left/right Householder vectors during phase 1 and reconstructs U₁ (m×n) / V₁ (n×n) by reverse application to thin identities (`dorgbr`-style — avoids the dimensional impossibility of maintaining a thin Q under right-multiplication by m×m Householders). Phase 2 diagonalizes the bidiagonal via symmetric eigendecomposition of Bᵀ·B (n×n tridiagonal) using the already-shipped `dsyev_full`, producing σ²/V₂; U₂ is recovered from B·V₂·diag(1/σ) with rank-deficient column zeroing. Final factors: U = U₁·U₂, V = V₁·V₂. `bi_svd` in r2-base + `.Internal("svd",…)` in engine now populate `$d`, `$u`, `$v` (R convention: $v is V itself, not Vᵀ). The old `dgesvd` (values-only) is retained for callers that don't need vectors. **Honest accuracy caveat:** Bᵀ·B route squares the condition number — well-conditioned matrices (κ ≲ 1e7) get ~1e-12 accuracy; ill-conditioned matrices lose accuracy on the small singular values. Proper LAPACK `dbdsqr` (implicit-shift bidiagonal QR with full Givens accumulation) would give κ-independent accuracy at higher implementation cost; tracked in KNOWN_LIMITATIONS. |
| Phase C.5 — JIT branchy vector code | `compile_vector_map_generic` rewritten from single-block to full multi-block IR lowering: every IR block becomes a Cranelift block nested inside the per-element row loop; `i` threaded as the first block param of every IR block; `Return` jumps to a shared `tail` block that stores `out[i]` and increments. `Branch`/`Jump`/`Phi` handled identically to the scalar Phase C.1 path. NA semantics preserved via input-bitmap reconstruction (NaN comparison → unordered → else branch, but output bitmap = AND of input bitmaps so NA inputs propagate to NA outputs regardless of branch taken). New `JitKind::VectorTernaryMap` + `compile_vector_ternary_map_generic` with `(*const f64, *const f64, *const f64, *mut f64, i64)` ABI unlocks 3-column ifelse-shape closures like `function(c, a, b) if (c > 0) a else b`. Engine adds `VectorTernaryMap` dispatch branch + `combine_ternary_output` helper. Unlocks user closures like `function(x) if (x > 0) x else -x` mapped over a vector. |

**Cumulative engine reduction:** `r2-engine/src/lib.rs` 7282 → ~4860 lines (~33%). Workspace tests: 168 → **233 passing**.

**v0.1.0 shipped.** Every roadmap item from the original Tier 0–4 dependency map is closed except the explicitly out-of-scope items (D&C, GPU, closure JIT, autograd) and the items intentionally bundled into Phase G hardware-awareness work (Oracle calibration).
