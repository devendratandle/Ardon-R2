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
| Oracle | Auto-scheduler | ✅ V1 built | `crates/r2-oracle/` — 5 tests passing. `dispatch(Op, Shape) → Backend{Serial\|Rayon}`. `bi_kmeans` already migrated. `bi_rf`, `bi_gbm`, `bi_cv` still inline; migrate next. V2 adds GPU/Cloud. |
| JIT | Cranelift backend | ◐ scalar + vector reductions + vector maps + composed bodies | `crates/r2-jit/src/lib.rs` + engine call path (Phases C through C.4-full part 2, 16 tests passing). Scalar functions; vector reductions; element-wise `v OP scalar`; element-wise `v OP w`; **generic 1-param composed arithmetic bodies (`(v+1)*2`, `v*v - 1`, etc.) lowered via IR into single fused native loops.** NA propagates via NaN. Next: C.5 matrices, C.6 `.Internal` direct lowering. |
| Rayon | Work-stealing | ✅ Phase D wrapped | Oracle-dispatched: `bi_rf`, `bi_kmeans`, reductions (`sum`/`mean`/`sd`/`var`/`min`/`max`/`prod`/`median`), `bi_cv`, `bi_summary`, **full apply family** (`lapply`/`sapply`/`apply`/`tapply`/`aggregate`) with pure-builtin allowlist (`sum`/`mean`/`sd`/`var`/`min`/`max`/`prod`/`length`/`sqrt`/`abs`/`exp`/`log`/`log2`/`log10`), `bi_gbm` (per-iteration row-work parallel; outer boosting loop sequential by algorithm), **inner tree split-search now parallel across features via `par_for` (v0.0.9 polish — Oracle dispatches per-node based on n_features × active_samples; trees with few features stay serial automatically)**. Closures and non-pure builtins fall back to serial path. `mapply`/`vapply` multi-arg pure-allowlist parallel = V2. |
| GPU | Dispatcher | ✗ not started | — |
| Cloud-RAM | Shards | ✗ not started | — |
| FFI Hub | 200–500 syscalls | ✗ not started | — |
| Memory | R2-ARROW columnar | ◐ F.1+F.2+F.3a+F.4 shipped; F.3 storage migration pending | `crates/r2-arrow/src/lib.rs` (8 reduction + 6 binary tests). **F.4 element-wise kernels now in place**: `binary`, `binary_scalar`, `add`/`sub`/`mul`/`div` shortcut methods. Dense × dense uses tight `for i in 0..n` over `&[f64]` slices for compiler auto-vectorization; sparse path ANDs validity bitmaps so NA propagates correctly. Same semantics as `r2_kernel::binary()` but zero-copy through the columnar representation — ready to be the default arithmetic path once F.3 lands. Previous lib.rs sketch — `ColumnarF64`, null bitmap, dense fast-path reductions. All 7 numeric reductions on columnar view. F.3a: `from_option_slice` + lazy-bitmap. **F.3 attempted but reverted in-session**: changing the `RVal::Numeric` variant to take `Reals` (Vec<Real> + cached `Arc<ColumnarF64>`) requires fixing ~75 construction sites across r2-base/r2-utils/r2-engine; could not complete safely in one session's budget. The `Reals` wrapper type is committed to `r2-types` for the eventual migration. **Verdict**: destructive F.3 needs a dedicated full-budget session — it's the kind of refactor that must finish atomically (every site updated) or the build breaks. F.4: element-wise vector ops on columnar; F.5: mmap-backed columns; F.6: i32/i64/bool/Utf8 dtypes. |
| Microkernel ① | Math kernel | ✅ built | `crates/r2-linalg/` (1,278 LoC, BLAS L1-L3, decompositions) |
| Microkernel ② | BLAS/LAPACK | ◐ partial | LU, Cholesky, QR, SVD, symmetric eigen (`dsyev`), non-symmetric eigen (`dgeev`), triangular solve (`backsolve`/`forwardsolve`), condition number (`rcond`/`kappa`) — done. Pivoting QR, Lanczos, complex type — todo |
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

| Phase | Adds | Unlocks | Status |
|---|---|---|---|
| **J.1** | counted `for(v in a:b)` lowering (real loop-carried phi for induction var + accumulators; ±1 step so ascending & R's descending `a:b` both match the interpreter) | scalar-arithmetic, **math-intrinsic**, and **nested** loops JIT (verified ~4000× on 1e7 iters) | ✅ **shipped** |
| **J.2** | unboxed vector element access in loops. **Bricks shipped:** (a) index-loop *folds* `for(i in 1:length(v)) s <- s <+/*> f(v[i])` route to the tested fused map-reduce codegen (`v[i]`→element) — `sum(v[i])` over 1e6 went *timeout*→0.024 s. (b) **binary map-reduce** `function(x,w) sum(f(x,w))` / `prod(...)` compiles to a fused two-pointer loop (`JitKind::Vector2ToScalar`, ABI `(*const f64,*const f64,i64)->f64`) — dot products / weighted sums `sum(x*w)` now native, verified bit-exact vs the interpreter (167167000 over 1:1000·1000:1). **Remaining:** general `Index{v,i}` stores (e.g. `y[i] <- f(x[i])`) need indexed-*store* codegen + a mutable vector-out ABI. | index-driven numeric kernels JIT | ◐ **in progress** |
| **J.3** | matrices / list access unboxed inside compiled code; guards + deopt for speculated types. **Bricks shipped:** (a) **matrix unboxing** at the JIT dispatch — a matrix's column-major buffer is fed to the existing vector kernels, so `sum(m*m)` / `sqrt(m)` / `A+B` / `sum(A*B)` JIT with zero new codegen (dim + dimnames preserved, bit-exact on 100×100). (b) **real indexed loads** — a new `IrInst::Load` + i64 pointer params let a scalar-returning counted loop reading `x[i]`/`w[i]` (index == the loop var over `1:length`, so provably in-bounds) compile to genuine native loads. This closes the imperative two-vector dot product `for(i in 1:length(x)) s<-s+x[i]*w[i]`, multi-statement folds (`sum((x-w)^2)`), and **scalar recurrences** (Horner) — none of which any map/reduce recogniser covers. Reuses the `Vector1/2ToScalar` ABI/dispatch (len passed as the fused count); empty input & `if`-without-`else` decline to the interpreter for safety. 1e7-elem dot loop: interpreter-timeout → 0.55 s (debug). (c) **indexed stores** — a new `IrInst::Store` + mutable output-pointer param compile `for(i in 1:length(x)) y[i] <- f(x[i][, w[i]])` returning a vector: two-input stores (`y[i]<-x[i]+w[i]`), multi-statement bodies, and `if`/`else`-valued stores (`IndexedStoreMap1/2`, engine allocates the out buffer, input NA bitmap AND-ed onto the result). The single-input pure form keeps its VectorMap path. **Remaining:** 2-D `m[i,j]` address arithmetic (needs nrow + nested-loop bounds), general in-bounds-checked / offset indices (`x[i-1]` recurrences), and list `x[[i]]`. | index-driven numeric kernels JIT | ◐ **in progress** |
| **J.4** | inline hot user + builtin calls into the loop | kills per-call dispatch → **iterative source libraries (r2sem) go near-native** | ◐ **bricks 1–2 shipped.** (1) pure user-helper *inlining* — a function composed of JIT-lowerable helpers (`function(a,b) sq(a)+sq(b)`) is inlined by AST substitution (depth-bounded → recursion falls back). (2) **multi-reduction scalar kernels** — functions that *combine* whole-vector reductions (`sum(x*y)/sum(x*x)` regression coef; `{m<-mean(x); sum((x-m)^2)/(length(x)-1)}` variance; covariance; Pearson `cor`) compile to several **fused loops** with scalar locals threaded between them — no intermediate vector materialised, reusing the `Vector1/2ToScalar` ABI. Verified bit-exact vs R (`var`/`cov`/`cor`). (3) **vector-local fusion + reduction hoisting** — a vector intermediate (`e <- pred-obs`) is *fused* away by substitution, and reductions nested inside expressions (`sum((x-mean(x))^2)`) are hoisted to scalar locals, so the *naive one-liner* forms of `var`/`cov`/`cor`/RMSE/R² JIT with zero intermediate allocation. Demonstrates the "one formula, one implementation" thesis: at small n a JIT'd R2-source `cor` reaches ~88% of the native builtin, and 1.5–3.3× the interpreted path (largest at small n). (4) **CSE + wave fusion** — identical reduction sub-trees (`mean(x)`, `d(x)=x-mean(x)`) are computed *once* and reused across variance/sd/cov/cor (common-subexpression elimination on a canonical key), and independent reductions sharing the same element loads run in a *single* pass with N accumulators (Cranelift GVN folds the shared `x[i]-mx`). This realises the "define the primitive `d` once, reuse everywhere; compute it once" design at the machine level, and matches a native `cor`'s pass structure (means, then `sxy`/`sxx`/`syy` in one pass). **(5) SIMD:** the reduction waves are F64X2-vectorised (2-wide main loop + horizontal reduce + scalar tail), and small integer powers `b^2..b^4` are expanded to multiplication so variance/correlation element exprs vectorise (the `pow` extern would block SIMD). Net: a JIT'd R2-source `cor` now **beats the hand-written native `cor` builtin** — 2.71 s vs 4.01 s at n=1000 (3.9× faster than the pre-SIMD kernel), 2.46 s vs 2.81 s at n=200. The "one formula, written once, reaches *past* native" goal is met for the scalar-reduction class. Transcendental/branchy element exprs fall back to the scalar wave. The single/binary **fused map-reduce** paths (`sum(f(x))`, dot products `sum(x*w)`) are also F64X2-vectorised with **4× unrolling** (4 independent accumulators break the fadd-latency chain, 8 elements/iter) + scalar tail. Payoff is on *compound* element expressions where the interpreter needs many passes: `sum(sqrt(x*x+w*w))` at n=10000 is 2.16 s (one fused SIMD pass) vs 5.37 s interpreted (5 passes) — **2.5×**. For a trivial single `sum(x*w)` the interpreter's LLVM-AVX2 reduce kernel is already optimal (Cranelift emits SSE2), so those tie — the JIT's edge is fusion, not raw lane width. **(6) vector-valued output (matrix/vector-lowering step 1):** a *vector-returning* kernel whose element expression embeds reductions — `d(x)=x-mean(x)`, z-score `(x-mean(x))/sd`, `x/sum(x)` — compiles to a reduction pass (fused waves → scalar locals) + a SIMD map pass writing the output buffer, reusing the `VectorMap`/`VectorBinaryMap` ABI. This establishes non-scalar JIT output (the prerequisite for matrix ops); the centring primitive `d` is now both a standalone native vector kernel and inlined into the scalar var/cov/cor formulas. **Remaining:** true matrix builtins (`%*%`→ the one `dgemm` kernel, `cor(matrix)`, `scale(matrix)`) producing matrix outputs on a scratch arena — the r2sem `fit_once` bottleneck, still the largest brick. **Remaining (the r2sem matrix bottleneck):** lower *matrix* builtins (`%*%`→dgemm, `cor(matrix)`, `scale`) inside a JIT'd region onto unboxed buffers + a scratch arena, so whole functions like `fit_once` compile without per-op RVal allocation. |
| **J.5** | profile-driven tiered dispatch + on-stack replacement | automatic, safe hot/cold tiering | ◐ groundwork: `explain(f)` |
| **J.6** | (optional) trace-based JIT across call boundaries | PyPy-class; the last ~2× | |

Why native builtins still exist: the standard dynamic-language pattern is
*interpreter/JIT for breadth + native kernels for the hot 5%*. First-party
hot kernels (`lm`, `plssem`, …) stay native builtins; J.4 is what lets
*third-party source* libraries reach the same class without engine edits.

### Phase P — parallel `apply` for library code (the *other* speed lever)

Profiling `r2sem` (source library) vs `plssem` (native builtin), both release,
showed a ~25× gap that splits **~equally** into two independent factors:
**~6× serial-vs-parallel bootstrap** and **~4× interpreted-vs-compiled per fit**.
The JIT (Phase J) attacks the ~4×. Phase P attacks the ~6× — **separately, and
at much lower risk**, because it's parallelism, not codegen.

**Deliver:** an `mclapply`/`par.sapply(x, f)` builtin that runs an R2 closure
over the elements of `x` across cores (Rayon lives in `r2-kernel`; this exposes
it to *library code*). Libraries call it for embarrassingly-parallel work —
bootstrap, permutation tests, cross-validation, Monte-Carlo — which is
ubiquitous in statistics.

**Why it's tractable + safe:** the hot types are already `Arc`-based (`RVal`,
`EnvRef`), closure calls create child envs copy-on-write, and the output sink is
thread-local — so each worker evaluates in an isolated context sharing the
registry + global env read-only via `Arc`. No shared mutation ⇒ no data race
⇒ no *silent-wrong-math* class (unlike JIT codegen); any bug surfaces as a
crash/incorrect result caught by tests. Scope: **pure** map closures (no `<<-`
to shared state), matching R's `parallel::` worker isolation.

**Impact:** ~6× for *any* embarrassingly-parallel library, as pure modular
source. `r2sem`: 0.80 s → ~0.13 s (~66× vs cSEM). Composes with the JIT later —
a `par.sapply` of a JIT-compiled closure runs parallel **and** native.

**Recommended order:** **P before J.3/J.4** — bigger single win, lower risk,
broadly applicable, and independent of the delicate codegen work.

**Brick 1 shipped:** `mclapply(x, FUN)` / `par.lapply` — runs a closure over
`x` across cores via `rayon`, returning a list. Unlock: `RVal`/`EnvRef` were
already `Send+Sync`; the only blocker was the JIT-handle cache, fixed by
requiring `JitHandle: Send+Sync` + a justified `unsafe impl` on the (immutable,
reentrant) compiled code. Each element evaluates in an isolated `fork_worker`
engine (shared read-only registry/globals via `Arc`, fresh scratch). Verified:
correct incl. captured variables across threads; **~4.5× on a 6-core interpreted
workload** (10.75 s → 2.40 s). Remaining: `par.sapply` simplification, chunking,
and an opt-in guard against `<<-` in the mapped closure.

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

