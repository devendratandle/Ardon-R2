# Agent task split — Fable 5 vs Opus (saved 2026-07-09)

Purpose: the maintainer runs two Claude agents with different usage windows —
**Fable 5** (long window, ~17–20 h reset; deepest reasoning; has computer-use
screen testing) and **Opus** (~5 h window; excellent focused coder). Divide the
roadmap so each session spends its budget where its strengths pay, and tokens
are never burned re-deriving context.

## Ground rules (both agents)

- Bootstrap from docs, never from old chat: `CLAUDE.md` §8, this file,
  `docs/NEXT-JIT-SESSION.md`, `docs/ARCHITECTURE.md`.
- One scoped task per session. Commit locally at every green checkpoint
  (commits are free; pushing needs the maintainer's explicit OK).
- Debug builds for iteration; release builds only for final measurement, run
  in the background (slow but token-free).
- End every session by updating the relevant resume doc — that is the next
  session's cheap bootstrap.
- GUI visual verification stays on Fable (computer-use screen testing).

## Fable 5 queue — deep, multi-step, high-risk (needs the long window)

1. **J.4 whole-function compilation** — the r2sem gap closer (~0.63 s source
   vs ~0.10 s native is per-op RVal allocation + dispatch across `fit_once`).
   Lower a whole numeric function onto reused buffers/arena calling shared
   kernels. Multi-session; real miscompile risk; scope a vertical slice first.
   (See `docs/NEXT-JIT-SESSION.md` option 2.)
2. **Arrow default-storage migration (F.3)** — switch `RVal::Numeric` storage
   to the staged `Reals` columnar wrapper; ~75 construction sites must be
   updated atomically or the build breaks. Needs one full uninterrupted budget.
3. **Mutable environments** — unblocks `<<-` in closures/counters and the
   closure-state factory pattern. Architectural evaluator change.
4. **CLI plot viewer (REPL blocking / macOS main-thread)** — design + skeleton:
   a small viewer process (reusing the GUI plot-window code) spawned by the
   CLI, plots streamed over IPC. Solves both the blocking-REPL problem and the
   macOS must-own-main-thread constraint, and gives the CLI a native graphics
   device. Fable designs the protocol + process lifecycle; Opus can then fill
   in per-platform polish against the spec.
5. **J.5 tiered dispatch** — profile-driven auto-JIT (engine counts hot
   closures, compiles in background, swaps handles). Design-heavy.
6. **GUI screen-test sweeps** — any visually-verified GUI work (fonts, DPI,
   layout), since computer-use lives here.

## Opus queue — well-scoped, spec-complete, lands inside 5 h

0. **Split closure.rs (1345) and reduce_kernel.rs (1123) per the 500–800 LoC policy** — mechanical, behavior-neutral (normalize passes → normalize.rs; loops/prologue/drivers → kernel_state.rs), all JIT tests must stay green.
1. **Peephole: `t(X) %*% v` → `dgemv_t`** — detect the transpose-then-multiply
   shape in `binary_op` (BinOp::MatMul arm, `crates/r2-engine/src/ops.rs`) and
   call `r2_linalg::dgemv_t` directly instead of materialising `t(X)`. Kernel
   already exists; r2sem hits this shape every iteration. Add a
   bit-exactness test vs the materialised path.
2. **Peephole: cheap `M[rows, ]` row-subset** — avoid the full-matrix copy in
   the common bootstrap-resample pattern (gather rows straight into the
   target buffer). Verify with r2sem timing.
3. **Library speedup pass (near-native modular libraries)** — run `explain()`
   over `devlib/` library code, refactor hot closures into JIT-recognised
   shapes (the J.1–J.4 primitive set), measure before/after. Mechanical and
   self-verifying.
4. **Docs/benchmarks upkeep** — CHANGELOG entries for the accumulated local
   work (JIT phases, dgemm fix, eigen/backsolve/rcond/kappa, GUI fixes),
   FUNCTIONS.md refresh, benchmark snapshot into `benchmarks/history/`.
5. **Package/plugin system polish** — `install.from.dir`/`library()` UX,
   manifest validation, error messages; small engine-side fixes with tests.
6. **Regression-battery expansion** — grow the base-R test scripts; run the
   full suite; fix small correctness gaps it surfaces.

## Sequencing note

Fable items 1–3 are serialized (each wants a fresh full session). Opus items
are independent — any can run while Fable's queue progresses. If an Opus task
uncovers something architectural, it stops and writes a note here instead of
improvising.

## Status snapshot (2026-07-09)

- All work through `0b5c859` LOCAL/UNPUSHED on `main`.
- GUI: selection inversion + long-hours palette + platform fonts (`ef5a325`),
  console snap-to-prompt on output (`0b5c859`) — all verified live on screen.
- JIT: J.1–J.4 scalar/vector-reduction class shipped (source `cor` beats
  native); matrix-multiply investigation ended in the dgemm small/thin
  fast-path fix (`055d9ba`) — see `docs/NEXT-JIT-SESSION.md`.
