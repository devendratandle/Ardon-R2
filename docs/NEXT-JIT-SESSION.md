# Resume point — after GUI work (saved 2026-07-08)

Context: the "matrix-multiply JIT brick" investigation ended with a finding —
matmul is already native BLAS through the interpreter, so JIT-compiling `%*%`
gives ~zero speedup. The real cost was a dgemm routing bug, fixed in `055d9ba`
(small/thin fast-path; 5×5 `%*%` ~34× faster, r2sem ~0.75s → ~0.63s).

The remaining r2sem gap (~0.63s source vs ~0.10s native) is NOT matmul — it is
broad interpreter overhead across `fit_once`: the `rawmat[rows,]` subset copy,
`scale`, `cbind`, `cor`, and per-op RVal allocations.

## The three options (pick one when resuming)

1. **Peepholes (small, safe, r2sem-relevant)** — in the spirit of the dgemm fix:
   - `t(X) %*% v` → transposed gemv (`dgemv_t` already exists in r2-linalg)
     without materialising the transpose. r2sem uses `t(blk) %*% inner` every
     iteration; today it allocates a 7×1000 transpose per call.
   - Cheaper `M[rows,]` row-subset (avoid the full copy where possible).

2. **Whole-function compilation (large, the only full-gap closer)** — lower an
   entire numeric function like `fit_once` onto reused buffers/arena, calling
   the shared kernels (dgemm, cor, scale) with no per-op RVal allocation.
   **Slices 1–3 SHIPPED (3e6c627, 622f956, 80cc6f8):** iterative kernels —
   whole functions with `for (1:K)` / `while (cond)` loops carrying SCALAR
   and VECTOR state across per-iteration reductions and fused map updates
   (gradient descent, Newton, EM, fixed-point, residual/shrinkage
   iterations), plus adaptive scalar `if/else` values. Carried vectors live
   in scratch buffers (alloc once per call via __r2_scratch_alloc/free —
   no ABI change). 3.1× release / 8.4× debug scalar-state; **64× debug** on
   vector-state (interpreter allocates 3–4 vectors per iteration, kernel
   none); bit-close incl. the `1:0` edge; read-before-define declines
   safely. **Remaining:** matrix-valued state inside compiled functions
   (`%*%`, `cor(matrix)`, `scale(matrix)` on the scratch arena) — the
   r2sem `fit_once` bottleneck and the final J.4 piece.

3. **Call matrix work done** — matmul is now native-fast everywhere; move to
   another phase (J.5 tiered dispatch, Arrow default-storage migration, etc.).

## State at save time
- All work LOCAL/UNPUSHED on `main` through `055d9ba`.
- JIT phase status is current in `docs/ARCHITECTURE.md` (Phase J table);
  implementation detail archived in `code-history/phase-j-jit-detail.md`.
- `target/debug` was cleared for disk space — first debug build recompiles
  deps (~5–8 min). Release binary is current at `target/release/r2.exe`.
