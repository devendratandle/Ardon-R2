# Swappable BLAS (runtime DLL dispatch)

R2 is **strictly pure-Rust — no C, no Fortran, no external BLAS.**
This feature does not change that.

`r2-linalg` ships a pure-Rust **reference kernel** that is always
available. From v0.2.1 it can also hand off the hot matrix-multiply
path to a **faster build of the *same pure-Rust kernel*, loaded at
runtime**. The intended variants are R2's own code compiled with
different CPU-SIMD targets:

- `r2_linalg_scalar.dll` — portable, no SIMD assumptions
- `r2_linalg_avx2.dll`   — AVX2 codegen
- `r2_linalg_avx512.dll` — AVX-512 codegen

Same Rust source, different `-C target-cpu` codegen. The installer
picks the one matching the host CPU. Everything stays Rust.

## Why a C ABI (even though both sides are Rust)

Rust has **no stable ABI**. Two *separately compiled* Rust `cdylib`s
cannot reliably call each other through Rust types (`Vec`, `Result`,
custom structs) — memory layouts aren't guaranteed across builds or
compiler versions. So even a Rust→Rust runtime-loaded DLL must cross
the boundary as **plain C**: flat `f64` buffers + integer dimensions.
This is purely an ABI-stability requirement, not a dependency on any
C library.

## The contract

A drop-in variant (one of R2's own CPU-targeted builds) must export
this symbol (C calling convention):

```c
// C(m×n) = alpha · A(m×k) · B(k×n) + beta · C   (column-major)
// returns 0 on success, non-zero on error
int32_t r2_dgemm(size_t m, size_t n, size_t k,
                 double alpha,
                 const double* a, const double* b,
                 double beta, double* c);
```

`r2_linalg.dll` already exports `r2_dgemm` (see
`crates/r2-linalg/src/blas_abi.rs`). A CPU-tuned variant is just R2's
own kernel rebuilt with different SIMD flags, exporting the same
symbol.

## Selecting a variant

Set `R2_BLAS` to the shared library's path before launching R2:

```sh
# Windows (PowerShell)
$env:R2_BLAS = "C:\path\to\r2_linalg_avx2.dll"; R2Gui.exe

# Linux / macOS
R2_BLAS=/path/to/libr2_blas.so  r2 script.r2
```

On the first matrix multiply R2 loads the library once, looks up
`r2_dgemm`, and routes `%*%` through it (it prints
`[r2-linalg] using external BLAS: <path>` to stderr). If `R2_BLAS` is
unset, the library can't be loaded, or the symbol is missing, R2
silently uses its built-in kernel — there is always a working
fallback.

## Roadmap

The intended end state (Phase 4) is **installer-time CPU dispatch**:
the installer detects the host's instruction-set extensions and drops
the right `r2_linalg_avx2.dll` / `_avx512.dll` / `_scalar.dll` next to
the executables, setting `R2_BLAS` automatically — so users get
CPU-optimal performance with no manual step. The v0.2.1 work puts the
load-and-dispatch mechanism in place; the optimized variants and the
installer wiring come next.
