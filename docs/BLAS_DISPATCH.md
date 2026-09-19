# Kernel dispatch: runtime SIMD selection in one binary

R2 is **strictly pure Rust — no C, no Fortran, no external BLAS.** Every
hot kernel in `r2-linalg`, `r2-tensor`, `r2-autograd` and `r2-train`
asks the CPU once, at first use, which instruction sets it has
(`is_x86_feature_detected!`) and branches to the matching build of the
same Rust source: a portable baseline, AVX2+FMA, or AVX-512. One
executable runs on every x86-64 machine and uses the widest vectors
that machine offers.

```rust
match simd_tier() {
    SimdTier::Avx512 => unsafe { macro_kernel_avx512(...) },
    SimdTier::Avx2   => unsafe { macro_kernel_avx2(...) },
    SimdTier::Scalar => macro_kernel_scalar(...),
}
```

The `unsafe` sits at that gate and nowhere else: calling a
`#[target_feature]` function is the one step the compiler cannot verify
(it cannot know the branch taken matches the feature the callee
requires), so the gate carries the proof in a `SAFETY` comment. The
kernels themselves are ordinary Rust.

## History

Until v0.4.0 this file described a runtime-loaded, per-CPU kernel
library selected by an installer (`R2_BLAS`). That mechanism and every
other plan to ship R2 as separately loaded libraries were removed in
v0.4.0: R2 is a single executable, and CPU selection happens inside it
as described above. The measured reason is in `CHANGELOG.md`.
