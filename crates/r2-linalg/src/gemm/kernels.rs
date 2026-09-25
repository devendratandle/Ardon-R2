//! The CPU checks and the hand-written micro-kernels the blocked GEMM
//! dispatches to: AVX2+FMA f32 (6x16) and f64 (6x8), AVX-512 f32 (12x32).

/// AVX2 + FMA on this CPU (the f32/f64 kernels' wide path).
#[cfg(target_arch = "x86_64")]
pub(super) fn avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}
#[cfg(not(target_arch = "x86_64"))]
pub(super) fn avx2_fma() -> bool { false }

/// AVX-512F on this CPU, and not capped by `R2_SIMD=avx2` / `sse2`.
#[cfg(target_arch = "x86_64")]
pub(super) fn avx512f_allowed() -> bool {
    let cap = std::env::var("R2_SIMD").ok();
    !matches!(cap.as_deref(), Some("avx2") | Some("sse2"))
        && std::arch::is_x86_feature_detected!("avx512f")
}
#[cfg(not(target_arch = "x86_64"))]
pub(super) fn avx512f_allowed() -> bool { false }

/// The f32 micro-kernel, written with AVX2 intrinsics instead of left to
/// the optimiser.
///
/// # Why this exists in a project that avoids `unsafe`
///
/// `micro_impl` is a plain Rust loop and LLVM vectorises it, but it does
/// not *guarantee* the twelve accumulators stay in registers across the
/// whole `kc` depth — and if even one spills, the innermost loop of the
/// whole library starts round-tripping through memory. Measured, that loop
/// reached 56-67 GFLOP/s against roughly 77 of single-core peak: 78%, with
/// the missing fifth exactly where a spill or a missed FMA fusion would
/// put it.
///
/// This is the same thing BLIS does. Its portable kernels are intrinsics,
/// not assembly, for precisely this reason — the register allocation of
/// the inner 6x16 tile is too important to delegate. It stays pure Rust
/// with no C dependency, which is the constraint that matters here
/// (`docs/BLAS_DISPATCH.md`); `unsafe` buys explicit registers, not a
/// foreign library.
///
/// The shape is 6 rows x 16 floats = **12 YMM accumulators**, plus 2 for
/// the B strip and 1 for the broadcast A scalar: 15 of the 16 architectural
/// YMM registers, with one to spare. `RESULTS`' own tile sweep found 12
/// independent FMA chains to be the peak — 4 chains gave 49.8 GFLOP/s, 12
/// gave 77.3, and 16 gave 60.3 because it spills.
///
/// # Safety
///
/// The caller passes packed buffers built by [`super::f32::pack_a`] and
/// [`super::f32::pack_b`], which are sized `panels * kc * MR` and
/// `strips * kc * NR` and always sliced to exactly one panel or strip. The
/// reads below walk `kc * MR` and `kc * NR` elements respectively, in
/// order, from the start of each — the debug assertions pin that. AVX2 and
/// FMA availability is established by the caller's `have_wide()` check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn micro_f32_avx2(kc: usize, apack: &[f32], bpack: &[f32]) -> [[f32; 16]; 6] {
    use std::arch::x86_64::*;
    debug_assert!(apack.len() >= kc * 6);
    debug_assert!(bpack.len() >= kc * 16);

    // Twelve accumulators, named so the register allocator has no choice.
    let (mut c00, mut c01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c20, mut c21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c30, mut c31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c40, mut c41) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut c50, mut c51) = (_mm256_setzero_ps(), _mm256_setzero_ps());

    let mut a = apack.as_ptr();
    let mut b = bpack.as_ptr();
    for _ in 0..kc {
        // One B strip: 16 floats, the two halves of the tile's width.
        let b0 = _mm256_loadu_ps(b);
        let b1 = _mm256_loadu_ps(b.add(8));
        // Each A element broadcasts across the strip. Packing put the six
        // rows of this depth step adjacent, so these six loads are one
        // cache line.
        let a0 = _mm256_set1_ps(*a);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        let a1 = _mm256_set1_ps(*a.add(1));
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        let a2 = _mm256_set1_ps(*a.add(2));
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        let a3 = _mm256_set1_ps(*a.add(3));
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        let a4 = _mm256_set1_ps(*a.add(4));
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        let a5 = _mm256_set1_ps(*a.add(5));
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
        a = a.add(6);
        b = b.add(16);
    }

    let mut out = [[0.0f32; 16]; 6];
    let p = out.as_mut_ptr() as *mut f32;
    _mm256_storeu_ps(p, c00);            _mm256_storeu_ps(p.add(8), c01);
    _mm256_storeu_ps(p.add(16), c10);    _mm256_storeu_ps(p.add(24), c11);
    _mm256_storeu_ps(p.add(32), c20);    _mm256_storeu_ps(p.add(40), c21);
    _mm256_storeu_ps(p.add(48), c30);    _mm256_storeu_ps(p.add(56), c31);
    _mm256_storeu_ps(p.add(64), c40);    _mm256_storeu_ps(p.add(72), c41);
    _mm256_storeu_ps(p.add(80), c50);    _mm256_storeu_ps(p.add(88), c51);
    out
}

/// The f64 micro-kernel: [`micro_f32_avx2`] at half the lanes, 6 rows x
/// 8 doubles = 12 YMM accumulators + 2 for the B strip + 1 broadcast.
///
/// # Safety
///
/// As for [`micro_f32_avx2`]: packed buffers from [`super::f64::pack_a`] /
/// [`super::f64::pack_b`], at least `kc * 6` and `kc * 8` elements, and AVX2+FMA
/// confirmed by the caller's `have_wide()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn micro_f64_avx2(kc: usize, apack: &[f64], bpack: &[f64]) -> [[f64; 8]; 6] {
    use std::arch::x86_64::*;
    debug_assert!(apack.len() >= kc * 6);
    debug_assert!(bpack.len() >= kc * 8);

    let (mut c00, mut c01) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut c10, mut c11) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut c20, mut c21) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut c30, mut c31) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut c40, mut c41) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let (mut c50, mut c51) = (_mm256_setzero_pd(), _mm256_setzero_pd());

    let mut a = apack.as_ptr();
    let mut b = bpack.as_ptr();
    for _ in 0..kc {
        let b0 = _mm256_loadu_pd(b);
        let b1 = _mm256_loadu_pd(b.add(4));
        let a0 = _mm256_set1_pd(*a);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        let a1 = _mm256_set1_pd(*a.add(1));
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        let a2 = _mm256_set1_pd(*a.add(2));
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);
        let a3 = _mm256_set1_pd(*a.add(3));
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);
        let a4 = _mm256_set1_pd(*a.add(4));
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);
        let a5 = _mm256_set1_pd(*a.add(5));
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
        a = a.add(6);
        b = b.add(8);
    }

    let mut out = [[0.0f64; 8]; 6];
    let p = out.as_mut_ptr() as *mut f64;
    _mm256_storeu_pd(p, c00);            _mm256_storeu_pd(p.add(4), c01);
    _mm256_storeu_pd(p.add(8), c10);     _mm256_storeu_pd(p.add(12), c11);
    _mm256_storeu_pd(p.add(16), c20);    _mm256_storeu_pd(p.add(20), c21);
    _mm256_storeu_pd(p.add(24), c30);    _mm256_storeu_pd(p.add(28), c31);
    _mm256_storeu_pd(p.add(32), c40);    _mm256_storeu_pd(p.add(36), c41);
    _mm256_storeu_pd(p.add(40), c50);    _mm256_storeu_pd(p.add(44), c51);
    out
}

/// The f32 micro-kernel for AVX-512: 12 rows x 32 floats.
///
/// The AVX2 kernel's tile is bounded by its 16 YMM registers (12
/// accumulators). AVX-512 doubles the vector width AND the register file
/// to 32 ZMM, so the tile grows in both directions: 32 floats across (two
/// ZMM per row, as the AVX2 kernel has two YMM) and 12 rows down — **24
/// accumulators + 2 for the B strip + 1 broadcast = 27 of 32**, leaving
/// headroom rather than spilling. That is the shape of BLIS's `skx` sgemm
/// kernel (its 32x12, transposed to R2's row-broadcast orientation). Per
/// depth step it issues 24 FMAs against 2 vector loads and 12 broadcasts:
/// twice the FMAs per load of the AVX2 tile, which is what lets two
/// 512-bit FMA ports stay fed.
///
/// The accumulators are an array indexed by constants in fully unrolled
/// loops, which LLVM promotes to registers; the emitted inner loop was
/// checked to contain no stack traffic (`cargo rustc --release -p
/// r2-linalg -- --emit asm`, search `micro_f32_avx512`).
///
/// # Safety
///
/// Packed buffers from [`f32_512::pack_a`] / [`f32_512::pack_b`], at least
/// `kc * 12` and `kc * 32` elements; AVX-512F confirmed by the caller's
/// `f32_512::have_wide()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn micro_f32_avx512(kc: usize, apack: &[f32], bpack: &[f32]) -> [[f32; 32]; 12] {
    use std::arch::x86_64::*;
    debug_assert!(apack.len() >= kc * 12);
    debug_assert!(bpack.len() >= kc * 32);

    let mut c = [[_mm512_setzero_ps(); 2]; 12];
    let mut a = apack.as_ptr();
    let mut b = bpack.as_ptr();
    for _ in 0..kc {
        let b0 = _mm512_loadu_ps(b);
        let b1 = _mm512_loadu_ps(b.add(16));
        for i in 0..12 {
            let ai = _mm512_set1_ps(*a.add(i));
            c[i][0] = _mm512_fmadd_ps(ai, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_ps(ai, b1, c[i][1]);
        }
        a = a.add(12);
        b = b.add(32);
    }

    let mut out = [[0.0f32; 32]; 12];
    for i in 0..12 {
        let p = out[i].as_mut_ptr();
        _mm512_storeu_ps(p, c[i][0]);
        _mm512_storeu_ps(p.add(16), c[i][1]);
    }
    out
}
