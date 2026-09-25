//! A blocked, packed GEMM — the Goto/BLIS structure, in safe Rust.
//!
//! This is BLAS `sgemm`, and it lives in the BLAS crate. `r2-tensor`'s
//! `ops::matmul` and the autograd tape's two gradient cases all call it,
//! so the LLM path has exactly one matrix-multiply implementation.
//!
//! The kernel below is a per-type macro, instantiated for f32 (the LLM
//! path) and f64 (`level3::dgemm`, the column-major BLAS entry point the
//! statistics path calls — `%*%` and the blocked LAPACK routines).
//!
//! # Why it exists
//!
//! `r2_tensor::ops::matmul` was a plain i-k-j triple loop, and
//! `level3::dgemm`'s packed path was no closer to its own ceiling.
//! Measured on the shapes an LLM step actually runs:
//!
//! ```text
//!                       GFLOP/s        % of that type's peak
//!   naive f32 loop       27 -  73          6 - 16
//!   level3 dgemm f64     11 -  28          5 - 12
//!   PyTorch (MKL)       171 - 286         37 - 62
//!   this kernel f32     118 - 195         26 - 42
//! ```
//!
//! MKL is hand-written assembly, which R2 will not ship — it is pure Rust
//! by design (`docs/BLAS_DISPATCH.md`). But **JAX is not MKL**: XLA:CPU
//! sends `dot` to Eigen (`__xla_cpu_runtime_EigenBatchMatMulF32`), which is
//! portable C++ templates with no assembly, and Eigen matches or beats MKL
//! on these shapes. The gap was never assembly. It is structure:
//!
//! 1. **Pack** A and B into contiguous, micro-kernel-ordered buffers, so
//!    the inner loop never strides and never TLB-misses.
//! 2. **Block** for three cache levels: a B panel sized for L3, an A block
//!    for L2, a micro-tile for registers.
//! 3. A **register micro-kernel** whose accumulators stay in registers for
//!    the whole depth of a panel.
//!
//! # The constants
//!
//! `MR = 6` rows and `NR = 16` columns — two AVX2 registers' worth of
//! floats. That is 12 accumulator registers, leaving 4 of the 16
//! architectural YMM for operands. `REPORT.md` measured this directly:
//! 4 independent FMA chains gave 49.8 GFLOP/s, 12 gave 77.3, and 16 gave
//! 60.3 because it spills. 12 is the peak and 6 x 2-vectors is how you get
//! it.
//!
//! `KC = 256`, `MC = 96`, `NC = 1024`: an A block of 96 KB for the 512 KB
//! L2, a B panel of 1 MB for the 8 MB L3.
//! The naive kernel fell from 63-73 to 26.8 GFLOP/s at exactly the shape
//! where B stopped fitting L3 (256x8000x4B = 8 MB); that cliff is what the
//! blocking removes.
//!
//! # Transposes are free
//!
//! `grad_A = g·Bᵀ` and `grad_B = Aᵀ·g` are the NT and TN cases. A packing
//! pass already reads every element and writes it elsewhere, so it can
//! read transposed at no cost — which is why [`Trans`] is an argument to
//! the packers rather than a separate materialising pass. `REPORT.md`
//! records materialising them instead as a REJECTED attempt: 393 ms
//! against 326 ms.
//!
//! # Vector width is chosen at RUNTIME
//!
//! The workspace sets no `target-cpu`, so everything compiles for baseline
//! x86-64: SSE2, and **no FMA at all** — every multiply-add is two
//! instructions. This kernel measured 29-59 GFLOP/s that way and 118-195
//! with AVX2+FMA. The build cannot simply enable AVX2 globally, because
//! the installer ships one binary to machines that may not have it and an
//! illegal instruction on a user's CPU is not a trade for throughput. So
//! the wide kernel is selected by `is_x86_feature_detected!`, which is what
//! MKL and Eigen do, and what `docs/BLAS_DISPATCH.md` describes as this
//! project's architecture — resolved in-process, one binary.
//!
//! Note that packing matters MORE with AVX2, not less: it took the naive
//! loop from 27-73 to 28-83 GFLOP/s (+30%) and this kernel from 29-59 to
//! 118-195 (+250%). Wide vector units only help when something is feeding
//! them contiguous data.

/// Whether an operand is stored transposed relative to the `m x k`,
/// `k x n` row-major convention.
///
/// `No` means the natural layout; `Yes` means the buffer holds the
/// transpose and the packer reads it strided. Nothing is materialised.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trans {
    No,
    Yes,
}

/// Worker count, resolved once. `rayon::current_num_threads()` is a
/// thread-pool query, and a training step issues thousands of GEMMs.
fn nthreads() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(rayon::current_num_threads)
}

impl Trans {
    /// The same buffer, read the other way round. Used to restate
    /// `C = A·B` as `Cᵀ = Bᵀ·Aᵀ` without touching any data.
    #[inline]
    fn flip(self) -> Trans {
        match self { Trans::No => Trans::Yes, Trans::Yes => Trans::No }
    }
}

/// Element count above which the packing passes thread. Lower than the
/// compute threshold: packing is a pure map, so the fork-join is the only
/// cost and there is no reassociation to worry about.
const PACK_PAR_MIN: usize = 1 << 13;

/// Per-call timing of the small GEMMs inside a real step, for
/// `--example gemm_insitu`. Off unless `R2_GEMM_STATS=1`; then every
/// parallel call records (shape key, microseconds). This
/// is the measurement that closed the small-shape question: op-level
/// timings with hot operands do not predict a step.
fn stats_on() -> bool {
    use std::sync::OnceLock;
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("R2_GEMM_STATS").map(|v| v == "1").unwrap_or(false))
}
static STATS: std::sync::Mutex<Vec<(u64, f32, bool)>> = std::sync::Mutex::new(Vec::new());

/// Drain the recorded per-call timings: `(m<<40 | k<<20 | n, microseconds, team)`.
pub fn take_gemm_stats() -> Vec<(u64, f32, bool)> {
    std::mem::take(&mut *STATS.lock().unwrap())
}


#[macro_use]
mod blocked;

/// The f32 kernel. A block 96x256x4B = 96 KB (L2), B panel
/// 256x1024x4B = 1 MB (L3).
pub mod f32 {
    use super::{nthreads, stats_on, Trans, PACK_PAR_MIN, STATS};
    blocked_gemm_for!(f32, 6, 16, 256, 96, 1024, "fma", super::avx2_fma());
}

/// The f64 kernel: the same structure at half the lanes. NR = 8 doubles is
/// two YMM registers, so the 6x8 tile is again 12 accumulators. The byte
/// sizes match the f32 kernel's: a 16 KB B strip and a 12 KB A panel per
/// tile (L1), an A block of 96x256x8B = 192 KB (L2), a B panel of
/// 256x512x8B = 1 MB (L3). `level3::dgemm` — `%*%` and the blocked LAPACK
/// routines — runs on this.
pub mod f64 {
    use super::{nthreads, stats_on, Trans, PACK_PAR_MIN, STATS};
    blocked_gemm_for!(f64, 6, 8, 256, 96, 512, "fma", super::avx2_fma());
}

/// The f32 kernel at the AVX-512 tile: 12 rows x 32 floats, 24 of the 32
/// ZMM registers as accumulators ([`micro_f32_avx512`]). Everything else —
/// packing, blocking, the partitions — is the same macro as [`f32`], so
/// each element of C sums its products in the same order with the same
/// fused multiply-adds, and the result is bit-identical to the AVX2
/// kernel's (`avx512_is_bit_identical_to_avx2`). A block of 96 rows is 8
/// panels here instead of 16.
///
/// `sgemm*` route here when [`f32_512::have_wide`] — AVX-512F on this CPU,
/// and `R2_SIMD` not set to `avx2` (the same cap `level3`'s `dot4` honours,
/// for a machine where AVX-512 downclocks the rest of the workload).
///
/// NOT yet tuned or timed: the machines R2 is developed on have no
/// AVX-512. It is checked for correctness under Intel SDE; `KC`/`MC`/`NC`
/// are the AVX2 kernel's until an AVX-512 machine measures them.
pub mod f32_512 {
    use super::{nthreads, stats_on, Trans, PACK_PAR_MIN, STATS};
    blocked_gemm_for!(f32, 12, 32, 256, 96, 1024, "avx512f", super::avx512f_allowed());
}

mod kernels;
use kernels::{avx2_fma, avx512f_allowed};
#[cfg(target_arch = "x86_64")]
use kernels::{micro_f32_avx2, micro_f32_avx512, micro_f64_avx2};

/// BLAS `sgemm`, row-major: `C = A·B` in single precision.
///
/// The one routine an LLM needs. The forward pass is NN, `grad_A = g·Bᵀ`
/// is NT, and `grad_B = Aᵀ·g` is TN.
pub fn sgemm(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
             m: usize, k: usize, n: usize, parallel: bool) -> Vec<f32> {
    if f32_512::have_wide() { return self::f32_512::gemm(a, ta, b, tb, m, k, n, parallel); }
    self::f32::gemm(a, ta, b, tb, m, k, n, parallel)
}

/// BLAS `sgemm` accumulating into an existing `C`.
pub fn sgemm_into(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
                  m: usize, k: usize, n: usize, c: &mut [f32], parallel: bool) {
    if f32_512::have_wide() { return self::f32_512::gemm_into(a, ta, b, tb, m, k, n, c, parallel); }
    self::f32::gemm_into(a, ta, b, tb, m, k, n, c, parallel)
}

/// BLAS `sgemm` writing `C = A·B` into an existing buffer whose prior
/// contents are ignored (no zeroing required).
pub fn sgemm_assign_into(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
                         m: usize, k: usize, n: usize, c: &mut [f32], parallel: bool) {
    if f32_512::have_wide() { return self::f32_512::gemm_assign_into(a, ta, b, tb, m, k, n, c, parallel); }
    self::f32::gemm_assign_into(a, ta, b, tb, m, k, n, c, parallel)
}

// The f64 instantiation above is reached through `level3::dgemm`, the
// column-major BLAS entry point: a column-major `C = A·B` is the row-major
// `Cᵀ = Bᵀ·Aᵀ`, i.e. this kernel with the operands swapped and no
// transposes materialised at all.

#[cfg(test)]
mod tests;
