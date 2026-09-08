//! A blocked, packed GEMM — the Goto/BLIS structure, in safe Rust.
//!
//! This is BLAS `sgemm`, and it lives in the BLAS crate. `r2-tensor`'s
//! `ops::matmul` and the autograd tape's two gradient cases all call it,
//! so the LLM path has exactly one matrix-multiply implementation.
//!
//! `level3::dgemm` is NOT touched. It is the column-major BLAS entry point
//! the statistics path calls, with its own fast path and its own tests.
//! The kernel below is written as a per-type macro so that instantiating
//! it for f64 later is a one-line change.
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
//! MKL and Eigen do, and what `docs/BLAS_DISPATCH.md` already describes as
//! this project's intended architecture — resolved in-process rather than
//! through a DLL.
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

/// Rows per register tile. See the module docs.
const MR: usize = 6;

/// Generate a complete blocked GEMM for one element type.
///
/// A macro rather than a trait: the micro-kernel has to be monomorphic for
/// the accumulator array to stay in registers, and `#[target_feature]`
/// cannot be applied through a generic. Rust has no numeric trait in std,
/// and pulling in `num-traits` would add a dependency to a crate that
/// deliberately has almost none. Only f32 is instantiated today; the macro
/// is what makes adding f64 a one-line change.
macro_rules! blocked_gemm_for {
    ($ty:ty, $nr:expr, $kc:expr, $mc:expr, $nc:expr, $feat:literal) => {
        /// Columns per register tile: two vector registers' worth.
        const NR: usize = $nr;
        /// Depth of one packed panel.
        const KC: usize = $kc;
        /// Rows of A per L2-resident block.
        const MC: usize = $mc;
        /// Columns of B per L3-resident panel.
        const NC_DEFAULT: usize = $nc;
        /// Panel width. Overridable via `R2_GEMM_NC` for the sweep in
        /// `--example gemm_nc_sweep`; the default is what ships.
        fn nc_width() -> usize {
            use std::sync::OnceLock;
            static W: OnceLock<usize> = OnceLock::new();
            *W.get_or_init(|| {
                std::env::var("R2_GEMM_NC").ok()
                    .and_then(|v| v.parse().ok())
                    .map(|v: usize| v.max(NR))
                    .unwrap_or(NC_DEFAULT)
            })
        }

        /// One `MR x NR` register tile, accumulated over `kc`.
        ///
        /// The accumulator is a LOCAL returned by value, never an
        /// out-parameter: behind a `&mut` the compiler must treat it as
        /// memory and spills every iteration, which is the difference
        /// between a register kernel and a memory kernel.
        #[inline(always)]
        fn micro_impl(kc: usize, apack: &[$ty], bpack: &[$ty]) -> [[$ty; NR]; MR] {
            let mut acc = [[0 as $ty; NR]; MR];
            for p in 0..kc {
                let b = &bpack[p * NR..p * NR + NR];
                let a = &apack[p * MR..p * MR + MR];
                for i in 0..MR {
                    let av = a[i];
                    let ai = &mut acc[i];
                    for j in 0..NR {
                        ai[j] += av * b[j];
                    }
                }
            }
            acc
        }

        /// The same source, compiled for AVX2 + FMA. Kept as the fallback
        /// for element types with no hand-written kernel.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2", enable = $feat)]
        #[allow(dead_code)]
        unsafe fn micro_wide(kc: usize, apack: &[$ty], bpack: &[$ty]) -> [[$ty; NR]; MR] {
            micro_impl(kc, apack, bpack)
        }

        /// Resolved once per process.
        #[inline]
        fn have_wide() -> bool {
            #[cfg(target_arch = "x86_64")]
            {
                use std::sync::OnceLock;
                static OK: OnceLock<bool> = OnceLock::new();
                *OK.get_or_init(|| {
                    std::arch::is_x86_feature_detected!("avx2")
                        && std::arch::is_x86_feature_detected!("fma")
                })
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        }

        /// Dispatch. The branch is per TILE, not per element.
        #[inline(always)]
        fn micro(kc: usize, apack: &[$ty], bpack: &[$ty], wide: bool) -> [[$ty; NR]; MR] {
            #[cfg(target_arch = "x86_64")]
            if wide {
                // SAFETY: `wide` is only true when `have_wide()` confirmed
                // AVX2+FMA on this CPU, and the packed buffers are sized by
                // `pack_a`/`pack_b` as the kernel's own safety note requires.
                //
                // MR x NR is 6 x 16 for f32, which is what the hand-written
                // kernel implements; the transmutes are identity casts that
                // let one macro body serve a type with a bespoke kernel and
                // a type without. Any other instantiation falls back to the
                // compiler-vectorised body.
                if std::mem::size_of::<$ty>() == 4 && MR == 6 && NR == 16 {
                    return unsafe {
                        let a: &[f32] = core::slice::from_raw_parts(
                            apack.as_ptr() as *const f32, apack.len());
                        let b: &[f32] = core::slice::from_raw_parts(
                            bpack.as_ptr() as *const f32, bpack.len());
                        let r = super::micro_f32_avx2(kc, a, b);
                        core::ptr::read(&r as *const _ as *const [[$ty; NR]; MR])
                    };
                }
                return unsafe { micro_wide(kc, apack, bpack) };
            }
            let _ = wide;
            micro_impl(kc, apack, bpack)
        }

        /// Pack a `kc x nc` slab of B into `NR`-wide strips, zero-padded.
        ///
        /// THREADED, and it matters more than it looks. This runs OUTSIDE
        /// the parallel region — the panel is packed once and then read by
        /// every row-block — so while it was serial it was pure Amdahl
        /// drag. This machine's parallel ceiling is 5.53x on six cores
        /// (`--example scaling_ceiling`, 92% efficient on register-resident
        /// work), but `sgemm` was reaching only 1.9-2.8x. A serial fraction
        /// of ~22% predicts 1/(0.22 + 0.78/6) = 2.85x, which is what was
        /// measured — the arithmetic pointed here before any code changed.
        ///
        /// Strips write disjoint slices of `out` and each output element
        /// comes from exactly one input element, so this is bit-identical
        /// however it is split.
        fn pack_b(b: &[$ty], k: usize, n: usize, trans: Trans,
                  pc: usize, kc: usize, jc: usize, nc: usize, out: &mut Vec<$ty>) {
            let strips = nc.div_ceil(NR);
            out.clear();
            out.resize(strips * kc * NR, 0 as $ty);
            let fill = |s: usize, strip: &mut [$ty]| {
                let j0 = jc + s * NR;
                for p in 0..kc {
                    let row = p * NR;
                    for jj in 0..NR {
                        let j = j0 + jj;
                        if j < jc + nc && j < n {
                            strip[row + jj] = match trans {
                                Trans::No => b[(pc + p) * n + j],
                                Trans::Yes => b[j * k + (pc + p)],
                            };
                        }
                    }
                }
            };
            if strips * kc * NR >= PACK_PAR_MIN {
                use rayon::prelude::*;
                out.par_chunks_mut(kc * NR).enumerate().for_each(|(s, st)| fill(s, st));
            } else {
                out.chunks_mut(kc * NR).enumerate().for_each(|(s, st)| fill(s, st));
            }
        }

        /// Pack an `mc x kc` block of A into `MR`-tall panels, zero-padded.
        fn pack_a(a: &[$ty], k: usize, m: usize, trans: Trans,
                  ic: usize, mc: usize, pc: usize, kc: usize, out: &mut Vec<$ty>) {
            let panels = mc.div_ceil(MR);
            out.clear();
            out.resize(panels * kc * MR, 0 as $ty);
            for pnl in 0..panels {
                let i0 = ic + pnl * MR;
                let base = pnl * kc * MR;
                for p in 0..kc {
                    let row = base + p * MR;
                    for ii in 0..MR {
                        let i = i0 + ii;
                        if i < ic + mc && i < m {
                            out[row + ii] = match trans {
                                Trans::No => a[i * k + (pc + p)],
                                Trans::Yes => a[(pc + p) * m + i],
                            };
                        }
                    }
                }
            }
        }

        /// `C += A·B`, row-major, accumulating into an existing buffer.
        ///
        /// Accumulating rather than assigning is what lets a caller fold a
        /// gradient in without a temporary.
        pub fn gemm_into(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                         m: usize, k: usize, n: usize, c: &mut [$ty], parallel: bool) {
            debug_assert_eq!(a.len(), m * k);
            debug_assert_eq!(b.len(), k * n);
            debug_assert_eq!(c.len(), m * n);
            if m == 0 || n == 0 || k == 0 {
                return;
            }
            use rayon::prelude::*;

            // ── short-M path ───────────────────────────────────────────
            //
            // Parallelism below runs over ROW-BLOCKS of C, so a short `m`
            // starves it however good the kernel is. That is not a corner
            // case, it is `grad_B = Aᵀ·g`, whose M is the weight's INPUT
            // dim: 256 against the forward's 2,048. Measured with
            // `--example gemm_scaling`, all three transpose cases run at
            // 56-64 GFLOP/s on ONE thread — the micro-kernel does not care
            // about the transpose at all — and then diverge entirely by M
            // once threaded:
            //
            //     case   M      1 thread   6 threads   scale
            //     NN     2048    55.9 GF    150.5 GF   2.69x
            //     NT     2048    60.6       148.0      2.44x
            //     TN      256    58.4        68.3      1.17x
            //     TN      768    62.9       167.3      2.66x   (ffn w2)
            //
            // The last row is the same explanation confirming itself: three
            // times the M, and most of the deficit goes away.
            //
            // The fix is to give the threaded loop a long dimension, and
            // `Cᵀ = Bᵀ·Aᵀ` does exactly that — it swaps M and N, so the
            // 8,000 becomes the row count and the 256 becomes the columns.
            // Neither operand moves: a transpose of an operand is just the
            // other `Trans` flag, which the packing pass honours for free.
            // Only the RESULT has to be turned back, one `m x n` pass
            // against an `m*k*n` multiply.
            //
            // Shrinking MC instead was tried and rejected: every row-block
            // re-streams the whole packed B panel, and for this case B is
            // the large operand, so more blocks made it worse (148 -> 159
            // ms).
            let blocks = m.div_ceil(MC);
            if parallel && blocks < nthreads() && n > m && m * n > 0 {
                let ct = gemm(b, tb.flip(), a, ta.flip(), n, k, m, true);
                // Transposed accumulate, tiled so neither side strides for
                // more than a tile: a flat i-j loop would stride one of them
                // by a whole row on every element.
                const T: usize = 32;
                for i0 in (0..m).step_by(T) {
                    let ih = T.min(m - i0);
                    for j0 in (0..n).step_by(T) {
                        let jh = T.min(n - j0);
                        for i in i0..i0 + ih {
                            for j in j0..j0 + jh {
                                c[i * n + j] += ct[j * m + i];
                            }
                        }
                    }
                }
                return;
            }

            let wide = have_wide();

            // Parallelism runs over ROW-BLOCKS of C, so a small `m`
            // starves it. That is not a corner case, it is `grad_B`:
            // `grad_B = Aᵀ·g` has M = the weight's INPUT dim (256, 768),
            // not the token count, so at MC = 96 it gets 3 blocks on 6
            // cores while `grad_A` gets 22. Measured, grad_B was the
            // slower of the two despite identical FLOPs — 148 ms against
            // 117 on the output-head shape.
            //
            // Shrink the block until there is at least one per worker,
            // never below one register tile. This trades a little L2
            // blocking quality for the other half of the machine, and it
            // only ever engages when `m` is too small to fill the cores —
            // at m = 2048 the arithmetic returns MC unchanged.
            let mc_blk = if parallel {
                let want = m.div_ceil(nthreads())
                    .next_multiple_of(MR).clamp(MR, MC);
                // ...but shrinking is not free. EVERY row-block streams the
                // whole packed B panel, so halving the block doubles the
                // traffic over it. That is a good trade when B is small and
                // a bad one when B is the large operand — which is exactly
                // `grad_B = Aᵀ·g`, where B is the upstream gradient (64 MB
                // on the output-head shape). Measured: shrinking there took
                // grad_B from 148 ms to 159, while it took grad_A from 117
                // to 95 and the forward from 201 to 159.
                //
                // So shrink only while the panel still fits L2 and
                // re-streaming it is cheap.
                let panel_bytes = KC.min(k) * nc_width().min(n) * std::mem::size_of::<$ty>();
                if panel_bytes <= (1 << 19) { want } else { MC }
            } else {
                MC
            };
            let mut bpack: Vec<$ty> = Vec::new();
            // ── L3 loop: a panel of B's columns ────────────────────────
            let ncw = nc_width();
            for jc in (0..n).step_by(ncw) {
                let nc = ncw.min(n - jc);
                // ── depth loop: a KC-deep slab ─────────────────────────
                for pc in (0..k).step_by(KC) {
                    let kc = KC.min(k - pc);
                    // Packed ONCE, then read by every row-block below and
                    // by every worker — hence packed outside the parallel
                    // region and shared immutably.
                    pack_b(b, k, n, tb, pc, kc, jc, nc, &mut bpack);
                    let bp = &bpack[..];

                    // ── L2 loop: a block of A's rows. Row-blocks of C are
                    // disjoint, so this is where the parallelism goes.
                    let block = |ic: usize, cband: &mut [$ty]| {
                        let mc = mc_blk.min(m - ic);
                        let mut apack: Vec<$ty> = Vec::new();
                        pack_a(a, k, m, ta, ic, mc, pc, kc, &mut apack);
                        let panels = mc.div_ceil(MR);
                        let strips = nc.div_ceil(NR);
                        for pnl in 0..panels {
                            let ap = &apack[pnl * kc * MR..(pnl + 1) * kc * MR];
                            let rows = MR.min(mc - pnl * MR);
                            for s in 0..strips {
                                let bpp = &bp[s * kc * NR..(s + 1) * kc * NR];
                                let acc = micro(kc, ap, bpp, wide);
                                // Write back only the live part; the rest
                                // of the tile is zero padding that must
                                // not reach C.
                                let cols = NR.min(nc - s * NR);
                                for ii in 0..rows {
                                    let crow = (pnl * MR + ii) * n + jc + s * NR;
                                    let dst = &mut cband[crow..crow + cols];
                                    for (d, v) in dst.iter_mut().zip(&acc[ii][..cols]) {
                                        *d += v;
                                    }
                                }
                            }
                        }
                    };

                    if parallel && m > mc_blk {
                        c.par_chunks_mut(mc_blk * n).enumerate()
                            .for_each(|(bi, cband)| block(bi * mc_blk, cband));
                    } else {
                        c.chunks_mut(mc_blk * n).enumerate()
                            .for_each(|(bi, cband)| block(bi * mc_blk, cband));
                    }
                }
            }
        }

        /// `C = A·B`, row-major.
        pub fn gemm(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                    m: usize, k: usize, n: usize, parallel: bool) -> Vec<$ty> {
            let mut c = vec![0 as $ty; m * n];
            gemm_into(a, ta, b, tb, m, k, n, &mut c, parallel);
            c
        }
    };
}

/// The f32 kernel. A block 96x256x4B = 96 KB (L2), B panel
/// 256x1024x4B = 1 MB (L3).
pub mod f32 {
    use super::{nthreads, Trans, MR, PACK_PAR_MIN};
    blocked_gemm_for!(f32, 16, 256, 96, 1024, "fma");
}

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
/// The caller passes packed buffers built by [`f32::pack_a`] and
/// [`f32::pack_b`], which are sized `panels * kc * MR` and
/// `strips * kc * NR` and always sliced to exactly one panel or strip. The
/// reads below walk `kc * MR` and `kc * NR` elements respectively, in
/// order, from the start of each — the debug assertions pin that. AVX2 and
/// FMA availability is established by the caller's `have_wide()` check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn micro_f32_avx2(kc: usize, apack: &[f32], bpack: &[f32]) -> [[f32; 16]; 6] {
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

/// BLAS `sgemm`, row-major: `C = A·B` in single precision.
///
/// The one routine an LLM needs. The forward pass is NN, `grad_A = g·Bᵀ`
/// is NT, and `grad_B = Aᵀ·g` is TN.
pub fn sgemm(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
             m: usize, k: usize, n: usize, parallel: bool) -> Vec<f32> {
    self::f32::gemm(a, ta, b, tb, m, k, n, parallel)
}

/// BLAS `sgemm` accumulating into an existing `C`.
pub fn sgemm_into(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
                  m: usize, k: usize, n: usize, c: &mut [f32], parallel: bool) {
    self::f32::gemm_into(a, ta, b, tb, m, k, n, c, parallel)
}

// NOT instantiated for f64, deliberately. `level3::dgemm` is the
// column-major BLAS entry point the statistics path already calls, it has
// its own small-shape fast path and its own tests, and changing it is a
// separate piece of work with a separate risk. The macro above is written
// per-type precisely so that instantiation is a one-line change when that
// work is scheduled — a column-major `C = A·B` is the row-major
// `Cᵀ = Bᵀ·Aᵀ`, i.e. this kernel with the operands swapped and no
// transposes materialised at all.

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, written out. Everything above is an optimisation of
    /// exactly this and must agree with it.
    fn naive(a: &[f32], ta: Trans, b: &[f32], tb: Trans,
             m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    let av = match ta { Trans::No => a[i * k + p], Trans::Yes => a[p * m + i] };
                    let bv = match tb { Trans::No => b[p * n + j], Trans::Yes => b[j * k + p] };
                    s += av * bv;
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn mk(n: usize, ph: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.031 + ph).sin()).collect()
    }

    /// Every transpose combination, at sizes that exercise the edge
    /// handling: dimensions that are not multiples of MR (6), NR (16) or
    /// KC (256) must still be exact.
    #[test]
    fn sgemm_matches_the_definition() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (6, 16, 1),          // exactly one tile
            (7, 17, 19),         // prime-ish, every edge ragged
            (13, 260, 33),       // k > KC, so the depth loop runs twice
            (96, 40, 1030),      // n > NC, so the panel loop runs twice
            (200, 70, 50),       // m > MC, so the row loop blocks
        ] {
            for &ta in &[Trans::No, Trans::Yes] {
                for &tb in &[Trans::No, Trans::Yes] {
                    let a = mk(m * k, 0.0);
                    let b = mk(k * n, 1.7);
                    let want = naive(&a, ta, &b, tb, m, k, n);
                    for &par in &[false, true] {
                        let got = sgemm(&a, ta, &b, tb, m, k, n, par);
                        for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                            assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()),
                                    "{m}x{k}x{n} ta={ta:?} tb={tb:?} par={par} \
                                     element {i}: {x} vs {y}");
                        }
                    }
                }
            }
        }
    }

    /// Threading must not change the answer. Row-blocks are disjoint and
    /// each accumulates its depth in the same order, so this is exact
    /// equality, not a tolerance.
    #[test]
    fn sgemm_is_thread_count_independent() {
        let (m, k, n) = (300, 300, 300);
        let a = mk(m * k, 0.3);
        let b = mk(k * n, 2.1);
        assert_eq!(sgemm(&a, Trans::No, &b, Trans::No, m, k, n, false),
                   sgemm(&a, Trans::No, &b, Trans::No, m, k, n, true),
                   "serial and parallel sgemm disagree");
    }

    /// `sgemm_into` ACCUMULATES. A caller folding a gradient in relies on
    /// it, and an assigning version would silently drop the other term.
    #[test]
    fn sgemm_into_accumulates() {
        let (m, k, n) = (20, 30, 40);
        let a = mk(m * k, 0.5);
        let b = mk(k * n, 1.1);
        let once = sgemm(&a, Trans::No, &b, Trans::No, m, k, n, false);
        let mut twice = once.clone();
        sgemm_into(&a, Trans::No, &b, Trans::No, m, k, n, &mut twice, false);
        for (t, o) in twice.iter().zip(&once) {
            assert!((t - 2.0 * o).abs() <= 1e-4 * (1.0 + o.abs()),
                    "sgemm_into did not accumulate: {t} vs {}", 2.0 * o);
        }
    }
}
