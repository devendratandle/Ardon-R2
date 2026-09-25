//! The blocked GEMM itself, as one macro instantiated per element type
//! and tile in `gemm.rs` (`f32`, `f64`, `f32_512`).

/// Generate a complete blocked GEMM for one element type.
///
/// A macro rather than a trait: the micro-kernel has to be monomorphic for
/// the accumulator array to stay in registers, and `#[target_feature]`
/// cannot be applied through a generic. Rust has no numeric trait in std,
/// and pulling in `num-traits` would add a dependency to a crate that
/// deliberately has almost none. Instantiated for f32 and f64 (AVX2) and
/// f32 again at the AVX-512 tile below. `$feat` is the `target_feature` the
/// fallback micro-kernel is compiled for and `$wide` the runtime check
/// that gates every wide path of this instantiation.
macro_rules! blocked_gemm_for {
    ($ty:ty, $mr:expr, $nr:expr, $kc:expr, $mc:expr, $nc:expr, $feat:literal, $wide:expr) => {
        /// Rows per register tile. See the module docs.
        const MR: usize = $mr;
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
        fn micro_wide(kc: usize, apack: &[$ty], bpack: &[$ty]) -> [[$ty; NR]; MR] {
            micro_impl(kc, apack, bpack)
        }

        /// Resolved once per process.
        #[inline]
        pub(crate) fn have_wide() -> bool {
            #[cfg(target_arch = "x86_64")]
            {
                use std::sync::OnceLock;
                static OK: OnceLock<bool> = OnceLock::new();
                *OK.get_or_init(|| $wide)
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
                if std::mem::size_of::<$ty>() == 4 && MR == 12 && NR == 32 {
                    return unsafe {
                        let a: &[f32] = core::slice::from_raw_parts(
                            apack.as_ptr() as *const f32, apack.len());
                        let b: &[f32] = core::slice::from_raw_parts(
                            bpack.as_ptr() as *const f32, bpack.len());
                        let r = super::micro_f32_avx512(kc, a, b);
                        core::ptr::read(&r as *const _ as *const [[$ty; NR]; MR])
                    };
                }
                if std::mem::size_of::<$ty>() == 8 && MR == 6 && NR == 8 {
                    return unsafe {
                        let a: &[f64] = core::slice::from_raw_parts(
                            apack.as_ptr() as *const f64, apack.len());
                        let b: &[f64] = core::slice::from_raw_parts(
                            bpack.as_ptr() as *const f64, bpack.len());
                        let r = super::micro_f64_avx2(kc, a, b);
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
                pack_b_into(b, k, n, trans, pc, kc, j0, NR.min(jc + nc - j0), strip);
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
            pack_a_into(a, k, m, trans, ic, mc, pc, kc, out);
        }

        /// The body of [`pack_a`], writing into a slice of at least
        /// `mc.div_ceil(MR) * kc * MR` elements. Ragged rows are written
        /// as zeros, so the slice need not be cleared first.
        fn pack_a_into(a: &[$ty], k: usize, m: usize, trans: Trans,
                       ic: usize, mc: usize, pc: usize, kc: usize, out: &mut [$ty]) {
            let panels = mc.div_ceil(MR);
            if trans == Trans::Yes {
                // A stored k x m: row `p` of the block is `mc` CONTIGUOUS
                // elements. Walk the rows once, sequentially, scattering
                // each into its `MR`-wide panel slot — instead of walking
                // all `kc` rows once per panel (sixteen strided passes over
                // the same lines; measured 2 ns per element, 0.5 GB/s).
                for p in 0..kc {
                    let src = (pc + p) * m + ic;
                    let row = &a[src..src + mc];
                    for pnl in 0..panels {
                        let dst = &mut out[pnl * kc * MR + p * MR..pnl * kc * MR + (p + 1) * MR];
                        let i0 = pnl * MR;
                        if i0 + MR <= mc {
                            *<&mut [$ty; MR]>::try_from(dst).unwrap() = row[i0..i0 + MR].try_into().unwrap();
                        } else {
                            for ii in 0..MR {
                                dst[ii] = if i0 + ii < mc { row[i0 + ii] } else { 0 as $ty };
                            }
                        }
                    }
                }
                return;
            }
            for pnl in 0..panels {
                let i0 = ic + pnl * MR;
                let base = pnl * kc * MR;
                for p in 0..kc {
                    let row = base + p * MR;
                    for ii in 0..MR {
                        let i = i0 + ii;
                        out[row + ii] = if i < ic + mc && i < m {
                            match trans {
                                Trans::No => a[i * k + (pc + p)],
                                Trans::Yes => a[(pc + p) * m + i],
                            }
                        } else { 0 as $ty };
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
            gemm_core(a, ta, b, tb, m, k, n, c, parallel, false)
        }

        /// `C = A·B` into an existing buffer whose contents are IGNORED.
        ///
        /// The first depth slab assigns and later slabs accumulate, so the
        /// caller need not zero `c` first. That is what lets a recycled
        /// buffer from the tape's pool be used directly: zeroing a 64 MB
        /// logits buffer before every output-head GEMM would cost the
        /// bandwidth the pool exists to save.
        pub fn gemm_assign_into(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                                m: usize, k: usize, n: usize, c: &mut [$ty], parallel: bool) {
            gemm_core(a, ta, b, tb, m, k, n, c, parallel, true)
        }

        #[allow(clippy::too_many_arguments)]
        fn gemm_core(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                     m: usize, k: usize, n: usize, c: &mut [$ty], parallel: bool,
                     assign: bool) {
            debug_assert_eq!(a.len(), m * k);
            debug_assert_eq!(b.len(), k * n);
            debug_assert_eq!(c.len(), m * n);
            if m == 0 || n == 0 || k == 0 {
                if assign { for v in c.iter_mut() { *v = 0 as $ty; } }
                return;
            }

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
            // (Not when the column-group form takes the call: it wants
            // the LONG side as `n`, and measured 3.3 vs 5.2 ms on
            // 256x2048x768 for keeping it there.)
            let blocks = m.div_ceil(MC);
            if parallel && blocks < nthreads() && n > m && m * n > 0 && !colgroups_ok(m, k, n) {
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
                                if assign { c[i * n + j] = ct[j * m + i]; }
                                else { c[i * n + j] += ct[j * m + i]; }
                            }
                        }
                    }
                }
                return;
            }

            let wide = have_wide();

            if stats_on() && parallel {
                let t = std::time::Instant::now();
                gemm_forkjoin(a, ta, b, tb, m, k, n, c, parallel, assign, wide);
                STATS.lock().unwrap().push(((m as u64) << 40 | (k as u64) << 20 | n as u64, t.elapsed().as_secs_f32() * 1e6, false));
                return;
            }
            gemm_forkjoin(a, ta, b, tb, m, k, n, c, parallel, assign, wide);
        }

        /// The fork-join path: pack B per (jc, pc) slab, then a fork over
        /// the row-blocks of C. What every parallel call runs unless the
        /// team is opted in.
        #[allow(clippy::too_many_arguments)]
        /// Pack the `kc x nc` slab of B at (`pc`, `jc`) into a slice of
        /// `nc.div_ceil(NR) * kc * NR` elements, serially — the body of
        /// [`pack_b`], for a caller that already runs inside a task.
        fn pack_b_into(b: &[$ty], k: usize, n: usize, trans: Trans,
                       pc: usize, kc: usize, jc: usize, nc: usize, out: &mut [$ty]) {
            let strips = nc.div_ceil(NR);
            for s in 0..strips {
                let strip = &mut out[s * kc * NR..(s + 1) * kc * NR];
                let j0 = jc + s * NR;
                let full = j0 + NR <= jc + nc && j0 + NR <= n;
                if full && trans == Trans::No {
                    // A full strip of an untransposed B is `kc` runs of
                    // `NR` contiguous elements: a copy per run, not an
                    // element loop with two bounds tests each.
                    // Fixed-size array moves: a `copy_from_slice` of a
                    // runtime length is a `memcpy` call per run, and at
                    // 64 bytes the call costs more than the copy.
                    for p in 0..kc {
                        let src = (pc + p) * n + j0;
                        let run: [$ty; NR] = b[src..src + NR].try_into().unwrap();
                        *<&mut [$ty; NR]>::try_from(&mut strip[p * NR..(p + 1) * NR]).unwrap() = run;
                    }
                    continue;
                }
                for p in 0..kc {
                    let row = p * NR;
                    for jj in 0..NR {
                        let j = j0 + jj;
                        strip[row + jj] = if j < jc + nc && j < n {
                            match trans {
                                Trans::No => b[(pc + p) * n + j],
                                Trans::Yes => b[j * k + (pc + p)],
                            }
                        } else { 0 as $ty };
                    }
                }
            }
        }

        /// The parallel form with PRIVATE B: tasks own column groups of C
        /// for the whole depth, and pack their own strips of B.
        ///
        /// [`gemm_forkjoin`] parallelises over row-blocks of C and packs
        /// one shared `KC x NC` panel of B per depth slab that every task
        /// then streams from L3, sixteen times per block. That is right
        /// when A is the large operand and `m` is long. It is wrong when
        /// `m` is short, and `grad_B = Aᵀ·g` is short by construction — M
        /// is the weight's input width (256, 768), K the token count.
        /// Kernel to kernel against MKL (`benchmarks/llm/gemm_cases.py`,
        /// both sides per shape in one window) every TN shape with
        /// M <= 768 ran at 0.50-0.84x of MKL while the same kernel at
        /// M >= 2048 ran 1.1-1.7x ahead; the plain NN case at 768x2048x768
        /// was just as slow, so it is the shape, not the transpose. Probes
        /// showed the micro-kernel 45% slower per FLOP on the short shape
        /// (the shared panel's cold fetch per slab and block) and three
        /// times the fork count (a fork per slab, ~8 tasks each).
        ///
        /// Here A is packed ONCE for the whole depth (in parallel), then
        /// each task takes a group of `NR`-wide column strips of C — and,
        /// when `n` is too short to make enough groups, a group of
        /// row-blocks as well — over all depth slabs, packing its own
        /// strips of B slab by slab. Every strip of B is still packed
        /// exactly once per row group, and it stays in that core's L2 for
        /// every row block it serves; A's packed panels stream from L3
        /// (or memory) once per column group, `gs` strips of reuse each.
        /// Three forks per call instead of three per slab. Two tasks never
        /// share a row of C, so each accumulates into its own contiguous
        /// tile and the tiles are folded into C once — one extra pass
        /// over `m x n` against `m*k*n` work. Each element sums its depth
        /// slabs in K order inside one task, so the result is bit-identical
        /// to the serial kernel ([`sgemm_is_thread_count_independent`]).
        ///
        /// Measured against the row-block form on this machine (TN unless
        /// said; us, best of two, same window): 2048x768x768 13.3 -> 10.2,
        /// 2048x768x2304 41.9 -> 25.5 (MKL 31.6), 2048x768x8000 145 -> 84
        /// (MKL 109), 2048x1024x4096 86 -> 60 (MKL 79), 8192x768x768 52 ->
        /// 38 (MKL 42); NN 2048x768x2304 33 -> 26, 2048x2048x2048 76 -> 62.
        /// The row-block form keeps the cases where A is the large
        /// operand (`grad_A` at 8,192 tokens: A = g, 75 MB) — see the
        /// dispatch in [`gemm_forkjoin`].
        fn gemm_colgroups(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                          m: usize, k: usize, n: usize, c: &mut [$ty],
                          assign: bool, wide: bool) {
            use rayon::prelude::*;
            // Scratch is kept per thread across calls. A fresh multi-MB
            // `Vec` per call is a fresh mapping from the OS each time —
            // ~1,000 page faults on a 4 MB working set, a millisecond on
            // a call that does the arithmetic in one. Nested calls (a
            // borrow already out) fall back to a fresh buffer.
            thread_local! {
                static A_SCRATCH: std::cell::RefCell<Vec<$ty>> = const { std::cell::RefCell::new(Vec::new()) };
                static P_SCRATCH: std::cell::RefCell<Vec<$ty>> = const { std::cell::RefCell::new(Vec::new()) };
                static B_SCRATCH: std::cell::RefCell<Vec<$ty>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            // Grows only; contents are whatever the last call left. Every
            // element read below is written first (packing writes its
            // padding as zeros; tiles are assigned on the first slab).
            fn take(cell: &'static std::thread::LocalKey<std::cell::RefCell<Vec<$ty>>>, len: usize) -> Vec<$ty> {
                let mut v = cell.with(|c| c.try_borrow_mut().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default());
                if v.len() < len { v.resize(len, 0 as $ty); }
                v
            }
            fn give(cell: &'static std::thread::LocalKey<std::cell::RefCell<Vec<$ty>>>, v: Vec<$ty>) {
                cell.with(|c| if let Ok(mut b) = c.try_borrow_mut() { if b.capacity() < v.capacity() { *b = v; } });
            }
            let nt = nthreads();
            let blocks = m.div_ceil(MC);
            const AP: usize = MC / MR;                  // panels per full block
            let nslab = k.div_ceil(KC);
            let strips = n.div_ceil(NR);
            // Strips per column group: at least four so each packed A
            // panel is reused across four tiles, at most eight so a task's
            // tile and B strips stay L2-resident; then row groups make up
            // the task count when `n` alone cannot.
            let gs = (strips / (2 * nt)).clamp(4, 8).min(strips);
            let ng = strips.div_ceil(gs);
            let gw = gs * NR;
            // Row groups: enough tasks, and a tile no larger than half of
            // L2 so the accumulation stays there.
            let tile_cap = (256 << 10) / (gw * std::mem::size_of::<$ty>());   // rows
            let rg = (2 * nt).div_ceil(ng).max(m.div_ceil(tile_cap.max(MC))).clamp(1, blocks);
            let rb = blocks.div_ceil(rg);                // blocks per row group
            let rg = blocks.div_ceil(rb);
            let rows_t = rb * MC;                        // tile rows, padded

            // ── A, the whole depth, packed once: [slab][block][panel] ──
            let slab_elems = blocks * AP * KC * MR;
            let mut apack = take(&A_SCRATCH, nslab * slab_elems);
            apack[..nslab * slab_elems].par_chunks_mut(AP * KC * MR).enumerate().for_each(|(t, out)| {
                let (si, bi) = (t / blocks, t % blocks);
                let (pc, ic) = (si * KC, bi * MC);
                let kc = KC.min(k - pc);
                // `pack_a_into` lays panels out `kc` deep; on a short last
                // slab they simply end early in this chunk.
                pack_a_into(a, k, m, ta, ic, MC.min(m - ic), pc, kc, out);
            });
            let ap_all = &apack[..];

            // ── one task per (column group, row group), all slabs ──
            let mut part = take(&P_SCRATCH, ng * rg * rows_t * gw);
            part[..ng * rg * rows_t * gw].par_chunks_mut(rows_t * gw).enumerate().for_each(|(t, tile)| {
                let (gi, ri) = (t / rg, t % rg);
                let (s0, s1) = (gi * gs, ((gi + 1) * gs).min(strips));
                let (b0, b1) = (ri * rb, ((ri + 1) * rb).min(blocks));
                let (jc, nc) = (s0 * NR, ((s1 - s0) * NR).min(n - s0 * NR));
                let mut bpack = take(&B_SCRATCH, (s1 - s0) * KC * NR);
                for si in 0..nslab {
                    let pc = si * KC;
                    let kc = KC.min(k - pc);
                    pack_b_into(b, k, n, tb, pc, kc, jc, nc, &mut bpack[..(s1 - s0) * kc * NR]);
                    let first = si == 0;
                    for bi in b0..b1 {
                        let ic = bi * MC;
                        let mc = MC.min(m - ic);
                        for pnl in 0..mc.div_ceil(MR) {
                            let off = si * slab_elems + bi * AP * KC * MR + pnl * kc * MR;
                            let ap = &ap_all[off..off + kc * MR];
                            let rows = MR.min(mc - pnl * MR);
                            for s in 0..s1 - s0 {
                                let bpp = &bpack[s * kc * NR..(s + 1) * kc * NR];
                                let acc = micro(kc, ap, bpp, wide);
                                for ii in 0..rows {
                                    let trow = ((bi - b0) * MC + pnl * MR + ii) * gw + s * NR;
                                    let dst = &mut tile[trow..trow + NR];
                                    for (d, v) in dst.iter_mut().zip(&acc[ii]) {
                                        if first { *d = *v; } else { *d += v; }
                                    }
                                }
                            }
                        }
                    }
                }
                give(&B_SCRATCH, bpack);
            });

            // ── fold the tiles into C, one row block per task ──
            let part_ref = &part[..];
            c.par_chunks_mut(MC * n).enumerate().for_each(|(bi, band)| {
                let mc = MC.min(m - bi * MC);
                let (ri, bl) = (bi / rb, bi % rb);
                for gi in 0..ng {
                    let t = gi * rg + ri;
                    let tile = &part_ref[t * rows_t * gw..(t + 1) * rows_t * gw];
                    let col0 = gi * gs * NR;
                    let cols = gw.min(n - col0);
                    for ii in 0..mc {
                        let dst = &mut band[ii * n + col0..ii * n + col0 + cols];
                        let src = &tile[(bl * MC + ii) * gw..(bl * MC + ii) * gw + cols];
                        if assign { dst.copy_from_slice(src); }
                        else { for (d, v) in dst.iter_mut().zip(src) { *d += v; } }
                    }
                }
            });
            give(&A_SCRATCH, apack);
            give(&P_SCRATCH, part);
        }

        /// Does a parallel call go to [`gemm_colgroups`]? Yes unless A is
        /// the large operand or the depth is shallow. The column-group
        /// form streams A's packed panels once per column group — fine
        /// from L3, and with four strips of reuse from memory up to a few
        /// tens of MB; past that the row-block form, which streams B and
        /// keeps A blocks private, wins (`grad_A` at 8,192 tokens, A = g
        /// = 75 MB: 107 vs 128 ms). And at one or two depth slabs the
        /// column-group form's extra pass over its tiles costs more than
        /// the fork it saves (the dim-256 forward, K = 256: 1.3 vs 2.3
        /// ms) — unless `m` is too short to fill the cores anyway. A short
        /// `n` with a long `m` also stays on the row-block form: column
        /// groups of four strips make too few tasks there, and the row
        /// groups that fill in replicate B (8192x768x256: 13 vs 16 ms).
        fn colgroups_ok(m: usize, k: usize, n: usize) -> bool {
            let nt = nthreads();
            let a_bytes = m * k * std::mem::size_of::<$ty>();
            let short_m = m.div_ceil(MC) < 2 * nt;
            let deep = k >= 3 * KC || short_m;
            let wide = n >= 2 * nt * 4 * NR || short_m;
            m * n > 0 && deep && wide && a_bytes <= 32 << 20
        }

        fn gemm_forkjoin(a: &[$ty], ta: Trans, b: &[$ty], tb: Trans,
                         m: usize, k: usize, n: usize, c: &mut [$ty], parallel: bool,
                         assign: bool, wide: bool) {
            use rayon::prelude::*;
            if parallel && colgroups_ok(m, k, n) {
                gemm_colgroups(a, ta, b, tb, m, k, n, c, assign, wide);
                return;
            }
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
                    // Only the FIRST depth slab may assign; every later slab
                    // adds its partial product to what the first one wrote.
                    let first = assign && pc == 0;
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
                                        if first { *d = *v; } else { *d += v; }
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
