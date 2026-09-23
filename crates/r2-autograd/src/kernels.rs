//! CPU kernels behind the tape's hot ops: the four-accumulator dot
//! product and the tiled attention forward / backward (portable and AVX2+FMA
//! builds of one generic body), plus the buffer helpers they share.

/// Dot product with four independent accumulators — see the note beside
/// the same helpers in `r2_tensor::ops`, which is where the shared copies
/// live.
///
/// This one is deliberately LOCAL rather than imported from there. It is
/// called from inside `#[target_feature(enable = "avx2")]` kernels, and a
/// function defined in another crate is not reliably inlined across that
/// boundary — when it is not, the hottest loop in attention silently loses
/// the wide codegen it was given. Measured: importing it cost the training
/// step 31.05 -> 33.29 s, with no other change.
#[inline(always)]
pub(crate) fn dot4(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let mut a = [0.0f32; 4];
    let full = n - n % 4;
    let mut i = 0;
    while i < full {
        a[0] += x[i] * y[i];
        a[1] += x[i + 1] * y[i + 1];
        a[2] += x[i + 2] * y[i + 2];
        a[3] += x[i + 3] * y[i + 3];
        i += 4;
    }
    let mut t = 0.0f32;
    while i < n { t += x[i] * y[i]; i += 1; }
    (a[0] + a[1]) + (a[2] + a[3]) + t
}

/// Is the wide kernel usable here? Resolved once per process.
///
/// The workspace sets no `target-cpu`, so this crate compiles for baseline
/// x86-64 — SSE2, and no FMA. `r2_linalg::gemm` already dispatches an
/// AVX2 micro-kernel at runtime for exactly this reason and gained 4x from
/// it; the attention kernels had been left on the baseline path.
#[inline]
fn have_avx2() -> bool {
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

/// `a*b + c`, fused when the kernel was compiled for FMA hardware.
///
/// Rust never contracts `a * b + c` into one FMA on its own — that would
/// change the rounding — so the AVX2 attention kernels below asked for
/// FMA and got separate `vmulps`/`vaddps` pairs (checked in the emitted
/// assembly: zero `vfmadd`). `mul_add` compiles to the one instruction
/// where the feature is enabled; where it is not, it would call libm's
/// software `fma`, hundreds of times slower, so the baseline build keeps
/// the two-instruction form.
#[inline(always)]
fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    if F { a.mul_add(b, c) } else { a * b + c }
}

/// One `BR x 16` tile of `A · Bᵀ`, B packed transposed (`bt[d*seqp + j]
/// = B[j][d]`): for each head-dim `d`, one row of sixteen keys is loaded
/// and `BR` elements of A are broadcast against it. `BR` is a constant so
/// the accumulator — `BR` x 2 vector registers — never leaves the register
/// file; a runtime row count made LLVM keep it on the stack, at three
/// memory operations per FMA. The ragged last block of a sequence runs
/// this with `BR = 1`.
#[inline(always)]
fn tile16<const BR: usize, const F: bool>(a: &[f32], aoff: &[usize; BR], bt: &[f32], seqp: usize,
                           hd: usize, j0: usize, out: &mut [[f32; 16]; BR]) {
    for d in 0..hd {
        let br = &bt[d * seqp + j0..d * seqp + j0 + 16];
        for ii in 0..BR {
            let x = a[aoff[ii] + d];
            let row = &mut out[ii];
            for jj in 0..16 { row[jj] = fma::<F>(x, br[jj], row[jj]); }
        }
    }
}

/// Fused causal attention, forward, for ONE (sequence, query head).
///
/// `orows` are that head's `seq` output pieces — row `i` of the output,
/// columns `qh*hd..(qh+1)*hd` — handed in as disjoint slices so that the
/// work units can be (sequence, head) rather than sequence: see
/// [`head_pieces`]. `vq`, `vk`, `vv` are the WHOLE tensors and `s`
/// selects the rows; a slice of them would be a copy.
///
/// # Why it is tiled the way a GEMM is
///
/// The scores of a block of queries against a tile of keys are formed
/// lane-parallel — one query element broadcast against a row of sixteen
/// keys — so no horizontal reduction ever happens ([`tile16`]). That
/// needs Kᵀ (a row of keys per head-dim), packed once per call; at seq
/// 256 it is 64 KB. The softmax over the finished score row is the plain
/// two-pass form (row max, `exp_shift_sum`, normalise); the row is `seq`
/// floats, so nothing quadratic is materialised, and the online (flash)
/// rescaling measured nothing here (REPORT.md).
///
/// P·V accumulates `CW` lanes of the output in registers over the keys;
/// `CW` is the widest of 64/32/16/8 that divides `hd`, because each lane
/// group is one dependent FMA chain and Zen 2 needs eight chains in
/// flight to reach its FMA throughput — sixteen lanes (two chains) ran
/// at a fifth of it.
///
/// The causal mask is structural: keys beyond the last query of a block
/// are never computed, and only the diagonal tile is checked per element.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attn_forward_head_impl<const CW: usize, const F: bool>(vq: &[f32], vk: &[f32], vv: &[f32],
                          orows: &mut [&mut [f32]],
                          s: usize, qh: usize, seq: usize, nh: usize, nkv: usize,
                          hd: usize, scale: f32) {
    /// Queries per score block.
    const BR: usize = 4;
    /// Keys per register tile.
    const BC: usize = 16;
    debug_assert_eq!(hd % CW, 0);

    let group = nh / nkv;
    let kvh = qh / group;
    let (qw, kw) = (nh * hd, nkv * hd);
    let base = s * seq;
    let seqp = seq.next_multiple_of(BC);
    let mut kt = vec![0.0f32; hd * seqp];          // kt[d * seqp + j] = K[j][d]
    let mut sc = vec![0.0f32; BR * seqp];
    let mut ex = vec![0.0f32; seqp];
    for j in 0..seq {
        let koff = (base + j) * kw + kvh * hd;
        let krow = &vk[koff..koff + hd];
        for d in 0..hd { kt[d * seqp + j] = krow[d]; }
    }
    let qoff = |i: usize| (base + i) * qw + qh * hd;

    for i0 in (0..seq).step_by(BR) {
        let br = BR.min(seq - i0);
        let jmax = i0 + br - 1;
        let tiles = (jmax / BC) + 1;
        // ── scores: BR x (tiles*BC), lane-parallel over keys ──
        for t in 0..tiles {
            let j0 = t * BC;
            let mut tile = [[0.0f32; BC]; BR];
            if br == BR {
                tile16::<BR, F>(vq, &[qoff(i0), qoff(i0 + 1), qoff(i0 + 2), qoff(i0 + 3)], &kt, seqp, hd, j0, &mut tile);
            } else {
                for ii in 0..br {
                    let mut one = [[0.0f32; BC]; 1];
                    tile16::<1, F>(vq, &[qoff(i0 + ii)], &kt, seqp, hd, j0, &mut one);
                    tile[ii] = one[0];
                }
            }
            for ii in 0..br {
                let dst = &mut sc[ii * seqp + j0..ii * seqp + j0 + BC];
                for jj in 0..BC { dst[jj] = tile[ii][jj] * scale; }
            }
        }
        // ── softmax + P·V per query ──
        for ii in 0..br {
            let qi = i0 + ii;
            let n = qi + 1;                                   // causal: keys 0..=qi
            let row = &sc[ii * seqp..ii * seqp + n];
            let mut m = f32::NEG_INFINITY;
            for &x in row { if x > m { m = x; } }
            let inv = 1.0 / r2_tensor::ops::exp_shift_sum(row, m, &mut ex[..n]);
            let dst = &mut orows[qi];
            for c0 in (0..hd).step_by(CW) {
                let mut acc = [0.0f32; CW];
                for j in 0..n {
                    let e = ex[j];
                    let voff = (base + j) * kw + kvh * hd + c0;
                    let vr = &vv[voff..voff + CW];
                    for c in 0..CW { acc[c] = fma::<F>(e, vr[c], acc[c]); }
                }
                for c in 0..CW { dst[c0 + c] = acc[c] * inv; }
            }
        }
    }
}

/// Fused causal attention, backward, for ONE (sequence, kv head) — every
/// query head of its GQA group, so dK and dV for the kv head are complete
/// on return and no unit writes another's rows.
///
/// `dqrows[i]` is row `i`'s `group*hd` columns of the q gradient for this
/// group; `dkrows[i]` / `dvrows[i]` are row `i`'s `hd` columns of the k /
/// v gradients for this kv head. All are ASSIGNED here (dK/dV zeroed at
/// the start, dQ complete per query), so the caller zeroes nothing.
///
/// Per query i, with P its softmax row and dP = dO_i·Vᵀ:
///
/// ```text
///   D_i  = Σ_j P_ij dP_ij
///   dS_j = P_ij (dP_ij − D_i) · scale
///   dQ_i  = Σ_j dS_j K_j      dK_j += dS_j Q_i      dV_j += P_ij dO_i
/// ```
///
/// Masked positions (`j > i`) never enter any sum, which is what stops a
/// token receiving gradient from its future.
///
/// # Why the accumulations are blocked
///
/// The scores and dP are [`tile16`] register tiles over packed Kᵀ / Vᵀ,
/// formed in two passes so each fits the register file. The three
/// accumulations used to be one loop over (query, key) pairs that
/// read-modify-wrote three `hd`-float rows per pair — 384 loads and 384
/// stores for 384 FLOPs at hd 64, on rows that live in L2 at seq 256.
/// Measured in situ at dim 768 / seq 256 that was 290 ms a step, 3.3x
/// PyTorch's SDPA backward. Now a block of `QB` queries is finished at
/// once: dV_j, then dK_j, are reduced over the block's queries in `CW`
/// register lanes and touched in memory once per block; dQ_i is reduced
/// over all its keys in registers and written once.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attn_backward_group_impl<const CW: usize, const F: bool>(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                            dqrows: &mut [&mut [f32]], dkrows: &mut [&mut [f32]],
                            dvrows: &mut [&mut [f32]],
                            s: usize, kvh: usize, seq: usize, nh: usize, nkv: usize,
                            hd: usize, scale: f32) {
    /// Queries per score tile.
    const BR: usize = 4;
    /// Keys per register tile.
    const BC: usize = 16;
    /// Queries finished per block (the dK/dV reduction width).
    const QB: usize = 16;
    debug_assert_eq!(hd % CW, 0);

    let group = nh / nkv;
    let (qw, kw) = (nh * hd, nkv * hd);
    let base = s * seq;
    let seqp = seq.next_multiple_of(BC);
    let mut kt = vec![0.0f32; hd * seqp];
    let mut vt = vec![0.0f32; hd * seqp];
    let mut sc = vec![0.0f32; QB * seqp];   // scores, then probabilities P
    let mut dp = vec![0.0f32; QB * seqp];   // dO·Vᵀ, then dS
    let mut ex = vec![0.0f32; seqp];
    for j in 0..seq {
        let off = (base + j) * kw + kvh * hd;
        let (krow, vrow) = (&vk[off..off + hd], &vv[off..off + hd]);
        for d in 0..hd { kt[d * seqp + j] = krow[d]; vt[d * seqp + j] = vrow[d]; }
        for x in dkrows[j].iter_mut() { *x = 0.0; }
        for x in dvrows[j].iter_mut() { *x = 0.0; }
    }

    for qh in kvh * group..(kvh + 1) * group {
        let qcol = (qh - kvh * group) * hd;                 // this head's columns in dqrows
        let qoff = |i: usize| (base + i) * qw + qh * hd;    // q and dO share the layout
        for i0 in (0..seq).step_by(QB) {
            let qb = QB.min(seq - i0);
            let jmax = i0 + qb - 1;
            // ── scores (into sc) and dP (into dp), BR x BC tiles ──
            for ii0 in (0..qb).step_by(BR) {
                let br = BR.min(qb - ii0);
                let tiles = ((i0 + ii0 + br - 1) / BC) + 1;
                for t in 0..tiles {
                    let j0 = t * BC;
                    let mut ts = [[0.0f32; BC]; BR];
                    let mut td = [[0.0f32; BC]; BR];
                    if br == BR {
                        let offs = [qoff(i0 + ii0), qoff(i0 + ii0 + 1), qoff(i0 + ii0 + 2), qoff(i0 + ii0 + 3)];
                        tile16::<BR, F>(vq, &offs, &kt, seqp, hd, j0, &mut ts);
                        tile16::<BR, F>(g, &offs, &vt, seqp, hd, j0, &mut td);
                    } else {
                        for ii in 0..br {
                            let (mut a, mut b) = ([[0.0f32; BC]; 1], [[0.0f32; BC]; 1]);
                            tile16::<1, F>(vq, &[qoff(i0 + ii0 + ii)], &kt, seqp, hd, j0, &mut a);
                            tile16::<1, F>(g, &[qoff(i0 + ii0 + ii)], &vt, seqp, hd, j0, &mut b);
                            ts[ii] = a[0]; td[ii] = b[0];
                        }
                    }
                    for ii in 0..br {
                        let r = (ii0 + ii) * seqp + j0;
                        let (ds, dd) = (&mut sc[r..r + BC], &mut dp[r..r + BC]);
                        for jj in 0..BC { ds[jj] = ts[ii][jj] * scale; dd[jj] = td[ii][jj]; }
                    }
                }
            }
            // ── per query: P, D, dS (over sc / dp); masked keys zeroed ──
            for ii in 0..qb {
                let qi = i0 + ii;
                let n = qi + 1;
                let prow = &mut sc[ii * seqp..ii * seqp + jmax + 1];
                let mut m = f32::NEG_INFINITY;
                for &x in &prow[..n] { if x > m { m = x; } }
                let inv = 1.0 / r2_tensor::ops::exp_shift_sum(&prow[..n], m, &mut ex[..n]);
                for j in 0..n { prow[j] = ex[j] * inv; }
                for j in n..jmax + 1 { prow[j] = 0.0; }
                let drow = &mut dp[ii * seqp..ii * seqp + jmax + 1];
                let mut dot = 0.0f32;
                for j in 0..n { dot += prow[j] * drow[j]; }
                for j in 0..n { drow[j] = prow[j] * (drow[j] - dot) * scale; }
                for j in n..jmax + 1 { drow[j] = 0.0; }
            }
            // ── dV_j += Σ_ii P_ij dO_i, then dK_j += Σ_ii dS_ij Q_i — key-outer,
            // reduced over the block's queries in registers, one
            // read-modify-write of each row per block. Two passes so each
            // has the whole register file. ──
            for j in 0..=jmax {
                let ii_lo = j.saturating_sub(i0);             // queries that can see key j
                let dv = &mut dvrows[j];
                for c0 in (0..hd).step_by(CW) {
                    let mut av = [0.0f32; CW];
                    for ii in ii_lo..qb {
                        let p = sc[ii * seqp + j];
                        let off = qoff(i0 + ii) + c0;
                        let gr = &g[off..off + CW];
                        for c in 0..CW { av[c] = fma::<F>(p, gr[c], av[c]); }
                    }
                    for c in 0..CW { dv[c0 + c] += av[c]; }
                }
                let dk = &mut dkrows[j];
                for c0 in (0..hd).step_by(CW) {
                    let mut ak = [0.0f32; CW];
                    for ii in ii_lo..qb {
                        let d = dp[ii * seqp + j];
                        let off = qoff(i0 + ii) + c0;
                        let qr = &vq[off..off + CW];
                        for c in 0..CW { ak[c] = fma::<F>(d, qr[c], ak[c]); }
                    }
                    for c in 0..CW { dk[c0 + c] += ak[c]; }
                }
            }
            // ── dQ_i = Σ_j dS_ij K_j — query-outer, every key in registers,
            // written once ──
            for ii in 0..qb {
                let qi = i0 + ii;
                let dq = &mut dqrows[qi];
                for c0 in (0..hd).step_by(CW) {
                    let mut aq = [0.0f32; CW];
                    for j in 0..=qi {
                        let d = dp[ii * seqp + j];
                        let koff = (base + j) * kw + kvh * hd + c0;
                        let kr = &vk[koff..koff + CW];
                        for c in 0..CW { aq[c] = fma::<F>(d, kr[c], aq[c]); }
                    }
                    for c in 0..CW { dq[qcol + c0 + c] = aq[c]; }
                }
            }
        }
    }
}

/// AVX2+FMA build of the forward kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
fn attn_forward_head_avx2<const CW: usize>(vq: &[f32], vk: &[f32], vv: &[f32],
                          orows: &mut [&mut [f32]],
                          s: usize, qh: usize, seq: usize, nh: usize, nkv: usize,
                          hd: usize, scale: f32) {
    attn_forward_head_impl::<CW, true>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale)
}

/// AVX2+FMA build of the backward kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
fn attn_backward_group_avx2<const CW: usize>(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                            dqrows: &mut [&mut [f32]], dkrows: &mut [&mut [f32]],
                            dvrows: &mut [&mut [f32]],
                            s: usize, kvh: usize, seq: usize, nh: usize, nkv: usize,
                            hd: usize, scale: f32) {
    attn_backward_group_impl::<CW, true>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale)
}

/// Dispatch: instruction set once per process, accumulator width by
/// `hd` (the widest of 64/32/16/8 lanes that divides it, else one — the
/// same code at every width). The branch is per work unit, not per
/// element.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_forward_head(vq: &[f32], vk: &[f32], vv: &[f32], orows: &mut [&mut [f32]],
                     s: usize, qh: usize, seq: usize, nh: usize, nkv: usize,
                     hd: usize, scale: f32) {
    macro_rules! go {
        ($f:ident $(, $F:literal)?) => {
            if hd % 64 == 0 { $f::<64 $(, $F)?>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale) }
            else if hd % 32 == 0 { $f::<32 $(, $F)?>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale) }
            else if hd % 16 == 0 { $f::<16 $(, $F)?>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale) }
            else if hd % 8 == 0 { $f::<8 $(, $F)?>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale) }
            else { $f::<1 $(, $F)?>(vq, vk, vv, orows, s, qh, seq, nh, nkv, hd, scale) }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; the callee's body
        // is the same safe code, compiled with wider instructions.
        unsafe { go!(attn_forward_head_avx2) };
        return;
    }
    go!(attn_forward_head_impl, false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_backward_group(vq: &[f32], vk: &[f32], vv: &[f32], g: &[f32],
                       dqrows: &mut [&mut [f32]], dkrows: &mut [&mut [f32]],
                       dvrows: &mut [&mut [f32]],
                       s: usize, kvh: usize, seq: usize, nh: usize, nkv: usize,
                       hd: usize, scale: f32) {
    macro_rules! go {
        ($f:ident $(, $F:literal)?) => {
            if hd % 64 == 0 { $f::<64 $(, $F)?>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale) }
            else if hd % 32 == 0 { $f::<32 $(, $F)?>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale) }
            else if hd % 16 == 0 { $f::<16 $(, $F)?>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale) }
            else if hd % 8 == 0 { $f::<8 $(, $F)?>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale) }
            else { $f::<1 $(, $F)?>(vq, vk, vv, g, dqrows, dkrows, dvrows, s, kvh, seq, nh, nkv, hd, scale) }
        };
    }
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: as above.
        unsafe { go!(attn_backward_group_avx2) };
        return;
    }
    go!(attn_backward_group_impl, false)
}

/// Split a row-major `[nseq*seq][nunits_per_row * piece]` buffer into
/// one `Vec` of row pieces per (sequence, unit): `out[s * per_row + u][i]`
/// is row `s*seq + i`, columns `u*piece..(u+1)*piece`. Plain reborrows —
/// no copy, no unsafe — so attention can hand each (sequence, head) to a
/// different worker although a head's columns are strided through the
/// tensor. `nseq * per_row * seq` pointers: at 8 x 12 x 256 that is 24 K.
pub(crate) fn head_pieces(buf: &mut [f32], seq: usize, per_row: usize, piece: usize) -> Vec<Vec<&mut [f32]>> {
    let nseq = buf.len() / (seq * per_row * piece);
    let mut out: Vec<Vec<&mut [f32]>> = (0..nseq * per_row).map(|_| Vec::with_capacity(seq)).collect();
    for (r, row) in buf.chunks_mut(per_row * piece).enumerate() {
        let s = r / seq;
        for (u, p) in row.chunks_mut(piece).enumerate() { out[s * per_row + u].push(p); }
    }
    out
}

/// Transpose a row-major `rows × cols` matrix. Used by the GPU backward
/// path to express both gradients as plain matmuls.
pub(crate) fn transpose_of(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols { out[j * rows + i] = x[i * cols + j]; }
    }
    out
}
