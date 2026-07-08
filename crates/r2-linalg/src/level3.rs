//! Level 3 BLAS: matrix-matrix operations
//! ========================================
//! Optimized with:
//!   - 3-level cache blocking (L1/L2/L3 aware)
//!   - 8×4 micro-kernel with register accumulation
//!   - Panel packing for contiguous memory access
//!   - 4-way unrolled inner loops for auto-SIMD
//!
//! All matrices column-major (Fortran convention).

use crate::LinalgError;

// Cache parameters — tuned for modern x86_64
const MR: usize = 8;    // micro-kernel rows
const NR: usize = 4;    // micro-kernel cols
const MC: usize = 256;  // L2 block rows
const KC: usize = 256;  // L2 block depth
const NC: usize = 512;  // L3 block cols

/// General matrix multiply: C = alpha*A*B + beta*C
pub fn dgemm(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], b: &[f64], beta: f64, c: &mut [f64],
) -> Result<(), LinalgError> {
    if a.len() != m * k { return Err(LinalgError::InvalidShape(format!("A: {}x{}", m, k))); }
    if b.len() != k * n { return Err(LinalgError::InvalidShape(format!("B: {}x{}", k, n))); }
    if c.len() != m * n { return Err(LinalgError::InvalidShape(format!("C: {}x{}", m, n))); }

    if beta == 0.0 { for ci in c.iter_mut() { *ci = 0.0; } }
    else if beta != 1.0 { for ci in c.iter_mut() { *ci *= beta; } }
    if alpha == 0.0 { return Ok(()); }

    // Small / thin matrix fast path. The blocked path below packs A- and
    // B-panels into scratch buffers and asks the Oracle whether to parallelise
    // — worthwhile amortised cost for large square matrices, but pure overhead
    // for the small and *thin* shapes that dominate statistics (`X %*% w`,
    // `t(X) %*% y`, small covariance blocks). `gemm_small` is a tight
    // column-major SAXPY loop the compiler auto-vectorises: it re-streams A
    // once per output column, so it wins whenever A (m·k) and the B-panel (k·n)
    // comfortably fit cache. Route there for genuinely small dims OR when both
    // panels are ≤ ~128 KB (16384 f64); use the blocked kernel only for the
    // large square case where its packing actually pays.
    if m.max(n).max(k) <= 96 || (m * k <= 16_384 && k * n <= 16_384) {
        gemm_small(m, n, k, alpha, a, b, c);
        return Ok(());
    }

    // Oracle decides serial vs multi-core based on m·n·k (hardware-scaled).
    // Parallelism is over disjoint COLUMN bands of C: column-major storage
    // means columns [j0, j0+w) of C occupy the contiguous slice
    // [j0*m, (j0+w)*m), so `par_chunks_mut` hands each thread its own
    // non-overlapping band — no locking, no false sharing across bands.
    let parallel = r2_oracle::should_parallelize(
        r2_oracle::Op::MatMul,
        r2_oracle::Shape::nmk(m, n, k),
    );

    if parallel {
        use rayon::prelude::*;
        let cores = r2_oracle::hw().cores.max(1);
        // ~2 bands per core: enough for Rayon load-balancing, but few
        // enough that the per-band packing-buffer allocation stays cheap
        // (too many tiny bands regressed medium matrices in benchmarks).
        let band = (n / (cores * 2)).max(1);
        c.par_chunks_mut(band * m).enumerate().for_each(|(blk, c_band)| {
            let nc_band = c_band.len() / m;
            gemm_band(m, k, blk * band, nc_band, alpha, a, b, c_band);
        });
    } else {
        gemm_band(m, k, 0, n, alpha, a, b, c);
    }
    Ok(())
}

/// Compute `C[:, j0..j0+band_n] += alpha · A · B[:, j0..j0+band_n]` for one
/// column band, writing into `c_band` (the band's own contiguous slice,
/// local column 0 = global column `j0`). Preserves the full L1/L2/L3 cache
/// blocking *within* the band, so a parallel band is as cache-efficient as
/// the serial whole. Each call owns its packing buffers — that's what makes
/// the bands safe to run concurrently.
#[inline]
fn gemm_band(
    m: usize, k: usize, j0: usize, band_n: usize,
    alpha: f64, a: &[f64], b: &[f64], c_band: &mut [f64],
) {
    let mut packed_a = vec![0.0f64; MC * KC];
    let mut packed_b = vec![0.0f64; KC * NC];
    let mut j_local = 0;
    while j_local < band_n {
        let nc = (band_n - j_local).min(NC);
        let jc_global = j0 + j_local;
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(KC);
            // B is read at the GLOBAL column offset; C is written at the
            // LOCAL offset within c_band (so the macro-kernel's `jc` is
            // j_local, and ldc stays `m`).
            pack_b(k, b, pc, jc_global, kc, nc, &mut packed_b);
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);
                pack_a(m, a, ic, pc, mc, kc, &mut packed_a);
                macro_kernel(mc, nc, kc, alpha, &packed_a, &packed_b, c_band, m, ic, j_local);
                ic += MC;
            }
            pc += KC;
        }
        j_local += NC;
    }
}

#[inline]
fn pack_a(lda: usize, a: &[f64], ic: usize, pc: usize, mc: usize, kc: usize, packed: &mut [f64]) {
    let mut pos = 0;
    let mut i = 0;
    while i + MR <= mc {
        for p in 0..kc {
            let col_start = (pc + p) * lda + ic + i;
            for ii in 0..MR { packed[pos] = a[col_start + ii]; pos += 1; }
        }
        i += MR;
    }
    if i < mc {
        let rem = mc - i;
        for p in 0..kc {
            let col_start = (pc + p) * lda + ic + i;
            for ii in 0..rem { packed[pos] = a[col_start + ii]; pos += 1; }
            for _ in rem..MR { packed[pos] = 0.0; pos += 1; }
        }
    }
}

#[inline]
fn pack_b(ldb: usize, b: &[f64], pc: usize, jc: usize, kc: usize, nc: usize, packed: &mut [f64]) {
    let mut pos = 0;
    let mut j = 0;
    while j + NR <= nc {
        for p in 0..kc {
            for jj in 0..NR { packed[pos] = b[(jc + j + jj) * ldb + pc + p]; pos += 1; }
        }
        j += NR;
    }
    if j < nc {
        let rem = nc - j;
        for p in 0..kc {
            for jj in 0..rem { packed[pos] = b[(jc + j + jj) * ldb + pc + p]; pos += 1; }
            for _ in rem..NR { packed[pos] = 0.0; pos += 1; }
        }
    }
}

/// The macro-kernel body. `#[inline(always)]` so it is *re-codegened*
/// inside each multiversion wrapper below — the AVX2+FMA wrapper compiles
/// this exact source with wider vectors + fused multiply-add, the baseline
/// with SSE2. Single source, two machine-code variants.
#[inline(always)]
fn macro_kernel_impl(
    mc: usize, nc: usize, kc: usize, alpha: f64,
    packed_a: &[f64], packed_b: &[f64],
    c: &mut [f64], ldc: usize, ic: usize, jc: usize,
) {
    let mr_count = (mc + MR - 1) / MR;
    let nr_count = (nc + NR - 1) / NR;
    for jr in 0..nr_count {
        let j = jr * NR;
        let actual_nr = (nc - j).min(NR);
        let b_off = jr * kc * NR;
        for ir in 0..mr_count {
            let i = ir * MR;
            let actual_mr = (mc - i).min(MR);
            let a_off = ir * kc * MR;
            if actual_mr == MR && actual_nr == NR {
                micro_kernel_8x4(kc, alpha, &packed_a[a_off..], &packed_b[b_off..], c, ldc, ic + i, jc + j);
            } else {
                micro_kernel_generic(actual_mr, actual_nr, kc, alpha, &packed_a[a_off..], &packed_b[b_off..], c, ldc, ic + i, jc + j);
            }
        }
    }
}

/// Runtime-multiversioned macro-kernel: dispatch once to the AVX2+FMA
/// build of `macro_kernel_impl` when the CPU has it (cached detection),
/// else the SSE2 baseline. Identical numerical result. Mirrors how
/// NumPy/OpenBLAS pick a CPU kernel at runtime — one binary, runs
/// everywhere, fast where the hardware allows.
#[inline]
fn macro_kernel(
    mc: usize, nc: usize, kc: usize, alpha: f64,
    packed_a: &[f64], packed_b: &[f64],
    c: &mut [f64], ldc: usize, ic: usize, jc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: each wrapper is entered only when its feature is detected.
        match simd_tier() {
            SimdTier::Avx512 => { unsafe { macro_kernel_avx512(mc, nc, kc, alpha, packed_a, packed_b, c, ldc, ic, jc); } return; }
            SimdTier::Avx2   => { unsafe { macro_kernel_avx2(mc, nc, kc, alpha, packed_a, packed_b, c, ldc, ic, jc); } return; }
            SimdTier::Sse2   => {}
        }
    }
    macro_kernel_impl(mc, nc, kc, alpha, packed_a, packed_b, c, ldc, ic, jc);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn macro_kernel_avx2(
    mc: usize, nc: usize, kc: usize, alpha: f64,
    packed_a: &[f64], packed_b: &[f64],
    c: &mut [f64], ldc: usize, ic: usize, jc: usize,
) {
    macro_kernel_impl(mc, nc, kc, alpha, packed_a, packed_b, c, ldc, ic, jc);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn macro_kernel_avx512(
    mc: usize, nc: usize, kc: usize, alpha: f64,
    packed_a: &[f64], packed_b: &[f64],
    c: &mut [f64], ldc: usize, ic: usize, jc: usize,
) {
    macro_kernel_impl(mc, nc, kc, alpha, packed_a, packed_b, c, ldc, ic, jc);
}

/// The best SIMD code path this CPU can run for the GEMM kernel.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq)]
enum SimdTier { Sse2, Avx2, Avx512 }

/// Cached, runtime SIMD-tier selection (AVX-512 → AVX2 → SSE2).
/// Knobs: `R2_NO_SIMD=1` forces SSE2 (A/B benchmarking); `R2_SIMD=avx2`
/// caps at AVX2 (e.g. to avoid AVX-512 frequency downclock on a mixed
/// workload), `R2_SIMD=sse2` forces baseline.
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_tier() -> SimdTier {
    use std::sync::OnceLock;
    static TIER: OnceLock<SimdTier> = OnceLock::new();
    *TIER.get_or_init(|| {
        if std::env::var_os("R2_NO_SIMD").is_some() { return SimdTier::Sse2; }
        let cap = std::env::var("R2_SIMD").ok();
        match cap.as_deref() {
            Some("sse2") => return SimdTier::Sse2,
            _ => {}
        }
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        let avx512 = std::is_x86_feature_detected!("avx512f");
        if avx512 && cap.as_deref() != Some("avx2") { SimdTier::Avx512 }
        else if avx2 { SimdTier::Avx2 }
        else { SimdTier::Sse2 }
    })
}

/// 8x4 micro-kernel: 32 register accumulators, 32 FMAs per iteration
#[inline(always)]
fn micro_kernel_8x4(
    kc: usize, alpha: f64, a: &[f64], b: &[f64],
    c: &mut [f64], ldc: usize, ci: usize, cj: usize,
) {
    let (mut c00, mut c10, mut c20, mut c30) = (0.0f64, 0.0, 0.0, 0.0);
    let (mut c40, mut c50, mut c60, mut c70) = (0.0, 0.0, 0.0, 0.0);
    let (mut c01, mut c11, mut c21, mut c31) = (0.0, 0.0, 0.0, 0.0);
    let (mut c41, mut c51, mut c61, mut c71) = (0.0, 0.0, 0.0, 0.0);
    let (mut c02, mut c12, mut c22, mut c32) = (0.0, 0.0, 0.0, 0.0);
    let (mut c42, mut c52, mut c62, mut c72) = (0.0, 0.0, 0.0, 0.0);
    let (mut c03, mut c13, mut c23, mut c33) = (0.0, 0.0, 0.0, 0.0);
    let (mut c43, mut c53, mut c63, mut c73) = (0.0, 0.0, 0.0, 0.0);

    for p in 0..kc {
        let ao = p * MR; let bo = p * NR;
        let (a0,a1,a2,a3) = (a[ao],a[ao+1],a[ao+2],a[ao+3]);
        let (a4,a5,a6,a7) = (a[ao+4],a[ao+5],a[ao+6],a[ao+7]);
        let (b0,b1,b2,b3) = (b[bo],b[bo+1],b[bo+2],b[bo+3]);
        c00+=a0*b0; c10+=a1*b0; c20+=a2*b0; c30+=a3*b0;
        c40+=a4*b0; c50+=a5*b0; c60+=a6*b0; c70+=a7*b0;
        c01+=a0*b1; c11+=a1*b1; c21+=a2*b1; c31+=a3*b1;
        c41+=a4*b1; c51+=a5*b1; c61+=a6*b1; c71+=a7*b1;
        c02+=a0*b2; c12+=a1*b2; c22+=a2*b2; c32+=a3*b2;
        c42+=a4*b2; c52+=a5*b2; c62+=a6*b2; c72+=a7*b2;
        c03+=a0*b3; c13+=a1*b3; c23+=a2*b3; c33+=a3*b3;
        c43+=a4*b3; c53+=a5*b3; c63+=a6*b3; c73+=a7*b3;
    }

    let co0 = cj*ldc+ci;
    c[co0]+=alpha*c00; c[co0+1]+=alpha*c10; c[co0+2]+=alpha*c20; c[co0+3]+=alpha*c30;
    c[co0+4]+=alpha*c40; c[co0+5]+=alpha*c50; c[co0+6]+=alpha*c60; c[co0+7]+=alpha*c70;
    let co1 = (cj+1)*ldc+ci;
    c[co1]+=alpha*c01; c[co1+1]+=alpha*c11; c[co1+2]+=alpha*c21; c[co1+3]+=alpha*c31;
    c[co1+4]+=alpha*c41; c[co1+5]+=alpha*c51; c[co1+6]+=alpha*c61; c[co1+7]+=alpha*c71;
    let co2 = (cj+2)*ldc+ci;
    c[co2]+=alpha*c02; c[co2+1]+=alpha*c12; c[co2+2]+=alpha*c22; c[co2+3]+=alpha*c32;
    c[co2+4]+=alpha*c42; c[co2+5]+=alpha*c52; c[co2+6]+=alpha*c62; c[co2+7]+=alpha*c72;
    let co3 = (cj+3)*ldc+ci;
    c[co3]+=alpha*c03; c[co3+1]+=alpha*c13; c[co3+2]+=alpha*c23; c[co3+3]+=alpha*c33;
    c[co3+4]+=alpha*c43; c[co3+5]+=alpha*c53; c[co3+6]+=alpha*c63; c[co3+7]+=alpha*c73;
}

#[inline(always)]
fn micro_kernel_generic(
    mr: usize, nr: usize, kc: usize, alpha: f64,
    a: &[f64], b: &[f64],
    c: &mut [f64], ldc: usize, ci: usize, cj: usize,
) {
    let mut acc = [0.0f64; MR * NR];
    for p in 0..kc {
        let ao = p * MR; let bo = p * NR;
        for j in 0..nr { let bv = b[bo+j]; for i in 0..mr { acc[j*MR+i] += a[ao+i]*bv; } }
    }
    for j in 0..nr { let col = (cj+j)*ldc+ci; for i in 0..mr { c[col+i] += alpha*acc[j*MR+i]; } }
}

#[inline]
fn gemm_small(m: usize, n: usize, k: usize, alpha: f64, a: &[f64], b: &[f64], c: &mut [f64]) {
    for j in 0..n {
        for p in 0..k {
            let bpj = alpha * b[j*k+p];
            if bpj == 0.0 { continue; }
            let ac = p*m; let cc = j*m;
            for i in 0..m { c[cc+i] += bpj * a[ac+i]; }
        }
    }
}

/// Symmetric rank-k update: C = alpha*A*At + beta*C
pub fn dsyrk(m: usize, k: usize, alpha: f64, a: &[f64], beta: f64, c: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m * k { return Err(LinalgError::InvalidShape(format!("A: {}x{}", m, k))); }
    if c.len() != m * m { return Err(LinalgError::InvalidShape(format!("C: {}x{}", m, m))); }
    if beta == 0.0 { for ci in c.iter_mut() { *ci = 0.0; } }
    else if beta != 1.0 { for ci in c.iter_mut() { *ci *= beta; } }
    if alpha == 0.0 { return Ok(()); }
    for j in 0..m {
        for i in 0..=j {
            let mut dot = 0.0;
            for p in 0..k { dot += a[p*m+i] * a[p*m+j]; }
            c[j*m+i] += alpha*dot; if i != j { c[i*m+j] = c[j*m+i]; }
        }
    }
    Ok(())
}

/// Matrix transpose with 8x8 cache blocking
pub fn dtranspose(m: usize, n: usize, a: &[f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m*n || b.len() != m*n { return Err(LinalgError::InvalidShape("transpose".into())); }
    const TB: usize = 8;
    let mut jj = 0;
    while jj < n { let jmax = (jj+TB).min(n); let mut ii = 0;
        while ii < m { let imax = (ii+TB).min(m);
            for j in jj..jmax { for i in ii..imax { b[i*n+j] = a[j*m+i]; } }
            ii += TB;
        } jj += TB;
    }
    Ok(())
}

/// Dot-product body — 4 independent accumulators so the f64 add chain
/// isn't serialized (enables ILP + SIMD). `#[inline(always)]` so the
/// multiversion wrappers below recodegen it under AVX2 / AVX-512.
#[inline(always)]
fn dot4_impl(x: &[f64], y: &[f64], m: usize) -> f64 {
    let mut acc = [0.0f64; 4];
    let main = m - (m % 4);
    let mut p = 0;
    while p < main {
        acc[0] += x[p] * y[p];
        acc[1] += x[p + 1] * y[p + 1];
        acc[2] += x[p + 2] * y[p + 2];
        acc[3] += x[p + 3] * y[p + 3];
        p += 4;
    }
    let mut dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while p < m { dot += x[p] * y[p]; p += 1; }
    dot
}

/// Runtime-multiversioned dot product (AVX-512 → AVX2 → SSE2), same tiers
/// as the GEMM kernel. Powers `crossprod` / XᵀX.
#[inline]
fn dot4(x: &[f64], y: &[f64], m: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: each wrapper entered only when its feature is detected.
        match simd_tier() {
            SimdTier::Avx512 => return unsafe { dot4_avx512(x, y, m) },
            SimdTier::Avx2 => return unsafe { dot4_avx2(x, y, m) },
            SimdTier::Sse2 => {}
        }
    }
    dot4_impl(x, y, m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(x: &[f64], y: &[f64], m: usize) -> f64 { dot4_impl(x, y, m) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot4_avx512(x: &[f64], y: &[f64], m: usize) -> f64 { dot4_impl(x, y, m) }

/// Crossproduct: C = Aᵀ·A (n×n) with unrolled dot products. Oracle-gated
/// multi-core: parallel over output columns (each is an independent set of
/// dot products writing the upper triangle of its own contiguous column),
/// then a cheap serial mirror fills the lower triangle. Powers
/// `crossprod()` / Xᵀ X for covariance and normal equations.
pub fn dcrossprod(m: usize, n: usize, a: &[f64], c: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != m*n { return Err(LinalgError::InvalidShape(format!("A: {}x{}", m, n))); }
    if c.len() != n*n { return Err(LinalgError::InvalidShape(format!("C: {}x{}", n, n))); }
    for ci in c.iter_mut() { *ci = 0.0; }

    // work ≈ n²·m (triangular, so ~2× over-counted — fine for the gate).
    let parallel = r2_oracle::should_parallelize(
        r2_oracle::Op::MatMul,
        r2_oracle::Shape::nmk(n, n, m),
    );

    if parallel {
        use rayon::prelude::*;
        // Each chunk is one column of C (`c[j*n .. (j+1)*n]`); write only
        // its upper-triangle entries (rows 0..=j) — disjoint per column.
        c.par_chunks_mut(n).enumerate().for_each(|(j, c_col)| {
            let cj = j * m;
            for i in 0..=j {
                c_col[i] = dot4(&a[i * m..], &a[cj..], m);
            }
        });
        // Mirror upper → lower (serial, O(n²), negligible vs the dots).
        for j in 0..n {
            for i in 0..j { c[i * n + j] = c[j * n + i]; }
        }
    } else {
        for j in 0..n {
            let cj = j * m;
            for i in 0..=j {
                let dot = dot4(&a[i * m..], &a[cj..], m);
                c[j*n+i] = dot;
                if i != j { c[i*n+j] = dot; }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dgemm_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        dgemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
    }
    #[test]
    fn dgemm_large_parallel_matches_naive() {
        // m·n·k ≈ 25.8M comfortably exceeds the MatMul parallel threshold on
        // a multi-core box, and n=521 > NC=512 exercises multi-block bands.
        // Non-multiples of MR/NR/NC catch the remainder/edge paths.
        let (m, k, n) = (257usize, 193usize, 521usize);
        let a: Vec<f64> = (0..m * k).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i * 5 % 11) as f64) - 5.0).collect();
        let mut c = vec![0.0; m * n];
        dgemm(m, n, k, 1.0, &a, &b, 0.0, &mut c).unwrap();
        // Naive column-major reference.
        let mut r = vec![0.0; m * n];
        for j in 0..n {
            for p in 0..k {
                let bpj = b[j * k + p];
                for i in 0..m {
                    r[j * m + i] += a[p * m + i] * bpj;
                }
            }
        }
        for idx in 0..m * n {
            assert!((c[idx] - r[idx]).abs() < 1e-9,
                "mismatch at {}: {} vs {}", idx, c[idx], r[idx]);
        }
    }

    #[test]
    fn dcrossprod_parallel_matches_naive() {
        // n²·m ≈ 55M → parallel path on a multi-core box; correct either way.
        let (m, n) = (613usize, 301usize);
        let a: Vec<f64> = (0..m * n).map(|i| ((i * 3 % 17) as f64) - 8.0).collect();
        let mut c = vec![0.0; n * n];
        dcrossprod(m, n, &a, &mut c).unwrap();
        // Naive Aᵀ·A, column-major: c[j*n+i] = Σ_p a[i*m+p]·a[j*m+p].
        let mut r = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut d = 0.0;
                for p in 0..m { d += a[i * m + p] * a[j * m + p]; }
                r[j * n + i] = d;
            }
        }
        for idx in 0..n * n {
            assert!((c[idx] - r[idx]).abs() < 1e-6, "mismatch at {}", idx);
        }
    }

    #[test]
    fn test_dgemm_16x16_identity() {
        let n = 16;
        let mut eye = vec![0.0; n*n];
        for i in 0..n { eye[i*n+i] = 1.0; }
        let a: Vec<f64> = (0..(n*n)).map(|i| (i+1) as f64).collect();
        let mut c = vec![0.0; n*n];
        dgemm(n, n, n, 1.0, &a, &eye, 0.0, &mut c).unwrap();
        assert!((c[0] - a[0]).abs() < 1e-10);
        assert!((c[n*n-1] - a[n*n-1]).abs() < 1e-10);
    }

    /// The small/thin fast path and the blocked path must agree on every shape
    /// straddling the routing threshold (naive triple-loop reference).
    #[test]
    fn dgemm_paths_agree_across_threshold() {
        fn naive(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
            let mut c = vec![0.0; m * n];
            for j in 0..n { for p in 0..k { for i in 0..m { c[j*m+i] += a[p*m+i] * b[j*k+p]; } } }
            c
        }
        // (m,n,k): small, thin (both orientations — the r2sem shapes), and large
        // square (blocked path).
        for &(m, n, k) in &[(5,5,5), (50,50,50), (100,100,100), (1000,1,7), (7,1,1000), (200,200,200)] {
            let a: Vec<f64> = (0..m*k).map(|i| ((i * 7 + 1) % 13) as f64 - 6.0).collect();
            let b: Vec<f64> = (0..k*n).map(|i| ((i * 5 + 2) % 11) as f64 - 5.0).collect();
            let mut c = vec![0.0; m * n];
            dgemm(m, n, k, 1.0, &a, &b, 0.0, &mut c).unwrap();
            let r = naive(m, n, k, &a, &b);
            let err = c.iter().zip(&r).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(err < 1e-9, "dgemm {}x{}x{} disagrees with naive by {}", m, n, k, err);
        }
    }
}
