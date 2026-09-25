//! Level 3 BLAS: matrix-matrix operations
//! ========================================
//! `dgemm` runs on the packed, blocked kernel in [`crate::gemm`] (the same
//! code as the LLM path's `sgemm`, instantiated for f64), with a SAXPY
//! fast path for small and thin shapes. `dcrossprod` uses runtime-
//! multiversioned dot products.
//!
//! All matrices column-major (Fortran convention).

use crate::LinalgError;

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

    // Thin-output fast path. `gemm_small` is a column-major SAXPY loop:
    // serial, no packing, it streams A once per output column. That wins
    // only when C has very few columns — matrix-vector `X %*% w` above all,
    // where the packed kernel would pad one column to a 6-row tile — or
    // when the call is too small for packing to pay. Measured against the
    // packed kernel (2026-09-25, `--example dgemm_cases`, m x k x n):
    // 2000x200x1 65 vs 327 us, 100000x100x1 4.7 vs 8.1 ms, 100x100x6 10
    // vs 23 us, 50x3x50 1.6 vs 2.5 us — but 128³ 368 vs 68 us, 64³ 44 vs
    // 15, 1000x32x32 143 vs 48, and 50000x5x5 329 vs 227 (the packed form
    // is threaded; this is not). So: n <= 2 always, n <= 8 while the work
    // is under ~1M, very shallow k, and the trivially small.
    let work = m * n * k;
    let small = n <= 2 || (n <= 8 && work < 1_000_000) || k <= 4 || work <= 512;
    if small {
        gemm_small(m, n, k, alpha, a, b, c);
        return Ok(());
    }

    // Everything else runs on the packed kernel `sgemm` is built from
    // (`gemm::f64`: 6x8 AVX2 register tile, three-level blocking, threaded
    // packing). It is row-major and column-major C = A·B is row-major
    // Cᵀ = Bᵀ·Aᵀ: B's buffer read row-major IS Bᵀ (n x k) and A's IS Aᵀ
    // (k x m), so the operands swap and nothing is transposed or copied.
    // The Oracle still decides serial vs multi-core from m·n·k.
    let parallel = r2_oracle::should_parallelize(
        r2_oracle::Op::MatMul,
        r2_oracle::Shape::nmk(m, n, k),
    );
    use crate::gemm::{f64 as g, Trans};
    if alpha == 1.0 {
        // beta == 0 was zeroed above; assigning skips re-reading those zeros.
        if beta == 0.0 { g::gemm_assign_into(b, Trans::No, a, Trans::No, n, k, m, c, parallel); }
        else { g::gemm_into(b, Trans::No, a, Trans::No, n, k, m, c, parallel); }
    } else {
        let t = g::gemm(b, Trans::No, a, Trans::No, n, k, m, parallel);
        for (ci, ti) in c.iter_mut().zip(&t) { *ci += alpha * ti; }
    }
    Ok(())
}

/// `a*b + c`, fused on the FMA build. Rust never contracts `a * b + c`
/// on its own — that changes the rounding — so the "AVX2+FMA" build of
/// this kernel was emitting separate multiplies and adds (the same
/// finding as the attention kernels, checked in the emitted assembly).
/// `mul_add` is the one instruction where the feature is enabled; where
/// it is not it would call libm's software `fma`, hundreds of times
/// slower, so the baseline keeps the two-instruction form.
#[inline(always)]
fn fma<const F: bool>(a: f64, b: f64, c: f64) -> f64 {
    if F { a.mul_add(b, c) } else { a * b + c }
}

/// The best SIMD code path this CPU can run for `dot4`.
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
fn dot4_impl<const F: bool>(x: &[f64], y: &[f64], m: usize) -> f64 {
    let mut acc = [0.0f64; 4];
    let main = m - (m % 4);
    let mut p = 0;
    while p < main {
        acc[0] = fma::<F>(x[p], y[p], acc[0]);
        acc[1] = fma::<F>(x[p + 1], y[p + 1], acc[1]);
        acc[2] = fma::<F>(x[p + 2], y[p + 2], acc[2]);
        acc[3] = fma::<F>(x[p + 3], y[p + 3], acc[3]);
        p += 4;
    }
    let mut dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while p < m { dot = fma::<F>(x[p], y[p], dot); p += 1; }
    dot
}

/// Runtime-multiversioned dot product (AVX-512 → AVX2 → SSE2), same tiers
/// as the GEMM kernel. Powers `crossprod` / XᵀX.
#[inline]
pub(crate) fn dot4(x: &[f64], y: &[f64], m: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: each wrapper entered only when its feature is detected.
        match simd_tier() {
            SimdTier::Avx512 => return unsafe { dot4_avx512(x, y, m) },
            SimdTier::Avx2 => return unsafe { dot4_avx2(x, y, m) },
            SimdTier::Sse2 => {}
        }
    }
    dot4_impl::<false>(x, y, m)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn dot4_avx2(x: &[f64], y: &[f64], m: usize) -> f64 { dot4_impl::<true>(x, y, m) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn dot4_avx512(x: &[f64], y: &[f64], m: usize) -> f64 { dot4_impl::<true>(x, y, m) }

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
        r2_oracle::Op::CrossProd,
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

    /// `C = alpha·A·B + beta·C` on the packed path, every alpha/beta branch,
    /// at shapes that take each of the kernel's partitions: column groups
    /// (deep k), row blocks (shallow k), and the short-M transposed form.
    #[test]
    fn dgemm_alpha_beta_on_packed_path() {
        for &(m, n, k) in &[(300, 250, 900), (517, 301, 130), (40, 700, 300), (130, 70, 1100)] {
            let a: Vec<f64> = (0..m*k).map(|i| ((i * 7 + 3) % 19) as f64 * 0.25 - 2.0).collect();
            let b: Vec<f64> = (0..k*n).map(|i| ((i * 5 + 1) % 23) as f64 * 0.5 - 5.0).collect();
            let c0: Vec<f64> = (0..m*n).map(|i| ((i * 3) % 7) as f64 - 3.0).collect();
            let mut ab = vec![0.0; m * n];
            for j in 0..n { for p in 0..k { for i in 0..m { ab[j*m+i] += a[p*m+i] * b[j*k+p]; } } }
            for &(alpha, beta) in &[(1.0, 0.0), (1.0, 1.0), (1.0, -0.5), (2.5, 0.0), (-0.75, 2.0)] {
                let mut c = c0.clone();
                dgemm(m, n, k, alpha, &a, &b, beta, &mut c).unwrap();
                for idx in 0..m * n {
                    let want = alpha * ab[idx] + beta * c0[idx];
                    assert!((c[idx] - want).abs() <= 1e-9 * want.abs().max(1.0),
                        "{m}x{n}x{k} alpha {alpha} beta {beta} at {idx}: {} vs {want}", c[idx]);
                }
            }
        }
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
