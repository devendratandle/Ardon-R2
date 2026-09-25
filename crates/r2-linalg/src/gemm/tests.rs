//! Tests for the blocked GEMM (`super` is `crate::gemm`).

use super::*;

fn transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; r * c];
    for i in 0..r { for j in 0..c { t[j * r + i] = x[i * c + j]; } }
    t
}

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

/// The AVX-512 kernel (12x32 tile) against the AVX2 kernel (6x16), BIT
/// for bit: same packing order, same per-element FMA sequence, so any
/// difference is a bug in the tile, its edges, or the partitions at a
/// different MR. Every transpose case, ragged sizes around both tiles,
/// both partitions (short and long m, shallow and deep k), serial and
/// threaded, assigning and accumulating. Returns early on a CPU without
/// AVX-512 — run it under Intel SDE (`sde -spr -- <test exe>`) there.
#[test]
fn avx512_is_bit_identical_to_avx2() {
    if !(f32_512::have_wide() && f32::have_wide()) {
        eprintln!("avx512_is_bit_identical_to_avx2: no AVX-512 here, skipped");
        return;
    }
    eprintln!("avx512_is_bit_identical_to_avx2: AVX-512 path active");
    let shapes = [
        (1usize, 1usize, 1usize), (12, 7, 32), (13, 33, 31), (11, 300, 65),
        (97, 256, 95), (200, 513, 40), (256, 2048, 768), (2048, 256, 768),
        (40, 900, 1100), (1030, 70, 20),
    ];
    let ts = [(Trans::No, Trans::No), (Trans::No, Trans::Yes), (Trans::Yes, Trans::No), (Trans::Yes, Trans::Yes)];
    for &(m, k, n) in &shapes {
        let a = mk(m * k, 0.3);
        let b = mk(k * n, 1.7);
        for &(ta, tb) in &ts {
            for &par in &[false, true] {
                let want = f32::gemm(&a, ta, &b, tb, m, k, n, par);
                let got = f32_512::gemm(&a, ta, &b, tb, m, k, n, par);
                let same = want.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "{m}x{k}x{n} {ta:?}{tb:?} par={par}: AVX-512 differs from AVX2");
                // accumulate into a non-zero C
                let c0 = mk(m * n, 2.9);
                let (mut c1, mut c2) = (c0.clone(), c0.clone());
                f32::gemm_into(&a, ta, &b, tb, m, k, n, &mut c1, par);
                f32_512::gemm_into(&a, ta, &b, tb, m, k, n, &mut c2, par);
                assert!(c1.iter().zip(&c2).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{m}x{k}x{n} {ta:?}{tb:?} par={par}: accumulate differs");
            }
        }
    }
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
