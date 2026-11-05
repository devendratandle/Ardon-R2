//! `sgemm` on CUDA: `C (=|+=) op(A) · op(B)`, row-major, in two forms.
//!
//! [`gemm_tiled`] is `kernels/gemm.cu`, the structure of the wgpu kernel in
//! `src/gemm.rs`: every element of C is one thread's fixed-order sum, so the
//! same inputs give the same bits on every run. [`gemm_cublas`] calls cuBLAS
//! if `libcublas.so.13` is on the machine, and is the faster of the two on a
//! Tesla T4. [`gemm`] takes cuBLAS and falls back to tiled when it is missing;
//! `R2_CUDA_GEMM=tiled|cublas` overrides the choice per call.

use super::ffi::{blas, Arg, Kernel};
use super::{loaded, Loaded, Tensor};
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

/// Rows and columns of C per block, as in `kernels/gemm.cu`.
const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 16;
/// Threads per block in x and y (BN/TN, BM/TM).
const BLOCK: [u32; 3] = [16, 16, 1];
const REDUCE_THREADS: u32 = 256;

const PTX: &str = include_str!("kernels/gemm.ptx");

struct Kernels { tiled: Kernel, reduce: Kernel }
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Result<&'static Kernels, &'static str> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels { tiled: m.kernel("sgemm_tiled", 15)?, reduce: m.kernel("sgemm_reduce", 6)? }))
}

/// Whether the module loads, for `module_status`.
pub(super) fn loads() -> Result<(), &'static str> { kernels().map(|_| ()) }

/// Below this many output tiles the depth is split, and never into chunks
/// under `MIN_CHUNK`: the wgpu kernel's constants, measured there.
const SPLIT_BELOW: u32 = 24;
const MIN_CHUNK: u32 = 256;

fn split(tiles: u32, k: u32) -> (u32, u32) {
    if tiles >= SPLIT_BELOW || k <= MIN_CHUNK { return (1, k.max(1)); }
    let want = SPLIT_BELOW.div_ceil(tiles);
    let chunk = k.div_ceil(want).max(MIN_CHUNK).div_ceil(BK) * BK;
    (k.div_ceil(chunk), chunk)
}

/// The split-K partials, grown on demand. The lock is held across a call's
/// two launches, so another thread's partials cannot land in between.
static SCRATCH: Mutex<Option<Tensor>> = Mutex::new(None);

/// Which GEMM a call used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path { Tiled, Cublas }

thread_local! { static LAST: Cell<Option<Path>> = const { Cell::new(None) }; }

/// The path this thread's most recent successful [`gemm`] took: what a
/// benchmark prints, so one row cannot be mistaken for the other.
pub fn last_path() -> Option<Path> { LAST.with(|l| l.get()) }

/// Operand sizes that fit the kernels' 32-bit indices and the buffers given.
fn fits(a: &Tensor, b: &Tensor, c: &Tensor, m: usize, k: usize, n: usize) -> bool {
    let small = |x: usize, y: usize| x.checked_mul(y).is_some_and(|p| u32::try_from(p).is_ok());
    m > 0 && k > 0 && n > 0 && small(m, k) && small(k, n) && small(m, n)
        && a.len() >= m * k && b.len() >= k * n && c.len() >= m * n
}

/// The tiled kernel. `a` holds A as `m x k` (or `k x m` when `ta`), `b` holds
/// B as `k x n` (or `n x k` when `tb`), `c` holds `m x n`. With `accumulate`
/// the product is added to C; otherwise C is assigned. `false` when it could
/// not be queued: no device, or sizes that do not fit.
#[allow(clippy::too_many_arguments)]
pub fn gemm_tiled(a: &Tensor, ta: bool, b: &Tensor, tb: bool, m: usize, k: usize, n: usize, c: &Tensor, accumulate: bool) -> bool {
    if !fits(a, b, c, m, k, n) { return false; }
    tiled_inner(a, ta, b, tb, m, k, n, c, accumulate).is_some()
}

#[allow(clippy::too_many_arguments)]
fn tiled_inner(a: &Tensor, ta: bool, b: &Tensor, tb: bool, m: usize, k: usize, n: usize, c: &Tensor, accumulate: bool) -> Option<()> {
    let ks = kernels().ok()?;
    let (mu, ku, nu) = (m as u32, k as u32, n as u32);
    let (ncols, nrows) = (nu.div_ceil(BN), mu.div_ceil(BM));
    let (ksplit, kchunk) = split(ncols.checked_mul(nrows)?, ku);
    let need = if ksplit > 1 { u32::try_from((ksplit as usize).checked_mul(m * n)?).ok()? as usize } else { 1 };
    let (lda, ldb) = (if ta { mu } else { ku }, if tb { ku } else { nu });

    let mut guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|t| t.len() < need) { *guard = Some(Tensor::zeros(need)?); }
    let part = guard.as_ref()?;
    let int = Arg::Int;
    ks.tiled.launch([ncols, nrows, ksplit], BLOCK, &[
        Arg::F(a), Arg::F(b), Arg::F(c), Arg::F(part),
        int(mu), int(ku), int(nu), int(lda), int(ldb), int(nu),
        int(ta as u32), int(tb as u32), int(accumulate as u32), int(ksplit), int(kchunk),
    ]).ok()?;
    if ksplit > 1 {
        ks.reduce.launch([(mu * nu).div_ceil(REDUCE_THREADS), 1, 1], [REDUCE_THREADS, 1, 1], &[
            Arg::F(part), Arg::F(c), int(mu), int(nu), int(ksplit), int(accumulate as u32),
        ]).ok()?;
    }
    Some(())
}

/// The same product through cuBLAS. `false` when it could not be queued,
/// which includes `libcublas.so.13` not being on this machine.
///
/// A row-major matrix is its transpose stored column-major, and
/// `(op(A) op(B))^T = op(B)^T op(A)^T`. So cuBLAS sees C as `n x m`, with
/// stored B as its first operand and stored A as its second, each transposed
/// exactly when the caller said so.
#[allow(clippy::too_many_arguments)]
pub fn gemm_cublas(a: &Tensor, ta: bool, b: &Tensor, tb: bool, m: usize, k: usize, n: usize, c: &Tensor, accumulate: bool) -> bool {
    if !fits(a, b, c, m, k, n) { return false; }
    let Ok(blas) = blas() else { return false };
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    blas.sgemm(tb, ta, n, m, k, b, ldb, a, lda, if accumulate { 1.0 } else { 0.0 }, c, n).is_ok()
}

/// The default path, by measurement on the Tesla T4 (`--example cuda_bench`,
/// 33 shape and transpose cases interleaved): cuBLAS 2.9-5.1 TFLOP/s, tiled
/// 1.4-3.6, cuBLAS ahead in every case by 1.2-3.2x. Its bits repeated across
/// two runs in all 33, but cuBLAS does not promise that, so a run that needs
/// the guarantee sets `R2_CUDA_GEMM=tiled`.
const DEFAULT_PATH: Path = Path::Cublas;

fn wanted() -> Path {
    // read per call, not cached: a benchmark alternates inside one window,
    // and this is microseconds against a GEMM
    match std::env::var("R2_CUDA_GEMM").as_deref() {
        Ok("tiled") => Path::Tiled,
        Ok("cublas") => Path::Cublas,
        _ => DEFAULT_PATH,
    }
}

/// `C (=|+=) op(A) · op(B)` on the device, by the wanted path, else the other
/// one. `false` (C untouched) when neither could run. Matches `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(a: &Tensor, ta: bool, b: &Tensor, tb: bool, m: usize, k: usize, n: usize, c: &Tensor, accumulate: bool) -> bool {
    let run = |p: Path| match p {
        Path::Tiled => gemm_tiled(a, ta, b, tb, m, k, n, c, accumulate),
        Path::Cublas => gemm_cublas(a, ta, b, tb, m, k, n, c, accumulate),
    };
    let first = wanted();
    let second = if first == Path::Tiled { Path::Cublas } else { Path::Tiled };
    for p in [first, second] {
        if run(p) { LAST.with(|l| l.set(Some(p))); return true; }
    }
    false
}

/// Host-slice convenience: upload, multiply, download. `None` on any failure,
/// including a fault found at the download. Matches `gemm::sgemm`.
pub fn sgemm(a: &[f32], ta: bool, b: &[f32], tb: bool, m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    let (da, db, dc) = (Tensor::upload(a)?, Tensor::upload(b)?, Tensor::zeros(m * n)?);
    if !gemm(&da, ta, &db, tb, m, k, n, &dc, false) { return None; }
    dc.to_vec().ok()
}

/// A(m×k)·B(k×n) → C(m×n); `None` on any failure so the caller keeps its CPU path.
pub fn try_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if a.len() != m.checked_mul(k)? || b.len() != k.checked_mul(n)? { return None; }
    sgemm(a, false, b, false, m, k, n)
}

/// A matrix that lives on the device across calls.
pub struct Resident { t: Tensor, pub rows: usize, pub cols: usize }

/// Upload a matrix once and keep it on the device.
pub fn upload(data: &[f32], rows: usize, cols: usize) -> Option<Resident> {
    if data.len() != rows.checked_mul(cols)? { return None; }
    Some(Resident { t: Tensor::upload(data)?, rows, cols })
}

/// A(m×k) · B(k×n) where B is already on the device.
pub fn matmul_resident(a: &[f32], b: &Resident, m: usize) -> Option<Vec<f32>> {
    let (k, n) = (b.rows, b.cols);
    if a.len() != m.checked_mul(k)? { return None; }
    let (ta, tc) = (Tensor::upload(a)?, Tensor::zeros(m.checked_mul(n)?)?);
    if !gemm(&ta, false, &b.t, false, m, k, n, &tc, false) { return None; }
    tc.to_vec().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::have_device;
    use r2_linalg::gemm::{sgemm as cpu_sgemm, Trans};

    fn mk(n: usize, ph: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.031 + ph).sin()).collect() }

    type Gemm = fn(&Tensor, bool, &Tensor, bool, usize, usize, usize, &Tensor, bool) -> bool;

    /// The wgpu test's shapes, which hit every ragged edge and split-K, in all
    /// four transpose cases, assign and accumulate, against the CPU kernel.
    fn matches_the_cpu(f: Gemm, what: &str) {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (64, 16, 64), (65, 17, 66), (7, 300, 19), (130, 70, 200),
                            (256, 2048, 128), (200, 768, 300), (256, 2048, 256), (700, 40, 640)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
                let (a, b) = (mk(m * k, 0.0), mk(k * n, 1.7));
                let t = |x: bool| if x { Trans::Yes } else { Trans::No };
                let want = cpu_sgemm(&a, t(ta), &b, t(tb), m, k, n, false);
                let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
                let (da, db) = (Tensor::upload(&a).unwrap(), Tensor::upload(&b).unwrap());
                let dc = Tensor::zeros(m * n).unwrap();
                assert!(f(&da, ta, &db, tb, m, k, n, &dc, false), "{what}: not queued");
                let got = dc.download();
                assert_eq!(got.len(), want.len(), "{what}: the device faulted");
                for i in 0..want.len() {
                    assert!((got[i] - want[i]).abs() <= 1e-5 * scale,
                            "{what} {m}x{k}x{n} ta={ta} tb={tb} at {i}: device {} vs cpu {}", got[i], want[i]);
                }
                assert!(f(&da, ta, &db, tb, m, k, n, &dc, true), "{what}: accumulate not queued");
                let twice = dc.download();
                for i in 0..want.len() {
                    assert!((twice[i] - 2.0 * want[i]).abs() <= 2e-5 * scale, "{what} accumulate {m}x{k}x{n} at {i}");
                }
            }
        }
    }

    #[test]
    fn tiled_matches_the_cpu_kernel() {
        if !have_device() { return; }
        matches_the_cpu(gemm_tiled, "tiled");
    }

    #[test]
    fn cublas_matches_the_cpu_kernel() {
        if !have_device() { return; }
        if let Err(why) = blas() {
            assert!(std::env::var_os("R2_REQUIRE_CUDA").is_none(), "R2_REQUIRE_CUDA is set but: {why}");
            eprintln!("{why}; skipped");
            return;
        }
        matches_the_cpu(gemm_cublas, "cuBLAS");
    }

    /// Same inputs, same bits, every run: what the tiled kernel is built for.
    #[test]
    fn tiled_is_bit_reproducible() {
        if !have_device() { return; }
        let (m, k, n) = (200usize, 1000usize, 150usize);
        let (a, b) = (Tensor::upload(&mk(m * k, 0.2)).unwrap(), Tensor::upload(&mk(k * n, 2.1)).unwrap());
        let run = || {
            let c = Tensor::zeros(m * n).unwrap();
            assert!(gemm_tiled(&a, false, &b, true, m, k, n, &c, false));
            c.download()
        };
        let first = run();
        for _ in 0..3 { assert_eq!(run(), first); }
    }

    /// Sizes that do not match the buffers are refused on the host, before
    /// anything can fault the context.
    #[test]
    fn operands_too_small_are_refused() {
        if !have_device() { return; }
        let (a, b, c) = (Tensor::zeros(10).unwrap(), Tensor::zeros(10).unwrap(), Tensor::zeros(10).unwrap());
        assert!(!gemm(&a, false, &b, false, 4, 4, 4, &c, false));
        assert!(!gemm(&a, false, &b, false, 0, 4, 4, &c, false));
        assert!(try_matmul(&[1.0; 3], &[1.0; 4], 2, 2, 2).is_none());
    }

    #[test]
    fn resident_matmul_matches_matmul() {
        if !have_device() { return; }
        let (m, k, n) = (33usize, 40usize, 21usize);
        let (a, b) = (mk(m * k, 0.4), mk(k * n, 0.9));
        let r = upload(&b, k, n).unwrap();
        assert_eq!((r.rows, r.cols), (k, n));
        let (got, want) = (matmul_resident(&a, &r, m).unwrap(), try_matmul(&a, &b, m, k, n).unwrap());
        assert_eq!(got.len(), want.len());
        // not bit equality: this may be cuBLAS, which does not promise it
        assert!(got.iter().zip(&want).all(|(x, y)| (x - y).abs() <= 1e-5));
    }
}
