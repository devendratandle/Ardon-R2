//! Level-1 kernels for the decompositions' inner loops, compiled 4 doubles
//! wide where the CPU has AVX2.
//!
//! The workspace builds for baseline x86-64 (SSE2, two doubles per
//! instruction, no FMA), so a plain Rust loop runs 2 wide unless it is
//! reached through a `#[target_feature]` function. Each kernel here is one
//! `#[inline(always)]` body compiled twice — baseline and AVX2 — and picked
//! at runtime. No FMA is enabled: every multiply and add keeps its own
//! rounding, so a kernel returns the same bits on either path.
//!
//! Measured first on the symmetric-eigen rotations (i5-12500, n = 1000):
//! the tridiagonal QR iteration 105 -> 69 ms from this alone.

/// AVX2 present (resolved once).
#[inline]
pub(crate) fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

macro_rules! dispatch {
    ($name:ident, $avx:ident, $imp:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        #[inline]
        pub(crate) fn $name($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            if avx2() {
                // SAFETY: avx2() confirmed AVX2 on this CPU.
                return unsafe { $avx($($arg),*) };
            }
            $imp($($arg),*)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn $avx($($arg: $ty),*) -> $ret { $imp($($arg),*) }
    };
}

dispatch!(dot, dot_avx2, dot_impl, (x: &[f64], y: &[f64]) -> f64);
dispatch!(axpy, axpy_avx2, axpy_impl, (y: &mut [f64], a: f64, x: &[f64]) -> ());
dispatch!(rank2, rank2_avx2, rank2_impl, (c: &mut [f64], v: &[f64], w: &[f64], wj: f64, vj: f64) -> ());

/// x·y over the shorter length, four accumulators so the adds overlap.
#[inline(always)]
fn dot_impl(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mut s = [0.0f64; 4];
    let (xc, yc) = (x.chunks_exact(4), y.chunks_exact(4));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        s[0] += a[0] * b[0]; s[1] += a[1] * b[1]; s[2] += a[2] * b[2]; s[3] += a[3] * b[3];
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (a, b) in xr.iter().zip(yr) { t += a * b; }
    t
}

/// y += a·x.
#[inline(always)]
fn axpy_impl(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) { *yi += a * xi; }
}

/// c −= v·wj + w·vj — one column of a symmetric rank-2 update.
#[inline(always)]
fn rank2_impl(c: &mut [f64], v: &[f64], w: &[f64], wj: f64, vj: f64) {
    for ((ci, vi), wi) in c.iter_mut().zip(v).zip(w) { *ci -= vi * wj + wi * vj; }
}
