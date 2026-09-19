//! CPU reference kernels for the transformer op set (Layer 0).
//!
//! These are the ACCURACY TRUTH: every GPU kernel (Pillar 1 / Opus) must
//! match these within f32 tolerance, exactly as the differential harness
//! pins the statistical surface against R. Correctness first, tuned later.
//! All compute is f32 (the neural type); shapes are row-major.

/// Matrix multiply: A(m×k) · B(k×n) → C(m×n), all row-major f32.
/// Loop order is i-k-j so the inner loop walks B and C contiguously and
/// vectorizes; the j-inner form strides both and runs several times
/// slower. Rows of C are independent, so the work splits across cores
/// once it is large enough to pay for the hand-off — below that threshold
/// thread setup costs more than the arithmetic saves.
/// Backend choice comes from the ORACLE, not from a threshold hidden in
/// this function. One component decides where work runs, for every op in
/// the system, so tuning is a single edit rather than a hunt through
/// kernels — and `explain()` can report the same decision the code makes.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    match r2_oracle::dispatch(r2_oracle::Op::TensorMatMul, r2_oracle::Shape::nmk(m, n, k)) {
        r2_oracle::Backend::Gpu => {
            // The Oracle decides policy; the device may still be absent or
            // fail, so a `None` falls through to the CPU path below rather
            // than erroring. Correctness never depends on a GPU existing.
            if let Some(out) = r2_gpu::matmul(a, b, m, k, n) {
                return out;
            }
            matmul_cpu(a, b, m, k, n, true)
        }
        r2_oracle::Backend::Rayon  => matmul_cpu(a, b, m, k, n, true),
        r2_oracle::Backend::Serial => matmul_cpu(a, b, m, k, n, false),
    }
}

/// `out = A·B` into a caller-owned buffer whose contents are ignored — the
/// form the tape's buffer pool needs, so a recycled output buffer is used
/// without a zeroing pass. Same dispatch as [`matmul`].
pub fn matmul_into(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), m * n);
    use r2_linalg::gemm::{sgemm_assign_into, Trans};
    match r2_oracle::dispatch(r2_oracle::Op::TensorMatMul, r2_oracle::Shape::nmk(m, n, k)) {
        r2_oracle::Backend::Gpu => {
            if let Some(v) = r2_gpu::matmul(a, b, m, k, n) { out.copy_from_slice(&v); return; }
            sgemm_assign_into(a, Trans::No, b, Trans::No, m, k, n, out, true)
        }
        r2_oracle::Backend::Rayon  => sgemm_assign_into(a, Trans::No, b, Trans::No, m, k, n, out, true),
        r2_oracle::Backend::Serial => sgemm_assign_into(a, Trans::No, b, Trans::No, m, k, n, out, false),
    }
}

/// CPU matmul — BLAS `sgemm`, from `r2_linalg::gemm`.
///
/// This was an i-k-j triple loop. That loop order is right for a naive
/// kernel (the inner loop walks B and C contiguously) but it re-streams
/// all of B for every row of C, so it falls off a cliff the moment B stops
/// fitting cache: measured 63-73 GFLOP/s on the small shapes and 26.8 on
/// the output head, where B is 256x8000x4B = 8 MB = exactly this machine's
/// L3. `gemm` packs instead, and dispatches an AVX2+FMA micro-kernel at
/// runtime: 118-195 GFLOP/s on the same five shapes, 5.1x overall.
///
/// The old loop carried `if aik == 0.0 { continue; }` for one-hot
/// embedding rows. Nothing builds a one-hot any more — `llm.rs` and
/// `transformer.rs` both use `Op::Embed`'s gather — so the branch is gone
/// with the loop rather than being carried into a kernel where a
/// data-dependent branch per element would cost more than it saves.
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, parallel: bool)
    -> Vec<f32>
{
    r2_linalg::gemm::sgemm(a, r2_linalg::gemm::Trans::No,
                           b, r2_linalg::gemm::Trans::No, m, k, n, parallel)
}

/// RMSNorm over the last dim: y = x / sqrt(mean(x²) + eps) * weight.
/// (Llama-family normalization — no mean-subtraction, unlike LayerNorm.)
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    rmsnorm_into(x, weight, eps, &mut out);
    out
}

/// [`rmsnorm`] into a caller-owned buffer; every element is written.
pub fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let d = weight.len();
    let rows = x.len() / d;
    debug_assert_eq!(out.len(), x.len());
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = sum_sq4(row) / d as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..d { out[r * d + j] = row[j] * scale * weight[j]; }
    }
}

/// Numerically-stable softmax over the last dim (subtract row max).
pub fn softmax(x: &[f32], d: usize) -> Vec<f32> {
    let rows = x.len() / d;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        // `f32::max` lowers to `llvm.maxnum`, which LLVM CAN vectorise as a
        // reduction — unlike `+`, whose non-associativity blocks it. So this
        // fold is already branchless SIMD and the hand-rolled `max4` below,
        // which uses compare-and-branch, measured WORSE. Not every serial
        // reduction is the same problem.
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // The exp and the sum stay FUSED in one pass, deliberately.
        //
        // Splitting them so the sum could use `sum4` was tried and measured
        // 12% WORSE on the training step (31.05 -> 34.77 s): at vocab 8,000
        // this runs over 16.4M elements per step, so a second pass costs
        // ~64 MB of memory traffic, while the serial add it removes was
        // already hidden behind the latency of the `exp` beside it. A
        // dependency chain only costs anything when something else is not
        // already the bottleneck.
        let inv = 1.0 / exp_shift_sum(row, m, &mut out[r * d..r * d + d]);
        for j in 0..d { out[r * d + j] *= inv; }
    }
    out
}

// ── vectorised exp, and the ops that are made of it ─────────────────
//
// `f32::exp` is a scalar libm call. A training step makes 32.8M of them in
// `softmax_ce` (vocab 8,000 x 2,048 rows, forward and backward) and 12.6M
// in `silu` — together about 19% of a step at the sustained clock, and
// none of it vectorises, because a call is a call.
//
// PyTorch does not pay that: MKL's VML supplies a vectorised `expf`, and
// that is most of why its softmax and its activations are quick. The
// answer is the same one the GEMM micro-kernel needed — write the thing by
// hand with intrinsics — and it is the classical Cephes reduction:
//
//   n = round(x * log2 e),  r = x - n*ln2   (ln2 split hi/lo for accuracy)
//   exp(x) = 2^n * P(r),    P a degree-6 minimax polynomial on |r| < ln2/2
//   2^n built directly in the exponent field
//
// The functions below take SLICES rather than scalars on purpose. A
// per-element vector helper would have to be inlined across a crate
// boundary into a `#[target_feature]` caller, and that is exactly the
// inlining that silently failed earlier in this work. Keeping the whole
// loop inside one `#[target_feature]` function in this crate removes the
// question.

/// `exp` for eight floats, Cephes-style range reduction plus a degree-6
/// minimax polynomial. Accurate to about 1 ULP over the representable
/// range; inputs are clamped to +/-88.376 so `2^n` cannot overflow the
/// exponent field.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // UNDERFLOW. `2^n` is assembled by writing `n + 127` into the exponent
    // field, which is only valid while that stays in 1..=254. exp(x) goes
    // subnormal below x = ln(MIN_POSITIVE) = -87.3365, and there `n + 127`
    // reaches 0 and then negative — the shift walks into the sign bit and
    // returns -inf instead of a small number. Clamp the computation to the
    // normal range and force the lanes that were below it to zero, which
    // is the right answer for every caller here: a softmax term that far
    // below the row max contributes nothing.
    let orig = x;
    let lo = _mm256_set1_ps(-87.336_54);
    let x = _mm256_min_ps(_mm256_set1_ps(88.376_26), x);
    let x = _mm256_max_ps(lo, x);
    // n = round(x / ln2)
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    // r = x - n*ln2, with ln2 split so the subtraction stays exact
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_38), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    // P(r)
    let mut y = _mm256_set1_ps(1.987_569_1e-4);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.000_000_1e-1));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    // 2^n, assembled straight into the exponent bits
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
    let r = _mm256_mul_ps(y, pow2n);
    // Zero wherever the input was below the normal range.
    _mm256_andnot_ps(_mm256_cmp_ps(orig, lo, _CMP_LT_OQ), r)
}

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
    { false }
}

/// `out[j] = exp(x[j] - m)`, returning the sum. The exp-and-accumulate
/// pass of every softmax.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn exp_shift_sum_avx2(x: &[f32], m: f32, out: &mut [f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let mv = _mm256_set1_ps(m);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mv));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), e);
        acc = _mm256_add_ps(acc, e);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    while i < n {
        let e = (x[i] - m).exp();
        out[i] = e;
        sum += e;
        i += 1;
    }
    sum
}

/// `out[j] = exp(x[j] - m)`, returning the sum. Dispatches to AVX2.
pub fn exp_shift_sum(x: &[f32], m: f32, out: &mut [f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; `out` is at least
        // as long as `x`, which the debug assertion pins.
        debug_assert!(out.len() >= x.len());
        return unsafe { exp_shift_sum_avx2(x, m, out) };
    }
    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(x) { let e = (v - m).exp(); *o = e; sum += e; }
    sum
}

/// `Σ exp(x[j] - m)` without storing the exponentials — the second pass of
/// a log-sum-exp, where the probabilities themselves are not wanted.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn exp_shift_sum_only_avx2(x: &[f32], m: f32) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let mv = _mm256_set1_ps(m);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_add_ps(acc,
            exp8(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mv)));
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    while i < n { sum += (x[i] - m).exp(); i += 1; }
    sum
}

/// `Σ exp(x[j] - m)`. Dispatches to AVX2.
pub fn exp_shift_sum_only(x: &[f32], m: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { exp_shift_sum_only_avx2(x, m) };
    }
    let mut sum = 0.0f32;
    for &v in x { sum += (v - m).exp(); }
    sum
}

/// `dst[j] = silu(src[j])`, vectorised.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_into_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        // x / (1 + exp(-x))
        let d = _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), v)));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_div_ps(v, d));
        i += 8;
    }
    while i < n { dst[i] = silu(src[i]); i += 1; }
}

/// `dst[j] = silu(src[j])`. Dispatches to AVX2.
pub fn silu_into(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; `dst` is at least
        // as long as `src`.
        debug_assert!(dst.len() >= src.len());
        unsafe { silu_into_avx2(src, dst) };
        return;
    }
    for (d, &v) in dst.iter_mut().zip(src) { *d = silu(v); }
}

/// EXPERIMENT (measured by `--example silu_forms`): the silu backward in
/// PyTorch's expression order, `dy * s * (1 + x * (1 - s))`, writing the
/// result fresh (no accumulate) exactly as ATen's `silu_backward_kernel`
/// does. Same `exp8`, same 8-wide loop; only the arithmetic order and the
/// store differ. Kept `pub` so the example can call it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silu_bwd_torch_form_avx2(v: &[f32], g: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = v.len();
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= n {
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        let gv = _mm256_loadu_ps(g.as_ptr().add(i));
        let s = _mm256_div_ps(one,
            _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), vv))));
        // dy * s * (1 + x * (1 - s))   — ATen's order, no fusion
        let t = _mm256_sub_ps(one, s);
        let t = _mm256_mul_ps(vv, t);
        let t = _mm256_add_ps(one, t);
        let t = _mm256_mul_ps(s, t);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(gv, t));
        i += 8;
    }
    while i < n {
        let s = 1.0 / (1.0 + (-v[i]).exp());
        out[i] = g[i] * s * (1.0 + v[i] * (1.0 - s));
        i += 1;
    }
}

/// Runtime-dispatched wrapper for the experiment above.
pub fn silu_bwd_torch_form(v: &[f32], g: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        debug_assert!(g.len() >= v.len() && out.len() >= v.len());
        // SAFETY: guarded by the runtime feature check; lengths asserted.
        unsafe { silu_bwd_torch_form_avx2(v, g, out) };
        return;
    }
    for ((o, &vv), &gg) in out.iter_mut().zip(v).zip(g) {
        let s = 1.0 / (1.0 + (-vv).exp());
        *o = gg * s * (1.0 + vv * (1.0 - s));
    }
}

/// `gout[j] += g[j] * d/dv silu(v[j])`, vectorised.
///
/// `d/dv [v*s] = s + v*s*(1-s)` with `s = sigmoid(v)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn silu_bwd_avx2(v: &[f32], g: &[f32], gout: &mut [f32], accumulate: bool) {
    use std::arch::x86_64::*;
    let n = v.len();
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= n {
        let vv = _mm256_loadu_ps(v.as_ptr().add(i));
        let gv = _mm256_loadu_ps(g.as_ptr().add(i));
        let s = _mm256_div_ps(one,
            _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), vv))));
        // s + v*s*(1-s)
        let d = _mm256_fmadd_ps(_mm256_mul_ps(vv, s), _mm256_sub_ps(one, s), s);
        // An FMA with a zero addend rounds once, exactly as the multiply
        // alone would, so the assign form is bit-identical to accumulating
        // into a zeroed buffer — and needs no zeroed buffer.
        let acc = if accumulate { _mm256_loadu_ps(gout.as_ptr().add(i)) } else { _mm256_setzero_ps() };
        _mm256_storeu_ps(gout.as_mut_ptr().add(i), _mm256_fmadd_ps(gv, d, acc));
        i += 8;
    }
    while i < n {
        let s = 1.0 / (1.0 + (-v[i]).exp());
        let d = g[i] * (s + v[i] * s * (1.0 - s));
        if accumulate { gout[i] += d; } else { gout[i] = d; }
        i += 1;
    }
}

/// `gout[j] += g[j] * silu'(v[j])`. Dispatches to AVX2.
pub fn silu_bwd(v: &[f32], g: &[f32], gout: &mut [f32]) {
    silu_bwd_acc(v, g, gout, true)
}

/// [`silu_bwd`] with a choice: `accumulate` adds into `gout`, otherwise
/// `gout` is ASSIGNED and its prior contents are ignored.
pub fn silu_bwd_acc(v: &[f32], g: &[f32], gout: &mut [f32], accumulate: bool) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; all three slices
        // are the same length.
        debug_assert!(g.len() >= v.len() && gout.len() >= v.len());
        unsafe { silu_bwd_avx2(v, g, gout, accumulate) };
        return;
    }
    for ((go, &vv), &gg) in gout.iter_mut().zip(v).zip(g) {
        let s = 1.0 / (1.0 + (-vv).exp());
        let d = gg * (s + vv * s * (1.0 - s));
        if accumulate { *go += d; } else { *go = d; }
    }
}

/// `gout[j] += inv * (exp(x[j] - lse) - [j == t])` — softmax
/// cross-entropy's gradient for one row, vectorised.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_ce_grad_avx2(x: &[f32], lse: f32, t: usize, inv: f32, gout: &mut [f32], accumulate: bool) {
    use std::arch::x86_64::*;
    let n = x.len();
    let l = _mm256_set1_ps(lse);
    let iv = _mm256_set1_ps(inv);
    let mut i = 0;
    while i + 8 <= n {
        let p = exp8(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), l));
        let acc = if accumulate { _mm256_loadu_ps(gout.as_ptr().add(i)) } else { _mm256_setzero_ps() };
        _mm256_storeu_ps(gout.as_mut_ptr().add(i), _mm256_fmadd_ps(iv, p, acc));
        i += 8;
    }
    while i < n {
        let d = inv * (x[i] - lse).exp();
        if accumulate { gout[i] += d; } else { gout[i] = d; }
        i += 1;
    }
    // the one-hot term, applied once
    gout[t] -= inv;
}

/// Softmax cross-entropy's gradient for one row. Dispatches to AVX2.
pub fn softmax_ce_grad(x: &[f32], lse: f32, t: usize, inv: f32, gout: &mut [f32]) {
    softmax_ce_grad_acc(x, lse, t, inv, gout, true)
}

/// [`softmax_ce_grad`] with a choice: `accumulate` adds into `gout`,
/// otherwise `gout` is ASSIGNED and its prior contents are ignored.
pub fn softmax_ce_grad_acc(x: &[f32], lse: f32, t: usize, inv: f32, gout: &mut [f32], accumulate: bool) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        // SAFETY: guarded by the runtime feature check; `gout` matches `x`
        // and `t` indexes inside both.
        debug_assert!(gout.len() >= x.len() && t < x.len());
        unsafe { softmax_ce_grad_avx2(x, lse, t, inv, gout, accumulate) };
        return;
    }
    for (j, (go, &v)) in gout.iter_mut().zip(x).enumerate() {
        let d = inv * ((v - lse).exp() - if j == t { 1.0 } else { 0.0 });
        if accumulate { *go += d; } else { *go = d; }
    }
}

// ── vectorisable reductions ─────────────────────────────────────────
//
// `x.iter().map(|v| v*v).sum::<f32>()` and `(0..d).map(..).sum()` are
// SERIAL reductions: each step depends on
// the previous one. Float addition is not associative and `f32::max` has
// NaN-ordering semantics, so the compiler is NOT ALLOWED to reassociate
// either — which means it cannot vectorise them either. The whole chain
// runs one scalar op at a time, at the LATENCY of the unit rather than
// its throughput.
//
// NOT every reduction has this problem, and hand-rolling the ones that do
// not measured WORSE: `f32::max` lowers to `llvm.maxnum`, which LLVM CAN
// vectorise as a reduction, and a fused `exp`+`sum` loop already hides the
// chain behind the exp's latency. Only `+` over a plain product needs
// help, which is why only these two helpers survive.
//
// Splitting into four independent chains is a choice about evaluation
// order that only the author may make. These do make it, and they make it
// DETERMINISTICALLY: the four chains are strided by a fixed pattern and
// combined in a fixed sequence, so the result is identical run to run,
// thread to thread, and machine to machine. That property is why the
// tape's gradients stay bit-reproducible.
//
// Measured in the attention kernel, where the same change was made first:
// it is worth about 12% of a whole training step.


/// Sum of squares, with four independent accumulators. The mean of this
/// is what RMSNorm normalises by.
#[inline(always)]
pub fn sum_sq4(x: &[f32]) -> f32 {
    let mut a = [0.0f32; 4];
    let n = x.len();
    let full = n - n % 4;
    let mut i = 0;
    while i < full {
        a[0] += x[i] * x[i];
        a[1] += x[i + 1] * x[i + 1];
        a[2] += x[i + 2] * x[i + 2];
        a[3] += x[i + 3] * x[i + 3];
        i += 4;
    }
    let mut t = 0.0f32;
    while i < n { t += x[i] * x[i]; i += 1; }
    (a[0] + a[1]) + (a[2] + a[3]) + t
}


/// Three-way product reduction `Σ x·y·z`, four accumulators. RMSNorm's
/// backward needs exactly this shape.
#[inline(always)]
pub fn dot3_4(x: &[f32], y: &[f32], z: &[f32]) -> f32 {
    let n = x.len().min(y.len()).min(z.len());
    let mut a = [0.0f32; 4];
    let full = n - n % 4;
    let mut i = 0;
    while i < full {
        a[0] += x[i] * y[i] * z[i];
        a[1] += x[i + 1] * y[i + 1] * z[i + 1];
        a[2] += x[i + 2] * y[i + 2] * z[i + 2];
        a[3] += x[i + 3] * y[i + 3] * z[i + 3];
        i += 4;
    }
    let mut t = 0.0f32;
    while i < n { t += x[i] * y[i] * z[i]; i += 1; }
    (a[0] + a[1]) + (a[2] + a[3]) + t
}

/// Maximum, with four independent chains.
///

/// SiLU (a.k.a. swish): x * sigmoid(x). The SwiGLU activation half.
#[inline]
pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// SwiGLU FFN gate: elementwise silu(gate) * up. Llama-family FFN.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// Every `(cos, sin)` a RoPE pass needs, indexed `[pos * (head_dim/2) + p]`.
///
/// The angle `pos / base^(2p/head_dim)` depends only on the POSITION and
/// the PAIR — never on the data. [`rope_inplace`] recomputes a `powf` and a
/// `sin_cos` for every element it touches, which is correct and, called
/// per row per head, enormously redundant: at 2,048 rows, 4 heads and
/// head_dim 64 that is 262,144 transcendental pairs where only 2,048 are
/// distinct. Measured, RoPE was 138.9 ms of a ~957 ms training step —
/// 14.5%, the second-largest item after the GEMM — for arithmetic that is
/// two multiplies and two adds per pair.
///
/// Build this once per call and the inner loop becomes exactly that.
/// The values are computed by the same expressions in the same order as
/// `rope_inplace`, so a table-driven pass is bit-identical to it.
pub fn rope_table(period: usize, head_dim: usize, base: f32) -> Vec<(f32, f32)> {
    let half = head_dim / 2;
    let mut t = Vec::with_capacity(period.max(1) * half);
    for pos in 0..period.max(1) {
        for p in 0..half {
            let freq = 1.0 / base.powf(2.0 * p as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            let (s, c) = theta.sin_cos();
            t.push((c, s));
        }
    }
    t
}

/// Rotary position embedding (RoPE) applied in place to one head vector
/// of even dim `d` at position `pos`. Rotates (x[2i], x[2i+1]) pairs by
/// θ = pos / base^(2i/d). The de-facto positional scheme for modern LLMs.
pub fn rope_inplace(x: &mut [f32], pos: usize, base: f32) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        let freq = 1.0 / base.powf(2.0 * i as f32 / d as f32);
        let theta = pos as f32 * freq;
        let (s, c) = theta.sin_cos();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// Embedding lookup: gather rows `ids` from a (vocab × d) table.
pub fn embed(table: &[f32], ids: &[usize], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * d];
    for (t, &id) in ids.iter().enumerate() {
        out[t * d..t * d + d].copy_from_slice(&table[id * d..id * d + d]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vectorised `exp` must match the scalar one closely enough that
    /// no downstream gradient notices. Checked across the whole range the
    /// clamp admits, plus the awkward values.
    #[test]
    fn vectorised_exp_matches_scalar() {
        let mut xs: Vec<f32> = Vec::new();
        let mut v = -88.0f32;
        while v <= 88.0 { xs.push(v); v += 0.0173; }
        xs.extend_from_slice(&[0.0, -0.0, 1.0, -1.0, 1e-8, -1e-8, 88.3, -88.3,
                               -87.0, -87.4, -100.0, -1000.0, f32::NEG_INFINITY,
                               f32::MIN_POSITIVE, 0.6931472, -0.6931472]);
        let mut got = vec![0.0f32; xs.len()];
        // `exp_shift_sum` with m = 0 is plain exp, and it exercises both
        // the vector body and the scalar tail. Its SUM is not checked here
        // — summing exp(x) up to x = 88 overflows f32 by construction, and
        // every real caller subtracts the row max first precisely so that
        // it cannot. The sum is checked below on a bounded range instead.
        exp_shift_sum(&xs, 0.0, &mut got);
        let mut worst = 0.0f64;
        for (&x, &g) in xs.iter().zip(&got) {
            let want = x.exp();
            if want < f32::MIN_POSITIVE {
                // Below the normal range the kernel flushes to zero rather
                // than producing subnormals — stated, and checked.
                assert_eq!(g, 0.0, "exp({x}) should flush to zero, got {g}");
                continue;
            }
            let rel = ((g as f64 - want as f64) / want as f64).abs();
            if rel > worst { worst = rel; }
        }
        // ~1 ULP for f32 is 1.2e-7; allow a small multiple of it.
        assert!(worst < 1e-6, "vectorised exp is {worst:.3e} off the scalar one");

        // And the returned sum, on a range where it cannot overflow —
        // which is what a softmax always hands it.
        let row: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.019).sin() * 12.0).collect();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut out = vec![0.0f32; row.len()];
        let got_sum = exp_shift_sum(&row, m, &mut out);
        let want_sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
        assert!((got_sum - want_sum).abs() <= 1e-4 * want_sum,
                "exp_shift_sum total {got_sum} vs {want_sum}");
        assert!((exp_shift_sum_only(&row, m) - want_sum).abs() <= 1e-4 * want_sum,
                "exp_shift_sum_only disagrees with exp_shift_sum");
    }

    /// `silu_into` and `silu_bwd` must agree with the scalar definitions,
    /// including on a length that is not a multiple of the vector width.
    #[test]
    fn vectorised_silu_matches_scalar() {
        let v: Vec<f32> = (0..101).map(|i| (i as f32 - 50.0) * 0.37).collect();
        let g: Vec<f32> = (0..101).map(|i| ((i as f32) * 0.11).sin()).collect();
        let mut fwd = vec![0.0f32; v.len()];
        silu_into(&v, &mut fwd);
        for (i, (&a, &b)) in fwd.iter().zip(&v).enumerate() {
            let want = silu(b);
            assert!((a - want).abs() <= 1e-6 * (1.0 + want.abs()),
                    "silu_into[{i}]: {a} vs {want}");
        }
        let mut got = vec![0.0f32; v.len()];
        silu_bwd(&v, &g, &mut got);
        for (i, ((&gg, &vv), &go)) in g.iter().zip(&v).zip(&got).enumerate() {
            let s = 1.0 / (1.0 + (-vv).exp());
            let want = gg * (s + vv * s * (1.0 - s));
            assert!((go - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "silu_bwd[{i}]: {go} vs {want}");
        }
    }

    /// The cross-entropy gradient row, against the definition — and it
    /// ACCUMULATES, which a caller folding two terms together relies on.
    #[test]
    fn vectorised_softmax_ce_grad_matches_scalar() {
        let d = 97usize;
        let x: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.23).sin() * 3.0).collect();
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + x.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        let (t, inv) = (17usize, 0.25f32);
        let mut got = vec![1.0f32; d];
        softmax_ce_grad(&x, lse, t, inv, &mut got);
        for j in 0..d {
            let want = 1.0 + inv * ((x[j] - lse).exp() - if j == t { 1.0 } else { 0.0 });
            assert!((got[j] - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "softmax_ce_grad[{j}]: {} vs {want}", got[j]);
        }
    }


    #[test]
    fn matmul_identity_and_known() {
        // [[1,2],[3,4]] · I = itself
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let id = vec![1.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul(&a, &id, 2, 2, 2), a);
        // [1,2,3]·[4,5,6]^T style: (1×3)·(3×1) = 32
        let r = matmul(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 1, 3, 1);
        assert_eq!(r, vec![32.0]);
    }

    #[test]
    fn softmax_sums_to_one_and_stable() {
        let out = softmax(&[1.0, 2.0, 3.0], 3);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // stable on large inputs (no overflow)
        let big = softmax(&[1000.0, 1001.0, 1002.0], 3);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        let x = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = 3.5355
        let w = vec![1.0, 1.0];
        let y = rmsnorm(&x, &w, 0.0);
        let rms_out = ((y[0] * y[0] + y[1] * y[1]) / 2.0).sqrt();
        assert!((rms_out - 1.0).abs() < 1e-5, "rms {}", rms_out);
    }

    #[test]
    fn silu_and_swiglu_known() {
        assert!((silu(0.0)).abs() < 1e-7);           // silu(0)=0
        assert!((silu(1.0) - 0.7310586).abs() < 1e-5); // 1*sigmoid(1)
        let g = swiglu(&[0.0, 1.0], &[2.0, 3.0]);
        assert!((g[0]).abs() < 1e-6 && (g[1] - 0.7310586 * 3.0).abs() < 1e-4);
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation is orthogonal → preserves the vector norm.
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let n0 = x.iter().map(|v| v * v).sum::<f32>();
        rope_inplace(&mut x, 5, 10000.0);
        let n1 = x.iter().map(|v| v * v).sum::<f32>();
        assert!((n0 - n1).abs() < 1e-4, "norm {} vs {}", n0, n1);
        // pos 0 is identity
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        rope_inplace(&mut y, 0, 10000.0);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn embed_gathers_rows() {
        let table = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]; // 3 vocab × 2
        let out = embed(&table, &[2, 0], 2);
        assert_eq!(out, vec![30.0, 31.0, 10.0, 11.0]);
    }
}
