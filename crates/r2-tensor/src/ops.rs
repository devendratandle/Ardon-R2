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
    let d = weight.len();
    let rows = x.len() / d;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = sum_sq4(row) / d as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..d { out[r * d + j] = row[j] * scale * weight[j]; }
    }
    out
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
        let mut sum = 0.0f32;
        for j in 0..d { let e = (row[j] - m).exp(); out[r * d + j] = e; sum += e; }
        let inv = 1.0 / sum;
        for j in 0..d { out[r * d + j] *= inv; }
    }
    out
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
