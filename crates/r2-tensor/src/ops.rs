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

/// CPU matmul. Loop order is i-k-j so the inner loop walks B and C
/// contiguously and vectorizes; the j-inner form strides both and runs
/// several times slower. Rows of C are independent, so they split across
/// cores when `parallel`.
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, parallel: bool)
    -> Vec<f32>
{
    use rayon::prelude::*;
    let mut c = vec![0.0f32; m * n];
    let row = |i: usize, crow: &mut [f32]| {
        for p in 0..k {
            let aik = a[i * k + p];
            if aik == 0.0 { continue; }   // one-hot embedding rows are mostly zero
            let brow = &b[p * n..p * n + n];
            for j in 0..n { crow[j] += aik * brow[j]; }
        }
    };
    if parallel {
        c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| row(i, crow));
    } else {
        c.chunks_mut(n).enumerate().for_each(|(i, crow)| row(i, crow));
    }
    c
}

/// RMSNorm over the last dim: y = x / sqrt(mean(x²) + eps) * weight.
/// (Llama-family normalization — no mean-subtraction, unlike LayerNorm.)
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let d = weight.len();
    let rows = x.len() / d;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
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
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..d { let e = (row[j] - m).exp(); out[r * d + j] = e; sum += e; }
        let inv = 1.0 / sum;
        for j in 0..d { out[r * d + j] *= inv; }
    }
    out
}

/// SiLU (a.k.a. swish): x * sigmoid(x). The SwiGLU activation half.
#[inline]
pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// SwiGLU FFN gate: elementwise silu(gate) * up. Llama-family FFN.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
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
