//! `sgemm` on the GPU: `C (=|+=) op(A) · op(B)`, the three cases a training
//! step runs (NN forward, NT `grad_A`, TN `grad_B`) in one kernel.
//!
//! # Shape of the kernel
//!
//! The CPU kernel's lessons carry over almost verbatim. A workgroup owns a
//! `BM x BN` = 64 x 64 tile of C and walks K in slabs of `BK` = 16: each
//! slab of A (64 x 16) and B (16 x 64) is staged in workgroup memory once
//! and then read by all 256 threads, and each thread keeps a `TM x TN` =
//! 4 x 4 register tile of C that it accumulates across the whole depth —
//! sixteen FMAs per eight shared-memory loads, which is what makes the
//! kernel arithmetic-bound rather than load-bound. The transposed cases
//! are handled where the slab is STAGED: the only difference between NN,
//! NT and TN is the index used to read the operand into shared memory, so
//! the inner loop is identical for all three and never strides.
//!
//! # Determinism
//!
//! Every element of C is produced by exactly one thread, which sums its
//! `K` products in a fixed order. No atomics, no split-K reductions. The
//! same inputs give the same bits on every run and on every device that
//! rounds FMA the same way (all of them) — the reproducibility cuBLAS
//! does not promise by default.
//!
//! # Accuracy
//!
//! Checked against `r2_linalg::gemm::sgemm` in the tests, at ragged
//! shapes, all three transpose cases, assign and accumulate. The CPU
//! kernel sums K in slabs of 256 with a different association, so the two
//! agree to f32 rounding, not bit for bit.

use crate::device::{gpu, Tensor};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Rows of C per workgroup.
pub const BM: u32 = 128;
/// Columns of C per workgroup.
pub const BN: u32 = 128;
/// Depth per staged slab.
pub const BK: u32 = 16;
/// Register tile per thread: TM rows x TN columns of C, held as
/// `TM * TN/4` named `vec4`s.
const TM: u32 = 8;
const TN: u32 = 8;
/// Threads per workgroup, fixed by the tile shape.
const THREADS: u32 = (BM / TM) * (BN / TN);

/// The kernel source. Generated rather than written out because the
/// register tile has to be NAMED variables: an `array<vec4>` indexed by a
/// loop counter went to scratch memory on AMD's compiler and ran at 14
/// GFLOP/s; the same arithmetic in named registers runs at 300. With the
/// accumulators, loads and stores unrolled here, the tile shape is a
/// constant to change rather than a kernel to rewrite.
fn shader() -> String {
    let (tm, tn4) = (TM as usize, (TN / 4) as usize);
    let tm4 = (TM / 4) as usize;
    let mut acc_decl = String::new();
    for i in 0..tm { for j in 0..tn4 { acc_decl += &format!("    var c{i}_{j} = vec4<f32>(0.0);\n"); } }
    let mut loads = String::new();
    for q in 0..tm4 { loads += &format!("            let a{q} = As[kk * (BM / 4u) + lid.y * {tm4}u + {q}u];\n"); }
    for j in 0..tn4 { loads += &format!("            let b{j} = Bs[kk * (BN / 4u) + lid.x * {tn4}u + {j}u];\n"); }
    let mut fmas = String::new();
    for i in 0..tm {
        let (q, comp) = (i / 4, ["x", "y", "z", "w"][i % 4]);
        for j in 0..tn4 {
            fmas += &format!("            c{i}_{j} = fma(vec4<f32>(a{q}.{comp}), b{j}, c{i}_{j});\n");
        }
    }
    let mut stores = String::new();
    for i in 0..tm {
        stores += &format!("    if (mbase + {i}u < d.m) {{\n");
        for j in 0..tn4 {
            for l in 0..4 {
                let comp = ["x", "y", "z", "w"][l];
                let col = j * 4 + l;
                stores += &format!(
                    "        if (nbase + {col}u < d.n) {{ let o = (mbase + {i}u) * d.ldc + nbase + {col}u; \
                     if (d.acc == 0u) {{ C[o] = c{i}_{j}.{comp}; }} else {{ C[o] = C[o] + c{i}_{j}.{comp}; }} }}\n");
            }
        }
        stores += "    }\n";
    }
    format!(r#"
struct Dims {{
    m: u32, k: u32, n: u32, lda: u32,
    ldb: u32, ldc: u32, ta: u32, tb: u32,
    acc: u32, pad0: u32, pad1: u32, pad2: u32,
}};
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const BM: u32 = {BM}u;
const BN: u32 = {BN}u;
const BK: u32 = {BK}u;
const TM: u32 = {TM}u;
const TN: u32 = {TN}u;
const THREADS: u32 = {THREADS}u;

// Slabs are stored k-major and packed four wide — As[k][m/4], Bs[k][n/4]
// — so a thread's TM rows of A and TN columns of B for one k are TM/4
// and TN/4 vec4 loads.
var<workgroup> As: array<vec4<f32>, {AS4}u>;
var<workgroup> Bs: array<vec4<f32>, {BS4}u>;

@compute @workgroup_size({WX}, {WY}, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {{
    let m0 = wg.y * BM;
    let n0 = wg.x * BN;
{acc_decl}
    for (var k0 = 0u; k0 < d.k; k0 = k0 + BK) {{
        // ── stage A: BM x BK elements, {LOADS_A} per thread. The slab
        // index is walked so that consecutive threads read consecutive
        // addresses of A whichever way A is stored: along k for NN, along
        // m for TN. ──
        for (var t = 0u; t < {LOADS_A}u; t = t + 1u) {{
            let idx = tid + t * THREADS;
            var r: u32; var c: u32;
            if (d.ta == 0u) {{ r = idx / BK; c = idx % BK; }} else {{ r = idx % BM; c = idx / BM; }}
            let mm = m0 + r;
            let kk = k0 + c;
            var v = 0.0;
            if (mm < d.m && kk < d.k) {{
                if (d.ta == 0u) {{ v = A[mm * d.lda + kk]; }} else {{ v = A[kk * d.lda + mm]; }}
            }}
            As[c * (BM / 4u) + r / 4u][r % 4u] = v;
        }}
        // ── stage B: BK x BN elements, {LOADS_B} per thread; along n for
        // NN, along k for NT ──
        for (var t = 0u; t < {LOADS_B}u; t = t + 1u) {{
            let idx = tid + t * THREADS;
            var r: u32; var c: u32;
            if (d.tb == 0u) {{ r = idx / BN; c = idx % BN; }} else {{ r = idx % BK; c = idx / BK; }}
            let kk = k0 + r;
            let nn = n0 + c;
            var v = 0.0;
            if (kk < d.k && nn < d.n) {{
                if (d.tb == 0u) {{ v = B[kk * d.ldb + nn]; }} else {{ v = B[nn * d.ldb + kk]; }}
            }}
            Bs[r * (BN / 4u) + c / 4u][c % 4u] = v;
        }}
        workgroupBarrier();

        // ── the TM x TN register tile over this slab ──
        for (var kk = 0u; kk < BK; kk = kk + 1u) {{
{loads}{fmas}        }}
        workgroupBarrier();
    }}

    // ── write the live part of the tile ──
    let mbase = m0 + lid.y * TM;
    let nbase = n0 + lid.x * TN;
{stores}}}
"#,
        BM = BM, BN = BN, BK = BK, TM = TM, TN = TN, THREADS = THREADS,
        WX = BN / TN, WY = BM / TM,
        AS4 = BK * BM / 4, BS4 = BK * BN / 4,
        LOADS_A = (BM * BK) / THREADS,
        LOADS_B = (BK * BN) / THREADS,
        acc_decl = acc_decl, loads = loads, fmas = fmas, stores = stores,
    )
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

static PIPELINE: OnceLock<Option<Pipeline>> = OnceLock::new();

fn pipeline() -> Option<&'static Pipeline> {
    PIPELINE.get_or_init(|| {
        let g = gpu()?;
        let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2gpu-sgemm"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        let pipeline = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("r2gpu-sgemm"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Some(Pipeline { pipeline, layout })
    }).as_ref()
}

/// `C (=|+=) op(A) · op(B)` on the device, row-major.
///
/// `a` holds A as `m x k` (or `k x m` when `ta`), `b` holds B as `k x n`
/// (or `n x k` when `tb`), `c` holds `m x n`. With `accumulate` the
/// product is added to C's current contents; otherwise C is assigned and
/// its prior contents are ignored — the same pair of entry points as the
/// CPU `sgemm_into` / `sgemm_assign_into`. Returns `false` (and leaves C
/// untouched) when no device is available.
pub fn gemm(a: &Tensor, ta: bool, b: &Tensor, tb: bool,
            m: usize, k: usize, n: usize, c: &Tensor, accumulate: bool) -> bool {
    debug_assert!(a.len >= m * k && b.len >= k * n && c.len >= m * n);
    let (Some(g), Some(p)) = (gpu(), pipeline()) else { return false };
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    let dims: [u32; 12] = [m as u32, k as u32, n as u32, lda as u32,
                           ldb as u32, n as u32, ta as u32, tb as u32,
                           accumulate as u32, 0, 0, 0];
    let ubuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-sgemm-dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(BN), (m as u32).div_ceil(BM), 1);
    }
    g.queue.submit(Some(enc.finish()));
    true
}

/// Host-slice convenience: upload, multiply, download. For tests and
/// one-off calls; a training loop keeps its tensors resident and calls
/// [`gemm`] directly.
pub fn sgemm(a: &[f32], ta: bool, b: &[f32], tb: bool,
             m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    let (ta_, tb_, tc) = (Tensor::upload(a)?, Tensor::upload(b)?, Tensor::zeros(m * n)?);
    if !gemm(&ta_, ta, &tb_, tb, m, k, n, &tc, false) { return None; }
    Some(tc.download())
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_linalg::gemm::{sgemm as cpu_sgemm, Trans};

    fn mk(n: usize, ph: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.031 + ph).sin()).collect()
    }

    /// Every transpose case at shapes that exercise every ragged edge —
    /// dimensions that are not multiples of BM, BN or BK — against the
    /// CPU kernel, and the accumulate path on top of an assign.
    #[test]
    fn gpu_sgemm_matches_the_cpu_kernel() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (64, 16, 64), (65, 17, 66),
                            (7, 300, 19), (130, 70, 200), (256, 2048, 128), (200, 768, 300)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
                let a = mk(m * k, 0.0);
                let b = mk(k * n, 1.7);
                let want = cpu_sgemm(&a, if ta { Trans::Yes } else { Trans::No },
                                     &b, if tb { Trans::Yes } else { Trans::No }, m, k, n, false);
                let got = sgemm(&a, ta, &b, tb, m, k, n).expect("gpu sgemm");
                assert_eq!(got.len(), want.len());
                let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
                for i in 0..want.len() {
                    assert!((got[i] - want[i]).abs() <= 1e-5 * scale,
                            "{m}x{k}x{n} ta={ta} tb={tb} at {i}: gpu {} vs cpu {}", got[i], want[i]);
                }
                // accumulate: C = want + want
                let ta_ = Tensor::upload(&a).unwrap();
                let tb_ = Tensor::upload(&b).unwrap();
                let tc = Tensor::upload(&want).unwrap();
                assert!(gemm(&ta_, ta, &tb_, tb, m, k, n, &tc, true));
                let twice = tc.download();
                for i in 0..want.len() {
                    assert!((twice[i] - 2.0 * want[i]).abs() <= 2e-5 * scale,
                            "accumulate {m}x{k}x{n} at {i}");
                }
            }
        }
    }

    /// Same inputs, same bits, every run: the determinism the kernel is
    /// built for.
    #[test]
    fn gpu_sgemm_is_bit_reproducible() {
        if gpu().is_none() { return; }
        let (m, k, n) = (200usize, 1000usize, 150usize);
        let a = mk(m * k, 0.2);
        let b = mk(k * n, 2.1);
        let first = sgemm(&a, false, &b, true, m, k, n).unwrap();
        for _ in 0..3 {
            assert_eq!(sgemm(&a, false, &b, true, m, k, n).unwrap(), first);
        }
    }
}
