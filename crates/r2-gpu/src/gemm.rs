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
//! `K` products in a fixed order. No atomics. When the output has too few
//! tiles to fill the device (`grad_B` of a 256-wide projection is a
//! 256 x 256 result over a depth of 2,048: four tiles on six CUs, and it
//! ran at 53-110 GFLOP/s), the depth is split across workgroups — but
//! each split writes its own partial tile and a second pass sums the
//! partials in split order, so the result is still a fixed-order sum.
//! The same inputs give the same bits on every run and on every device
//! that rounds FMA the same way (all of them) — the reproducibility
//! cuBLAS does not promise by default.
//!
//! Split-K was chosen against a 64 x 64 tile and a register-prefetching
//! slab loop, all three interleaved in one process: the small tile gained
//! nothing over the split, and prefetching cost 25-30% everywhere (the
//! slot tables and the staged registers spilled the occupancy the big
//! tile lives on).
//!
//! # Storage types
//!
//! A and B may be f16 (`Tensor::dtype`); they are converted to f32 as
//! they are staged, and the accumulators are f32 whatever the operands.
//! C is written in its own type. One kernel per (A, B, C) signature,
//! compiled on first use. f16 operands halve the bytes a GEMM reads —
//! the point of mixed precision on a bandwidth-bound device — and are
//! what a cooperative-matrix kernel will consume.
//!
//! # Accuracy
//!
//! Checked against `r2_linalg::gemm::sgemm` in the tests, at ragged
//! shapes, all three transpose cases, assign and accumulate. The CPU
//! kernel sums K in slabs of 256 with a different association, so the two
//! agree to f32 rounding, not bit for bit.

use crate::device::{gpu, Dtype, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;
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
/// The operand and result types of one compiled kernel: A and B are read
/// (f32 or f16, converted to f32 as they are staged — the arithmetic and
/// the accumulators are always f32), C is written in its own type. The
/// split-K partials stay f32.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Sig { a: Dtype, b: Dtype, c: Dtype }

fn shader(sig: Sig) -> String {
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
                     if (d.ksplit > 1u) {{ P[wg.z * d.m * d.n + o] = c{i}_{j}.{comp}; }} \
                     else if (d.acc == 0u) {{ C[o] = {tc}(c{i}_{j}.{comp}); }} else {{ C[o] = {tc}(f32(C[o]) + c{i}_{j}.{comp}); }} }}\n",
                    tc = sig.c.wgsl());
            }
        }
        stores += "    }\n";
    }
    let enable = if [sig.a, sig.b, sig.c].contains(&Dtype::F16) { "enable f16;\n" } else { "" };
    format!(r#"{enable}
struct Dims {{
    m: u32, k: u32, n: u32, lda: u32,
    ldb: u32, ldc: u32, ta: u32, tb: u32,
    acc: u32, ksplit: u32, kchunk: u32, pad2: u32,
}};
@group(0) @binding(0) var<storage, read> A: array<{ta}>;
@group(0) @binding(1) var<storage, read> B: array<{tb}>;
@group(0) @binding(2) var<storage, read_write> C: array<{tc}>;
@group(0) @binding(3) var<uniform> d: Dims;
// split-K partials, [ksplit][m][n]; written instead of C when ksplit > 1
@group(0) @binding(4) var<storage, read_write> P: array<f32>;

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
    // this workgroup's share of the depth (all of it when ksplit == 1)
    let kbeg = wg.z * d.kchunk;
    let kend = min(kbeg + d.kchunk, d.k);
{acc_decl}
    for (var k0 = kbeg; k0 < kend; k0 = k0 + BK) {{
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
            if (mm < d.m && kk < kend) {{
                if (d.ta == 0u) {{ v = f32(A[mm * d.lda + kk]); }} else {{ v = f32(A[kk * d.lda + mm]); }}
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
            if (kk < kend && nn < d.n) {{
                if (d.tb == 0u) {{ v = f32(B[kk * d.ldb + nn]); }} else {{ v = f32(B[nn * d.ldb + kk]); }}
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
        enable = enable, ta = sig.a.wgsl(), tb = sig.b.wgsl(), tc = sig.c.wgsl(),
        WX = BN / TN, WY = BM / TM,
        AS4 = BK * BM / 4, BS4 = BK * BN / 4,
        LOADS_A = (BM * BK) / THREADS,
        LOADS_B = (BK * BN) / THREADS,
        acc_decl = acc_decl, loads = loads, fmas = fmas, stores = stores,
    )
}

/// The split-K reduction: `C (=|+=) Σ_z P[z]`, splits in order, one
/// thread per element — the fixed-order sum that keeps split-K
/// deterministic.
fn reduce_shader(c: Dtype) -> String {
    format!(r#"{enable}
struct Dims {{
    m: u32, k: u32, n: u32, lda: u32,
    ldb: u32, ldc: u32, ta: u32, tb: u32,
    acc: u32, ksplit: u32, kchunk: u32, pad2: u32,
}};
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<{tc}>;
@group(0) @binding(2) var<uniform> d: Dims;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let o = gid.x;
    let mn = d.m * d.n;
    if (o >= mn) {{ return; }}
    var s = 0.0;
    for (var z = 0u; z < d.ksplit; z = z + 1u) {{ s = s + P[z * mn + o]; }}
    if (d.acc == 0u) {{ C[o] = {tc}(s); }} else {{ C[o] = {tc}(f32(C[o]) + s); }}
}}
"#, enable = if c == Dtype::F16 { "enable f16;\n" } else { "" }, tc = c.wgsl())
}

/// Below this many output tiles the depth is split. 24 measured best of
/// {none, 24, 48} interleaved: 48 started to cost the shapes that were
/// already fine (2048 x 256 x 256 NN, 199 -> 158 GFLOP/s).
const SPLIT_BELOW: u32 = 24;
/// Never split the depth finer than this (a multiple of BK): a partial
/// tile costs a write and a read of m x n, and a chunk this deep pays
/// for it. It also means a depth of 256 or less is never split.
const MIN_CHUNK: u32 = 256;

/// How many ways to split the depth `k` for an output of `tiles`
/// workgroups, and the chunk each takes.
fn split(tiles: u32, k: u32) -> (u32, u32) {
    if tiles >= SPLIT_BELOW || k <= MIN_CHUNK { return (1, k.max(1)); }
    let want = SPLIT_BELOW.div_ceil(tiles);
    let chunk = k.div_ceil(want).max(MIN_CHUNK).div_ceil(BK) * BK;
    (k.div_ceil(chunk), chunk)
}

struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The GEMM and the split-K reduction, one pair per type signature,
/// compiled on first use.
static PIPELINES: Mutex<Option<HashMap<Sig, &'static [Pipeline; 2]>>> = Mutex::new(None);

fn pipelines(sig: Sig) -> Option<&'static [Pipeline; 2]> {
    let mut guard = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(p) = table.get(&sig) { return Some(p); }
    let p: &'static [Pipeline; 2] = Box::leak(Box::new([build(&shader(sig))?, build(&reduce_shader(sig.c))?]));
    table.insert(sig, p);
    Some(p)
}

fn build(src: &str) -> Option<Pipeline> {
    let g = gpu()?;
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("r2gpu-sgemm"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
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
}

/// The split-K partials buffer, grown on demand and reused: queue order
/// keeps one call's partials safe from the next.
static SCRATCH: std::sync::Mutex<Option<Tensor>> = std::sync::Mutex::new(None);

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
    let sig = Sig { a: a.dtype, b: b.dtype, c: c.dtype };
    let (Some(g), Some(ps)) = (gpu(), pipelines(sig)) else { return false };
    let p = &ps[0];
    let tiles = (m as u32).div_ceil(BM) * (n as u32).div_ceil(BN);
    let (ksplit, kchunk) = split(tiles, k as u32);
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    let dims: [u32; 12] = [m as u32, k as u32, n as u32, lda as u32,
                           ldb as u32, n as u32, ta as u32, tb as u32,
                           accumulate as u32, ksplit, kchunk, 0];
    let need = if ksplit > 1 { ksplit as usize * m * n } else { 1 };
    let mut scratch = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if scratch.as_ref().map(|t| t.len < need).unwrap_or(true) {
        let Some(t) = Tensor::zeros(need) else { return false };
        *scratch = Some(t);
    }
    let part = scratch.as_ref().unwrap();
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
            wgpu::BindGroupEntry { binding: 4, resource: part.buf.as_entire_binding() },
        ],
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(BN), (m as u32).div_ceil(BM), ksplit);
    }
    if ksplit > 1 {
        let r = &ps[1];
        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &r.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: part.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: c.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(((m * n) as u32).div_ceil(256), 1, 1);
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
                            (7, 300, 19), (130, 70, 200), (256, 2048, 128), (200, 768, 300),
                            (256, 2048, 256), (700, 40, 640)] {  // split-K (4 tiles, depth 2048) and unsplit
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
    /// f16 operands and an f16 result against the f32 CPU kernel run on
    /// the SAME rounded inputs: the only differences are f32 accumulation
    /// order and the final f16 rounding of C.
    #[test]
    fn gpu_sgemm_reads_f16_operands_and_writes_either_type() {
        if gpu().map(|g| g.f16) != Some(true) { eprintln!("no f16 on this adapter; skipped"); return; }
        use crate::half::{f16_to_f32, f32_to_f16};
        let round = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect() };
        for &(m, k, n) in &[(65usize, 130usize, 70usize), (256, 2048, 256), (200, 768, 300)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false)] {
                let a = round(&mk(m * k, 0.3)); let b = round(&mk(k * n, 1.7));
                let want = cpu_sgemm(&a, if ta { Trans::Yes } else { Trans::No }, &b, if tb { Trans::Yes } else { Trans::No }, m, k, n, true);
                let (da, db) = (Tensor::upload_as(&a, Dtype::F16).unwrap(), Tensor::upload_as(&b, Dtype::F16).unwrap());
                for cdt in [Dtype::F32, Dtype::F16] {
                    let dc = Tensor::zeros_as(m * n, cdt).unwrap();
                    assert!(gemm(&da, ta, &db, tb, m, k, n, &dc, false));
                    let got = dc.download();
                    let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
                    let tol = if cdt == Dtype::F16 { 2e-3 } else { 1e-5 };
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!((x - y).abs() <= tol * scale, "{m}x{k}x{n} ta {ta} tb {tb} C {cdt:?} [{i}]: {x} vs {y}");
                    }
                }
            }
        }
    }

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
