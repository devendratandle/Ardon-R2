//! Fused causal grouped-query attention on the GPU, forward and backward,
//! in flash form: nothing of size `seq x seq` is ever written.
//!
//! Layouts are the tape's (`r2_autograd::Op::Attention`): `q` is
//! `[nseq*seq][nh*hd]`, `k` and `v` are `[nseq*seq][nkv*hd]`, the output
//! and its gradient are shaped like `q`. `nh / nkv` query heads share one
//! kv head.
//!
//! # Forward
//!
//! One workgroup per (sequence, query head, block of `BQ` queries); one
//! thread per query. Key/value blocks of `BKV` rows are staged in
//! workgroup memory and every thread walks them with the online softmax —
//! running max `m`, running sum `l`, accumulator rescaled when the max
//! rises — then writes `o = acc / l` and the log-sum-exp `L = m + ln l`
//! the backward reuses. The causal limit is structural: a query block
//! visits key blocks up to its own diagonal only, and the diagonal block
//! is masked per element.
//!
//! # Backward (the FlashAttention-2 decomposition, without atomics)
//!
//! With `L` from the forward and `D_i = dO_i · O_i` from a small
//! pre-kernel, `P_ij = exp(S_ij - L_i)` is recomputed wherever it is
//! needed and never stored:
//!
//! ```text
//!   dV_j  = Σ_i P_ij dO_i                 (one thread per key)
//!   dS_ij = P_ij (dO_i·V_j − D_i)
//!   dK_j  = scale Σ_i dS_ij Q_i           (one thread per key)
//!   dQ_i  = scale Σ_j dS_ij K_j           (one thread per query)
//! ```
//!
//! Each output row is owned by exactly one thread, which sums its terms
//! in a fixed order — across query blocks, and across the heads of a GQA
//! group for dK/dV — so every result is bit-reproducible run to run.
//!
//! # What this costs, and what did not help
//!
//! Every thread re-reads the staged rows it pairs with — one workgroup-
//! memory load per FMA — so these kernels run at 20-30 GFLOP/s here
//! against the GEMM's 400: at the small model (32 x 64 tokens, hd 64)
//! the forward is 2.1 ms and the backward 14.8 (dV 3.9, dK 7.9, dQ 5.9),
//! 14% of a training step. Splitting the reduction axis into chunks with
//! partial slabs and a fixed-order sum (the GEMM's split-K, for more
//! workgroups) was measured interleaved at splits 1/2/4/8/16: 14.8, 16.2,
//! 19.6, 26.1, 40.7 ms — monotonically worse, the per-workgroup staging
//! and row loads outweighing any occupancy gained. Closed. What remains
//! is the register-tiled backward (`attn_tiled`'s shape for dV, dK, dQ),
//! which cuts the loads per FMA the way the GEMM does.
//!
//! # Registers, not arrays
//!
//! The head dimension is a compile-time constant of a generated shader,
//! one pipeline per `hd`, so every per-row vector (a query, an
//! accumulator, a gradient row) is a set of named `vec4` variables. A
//! dynamically indexed `array` in a thread's private storage is where the
//! GEMM lost a factor of twenty on this compiler.

use crate::device::{gpu, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// Queries per forward / dQ workgroup (one per thread).
const BQ: u32 = 64;
/// Keys per staged block, and per dK / dV workgroup (one per thread).
const BKV: u32 = 64;

/// Attention problem dimensions, shared by all four kernels.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub nseq: usize,
    pub seq: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub scale: f32,
}

struct Kernels {
    forward: wgpu::ComputePipeline,
    delta: wgpu::ComputePipeline,
    dv: wgpu::ComputePipeline,
    dk: wgpu::ComputePipeline,
    dq: wgpu::ComputePipeline,
}

/// Whether `attn_tiled`'s register-tiled forward is used where it
/// applies (`hd % 32 == 0`). Off: measured against this reference kernel
/// in one process it was SLOWER on this adapter — 2.24 vs 2.12 ms at
/// 32 x 64 tokens (4/2 heads, hd 64) and 11.70 vs 9.37 ms at 8 x 256
/// (12/4 heads) — its 128-thread workgroups and 23 KB of workgroup
/// memory leave the six CUs emptier than the reference's do. It stays as
/// the FlashAttention-2 shape to build the backward on for a device with
/// more to fill.
const TILED_FORWARD: bool = false;

static KERNELS: Mutex<Option<HashMap<usize, &'static Kernels>>> = Mutex::new(None);

fn kernels(hd: usize) -> Option<&'static Kernels> {
    let mut guard = KERNELS.lock().ok()?;
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(k) = table.get(&hd) { return Some(k); }
    let g = gpu()?;
    let make = |label: &str, src: String| -> wgpu::ComputePipeline {
        let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        })
    };
    let k: &'static Kernels = Box::leak(Box::new(Kernels {
        forward: make("r2gpu-attn-fwd", if TILED_FORWARD && crate::attn_tiled::supports(hd) { crate::attn_tiled::forward_wgsl(hd) } else { forward_wgsl(hd) }),
        delta: make("r2gpu-attn-delta", delta_wgsl(hd)),
        dv: make("r2gpu-attn-dv", dkv_wgsl(hd, false)),
        dk: make("r2gpu-attn-dk", dkv_wgsl(hd, true)),
        dq: make("r2gpu-attn-dq", dq_wgsl(hd)),
    }));
    table.insert(hd, k);
    Some(k)
}

// ── shader generation ────────────────────────────────────────────────────

/// The uniform block and bindings every kernel shares.
fn header(hd: usize, bindings: &[(&str, &str)]) -> String {
    let mut s = String::from(
        "struct Dims { nseq: u32, seq: u32, nh: u32, nkv: u32, hd: u32, group: u32, pad0: u32, pad1: u32, scale: f32, pad2: f32, pad3: f32, pad4: f32 };\n");
    for (i, (name, access)) in bindings.iter().enumerate() {
        s += &format!("@group(0) @binding({i}) var<storage, {access}> {name}: array<f32>;\n");
    }
    s += &format!("@group(0) @binding({}) var<uniform> d: Dims;\n", bindings.len());
    s += &format!("const HD: u32 = {hd}u;\nconst V4: u32 = {}u;\nconst BQ: u32 = {BQ}u;\nconst BKV: u32 = {BKV}u;\n", hd / 4);
    s
}

/// `let name{i} = vec4(src[off + 4i .. +4])` for i in 0..hd/4.
fn load_row(name: &str, src: &str, off: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!(
        "    let {name}{i} = vec4<f32>({src}[{off} + {a}u], {src}[{off} + {b}u], {src}[{off} + {c}u], {src}[{off} + {d}u]);\n",
        a = 4 * i, b = 4 * i + 1, c = 4 * i + 2, d = 4 * i + 3)).collect()
}

/// `var name{i} = vec4(0)` for i in 0..hd/4.
fn zero_row(name: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!("    var {name}{i} = vec4<f32>(0.0);\n")).collect()
}

/// A workgroup-memory row `ws[base + 4i..]` read as vec4s into `name{i}`.
fn load_ws_row(name: &str, ws: &str, base: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!("        let {name}{i} = {ws}[({base}) / 4u + {i}u];\n")).collect()
}

/// `Σ_i dot(a{i}, b{i})` as an expression.
fn dot(a: &str, b: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!("dot({a}{i}, {b}{i})")).collect::<Vec<_>>().join(" + ")
}

/// `acc{i} = fma(vec4(s), row{i}, acc{i})` for all i.
fn axpy(acc: &str, s: &str, row: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!("        {acc}{i} = fma(vec4<f32>({s}), {row}{i}, {acc}{i});\n")).collect()
}

/// `acc{i} = acc{i} * s`.
fn scale_row(acc: &str, s: &str, hd: usize) -> String {
    (0..hd / 4).map(|i| format!("            {acc}{i} = {acc}{i} * {s};\n")).collect()
}

/// Store `row{i}` (optionally times `mul`) to `dst[off + 4i..]`.
fn store_row(dst: &str, off: &str, row: &str, mul: &str, hd: usize) -> String {
    let mut s = String::new();
    for i in 0..hd / 4 {
        for (l, c) in ["x", "y", "z", "w"].iter().enumerate() {
            s += &format!("    {dst}[{off} + {}u] = {row}{i}.{c}{mul};\n", 4 * i + l);
        }
    }
    s
}

/// Stage `rows` rows of `hd` floats from `src` (row `r` at
/// `src[(base + r) * stride + col0 ..]`) into `ws`, cooperatively.
fn stage(ws: &str, src: &str, rows: u32, hd: usize, threads: u32) -> String {
    let total = rows as usize * hd / 4;
    let per = total.div_ceil(threads as usize);
    format!(r#"
        for (var t = 0u; t < {per}u; t = t + 1u) {{
            let idx = tid + t * {threads}u;
            if (idx < {total}u) {{
                let r = idx / V4;
                let c4 = idx % V4;
                let row = kb * BKV + r;
                if (row < d.seq) {{
                    let off = (base + row) * kw + kvh * HD + c4 * 4u;
                    {ws}[idx] = vec4<f32>({src}[off], {src}[off + 1u], {src}[off + 2u], {src}[off + 3u]);
                }} else {{
                    {ws}[idx] = vec4<f32>(0.0);
                }}
            }}
        }}
"#)
}

fn forward_wgsl(hd: usize) -> String {
    let mut s = header(hd, &[("Q", "read"), ("K", "read"), ("V", "read"), ("O", "read_write"), ("L", "read_write")]);
    s += &format!("var<workgroup> Ks: array<vec4<f32>, {n}u>;\nvar<workgroup> Vs: array<vec4<f32>, {n}u>;\n", n = BKV as usize * hd / 4);
    s += &format!(r#"
@compute @workgroup_size({BQ}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let sq = wg.z;                       // sequence
    let qh = wg.y;                       // query head
    let kvh = qh / d.group;
    let qb = wg.x;                       // query block
    let base = sq * d.seq;
    let qw = d.nh * HD;
    let kw = d.nkv * HD;
    let qi = qb * BQ + tid;              // this thread's query (may be past seq)
    let live = qi < d.seq;
    let qoff = (base + min(qi, d.seq - 1u)) * qw + qh * HD;
"#);
    s += &load_row("q", "Q", "qoff", hd);
    s += &zero_row("acc", hd);
    s += r#"
    var m = -3.0e38;
    var l = 0.0;
    // key blocks up to and including this query block's diagonal
    for (var kb = 0u; kb <= qb; kb = kb + 1u) {
"#;
    s += &stage("Ks", "K", BKV, hd, BQ);
    s += &stage("Vs", "V", BKV, hd, BQ);
    s += r#"
        workgroupBarrier();
        let jmax = min(BKV, d.seq - kb * BKV);
        for (var j = 0u; j < jmax; j = j + 1u) {
            let kj = kb * BKV + j;
            if (live && kj <= qi) {
"#;
    s += &load_ws_row("k", "Ks", "j * HD", hd);
    s += &format!("        let sc = ({}) * d.scale;\n", dot("q", "k", hd));
    s += r#"
        if (sc > m) {
            let corr = exp(m - sc);
            l = l * corr;
"#;
    s += &scale_row("acc", "corr", hd);
    s += r#"
            m = sc;
        }
        let p = exp(sc - m);
        l = l + p;
"#;
    s += &load_ws_row("v", "Vs", "j * HD", hd);
    s += &axpy("acc", "p", "v", hd);
    s += r#"
            }
        }
        workgroupBarrier();
    }
    if (live) {
        let inv = 1.0 / l;
"#;
    s += &store_row("O", "qoff", "acc", " * inv", hd);
    s += r#"
        L[(base + qi) * d.nh + qh] = m + log(l);
    }
}
"#;
    s
}

/// `D_i = dO_i · O_i`, one thread per (row, head).
fn delta_wgsl(hd: usize) -> String {
    let mut s = header(hd, &[("O", "read"), ("dO", "read"), ("Dl", "read_write")]);
    s += &format!(r#"
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;                       // row * nh + head
    let rows = d.nseq * d.seq;
    if (idx >= rows * d.nh) {{ return; }}
    let row = idx / d.nh;
    let h = idx % d.nh;
    let off = row * d.nh * HD + h * HD;
"#);
    s += &load_row("o", "O", "off", hd);
    s += &load_row("g", "dO", "off", hd);
    s += &format!("    Dl[idx] = {};\n}}\n", dot("o", "g", hd));
    s
}

/// dV (`dk = false`) or dK (`dk = true`): one thread per key of a kv head,
/// looping over every query head of the group and every query block at
/// or after the key's block.
fn dkv_wgsl(hd: usize, dk: bool) -> String {
    let out = if dk { "dK" } else { "dV" };
    let mut s = header(hd, &[("Q", "read"), ("K", "read"), ("V", "read"), ("dO", "read"),
                             ("L", "read"), ("Dl", "read"), (out, "read_write")]);
    s += &format!("var<workgroup> Qs: array<vec4<f32>, {n}u>;\nvar<workgroup> Gs: array<vec4<f32>, {n}u>;\nvar<workgroup> Ls: array<f32, {BQ}u>;\nvar<workgroup> Ds: array<f32, {BQ}u>;\n", n = BQ as usize * hd / 4);
    s += &format!(r#"
@compute @workgroup_size({BKV}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let sq = wg.z;
    let kvh = wg.y;
    let kb = wg.x;                       // key block
    let base = sq * d.seq;
    let qw = d.nh * HD;
    let kw = d.nkv * HD;
    let kj = kb * BKV + tid;             // this thread's key
    let live = kj < d.seq;
    let koff = (base + min(kj, d.seq - 1u)) * kw + kvh * HD;
"#);
    s += &load_row("k", "K", "koff", hd);
    s += &load_row("v", "V", "koff", hd);
    s += &zero_row("acc", hd);
    s += r#"
    let nqb = (d.seq + BQ - 1u) / BQ;
    for (var qh = kvh * d.group; qh < (kvh + 1u) * d.group; qh = qh + 1u) {
        for (var qb = kb; qb < nqb; qb = qb + 1u) {
            // stage the query block: Q, dO, L, D
            for (var t = 0u; t < V4; t = t + 1u) {
                let idx = tid + t * BQ;
                let r = idx / V4;
                let c4 = idx % V4;
                let row = qb * BQ + r;
                if (row < d.seq) {
                    let off = (base + row) * qw + qh * HD + c4 * 4u;
                    Qs[idx] = vec4<f32>(Q[off], Q[off + 1u], Q[off + 2u], Q[off + 3u]);
                    Gs[idx] = vec4<f32>(dO[off], dO[off + 1u], dO[off + 2u], dO[off + 3u]);
                } else {
                    Qs[idx] = vec4<f32>(0.0);
                    Gs[idx] = vec4<f32>(0.0);
                }
            }
            let lrow = qb * BQ + tid;
            if (lrow < d.seq) {
                Ls[tid] = L[(base + lrow) * d.nh + qh];
                Ds[tid] = Dl[(base + lrow) * d.nh + qh];
            }
            workgroupBarrier();
            let imax = min(BQ, d.seq - qb * BQ);
            for (var i = 0u; i < imax; i = i + 1u) {
                let qi = qb * BQ + i;
                if (live && kj <= qi) {
"#;
    s += &load_ws_row("q", "Qs", "i * HD", hd);
    s += &format!("        let p = exp(({}) * d.scale - Ls[i]);\n", dot("q", "k", hd));
    s += &load_ws_row("g", "Gs", "i * HD", hd);
    if dk {
        s += &format!("        let ds = p * (({}) - Ds[i]) * d.scale;\n", dot("g", "v", hd));
        s += &axpy("acc", "ds", "q", hd);
    } else {
        s += &axpy("acc", "p", "g", hd);
    }
    s += r#"
                }
            }
            workgroupBarrier();
        }
    }
    if (live) {
"#;
    s += &store_row(out, "koff", "acc", "", hd);
    s += "    }\n}\n";
    s
}

/// dQ: one thread per query, looping over the key blocks it can see.
fn dq_wgsl(hd: usize) -> String {
    let mut s = header(hd, &[("Q", "read"), ("K", "read"), ("V", "read"), ("dO", "read"),
                             ("L", "read"), ("Dl", "read"), ("dQ", "read_write")]);
    s += &format!("var<workgroup> Ks: array<vec4<f32>, {n}u>;\nvar<workgroup> Vs: array<vec4<f32>, {n}u>;\n", n = BKV as usize * hd / 4);
    s += &format!(r#"
@compute @workgroup_size({BQ}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let sq = wg.z;
    let qh = wg.y;
    let kvh = qh / d.group;
    let qb = wg.x;
    let base = sq * d.seq;
    let qw = d.nh * HD;
    let kw = d.nkv * HD;
    let qi = qb * BQ + tid;
    let live = qi < d.seq;
    let qoff = (base + min(qi, d.seq - 1u)) * qw + qh * HD;
    let li = L[(base + min(qi, d.seq - 1u)) * d.nh + qh];
    let di = Dl[(base + min(qi, d.seq - 1u)) * d.nh + qh];
"#);
    s += &load_row("q", "Q", "qoff", hd);
    s += &load_row("g", "dO", "qoff", hd);
    s += &zero_row("acc", hd);
    s += r#"
    for (var kb = 0u; kb <= qb; kb = kb + 1u) {
"#;
    s += &stage("Ks", "K", BKV, hd, BQ);
    s += &stage("Vs", "V", BKV, hd, BQ);
    s += r#"
        workgroupBarrier();
        let jmax = min(BKV, d.seq - kb * BKV);
        for (var j = 0u; j < jmax; j = j + 1u) {
            let kj = kb * BKV + j;
            if (live && kj <= qi) {
"#;
    s += &load_ws_row("k", "Ks", "j * HD", hd);
    s += &load_ws_row("v", "Vs", "j * HD", hd);
    s += &format!("        let p = exp(({}) * d.scale - li);\n", dot("q", "k", hd));
    s += &format!("        let ds = p * (({}) - di) * d.scale;\n", dot("g", "v", hd));
    s += &axpy("acc", "ds", "k", hd);
    s += r#"
            }
        }
        workgroupBarrier();
    }
    if (live) {
"#;
    s += &store_row("dQ", "qoff", "acc", "", hd);
    s += "    }\n}\n";
    s
}

// ── host side ────────────────────────────────────────────────────────────

fn dims_buffer(sh: &Shape) -> Option<wgpu::Buffer> {
    let g = gpu()?;
    let group = (sh.nh / sh.nkv) as u32;
    let words: [u32; 12] = [sh.nseq as u32, sh.seq as u32, sh.nh as u32, sh.nkv as u32,
                            sh.hd as u32, group, 0, 0, sh.scale.to_bits(), 0, 0, 0];
    Some(g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-attn-dims"),
        contents: bytemuck::cast_slice(&words),
        usage: wgpu::BufferUsages::UNIFORM,
    }))
}

fn launch(p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], dims: &wgpu::Buffer, wgs: (u32, u32, u32)) -> Option<()> {
    let g = gpu()?;
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: dims.as_entire_binding() });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &p.get_bind_group_layout(0), entries: &entries,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(wgs.0, wgs.1, wgs.2);
    }
    g.queue.submit(Some(enc.finish()));
    Some(())
}

fn valid(sh: &Shape) -> bool {
    sh.hd % 4 == 0 && sh.hd > 0 && sh.nkv > 0 && sh.nh % sh.nkv == 0 && sh.seq > 0 && sh.nseq > 0
}

/// Forward: `o` receives the attention output, `lse` (length
/// `nseq*seq*nh`) the per-query log-sum-exp for the backward. Returns
/// `false` and touches nothing when there is no device or the shape is
/// unsupported (`hd` must be a multiple of 4).
pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, sh: &Shape) -> bool {
    if !valid(sh) { return false; }
    let (Some(ks), Some(dims)) = (kernels(sh.hd), dims_buffer(sh)) else { return false };
    let qblocks = (sh.seq as u32).div_ceil(BQ);
    launch(&ks.forward, &[&q.buf, &k.buf, &v.buf, &o.buf, &lse.buf], &dims,
           (qblocks, sh.nh as u32, sh.nseq as u32)).is_some()
}

/// Backward from the upstream gradient `g` (shaped like `o`): writes
/// `dq`, `dk`, `dv` (assigned, not accumulated). `o` and `lse` are the
/// forward's outputs; `delta` is scratch of length `nseq*seq*nh`.
#[allow(clippy::too_many_arguments)]
pub fn backward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, g: &Tensor,
                delta: &Tensor, dq: &Tensor, dk: &Tensor, dv: &Tensor, sh: &Shape) -> bool {
    valid(sh) && backward_inner(q, k, v, o, lse, g, delta, dq, dk, dv, sh).is_some()
}

#[allow(clippy::too_many_arguments)]
fn backward_inner(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, g: &Tensor,
                  delta: &Tensor, dq: &Tensor, dk: &Tensor, dv: &Tensor, sh: &Shape) -> Option<()> {
    let (ks, dims) = (kernels(sh.hd)?, dims_buffer(sh)?);
    let rows = (sh.nseq * sh.seq * sh.nh) as u32;
    let qblocks = (sh.seq as u32).div_ceil(BQ);
    let kblocks = (sh.seq as u32).div_ceil(BKV);
    launch(&ks.delta, &[&o.buf, &g.buf, &delta.buf], &dims, (rows.div_ceil(64), 1, 1))?;
    let inputs = [&q.buf, &k.buf, &v.buf, &g.buf, &lse.buf, &delta.buf];
    let mut b = inputs.to_vec(); b.push(&dv.buf);
    launch(&ks.dv, &b, &dims, (kblocks, sh.nkv as u32, sh.nseq as u32))?;
    let mut b = inputs.to_vec(); b.push(&dk.buf);
    launch(&ks.dk, &b, &dims, (kblocks, sh.nkv as u32, sh.nseq as u32))?;
    let mut b = inputs.to_vec(); b.push(&dq.buf);
    launch(&ks.dq, &b, &dims, (qblocks, sh.nh as u32, sh.nseq as u32))?;
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_autograd::Tape;

    fn mk(n: usize, ph: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.8).collect()
    }

    /// Forward output and all three gradients against the tape's CPU
    /// attention, at shapes that cross the block boundaries (seq 65, 130),
    /// with GQA grouping, ragged sequences and batch > 1.
    #[test]
    fn gpu_attention_matches_the_tape() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        for &(nseq, seq, nh, nkv, hd) in &[(1usize, 8usize, 2usize, 1usize, 8usize),
                                           (2, 65, 4, 2, 16), (3, 130, 3, 3, 32), (2, 70, 4, 2, 64)] {
            let rows = nseq * seq;
            let (qv, kv, vv) = (mk(rows * nh * hd, 0.0), mk(rows * nkv * hd, 1.3), mk(rows * nkv * hd, 2.7));
            let gv = mk(rows * nh * hd, 0.9);
            let scale = 1.0 / (hd as f32).sqrt();
            let sh = Shape { nseq, seq, nh, nkv, hd, scale };

            let mut tape = Tape::new();
            let (q, k, v) = (tape.leaf(qv.clone(), true), tape.leaf(kv.clone(), true), tape.leaf(vv.clone(), true));
            let o = tape.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
            tape.backward_from(o, &gv);

            let (tq, tk, tv) = (Tensor::upload(&qv).unwrap(), Tensor::upload(&kv).unwrap(), Tensor::upload(&vv).unwrap());
            let to = Tensor::zeros(rows * nh * hd).unwrap();
            let tl = Tensor::zeros(rows * nh).unwrap();
            assert!(forward(&tq, &tk, &tv, &to, &tl, &sh));
            let tg = Tensor::upload(&gv).unwrap();
            let td = Tensor::zeros(rows * nh).unwrap();
            let (dq, dk, dv) = (Tensor::zeros(rows * nh * hd).unwrap(), Tensor::zeros(rows * nkv * hd).unwrap(), Tensor::zeros(rows * nkv * hd).unwrap());
            assert!(backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh));

            let close = |got: &[f32], want: &[f32], what: &str| {
                assert_eq!(got.len(), want.len(), "{what}: length");
                let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
                for (i, (a, b)) in got.iter().zip(want).enumerate() {
                    assert!((a - b).abs() <= 3e-5 * scale,
                            "{what}[{i}] gpu {a} vs cpu {b} (nseq {nseq} seq {seq} nh {nh} nkv {nkv} hd {hd})");
                }
            };
            close(&to.download(), tape.value(o), "output");
            close(&dq.download(), tape.grad(q), "grad_q");
            close(&dk.download(), tape.grad(k), "grad_k");
            close(&dv.download(), tape.grad(v), "grad_v");

            // the tiled forward, kept off the dispatch on this adapter,
            // still has to agree
            if crate::attn_tiled::supports(hd) {
                let g = gpu().unwrap();
                let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("r2gpu-attn-fwd-tiled"),
                    source: wgpu::ShaderSource::Wgsl(crate::attn_tiled::forward_wgsl(hd).into()),
                });
                let tiled = g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("r2gpu-attn-fwd-tiled"), layout: None, module: &module,
                    entry_point: Some("main"), compilation_options: Default::default(), cache: None,
                });
                let (to2, tl2) = (Tensor::zeros(rows * nh * hd).unwrap(), Tensor::zeros(rows * nh).unwrap());
                let dims = dims_buffer(&sh).unwrap();
                launch(&tiled, &[&tq.buf, &tk.buf, &tv.buf, &to2.buf, &tl2.buf], &dims,
                       ((seq as u32).div_ceil(BQ), nh as u32, nseq as u32)).unwrap();
                close(&to2.download(), tape.value(o), "tiled output");
                close(&tl2.download(), &tl.download(), "tiled lse");
            }
        }
    }
}
