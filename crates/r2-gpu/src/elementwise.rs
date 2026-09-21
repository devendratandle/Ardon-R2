//! The elementwise tail of a transformer step on the GPU: rmsnorm, SiLU,
//! products and sums, RoPE, softmax cross-entropy and the embedding
//! gather — forward and backward — one launch each.
//!
//! On the CPU these are the "bandwidth tail": ~20% of a step moving bytes
//! twice. Here each is a single pass over its operands, and the backward
//! arms take an `assign` flag exactly as the tape's first-writer rule
//! does, so gradient buffers are never zeroed.
//!
//! # Determinism
//!
//! Row reductions (rmsnorm's mean square, softmax-CE's log-sum-exp) are
//! fixed-order tree reductions in workgroup memory. The two backward ops
//! that scatter on a CPU — rmsnorm's `dw` (a sum over rows per column)
//! and the embedding table's gradient (a sum over the positions of each
//! token) — are written as GATHERS: one thread owns one output element and
//! sums its contributions in order. No atomics anywhere, so every result
//! is bit-reproducible.
//!
//! # Accuracy
//!
//! Checked against the CPU kernels in `r2_tensor::ops` and the tape's
//! backward arms in the tests; `exp`, `sin` and `cos` differ from the
//! CPU's by a few ULP, so the comparisons are to f32 rounding.

use crate::device::{gpu, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// Threads per workgroup for the row kernels (one workgroup per row).
const ROW_THREADS: u32 = 256;
/// Threads per workgroup for the flat kernels (one element per thread).
const FLAT: u32 = 256;

static PIPES: Mutex<Option<HashMap<&'static str, &'static wgpu::ComputePipeline>>> = Mutex::new(None);

fn pipeline(name: &'static str, src: fn() -> String) -> Option<&'static wgpu::ComputePipeline> {
    let mut guard = PIPES.lock().ok()?;
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(p) = table.get(name) { return Some(p); }
    let g = gpu()?;
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name), source: wgpu::ShaderSource::Wgsl(src().into()),
    });
    let p: &'static wgpu::ComputePipeline = Box::leak(Box::new(
        g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        })));
    table.insert(name, p);
    Some(p)
}

/// Every kernel's uniform block: eight integers and four floats, meaning
/// per kernel.
const PARAMS: &str = "struct P { n: u32, d: u32, rows: u32, a: u32, b: u32, c: u32, e: u32, f: u32, x: f32, y: f32, z: f32, w: f32 };\n";

fn launch(p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], ints: [u32; 8], floats: [f32; 4], groups: u32) -> Option<()> {
    let g = gpu()?;
    let mut words = [0u32; 12];
    words[..8].copy_from_slice(&ints);
    for (i, f) in floats.iter().enumerate() { words[8 + i] = f.to_bits(); }
    let ubuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-params"), contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: ubuf.as_entire_binding() });
    let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &p.get_bind_group_layout(0), entries: &entries,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    g.queue.submit(Some(enc.finish()));
    Some(())
}

fn bindings(names: &[(&str, &str)]) -> String {
    let mut s = String::from(PARAMS);
    for (i, (name, access)) in names.iter().enumerate() {
        s += &format!("@group(0) @binding({i}) var<storage, {access}> {name}: array<{ty}>;\n",
                      ty = if *name == "ids" || *name == "offs" || *name == "pos" { "u32" } else { "f32" });
    }
    s += &format!("@group(0) @binding({}) var<uniform> p: P;\n", names.len());
    s
}

/// A fixed-order sum of `red[0..ROW_THREADS]` into `red[0]` (workgroup
/// memory), halving each round; the caller has written `red[tid]`.
fn tree_sum() -> String {
    format!(r#"
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = red[tid] + red[tid + s]; }}
        workgroupBarrier();
    }}
"#, half = ROW_THREADS / 2)
}

// ── shaders ─────────────────────────────────────────────────────────────

fn rmsnorm_fwd_src() -> String {
    bindings(&[("X", "read"), ("W", "read"), ("Y", "read_write")]) + &format!(r#"
var<workgroup> red: array<f32, {T}u>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let r = wg.x;
    var ss = 0.0;
    for (var j = tid; j < p.d; j = j + {T}u) {{ let v = X[r * p.d + j]; ss = fma(v, v, ss); }}
    red[tid] = ss;
{tree}
    let scale = 1.0 / sqrt(red[0] / f32(p.d) + p.x);
    for (var j = tid; j < p.d; j = j + {T}u) {{ Y[r * p.d + j] = X[r * p.d + j] * scale * W[j]; }}
}}
"#, T = ROW_THREADS, tree = tree_sum())
}

/// dX (assign or accumulate) and the per-row `rinv` the dW pass needs.
fn rmsnorm_bwd_x_src() -> String {
    bindings(&[("X", "read"), ("W", "read"), ("G", "read"), ("GX", "read_write"), ("RI", "read_write")]) + &format!(r#"
var<workgroup> red: array<f32, {T}u>;
var<workgroup> red2: array<f32, {T}u>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let r = wg.x;
    var ss = 0.0;
    var sg = 0.0;
    for (var j = tid; j < p.d; j = j + {T}u) {{
        let v = X[r * p.d + j];
        ss = fma(v, v, ss);
        sg = fma(G[r * p.d + j] * W[j], v, sg);
    }}
    red[tid] = ss;
    red2[tid] = sg;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = red[tid] + red[tid + s]; red2[tid] = red2[tid] + red2[tid + s]; }}
        workgroupBarrier();
    }}
    let ri = 1.0 / sqrt(red[0] / f32(p.d) + p.x);
    let coef = ri * ri * ri / f32(p.d) * red2[0];
    if (tid == 0u) {{ RI[r] = ri; }}
    for (var j = tid; j < p.d; j = j + {T}u) {{
        let o = r * p.d + j;
        let dx = G[o] * W[j] * ri - coef * X[o];
        if (p.a == 0u) {{ GX[o] = dx; }} else {{ GX[o] = GX[o] + dx; }}
    }}
}}
"#, T = ROW_THREADS, half = ROW_THREADS / 2)
}

/// dW[j] += Σ_r G[r][j] X[r][j] rinv[r] — one thread per column, rows in
/// order.
fn rmsnorm_bwd_w_src() -> String {
    bindings(&[("X", "read"), ("G", "read"), ("RI", "read"), ("GW", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let j = gid.x;
    if (j >= p.d) {{ return; }}
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {{ acc = fma(G[r * p.d + j] * X[r * p.d + j], RI[r], acc); }}
    if (p.a == 0u) {{ GW[j] = acc; }} else {{ GW[j] = GW[j] + acc; }}
}}
"#, T = FLAT)
}

/// `Y = f(A, B)` over `n` elements; `p.b` selects the op:
/// 0 silu(A), 1 A*B, 2 A+B, 3 silu(A)*B.
fn flat_fwd_src() -> String {
    bindings(&[("A", "read"), ("B", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let a = A[i];
    var y = 0.0;
    switch p.b {{
        case 0u: {{ y = a / (1.0 + exp(-a)); }}
        case 1u: {{ y = a * B[i]; }}
        case 2u: {{ y = a + B[i]; }}
        default: {{ y = a / (1.0 + exp(-a)) * B[i]; }}
    }}
    Y[i] = y;
}}
"#, T = FLAT)
}

/// Backward of the flat ops. `p.b`: 0 silu → GA (=|+=) G·silu'(A);
/// 1 mul → GA (=|+=) G·B and GB (=|+=) G·A; 2 add → GA, GB (=|+=) G.
/// `p.a` / `p.c` are the assign flags for GA / GB; `p.e` / `p.f` say
/// whether each is wanted at all.
fn flat_bwd_src() -> String {
    bindings(&[("A", "read"), ("B", "read"), ("G", "read"), ("GA", "read_write"), ("GB", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let g = G[i];
    var da = 0.0;
    var db = 0.0;
    switch p.b {{
        case 0u: {{ let a = A[i]; let s = 1.0 / (1.0 + exp(-a)); da = g * (s + a * s * (1.0 - s)); }}
        case 1u: {{ da = g * B[i]; db = g * A[i]; }}
        default: {{ da = g; db = g; }}
    }}
    if (p.e == 1u) {{ if (p.a == 0u) {{ GA[i] = da; }} else {{ GA[i] = GA[i] + da; }} }}
    if (p.f == 1u) {{ if (p.c == 0u) {{ GB[i] = db; }} else {{ GB[i] = GB[i] + db; }} }}
}}
"#, T = FLAT)
}

/// RoPE, one thread per (row, head, pair). `p.b` = 1 for the backward
/// (the inverse rotation), `p.a` the assign flag (backward only).
fn rope_src() -> String {
    bindings(&[("X", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let half = p.d / 2u;                    // pairs per head; p.d = head_dim
    let i = gid.x;                          // (row, head, pair)
    let total = p.rows * p.c * half;        // p.c = heads
    if (i >= total) {{ return; }}
    let pr = i % half;
    let h = (i / half) % p.c;
    let r = i / (half * p.c);
    let pos = f32(r % p.a);                 // p.a = period
    let freq = 1.0 / pow(p.x, 2.0 * f32(pr) / f32(p.d));
    let th = pos * freq;
    let c = cos(th);
    let s = sin(th);
    let o = r * p.c * p.d + h * p.d + 2u * pr;
    let a = X[o];
    let b = X[o + 1u];
    if (p.b == 0u) {{
        Y[o] = a * c - b * s;
        Y[o + 1u] = a * s + b * c;
    }} else {{
        let ra = a * c + b * s;
        let rb = -a * s + b * c;
        if (p.e == 0u) {{ Y[o] = ra; Y[o + 1u] = rb; }} else {{ Y[o] = Y[o] + ra; Y[o + 1u] = Y[o + 1u] + rb; }}
    }}
}}
"#, T = FLAT)
}

/// Per row: `LSE[r] = logsumexp(row)`, `LOSS[r] = LSE[r] - x[t]`.
fn softmax_ce_fwd_src() -> String {
    bindings(&[("X", "read"), ("ids", "read"), ("LSE", "read_write"), ("LOSS", "read_write")]) + &format!(r#"
var<workgroup> red: array<f32, {T}u>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let r = wg.x;
    var m = -3.0e38;
    for (var j = tid; j < p.d; j = j + {T}u) {{ m = max(m, X[r * p.d + j]); }}
    red[tid] = m;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = max(red[tid], red[tid + s]); }}
        workgroupBarrier();
    }}
    let mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = tid; j < p.d; j = j + {T}u) {{ sum = sum + exp(X[r * p.d + j] - mx); }}
    red[tid] = sum;
{tree}
    if (tid == 0u) {{
        let lse = mx + log(red[0]);
        LSE[r] = lse;
        LOSS[r] = lse - X[r * p.d + ids[r]];
    }}
}}
"#, T = ROW_THREADS, half = ROW_THREADS / 2, tree = tree_sum())
}

/// `GX (=|+=) inv * (exp(x - lse) - [j == t])`.
fn softmax_ce_bwd_src() -> String {
    bindings(&[("X", "read"), ("ids", "read"), ("LSE", "read"), ("GX", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let r = i / p.d;
    let j = i % p.d;
    var v = exp(X[i] - LSE[r]);
    if (j == ids[r]) {{ v = v - 1.0; }}
    v = v * p.x;
    if (p.a == 0u) {{ GX[i] = v; }} else {{ GX[i] = GX[i] + v; }}
}}
"#, T = FLAT)
}

/// `Y[i][:] = T[ids[i]][:]`.
fn embed_fwd_src() -> String {
    bindings(&[("T", "read"), ("ids", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let row = i / p.d;
    let j = i % p.d;
    Y[i] = T[ids[row] * p.d + j];
}}
"#, T = FLAT)
}

/// `GT[v][j] (=|+=) Σ_{{i : ids[i] = v}} G[i][j]`, the positions of each
/// token given as CSR (`offs[v]..offs[v+1]` index into `pos`), built on
/// the host per batch. One thread per (v, j); rows in order.
fn embed_bwd_src() -> String {
    bindings(&[("G", "read"), ("offs", "read"), ("pos", "read"), ("GT", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;                          // v * d + j
    if (i >= p.n) {{ return; }}
    let v = i / p.d;
    let j = i % p.d;
    var acc = 0.0;
    for (var k = offs[v]; k < offs[v + 1u]; k = k + 1u) {{ acc = acc + G[pos[k] * p.d + j]; }}
    if (p.a == 0u) {{ GT[i] = acc; }} else {{ GT[i] = GT[i] + acc; }}
}}
"#, T = FLAT)
}

// ── host API ────────────────────────────────────────────────────────────

fn ceil_div(n: usize, d: u32) -> u32 { (n as u32).div_ceil(d) }

/// `y[r] = x[r] / rms(x[r]) * w`, rows of `d`.
pub fn rmsnorm_fwd(x: &Tensor, w: &Tensor, y: &Tensor, rows: usize, d: usize, eps: f32) -> bool {
    let Some(p) = pipeline("rmsnorm_fwd", rmsnorm_fwd_src) else { return false };
    launch(p, &[&x.buf, &w.buf, &y.buf], [0, d as u32, rows as u32, 0, 0, 0, 0, 0], [eps, 0.0, 0.0, 0.0], rows as u32).is_some()
}

/// rmsnorm backward: `gx (=|+=)` per `assign_x`, `gw (=|+=)` per
/// `assign_w`; `rinv` is scratch of length `rows`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bwd(x: &Tensor, w: &Tensor, g: &Tensor, gx: &Tensor, gw: &Tensor, rinv: &Tensor,
                   rows: usize, d: usize, eps: f32, assign_x: bool, assign_w: bool) -> bool {
    let (Some(px), Some(pw)) = (pipeline("rmsnorm_bwd_x", rmsnorm_bwd_x_src), pipeline("rmsnorm_bwd_w", rmsnorm_bwd_w_src)) else { return false };
    launch(px, &[&x.buf, &w.buf, &g.buf, &gx.buf, &rinv.buf], [0, d as u32, rows as u32, (!assign_x) as u32, 0, 0, 0, 0], [eps, 0.0, 0.0, 0.0], rows as u32).is_some()
        && launch(pw, &[&x.buf, &g.buf, &rinv.buf, &gw.buf], [0, d as u32, rows as u32, (!assign_w) as u32, 0, 0, 0, 0], [0.0; 4], ceil_div(d, FLAT)).is_some()
}

/// The flat forward ops.
#[derive(Clone, Copy)]
pub enum Flat { Silu, Mul, Add, SiluMul }

/// `y = op(a, b)` over `n` elements (`b` is ignored for `Silu`).
pub fn flat_fwd(op: Flat, a: &Tensor, b: &Tensor, y: &Tensor, n: usize) -> bool {
    let Some(p) = pipeline("flat_fwd", flat_fwd_src) else { return false };
    let code = match op { Flat::Silu => 0, Flat::Mul => 1, Flat::Add => 2, Flat::SiluMul => 3 };
    launch(p, &[&a.buf, &b.buf, &y.buf], [n as u32, 0, 0, 0, code, 0, 0, 0], [0.0; 4], ceil_div(n, FLAT)).is_some()
}

/// Backward of `Silu` / `Mul` / `Add`: `ga`, `gb` each `Some((buffer,
/// assign))` when wanted.
pub fn flat_bwd(op: Flat, a: &Tensor, b: &Tensor, g: &Tensor, ga: Option<(&Tensor, bool)>, gb: Option<(&Tensor, bool)>, n: usize) -> bool {
    let Some(p) = pipeline("flat_bwd", flat_bwd_src) else { return false };
    let code = match op { Flat::Silu => 0, Flat::Mul => 1, Flat::Add => 2, Flat::SiluMul => 1 };
    // An output nobody wants still needs a distinct binding (binding `g`
    // twice, once read-only and once read-write, is a usage conflict), so
    // it gets a shared one-element scratch buffer that is never touched.
    static DUMMY: std::sync::OnceLock<Option<Tensor>> = std::sync::OnceLock::new();
    let Some(dummy) = DUMMY.get_or_init(|| Tensor::zeros(1)).as_ref() else { return false };
    let (gab, ga_acc, ga_on) = match ga { Some((t, assign)) => (&t.buf, (!assign) as u32, 1), None => (&dummy.buf, 0, 0) };
    let (gbb, gb_acc, gb_on) = match gb { Some((t, assign)) => (&t.buf, (!assign) as u32, 1), None => (&dummy.buf, 0, 0) };
    launch(p, &[&a.buf, &b.buf, &g.buf, gab, gbb], [n as u32, 0, 0, ga_acc, code, gb_acc, ga_on, gb_on], [0.0; 4], ceil_div(n, FLAT)).is_some()
}

/// RoPE forward: `y = rotate(x)`, positions restarting every `period`
/// rows; rows are `heads x head_dim` wide.
pub fn rope_fwd(x: &Tensor, y: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32) -> bool {
    let Some(p) = pipeline("rope", rope_src) else { return false };
    let total = rows * heads * (head_dim / 2);
    launch(p, &[&x.buf, &y.buf], [0, head_dim as u32, rows as u32, period.max(1) as u32, 0, heads as u32, 0, 0], [base, 0.0, 0.0, 0.0], ceil_div(total, FLAT)).is_some()
}

/// RoPE backward: `gx (=|+=) rotate⁻¹(g)`.
pub fn rope_bwd(g: &Tensor, gx: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32, assign: bool) -> bool {
    let Some(p) = pipeline("rope", rope_src) else { return false };
    let total = rows * heads * (head_dim / 2);
    launch(p, &[&g.buf, &gx.buf], [0, head_dim as u32, rows as u32, period.max(1) as u32, 1, heads as u32, (!assign) as u32, 0], [base, 0.0, 0.0, 0.0], ceil_div(total, FLAT)).is_some()
}

/// Token ids on the device.
pub fn upload_ids(ids: &[usize]) -> Option<Tensor> {
    let words: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let g = gpu()?;
    let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-ids"), contents: bytemuck::cast_slice(&words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    Some(Tensor { buf, len: words.len() })
}

/// Softmax cross-entropy forward: per row `lse` and `loss = lse - x[t]`;
/// the mean loss is `Σ loss / rows` on the host.
pub fn softmax_ce_fwd(x: &Tensor, ids: &Tensor, lse: &Tensor, loss: &Tensor, rows: usize, d: usize) -> bool {
    let Some(p) = pipeline("softmax_ce_fwd", softmax_ce_fwd_src) else { return false };
    launch(p, &[&x.buf, &ids.buf, &lse.buf, &loss.buf], [0, d as u32, rows as u32, 0, 0, 0, 0, 0], [0.0; 4], rows as u32).is_some()
}

/// `gx (=|+=) inv * (softmax(x) - onehot)`.
pub fn softmax_ce_bwd(x: &Tensor, ids: &Tensor, lse: &Tensor, gx: &Tensor, rows: usize, d: usize, inv: f32, assign: bool) -> bool {
    let Some(p) = pipeline("softmax_ce_bwd", softmax_ce_bwd_src) else { return false };
    launch(p, &[&x.buf, &ids.buf, &lse.buf, &gx.buf], [(rows * d) as u32, d as u32, rows as u32, (!assign) as u32, 0, 0, 0, 0], [inv, 0.0, 0.0, 0.0], ceil_div(rows * d, FLAT)).is_some()
}

/// `y[i] = table[ids[i]]`.
pub fn embed_fwd(table: &Tensor, ids: &Tensor, y: &Tensor, rows: usize, d: usize) -> bool {
    let Some(p) = pipeline("embed_fwd", embed_fwd_src) else { return false };
    launch(p, &[&table.buf, &ids.buf, &y.buf], [(rows * d) as u32, d as u32, rows as u32, 0, 0, 0, 0, 0], [0.0; 4], ceil_div(rows * d, FLAT)).is_some()
}

/// The positions of each token id as CSR, for [`embed_bwd`].
pub struct TokenIndex { offs: Tensor, pos: Tensor }

impl TokenIndex {
    pub fn build(ids: &[usize], vocab: usize) -> Option<TokenIndex> {
        let mut counts = vec![0u32; vocab + 1];
        for &t in ids { counts[t + 1] += 1; }
        for v in 0..vocab { counts[v + 1] += counts[v]; }
        let mut fill = counts.clone();
        let mut pos = vec![0u32; ids.len()];
        for (i, &t) in ids.iter().enumerate() { pos[fill[t] as usize] = i as u32; fill[t] += 1; }
        let g = gpu()?;
        let mk = |w: &[u32]| g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-csr"), contents: bytemuck::cast_slice(w), usage: wgpu::BufferUsages::STORAGE,
        });
        Some(TokenIndex { offs: Tensor { buf: mk(&counts), len: counts.len() }, pos: Tensor { buf: mk(&pos), len: pos.len().max(1) } })
    }
}

/// `gtable[v] (=|+=) Σ_{i: ids[i]=v} g[i]`.
pub fn embed_bwd(g: &Tensor, index: &TokenIndex, gtable: &Tensor, vocab: usize, d: usize, assign: bool) -> bool {
    let Some(p) = pipeline("embed_bwd", embed_bwd_src) else { return false };
    launch(p, &[&g.buf, &index.offs.buf, &index.pos.buf, &gtable.buf], [(vocab * d) as u32, d as u32, 0, (!assign) as u32, 0, 0, 0, 0], [0.0; 4], ceil_div(vocab * d, FLAT)).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_autograd::Tape;

    fn mk(n: usize, ph: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.9).collect() }
    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() <= tol * scale, "{what}[{i}]: gpu {a} vs cpu {b}");
        }
    }

    /// Every kernel against the tape, forward value and every gradient,
    /// with accumulate-onto-existing exercised for the backward arms.
    #[test]
    fn elementwise_kernels_match_the_tape() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let (rows, d) = (37usize, 96usize);
        let x = mk(rows * d, 0.0); let w = mk(d, 1.0); let b = mk(rows * d, 2.0); let g = mk(rows * d, 3.0);
        let (tx, tw, tb, tg) = (Tensor::upload(&x).unwrap(), Tensor::upload(&w).unwrap(), Tensor::upload(&b).unwrap(), Tensor::upload(&g).unwrap());
        let y = Tensor::zeros(rows * d).unwrap();

        // rmsnorm
        let mut tape = Tape::new();
        let (vx, vw) = (tape.leaf(x.clone(), true), tape.leaf(w.clone(), true));
        let vy = tape.rmsnorm(vx, vw, d, 1e-5);
        tape.backward_from(vy, &g);
        assert!(rmsnorm_fwd(&tx, &tw, &y, rows, d, 1e-5));
        close(&y.download(), tape.value(vy), 1e-5, "rmsnorm");
        let (gx, gw, ri) = (Tensor::upload(&b).unwrap(), Tensor::upload(&w).unwrap(), Tensor::zeros(rows).unwrap());
        assert!(rmsnorm_bwd(&tx, &tw, &tg, &gx, &gw, &ri, rows, d, 1e-5, false, false));
        let want_gx: Vec<f32> = tape.grad(vx).iter().zip(&b).map(|(a, c)| a + c).collect();
        let want_gw: Vec<f32> = tape.grad(vw).iter().zip(&w).map(|(a, c)| a + c).collect();
        close(&gx.download(), &want_gx, 2e-5, "rmsnorm dx (accumulate)");
        close(&gw.download(), &want_gw, 2e-5, "rmsnorm dw (accumulate)");

        // silu, mul, add
        for (op, name) in [(Flat::Silu, "silu"), (Flat::Mul, "mul"), (Flat::Add, "add")] {
            let mut tape = Tape::new();
            let (va, vb) = (tape.leaf(x.clone(), true), tape.leaf(b.clone(), true));
            let vy = match op { Flat::Silu => tape.silu(va), Flat::Mul => tape.mul(va, vb), _ => tape.add(va, vb) };
            tape.backward_from(vy, &g);
            assert!(flat_fwd(op, &tx, &tb, &y, rows * d));
            close(&y.download(), tape.value(vy), 1e-5, name);
            let (ga, gb) = (Tensor::zeros(rows * d).unwrap(), Tensor::zeros(rows * d).unwrap());
            assert!(flat_bwd(op, &tx, &tb, &tg, Some((&ga, true)), if matches!(op, Flat::Silu) { None } else { Some((&gb, true)) }, rows * d));
            close(&ga.download(), tape.grad(va), 2e-5, &format!("{name} da"));
            if !matches!(op, Flat::Silu) { close(&gb.download(), tape.grad(vb), 2e-5, &format!("{name} db")); }
        }
        // silu*mul fused forward
        assert!(flat_fwd(Flat::SiluMul, &tx, &tb, &y, rows * d));
        let want: Vec<f32> = x.iter().zip(&b).map(|(a, c)| a / (1.0 + (-a).exp()) * c).collect();
        close(&y.download(), &want, 1e-5, "silu*mul");

        // rope: rows = 5 sequences of 8, 3 heads of 32
        let (nseq, seq, nh, hd) = (5usize, 8usize, 3usize, 32usize);
        let xr = mk(nseq * seq * nh * hd, 0.4);
        let gr = mk(nseq * seq * nh * hd, 1.4);
        let mut tape = Tape::new();
        let vx = tape.leaf(xr.clone(), true);
        let vy = tape.rope_seq(vx, nseq * seq, seq, nh, hd, 10000.0);
        tape.backward_from(vy, &gr);
        let (txr, tgr) = (Tensor::upload(&xr).unwrap(), Tensor::upload(&gr).unwrap());
        let yr = Tensor::zeros(xr.len()).unwrap();
        assert!(rope_fwd(&txr, &yr, nseq * seq, seq, nh, hd, 10000.0));
        close(&yr.download(), tape.value(vy), 2e-5, "rope");
        assert!(rope_bwd(&tgr, &yr, nseq * seq, seq, nh, hd, 10000.0, true));
        close(&yr.download(), tape.grad(vx), 2e-5, "rope dx");

        // softmax cross-entropy: 19 rows of vocab 300
        let (r, v) = (19usize, 300usize);
        let logits = mk(r * v, 0.7);
        let targets: Vec<usize> = (0..r).map(|i| (i * 37) % v).collect();
        let mut tape = Tape::new();
        let vl = tape.leaf(logits.clone(), true);
        let loss = tape.softmax_ce(vl, v, targets.clone());
        tape.backward_from(loss, &[1.0]);
        let tl = Tensor::upload(&logits).unwrap();
        let ids = upload_ids(&targets).unwrap();
        let (lse, lossv) = (Tensor::zeros(r).unwrap(), Tensor::zeros(r).unwrap());
        assert!(softmax_ce_fwd(&tl, &ids, &lse, &lossv, r, v));
        let mean = lossv.download().iter().sum::<f32>() / r as f32;
        assert!((mean - tape.value(loss)[0]).abs() < 1e-4, "loss {mean} vs {}", tape.value(loss)[0]);
        let gl = Tensor::zeros(r * v).unwrap();
        assert!(softmax_ce_bwd(&tl, &ids, &lse, &gl, r, v, 1.0 / r as f32, true));
        close(&gl.download(), tape.grad(vl), 2e-5, "softmax_ce dx");

        // embedding: vocab 50, 24 tokens, d 16
        let (vocab, n, de) = (50usize, 24usize, 16usize);
        let table = mk(vocab * de, 0.1);
        let toks: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % vocab).collect();
        let ge = mk(n * de, 2.2);
        let mut tape = Tape::new();
        let vt = tape.leaf(table.clone(), true);
        let ve = tape.embed(vt, &toks, de);
        tape.backward_from(ve, &ge);
        let (tt, tids, ye) = (Tensor::upload(&table).unwrap(), upload_ids(&toks).unwrap(), Tensor::zeros(n * de).unwrap());
        assert!(embed_fwd(&tt, &tids, &ye, n, de));
        close(&ye.download(), tape.value(ve), 0.0, "embed");
        let idx = TokenIndex::build(&toks, vocab).unwrap();
        let gt = Tensor::zeros(vocab * de).unwrap();
        assert!(embed_bwd(&Tensor::upload(&ge).unwrap(), &idx, &gt, vocab, de, true));
        close(&gt.download(), tape.grad(vt), 1e-6, "embed dtable");
    }
}
