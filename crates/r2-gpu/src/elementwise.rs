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

use crate::device::{gpu, Dtype, Tensor};
use std::collections::HashMap;
use std::sync::Mutex;
use wgpu::util::DeviceExt;

/// Threads per workgroup for the row kernels (one workgroup per row).
const ROW_THREADS: u32 = 256;
/// Threads per workgroup for the flat kernels (one element per thread).
const FLAT: u32 = 256;

/// Compiled kernels, one per (name, storage-type signature): the same
/// source compiles once for f32 buffers and again for each mix of f16
/// ones a caller hands it.
static PIPES: Mutex<Option<HashMap<String, &'static wgpu::ComputePipeline>>> = Mutex::new(None);

/// The storage types of a kernel's bindings, in binding order.
type Sig<'a> = &'a [Dtype];

fn pipeline(name: &'static str, sig: Sig, src: fn(Sig) -> String) -> Option<&'static wgpu::ComputePipeline> {
    let g = gpu()?;
    if sig.contains(&Dtype::F16) && !g.f16 { return None; }
    let key: String = std::iter::once(name.to_string()).chain(sig.iter().map(|d| d.tag().to_string())).collect();
    let mut guard = PIPES.lock().ok()?;
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(p) = table.get(&key) { return Some(p); }
    let module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name), source: wgpu::ShaderSource::Wgsl(src(sig).into()),
    });
    let p: &'static wgpu::ComputePipeline = Box::leak(Box::new(
        g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        })));
    table.insert(key, p);
    Some(p)
}

/// The signature of a launch: the dtypes of its tensors in binding order.
fn sig_of(ts: &[&Tensor]) -> Vec<Dtype> { ts.iter().map(|t| t.dtype).collect() }

/// Every kernel's uniform block: eight integers and four floats, meaning
/// per kernel.
const PARAMS: &str = "struct P { n: u32, d: u32, rows: u32, a: u32, b: u32, c: u32, e: u32, f: u32, x: f32, y: f32, z: f32, w: f32 };\n";

fn launch(p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], ints: [u32; 8], floats: [f32; 4], groups: u32) -> Option<()> {
    let res: Vec<wgpu::BindingResource> = bufs.iter().map(|b| b.as_entire_binding()).collect();
    launch_bufs(p, &res, ints, floats, groups)
}

/// A whole-buffer or sub-range binding at byte `off` (which must be a
/// multiple of 256, the storage-binding alignment every adapter honours).
fn range(buf: &wgpu::Buffer, off: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buf, offset: off, size: None })
}

fn launch_bufs(p: &wgpu::ComputePipeline, bufs: &[wgpu::BindingResource], ints: [u32; 8], floats: [f32; 4], groups: u32) -> Option<()> {
    let g = gpu()?;
    let mut words = [0u32; 12];
    words[..8].copy_from_slice(&ints);
    for (i, f) in floats.iter().enumerate() { words[8 + i] = f.to_bits(); }
    let ubuf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-params"), contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.clone() }).collect();
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

/// The uniform block, the bindings (each in its storage type from `sig`;
/// `ids` / `offs` / `pos` are u32 whatever `sig` says) and, for every
/// float binding, `ld_NAME(i) -> f32` and — when writable — `st_NAME(i,
/// v)`. Kernel bodies go through these, so one body serves every mix of
/// f32 and f16 buffers and every arithmetic stays f32.
fn bindings(sig: Sig, names: &[(&str, &str)]) -> String {
    let mut s = String::new();
    if sig.contains(&Dtype::F16) { s += "enable f16;\n"; }
    s += &crate::numerics::prelude();          // exp_r2, div_cr, sqrt_cr: the CPU's bits
    s += PARAMS;
    for (i, (name, access)) in names.iter().enumerate() {
        let ints = *name == "ids" || *name == "offs" || *name == "pos";
        let ty = if ints { "u32" } else { sig.get(i).copied().unwrap_or(Dtype::F32).wgsl() };
        s += &format!("@group(0) @binding({i}) var<storage, {access}> {name}: array<{ty}>;\n");
        if !ints {
            s += &format!("fn ld_{name}(i: u32) -> f32 {{ return f32({name}[i]); }}\n");
            if *access == "read_write" {
                s += &format!("fn st_{name}(i: u32, v: f32) {{ {name}[i] = {ty}(v); }}\n");
            }
        }
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

/// The row layout of the rmsnorm kernels: `RM_LANES` threads per row,
/// `RM_ROWS` rows per workgroup. One workgroup per row (the obvious
/// layout) put 2,048 workgroups of 256 threads through an 8-stage barrier
/// tree to reduce 256 elements each — 1.5 ms for 2 MB, latency, not
/// bandwidth. Eight rows per workgroup with 32 lanes each amortise the
/// barriers eight ways and shorten the tree to five stages.
const RM_LANES: u32 = 32;
const RM_ROWS: u32 = 8;
/// Row blocks the dW reduction is split across (a fixed-order two-stage
/// sum, so the result does not depend on scheduling).
const RM_PARTS: u32 = 64;

/// Scratch `rmsnorm_bwd` needs for `rows` x `d`: the per-row `rinv`, then
/// `RM_PARTS x d` dW partials.
pub fn rmsnorm_scratch_len(rows: usize, d: usize) -> usize { rm_part_off(rows) + RM_PARTS as usize * d }

/// Where the dW partials start in the scratch: after `rinv`, rounded up to
/// the 256-byte storage-binding alignment (in floats).
fn rm_part_off(rows: usize) -> usize { rows.div_ceil(64) * 64 }

/// The per-row prologue shared by the forward and the dX pass: which row
/// this thread serves, whether it exists, and where it starts (an absent
/// row reads row 0 so every lane keeps to the barriers in step).
const RM_ROW: &str = r#"
    let lane = tid % LANES;
    let r = wg.x * ROWS + tid / LANES;
    let valid = r < p.rows;
    let base = select(0u, r * p.d, valid);
"#;

/// A fixed-order sum of each row's `LANES` partials into its first lane.
const RM_TREE: &str = r#"
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s = s >> 1u) {
        if (lane < s) { red[tid] = red[tid] + red[tid + s]; }
        workgroupBarrier();
    }
"#;

fn rm_consts() -> String {
    format!("const LANES: u32 = {RM_LANES}u;
const ROWS: u32 = {RM_ROWS}u;
const T: u32 = {}u;
", RM_LANES * RM_ROWS)
}

fn rmsnorm_fwd_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("W", "read"), ("Y", "read_write")]) + &rm_consts() + &format!(r#"
var<workgroup> red: array<f32, T>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
{RM_ROW}
    var ss = 0.0;
    for (var j = lane; j < p.d; j = j + LANES) {{ let v = ld_X(base + j); ss = fma(v, v, ss); }}
    red[tid] = ss;
{RM_TREE}
    let scale = 1.0 / sqrt(red[tid - lane] / f32(p.d) + p.x);
    if (valid) {{
        for (var j = lane; j < p.d; j = j + LANES) {{ st_Y(base + j, ld_X(base + j) * scale * ld_W(j)); }}
    }}
}}
"#, T = RM_LANES * RM_ROWS)
}

/// dX (assign or accumulate) and the per-row `rinv` the dW pass needs.
fn rmsnorm_bwd_x_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("W", "read"), ("G", "read"), ("GX", "read_write"), ("RI", "read_write")]) + &rm_consts() + &format!(r#"
var<workgroup> red: array<f32, T>;
var<workgroup> red2: array<f32, T>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
{RM_ROW}
    var ss = 0.0;
    var sg = 0.0;
    for (var j = lane; j < p.d; j = j + LANES) {{
        let v = ld_X(base + j);
        ss = fma(v, v, ss);
        sg = fma(ld_G(base + j) * ld_W(j), v, sg);
    }}
    red[tid] = ss;
    red2[tid] = sg;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s = s >> 1u) {{
        if (lane < s) {{ red[tid] = red[tid] + red[tid + s]; red2[tid] = red2[tid] + red2[tid + s]; }}
        workgroupBarrier();
    }}
    let ri = 1.0 / sqrt(red[tid - lane] / f32(p.d) + p.x);
    let coef = ri * ri * ri / f32(p.d) * red2[tid - lane];
    if (valid) {{
        if (lane == 0u) {{ st_RI(r, ri); }}
        for (var j = lane; j < p.d; j = j + LANES) {{
            let o = base + j;
            let dx = ld_G(o) * ld_W(j) * ri - coef * ld_X(o);
            if (p.a == 0u) {{ st_GX(o, dx); }} else {{ st_GX(o, ld_GX(o) + dx); }}
        }}
    }}
}}
"#, T = RM_LANES * RM_ROWS)
}

/// dW, stage one: `PART[b][j] = Σ_(r in block b) G[r][j] X[r][j] rinv[r]`,
/// one thread per (block, column), rows in order. One thread per column
/// over ALL rows was a single workgroup walking 2,048 rows — 4 ms.
fn rmsnorm_bwd_w_part_src(sig: Sig) -> String {
    // RI and PART are two ranges of ONE scratch buffer; wgpu tracks usage
    // per buffer, so both are declared read_write (one usage merges, a
    // read beside a read_write does not)
    bindings(sig, &[("X", "read"), ("G", "read"), ("RI", "read_write"), ("PART", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let cg = (p.d + {T}u - 1u) / {T}u;
    let b = wg.x / cg;
    let j = (wg.x % cg) * {T}u + tid;
    if (j >= p.d) {{ return; }}
    let chunk = (p.rows + p.b - 1u) / p.b;
    let r0 = b * chunk;
    let r1 = min(r0 + chunk, p.rows);
    var acc = 0.0;
    for (var r = r0; r < r1; r = r + 1u) {{ acc = fma(ld_G(r * p.d + j) * ld_X(r * p.d + j), ld_RI(r), acc); }}
    st_PART(b * p.d + j, acc);
}}
"#, T = FLAT)
}

/// dW, stage two: `GW[j] (=|+=) Σ_b PART[b][j]`, blocks in order.
fn rmsnorm_bwd_w_sum_src(sig: Sig) -> String {
    bindings(sig, &[("PART", "read"), ("GW", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let j = gid.x;
    if (j >= p.d) {{ return; }}
    var acc = 0.0;
    for (var b = 0u; b < p.b; b = b + 1u) {{ acc = acc + ld_PART(b * p.d + j); }}
    if (p.a == 0u) {{ st_GW(j, acc); }} else {{ st_GW(j, ld_GW(j) + acc); }}
}}
"#, T = FLAT)
}

/// `Y = f(A, B)` over `n` elements; `p.b` selects the op:
/// 0 silu(A), 1 A*B, 2 A+B, 3 silu(A)*B, 4 Y + A (in place; B unused),
/// 5 A (a copy — between storage types, this is the f32 -> f16 cast).
fn flat_fwd_src(sig: Sig) -> String {
    bindings(sig, &[("A", "read"), ("B", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let a = ld_A(i);
    var y = 0.0;
    switch p.b {{
        case 0u: {{ y = div_cr(a, 1.0 + exp_r2(-a)); }}
        case 1u: {{ y = a * ld_B(i); }}
        case 2u: {{ y = a + ld_B(i); }}
        case 3u: {{ y = div_cr(a, 1.0 + exp_r2(-a)) * ld_B(i); }}
        case 4u: {{ y = ld_Y(i) + a; }}
        default: {{ y = a; }}
    }}
    st_Y(i, y);
}}
"#, T = FLAT)
}

/// Backward of the flat ops. `p.b`: 0 silu → GA (=|+=) G·silu'(A);
/// 1 mul → GA (=|+=) G·B and GB (=|+=) G·A; 2 add → GA, GB (=|+=) G.
/// `p.a` / `p.c` are the assign flags for GA / GB; `p.e` / `p.f` say
/// whether each is wanted at all.
fn flat_bwd_src(sig: Sig) -> String {
    bindings(sig, &[("A", "read"), ("B", "read"), ("G", "read"), ("GA", "read_write"), ("GB", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let g = ld_G(i);
    var da = 0.0;
    var db = 0.0;
    switch p.b {{
        case 0u: {{ let a = ld_A(i); let s = div_cr(1.0, 1.0 + exp_r2(-a)); da = g * (s + a * s * (1.0 - s)); }}
        case 1u: {{ da = g * ld_B(i); db = g * ld_A(i); }}
        default: {{ da = g; db = g; }}
    }}
    if (p.e == 1u) {{ if (p.a == 0u) {{ st_GA(i, da); }} else {{ st_GA(i, ld_GA(i) + da); }} }}
    if (p.f == 1u) {{ if (p.c == 0u) {{ st_GB(i, db); }} else {{ st_GB(i, ld_GB(i) + db); }} }}
}}
"#, T = FLAT)
}

/// RoPE, one thread per (row, head, pair). `p.b` = 1 for the backward
/// (the inverse rotation), `p.a` the assign flag (backward only).
///
/// The angles come from TAB, `(cos, sin)` per (position, pair), built on
/// the host by [`rope_table`] — the CPU tape's own table, value for value.
/// This kernel used to compute `pow`, `cos` and `sin` itself; the probe
/// found this adapter's `sin` 2.1 million ULP off near its zeros (Vulkan
/// only bounds it in absolute terms), and those are vendor functions that
/// no two devices share. With the table there is no transcendental on the
/// device at all, the rotation is the tape's `a*c - b*s` / `a*s + b*c`,
/// and the GPU's RoPE gives the CPU's bits.
fn rope_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("TAB", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let half = p.d / 2u;                    // pairs per head; p.d = head_dim
    let i = gid.x;                          // (row, head, pair)
    let total = p.rows * p.c * half;        // p.c = heads
    if (i >= total) {{ return; }}
    let pr = i % half;
    let h = (i / half) % p.c;
    let r = i / (half * p.c);
    let t = ((r % p.a) * half + pr) * 2u;   // p.a = period
    let c = ld_TAB(t);
    let s = ld_TAB(t + 1u);
    let o = r * p.c * p.d + h * p.d + 2u * pr;
    let a = ld_X(o);
    let b = ld_X(o + 1u);
    if (p.b == 0u) {{
        st_Y(o, a * c - b * s);
        st_Y(o + 1u, a * s + b * c);
    }} else {{
        let ra = a * c + b * s;
        let rb = -a * s + b * c;
        if (p.e == 0u) {{ st_Y(o, ra); st_Y(o + 1u, rb); }} else {{ st_Y(o, ld_Y(o) + ra); st_Y(o + 1u, ld_Y(o + 1u) + rb); }}
    }}
}}
"#, T = FLAT)
}

/// Per row: `LSE[r] = logsumexp(row)`, `LOSS[r] = LSE[r] - x[t]`.
fn softmax_ce_fwd_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("ids", "read"), ("LSE", "read_write"), ("LOSS", "read_write")]) + &format!(r#"
var<workgroup> red: array<f32, {T}u>;
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let r = wg.x;
    var m = -3.0e38;
    for (var j = tid; j < p.d; j = j + {T}u) {{ m = max(m, ld_X(r * p.d + j)); }}
    red[tid] = m;
    workgroupBarrier();
    for (var s = {half}u; s > 0u; s = s >> 1u) {{
        if (tid < s) {{ red[tid] = max(red[tid], red[tid + s]); }}
        workgroupBarrier();
    }}
    let mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = tid; j < p.d; j = j + {T}u) {{ sum = sum + exp_r2(ld_X(r * p.d + j) - mx); }}
    red[tid] = sum;
{tree}
    if (tid == 0u) {{
        let lse = mx + log(red[0]);
        st_LSE(r, lse);
        st_LOSS(r, lse - ld_X(r * p.d + ids[r]));
    }}
}}
"#, T = ROW_THREADS, half = ROW_THREADS / 2, tree = tree_sum())
}

/// `GX (=|+=) inv * (exp(x - lse) - [j == t])`.
fn softmax_ce_bwd_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("ids", "read"), ("LSE", "read"), ("GX", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let r = i / p.d;
    let j = i % p.d;
    var v = exp_r2(ld_X(i) - ld_LSE(r));
    if (j == ids[r]) {{ v = v - 1.0; }}
    v = v * p.x;
    if (p.a == 0u) {{ st_GX(i, v); }} else {{ st_GX(i, ld_GX(i) + v); }}
}}
"#, T = FLAT)
}

/// `FLAG[p.a] = 1` if any of `X[0..n]` is inf, nan, or at least `p.x`
/// in magnitude. Non-finiteness is tested on the bits (WGSL has no
/// isinf/isnan, and `v != v` is not safe under the fast-math a driver
/// may apply). The magnitude test is what catches an f16 overflow on
/// an adapter that SATURATES an overflowing f32 -> f16 store to
/// ±65,504 instead of producing inf — this one does, so an inf-only
/// check would never fire. Racing writes of the same 1.0 are benign.
fn nonfinite_src(sig: Sig) -> String {
    bindings(sig, &[("X", "read"), ("FLAG", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let v = ld_X(i);
    let b = bitcast<u32>(v);
    if ((b & 0x7f800000u) == 0x7f800000u || abs(v) >= p.x) {{ st_FLAG(p.a, 1.0); }}
}}
"#, T = FLAT)
}

/// `Y[i][:] = T[ids[i]][:]`.
fn embed_fwd_src(sig: Sig) -> String {
    bindings(sig, &[("T", "read"), ("ids", "read"), ("Y", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let row = i / p.d;
    let j = i % p.d;
    st_Y(i, ld_T(ids[row] * p.d + j));
}}
"#, T = FLAT)
}

/// `GT[v][j] (=|+=) Σ_{{i : ids[i] = v}} G[i][j]`, the positions of each
/// token given as CSR (`offs[v]..offs[v+1]` index into `pos`), built on
/// the host per batch. One thread per (v, j); rows in order.
fn embed_bwd_src(sig: Sig) -> String {
    bindings(sig, &[("G", "read"), ("offs", "read"), ("pos", "read"), ("GT", "read_write")]) + &format!(r#"
@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;                          // v * d + j
    if (i >= p.n) {{ return; }}
    let v = i / p.d;
    let j = i % p.d;
    var acc = 0.0;
    for (var k = offs[v]; k < offs[v + 1u]; k = k + 1u) {{ acc = acc + ld_G(pos[k] * p.d + j); }}
    if (p.a == 0u) {{ st_GT(i, acc); }} else {{ st_GT(i, ld_GT(i) + acc); }}
}}
"#, T = FLAT)
}

// ── host API ────────────────────────────────────────────────────────────

fn ceil_div(n: usize, d: u32) -> u32 { (n as u32).div_ceil(d) }

/// `y[r] = x[r] / rms(x[r]) * w`, rows of `d`.
pub fn rmsnorm_fwd(x: &Tensor, w: &Tensor, y: &Tensor, rows: usize, d: usize, eps: f32) -> bool {
    let Some(p) = pipeline("rmsnorm_fwd", &sig_of(&[x, w, y]), rmsnorm_fwd_src) else { return false };
    launch(p, &[&x.buf, &w.buf, &y.buf], [0, d as u32, rows as u32, 0, 0, 0, 0, 0], [eps, 0.0, 0.0, 0.0], ceil_div(rows, RM_ROWS)).is_some()
}

/// rmsnorm backward: `gx (=|+=)` per `assign_x`, `gw (=|+=)` per
/// `assign_w`; `rinv` is scratch of length `rows`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bwd(x: &Tensor, w: &Tensor, g: &Tensor, gx: &Tensor, gw: &Tensor, scratch: &Tensor,
                   rows: usize, d: usize, eps: f32, assign_x: bool, assign_w: bool) -> bool {
    if scratch.len < rmsnorm_scratch_len(rows, d) { return false; }
    let (Some(px), Some(pp), Some(ps)) = (pipeline("rmsnorm_bwd_x", &sig_of(&[x, w, g, gx, scratch]), rmsnorm_bwd_x_src),
                                          pipeline("rmsnorm_bwd_w_part", &sig_of(&[x, g, scratch, scratch]), rmsnorm_bwd_w_part_src),
                                          pipeline("rmsnorm_bwd_w_sum", &sig_of(&[scratch, gw]), rmsnorm_bwd_w_sum_src)) else { return false };
    // the scratch, split: rinv is its first `rows` floats, the partials follow
    let po = (rm_part_off(rows) * 4) as u64;
    let (ri, part) = (range(&scratch.buf, 0), range(&scratch.buf, po));
    let parts = RM_PARTS.min(rows as u32).max(1);
    launch_bufs(px, &[x.buf.as_entire_binding(), w.buf.as_entire_binding(), g.buf.as_entire_binding(), gx.buf.as_entire_binding(), ri.clone()],
                [0, d as u32, rows as u32, (!assign_x) as u32, 0, 0, 0, 0], [eps, 0.0, 0.0, 0.0], ceil_div(rows, RM_ROWS)).is_some()
        && launch_bufs(pp, &[x.buf.as_entire_binding(), g.buf.as_entire_binding(), ri, part.clone()],
                       [0, d as u32, rows as u32, 0, parts, 0, 0, 0], [0.0; 4], ceil_div(d, FLAT) * parts).is_some()
        && launch_bufs(ps, &[part, gw.buf.as_entire_binding()],
                       [0, d as u32, rows as u32, (!assign_w) as u32, parts, 0, 0, 0], [0.0; 4], ceil_div(d, FLAT)).is_some()
}

/// The flat forward ops. `AddInto` is `y += a` in place — the residual
/// connection — and ignores `b`; `Silu` and `Copy` ignore `b` too.
/// `Copy` between an f32 and an f16 tensor is the cast.
#[derive(Clone, Copy)]
pub enum Flat { Silu, Mul, Add, SiluMul, AddInto, Copy }

/// The one-element scratch bound wherever a kernel has an operand nobody
/// reads: binding a buffer twice in one dispatch (once read-only, once
/// read-write) is a usage conflict, so unused slots get this instead.
fn dummy() -> Option<&'static Tensor> {
    static DUMMY: std::sync::OnceLock<Option<Tensor>> = std::sync::OnceLock::new();
    DUMMY.get_or_init(|| Tensor::zeros(1)).as_ref()
}

/// `y = op(a, b)` over `n` elements.
pub fn flat_fwd(op: Flat, a: &Tensor, b: &Tensor, y: &Tensor, n: usize) -> bool {
    let code = match op { Flat::Silu => 0, Flat::Mul => 1, Flat::Add => 2, Flat::SiluMul => 3, Flat::AddInto => 4, Flat::Copy => 5 };
    // an unused read-only slot aliases A: two read bindings of one buffer
    // are allowed, a read and a read_write of one buffer are not
    let (bb, bt) = if matches!(op, Flat::Silu | Flat::AddInto | Flat::Copy) { (&a.buf, a) } else { (&b.buf, b) };
    let Some(p) = pipeline("flat_fwd", &sig_of(&[a, bt, y]), flat_fwd_src) else { return false };
    launch(p, &[&a.buf, bb, &y.buf], [n as u32, 0, 0, 0, code, 0, 0, 0], [0.0; 4], ceil_div(n, FLAT)).is_some()
}

/// Backward of `Silu` / `Mul` / `Add`: `ga`, `gb` each `Some((buffer,
/// assign))` when wanted.
pub fn flat_bwd(op: Flat, a: &Tensor, b: &Tensor, g: &Tensor, ga: Option<(&Tensor, bool)>, gb: Option<(&Tensor, bool)>, n: usize) -> bool {
    let code = match op { Flat::Silu => 0, Flat::Mul => 1, Flat::Add | Flat::AddInto | Flat::Copy => 2, Flat::SiluMul => 1 };
    let Some(dummy) = dummy() else { return false };
    let (gat, ga_acc, ga_on) = match ga { Some((t, assign)) => (t, (!assign) as u32, 1), None => (dummy, 0, 0) };
    let (gbt, gb_acc, gb_on) = match gb { Some((t, assign)) => (t, (!assign) as u32, 1), None => (dummy, 0, 0) };
    let (gab, gbb) = (&gat.buf, &gbt.buf);
    let (bb, bt) = if matches!(op, Flat::Silu) { (&a.buf, a) } else { (&b.buf, b) };
    let Some(p) = pipeline("flat_bwd", &sig_of(&[a, bt, g, gat, gbt]), flat_bwd_src) else { return false };
    launch(p, &[&a.buf, bb, &g.buf, gab, gbb], [n as u32, 0, 0, ga_acc, code, gb_acc, ga_on, gb_on], [0.0; 4], ceil_div(n, FLAT)).is_some()
}

/// `(cos, sin)` per (position, pair), flattened — `r2_tensor::ops::
/// rope_table`, the CPU tape's table, by the same expressions in the same
/// order (so the same bits; `rope_table_matches_the_cpu` pins it). It is
/// computed on the host, where `powf` and `sin_cos` are one library for
/// both the CPU tape and the device.
pub fn rope_table(period: usize, head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut t = Vec::with_capacity(period.max(1) * half * 2);
    for pos in 0..period.max(1) {
        for p in 0..half {
            let freq = 1.0 / base.powf(2.0 * p as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            let (s, c) = theta.sin_cos();
            t.push(c);
            t.push(s);
        }
    }
    t
}

/// The table on the device, built once per (period, head_dim, base).
fn rope_table_dev(period: usize, head_dim: usize, base: f32) -> Option<&'static Tensor> {
    static TABLES: Mutex<Option<HashMap<(usize, usize, u32), &'static Tensor>>> = Mutex::new(None);
    let key = (period.max(1), head_dim, base.to_bits());
    let mut guard = TABLES.lock().ok()?;
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(t) = table.get(&key) { return Some(t); }
    let t: &'static Tensor = Box::leak(Box::new(Tensor::upload(&rope_table(period, head_dim, base))?));
    table.insert(key, t);
    Some(t)
}

/// RoPE forward: `y = rotate(x)`, positions restarting every `period`
/// rows; rows are `heads x head_dim` wide.
pub fn rope_fwd(x: &Tensor, y: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32) -> bool {
    let Some(tab) = rope_table_dev(period, head_dim, base) else { return false };
    let Some(p) = pipeline("rope", &sig_of(&[x, tab, y]), rope_src) else { return false };
    let total = rows * heads * (head_dim / 2);
    launch(p, &[&x.buf, &tab.buf, &y.buf], [0, head_dim as u32, rows as u32, period.max(1) as u32, 0, heads as u32, 0, 0], [0.0; 4], ceil_div(total, FLAT)).is_some()
}

/// RoPE backward: `gx (=|+=) rotate⁻¹(g)`.
pub fn rope_bwd(g: &Tensor, gx: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32, assign: bool) -> bool {
    let Some(tab) = rope_table_dev(period, head_dim, base) else { return false };
    let Some(p) = pipeline("rope", &sig_of(&[g, tab, gx]), rope_src) else { return false };
    let total = rows * heads * (head_dim / 2);
    launch(p, &[&g.buf, &tab.buf, &gx.buf], [0, head_dim as u32, rows as u32, period.max(1) as u32, 1, heads as u32, (!assign) as u32, 0], [0.0; 4], ceil_div(total, FLAT)).is_some()
}

/// Token ids on the device.
pub fn upload_ids(ids: &[usize]) -> Option<Tensor> {
    let words: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    let g = gpu()?;
    let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-ids"), contents: bytemuck::cast_slice(&words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    Some(Tensor { buf, len: words.len(), dtype: crate::device::Dtype::F32 })
}

/// Softmax cross-entropy forward: per row `lse` and `loss = lse - x[t]`;
/// the mean loss is `Σ loss / rows` on the host.
pub fn softmax_ce_fwd(x: &Tensor, ids: &Tensor, lse: &Tensor, loss: &Tensor, rows: usize, d: usize) -> bool {
    let Some(p) = pipeline("softmax_ce_fwd", &sig_of(&[x, ids, lse, loss]), softmax_ce_fwd_src) else { return false };
    launch(p, &[&x.buf, &ids.buf, &lse.buf, &loss.buf], [0, d as u32, rows as u32, 0, 0, 0, 0, 0], [0.0; 4], rows as u32).is_some()
}

/// `gx (=|+=) inv * (softmax(x) - onehot)`.
pub fn softmax_ce_bwd(x: &Tensor, ids: &Tensor, lse: &Tensor, gx: &Tensor, rows: usize, d: usize, inv: f32, assign: bool) -> bool {
    let Some(p) = pipeline("softmax_ce_bwd", &sig_of(&[x, ids, lse, gx]), softmax_ce_bwd_src) else { return false };
    launch(p, &[&x.buf, &ids.buf, &lse.buf, &gx.buf], [(rows * d) as u32, d as u32, rows as u32, (!assign) as u32, 0, 0, 0, 0], [inv, 0.0, 0.0, 0.0], ceil_div(rows * d, FLAT)).is_some()
}

/// Sets `flag[at] = 1` when any of `x[0..n]` is inf or nan — or, for an
/// f16 tensor, saturated at the f16 maximum (65,504), which is what an
/// overflowing store becomes on an adapter that clamps instead of
/// producing inf. Leaves the flag alone otherwise. The caller zeroes the
/// slot first and reads it back with whatever else it downloads — the
/// overflow test of dynamic loss scaling without a readback of its own.
pub fn nonfinite_flag(x: &Tensor, flag: &Tensor, at: usize, n: usize) -> bool {
    let Some(p) = pipeline("nonfinite", &sig_of(&[x, flag]), nonfinite_src) else { return false };
    let limit = if x.dtype == Dtype::F16 { 65504.0 } else { f32::INFINITY };
    launch(p, &[&x.buf, &flag.buf], [n as u32, 0, 0, at as u32, 0, 0, 0, 0], [limit, 0.0, 0.0, 0.0], ceil_div(n, FLAT)).is_some()
}

/// `y[i] = table[ids[i]]`.
pub fn embed_fwd(table: &Tensor, ids: &Tensor, y: &Tensor, rows: usize, d: usize) -> bool {
    let Some(p) = pipeline("embed_fwd", &sig_of(&[table, ids, y]), embed_fwd_src) else { return false };
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
        Some(TokenIndex { offs: Tensor { buf: mk(&counts), len: counts.len(), dtype: crate::device::Dtype::F32 }, pos: Tensor { buf: mk(&pos), len: pos.len().max(1), dtype: crate::device::Dtype::F32 } })
    }
}

/// `gtable[v] (=|+=) Σ_{i: ids[i]=v} g[i]`.
pub fn embed_bwd(g: &Tensor, index: &TokenIndex, gtable: &Tensor, vocab: usize, d: usize, assign: bool) -> bool {
    let Some(p) = pipeline("embed_bwd", &sig_of(&[g, &index.offs, &index.pos, gtable]), embed_bwd_src) else { return false };
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

    /// The device's RoPE table is the CPU tape's, value for value.
    #[test]
    fn rope_table_matches_the_cpu() {
        for &(per, hd, base) in &[(64usize, 64usize, 10000.0f32), (256, 64, 10000.0), (8, 32, 500000.0)] {
            let flat: Vec<f32> = r2_tensor::ops::rope_table(per, hd, base).into_iter().flat_map(|(c, s)| [c, s]).collect();
            let ours = rope_table(per, hd, base);
            assert!(ours.iter().zip(&flat).all(|(a, b)| a.to_bits() == b.to_bits()) && ours.len() == flat.len(),
                    "table differs at period {per}, head_dim {hd}");
        }
    }

    /// Every kernel against the tape, forward value and every gradient,
    /// with accumulate-onto-existing exercised for the backward arms.
    #[test]
    fn elementwise_kernels_match_the_tape() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        kernels_match_the_tape(Dtype::F32, 1.0);
    }

    /// The same kernels with every activation-role buffer stored f16:
    /// inputs are rounded to f16 on the host first so the tape sees the
    /// same numbers, and the tolerance is one f16 ULP of an output (2^-10
    /// relative: this adapter's f32 -> f16 store truncates rather than
    /// rounds to nearest, so the bound is a whole ULP, not half).
    #[test]
    fn elementwise_kernels_match_the_tape_in_f16_storage() {
        if gpu().map(|g| g.f16) != Some(true) { eprintln!("no f16 on this adapter; skipped"); return; }
        kernels_match_the_tape(Dtype::F16, 110.0);
    }

    fn kernels_match_the_tape(dt: Dtype, tol_x: f32) {
        use crate::half::{f16_to_f32, f32_to_f16};
        let close = |got: &[f32], want: &[f32], tol: f32, what: &str| close(got, want, tol * tol_x, what);
        // activations in `dt`, rounded on the host when dt is f16
        let mk = |n: usize, ph: f32| -> Vec<f32> {
            let v = mk(n, ph);
            if dt == Dtype::F16 { v.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect() } else { v }
        };
        let up = |v: &[f32]| Tensor::upload_as(v, dt).unwrap();
        let zeros = |n: usize| Tensor::zeros_as(n, dt).unwrap();
        let (rows, d) = (37usize, 96usize);
        let x = mk(rows * d, 0.0); let w = mk(d, 1.0); let b = mk(rows * d, 2.0); let g = mk(rows * d, 3.0);
        // norm weights and the residual stream stay f32 in the trainer; x plays the residual here
        let (tx, tw, tb, tg) = (Tensor::upload(&x).unwrap(), Tensor::upload(&w).unwrap(), up(&b), up(&g));
        let y = zeros(rows * d);

        // rmsnorm
        let mut tape = Tape::new();
        let (vx, vw) = (tape.leaf(x.clone(), true), tape.leaf(w.clone(), true));
        let vy = tape.rmsnorm(vx, vw, d, 1e-5);
        tape.backward_from(vy, &g);
        assert!(rmsnorm_fwd(&tx, &tw, &y, rows, d, 1e-5));
        close(&y.download(), tape.value(vy), 1e-5, "rmsnorm");
        let (gx, gw, ri) = (Tensor::upload(&b).unwrap(), Tensor::upload(&w).unwrap(), Tensor::zeros(rmsnorm_scratch_len(rows, d)).unwrap());
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
            let (ga, gb) = (zeros(rows * d), zeros(rows * d));
            assert!(flat_bwd(op, &tx, &tb, &tg, Some((&ga, true)), if matches!(op, Flat::Silu) { None } else { Some((&gb, true)) }, rows * d));
            close(&ga.download(), tape.grad(va), 2e-5, &format!("{name} da"));
            if !matches!(op, Flat::Silu) { close(&gb.download(), tape.grad(vb), 2e-5, &format!("{name} db")); }
        }
        // the same flat ops with A stored in `dt` too (gate, up, sg, act are all `dt` in the trainer)
        let txd = up(&x);
        for (op, name) in [(Flat::Silu, "silu(dt)"), (Flat::Mul, "mul(dt)")] {
            let mut tape = Tape::new();
            let (va, vb) = (tape.leaf(x.clone(), true), tape.leaf(b.clone(), true));
            let vy = if matches!(op, Flat::Silu) { tape.silu(va) } else { tape.mul(va, vb) };
            tape.backward_from(vy, &g);
            assert!(flat_fwd(op, &txd, &tb, &y, rows * d));
            close(&y.download(), tape.value(vy), 1e-5, name);
            let ga = zeros(rows * d);
            assert!(flat_bwd(op, &txd, &tb, &tg, Some((&ga, true)), None, rows * d));
            close(&ga.download(), tape.grad(va), 2e-5, &format!("{name} da"));
        }
        // the cast: f32 -> dt -> f32 round-trips to the rounded input
        assert!(flat_fwd(Flat::Copy, &tx, &tx, &y, rows * d));
        close(&y.download(), &x, 0.0, "copy/cast");
        // the overflow flag: clean data leaves it, one inf sets it
        let flag = Tensor::zeros(4).unwrap();
        assert!(nonfinite_flag(&tb, &flag, 2, rows * d));
        assert_eq!(flag.download(), vec![0.0; 4], "clean data must not raise the flag");
        let mut bad = b.clone(); bad[rows * d / 2] = f32::INFINITY;
        let tbad = up(&bad);
        assert!(nonfinite_flag(&tbad, &flag, 2, rows * d));
        assert_eq!(flag.download(), vec![0.0, 0.0, 1.0, 0.0], "an inf must raise the flag at `at`");
        if dt == Dtype::F16 {
            // an overflowing store saturates on this adapter: that must be flagged too
            let flag = Tensor::zeros(1).unwrap();
            let sat = Tensor::zeros_as(4, Dtype::F16).unwrap();
            assert!(flat_fwd(Flat::Copy, &Tensor::upload(&[1.0, 1e30, 2.0, 3.0]).unwrap(), &tx, &sat, 4));
            assert!(nonfinite_flag(&sat, &flag, 0, 4));
            assert_eq!(flag.download(), vec![1.0], "a saturated f16 must raise the flag");
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
        let (txr, tgr) = (up(&xr), up(&gr));
        let yr = zeros(xr.len());
        assert!(rope_fwd(&txr, &yr, nseq * seq, seq, nh, hd, 10000.0));
        // with the host table there is no device transcendental left: in f32
        // storage the GPU's RoPE is the tape's, bit for bit
        let rope_tol = if dt == Dtype::F32 { 0.0 } else { 2e-5 };
        close(&yr.download(), tape.value(vy), rope_tol, "rope");
        assert!(rope_bwd(&tgr, &yr, nseq * seq, seq, nh, hd, 10000.0, true));
        close(&yr.download(), tape.grad(vx), rope_tol, "rope dx");

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
        let gl = zeros(r * v);
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
        let (tt, tids, ye) = (up(&table), upload_ids(&toks).unwrap(), Tensor::zeros(n * de).unwrap());
        assert!(embed_fwd(&tt, &tids, &ye, n, de));
        close(&ye.download(), tape.value(ve), 0.0, "embed");
        let idx = TokenIndex::build(&toks, vocab).unwrap();
        let gt = Tensor::zeros(vocab * de).unwrap();
        assert!(embed_bwd(&Tensor::upload(&ge).unwrap(), &idx, &gt, vocab, de, true));
        close(&gt.download(), tape.grad(vt), 1e-6, "embed dtable");
    }
}
