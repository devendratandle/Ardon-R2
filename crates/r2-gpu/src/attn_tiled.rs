//! The tiled attention forward: the score block and the P·V update as
//! register-tiled products across the workgroup — FlashAttention-2's
//! shape — for head dimensions that are a multiple of 32.
//!
//! The one-thread-per-query kernel in `attention.rs` is the reference:
//! simple, and correct for any `hd`, but every thread re-reads the whole
//! staged K/V block, one load per FMA, so it is bound by workgroup-memory
//! bandwidth at a fifth of the GEMM's rate. Here a workgroup of 128
//! threads owns 64 queries and walks the keys 32 at a time:
//!
//! 1. `S = Q·Kᵀ` (64 x 32) as a GEMM: Q and K are staged transposed in
//!    16-deep slabs of `hd`, each thread accumulates a 4 x 4 sub-tile in
//!    named `vec4`s — two loads per four FMAs, as in `gemm.rs`.
//! 2. The scaled, causally masked scores go to workgroup memory; one
//!    thread per row does the online-softmax update (new max, rescale
//!    factor, running sum) and rewrites the row as `P`.
//! 3. `O += P·V`: each thread's 4 rows x (hd/8) columns of the output tile
//!    stay in registers for the whole key walk, rescaled by the row's
//!    factor when its max moves.
//!
//! Every output element is owned by one thread and every row statistic by
//! one thread, keys visited in order: bit-reproducible, no atomics. The
//! whole working set — two operand slabs, the score tile, the V block and
//! the row statistics — is 23 KB, inside the 32 KB this adapter offers.

/// Queries per workgroup.
pub const TQ: u32 = 64;
/// Keys per step.
pub const TK: u32 = 32;
/// Depth of one staged slab of `hd`.
const DC: u32 = 16;
/// Threads: a 16 x 8 grid of 4 x 4 score sub-tiles.
const THREADS: u32 = (TQ / 4) * (TK / 4);

/// Is `hd` served by this kernel? (Multiples of 32: the P·V register tile
/// is `hd / 8` columns per thread, in vec4s.)
pub fn supports(hd: usize) -> bool { hd % 32 == 0 && hd >= 32 }

pub fn forward_wgsl(hd: usize) -> String {
    let v4 = hd / 4;                 // vec4s per row of V / O
    let cpt = hd / 8;                // output columns per thread
    let cv4 = cpt / 4;               // vec4s of output per row per thread
    let dchunks = hd / DC as usize;

    // O accumulators o{i}_{c}, i in 0..4 rows, c in 0..cv4
    let mut o_decl = String::new();
    for i in 0..4 { for c in 0..cv4 { o_decl += &format!("    var o{i}_{c} = vec4<f32>(0.0);\n"); } }
    let mut o_rescale = String::new();
    for i in 0..4 { for c in 0..cv4 { o_rescale += &format!("        o{i}_{c} = o{i}_{c} * cr{i};\n"); } }
    let mut pv = String::new();
    for c in 0..cv4 { pv += &format!("            let v{c} = Vs[j * {v4}u + tx * {cv4}u + {c}u];\n"); }
    for i in 0..4 {
        pv += &format!("            let p{i} = Ss[(ty * 4u + {i}u) * TK + j];\n");
        for c in 0..cv4 { pv += &format!("            o{i}_{c} = fma(vec4<f32>(p{i}), v{c}, o{i}_{c});\n"); }
    }
    let mut o_store = String::new();
    for i in 0..4 {
        o_store += &format!("    if (qb * TQ + ty * 4u + {i}u < d.seq) {{\n        let inv{i} = 1.0 / ls[ty * 4u + {i}u];\n        let off{i} = (base + qb * TQ + ty * 4u + {i}u) * qw + qh * HD + tx * {cpt}u;\n");
        for c in 0..cv4 {
            for (l, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                o_store += &format!("        O[off{i} + {}u] = o{i}_{c}.{comp} * inv{i};\n", c * 4 + l);
            }
        }
        o_store += &format!("        if (tx == 0u) {{ L[(base + qb * TQ + ty * 4u + {i}u) * d.nh + qh] = ms[ty * 4u + {i}u] + log(ls[ty * 4u + {i}u]); }}\n    }}\n");
    }

    format!(r#"
struct Dims {{ nseq: u32, seq: u32, nh: u32, nkv: u32, hd: u32, group: u32, pad0: u32, pad1: u32, scale: f32, pad2: f32, pad3: f32, pad4: f32 }};
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read_write> L: array<f32>;
@group(0) @binding(5) var<uniform> d: Dims;

const HD: u32 = {hd}u;
const TQ: u32 = {TQ}u;
const TK: u32 = {TK}u;
const DC: u32 = {DC}u;
const NEG: f32 = -1.0e30;

// Qt[dd][i] and Kt[dd][j] for one 16-deep slab, vec4 over i / j.
var<workgroup> Qt: array<vec4<f32>, {qt_len}u>;
var<workgroup> Kt: array<vec4<f32>, {kt_len}u>;
// Scores, then probabilities: Ss[i][j], row-major.
var<workgroup> Ss: array<f32, {ss_len}u>;
// V block, vec4 over the columns: Vs[j][c/4].
var<workgroup> Vs: array<vec4<f32>, {vs_len}u>;
var<workgroup> ms: array<f32, {TQ}u>;
var<workgroup> ls: array<f32, {TQ}u>;
var<workgroup> cs: array<f32, {TQ}u>;

@compute @workgroup_size({THREADS}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let sq = wg.z;
    let qh = wg.y;
    let kvh = qh / d.group;
    let qb = wg.x;
    let base = sq * d.seq;
    let qw = d.nh * HD;
    let kw = d.nkv * HD;
    let ty = tid / 8u;                 // 0..16: rows ty*4 .. +4
    let tx = tid % 8u;                 // 0..8:  keys tx*4 .. +4 (scores), cols tx*{cpt} .. (output)
    if (tid < TQ) {{ ms[tid] = NEG; ls[tid] = 0.0; }}
{o_decl}
    let qlast = min(qb * TQ + TQ, d.seq) - 1u;        // last query of this block
    let nkb = qlast / TK + 1u;                          // key steps that reach it
    for (var kb = 0u; kb < nkb; kb = kb + 1u) {{
        // ── S = Q·Kᵀ over hd, in slabs of DC ──
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var dc = 0u; dc < {dchunks}u; dc = dc + 1u) {{
            let d0 = dc * DC;
            // stage Qt: TQ rows x DC depth vec4 reads, {q_per} per thread,
            // each a contiguous vec4 of a row, scattered transposed
            for (var t = 0u; t < {q_per}u; t = t + 1u) {{
                let idx = tid + t * {THREADS}u;
                let i = idx / {D4}u;
                let d4 = idx % {D4}u;
                let row = qb * TQ + i;
                var qv = vec4<f32>(0.0);
                if (row < d.seq) {{
                    let off = (base + row) * qw + qh * HD + d0 + d4 * 4u;
                    qv = vec4<f32>(Q[off], Q[off + 1u], Q[off + 2u], Q[off + 3u]);
                }}
                Qt[(d4 * 4u + 0u) * (TQ / 4u) + i / 4u][i % 4u] = qv.x;
                Qt[(d4 * 4u + 1u) * (TQ / 4u) + i / 4u][i % 4u] = qv.y;
                Qt[(d4 * 4u + 2u) * (TQ / 4u) + i / 4u][i % 4u] = qv.z;
                Qt[(d4 * 4u + 3u) * (TQ / 4u) + i / 4u][i % 4u] = qv.w;
            }}
            // stage Kt: TK keys x DC depth vec4 reads, {k_per} per thread
            for (var t = 0u; t < {k_per}u; t = t + 1u) {{
                let idx = tid + t * {THREADS}u;
                let j = idx / {D4}u;
                let d4 = idx % {D4}u;
                let row = kb * TK + j;
                var kv = vec4<f32>(0.0);
                if (row < d.seq) {{
                    let off = (base + row) * kw + kvh * HD + d0 + d4 * 4u;
                    kv = vec4<f32>(K[off], K[off + 1u], K[off + 2u], K[off + 3u]);
                }}
                Kt[(d4 * 4u + 0u) * (TK / 4u) + j / 4u][j % 4u] = kv.x;
                Kt[(d4 * 4u + 1u) * (TK / 4u) + j / 4u][j % 4u] = kv.y;
                Kt[(d4 * 4u + 2u) * (TK / 4u) + j / 4u][j % 4u] = kv.z;
                Kt[(d4 * 4u + 3u) * (TK / 4u) + j / 4u][j % 4u] = kv.w;
            }}
            workgroupBarrier();
            for (var dd = 0u; dd < DC; dd = dd + 1u) {{
                let a = Qt[dd * (TQ / 4u) + ty];
                let b = Kt[dd * (TK / 4u) + tx];
                s0 = fma(vec4<f32>(a.x), b, s0);
                s1 = fma(vec4<f32>(a.y), b, s1);
                s2 = fma(vec4<f32>(a.z), b, s2);
                s3 = fma(vec4<f32>(a.w), b, s3);
            }}
            workgroupBarrier();
        }}
        // ── scaled, causally masked scores to Ss; V block staged ──
        for (var i = 0u; i < 4u; i = i + 1u) {{
            var sr: vec4<f32>;
            if (i == 0u) {{ sr = s0; }} else if (i == 1u) {{ sr = s1; }} else if (i == 2u) {{ sr = s2; }} else {{ sr = s3; }}
            let qi = qb * TQ + ty * 4u + i;
            for (var l = 0u; l < 4u; l = l + 1u) {{
                let kj = kb * TK + tx * 4u + l;
                var sv = sr[l] * d.scale;
                if (kj > qi || kj >= d.seq) {{ sv = NEG; }}
                Ss[(ty * 4u + i) * TK + tx * 4u + l] = sv;
            }}
        }}
        for (var t = 0u; t < {vs_per}u; t = t + 1u) {{
            let idx = tid + t * {THREADS}u;
            if (idx < {vs_len}u) {{
                let j = idx / {v4}u;
                let c4 = idx % {v4}u;
                let row = kb * TK + j;
                var vv = vec4<f32>(0.0);
                if (row < d.seq) {{
                    let off = (base + row) * kw + kvh * HD + c4 * 4u;
                    vv = vec4<f32>(V[off], V[off + 1u], V[off + 2u], V[off + 3u]);
                }}
                Vs[idx] = vv;
            }}
        }}
        workgroupBarrier();
        // ── online softmax, one thread per row ──
        if (tid < TQ) {{
            var mloc = NEG;
            for (var j = 0u; j < TK; j = j + 1u) {{ mloc = max(mloc, Ss[tid * TK + j]); }}
            let mnew = max(ms[tid], mloc);
            let corr = exp(ms[tid] - mnew);
            var sum = 0.0;
            for (var j = 0u; j < TK; j = j + 1u) {{
                let p = exp(Ss[tid * TK + j] - mnew);
                Ss[tid * TK + j] = p;
                sum = sum + p;
            }}
            ls[tid] = ls[tid] * corr + sum;
            ms[tid] = mnew;
            cs[tid] = corr;
        }}
        workgroupBarrier();
        // ── O = O * corr + P·V ──
        let cr0 = cs[ty * 4u];
        let cr1 = cs[ty * 4u + 1u];
        let cr2 = cs[ty * 4u + 2u];
        let cr3 = cs[ty * 4u + 3u];
{o_rescale}
        for (var j = 0u; j < TK; j = j + 1u) {{
{pv}        }}
        workgroupBarrier();
    }}
{o_store}}}
"#,
        hd = hd, TQ = TQ, TK = TK, DC = DC, THREADS = THREADS,
        qt_len = DC as usize * TQ as usize / 4, kt_len = DC as usize * TK as usize / 4,
        ss_len = TQ as usize * TK as usize, vs_len = TK as usize * v4,
        vs_per = (TK as usize * v4).div_ceil(THREADS as usize),
        v4 = v4, cpt = cpt, dchunks = dchunks, D4 = DC / 4,
        q_per = (TQ * DC / 4) / THREADS, k_per = (TK * DC / 4) / THREADS,
        o_decl = o_decl, o_rescale = o_rescale, pv = pv, o_store = o_store,
    )
}
