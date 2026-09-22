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

/// The tiled backward's block: keys per workgroup, queries per step.
/// Square, so the causal walk starts at the diagonal block.
pub const BT: u32 = 32;
/// Threads of the backward kernel: an 8 x 8 grid of 4 x 4 sub-tiles.
const BTHREADS: u32 = (BT / 4) * (BT / 4);

/// Is `hd` served by this kernel? (Multiples of 32: the P·V register tile
/// is `hd / 8` columns per thread, in vec4s.)
pub fn supports(hd: usize) -> bool { hd % 32 == 0 && hd >= 32 }

/// Is `hd` served by the tiled dK/dV kernel? Multiples of 32 up to 64:
/// each thread carries 4 keys x `hd/8` columns of BOTH dK and dV in
/// registers for the whole walk, and past 64 that spills to scratch —
/// which is what cost the GEMM a factor of twenty when it was tried.
pub fn supports_bwd(hd: usize) -> bool { hd % 32 == 0 && (32..=64).contains(&hd) }

pub fn forward_wgsl(hd: usize, act: crate::device::Dtype) -> String {
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
                o_store += &format!("        O[off{i} + {}u] = {ty}(o{i}_{c}.{comp} * inv{i});\n", c * 4 + l, ty = act.wgsl());
            }
        }
        o_store += &format!("        if (tx == 0u) {{ L[(base + qb * TQ + ty * 4u + {i}u) * d.nh + qh] = ms[ty * 4u + {i}u] + log(ls[ty * 4u + {i}u]); }}\n    }}\n");
    }

    format!(r#"
{enable}struct Dims {{ nseq: u32, seq: u32, nh: u32, nkv: u32, hd: u32, group: u32, pad0: u32, pad1: u32, scale: f32, pad2: f32, pad3: f32, pad4: f32 }};
@group(0) @binding(0) var<storage, read> Q: array<{ty}>;
@group(0) @binding(1) var<storage, read> K: array<{ty}>;
@group(0) @binding(2) var<storage, read> V: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> O: array<{ty}>;
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
                    qv = vec4<f32>(f32(Q[off]), f32(Q[off + 1u]), f32(Q[off + 2u]), f32(Q[off + 3u]));
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
                    kv = vec4<f32>(f32(K[off]), f32(K[off + 1u]), f32(K[off + 2u]), f32(K[off + 3u]));
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
                    vv = vec4<f32>(f32(V[off]), f32(V[off + 1u]), f32(V[off + 2u]), f32(V[off + 3u]));
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
        enable = if act == crate::device::Dtype::F16 { "enable f16;\n" } else { "" }, ty = act.wgsl(),
        qt_len = DC as usize * TQ as usize / 4, kt_len = DC as usize * TK as usize / 4,
        ss_len = TQ as usize * TK as usize, vs_len = TK as usize * v4,
        vs_per = (TK as usize * v4).div_ceil(THREADS as usize),
        v4 = v4, cpt = cpt, dchunks = dchunks, D4 = DC / 4,
        q_per = (TQ * DC / 4) / THREADS, k_per = (TK * DC / 4) / THREADS,
        o_decl = o_decl, o_rescale = o_rescale, pv = pv, o_store = o_store,
    )
}

/// The tiled backward for dK and dV, in one kernel.
///
/// The reference kernels in `attention.rs` give one thread a whole key
/// row and walk the queries: every thread re-reads the staged Q and dO
/// rows, one workgroup-memory load per FMA, and dK and dV are two
/// launches that each recompute `P` from scratch. Here a workgroup owns
/// `BT` keys and walks the query blocks once, computing both:
///
/// 1. `Sᵀ = K·Qᵀ` and `dPᵀ = V·dOᵀ`, each a `BT x BT` tile accumulated
///    as a GEMM over `hd` in 16-deep slabs — operands staged transposed,
///    each thread holding a 4 x 4 sub-tile in `vec4`s, two loads per
///    four FMAs.
/// 2. `Pᵀ = exp(scale·Sᵀ − L)` and `dSᵀ = scale·Pᵀ·(dPᵀ − D)`,
///    causally masked, into workgroup memory.
/// 3. `dV += Pᵀ·dO` and `dK += dSᵀ·Q`, the `BT x hd` result tiles held in
///    registers — 4 keys x `hd/8` columns per thread — for the whole
///    walk, including across the query heads that share this kv head.
///
/// Every output element is produced by one thread summing in a fixed
/// order (query heads, then query blocks, then queries), so this is as
/// reproducible as the reference kernel, and it agrees with it to f32
/// rounding — the arithmetic is the same, the association differs.
///
/// dQ is not here: its rows belong to a different workgroup, and
/// accumulating them would need atomics. It stays a kernel of its own.
pub fn dkv_wgsl(hd: usize, act: crate::device::Dtype) -> String {
    let v4 = hd / 4;                       // vec4s per row of a BT x hd block
    let cpt = hd / 8;                      // columns of dK/dV per thread
    let cv4 = cpt / 4;                     // and in vec4s
    let dchunks = hd / DC as usize;
    let elt = act.wgsl();

    let mut acc_decl = String::new();
    for r in 0..4 {
        for c in 0..cv4 {
            acc_decl += &format!("    var dv{r}_{c} = vec4<f32>(0.0);\n    var dk{r}_{c} = vec4<f32>(0.0);\n");
        }
    }

    // One transposed slab of `src` (rows of the block starting at `blk`)
    // into `dst`: `dst[dd * (BT/4) + i/4][i%4]`, two vec4 reads per thread.
    let stage_t = |dst: &str, src: &str, blk: &str, w: &str, h: &str| -> String {
        format!(r#"
            for (var t = 0u; t < {per}u; t = t + 1u) {{
                let idx = tid + t * {BTHREADS}u;
                let i = idx / {d4}u;
                let dd4 = idx % {d4}u;
                let row = {blk} + i;
                var vv = vec4<f32>(0.0);
                if (row < d.seq) {{
                    let off = (base + row) * {w} + {h} * HD + d0 + dd4 * 4u;
                    vv = vec4<f32>(f32({src}[off]), f32({src}[off + 1u]), f32({src}[off + 2u]), f32({src}[off + 3u]));
                }}
                {dst}[(dd4 * 4u + 0u) * (BT / 4u) + i / 4u][i % 4u] = vv.x;
                {dst}[(dd4 * 4u + 1u) * (BT / 4u) + i / 4u][i % 4u] = vv.y;
                {dst}[(dd4 * 4u + 2u) * (BT / 4u) + i / 4u][i % 4u] = vv.z;
                {dst}[(dd4 * 4u + 3u) * (BT / 4u) + i / 4u][i % 4u] = vv.w;
            }}"#,
            per = (BT * DC / 4) / BTHREADS, BTHREADS = BTHREADS, d4 = DC / 4)
    };
    // The 4 x 4 sub-tile of `At·Bt` over one staged slab.
    let tile = |acc: &str| -> String {
        let mut s = String::from("            for (var dd = 0u; dd < DC; dd = dd + 1u) {\n                let a = Kt[dd * (BT / 4u) + ty];\n                let b = Qt[dd * (BT / 4u) + tx];\n");
        for (r, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            s += &format!("                {acc}{r} = fma(vec4<f32>(a.{comp}), b, {acc}{r});\n");
        }
        s + "            }\n"
    };
    // Pᵀ and dSᵀ for this thread's sub-tile, causally masked.
    let mut elem = String::new();
    for r in 0..4 {
        elem += &format!(r#"        {{
            var sv = s{r};
            var gv = g{r};
            let kj = kb * BT + ty * 4u + {r}u;
            for (var c = 0u; c < 4u; c = c + 1u) {{
                let qi = qb * BT + tx * 4u + c;
                var p = 0.0;
                var ds = 0.0;
                if (kj <= qi && qi < d.seq && kj < d.seq) {{
                    p = exp(sv[c] * d.scale - Ls[tx * 4u + c]);
                    ds = p * (gv[c] - Ds[tx * 4u + c]) * d.scale;
                }}
                Ps[(ty * 4u + {r}u) * BT + tx * 4u + c] = p;
                Ss[(ty * 4u + {r}u) * BT + tx * 4u + c] = ds;
            }}
        }}
"#);
    }
    // A BT x hd block of `src`, row-major in vec4s, into Bs.
    let stage_b = |src: &str, w: &str, h: &str| -> String {
        format!(r#"        for (var t = 0u; t < {per}u; t = t + 1u) {{
            let idx = tid + t * {BTHREADS}u;
            if (idx < {bs_len}u) {{
                let i = idx / {v4}u;
                let c4 = idx % {v4}u;
                let row = qb * BT + i;
                var vv = vec4<f32>(0.0);
                if (row < d.seq) {{
                    let off = (base + row) * {w} + {h} * HD + c4 * 4u;
                    vv = vec4<f32>(f32({src}[off]), f32({src}[off + 1u]), f32({src}[off + 2u]), f32({src}[off + 3u]));
                }}
                Bs[idx] = vv;
            }}
        }}"#,
            per = (BT as usize * v4).div_ceil(BTHREADS as usize), BTHREADS = BTHREADS,
            bs_len = BT as usize * v4, v4 = v4)
    };
    // `acc += mat[key][i] * Bs[i][col]`, the whole query block.
    let accum = |acc: &str, mat: &str| -> String {
        let mut s = String::from("        for (var i = 0u; i < BT; i = i + 1u) {\n");
        for c in 0..cv4 { s += &format!("            let b{c} = Bs[i * {v4}u + cx * {cv4}u + {c}u];\n"); }
        for r in 0..4 {
            s += &format!("            let m{r} = {mat}[(cy * 4u + {r}u) * BT + i];\n");
            for c in 0..cv4 { s += &format!("            {acc}{r}_{c} = fma(vec4<f32>(m{r}), b{c}, {acc}{r}_{c});\n"); }
        }
        s + "        }\n"
    };
    let mut store = String::new();
    for r in 0..4 {
        store += &format!("    {{\n        let kj = kb * BT + cy * 4u + {r}u;\n        if (kj < d.seq) {{\n            let off = (base + kj) * kw + kvh * HD + cx * {cpt}u;\n");
        for c in 0..cv4 {
            for (l, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                let o = c * 4 + l;
                store += &format!("            dV[off + {o}u] = {elt}(dv{r}_{c}.{comp});\n            dK[off + {o}u] = {elt}(dk{r}_{c}.{comp});\n");
            }
        }
        store += "        }\n    }\n";
    }

    format!(r#"
{enable}struct Dims {{ nseq: u32, seq: u32, nh: u32, nkv: u32, hd: u32, group: u32, pad0: u32, pad1: u32, scale: f32, pad2: f32, pad3: f32, pad4: f32 }};
@group(0) @binding(0) var<storage, read> Q: array<{elt}>;
@group(0) @binding(1) var<storage, read> K: array<{elt}>;
@group(0) @binding(2) var<storage, read> V: array<{elt}>;
@group(0) @binding(3) var<storage, read> dO: array<{elt}>;
@group(0) @binding(4) var<storage, read> L: array<f32>;
@group(0) @binding(5) var<storage, read> Dl: array<f32>;
@group(0) @binding(6) var<storage, read_write> dK: array<{elt}>;
@group(0) @binding(7) var<storage, read_write> dV: array<{elt}>;
@group(0) @binding(8) var<uniform> d: Dims;

const HD: u32 = {hd}u;
const BT: u32 = {BT}u;
const DC: u32 = {DC}u;

// One staged slab of hd, transposed: Kt[dd][j], Qt[dd][i].
var<workgroup> Kt: array<vec4<f32>, {slab}u>;
var<workgroup> Qt: array<vec4<f32>, {slab}u>;
// Pᵀ and dSᵀ, both BT x BT, key-major.
var<workgroup> Ps: array<f32, {tile_len}u>;
var<workgroup> Ss: array<f32, {tile_len}u>;
// The query block's dO, then its Q: BT x hd, vec4 over the columns.
var<workgroup> Bs: array<vec4<f32>, {bs_len}u>;
var<workgroup> Ls: array<f32, {BT}u>;
var<workgroup> Ds: array<f32, {BT}u>;

@compute @workgroup_size({BTHREADS}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
    let sq = wg.z;
    let kvh = wg.y;
    let kb = wg.x;
    let base = sq * d.seq;
    let qw = d.nh * HD;
    let kw = d.nkv * HD;
    let ty = tid / 8u;                 // keys ty*4 .. +4 (both phases)
    let tx = tid % 8u;                 // queries tx*4 .. +4 (the score tile)
    let cy = ty;                       // keys cy*4 .. +4 (the dK/dV tile)
    let cx = tx;                       // columns cx*{cpt} .. +{cpt}
{acc_decl}
    let nqb = (d.seq + BT - 1u) / BT;
    for (var qh = kvh * d.group; qh < (kvh + 1u) * d.group; qh = qh + 1u) {{
        // causal: BT queries per block and BT keys per workgroup, so the
        // first block that can see this one is its own diagonal
        for (var qb = kb; qb < nqb; qb = qb + 1u) {{
            // ── Sᵀ = K·Qᵀ and dPᵀ = V·dOᵀ, over hd in slabs ──
            var s0 = vec4<f32>(0.0);
            var s1 = vec4<f32>(0.0);
            var s2 = vec4<f32>(0.0);
            var s3 = vec4<f32>(0.0);
            var g0 = vec4<f32>(0.0);
            var g1 = vec4<f32>(0.0);
            var g2 = vec4<f32>(0.0);
            var g3 = vec4<f32>(0.0);
            for (var dc = 0u; dc < {dchunks}u; dc = dc + 1u) {{
                let d0 = dc * DC;
{stage_kq}
                workgroupBarrier();
{tile_s}                workgroupBarrier();
{stage_vg}
                workgroupBarrier();
{tile_g}                workgroupBarrier();
            }}
            if (tid < BT) {{
                let row = qb * BT + tid;
                if (row < d.seq) {{
                    Ls[tid] = L[(base + row) * d.nh + qh];
                    Ds[tid] = Dl[(base + row) * d.nh + qh];
                }}
            }}
            workgroupBarrier();
            // ── Pᵀ, dSᵀ ──
{elem}            workgroupBarrier();
            // ── dV += Pᵀ·dO ──
{stage_do}
            workgroupBarrier();
{acc_dv}            workgroupBarrier();
            // ── dK += dSᵀ·Q ──
{stage_q}
            workgroupBarrier();
{acc_dk}            workgroupBarrier();
        }}
    }}
{store}}}
"#,
        hd = hd, BT = BT, DC = DC, BTHREADS = BTHREADS, cpt = cpt,
        enable = if act == crate::device::Dtype::F16 { "enable f16;\n" } else { "" }, elt = elt,
        slab = DC as usize * BT as usize / 4,
        tile_len = BT as usize * BT as usize,
        bs_len = BT as usize * v4,
        dchunks = dchunks,
        acc_decl = acc_decl,
        stage_kq = stage_t("Kt", "K", "kb * BT", "kw", "kvh") + &stage_t("Qt", "Q", "qb * BT", "qw", "qh"),
        stage_vg = stage_t("Kt", "V", "kb * BT", "kw", "kvh") + &stage_t("Qt", "dO", "qb * BT", "qw", "qh"),
        tile_s = tile("s"), tile_g = tile("g"),
        elem = elem,
        stage_do = stage_b("dO", "qw", "qh"), stage_q = stage_b("Q", "qw", "qh"),
        acc_dv = accum("dv", "Ps"), acc_dk = accum("dk", "Ss"),
        store = store,
    )
}
