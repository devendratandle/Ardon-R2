//! The elementwise tail of a transformer step on CUDA, forward and backward:
//! the `elementwise` module's public API over `kernels/elementwise.cu`, so the
//! trainer in `r2_train::gpu_llm` runs on either backend. f32 storage only.
//!
//! Every function returns `false`, with nothing queued, when there is no device
//! or a buffer is too short: a kernel past the end of a buffer would fault the
//! context, and that would end every later call too.

use super::ffi::{device, Arg, DeviceBuffer, Kernel};
use super::{loaded, Loaded, Tensor};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// As in `kernels/elementwise.cu`: 8 rows to a block, a warp of 32 threads to each.
const ROW_BLOCK: u32 = 256;
const LANES: usize = 32;
const FLAT: u32 = 256;
/// Row blocks the rmsnorm dW reduction is split across.
const RM_PARTS: usize = 64;

const PTX: &str = include_str!("kernels/elementwise.ptx");

struct Kernels {
    rm_fwd: Kernel, rm_bwd_x: Kernel, rm_part: Kernel, rm_sum: Kernel,
    flat_fwd: Kernel, flat_bwd: Kernel, rope: Kernel,
    ce_fwd: Kernel, ce_bwd: Kernel, nonfinite: Kernel, embed_fwd: Kernel, embed_bwd: Kernel,
}
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Option<&'static Kernels> { all().ok() }

fn all() -> Result<&'static Kernels, &'static str> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels {
        rm_fwd: m.kernel("rmsnorm_fwd", 6)?,
        rm_bwd_x: m.kernel("rmsnorm_bwd_x", 9)?,
        rm_part: m.kernel("rmsnorm_bwd_w_part", 7)?,
        rm_sum: m.kernel("rmsnorm_bwd_w_sum", 6)?,
        flat_fwd: m.kernel("flat_fwd", 5)?,
        flat_bwd: m.kernel("flat_bwd", 11)?,
        rope: m.kernel("rope", 9)?,
        ce_fwd: m.kernel("softmax_ce_fwd", 6)?,
        ce_bwd: m.kernel("softmax_ce_bwd", 8)?,
        nonfinite: m.kernel("nonfinite", 4)?,
        embed_fwd: m.kernel("embed_fwd", 5)?,
        embed_bwd: m.kernel("embed_bwd", 7)?,
    }))
}

/// Whether the module loads, for `module_status`.
pub(super) fn loads() -> Result<(), &'static str> { all().map(|_| ()) }

/// `threads` one-thread-per-item launches in blocks of `block`. `false` when
/// the count does not fit the kernels' 32-bit indices.
fn run(k: &Kernel, threads: usize, block: u32, args: &[Arg]) -> bool {
    let Ok(t) = u32::try_from(threads) else { return false };
    t == 0 || k.launch([t.div_ceil(block), 1, 1], [block, 1, 1], args).is_ok()
}

fn int(x: usize) -> Arg<'static> { Arg::Int(x as u32) }

/// Scratch `rmsnorm_bwd` needs for `rows` x `d`: the per-row `rinv`, then the
/// dW partials.
pub fn rmsnorm_scratch_len(rows: usize, d: usize) -> usize { rows + RM_PARTS * d }

/// `y[r] = x[r] / rms(x[r]) * w`, rows of `d`.
pub fn rmsnorm_fwd(x: &Tensor, w: &Tensor, y: &Tensor, rows: usize, d: usize, eps: f32) -> bool {
    let Some(ks) = kernels() else { return false };
    if x.len() < rows * d || y.len() < rows * d || w.len() < d { return false; }
    run(&ks.rm_fwd, rows * LANES, ROW_BLOCK, &[Arg::F(x), Arg::F(w), Arg::F(y), int(rows), int(d), Arg::Real(eps)])
}

/// rmsnorm backward: `gx (=|+=)` per `assign_x`, `gw (=|+=)` per `assign_w`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bwd(x: &Tensor, w: &Tensor, g: &Tensor, gx: &Tensor, gw: &Tensor, scratch: &Tensor,
                   rows: usize, d: usize, eps: f32, assign_x: bool, assign_w: bool) -> bool {
    let Some(ks) = kernels() else { return false };
    if [x, g, gx].iter().any(|t| t.len() < rows * d) || w.len() < d || gw.len() < d
        || scratch.len() < rmsnorm_scratch_len(rows, d) { return false; }
    let parts = RM_PARTS.min(rows).max(1);
    let (off, s) = (int(rows), Arg::F(scratch));
    run(&ks.rm_bwd_x, rows * LANES, ROW_BLOCK, &[Arg::F(x), Arg::F(w), Arg::F(g), Arg::F(gx), s,
                                           int(rows), int(d), Arg::Real(eps), int(!assign_x as usize)])
        && run(&ks.rm_part, parts * d, FLAT, &[Arg::F(x), Arg::F(g), s, off, int(rows), int(d), int(parts)])
        && run(&ks.rm_sum, d, FLAT, &[s, off, Arg::F(gw), int(d), int(parts), int(!assign_w as usize)])
}

/// The flat forward ops. `AddInto` is `y += a` in place and ignores `b`; `Silu`
/// and `Copy` ignore `b` too.
#[derive(Clone, Copy)]
pub enum Flat { Silu, Mul, Add, SiluMul, AddInto, Copy }

/// `y = op(a, b)` over `n` elements.
pub fn flat_fwd(op: Flat, a: &Tensor, b: &Tensor, y: &Tensor, n: usize) -> bool {
    let Some(ks) = kernels() else { return false };
    let code = match op { Flat::Silu => 0, Flat::Mul => 1, Flat::Add => 2, Flat::SiluMul => 3, Flat::AddInto => 4, Flat::Copy => 5 };
    if a.len() < n || b.len() < n && matches!(op, Flat::Mul | Flat::Add | Flat::SiluMul) || y.len() < n { return false; }
    run(&ks.flat_fwd, n, FLAT, &[Arg::F(a), Arg::F(b), Arg::F(y), int(n), int(code)])
}

/// Backward of `Silu` / `Mul` / `Add`: `ga`, `gb` each `Some((buffer, assign))`
/// when wanted.
pub fn flat_bwd(op: Flat, a: &Tensor, b: &Tensor, g: &Tensor, ga: Option<(&Tensor, bool)>, gb: Option<(&Tensor, bool)>, n: usize) -> bool {
    let Some(ks) = kernels() else { return false };
    let code = match op { Flat::Silu => 0, Flat::Mul | Flat::SiluMul => 1, Flat::Add | Flat::AddInto | Flat::Copy => 2 };
    // an unwanted output is never touched by the kernel, so `g` stands in
    let (gat, ga_add, ga_on) = ga.map_or((g, 0, 0), |(t, assign)| (t, !assign as usize, 1));
    let (gbt, gb_add, gb_on) = gb.map_or((g, 0, 0), |(t, assign)| (t, !assign as usize, 1));
    if [a, b, g, gat, gbt].iter().any(|t| t.len() < n) { return false; }
    run(&ks.flat_bwd, n, FLAT, &[Arg::F(a), Arg::F(b), Arg::F(g), Arg::F(gat), Arg::F(gbt), int(n),
                                 int(code), int(ga_add), int(gb_add), int(ga_on), int(gb_on)])
}

/// `(cos, sin)` per (position, pair), flattened: the CPU tape's table, by the
/// expressions `elementwise::rope_table` documents.
pub fn rope_table(period: usize, head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut t = Vec::with_capacity(period.max(1) * half * 2);
    for pos in 0..period.max(1) {
        for p in 0..half {
            let freq = (base as f64).powf(-2.0 * p as f64 / head_dim as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            t.push(c as f32);
            t.push(s as f32);
        }
    }
    t
}

/// The table on the device, built once per (period, head_dim, base).
fn rope_table_dev(period: usize, head_dim: usize, base: f32) -> Option<&'static Tensor> {
    static TABLES: Mutex<Option<HashMap<(usize, usize, u32), &'static Tensor>>> = Mutex::new(None);
    let key = (period.max(1), head_dim, base.to_bits());
    let mut guard = TABLES.lock().unwrap_or_else(|e| e.into_inner());
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(t) = table.get(&key) { return Some(t); }
    let t: &'static Tensor = Box::leak(Box::new(Tensor::upload(&rope_table(period, head_dim, base))?));
    table.insert(key, t);
    Some(t)
}

#[allow(clippy::too_many_arguments)]
fn rope(x: &Tensor, y: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32, bwd: bool, add: bool) -> bool {
    let Some(ks) = kernels() else { return false };
    if head_dim < 2 || heads == 0 || x.len() < rows * heads * head_dim || y.len() < rows * heads * head_dim { return false; }
    let Some(tab) = rope_table_dev(period, head_dim, base) else { return false };
    run(&ks.rope, rows * heads * (head_dim / 2), FLAT, &[Arg::F(x), Arg::F(tab), Arg::F(y), int(rows), int(head_dim),
                                                         int(period.max(1)), int(heads), int(bwd as usize), int(add as usize)])
}

/// RoPE forward: `y = rotate(x)`, positions restarting every `period` rows; rows
/// are `heads x head_dim` wide.
pub fn rope_fwd(x: &Tensor, y: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32) -> bool {
    rope(x, y, rows, period, heads, head_dim, base, false, false)
}

/// RoPE backward: `gx (=|+=) rotate⁻¹(g)`.
#[allow(clippy::too_many_arguments)]
pub fn rope_bwd(g: &Tensor, gx: &Tensor, rows: usize, period: usize, heads: usize, head_dim: usize, base: f32, assign: bool) -> bool {
    rope(g, gx, rows, period, heads, head_dim, base, true, !assign)
}

/// Token ids on the device. `IGNORE` (`usize::MAX`) narrows to the kernels'
/// `0xFFFFFFFF`.
pub type Ids = DeviceBuffer<u32>;

pub fn upload_ids(ids: &[usize]) -> Option<Ids> {
    let words: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    Ids::from_slice(device().ok()?, &words).ok()
}

/// Softmax cross-entropy forward: per row `lse` and `loss = lse - x[t]`.
pub fn softmax_ce_fwd(x: &Tensor, ids: &Ids, lse: &Tensor, loss: &Tensor, rows: usize, d: usize) -> bool {
    let Some(ks) = kernels() else { return false };
    if x.len() < rows * d || ids.len() < rows || lse.len() < rows || loss.len() < rows { return false; }
    run(&ks.ce_fwd, rows * LANES, ROW_BLOCK, &[Arg::F(x), Arg::U(ids), Arg::F(lse), Arg::F(loss), int(rows), int(d)])
}

/// `gx (=|+=) inv * (softmax(x) - onehot)`.
#[allow(clippy::too_many_arguments)]
pub fn softmax_ce_bwd(x: &Tensor, ids: &Ids, lse: &Tensor, gx: &Tensor, rows: usize, d: usize, inv: f32, assign: bool) -> bool {
    let Some(ks) = kernels() else { return false };
    if x.len() < rows * d || gx.len() < rows * d || ids.len() < rows || lse.len() < rows { return false; }
    run(&ks.ce_bwd, rows * d, FLAT, &[Arg::F(x), Arg::U(ids), Arg::F(lse), Arg::F(gx), int(rows * d), int(d), Arg::Real(inv), int(!assign as usize)])
}

/// Sets `flag[at] = 1` when any of `x[0..n]` is inf or nan; leaves it otherwise.
pub fn nonfinite_flag(x: &Tensor, flag: &Tensor, at: usize, n: usize) -> bool {
    let Some(ks) = kernels() else { return false };
    if x.len() < n || flag.len() <= at { return false; }
    run(&ks.nonfinite, n, FLAT, &[Arg::F(x), Arg::F(flag), int(n), int(at)])
}

/// `y[i] = table[ids[i]]`.
pub fn embed_fwd(table: &Tensor, ids: &Ids, y: &Tensor, rows: usize, d: usize) -> bool {
    let Some(ks) = kernels() else { return false };
    if ids.len() < rows || y.len() < rows * d || d == 0 { return false; }
    run(&ks.embed_fwd, rows * d, FLAT, &[Arg::F(table), Arg::U(ids), Arg::F(y), int(rows * d), int(d)])
}

/// The positions of each token id as CSR, for [`embed_bwd`].
pub struct TokenIndex { offs: Ids, pos: Ids }

impl TokenIndex {
    pub fn build(ids: &[usize], vocab: usize) -> Option<TokenIndex> {
        let mut counts = vec![0u32; vocab + 1];
        for &t in ids { *counts.get_mut(t + 1)? += 1; }
        for v in 0..vocab { counts[v + 1] += counts[v]; }
        let mut fill = counts.clone();
        let mut pos = vec![0u32; ids.len()];
        for (i, &t) in ids.iter().enumerate() { pos[fill[t] as usize] = i as u32; fill[t] += 1; }
        let dev = device().ok()?;
        Some(TokenIndex { offs: Ids::from_slice(dev, &counts).ok()?, pos: Ids::from_slice(dev, &pos).ok()? })
    }
}

/// `gtable[v] (=|+=) Σ_{i: ids[i]=v} g[i]`.
pub fn embed_bwd(g: &Tensor, index: &TokenIndex, gtable: &Tensor, vocab: usize, d: usize, assign: bool) -> bool {
    let Some(ks) = kernels() else { return false };
    if gtable.len() < vocab * d || index.offs.len() < vocab + 1 || d == 0 { return false; }
    run(&ks.embed_bwd, vocab * d, FLAT, &[Arg::F(g), Arg::U(&index.offs), Arg::U(&index.pos), Arg::F(gtable),
                                          int(vocab * d), int(d), int(!assign as usize)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::have_device;
    use r2_autograd::Tape;

    fn mk(n: usize, ph: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.9).collect() }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length (a fault downloads nothing)");
        let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() <= tol * scale, "{what}[{i}]: device {a} vs cpu {b}");
        }
    }

    #[test]
    fn rope_table_matches_the_cpu() {
        for &(per, hd, base) in &[(64usize, 64usize, 10000.0f32), (256, 64, 10000.0), (8, 32, 500000.0)] {
            let flat: Vec<f32> = r2_tensor::ops::rope_table(per, hd, base).into_iter().flat_map(|(c, s)| [c, s]).collect();
            assert!(rope_table(per, hd, base).iter().zip(&flat).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// Every kernel against the tape, forward value and every gradient, with
    /// accumulate-onto-existing exercised for the backward arms.
    #[test]
    fn kernels_match_the_tape() {
        if !have_device() { return; }
        let up = |v: &[f32]| Tensor::upload(v).unwrap();
        let z = |n: usize| Tensor::zeros(n).unwrap();
        let (rows, d) = (37usize, 96usize);
        let (x, w, b, g) = (mk(rows * d, 0.0), mk(d, 1.0), mk(rows * d, 2.0), mk(rows * d, 3.0));
        let (tx, tw, tb, tg, y) = (up(&x), up(&w), up(&b), up(&g), z(rows * d));

        let mut tape = Tape::new();
        let (vx, vw) = (tape.leaf(x.clone(), true), tape.leaf(w.clone(), true));
        let vy = tape.rmsnorm(vx, vw, d, 1e-5);
        tape.backward_from(vy, &g);
        assert!(rmsnorm_fwd(&tx, &tw, &y, rows, d, 1e-5));
        close(&y.download(), tape.value(vy), 1e-5, "rmsnorm");
        let (gx, gw, ri) = (up(&b), up(&w), z(rmsnorm_scratch_len(rows, d)));
        assert!(rmsnorm_bwd(&tx, &tw, &tg, &gx, &gw, &ri, rows, d, 1e-5, false, false));
        let plus = |a: &[f32], c: &[f32]| -> Vec<f32> { a.iter().zip(c).map(|(p, q)| p + q).collect() };
        close(&gx.download(), &plus(tape.grad(vx), &b), 2e-5, "rmsnorm dx (accumulate)");
        close(&gw.download(), &plus(tape.grad(vw), &w), 2e-5, "rmsnorm dw (accumulate)");
        let (gx, gw) = (z(rows * d), z(d));
        assert!(rmsnorm_bwd(&tx, &tw, &tg, &gx, &gw, &ri, rows, d, 1e-5, true, true));
        close(&gx.download(), tape.grad(vx), 2e-5, "rmsnorm dx");
        close(&gw.download(), tape.grad(vw), 2e-5, "rmsnorm dw");

        for (op, name) in [(Flat::Silu, "silu"), (Flat::Mul, "mul"), (Flat::Add, "add")] {
            let mut tape = Tape::new();
            let (va, vb) = (tape.leaf(x.clone(), true), tape.leaf(b.clone(), true));
            let vy = match op { Flat::Silu => tape.silu(va), Flat::Mul => tape.mul(va, vb), _ => tape.add(va, vb) };
            tape.backward_from(vy, &g);
            assert!(flat_fwd(op, &tx, &tb, &y, rows * d));
            close(&y.download(), tape.value(vy), 1e-5, name);
            let (ga, gb) = (z(rows * d), z(rows * d));
            let want_b = !matches!(op, Flat::Silu);
            assert!(flat_bwd(op, &tx, &tb, &tg, Some((&ga, true)), want_b.then_some((&gb, true)), rows * d));
            close(&ga.download(), tape.grad(va), 2e-5, name);
            if want_b { close(&gb.download(), tape.grad(vb), 2e-5, name); }
        }
        assert!(flat_fwd(Flat::Copy, &tx, &tx, &y, rows * d));
        close(&y.download(), &x, 0.0, "copy");
        assert!(flat_fwd(Flat::AddInto, &tb, &tb, &y, rows * d));
        close(&y.download(), &plus(&x, &b), 0.0, "add into");
        assert!(flat_fwd(Flat::SiluMul, &tx, &tb, &y, rows * d));
        let want: Vec<f32> = x.iter().zip(&b).map(|(a, c)| a / (1.0 + (-a).exp()) * c).collect();
        close(&y.download(), &want, 1e-5, "silu*mul");

        let flag = z(4);
        assert!(nonfinite_flag(&tb, &flag, 2, rows * d));
        assert_eq!(flag.download(), vec![0.0; 4], "clean data must not raise the flag");
        let mut bad = b.clone();
        bad[rows * d / 2] = f32::INFINITY;
        assert!(nonfinite_flag(&up(&bad), &flag, 2, rows * d));
        assert_eq!(flag.download(), vec![0.0, 0.0, 1.0, 0.0], "an inf must raise the flag at `at`");

        // rope: 5 sequences of 8, 3 heads of 32. With the host's table there is
        // no device transcendental, so this is the tape's bits.
        let (nseq, seq, nh, hd) = (5usize, 8usize, 3usize, 32usize);
        let (xr, gr) = (mk(nseq * seq * nh * hd, 0.4), mk(nseq * seq * nh * hd, 1.4));
        let mut tape = Tape::new();
        let vx = tape.leaf(xr.clone(), true);
        let vy = tape.rope_seq(vx, nseq * seq, seq, nh, hd, 10000.0);
        tape.backward_from(vy, &gr);
        let (txr, tgr, yr) = (up(&xr), up(&gr), z(xr.len()));
        assert!(rope_fwd(&txr, &yr, nseq * seq, seq, nh, hd, 10000.0));
        close(&yr.download(), tape.value(vy), 0.0, "rope");
        assert!(rope_bwd(&tgr, &yr, nseq * seq, seq, nh, hd, 10000.0, true));
        close(&yr.download(), tape.grad(vx), 0.0, "rope dx");

        // softmax cross-entropy: 19 rows of vocab 300
        let (r, v) = (19usize, 300usize);
        let logits = mk(r * v, 0.7);
        let targets: Vec<usize> = (0..r).map(|i| (i * 37) % v).collect();
        let mut tape = Tape::new();
        let vl = tape.leaf(logits.clone(), true);
        let loss = tape.softmax_ce(vl, v, targets.clone());
        tape.backward_from(loss, &[1.0]);
        let (tl, ids, lse, lossv, gl) = (up(&logits), upload_ids(&targets).unwrap(), z(r), z(r), z(r * v));
        assert!(softmax_ce_fwd(&tl, &ids, &lse, &lossv, r, v));
        let mean = lossv.download().iter().sum::<f32>() / r as f32;
        assert!((mean - tape.value(loss)[0]).abs() < 1e-4, "loss {mean} vs {}", tape.value(loss)[0]);
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
        let (tt, tids, ye) = (up(&table), upload_ids(&toks).unwrap(), z(n * de));
        assert!(embed_fwd(&tt, &tids, &ye, n, de));
        close(&ye.download(), tape.value(ve), 0.0, "embed");
        let idx = TokenIndex::build(&toks, vocab).unwrap();
        let gt = z(vocab * de);
        assert!(embed_bwd(&up(&ge), &idx, &gt, vocab, de, true));
        close(&gt.download(), tape.grad(vt), 1e-6, "embed dtable");
    }

    /// Short buffers and a token past the vocabulary are refused on the host,
    /// before anything can fault the context.
    #[test]
    fn short_buffers_are_refused() {
        if !have_device() { return; }
        let (a, b) = (Tensor::zeros(4).unwrap(), Tensor::zeros(100).unwrap());
        assert!(!rmsnorm_fwd(&a, &b, &b, 10, 10, 1e-5));
        assert!(!flat_fwd(Flat::Mul, &b, &a, &b, 100));
        assert!(!rope_fwd(&a, &b, 1, 1, 1, 8, 10000.0));
        assert!(TokenIndex::build(&[5], 3).is_none());
    }
}
