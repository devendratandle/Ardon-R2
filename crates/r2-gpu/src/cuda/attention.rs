//! Fused causal grouped-query attention on CUDA, forward and backward: the
//! `attention` module's public API over `kernels/attention.cu`, which explains
//! the form. Scores are stored per (sequence, head), `seq x seq` each, in scratch
//! held here, and the products are batched GEMMs: cuBLAS's by default, the
//! register-tiled kernel without it or with `R2_CUDA_GEMM=tiled`.

use super::ffi::{blas, Arg, Kernel, Strided};
use super::gemm::{note_path, wanted, Path};
use super::{loaded, sync, Loaded, Tensor};
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Attention problem dimensions, as `attention::Shape`.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub nseq: usize,
    pub seq: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub scale: f32,
}

/// As `BM`, `BN` in `kernels/attention.cu`, and its thread grid.
const TILE: u32 = 64;
const GEMM_BLOCK: [u32; 3] = [8, 8, 1];
const ROW_THREADS: u32 = 64;
const FLAT: u32 = 256;

const PTX: &str = include_str!("kernels/attention.ptx");

struct Kernels { gemm: Kernel, softmax: Kernel, probs: Kernel, dscore: Kernel, delta: Kernel, group_sum: Kernel }
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Result<&'static Kernels, &'static str> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels {
        gemm: m.kernel("attn_gemm", 24)?,
        softmax: m.kernel("attn_softmax", 5)?,
        probs: m.kernel("attn_probs", 5)?,
        dscore: m.kernel("attn_dscore", 6)?,
        delta: m.kernel("attn_delta", 5)?,
        group_sum: m.kernel("attn_group_sum", 6)?,
    }))
}

/// Whether the module loads, for `module_status`.
pub(super) fn loads() -> Result<(), &'static str> { kernels().map(|_| ()) }

/// Elements in one score matrix per (sequence, head) over the whole batch.
fn pair_len(sh: &Shape) -> Option<usize> {
    sh.nseq.checked_mul(sh.nh)?.checked_mul(sh.seq)?.checked_mul(sh.seq)
}

/// The wgpu path's validity rule, and sizes that fit the kernels' 32-bit indices.
fn valid(sh: &Shape) -> bool {
    let fits = |x: Option<usize>| x.is_some_and(|n| u32::try_from(n).is_ok());
    let rows = sh.nseq.checked_mul(sh.seq);
    sh.hd % 4 == 0 && sh.hd > 0 && sh.nkv > 0 && sh.nh % sh.nkv == 0 && sh.nh > 0 && sh.seq > 0 && sh.nseq > 0
        && fits(pair_len(sh)) && fits(rows.and_then(|r| r.checked_mul(sh.nh.max(sh.nkv))).and_then(|r| r.checked_mul(sh.hd)))
}

/// How an operand is addressed: `base = sequence * ss + head * hs`, then
/// `first * index1 + second * index2`.
#[derive(Clone, Copy)]
struct Layout { ss: u32, hs: u32, first: u32, second: u32, kv: bool }

impl Layout {
    /// `[nseq*seq][width*hd]`: (row, column) within a head of `width` heads.
    fn rows(sh: &Shape, width: usize, kv: bool) -> Layout {
        let row = (width * sh.hd) as u32;
        Layout { ss: sh.seq as u32 * row, hs: sh.hd as u32, first: row, second: 1, kv }
    }
    /// `[nseq][nh][seq][seq]`: the score matrices.
    fn pairs(sh: &Shape) -> Layout {
        let (seq, nh) = (sh.seq as u32, sh.nh as u32);
        Layout { ss: nh * seq * seq, hs: seq * seq, first: seq, second: 1, kv: false }
    }
    /// The same operand with its two indices exchanged.
    fn t(self) -> Layout { Layout { first: self.second, second: self.first, ..self } }
}

/// `C(i, j) = alpha sum_k A(i, k) B(k, j)` for every (sequence, head): cuBLAS's
/// batched GEMM when it is wanted and there, else the tiled kernel.
#[allow(clippy::too_many_arguments)]
fn product(ks: &Kernels, sh: &Shape, tri: bool, a: (&Tensor, Layout), b: (&Tensor, Layout), c: (&Tensor, Layout),
           m: usize, k: usize, n: usize, alpha: f32) -> Option<()> {
    let path = if wanted() == Path::Cublas && batched(sh, a, b, c, m, k, n, alpha).is_some() {
        Path::Cublas
    } else {
        tiled(ks, sh, tri, a, b, c, m, k, n, alpha)?;
        Path::Tiled
    };
    note_path(path);
    mark(0);
    Some(())
}

/// One operand of head `h` in cuBLAS's column-major terms, whose rows are this
/// layout's second index when that is contiguous, and its first otherwise.
fn strided<'a>((t, l): (&'a Tensor, Layout), h: usize, group: usize) -> Strided<'a> {
    let head = if l.kv { h / group } else { h };
    let packed = l.second == 1;
    Strided { buf: t, offset: head * l.hs as usize, trans: !packed, ld: (if packed { l.first } else { l.second }) as usize, stride: l.ss as usize }
}

/// All outputs here are contiguous along `j`, so cuBLAS holds `C` as its
/// transpose and computes `C^T = B^T A^T`. A batch has one stride, and heads and
/// sequences have different ones, so it is one batch of sequences per head, with
/// no copy. Above the diagonal the scores are computed and then thrown away by
/// the softmax, as `tri` only saves work in the tiled kernel.
#[allow(clippy::too_many_arguments)]
fn batched(sh: &Shape, a: (&Tensor, Layout), b: (&Tensor, Layout), c: (&Tensor, Layout),
           m: usize, k: usize, n: usize, alpha: f32) -> Option<()> {
    let blas = blas().ok()?;
    let group = sh.nh / sh.nkv;
    for h in 0..sh.nh {
        blas.sgemm_strided(n, m, k, alpha, strided(b, h, group), strided(a, h, group), strided(c, h, group), sh.nseq).ok()?;
    }
    Some(())
}

#[allow(clippy::too_many_arguments)]
fn tiled(ks: &Kernels, sh: &Shape, tri: bool, a: (&Tensor, Layout), b: (&Tensor, Layout), c: (&Tensor, Layout),
         m: usize, k: usize, n: usize, alpha: f32) -> Option<()> {
    let int = Arg::Int;
    let (la, lb, lc) = (a.1, b.1, c.1);
    ks.gemm.launch([(n as u32).div_ceil(TILE), (m as u32).div_ceil(TILE), (sh.nseq * sh.nh) as u32], GEMM_BLOCK, &[
        Arg::F(a.0), Arg::F(b.0), Arg::F(c.0),
        int(m as u32), int(k as u32), int(n as u32), int(sh.nh as u32), int((sh.nh / sh.nkv) as u32), int(tri as u32),
        int(la.ss), int(la.hs), int(la.first), int(la.second), int(la.kv as u32),
        int(lb.ss), int(lb.hs), int(lb.first), int(lb.second), int(lb.kv as u32),
        int(lc.ss), int(lc.hs), int(lc.first), int(lc.second), Arg::Real(alpha),
    ]).ok()
}

fn rows_launch(k: &Kernel, threads: usize, block: u32, args: &[Arg]) -> Option<()> {
    k.launch([(threads as u32).div_ceil(block), 1, 1], [block, 1, 1], args).ok()?;
    mark(1);
    Some(())
}

thread_local! { static PROFILE: Cell<Option<(Instant, [f64; 2])>> = const { Cell::new(None) }; }

/// Starts this thread's split of attention into its products and row kernels.
/// It drains the queue after every launch, so it says where the time goes and
/// not how long a call takes.
pub fn profile_start() { PROFILE.with(|p| p.set(Some((Instant::now(), [0.0; 2])))); }

/// Ends the split: milliseconds in the products and in the row kernels.
pub fn profile_take() -> Option<[f64; 2]> { PROFILE.with(|p| p.take()).map(|(_, ms)| ms) }

fn mark(slot: usize) {
    PROFILE.with(|p| if let Some((t, mut ms)) = p.get() {
        sync();
        ms[slot] += t.elapsed().as_secs_f64() * 1e3;
        p.set(Some((Instant::now(), ms)));
    });
}

/// Score matrices and gradient partials, grown on demand. The lock is held
/// across a call's launches, so another thread's cannot land in between.
#[derive(Default)]
struct Scratch { p: Option<Tensor>, ds: Option<Tensor>, part_k: Option<Tensor>, part_v: Option<Tensor> }
static SCRATCH: Mutex<Option<Scratch>> = Mutex::new(None);

fn grown(slot: &mut Option<Tensor>, len: usize) -> Option<()> {
    if slot.as_ref().is_none_or(|t| t.len() < len) { *slot = Some(Tensor::zeros(len)?); }
    Some(())
}

/// Forward: `o` receives the attention output, `lse` (length `nseq*seq*nh`) the
/// per-query log-sum-exp for the backward. `false` and nothing queued when there
/// is no device or the shape or a buffer does not fit.
pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, sh: &Shape) -> bool {
    if !valid(sh) { return false; }
    let (rows, qw, kw) = (sh.nseq * sh.seq, sh.nh * sh.hd, sh.nkv * sh.hd);
    if q.len() < rows * qw || k.len() < rows * kw || v.len() < rows * kw || o.len() < rows * qw || lse.len() < rows * sh.nh { return false; }
    forward_inner(q, k, v, o, lse, sh).is_some()
}

fn forward_inner(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, sh: &Shape) -> Option<()> {
    let ks = kernels().ok()?;
    let (seq, hd) = (sh.seq, sh.hd);
    let (lq, lkv, lp) = (Layout::rows(sh, sh.nh, false), Layout::rows(sh, sh.nkv, true), Layout::pairs(sh));
    let mut guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    let sc = guard.get_or_insert_with(Scratch::default);
    grown(&mut sc.p, pair_len(sh)?)?;
    let p = sc.p.as_ref()?;
    product(ks, sh, true, (q, lq), (k, lkv.t()), (p, lp), seq, hd, seq, sh.scale)?;
    rows_launch(&ks.softmax, sh.nseq * sh.nh * seq, ROW_THREADS,
                &[Arg::F(p), Arg::F(lse), Arg::Int(sh.nseq as u32), Arg::Int(seq as u32), Arg::Int(sh.nh as u32)])?;
    product(ks, sh, false, (p, lp), (v, lkv), (o, lq), seq, seq, hd, 1.0)
}

/// Backward from the upstream gradient `g` (shaped like `o`): writes `dq`, `dk`,
/// `dv`, assigned and not accumulated. `o` and `lse` are the forward's outputs;
/// `delta` is scratch of length `nseq*seq*nh`.
#[allow(clippy::too_many_arguments)]
pub fn backward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, g: &Tensor,
                delta: &Tensor, dq: &Tensor, dk: &Tensor, dv: &Tensor, sh: &Shape) -> bool {
    if !valid(sh) { return false; }
    let (rows, qw, kw) = (sh.nseq * sh.seq, sh.nh * sh.hd, sh.nkv * sh.hd);
    if [q, o, g, dq].iter().any(|t| t.len() < rows * qw) || [k, v, dk, dv].iter().any(|t| t.len() < rows * kw)
        || [lse, delta].iter().any(|t| t.len() < rows * sh.nh) { return false; }
    backward_inner(q, k, v, o, lse, g, delta, dq, dk, dv, sh).is_some()
}

#[allow(clippy::too_many_arguments)]
fn backward_inner(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, g: &Tensor,
                  delta: &Tensor, dq: &Tensor, dk: &Tensor, dv: &Tensor, sh: &Shape) -> Option<()> {
    let ks = kernels().ok()?;
    let (seq, hd, rows) = (sh.seq, sh.hd, sh.nseq * sh.seq);
    let group = sh.nh / sh.nkv;
    let (lq, lkv, lp) = (Layout::rows(sh, sh.nh, false), Layout::rows(sh, sh.nkv, true), Layout::pairs(sh));
    let int = Arg::Int;
    let (nseq_a, seq_a, nh_a) = (int(sh.nseq as u32), int(seq as u32), int(sh.nh as u32));
    let row_threads = sh.nseq * sh.nh * seq;

    let mut guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    let sc = guard.get_or_insert_with(Scratch::default);
    let pairs = pair_len(sh)?;
    grown(&mut sc.p, pairs)?;
    grown(&mut sc.ds, pairs)?;
    if group > 1 {
        grown(&mut sc.part_k, rows * sh.nh * hd)?;
        grown(&mut sc.part_v, rows * sh.nh * hd)?;
    }
    let (p, ds) = (sc.p.as_ref()?, sc.ds.as_ref()?);

    rows_launch(&ks.delta, rows * sh.nh, ROW_THREADS, &[Arg::F(o), Arg::F(g), Arg::F(delta), int((rows * sh.nh) as u32), int(hd as u32)])?;
    // P from the forward's L
    product(ks, sh, true, (q, lq), (k, lkv.t()), (p, lp), seq, hd, seq, sh.scale)?;
    rows_launch(&ks.probs, row_threads, ROW_THREADS, &[Arg::F(p), Arg::F(lse), nseq_a, seq_a, nh_a])?;
    // dS = P (dO.V - D)
    product(ks, sh, true, (g, lq), (v, lkv.t()), (ds, lp), seq, hd, seq, 1.0)?;
    rows_launch(&ks.dscore, row_threads, ROW_THREADS, &[Arg::F(ds), Arg::F(p), Arg::F(delta), nseq_a, seq_a, nh_a])?;
    // dV and dK per head: straight into the output when each kv head has one
    // query head, into a partial when a group has to be summed
    let (tv, tk) = if group > 1 { (sc.part_v.as_ref()?, sc.part_k.as_ref()?) } else { (dv, dk) };
    product(ks, sh, false, (p, lp.t()), (g, lq), (tv, lq), seq, seq, hd, 1.0)?;
    product(ks, sh, false, (ds, lp.t()), (q, lq), (tk, lq), seq, seq, hd, sh.scale)?;
    if group > 1 {
        let n = rows * sh.nkv * hd;
        let args = |part, out| [Arg::F(part), Arg::F(out), int(rows as u32), int(sh.nkv as u32), int(hd as u32), int(group as u32)];
        rows_launch(&ks.group_sum, n, FLAT, &args(tv, dv))?;
        rows_launch(&ks.group_sum, n, FLAT, &args(tk, dk))?;
    }
    product(ks, sh, false, (ds, lp), (k, lkv), (dq, lq), seq, seq, hd, sh.scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::gemm::{force_path, last_path};
    use crate::cuda::have_device;
    use r2_autograd::Tape;

    /// Runs `f` on the tiled kernel and, where cuBLAS loads, on its batched GEMM.
    fn on_both_paths(mut f: impl FnMut(Path)) {
        for p in [Path::Tiled, Path::Cublas] {
            if p == Path::Cublas && blas().is_err() { continue; }
            force_path(Some(p));
            f(p);
            force_path(None);
        }
    }

    fn mk(n: usize, ph: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.8).collect() }

    /// Output and all three gradients against the tape's CPU attention, at the
    /// wgpu test's shapes: across tile boundaries, GQA grouping, batch > 1.
    #[test]
    fn matches_the_tape() {
        if !have_device() { return; }
        on_both_paths(|path| for &(nseq, seq, nh, nkv, hd) in &[(1usize, 8usize, 2usize, 1usize, 8usize), (2, 65, 4, 2, 16),
                                           (3, 130, 3, 3, 32), (2, 70, 4, 2, 64), (1, 40, 2, 1, 256), (1, 1, 2, 2, 4)] {
            let rows = nseq * seq;
            let (qv, kv, vv) = (mk(rows * nh * hd, 0.0), mk(rows * nkv * hd, 1.3), mk(rows * nkv * hd, 2.7));
            let gv = mk(rows * nh * hd, 0.9);
            let scale = 1.0 / (hd as f32).sqrt();
            let sh = Shape { nseq, seq, nh, nkv, hd, scale };

            let mut tape = Tape::new();
            let (q, k, v) = (tape.leaf(qv.clone(), true), tape.leaf(kv.clone(), true), tape.leaf(vv.clone(), true));
            let o = tape.attention(q, k, v, nseq, seq, nh, nkv, hd, scale);
            tape.backward_from(o, &gv);

            let up = |x: &[f32]| Tensor::upload(x).unwrap();
            let z = |n: usize| Tensor::zeros(n).unwrap();
            let (tq, tk, tv, tg) = (up(&qv), up(&kv), up(&vv), up(&gv));
            let (to, tl, td) = (z(rows * nh * hd), z(rows * nh), z(rows * nh));
            let (dq, dk, dv) = (z(rows * nh * hd), z(rows * nkv * hd), z(rows * nkv * hd));
            assert!(forward(&tq, &tk, &tv, &to, &tl, &sh));
            assert!(backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh));
            assert_eq!(last_path(), Some(path), "the path this test claims to cover");

            let close = |got: &[f32], want: &[f32], what: &str| {
                assert_eq!(got.len(), want.len(), "{what}: length (a fault downloads nothing)");
                let scale = want.iter().fold(0.0f32, |s, x| s.max(x.abs())).max(1.0);
                for (i, (a, b)) in got.iter().zip(want).enumerate() {
                    assert!((a - b).abs() <= 3e-5 * scale, "{what}[{i}] device {a} vs cpu {b} ({sh:?})");
                }
            };
            close(&to.download(), tape.value(o), "output");
            close(&dq.download(), tape.grad(q), "grad_q");
            close(&dk.download(), tape.grad(k), "grad_k");
            close(&dv.download(), tape.grad(v), "grad_v");
        });
    }

    /// The tiled kernel promises this: each element is one thread's fixed-order
    /// sum. cuBLAS does not promise it, so this reports what it does.
    #[test]
    fn backward_is_bit_reproducible() {
        if !have_device() { return; }
        on_both_paths(|_| one_reproducibility_run());
    }

    fn one_reproducibility_run() {
        let sh = Shape { nseq: 2, seq: 70, nh: 4, nkv: 2, hd: 64, scale: 0.125 };
        let rows = 140usize;
        let (tq, tk, tv, tg) = (Tensor::upload(&mk(rows * 256, 0.0)).unwrap(), Tensor::upload(&mk(rows * 128, 1.3)).unwrap(),
                                Tensor::upload(&mk(rows * 128, 2.7)).unwrap(), Tensor::upload(&mk(rows * 256, 0.9)).unwrap());
        let run = || {
            let z = |n: usize| Tensor::zeros(n).unwrap();
            let (to, tl, td, dq, dk, dv) = (z(rows * 256), z(rows * 4), z(rows * 4), z(rows * 256), z(rows * 128), z(rows * 128));
            assert!(forward(&tq, &tk, &tv, &to, &tl, &sh));
            assert!(backward(&tq, &tk, &tv, &to, &tl, &tg, &td, &dq, &dk, &dv, &sh));
            (dq.download(), dk.download(), dv.download())
        };
        let first = run();
        assert!(!first.0.is_empty());
        for _ in 0..3 { assert_eq!(run(), first); }
    }

    /// Shapes the kernels do not serve, and undersized buffers, are refused
    /// without touching the device.
    #[test]
    fn unsupported_shapes_are_refused() {
        if !have_device() { return; }
        let t = Tensor::zeros(16).unwrap();
        let bad = |nseq, seq, nh, nkv, hd| Shape { nseq, seq, nh, nkv, hd, scale: 1.0 };
        for sh in [bad(1, 1, 1, 1, 3), bad(1, 1, 1, 1, 0), bad(1, 1, 3, 2, 4), bad(1, 1, 1, 0, 4), bad(0, 1, 1, 1, 4),
                   bad(1, 1 << 17, 1, 1, 4)] {
            assert!(!forward(&t, &t, &t, &t, &t, &sh), "{sh:?}");
        }
        assert!(!forward(&t, &t, &t, &t, &t, &bad(4, 4, 1, 1, 4)), "buffers too small");
    }
}
