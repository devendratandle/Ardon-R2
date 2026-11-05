//! Fused causal grouped-query attention on CUDA, forward and backward: the
//! `attention` module's public API over `kernels/attention.cu`.
//!
//! Layouts and the FlashAttention-2 decomposition are as documented there. A
//! `Shape` with `hd` over [`MAX_HD`] is refused and the caller keeps its CPU
//! path, because that bound sizes the per-thread rows in the kernels.

use super::ffi::{Arg, Kernel};
use super::{loaded, Loaded, Tensor};
use std::sync::OnceLock;

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

/// As `MAX_HD` in `kernels/attention.cu`.
pub const MAX_HD: usize = 256;
/// As `BLOCK` in `kernels/attention.cu`.
const BLOCK: u32 = 64;

const PTX: &str = include_str!("kernels/attention.ptx");

struct Kernels { forward: Kernel, delta: Kernel, dkv: Kernel, dq: Kernel }
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Result<&'static Kernels, &'static str> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels {
        forward: m.kernel("attn_forward", 12)?,
        delta: m.kernel("attn_delta", 5)?,
        dkv: m.kernel("attn_dkv", 15)?,
        dq: m.kernel("attn_dq", 14)?,
    }))
}

/// Whether the module loads, for `module_status`.
pub(super) fn loads() -> Result<(), &'static str> { kernels().map(|_| ()) }

/// The wgpu path's validity rule, the row-array bound, and sizes that fit the
/// kernels' 32-bit indices.
fn valid(sh: &Shape) -> bool {
    let fits = || {
        let rows = sh.nseq.checked_mul(sh.seq)?;
        u32::try_from(rows.checked_mul(sh.nh.max(sh.nkv))?.checked_mul(sh.hd)?).ok()
    };
    sh.hd % 4 == 0 && sh.hd > 0 && sh.hd <= MAX_HD && sh.nkv > 0 && sh.nh % sh.nkv == 0 && sh.nh > 0
        && sh.seq > 0 && sh.nseq > 0 && fits().is_some()
}

fn launch(k: &Kernel, threads: usize, args: &[Arg]) -> Option<()> {
    k.launch([(threads as u32).div_ceil(BLOCK), 1, 1], [BLOCK, 1, 1], args).ok()
}

/// The shape's trailing arguments, shared by every kernel but `attn_delta`.
fn dims(sh: &Shape) -> [Arg<'static>; 7] {
    [Arg::Int(sh.nseq as u32), Arg::Int(sh.seq as u32), Arg::Int(sh.nh as u32), Arg::Int(sh.nkv as u32),
     Arg::Int(sh.hd as u32), Arg::Int((sh.nh / sh.nkv) as u32), Arg::Real(sh.scale)]
}

/// Forward: `o` receives the attention output, `lse` (length `nseq*seq*nh`)
/// the per-query log-sum-exp for the backward. `false` and nothing queued
/// when there is no device or the shape or a buffer does not fit.
pub fn forward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, sh: &Shape) -> bool {
    if !valid(sh) { return false; }
    let (rows, qw, kw) = (sh.nseq * sh.seq, sh.nh * sh.hd, sh.nkv * sh.hd);
    if q.len() < rows * qw || k.len() < rows * kw || v.len() < rows * kw || o.len() < rows * qw || lse.len() < rows * sh.nh { return false; }
    let Ok(ks) = kernels() else { return false };
    let mut args = vec![Arg::F(q), Arg::F(k), Arg::F(v), Arg::F(o), Arg::F(lse)];
    args.extend(dims(sh));
    launch(&ks.forward, rows * sh.nh, &args).is_some()
}

/// Backward from the upstream gradient `g` (shaped like `o`): writes `dq`,
/// `dk`, `dv`, assigned and not accumulated. `o` and `lse` are the forward's
/// outputs; `delta` is scratch of length `nseq*seq*nh`.
#[allow(clippy::too_many_arguments)]
pub fn backward(q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, lse: &Tensor, g: &Tensor,
                delta: &Tensor, dq: &Tensor, dk: &Tensor, dv: &Tensor, sh: &Shape) -> bool {
    if !valid(sh) { return false; }
    let (rows, qw, kw) = (sh.nseq * sh.seq, sh.nh * sh.hd, sh.nkv * sh.hd);
    if [q, o, g, dq].iter().any(|t| t.len() < rows * qw) || [k, v, dk, dv].iter().any(|t| t.len() < rows * kw)
        || [lse, delta].iter().any(|t| t.len() < rows * sh.nh) { return false; }
    let Ok(ks) = kernels() else { return false };
    let head = rows * sh.nh;
    let queued = (|| {
        launch(&ks.delta, head, &[Arg::F(o), Arg::F(g), Arg::F(delta), Arg::Int(head as u32), Arg::Int(sh.hd as u32)])?;
        let mut a = vec![Arg::F(q), Arg::F(k), Arg::F(v), Arg::F(g), Arg::F(lse), Arg::F(delta), Arg::F(dk), Arg::F(dv)];
        a.extend(dims(sh));
        launch(&ks.dkv, rows * sh.nkv, &a)?;
        let mut a = vec![Arg::F(q), Arg::F(k), Arg::F(v), Arg::F(g), Arg::F(lse), Arg::F(delta), Arg::F(dq)];
        a.extend(dims(sh));
        launch(&ks.dq, head, &a)
    })();
    queued.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::have_device;
    use r2_autograd::Tape;

    fn mk(n: usize, ph: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37 + ph).sin() * 0.8).collect() }

    /// Output and all three gradients against the tape's CPU attention, at the
    /// wgpu test's shapes: across block boundaries, GQA grouping, batch > 1.
    #[test]
    fn matches_the_tape() {
        if !have_device() { return; }
        for &(nseq, seq, nh, nkv, hd) in &[(1usize, 8usize, 2usize, 1usize, 8usize), (2, 65, 4, 2, 16),
                                           (3, 130, 3, 3, 32), (2, 70, 4, 2, 64), (1, 40, 2, 1, 256)] {
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
        }
    }

    /// Each output row is one thread's fixed-order sum: same bits, every run.
    #[test]
    fn backward_is_bit_reproducible() {
        if !have_device() { return; }
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
                   bad(1, 1, 1, 1, MAX_HD + 4)] {
            assert!(!forward(&t, &t, &t, &t, &t, &sh), "{sh:?}");
        }
        assert!(!forward(&t, &t, &t, &t, &t, &bad(4, 4, 1, 1, 4)), "buffers too small");
    }
}
