//! Adam and the gradient norm on CUDA: the `optim` module's public API over
//! `kernels/optim.cu`. The update is bit-identical to `r2_train`'s CPU Adam.

use super::ffi::{Arg, Kernel};
use super::{loaded, Loaded, Tensor};
use std::sync::OnceLock;

const THREADS: u32 = 256;
/// Gradient values per chunk, as `NORM_CHUNK` in `kernels/optim.cu`.
const NORM_CHUNK: usize = 256;
const NORM_THREADS: u32 = 64;

const PTX: &str = include_str!("kernels/optim.ptx");

struct Kernels { adam: Kernel, norm: Kernel, mean: Kernel }
static KERNELS: Loaded<Kernels> = OnceLock::new();

fn kernels() -> Result<&'static Kernels, &'static str> {
    loaded(&KERNELS, PTX, |m| Ok(Kernels { adam: m.kernel("adam_step", 13)?, norm: m.kernel("grad_sq_chunk", 4)?, mean: m.kernel("grad_mean", 4)? }))
}

/// Whether the module loads, for `module_status`.
pub(super) fn loads() -> Result<(), &'static str> { kernels().map(|_| ()) }

/// One Adam update over `n` parameters, the moments updated in place. `t` is
/// the 1-based step number for the bias corrections; `scale` divides the
/// gradient first; `c` is AdamW's `lr * weight_decay` for this block, 0 to skip
/// it. `false` and nothing queued when there is no device or a buffer is short.
#[allow(clippy::too_many_arguments)]
pub fn adam_step(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, n: usize,
                 lr: f32, beta1: f32, beta2: f32, eps: f32, t: u32, scale: f32, c: f32) -> bool {
    if [w, g, m, v].iter().any(|b| b.len() < n) { return false; }
    let Ok(nu) = u32::try_from(n) else { return false };
    if n == 0 { return true; }
    let Ok(ks) = kernels() else { return false };
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    let r = Arg::Real;
    ks.adam.launch([nu.div_ceil(THREADS), 1, 1], [THREADS, 1, 1], &[
        Arg::F(w), Arg::F(g), Arg::F(m), Arg::F(v), Arg::Int(nu),
        r(beta1), r(beta2), r(bc1), r(bc2), r(lr), r(eps), r(scale), r(c),
    ]).is_ok()
}

/// `g[..n] = (g[..n] + peer[at..at + n]) / 2`: one rank's half of a two-rank mean.
/// `false` and nothing queued when there is no device or a buffer is short.
pub fn grad_mean(g: &Tensor, peer: &Tensor, at: usize, n: usize) -> bool {
    if g.len() < n || at.checked_add(n).is_none_or(|e| e > peer.len()) { return false; }
    let (Ok(nu), Ok(base)) = (u32::try_from(n), u32::try_from(at)) else { return false };
    if n == 0 { return true; }
    let Ok(ks) = kernels() else { return false };
    ks.mean.launch([nu.div_ceil(THREADS), 1, 1], [THREADS, 1, 1], &[Arg::F(g), Arg::F(peer), Arg::Int(base), Arg::Int(nu)]).is_ok()
}

/// Values [`grad_sq_norm`]'s scratch tensor must hold for these block sizes.
pub fn norm_scratch_len(sizes: &[usize]) -> usize {
    2 * sizes.iter().map(|n| n.div_ceil(NORM_CHUNK).max(1)).sum::<usize>()
}

/// `Σ g²` over every block, in f64: what the CPU's clipping computes, without
/// bringing the gradients back to the host.
pub fn grad_sq_norm(grads: &[Tensor], sizes: &[usize], scratch: &Tensor) -> Option<f64> {
    if grads.len() != sizes.len() || scratch.len() < norm_scratch_len(sizes) { return None; }
    let ks = kernels().ok()?;
    scratch.copy_from(0, &vec![0.0f32; scratch.len()]).ok()?;
    let mut base = 0u32;
    for (g, &n) in grads.iter().zip(sizes) {
        if g.len() < n { return None; }
        let chunks = n.div_ceil(NORM_CHUNK).max(1) as u32;
        ks.norm.launch([chunks.div_ceil(NORM_THREADS), 1, 1], [NORM_THREADS, 1, 1],
                       &[Arg::F(g), Arg::F(scratch), Arg::Int(u32::try_from(n).ok()?), Arg::Int(base)]).ok()?;
        base += chunks;
    }
    Some(scratch.to_vec().ok()?.iter().map(|&x| x as f64).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::have_device;
    use r2_train::optim::Adam;

    /// Three steps against the CPU's, from the same state: every bit of every
    /// parameter. Every seventh parameter never gets a gradient, so `v` stays 0
    /// and `sqrt(v) = 0`, which once turned such weights into NaN on wgpu.
    #[test]
    fn adam_is_bit_identical_to_the_cpu() {
        if !have_device() { return; }
        let n = 10_007usize;
        let init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
        let grad = |t: u32| -> Vec<f32> {
            (0..n).map(|i| if i % 7 == 0 { 0.0 } else { ((i as f32) * 0.07 + t as f32).cos() * 0.5 }).collect()
        };
        let (tw, tm, tv) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
        for t in 1..=3u32 {
            let tg = Tensor::upload(&grad(t)).unwrap();
            assert!(adam_step(&tw, &tg, &tm, &tv, n, 3e-4, 0.9, 0.999, 1e-8, t, 2.0, 0.0));
        }
        let mut w = init.clone();
        let mut cpu = Adam::new(n, 3e-4);
        for t in 1..=3u32 {
            let g = grad(t);
            let mut blocks = vec![std::mem::take(&mut w)];
            cpu.step_blocks(&mut blocks, &[&g], 2.0).unwrap();
            w = blocks.remove(0);
        }
        let got = tw.download();
        assert_eq!(got.len(), n, "a fault downloads nothing");
        let off: Vec<usize> = (0..n).filter(|&i| got[i].to_bits() != w[i].to_bits()).collect();
        assert!(off.is_empty(), "{} of {n} differ from the CPU; first: {:?}", off.len(), off.first().map(|&i| (i, got[i], w[i])));
    }

    /// AdamW: the decay is the CPU's one fused multiply-add, per block; a decay
    /// of 0 leaves that block plain Adam.
    #[test]
    fn adamw_is_bit_identical_to_the_cpu() {
        if !have_device() { return; }
        let n = 4099usize;
        let init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos() * 0.5).collect();
        let (lr, wd) = (3e-4f32, 0.1f32);
        let z = || Tensor::zeros(n).unwrap();
        let (tw, tm, tv) = (Tensor::upload(&init).unwrap(), z(), z());
        let (uw, um, uv) = (Tensor::upload(&init).unwrap(), z(), z());
        let tg = Tensor::upload(&grad).unwrap();
        for t in 1..=3u32 {
            assert!(adam_step(&tw, &tg, &tm, &tv, n, lr, 0.9, 0.999, 1e-8, t, 2.0, lr * wd));
            assert!(adam_step(&uw, &tg, &um, &uv, n, lr, 0.9, 0.999, 1e-8, t, 2.0, 0.0));
        }
        let mut blocks = vec![init.clone(), init.clone()];
        let mut cpu = Adam::new(2 * n, lr);
        cpu.weight_decay = wd;
        for _ in 0..3 {
            cpu.step_blocks_masked(&mut blocks, &[&grad, &grad], 2.0, Some(&[true, false])).unwrap();
        }
        let same = |t: &Tensor, b: &[f32]| { let d = t.download(); d.len() == b.len() && d.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) };
        assert!(same(&tw, &blocks[0]), "decayed block");
        assert!(same(&uw, &blocks[1]), "undecayed block");
    }

    /// Blocks of very different magnitudes, where one f32 sum would lose the
    /// small squares, against f64 on the host at the wgpu test's tolerance.
    #[test]
    fn grad_norm_matches_f64_on_the_host() {
        if !have_device() { return; }
        let blocks: Vec<Vec<f32>> = vec![(0..100_003).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect(),
                                         (0..17).map(|i| 1e-4 * (i as f32 - 8.0)).collect(),
                                         (0..4096).map(|i| 1e-3 * ((i as f32) * 0.11).cos()).collect()];
        let sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let tensors: Vec<Tensor> = blocks.iter().map(|b| Tensor::upload(b).unwrap()).collect();
        let scratch = Tensor::zeros(norm_scratch_len(&sizes)).unwrap();
        let got = grad_sq_norm(&tensors, &sizes, &scratch).unwrap();
        let want: f64 = blocks.iter().flatten().map(|&x| x as f64 * x as f64).sum();
        assert!((got - want).abs() < 2e-8 * want, "device {got}, host {want}");
    }

    #[test]
    fn short_buffers_are_refused() {
        if !have_device() { return; }
        let (a, b) = (Tensor::zeros(4).unwrap(), Tensor::zeros(8).unwrap());
        assert!(!adam_step(&a, &b, &b, &b, 8, 1e-3, 0.9, 0.999, 1e-8, 1, 1.0, 0.0));
        assert!(grad_sq_norm(std::slice::from_ref(&b), &[8], &a).is_some());
        assert!(grad_sq_norm(std::slice::from_ref(&a), &[8], &b).is_none());
    }
}
