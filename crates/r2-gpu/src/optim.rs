//! Adam on the device: one launch over every parameter, the moments
//! resident, the update in place.
//!
//! The arithmetic is the CPU kernel's (`r2_train::optim::adam_scalar`)
//! operation for operation, in the same association, and **bit-identical
//! to it** — which took one correction.
//!
//! This doc used to say the gap was the driver contracting `a * b + c`
//! into an FMA. It was not: `numerics::probe` shows this adapter never
//! contracts (0 of 1,048,576), but its builtin `/` is off in 29% of cases
//! (up to 2 ULP — WGSL allows 2.5) and `sqrt` in 15%. Adam has four
//! divisions and a square root per element. They now go through
//! `numerics::CR_WGSL`'s `div_cr` / `sqrt_cr` — the builtin's answer plus
//! one exact-`fma` correction step, 100% correctly rounded on the probe —
//! and the update matches the CPU to the bit.

use crate::device::{gpu, Tensor};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

const THREADS: u32 = 256;

fn src() -> String {
    format!(r#"{CR}
struct P {{ n: u32, pad0: u32, pad1: u32, pad2: u32, b1: f32, b2: f32, bc1: f32, bc2: f32, lr: f32, eps: f32, scale: f32, c: f32 }};
@group(0) @binding(0) var<storage, read_write> W: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let gi = div_cr(G[i], p.scale);
    let m = p.b1 * M[i] + (1.0 - p.b1) * gi;
    let v = p.b2 * V[i] + (1.0 - p.b2) * gi * gi;
    M[i] = m;
    V[i] = v;
    let mh = div_cr(m, p.bc1);
    let vh = div_cr(v, p.bc2);
    var w = W[i];
    if (p.c != 0.0) {{ w = fma(-p.c, w, w); }}
    W[i] = w - div_cr(p.lr * mh, sqrt_cr(vh) + p.eps);
}}
"#, T = THREADS, CR = crate::numerics::CR_WGSL)
}

static PIPE: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

/// One Adam update over `n` parameters: `w -= lr * m̂ / (sqrt(v̂) + eps)`
/// with the moments updated in place. `t` is the step number (1-based)
/// for the bias corrections; `scale` divides the gradient first (loss
/// scaling / accumulation). `c` is AdamW's `lr * weight_decay` for this block,
/// applied as the CPU kernel applies it (one fused multiply-add); 0 skips it.
#[allow(clippy::too_many_arguments)]
pub fn adam_step(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, n: usize,
                 lr: f32, beta1: f32, beta2: f32, eps: f32, t: u32, scale: f32, c: f32) -> bool {
    let Some(gpu) = gpu() else { return false };
    let Some(p) = PIPE.get_or_init(|| {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2gpu-adam"), source: wgpu::ShaderSource::Wgsl(src().into()),
        });
        Some(gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("r2gpu-adam"), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }))
    }).as_ref() else { return false };
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    let words: [u32; 12] = [n as u32, 0, 0, 0, beta1.to_bits(), beta2.to_bits(), bc1.to_bits(), bc2.to_bits(),
                            lr.to_bits(), eps.to_bits(), scale.to_bits(), c.to_bits()];
    let ubuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("r2gpu-adam-p"), contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: w.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: g.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: m.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v.buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ubuf.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(THREADS), 1, 1);
    }
    gpu.queue.submit(Some(enc.finish()));
    true
}

/// Gradient values summed per chunk. The host adds the chunks in f64. Short
/// chunks are the accuracy that holds on every driver, because a sum over 256
/// values cannot drift far, whatever happens to the compensation below.
const NORM_CHUNK: usize = 256;

/// `Σ g²` per chunk, in f32 with its errors kept: `fma` recovers what each
/// square rounded away and TwoSum what each addition did. Plain f32 over a
/// whole block would lose the small squares among 7M values, and the CPU's
/// norm is f64, so clipping would not scale by the same factor.
///
/// The compensation is a bonus, not the guarantee. WGSL lets a driver simplify
/// float expressions, and llvmpipe's `(hi - (s - b)) + (a - b)` comes out as 0:
/// at chunks of 4096 that cost it 4.9e-8 where the Vega has 5e-13.
fn norm_src() -> String {
    format!(r#"
struct P {{ n: u32, base: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let start = gid.x * {C}u;
    if (start >= p.n) {{ return; }}
    let end = min(start + {C}u, p.n);
    var hi = 0.0;
    var lo = 0.0;
    for (var i = start; i < end; i = i + 1u) {{
        let x = G[i];
        let a = x * x;
        let s = hi + a;
        let b = s - hi;
        lo = lo + ((hi - (s - b)) + (a - b)) + fma(x, x, -a);
        hi = s;
    }}
    OUT[2u * (p.base + gid.x)] = hi;
    OUT[2u * (p.base + gid.x) + 1u] = lo;
}}
"#, C = NORM_CHUNK)
}

static NORM_PIPE: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

/// Values [`grad_sq_norm`]'s scratch tensor must hold for these block sizes.
pub fn norm_scratch_len(sizes: &[usize]) -> usize {
    2 * sizes.iter().map(|n| n.div_ceil(NORM_CHUNK).max(1)).sum::<usize>()
}

/// `Σ g²` over every block, in f64: what `r2_train`'s CPU clipping computes,
/// without bringing 29 MB of gradients back to read 226 KB of sums.
pub fn grad_sq_norm(grads: &[Tensor], sizes: &[usize], scratch: &Tensor) -> Option<f64> {
    let gpu = gpu()?;
    let p = NORM_PIPE.get_or_init(|| {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2gpu-norm"), source: wgpu::ShaderSource::Wgsl(norm_src().into()),
        });
        Some(gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("r2gpu-norm"), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }))
    }).as_ref()?;
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let mut base = 0u32;
    for (g, &n) in grads.iter().zip(sizes) {
        let words: [u32; 4] = [n as u32, base, 0, 0];
        let ubuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-norm-p"), contents: bytemuck::cast_slice(&words), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &p.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: g.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        });
        let chunks = n.div_ceil(NORM_CHUNK).max(1) as u32;
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(chunks.div_ceil(64), 1, 1);
        drop(pass);
        base += chunks;
    }
    gpu.queue.submit(Some(enc.finish()));
    Some(scratch.download().iter().map(|&x| x as f64).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_train::optim::Adam;

    /// AdamW on the device: the decay is the CPU's one fused multiply-add,
    /// per block, and so is every bit of the result. A decay of 0 in one
    /// block leaves that block plain Adam.
    #[test]
    fn gpu_adamw_is_bit_identical_to_the_cpu_adamw() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        if crate::numerics::fma_exact() == Some(false) {
            eprintln!("adapter's fma is not fused: bit-identity is not promised here; skipped"); return;
        }
        let n = 4099usize;
        let init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos() * 0.5).collect();
        let (lr, wd) = (3e-4f32, 0.1f32);
        let (tw, tm, tv) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
        let (uw, um, uv) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
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
        assert!(tw.download().iter().zip(&blocks[0]).all(|(a, b)| a.to_bits() == b.to_bits()), "decayed block");
        assert!(uw.download().iter().zip(&blocks[1]).all(|(a, b)| a.to_bits() == b.to_bits()), "undecayed block");
    }

    /// Blocks of very different magnitudes, where one f32 sum over all of
    /// them would lose the small squares.
    fn norm_blocks() -> Vec<Vec<f32>> {
        vec![(0..100_003).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect(),
             (0..17).map(|i| 1e-4 * (i as f32 - 8.0)).collect(),
             (0..4096).map(|i| 1e-3 * ((i as f32) * 0.11).cos()).collect()]
    }

    /// What every driver must reach, compensation or not. In f32, on the
    /// worst case below, this is about 1e-8; the norm's error is half that.
    const NORM_TOLERANCE: f64 = 2e-8;

    fn norm_host(blocks: &[Vec<f32>]) -> f64 {
        blocks.iter().flatten().map(|&x| x as f64 * x as f64).sum()
    }

    /// The device's Σg² against f64 on the host.
    #[test]
    fn gpu_grad_norm_matches_f64_on_the_host() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let blocks = norm_blocks();
        let sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let tensors: Vec<Tensor> = blocks.iter().map(|b| Tensor::upload(b).unwrap()).collect();
        let scratch = Tensor::zeros(norm_scratch_len(&sizes)).unwrap();
        let got = grad_sq_norm(&tensors, &sizes, &scratch).unwrap();
        let want = norm_host(&blocks);
        assert!((got - want).abs() < NORM_TOLERANCE * want, "device {got}, host {want}");
    }

    /// The driver that cancels the shader's compensation: plain f32 sums over
    /// the same chunks, on the host, needing no device. llvmpipe reached
    /// 4.9e-8 at chunks of 4096, which is what this gives there.
    #[test]
    fn the_chunks_alone_hold_the_norm_without_any_compensation() {
        let blocks = norm_blocks();
        let plain: f64 = blocks.iter()
            .flat_map(|b| b.chunks(NORM_CHUNK))
            .map(|c| c.iter().fold(0.0f32, |s, &x| s + x * x) as f64).sum();
        let want = norm_host(&blocks);
        assert!((plain - want).abs() < NORM_TOLERANCE * want, "chunked f32 {plain}, f64 {want}");
    }

    /// Three steps on the device against three on the CPU, from the same
    /// state: bit-identical, weights and all — the correctly rounded
    /// division and square root close the gap the builtins left.
    #[test]
    fn gpu_adam_is_bit_identical_to_the_cpu_adam() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        if crate::numerics::fma_exact() == Some(false) {
            eprintln!("adapter's fma is not fused: bit-identity is not promised here; skipped"); return;
        }
        let n = 10_007usize;
        let init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
        // every seventh parameter never gets a gradient — an embedding row
        // no batch contained: v stays 0, and sqrt(v) = 0 is exactly the case
        // that once turned such weights into NaN on the device
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
        let off: Vec<usize> = got.iter().zip(&w).enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits()).map(|(i, _)| i).collect();
        assert!(off.is_empty(), "{} of {n} parameters differ from the CPU; first: {:?}", off.len(),
                off.first().map(|&i| (i, got[i], w[i])));
        // reproducibility: the same three steps again give the same bits
        let (tw2, tm2, tv2) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
        for t in 1..=3u32 {
            let tg = Tensor::upload(&grad(t)).unwrap();
            assert!(adam_step(&tw2, &tg, &tm2, &tv2, n, 3e-4, 0.9, 0.999, 1e-8, t, 2.0, 0.0));
        }
        assert_eq!(tw2.download(), got, "the device did not reproduce its own result");
    }
}
