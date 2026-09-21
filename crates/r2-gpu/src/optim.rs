//! Adam on the device: one launch over every parameter, the moments
//! resident, the update in place.
//!
//! The arithmetic is the CPU kernel's (`r2_train::optim::adam_scalar`)
//! operation for operation — `div` and `sqrt`, the same association — and
//! `/` and `sqrt` are IEEE-correct on every backend wgpu ships. It is
//! still not bit-identical to the CPU: WGSL lets the driver's compiler
//! contract `a * b + c` into an FMA (the language has no `precise`, and
//! naga cannot emit SPIR-V's `NoContraction`), and on this adapter it
//! does so for ~0.2% of elements, a few ULP each. So the contract is:
//! the device agrees with the CPU to a few ULP, and with itself to the bit —
//! the same run gives the same weights every time.

use crate::device::{gpu, Tensor};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

const THREADS: u32 = 256;

fn src() -> String {
    format!(r#"
struct P {{ n: u32, pad0: u32, pad1: u32, pad2: u32, b1: f32, b2: f32, bc1: f32, bc2: f32, lr: f32, eps: f32, scale: f32, pad3: f32 }};
@group(0) @binding(0) var<storage, read_write> W: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size({T}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= p.n) {{ return; }}
    let gi = G[i] / p.scale;
    let m = p.b1 * M[i] + (1.0 - p.b1) * gi;
    let v = p.b2 * V[i] + (1.0 - p.b2) * gi * gi;
    M[i] = m;
    V[i] = v;
    let mh = m / p.bc1;
    let vh = v / p.bc2;
    W[i] = W[i] - p.lr * mh / (sqrt(vh) + p.eps);
}}
"#, T = THREADS)
}

static PIPE: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

/// One Adam update over `n` parameters: `w -= lr * m̂ / (sqrt(v̂) + eps)`
/// with the moments updated in place. `t` is the step number (1-based)
/// for the bias corrections; `scale` divides the gradient first (loss
/// scaling / accumulation).
#[allow(clippy::too_many_arguments)]
pub fn adam_step(w: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, n: usize,
                 lr: f32, beta1: f32, beta2: f32, eps: f32, t: u32, scale: f32) -> bool {
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
                            lr.to_bits(), eps.to_bits(), scale.to_bits(), 0];
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

#[cfg(test)]
mod tests {
    use super::*;
    use r2_train::optim::Adam;

    /// Three steps on the device against three on the CPU, from the same
    /// state: within a few ULP everywhere (the driver may fuse a multiply-
    /// add the CPU keeps separate), and the device agrees with itself to
    /// the bit.
    #[test]
    fn gpu_adam_matches_the_cpu_adam_to_a_few_ulp_and_itself_to_the_bit() {
        if gpu().is_none() { eprintln!("no GPU adapter; skipped"); return; }
        let n = 10_007usize;
        let init: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
        let grad = |t: u32| -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.07 + t as f32).cos() * 0.5).collect() };

        let (tw, tm, tv) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
        for t in 1..=3u32 {
            let tg = Tensor::upload(&grad(t)).unwrap();
            assert!(adam_step(&tw, &tg, &tm, &tv, n, 3e-4, 0.9, 0.999, 1e-8, t, 2.0));
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
        for (i, (a, b)) in got.iter().zip(&w).enumerate() {
            let ulps = (a.to_bits() as i64 - b.to_bits() as i64).abs();
            assert!(ulps <= 4, "parameter {i}: gpu {a} vs cpu {b}, {ulps} ULP apart");
        }
        // reproducibility: the same three steps again give the same bits
        let (tw2, tm2, tv2) = (Tensor::upload(&init).unwrap(), Tensor::zeros(n).unwrap(), Tensor::zeros(n).unwrap());
        for t in 1..=3u32 {
            let tg = Tensor::upload(&grad(t)).unwrap();
            assert!(adam_step(&tw2, &tg, &tm2, &tv2, n, 3e-4, 0.9, 0.999, 1e-8, t, 2.0));
        }
        assert_eq!(tw2.download(), got, "the device did not reproduce its own result");
    }
}
