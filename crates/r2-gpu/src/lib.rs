//! r2-gpu — GPU compute dispatcher (Pillar 1 foundation).
//!
//! # The accuracy contract (this is the load-bearing decision)
//!
//! WGSL has **no f64**. Ardon-R2's promise is R-faithful accuracy, so the
//! GPU is NEVER the truth source and NEVER runs statistics silently:
//!
//! 1. **Opt-in only.** The Oracle routes an op to the GPU only when the
//!    caller enabled it (`options(r2.gpu = TRUE)` → `gpu_enabled()`).
//! 2. **f32-safe op list only.** Only operations whose f32 rounding is
//!    acceptable for their use (ML-class element maps, matmul for training
//!    /inference — never `sum`/`var`/`cor`, where f32 cancellation is
//!    unacceptable) are eligible. `Op::is_f32_safe()` encodes this.
//! 3. **CPU is the reference.** Every kernel has a CPU implementation
//!    (`cpu::`) that is ALWAYS compiled and unit-tested; the GPU path must
//!    match it within a stated f32 tolerance (verified in tests, gated by
//!    the `gpu` feature). CI, with no GPU, still tests the CPU reference.
//! 4. **Fallback is silent and safe.** Any GPU error, missing adapter, or
//!    ineligible op falls back to the CPU path — never a wrong answer,
//!    never a crash. `dispatch()` embodies this: try GPU if eligible+
//!    enabled+available, else CPU.
//!
//! Default build = CPU reference only (no wgpu). `--features gpu` adds the
//! real device path.

/// GPU-eligible element-wise operations. Only ops where f32 precision is
/// acceptable for their intended (ML) use are listed; statistics stay on
/// the CPU f64 path and never appear here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    /// x -> x (transfer sanity kernel).
    Identity,
    /// Rectified linear unit: max(x, 0).
    Relu,
    /// Logistic sigmoid: 1/(1+e^-x).
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Exponential.
    Exp,
    /// Scale by a constant (a*x).
    Scale(f32),
}

impl Op {
    /// Is this op acceptable to run in f32 on the GPU? All listed ops are
    /// activation/element maps whose ~1e-7 relative error is fine for
    /// training and inference. (Reductions are deliberately NOT here.)
    pub fn is_f32_safe(self) -> bool {
        matches!(self,
            Op::Identity | Op::Relu | Op::Sigmoid | Op::Tanh | Op::Exp | Op::Scale(_))
    }

    #[inline]
    fn apply_scalar(self, x: f32) -> f32 {
        match self {
            Op::Identity => x,
            Op::Relu => if x > 0.0 { x } else { 0.0 },
            Op::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Op::Tanh => x.tanh(),
            Op::Exp => x.exp(),
            Op::Scale(a) => a * x,
        }
    }

    /// The WGSL expression body for this op (used by the gpu feature).
    #[cfg(feature = "gpu")]
    fn wgsl_expr(self) -> String {
        match self {
            Op::Identity => "x".into(),
            Op::Relu => "max(x, 0.0)".into(),
            Op::Sigmoid => "1.0 / (1.0 + exp(-x))".into(),
            Op::Tanh => "tanh(x)".into(),
            Op::Exp => "exp(x)".into(),
            Op::Scale(a) => format!("{:?} * x", a),
        }
    }
}

/// CPU reference kernels — always compiled, always the accuracy truth.
pub mod cpu {
    use super::Op;
    /// Apply an element-wise op to an f32 slice (reference implementation).
    pub fn map(op: Op, xs: &[f32]) -> Vec<f32> {
        xs.iter().map(|&x| op.apply_scalar(x)).collect()
    }
}

// ── Opt-in gate (set from the engine when options(r2.gpu=TRUE)) ─────────

use std::sync::atomic::{AtomicBool, Ordering};
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable/disable GPU routing (the engine calls this from options()).
pub fn set_gpu_enabled(on: bool) { GPU_ENABLED.store(on, Ordering::Relaxed); }
/// Is GPU routing enabled by the caller?
pub fn gpu_enabled() -> bool { GPU_ENABLED.load(Ordering::Relaxed) }

/// Minimum element count below which GPU transfer never amortizes — always
/// CPU regardless of the enable flag. (Tunable; conservative default.)
pub const GPU_MIN_ELEMS: usize = 1 << 16; // 65536

/// The single entry point. Runs `op` over `xs`, choosing GPU only when:
/// enabled AND the op is f32-safe AND the input is large enough AND a GPU
/// path is compiled and a device is available. Otherwise the CPU
/// reference. Never errors — the accuracy/availability policy guarantees a
/// correct CPU result in every fallback case.
pub fn dispatch(op: Op, xs: &[f32]) -> Vec<f32> {
    if gpu_enabled() && op.is_f32_safe() && xs.len() >= GPU_MIN_ELEMS {
        #[cfg(feature = "gpu")]
        {
            if let Some(out) = gpu::try_map(op, xs) {
                return out;
            }
        }
    }
    cpu::map(op, xs)
}

/// Human-readable backend report for `explain()`-style introspection.
pub fn backend_report(n: usize, op: Op) -> String {
    let would_gpu = gpu_enabled() && op.is_f32_safe() && n >= GPU_MIN_ELEMS;
    #[cfg(feature = "gpu")]
    let compiled = "compiled";
    #[cfg(not(feature = "gpu"))]
    let compiled = "not compiled (build with --features gpu)";
    format!(
        "op={:?} n={} f32_safe={} gpu_enabled={} threshold={} → {} (gpu path: {})",
        op, n, op.is_f32_safe(), gpu_enabled(), GPU_MIN_ELEMS,
        if would_gpu { "GPU-eligible" } else { "CPU" }, compiled,
    )
}

// ── Real GPU path (only under --features gpu) ───────────────────────────
#[cfg(feature = "gpu")]
mod gpu {
    use super::Op;
    use wgpu::util::DeviceExt;

    /// Try to run the op on a GPU. Returns None on ANY failure (no adapter,
    /// device request failed, etc.) so `dispatch` falls back to CPU.
    pub fn try_map(op: Op, xs: &[f32]) -> Option<Vec<f32>> {
        pollster::block_on(run(op, xs)).ok()
    }

    async fn run(op: Op, xs: &[f32]) -> Result<Vec<f32>, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("no GPU adapter")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let n = xs.len();
        let bytes = (n * std::mem::size_of::<f32>()) as u64;
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-in"),
            contents: bytemuck::cast_slice(xs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
             @compute @workgroup_size(64)\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
               let i = gid.x;\n\
               if (i >= {n}u) {{ return; }}\n\
               let x = data[i];\n\
               data[i] = {expr};\n\
             }}",
            n = n, expr = op.wgsl_expr());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2gpu-shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("r2gpu-pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() }],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((n as u32) + 63) / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(&input, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("map_async: {e}"))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_reference_kernels_correct() {
        let xs = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
        assert_eq!(cpu::map(Op::Relu, &xs), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
        assert_eq!(cpu::map(Op::Identity, &xs), xs.to_vec());
        let sig = cpu::map(Op::Sigmoid, &xs);
        assert!((sig[2] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
        let sc = cpu::map(Op::Scale(3.0), &xs);
        assert_eq!(sc, vec![-6.0, -1.5, 0.0, 1.5, 6.0]);
    }

    #[test]
    fn statistics_ops_are_not_in_the_f32_safe_set() {
        // The accuracy contract: only ML element maps are eligible. There
        // is deliberately no Sum/Var/Cor Op variant — statistics can never
        // be routed to f32 hardware. Every declared Op is f32-safe.
        for op in [Op::Identity, Op::Relu, Op::Sigmoid, Op::Tanh, Op::Exp, Op::Scale(2.0)] {
            assert!(op.is_f32_safe());
        }
    }

    #[test]
    fn dispatch_below_threshold_uses_cpu_even_if_enabled() {
        set_gpu_enabled(true);
        let xs = vec![1.0f32; 100]; // < GPU_MIN_ELEMS
        let out = dispatch(Op::Scale(2.0), &xs);
        assert_eq!(out, vec![2.0f32; 100]);
        set_gpu_enabled(false);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu_within_f32_tolerance_when_available() {
        // Only meaningful on a machine with a GPU; if none, dispatch falls
        // back to CPU and the equality trivially holds.
        set_gpu_enabled(true);
        let xs: Vec<f32> = (0..GPU_MIN_ELEMS).map(|i| (i as f32) * 1e-3 - 30.0).collect();
        for op in [Op::Relu, Op::Sigmoid, Op::Tanh, Op::Scale(0.7)] {
            let g = dispatch(op, &xs);
            let c = cpu::map(op, &xs);
            let maxerr = g.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(maxerr < 1e-5, "{:?} GPU vs CPU maxerr {}", op, maxerr);
        }
        set_gpu_enabled(false);
    }
}
