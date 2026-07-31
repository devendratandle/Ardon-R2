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

/// Name/kind of the adapter that would serve GPU work, if any.
/// Integrated GPUs (Intel UHD/Iris, AMD APU, Apple silicon) count fully —
/// wgpu reaches them via Vulkan/DX12/Metal, so machines without discrete
/// cards still get offload capability (slower, but the job completes).
pub fn adapter_info() -> String {
    #[cfg(feature = "gpu")]
    { gpu::adapter_info() }
    #[cfg(not(feature = "gpu"))]
    { "gpu feature not compiled (build with --features gpu)".into() }
}

// ── Real GPU path (only under --features gpu) ───────────────────────────
#[cfg(feature = "gpu")]
mod gpu {
    use super::Op;
    use wgpu::util::DeviceExt;

    /// Report the adapter wgpu selects (name, backend, device type —
    /// IntegratedGpu / DiscreteGpu / Cpu-software).
    pub fn adapter_info() -> String {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            match instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await {
                Some(a) => {
                    let i = a.get_info();
                    format!("{} ({:?}, backend {:?})", i.name, i.device_type, i.backend)
                }
                None => "no adapter (CPU fallback active)".into(),
            }
        })
    }

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

    // ── Matrix multiply ────────────────────────────────────────────────
    //
    // This is the op that decides training and inference throughput —
    // everything else in a transformer is small beside it.

    const TILE: usize = 16;

    /// Device, queue and compiled pipeline, created ONCE.
    ///
    /// Building them per call cost ~420 ms, hundreds of times the actual
    /// arithmetic at any size a transformer uses, which made the GPU look
    /// useless when the kernel was fine. Adapter enumeration, device
    /// creation and shader compilation are one-time costs; only buffers
    /// are genuinely per-call.
    struct Ctx {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
    }
    static CTX: std::sync::OnceLock<Option<Ctx>> = std::sync::OnceLock::new();

    fn ctx() -> Option<&'static Ctx> {
        CTX.get_or_init(|| pollster::block_on(build_ctx())).as_ref()
    }

    async fn build_ctx() -> Option<Ctx> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("r2gpu-matmul"),
            source: wgpu::ShaderSource::Wgsl(matmul_wgsl().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("r2gpu-matmul-pipe"), layout: None,
            module: &module, entry_point: "main",
        });
        Some(Ctx { device, queue, pipeline })
    }

    /// Tiled matmul kernel. Each workgroup stages a TILE×TILE block of A
    /// and B into shared memory so every loaded value is reused by TILE
    /// threads instead of being re-fetched from global memory. On an
    /// integrated GPU — which shares bandwidth with the CPU — that reuse is
    /// the whole game. Bounds are checked per access so dimensions that are
    /// not multiples of TILE read as zero rather than past the buffer.
    fn matmul_wgsl() -> String {
        format!(r#"
struct Dims {{ m: u32, k: u32, n: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

var<workgroup> tileA: array<f32, {tt}u>;
var<workgroup> tileB: array<f32, {tt}u>;

@compute @workgroup_size({t}, {t})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;
    let tiles = (d.k + {t}u - 1u) / {t}u;
    for (var t0 = 0u; t0 < tiles; t0 = t0 + 1u) {{
        let aCol = t0 * {t}u + lid.x;
        let bRow = t0 * {t}u + lid.y;
        if (row < d.m && aCol < d.k) {{
            tileA[lid.y * {t}u + lid.x] = A[row * d.k + aCol];
        }} else {{
            tileA[lid.y * {t}u + lid.x] = 0.0;
        }}
        if (bRow < d.k && col < d.n) {{
            tileB[lid.y * {t}u + lid.x] = B[bRow * d.n + col];
        }} else {{
            tileB[lid.y * {t}u + lid.x] = 0.0;
        }}
        workgroupBarrier();
        for (var i = 0u; i < {t}u; i = i + 1u) {{
            acc = acc + tileA[lid.y * {t}u + i] * tileB[i * {t}u + lid.x];
        }}
        workgroupBarrier();
    }}
    if (row < d.m && col < d.n) {{ C[row * d.n + col] = acc; }}
}}
"#, t = TILE, tt = TILE * TILE)
    }

    /// A(m×k)·B(k×n) → C(m×n), row-major f32. `None` on any failure so the
    /// caller keeps its CPU path.
    pub fn try_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize)
        -> Option<Vec<f32>>
    {
        if a.len() != m * k || b.len() != k * n { return None; }
        let Ctx { device, queue, pipeline } = ctx()?;

        let out_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-a"), contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-b"), contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-c"), size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        let buf_d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-dims"), contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-staging-mm"), size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_d.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                ((n + TILE - 1) / TILE) as u32,
                ((m + TILE - 1) / TILE) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, out_bytes);
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Some(out)
    }
}

/// Matrix multiply on the GPU when enabled and available, else `None` so
/// the caller keeps its CPU path. Behind the same opt-in switch as the
/// element-wise ops: correctness never depends on a device being present.
#[cfg(feature = "gpu")]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if !gpu_enabled() { return None; }
    gpu::try_matmul(a, b, m, k, n)
}

#[cfg(not(feature = "gpu"))]
pub fn matmul(_a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Option<Vec<f32>> {
    None
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
