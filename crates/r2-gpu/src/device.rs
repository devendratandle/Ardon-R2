//! The device: one adapter, one queue, one place buffers live.
//!
//! Everything in the GPU path shares this context. Adapter enumeration,
//! device creation and shader compilation happen once per process
//! (measured at ~420 ms together — hundreds of times any kernel here);
//! only buffers are per-call, and a training run keeps even those
//! resident across steps.
//!
//! No f64 exists in WGSL, so nothing statistical ever comes here: the
//! accuracy contract in `lib.rs` stands, and every kernel in this crate
//! is checked against its CPU reference in the tests.

use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// The process-wide device.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// The device, created on first use; `None` when this machine has no
/// usable adapter, in which case every caller keeps its CPU path.
pub fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(|| pollster::block_on(open())).as_ref()
}

async fn open() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok()?;
    let info = adapter.get_info();
    // Ask for the adapter's own limits rather than WebGPU's defaults:
    // the GEMM's workgroup tiles and a training run's buffers are sized
    // for real hardware, not the browser floor.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("r2-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .ok()?;
    Some(Gpu { device, queue, info })
}

/// Whether the adapter can run `enable f16;` shaders (asked of a fresh
/// adapter, so it says what the hardware offers, not what was requested).
pub fn adapter_offers_f16() -> bool {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default()
        }).await else { return false };
        adapter.features().contains(wgpu::Features::SHADER_F16)
    })
}

/// One line about the adapter, for reports.
pub fn adapter_line() -> String {
    match gpu() {
        Some(g) => format!("{} ({:?}, backend {:?})", g.info.name, g.info.device_type, g.info.backend),
        None => "no adapter (CPU fallback active)".into(),
    }
}

/// An f32 buffer that lives on the device.
///
/// Uploaded once, used by any number of kernels, read back only when the
/// host needs the numbers. A training step keeps weights, activations,
/// gradients and optimizer state here for the whole run; only tokens go
/// in and a loss comes out.
pub struct Tensor {
    pub(crate) buf: wgpu::Buffer,
    pub len: usize,
}

impl Tensor {
    /// Upload host data.
    pub fn upload(data: &[f32]) -> Option<Tensor> {
        let g = gpu()?;
        let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-tensor"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Some(Tensor { buf, len: data.len() })
    }

    /// A zero-filled buffer of `len` elements.
    pub fn zeros(len: usize) -> Option<Tensor> {
        let g = gpu()?;
        let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-tensor"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Tensor { buf, len })
    }

    /// Overwrite the buffer's contents from the host.
    pub fn write(&self, data: &[f32]) {
        if let Some(g) = gpu() {
            g.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(data));
        }
    }

    /// Read the buffer back to the host. Waits for every submitted
    /// command to finish first, so this is also the synchronisation point
    /// a caller uses to time a kernel.
    pub fn download(&self) -> Vec<f32> {
        let Some(g) = gpu() else { return Vec::new() };
        let bytes = (self.len * 4) as u64;
        let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-staging"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = g.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf, 0, &staging, 0, bytes);
        g.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().ok().and_then(|r| r.ok()).is_none() { return Vec::new(); }
        let Ok(data) = slice.get_mapped_range() else { return Vec::new() };
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

/// Block until the queue is idle — the fence a benchmark closes on.
pub fn sync() {
    if let Some(g) = gpu() {
        let _ = g.device.poll(wgpu::PollType::wait_indefinitely());
    }
}
