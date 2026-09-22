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
//!
//! Two storage types: f32, and f16 where the adapter offers `shader-f16`
//! (asked for at device creation when it does). A kernel reads either
//! through a generated load helper and always computes in f32; f16 is a
//! storage format here — half the bytes for activations and the weight
//! copies the GEMMs read — not an arithmetic one.

use crate::half::{f16_to_f32, f32_to_f16};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// The process-wide device.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    /// Whether f16 storage is available (`shader-f16` was granted).
    pub f16: bool,
}

/// Element type of a [`Tensor`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dtype { F32, F16 }

impl Dtype {
    pub fn bytes(self) -> usize { match self { Dtype::F32 => 4, Dtype::F16 => 2 } }
    /// The WGSL element type.
    pub fn wgsl(self) -> &'static str { match self { Dtype::F32 => "f32", Dtype::F16 => "f16" } }
    /// A one-letter tag for pipeline cache keys.
    pub fn tag(self) -> char { match self { Dtype::F32 => 's', Dtype::F16 => 'h' } }
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
    let f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("r2-gpu"),
            required_features: if f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() },
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .ok()?;
    Some(Gpu { device, queue, info, f16 })
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

/// A buffer of f32 or f16 elements that lives on the device.
///
/// Uploaded once, used by any number of kernels, read back only when the
/// host needs the numbers. A training step keeps weights, activations,
/// gradients and optimizer state here for the whole run; only tokens go
/// in and a loss comes out. The host side is always f32: an f16 tensor
/// converts on the way in and out.
pub struct Tensor {
    pub(crate) buf: wgpu::Buffer,
    pub len: usize,
    pub dtype: Dtype,
}

fn as_bytes(data: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => bytemuck::cast_slice(data).to_vec(),
        Dtype::F16 => data.iter().flat_map(|&x| f32_to_f16(x).to_le_bytes()).collect(),
    }
}

impl Tensor {
    /// Upload host data as f32.
    pub fn upload(data: &[f32]) -> Option<Tensor> { Tensor::upload_as(data, Dtype::F32) }

    /// Upload host data, stored as `dtype` (`None` when the device lacks
    /// f16 and f16 was asked for).
    pub fn upload_as(data: &[f32], dtype: Dtype) -> Option<Tensor> {
        let g = gpu()?;
        if dtype == Dtype::F16 && !g.f16 { return None; }
        let bytes = as_bytes(data, dtype);
        let buf = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("r2gpu-tensor"),
            contents: if bytes.is_empty() { &[0u8; 4] } else { &bytes },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Some(Tensor { buf, len: data.len(), dtype })
    }

    /// A zero-filled f32 buffer of `len` elements.
    pub fn zeros(len: usize) -> Option<Tensor> { Tensor::zeros_as(len, Dtype::F32) }

    /// A zero-filled buffer of `len` elements of `dtype`.
    pub fn zeros_as(len: usize, dtype: Dtype) -> Option<Tensor> {
        let g = gpu()?;
        if dtype == Dtype::F16 && !g.f16 { return None; }
        let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r2gpu-tensor"),
            size: ((len * dtype.bytes()).max(4)).div_ceil(4) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Tensor { buf, len, dtype })
    }

    /// Overwrite the buffer's contents from the host.
    pub fn write(&self, data: &[f32]) { self.write_at(0, data) }

    /// Overwrite `data.len()` elements starting at element `at`. Queued
    /// in order with the kernels, like everything else.
    pub fn write_at(&self, at: usize, data: &[f32]) {
        if let Some(g) = gpu() {
            g.queue.write_buffer(&self.buf, (at * self.dtype.bytes()) as u64, &as_bytes(data, self.dtype));
        }
    }

    /// Read the buffer back to the host as f32. Waits for every submitted
    /// command to finish first, so this is also the synchronisation point
    /// a caller uses to time a kernel.
    pub fn download(&self) -> Vec<f32> {
        let Some(g) = gpu() else { return Vec::new() };
        let bytes = ((self.len * self.dtype.bytes()).div_ceil(4) * 4) as u64;
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
        let out: Vec<f32> = match self.dtype {
            Dtype::F32 => bytemuck::cast_slice(&data).to_vec(),
            Dtype::F16 => data.chunks_exact(2).take(self.len)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        };
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
