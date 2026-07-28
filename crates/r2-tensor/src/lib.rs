//! r2-tensor — the LLM tensor substrate (Layer 0 of the trillion-scale
//! architecture; see docs/LLM_TRILLION_ARCHITECTURE.md).
//!
//! Three pieces:
//!   * `dtype` — f32/bf16/f16 conversions + Q8_0/Q4_0 quantized blocks
//!     (why a 32B model fits in ~18 GB, a 1T in ~500 GB mmap'd).
//!   * `ops`   — CPU reference kernels for the transformer op set
//!     (matmul/rmsnorm/softmax/rope/swiglu/embed). These are the accuracy
//!     truth every GPU kernel is checked against.
//!   * `MmapWeights` — a read-only memory-mapped weight file, so the model
//!     data never fully materialises in RAM (bridges to Pillar 2's
//!     out-of-core design; the mesh's `Shard` layer decides placement).
//!
//! All neural compute is f32; the f64 statistical surface is elsewhere and
//! never mixed. This crate has NO heavy deps (only memmap2) and no GPU —
//! the GPU kernels live in r2-gpu behind the accuracy contract.

pub mod dtype;
pub mod infer;
pub mod json;
pub mod safetensors;
pub mod model;
pub mod ops;

use std::fs::File;
use std::path::Path;

/// A memory-mapped weight file: raw bytes on disk, viewed as tensors
/// on demand. The OS pages in only what compute touches — a 500 GB
/// weight file has a bounded RSS. Read-only; the file is the source of
/// truth. (Weight *format* parsing — GGUF/safetensors headers — is Opus;
/// this is the byte-level substrate they build on.)
pub struct MmapWeights {
    map: memmap2::Mmap,
}

impl MmapWeights {
    /// Map a weight file read-only. O(1) — no bytes are read until touched.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let f = File::open(path)?;
        // SAFETY: read-only map of a file we hold open; standard mmap use.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        Ok(MmapWeights { map })
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn bytes(&self) -> &[u8] { &self.map }

    /// View `count` f32 values at byte `offset` WITHOUT copying, when the
    /// region is 4-byte aligned and in bounds. Returns None if it isn't
    /// (caller falls back to `read_f32` which copies). Native endianness.
    pub fn view_f32(&self, offset: usize, count: usize) -> Option<&[f32]> {
        let bytes = count.checked_mul(4)?;
        if offset.checked_add(bytes)? > self.map.len() { return None; }
        let ptr = self.map.as_ptr();
        // SAFETY: bounds checked above; alignment checked here.
        if (unsafe { ptr.add(offset) } as usize) % std::mem::align_of::<f32>() != 0 {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts(ptr.add(offset) as *const f32, count)
        })
    }

    /// Copy `count` f32 values at `offset` (works regardless of alignment;
    /// little-endian, the GGUF/safetensors convention).
    pub fn read_f32(&self, offset: usize, count: usize) -> Option<Vec<f32>> {
        let end = offset.checked_add(count.checked_mul(4)?)?;
        if end > self.map.len() { return None; }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let b = offset + i * 4;
            out.push(f32::from_le_bytes([
                self.map[b], self.map[b + 1], self.map[b + 2], self.map[b + 3],
            ]));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn mmap_reads_f32_back() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("r2tensor_test_{}.bin", std::process::id()));
        let vals: Vec<f32> = vec![1.5, -2.0, 3.25, 0.0];
        {
            let mut f = File::create(&path).unwrap();
            for v in &vals { f.write_all(&v.to_le_bytes()).unwrap(); }
        }
        let w = MmapWeights::open(&path).unwrap();
        assert_eq!(w.len(), 16);
        assert_eq!(w.read_f32(0, 4).unwrap(), vals);
        assert_eq!(w.read_f32(4, 2).unwrap(), vec![-2.0, 3.25]);
        assert!(w.read_f32(0, 100).is_none()); // out of bounds → None
        let _ = std::fs::remove_file(&path);
    }
}
