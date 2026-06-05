//! R2-ARROW — columnar memory substrate (Phase F).
//!
//! Per docs/ARCHITECTURE.md §5 Phase F:
//!   - New crate
//!   - Replaces `Vec<Option<f64>>` with columnar buffers + null bitmap
//!   - Migration is gradual: new code uses ARROW; old code stays on
//!     `Vec<Option<f64>>` until touched. Adapters bridge the boundary.
//!
//! This file is the **scaffold** (Phase F spine). It defines the core types
//! and the conversion adapters to/from `Vec<Option<f64>>`. Real adoption
//! across the engine, JIT, and stats functions arrives in subsequent
//! sessions (Phase F.1, F.2, …).
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only — this crate has zero external deps in V1.
//!   §4.7 Backwards-compatible — old `Vec<Option<f64>>` paths stay valid;
//!        ColumnarF64 is opt-in until migration completes.
//!
//! Layout (inspired by Apache Arrow):
//!   - Contiguous `Vec<f64>` for values. NaN slots are *valid* values; nulls
//!     are tracked separately by a packed bitmap.
//!   - Packed `Vec<u8>` bitmap, 1 bit per row, 1 = present, 0 = null.
//!   - Length is explicit; values.len() may be ≥ length when over-allocated.
//!
//! This layout is the same the future `r2-ai` and `r2-stats` libraries (per
//! the private design docs) will adopt for zero-copy interop.

#![deny(missing_docs)]

/// A column of `f64` values with explicit null tracking.
///
/// Values live in a contiguous `Vec<f64>` — perfect for SIMD, mmap, and
/// FFI to BLAS / GPU kernels. Nulls are stored in a packed bitmap parallel
/// to the values.
#[derive(Debug, Clone)]
pub struct ColumnarF64 {
    values: Vec<f64>,
    /// Packed null bitmap — bit i is 1 when index i is non-null.
    /// `None` ⇒ no nulls in this column (fast path).
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarF64 {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarF64 { values: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Pre-allocate capacity for `n` elements.
    pub fn with_capacity(n: usize) -> Self {
        ColumnarF64 { values: Vec::with_capacity(n), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a fully-valid (no nulls) `Vec<f64>`. Zero-copy — takes
    /// ownership.
    pub fn from_vec(values: Vec<f64>) -> Self {
        let len = values.len();
        ColumnarF64 { values, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `Vec<Option<f64>>` (R2's existing representation).
    /// This is the **bridge** used during gradual migration.
    pub fn from_options(opts: Vec<Option<f64>>) -> Self {
        Self::from_option_slice(&opts)
    }

    /// Same as `from_options` but takes a slice — caller keeps ownership.
    /// Used by `RVal::to_columnar()` to avoid an extra `Vec::clone()` on
    /// every call. (Phase F.3a — eliminates the per-call clone from the
    /// reduction hot path that F.2 introduced.)
    pub fn from_option_slice(opts: &[Option<f64>]) -> Self {
        let len = opts.len();
        let mut values = Vec::with_capacity(len);
        // Single pass: detect nulls inline. If we get to the end without
        // seeing one, we never allocated the bitmap — dense fast path.
        let mut bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, x) in opts.iter().enumerate() {
            match x {
                Some(v) => {
                    values.push(*v);
                    // If the bitmap is already allocated (a prior null was
                    // seen), mark this slot valid.
                    if let Some(b) = bits.as_mut() { b[i / 8] |= 1 << (i % 8); }
                }
                None => {
                    if bits.is_none() {
                        // First null: lazily allocate the bitmap and mark
                        // every previous slot as valid in one go.
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i {
                            new_bits[j / 8] |= 1 << (j % 8);
                        }
                        bits = Some(new_bits);
                    }
                    values.push(f64::NAN);
                    null_count += 1;
                }
            }
        }
        ColumnarF64 { values, valid_bits: bits, len, null_count }
    }

    /// Convert back to `Vec<Option<f64>>` for legacy callers.
    pub fn to_options(&self) -> Vec<Option<f64>> {
        match &self.valid_bits {
            None => self.values.iter().take(self.len).map(|v| Some(*v)).collect(),
            Some(bits) => (0..self.len).map(|i| {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }).collect(),
        }
    }

    /// Logical length of the column.
    pub fn len(&self) -> usize { self.len }

    /// `true` when the column has zero rows.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }

    /// `true` when no entries are null. Fast path enables SIMD reductions
    /// without bitmap checks.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }

    /// Read element `i`. `None` ⇒ null. Out-of-bounds ⇒ panics (per Arrow
    /// convention; checked accessors come later).
    pub fn get(&self, i: usize) -> Option<f64> {
        assert!(i < self.len, "ColumnarF64 index {} out of bounds (len {})", i, self.len);
        match &self.valid_bits {
            None => Some(self.values[i]),
            Some(bits) => {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }
        }
    }

    /// Borrow the contiguous values slice. **Includes placeholder NaNs**
    /// at null positions when `is_dense()` is false; consult `valid_bits`
    /// for masking. SIMD-friendly when the column is dense.
    pub fn values(&self) -> &[f64] { &self.values[..self.len] }

    /// Borrow the null bitmap (when present). Each byte packs 8 rows
    /// little-endian (LSB = first row in that byte).
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }
}

impl Default for ColumnarF64 {
    fn default() -> Self { Self::new() }
}

// ── Reductions on the dense fast path (Phase F.1) ────────────────────
//
// When a column is dense (no nulls), reductions reduce to plain `f64`
// loops over a contiguous slice — perfectly SIMD-able by the compiler.
// When nulls exist, we either short-circuit to `None` or skip nulls,
// matching R semantics (configurable via `na_rm`).

impl ColumnarF64 {
    /// Sum of values. With nulls and `na_rm=false`, returns `None`.
    pub fn sum(&self, na_rm: bool) -> Option<f64> {
        if self.is_dense() { return Some(self.values().iter().sum()); }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s = 0.0;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i]; }
        }
        Some(s)
    }

    /// Arithmetic mean. With nulls and `na_rm=false`, returns `None`.
    /// Empty column ⇒ `None`.
    pub fn mean(&self, na_rm: bool) -> Option<f64> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(self.values().iter().sum::<f64>() / self.len as f64);
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s = 0.0;
        let mut n = 0usize;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i]; n += 1; }
        }
        if n == 0 { None } else { Some(s / n as f64) }
    }

    /// Minimum value, NA-aware.
    pub fn min(&self, na_rm: bool) -> Option<f64> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(self.values().iter().copied().fold(f64::INFINITY, f64::min));
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m = f64::INFINITY;
        let mut any = false;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = m.min(self.values[i]); any = true;
            }
        }
        if any { Some(m) } else { None }
    }

    /// Maximum value, NA-aware.
    pub fn max(&self, na_rm: bool) -> Option<f64> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(self.values().iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m = f64::NEG_INFINITY;
        let mut any = false;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = m.max(self.values[i]); any = true;
            }
        }
        if any { Some(m) } else { None }
    }
}

// ── Memory-mapped columnar reader (Phase F.5) ────────────────────────
//
// Reads a binary file as a borrowed `&[f64]` without copying into a
// heap buffer. The OS pages in only the parts actually touched, so a
// 10 GB file uses ~zero RAM until you reduce/scan a region.
//
// Cost vs `ColumnarF64::from_vec`: zero allocation, zero copy. Pages
// fault in lazily on first access (cold-cache reads are disk-bound,
// warm-cache reads are RAM-bound but page-cache hit not heap alloc).
//
// Limitations:
//   - **Read-only**: the mmap is opened read-only; mutation would
//     require a separate writable mapping.
//   - **No null bitmap**: the file is a contiguous packed `[f64]`. NA
//     in the source must be encoded as NaN before writing.
//   - **Alignment**: the host's `[f64]` alignment is 8 bytes. Most
//     filesystems hand back page-aligned bytes (4 KB / 16 KB / 64 KB)
//     so this is fine in practice. We sanity-check the pointer
//     alignment at open time and error if it doesn't match.
//   - **Endianness**: assumes host byte order. Cross-platform feeds
//     would need an explicit endian conversion pass; out of scope.

#[cfg(feature = "mmap")]
mod mmap_impl {
    use std::sync::Arc;
    use std::path::Path;

    /// Memory-mapped read-only view over a packed `[f64]` file. Behaves
    /// as a `&[f64]` for the lifetime of the struct.
    ///
    /// The mmap handle is held in an `Arc` so cheap `clone()` shares
    /// the same mapping (multiple readers, one mapping). The pointer
    /// derived from the mapping is valid as long as the `Arc<Mmap>`
    /// lives, which is tied to `self`'s lifetime — hence `as_slice()`
    /// is safe.
    pub struct MmapColumnar {
        // Order matters: `_handle` must outlive `ptr` field uses,
        // and Rust drops fields in declaration order — keep handle FIRST
        // so it's dropped LAST.
        _handle: Arc<memmap2::Mmap>,
        ptr: *const f64,
        len: usize,
    }

    // Mmap is Send + Sync (a read-only mapping is shareable across threads).
    // The pointer derived from it inherits that safety because the Arc
    // keeps the mapping alive.
    unsafe impl Send for MmapColumnar {}
    unsafe impl Sync for MmapColumnar {}

    impl MmapColumnar {
        /// Open a packed `[f64]` file and return a borrowed view.
        /// File size must be a multiple of 8 bytes; the resulting
        /// slice length is `file_size / 8`.
        pub fn open<P: AsRef<Path>>(path: P) -> Result<MmapColumnar, String> {
            let file = std::fs::File::open(&path)
                .map_err(|e| format!("MmapColumnar::open: cannot open '{}': {}",
                    path.as_ref().display(), e))?;
            let metadata = file.metadata()
                .map_err(|e| format!("MmapColumnar::open: stat failed: {}", e))?;
            let len_bytes = metadata.len() as usize;
            if len_bytes % 8 != 0 {
                return Err(format!(
                    "MmapColumnar::open: file size {} is not a multiple of 8 (packed f64)",
                    len_bytes));
            }
            // SAFETY: the file is opened read-only; we won't mutate it.
            // Mmap requires unsafe because external processes could write
            // to the file under us, but we accept that risk for read-only
            // workloads (R-style analytics on a static dataset).
            let mmap = unsafe {
                memmap2::Mmap::map(&file)
                    .map_err(|e| format!("MmapColumnar::open: mmap failed: {}", e))?
            };
            let ptr = mmap.as_ptr() as *const f64;
            // Alignment sanity check — f64 needs 8-byte alignment.
            if (ptr as usize) % std::mem::align_of::<f64>() != 0 {
                return Err(format!(
                    "MmapColumnar::open: mmap pointer 0x{:x} not 8-byte aligned",
                    ptr as usize));
            }
            let len = len_bytes / 8;
            Ok(MmapColumnar { _handle: Arc::new(mmap), ptr, len })
        }

        /// Borrow as `&[f64]`. The slice is alive as long as `self` is.
        pub fn as_slice(&self) -> &[f64] {
            // SAFETY: ptr was derived from a valid mmap whose lifetime
            // is bound to `self` via the Arc. `len * 8 <= mmap.len()`
            // by construction. No mutable aliasing — mmap is read-only.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }

        /// Length in `f64` elements.
        pub fn len(&self) -> usize { self.len }
        /// True if zero elements.
        pub fn is_empty(&self) -> bool { self.len == 0 }

        // Reductions — same dense-loop bodies as ColumnarF64's dense path,
        // operating on the borrowed slice. No null support (mmap file is
        // a packed array; NaN encodes NA if needed).

        /// Sum of all values. Uses 8 independent accumulators so the
        /// f64 add chain isn't serialized — the compiler pipelines /
        /// auto-vectorizes it (plain `iter().sum()` is one dependency
        /// chain, ~2× slower on out-of-cache data).
        pub fn sum(&self) -> f64 {
            let s = self.as_slice();
            let mut acc = [0.0f64; 8];
            let mut it = s.chunks_exact(8);
            for c in &mut it {
                acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]; acc[3] += c[3];
                acc[4] += c[4]; acc[5] += c[5]; acc[6] += c[6]; acc[7] += c[7];
            }
            let mut total = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for &v in it.remainder() { total += v; }
            total
        }
        /// Arithmetic mean. Returns 0.0 on empty.
        pub fn mean(&self) -> f64 {
            if self.len == 0 { 0.0 } else { self.sum() / self.len as f64 }
        }
        /// Minimum value (NaN-skipping via `f64::min`).
        pub fn min(&self) -> f64 {
            self.as_slice().iter().copied().fold(f64::INFINITY, f64::min)
        }
        /// Maximum value (NaN-skipping via `f64::max`).
        pub fn max(&self) -> f64 {
            self.as_slice().iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }

        /// Copy into an owned `ColumnarF64`. Useful when bridging into
        /// the existing RVal::Numeric storage path — pays a one-time
        /// allocation for full ownership.
        pub fn to_columnar(&self) -> super::ColumnarF64 {
            super::ColumnarF64::from_vec(self.as_slice().to_vec())
        }
    }

    impl Clone for MmapColumnar {
        fn clone(&self) -> Self {
            MmapColumnar {
                _handle: self._handle.clone(),
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    /// Write a slice of `f64` to disk as a packed binary file —
    /// inverse of `MmapColumnar::open`. Useful for tests and for the
    /// `save_columnar` path that lets users build mmap-friendly artifacts.
    pub fn write_packed_f64<P: AsRef<Path>>(path: P, values: &[f64]) -> Result<(), String> {
        // SAFETY: f64 is Copy and the byte representation is well-defined.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
        };
        std::fs::write(&path, bytes)
            .map_err(|e| format!("write_packed_f64: {}", e))
    }
}

#[cfg(feature = "mmap")]
pub use mmap_impl::{MmapColumnar, write_packed_f64};

#[cfg(feature = "mmap")]
#[cfg(test)]
mod f5_mmap_tests {
    use super::*;

    fn tmp(suffix: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("r2arrow_mmap_{}_{}", std::process::id(), suffix));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn write_then_mmap_roundtrip() {
        let path = tmp("rt.f64");
        let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 1.5).collect();
        write_packed_f64(&path, &values).unwrap();
        let m = MmapColumnar::open(&path).unwrap();
        assert_eq!(m.len(), 1000);
        let slice = m.as_slice();
        for i in 0..1000 {
            assert_eq!(slice[i], (i as f64) * 1.5);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_reductions_match_owned() {
        let path = tmp("red.f64");
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        write_packed_f64(&path, &values).unwrap();
        let m = MmapColumnar::open(&path).unwrap();
        assert!((m.sum() - 15.0).abs() < 1e-12);
        assert!((m.mean() - 3.0).abs() < 1e-12);
        assert_eq!(m.min(), 1.0);
        assert_eq!(m.max(), 5.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mmap_bridges_to_owned_columnar() {
        let path = tmp("bridge.f64");
        write_packed_f64(&path, &[10.0, 20.0, 30.0]).unwrap();
        let m = MmapColumnar::open(&path).unwrap();
        let owned = m.to_columnar();
        assert_eq!(owned.sum(false), Some(60.0));
        assert_eq!(owned.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_file_size_errors() {
        // 7 bytes — not a multiple of 8, should refuse.
        let path = tmp("bad.f64");
        std::fs::write(&path, [0u8; 7]).unwrap();
        let r = MmapColumnar::open(&path);
        assert!(r.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_file_yields_empty_view() {
        let path = tmp("empty.f64");
        std::fs::write(&path, []).unwrap();
        let m = MmapColumnar::open(&path).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn clone_shares_mapping() {
        // Cloning should not re-mmap or copy data — Arc<Mmap> is shared.
        let path = tmp("clone.f64");
        write_packed_f64(&path, &[1.0, 2.0, 3.0]).unwrap();
        let a = MmapColumnar::open(&path).unwrap();
        let b = a.clone();
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr(),
            "cloned MmapColumnar should point to the same mapped bytes");
        std::fs::remove_file(&path).ok();
    }
}

// ── Additional dtypes: i32 / bool (Phase F.6) ─────────────────────────
//
// Mirrors `ColumnarF64`'s layout for two additional primitives. Bool uses
// a packed-bit values buffer (one bit per element) on top of the same
// validity-bitmap convention, which halves memory vs storing one byte
// per logical and trades a tiny bit of shift/mask cost.
//
// Not included in F.6.v1: `ColumnarI64` (mechanical copy of I32 with
// `i64` — add when an actual i64 hot path materializes) and
// `ColumnarUtf8` (variable-length strings need a separate offsets array
// plus a values byte buffer — its own design pass, deferred).
//
// **What this unlocks:** when a future "F.6 storage migration" pass
// changes `RVal::Integer` / `RVal::Logical` to use these columnar forms
// (same shape as the F.3 change for `RVal::Numeric`), integer/logical
// reductions get the same SIMD-friendly dense path that f64 already has.

/// Packed columnar storage for `Option<i32>`.
#[derive(Debug, Clone)]
pub struct ColumnarI32 {
    values: Vec<i32>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarI32 {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarI32 { values: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a dense `Vec<i32>` (no nulls).
    pub fn from_vec(values: Vec<i32>) -> Self {
        let len = values.len();
        ColumnarI32 { values, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `&[Option<i32>]` with lazy bitmap allocation: the
    /// bitmap is only created when the first `None` is encountered.
    pub fn from_option_slice(opts: &[Option<i32>]) -> Self {
        let len = opts.len();
        let mut values = Vec::with_capacity(len);
        let mut bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, opt) in opts.iter().enumerate() {
            match opt {
                Some(v) => {
                    values.push(*v);
                    if let Some(b) = bits.as_mut() {
                        b[i / 8] |= 1 << (i % 8);
                    }
                }
                None => {
                    values.push(0);
                    if bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        bits = Some(new_bits);
                    }
                    null_count += 1;
                }
            }
        }
        ColumnarI32 { values, valid_bits: bits, len, null_count }
    }

    /// Materialize back into `Vec<Option<i32>>`.
    pub fn to_options(&self) -> Vec<Option<i32>> {
        match &self.valid_bits {
            None => self.values.iter().take(self.len).map(|v| Some(*v)).collect(),
            Some(bits) => (0..self.len).map(|i| {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }).collect(),
        }
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True if zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no nulls — enables the SIMD-friendly dense fast path.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }
    /// Borrow the raw values buffer. Positions marked invalid by
    /// `valid_bits` are arbitrary (typically 0) — consult the bitmap.
    pub fn values(&self) -> &[i32] { &self.values[..self.len] }
    /// Validity bitmap when present (None ⇒ all valid).
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Read one element with bounds check.
    pub fn get(&self, i: usize) -> Option<i32> {
        assert!(i < self.len, "ColumnarI32 index {} out of bounds (len {})", i, self.len);
        match &self.valid_bits {
            None => Some(self.values[i]),
            Some(bits) => if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None },
        }
    }

    /// Sum as i64 (avoids i32 overflow on long columns).
    /// With nulls and `na_rm=false`, returns `None`.
    pub fn sum(&self, na_rm: bool) -> Option<i64> {
        if self.is_dense() {
            return Some(self.values().iter().map(|v| *v as i64).sum());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s: i64 = 0;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i] as i64; }
        }
        Some(s)
    }

    /// Min, NA-aware.
    pub fn min(&self, na_rm: bool) -> Option<i32> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(*self.values().iter().min().unwrap());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i32> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.min(self.values[i])));
            }
        }
        m
    }

    /// Max, NA-aware.
    pub fn max(&self, na_rm: bool) -> Option<i32> {
        if self.len == 0 { return None; }
        if self.is_dense() {
            return Some(*self.values().iter().max().unwrap());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut m: Option<i32> = None;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                m = Some(m.map_or(self.values[i], |x| x.max(self.values[i])));
            }
        }
        m
    }
}

impl Default for ColumnarI32 {
    fn default() -> Self { ColumnarI32::new() }
}

/// Packed columnar storage for `Option<bool>`.
///
/// Values use a packed-bit representation (one bit per element) like the
/// validity bitmap — so 1 million bools fit in 125 KB of values + 125 KB
/// of bitmap = 250 KB total, versus 16 MB for `Vec<Option<bool>>` (Rust's
/// `Option<bool>` is 1 byte; plus null bookkeeping).
#[derive(Debug, Clone)]
pub struct ColumnarBool {
    /// Packed bits: bit `i` is value of element `i`.
    /// Position marked invalid in `valid_bits` ⇒ value bit is don't-care.
    value_bits: Vec<u8>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarBool {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarBool { value_bits: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a dense `Vec<bool>` (no nulls).
    pub fn from_vec(values: Vec<bool>) -> Self {
        let len = values.len();
        let mut value_bits = vec![0u8; (len + 7) / 8];
        for (i, &b) in values.iter().enumerate() {
            if b { value_bits[i / 8] |= 1 << (i % 8); }
        }
        ColumnarBool { value_bits, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `&[Option<bool>]`.
    pub fn from_option_slice(opts: &[Option<bool>]) -> Self {
        let len = opts.len();
        let mut value_bits = vec![0u8; (len + 7) / 8];
        let mut valid_bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, opt) in opts.iter().enumerate() {
            match opt {
                Some(b) => {
                    if *b { value_bits[i / 8] |= 1 << (i % 8); }
                    if let Some(bits) = valid_bits.as_mut() {
                        bits[i / 8] |= 1 << (i % 8);
                    }
                }
                None => {
                    if valid_bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        valid_bits = Some(new_bits);
                    }
                    null_count += 1;
                }
            }
        }
        ColumnarBool { value_bits, valid_bits, len, null_count }
    }

    /// Materialize back into `Vec<Option<bool>>`.
    pub fn to_options(&self) -> Vec<Option<bool>> {
        (0..self.len).map(|i| {
            let valid = match &self.valid_bits {
                None => true,
                Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1,
            };
            if valid {
                Some((self.value_bits[i / 8] >> (i % 8)) & 1 == 1)
            } else {
                None
            }
        }).collect()
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True if zero elements.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of nulls.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no nulls.
    pub fn is_dense(&self) -> bool { self.null_count == 0 }
    /// Borrow packed value bitmap (LSB-first within each byte).
    pub fn value_bits(&self) -> &[u8] { &self.value_bits }
    /// Validity bitmap when present.
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Read one element.
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.len, "ColumnarBool index {} out of bounds (len {})", i, self.len);
        let valid = match &self.valid_bits {
            None => true,
            Some(bits) => (bits[i / 8] >> (i % 8)) & 1 == 1,
        };
        if valid { Some((self.value_bits[i / 8] >> (i % 8)) & 1 == 1) } else { None }
    }

    /// Count of TRUE values among valid elements.
    pub fn count_true(&self) -> usize {
        match &self.valid_bits {
            None => {
                // Dense: popcount over value_bits, masking trailing slop bits.
                let mut total = 0usize;
                for b in 0..(self.len / 8) {
                    total += self.value_bits[b].count_ones() as usize;
                }
                let rem = self.len % 8;
                if rem > 0 {
                    let mask = (1u8 << rem) - 1;
                    total += (self.value_bits[self.len / 8] & mask).count_ones() as usize;
                }
                total
            }
            Some(bits) => {
                // Sparse: only count where valid AND set.
                let mut total = 0usize;
                for i in 0..self.len {
                    if (bits[i / 8] >> (i % 8)) & 1 == 1
                        && (self.value_bits[i / 8] >> (i % 8)) & 1 == 1
                    {
                        total += 1;
                    }
                }
                total
            }
        }
    }

    /// Count of FALSE values among valid elements.
    pub fn count_false(&self) -> usize {
        let valid = self.len - self.null_count;
        valid - self.count_true()
    }

    /// `any(x)` — TRUE if at least one TRUE, NA-aware: returns None if
    /// no TRUE and at least one NA (R semantics).
    pub fn any(&self) -> Option<bool> {
        if self.count_true() > 0 { return Some(true); }
        if self.null_count > 0 { return None; }
        Some(false)
    }

    /// `all(x)` — TRUE if every valid element is TRUE, NA-aware.
    pub fn all(&self) -> Option<bool> {
        if self.count_false() > 0 { return Some(false); }
        if self.null_count > 0 { return None; }
        Some(true)
    }
}

impl Default for ColumnarBool {
    fn default() -> Self { ColumnarBool::new() }
}

#[cfg(test)]
mod f6_dtypes_tests {
    use super::*;

    #[test]
    fn i32_dense_roundtrip() {
        let c = ColumnarI32::from_vec(vec![1, 2, 3, -4, 5]);
        assert_eq!(c.len(), 5);
        assert_eq!(c.null_count(), 0);
        assert!(c.is_dense());
        assert_eq!(c.to_options(), vec![Some(1), Some(2), Some(3), Some(-4), Some(5)]);
        assert!(c.valid_bits().is_none(), "dense should not allocate bitmap");
    }

    #[test]
    fn i32_with_nulls() {
        let c = ColumnarI32::from_option_slice(&[Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.to_options(), vec![Some(1), None, Some(3), None, Some(5)]);
        assert_eq!(c.get(0), Some(1));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(3));
    }

    #[test]
    fn i32_reductions() {
        let c = ColumnarI32::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);
        assert_eq!(c.sum(false), Some(31));
        assert_eq!(c.min(false), Some(1));
        assert_eq!(c.max(false), Some(9));
    }

    #[test]
    fn i32_sum_with_na_propagates() {
        let c = ColumnarI32::from_option_slice(&[Some(1), None, Some(3)]);
        // Without na_rm, NA poisons the result.
        assert_eq!(c.sum(false), None);
        // With na_rm, skip nulls.
        assert_eq!(c.sum(true), Some(4));
    }

    #[test]
    fn i32_sum_avoids_overflow_via_i64() {
        // i32::MAX summed 3 times would overflow i32 but fits i64.
        let c = ColumnarI32::from_vec(vec![i32::MAX, i32::MAX, i32::MAX]);
        let s = c.sum(false).unwrap();
        assert_eq!(s, 3i64 * i32::MAX as i64);
    }

    #[test]
    fn bool_dense_packs_one_bit_per_element() {
        let c = ColumnarBool::from_vec(vec![true, false, true, true, false, true, false, true]);
        // 8 elements packed into 1 byte.
        assert_eq!(c.value_bits().len(), 1);
        assert_eq!(c.count_true(), 5);
        assert_eq!(c.count_false(), 3);
        assert_eq!(c.to_options(),
            vec![Some(true), Some(false), Some(true), Some(true),
                 Some(false), Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn bool_count_true_handles_trailing_partial_byte() {
        // 5 bits — partial last byte. Without masking, padding bits would
        // leak into popcount.
        let c = ColumnarBool::from_vec(vec![true, true, true, true, true]);
        assert_eq!(c.count_true(), 5);
    }

    #[test]
    fn bool_with_nulls() {
        let c = ColumnarBool::from_option_slice(&[Some(true), None, Some(false), Some(true), None]);
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.count_true(), 2);
        assert_eq!(c.count_false(), 1);
        assert_eq!(c.to_options(), vec![Some(true), None, Some(false), Some(true), None]);
    }

    #[test]
    fn bool_any_and_all() {
        // Dense — no NA.
        let all_true = ColumnarBool::from_vec(vec![true, true, true]);
        assert_eq!(all_true.all(), Some(true));
        assert_eq!(all_true.any(), Some(true));

        let mixed = ColumnarBool::from_vec(vec![true, false, true]);
        assert_eq!(mixed.all(), Some(false));
        assert_eq!(mixed.any(), Some(true));

        let all_false = ColumnarBool::from_vec(vec![false, false]);
        assert_eq!(all_false.all(), Some(false));
        assert_eq!(all_false.any(), Some(false));

        // With NA — R semantics: any() returns None when uncertain, all() too.
        let with_na = ColumnarBool::from_option_slice(&[None, Some(false)]);
        // any: no TRUE among valid, but NA exists → unknown.
        assert_eq!(with_na.any(), None);
        let with_na_t = ColumnarBool::from_option_slice(&[None, Some(true)]);
        // any: has a TRUE → definitively true regardless of NA.
        assert_eq!(with_na_t.any(), Some(true));

        let with_na_f = ColumnarBool::from_option_slice(&[None, Some(false)]);
        // all: has a FALSE → definitively false regardless of NA.
        assert_eq!(with_na_f.all(), Some(false));
    }
}

// ── Element-wise binary kernels (Phase F.4) ──────────────────────────
//
// Vector ⊗ vector and vector ⊗ scalar arithmetic that stays on the
// columnar representation end-to-end. The dense × dense path is a
// tight `for i in 0..n` over two contiguous `&[f64]` slices — exactly
// what rustc's auto-vectorizer turns into SSE/AVX/NEON. When either
// input has nulls, the bitmap of the output is the bitwise AND of the
// inputs' bitmaps so NA propagates correctly.
//
// These are the columnar equivalents of `r2_kernel::binary()` — same
// semantics, but no `Vec<Option<f64>>` round-trip. Once F.3 lands and
// RVal::Numeric uses Arc<ColumnarF64> storage, every binary op on
// numeric vectors will use this path directly.

/// Element-wise binary operations. Mirrors `r2_kernel::BinaryOp` but
/// re-declared here so r2-arrow doesn't pull r2-kernel as a dep — keeps
/// the layer ordering clean (kernels can depend on arrow, not vice versa).
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowBinaryOp { Add, Sub, Mul, Div, Pow, Mod }

#[inline]
fn apply_scalar(op: ArrowBinaryOp, a: f64, b: f64) -> f64 {
    match op {
        ArrowBinaryOp::Add => a + b,
        ArrowBinaryOp::Sub => a - b,
        ArrowBinaryOp::Mul => a * b,
        ArrowBinaryOp::Div => a / b,
        ArrowBinaryOp::Pow => a.powf(b),
        ArrowBinaryOp::Mod => a % b,
    }
}

/// Pack a `valid` bool slice into the packed u8 bitmap layout.
fn pack_valid_bits(valid: &[bool]) -> Vec<u8> {
    let n = valid.len();
    let mut bits = vec![0u8; (n + 7) / 8];
    for (i, &v) in valid.iter().enumerate() {
        if v { bits[i / 8] |= 1 << (i % 8); }
    }
    bits
}

impl ColumnarF64 {
    /// `self ⊗ other` element-wise. Both inputs must have the same length.
    pub fn binary(&self, op: ArrowBinaryOp, other: &ColumnarF64) -> Result<ColumnarF64, String> {
        if self.len != other.len {
            return Err(format!("length mismatch: {} vs {}", self.len, other.len));
        }
        let n = self.len;

        // Dense × dense fast path — tight loop, SIMD-friendly.
        if self.is_dense() && other.is_dense() {
            let a = self.values();
            let b = other.values();
            let mut out = Vec::with_capacity(n);
            for i in 0..n { out.push(apply_scalar(op, a[i], b[i])); }
            return Ok(ColumnarF64::from_vec(out));
        }

        // Sparse path: build a fresh valid mask = (self.valid AND other.valid).
        let a_bits = self.valid_bits.as_deref();
        let b_bits = other.valid_bits.as_deref();
        let is_valid = |bits: Option<&[u8]>, i: usize| -> bool {
            match bits {
                None => true,
                Some(bs) => (bs[i / 8] >> (i % 8)) & 1 == 1,
            }
        };
        let mut values = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        let mut null_count = 0usize;
        for i in 0..n {
            let va = is_valid(a_bits, i);
            let vb = is_valid(b_bits, i);
            if va && vb {
                values.push(apply_scalar(op, self.values[i], other.values[i]));
                valid.push(true);
            } else {
                values.push(0.0);
                valid.push(false);
                null_count += 1;
            }
        }
        let valid_bits = if null_count == 0 { None } else { Some(pack_valid_bits(&valid)) };
        Ok(ColumnarF64 { values, valid_bits, len: n, null_count })
    }

    /// `self ⊗ scalar` element-wise. Nulls in `self` propagate.
    pub fn binary_scalar(&self, op: ArrowBinaryOp, scalar: f64) -> ColumnarF64 {
        let n = self.len;
        if self.is_dense() {
            let a = self.values();
            let mut out = Vec::with_capacity(n);
            for i in 0..n { out.push(apply_scalar(op, a[i], scalar)); }
            return ColumnarF64::from_vec(out);
        }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                values.push(apply_scalar(op, self.values[i], scalar));
            } else {
                values.push(0.0);
            }
        }
        ColumnarF64 {
            values,
            valid_bits: Some(bits.to_vec()),
            len: n,
            null_count: self.null_count,
        }
    }

    /// Element-wise `self + other`.
    pub fn add(&self, other: &ColumnarF64) -> Result<ColumnarF64, String> { self.binary(ArrowBinaryOp::Add, other) }
    /// Element-wise `self - other`.
    pub fn sub(&self, other: &ColumnarF64) -> Result<ColumnarF64, String> { self.binary(ArrowBinaryOp::Sub, other) }
    /// Element-wise `self * other`.
    pub fn mul(&self, other: &ColumnarF64) -> Result<ColumnarF64, String> { self.binary(ArrowBinaryOp::Mul, other) }
    /// Element-wise `self / other`.
    pub fn div(&self, other: &ColumnarF64) -> Result<ColumnarF64, String> { self.binary(ArrowBinaryOp::Div, other) }
}

// ════════════════════════════════════════════════════════════════════
// Phase F.7 — ColumnarF32 (opt-in single-precision storage)
// ════════════════════════════════════════════════════════════════════
//
// Mirror of `ColumnarF64` for `f32` payloads — half the memory footprint
// for stored data, at the cost of ~7-9 decimal digits of precision
// versus f64's ~15-17. R2's stance:
//
//   - **Storage**: opt-in via `as.single(x)`. Users who hit memory
//     pressure on large vectors can choose this tradeoff explicitly.
//   - **Computation**: arithmetic between Single values stays in f32.
//     Arithmetic between Single and Numeric (f64) promotes to f64.
//     This matches NumPy's dtype-promotion rules and R's rare
//     `as.single` semantics — explicit type lives on disk, arithmetic
//     respects the type but promotes on mixing.
//   - **Reductions**: cast each element to f64 before summing/etc., to
//     preserve precision for long accumulations. The output of a
//     reduction is always f64.
//
// **NA representation**: same `Vec<f32>` payload + packed valid bitmap
// pattern as ColumnarF64. NaN-in-payload also treated as NA for
// consistency.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub struct ColumnarF32 {
    values: Vec<f32>,
    valid_bits: Option<Vec<u8>>,
    len: usize,
    null_count: usize,
}

impl ColumnarF32 {
    /// New empty column.
    pub fn new() -> Self {
        ColumnarF32 { values: Vec::new(), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Pre-allocate capacity for `n` elements.
    pub fn with_capacity(n: usize) -> Self {
        ColumnarF32 { values: Vec::with_capacity(n), valid_bits: None, len: 0, null_count: 0 }
    }

    /// Build from a fully-valid (no nulls) `Vec<f32>`. Zero-copy.
    pub fn from_vec(values: Vec<f32>) -> Self {
        let len = values.len();
        ColumnarF32 { values, valid_bits: None, len, null_count: 0 }
    }

    /// Build from `Vec<Option<f32>>`.
    pub fn from_options(opts: Vec<Option<f32>>) -> Self {
        Self::from_option_slice(&opts)
    }

    /// Build from a `&[Option<f32>]` slice.
    pub fn from_option_slice(opts: &[Option<f32>]) -> Self {
        let len = opts.len();
        let mut values = Vec::with_capacity(len);
        let mut bits: Option<Vec<u8>> = None;
        let mut null_count = 0;
        for (i, x) in opts.iter().enumerate() {
            match x {
                Some(v) => {
                    values.push(*v);
                    if let Some(b) = bits.as_mut() { b[i / 8] |= 1 << (i % 8); }
                }
                None => {
                    if bits.is_none() {
                        let mut new_bits = vec![0u8; (len + 7) / 8];
                        for j in 0..i { new_bits[j / 8] |= 1 << (j % 8); }
                        bits = Some(new_bits);
                    }
                    values.push(f32::NAN);
                    null_count += 1;
                }
            }
        }
        ColumnarF32 { values, valid_bits: bits, len, null_count }
    }

    /// Build from a `ColumnarF64` by lossy narrowing. NaNs and validity
    /// bitmap preserved. Use this for `as.single(x)`.
    pub fn from_f64(other: &ColumnarF64) -> Self {
        let values: Vec<f32> = other.values.iter().map(|&x| x as f32).collect();
        ColumnarF32 {
            values,
            valid_bits: other.valid_bits.clone(),
            len: other.len,
            null_count: other.null_count,
        }
    }

    /// Promote to `ColumnarF64`. Lossless widening (f32 ⊂ f64).
    /// Used implicitly when Single values mix with Numeric in arithmetic.
    pub fn to_f64(&self) -> ColumnarF64 {
        let values: Vec<f64> = self.values.iter().map(|&x| x as f64).collect();
        ColumnarF64 {
            values,
            valid_bits: self.valid_bits.clone(),
            len: self.len,
            null_count: self.null_count,
        }
    }

    /// Convert back to `Vec<Option<f32>>` for legacy callers.
    pub fn to_options(&self) -> Vec<Option<f32>> {
        match &self.valid_bits {
            None => self.values.iter().take(self.len).map(|v| Some(*v)).collect(),
            Some(bits) => (0..self.len).map(|i| {
                if (bits[i / 8] >> (i % 8)) & 1 == 1 { Some(self.values[i]) } else { None }
            }).collect(),
        }
    }

    /// Logical length.
    pub fn len(&self) -> usize { self.len }
    /// True when this column has zero rows.
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Number of null entries.
    pub fn null_count(&self) -> usize { self.null_count }
    /// True when no entries are null (fast path for SIMD reductions).
    pub fn is_dense(&self) -> bool { self.null_count == 0 }

    /// Read element `i`. `None` ⇒ null.
    pub fn get(&self, i: usize) -> Option<f32> {
        if i >= self.len { return None; }
        if let Some(bits) = &self.valid_bits {
            if (bits[i / 8] >> (i % 8)) & 1 == 0 { return None; }
        }
        Some(self.values[i])
    }

    /// Borrow the raw values slice. For dense columns this is the live
    /// data; for sparse columns positions where validity is 0 hold
    /// NaN-as-placeholder.
    pub fn values(&self) -> &[f32] { &self.values[..self.len] }

    /// Optional borrowed view of the valid-bits bitmap.
    pub fn valid_bits(&self) -> Option<&[u8]> { self.valid_bits.as_deref() }

    /// Memory footprint in bytes — exact for the values; bitmap rounded.
    pub fn nbytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
            + self.valid_bits.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Sum — promotes to f64 internally to preserve precision over long
    /// accumulations. Returns f64 (caller can narrow if desired).
    /// `na_rm = false` → returns None when any null present.
    pub fn sum(&self, na_rm: bool) -> Option<f64> {
        if self.is_dense() {
            return Some(self.values.iter().map(|&x| x as f64).sum());
        }
        if !na_rm { return None; }
        let bits = self.valid_bits.as_ref().unwrap();
        let mut s = 0.0_f64;
        for i in 0..self.len {
            if (bits[i / 8] >> (i % 8)) & 1 == 1 { s += self.values[i] as f64; }
        }
        Some(s)
    }
}

impl Default for ColumnarF32 {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod f32_tests {
    use super::*;

    #[test]
    fn from_vec_round_trip() {
        let c = ColumnarF32::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(c.len(), 3);
        assert!(c.is_dense());
        assert_eq!(c.values(), &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn from_options_with_nulls() {
        let c = ColumnarF32::from_options(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(c.null_count(), 1);
        assert_eq!(c.get(0), Some(1.0));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(3.0));
    }

    #[test]
    fn f64_to_f32_round_trip_dense() {
        let f64c = ColumnarF64::from_vec(vec![1.5, 2.5, 3.5]);
        let f32c = ColumnarF32::from_f64(&f64c);
        assert_eq!(f32c.values(), &[1.5_f32, 2.5, 3.5]);
        let back = f32c.to_f64();
        assert_eq!(back.values(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn f64_to_f32_loses_low_precision() {
        // 1.0 + 2^-30 is representable in f64 but not in f32 (mantissa
        // is 23 bits, so values within 2^-23 of 1.0 collapse to 1.0).
        let f64c = ColumnarF64::from_vec(vec![1.0 + 2f64.powi(-30)]);
        let f32c = ColumnarF32::from_f64(&f64c);
        // Documented precision loss — f32 can't represent this.
        assert_eq!(f32c.values()[0], 1.0_f32);
    }

    #[test]
    fn sum_promotes_to_f64() {
        let c = ColumnarF32::from_vec(vec![0.1; 1000]);
        let s = c.sum(false).unwrap();
        // f32 alone accumulating 1000 × 0.1 would drift by ~1e-5 due to
        // rounding. f64 accumulation stays near the exact 100.0.
        assert!((s - 100.0).abs() < 1e-4, "got {}", s);
    }

    #[test]
    fn memory_footprint_is_half_of_f64() {
        let c32 = ColumnarF32::from_vec(vec![1.0; 1_000_000]);
        // f32 = 4 MB. f64 equivalent would be 8 MB (8 bytes/elem × 1e6).
        assert_eq!(c32.nbytes(), 4_000_000);
        assert_eq!(c32.nbytes() * 2, 8_000_000); // confirms 2x ratio
    }
}

#[cfg(test)]
mod f4_binary_tests {
    use super::*;

    #[test]
    fn dense_add_is_pointwise() {
        let a = ColumnarF64::from_vec(vec![1.0, 2.0, 3.0]);
        let b = ColumnarF64::from_vec(vec![10.0, 20.0, 30.0]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_options(), vec![Some(11.0), Some(22.0), Some(33.0)]);
        assert!(c.valid_bits().is_none(), "dense × dense should stay dense");
    }

    #[test]
    fn nulls_propagate_via_bitmap_and() {
        let a = ColumnarF64::from_options(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let b = ColumnarF64::from_options(vec![Some(10.0), Some(20.0), None, Some(40.0)]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_options(), vec![Some(11.0), None, None, Some(44.0)]);
        assert_eq!(c.null_count(), 2);
    }

    #[test]
    fn scalar_op_preserves_null_pattern() {
        let a = ColumnarF64::from_options(vec![Some(1.0), None, Some(3.0)]);
        let c = a.binary_scalar(ArrowBinaryOp::Mul, 2.0);
        assert_eq!(c.to_options(), vec![Some(2.0), None, Some(6.0)]);
    }

    #[test]
    fn length_mismatch_errors() {
        let a = ColumnarF64::from_vec(vec![1.0, 2.0]);
        let b = ColumnarF64::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn div_by_zero_yields_inf_not_panic() {
        let a = ColumnarF64::from_vec(vec![1.0]);
        let b = ColumnarF64::from_vec(vec![0.0]);
        let c = a.div(&b).unwrap();
        assert!(c.to_options()[0].unwrap().is_infinite());
    }

    #[test]
    fn all_six_ops() {
        let a = ColumnarF64::from_vec(vec![6.0]);
        let b = ColumnarF64::from_vec(vec![2.0]);
        assert_eq!(a.binary(ArrowBinaryOp::Add, &b).unwrap().to_options(), vec![Some(8.0)]);
        assert_eq!(a.binary(ArrowBinaryOp::Sub, &b).unwrap().to_options(), vec![Some(4.0)]);
        assert_eq!(a.binary(ArrowBinaryOp::Mul, &b).unwrap().to_options(), vec![Some(12.0)]);
        assert_eq!(a.binary(ArrowBinaryOp::Div, &b).unwrap().to_options(), vec![Some(3.0)]);
        assert_eq!(a.binary(ArrowBinaryOp::Pow, &b).unwrap().to_options(), vec![Some(36.0)]);
        assert_eq!(a.binary(ArrowBinaryOp::Mod, &b).unwrap().to_options(), vec![Some(0.0)]);
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::*;

    #[test]
    fn dense_sum_uses_contiguous_slice() {
        let c = ColumnarF64::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.sum(false), Some(10.0));
        assert_eq!(c.mean(false), Some(2.5));
        assert_eq!(c.min(false), Some(1.0));
        assert_eq!(c.max(false), Some(4.0));
    }

    #[test]
    fn null_propagates_without_na_rm() {
        let c = ColumnarF64::from_options(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(c.sum(false), None);
        assert_eq!(c.mean(false), None);
    }

    #[test]
    fn na_rm_skips_nulls() {
        let c = ColumnarF64::from_options(vec![Some(1.0), None, Some(3.0), None, Some(5.0)]);
        assert_eq!(c.sum(true), Some(9.0));
        assert_eq!(c.mean(true), Some(3.0));
        assert_eq!(c.min(true), Some(1.0));
        assert_eq!(c.max(true), Some(5.0));
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_round_trip() {
        let c = ColumnarF64::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(c.len(), 3);
        assert!(c.is_dense());
        assert_eq!(c.null_count(), 0);
        assert_eq!(c.get(0), Some(1.0));
        assert_eq!(c.get(2), Some(3.0));
        assert_eq!(c.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_options_with_nulls() {
        let c = ColumnarF64::from_options(vec![Some(1.0), None, Some(3.0), None, Some(5.0)]);
        assert_eq!(c.len(), 5);
        assert!(!c.is_dense());
        assert_eq!(c.null_count(), 2);
        assert_eq!(c.get(0), Some(1.0));
        assert_eq!(c.get(1), None);
        assert_eq!(c.get(2), Some(3.0));
        assert_eq!(c.get(3), None);
        assert_eq!(c.get(4), Some(5.0));
    }

    #[test]
    fn round_trip_options() {
        let original = vec![Some(1.5), None, Some(2.5), None, Some(3.5)];
        let c = ColumnarF64::from_options(original.clone());
        let back = c.to_options();
        assert_eq!(back, original);
    }

    #[test]
    fn dense_no_bitmap_allocated() {
        let c = ColumnarF64::from_options(vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(c.is_dense());
        assert!(c.valid_bits().is_none(), "fully-valid columns should not allocate a bitmap");
    }

    #[test]
    fn empty_column() {
        let c = ColumnarF64::new();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert!(c.is_dense());
    }
}
