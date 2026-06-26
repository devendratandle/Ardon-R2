//! Memory-mapped columnar reader/writer (Phase F.5).

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

        /// Product of all values (8 independent accumulators, like `sum`).
        pub fn prod(&self) -> f64 {
            let s = self.as_slice();
            let mut acc = [1.0f64; 8];
            let mut it = s.chunks_exact(8);
            for c in &mut it {
                acc[0] *= c[0]; acc[1] *= c[1]; acc[2] *= c[2]; acc[3] *= c[3];
                acc[4] *= c[4]; acc[5] *= c[5]; acc[6] *= c[6]; acc[7] *= c[7];
            }
            let mut p = ((acc[0] * acc[1]) * (acc[2] * acc[3]))
                * ((acc[4] * acc[5]) * (acc[6] * acc[7]));
            for &v in it.remainder() { p *= v; }
            p
        }

        /// `(min, max)` in a single sweep (NaN-skipping via `f64::min`/`max`).
        pub fn range(&self) -> (f64, f64) {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for &v in self.as_slice() { mn = mn.min(v); mx = mx.max(v); }
            (mn, mx)
        }

        /// Sample/population variance with `ddof` delta degrees of freedom
        /// (`ddof=1` → R's `var()`; `ddof=0` → population). Two-pass for
        /// numerical fidelity with R: pass 1 is `mean()` (the 8-accumulator
        /// sum), pass 2 is the centered sum of squares with 8 accumulators.
        /// Two sweeps over the mmap means ~2× disk read for a >RAM file —
        /// the price of matching R's stable two-pass result. Returns NaN
        /// when `n <= ddof`.
        pub fn var(&self, ddof: usize) -> f64 {
            let s = self.as_slice();
            let n = s.len();
            if n <= ddof { return f64::NAN; }
            let mean = self.mean();
            let mut acc = [0.0f64; 8];
            let mut it = s.chunks_exact(8);
            for c in &mut it {
                let d0 = c[0] - mean; let d1 = c[1] - mean;
                let d2 = c[2] - mean; let d3 = c[3] - mean;
                let d4 = c[4] - mean; let d5 = c[5] - mean;
                let d6 = c[6] - mean; let d7 = c[7] - mean;
                acc[0] += d0 * d0; acc[1] += d1 * d1;
                acc[2] += d2 * d2; acc[3] += d3 * d3;
                acc[4] += d4 * d4; acc[5] += d5 * d5;
                acc[6] += d6 * d6; acc[7] += d7 * d7;
            }
            let mut ss = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for &v in it.remainder() { let d = v - mean; ss += d * d; }
            ss / (n - ddof) as f64
        }

        /// Standard deviation = `sqrt(var(ddof))`.
        pub fn sd(&self, ddof: usize) -> f64 { self.var(ddof).sqrt() }

        /// Approximate quantiles via a two-pass histogram — true
        /// out-of-core (bounded RAM = `bins` counters, independent of n).
        /// Pass 1 = `range()` (min/max); pass 2 bins values into `bins`
        /// buckets; each probability is located by cumulative count with
        /// linear interpolation inside its bucket. `p=0`→min, `p=1`→max are
        /// exact; interior quantiles are accurate to ≈ (max−min)/bins.
        /// NaN (NA) values are skipped. Returns one value per probability.
        pub fn quantile_hist(&self, probs: &[f64], bins: usize) -> Vec<f64> {
            let s = self.as_slice();
            if s.is_empty() { return vec![f64::NAN; probs.len()]; }
            let (mn, mx) = self.range();
            if !(mx > mn) {
                // Constant (or single) column — every quantile is that value.
                return probs.iter().map(|_| mn).collect();
            }
            let bins = bins.max(1);
            let width = (mx - mn) / bins as f64;
            let mut hist = vec![0u64; bins];
            for &v in s {
                if v.is_nan() { continue; }
                let mut b = ((v - mn) / width) as usize;
                if b >= bins { b = bins - 1; }
                hist[b] += 1;
            }
            let total: u64 = hist.iter().sum();
            probs.iter().map(|&p| {
                let p = p.clamp(0.0, 1.0);
                if p <= 0.0 { return mn; }
                if p >= 1.0 { return mx; }
                let target = p * total as f64;
                let mut cum = 0u64;
                let mut q = mx;
                for b in 0..bins {
                    let next = cum + hist[b];
                    if next as f64 >= target {
                        let bin_lo = mn + b as f64 * width;
                        let within = if hist[b] > 0 {
                            (target - cum as f64) / hist[b] as f64
                        } else { 0.0 };
                        q = bin_lo + within * width;
                        break;
                    }
                    cum = next;
                }
                q
            }).collect()
        }

        /// Out-of-core scalar map: apply `f` to every element and stream
        /// the result to a new packed-f64 file at `path`. Input is paged
        /// by the OS; output is written through a small fixed-size buffer,
        /// so peak RSS stays bounded regardless of column size (>RAM in →
        /// >RAM out). Returns the number of elements written.
        pub fn map_to<P: AsRef<Path>, F: Fn(f64) -> f64>(
            &self, path: P, f: F,
        ) -> Result<usize, String> {
            const CHUNK: usize = 1 << 16; // 65_536 f64 = 512 KiB out-buffer
            let s = self.as_slice();
            let mut w = MmapWriter::create(path)?;
            let mut buf: Vec<f64> = Vec::with_capacity(CHUNK);
            for block in s.chunks(CHUNK) {
                buf.clear();
                buf.extend(block.iter().map(|&x| f(x)));
                w.append(&buf)?;
            }
            w.finish()
        }

        /// Copy into an owned `ColumnarF64`. Useful when bridging into
        /// the existing RVal::Numeric storage path — pays a one-time
        /// allocation for full ownership.
        pub fn to_columnar(&self) -> super::ColumnarF64 {
            super::ColumnarF64::from_vec(self.as_slice().to_vec())
        }
    }

    /// Streaming/chunked writer for a packed-f64 file. Lets a
    /// larger-than-RAM column be *built* block by block without ever
    /// holding the whole thing in memory — the inverse capability of
    /// `MmapColumnar` (which reads >RAM), closing the out-of-core loop.
    ///
    /// Bytes go through a `BufWriter`, so many small `append` calls are
    /// coalesced into large sequential writes. Call `finish()` to flush;
    /// dropping without `finish()` still flushes via `BufWriter`'s Drop,
    /// but `finish()` surfaces any final I/O error.
    pub struct MmapWriter {
        w: std::io::BufWriter<std::fs::File>,
        count: usize,
    }

    impl MmapWriter {
        /// Create (or truncate) the file at `path` for streaming writes.
        pub fn create<P: AsRef<Path>>(path: P) -> Result<MmapWriter, String> {
            let f = std::fs::File::create(&path).map_err(|e| {
                format!("MmapWriter::create: cannot create '{}': {}",
                    path.as_ref().display(), e)
            })?;
            Ok(MmapWriter { w: std::io::BufWriter::with_capacity(1 << 20, f), count: 0 })
        }

        /// Append a block of values. Their packed little-/native-endian
        /// f64 bytes are appended verbatim (same layout `MmapColumnar`
        /// reads back).
        pub fn append(&mut self, vals: &[f64]) -> Result<(), String> {
            use std::io::Write;
            // SAFETY: f64 is Copy with a well-defined byte representation.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 8)
            };
            self.w.write_all(bytes)
                .map_err(|e| format!("MmapWriter::append: {}", e))?;
            self.count += vals.len();
            Ok(())
        }

        /// Number of f64 elements appended so far.
        pub fn len(&self) -> usize { self.count }
        /// True if nothing has been appended.
        pub fn is_empty(&self) -> bool { self.count == 0 }

        /// Flush and finish; returns the total element count written.
        pub fn finish(mut self) -> Result<usize, String> {
            use std::io::Write;
            self.w.flush().map_err(|e| format!("MmapWriter::finish: {}", e))?;
            Ok(self.count)
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
