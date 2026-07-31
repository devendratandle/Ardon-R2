//! safetensors weight loading — read a real checkpoint.
//!
//! safetensors is the format HuggingFace ships model weights in, and it is
//! deliberately trivial to read: an 8-byte little-endian header length, a
//! JSON header describing every tensor (dtype, shape, byte range), then the
//! raw tensor bytes. No pickle, no code execution — which is the whole
//! reason it replaced `.bin`, and why it is the right first format for a
//! project whose thesis is safety.
//!
//! The file is **memory-mapped**: tensor bytes are paged in by the OS on
//! demand rather than read into a buffer, so opening a 60 GB checkpoint
//! costs almost nothing and only the tensors actually touched occupy RAM.
//! Dequantization to `f32` happens per tensor, when requested.
//!
//! Scope: reading. F32 / F16 / BF16 / I8 / I32 / I64 / U8 / BOOL are
//! understood; writing is not implemented (training checkpoints are a
//! separate concern from serving a downloaded model).

use std::collections::BTreeMap;
use std::path::Path;

use crate::dtype::{bf16_to_f32, f16_to_f32};
use crate::json::Json;
use crate::MmapWeights;

/// Element type as named in a safetensors header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype { F32, F16, BF16, I8, U8, I32, I64, Bool }

impl Dtype {
    pub fn parse(s: &str) -> Result<Dtype, String> {
        Ok(match s {
            "F32"  => Dtype::F32,
            "F16"  => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I8"   => Dtype::I8,
            "U8"   => Dtype::U8,
            "I32"  => Dtype::I32,
            "I64"  => Dtype::I64,
            "BOOL" => Dtype::Bool,
            // F64/U16/U32/U64 exist in the spec but no LLM checkpoint uses
            // them for weights; refuse rather than guess.
            other  => return Err(format!("safetensors: unsupported dtype '{}'", other)),
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }
}

/// Where one tensor lives in the file and what shape it is.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the data section (after the header).
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    /// Element count implied by the shape.
    pub fn numel(&self) -> usize { self.shape.iter().product() }
}

/// An opened safetensors file: the tensor index plus the mapped bytes.
pub struct SafeTensors {
    // (MmapWeights is not Debug; the index fields below are what matter
    // for diagnostics, so Debug is derived over them via a manual impl.)
    tensors: BTreeMap<String, TensorInfo>,
    /// Free-form `__metadata__` strings from the header.
    metadata: BTreeMap<String, String>,
    map: MmapWeights,
    /// Offset of the data section (8 + header length).
    data_start: usize,
}

impl SafeTensors {
    /// Open a checkpoint and parse its header. The tensor payload is NOT
    /// read here — it is mapped, so this is fast even for a 60 GB file.
    pub fn open(path: impl AsRef<Path>) -> Result<SafeTensors, String> {
        let map = MmapWeights::open(path)
            .map_err(|e| format!("safetensors: cannot open file: {}", e))?;
        let bytes = map.bytes();
        if bytes.len() < 8 {
            return Err("safetensors: file shorter than its 8-byte length prefix".into());
        }
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_start = 8usize.checked_add(hdr_len)
            .ok_or("safetensors: header length overflows")?;
        if data_start > bytes.len() {
            return Err(format!(
                "safetensors: header claims {} bytes but file holds {}",
                hdr_len, bytes.len().saturating_sub(8)));
        }
        let hdr = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|_| "safetensors: header is not valid UTF-8".to_string())?;
        let json = Json::parse(hdr).map_err(|e| format!("safetensors: {}", e))?;
        let obj = json.as_obj().ok_or("safetensors: header is not a JSON object")?;

        let data_len = bytes.len() - data_start;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();

        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_obj() {
                    for (k, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = Dtype::parse(v.get("dtype").and_then(|d| d.as_str())
                .ok_or_else(|| format!("safetensors: '{}' has no dtype", name))?)?;
            let shape = v.get("shape").and_then(|s| s.as_usize_vec())
                .ok_or_else(|| format!("safetensors: '{}' has no valid shape", name))?;
            let off = v.get("data_offsets").and_then(|o| o.as_usize_vec())
                .ok_or_else(|| format!("safetensors: '{}' has no data_offsets", name))?;
            if off.len() != 2 {
                return Err(format!("safetensors: '{}' data_offsets must have 2 entries", name));
            }
            let (start, end) = (off[0], off[1]);
            // Validate NOW, not at read time: a truncated or overlapping
            // file should fail on open with the offending tensor named,
            // not produce silent garbage deep inside a forward pass.
            if end < start || end > data_len {
                return Err(format!(
                    "safetensors: '{}' range {}..{} outside data section of {} bytes",
                    name, start, end, data_len));
            }
            let expect = shape.iter().product::<usize>() * dtype.size();
            if end - start != expect {
                return Err(format!(
                    "safetensors: '{}' shape {:?} as {:?} needs {} bytes, header gives {}",
                    name, shape, dtype, expect, end - start));
            }
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start, end });
        }

        Ok(SafeTensors { tensors, metadata, map, data_start })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> Vec<&str> { self.tensors.keys().map(|s| s.as_str()).collect() }
    pub fn len(&self) -> usize { self.tensors.len() }
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> { self.tensors.get(name) }
    pub fn metadata(&self) -> &BTreeMap<String, String> { &self.metadata }

    /// Total elements across every tensor — the checkpoint's parameter
    /// count, which should match the model config's `n_params`.
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
    }

    /// Raw bytes of one tensor (a slice of the mapping — no copy).
    pub fn raw(&self, name: &str) -> Result<&[u8], String> {
        let t = self.tensors.get(name)
            .ok_or_else(|| format!("safetensors: no tensor named '{}'", name))?;
        Ok(&self.map.bytes()[self.data_start + t.start..self.data_start + t.end])
    }

    /// Read one tensor as `f32`, converting from whatever it is stored as.
    /// This is the only place a copy happens, and only for the tensors a
    /// caller actually asks for.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensors.get(name)
            .ok_or_else(|| format!("safetensors: no tensor named '{}'", name))?;
        let b = self.raw(name)?;
        let n = t.numel();
        let mut out = Vec::with_capacity(n);
        match t.dtype {
            Dtype::F32 => for c in b.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            },
            Dtype::F16 => for c in b.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            },
            Dtype::BF16 => for c in b.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            },
            Dtype::I8   => for &x in b { out.push(x as i8 as f32); },
            Dtype::U8   => for &x in b { out.push(x as f32); },
            Dtype::Bool => for &x in b { out.push(if x != 0 { 1.0 } else { 0.0 }); },
            Dtype::I32  => for c in b.chunks_exact(4) {
                out.push(i32::from_le_bytes(c.try_into().unwrap()) as f32);
            },
            Dtype::I64  => for c in b.chunks_exact(8) {
                out.push(i64::from_le_bytes(c.try_into().unwrap()) as f32);
            },
        }
        debug_assert_eq!(out.len(), n);
        Ok(out)
    }

    /// Read a tensor and require an exact shape — the check that turns a
    /// silently-wrong load into an error naming the tensor.
    pub fn tensor_f32_shaped(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.info(name)
            .ok_or_else(|| format!("safetensors: no tensor named '{}'", name))?;
        if t.shape != shape {
            return Err(format!("safetensors: '{}' has shape {:?}, expected {:?}",
                               name, t.shape, shape));
        }
        self.tensor_f32(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a valid safetensors file in a temp dir. Writing the format by
    /// hand in the test is deliberate: it pins the reader to the SPEC, not
    /// to a writer we also wrote.
    fn write_file(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("r2st-{}-{}.safetensors", name, std::process::id()));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        p
    }

    #[test]
    fn reads_f32_f16_and_bf16_tensors() {
        // a: F32 [2] = 1.5, -2.25 | b: F16 [2] = 1.0, 2.0 | c: BF16 [1] = 1.0
        let mut data = Vec::new();
        data.extend_from_slice(&1.5f32.to_le_bytes());
        data.extend_from_slice(&(-2.25f32).to_le_bytes());
        data.extend_from_slice(&0x3C00u16.to_le_bytes()); // f16 1.0
        data.extend_from_slice(&0x4000u16.to_le_bytes()); // f16 2.0
        data.extend_from_slice(&0x3F80u16.to_le_bytes()); // bf16 1.0
        let header = r#"{"__metadata__":{"format":"pt"},
            "a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "b":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},
            "c":{"dtype":"BF16","shape":[1],"data_offsets":[12,14]}}"#;
        let p = write_file("mixed", header, &data);
        let st = SafeTensors::open(&p).unwrap();

        assert_eq!(st.len(), 3);
        assert_eq!(st.names(), vec!["a", "b", "c"]);
        assert_eq!(st.metadata().get("format").map(|s| s.as_str()), Some("pt"));
        assert_eq!(st.tensor_f32("a").unwrap(), vec![1.5, -2.25]);
        assert_eq!(st.tensor_f32("b").unwrap(), vec![1.0, 2.0]);
        assert_eq!(st.tensor_f32("c").unwrap(), vec![1.0]);
        assert_eq!(st.total_params(), 5);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn shape_and_offsets_are_validated_on_open() {
        // Header claims 4 elements but only provides bytes for 2.
        let data = vec![0u8; 8];
        let bad = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let p = write_file("badshape", bad, &data);
        let err = SafeTensors::open(&p).unwrap_err();
        assert!(err.contains("needs 16 bytes"), "got: {err}");
        let _ = std::fs::remove_file(p);

        // Range past the end of the data section.
        let past = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let p2 = write_file("past", past, &[0u8; 4]);
        assert!(SafeTensors::open(&p2).unwrap_err().contains("outside data section"));
        let _ = std::fs::remove_file(p2);
    }

    #[test]
    fn shaped_read_rejects_a_mismatch() {
        let mut data = Vec::new();
        for i in 0..6 { data.extend_from_slice(&(i as f32).to_le_bytes()); }
        let header = r#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let p = write_file("shaped", header, &data);
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.tensor_f32_shaped("w", &[2, 3]).unwrap().len(), 6);
        let err = st.tensor_f32_shaped("w", &[3, 2]).unwrap_err();
        assert!(err.contains("expected [3, 2]"), "got: {err}");
        assert!(st.tensor_f32("nope").unwrap_err().contains("no tensor named"));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn corrupt_files_error_instead_of_panicking() {
        // Truncated length prefix.
        let p = std::env::temp_dir().join(format!("r2st-tiny-{}.safetensors", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert!(SafeTensors::open(&p).unwrap_err().contains("shorter than"));
        let _ = std::fs::remove_file(&p);

        // Header length longer than the file.
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&9999u64.to_le_bytes()).unwrap();
        f.write_all(b"{}").unwrap();
        drop(f);
        assert!(SafeTensors::open(&p).unwrap_err().contains("header claims"));
        let _ = std::fs::remove_file(&p);

        // Not JSON.
        let p2 = write_file("notjson", "not json at all", &[]);
        assert!(SafeTensors::open(&p2).is_err());
        let _ = std::fs::remove_file(p2);

        // Unsupported dtype named explicitly.
        let p3 = write_file("f64", r#"{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#, &[0u8; 8]);
        assert!(SafeTensors::open(&p3).unwrap_err().contains("unsupported dtype"));
        let _ = std::fs::remove_file(p3);
    }
}

/// Write named f32 tensors as a safetensors file.
///
/// R2 uses this as its OWN model format, not as an interop concession:
/// the container is open, trivially parseable, executes no code on load,
/// and — the practical reason — is **memory-mappable**, so a 30 GB model
/// can be served without ever reading it into RAM. Any tool that reads
/// safetensors can read an R2 model, which is what makes "download a
/// model and run it" free of lock-in.
///
/// Writes atomically (temp file → fsync → rename), so an interrupted save
/// cannot leave a half-written model that loads as garbage.
pub fn save(path: impl AsRef<Path>, tensors: &[(String, Vec<usize>, Vec<f32>)])
    -> Result<(), String>
{
    use std::io::Write;

    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(format!(
                "safetensors::save: '{}' shape {:?} implies {} values, got {}",
                name, shape, n, data.len()));
        }
        if i > 0 { header.push(','); }
        let bytes = data.len() * 4;
        header.push_str(&format!(
            r#""{}":{{"dtype":"F32","shape":{:?},"data_offsets":[{},{}]}}"#,
            name, shape, offset, offset + bytes));
        offset += bytes;
    }
    header.push('}');

    let path = path.as_ref();
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("safetensors::save: {}", e))?;
        }
    }
    let tmp = path.with_extension("tmp");
    {
        let f = std::fs::File::create(&tmp)
            .map_err(|e| format!("safetensors::save: create: {}", e))?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|_| w.write_all(header.as_bytes()))
            .map_err(|e| format!("safetensors::save: header: {}", e))?;
        for (_, _, data) in tensors {
            for x in data {
                w.write_all(&x.to_le_bytes())
                    .map_err(|e| format!("safetensors::save: data: {}", e))?;
            }
        }
        let f = w.into_inner().map_err(|e| format!("safetensors::save: flush: {}", e))?;
        f.sync_all().map_err(|e| format!("safetensors::save: sync: {}", e))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("safetensors::save: commit: {}", e))
}

impl std::fmt::Debug for SafeTensors {
    /// Show the index, not the mapping — the tensor list is what a
    /// diagnostic needs, and the payload could be gigabytes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensors")
            .field("tensors", &self.tensors.len())
            .field("total_params", &self.total_params())
            .field("metadata", &self.metadata)
            .finish()
    }
}
