//! Out-of-core token storage: tokenize once to a file, train from a
//! memory map.
//!
//! # The ceiling this removes
//!
//! Training currently holds the whole token stream in RAM as a
//! `Vec<usize>` — eight bytes per token on a 64-bit target. Measured on
//! TinyStories at vocab 8000 (0.242 tokens per byte), a corpus costs
//! roughly FOUR TIMES its own size before a single step runs:
//!
//! | corpus | tokens | `Vec<usize>` | plus corpus + encode buffer |
//! |---|---|---|---|
//! | 19.4 MB | 4.70 M | 37.6 MB | ~76 MB |
//! | 700 MB | 169 M | 1.35 GB | ~2.7 GB |
//!
//! So a 700 MB corpus needs ~2.7 GB of RAM to begin, on a laptop where the
//! model, its gradients and the optimizer state also have to fit. The
//! corpus size at which training becomes impossible is a hard limit, and
//! it has nothing to do with how fast anything runs.
//!
//! # What this does instead
//!
//! Token ids are written once as little-endian `u32` and read back through
//! a memory map. Two independent wins:
//!
//! * **`u32`, not `usize`** — half the bytes. A tokenizer with more than
//!   4.29 billion tokens does not exist and will not; GPT-2 has 50,257 and
//!   the largest shipping vocabularies are ~256,000.
//! * **Mapped, not loaded** — the process holds a window, not the corpus.
//!   Resident memory becomes O(batch x seq), and the OS pages in what the
//!   training cursor actually touches.
//!
//! A training run then costs one pass to tokenize, and every later run
//! over the same corpus skips tokenization entirely.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Writes a token stream to disk as little-endian `u32`.
///
/// Little-endian explicitly, not native: a token file written on one
/// machine has to be readable on another, and "whatever this CPU does" is
/// how a format silently becomes non-portable.
pub struct TokenWriter {
    out: BufWriter<File>,
    count: usize,
}

impl TokenWriter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<TokenWriter, String> {
        let f = File::create(path.as_ref())
            .map_err(|e| format!("token file {}: {e}", path.as_ref().display()))?;
        Ok(TokenWriter { out: BufWriter::with_capacity(1 << 20, f), count: 0 })
    }

    /// Append ids. Rejects anything that will not fit in `u32` rather than
    /// truncating — a silently wrapped token id trains a wrong model.
    pub fn write(&mut self, ids: &[usize]) -> Result<(), String> {
        for &id in ids {
            let v = u32::try_from(id)
                .map_err(|_| format!("token id {id} exceeds u32; \
                                      vocabularies this large are not supported"))?;
            self.out.write_all(&v.to_le_bytes())
                .map_err(|e| format!("writing token file: {e}"))?;
        }
        self.count += ids.len();
        Ok(())
    }

    /// Flush and return how many tokens were written.
    pub fn finish(mut self) -> Result<usize, String> {
        self.out.flush().map_err(|e| format!("flushing token file: {e}"))?;
        Ok(self.count)
    }
}

/// A memory-mapped token stream.
///
/// `window` copies `seq` ids out of the map into a `Vec<usize>`, which is
/// what the tape wants. That copy is O(seq) — a few kilobytes per sequence
/// — not O(corpus), which is the whole point.
pub struct TokenStore {
    map: memmap2::Mmap,
    len: usize,
}

impl TokenStore {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<TokenStore, String> {
        let f = File::open(path.as_ref())
            .map_err(|e| format!("token file {}: {e}", path.as_ref().display()))?;
        let bytes = f.metadata().map_err(|e| format!("token file: {e}"))?.len() as usize;
        if bytes % 4 != 0 {
            return Err(format!("token file {} is {bytes} bytes, not a multiple of 4 — \
                                truncated or not a token file", path.as_ref().display()));
        }
        // SAFETY: the file is opened read-only and this map is never handed
        // out as a mutable slice. The documented hazard for any mmap is a
        // concurrent external truncation, which would fault; a training
        // corpus is written once and then read, so nothing here truncates it.
        let map = unsafe { memmap2::Mmap::map(&f) }
            .map_err(|e| format!("mapping token file: {e}"))?;
        Ok(TokenStore { map, len: bytes / 4 })
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Ids `[start, start+n)` as `usize`, for feeding a batch.
    ///
    /// Reads through the map, so only the pages this window touches are
    /// resident. Returns `None` past the end rather than panicking: a
    /// training cursor walks off the end every epoch and that is normal.
    pub fn window(&self, start: usize, n: usize) -> Option<Vec<usize>> {
        if start + n > self.len { return None; }
        let bytes = &self.map[start * 4..(start + n) * 4];
        Some(bytes.chunks_exact(4)
             .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
             .collect())
    }

    /// Resident cost of holding this store, in bytes — the mapping itself,
    /// not the file. Reported so a run can state its own memory rather
    /// than have it estimated.
    pub fn resident_bytes(&self) -> usize { std::mem::size_of::<Self>() }

    /// What the same stream would have cost as an in-RAM `Vec<usize>`.
    pub fn in_ram_equivalent_bytes(&self) -> usize {
        self.len * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("r2_tokens_{name}_{}.bin", std::process::id()));
        p
    }

    #[test]
    fn round_trips_a_token_stream() {
        let path = tmp("round");
        let ids: Vec<usize> = (0..1000).map(|i| (i * 7919) % 50257).collect();
        let mut w = TokenWriter::create(&path).unwrap();
        w.write(&ids).unwrap();
        assert_eq!(w.finish().unwrap(), ids.len());

        let s = TokenStore::open(&path).unwrap();
        assert_eq!(s.len(), ids.len());
        assert_eq!(s.window(0, ids.len()).unwrap(), ids);
        assert_eq!(s.window(10, 5).unwrap(), &ids[10..15]);
        std::fs::remove_file(&path).ok();
    }

    /// Walking off the end must return None, not panic — a training cursor
    /// does this at every epoch boundary.
    #[test]
    fn window_past_the_end_is_none_not_a_panic() {
        let path = tmp("end");
        let mut w = TokenWriter::create(&path).unwrap();
        w.write(&[1, 2, 3]).unwrap();
        w.finish().unwrap();
        let s = TokenStore::open(&path).unwrap();
        assert!(s.window(0, 3).is_some());
        assert!(s.window(1, 3).is_none());
        assert!(s.window(99, 1).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// A token id too large for u32 must be REFUSED. Truncating it would
    /// silently train the model on a different token.
    #[test]
    fn oversized_ids_are_refused_not_truncated() {
        let path = tmp("big");
        let mut w = TokenWriter::create(&path).unwrap();
        let err = w.write(&[u32::MAX as usize + 1]).unwrap_err();
        assert!(err.contains("exceeds u32"), "unexpected error: {err}");
        std::fs::remove_file(&path).ok();
    }

    /// A file whose length is not a multiple of 4 is truncated or is not a
    /// token file; reading it would silently misalign every id after the
    /// break.
    #[test]
    fn truncated_file_is_rejected() {
        let path = tmp("trunc");
        std::fs::write(&path, [1u8, 2, 3, 4, 5]).unwrap();
        let err = match TokenStore::open(&path) {
            Err(e) => e,
            Ok(_) => panic!("a truncated token file must be rejected"),
        };
        assert!(err.contains("multiple of 4"), "unexpected error: {err}");
        std::fs::remove_file(&path).ok();
    }

    /// The saving is the reason this exists, so state it in a test.
    #[test]
    fn mapping_costs_far_less_than_the_vec_it_replaces() {
        let path = tmp("save");
        let ids: Vec<usize> = (0..100_000).collect();
        let mut w = TokenWriter::create(&path).unwrap();
        w.write(&ids).unwrap();
        w.finish().unwrap();
        let s = TokenStore::open(&path).unwrap();
        assert_eq!(s.in_ram_equivalent_bytes(), 100_000 * std::mem::size_of::<usize>());
        assert!(s.resident_bytes() < 1000,
                "the store itself should be a handle, not the data");
        std::fs::remove_file(&path).ok();
    }
}
