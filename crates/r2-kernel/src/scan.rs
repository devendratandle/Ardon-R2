//! Scan / cumulative kernel — Phase K.7.

use crate::{SerialBackend, RayonBackend};
use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════
// Phase K.7 — Scan / cumulative operations
// ════════════════════════════════════════════════════════════════════
//
// Element-wise prefix-style reductions: each output position holds the
// reduction of all input positions up to and including it. Common in
// stats (running totals, density integration, cumulative regression).
//
//   cumsum:  out[i] = sum(in[0..=i])
//   cumprod: out[i] = prod(in[0..=i])
//   cummax:  out[i] = max(in[0..=i])
//   cummin:  out[i] = min(in[0..=i])
//
// NA propagation: once a None is seen, every subsequent output is None
// (matches R's `cumsum(c(1, NA, 3))` → `c(1, NA, NA)`).
//
// Serial: trivial O(n) loop.
// Rayon: two-pass Blelloch-style parallel scan. For workloads ≥ ~10K
//   the parallel pass amortises the chunk-merge overhead.

/// Cumulative / scan operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOp {
    /// `out[i] = sum(in[0..=i])`
    Cumsum,
    /// `out[i] = prod(in[0..=i])`
    Cumprod,
    /// `out[i] = max(in[0..=i])`
    Cummax,
    /// `out[i] = min(in[0..=i])`
    Cummin,
}

#[inline]
fn scan_identity(op: ScanOp) -> f64 {
    match op {
        ScanOp::Cumsum => 0.0,
        ScanOp::Cumprod => 1.0,
        ScanOp::Cummax => f64::NEG_INFINITY,
        ScanOp::Cummin => f64::INFINITY,
    }
}

#[inline]
fn scan_combine(op: ScanOp, acc: f64, v: f64) -> f64 {
    match op {
        ScanOp::Cumsum => acc + v,
        ScanOp::Cumprod => acc * v,
        ScanOp::Cummax => acc.max(v),
        ScanOp::Cummin => acc.min(v),
    }
}

pub trait ScanBackend: Send + Sync {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl ScanBackend for SerialBackend {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = scan_identity(op);
        let mut hit_na = false;
        for x in data {
            if hit_na { out.push(None); continue; }
            match x {
                Some(v) => {
                    acc = scan_combine(op, acc, *v);
                    out.push(Some(acc));
                }
                None => { hit_na = true; out.push(None); }
            }
        }
        out
    }
}

impl ScanBackend for RayonBackend {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        // Two-pass parallel scan:
        //   Pass 1: split into chunks, reduce each chunk in parallel.
        //   Sequential merge: prefix-combine chunk totals.
        //   Pass 2: re-scan each chunk in parallel, seeded with its prefix.
        //
        // NA handling: a None in chunk k poisons everything from that
        // position onwards. We track per-chunk "first NA index" so the
        // pass-2 scan emits None for the rest of that chunk and all
        // subsequent chunks.
        let n = data.len();
        if n == 0 { return Vec::new(); }
        // For small inputs the parallel overhead loses — fall through to serial.
        if n < 4096 { return SerialBackend.scan(op, data); }

        let n_chunks = num_chunks(n);
        let chunk_size = n.div_ceil(n_chunks);

        // Pass 1: per-chunk reduction + first-NA index.
        #[derive(Clone, Copy)]
        struct ChunkInfo {
            total: f64,
            first_na: Option<usize>, // relative to chunk start
        }
        let infos: Vec<ChunkInfo> = (0..n_chunks).into_par_iter().map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut acc = scan_identity(op);
            let mut first_na = None;
            for (i, v) in data[start..end].iter().enumerate() {
                match v {
                    Some(x) => { acc = scan_combine(op, acc, *x); }
                    None => { first_na = Some(i); break; }
                }
            }
            ChunkInfo { total: acc, first_na }
        }).collect();

        // Sequential prefix combine over chunk totals — gives the
        // "scan up to but not including chunk c" seed.
        let mut prefixes = Vec::with_capacity(n_chunks);
        let mut acc = scan_identity(op);
        let mut prefix_na_at = None;
        for (c, info) in infos.iter().enumerate() {
            prefixes.push(acc);
            if prefix_na_at.is_none() && info.first_na.is_some() {
                prefix_na_at = Some(c);
            }
            if prefix_na_at.is_some() {
                // Once any chunk has NA, all subsequent prefixes are
                // "NA from offset 0" — we represent this with a poison.
                acc = scan_identity(op); // doesn't matter; will be masked
            } else {
                acc = scan_combine(op, acc, info.total);
            }
        }

        // Pass 2: per-chunk re-scan seeded with the chunk's prefix.
        let chunks: Vec<Vec<Option<f64>>> = (0..n_chunks).into_par_iter().map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let seed = prefixes[c];
            // If any prior chunk had NA, the whole of this chunk is None.
            let chunk_poisoned = prefix_na_at.map_or(false, |na_c| c > na_c);
            let mut out = Vec::with_capacity(end - start);
            let mut acc = seed;
            let mut hit_na = chunk_poisoned;
            for (i, v) in data[start..end].iter().enumerate() {
                if hit_na { out.push(None); continue; }
                match v {
                    Some(x) => {
                        acc = scan_combine(op, acc, *x);
                        out.push(Some(acc));
                    }
                    None => { hit_na = true; out.push(None); let _ = i; }
                }
            }
            out
        }).collect();

        chunks.into_iter().flatten().collect()
    }
}

/// Compute a reasonable chunk count: roughly one chunk per core, but
/// bounded so chunks aren't pointless-tiny.
#[inline]
fn num_chunks(n: usize) -> usize {
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    let min_chunk = 1024;
    let max_chunks = (n + min_chunk - 1) / min_chunk;
    cores.min(max_chunks).max(1)
}

/// Public scan dispatcher. Oracle decides Serial vs Rayon based on
/// input length (`Op::Reduction` threshold reused — scan has similar
/// memory-bandwidth profile to reduction).
pub fn scan(op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.scan(op, data),
        Backend::Rayon => RayonBackend.scan(op, data),
    }
}
