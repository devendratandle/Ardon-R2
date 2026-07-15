//! Memory-budget manager (Pillar 2 foundation).
//!
//! Tracks the approximate live in-RAM columnar footprint and enforces an
//! optional budget by SPILLING cold numeric columns to disk (a packed-f64
//! mmap file) and handing back an `mmapcol` handle — which already
//! composes with sum/mean/var/quantile/mmap.map, so downstream compute is
//! unchanged. This is the accounting + policy core; auto-spill-on-assign
//! wiring is left to Opus (it touches the hot assignment path and wants a
//! separate, carefully-benchmarked pass).
//!
//! Design notes:
//! - The budget is a soft cap. Spilling is opportunistic and explicit in
//!   v1 (`mem.spill(x)`); the manager reports pressure via `mem.status()`.
//! - Bytes are ESTIMATES (8 × element count for numeric/int, 1 × for
//!   logical) — enough to drive an LRU decision, not an allocator.
//! - A spilled column is a real `mmapcol`; restoring it (`mem.restore`)
//!   reads it back into a dense numeric vector. No data is ever lost:
//!   spill writes before it drops the in-RAM copy.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Global live-bytes counter and budget. Process-wide (one engine per
/// process in server mode; the interactive CLI is single-engine too).
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static BUDGET_BYTES: AtomicU64 = AtomicU64::new(0); // 0 = unlimited
static SPILLS: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Parse a human budget string: "8G", "512M", "2048K", "1073741824",
/// case-insensitive, optional trailing "B". Returns bytes, or None.
pub fn parse_budget(s: &str) -> Option<u64> {
    let t = s.trim().trim_end_matches(['B', 'b']);
    if t.is_empty() { return None; }
    let (num, mult) = match t.chars().last().unwrap().to_ascii_uppercase() {
        'K' => (&t[..t.len() - 1], 1024u64),
        'M' => (&t[..t.len() - 1], 1024 * 1024),
        'G' => (&t[..t.len() - 1], 1024 * 1024 * 1024),
        'T' => (&t[..t.len() - 1], 1024u64 * 1024 * 1024 * 1024),
        _ => (t, 1),
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult as f64) as u64)
}

/// Set the budget (0 disables enforcement but keeps accounting).
pub fn set_budget(bytes: u64) {
    BUDGET_BYTES.store(bytes, Ordering::Relaxed);
    ENABLED.store(bytes > 0, Ordering::Relaxed);
}
pub fn budget() -> u64 { BUDGET_BYTES.load(Ordering::Relaxed) }
pub fn live_bytes() -> u64 { LIVE_BYTES.load(Ordering::Relaxed) }
pub fn spill_count() -> u64 { SPILLS.load(Ordering::Relaxed) }
pub fn enabled() -> bool { ENABLED.load(Ordering::Relaxed) }

/// Record an allocation / deallocation of `bytes` live columnar memory.
/// Saturating so a mis-estimate can never underflow the counter.
pub fn add(bytes: u64) { LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed); }
pub fn sub(bytes: u64) {
    let mut cur = LIVE_BYTES.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(bytes);
        match LIVE_BYTES.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(c) => cur = c,
        }
    }
}
pub fn note_spill() { SPILLS.fetch_add(1, Ordering::Relaxed); }

/// True when live usage is over the budget (and a budget is set).
pub fn over_budget() -> bool {
    enabled() && live_bytes() > budget()
}

/// Approximate live bytes of an RVal's columnar payload (0 for scalars,
/// strings, and structures we don't spill).
pub fn estimate_bytes(v: &r2_types::RVal) -> u64 {
    use r2_types::RVal::*;
    match v {
        Numeric(d, _) => (d.len_fast() as u64) * 8,
        Integer(d, _) => (d.len_fast() as u64) * 4,
        Logical(d, _) => d.as_vec().len() as u64,
        Matrix(m) => (m.data.len() as u64) * 8,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_budget_units() {
        assert_eq!(parse_budget("1024"), Some(1024));
        assert_eq!(parse_budget("1K"), Some(1024));
        assert_eq!(parse_budget("8G"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_budget("512M"), Some(512 * 1024 * 1024));
        assert_eq!(parse_budget("2GB"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_budget(""), None);
        assert_eq!(parse_budget("xyz"), None);
    }
    #[test]
    fn accounting_saturates() {
        let base = live_bytes();
        add(1000); assert_eq!(live_bytes(), base + 1000);
        sub(10_000); // over-subtract
        assert!(live_bytes() <= base); // never underflows below what we added
    }
}
