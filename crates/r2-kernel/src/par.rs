//! Parallel-for kernel — Phase K.4.

use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════
// Parallel-for kernel — Phase K.4
// ════════════════════════════════════════════════════════════════════
//
// Backend-dispatched parallel-for-each. Caller passes the work `kind`
// (used by Oracle to pick the threshold) and a closure indexed by `i`;
// kernel runs the closure for each `i` in `0..n` and collects results.
// Builtins that previously called `(0..n).into_par_iter().map(...)`
// directly now call `par_for(kind, n, f)` — Rayon stays below this
// layer (§4.9 locked decision).
//
// SAFETY: closures must be `Send + Sync`. Output type must be `Send`.
// Order of result indexing is preserved (Rayon's collect is stable).

/// Parallel-for-each. Backend chosen by Oracle. Result is a `Vec<T>`
/// indexed by `0..n`.
pub fn par_for<T, F>(kind: Op, n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Send + Sync,
    T: Send,
{
    match r2_oracle::dispatch(kind, Shape::n(n)) {
        Backend::Serial => (0..n).map(f).collect(),
        // The Oracle only returns Gpu for matmul, never for these ops;

        // if that ever changes, parallel CPU is the safe equivalent.

        Backend::Gpu | Backend::Rayon => (0..n).into_par_iter().map(f).collect(),
    }
}

/// Force parallel iteration without consulting Oracle. Used by callers
/// that have already made the dispatch decision themselves (e.g. the
/// list-aware apply path that computes aggregate work across
/// heterogeneous components and decides via `Op::ListMap` separately).
pub fn par_for_rayon<T, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Send + Sync,
    T: Send,
{
    (0..n).into_par_iter().map(f).collect()
}
