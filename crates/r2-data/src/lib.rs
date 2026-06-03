//! R2 Data — domain crate for data-shaping builtins (Phase R.2).
//!
//! Per docs/ARCHITECTURE.md §5 Phase R, this crate hosts:
//!   - bind operations (cbind, rbind)
//!   - column ops (select, mutate, arrange)
//!   - row ops (filter, slice)
//!   - frame summarisation (summary)
//!   - apply family (apply, lapply, sapply, ...)
//!
//! All builtins follow the locked pattern: pure
//! `fn(&[EvalArg]) -> Result<RVal, R2Err>` signature; engine wraps via
//! 1-line adapter to its `BuiltinFn`. No engine reference needed.
//!
//! Phase R.2 spine: cbind + rbind + their shared `coerce_to_columns`
//! helper. Other domain functions migrate in subsequent sessions.

// Routed output macros — see r2_types::out. Send formatted output to the
// GUI/CLI sink instead of raw stdout (a windowed GUI has no console).
macro_rules! soutln {
    () => { $crate::__rout("\n") };
    ($($arg:tt)*) => { $crate::__rout(&format!("{}\n", format_args!($($arg)*))) };
}
#[allow(unused_macros)]
macro_rules! sout {
    ($($arg:tt)*) => { $crate::__rout(&format!("{}", format_args!($($arg)*))) };
}
#[doc(hidden)]
pub fn __rout(s: &str) { r2_types::out::rout(s); }

pub mod bind;
pub mod concat;
pub mod dplyr;
pub mod summary;
pub mod apply;
pub mod table;
pub mod meta;
pub mod clean;
pub mod order;

use r2_types::{EvalArg, R2Err, RVal};

/// Returns this crate's exported builtins as `(name, fn-pointer)` pairs.
/// r2-engine adapts the signature at registration time.
pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("cbind",   bind::bi_cbind),
        ("rbind",   bind::bi_rbind),
        ("c",       concat::bi_c),
        ("filter",  dplyr::bi_filter),
        ("select",  dplyr::bi_select),
        ("arrange", dplyr::bi_arrange),
        ("mutate",  dplyr::bi_mutate),
    ]
}
