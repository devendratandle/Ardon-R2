//! R2 Machine Learning — domain crate (Phase R.1).
//!
//! Per docs/ARCHITECTURE.md §5 Phase R:
//!   - Each domain crate hosts its own `bi_*` functions and exposes
//!     `register_builtins() -> Vec<(name, fn)>` for r2-engine to consume.
//!   - Math lives here; engine becomes a thin orchestrator.
//!
//! ## Status: scaffold + ABI established
//!
//! This crate currently exports an empty registration list. The eight ML
//! builtins (`rf`, `gbm`, `kmeans`, `rpart`, `knn`, `naive.bayes`, `cv`,
//! `prcomp`) cannot be moved here as drop-ins yet — they're tangled with
//! engine-private helpers:
//!
//!   - `extract_ml_data(e, args)`   — formula+data → (y, matrix, names)
//!   - `build_tree(...)`            — recursive CART builder
//!   - `tree_predict_one(...)`      — single-row tree prediction
//!   - `parallel_random(seed)`      — thread-local RNG
//!   - `count_splits(node)`         — tree-importance counter
//!   - `serialize_tree(...)`        — flatten tree for storage
//!
//! These helpers live inside `r2-engine` and use `&mut Engine` for type
//! coercion (`as_reals`, `scalar_f64`). The R.1 completion plan is:
//!
//!   1. Extract pure helpers (`build_tree`, `tree_predict_one`,
//!      `count_splits`, `serialize_tree`, `parallel_random`) into
//!      `r2-ml::tree` — they're algorithmic and don't need an engine.
//!   2. Refactor `extract_ml_data` to use `r2-types`-only APIs (move
//!      `as_reals` / `scalar_f64` into `r2-types` as RVal methods).
//!   3. Move each `bi_*` ML function in turn, one per session.
//!
//! Step 1 is mechanical (~200 LoC moved). Step 2 needs careful naming
//! coordination with engine. Step 3 is mechanical per function.
//!
//! See r2-stats for the locked-in pattern this crate will follow.

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

pub mod tree;
pub mod data;
pub mod dispatch;

use r2_types::{EvalArg, R2Err, RVal};

/// Returns this crate's exported builtins as `(name, fn-pointer)` pairs.
/// r2-engine adapts the signature at registration time.
pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("rpart",       dispatch::bi_rpart),
        ("rf",          dispatch::bi_rf),
        ("gbm",         dispatch::bi_gbm),
        ("kmeans",      dispatch::bi_kmeans),
        ("cv",          dispatch::bi_cv),
        ("knn",         dispatch::bi_knn),
        ("naive.bayes", dispatch::bi_naive_bayes),
        ("prcomp",      dispatch::bi_prcomp),
    ]
}
