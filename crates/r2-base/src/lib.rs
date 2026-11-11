// R2 Base Library — built-in datasets and linalg-domain builtins.
//
// The datasets are R's whole `datasets` package, written by R into
// `datasets/datasets.r2o` and embedded (see `datasets.rs`).
//
// Phase R.4: linalg builtins (matrix/tensor/t/crossprod/svd/eigen) live
// in the `linalg_ops` submodule — pure `fn(&[EvalArg]) -> Result<RVal,
// R2Err>`, thin wrappers over r2-linalg kernels.

pub mod datasets;
pub mod linalg_ops;
pub mod qr_ops;
pub mod r2d;

use r2_types::*;
use std::sync::Arc;

/// Bind all of R's datasets into the base environment.
pub fn register_datasets(env: &mut std::collections::HashMap<Arc<str>, RVal>) {
    datasets::register_all(env);
}
