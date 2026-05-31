//! Builtin function shims, grouped by domain. Each submodule
//! exposes `pub(crate) fn bi_*(&mut Engine, &[EvalArg], &EnvRef) ->
//! Result<RVal, R2Err>` functions that get registered into the
//! function table in `crate::Engine::new` and friends.
//!
//! The split exists to keep `lib.rs` browsable. Engine logic
//! (struct, eval, registry, package layer) stays in `lib.rs`;
//! call-site bodies that just dispatch into a sibling crate
//! (r2-graphics, r2-stats, etc.) live here.
//!
//! Migration policy: move one cohesive domain at a time, verify the
//! build, ship. Never carry incomplete moves across a commit.

pub(crate) mod data;
pub(crate) mod graphics;
pub(crate) mod io;
pub(crate) mod ml;
pub(crate) mod stats;
pub(crate) mod strings;
