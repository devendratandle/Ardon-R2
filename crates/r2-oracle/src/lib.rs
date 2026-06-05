//! R2 Oracle — auto-scheduler V1 (Phase E).
//!
//! Per docs/ARCHITECTURE.md §5 Phase E:
//!   - One central function: `dispatch(op, shape) -> Backend`
//!   - Replaces hand-coded thresholds in `bi_rf`, `bi_kmeans`, `bi_gbm`, `bi_cv`.
//!
//! V1 returns `Serial` or `Rayon`. V2 adds `Gpu` and `CloudShard`.
//!
//! Locked decisions honoured:
//!   §4.6 Oracle V1 is a threshold dispatcher (not a calibrated cost model).
//!   §4.5 Pure-Rust deps only — this crate has zero deps.
//!
//! Design rule: thresholds live HERE, not at call sites. Tuning happens in
//! one place; every parallelizable builtin asks the same Oracle.

#![deny(missing_docs)]
#![allow(missing_docs)] // V1 keeps doc-comments lightweight; tighten in V2.

/// What kind of work the caller wants scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Element-wise vector op `f(v[i])` or `f(a[i], b[i])`.
    PerElementMap,
    /// Reduction over a vector — `sum`, `mean`, `prod`, etc.
    Reduction,
    /// Per-row distance / nearest-centroid / classification scoring.
    PerPointDistance,
    /// Tree construction (random forest, gbm one tree).
    TreeBuild,
    /// K-fold cross-validation (each fold independent).
    KFoldCV,
    /// Per-list-component dispatch — `lapply(lst, f)`-shaped work
    /// where each component is an independent unit. Crossover depends
    /// on aggregate work (not component count), since one big numeric
    /// component is worth parallelising even if there are only 2 of them.
    ListMap,
    /// Catch-all for ops not yet modeled.
    Unknown,
}

/// The dimensions the work runs over. Set fields you know; leave the rest 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct Shape {
    /// Number of items (rows, points, trees, folds…).
    pub n: usize,
    /// Secondary dimension — columns, k-clusters, depth.
    pub m: usize,
    /// Tertiary dimension — features per point, etc.
    pub k: usize,
}

impl Shape {
    pub fn n(n: usize) -> Self { Shape { n, m: 0, k: 0 } }
    pub fn nm(n: usize, m: usize) -> Self { Shape { n, m, k: 0 } }
    pub fn nmk(n: usize, m: usize, k: usize) -> Self { Shape { n, m, k } }
    /// Estimated total work units for the operation. Caller may pass a
    /// custom value; default multiplies the known dimensions.
    pub fn work(&self) -> usize {
        let n = self.n.max(1);
        let m = self.m.max(1);
        let k = self.k.max(1);
        n.saturating_mul(m).saturating_mul(k)
    }
}

/// Where the Oracle says to run the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Single-threaded execution.
    Serial,
    /// Rayon work-stealing thread pool.
    Rayon,
}

// ════════════════════════════════════════════════════════════════════
// Hardware introspection — Phase G partial (v0.1.0)
// ════════════════════════════════════════════════════════════════════
//
// A tiny one-shot probe that runs once at startup and exposes a frozen
// `Hw` struct to dispatch decisions. Captures the deployment signals
// that determine when parallelism wins:
//
//   - **cores**: parallel crossover scales inversely with core count.
//     A 2-core VM should stay serial for medium workloads; a 64-core
//     server should go parallel earlier.
//   - **CPU features**: FMA/AVX2/AVX-512 availability affects per-core
//     throughput. Used by JIT and kernel paths that have SIMD variants.
//   - **RAM (best-effort)**: large-allocation heuristics; not yet wired
//     into dispatch but available for future cost models.
//
// **Why partial**: full Phase G would also detect cache sizes (L1/L2/L3),
// NUMA topology, and ISA-specific SIMD widths. Those need an extra dep
// (`raw-cpuid` or `cache-size`) and finer detection logic. v0.1.0 ships
// the 80% that's free (`std::thread::available_parallelism`,
// `std::is_x86_feature_detected!`, simple env-var RAM hints).
//
// **Pure-Rust deps**: nothing new. Cores via stdlib. SIMD via cfg-gated
// `is_x86_feature_detected!` macro. RAM via env-var override (no probe).

/// Snapshot of the deployment hardware. Built once at process start via
/// [`hw()`]; subsequent calls return the same cached struct.
#[derive(Debug, Clone, Copy)]
pub struct Hw {
    /// Number of available logical cores (via `std::thread::available_parallelism`).
    /// Falls back to 1 if the OS doesn't report it.
    pub cores: usize,
    /// True if the CPU advertises FMA3 / AVX2. (x86 only — false on ARM
    /// since we don't yet detect SVE/NEON features.)
    pub has_fma: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    /// User-hinted RAM in MB via `R2_RAM_MB` env var, else 0 (unknown).
    /// Auto-detection deferred to Phase G proper (needs `sysinfo` dep).
    pub ram_mb_hint: usize,
    /// Architecture name: "x86_64", "aarch64", etc.
    pub arch: &'static str,
    /// OS name: "linux", "windows", "macos", etc.
    pub os: &'static str,
}

impl Hw {
    fn detect() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // SIMD feature detection — cfg-gated so non-x86 builds skip.
        #[cfg(target_arch = "x86_64")]
        let (has_fma, has_avx2, has_avx512) = (
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f"),
        );
        #[cfg(not(target_arch = "x86_64"))]
        let (has_fma, has_avx2, has_avx512) = (false, false, false);

        let ram_mb_hint: usize = std::env::var("R2_RAM_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Hw {
            cores,
            has_fma,
            has_avx2,
            has_avx512,
            ram_mb_hint,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
        }
    }
}

/// Returns the cached hardware snapshot. First call probes; subsequent
/// calls return the same value (O(1) load). Safe to call from any thread.
pub fn hw() -> &'static Hw {
    use std::sync::OnceLock;
    static HW: OnceLock<Hw> = OnceLock::new();
    HW.get_or_init(Hw::detect)
}

/// V1 dispatch — threshold-based, **now hardware-aware**.
///
/// Base thresholds were calibrated against ~3 GHz x86_64 with 8 cores.
/// We scale them by the deployment's actual core count: more cores ⇒
/// parallel becomes profitable at smaller N (overhead amortizes faster).
/// The formula is `scaled = base * (8 / cores).max(0.25)` clamped, so:
///   - 1 core   → 8× the base threshold (effectively serial-only)
///   - 4 cores  → 2× the base threshold
///   - 8 cores  → base
///   - 64 cores → 0.25× the base threshold
///
/// This is a closed-form heuristic, not a calibration. Real `r2-bench`
/// calibration is deferred to Phase G proper.
pub fn dispatch(op: Op, shape: Shape) -> Backend {
    // Measurement / debug override: R2_FORCE_SERIAL=1 makes the oracle
    // always pick Serial, so the parallel speedup can be A/B benchmarked
    // (and as an escape hatch on flaky multi-core environments).
    if std::env::var_os("R2_FORCE_SERIAL").is_some() {
        return Backend::Serial;
    }
    let work = shape.work();
    let base: usize = match op {
        Op::PerElementMap     => 50_000,
        Op::Reduction         => 200_000,
        Op::PerPointDistance  => 10_000,
        Op::TreeBuild         => 1,
        Op::KFoldCV           => 2,
        // ListMap: aggregate-work threshold. Set lower than PerElementMap
        // because per-component spawn overhead is already amortised by
        // having distinct components (vs N tiny per-element iterations).
        Op::ListMap           => 10_000,
        Op::Unknown           => 100_000,
    };
    // Trees and CV stay always-parallel; scaling them is meaningless.
    let threshold = if matches!(op, Op::TreeBuild | Op::KFoldCV) {
        base
    } else {
        scale_threshold(base, hw().cores)
    };
    if work >= threshold { Backend::Rayon } else { Backend::Serial }
}

/// Scale a base threshold by the actual core count. 8-core machine is
/// the reference; fewer cores raise the bar, more cores lower it.
/// Clamps to [0.25×, 8×] so extreme platforms don't get pathological.
#[inline]
fn scale_threshold(base: usize, cores: usize) -> usize {
    let cores = cores.max(1) as f64;
    let factor = (8.0 / cores).clamp(0.25, 8.0);
    ((base as f64) * factor) as usize
}

/// Convenience: returns `true` if dispatch picks Rayon.
pub fn should_parallelize(op: Op, shape: Shape) -> bool {
    matches!(dispatch(op, shape), Backend::Rayon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_per_element_stays_serial() {
        assert_eq!(dispatch(Op::PerElementMap, Shape::n(100)), Backend::Serial);
    }

    #[test]
    fn large_per_element_goes_parallel() {
        assert_eq!(dispatch(Op::PerElementMap, Shape::n(1_000_000)), Backend::Rayon);
    }

    #[test]
    fn kmeans_shape_threshold() {
        // Hardware-aware: base PerPointDistance threshold = 10K, scaled
        // to [2.5K, 80K] depending on core count. Pick N values clearly
        // outside that envelope to avoid machine-dependent test flakes.
        // m*k*n = 1000*10*10 = 100K → above any scaled threshold.
        assert_eq!(dispatch(Op::PerPointDistance, Shape::nmk(1000, 10, 10)), Backend::Rayon);
        // m*k*n = 100*5*4 = 2000 → below any scaled threshold.
        assert_eq!(dispatch(Op::PerPointDistance, Shape::nmk(100, 5, 4)), Backend::Serial);
    }

    #[test]
    fn cv_always_parallel_for_multiple_folds() {
        assert_eq!(dispatch(Op::KFoldCV, Shape::n(2)), Backend::Rayon);
        assert_eq!(dispatch(Op::KFoldCV, Shape::n(10)), Backend::Rayon);
    }

    #[test]
    fn tree_build_always_parallel() {
        assert_eq!(dispatch(Op::TreeBuild, Shape::n(1)), Backend::Rayon);
    }

    // ── Hardware-aware Oracle (v0.1.0 partial Phase G) ──────────────

    #[test]
    fn hw_snapshot_is_consistent() {
        let h1 = hw();
        let h2 = hw();
        assert_eq!(h1.cores, h2.cores, "Hw should be cached and consistent");
        assert!(h1.cores >= 1, "must report at least 1 core");
        // Arch and OS strings are from std::env::consts — never empty.
        assert!(!h1.arch.is_empty());
        assert!(!h1.os.is_empty());
    }

    #[test]
    fn scale_threshold_extreme_clamps() {
        // 1-core machine: factor clamped to 8× the base.
        assert_eq!(scale_threshold(10_000, 1), 80_000);
        // 100-core machine: factor clamped to 0.25× the base (not 0.08×).
        assert_eq!(scale_threshold(10_000, 100), 2_500);
        // Reference 8-core: no scaling.
        assert_eq!(scale_threshold(10_000, 8), 10_000);
        // 4-core: 2× the base.
        assert_eq!(scale_threshold(10_000, 4), 20_000);
        // 0 cores (impossible but defensive): clamped to 1 → 8× the base.
        assert_eq!(scale_threshold(10_000, 0), 80_000);
    }

    #[test]
    fn tree_build_and_cv_ignore_core_scaling() {
        // TreeBuild and KFoldCV are always-parallel by design; scaling
        // them would be a regression.
        assert_eq!(dispatch(Op::TreeBuild, Shape::n(1)), Backend::Rayon);
        assert_eq!(dispatch(Op::KFoldCV,   Shape::n(2)), Backend::Rayon);
    }
}
