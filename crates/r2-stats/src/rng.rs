//! RNG primitives + random-variate builtins — Phase R.12.
//!
//! Home for the global RNG seed and all random-variate generators:
//! `rnorm`, `runif`, `sample`, `rbinom`, `rpois`, `set.seed`.
//!
//! **Why here:** the seed and the generators are statistical primitives.
//! Previously the seed lived in `r2_ml::tree::SEED_STATE` for historical
//! reasons (rf/gbm needed it first), and the generator builtins lived in
//! r2-engine. Both consolidate here so the random-stats cluster has one
//! owner. `r2_ml::tree` now re-exports from this module so existing
//! `r2_ml::tree::next_random` / `parallel_random` call sites keep working.
//!
//! **Algorithm:** PCG-flavoured LCG step
//! (`s ← s·6364136223846793005 + 1442695040888963407`), top 53 bits → f64
//! in `[0, 1)`. Good enough for everyday Monte Carlo at v0.x; not
//! cryptographic and not designed for parallel streams (use the
//! `parallel_random(seed)` form which takes a thread-local seed instead
//! of touching the global atomic).

use r2_types::{Attrs, ErrKind, EvalArg, Integer, R2Err, RVal, Real};
use std::sync::atomic::{AtomicU64, Ordering};

#[inline]
fn first(a: &[EvalArg]) -> RVal { a.first().map(|x| x.value.clone()).unwrap_or(RVal::Null) }

#[inline]
fn arg_named(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some(name)).map(|x| x.value.clone())
}

/// R-style positional/named argument matcher for the distribution
/// builtins. Formals are resolved in declaration order: a formal takes its
/// named argument if one was supplied, otherwise the next not-yet-consumed
/// unnamed (positional) argument. This matches R for both
/// `rnorm(100, 10, 3)` and `rnorm(n = 100, sd = 3)` call styles — the old
/// code read `mean`/`sd`/`min`/`max`/etc. by name only, silently dropping
/// positionally-passed values (`rnorm(n, 10, 3)` came out standard-normal).
struct ArgMatcher<'a> {
    args: &'a [EvalArg],
    /// How many unnamed (positional) args have been consumed so far.
    pos: usize,
}

impl<'a> ArgMatcher<'a> {
    fn new(args: &'a [EvalArg]) -> Self {
        Self { args, pos: 0 }
    }

    /// Resolve formal `name`: a matching named argument wins; otherwise the
    /// next unconsumed positional argument is taken (and the cursor advances).
    fn get(&mut self, name: &str) -> Option<RVal> {
        if let Some(v) = arg_named(self.args, name) {
            return Some(v);
        }
        let mut seen = 0usize;
        let mut found = None;
        for x in self.args {
            if x.name.is_none() {
                if seen == self.pos {
                    found = Some(x.value.clone());
                    break;
                }
                seen += 1;
            }
        }
        self.pos += 1;
        found
    }

    fn get_f64(&mut self, name: &str, default: f64) -> f64 {
        self.get(name).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(default)
    }

    fn get_usize(&mut self, name: &str, default: usize) -> usize {
        self.get(name)
            .and_then(|v| v.scalar_f64().ok().flatten())
            .map(|x| x as usize)
            .unwrap_or(default)
    }
}

const MAX_ALLOC_BYTES: usize = 500_000_000;
fn check_alloc(elements: usize, elem_size: usize) -> Result<(), R2Err> {
    let bytes = elements.saturating_mul(elem_size);
    if bytes > MAX_ALLOC_BYTES {
        return Err(R2Err {
            msg: format!("allocation of {} bytes exceeds limit (max {} MB).", bytes, MAX_ALLOC_BYTES / 1_000_000),
            kind: ErrKind::Runtime,
        });
    }
    Ok(())
}

// ── Global RNG state ────────────────────────────────────────────────

/// Global LCG seed. Mutated atomically via CAS for thread-safety.
/// `set.seed(k)` stores into this; `next_random()` advances it.
pub static SEED_STATE: AtomicU64 = AtomicU64::new(12345);

/// Read the current seed (snapshot — non-atomic, fine for diagnostics).
pub fn current_seed() -> u64 { SEED_STATE.load(Ordering::Relaxed) }

/// Set the global seed. Used by `bi_set_seed` and by tests.
pub fn set_seed(v: u64) { SEED_STATE.store(v, Ordering::Relaxed) }

thread_local! {
    /// Phase P — per-worker RNG stream. When `Some`, `next_random()` advances
    /// this thread-local seed instead of the shared atomic, giving parallel
    /// workers independent, reproducible streams (seeded per task in
    /// `mclapply`). `None` = use the global stream (the normal serial case).
    static WORKER_SEED: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Install (or clear) the current thread's per-worker RNG stream (Phase P).
pub fn set_worker_seed(s: Option<u64>) { WORKER_SEED.with(|c| c.set(s)); }

/// f64 in `[0, 1)`. Thread-safe. Uses the per-worker stream if one is set
/// (parallel apply), else the shared atomic (serial).
pub fn next_random() -> f64 {
    if let Some(s) = WORKER_SEED.with(|c| c.get()) {
        let new = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        WORKER_SEED.with(|c| c.set(Some(new)));
        return (new >> 11) as f64 / (1u64 << 53) as f64;
    }
    let mut old = SEED_STATE.load(Ordering::Relaxed);
    loop {
        let new = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        match SEED_STATE.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return (new >> 11) as f64 / (1u64 << 53) as f64,
            Err(x) => old = x,
        }
    }
}

/// Advance a CALLER-OWNED seed (no atomic). Use in parallel sections
/// where each thread carries its own seed; cheaper than `next_random`
/// because no CAS.
pub fn parallel_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

// ── Random-variate builtins ─────────────────────────────────────────

pub fn bi_rnorm(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut m = ArgMatcher::new(a);
    let n = m.get_usize("n", 1);
    let mean = m.get_f64("mean", 0.0);
    let sd = m.get_f64("sd", 1.0);
    check_alloc(n, 8)?;
    // F.3 native-columnar path: rnorm produces no NAs, so build the
    // dense `Vec<f64>` directly and wrap via `Reals::from_dense_f64`.
    // This skips the `Option<f64>` allocation and the from_option_slice
    // re-pack that would happen on the first `.columnar()` call.
    let mut results: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        // Box–Muller transform.
        let u1 = next_random().max(1e-15);
        let u2 = next_random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        results.push(mean + sd * z);
    }
    Ok(RVal::Numeric(r2_types::Reals::from_dense_f64(results), Attrs::default()))
}

pub fn bi_runif(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut m = ArgMatcher::new(a);
    let n = m.get_usize("n", 1);
    let min = m.get_f64("min", 0.0);
    let max = m.get_f64("max", 1.0);
    check_alloc(n, 8)?;
    // F.3 native-columnar path: see bi_rnorm.
    let mut results: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let u = next_random();
        results.push(min + (max - min) * u);
    }
    Ok(RVal::Numeric(r2_types::Reals::from_dense_f64(results), Attrs::default()))
}

pub fn bi_rexp(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut m = ArgMatcher::new(a);
    let n = m.get_usize("n", 1);
    let rate = m.get_f64("rate", 1.0);
    check_alloc(n, 8)?;
    let mut results: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        // Inverse-CDF: X = -ln(U)/rate.
        let u = next_random().max(1e-15);
        results.push(-u.ln() / rate);
    }
    Ok(RVal::Numeric(r2_types::Reals::from_dense_f64(results), Attrs::default()))
}

pub fn bi_sample(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x = first(a).as_reals()?;
    let nth = a.get(1).map(|x| x.value.clone()).unwrap_or(RVal::Null);
    let n_default = x.len() as f64;
    let n = arg_named(a, "size")
        .or(if a.len() > 1 && a[1].name.is_none() { Some(nth) } else { None })
        .and_then(|v| v.scalar_f64().ok().flatten())
        .unwrap_or(n_default) as usize;
    if x.is_empty() {
        return Err(R2Err { msg: "sample: cannot sample from empty vector".into(), kind: ErrKind::Runtime });
    }
    let mut result: Vec<Real> = Vec::with_capacity(n);
    // Drive from the global stream for reproducibility under set.seed.
    for _ in 0..n {
        let u = next_random();
        let idx = (u * x.len() as f64) as usize % x.len();
        result.push(x[idx]);
    }
    Ok(RVal::Numeric(result.into(), Attrs::default()))
}

pub fn bi_rbinom(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut m = ArgMatcher::new(a);
    let n = m.get_usize("n", 1);
    let size = m.get_usize("size", 1);
    let prob = m.get_f64("prob", 0.5);
    let mut results: Vec<Integer> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut successes = 0i32;
        for _ in 0..size {
            if next_random() < prob { successes += 1; }
        }
        results.push(Some(successes));
    }
    Ok(RVal::Integer(results.into(), Attrs::default()))
}

pub fn bi_rpois(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mut m = ArgMatcher::new(a);
    let n = m.get_usize("n", 1);
    let lambda = m.get_f64("lambda", 1.0);
    let mut results: Vec<Integer> = Vec::with_capacity(n);
    for _ in 0..n {
        // Knuth's algorithm. Acceptable for small-to-moderate lambda
        // (≤ 30); a normal approximation would be tighter for large lambda.
        let l = (-lambda).exp();
        let mut k = 0i32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= next_random();
            if p <= l { break; }
        }
        results.push(Some(k - 1));
    }
    Ok(RVal::Integer(results.into(), Attrs::default()))
}

pub fn bi_set_seed(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let seed = first(a).scalar_f64()?.unwrap_or(42.0) as u64;
    set_seed(seed);
    soutln!("Random seed set to {}", seed);
    Ok(RVal::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }
    fn n(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }

    #[test]
    fn set_seed_is_reproducible() {
        // Use thread-local seed so this test doesn't race with other
        // rng tests in the same crate that mutate the global SEED_STATE
        // (cargo test runs threads in parallel by default).
        let mut s1 = 123u64;
        let a = parallel_random(&mut s1);
        let b = parallel_random(&mut s1);
        let mut s2 = 123u64;
        let a2 = parallel_random(&mut s2);
        let b2 = parallel_random(&mut s2);
        assert!((a - a2).abs() < 1e-15);
        assert!((b - b2).abs() < 1e-15);
    }

    #[test]
    fn runif_in_range() {
        set_seed(42);
        let r = bi_runif(&[evarg(n(1000.0))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                for x in v.iter() {
                    let x = x.unwrap();
                    assert!(x >= 0.0 && x < 1.0, "out of range: {}", x);
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rnorm_mean_and_sd_approx() {
        set_seed(7);
        let r = bi_rnorm(&[evarg(n(10_000.0))]).unwrap();
        match r {
            RVal::Numeric(v, _) => {
                let values: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                // ±0.03 mean and 5% variance tolerance for n=10k Box-Muller.
                assert!(mean.abs() < 0.05, "mean = {}", mean);
                assert!((var - 1.0).abs() < 0.05, "var = {}", var);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn rbinom_within_bounds() {
        set_seed(11);
        let r = bi_rbinom(&[
            evarg(n(100.0)),
            EvalArg { name: Some(std::sync::Arc::from("size")), value: n(10.0) },
            EvalArg { name: Some(std::sync::Arc::from("prob")), value: n(0.5) },
        ]).unwrap();
        match r {
            RVal::Integer(v, _) => for x in &v {
                let x = x.unwrap();
                assert!((0..=10).contains(&x), "{} not in [0,10]", x);
            },
            _ => panic!(),
        }
    }
}
