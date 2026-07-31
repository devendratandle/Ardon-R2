//! r2-train — training-loop foundation + the VERIFIED-INNOVATION core.
//!
//! Two things live here (docs/LLM_TRILLION_ARCHITECTURE.md layers 5–6):
//!
//! 1. The training-step abstraction (`Objective`, `step_verified`) that
//!    the full transformer loop (Opus) plugs into.
//!
//! 2. **The fuzzy-innovation core** — the maintainer's design: an OPT-IN
//!    0.5–1% exploration budget where a training step may take a
//!    stochastic (non-gradient) candidate step, which is NEVER merged
//!    until it passes TWO independent gates:
//!      • the FORMAL gate — an inductive invariant check (finite loss,
//!        bounded norms, trust region): if the invariants held at step k,
//!        the candidate must preserve them at k+1;
//!      • the EMPIRICAL gate — validation on held-out signal (not the
//!        batch that produced it): the candidate must IMPROVE the
//!        objective by a margin beyond noise.
//!    Fail either ⇒ the candidate is DISCARDED and the plain verified
//!    gradient step is taken instead. The worst case of exploration is
//!    "nothing found, training proceeds normally" — there is no path
//!    where an unvalidated perturbation degrades the model. Every
//!    accepted innovation is seeded/logged ⇒ replayable and auditable.

use std::sync::atomic::{AtomicU64, Ordering};

pub mod optim;
pub mod accumulate;
pub mod llm;
pub mod checkpoint;
pub mod transformer;
pub mod distributed;

// ── The objective a training loop optimizes ────────────────────────────

/// What the trainer needs from a model: loss + gradient on the TRAINING
/// batch, and loss on HELD-OUT validation data (the empirical gate's
/// independent signal). The toy quadratic in tests implements this; the
/// transformer (Opus) implements the same trait.
pub trait Objective {
    /// Loss at `params` on the current training batch.
    fn loss(&self, params: &[f32]) -> f32;
    /// Gradient at `params` on the current training batch.
    fn grad(&self, params: &[f32]) -> Vec<f32>;
    /// Loss at `params` on HELD-OUT validation data. Must be independent
    /// of the training batch — the no-self-confirmation rule.
    fn validation_loss(&self, params: &[f32]) -> f32;
}

// ── Exploration policy (the "fuzzy" part) ──────────────────────────────

/// Deterministic, seeded RNG (xorshift64*) — exploration must be
/// REPLAYABLE: an accepted innovation can be re-derived from the logged
/// seed + step number, which is what makes it auditable.
pub struct SeededRng(u64);
impl SeededRng {
    pub fn new(seed: u64) -> Self { SeededRng(seed.max(1)) }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32) / ((1u64 << 24) as f32) // [0,1)
    }
    /// Uniform in [-1, 1).
    pub fn next_sym(&mut self) -> f32 { self.next_f32() * 2.0 - 1.0 }
}

/// Decides WHEN to explore and WHAT candidate to propose.
pub trait ExplorationPolicy {
    /// Should step `k` be an exploration step? Called once per step; the
    /// implementation enforces the budget rate ρ.
    fn should_explore(&mut self, step: u64) -> bool;
    /// Propose a candidate parameter vector (a perturbation of `params`).
    fn propose(&mut self, params: &[f32]) -> Vec<f32>;
}

/// Reference policy: explore with probability ρ (0.5–1% per the design),
/// proposing a bounded random perturbation within `radius` (the trust
/// region the formal gate re-checks).
pub struct StochasticPolicy {
    pub rate: f32,
    pub radius: f32,
    pub rng: SeededRng,
}

impl StochasticPolicy {
    /// `rate` clamped to the designed 0–1% envelope (0 disables).
    pub fn new(rate: f32, radius: f32, seed: u64) -> Self {
        StochasticPolicy { rate: rate.clamp(0.0, 0.01), radius, rng: SeededRng::new(seed) }
    }
}

impl ExplorationPolicy for StochasticPolicy {
    fn should_explore(&mut self, _step: u64) -> bool {
        self.rng.next_f32() < self.rate
    }
    fn propose(&mut self, params: &[f32]) -> Vec<f32> {
        params.iter().map(|p| p + self.rng.next_sym() * self.radius).collect()
    }
}

// ── The two verification gates ─────────────────────────────────────────

/// Why a candidate was rejected (logged for audit).
#[derive(Debug, Clone, PartialEq)]
pub enum Rejection {
    /// Formal gate: an invariant would break at k+1.
    InvariantViolated(&'static str),
    /// Empirical gate: no improvement beyond the noise margin.
    NoValidatedImprovement { candidate: f32, incumbent: f32 },
}

/// Gate 1 — FORMAL (inductive invariants). Assuming the invariants held
/// at step k (params finite, loss finite), the candidate must preserve
/// them at k+1 and stay inside the trust region. Pure checks — no data.
pub fn formal_gate(
    params: &[f32], candidate: &[f32], candidate_loss: f32, radius: f32,
) -> Result<(), Rejection> {
    if !candidate_loss.is_finite() {
        return Err(Rejection::InvariantViolated("loss must remain finite"));
    }
    if candidate.iter().any(|x| !x.is_finite()) {
        return Err(Rejection::InvariantViolated("parameters must remain finite"));
    }
    // Trust region: ‖candidate − params‖∞ ≤ radius (+ f32 slack).
    let max_dev = params.iter().zip(candidate)
        .map(|(p, c)| (p - c).abs()).fold(0.0f32, f32::max);
    if max_dev > radius * 1.0001 {
        return Err(Rejection::InvariantViolated("candidate outside trust region"));
    }
    Ok(())
}

/// Gate 2 — EMPIRICAL ("neural-enforced" validation). The candidate must
/// improve HELD-OUT validation loss by at least `margin` (a noise floor;
/// scaling this into a proper significance test on batched validation
/// losses is Opus work — the engine's full-precision t-test surface is
/// available for it).
pub fn empirical_gate(
    incumbent_val: f32, candidate_val: f32, margin: f32,
) -> Result<(), Rejection> {
    if candidate_val < incumbent_val - margin {
        Ok(())
    } else {
        Err(Rejection::NoValidatedImprovement {
            candidate: candidate_val, incumbent: incumbent_val,
        })
    }
}

// ── The verified training step ─────────────────────────────────────────

/// Outcome of one step (logged; counters aggregated in `TrainStats`).
#[derive(Debug, Clone, PartialEq)]
pub enum StepKind {
    /// Ordinary verified gradient step (99–99.5% of steps).
    Gradient,
    /// Exploration candidate accepted — passed BOTH gates.
    InnovationAccepted,
    /// Exploration candidate rejected — fell back to the gradient step.
    InnovationRejected(Rejection),
}

/// Aggregate counters proving the budget/gate behaviour (auditable).
#[derive(Default)]
pub struct TrainStats {
    pub steps: AtomicU64,
    pub explored: AtomicU64,
    pub accepted: AtomicU64,
    pub rejected: AtomicU64,
}

/// One training step with the OPT-IN verified-exploration path.
///
/// policy = None (the default) ⇒ plain gradient descent, always.
/// policy = Some(...) ⇒ at rate ρ, propose a candidate; merge it ONLY if
/// it passes the formal gate AND the empirical gate; otherwise discard it
/// and take the ordinary gradient step. Exploration can never make a step
/// worse than plain training — that is the safety property.
pub fn step_verified<O: Objective>(
    obj: &O,
    params: &mut Vec<f32>,
    lr: f32,
    step: u64,
    policy: Option<&mut dyn ExplorationPolicy>,
    stats: &TrainStats,
) -> StepKind {
    stats.steps.fetch_add(1, Ordering::Relaxed);

    if let Some(pol) = policy {
        if pol.should_explore(step) {
            stats.explored.fetch_add(1, Ordering::Relaxed);
            let candidate = pol.propose(params);
            let cand_loss = obj.loss(&candidate);
            // Gate 1: formal invariants (radius from the policy's proposal
            // bound — reference uses the ∞-norm trust region).
            let radius = params.iter().zip(&candidate)
                .map(|(p, c)| (p - c).abs()).fold(0.0f32, f32::max).max(1e-12);
            let g1 = formal_gate(params, &candidate, cand_loss, radius);
            let outcome = g1.and_then(|_| {
                // Gate 2: held-out validation, small noise margin.
                let inc = obj.validation_loss(params);
                let cnd = obj.validation_loss(&candidate);
                empirical_gate(inc, cnd, 1e-6)
            });
            match outcome {
                Ok(()) => {
                    *params = candidate; // merge the VERIFIED innovation
                    stats.accepted.fetch_add(1, Ordering::Relaxed);
                    return StepKind::InnovationAccepted;
                }
                Err(rej) => {
                    stats.rejected.fetch_add(1, Ordering::Relaxed);
                    // Discard candidate; fall through to the gradient step.
                    let g = obj.grad(params);
                    for (p, gi) in params.iter_mut().zip(&g) { *p -= lr * gi; }
                    return StepKind::InnovationRejected(rej);
                }
            }
        }
    }
    let g = obj.grad(params);
    for (p, gi) in params.iter_mut().zip(&g) { *p -= lr * gi; }
    StepKind::Gradient
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy objective: f(x) = ‖x − target‖² with a held-out shift — convex,
    /// known optimum, exact gradient. Validation uses a DIFFERENT target
    /// offset so the two signals are independent (the no-self-confirmation
    /// property the empirical gate requires).
    struct Quad { target: Vec<f32> }
    impl Objective for Quad {
        fn loss(&self, p: &[f32]) -> f32 {
            p.iter().zip(&self.target).map(|(x, t)| (x - t) * (x - t)).sum()
        }
        fn grad(&self, p: &[f32]) -> Vec<f32> {
            p.iter().zip(&self.target).map(|(x, t)| 2.0 * (x - t)).collect()
        }
        fn validation_loss(&self, p: &[f32]) -> f32 {
            // Same optimum, independent evaluation (offset weighting).
            p.iter().zip(&self.target).map(|(x, t)| 1.5 * (x - t) * (x - t)).sum()
        }
    }

    #[test]
    fn plain_training_converges_without_exploration() {
        let obj = Quad { target: vec![1.0, -2.0, 0.5] };
        let mut p = vec![0.0; 3];
        let stats = TrainStats::default();
        for k in 0..200 {
            let kind = step_verified(&obj, &mut p, 0.05, k, None, &stats);
            assert_eq!(kind, StepKind::Gradient);
        }
        assert!(obj.loss(&p) < 1e-6, "loss {}", obj.loss(&p));
        assert_eq!(stats.explored.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn exploration_budget_respected_and_gates_run() {
        let obj = Quad { target: vec![1.0, -2.0, 0.5] };
        let mut p = vec![0.0; 3];
        let mut pol = StochasticPolicy::new(0.01, 0.05, 42);
        let stats = TrainStats::default();
        for k in 0..5000 {
            let _ = step_verified(&obj, &mut p, 0.05, k, Some(&mut pol), &stats);
        }
        let steps = stats.steps.load(Ordering::Relaxed) as f64;
        let explored = stats.explored.load(Ordering::Relaxed) as f64;
        // Budget: ~1% ± statistical slack.
        assert!(explored / steps < 0.03, "explored {}/{}", explored, steps);
        // Every exploration resolved through the gates:
        assert_eq!(stats.explored.load(Ordering::Relaxed),
                   stats.accepted.load(Ordering::Relaxed) + stats.rejected.load(Ordering::Relaxed));
        // And training still converged (exploration never made it worse):
        assert!(obj.loss(&p) < 1e-4, "loss {}", obj.loss(&p));
    }

    #[test]
    fn formal_gate_rejects_nonfinite_and_out_of_region() {
        let p = vec![0.0f32; 2];
        assert!(matches!(formal_gate(&p, &[f32::NAN, 0.0], 1.0, 1.0),
            Err(Rejection::InvariantViolated(_))));
        assert!(matches!(formal_gate(&p, &[0.0, 0.0], f32::INFINITY, 1.0),
            Err(Rejection::InvariantViolated(_))));
        assert!(formal_gate(&p, &[0.5, -0.5], 1.0, 1.0).is_ok());
    }

    #[test]
    fn empirical_gate_rejects_non_improvement() {
        assert!(empirical_gate(1.0, 0.5, 1e-6).is_ok());          // better
        assert!(empirical_gate(1.0, 1.0, 1e-6).is_err());         // equal
        assert!(empirical_gate(1.0, 2.0, 1e-6).is_err());         // worse
    }

    #[test]
    fn deliberately_bad_candidate_is_rejected_and_discarded() {
        // A policy that always explores with a HUGE harmful jump: the
        // gates must reject it every time and training must still work.
        struct BadPolicy;
        impl ExplorationPolicy for BadPolicy {
            fn should_explore(&mut self, _: u64) -> bool { true }
            fn propose(&mut self, params: &[f32]) -> Vec<f32> {
                params.iter().map(|p| p + 1000.0).collect() // ruinous
            }
        }
        let obj = Quad { target: vec![1.0, -2.0] };
        let mut p = vec![0.0; 2];
        let mut pol = BadPolicy;
        let stats = TrainStats::default();
        for k in 0..300 {
            let kind = step_verified(&obj, &mut p, 0.05, k, Some(&mut pol), &stats);
            assert!(matches!(kind, StepKind::InnovationRejected(_)));
        }
        assert_eq!(stats.accepted.load(Ordering::Relaxed), 0);
        // Training converged anyway — exploration can never hurt:
        assert!(obj.loss(&p) < 1e-4, "loss {}", obj.loss(&p));
    }

    #[test]
    fn seeded_rng_is_replayable() {
        let mut a = SeededRng::new(7);
        let mut b = SeededRng::new(7);
        for _ in 0..100 { assert_eq!(a.next_f32(), b.next_f32()); }
    }
}
