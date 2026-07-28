//! Optimizers — Adam, and why its state is not optional.
//!
//! Plain SGD is stateless: `p -= lr * g`. Adam is not. It carries a
//! per-parameter first moment (`m`, momentum) and second moment (`v`,
//! the running scale of recent gradients), plus a step counter used to
//! bias-correct both. That state IS part of the training trajectory —
//! resuming a run without it does not continue the same optimization, it
//! starts a differently-conditioned one that happens to share the current
//! weights. Every serious training loop therefore checkpoints the
//! optimizer, and `r2_train::checkpoint` does.

/// Adam (Kingma & Ba), with the standard bias correction.
#[derive(Debug, Clone)]
pub struct Adam {
    /// First moment (momentum) per parameter.
    pub m: Vec<f32>,
    /// Second moment (uncentered variance) per parameter.
    pub v: Vec<f32>,
    /// Steps taken — drives bias correction, so it must survive a resume.
    pub t: u64,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    /// Standard defaults (β₁=0.9, β₂=0.999, ε=1e-8) for `n` parameters.
    pub fn new(n: usize, lr: f32) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0,
               lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }

    /// Number of parameters this optimizer is sized for.
    pub fn len(&self) -> usize { self.m.len() }
    pub fn is_empty(&self) -> bool { self.m.is_empty() }

    /// One update, in place. Errors on a length mismatch rather than
    /// updating a prefix — a shape error here would corrupt training
    /// silently and be very hard to trace back.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), String> {
        if params.len() != self.m.len() || grads.len() != self.m.len() {
            return Err(format!(
                "Adam::step: sized for {} params, got params={} grads={}",
                self.m.len(), params.len(), grads.len()));
        }
        self.t += 1;
        // Bias correction: m and v start at zero, so early estimates are
        // biased toward zero; dividing by (1-β^t) removes exactly that.
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            params[i] -= self.lr * mh / (vh.sqrt() + self.eps);
        }
        Ok(())
    }

    /// Discard momentum/variance but keep the hyper-parameters — what a
    /// weights-only "resume" effectively does. Exposed so the cost of
    /// doing that can be measured rather than argued about (see the
    /// checkpoint tests).
    pub fn reset_state(&mut self) {
        for x in self.m.iter_mut() { *x = 0.0; }
        for x in self.v.iter_mut() { *x = 0.0; }
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradient of f(p) = Σ (p_i - target_i)² — a convex bowl, so a
    /// correct optimizer must approach the target monotonically.
    fn grad(params: &[f32], target: &[f32]) -> Vec<f32> {
        params.iter().zip(target).map(|(p, t)| 2.0 * (p - t)).collect()
    }

    #[test]
    fn converges_on_a_convex_objective() {
        let target = vec![1.0f32, -2.0, 0.5];
        let mut p = vec![0.0f32; 3];
        let mut opt = Adam::new(3, 0.1);
        for _ in 0..500 {
            let g = grad(&p, &target);
            opt.step(&mut p, &g).unwrap();
        }
        for (a, b) in p.iter().zip(&target) {
            assert!((a - b).abs() < 1e-3, "Adam should reach the minimum: {p:?}");
        }
        assert_eq!(opt.t, 500, "step count must track updates");
    }

    #[test]
    fn state_evolves_and_bias_correction_applies_on_step_one() {
        // With m=v=0 and constant g, the FIRST bias-corrected step is
        // almost exactly -lr (the classic Adam property). Catches a
        // missing or wrong bias correction, which otherwise only shows up
        // as slightly-off early training.
        let mut p = vec![0.0f32];
        let mut opt = Adam::new(1, 0.01);
        opt.step(&mut p, &[3.0]).unwrap();
        assert!((p[0] + 0.01).abs() < 1e-6, "first step should move ~-lr, got {}", p[0]);
        assert!(opt.m[0] != 0.0 && opt.v[0] != 0.0, "moments must update");
    }

    #[test]
    fn length_mismatch_is_an_error_not_a_partial_update() {
        let mut opt = Adam::new(3, 0.1);
        let mut p = vec![0.0f32; 3];
        assert!(opt.step(&mut p, &[1.0, 2.0]).is_err());
        assert_eq!(p, vec![0.0; 3], "no parameter may change on a rejected step");
        assert_eq!(opt.t, 0, "a rejected step must not advance the counter");
    }

    #[test]
    fn reset_state_clears_moments_but_keeps_hyperparameters() {
        let mut opt = Adam::new(2, 0.05);
        opt.step(&mut [0.0, 0.0], &[1.0, 1.0]).unwrap();
        opt.reset_state();
        assert_eq!(opt.t, 0);
        assert!(opt.m.iter().all(|&x| x == 0.0) && opt.v.iter().all(|&x| x == 0.0));
        assert_eq!(opt.lr, 0.05);
    }
}
