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

/// Is AVX2+FMA available? Resolved once.
///
/// Runtime dispatch, not `-C target-cpu`: the workspace sets no target CPU,
/// so without this every kernel compiles baseline x86-64 (SSE2, no FMA) —
/// and one binary still runs everywhere.
#[cfg(target_arch = "x86_64")]
fn have_avx2() -> bool {
    use std::sync::OnceLock;
    static YES: OnceLock<bool> = OnceLock::new();
    *YES.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

/// The Adam update for one block. The reference form.
#[allow(clippy::too_many_arguments)]
fn adam_scalar(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
               b1: f32, b2: f32, bc1: f32, bc2: f32, lr: f32, eps: f32, scale: f32) {
    for i in 0..p.len() {
        let gi = g[i] / scale;
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        p[i] -= lr * mh / (vh.sqrt() + eps);
    }
}

/// The same update, eight lanes at a time.
///
/// Every operation is the IEEE one the scalar form uses, in the same order:
/// `div` and `sqrt` rather than a reciprocal approximation, and separate
/// multiply-add rather than FMA. So this is bit-identical to
/// [`adam_scalar`], which matters because the loss trajectory is the
/// project's regression check — an optimizer that drifts in the last ULP
/// would make every future comparison ambiguous.
///
/// The scalar loop could not be vectorised by the compiler on its own
/// terms: the two bias-correction divisions are by loop-invariant values,
/// but turning a division into a multiply-by-reciprocal changes rounding,
/// so LLVM may not do it without fast-math, and `sqrt` plus three divisions
/// per parameter over 7.24M parameters is what that costs.
///
/// # Safety
///
/// Requires AVX2 and FMA (checked by [`have_avx2`]). All four slices must
/// be the same length; the caller checks that.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn adam_avx2(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
                    b1: f32, b2: f32, bc1: f32, bc2: f32, lr: f32, eps: f32, scale: f32) {
    use std::arch::x86_64::*;
    let (vb1, vb2) = (_mm256_set1_ps(b1), _mm256_set1_ps(b2));
    let (v1b1, v1b2) = (_mm256_set1_ps(1.0 - b1), _mm256_set1_ps(1.0 - b2));
    let (vbc1, vbc2) = (_mm256_set1_ps(bc1), _mm256_set1_ps(bc2));
    let (vlr, veps, vsc) = (_mm256_set1_ps(lr), _mm256_set1_ps(eps), _mm256_set1_ps(scale));
    let n = p.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let gi = _mm256_div_ps(_mm256_loadu_ps(g.as_ptr().add(i)), vsc);
        let mv = _mm256_add_ps(_mm256_mul_ps(vb1, _mm256_loadu_ps(m.as_ptr().add(i))),
                               _mm256_mul_ps(v1b1, gi));
        // `(1-b2) * gi * gi` associates LEFT in the scalar form —
        // `((1-b2)*gi)*gi`, not `(1-b2)*(gi*gi)`. Those differ in the last
        // bit, and the bit-identity test caught it.
        let vv = _mm256_add_ps(_mm256_mul_ps(vb2, _mm256_loadu_ps(v.as_ptr().add(i))),
                               _mm256_mul_ps(_mm256_mul_ps(v1b2, gi), gi));
        _mm256_storeu_ps(m.as_mut_ptr().add(i), mv);
        _mm256_storeu_ps(v.as_mut_ptr().add(i), vv);
        let den = _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(vv, vbc2)), veps);
        let upd = _mm256_div_ps(_mm256_mul_ps(vlr, _mm256_div_ps(mv, vbc1)), den);
        _mm256_storeu_ps(p.as_mut_ptr().add(i),
                         _mm256_sub_ps(_mm256_loadu_ps(p.as_ptr().add(i)), upd));
        i += 8;
    }
    if i < n {
        adam_scalar(&mut p[i..], &g[i..], &mut m[i..], &mut v[i..],
                    b1, b2, bc1, bc2, lr, eps, scale);
    }
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

    /// One update over the parameter BLOCKS, in place.
    ///
    /// The flat [`Adam::step`] below needs its inputs contiguous, and a
    /// model does not store them that way. Reaching it therefore cost, per
    /// step at the shipping shape: a 29 MB gradient buffer built with a
    /// division per element, a 29 MB copy of every parameter in, and a
    /// 29 MB copy back out — about 116 MB of traffic and 7.24M divisions
    /// to satisfy an API shape. Measured together with the update itself,
    /// `optimizer + flatten + writeback` was **32.3 ms, 5.6% of a step**.
    ///
    /// This walks the blocks against a running offset into `m`/`v`
    /// instead. Nothing is flattened, nothing is copied, and the gradient
    /// scaling folds into the update. It is what PyTorch's `foreach` and
    /// `fused` Adam do for the same reason.
    ///
    /// `grads[i]` must match `params[i]` in length, and the blocks together
    /// must cover exactly the state this optimizer was sized for —
    /// otherwise the offsets would silently pair a parameter with another
    /// parameter's momentum.
    pub fn step_blocks(&mut self, params: &mut [Vec<f32>], grads: &[&[f32]], scale: f32)
        -> Result<(), String>
    {
        if params.len() != grads.len() {
            return Err(format!("Adam::step_blocks: {} parameter blocks, {} gradient blocks",
                               params.len(), grads.len()));
        }
        let total: usize = params.iter().map(|p| p.len()).sum();
        if total != self.m.len() {
            return Err(format!("Adam::step_blocks: sized for {} params, blocks total {total}",
                               self.m.len()));
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut off = 0usize;
        for (p, g) in params.iter_mut().zip(grads) {
            if p.len() != g.len() {
                return Err(format!("Adam::step_blocks: block is {} params, {} gradients",
                                   p.len(), g.len()));
            }
            let n = p.len();
            let (m, v) = (&mut self.m[off..off + n], &mut self.v[off..off + n]);
            #[cfg(target_arch = "x86_64")]
            if have_avx2() {
                // SAFETY: `have_avx2()` confirmed AVX2+FMA on this CPU, and
                // the four slices are all exactly `n` long (checked above).
                unsafe { adam_avx2(p, g, m, v, self.beta1, self.beta2, bc1, bc2,
                                   self.lr, self.eps, scale) };
                off += n;
                continue;
            }
            adam_scalar(p, g, m, v, self.beta1, self.beta2, bc1, bc2,
                        self.lr, self.eps, scale);
            off += n;
        }
        Ok(())
    }

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

    /// `step_blocks` must be BIT-IDENTICAL to the flat `step` it replaces,
    /// including the AVX2 path.
    ///
    /// Not "close": the loss trajectory is this project's regression check
    /// after every change, so an optimizer that drifts by a single ULP
    /// would make every later comparison ambiguous — a real regression and
    /// a rounding difference would look the same. Sizes here are chosen to
    /// exercise a block that is not a multiple of the 8-wide lane, so the
    /// scalar tail is covered too.
    #[test]
    fn step_blocks_is_bit_identical_to_the_flat_step() {
        let shapes = [1usize, 7, 8, 9, 1000, 4097];
        let total: usize = shapes.iter().sum();
        // Something with structure and both signs, so a sign or ordering
        // slip cannot pass by symmetry.
        let val = |i: usize| ((i % 37) as f32 - 18.0) * 0.031;
        let grd = |i: usize| ((i % 23) as f32 - 11.0) * 0.017;

        let mut flat_p: Vec<f32> = (0..total).map(val).collect();
        let flat_g: Vec<f32> = (0..total).map(grd).collect();
        let mut flat_opt = Adam::new(total, 0.003);

        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut gblocks: Vec<Vec<f32>> = Vec::new();
        let mut off = 0;
        for &n in &shapes {
            blocks.push((off..off + n).map(val).collect());
            gblocks.push((off..off + n).map(grd).collect());
            off += n;
        }
        let mut blk_opt = Adam::new(total, 0.003);

        // Several steps: bias correction changes with `t`, so a single
        // step would not catch a counter that advances differently.
        for step in 1..=5 {
            flat_opt.step(&mut flat_p, &flat_g).unwrap();
            let refs: Vec<&[f32]> = gblocks.iter().map(|g| g.as_slice()).collect();
            blk_opt.step_blocks(&mut blocks, &refs, 1.0).unwrap();

            let joined: Vec<f32> = blocks.iter().flatten().copied().collect();
            assert_eq!(joined, flat_p, "step {step}: block update diverged from the flat one");
            assert_eq!(blk_opt.m, flat_opt.m, "step {step}: first moments diverged");
            assert_eq!(blk_opt.v, flat_opt.v, "step {step}: second moments diverged");
            assert_eq!(blk_opt.t, flat_opt.t);
        }
    }

    /// `scale` divides the gradient, and must do it exactly as the caller's
    /// own division would — the non-uniform batch path relies on it.
    #[test]
    fn step_blocks_scale_matches_dividing_the_gradient_first() {
        let g: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.13).collect();
        let scale = 3.0f32;

        let mut a = vec![vec![0.5f32; 40]];
        let mut oa = Adam::new(40, 0.01);
        oa.step_blocks(&mut a, &[g.as_slice()], scale).unwrap();

        let pre: Vec<f32> = g.iter().map(|x| x / scale).collect();
        let mut b = vec![vec![0.5f32; 40]];
        let mut ob = Adam::new(40, 0.01);
        ob.step_blocks(&mut b, &[pre.as_slice()], 1.0).unwrap();

        assert_eq!(a, b, "scaling inside the update must match scaling before it");
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
