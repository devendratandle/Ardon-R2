//! r2-autograd — reverse-mode automatic differentiation over the
//! r2-tensor op set (Layer 4 of the trillion-scale architecture;
//! docs/LLM_TRILLION_ARCHITECTURE.md).
//!
//! A `Tape` records the forward computation as a DAG of ops; `backward()`
//! walks it in reverse, accumulating gradients into every leaf marked
//! `requires_grad`. The op set is the transformer-critical one (matmul,
//! elementwise add/mul, SiLU, RMSNorm, softmax-cross-entropy, MSE).
//!
//! THE ACCURACY DISCIPLINE (mirrors the differential harness for stats,
//! and the GPU-vs-CPU contract): every backward is checked against
//! FINITE DIFFERENCES in the tests. Analytic gradient must match the
//! numeric gradient to a tight tolerance, or it doesn't ship. A wrong
//! gradient trains a wrong model silently — so gradients are gated, not
//! trusted.
//!
//! Shard-awareness (a grad on a `TensorParallel` tensor triggering the
//! right collective) is the integration point with r2-mesh — wired by
//! r2-train / Opus; the local tape here is the correctness reference.

/// Index of a value node on the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

/// One recorded operation. Holds the input `Var`s and any constants the
/// backward pass needs. Backward math lives in `Tape::backward`.
enum Op {
    Leaf,
    Add(Var, Var),
    Mul(Var, Var),
    MatMul { a: Var, b: Var, m: usize, k: usize, n: usize },
    Silu(Var),
    /// RMSNorm over the last dim `d`, with weight and eps. Rows = len/d.
    Rmsnorm { x: Var, w: Var, d: usize, eps: f32 },
    /// Transpose a rows×cols matrix → cols×rows.
    Transpose { x: Var, rows: usize, cols: usize },
    /// Softmax over each `d`-wide row (a differentiable op, distinct from
    /// the fused SoftmaxCE loss — this one is used INSIDE attention).
    SoftmaxRows { x: Var, d: usize },
    /// Σ of all elements → scalar.
    SumAll(Var),
    /// Mean squared error against a constant target (target has no grad).
    Mse { pred: Var, target: Vec<f32> },
    /// Softmax over `d`-wide rows then cross-entropy against per-row class
    /// indices. Scalar loss; the classic fused backward (softmax − onehot).
    SoftmaxCE { logits: Var, d: usize, targets: Vec<usize> },
}

/// The autograd tape: values + grads + the op that produced each node.
pub struct Tape {
    vals: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
    ops: Vec<Op>,
    requires: Vec<bool>,
}

impl Tape {
    pub fn new() -> Self {
        Tape { vals: Vec::new(), grads: Vec::new(), ops: Vec::new(), requires: Vec::new() }
    }

    fn push(&mut self, val: Vec<f32>, op: Op, requires: bool) -> Var {
        let idx = self.vals.len();
        self.grads.push(vec![0.0; val.len()]);
        self.vals.push(val);
        self.ops.push(op);
        self.requires.push(requires);
        Var(idx)
    }

    /// A leaf parameter (or input). `requires_grad` marks it for gradient
    /// accumulation (weights = true; fixed inputs = false).
    pub fn leaf(&mut self, val: Vec<f32>, requires_grad: bool) -> Var {
        self.push(val, Op::Leaf, requires_grad)
    }

    pub fn value(&self, v: Var) -> &[f32] { &self.vals[v.0] }
    pub fn grad(&self, v: Var) -> &[f32] { &self.grads[v.0] }

    // ── forward ops (each records enough for backward) ─────────────────

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let val: Vec<f32> = self.vals[a.0].iter().zip(&self.vals[b.0]).map(|(x, y)| x + y).collect();
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::Add(a, b), req)
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let val: Vec<f32> = self.vals[a.0].iter().zip(&self.vals[b.0]).map(|(x, y)| x * y).collect();
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::Mul(a, b), req)
    }

    pub fn matmul(&mut self, a: Var, b: Var, m: usize, k: usize, n: usize) -> Var {
        let val = r2_tensor::ops::matmul(&self.vals[a.0], &self.vals[b.0], m, k, n);
        let req = self.requires[a.0] || self.requires[b.0];
        self.push(val, Op::MatMul { a, b, m, k, n }, req)
    }

    pub fn silu(&mut self, x: Var) -> Var {
        let val: Vec<f32> = self.vals[x.0].iter().map(|&v| r2_tensor::ops::silu(v)).collect();
        let req = self.requires[x.0];
        self.push(val, Op::Silu(x), req)
    }

    pub fn rmsnorm(&mut self, x: Var, w: Var, d: usize, eps: f32) -> Var {
        let val = r2_tensor::ops::rmsnorm(&self.vals[x.0], &self.vals[w.0], eps);
        let req = self.requires[x.0] || self.requires[w.0];
        self.push(val, Op::Rmsnorm { x, w, d, eps }, req)
    }

    pub fn transpose(&mut self, x: Var, rows: usize, cols: usize) -> Var {
        let vx = &self.vals[x.0];
        let mut val = vec![0.0f32; rows * cols];
        for i in 0..rows { for j in 0..cols { val[j * rows + i] = vx[i * cols + j]; } }
        let req = self.requires[x.0];
        self.push(val, Op::Transpose { x, rows, cols }, req)
    }

    pub fn softmax_rows(&mut self, x: Var, d: usize) -> Var {
        let val = r2_tensor::ops::softmax(&self.vals[x.0], d);
        let req = self.requires[x.0];
        self.push(val, Op::SoftmaxRows { x, d }, req)
    }

    pub fn sum_all(&mut self, x: Var) -> Var {
        let s: f32 = self.vals[x.0].iter().sum();
        let req = self.requires[x.0];
        self.push(vec![s], Op::SumAll(x), req)
    }

    pub fn mse(&mut self, pred: Var, target: Vec<f32>) -> Var {
        let n = target.len() as f32;
        let s: f32 = self.vals[pred.0].iter().zip(&target).map(|(p, t)| (p - t) * (p - t)).sum();
        let req = self.requires[pred.0];
        self.push(vec![s / n], Op::Mse { pred, target }, req)
    }

    pub fn softmax_ce(&mut self, logits: Var, d: usize, targets: Vec<usize>) -> Var {
        let sm = r2_tensor::ops::softmax(&self.vals[logits.0], d);
        let mut loss = 0.0f32;
        for (r, &t) in targets.iter().enumerate() {
            loss += -(sm[r * d + t].max(1e-30)).ln();
        }
        loss /= targets.len() as f32;
        let req = self.requires[logits.0];
        self.push(vec![loss], Op::SoftmaxCE { logits, d, targets }, req)
    }

    // ── backward: reverse-mode gradient accumulation ───────────────────

    /// Seed the given scalar output with grad 1 and propagate to all
    /// `requires_grad` leaves. `loss` must be a length-1 node.
    pub fn backward(&mut self, loss: Var) {
        assert_eq!(self.vals[loss.0].len(), 1, "backward() expects a scalar loss");
        for g in self.grads.iter_mut() { for x in g.iter_mut() { *x = 0.0; } }
        self.grads[loss.0][0] = 1.0;

        // Nodes were pushed in topological order → reverse index order is
        // a valid reverse-topological walk.
        for i in (0..self.ops.len()).rev() {
            // Take this node's incoming grad (clone: we index other nodes).
            let g = self.grads[i].clone();
            match &self.ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    let (a, b) = (a.0, b.0);
                    for (ga, gi) in self.grads[a].iter_mut().zip(&g) { *ga += gi; }
                    for (gb, gi) in self.grads[b].iter_mut().zip(&g) { *gb += gi; }
                }
                Op::Mul(a, b) => {
                    let (a, b) = (a.0, b.0);
                    let (va, vb) = (self.vals[a].clone(), self.vals[b].clone());
                    for (ga, (gi, vbi)) in self.grads[a].iter_mut().zip(g.iter().zip(&vb)) { *ga += gi * vbi; }
                    for (gb, (gi, vai)) in self.grads[b].iter_mut().zip(g.iter().zip(&va)) { *gb += gi * vai; }
                }
                Op::MatMul { a, b, m, k, n } => {
                    let (ai, bi, m, k, n) = (a.0, b.0, *m, *k, *n);
                    let va = self.vals[ai].clone();
                    let vb = self.vals[bi].clone();
                    // grad_A(m×k) = g(m×n) · B^T(n×k)
                    for i2 in 0..m { for p in 0..k {
                        let mut acc = 0.0f32;
                        for j in 0..n { acc += g[i2 * n + j] * vb[p * n + j]; }
                        self.grads[ai][i2 * k + p] += acc;
                    }}
                    // grad_B(k×n) = A^T(k×m) · g(m×n)
                    for p in 0..k { for j in 0..n {
                        let mut acc = 0.0f32;
                        for i2 in 0..m { acc += va[i2 * k + p] * g[i2 * n + j]; }
                        self.grads[bi][p * n + j] += acc;
                    }}
                }
                Op::Silu(x) => {
                    let xi = x.0;
                    let vx = self.vals[xi].clone();
                    for (gx, (gi, &v)) in self.grads[xi].iter_mut().zip(g.iter().zip(&vx)) {
                        let s = 1.0 / (1.0 + (-v).exp());
                        // d/dv [v*s] = s + v*s*(1-s)
                        *gx += gi * (s + v * s * (1.0 - s));
                    }
                }
                Op::Rmsnorm { x, w, d, eps } => {
                    let (xi, wi, d, eps) = (x.0, w.0, *d, *eps);
                    let vx = self.vals[xi].clone();
                    let vw = self.vals[wi].clone();
                    let rows = vx.len() / d;
                    for r in 0..rows {
                        let xr = &vx[r * d..r * d + d];
                        let gr = &g[r * d..r * d + d];
                        let ms = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
                        let rinv = 1.0 / (ms + eps).sqrt();
                        // s = Σ_j g_j w_j x_j
                        let s: f32 = (0..d).map(|j| gr[j] * vw[j] * xr[j]).sum();
                        let coef = rinv * rinv * rinv / d as f32;
                        for j in 0..d {
                            // dL/dx_i = g_i w_i r  -  r³ x_i/d * s
                            self.grads[xi][r * d + j] += gr[j] * vw[j] * rinv - coef * xr[j] * s;
                            // dL/dw_j = g_j * x_j * r
                            self.grads[wi][j] += gr[j] * xr[j] * rinv;
                        }
                    }
                }
                Op::Transpose { x, rows, cols } => {
                    let (xi, rows, cols) = (x.0, *rows, *cols);
                    // grad_x[i,j] += g[j,i]
                    for i in 0..rows { for j in 0..cols {
                        self.grads[xi][i * cols + j] += g[j * rows + i];
                    }}
                }
                Op::SoftmaxRows { x, d } => {
                    let (xi, d) = (x.0, *d);
                    let y = self.vals[i].clone(); // this node's value = softmax
                    let rows = y.len() / d;
                    for r in 0..rows {
                        let yr = &y[r * d..r * d + d];
                        let gr = &g[r * d..r * d + d];
                        // dot = Σ_j g_j y_j ; dL/dx_i = y_i (g_i − dot)
                        let dot: f32 = (0..d).map(|j| gr[j] * yr[j]).sum();
                        for j in 0..d {
                            self.grads[xi][r * d + j] += yr[j] * (gr[j] - dot);
                        }
                    }
                }
                Op::SumAll(x) => {
                    let xi = x.0;
                    for gx in self.grads[xi].iter_mut() { *gx += g[0]; }
                }
                Op::Mse { pred, target } => {
                    let pi = pred.0;
                    let n = target.len() as f32;
                    let vp = self.vals[pi].clone();
                    let target = target.clone();
                    for (gp, (p, t)) in self.grads[pi].iter_mut().zip(vp.iter().zip(&target)) {
                        *gp += g[0] * 2.0 * (p - t) / n;
                    }
                }
                Op::SoftmaxCE { logits, d, targets } => {
                    let (li, d, targets) = (logits.0, *d, targets.clone());
                    let sm = r2_tensor::ops::softmax(&self.vals[li], d);
                    let inv = g[0] / targets.len() as f32;
                    // grad = (softmax − onehot) / batch, scaled by upstream g.
                    for (r, &t) in targets.iter().enumerate() {
                        for j in 0..d {
                            let mut val = sm[r * d + j];
                            if j == t { val -= 1.0; }
                            self.grads[li][r * d + j] += inv * val;
                        }
                    }
                }
            }
        }
    }
}

impl Default for Tape { fn default() -> Self { Tape::new() } }

/// Finite-difference gradient of a scalar function of `params` — the
/// numeric reference the analytic backward is checked against. Central
/// difference (O(h²)), h chosen for f32. Rebuilds the graph each eval via
/// the caller's closure.
pub fn finite_diff<F: Fn(&[f32]) -> f32>(params: &[f32], f: F) -> Vec<f32> {
    let h = 1e-3f32;
    let mut g = vec![0.0f32; params.len()];
    let mut p = params.to_vec();
    for i in 0..params.len() {
        let orig = p[i];
        p[i] = orig + h; let fp = f(&p);
        p[i] = orig - h; let fm = f(&p);
        p[i] = orig;
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert analytic grad (from a fresh tape built by `build`) matches
    /// the finite-difference grad of the same scalar function.
    fn check_grad<B>(params: &[f32], build: B)
    where B: Fn(&mut Tape, &[f32]) -> (Var, Var) {
        // Analytic: build tape, backward, read leaf grad.
        let mut t = Tape::new();
        let (leaf, loss) = build(&mut t, params);
        t.backward(loss);
        let analytic = t.grad(leaf).to_vec();
        // Numeric: scalar loss as a function of the leaf's params.
        let numeric = finite_diff(params, |p| {
            let mut t = Tape::new();
            let (_, loss) = build(&mut t, p);
            t.value(loss)[0]
        });
        let maxerr = analytic.iter().zip(&numeric)
            .map(|(a, n)| (a - n).abs()).fold(0.0f32, f32::max);
        assert!(maxerr < 2e-2, "grad mismatch: analytic {:?} numeric {:?}", analytic, numeric);
    }

    #[test]
    fn add_mul_chain_grad() {
        check_grad(&[1.5, -2.0, 0.5], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let c = t.leaf(vec![2.0, 3.0, -1.0], false);
            let y = t.mul(x, c);       // x*c
            let z = t.add(y, x);       // x*c + x
            let loss = t.sum_all(z);
            (x, loss)
        });
    }

    #[test]
    fn matmul_grad() {
        check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
            let a = t.leaf(p.to_vec(), true);          // 2×3
            let b = t.leaf(vec![1.0, 0.5, -1.0, 2.0, 0.0, 1.5], false); // 3×2
            let c = t.matmul(a, b, 2, 3, 2);           // 2×2
            let loss = t.sum_all(c);
            (a, loss)
        });
    }

    #[test]
    fn silu_grad() {
        check_grad(&[-1.0, 0.3, 2.0, -0.5], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let y = t.silu(x);
            let loss = t.sum_all(y);
            (x, loss)
        });
    }

    #[test]
    fn rmsnorm_grad_wrt_x() {
        check_grad(&[0.5, -1.5, 2.0, 0.25], |t, p| {
            let x = t.leaf(p.to_vec(), true);
            let w = t.leaf(vec![1.0, 0.5, 1.5, 2.0], false);
            let y = t.rmsnorm(x, w, 4, 1e-5);
            let loss = t.sum_all(y);
            (x, loss)
        });
    }

    #[test]
    fn rmsnorm_grad_wrt_w() {
        check_grad(&[1.0, 0.5, 1.5, 2.0], |t, p| {
            let x = t.leaf(vec![0.5, -1.5, 2.0, 0.25], false);
            let w = t.leaf(p.to_vec(), true);
            let y = t.rmsnorm(x, w, 4, 1e-5);
            let loss = t.sum_all(y);
            (w, loss)
        });
    }

    #[test]
    fn mse_grad() {
        check_grad(&[0.2, 0.8, -0.4], |t, p| {
            let pred = t.leaf(p.to_vec(), true);
            let loss = t.mse(pred, vec![1.0, 0.0, -1.0]);
            (pred, loss)
        });
    }

    #[test]
    fn transpose_grad() {
        check_grad(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |t, p| {
            let x = t.leaf(p.to_vec(), true);       // 2×3
            let xt = t.transpose(x, 2, 3);          // 3×2
            let c = t.leaf(vec![1.0, -2.0, 0.5, 1.5, -1.0, 2.0], false);
            let prod = t.mul(xt, c);
            let loss = t.sum_all(prod);
            (x, loss)
        });
    }

    #[test]
    fn softmax_rows_grad() {
        check_grad(&[1.0, 2.0, 0.5, -1.0, 0.3, 2.0], |t, p| {
            let x = t.leaf(p.to_vec(), true);       // 2 rows × 3
            let sm = t.softmax_rows(x, 3);
            let c = t.leaf(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.5], false);
            let prod = t.mul(sm, c);
            let loss = t.sum_all(prod);
            (x, loss)
        });
    }

    #[test]
    fn softmax_cross_entropy_grad() {
        // 2 rows × 3 classes; targets [0, 2].
        check_grad(&[2.0, 1.0, 0.1, -1.0, 0.5, 3.0], |t, p| {
            let logits = t.leaf(p.to_vec(), true);
            let loss = t.softmax_ce(logits, 3, vec![0, 2]);
            (logits, loss)
        });
    }

    #[test]
    fn tiny_mlp_trains() {
        // A 1-layer net loss must DECREASE under gradient steps — proves
        // forward+backward compose into real learning.
        let mut w = vec![0.1f32; 6]; // 3→2
        let x = vec![1.0, 2.0, -1.0]; // 1×3
        let target = vec![1.0, -1.0];
        let mut prev = f32::INFINITY;
        for _ in 0..50 {
            let mut t = Tape::new();
            let wv = t.leaf(w.clone(), true);
            let xv = t.leaf(x.clone(), false);
            let y = t.matmul(xv, wv, 1, 3, 2);
            let a = t.silu(y);
            let loss = t.mse(a, target.clone());
            t.backward(loss);
            let g = t.grad(wv).to_vec();
            for (wi, gi) in w.iter_mut().zip(&g) { *wi -= 0.1 * gi; }
            let l = t.value(loss)[0];
            assert!(l <= prev + 1e-5, "loss went up: {} -> {}", prev, l);
            prev = l;
        }
        assert!(prev < 1.0, "final loss {}", prev);
    }
}
