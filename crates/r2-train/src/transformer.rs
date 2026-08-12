//! End-to-end proof: a real causal decoder transformer built ENTIRELY
//! from the Ardon-R2 foundation layers — r2-tensor ops, r2-autograd tape,
//! r2-train step — in pure Rust, no Python, no framework.
//!
//! This is the artifact that turns "the architecture exists" into "we
//! trained a transformer with it." A tiny single-head causal-attention
//! model (RMSNorm → self-attention → SwiGLU FFN, ×2 blocks, residuals)
//! learns a next-token task; the test asserts the loss drops far below
//! the untrained baseline. Every gradient flowing through it is
//! finite-difference-verified by the r2-autograd test suite, so this is
//! learning on VERIFIED gradients — the project's accuracy discipline
//! carried into training.
//!
//! Deliberately small (D=16, T=6, vocab=12, 2 blocks) so it trains in a
//! second on a CPU with no GPU. The SAME code — same ops, same tape — is
//! what the mesh (r2-mesh Shard/Collective) scales to 32B/1T: only the
//! tensor sizes and the transport change, not the math.

use r2_autograd::{Tape, Var};

/// Learnable parameters of the toy transformer (all flat f32 vectors).
pub struct Model {
    pub d: usize,      // embed dim
    pub t: usize,      // sequence length
    pub vocab: usize,
    pub h: usize,      // FFN hidden
    pub params: Vec<Vec<f32>>, // flat param blocks (see `layout`)
}

/// Xorshift RNG for reproducible init.
struct R(u64);
impl R {
    fn n(&mut self) -> f32 {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

impl Model {
    /// Parameter blocks, in order (index = position in `params`):
    /// 0 tok_embed (vocab×d)   1 pos_embed (t×d)
    /// per block b (2 blocks), 9 blocks each starting at 2 + b*9:
    ///   +0 norm1 (d)  +1 Wq +2 Wk +3 Wv +4 Wo (each d×d)
    ///   +5 norm2 (d)  +6 W1 (d×h) +7 W3 (d×h) +8 W2 (h×d)
    /// last: 2 + 2*9 = 20  Wout (d×vocab)
    pub fn new(d: usize, t: usize, vocab: usize, h: usize, seed: u64) -> Self {
        let mut r = R(seed.max(1));
        let scale = (1.0 / d as f32).sqrt();
        let mut mk = |n: usize, s: f32| (0..n).map(|_| r.n() * s).collect::<Vec<f32>>();
        let mut params = Vec::new();
        params.push(mk(vocab * d, 0.1));   // tok_embed
        params.push(mk(t * d, 0.1));       // pos_embed
        for _ in 0..2 {
            params.push(vec![1.0; d]);          // norm1 (init 1)
            params.push(mk(d * d, scale));      // Wq
            params.push(mk(d * d, scale));      // Wk
            params.push(mk(d * d, scale));      // Wv
            params.push(mk(d * d, scale));      // Wo
            params.push(vec![1.0; d]);          // norm2
            params.push(mk(d * h, scale));      // W1
            params.push(mk(d * h, scale));      // W3
            params.push(mk(h * d, scale));      // W2
        }
        params.push(mk(d * vocab, scale));      // Wout
        Model { d, t, vocab, h, params }
    }

    #[allow(dead_code)] // superseded by llm::Trainer; kept for the older demo path
    fn n_params(&self) -> usize { self.params.iter().map(|p| p.len()).sum() }

    /// Build the forward graph on a fresh tape for input token ids `ids`
    /// (len t) with next-token `targets` (len t); return (leaf vars, loss).
    /// Leaf vars are returned in the same order as `self.params` so grads
    /// map back cleanly.
    fn forward(&self, tape: &mut Tape, ids: &[usize], targets: &[usize]) -> (Vec<Var>, Var) {
        let (d, t, vocab, h) = (self.d, self.t, self.vocab, self.h);
        // Register every param block as a requires_grad leaf.
        let leaves: Vec<Var> = self.params.iter()
            .map(|p| tape.leaf(p.clone(), true)).collect();

        // Embedding via one-hot matmul (differentiable, reuses matmul):
        // onehot(t×vocab) · tok_embed(vocab×d) = (t×d).
        let mut onehot = vec![0.0f32; t * vocab];
        for (i, &id) in ids.iter().enumerate() { onehot[i * vocab + id] = 1.0; }
        let oh = tape.leaf(onehot, false);
        let tok = tape.matmul(oh, leaves[0], t, vocab, d);
        let mut x = tape.add(tok, leaves[1]); // + pos_embed (t×d)

        // Causal mask constant (t×t): 0 on allowed, big-negative on future.
        let mut mask = vec![0.0f32; t * t];
        for i in 0..t { for j in 0..t { if j > i { mask[i * t + j] = -1e9; } } }
        let inv_sqrt_d = vec![1.0 / (d as f32).sqrt(); t * t];

        for b in 0..2 {
            let base = 2 + b * 9;
            // ── self-attention sublayer ──
            let normed = tape.rmsnorm(x, leaves[base], d, 1e-5);
            let q = tape.matmul(normed, leaves[base + 1], t, d, d);
            let k = tape.matmul(normed, leaves[base + 2], t, d, d);
            let v = tape.matmul(normed, leaves[base + 3], t, d, d);
            let kt = tape.transpose(k, t, d);                 // d×t
            let scores = tape.matmul(q, kt, t, d, t);         // t×t
            let sc = tape.leaf(inv_sqrt_d.clone(), false);
            let scaled = tape.mul(scores, sc);
            let mv = tape.leaf(mask.clone(), false);
            let masked = tape.add(scaled, mv);
            let attn = tape.softmax_rows(masked, t);          // t×t
            let ctx = tape.matmul(attn, v, t, t, d);          // t×d
            let out = tape.matmul(ctx, leaves[base + 4], t, d, d);
            x = tape.add(x, out);                             // residual

            // ── SwiGLU FFN sublayer ──
            let normed2 = tape.rmsnorm(x, leaves[base + 5], d, 1e-5);
            let g = tape.matmul(normed2, leaves[base + 6], t, d, h);
            let u = tape.matmul(normed2, leaves[base + 7], t, d, h);
            let sg = tape.silu(g);
            let hh = tape.mul(sg, u);                         // silu(g)*u
            let ffn = tape.matmul(hh, leaves[base + 8], t, h, d);
            x = tape.add(x, ffn);                             // residual
        }

        // Output projection → logits (t×vocab), then softmax-CE loss.
        let logits = tape.matmul(x, leaves[20], t, d, vocab);
        let loss = tape.softmax_ce(logits, vocab, targets.to_vec());
        (leaves, loss)
    }

    /// One SGD step over a batch of (ids, targets) sequences; returns the
    /// mean loss. Grads for each param block accumulate across the batch.
    pub fn train_step(&mut self, batch: &[(Vec<usize>, Vec<usize>)], lr: f32) -> f32 {
        let np = self.params.len();
        let mut grad_acc: Vec<Vec<f32>> = self.params.iter().map(|p| vec![0.0; p.len()]).collect();
        let mut total = 0.0f32;
        for (ids, targets) in batch {
            let mut tape = Tape::new();
            let (leaves, loss) = self.forward(&mut tape, ids, targets);
            tape.backward(loss);
            total += tape.value(loss)[0];
            for p in 0..np {
                let g = tape.grad(leaves[p]);
                for (a, gi) in grad_acc[p].iter_mut().zip(g) { *a += gi; }
            }
        }
        let scale = lr / batch.len() as f32;
        for p in 0..np {
            for (w, g) in self.params[p].iter_mut().zip(&grad_acc[p]) { *w -= scale * g; }
        }
        total / batch.len() as f32
    }

    /// Loss on one sequence without training (baseline / eval).
    pub fn eval_loss(&self, ids: &[usize], targets: &[usize]) -> f32 {
        let mut tape = Tape::new();
        let (_, loss) = self.forward(&mut tape, ids, targets);
        tape.value(loss)[0]
    }

    /// Predicted next token at each position (argmax of the logits) — for
    /// checking the model actually learned the RULE, not just low loss.
    /// Forward through logits only (no loss node).
    pub fn predict(&self, ids: &[usize]) -> Vec<usize> {
        let (d, t, vocab, h) = (self.d, self.t, self.vocab, self.h);
        let mut tape = Tape::new();
        let leaves: Vec<Var> = self.params.iter().map(|p| tape.leaf(p.clone(), false)).collect();
        let mut onehot = vec![0.0f32; t * vocab];
        for (i, &id) in ids.iter().enumerate() { onehot[i * vocab + id] = 1.0; }
        let oh = tape.leaf(onehot, false);
        let tok = tape.matmul(oh, leaves[0], t, vocab, d);
        let mut x = tape.add(tok, leaves[1]);
        let mut mask = vec![0.0f32; t * t];
        for i in 0..t { for j in 0..t { if j > i { mask[i * t + j] = -1e9; } } }
        let inv = vec![1.0 / (d as f32).sqrt(); t * t];
        for b in 0..2 {
            let base = 2 + b * 9;
            let normed = tape.rmsnorm(x, leaves[base], d, 1e-5);
            let q = tape.matmul(normed, leaves[base + 1], t, d, d);
            let k = tape.matmul(normed, leaves[base + 2], t, d, d);
            let v = tape.matmul(normed, leaves[base + 3], t, d, d);
            let kt = tape.transpose(k, t, d);
            let scores = tape.matmul(q, kt, t, d, t);
            let sc = tape.leaf(inv.clone(), false);
            let scaled = tape.mul(scores, sc);
            let mv = tape.leaf(mask.clone(), false);
            let masked = tape.add(scaled, mv);
            let attn = tape.softmax_rows(masked, t);
            let ctx = tape.matmul(attn, v, t, t, d);
            let out = tape.matmul(ctx, leaves[base + 4], t, d, d);
            x = tape.add(x, out);
            let normed2 = tape.rmsnorm(x, leaves[base + 5], d, 1e-5);
            let g = tape.matmul(normed2, leaves[base + 6], t, d, h);
            let u = tape.matmul(normed2, leaves[base + 7], t, d, h);
            let sg = tape.silu(g);
            let hh = tape.mul(sg, u);
            let ffn = tape.matmul(hh, leaves[base + 8], t, h, d);
            x = tape.add(x, ffn);
        }
        let logits = tape.matmul(x, leaves[20], t, d, vocab);
        let lg = tape.value(logits);
        (0..t).map(|r| {
            let row = &lg[r * vocab..r * vocab + vocab];
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE END-TO-END PROOF. A causal transformer, built only from the
    /// foundation layers, learns a deterministic next-token rule
    /// (target = (id + 1) mod vocab) on fixed sequences. Loss must fall
    /// far below the random baseline ln(vocab) — i.e. it genuinely
    /// learned, on finite-difference-verified gradients, in pure Rust.
    #[test]
    fn transformer_learns_next_token_rule() {
        let (d, t, vocab, h) = (16usize, 6usize, 12usize, 32usize);
        let mut model = Model::new(d, t, vocab, h, 1234);

        // Deterministic task: each token's target is the next id (mod vocab).
        let make = |start: usize| {
            let ids: Vec<usize> = (0..t).map(|i| (start + i) % vocab).collect();
            let targets: Vec<usize> = ids.iter().map(|&x| (x + 1) % vocab).collect();
            (ids, targets)
        };
        let batch: Vec<_> = (0..vocab).map(make).collect();

        let random_baseline = (vocab as f32).ln(); // ≈ 2.485
        let start_loss = model.eval_loss(&batch[0].0, &batch[0].1);

        let mut last = f32::INFINITY;
        for epoch in 0..400 {
            last = model.train_step(&batch, 0.05);
            assert!(last.is_finite(), "loss diverged at epoch {}", epoch);
        }

        // It must have learned: final loss far below random guessing.
        assert!(last < random_baseline * 0.5,
            "transformer did not learn: final loss {} vs baseline {} (start {})",
            last, random_baseline, start_loss);

        // And it must actually PREDICT the rule: argmax of each position's
        // logits equals the correct next token, across all sequences.
        let mut correct = 0usize;
        let mut total = 0usize;
        for (ids, targets) in &batch {
            let pred = model.predict(ids);
            for (p, t) in pred.iter().zip(targets) { total += 1; if p == t { correct += 1; } }
        }
        let acc = correct as f32 / total as f32;
        assert!(acc > 0.9, "prediction accuracy only {:.2} ({}/{})", acc, correct, total);

        eprintln!("[transformer proof] start_loss={:.3} final_loss={:.3} baseline={:.3} accuracy={:.1}%",
            start_loss, last, random_baseline, acc * 100.0);
    }

    #[test]
    fn transformer_params_count_sane() {
        let m = Model::new(16, 6, 12, 32, 1);
        // 2 embeddings + 2 blocks×9 + output = 21 param blocks.
        assert_eq!(m.params.len(), 21);
        assert!(m.n_params() > 0);
    }
}
