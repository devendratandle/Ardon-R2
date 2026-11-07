//! A look inside the model on one batch: gradient and activation summaries,
//! per block, on request.
//!
//! Nothing here runs during training. `Trainer::train_step` keeps no
//! activations and frees its gradients off the thread, because keeping them
//! would cost a step's memory, and a pass over every gradient per step (see
//! `Trainer::last_grad_norm`). So an inspection is its own forward and
//! backward, on the batch it is given:
//!
//! - **time:** one extra training pass without the optimizer, per call;
//! - **memory:** the tape of that pass, values and gradients, about what a
//!   step already peaks at. The weights are lent to the tape, not copied.
//!
//! A run on the GPU is inspected here too, from weights downloaded to the
//! trainer: `GpuTrainer` matches this tape's loss and every gradient block to
//! f32 rounding, and it reuses one activation set per layer that holds only
//! the last batch's values, so reading the device would save no pass.

use r2_autograd::{Tape, Var};

use crate::llm::Trainer;

/// Mean, spread and extremes of one tensor. Computed in f64, in a fixed
/// order, so the same values give the same summary on every machine.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    /// Largest absolute value among the finite ones.
    pub max_abs: f64,
    /// How many values are NaN or infinite; left out of the other figures.
    pub nonfinite: usize,
}

impl Summary {
    pub fn of(values: &[f32]) -> Summary {
        let (mut n, mut sum, mut sq, mut max_abs) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        for &x in values {
            if !x.is_finite() { continue; }
            let x = x as f64;
            n += 1;
            sum += x;
            sq += x * x;
            max_abs = max_abs.max(x.abs());
        }
        let mean = if n > 0 { sum / n as f64 } else { f64::NAN };
        // population sd: the tensor is the whole of what is being described
        let sd = if n > 0 { (sq / n as f64 - mean * mean).max(0.0).sqrt() } else { f64::NAN };
        Summary { n: values.len(), mean, sd, max_abs, nonfinite: values.len() - n }
    }
}

/// One parameter block.
#[derive(Debug, Clone)]
pub struct BlockStats {
    pub name: String,
    pub params: usize,
    /// False for a frozen block, which has no gradient.
    pub trainable: bool,
    pub weight_norm: f64,
    pub grad_norm: Option<f64>,
    pub grad_max: Option<f64>,
}

/// One activation: the output of a stage of the forward pass.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub stage: String,
    pub summary: Summary,
}

#[derive(Debug, Clone)]
pub struct Inspection {
    pub loss: f32,
    pub blocks: Vec<BlockStats>,
    pub activations: Vec<ActivationStats>,
}

fn l2(values: &[f32]) -> f64 {
    values.iter().map(|&x| { let x = x as f64; x * x }).sum::<f64>().sqrt()
}

impl Trainer {
    /// Summaries of one forward and backward pass on `batch`, equal-length
    /// sequences only. Changes no weight and no optimizer state: training
    /// after an inspection is bit-identical to training without one.
    pub fn inspect(&mut self, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<Inspection, String> {
        if batch.is_empty() { return Err("inspect: empty batch".into()); }
        let seq = batch[0].0.len();
        if !batch.iter().all(|(i, t)| i.len() == seq && t.len() == seq) {
            return Err("inspect: needs sequences of equal length".into());
        }
        let mut toks = Vec::with_capacity(batch.len() * seq);
        let mut tgts = Vec::with_capacity(batch.len() * seq);
        for (i, t) in batch { toks.extend_from_slice(i); tgts.extend_from_slice(t); }

        let names = self.block_names();
        let trainable: Vec<bool> = (0..self.params.len()).map(|i| !self.is_frozen(i)).collect();
        let mut tape = Tape::with_pool(std::mem::take(&mut self.pool));
        let leaves: Vec<Var> = self.params.iter_mut().zip(&trainable)
            .map(|(p, &train)| tape.leaf(std::mem::take(p), train)).collect();

        let mut stages = Vec::new();
        let logits = self.forward_traced(&mut tape, &toks, seq, &leaves, Some(&mut stages));
        let loss_var = tape.softmax_ce(logits, self.cfg.vocab, tgts);
        let loss = tape.value(loss_var)[0];
        tape.backward(loss_var);

        let activations = stages.into_iter()
            .map(|(stage, v)| ActivationStats { stage, summary: Summary::of(tape.value(v)) })
            .collect();
        let blocks = leaves.iter().enumerate().map(|(i, &lv)| {
            let grad = if trainable[i] { Some(tape.grad(lv)) } else { None };
            let value = tape.value(lv);
            BlockStats {
                name: names[i].clone(),
                params: value.len(),
                trainable: trainable[i],
                weight_norm: l2(value),
                grad_norm: grad.map(l2),
                grad_max: grad.map(|g| Summary::of(g).max_abs),
            }
        }).collect();

        for (i, &lv) in leaves.iter().enumerate() {
            self.params[i] = tape.take_value(lv);
        }
        self.pool = tape.into_pool();
        Ok(Inspection { loss, blocks, activations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2_tensor::model::Config;

    fn trainer() -> Trainer {
        let cfg = Config { dim: 16, n_heads: 2, n_kv_heads: 1, n_layers: 2, vocab: 32,
                           ffn_hidden: 32, max_seq: 16, rope_base: 10000.0, eps: 1e-5 };
        Trainer::new(cfg, 0.01, 7).unwrap()
    }

    fn batch() -> Vec<(Vec<usize>, Vec<usize>)> {
        (0..3).map(|b| ((0..8).map(|i| (i + b) % 32).collect(), (1..9).map(|i| (i + b) % 32).collect())).collect()
    }

    #[test]
    fn the_summary_matches_a_hand_computation() {
        let s = Summary::of(&[1.0, -3.0, f32::NAN, f32::INFINITY]);
        assert_eq!((s.n, s.nonfinite), (4, 2));
        assert_eq!((s.mean, s.sd, s.max_abs), (-1.0, 2.0, 3.0));
    }

    #[test]
    fn gradients_and_the_loss_are_those_of_the_tape() {
        let mut t = trainer();
        let (loss, _, grads) = t.loss_logits_grads(&batch()).unwrap();
        let look = t.inspect(&batch()).unwrap();
        assert_eq!(look.loss, loss);
        assert_eq!(look.blocks.len(), grads.len());
        for (b, g) in look.blocks.iter().zip(&grads) {
            assert_eq!(b.grad_norm, Some(l2(g)), "{}", b.name);
            assert_eq!(b.params, g.len());
        }
    }

    #[test]
    fn every_layer_reports_its_stages() {
        let look = trainer().inspect(&batch()).unwrap();
        let stages: Vec<&str> = look.activations.iter().map(|a| a.stage.as_str()).collect();
        assert_eq!(stages, ["embed", "layer.0.attn", "layer.0.ffn", "layer.0.out",
                            "layer.1.attn", "layer.1.ffn", "layer.1.out", "final"]);
        assert!(look.activations.iter().all(|a| a.summary.n > 0 && a.summary.sd > 0.0));
    }

    #[test]
    fn a_frozen_block_has_no_gradient() {
        let mut t = trainer();
        t.freeze(&["tok_embed"]).unwrap();
        let look = t.inspect(&batch()).unwrap();
        assert!(!look.blocks[0].trainable && look.blocks[0].grad_norm.is_none());
        assert!(look.blocks[0].weight_norm > 0.0);
        assert!(look.blocks[1].grad_norm.is_some());
    }

    /// The promise that makes it safe to call from a callback.
    #[test]
    fn inspecting_between_steps_leaves_training_bit_identical() {
        let (mut plain, mut looked) = (trainer(), trainer());
        for _ in 0..5 {
            let a = plain.train_step(&batch()).unwrap();
            looked.inspect(&batch()).unwrap();
            let b = looked.train_step(&batch()).unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(plain.params, looked.params);
        assert_eq!(plain.opt.m, looked.opt.m);
        assert_eq!(plain.opt.v, looked.opt.v);
    }
}
