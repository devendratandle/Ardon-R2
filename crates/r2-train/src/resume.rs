//! A [`Trainer`] on disk: continue a run exactly, or start from saved weights.
//!
//! Two different things, kept apart on purpose:
//!
//! * **Resume** ([`Trainer::save_checkpoint`] / [`Trainer::resume_latest`]):
//!   weights, Adam state and every setting. The continued run is bit for bit
//!   the run that was interrupted, because a batch is a function of the step
//!   index (`crate::data`) and all the rest of the state is here.
//! * **Fine-tune** ([`Trainer::from_model`]): weights only, from a saved
//!   model, with a new optimizer. A different run that starts where another
//!   finished.

use std::path::{Path, PathBuf};

use r2_tensor::model::{Config, Layer, Model};

use crate::checkpoint::{load_latest, save_parts_rotating, Checkpoint};
use crate::llm::Trainer;
use crate::optim::{Adam, LrSchedule};

/// `Config` as scalars. Each value is an integer or an f32, so f64 holds it
/// exactly.
fn config_scalars(c: &Config) -> [(&'static str, f64); 9] {
    [("cfg.dim", c.dim as f64), ("cfg.n_heads", c.n_heads as f64), ("cfg.n_kv_heads", c.n_kv_heads as f64),
     ("cfg.n_layers", c.n_layers as f64), ("cfg.vocab", c.vocab as f64), ("cfg.ffn_hidden", c.ffn_hidden as f64),
     ("cfg.max_seq", c.max_seq as f64), ("cfg.rope_base", c.rope_base as f64), ("cfg.eps", c.eps as f64)]
}

fn config_from(ck: &Checkpoint) -> Result<Config, String> {
    let n = |k: &str| ck.scalar(k).map(|v| v as usize);
    Ok(Config {
        dim: n("cfg.dim")?, n_heads: n("cfg.n_heads")?, n_kv_heads: n("cfg.n_kv_heads")?,
        n_layers: n("cfg.n_layers")?, vocab: n("cfg.vocab")?, ffn_hidden: n("cfg.ffn_hidden")?,
        max_seq: n("cfg.max_seq")?, rope_base: ck.scalar("cfg.rope_base")? as f32,
        eps: ck.scalar("cfg.eps")? as f32,
    })
}

impl Trainer {
    /// Everything needed to continue, as `dir/step-<n>.ckpt`, keeping the
    /// newest `keep` (0 keeps all). Writes the tensors from where they live.
    pub fn save_checkpoint(&self, dir: impl AsRef<Path>, keep: usize) -> Result<PathBuf, String> {
        let names: Vec<String> = self.block_names().iter().map(|n| format!("p.{n}")).collect();
        let mut tensors: Vec<(&str, &[f32])> = names.iter().map(String::as_str)
            .zip(self.params.iter().map(Vec::as_slice)).collect();
        tensors.push(("adam.m", &self.opt.m));
        tensors.push(("adam.v", &self.opt.v));
        let o = &self.opt;
        let mut scalars: Vec<(&str, f64)> = config_scalars(&self.cfg).to_vec();
        scalars.extend([("adam.t", o.t as f64), ("adam.lr", o.lr as f64), ("adam.beta1", o.beta1 as f64),
                        ("adam.beta2", o.beta2 as f64), ("adam.eps", o.eps as f64),
                        ("adam.weight_decay", o.weight_decay as f64)]);
        if let Some(c) = self.clip { scalars.push(("clip", c as f64)); }
        if let Some(s) = self.sched {
            scalars.extend([("sched.base", s.base as f64), ("sched.min", s.min as f64),
                            ("sched.warmup", s.warmup as f64), ("sched.total", s.total as f64)]);
        }
        save_parts_rotating(dir, self.step, keep, &tensors, &scalars)
    }

    /// Rebuild the trainer a checkpoint was taken from, moving its tensors
    /// rather than copying them.
    pub fn from_checkpoint(mut ck: Checkpoint) -> Result<Trainer, String> {
        let cfg = config_from(&ck)?;
        let mut t = Trainer::empty(cfg)?;
        let sizes = Trainer::block_shapes(&cfg);
        for (i, name) in t.block_names().into_iter().enumerate() {
            let p = ck.take_tensor(&format!("p.{name}"))?;
            if p.len() != sizes[i] {
                return Err(format!("checkpoint: '{name}' holds {} values, the config says {}", p.len(), sizes[i]));
            }
            t.params[i] = p;
        }
        let (m, v) = (ck.take_tensor("adam.m")?, ck.take_tensor("adam.v")?);
        let total: usize = sizes.iter().sum();
        if m.len() != total || v.len() != total {
            return Err(format!("checkpoint: Adam state holds {}/{} values for {total} parameters", m.len(), v.len()));
        }
        t.opt = Adam {
            m, v,
            t: ck.scalar("adam.t")? as u64, lr: ck.scalar("adam.lr")? as f32,
            beta1: ck.scalar("adam.beta1")? as f32, beta2: ck.scalar("adam.beta2")? as f32,
            eps: ck.scalar("adam.eps")? as f32, weight_decay: ck.scalar("adam.weight_decay")? as f32,
        };
        t.step = ck.step;
        t.clip = ck.scalar("clip").ok().map(|c| c as f32);
        if ck.scalar("sched.base").is_ok() {
            t.sched = Some(LrSchedule {
                base: ck.scalar("sched.base")? as f32, min: ck.scalar("sched.min")? as f32,
                warmup: ck.scalar("sched.warmup")? as u64, total: ck.scalar("sched.total")? as u64,
            });
        }
        Ok(t)
    }

    /// The trainer from the newest checkpoint in `dir`; `None` if there is none.
    pub fn resume_latest(dir: impl AsRef<Path>) -> Result<Option<Trainer>, String> {
        load_latest(dir)?.map(Trainer::from_checkpoint).transpose()
    }

    /// Start a NEW run from saved weights: the weights move in, the optimizer
    /// starts empty at `lr`. This is fine-tuning's starting point.
    pub fn from_model(m: Model, lr: f32) -> Result<Trainer, String> {
        m.validate()?;
        let mut t = Trainer::empty(m.cfg)?;
        t.params[0] = m.tok_embed;
        t.params[1] = m.final_norm;
        t.params[2] = m.output;
        for (l, ly) in m.layers.into_iter().enumerate() {
            let parts = [ly.attn_norm, ly.wq, ly.wk, ly.wv, ly.wo, ly.ffn_norm, ly.w1, ly.w2, ly.w3];
            for (p, data) in parts.into_iter().enumerate() {
                t.params[3 + l * 9 + p] = data;
            }
        }
        t.opt = Adam::new(t.n_params(), lr);
        Ok(t)
    }

    /// Lend the weights to code that needs a [`Model`], without copying them:
    /// they move into a `Model` for the call and back afterwards.
    pub fn with_model<R>(&mut self, f: impl FnOnce(&Model) -> R) -> R {
        let c = self.cfg;
        let mut take = |i: usize| std::mem::take(&mut self.params[i]);
        let m = Model {
            cfg: c,
            tok_embed: take(0), final_norm: take(1), output: take(2),
            layers: (0..c.n_layers).map(|l| {
                let mut g = |p: usize| take(3 + l * 9 + p);
                Layer { attn_norm: g(0), wq: g(1), wk: g(2), wv: g(3), wo: g(4),
                        ffn_norm: g(5), w1: g(6), w2: g(7), w3: g(8) }
            }).collect(),
        };
        let out = f(&m);
        self.params[0] = m.tok_embed;
        self.params[1] = m.final_norm;
        self.params[2] = m.output;
        for (l, ly) in m.layers.into_iter().enumerate() {
            let parts = [ly.attn_norm, ly.wq, ly.wk, ly.wv, ly.wo, ly.ffn_norm, ly.w1, ly.w2, ly.w3];
            for (p, data) in parts.into_iter().enumerate() {
                self.params[3 + l * 9 + p] = data;
            }
        }
        out
    }

    /// Save as a servable model directory (`Model::save_dir`), without
    /// copying the weights.
    pub fn save_dir(&mut self, dir: impl AsRef<Path>) -> Result<(), String> {
        self.with_model(|m| m.save_dir(dir))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Batcher, Order};
    use crate::optim::LrSchedule;

    fn cfg() -> Config {
        Config { dim: 16, n_heads: 2, n_kv_heads: 1, n_layers: 2, vocab: 40, ffn_hidden: 32,
                 max_seq: 16, rope_base: 10000.0, eps: 1e-5 }
    }

    fn corpus() -> Vec<u32> { (0..600u32).map(|i| (i * 7 + i / 5) % 40).collect() }

    fn trainer() -> Trainer {
        let mut t = Trainer::new(cfg(), 0.01, 3).unwrap();
        t.opt.weight_decay = 0.05;
        t.clip = Some(1.0);
        t.sched = Some(LrSchedule { base: 0.01, min: 0.001, warmup: 2, total: 12 });
        t
    }

    fn run(t: &mut Trainer, steps: usize) {
        let s = corpus();
        let b = Batcher { batch: 3, seq: 8, order: Order::Shuffled(5) };
        for _ in 0..steps {
            t.train_step(&b.at(&s[..], t.step as usize)).unwrap();
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("r2resume-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn bits(t: &Trainer) -> Vec<u32> { t.params.iter().flatten().map(|x| x.to_bits()).collect() }

    #[test]
    fn a_resumed_run_is_bit_identical_to_an_uninterrupted_one() {
        let mut whole = trainer();
        run(&mut whole, 12);

        let dir = tmp("identical");
        let mut first = trainer();
        run(&mut first, 5);
        first.save_checkpoint(&dir, 2).unwrap();
        drop(first);
        let mut second = Trainer::resume_latest(&dir).unwrap().unwrap();
        assert_eq!(second.step, 5);
        run(&mut second, 7);

        assert_eq!(bits(&whole), bits(&second), "weights differ after resuming");
        assert_eq!(whole.opt.m, second.opt.m);
        assert_eq!(whole.opt.v, second.opt.v);
        assert_eq!(whole.opt.t, second.opt.t);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_file_written_from_borrowed_tensors_is_the_one_the_format_defines() {
        let dir = tmp("format");
        let mut t = trainer();
        run(&mut t, 1);
        let path = t.save_checkpoint(&dir, 0).unwrap();
        let written = std::fs::read(&path).unwrap();
        let ck = Checkpoint::load(&path).unwrap();
        assert_eq!(ck.to_bytes(), written, "load + to_bytes must give the same bytes");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_checkpoint_of_another_shape_is_refused() {
        let dir = tmp("shape");
        let t = trainer();
        let path = t.save_checkpoint(&dir, 0).unwrap();
        let mut ck = Checkpoint::load(&path).unwrap();
        ck.put_scalar("cfg.ffn_hidden", 48.0);
        assert!(Trainer::from_checkpoint(ck).err().unwrap().contains("holds"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fine_tuning_starts_from_the_saved_weights_with_a_new_optimizer() {
        let mut base = trainer();
        run(&mut base, 3);
        let dir = tmp("model");
        base.save_dir(&dir).unwrap();
        let before = bits(&base);
        let t = Trainer::from_model(Model::load_dir(&dir).unwrap(), 0.001).unwrap();
        assert_eq!(bits(&t), before, "the weights must arrive unchanged");
        assert_eq!((t.step, t.opt.t), (0, 0));
        assert_eq!(t.opt.lr, 0.001);
        assert!(t.opt.m.iter().all(|&x| x == 0.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lending_the_model_changes_nothing_and_copies_nothing() {
        let mut t = trainer();
        run(&mut t, 2);
        let before = bits(&t);
        let ptr = t.params[0].as_ptr();
        let (via_with, ptr_inside) = t.with_model(|m| (m.tok_embed.clone(), m.tok_embed.as_ptr()));
        assert_eq!(ptr, ptr_inside, "the model must hold the trainer's own buffer");
        assert_eq!(via_with, t.params[0]);
        assert_eq!(bits(&t), before);
        assert_eq!(t.params[0].as_ptr(), ptr, "and hand it back");
    }
}
