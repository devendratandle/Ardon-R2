//! LoRA: fine-tune a frozen model through small low-rank adapters.
//!
//! A chosen projection `x W` becomes `x W + (alpha / rank) (x A) B`, with
//! `A` of `k x rank` and `B` of `rank x n`. Only `A` and `B` train. So a
//! fine-tune needs gradients and Adam state for the adapters alone, its
//! checkpoints hold only them, and what it produces is a small file that
//! applies to the model it came from.
//!
//! `B` starts at zero, so training begins exactly at the base model. Serving
//! needs nothing new: `W + (alpha / rank) A B` is merged into ordinary weights
//! and `r2_tensor::model::Model` runs them.
//!
//! An R2 model fine-tunes only from an R2 model: there is no importer for
//! other projects' weights, by project decision.

use std::path::Path;

use r2_tensor::json::Json;
use r2_tensor::model::{Config, Model};
use r2_tensor::safetensors::{self, SafeTensors};

use crate::llm::{Trainer, PER_LAYER};
use crate::optim::Adam;

const ADAPTER_FILE: &str = "adapter.safetensors";
const ADAPTER_CONFIG: &str = "adapter.json";

/// The projections an adapter may wrap (not the two norm gains).
const TARGETS: [usize; 7] = [1, 2, 3, 4, 6, 7, 8];

/// The adapters a trainer holds, after its base blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Lora {
    pub rank: usize,
    pub alpha: f32,
    /// Per-layer block parts wrapped, as indices into `PER_LAYER`, ascending.
    pub parts: Vec<usize>,
    /// Identity of the base weights these adapters were trained on.
    pub base_fp: u64,
}

/// `(in, out)` of per-layer block `part`.
fn part_shape(c: &Config, part: usize) -> (usize, usize) {
    let (d, kv, h) = (c.dim, c.kv_dim(), c.ffn_hidden);
    match part {
        1 | 4 => (d, d),
        2 | 3 => (d, kv),
        6 | 8 => (d, h),
        7 => (h, d),
        _ => unreachable!("part {part} has no adapter"),
    }
}

impl Lora {
    /// Blocks `(A, B)` of the adapter on `part` of `layer`, if it has one.
    pub(crate) fn blocks(&self, c: &Config, layer: usize, part: usize) -> Option<(usize, usize)> {
        let p = self.parts.iter().position(|&x| x == part)?;
        let a = Trainer::block_shapes(c).len() + (layer * self.parts.len() + p) * 2;
        Some((a, a + 1))
    }

    pub(crate) fn scale(&self) -> f32 { self.alpha / self.rank as f32 }

    /// Adapter blocks, in block order, as `(name, shape)`.
    fn layout(&self, c: &Config) -> Vec<(String, Vec<usize>)> {
        let mut v = Vec::new();
        for l in 0..c.n_layers {
            for &p in &self.parts {
                let (k, n) = part_shape(c, p);
                v.push((format!("layer.{l}.{}.lora_a", PER_LAYER[p]), vec![k, self.rank]));
                v.push((format!("layer.{l}.{}.lora_b", PER_LAYER[p]), vec![self.rank, n]));
            }
        }
        v
    }
}

/// Parts for target names. `None` means the usual four attention projections.
fn parse_targets(targets: Option<&[&str]>) -> Result<Vec<usize>, String> {
    let Some(names) = targets else { return Ok(vec![1, 2, 3, 4]); };
    let mut parts = Vec::with_capacity(names.len());
    for t in names {
        let p = PER_LAYER.iter().position(|n| n == t).filter(|p| TARGETS.contains(p))
            .ok_or_else(|| format!("lora: no adapter can wrap '{t}' (wq, wk, wv, wo, w1, w2 or w3)"))?;
        parts.push(p);
    }
    parts.sort();
    parts.dedup();
    if parts.is_empty() { return Err("lora: no targets".into()); }
    Ok(parts)
}

/// Fingerprint of the base blocks of `params`, frozen or not.
fn base_fingerprint(params: &[Vec<f32>], base_blocks: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in &params[..base_blocks] {
        for x in p { h = (h ^ x.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01B3); }
    }
    h
}

impl Trainer {
    /// Freeze the whole model and add adapters on `targets` (`None`: wq, wk,
    /// wv, wo). The optimizer restarts, holding state for the adapters only.
    pub fn enable_lora(&mut self, rank: usize, alpha: f32, targets: Option<&[&str]>, seed: u64) -> Result<usize, String> {
        if self.lora.is_some() { return Err("lora: this model already has adapters".into()); }
        let parts = parse_targets(targets)?;
        let c = self.cfg;
        for &p in &parts {
            let (k, n) = part_shape(&c, p);
            if rank == 0 || rank > k.min(n) {
                return Err(format!("lora: rank {rank} does not fit '{}' ({k} x {n})", PER_LAYER[p]));
            }
        }
        let base_blocks = self.params.len();
        let lora = Lora { rank, alpha, parts, base_fp: base_fingerprint(&self.params, base_blocks) };
        let mut rng = seed | 1;
        let mut next = || {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            ((rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u32 << 24) as f32) - 0.5
        };
        // A takes the scale Trainer::new gives a matrix; B is zero, so the
        // adapter starts as nothing at all.
        for (name, shape) in lora.layout(&c) {
            let n = shape[0] * shape[1];
            self.params.push(if name.ends_with("lora_a") {
                let s = 1.0 / (shape[0] as f32).sqrt();
                (0..n).map(|_| next() * 2.0 * s).collect()
            } else {
                vec![0.0; n]
            });
        }
        let total: usize = self.params[base_blocks..].iter().map(Vec::len).sum();
        let mut frozen = vec![true; base_blocks];
        frozen.resize(self.params.len(), false);
        self.frozen = frozen;
        self.lora = Some(lora);
        self.frozen_fp = self.fingerprint_frozen();
        let (lr, wd) = (self.opt.lr, self.opt.weight_decay);
        self.opt = Adam::new(total, lr);
        self.opt.weight_decay = wd;
        Ok(total)
    }

    /// `W + (alpha / rank) A B` for per-layer block `part` of `layer`, when it
    /// has an adapter.
    pub(crate) fn merged_block(&self, layer: usize, part: usize) -> Option<Vec<f32>> {
        let lora = self.lora.as_ref()?;
        let (a, b) = lora.blocks(&self.cfg, layer, part)?;
        let (k, n) = part_shape(&self.cfg, part);
        let mut delta = vec![0.0f32; k * n];
        r2_tensor::ops::matmul_into(&self.params[a], &self.params[b], k, lora.rank, n, &mut delta);
        let s = lora.scale();
        let w = &self.params[3 + layer * 9 + part];
        Some(w.iter().zip(&delta).map(|(w, d)| w + s * d).collect())
    }

    /// Write only the adapters: `adapter.safetensors` + `adapter.json`.
    pub fn save_adapter(&self, dir: impl AsRef<Path>) -> Result<(), String> {
        let lora = self.lora.as_ref().ok_or("lora: this model has no adapters")?;
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| format!("lora: {e}"))?;
        let base_blocks = Trainer::block_shapes(&self.cfg).len();
        let tensors: Vec<(String, Vec<usize>, &[f32])> = lora.layout(&self.cfg).into_iter().enumerate()
            .map(|(i, (name, shape))| (name, shape, self.params[base_blocks + i].as_slice())).collect();
        safetensors::save(dir.join(ADAPTER_FILE), &tensors)?;
        // Written last, as model.json is: it commits the adapter.
        let targets: Vec<String> = lora.parts.iter().map(|&p| format!("\"{}\"", PER_LAYER[p])).collect();
        std::fs::write(dir.join(ADAPTER_CONFIG), format!(
            "{{\n  \"format\": \"r2-adapter-v1\",\n  \"rank\": {},\n  \"alpha\": {},\n  \"targets\": [{}],\n  \
             \"base_fingerprint\": \"{:016x}\"\n}}\n",
            lora.rank, lora.alpha, targets.join(", "), lora.base_fp))
            .map_err(|e| format!("lora: {e}"))
    }

    /// The trainer of `base` with the adapters saved in `dir`. Serve it
    /// (`with_model` merges) or keep training. Refuses another base model.
    pub fn with_adapter(base: Model, lr: f32, dir: impl AsRef<Path>) -> Result<Trainer, String> {
        let dir = dir.as_ref();
        let src = std::fs::read_to_string(dir.join(ADAPTER_CONFIG))
            .map_err(|e| format!("lora: {}: {e}", dir.join(ADAPTER_CONFIG).display()))?;
        let j = Json::parse(&src).map_err(|e| format!("adapter.json: {e}"))?;
        let rank = j.get("rank").and_then(|v| v.as_usize()).ok_or("adapter.json: no rank")?;
        let alpha = j.get("alpha").and_then(|v| v.as_f64()).ok_or("adapter.json: no alpha")? as f32;
        let names: Vec<String> = j.get("targets").and_then(|v| v.as_arr())
            .ok_or("adapter.json: no targets")?
            .iter().filter_map(|t| t.as_str().map(str::to_string)).collect();
        let want = j.get("base_fingerprint").and_then(|v| v.as_str()).ok_or("adapter.json: no base_fingerprint")?;

        let mut t = Trainer::from_model(base, lr)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        t.enable_lora(rank, alpha, Some(&refs), 1)?;
        let lora = t.lora.as_ref().expect("just enabled");
        if format!("{:016x}", lora.base_fp) != want {
            return Err("lora: this adapter was trained on a different base model".into());
        }
        let st = SafeTensors::open(dir.join(ADAPTER_FILE))?;
        let base_blocks = Trainer::block_shapes(&t.cfg).len();
        for (i, (name, shape)) in lora.layout(&t.cfg).into_iter().enumerate() {
            t.params[base_blocks + i] = st.tensor_f32_shaped(&name, &shape)?;
        }
        Ok(t)
    }
}

/// Checkpoint scalars for the adapters.
pub(crate) fn lora_scalars(l: &Lora) -> [(&'static str, f64); 5] {
    [("lora.rank", l.rank as f64), ("lora.alpha", l.alpha as f64),
     ("lora.parts", l.parts.iter().map(|&p| (1u32 << p) as f64).sum()),
     ("lora.fp_hi", (l.base_fp >> 32) as f64), ("lora.fp_lo", (l.base_fp & 0xFFFF_FFFF) as f64)]
}

/// The adapters a checkpoint was taken with, if any.
pub(crate) fn lora_from(ck: &crate::checkpoint::Checkpoint) -> Result<Option<Lora>, String> {
    if ck.scalar("lora.rank").is_err() { return Ok(None); }
    let mask = ck.scalar("lora.parts")? as u32;
    Ok(Some(Lora {
        rank: ck.scalar("lora.rank")? as usize,
        alpha: ck.scalar("lora.alpha")? as f32,
        parts: (0..PER_LAYER.len()).filter(|p| mask & (1 << p) != 0).collect(),
        base_fp: ((ck.scalar("lora.fp_hi")? as u64) << 32) | ck.scalar("lora.fp_lo")? as u64,
    }))
}

/// Sizes of the adapter blocks, in block order.
pub(crate) fn adapter_sizes(l: &Lora, c: &Config) -> Vec<usize> {
    l.layout(c).iter().map(|(_, s)| s[0] * s[1]).collect()
}

/// Names of the adapter blocks, in block order.
pub(crate) fn adapter_names(l: &Lora, c: &Config) -> Vec<String> {
    l.layout(c).into_iter().map(|(n, _)| n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Order, PairSet};
    use std::path::PathBuf;

    fn cfg() -> Config {
        Config { dim: 16, n_heads: 2, n_kv_heads: 1, n_layers: 2, vocab: 40, ffn_hidden: 32,
                 max_seq: 16, rope_base: 10000.0, eps: 1e-5 }
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("r2lora-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn batch() -> Vec<(Vec<usize>, Vec<usize>)> {
        vec![(vec![1, 2, 3, 4, 5], vec![2, 3, 4, 5, 6]), (vec![7, 8, 9, 10, 11], vec![8, 9, 10, 11, 12])]
    }

    fn examples() -> PairSet {
        let pairs: Vec<_> = (0..12u32).map(|i| (vec![i + 1, i + 2], vec![(i * 3) % 40, (i * 5) % 40])).collect();
        PairSet::new(&pairs, 8, 0).unwrap()
    }

    fn base() -> Trainer {
        let mut t = Trainer::new(cfg(), 0.01, 3).unwrap();
        for _ in 0..5 { t.train_step(&batch()).unwrap(); }
        t.opt.lr = 0.02;
        t
    }

    fn bits(p: &[Vec<f32>]) -> Vec<u32> { p.iter().flatten().map(|x| x.to_bits()).collect() }

    #[test]
    fn a_new_adapter_is_exactly_the_base_model() {
        let plain = base();
        let (l0, g0, _) = plain.loss_logits_grads(&batch()).unwrap();
        let mut t = base();
        t.enable_lora(4, 8.0, None, 1).unwrap();
        let (l1, g1, _) = t.loss_logits_grads(&batch()).unwrap();
        assert_eq!(l0, l1);
        assert!(g0 == g1, "B = 0 must leave every logit where it was");
    }

    #[test]
    fn only_the_adapters_train_and_they_are_small() {
        let mut t = base();
        let total = t.enable_lora(4, 8.0, None, 1).unwrap();
        let base_blocks = Trainer::block_shapes(&t.cfg).len();
        let before = bits(&t.params[..base_blocks]);
        let params: usize = t.params[..base_blocks].iter().map(Vec::len).sum();
        assert_eq!(t.opt.m.len(), total, "Adam state for the adapters only");
        assert!(total * 3 < params, "{total} adapter values for {params} weights");
        let s = examples();
        let first = t.train_step(&s.at(4, 0, Order::Shuffled(1))).unwrap();
        let mut last = first;
        for step in 1..120 { last = t.train_step(&s.at(4, step, Order::Shuffled(1))).unwrap(); }
        assert_eq!(before, bits(&t.params[..base_blocks]), "the base moved");
        assert!(last < first * 0.8, "loss {first} -> {last}");
    }

    #[test]
    fn the_adapter_gradient_matches_finite_differences() {
        let mut t = base();
        t.enable_lora(3, 6.0, Some(&["wq", "w2"]), 1).unwrap();
        // B away from zero, so A has a gradient too
        let base_blocks = Trainer::block_shapes(&t.cfg).len();
        for (i, p) in t.params[base_blocks..].iter_mut().enumerate() {
            if i % 2 == 1 { for (j, x) in p.iter_mut().enumerate() { *x = 0.05 * ((j % 5) as f32 - 2.0); } }
        }
        let b = batch();
        let (_, _, grads) = t.loss_logits_grads(&b).unwrap();
        let h = 1e-2f32;
        for block in [base_blocks, base_blocks + 1, base_blocks + 3] {
            for i in [0usize, 5] {
                let orig = t.params[block][i];
                t.params[block][i] = orig + h;
                let up = t.loss_logits_grads(&b).unwrap().0;
                t.params[block][i] = orig - h;
                let dn = t.loss_logits_grads(&b).unwrap().0;
                t.params[block][i] = orig;
                let numeric = (up - dn) / (2.0 * h);
                let g = grads[block][i];
                assert!((g - numeric).abs() < 0.02 + 0.1 * numeric.abs(),
                        "block {block}[{i}]: tape {g}, numeric {numeric}");
            }
        }
    }

    #[test]
    fn serving_the_merged_weights_gives_the_trained_logits() {
        let mut t = base();
        t.enable_lora(4, 8.0, Some(&["wq", "wv", "w1"]), 1).unwrap();
        let s = examples();
        for step in 0..30 { t.train_step(&s.at(4, step, Order::Sequential)).unwrap(); }
        let tokens = vec![1usize, 2, 3, 4];
        let next = vec![2usize, 3, 4, 5];
        let (_, logits, _) = t.loss_logits_grads(&[(tokens.clone(), next)]).unwrap();
        let before = bits(&t.params);
        let vocab = t.cfg.vocab;
        let check = |m: &Model| {
            let mut caches = m.new_caches().unwrap();
            for (i, &tok) in tokens.iter().enumerate() {
                let served = m.forward_step(tok, i, &mut caches).unwrap();
                for (a, b) in served.iter().zip(&logits[i * vocab..(i + 1) * vocab]) {
                    assert!((a - b).abs() < 1e-3, "position {i}: served {a}, trained {b}");
                }
            }
        };
        check(&t.to_model().unwrap());
        t.with_model(check);
        assert_eq!(before, bits(&t.params), "merging for a call must leave the base and adapters alone");
    }

    #[test]
    fn an_adapter_file_applies_to_its_base_and_no_other() {
        let mut t = base();
        let plain = base().to_model().unwrap();
        t.enable_lora(4, 8.0, None, 1).unwrap();
        let s = examples();
        for step in 0..10 { t.train_step(&s.at(4, step, Order::Sequential)).unwrap(); }
        let dir = tmp("adapter");
        t.save_adapter(&dir).unwrap();
        let size = std::fs::metadata(dir.join(ADAPTER_FILE)).unwrap().len() as usize;
        assert!(size < t.params[..Trainer::block_shapes(&t.cfg).len()].iter().map(Vec::len).sum::<usize>() * 4 / 3);

        let back = Trainer::with_adapter(plain.clone(), 0.01, &dir).unwrap();
        assert_eq!(bits(&back.params), bits(&t.params), "base + file must rebuild the trainer");

        let mut other = plain;
        other.tok_embed[0] += 1.0;
        assert!(Trainer::with_adapter(other, 0.01, &dir).err().unwrap().contains("different base"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_resumed_adapter_run_is_the_same_run_and_stores_only_the_adapters() {
        let plain = base().to_model().unwrap();
        let start = || {
            let mut t = Trainer::from_model(plain.clone(), 0.02).unwrap();
            t.enable_lora(4, 8.0, None, 1).unwrap();
            t.clip = Some(1.0);
            t
        };
        let s = examples();
        let run = |t: &mut Trainer, n: usize| for _ in 0..n {
            t.train_step(&s.at(4, t.step as usize, Order::Shuffled(2))).unwrap();
        };
        let mut whole = start();
        run(&mut whole, 8);

        let dir = tmp("resume");
        let mut first = start();
        run(&mut first, 3);
        let trainable = first.opt.m.len();
        let path = first.save_checkpoint(&dir, 0).unwrap();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        assert!(size < (trainable * 3 + 2000) * 4, "the base was stored: {size} bytes");
        drop(first);
        let mut second = Trainer::resume_latest(&dir, Some(plain.clone())).unwrap().unwrap();
        run(&mut second, 5);
        assert_eq!(bits(&whole.params), bits(&second.params));
        assert_eq!(whole.lora, second.lora);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_requests_are_refused() {
        let mut t = base();
        assert!(t.enable_lora(0, 1.0, None, 1).unwrap_err().contains("rank 0"));
        assert!(t.enable_lora(99, 1.0, None, 1).unwrap_err().contains("rank 99"));
        assert!(t.enable_lora(2, 1.0, Some(&["attn_norm"]), 1).unwrap_err().contains("attn_norm"));
        t.enable_lora(2, 1.0, None, 1).unwrap();
        assert!(t.enable_lora(2, 1.0, None, 1).unwrap_err().contains("already"));
    }
}
