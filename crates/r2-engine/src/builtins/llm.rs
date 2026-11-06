//! `llm.*` — train, save, load and serve language models from R2 script.
//!
//! This is the language-level surface over `r2-train` and `r2-tensor`: the
//! whole lifecycle without leaving R2 and without a line of Rust.
//!
//! ```r
//! m <- llm.new(dim = 128, layers = 5, heads = 4, kv.heads = 2, ffn = 384)
//! llm.train(m, "some training text", steps = 120, lr = 0.003)
//! llm.save(m, "mymodel")
//! s <- llm.load("mymodel")
//! llm.generate(s, "prompt", 32)
//! ```
//!
//! Models live in a process registry and R2 sees an integer handle, the
//! same shape R uses for connections and devices. A handle is cheap to
//! pass around and cannot be accidentally deep-copied, which matters when
//! the thing behind it is hundreds of megabytes.

use std::collections::HashMap;
use std::sync::Mutex;

use r2_tensor::infer::{Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;
use r2_types::*;

use crate::{err, gn, gv, Engine};

pub(super) const TOKENIZER_FILE: &str = "tokenizer.json";

/// What a handle refers to: a model being trained, or one loaded purely
/// for serving (no optimizer state, so it costs a third of the memory).
pub(super) enum Slot {
    Training(Box<Trainer>),
    Serving(Box<Model>),
}

/// The tokenizer a handle was created with.
///
/// Held BESIDE the model rather than rebuilt at each call site. `llm.train`
/// and `llm.generate` used to each construct their own `byte_level()`, which
/// is only safe because both happened to pick the same one; the moment a
/// model has a learned vocabulary, a generate that rebuilt a byte tokenizer
/// would decode the model's ids against the wrong table and emit plausible
/// nonsense. Pairing the tokenizer with the handle makes that unrepresentable.
pub(super) type Toks = std::collections::HashMap<u32, Tokenizer>;
pub(super) fn with_tokenizers<R>(f: impl FnOnce(&mut Toks) -> R) -> R {
    use std::sync::{Mutex, OnceLock};
    static T: OnceLock<Mutex<Toks>> = OnceLock::new();
    f(&mut T.get_or_init(|| Mutex::new(Toks::new())).lock().unwrap())
}

/// The tokenizer for `id`, or byte-level if none was attached.
pub(super) fn tokenizer_for(id: u32) -> Tokenizer {
    with_tokenizers(|t| t.get(&id).cloned()).unwrap_or_else(Tokenizer::byte_level)
}

impl Slot {
    pub(super) fn config(&self) -> Config {
        match self {
            Slot::Training(t) => t.cfg,
            Slot::Serving(m) => m.cfg,
        }
    }
    /// Run `f` on a servable model. Nothing is copied: a serving handle
    /// already is one, and a trainer lends its weights for the call.
    fn with_model<R>(&mut self, f: impl FnOnce(&Model) -> R) -> R {
        match self {
            Slot::Training(t) => t.with_model(f),
            Slot::Serving(m) => f(m),
        }
    }
}

static REGISTRY: Mutex<Option<HashMap<u32, Slot>>> = Mutex::new(None);
static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

pub(super) fn with_registry<R>(f: impl FnOnce(&mut HashMap<u32, Slot>) -> R) -> R {
    let mut g = REGISTRY.lock().unwrap();
    f(g.get_or_insert_with(HashMap::new))
}

pub(super) fn insert(slot: Slot) -> u32 {
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    with_registry(|r| r.insert(id, slot));
    id
}

pub(super) fn handle(v: &RVal) -> Result<u32, R2Err> {
    match v.scalar_f64() {
        Ok(Some(x)) if x >= 1.0 => Ok(x as u32),
        _ => err!(Runtime, "llm: expected a model handle from llm.new() or llm.load()"),
    }
}

pub(super) fn num_arg(a: &[EvalArg], name: &str, default: f64) -> f64 {
    gn(a, name).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(default)
}

pub(super) fn str_arg(v: &RVal) -> Result<String, R2Err> {
    match v {
        RVal::Character(s, _) => match s.first().and_then(|x| x.clone()) {
            Some(t) => Ok(t.to_string()),
            None => err!(Runtime, "llm: expected a string"),
        },
        _ => err!(Runtime, "llm: expected a string, got {}", v.type_name()),
    }
}

pub(super) fn bool_arg(a: &[EvalArg], name: &str, default: bool) -> bool {
    gn(a, name).and_then(|v| v.as_logicals().ok()?.first().copied().flatten()).unwrap_or(default)
}

pub(super) fn as_num(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }

/// Handles a callback is training right now. While one is listed, the script
/// can read that model but not change it: the loop holds a step count, a
/// schedule and possibly a device copy that a change would leave behind.
/// Only a run with a callback ever enters this list.
static RUNNING: Mutex<Vec<u32>> = Mutex::new(Vec::new());

pub(super) fn refuse_running(ctx: &str, id: u32) -> Result<(), R2Err> {
    if RUNNING.lock().unwrap().contains(&id) {
        return err!(Runtime, "{}: model {} is still training: its callback can read it, not change it", ctx, id);
    }
    Ok(())
}

/// Lists a handle in [`RUNNING`] until dropped, however the run ends.
pub(super) struct Running(u32);

impl Running {
    pub(super) fn enter(id: u32) -> Running {
        RUNNING.lock().unwrap().push(id);
        Running(id)
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        RUNNING.lock().unwrap().retain(|&x| x != self.0);
    }
}

/// A shape `llm.new(preset =)` can name.
struct Shape { dim: usize, layers: usize, heads: usize, kv_heads: usize, ffn: usize, ctx: usize, vocab: usize }

/// Shape only: the weights still come from `seed` and training, and every
/// field is a named argument's default. These are the shapes R2 has been
/// benchmarked at (`benchmarks/llm/`), under the names its reports use.
const PRESETS: [(&str, Shape); 4] = [
    ("tiny", Shape { dim: 128, layers: 4, heads: 4, kv_heads: 2, ffn: 384, ctx: 64, vocab: 2000 }),      // 1.30M
    ("small", Shape { dim: 256, layers: 4, heads: 4, kv_heads: 2, ffn: 768, ctx: 64, vocab: 8000 }),     // 7.24M
    ("medium", Shape { dim: 768, layers: 4, heads: 12, kv_heads: 4, ffn: 2304, ctx: 256, vocab: 8000 }), // 39.8M
    ("large", Shape { dim: 768, layers: 12, heads: 12, kv_heads: 12, ffn: 3072, ctx: 512, vocab: 8000 }), // 125.5M
];

fn preset_arg(a: &[EvalArg]) -> Result<Option<&'static Shape>, R2Err> {
    let Some(name) = gn(a, "preset").filter(|v| !matches!(v, RVal::Null)) else { return Ok(None) };
    let name = str_arg(&name)?;
    match PRESETS.iter().find(|(n, _)| *n == name) {
        Some((_, shape)) => Ok(Some(shape)),
        None => {
            let names: Vec<&str> = PRESETS.iter().map(|(n, _)| *n).collect();
            err!(Runtime, "llm.new: unknown preset '{}' ({})", name, names.join(", "))
        }
    }
}

/// `llm.new(preset=, dim=, layers=, heads=, kv.heads=, ffn=, ctx=)` — create a
/// trainable model. Vocabulary is the 256 byte values, so any text (any
/// language, any bytes) is representable with no tokenizer training step.
pub(crate) fn bi_llm_new(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let preset = preset_arg(a)?;
    // a named argument wins, then the preset, then the plain default
    let arg = |name: &str, of: fn(&Shape) -> usize, plain: f64| {
        num_arg(a, name, preset.map_or(plain, |s| of(s) as f64))
    };
    let dim = arg("dim", |s| s.dim, 128.0) as usize;
    let n_heads = arg("heads", |s| s.heads, 4.0) as usize;
    let cfg = Config {
        dim,
        n_heads,
        n_kv_heads: arg("kv.heads", |s| s.kv_heads, n_heads as f64) as usize,
        n_layers: arg("layers", |s| s.layers, 4.0) as usize,
        // `vocab` is the tokenizer's size, and the DEFAULT IS BPE.
        //
        // 8,000 measures 0.242 tokens per byte on TinyStories — 4.13 bytes
        // per token, which is the ~4x compression a real tokenizer is
        // supposed to deliver. The 256 byte values remain INSIDE that
        // vocabulary as the fallback, so nothing is unrepresentable: an
        // emoji no merge covers still decomposes into its UTF-8 bytes.
        //
        // `vocab = 256` is still accepted and gives a pure byte-level
        // tokenizer — one token per byte, no merges. That is a fallback
        // for the rare case where merges are unwanted, NOT the default it
        // used to be. A byte-level default makes every sequence ~4x longer
        // than it needs to be, and attention is O(seq^2), so it was costing
        // roughly 16x the attention work for the same text.
        //
        // Measured on 0.33 MB of prose, 4-layer dim-256 model
        // (`--example tokenizer_tradeoff`):
        //     256   1.000 tok/byte  5.42 bits/byte   752 text bytes/s
        //    1000   0.439            3.92           1516
        //    4000   0.311            3.31           1588  <- throughput peak
        //   14492   0.250            2.89            790
        // Bigger vocabularies keep improving quality per byte, but the
        // output projection is dim*vocab per token, so on a SMALL model it
        // starts costing more than the shorter sequence saves.
        vocab: arg("vocab", |s| s.vocab, 8000.0) as usize,
        ffn_hidden: arg("ffn", |s| s.ffn, (dim * 3) as f64) as usize,
        max_seq: arg("ctx", |s| s.ctx, 64.0) as usize,
        rope_base: 10000.0,
        eps: 1e-5,
    };
    let lr = num_arg(a, "lr", 0.003) as f32;
    let seed = num_arg(a, "seed", 42.0) as u64;
    match Trainer::new(cfg, lr, seed) {
        Ok(t) => Ok(as_num(insert(Slot::Training(Box::new(t))) as f64)),
        Err(e) => err!(Runtime, "llm.new: {}", e),
    }
}

/// `llm.generate(model, prompt, n=, temperature=, top.k=, seed=)`.
pub(crate) fn bi_llm_generate(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let prompt = str_arg(&gv(a, 1))?;
    let n = if a.len() > 2 && a[2].name.is_none() {
        gv(a, 2).scalar_f64().ok().flatten().unwrap_or(32.0) as usize
    } else { num_arg(a, "n", 32.0) as usize };
    let temperature = num_arg(a, "temperature", 0.0) as f32;
    let top_k = num_arg(a, "top.k", 0.0) as usize;
    let seed = num_arg(a, "seed", 1.0) as u64;

    // The handle's own tokenizer — see `Toks`. Rebuilding a byte-level one
    // here would decode a BPE model's ids against the wrong table.
    let tok = tokenizer_for(id);
    let ids: Vec<usize> = tok.encode(&prompt)
        .map_err(|e| R2Err { msg: format!("llm.generate: {}", e), kind: ErrKind::Runtime })?
        .iter().map(|&i| i as usize).collect();
    if ids.is_empty() {
        return err!(Runtime, "llm.generate: prompt must not be empty");
    }
    let mut sampler = Sampler::new(seed, SamplerConfig {
        temperature, top_k, ..Default::default()
    });
    let out = with_registry(|r| match r.get_mut(&id) {
        Some(s) => s.with_model(|m| m.generate(&ids, n, &mut sampler, None)),
        None => Err(format!("unknown model handle {}", id)),
    }).map_err(|e| R2Err { msg: format!("llm.generate: {}", e), kind: ErrKind::Runtime })?;
    let text = tok.decode(&out.iter().map(|&i| i as u32).collect::<Vec<_>>());
    Ok(RVal::Character(vec![Some(std::sync::Arc::from(text.as_str()))], Attrs::default()))
}

/// `llm.save(model, dir)` — write `model.json` + `model.safetensors`.
pub(crate) fn bi_llm_save(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let dir = str_arg(&gv(a, 1))?;
    with_registry(|r| match r.get_mut(&id) {
        Some(s) => s.with_model(|m| m.save_dir(&dir)),
        None => Err(format!("unknown model handle {}", id)),
    }).map_err(|e| R2Err { msg: format!("llm.save: {}", e), kind: ErrKind::Runtime })?;
    // The ids mean nothing without their tokenizer: a model saved alone
    // would decode against the wrong table and generate nonsense.
    std::fs::write(std::path::Path::new(&dir).join(TOKENIZER_FILE), tokenizer_for(id).to_tokenizer_json())
        .map_err(|e| R2Err { msg: format!("llm.save: {}", e), kind: ErrKind::Runtime })?;
    Ok(RVal::Character(vec![Some(std::sync::Arc::from(dir.as_str()))], Attrs::default()))
}

/// The tokenizer saved beside a model. Only a pure byte-level model can do
/// without one; any other would decode its ids against the wrong table.
pub(super) fn load_tokenizer(dir: &str, cfg: &Config) -> Result<Tokenizer, String> {
    let path = std::path::Path::new(dir).join(TOKENIZER_FILE);
    match std::fs::read_to_string(&path) {
        Ok(src) => Tokenizer::from_tokenizer_json(&src),
        Err(_) if cfg.vocab <= 256 => Ok(Tokenizer::byte_level()),
        Err(e) => Err(format!("{}: {} (a model with a learned vocabulary needs its tokenizer)",
                              path.display(), e)),
    }
}

/// `llm.load(dir)` — load a saved model for serving.
pub(crate) fn bi_llm_load(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let dir = str_arg(&gv(a, 0))?;
    let m = Model::load_dir(&dir)
        .map_err(|e| R2Err { msg: format!("llm.load: {}", e), kind: ErrKind::Runtime })?;
    let tok = load_tokenizer(&dir, &m.cfg)
        .map_err(|e| R2Err { msg: format!("llm.load: {}", e), kind: ErrKind::Runtime })?;
    // train = TRUE starts a new run from these weights: fine-tuning.
    let slot = if bool_arg(a, "train", false) {
        let t = Trainer::from_model(m, num_arg(a, "lr", 3e-4) as f32)
            .map_err(|e| R2Err { msg: format!("llm.load: {}", e), kind: ErrKind::Runtime })?;
        Slot::Training(Box::new(t))
    } else {
        Slot::Serving(Box::new(m))
    };
    let id = insert(slot);
    with_tokenizers(|t| t.insert(id, tok));
    Ok(as_num(id as f64))
}

/// `llm.resume(dir)` — the newest checkpoint `llm.train(checkpoint = dir)`
/// wrote there, as a handle that continues the run exactly.
pub(crate) fn bi_llm_resume(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let dir = str_arg(&gv(a, 0))?;
    let fail = |e: String| R2Err { msg: format!("llm.resume: {}", e), kind: ErrKind::Runtime };
    let base = gn(a, "base").map(|v| str_arg(&v)).transpose()?
        .map(|b| Model::load_dir(&b).map_err(fail)).transpose()?;
    let t = Trainer::resume_latest(&dir, base).map_err(fail)?
        .ok_or_else(|| fail(format!("no checkpoint in '{}'", dir)))?;
    let tok = load_tokenizer(&dir, &t.cfg).map_err(fail)?;
    let id = insert(Slot::Training(Box::new(t)));
    with_tokenizers(|m| m.insert(id, tok));
    Ok(as_num(id as f64))
}

/// `llm.info(model)` — a named list of the model's shape and size.
pub(crate) fn bi_llm_info(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let (cfg, serving) = with_registry(|r| match r.get(&id) {
        Some(s) => Ok((s.config(), matches!(s, Slot::Serving(_)))),
        None => Err(R2Err { msg: format!("llm.info: unknown model handle {}", id), kind: ErrKind::Runtime }),
    })?;
    let items: Vec<(Option<std::sync::Arc<str>>, RVal)> = vec![
        (Some("params".into()), as_num(cfg.n_params() as f64)),
        (Some("dim".into()), as_num(cfg.dim as f64)),
        (Some("layers".into()), as_num(cfg.n_layers as f64)),
        (Some("heads".into()), as_num(cfg.n_heads as f64)),
        (Some("kv.heads".into()), as_num(cfg.n_kv_heads as f64)),
        (Some("ffn".into()), as_num(cfg.ffn_hidden as f64)),
        (Some("ctx".into()), as_num(cfg.max_seq as f64)),
        (Some("vocab".into()), as_num(cfg.vocab as f64)),
        (Some("mode".into()), RVal::Character(
            vec![Some(std::sync::Arc::from(if serving { "serving" } else { "training" }))],
            Attrs::default())),
    ];
    Ok(RVal::list(items))
}

/// `llm.free(model)` — drop a model and release its memory. Explicit
/// because a model can be gigabytes and R2 has no way to know when a
/// script is done with a handle.
pub(crate) fn bi_llm_free(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    refuse_running("llm.free", id)?;
    let existed = with_registry(|r| r.remove(&id).is_some());
    Ok(RVal::Logical(vec![Some(existed)].into(), Attrs::default()))
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    /// The sizes the preset names promise, without allocating 125M weights.
    #[test]
    fn each_preset_has_the_size_it_was_benchmarked_at() {
        let params: Vec<(&str, usize)> = PRESETS.iter().map(|(name, s)| {
            let cfg = Config { dim: s.dim, n_heads: s.heads, n_kv_heads: s.kv_heads, n_layers: s.layers,
                               vocab: s.vocab, ffn_hidden: s.ffn, max_seq: s.ctx, rope_base: 10000.0, eps: 1e-5 };
            cfg.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            (*name, cfg.n_params())
        }).collect();
        for ((name, n), millions) in params.iter().zip([1.30, 7.24, 39.8, 125.5]) {
            assert!((*n as f64 / 1e6 - millions).abs() < 0.1, "{name}: {n}");
        }
    }
}
