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

/// What a handle refers to: a model being trained, or one loaded purely
/// for serving (no optimizer state, so it costs a third of the memory).
enum Slot {
    Training(Box<Trainer>),
    Serving(Box<Model>),
}

impl Slot {
    fn config(&self) -> Config {
        match self {
            Slot::Training(t) => t.cfg,
            Slot::Serving(m) => m.cfg,
        }
    }
    /// A servable model — exported from the trainer if needed.
    fn model(&self) -> Result<Model, String> {
        match self {
            Slot::Training(t) => t.to_model(),
            Slot::Serving(m) => Ok((**m).clone()),
        }
    }
}

static REGISTRY: Mutex<Option<HashMap<u32, Slot>>> = Mutex::new(None);
static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn with_registry<R>(f: impl FnOnce(&mut HashMap<u32, Slot>) -> R) -> R {
    let mut g = REGISTRY.lock().unwrap();
    f(g.get_or_insert_with(HashMap::new))
}

fn insert(slot: Slot) -> u32 {
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    with_registry(|r| r.insert(id, slot));
    id
}

fn handle(v: &RVal) -> Result<u32, R2Err> {
    match v.scalar_f64() {
        Ok(Some(x)) if x >= 1.0 => Ok(x as u32),
        _ => err!(Runtime, "llm: expected a model handle from llm.new() or llm.load()"),
    }
}

fn num_arg(a: &[EvalArg], name: &str, default: f64) -> f64 {
    gn(a, name).and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(default)
}

fn str_arg(v: &RVal) -> Result<String, R2Err> {
    match v {
        RVal::Character(s, _) => match s.first().and_then(|x| x.clone()) {
            Some(t) => Ok(t.to_string()),
            None => err!(Runtime, "llm: expected a string"),
        },
        _ => err!(Runtime, "llm: expected a string, got {}", v.type_name()),
    }
}

fn as_num(x: f64) -> RVal { RVal::Numeric(vec![Some(x)].into(), Attrs::default()) }

/// `llm.new(dim=, layers=, heads=, kv.heads=, ffn=, ctx=)` — create a
/// trainable model. Vocabulary is the 256 byte values, so any text (any
/// language, any bytes) is representable with no tokenizer training step.
pub(crate) fn bi_llm_new(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let dim = num_arg(a, "dim", 128.0) as usize;
    let n_heads = num_arg(a, "heads", 4.0) as usize;
    let cfg = Config {
        dim,
        n_heads,
        n_kv_heads: num_arg(a, "kv.heads", n_heads as f64) as usize,
        n_layers: num_arg(a, "layers", 4.0) as usize,
        vocab: 256,
        ffn_hidden: num_arg(a, "ffn", (dim * 3) as f64) as usize,
        max_seq: num_arg(a, "ctx", 64.0) as usize,
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

/// `llm.train(model, text, steps=, seq=, batch=)` — train on a string.
/// Returns the final loss so a script can decide whether to keep going.
pub(crate) fn bi_llm_train(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let text = str_arg(&gv(a, 1))?;
    let steps = num_arg(a, "steps", 50.0) as usize;
    let seq = num_arg(a, "seq", 16.0) as usize;
    let max_batch = num_arg(a, "batch", 8.0) as usize;
    let report = num_arg(a, "report", 0.0) as usize;

    let tok = Tokenizer::byte_level();
    let ids: Vec<usize> = tok.encode(&text)
        .map_err(|e| R2Err { msg: format!("llm.train: {}", e), kind: ErrKind::Runtime })?
        .iter().map(|&i| i as usize).collect();
    if ids.len() < seq + 2 {
        return err!(Runtime,
            "llm.train: text is too short — needs more than {} bytes for seq={}",
            seq + 1, seq);
    }
    // Sliding windows of `seq` tokens; target is the text shifted by one,
    // which is what "predict the next token" means concretely.
    let mut batch = Vec::new();
    for s in 0..(ids.len() - seq - 1).min(max_batch) {
        batch.push((ids[s..s + seq].to_vec(), ids[s + 1..s + seq + 1].to_vec()));
    }

    with_registry(|r| {
        let slot = match r.get_mut(&id) {
            Some(s) => s,
            None => return err!(Runtime, "llm.train: unknown model handle {}", id),
        };
        let t = match slot {
            Slot::Training(t) => t,
            Slot::Serving(_) => return err!(Runtime,
                "llm.train: this handle is a loaded model (serving only). \
                 Create one with llm.new() to train."),
        };
        let mut last = 0.0f32;
        for s in 1..=steps {
            last = t.train_step(&batch)
                .map_err(|e| R2Err { msg: format!("llm.train: {}", e), kind: ErrKind::Runtime })?;
            if report > 0 && (s % report == 0 || s == 1) {
                r2_types::out::rout(&format!("step {:>5}  loss {:.4}\n", s, last));
            }
        }
        Ok(as_num(last as f64))
    })
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

    let model = with_registry(|r| match r.get(&id) {
        Some(s) => s.model().map_err(|e| R2Err { msg: format!("llm.generate: {}", e), kind: ErrKind::Runtime }),
        None => Err(R2Err { msg: format!("llm.generate: unknown model handle {}", id), kind: ErrKind::Runtime }),
    })?;

    let tok = Tokenizer::byte_level();
    let ids: Vec<usize> = tok.encode(&prompt)
        .map_err(|e| R2Err { msg: format!("llm.generate: {}", e), kind: ErrKind::Runtime })?
        .iter().map(|&i| i as usize).collect();
    if ids.is_empty() {
        return err!(Runtime, "llm.generate: prompt must not be empty");
    }
    let mut sampler = Sampler::new(seed, SamplerConfig {
        temperature, top_k, ..Default::default()
    });
    let out = model.generate(&ids, n, &mut sampler, None)
        .map_err(|e| R2Err { msg: format!("llm.generate: {}", e), kind: ErrKind::Runtime })?;
    let text = tok.decode(&out.iter().map(|&i| i as u32).collect::<Vec<_>>());
    Ok(RVal::Character(vec![Some(std::sync::Arc::from(text.as_str()))], Attrs::default()))
}

/// `llm.save(model, dir)` — write `model.json` + `model.safetensors`.
pub(crate) fn bi_llm_save(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let dir = str_arg(&gv(a, 1))?;
    let model = with_registry(|r| match r.get(&id) {
        Some(s) => s.model().map_err(|e| R2Err { msg: format!("llm.save: {}", e), kind: ErrKind::Runtime }),
        None => Err(R2Err { msg: format!("llm.save: unknown model handle {}", id), kind: ErrKind::Runtime }),
    })?;
    model.save_dir(&dir)
        .map_err(|e| R2Err { msg: format!("llm.save: {}", e), kind: ErrKind::Runtime })?;
    Ok(RVal::Character(vec![Some(std::sync::Arc::from(dir.as_str()))], Attrs::default()))
}

/// `llm.load(dir)` — load a saved model for serving.
pub(crate) fn bi_llm_load(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let dir = str_arg(&gv(a, 0))?;
    let m = Model::load_dir(&dir)
        .map_err(|e| R2Err { msg: format!("llm.load: {}", e), kind: ErrKind::Runtime })?;
    Ok(as_num(insert(Slot::Serving(Box::new(m))) as f64))
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
    Ok(RVal::List(items))
}

/// `llm.free(model)` — drop a model and release its memory. Explicit
/// because a model can be gigabytes and R2 has no way to know when a
/// script is done with a handle.
pub(crate) fn bi_llm_free(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    let existed = with_registry(|r| r.remove(&id).is_some());
    Ok(RVal::Logical(vec![Some(existed)].into(), Attrs::default()))
}
