//! `llm.train`, `llm.finetune`, `llm.freeze`: everything that changes weights.
//!
//! ```r
//! base <- llm.load("mymodel", train = TRUE, lr = 3e-4)
//! llm.freeze(base, c("tok_embed", "layer.0."))
//! llm.finetune(base, prompts, completions, steps = 200, batch = 4, seq = 64,
//!              wd = 0.01, clip = 1, schedule = "cosine", warmup = 20,
//!              checkpoint = "run1", every = 50)
//! ```

use r2_tensor::tokenizer::Tokenizer;
use r2_train::data::{Batch, Batcher, Order, PairSet};
use r2_train::llm::Trainer;
use r2_train::optim::LrSchedule;
use r2_types::*;

use super::llm::{as_num, bool_arg, handle, insert, load_tokenizer, num_arg, refuse_running, str_arg,
                 tokenizer_for, with_registry, with_tokenizers, Running, Slot, TOKENIZER_FILE};
use crate::{err, gn, gv, Engine};

pub(super) fn fail(ctx: &str, e: impl std::fmt::Display) -> R2Err {
    R2Err { msg: format!("{ctx}: {e}"), kind: ErrKind::Runtime }
}

// The GPU's side of the loop, so the loop itself reads the same in every build.
#[cfg(any(feature = "gpu", feature = "cuda"))]
type Device = r2_train::gpu_llm::GpuTrainer;
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
type Device = std::convert::Infallible;

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn open_device(ctx: &str, s: &Settings, t: &Trainer) -> Result<Option<Device>, R2Err> {
    if !s.gpu { return Ok(None); }
    if s.accum > 1 { return err!(Runtime, "{}: accum is CPU only so far", ctx); }
    Device::from_trainer(t).map(Some).map_err(|e| fail(ctx, e))
}
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn open_device(ctx: &str, s: &Settings, _: &Trainer) -> Result<Option<Device>, R2Err> {
    if s.gpu { return err!(Runtime, "{}: this build has no GPU support (build with --features gpu or cuda)", ctx); }
    Ok(None)
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
fn device_step(device: &mut Option<Device>, batch: &Batch) -> Option<Result<f32, String>> {
    device.as_mut().map(|g| g.train_step(batch))
}
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn device_step(_: &mut Option<Device>, _: &Batch) -> Option<Result<f32, String>> { None }

/// Weights and optimizer state back from the device, if there is one.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn bring_back(device: &Option<Device>, t: &mut Trainer) {
    if let Some(g) = device { g.download_into(t); }
}
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn bring_back(_: &Option<Device>, _: &mut Trainer) {}

/// The handle's trainer, or the reason it has none.
fn trainer_in<'a>(r: &'a mut std::collections::HashMap<u32, Slot>, ctx: &str, id: u32) -> Result<&'a mut Trainer, R2Err> {
    match r.get_mut(&id) {
        Some(Slot::Training(t)) => Ok(t),
        Some(Slot::Serving(_)) => err!(Runtime,
            "{}: this handle is a loaded model (serving only). \
             Use llm.load(dir, train = TRUE) to fine-tune it.", ctx),
        None => err!(Runtime, "{}: unknown model handle {}", ctx, id),
    }
}

/// The settings `llm.train` and `llm.finetune` share.
struct Settings {
    gpu: bool,
    steps: usize,
    report: usize,
    accum: usize,
    lr: Option<f32>,
    wd: Option<f32>,
    clip: Option<f32>,
    cosine: bool,
    warmup: u64,
    min_lr: Option<f32>,
    checkpoint: Option<String>,
    every: usize,
    keep: usize,
    callback: Option<RVal>,
}

impl Settings {
    fn parse(ctx: &str, a: &[EvalArg], default_steps: f64) -> Result<Settings, R2Err> {
        let steps = num_arg(a, "steps", default_steps) as usize;
        let report = num_arg(a, "report", 0.0) as usize;
        let callback = gn(a, "callback").filter(|v| !matches!(v, RVal::Null));
        match &callback {
            None | Some(RVal::Closure(_) | RVal::BuiltinFn(_)) => {}
            Some(other) => return err!(Runtime, "{}: callback must be a function, got {}", ctx, other.type_name()),
        }
        if callback.is_some() && report == 0 {
            return err!(Runtime, "{}: callback needs report = n, the steps between its calls", ctx);
        }
        let opt = |name: &str| gn(a, name).and_then(|v| v.scalar_f64().ok().flatten()).map(|x| x as f32);
        let cosine = match gn(a, "schedule").map(|v| str_arg(&v)).transpose()?.as_deref() {
            None | Some("constant") => false,
            Some("cosine") => true,
            Some(other) => return err!(Runtime, "{}: unknown schedule '{}' (\"constant\" or \"cosine\")", ctx, other),
        };
        let gpu = match gn(a, "device").map(|v| str_arg(&v)).transpose()?.as_deref() {
            None | Some("cpu") => false,
            Some("gpu") => true,
            Some(other) => return err!(Runtime, "{}: unknown device '{}' (\"cpu\" or \"gpu\")", ctx, other),
        };
        Ok(Settings {
            gpu,
            steps,
            report,
            callback,
            accum: (num_arg(a, "accum", 1.0) as usize).max(1),
            lr: opt("lr"), wd: opt("wd"), clip: opt("clip"),
            cosine,
            warmup: num_arg(a, "warmup", 0.0) as u64,
            min_lr: opt("min.lr"),
            checkpoint: gn(a, "checkpoint").map(|v| str_arg(&v)).transpose()?,
            every: (num_arg(a, "every", (steps / 10).max(1) as f64) as usize).max(1),
            keep: num_arg(a, "keep", 3.0) as usize,
        })
    }

    fn apply(&self, t: &mut Trainer) {
        if let Some(lr) = self.lr { t.opt.lr = lr; }
        if let Some(wd) = self.wd { t.opt.weight_decay = wd; }
        if let Some(c) = self.clip { t.clip = (c > 0.0).then_some(c); }
        if self.cosine {
            let base = t.opt.lr;
            t.sched = Some(LrSchedule { base, min: self.min_lr.unwrap_or(base / 10.0), warmup: self.warmup,
                                        total: t.step + self.steps as u64 });
        }
    }

    /// Does step `n` of this call report: print its loss, or call back?
    fn reports(&self, n: usize) -> bool {
        self.report > 0 && (n % self.report == 0 || n == 1)
    }

    /// The last step to run before the script gets control again: the next
    /// reporting step with a callback, the end of the run without one.
    fn stretch_end(&self, n: usize) -> usize {
        if self.callback.is_none() { return self.steps; }
        let next = if n == 1 { 1 } else { n.div_ceil(self.report) * self.report };
        next.min(self.steps)
    }
}

/// The training loop: one batch per step, from the trainer's own step count,
/// so a later call or a resumed run continues the same sequence.
///
/// The registry stays locked while steps run, and is free only while a
/// callback does: the callback may call any `llm.*`, which takes that lock.
/// So the steps go in stretches between callbacks, and a run without one is a
/// single stretch, with nothing extra per step.
fn run_steps(e: &mut Engine, env: &EnvRef, ctx: &str, id: u32, s: &Settings, tok: &Tokenizer,
             batch_at: impl Fn(usize) -> Batch) -> Result<RVal, R2Err>
{
    if let Some(dir) = &s.checkpoint {
        // the tokenizer sits beside the checkpoints, for llm.resume
        std::fs::create_dir_all(dir)
            .and_then(|_| std::fs::write(std::path::Path::new(dir).join(TOKENIZER_FILE), tok.to_tokenizer_json()))
            .map_err(|e| fail(ctx, format!("checkpoint: {e}")))?;
    }
    let _running = s.callback.as_ref().map(|_| Running::enter(id));
    let (first_step, mut device) = with_registry(|r| {
        let t = trainer_in(r, ctx, id)?;
        s.apply(t);
        Ok::<_, R2Err>((t.step as usize, open_device(ctx, s, t)?))
    })?;
    let mut last = 0.0f32;
    let mut next = 1;
    while next <= s.steps {
        let end = s.stretch_end(next);
        let call_back = s.callback.as_ref().filter(|_| s.reports(end));
        with_registry(|r| {
            let t = trainer_in(r, ctx, id)?;
            for n in next..=end {
                let batch = batch_at(first_step + n - 1);
                last = match device_step(&mut device, &batch) {
                    Some(r) => r,
                    None if s.accum > 1 => {
                        let parts: Vec<&[(Vec<usize>, Vec<usize>)]> = batch.chunks(batch.len().div_ceil(s.accum).max(1)).collect();
                        t.train_step_parts(&parts)
                    }
                    None => t.train_step(&batch),
                }.map_err(|e| fail(ctx, e))?;
                // a callback decides for itself what to show
                if s.callback.is_none() && s.reports(n) {
                    r2_types::out::rout(&format!("step {:>5}  loss {:.4}\n", n, last));
                }
                if let Some(dir) = s.checkpoint.as_ref().filter(|_| n % s.every == 0 || n == s.steps) {
                    bring_back(&device, t);
                    t.save_checkpoint(dir, s.keep).map_err(|e| fail(ctx, e))?;
                }
            }
            // The weights live on the device until here. A callback sees them
            // current, at the price of this read-back on each of its steps.
            if end == s.steps || call_back.is_some() { bring_back(&device, t); }
            Ok::<_, R2Err>(())
        })?;
        if let Some(f) = call_back {
            e.call_fn(f, &[EvalArg { name: None, value: as_num(end as f64) },
                           EvalArg { name: None, value: as_num(last as f64) }], env)?;
        }
        next = end + 1;
    }
    Ok(as_num(last as f64))
}

/// Attach a tokenizer to this handle the first time it trains. A model asked
/// for vocab > 256 learns a BPE from `text` — the merges have to come from
/// somewhere, and the training text is the only corpus the builtin is given.
fn ensure_tokenizer(ctx: &str, id: u32, text: &str, a: &[EvalArg]) -> Result<Tokenizer, R2Err> {
    let want_vocab = with_registry(|r| r.get(&id).map(|s| s.config().vocab)).unwrap_or(256);
    if with_tokenizers(|t| !t.contains_key(&id)) {
        let tok = if want_vocab > 256 {
            Tokenizer::from_trained(&r2_tensor::bpe::train(text, want_vocab))
        } else {
            Tokenizer::byte_level()
        };
        // The LEARNED vocabulary can be smaller than the one requested: a
        // corpus only contains so many useful merges, and a short one runs
        // out early (0.33 MB of prose caps out near 14,500 however much you
        // ask for). Resize the model to match rather than leaving it with a
        // wider output layer than the tokenizer has tokens — that costs
        // `dim * unused` per token forever, and lets the model emit ids the
        // tokenizer cannot decode, which drops characters silently.
        //
        // Safe here and only here: this is the first train call, so the
        // model has seen no data and nothing is discarded.
        let actual = tok.vocab_size();
        if actual != want_vocab {
            let (mut cfg, old_lr) = with_registry(|r| match r.get(&id) {
                Some(Slot::Training(t)) => Some((t.cfg, t.opt.lr)),
                _ => None,
            }).ok_or_else(|| fail(ctx, format!("unknown model handle {id}")))?;
            cfg.vocab = actual;
            let lr = num_arg(a, "lr", old_lr as f64) as f32;
            let rebuilt = Trainer::new(cfg, lr, num_arg(a, "seed", 42.0) as u64).map_err(|e| fail(ctx, e))?;
            with_registry(|r| { r.insert(id, Slot::Training(Box::new(rebuilt))); });
        }
        with_tokenizers(|t| t.insert(id, tok));
    }
    Ok(tokenizer_for(id))
}

/// Training text: the text itself, or the path of a file holding it.
pub(super) fn corpus_text(s: String) -> Result<String, R2Err> {
    if s.len() < 4096 && !s.contains('\n') && std::path::Path::new(&s).is_file() {
        return std::fs::read_to_string(&s).map_err(|e| fail("llm.train", format!("cannot read '{s}': {e}")));
    }
    Ok(s)
}

fn order_arg(a: &[EvalArg], default: bool) -> Order {
    if bool_arg(a, "shuffle", default) { Order::Shuffled(num_arg(a, "seed", 42.0) as u64) } else { Order::Sequential }
}

/// `llm.train(model, text, steps=, seq=, batch=, ...)` — train on a text (a
/// string or a file). Returns the final loss so a script can decide whether
/// to keep going.
pub(crate) fn bi_llm_train(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let id = handle(&gv(a, 0))?;
    refuse_running("llm.train", id)?;
    let text = corpus_text(str_arg(&gv(a, 1))?)?;
    let settings = Settings::parse("llm.train", a, 50.0)?;
    let seq = num_arg(a, "seq", 16.0) as usize;
    let max_batch = num_arg(a, "batch", 8.0) as usize;

    let tok = ensure_tokenizer("llm.train", id, &text, a)?;
    let ids: Vec<u32> = tok.encode(&text).map_err(|e| fail("llm.train", e))?;
    drop(text);
    if ids.len() < seq + 2 {
        return err!(Runtime, "llm.train: text is too short — needs more than {} tokens for seq={}", seq + 1, seq);
    }
    // Each step takes the next windows of the text; the target is the window
    // shifted by one, which is what "predict the next token" means.
    let batcher = Batcher { batch: max_batch.min(ids.len() - seq - 1).max(1), seq, order: order_arg(a, false) };
    run_steps(e, env, "llm.train", id, &settings, &tok, |step| batcher.at(&ids[..], step))
}

pub(super) fn strings(ctx: &str, v: &RVal) -> Result<Vec<String>, R2Err> {
    let RVal::Character(s, _) = v else { return err!(Runtime, "{}: expected text, got {}", ctx, v.type_name()) };
    s.iter().map(|x| x.as_ref().map(|t| t.to_string()).ok_or_else(|| fail(ctx, "NA in the examples"))).collect()
}

/// `llm.finetune(model, prompts, completions, steps=, batch=, seq=, ...)` —
/// teach what follows each prompt. Only completion tokens count in the loss;
/// batches are padded to their longest example. Returns the final loss.
pub(crate) fn bi_llm_finetune(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.finetune";
    let id = handle(&gv(a, 0))?;
    refuse_running(ctx, id)?;
    let (prompts, completions) = (strings(ctx, &gv(a, 1))?, strings(ctx, &gv(a, 2))?);
    if prompts.len() != completions.len() {
        return err!(Runtime, "{}: {} prompts but {} completions", ctx, prompts.len(), completions.len());
    }
    let settings = Settings::parse(ctx, a, 100.0)?;
    let seq = num_arg(a, "seq", 64.0) as usize;
    let batch = (num_arg(a, "batch", 4.0) as usize).max(1);

    let joined: String = prompts.iter().zip(&completions).flat_map(|(p, c)| [p.as_str(), c.as_str()]).collect();
    let tok = ensure_tokenizer(ctx, id, &joined, a)?;
    let encode = |s: &str| tok.encode(s).map_err(|e| fail(ctx, e));
    let pairs: Vec<(Vec<u32>, Vec<u32>)> = prompts.iter().zip(&completions)
        .map(|(p, c)| Ok((encode(p)?, encode(c)?))).collect::<Result<_, R2Err>>()?;
    // seq + 1 ids make `seq` inputs. The pad id is arbitrary: it cannot
    // reach the loss or a gradient (r2_train::data::PairSet).
    let set = PairSet::new(&pairs, seq + 1, 0).map_err(|e| fail(ctx, e))?;
    let order = order_arg(a, true);
    run_steps(e, env, ctx, id, &settings, &tok, |step| set.at(batch, step, order))
}

/// `llm.lora(model, rank=8, alpha=2*rank, targets=)` — freeze the model and
/// add low-rank adapters. Returns how many values now train.
pub(crate) fn bi_llm_lora(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.lora";
    let id = handle(&gv(a, 0))?;
    refuse_running(ctx, id)?;
    let rank = num_arg(a, "rank", 8.0) as usize;
    let alpha = num_arg(a, "alpha", 2.0 * rank as f64) as f32;
    let names = gn(a, "targets").map(|v| strings(ctx, &v)).transpose()?;
    let refs: Option<Vec<&str>> = names.as_ref().map(|n| n.iter().map(String::as_str).collect());
    let seed = num_arg(a, "seed", 42.0) as u64;
    with_registry(|r| match r.get_mut(&id) {
        Some(Slot::Training(t)) => t.enable_lora(rank, alpha, refs.as_deref(), seed)
            .map(|n| as_num(n as f64)).map_err(|e| fail(ctx, e)),
        Some(Slot::Serving(_)) => err!(Runtime, "{}: this handle is serving only; \
                                                 use llm.load(dir, train = TRUE)", ctx),
        None => err!(Runtime, "{}: unknown model handle {}", ctx, id),
    })
}

/// `llm.save.adapter(model, dir)` — write only the adapters. A small file
/// that applies to the model they were trained on. `llm.save` writes the
/// whole model with them merged in.
pub(crate) fn bi_llm_save_adapter(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.save.adapter";
    let id = handle(&gv(a, 0))?;
    let dir = str_arg(&gv(a, 1))?;
    with_registry(|r| match r.get(&id) {
        Some(Slot::Training(t)) => t.save_adapter(&dir).map_err(|e| fail(ctx, e)),
        Some(Slot::Serving(_)) => err!(Runtime, "{}: this handle has no adapters", ctx),
        None => err!(Runtime, "{}: unknown model handle {}", ctx, id),
    })?;
    Ok(RVal::Character(vec![Some(std::sync::Arc::from(dir.as_str()))], Attrs::default()))
}

/// `llm.load.adapter(base, adapter, lr=)` — the model saved in `base` with the
/// adapters saved in `adapter`, as a handle to generate from or keep training.
/// Refuses a base the adapters were not trained on.
pub(crate) fn bi_llm_load_adapter(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.load.adapter";
    let (base, adapter) = (str_arg(&gv(a, 0))?, str_arg(&gv(a, 1))?);
    let m = r2_tensor::model::Model::load_dir(&base).map_err(|e| fail(ctx, e))?;
    let tok = load_tokenizer(&base, &m.cfg).map_err(|e| fail(ctx, e))?;
    let t = Trainer::with_adapter(m, num_arg(a, "lr", 3e-4) as f32, &adapter).map_err(|e| fail(ctx, e))?;
    let id = insert(Slot::Training(Box::new(t)));
    with_tokenizers(|m| m.insert(id, tok));
    Ok(as_num(id as f64))
}

/// `llm.freeze(model, blocks)` — stop training every block whose name starts
/// with one of `blocks`. Returns how many blocks are frozen.
pub(crate) fn bi_llm_freeze(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.freeze";
    let id = handle(&gv(a, 0))?;
    refuse_running(ctx, id)?;
    let names = strings(ctx, &gv(a, 1))?;
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    with_registry(|r| match r.get_mut(&id) {
        Some(Slot::Training(t)) => t.freeze(&refs).map(|n| as_num(n as f64)).map_err(|e| fail(ctx, e)),
        Some(Slot::Serving(_)) => err!(Runtime, "{}: this handle is serving only", ctx),
        None => err!(Runtime, "{}: unknown model handle {}", ctx, id),
    })
}
