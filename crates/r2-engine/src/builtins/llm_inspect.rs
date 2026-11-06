//! `llm.inspect`: gradient and activation summaries per block, as data frames.
//!
//! ```r
//! look <- llm.inspect(m, text, seq = 32, batch = 4)
//! look$blocks        # block, params, trainable, weight.norm, grad.norm, ...
//! look$activations   # stage, mean, sd, max.abs, nonfinite
//! ```
//!
//! Costs one forward and backward pass per call and that pass's memory
//! (`r2_train::inspect`); a run that never calls it pays nothing.

use std::sync::Arc;

use r2_train::data::{Batch, Batcher, Order, PairSet};
use r2_train::inspect::Inspection;
use r2_types::*;

use super::llm::{as_num, handle, num_arg, str_arg, tokenizer_for, with_registry, Slot};
use super::llm_train::{corpus_text, fail, strings};
use crate::{err, gn, gv, Engine};

/// The batch to look at: the first windows of a text, as `llm.train` takes
/// them, or the first examples of `completions =`, as `llm.finetune` does.
fn batch_of(ctx: &str, id: u32, a: &[EvalArg]) -> Result<Batch, R2Err> {
    let tok = tokenizer_for(id);
    let encode = |s: &str| tok.encode(s).map_err(|e| fail(ctx, e));
    if let Some(answers) = gn(a, "completions") {
        let (prompts, answers) = (strings(ctx, &gv(a, 1))?, strings(ctx, &answers)?);
        if prompts.len() != answers.len() {
            return err!(Runtime, "{}: {} prompts but {} completions", ctx, prompts.len(), answers.len());
        }
        let seq = num_arg(a, "seq", 64.0) as usize;
        let batch = (num_arg(a, "batch", 4.0) as usize).max(1);
        let pairs: Vec<(Vec<u32>, Vec<u32>)> = prompts.iter().zip(&answers)
            .map(|(p, c)| Ok((encode(p)?, encode(c)?))).collect::<Result<_, R2Err>>()?;
        let set = PairSet::new(&pairs, seq + 1, 0).map_err(|e| fail(ctx, e))?;
        return Ok(set.at(batch, 0, Order::Sequential));
    }
    let text = corpus_text(str_arg(&gv(a, 1))?)?;
    let seq = num_arg(a, "seq", 16.0) as usize;
    let batch = (num_arg(a, "batch", 8.0) as usize).max(1);
    let ids = encode(&text)?;
    if ids.len() < seq + 2 {
        return err!(Runtime, "{}: text is too short: needs more than {} tokens for seq={}", ctx, seq + 1, seq);
    }
    let windows = Batcher { batch: batch.min(ids.len() - seq - 1).max(1), seq, order: Order::Sequential };
    Ok(windows.at(&ids[..], 0))
}

fn numbers(xs: impl Iterator<Item = Option<f64>>) -> RVal {
    RVal::Numeric(xs.collect::<Vec<_>>().into(), Attrs::default())
}

fn frame(columns: Vec<(&str, RVal)>) -> RVal {
    RVal::DataFrame(DataFrame {
        columns: columns.into_iter().map(|(n, v)| (Arc::from(n), v)).collect(),
        row_names: None,
    })
}

fn labels<'a>(xs: impl Iterator<Item = &'a str>) -> RVal {
    RVal::Character(xs.map(|s| Some(Arc::from(s))).collect(), Attrs::default())
}

fn as_list(look: &Inspection) -> RVal {
    let b = &look.blocks;
    let blocks = frame(vec![
        ("block", labels(b.iter().map(|x| x.name.as_str()))),
        ("params", numbers(b.iter().map(|x| Some(x.params as f64)))),
        ("trainable", RVal::Logical(b.iter().map(|x| Some(x.trainable)).collect::<Vec<_>>().into(), Attrs::default())),
        ("weight.norm", numbers(b.iter().map(|x| Some(x.weight_norm)))),
        ("grad.norm", numbers(b.iter().map(|x| x.grad_norm))),
        ("grad.max", numbers(b.iter().map(|x| x.grad_max))),
        // scale free, so blocks of different sizes can be compared
        ("grad.ratio", numbers(b.iter().map(|x| x.grad_norm.map(|g| g / x.weight_norm)))),
    ]);
    let s = &look.activations;
    let activations = frame(vec![
        ("stage", labels(s.iter().map(|x| x.stage.as_str()))),
        ("mean", numbers(s.iter().map(|x| Some(x.summary.mean)))),
        ("sd", numbers(s.iter().map(|x| Some(x.summary.sd)))),
        ("max.abs", numbers(s.iter().map(|x| Some(x.summary.max_abs)))),
        ("nonfinite", numbers(s.iter().map(|x| Some(x.summary.nonfinite as f64)))),
    ]);
    RVal::list(vec![
        (Some("loss".into()), as_num(look.loss as f64)),
        (Some("blocks".into()), blocks),
        (Some("activations".into()), activations),
    ])
}

/// `llm.inspect(model, text, seq=16, batch=8)`, or
/// `llm.inspect(model, prompts, completions=, seq=64, batch=4)`.
pub(crate) fn bi_llm_inspect(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let ctx = "llm.inspect";
    let id = handle(&gv(a, 0))?;
    let batch = batch_of(ctx, id, a)?;
    let look = with_registry(|r| match r.get_mut(&id) {
        Some(Slot::Training(t)) => t.inspect(&batch).map_err(|e| fail(ctx, e)),
        Some(Slot::Serving(_)) => err!(Runtime,
            "{}: this handle is a loaded model (serving only). \
             Use llm.load(dir, train = TRUE) to look at its gradients.", ctx),
        None => err!(Runtime, "{}: unknown model handle {}", ctx, id),
    })?;
    Ok(as_list(&look))
}
