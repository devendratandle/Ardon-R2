//! One training step split into its three phases, each timed on its own:
//!
//!   forward    parameters onto the tape, the fused forward, softmax_ce
//!   backward   `Tape::backward` — every gradient
//!   optimizer  weights back off the tape, Adam in place
//!
//! The same split as `benchmarks/llm/phase_split.py` runs in PyTorch, on
//! the same model and the same 32 x 64 = 2,048 tokens per step, so the two
//! tables line up phase for phase. Warms to the sustained clock first (see
//! `step_census` for why), then reports the median of `R2_STEPS` steps.
//!
//! The phase sum is also checked against a plain `train_step` timed whole:
//! if the phases do not add up to the step, the split is wrong.
//!
//!     cargo run --release -p r2-train --example phase_split

use r2_autograd::{Tape, Var};
use r2_tensor::model::Config;
use r2_train::llm::Trainer;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    let cfg = Config { dim: env("R2_DIM", 256), n_heads: env("R2_HEADS", 4), n_kv_heads: env("R2_KV", 2),
                       n_layers: env("R2_LAYERS", 4), vocab: env("R2_VOCAB", 8000),
                       ffn_hidden: env("R2_FFN", 768), max_seq: env("R2_SEQ", 64usize).max(64),
                       rope_base: 10000.0, eps: 1e-5 };
    let (bn, seq) = (env("R2_BATCH", 32usize), env("R2_SEQ", 64usize));
    let steps: usize = env("R2_STEPS", 20);
    let mut tr = Trainer::new(cfg, 3e-4, 7).expect("trainer");
    let batch: Vec<(Vec<usize>, Vec<usize>)> = (0..bn).map(|b| {
        let inp: Vec<usize> = (0..seq).map(|i| (b * 131 + i * 7919 + 1) % cfg.vocab).collect();
        let tgt: Vec<usize> = (0..seq).map(|i| (b * 131 + i * 7919 + 2) % cfg.vocab).collect();
        (inp, tgt)
    }).collect();
    let mut toks = Vec::with_capacity(bn * seq);
    let mut tgts = Vec::with_capacity(bn * seq);
    for (i, t) in &batch { toks.extend_from_slice(i); tgts.extend_from_slice(t); }

    println!("R2  — dim {} x {} layers, ffn {}, vocab {}, {bn} x {seq} = {} tokens/step, {} params",
             cfg.dim, cfg.n_layers, cfg.ffn_hidden, cfg.vocab, bn * seq, tr.n_params());

    // warm to the sustained clock
    let t0 = std::time::Instant::now();
    while t0.elapsed().as_secs_f64() < 8.0 { let _ = tr.train_step(&batch).expect("warm"); }

    let (mut fw, mut bw, mut op, mut whole) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut take, mut adam, mut dropt) = (Vec::new(), Vec::new(), Vec::new());
    let use_pool = std::env::var("R2_POOL").map(|v| v != "0").unwrap_or(true);
    let mut pool = r2_autograd::BufPool::new();
    for _ in 0..steps {
        // ── phase-split step: the body of Trainer::train_step, timed ──
        let s0 = std::time::Instant::now();
        let mut tape = if use_pool { Tape::with_pool(std::mem::take(&mut pool)) } else { Tape::new() };
        let leaves: Vec<Var> = tr.params.iter_mut()
            .map(|p| tape.leaf(std::mem::take(p), true)).collect();
        let logits = tr.forward_fused_census(&mut tape, &toks, seq, &leaves);
        let loss = tape.softmax_ce(logits, cfg.vocab, tgts.clone());
        std::hint::black_box(tape.value(loss)[0]);
        let t_f = s0.elapsed().as_secs_f64() * 1e3;

        let s1 = std::time::Instant::now();
        tape.backward(loss);
        let t_b = s1.elapsed().as_secs_f64() * 1e3;

        let s2 = std::time::Instant::now();
        for (i, &lv) in leaves.iter().enumerate() { tr.params[i] = tape.take_value(lv); }
        let grads: Vec<&[f32]> = leaves.iter().map(|&lv| tape.grad(lv)).collect();
        let t_take = s2.elapsed().as_secs_f64() * 1e3;
        let s2b = std::time::Instant::now();
        let Trainer { params, opt, .. } = &mut tr;
        opt.step_blocks(params, &grads, 1.0).expect("adam");
        let t_adam = s2b.elapsed().as_secs_f64() * 1e3;
        let s2c = std::time::Instant::now();
        drop(grads);
        if use_pool { pool = tape.into_pool(); } else { drop(tape); }
        let t_drop = s2c.elapsed().as_secs_f64() * 1e3;
        let t_o = s2.elapsed().as_secs_f64() * 1e3;
        fw.push(t_f); bw.push(t_b); op.push(t_o);
        take.push(t_take); adam.push(t_adam); dropt.push(t_drop);

        // ── the same step unsplit, for the cross-check ──
        let s3 = std::time::Instant::now();
        let _ = tr.train_step(&batch).expect("step");
        whole.push(s3.elapsed().as_secs_f64() * 1e3);
    }
    let (f, b, o, w) = (median(fw), median(bw), median(op), median(whole));
    println!("\n{:<12} {:>10} {:>8}", "phase", "ms/step", "share");
    println!("{}", "-".repeat(32));
    for (l, v) in [("forward", f), ("backward", b), ("optimizer", o)] {
        println!("{l:<12} {v:>10.1} {:>7.1}%", v / (f + b + o) * 100.0);
    }
    println!("{:<12} {:>10.1}   of which: weights off tape {:.1}, Adam {:.1}, tape drop {:.1}",
             "", o, median(take.clone()), median(adam.clone()), median(dropt.clone()));
    println!("{}", "-".repeat(32));
    println!("{:<12} {:>10.1}", "phase sum", f + b + o);
    println!("{:<12} {:>10.1}   (train_step timed whole; should match the sum)", "whole step", w);
    println!("\nR2_PHASES forward={f:.2} backward={b:.2} optimizer={o:.2} whole={w:.2}");

    // In-situ backward census (R2_TAPE_STATS=1): ms per step by op kind,
    // over the split steps AND the whole steps timed above (2 x steps).
    let stats = r2_autograd::take_backward_stats();
    if !stats.is_empty() {
        let mut by: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
        for (k, ms) in &stats { let e = by.entry(k).or_insert((0, 0.0)); e.0 += 1; e.1 += *ms as f64; }
        let denom = (2 * steps) as f64;
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        println!("\nbackward census (in situ), ms per step:");
        println!("{:<14} {:>8} {:>10} {:>8}", "op", "arms", "ms/step", "share");
        let total: f64 = rows.iter().map(|r| r.1 .1).sum::<f64>() / denom;
        for (k, (n, ms)) in &rows {
            println!("{k:<14} {:>8} {:>10.1} {:>7.1}%", n / (2 * steps), ms / denom, ms / denom / total * 100.0);
        }
        println!("{:<14} {:>8} {:>10.1}", "total", "", total);
    }
}
