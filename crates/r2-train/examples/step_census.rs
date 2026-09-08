//! Where does a training step go NOW, after sgemm and the fused attention?
//!
//! The three GEMM cases account for ~500 ms of a ~1,095 ms step and
//! attention for ~50 ms. This finds the rest, because an unattributed half
//! of the step is where the next win is hiding — and guessing at it is how
//! the last several targets were picked wrongly.
//!
//!     cargo run --release -p r2-train --example step_census

use r2_autograd::Tape;
use r2_tensor::model::Config;
use r2_train::llm::Trainer;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn t_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    median((0..5).map(|_| {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        s.elapsed().as_secs_f64() / reps as f64 * 1e3
    }).collect())
}

fn main() {
    let cfg = Config { dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4,
                       vocab: 8000, ffn_hidden: 768, max_seq: 64,
                       rope_base: 10000.0, eps: 1e-5 };
    let (bn, seq) = (32usize, 64usize);
    let t = bn * seq;
    let mut tr = Trainer::new(cfg, 3e-4, 7).expect("trainer");
    let batch: Vec<(Vec<usize>, Vec<usize>)> = (0..bn).map(|b| {
        let inp: Vec<usize> = (0..seq).map(|i| (b * 131 + i * 7919 + 1) % cfg.vocab).collect();
        let tgt: Vec<usize> = (0..seq).map(|i| (b * 131 + i * 7919 + 2) % cfg.vocab).collect();
        (inp, tgt)
    }).collect();
    let tgts: Vec<usize> = batch.iter().flat_map(|(_, t)| t.clone()).collect();

    println!("dim {} x {} layers, vocab {}, {bn} x {seq} = {t} tokens/step",
             cfg.dim, cfg.n_layers, cfg.vocab);
    println!("{} params in {} blocks\n", tr.n_params(), tr.params.len());

    // WARM TO THE SUSTAINED CLOCK BEFORE MEASURING ANYTHING.
    //
    // This is a 15 W part: the first measurement of a run gets boost clock
    // and everything after it does not. An earlier version timed the full
    // step first and the stages afterwards, and the stages summed to 155%
    // of the step -- not because the accounting was wrong but because the
    // step had been timed fast and the stages slow. Shares taken that way
    // are biased toward whatever is measured last.
    {
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() < 8.0 {
            let _ = tr.train_step(&batch).expect("warmup");
        }
    }
    let full = t_ms(2, || { let _ = tr.train_step(&batch).expect("step"); });
    let fb = t_ms(2, || { let _ = tr.loss_logits_grads(&batch).expect("fb"); });

    // The tape's copy of every parameter, once per step.
    let params = tr.params.clone();
    let leaves = t_ms(3, || {
        let mut tp = Tape::new();
        for p in &params { std::hint::black_box(tp.leaf(p.clone(), true)); }
    });

    // softmax_ce over the vocabulary: 2,048 x 8,000 = 16.4M exp() calls.
    let logits: Vec<f32> = (0..t * cfg.vocab).map(|i| (i as f32 * 0.0001).sin()).collect();
    let ce = t_ms(3, || {
        let mut tp = Tape::new();
        let l = tp.leaf(logits.clone(), true);
        let loss = tp.softmax_ce(l, cfg.vocab, tgts.clone());
        tp.backward(loss);
        std::hint::black_box(tp.grad(l).len());
    });
    let ce_leaf = t_ms(3, || {
        let mut tp = Tape::new();
        std::hint::black_box(tp.leaf(logits.clone(), true));
    });

    // The REAL tape a step builds — activations included, not just params —
    // because backward() zeroes every element of it before it starts.
    let (elems, nodes, zero_ms) = {
        let mut tp = Tape::new();
        let lv: Vec<_> = tr.params.iter().map(|p| tp.leaf(p.clone(), true)).collect();
        let toks: Vec<usize> = batch.iter().flat_map(|(i, _)| i.clone()).collect();
        let logits = tr.forward_fused_census(&mut tp, &toks, seq, &lv);
        let loss = tp.softmax_ce(logits, cfg.vocab, tgts.clone());
        let e = tp.elements();
        let n = tp.len();
        // backward() twice: the second pays only the zeroing plus the same
        // arithmetic, so this is an upper bound on the zeroing, not it.
        let z = t_ms(2, || { tp.zero_grads_census(); });
        let _ = loss;
        (e, n, z)
    };

    // The elementwise ops nothing else has accounted for.
    let x: Vec<f32> = (0..t * cfg.dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let w: Vec<f32> = (0..cfg.dim).map(|i| (i as f32 * 0.01).cos()).collect();
    let rms = t_ms(20, || { std::hint::black_box(r2_tensor::ops::rmsnorm(&x, &w, 1e-5)); });
    let h: Vec<f32> = (0..t * cfg.ffn_hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let silu = t_ms(20, || {
        std::hint::black_box(h.iter().map(|&v| r2_tensor::ops::silu(v)).collect::<Vec<f32>>());
    });

    // Is the unattributed remainder the tape's ALLOCATION churn? Replay the
    // same node-size distribution the real step produces: two Vecs per node
    // (value + gradient), allocated fresh and first-touched, then dropped.
    let sizes: Vec<usize> = {
        let mut tp = Tape::new();
        let lv: Vec<_> = tr.params.iter().map(|p| tp.leaf(p.clone(), true)).collect();
        let toks: Vec<usize> = batch.iter().flat_map(|(i, _)| i.clone()).collect();
        let lg = tr.forward_fused_census(&mut tp, &toks, seq, &lv);
        let _ = tp.softmax_ce(lg, cfg.vocab, tgts.clone());
        (0..tp.len()).map(|i| tp.value(r2_autograd::Var(i)).len()).collect()
    };
    let churn = t_ms(2, || {
        let mut keep: Vec<Vec<f32>> = Vec::with_capacity(sizes.len() * 2);
        for &n in &sizes {
            let mut v = vec![0.0f32; n];
            let mut g = vec![0.0f32; n];
            // first touch, which is where the page faults are
            if n > 0 { v[0] = 1.0; g[0] = 1.0; v[n - 1] = 1.0; g[n - 1] = 1.0; }
            keep.push(v); keep.push(g);
        }
        std::hint::black_box(keep.len());
    });

    // sgemm, measured live rather than pasted in: the three cases at the
    // five shapes a step runs, weighted by how often each is called.
    let gemm_ms: f64 = [(2048usize, 256usize, 8000usize, 1usize),
                        (2048, 256, 768, 8), (2048, 768, 256, 4),
                        (2048, 256, 256, 8), (2048, 256, 128, 8)]
        .iter().map(|&(m, k, n, cnt)| {
            use r2_linalg::gemm::{sgemm, Trans};
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002).cos()).collect();
            let gg: Vec<f32> = (0..m * n).map(|i| (i as f32 * 0.003).sin()).collect();
            let nn = t_ms(2, || { std::hint::black_box(sgemm(&a, Trans::No, &b, Trans::No, m, k, n, true)); });
            let nt = t_ms(2, || { std::hint::black_box(sgemm(&gg, Trans::No, &b, Trans::Yes, m, n, k, true)); });
            let tn = t_ms(2, || { std::hint::black_box(sgemm(&a, Trans::Yes, &gg, Trans::No, k, m, n, true)); });
            (nn + nt + tn) * cnt as f64
        }).sum();

    // ── the elementwise ops, forward AND backward, at the per-step counts
    // a step actually runs. This is the unattributed remainder.
    let d = cfg.dim;
    let h = cfg.ffn_hidden;
    // one op on the tape, timed with its backward, minus the leaf cost
    let op = |n: usize, build: &dyn Fn(&mut Tape, r2_autograd::Var, r2_autograd::Var) -> r2_autograd::Var| -> f64 {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).cos()).collect();
        let seed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003).sin()).collect();
        let whole = t_ms(5, || {
            let mut tp = Tape::new();
            let x = tp.leaf(a.clone(), true);
            let y = tp.leaf(b.clone(), true);
            let o = build(&mut tp, x, y);
            tp.backward_from(o, &seed);
            std::hint::black_box(tp.grad(x).len());
        });
        let bare = t_ms(5, || {
            let mut tp = Tape::new();
            std::hint::black_box((tp.leaf(a.clone(), true), tp.leaf(b.clone(), true)));
        });
        (whole - bare).max(0.0)
    };
    let add_ms  = op(t * d, &|tp, x, y| tp.add(x, y)) * 8.0;
    let mul_ms  = op(t * h, &|tp, x, y| { let _ = y; tp.mul(x, x) }) * 4.0;
    let silu_ms = op(t * h, &|tp, x, y| { let _ = y; tp.silu(x) }) * 4.0;
    let rms_ms = {
        let xv: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.001).sin()).collect();
        let wv: Vec<f32> = (0..d).map(|i| (i as f32 * 0.01).cos()).collect();
        let seed: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.003).sin()).collect();
        t_ms(5, || {
            let mut tp = Tape::new();
            let x = tp.leaf(xv.clone(), true);
            let w = tp.leaf(wv.clone(), true);
            let o = tp.rmsnorm(x, w, d, 1e-5);
            tp.backward_from(o, &seed);
            std::hint::black_box(tp.grad(x).len());
        }) * 8.0
    };
    let rope_ms = {
        let qv: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.001).sin()).collect();
        let seed: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.003).sin()).collect();
        t_ms(5, || {
            let mut tp = Tape::new();
            let q = tp.leaf(qv.clone(), true);
            let o = tp.rope_seq(q, t, seq, cfg.n_heads, cfg.head_dim(), 10000.0);
            tp.backward_from(o, &seed);
            std::hint::black_box(tp.grad(q).len());
        }) * 8.0
    };
    let embed_ms = {
        let tab = tr.params[0].clone();
        let toks: Vec<usize> = batch.iter().flat_map(|(i, _)| i.clone()).collect();
        let seed: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.003).sin()).collect();
        t_ms(5, || {
            let mut tp = Tape::new();
            let w = tp.leaf(tab.clone(), true);
            let o = tp.embed(w, &toks, d);
            tp.backward_from(o, &seed);
            std::hint::black_box(tp.grad(w).len());
        })
    };

    let pct = |x: f64| x / full * 100.0;
    println!("{:>34} {:>10} {:>8}", "stage", "ms/step", "share");
    println!("{}", "-".repeat(54));
    println!("{:>34} {:>10.1} {:>7.1}%", "FULL train_step", full, 100.0);
    println!("{:>34} {:>10.1} {:>7.1}%", "  forward + backward + grads", fb, pct(fb));
    println!("{:>34} {:>10.1} {:>7.1}%", "  optimizer + flatten + writeback", full - fb, pct(full - fb));
    println!("{}", "-".repeat(54));
    println!("  inside forward+backward:");
    println!("{:>34} {:>10.1} {:>7.1}%", "sgemm x3 (fwd, grad_A, grad_B)", gemm_ms, pct(gemm_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "attention (4 layers)", 50.0, pct(50.0));
    println!("{:>34} {:>10.1} {:>7.1}%", "softmax_ce fwd+bwd", ce - ce_leaf, pct(ce - ce_leaf));
    println!("{:>34} {:>10.1} {:>7.1}%", "tape's copy of every parameter", leaves, pct(leaves));
    println!("{:>34} {:>10.1} {:>7.1}%", "backward()'s blanket grad zeroing", zero_ms, pct(zero_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "tape alloc churn (2 Vecs/node)", churn, pct(churn));
    println!("  -- elementwise, fwd+bwd, at per-step counts --");
    println!("{:>34} {:>10.1} {:>7.1}%", "silu x4", silu_ms, pct(silu_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "mul x4", mul_ms, pct(mul_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "add x8", add_ms, pct(add_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "rmsnorm x8", rms_ms, pct(rms_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "rope_seq x8", rope_ms, pct(rope_ms));
    println!("{:>34} {:>10.1} {:>7.1}%", "embed x1", embed_ms, pct(embed_ms));
    let named = gemm_ms + 50.0 + (ce - ce_leaf) + (full - fb) + zero_ms + churn
        + silu_ms + mul_ms + add_ms + rms_ms + rope_ms + embed_ms;
    println!("{}", "-".repeat(54));
    println!("{:>34} {:>10.1} {:>7.1}%", "accounted for", named, pct(named));
    println!("{:>34} {:>10.1} {:>7.1}%", "STILL UNATTRIBUTED", full - named, pct(full - named));
    // Re-take the step total AFTER the stage timings, so a run that drifted
    // anyway is visible instead of silently wrong.
    let full_end = t_ms(2, || { let _ = tr.train_step(&batch).expect("step"); });
    println!("  step total {full:.1} ms before the stage timings, {full_end:.1} ms after ({:+.1}% drift)",
             (full_end - full) / full * 100.0);
    println!("\n  tape holds {elems} elements; backward() zeroes every one of");
    println!("  them before it starts, and PyTorch has no equivalent step.");
}
