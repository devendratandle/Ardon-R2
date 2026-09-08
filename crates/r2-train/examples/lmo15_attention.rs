//! LMO-15 — the attention block, forward and backward, and nothing else.
//!
//! The queue (`benchmarks/llm/REPORT.md`) has carried this line for a
//! long time:
//!
//!     LMO-15  Attention backward  26.4%  OPEN — never compared to PyTorch
//!
//! It is the largest open component and it has never had a reference
//! number. This produces one, against PyTorch AND JAX, on the shapes R2
//! actually trains at.
//!
//! WHAT IS MEASURED. `Op::Attention` — exactly what `llm.rs::forward_fused`
//! runs between the q/k/v projections and the output projection, with
//! nothing else on the tape. Grouped-query, causal, scale 1/sqrt(head_dim).
//! The decomposition it replaced is not measured here; it survives as the
//! oracle in `attention_matches_the_decomposition`, where it is checked
//! against rather than timed.
//!
//! The backward is seeded with `backward_from(ctx, g)` — PyTorch's
//! `.backward(g)` — so both sides differentiate the same function from the
//! same upstream gradient, with no manufactured scalar loss in between.
//!
//! The table leaves are built ONCE, outside the timer, because q/k/v are
//! activations that already exist when attention runs; timing their
//! creation would measure the projections, which are LMO-3/6/8/10/11.
//!
//!     cargo run --release -p r2-train --example lmo15_attention
//!     python benchmarks/llm/lmo15_attention.py

use r2_autograd::Tape;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let hd = env("R2_LMO_HD", 64);
    let n_heads = env("R2_LMO_HEADS", 4);
    let n_kv = env("R2_LMO_KV", 2);
    let seq = env("R2_LMO_SEQ", 64);
    // Batch counts to sweep: the shipping step is 32 sequences of 64
    // (2,048 tokens); the queue's profile was taken at 4,096.
    let batches: Vec<usize> = std::env::var("R2_LMO_BATCHES").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 32, 64]);

    println!("LMO-15 — attention block only, R2");
    println!("  seq {seq}, {n_heads} query heads, {n_kv} kv heads, head_dim {hd}");
    println!("  microseconds, median of 9\n");
    println!("{:>8} {:>8} {:>12} {:>14} {:>10}",
             "batch", "tokens", "fwd", "fwd+bwd", "bwd only");
    println!("{}", "-".repeat(56));

    let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new();
    for &bn in &batches {
        let t = bn * seq;
        let mk = |n: usize, ph: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.0013 + ph).sin()).collect()
        };
        let qv = mk(t * n_heads * hd, 0.0);
        let kv_ = mk(t * n_kv * hd, 1.0);
        let vv = mk(t * n_kv * hd, 2.0);
        let g = mk(t * n_heads * hd, 3.0);
        // Total attention work is O(bn * seq^2), so hold it roughly fixed.
        let reps = ((2e7 / (bn as f64 * (seq * seq) as f64)).ceil() as usize).clamp(2, 20);
        let scale = 1.0 / (hd as f32).sqrt();

        // ── forward only ───────────────────────────────────────────────
        // q/k/v are ACTIVATIONS: they exist before attention runs, so the
        // leaves are created outside the timer. Timing them would be
        // timing the projections, which are a different LMO stage.
        let fwd = {
            let mut r = Vec::new();
            for _ in 0..9 {
                let mut tp = Tape::new();
                let (a, b, c) = (tp.leaf(qv.clone(), true), tp.leaf(kv_.clone(), true),
                                 tp.leaf(vv.clone(), true));
                for _ in 0..2 { std::hint::black_box(tp.attention(a, b, c, bn, seq, n_heads, n_kv, hd, scale)); }
                let s = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(tp.attention(a, b, c, bn, seq, n_heads, n_kv, hd, scale));
                }
                r.push(s.elapsed().as_secs_f64() / reps as f64 * 1e6);
            }
            median(r)
        };

        // ── forward + backward, seeded exactly as `.backward(g)` ───────
        let fb = {
            let mut r = Vec::new();
            for _ in 0..9 {
                let s = std::time::Instant::now();
                for _ in 0..reps {
                    let mut tp = Tape::new();
                    let (a, b, c) = (tp.leaf(qv.clone(), true), tp.leaf(kv_.clone(), true),
                                     tp.leaf(vv.clone(), true));
                    let ctx = tp.attention(a, b, c, bn, seq, n_heads, n_kv, hd, scale);
                    tp.backward_from(ctx, &g);
                    std::hint::black_box(tp.grad(a).len());
                }
                r.push(s.elapsed().as_secs_f64() / reps as f64 * 1e6);
            }
            median(r)
        };
        // The leaf creation that `fb` pays and `fwd` does not, so the
        // backward can be reported on its own.
        let leaves = {
            let mut r = Vec::new();
            for _ in 0..9 {
                let s = std::time::Instant::now();
                for _ in 0..reps {
                    let mut tp = Tape::new();
                    std::hint::black_box((tp.leaf(qv.clone(), true), tp.leaf(kv_.clone(), true),
                                          tp.leaf(vv.clone(), true)));
                }
                r.push(s.elapsed().as_secs_f64() / reps as f64 * 1e6);
            }
            median(r)
        };
        let bwd = (fb - fwd - leaves).max(0.0);
        println!("{bn:>8} {t:>8} {fwd:>12.1} {fb:>14.1} {bwd:>10.1}");
        rows.push((bn, t, fwd, fb));
    }

    let mut js = format!("{{\n  \"seq\": {seq}, \"heads\": {n_heads}, \"kv\": {n_kv}, \"hd\": {hd},\n  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 { js.push_str(",\n"); }
        js.push_str(&format!("    {{\"batch\": {}, \"tokens\": {}, \"fwd\": {:.1}, \"fb\": {:.1}}}",
                             r.0, r.1, r.2, r.3));
    }
    js.push_str("\n  ]\n}\n");
    match std::fs::write("lmo15_r2.json", js) {
        Ok(()) => println!("\nwrote lmo15_r2.json — run `python benchmarks/llm/lmo15_attention.py`\n\
                            for the joint R2 / PyTorch / JAX table and the verdict."),
        Err(e) => eprintln!("could not write lmo15_r2.json: {e}"),
    }
}
