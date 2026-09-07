//! LMO-1 — the embedding step, R2 against PyTorch's, and nothing else.
//!
//! R2 computes the token embedding as a ONE-HOT MATMUL (`llm.rs`):
//!
//!     onehot[t x vocab]  with a single 1.0 per row
//!     x = onehot @ table          (t x vocab) @ (vocab x d)
//!
//! PyTorch computes it as a GATHER — `aten::embedding` dispatches to
//! `index_select`, which copies row `tok_i` of the table into row `i` of
//! the output, and its backward is a scatter-add into the touched rows.
//!
//! The two produce the SAME numbers. They do wildly different amounts of
//! work, and the difference scales with the VOCABULARY:
//!
//!     one-hot matmul   2 * t * vocab * d  multiply-adds, and a
//!                      t x vocab buffer built and zeroed every forward
//!     gather           t * d              element copies, no buffer
//!
//! At vocab 256 that ratio is 256; at vocab 8000 it is 8000. The
//! vocabulary was 256 until a BPE tokenizer existed, which is why this
//! stage could be ignored — it cannot be now.
//!
//! Timed through the SHIPPING `Tape`, not a reimplementation, so what is
//! measured is what training runs.
//!
//!     R2_LMO_TOKENS=2048 R2_LMO_VOCABS=256,8000,32000 \
//!       cargo run --release -p r2-train --example lmo1_embedding
//!     python benchmarks/llm/lmo1_embedding.py

use r2_autograd::Tape;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median of 9 batches. A single reading on this machine is an OS stall
/// until proven otherwise.
fn t_us(reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..reps.min(5) { f(); }
    let mut r = Vec::new();
    for _ in 0..9 {
        let s = std::time::Instant::now();
        for _ in 0..reps { f(); }
        r.push(s.elapsed().as_secs_f64() / reps as f64 * 1e6);
    }
    median(r)
}

fn main() {
    let t: usize = std::env::var("R2_LMO_TOKENS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(2048);
    let d: usize = std::env::var("R2_LMO_DIM").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(256);
    let vocabs: Vec<usize> = std::env::var("R2_LMO_VOCABS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 8000, 32000]);

    println!("LMO-1 — embedding step only, R2's one-hot matmul");
    println!("  {t} tokens, dim {d}, microseconds, median of 9\n");
    println!("{:>7} {:>12} {:>9} {:>10} {:>11} {:>12} {:>12}",
             "vocab", "onehot MB", "leaf", "build", "forward", "fwd+bwd", "MFLOP fwd");
    println!("  (leaf is the table copy onto the tape, SUBTRACTED from the two right columns)");
    println!("{}", "-".repeat(80));

    for &vocab in &vocabs {
        // Total work per rep is O(t*vocab*d), so hold it roughly fixed.
        let reps = ((2e9 / (t as f64 * vocab as f64 * d as f64)).ceil() as usize)
            .clamp(3, 200);
        let table: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.0007).sin()).collect();
        let tokens: Vec<usize> = (0..t).map(|i| (i * 7919) % vocab).collect();
        let g: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.005).sin()).collect();

        // ── build the one-hot, exactly as `forward_fused` does ─────────
        let build = t_us(reps, || {
            let mut onehot = vec![0.0f32; t * vocab];
            for (i, &tok) in tokens.iter().enumerate() { onehot[i * vocab + tok] = 1.0; }
            std::hint::black_box(onehot.len());
        });

        // ── the table leaf, measured so it can be SUBTRACTED ───────────
        // `tape.leaf(p.clone(), true)` copies the whole table onto the tape
        // — 8 MB at vocab 8000. That is the tape's ownership cost, paid by
        // every parameter block, not the embedding's. Charging it here
        // would inflate this stage and is exactly the attribution error
        // that produced a wrong LMO-1 number once already.
        let mut tl = Tape::new();
        let leaf_only = t_us(reps, || {
            tl = Tape::new();
            std::hint::black_box(tl.leaf(table.clone(), true));
        });

        // ── forward: build + leaf + matmul ─────────────────────────────
        let mut tp = Tape::new();
        let fwd = t_us(reps, || {
            tp = Tape::new();
            let mut onehot = vec![0.0f32; t * vocab];
            for (i, &tok) in tokens.iter().enumerate() { onehot[i * vocab + tok] = 1.0; }
            let oh = tp.leaf(onehot, false);
            let w = tp.leaf(table.clone(), true);
            let x = tp.matmul(oh, w, t, vocab, d);
            std::hint::black_box(tp.value(x).len());
        });

        // ── forward + backward, which is what a training step runs ─────
        let fb = t_us(reps, || {
            let mut tb = Tape::new();
            let mut onehot = vec![0.0f32; t * vocab];
            for (i, &tok) in tokens.iter().enumerate() { onehot[i * vocab + tok] = 1.0; }
            let oh = tb.leaf(onehot, false);
            let w = tb.leaf(table.clone(), true);
            let x = tb.matmul(oh, w, t, vocab, d);
            let s = tb.leaf(g.clone(), false);
            let p = tb.mul(x, s);
            let l = tb.sum_all(p);
            tb.backward(l);
            std::hint::black_box(tb.grad(w).len());
        });

        let mflop = 2.0 * t as f64 * vocab as f64 * d as f64 / 1e6;
        // Table leaf subtracted from both, so what remains is the
        // embedding: build the one-hot, multiply, and its backward.
        let fwd_net = (fwd - leaf_only).max(0.0);
        let fb_net = (fb - leaf_only).max(0.0);
        println!("{vocab:>7} {:>12.1} {leaf_only:>9.1} {build:>10.1} {fwd_net:>11.1} {fb_net:>12.1} {mflop:>12.1}", (t * vocab * 4) as f64 / 1e6);
    }

    println!("\nA gather does t*d = {} element copies for ANY vocabulary.", t * d);
    println!("The one-hot form's cost grows linearly with the vocabulary, so");
    println!("moving from a byte tokenizer to BPE multiplies this stage by ~31x.");
}
