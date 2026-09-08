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
    println!("{:>7} {:>10} {:>8} {:>11} {:>12} {:>10} {:>10} {:>10}",
             "vocab", "onehot MB", "leaf", "OH fwd", "OH fwd+bwd",
             "GA fwd", "GA fwd+bwd", "speedup");
    println!("  OH = the one-hot matmul this replaces; GA = the gather that ships.");
    println!("  The table's leaf copy is measured and SUBTRACTED from all four.");
    println!("{}", "-".repeat(84));

    // (vocab, oh_fwd, oh_fb, ga_fwd, ga_fb, leaf) kept for the JSON that
    // the PyTorch side joins against.
    let mut rows: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();
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
            tb.backward_from(x, &g);
            std::hint::black_box(tb.grad(w).len());
        });

        // ── the GATHER, which is what ships now ────────────────────────
        // NO subtraction here. The table leaf is created ONCE, outside the
        // timer, which is exactly PyTorch's situation: its parameter tensor
        // persists across calls and only the output is allocated per call.
        // Timing `Tape::new() + leaf + embed` and subtracting a separately
        // measured `leaf` is a difference of two large noisy numbers — it
        // printed 658.6 / 69.9 / 3167.2 us for IDENTICAL work across the
        // three vocabularies, which is not a measurement.
        let g_fwd = {
            let mut r = Vec::new();
            for _ in 0..9 {
                let mut tg = Tape::new();
                let w = tg.leaf(table.clone(), true);
                for _ in 0..reps.min(5) { std::hint::black_box(tg.embed(w, &tokens, d)); }
                let s = std::time::Instant::now();
                for _ in 0..reps { std::hint::black_box(tg.embed(w, &tokens, d)); }
                r.push(s.elapsed().as_secs_f64() / reps as f64 * 1e6);
            }
            median(r)
        };
        // `backward_from(x, g)` IS PyTorch's `.backward(g)`. This used to
        // build `mul(x, g)` then `sum_all` to manufacture a scalar for
        // `backward` — an extra leaf, an extra elementwise multiply, a
        // reduction and all three of their backwards, none of which the
        // PyTorch side ran. It measured ~4 ms of a ~6.6 ms reading, so the
        // published fwd+bwd ratio was a comparison of R2's HARNESS against
        // PyTorch's embedding.
        let g_fb = (t_us(reps, || {
            let mut tg = Tape::new();
            let w = tg.leaf(table.clone(), true);
            let x = tg.embed(w, &tokens, d);
            tg.backward_from(x, &g);
            std::hint::black_box(tg.grad(w).len());
        }) - leaf_only).max(0.0);

        let mflop = 2.0 * t as f64 * vocab as f64 * d as f64 / 1e6;
        // Table leaf subtracted from both, so what remains is the
        // embedding: build the one-hot, multiply, and its backward.
        let fwd_net = (fwd - leaf_only).max(0.0);
        let fb_net = (fb - leaf_only).max(0.0);
        println!("{vocab:>7} {:>10.1} {leaf_only:>8.1} {fwd_net:>11.1} {fb_net:>12.1} {g_fwd:>10.1} {g_fb:>10.1} {:>9.0}x", (t * vocab * 4) as f64 / 1e6,
                 if g_fb > 0.0 { fb_net / g_fb } else { 0.0 });
        rows.push((vocab, fwd_net, fb_net, g_fwd, g_fb, leaf_only));
        let _ = (build, mflop);
    }

    // Emit the numbers so the PyTorch side can JOIN them and print ONE
    // table with a verdict per row. Two programs printing two tables
    // leaves the comparison to a human, and the comparison is the whole
    // point: an R2-versus-R2 improvement that is still behind PyTorch is
    // not a result. `benchmarks/llm/lmo1_embedding.py` reads this file.
    let mut js = format!("{{\n  \"tokens\": {t}, \"dim\": {d},\n  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 { js.push_str(",\n"); }
        js.push_str(&format!(
            "    {{\"vocab\": {}, \"oh_fwd\": {:.1}, \"oh_fb\": {:.1}, \"ga_fwd\": {:.1}, \"ga_fb\": {:.1}, \"leaf\": {:.1}}}",
            r.0, r.1, r.2, r.3, r.4, r.5));
    }
    js.push_str("\n  ]\n}\n");
    match std::fs::write("lmo1_r2.json", js) {
        Ok(()) => println!("\nwrote lmo1_r2.json — run `python benchmarks/llm/lmo1_embedding.py`\n\
                            for the joint R2-vs-PyTorch table and the per-row verdict."),
        Err(e) => eprintln!("could not write lmo1_r2.json: {e}"),
    }

    println!("\nA gather does t*d = {} element copies for ANY vocabulary.", t * d);
    println!("The one-hot form's cost grows linearly with the vocabulary, so");
    println!("moving from a byte tokenizer to BPE multiplies this stage by ~31x.");
}
