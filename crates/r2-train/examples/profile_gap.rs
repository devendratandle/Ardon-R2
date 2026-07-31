//! Where does a training step's time actually go, and how far is each
//! part from what this hardware can do?
use r2_tensor::model::Config;
use r2_tensor::ops;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warm
    let t = std::time::Instant::now();
    for _ in 0..reps { f(); }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let v = Tokenizer::byte_level().vocab_size();
    let cfg = Config { dim: 384, n_heads: 6, n_kv_heads: 2, n_layers: 6, vocab: v,
                       ffn_hidden: 1024, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    let (d, kv, h, seq, bn) = (cfg.dim, cfg.kv_dim(), cfg.ffn_hidden, 16usize, 6usize);
    let rows = seq * bn;

    // ---- 1. What the machine can do on the shapes we actually use ----
    println!("PER-OP RATE AT TRAINING SHAPES (measured)");
    let mk = |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32 * 0.001).sin()).collect() };
    let mut mm_rate = 0.0;
    for &(m, k, n, label) in &[
        (rows, d, d, "q/k/v/o proj"), (rows, d, h, "ffn up/gate"), (rows, h, d, "ffn down"),
    ] {
        let (a, b) = (mk(m * k), mk(k * n));
        let dt = time(20, || { std::hint::black_box(ops::matmul(&a, &b, m, k, n)); });
        let gf = 2.0 * m as f64 * k as f64 * n as f64 / dt / 1e9;
        if label == "ffn up/gate" { mm_rate = gf; }
        println!("  {:<14} {:>4}x{:<4}x{:<5} {:>8.2} ms  {:>7.1} GFLOP/s", label, m, k, n, dt*1e3, gf);
    }
    // Non-matmul ops at the same width.
    let x = mk(rows * d);
    let w = vec![1.0f32; d];
    let dt_rms = time(200, || { std::hint::black_box(ops::rmsnorm(&x, &w, 1e-5)); });
    let sc = mk(rows * seq);
    let dt_sm = time(200, || { std::hint::black_box(ops::softmax(&sc, seq)); });
    let g1 = mk(rows * h); let g2 = mk(rows * h);
    let dt_sw = time(200, || { std::hint::black_box(ops::swiglu(&g1, &g2)); });
    println!("  {:<14} {:>15} {:>8.3} ms", "rmsnorm", "", dt_rms*1e3);
    println!("  {:<14} {:>15} {:>8.3} ms", "softmax", "", dt_sm*1e3);
    println!("  {:<14} {:>15} {:>8.3} ms", "swiglu", "", dt_sw*1e3);

    // ---- 2. A real training step ----
    let corpus = "ardon r2 is a statistical runtime written in pure rust. ";
    let tok = Tokenizer::byte_level();
    let ids: Vec<usize> = tok.encode(corpus).unwrap().iter().map(|&i| i as usize).collect();
    let mut batch = Vec::new();
    for s in 0..bn { batch.push((ids[s..s+seq].to_vec(), ids[s+1..s+seq+1].to_vec())); }
    let mut tr = Trainer::new(cfg, 0.003, 1).unwrap();
    let dt_step = time(6, || { tr.train_step(&batch).unwrap(); });

    // ---- 3. How much of that step SHOULD be matmul? ----
    // Per layer, forward: q,k,v (d->d,kv,kv), o (d->d), gate,up (d->h), down (h->d)
    let per_layer_fwd = 2.0 * rows as f64 * (
        (d*d) as f64 + 2.0*(d*kv) as f64 + (d*d) as f64 + 2.0*(d*h) as f64 + (h*d) as f64);
    let embed_out = 2.0 * rows as f64 * (cfg.vocab * d) as f64 * 2.0;
    let fwd = per_layer_fwd * cfg.n_layers as f64 + embed_out;
    let total_mm = fwd * 3.0;                  // backward is ~2x forward
    let predicted = total_mm / (mm_rate * 1e9);

    println!("\nTRAINING STEP BREAKDOWN");
    println!("  measured step time         {:>9.1} ms", dt_step*1e3);
    println!("  matmul FLOPs in the step   {:>9.2} GFLOP", total_mm/1e9);
    println!("  time IF matmul ran at its  {:>9.1} ms   ({:.0}% of the step)",
             predicted*1e3, predicted/dt_step*100.0);
    println!("  measured rate              {:>9.1} GFLOP/s", total_mm/dt_step/1e9);
    println!("  everything else (overhead) {:>9.1} ms   ({:.0}% of the step)",
             (dt_step-predicted)*1e3, (dt_step-predicted)/dt_step*100.0);

    // ---- 4. Distance to hardware ----
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // AVX2: 8 lanes x 2 (FMA) x 2 FMA units, ~2.5 GHz sustained all-core.
    let peak = cores as f64 * 8.0 * 2.0 * 2.0 * 2.5;
    println!("\nDISTANCE TO HARDWARE ({} cores, AVX2+FMA)", cores);
    println!("  theoretical peak           {:>9.0} GFLOP/s", peak);
    println!("  our matmul                 {:>9.1} GFLOP/s  ({:.1}% of peak)", mm_rate, mm_rate/peak*100.0);
    println!("  our training               {:>9.1} GFLOP/s  ({:.1}% of peak)",
             total_mm/dt_step/1e9, total_mm/dt_step/1e9/peak*100.0);
}
