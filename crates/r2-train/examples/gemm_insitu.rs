//! Per-call GEMM latency INSIDE real training steps, by shape.
//!
//! The counterpart of `benchmarks/llm/mm_profile.py` (PyTorch's GEMMs
//! profiled inside its steps). Op-level timings with hot operands do not
//! predict a step — the pack-free experiment ran 0.7x in isolation and
//! +9..22% worse in training — so both sides are compared here, in situ.
//! `R2_GEMM_STATS=1` makes `sgemm` record every parallel call under
//! 0.5 GFLOP; this prints percentiles per shape and the step total.
//! Takes the same shape knobs as `tinystories_train`.
//!
//!     R2_GEMM_STATS=1 cargo run --release -p r2-train --example gemm_insitu
//!     R2_GEMM_STATS=1 R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 \
//!         R2_SEQ=256 R2_BATCH=8 cargo run --release -p r2-train --example gemm_insitu

use r2_tensor::model::Config;
use r2_train::llm::Trainer;
use std::collections::BTreeMap;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn pct(v: &[f32], p: f64) -> f32 { v[((v.len() - 1) as f64 * p).round() as usize] }

fn main() {
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
    println!("gemm_insitu — dim {} x {} layers, ffn {}, {}/{} heads, {bn} x {seq} = {} tokens/step, {steps} steps after an 8 s warm-up\n",
             cfg.dim, cfg.n_layers, cfg.ffn_hidden, cfg.n_heads, cfg.n_kv_heads, bn * seq);
    if std::env::var("R2_GEMM_STATS").map(|v| v != "1").unwrap_or(true) {
        println!("  (set R2_GEMM_STATS=1 or no per-call rows will be recorded)\n");
    }

    let t0 = std::time::Instant::now();
    while t0.elapsed().as_secs_f64() < 8.0 { let _ = tr.train_step(&batch).expect("warm"); }
    let _ = r2_linalg::gemm::take_gemm_stats();

    let mut step_ms = Vec::new();
    for _ in 0..steps {
        let s = std::time::Instant::now();
        let _ = tr.train_step(&batch).expect("step");
        step_ms.push(s.elapsed().as_secs_f32() * 1e3);
    }
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut by: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
    for (key, us, _) in r2_linalg::gemm::take_gemm_stats() { by.entry(key).or_default().push(us); }
    println!("{:>18} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9}", "m x k x n", "calls", "p50 us", "p90", "p99", "max", "ms/step");
    println!("{}", "-".repeat(74));
    let mut total = 0.0f64;
    for (key, mut v) in by {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (m, k, n) = (key >> 40, (key >> 20) & 0xFFFFF, key & 0xFFFFF);
        let ms = v.iter().map(|&x| x as f64).sum::<f64>() / 1e3 / steps as f64;
        total += ms;
        println!("{:>18} {:>7} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.2}",
                 format!("{m}x{k}x{n}"), v.len() / steps, pct(&v, 0.5), pct(&v, 0.9), pct(&v, 0.99), v[v.len() - 1], ms);
    }
    println!("{}", "-".repeat(74));
    println!("small GEMMs (< 0.5 GFLOP): {total:.1} ms/step;  step p50 {:.1} ms, p90 {:.1}, max {:.1}",
             pct(&step_ms, 0.5), pct(&step_ms, 0.9), step_ms[step_ms.len() - 1]);
}
