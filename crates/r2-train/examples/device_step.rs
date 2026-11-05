//! Seconds per training step, the device against the CPU trainer, on synthetic
//! tokens, labelled with the device that ran. The first device steps compile
//! kernels and fill buffers, so they are dropped; the figure is the median of
//! the rest. Set `R2_GPU_STATS=<step>` for the per-kernel census of that step.
//!
//!     cargo run --release -p r2-train --features cuda --example device_step

use r2_tensor::model::Config;
use r2_train::gpu_llm::GpuTrainer;
use r2_train::llm::Trainer;
use std::time::Instant;

fn batch(nseq: usize, seq: usize, vocab: usize, step: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut s = 0x9E3779B97F4A7C15u64 ^ step.wrapping_mul(0xD1342543DE82EF95);
    let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 33) as usize % vocab };
    (0..nseq).map(|_| {
        let toks: Vec<usize> = (0..=seq).map(|_| next()).collect();
        (toks[..seq].to_vec(), toks[1..].to_vec())
    }).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("device: {}", r2_gpu::adapter_info());
    println!("{:<8} {:>8} {:>10} {:>10} {:>7} {:>11} {:>11}", "model", "params", "device s", "CPU s", "ratio", "device loss", "CPU loss");
    // the two models of benchmarks/llm/REPORT.md
    let small = Config { dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4, vocab: 8000, ffn_hidden: 768, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    let medium = Config { dim: 768, n_heads: 12, n_kv_heads: 4, n_layers: 4, vocab: 8000, ffn_hidden: 2304, max_seq: 256, rope_base: 10000.0, eps: 1e-5 };
    for (name, c, nseq, seq) in [("small", small, 32usize, 64usize), ("medium", medium, 8, 256)] {
        let mut cpu = Trainer::new(c, 3e-4, 7).expect("model");
        let params: usize = cpu.params.iter().map(Vec::len).sum();
        let mut dev = match GpuTrainer::from_trainer(&cpu) {
            Ok(d) => d,
            Err(e) => { println!("no device figures: {e}"); return; }
        };
        let (mut dt, mut dl) = (Vec::new(), 0.0);
        for s in 0..23u64 {
            let b = batch(nseq, seq, c.vocab, s);
            let t = Instant::now();
            dl = dev.train_step(&b).expect("device step");
            if s >= 3 { dt.push(t.elapsed().as_secs_f64()); }
        }
        let (mut ct, mut cl) = (Vec::new(), 0.0);
        for s in 0..23u64 {
            let b = batch(nseq, seq, c.vocab, s);
            let t = Instant::now();
            cl = cpu.train_step(&b).expect("cpu step");
            if s >= 3 { ct.push(t.elapsed().as_secs_f64()); }
        }
        let (d, k) = (median(dt), median(ct));
        println!("{name:<8} {:>7.1}M {d:>10.4} {k:>10.4} {:>6.1}x {dl:>11.4} {cl:>11.4}", params as f64 / 1e6, k / d);
    }
}
