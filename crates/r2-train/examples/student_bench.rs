//! What a student on a basic laptop actually experiences:
//! how long to train something, how fast does it answer, does it learn.
use r2_tensor::infer::{argmax, Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn accuracy(m: &Model, batch: &[(Vec<usize>, Vec<usize>)]) -> f64 {
    let (mut ok, mut n) = (0usize, 0usize);
    for (inp, tgt) in batch {
        let mut c = m.new_caches().unwrap();
        for (i, (&tk, &want)) in inp.iter().zip(tgt).enumerate() {
            if argmax(&m.forward_step(tk, i, &mut c).unwrap()) == want { ok += 1; }
            n += 1;
        }
    }
    ok as f64 / n.max(1) as f64
}

fn run(label: &str, cfg: Config, steps: usize) {
    let tok = Tokenizer::byte_level();
    let corpus = "ardon r2 is a statistical runtime written in pure rust. ";
    let ids: Vec<usize> = tok.encode(corpus).unwrap().iter().map(|&i| i as usize).collect();
    let seq = 16;
    let mut batch = Vec::new();
    for s in 0..6 { batch.push((ids[s..s+seq].to_vec(), ids[s+1..s+seq+1].to_vec())); }

    let mut tr = Trainer::new(cfg, 0.003, 42).unwrap();
    let t0 = std::time::Instant::now();
    let mut loss = 0.0;
    for _ in 0..steps { loss = tr.train_step(&batch).unwrap(); }
    let train_s = t0.elapsed().as_secs_f64();

    let m = tr.to_model().unwrap();
    let acc = accuracy(&m, &batch);

    // Response: latency to the FIRST token, then sustained rate.
    let prompt: Vec<usize> = tok.encode("ardon r2 is").unwrap().iter().map(|&i| i as usize).collect();
    let mut s = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
    let t1 = std::time::Instant::now();
    let _ = m.generate(&prompt, 1, &mut s, None).unwrap();
    let first_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = std::time::Instant::now();
    let out = m.generate(&prompt, 64, &mut s, None).unwrap();
    let tps = out.len() as f64 / t2.elapsed().as_secs_f64();

    // Memory: weights for serving, ~5x that for training (Adam state).
    let wt_mb = cfg.n_params() as f64 * 4.0 / 1e6;
    println!("{:<7} {:>11} {:>8.1}s {:>8.3} {:>7.0}% {:>10.1} {:>9.0} {:>8.0}",
             label, cfg.n_params(), train_s, loss, acc * 100.0, first_ms, tps, wt_mb * 5.0);
}

fn main() {
    let v = Tokenizer::byte_level().vocab_size();
    println!("{:<7} {:>11} {:>9} {:>8} {:>8} {:>10} {:>9} {:>8}",
             "model", "params", "train", "loss", "acc", "1st tok ms", "tok/s", "train MB");
    println!("{}", "-".repeat(78));
    run("1M",  Config { dim:128, n_heads:4, n_kv_heads:2, n_layers:5, vocab:v,
                        ffn_hidden:384, max_seq:128, rope_base:10000.0, eps:1e-5 }, 120);
    run("3M",  Config { dim:192, n_heads:6, n_kv_heads:2, n_layers:6, vocab:v,
                        ffn_hidden:512, max_seq:128, rope_base:10000.0, eps:1e-5 }, 60);
    run("10M", Config { dim:384, n_heads:6, n_kv_heads:2, n_layers:6, vocab:v,
                        ffn_hidden:1024, max_seq:128, rope_base:10000.0, eps:1e-5 }, 30);
    println!("\n(train MB = weights + Adam optimizer state; serving needs ~1/5 of it)");
}
