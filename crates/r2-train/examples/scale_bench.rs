//! Scaling benchmark: 1M / 10M / 100M parameters.
//!
//! Reports next-token accuracy, memory accounting, and achieved FLOP/s.
//! FLOP/s is the hardware-independent figure: training compute is
//! ~6*N*D (2ND forward + 4ND backward, N params over D tokens), so
//! FLOP/s can be compared against any other implementation on any machine.
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn mem_report(cfg: &Config) -> (f64, f64, f64, f64) {
    let n = cfg.n_params() as f64;
    let params = n * 4.0;              // f32 weights
    let adam = n * 8.0;                // Adam m + v
    let grads = n * 4.0;               // accumulated gradients
    let flat = n * 4.0;                // flattened copy for the optimizer step
    (params, adam, grads, flat)
}

/// Next-token accuracy of the SERVED model over the training sequences:
/// fraction of positions whose argmax equals the target.
fn accuracy(m: &Model, batch: &[(Vec<usize>, Vec<usize>)]) -> Result<f64, String> {
    let (mut right, mut total) = (0usize, 0usize);
    for (inp, tgt) in batch {
        let mut caches = m.new_caches()?;
        for (i, (&tok, &want)) in inp.iter().zip(tgt).enumerate() {
            let logits = m.forward_step(tok, i, &mut caches)?;
            let got = r2_tensor::infer::argmax(&logits);
            if got == want { right += 1; }
            total += 1;
        }
    }
    Ok(right as f64 / total.max(1) as f64)
}

fn run(name: &str, cfg: Config, steps: usize, batch_n: usize, seq: usize)
    -> Result<(), String>
{
    let tok = Tokenizer::byte_level();
    let corpus = "ardon r2 is a statistical runtime built in pure rust. ";
    let ids: Vec<usize> = tok.encode(corpus)?.iter().map(|&i| i as usize).collect();
    let mut batch = Vec::new();
    for s in 0..batch_n.min(ids.len().saturating_sub(seq + 1)) {
        batch.push((ids[s..s + seq].to_vec(), ids[s + 1..s + seq + 1].to_vec()));
    }

    let n = cfg.n_params();
    let (p, a, g, f) = mem_report(&cfg);
    let r2_total = (p + a + g + f) / 1e6;

    let mut tr = Trainer::new(cfg, 0.003, 42)?;
    let t0 = std::time::Instant::now();
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for s in 1..=steps {
        last = tr.train_step(&batch)?;
        if s == 1 { first = last; }
    }
    let dt = t0.elapsed().as_secs_f64();

    let tokens = (steps * batch.len() * seq) as f64;
    let flops = 6.0 * n as f64 * tokens;      // 2ND forward + 4ND backward
    let acc = accuracy(&tr.to_model()?, &batch)?;

    println!("{:<8} {:>12} {:>7} {:>9.3} {:>9.3} {:>7.1}% {:>9.1} {:>8.2} {:>9.2}",
             name, n, steps, first, last, acc * 100.0, r2_total, dt,
             flops / dt / 1e9);
    Ok(())
}

fn main() -> Result<(), String> {
    println!("{:<8} {:>12} {:>7} {:>9} {:>9} {:>8} {:>9} {:>8} {:>9}",
             "size", "params", "steps", "loss0", "loss1", "acc", "R2 MB", "secs", "GFLOP/s");
    println!("{}", "-".repeat(92));

    let v = Tokenizer::byte_level().vocab_size();
    // ~1M
    run("1M", Config { dim: 128, n_heads: 4, n_kv_heads: 2, n_layers: 5, vocab: v,
                       ffn_hidden: 384, max_seq: 64, rope_base: 10000.0, eps: 1e-5 },
        60, 4, 16)?;
    // ~10M
    run("10M", Config { dim: 384, n_heads: 6, n_kv_heads: 2, n_layers: 6, vocab: v,
                        ffn_hidden: 1024, max_seq: 64, rope_base: 10000.0, eps: 1e-5 },
        20, 4, 16)?;
    // ~100M
    run("100M", Config { dim: 768, n_heads: 12, n_kv_heads: 4, n_layers: 12, vocab: v,
                         ffn_hidden: 3072, max_seq: 64, rope_base: 10000.0, eps: 1e-5 },
        4, 2, 16)?;
    // Same 10M model, but WIDER work per step: 16 sequences x 32 tokens
    // instead of 4 x 16. Same arithmetic per token; far better matmul
    // shape, which is what decides hardware utilization.
    run("10M-wide", Config { dim: 384, n_heads: 6, n_kv_heads: 2, n_layers: 6, vocab: v,
                             ffn_hidden: 1024, max_seq: 64, rope_base: 10000.0, eps: 1e-5 },
        6, 16, 32)?;
    // ~1B — projected only; see the note printed below.
    let b1 = Config { dim: 2048, n_heads: 16, n_kv_heads: 4, n_layers: 24, vocab: v,
                      ffn_hidden: 8192, max_seq: 64, rope_base: 10000.0, eps: 1e-5 };
    let (p, a, gr, f) = mem_report(&b1);
    println!("{:<8} {:>12} {:>7} {:>9} {:>9} {:>8} {:>9.1} {:>8} {:>9}",
             "1B", b1.n_params(), "-", "-", "-", "-", (p + a + gr + f) / 1e6, "-", "-");
    Ok(())
}
