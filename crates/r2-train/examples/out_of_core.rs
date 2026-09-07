//! Train from a memory-mapped token file instead of an in-RAM `Vec`.
//!
//! The limit this removes is a capability limit, not a speed one: holding
//! the token stream as `Vec<usize>` costs eight bytes per token, so the
//! corpus size at which training becomes impossible is fixed by RAM and
//! has nothing to do with how fast anything runs.
//!
//! Both arms train the SAME model on the SAME tokens for the SAME steps.
//! The only difference is where the tokens live. Loss must match to the
//! last digit — if it does not, the store is not returning what the Vec
//! returned, and every number here is meaningless.
//!
//!     R2_CORPUS=corpus.txt R2_STEPS=10 R2_SEQ=64 R2_BATCH=16 R2_VOCAB=8000 \
//!       cargo run --release -p r2-train --example out_of_core

use r2_tensor::bpe;
use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;
use r2_train::tokens::{TokenStore, TokenWriter};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn mb(bytes: usize) -> f64 { bytes as f64 / 1e6 }

fn main() {
    let path: String = std::env::var("R2_CORPUS").unwrap_or_else(|_| "corpus.txt".into());
    let steps: usize = env("R2_STEPS", 10);
    let seq: usize = env("R2_SEQ", 64);
    let bn: usize = env("R2_BATCH", 16);
    let vocab: usize = env("R2_VOCAB", 8000);
    let tok_path: String = std::env::var("R2_TOKENS")
        .unwrap_or_else(|_| "corpus.tokens".into());

    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut end = raw.len();
    while end > 0 && std::str::from_utf8(&raw[..end]).is_err() { end -= 1; }
    let corpus = String::from_utf8_lossy(&raw[..end]).into_owned();

    println!("Out-of-core token storage");
    println!("  corpus {path}  ({:.1} MB)\n", mb(corpus.len()));

    let tk = if vocab <= 256 {
        Tokenizer::byte_level()
    } else {
        let mut cut = corpus.len().min(4_000_000);
        while cut > 0 && !corpus.is_char_boundary(cut) { cut -= 1; }
        Tokenizer::from_trained(&bpe::train(&corpus[..cut], vocab))
    };
    let ids: Vec<usize> = tk.encode(&corpus).expect("encode")
        .iter().map(|&i| i as usize).collect();

    // ── write once ─────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let mut w = TokenWriter::create(&tok_path).expect("create token file");
    w.write(&ids).expect("write tokens");
    let written = w.finish().expect("finish");
    let write_s = t0.elapsed().as_secs_f64();

    let store = TokenStore::open(&tok_path).expect("open token file");
    assert_eq!(store.len(), written, "token file length disagrees with what was written");

    println!("{:<34}{:>14}{:>14}", "", "in RAM", "memory-mapped");
    println!("{}", "-".repeat(62));
    println!("{:<34}{:>14}{:>14}", "tokens", ids.len(), store.len());
    println!("{:<34}{:>13.1}M{:>13.1}M", "held by the token stream",
             mb(ids.len() * std::mem::size_of::<usize>()),
             mb(store.resident_bytes()));
    println!("{:<34}{:>13.1}M{:>13.1}M", "on disk", 0.0, mb(store.len() * 4));
    println!("\nwrote {} tokens in {write_s:.2}s ({:.1} MB/s)",
             written, mb(store.len() * 4) / write_s);

    // ── both arms train identically ────────────────────────────────────
    let cfg = Config {
        dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4,
        vocab: tk.vocab_size(), ffn_hidden: 768, max_seq: seq.max(64),
        rope_base: 10000.0, eps: 1e-5,
    };
    let need = steps * bn * (seq + 1);
    assert!(ids.len() >= need, "corpus yields {} tokens, needs {need}", ids.len());

    let run_vec = || {
        let mut tr = Trainer::new(cfg, 3e-4, 1).expect("trainer");
        let (mut cursor, mut loss) = (0usize, 0.0f32);
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut batch = Vec::with_capacity(bn);
            for _ in 0..bn {
                if cursor + seq + 1 >= ids.len() { cursor = 0; }
                batch.push((ids[cursor..cursor + seq].to_vec(),
                            ids[cursor + 1..cursor + seq + 1].to_vec()));
                cursor += seq;
            }
            loss = tr.train_step(&batch).expect("step");
        }
        (loss, t0.elapsed().as_secs_f64())
    };

    let run_mapped = || {
        let mut tr = Trainer::new(cfg, 3e-4, 1).expect("trainer");
        let (mut cursor, mut loss) = (0usize, 0.0f32);
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut batch = Vec::with_capacity(bn);
            for _ in 0..bn {
                if cursor + seq + 1 >= store.len() { cursor = 0; }
                // Two windows of `seq`: the inputs and the same shifted by
                // one. Each is a few kilobytes — the cost is O(seq), never
                // O(corpus), which is the whole point of the exercise.
                let x = store.window(cursor, seq).expect("window");
                let y = store.window(cursor + 1, seq).expect("window");
                batch.push((x, y));
                cursor += seq;
            }
            loss = tr.train_step(&batch).expect("step");
        }
        (loss, t0.elapsed().as_secs_f64())
    };

    let (loss_v, secs_v) = run_vec();
    let (loss_m, secs_m) = run_mapped();

    println!("\n{:<34}{:>14}{:>14}", "", "in RAM", "memory-mapped");
    println!("{}", "-".repeat(62));
    println!("{:<34}{:>14.6}{:>14.6}", format!("loss after {steps} steps"), loss_v, loss_m);
    println!("{:<34}{:>13.2}s{:>13.2}s", "training", secs_v, secs_m);
    println!("{:<34}{:>14.0}{:>14.0}", "tokens/s",
             (steps * bn * seq) as f64 / secs_v, (steps * bn * seq) as f64 / secs_m);

    // The identity check. Anything else and the comparison is void.
    assert_eq!(loss_v, loss_m,
               "mapped training diverged from in-RAM training — the store is \
                not returning the same tokens");
    println!("\nloss is BIT-IDENTICAL between the two paths.");
    // Quote the SHAPE of the saving, not a ratio against a handle size —
    // "784930x" is arithmetic on a 48-byte struct and means nothing.
    println!("RAM for tokens: {:.1} MB of Vec<usize> -> a {}-byte handle.",
             mb(ids.len() * std::mem::size_of::<usize>()), store.resident_bytes());
    println!("Resident cost is O(batch x seq) = {} tokens, not O(corpus) = {}.",
             bn * seq, store.len());
    println!("On disk the stream is {:.1} MB, half the Vec<usize> it replaces,",
             mb(store.len() * 4));
    println!("and a second run over this corpus skips tokenization entirely.");
}
