//! Byte-level or BPE? Decide it by measurement, not by the 4x folklore.
//!
//! # Why raw loss cannot answer this
//!
//! A vocab-256 model and a vocab-8000 model do not have comparable losses.
//! Cross-entropy is per TOKEN, and the two are predicting different things:
//! guessing 1 of 256 bytes is an easier question than guessing 1 of 8000
//! word-pieces, so the byte model's loss is lower while it says LESS about
//! the text. Comparing them directly would pick the wrong tokenizer.
//!
//! The comparable quantity is **bits per byte**:
//!
//! ```text
//! bpb = loss_nats_per_token * (tokens / bytes) / ln(2)
//! ```
//!
//! which is how much information the model needs to reproduce the raw text,
//! independent of how that text was chopped up. It is the standard metric
//! for exactly this comparison.
//!
//! # What is held fixed
//!
//! A fixed TEXT budget, not a fixed token budget. The whole claim for BPE
//! is that it covers the same text in fewer tokens, so pinning tokens would
//! assume the answer. Every arm sees the same bytes of the same corpus and
//! the same number of optimizer steps; what differs is how much text each
//! step contains, which is the effect being measured.
//!
//!     R2_CORPUS=corpus.txt R2_STEPS=60 R2_SEQ=64 R2_BATCH=16 \
//!       cargo run --release -p r2-train --example tokenizer_tradeoff

use r2_tensor::bpe;
use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

struct Arm {
    label: String,
    vocab: usize,
    tokens: Vec<usize>,
    /// Bytes of source text the token stream covers — the denominator that
    /// makes the arms comparable.
    bytes: usize,
    train_s: f64,
}

fn main() {
    let path: String = std::env::var("R2_CORPUS").unwrap_or_else(|_| "corpus.txt".into());
    let steps: usize = env("R2_STEPS", 60);
    let seq: usize = env("R2_SEQ", 64);
    let bn: usize = env("R2_BATCH", 16);
    let vocabs: Vec<usize> = std::env::var("R2_VOCABS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 2000, 8000, 16000]);

    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}\nset R2_CORPUS to a UTF-8 text file");
        std::process::exit(1);
    });
    let mut end = raw.len();
    while end > 0 && std::str::from_utf8(&raw[..end]).is_err() { end -= 1; }
    let corpus = String::from_utf8_lossy(&raw[..end]).into_owned();

    println!("Byte-level vs BPE — same corpus, same steps, same model shape");
    println!("  corpus  {path}  ({:.2} MB)", corpus.len() as f64 / 1e6);
    println!("  schedule {steps} steps x {bn} x {seq}\n");

    let mut arms: Vec<Arm> = Vec::new();
    for &v in &vocabs {
        let (label, tok) = if v <= 256 {
            ("byte-level".to_string(), Tokenizer::byte_level())
        } else {
            let t0 = std::time::Instant::now();
            let trained = bpe::train(&corpus, v);
            let s = t0.elapsed().as_secs_f64();
            (format!("BPE {v} (trained in {s:.1}s)"), Tokenizer::from_trained(&trained))
        };
        let ids: Vec<usize> = tok.encode(&corpus)
            .expect("encode").iter().map(|&i| i as usize).collect();
        arms.push(Arm { label, vocab: tok.vocab_size(), tokens: ids,
                        bytes: corpus.len(), train_s: 0.0 });
    }

    // ── train one model per arm, identical shape and schedule ──────────
    println!("{:<34} {:>8} {:>10} {:>9} {:>9} {:>9}",
             "arm", "vocab", "tok/byte", "loss", "bits/byte", "sec");
    println!("{}", "-".repeat(84));

    let mut rows: Vec<(String, usize, f64, f32, f64, f64, f64)> = Vec::new();
    for arm in arms.iter_mut() {
        let cfg = Config {
            dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4,
            vocab: arm.vocab, ffn_hidden: 768, max_seq: seq.max(64),
            rope_base: 10000.0, eps: 1e-5,
        };
        let mut tr = match Trainer::new(cfg, 3e-4, 1) {
            Ok(t) => t,
            Err(e) => { println!("{:<34} trainer failed: {e}", arm.label); continue; }
        };

        let mut cursor = 0usize;
        let mut loss = 0.0f32;
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let mut batch = Vec::with_capacity(bn);
            for _ in 0..bn {
                if cursor + seq + 1 >= arm.tokens.len() { cursor = 0; }
                batch.push((arm.tokens[cursor..cursor + seq].to_vec(),
                            arm.tokens[cursor + 1..cursor + seq + 1].to_vec()));
                cursor += seq;
            }
            loss = tr.train_step(&batch).expect("step");
        }
        arm.train_s = t0.elapsed().as_secs_f64();

        // tokens per byte, and the conversion that makes arms comparable.
        let tpb = arm.tokens.len() as f64 / arm.bytes as f64;
        let bpb = loss as f64 * tpb / std::f64::consts::LN_2;
        // Text actually covered by the run: steps * batch * seq tokens,
        // converted back to bytes. This is the throughput a USER sees.
        let text_bytes = (steps * bn * seq) as f64 / tpb;
        rows.push((arm.label.clone(), arm.vocab, tpb, loss, bpb,
                   arm.train_s, text_bytes / arm.train_s));
        println!("{:<34} {:>8} {:>10.3} {:>9.4} {:>9.4} {:>9.1}",
                 arm.label, arm.vocab, tpb, loss, bpb, arm.train_s);
    }

    println!("\n{:<34} {:>14} {:>16}", "arm", "text bytes/s", "vs byte-level");
    println!("{}", "-".repeat(68));
    let base = rows.first().map(|r| r.6).unwrap_or(1.0);
    for (label, _, _, _, _, _, bps) in &rows {
        println!("{:<34} {:>14.0} {:>15.2}x", label, bps, bps / base);
    }

    println!("\nbits/byte is the only cross-tokenizer quality metric here.");
    println!("LOWER is better; raw loss is NOT comparable across vocabularies.");
}
