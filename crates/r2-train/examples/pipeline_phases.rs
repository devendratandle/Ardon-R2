//! Where does a training run's wall time ACTUALLY go, phase by phase?
//!
//! The question this answers: how much resistance does the tokenizer create
//! in a real pipeline? Not from a standalone throughput figure and not from
//! arithmetic — by timing every phase of the same run.
//!
//!   read      pull the corpus off disk
//!   learn     train BPE merges (BPE arms only)
//!   tokenize  text -> token ids
//!   train     the optimizer steps
//!
//! Two arms on the same corpus and the same STEP COUNT: byte-level
//! (vocab 256, one token per byte) and BPE. Holding steps fixed rather than
//! text fixed is deliberate — it is what a training script actually
//! controls, and it makes "text covered per second" the output rather than
//! an input.
//!
//! Reported per arm: each phase's seconds and share, tokens/s, and TEXT
//! bytes/s. The last is the one that matters, because a token is not a
//! fixed amount of text and comparing tokens/s across vocabularies is
//! meaningless.
//!
//!     R2_CORPUS=corpus.txt R2_STEPS=40 R2_SEQ=64 R2_BATCH=32 \
//!       cargo run --release -p r2-train --example pipeline_phases

use r2_tensor::bpe;
use r2_tensor::model::Config;
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let path: String = std::env::var("R2_CORPUS").unwrap_or_else(|_| "corpus.txt".into());
    let steps: usize = env("R2_STEPS", 40);
    let seq: usize = env("R2_SEQ", 64);
    let bn: usize = env("R2_BATCH", 32);
    let vocabs: Vec<usize> = std::env::var("R2_VOCABS").ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 8000]);
    // How much text to LEARN merges from. Merge quality saturates long
    // before the data does, so this is capped independently of the corpus.
    let learn_cap: usize = env("R2_LEARN_BYTES", 4_000_000usize);

    let t0 = std::time::Instant::now();
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut end = raw.len();
    while end > 0 && std::str::from_utf8(&raw[..end]).is_err() { end -= 1; }
    let corpus = String::from_utf8_lossy(&raw[..end]).into_owned();
    let read_s = t0.elapsed().as_secs_f64();

    println!("Pipeline phases — same corpus, same step count");
    println!("  corpus   {path}  ({:.1} MB)", corpus.len() as f64 / 1e6);
    println!("  schedule {steps} steps x {bn} x {seq} = {} tokens/arm\n",
             steps * bn * seq);

    println!("{:<12} {:>8} {:>8} {:>9} {:>9} {:>9} {:>10} {:>12} {:>11}",
             "arm", "read", "learn", "tokenize", "train", "TOTAL",
             "tok/byte", "tokens/s", "text B/s");
    println!("{}", "-".repeat(96));

    for &v in &vocabs {
        // ── learn ──────────────────────────────────────────────────────
        let t0 = std::time::Instant::now();
        let tok = if v <= 256 {
            Tokenizer::byte_level()
        } else {
            let mut cut = corpus.len().min(learn_cap);
            while cut > 0 && !corpus.is_char_boundary(cut) { cut -= 1; }
            Tokenizer::from_trained(&bpe::train(&corpus[..cut], v))
        };
        let learn_s = t0.elapsed().as_secs_f64();

        // ── tokenize ───────────────────────────────────────────────────
        // The WHOLE corpus, as a real run does once before training.
        let t0 = std::time::Instant::now();
        let ids: Vec<usize> = tok.encode(&corpus)
            .expect("encode").iter().map(|&i| i as usize).collect();
        let tokenize_s = t0.elapsed().as_secs_f64();

        let need = steps * bn * (seq + 1);
        if ids.len() < need {
            println!("{:<12} corpus yields only {} tokens, needs {need}",
                     format!("vocab {v}"), ids.len());
            continue;
        }

        // ── train ──────────────────────────────────────────────────────
        let cfg = Config {
            dim: 256, n_heads: 4, n_kv_heads: 2, n_layers: 4,
            vocab: tok.vocab_size(), ffn_hidden: 768, max_seq: seq.max(64),
            rope_base: 10000.0, eps: 1e-5,
        };
        let mut tr = Trainer::new(cfg, 3e-4, 1).expect("trainer");
        let mut cursor = 0usize;
        let mut loss = 0.0f32;
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
        let train_s = t0.elapsed().as_secs_f64();

        let total = read_s + learn_s + tokenize_s + train_s;
        let tpb = ids.len() as f64 / corpus.len() as f64;
        let toks = (steps * bn * seq) as f64;
        // Text covered by the run, in bytes, over the WHOLE pipeline time.
        let text_bytes = toks / tpb;
        println!("{:<12} {:>8.2} {:>8.2} {:>9.2} {:>9.2} {:>9.2} {:>10.3} {:>12.0} {:>11.0}",
                 format!("vocab {}", tok.vocab_size()),
                 read_s, learn_s, tokenize_s, train_s, total,
                 tpb, toks / train_s, text_bytes / total);
        println!("{:<12} {:>8} {:>7.1}% {:>8.1}% {:>8.1}% {:>9} {:>10} {:>12} {:>11}",
                 "  share", "", learn_s / total * 100.0,
                 tokenize_s / total * 100.0, train_s / total * 100.0,
                 "", "", "", format!("loss {loss:.3}"));
    }

    println!("\ntext bytes/s over the WHOLE pipeline is the honest figure:");
    println!("tokens/s cannot be compared across vocabularies, and a run that");
    println!("spends its time learning merges has paid for them either way.");
}
