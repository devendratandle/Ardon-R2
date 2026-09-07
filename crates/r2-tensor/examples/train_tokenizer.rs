//! Train a GPT-2-style byte-level BPE tokenizer on a corpus and report
//! what it actually buys.
//!
//! R2's `Tokenizer::byte_level()` is 256 byte tokens with no merges — one
//! token per byte. English is roughly 4 bytes per GPT-2 token, so a
//! byte-level model reads ~4x the tokens for the same text. Since attention
//! is O(seq²), that is not a 4x cost, it is worse.
//!
//! This trains merges from real text and prints bytes/token, so the gain is
//! measured on YOUR corpus rather than quoted from a paper. Writes a
//! HuggingFace `tokenizer.json` that `tokenizers`/`transformers` can load.
//!
//!     R2_CORPUS=path/to/text.txt R2_VOCAB=32000 \
//!       cargo run --release -p r2-tensor --example train_tokenizer
//!
//! With no corpus it trains on a small built-in sample, which demonstrates
//! the mechanism but says nothing about a real vocabulary.

use r2_tensor::bpe;
use r2_tensor::tokenizer::Tokenizer;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const SAMPLE: &str = "\
the quick brown fox jumps over the lazy dog. the dog barks and the fox runs.
a cat sat on a mat while the dog watched the cat and the cat watched the dog.
training a tokenizer needs text, and more text makes better merges.
the more the text repeats its words, the more the merges can compress it.
";

fn main() {
    let vocab_size: usize = env("R2_VOCAB", 8000);
    let path = std::env::var("R2_CORPUS").ok();
    // A tokenizer is trained on a SAMPLE of the corpus, not all of it —
    // merge quality saturates long before the data does, and GPT-2 itself
    // was trained on a subset. Capped so this stays minutes, not hours.
    let cap: usize = env("R2_CORPUS_BYTES", 50_000_000usize);

    let (corpus, label) = match &path {
        Some(p) => {
            let raw = std::fs::read(p).unwrap_or_else(|e| {
                eprintln!("cannot read {p}: {e}");
                std::process::exit(1);
            });
            let n = raw.len().min(cap);
            // Truncate on a char boundary — a split multi-byte character
            // would train merges on a byte sequence that never occurs.
            let mut end = n;
            while end > 0 && std::str::from_utf8(&raw[..end]).is_err() { end -= 1; }
            (String::from_utf8_lossy(&raw[..end]).into_owned(),
             format!("{p} (first {:.1} MB of {:.1} MB)",
                     end as f64 / 1e6, raw.len() as f64 / 1e6))
        }
        None => (SAMPLE.to_string(), "built-in sample".to_string()),
    };

    println!("Training byte-level BPE, GPT-2 style");
    println!("  corpus     {label}");
    println!("  vocab_size {vocab_size} (256 byte tokens + {} merges)",
             vocab_size.saturating_sub(256));

    let words = bpe::pretokenize(&corpus).len();
    println!("  pre-tokens {words}\n");

    let t0 = std::time::Instant::now();
    let trained = bpe::train(&corpus, vocab_size);
    let train_s = t0.elapsed().as_secs_f64();
    let tok = Tokenizer::from_trained(&trained);
    println!("trained {} merges in {train_s:.1}s", trained.merges.len());

    // ── what it buys, measured on the corpus itself ────────────────────
    // Walk back to a char boundary — slicing at a fixed byte offset panics
    // the moment the corpus contains a multi-byte character, which real
    // text always does.
    let mut cut = corpus.len().min(2_000_000);
    while cut > 0 && !corpus.is_char_boundary(cut) { cut -= 1; }
    let sample: &str = &corpus[..cut];
    let t0 = std::time::Instant::now();
    let ids = tok.encode(sample).expect("encode");
    let enc_s = t0.elapsed().as_secs_f64();

    let byte_tok = Tokenizer::byte_level();
    let byte_ids = byte_tok.encode(sample).expect("encode");

    let bpt = sample.len() as f64 / ids.len() as f64;
    let byte_bpt = sample.len() as f64 / byte_ids.len() as f64;
    println!("\n{:<28} {:>12} {:>12}", "", "byte-level", "trained BPE");
    println!("{:<28} {:>12} {:>12}", "tokens for the sample",
             byte_ids.len(), ids.len());
    println!("{:<28} {:>12.2} {:>12.2}", "bytes per token", byte_bpt, bpt);
    // How much SHORTER the sequence gets. Attention is O(seq^2), so the
    // saving in attention work is the square of this.
    println!("{:<28} {:>12} {:>12.2}x", "sequence is shorter by", "1.00", bpt / byte_bpt);
    println!("{:<28} {:>12} {:>12.2}x", "  => attention work O(seq^2)", "1.00",
             (bpt / byte_bpt).powi(2));
    println!("\nencode threw {:.1} MB/s", sample.len() as f64 / enc_s / 1e6);

    // Round-trip is the property that makes the tokenizer safe to ship.
    let back = tok.decode(&ids);
    assert_eq!(back, sample, "round-trip failed — tokenizer is not safe to use");
    println!("round-trip on {:.1} MB: EXACT", sample.len() as f64 / 1e6);

    let out = std::env::var("R2_TOKENIZER_OUT")
        .unwrap_or_else(|_| "tokenizer.json".to_string());
    std::fs::write(&out, tok.to_tokenizer_json()).expect("write tokenizer.json");
    println!("wrote {out} ({} tokens) — loadable by HuggingFace `tokenizers`",
             tok.vocab_size());
}
