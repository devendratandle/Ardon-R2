//! Load GPT-2's real `tokenizer.json` and dump R2's token ids, so they can
//! be diffed against HuggingFace's on the SAME file and the SAME input.
//!
//! This is the check that settles whether R2 "matches the GPT-2 standard".
//! Everything else — the byte alphabet, the merge ranks, the pre-tokenizer
//! regex — is an argument from the published algorithm. Only feeding both
//! implementations one vocabulary and comparing ids is evidence.
//!
//!     R2_GPT2_JSON=gpt2_tokenizer.json \
//!       cargo run --release -p r2-tensor --example gpt2_compare
//!
//! Prints one line per case: `CASE <n> <ids...>`, plus a throughput figure
//! on a larger body of text. `gpt2_compare.py` reads it and diffs.

use r2_tensor::tokenizer::Tokenizer;

/// The cases that separate a correct ByteLevel BPE from a plausible one.
/// Every entry exercises a specific clause of GPT-2's pre-tokenizer regex.
const CASES: &[&str] = &[
    "hello world",
    " hello",
    "hello",
    "don't stop",
    "It's 42 apples, isn't it?",
    "a   b",                       // \s+(?!\S): run of spaces
    "  leading",                   // leading whitespace run
    "trailing  ",                  // trailing whitespace run
    "x=1;y=2",                     // punctuation runs
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "CamelCaseIdentifier",
    "snake_case_name",
    "tabs\tand\nnewlines",
    "日本語のテキスト",              // CJK, multi-byte
    "emoji 🙂 and 🎉 mixed",        // 4-byte UTF-8
    "café naïve résumé",           // Latin-1 range
    "  ",                          // whitespace only
    "a",
    "",
];

fn main() {
    let path = std::env::var("R2_GPT2_JSON")
        .unwrap_or_else(|_| "gpt2_tokenizer.json".into());
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let tok = Tokenizer::from_tokenizer_json(&src).unwrap_or_else(|e| {
        eprintln!("cannot parse {path}: {e}");
        std::process::exit(1);
    });
    eprintln!("loaded {} tokens, encoding {:?}", tok.vocab_size(), tok.encoding());

    for (i, case) in CASES.iter().enumerate() {
        match tok.encode(case) {
            Ok(ids) => {
                let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                println!("CASE {i} {}", s.join(","));
            }
            Err(e) => println!("CASE {i} ERROR {e}"),
        }
    }

    // SCALING. A prior analysis of this code measured encode as O(n^2)
    // (3,808 bytes in 0.94 s, extrapolating to ~18 hours per MB) because BPE
    // ran over the whole document instead of per pre-token. If that is
    // fixed, time must roughly DOUBLE as the input doubles, not quadruple.
    println!("SCALING");
    let unit = CASES.join(" ");
    for k in [1usize, 2, 4, 8, 16, 32, 64] {
        let body = unit.repeat(k * 8);
        let t0 = std::time::Instant::now();
        let ids = tok.encode(&body).expect("encode");
        let secs = t0.elapsed().as_secs_f64();
        println!("  bytes={:>8} secs={:.5} mbps={:>6.2} tokens={}",
                 body.len(), secs, body.len() as f64 / secs / 1e6, ids.len());
    }

    // PHASE SPLIT: how much of encode is the pre-tokenizer, and how much
    // is the merge loop? Optimising the wrong half is the standing risk.
    {
        let body = CASES.join(" ").repeat(2000);
        let reps = 5;
        let t0 = std::time::Instant::now();
        let mut words = 0usize;
        for _ in 0..reps { words = r2_tensor::bpe::pretokenize(&body).len(); }
        let pre = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { std::hint::black_box(tok.encode(&body).unwrap().len()); }
        let full = t0.elapsed().as_secs_f64() / reps as f64;
        println!("PHASES bytes={} words={words} pretokenize={:.4}s ({:.1}%) rest={:.4}s ({:.1}%) total={:.4}s",
                 body.len(), pre, pre / full * 100.0,
                 full - pre, (full - pre) / full * 100.0, full);
    }

    // Throughput on a larger body, and the round-trip that makes it usable.
    let body = CASES.join(" ").repeat(2000);
    let t0 = std::time::Instant::now();
    let ids = tok.encode(&body).expect("encode");
    let secs = t0.elapsed().as_secs_f64();
    println!("THROUGHPUT bytes={} tokens={} secs={:.4} mbps={:.2}",
             body.len(), ids.len(), secs, body.len() as f64 / secs / 1e6);
    println!("ROUNDTRIP {}", if tok.decode(&ids) == body { "EXACT" } else { "MISMATCH" });
}
