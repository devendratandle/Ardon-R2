//! Train a TinyStories model end to end in R2, and lay down everything the
//! PyTorch half needs to train the SAME model, on the SAME tokens, from the
//! SAME initial weights.
//!
//! Pairs with `benchmarks/llm/tinystories_train.py`, which reads this run's
//! manifest, trains its own copy, and prints ONE joined table. Neither half
//! reports a comparison on its own — `benchmarks/llm/REPORT.md` records that
//! pairing two halves by hand across two windows has already produced a
//! wrong claim once.
//!
//! # What makes this a comparison rather than two runs
//!
//! Three things are held identical, and each one removes a way the result
//! could be an artefact:
//!
//!   * **The token stream.** R2 learns the BPE merges, writes the vocabulary
//!     as `tokenizer.json`, and dumps the encoded ids; PyTorch reads those
//!     ids. Two tokenizers would give two vocabularies, and cross-entropy is
//!     per token — guessing 1 of 8,000 is a different question from guessing
//!     1 of 8,143, so the lower loss can belong to the worse model.
//!   * **The initial weights.** Every parameter block is dumped and loaded on
//!     the other side. Different init RNGs put ~0.11 between two starting
//!     losses in an earlier measurement — as large as the final gap that
//!     measurement was arguing about.
//!   * **The batch order.** Windows are a deterministic function of the step
//!     index, so both sides see the same sequences in the same order.
//!
//! What is left free is the only thing under test: the arithmetic, and how
//! fast each library does it.
//!
//! # What is reported as accuracy
//!
//! Held-out loss, on the tail of the corpus neither side trained on. The
//! training loss is what the model had *before* each update, on the very
//! batch it was about to fit; it measures memorisation of the stream.
//!
//!     cargo run --release -p r2-train --example tinystories_train
//!     python benchmarks/llm/tinystories_train.py

use r2_tensor::bpe;
use r2_tensor::infer::{Sampler, SamplerConfig};
use r2_tensor::model::{Config, Model};
use r2_tensor::tokenizer::Tokenizer;
use r2_train::llm::Trainer;
use r2_train::tokens::{TokenStore, TokenWriter};

/// Everything this run produces, so a repository listing stays readable and
/// one directory removal undoes the run.
const OUT: &str = "ts_run";

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn die(ctx: &str, e: impl std::fmt::Display) -> ! {
    eprintln!("{ctx}: {e}");
    std::process::exit(1)
}

fn write_f32(path: &str, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(path, b).unwrap_or_else(|e| die(&format!("writing {path}"), e));
}

/// The batch for step `s`: `bn` consecutive, non-overlapping windows of
/// `seq` tokens, starting where step `s-1` stopped.
///
/// A function of the step index alone, so the PyTorch half reproduces it in
/// the same two lines rather than being handed the batches. It wraps modulo
/// the stream, so a step count larger than the corpus is a second epoch
/// rather than a panic.
///
/// Reads through the memory map, so the resident cost is O(batch x seq)
/// rather than O(corpus) — the whole point of the token store.
fn batch_at(ids: &TokenStore, s: usize, bn: usize, seq: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let span = ids.len() - seq - 1;
    (0..bn).map(|b| {
        let o = ((s * bn + b) * seq) % span;
        let w = ids.window(o, seq + 1).expect("window inside the stream");
        (w[..seq].to_vec(), w[1..].to_vec())
    }).collect()
}

/// Read bytes `[from, to)` of a file as text.
///
/// Used only for the capped BPE-training prefix. Everything else streams.
fn read_span(path: &str, from: u64, to: u64) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot read {path}"), e));
    f.seek(SeekFrom::Start(from)).unwrap_or_else(|e| die("seek", e));
    let mut buf = vec![0u8; (to - from) as usize];
    let n = f.read(&mut buf).unwrap_or_else(|e| die("read", e));
    buf.truncate(n);
    // Trim any partial UTF-8 sequence the span cut through.
    while !buf.is_empty() && std::str::from_utf8(&buf).is_err() { buf.pop(); }
    String::from_utf8_lossy(&buf).into_owned()
}

/// The start of the line containing (or ending at) `target`.
///
/// Reads a small window and scans back, rather than the whole corpus: the
/// answer is the same cut a full-file `rfind('\n')` gives, at a cost that
/// does not grow with the file.
fn line_start_at_or_before(path: &str, target: u64) -> u64 {
    const W: u64 = 1 << 16;
    let from = target.saturating_sub(W);
    let span = read_span(path, from, target);
    match span.rfind('\n') {
        Some(i) => from + i as u64 + 1,
        // No newline in the whole window — treat the target as the cut
        // rather than search backwards forever.
        None => target,
    }
}

/// Encode as much of `chunk` as ends on a pre-token boundary, write it, and
/// return the bytes left over for the next chunk.
///
/// The cut point is the whole reason this is correct. GPT-2 splits text
/// into pre-tokens BEFORE merging and **a merge never crosses a pre-token
/// boundary** (`bpe`'s module docs), so a chunk that ends on one encodes
/// exactly as the same text would inside the whole stream. A chunk that
/// ends anywhere else does not — splitting on newlines instead was measured
/// wrong: over 3 MB at 64 KB chunks the ids diverged, `[1293, 400]` whole
/// against `[10, 470]` chunked for identical text, because a newline plus
/// the following space is ONE pre-token and the split cut through it. At 1 MB the same test passed,
/// which is luck about where the boundaries land, not correctness.
fn flush_chunk(chunk: &[u8], tok: &Tokenizer, w: &mut TokenWriter, last: bool) -> Vec<u8> {
    if chunk.is_empty() { return Vec::new(); }
    // A chunk can also end mid-character; those bytes belong to the next.
    let mut end = chunk.len();
    while end > 0 && std::str::from_utf8(&chunk[..end]).is_err() { end -= 1; }
    let text = std::str::from_utf8(&chunk[..end]).unwrap_or("");
    let cut = if last {
        text.len()
    } else {
        bpe::pretokenize(text).last().map_or(text.len(), |p| text.len() - p.len())
    };
    if cut > 0 {
        let ids = tok.encode(&text[..cut]).unwrap_or_else(|e| die("encoding", e));
        let ids: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
        w.write(&ids).unwrap_or_else(|e| die("writing tokens", e));
    }
    chunk[cut..].to_vec()
}

/// Tokenize `[from, to)` of `path` straight into a token file.
///
/// Neither the text nor the id stream is ever fully resident: the peak is
/// one chunk. Identical ids to encoding the range in one piece — see
/// [`flush_chunk`] for why that is not automatic.
fn tokenize_range(path: &str, from: u64, to: u64, tok: &Tokenizer, out: &str) -> usize {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    const CHUNK: usize = 1 << 20;
    let f = std::fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot read {path}"), e));
    let mut r = BufReader::with_capacity(CHUNK, f);
    r.seek(SeekFrom::Start(from)).unwrap_or_else(|e| die("seek", e));
    let mut w = TokenWriter::create(out).unwrap_or_else(|e| die("creating token file", e));

    let (mut pos, mut line) = (from, Vec::new());
    let mut chunk: Vec<u8> = Vec::with_capacity(CHUNK * 2);
    while pos < to {
        line.clear();
        let n = r.read_until(b'\n', &mut line).unwrap_or_else(|e| die("read", e));
        if n == 0 { break; }
        let take = ((to - pos) as usize).min(n);
        chunk.extend_from_slice(&line[..take]);
        pos += n as u64;
        if chunk.len() >= CHUNK { chunk = flush_chunk(&chunk, tok, &mut w, false); }
    }
    let rest = flush_chunk(&chunk, tok, &mut w, true);
    debug_assert!(rest.is_empty(), "the final flush must consume everything");
    w.finish().unwrap_or_else(|e| die("finishing token file", e))
}

fn main() {
    let path: String = std::env::var("R2_CORPUS").unwrap_or_else(|_| "corpus.txt".into());
    let steps: usize = env("R2_STEPS", 300);
    let seq: usize = env("R2_SEQ", 64);
    let bn: usize = env("R2_BATCH", 32);
    let vocab_target: usize = env("R2_VOCAB", 8000);
    let lr: f32 = env("R2_LR", 3e-4f32);
    let seed: u64 = env("R2_SEED", 42);
    // Merge quality saturates long before a 19 MB corpus does, so what BPE
    // learns from is capped independently of what training reads.
    let learn_cap: usize = env("R2_LEARN_BYTES", 4_000_000usize);
    // The held-out tail. 2% of 19.4 MB is ~390 KB, ~94k tokens — more than
    // the ~16k a fixed eval set reads, so the rest is headroom.
    let val_frac: f64 = env("R2_VAL_FRAC", 0.02f64);
    let val_batches: usize = env("R2_VAL_BATCHES", 8);

    std::fs::create_dir_all(OUT).unwrap_or_else(|e| die("creating ts_run", e));

    // ── corpus: streamed, never held whole ──────────────────────────────
    //
    // The corpus is read in line-aligned chunks and tokenized straight to a
    // file. Nothing here materialises either the text or the id stream, so
    // the peak is a ~1 MB chunk rather than the corpus — which is the
    // difference between "a 700 MB corpus needs ~2.7 GB to begin" and "it
    // needs a window". `r2_train::tokens` documents that arithmetic.
    let flen = std::fs::metadata(&path)
        .unwrap_or_else(|e| die(&format!("cannot stat {path}"), e)).len();

    // Split at a LINE boundary at or before the target, so a story is never
    // half in training and half in validation. Found by reading a small
    // window around the target rather than the whole file — the same cut a
    // full-corpus `rfind` would produce.
    let want = (flen as f64 * (1.0 - val_frac)) as u64;
    let cut = line_start_at_or_before(&path, want);

    println!("Ardon-R2 — TinyStories training run");
    println!("  corpus     {path}  ({:.1} MB, streamed — never held whole)",
             flen as f64 / 1e6);
    println!("  split      train {:.1} MB / held-out {:.1} MB (cut at a line boundary)",
             cut as f64 / 1e6, (flen - cut) as f64 / 1e6);

    // ── tokenizer and token files: built once, then reused ──────────────
    //
    // The stamp records the INPUTS a token file was built from — corpus
    // path, its size, the split point, and the two tokenizer settings — so
    // a later run over the same corpus skips both the BPE training and the
    // tokenization pass and maps the existing file. It deliberately names
    // inputs rather than the trained vocabulary: the point is to decide
    // whether to do the work *before* doing it, and reusing a token stream
    // against a different vocabulary would train the model on ids nothing
    // can decode.
    let learn_bytes = learn_cap.min(cut as usize);
    let tok_path = format!("{OUT}/tokenizer.json");
    let (train_path, val_path) = (format!("{OUT}/train_ids.bin"), format!("{OUT}/val_ids.bin"));
    let stamp_path = format!("{OUT}/tokens.stamp");
    let stamp = format!("{path}|{flen}|{cut}|{vocab_target}|{learn_bytes}");
    let reusable = std::fs::read_to_string(&stamp_path).map(|s| s == stamp).unwrap_or(false)
        && [&tok_path, &train_path, &val_path].iter().all(|p| std::path::Path::new(p).exists());

    let (tok, learn_s, tokenize_s);
    if reusable {
        let src = std::fs::read_to_string(&tok_path)
            .unwrap_or_else(|e| die("reading tokenizer.json", e));
        tok = Tokenizer::from_tokenizer_json(&src)
            .unwrap_or_else(|e| die("parsing tokenizer.json", e));
        learn_s = 0.0;
        tokenize_s = 0.0;
        println!("  reused     tokenizer.json + token files — same corpus, same settings");
    } else {
        // Merge quality saturates long before a 19 MB corpus does, so only
        // the capped prefix is read — the one place text is held in memory,
        // and it is bounded by `R2_LEARN_BYTES`, not by the corpus.
        let t0 = std::time::Instant::now();
        let learn_on = read_span(&path, 0, learn_bytes as u64);
        tok = Tokenizer::from_trained(&bpe::train(&learn_on, vocab_target));
        learn_s = t0.elapsed().as_secs_f64();
        drop(learn_on);
        std::fs::write(&tok_path, tok.to_tokenizer_json())
            .unwrap_or_else(|e| die("writing tokenizer.json", e));

        let t0 = std::time::Instant::now();
        tokenize_range(&path, 0, cut, &tok, &train_path);
        tokenize_range(&path, cut, flen, &tok, &val_path);
        tokenize_s = t0.elapsed().as_secs_f64();
        std::fs::write(&stamp_path, &stamp).unwrap_or_else(|e| die("writing stamp", e));
    }

    // ── the token stores: mapped, not loaded ────────────────────────────
    let train = TokenStore::open(&train_path).unwrap_or_else(|e| die("opening train tokens", e));
    let val_store = TokenStore::open(&val_path).unwrap_or_else(|e| die("opening held-out tokens", e));
    let (n_train, n_val) = (train.len(), val_store.len());
    if reusable {
        println!("  tokenizer  vocab {} (loaded)", tok.vocab_size());
    } else {
        println!("  tokenizer  vocab {} learned from {:.1} MB in {learn_s:.2} s",
                 tok.vocab_size(), learn_bytes as f64 / 1e6);
        println!("  tokenize   {n_train} train + {n_val} held-out ids in {tokenize_s:.2} s  \
                  ({:.3} tokens/byte)", (n_train + n_val) as f64 / flen as f64);
    }
    println!("  memory     token stream mapped: {} B resident vs {:.1} MB as a Vec<usize> \
              ({:.0}x)",
             train.resident_bytes() + val_store.resident_bytes(),
             (train.in_ram_equivalent_bytes() + val_store.in_ram_equivalent_bytes()) as f64 / 1e6,
             (train.in_ram_equivalent_bytes() + val_store.in_ram_equivalent_bytes()) as f64
                 / (train.resident_bytes() + val_store.resident_bytes()) as f64);

    // ── model ───────────────────────────────────────────────────────────
    let cfg = Config {
        dim: env("R2_DIM", 256), n_heads: 4, n_kv_heads: 2,
        n_layers: env("R2_LAYERS", 4), vocab: tok.vocab_size(),
        ffn_hidden: env("R2_FFN", 768), max_seq: seq.max(64),
        rope_base: 10000.0, eps: 1e-5,
    };
    let mut tr = Trainer::new(cfg, lr, seed).unwrap_or_else(|e| die("trainer", e));
    assert_eq!(tr.n_params(), cfg.n_params(), "trainer and config disagree on size");
    println!("  model      {:.2}M parameters, dim {} x {} layers, ffn {}, {} q / {} kv heads",
             cfg.n_params() as f64 / 1e6, cfg.dim, cfg.n_layers, cfg.ffn_hidden,
             cfg.n_heads, cfg.n_kv_heads);
    println!("  schedule   {steps} steps x {bn} x {seq} = {} tokens, Adam lr {lr}\n",
             steps * bn * seq);

    // The initial weights, so the other side starts from this exact point
    // rather than from its own RNG.
    let names = tr.block_names();
    for (i, p) in tr.params.iter().enumerate() {
        write_f32(&format!("{OUT}/init_{i}.bin"), p);
    }

    // ── held-out set: fixed, and never trained on ───────────────────────
    let val: Vec<(Vec<usize>, Vec<usize>)> =
        (0..val_batches).flat_map(|s| batch_at(&val_store, s, bn, seq)).collect();
    let val_before = tr.eval_loss(&val).unwrap_or_else(|e| die("eval", e));

    // ── train ───────────────────────────────────────────────────────────
    let mut curve: Vec<(usize, f32)> = Vec::new();
    let every = (steps / 10).max(1);
    let t0 = std::time::Instant::now();
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for s in 0..steps {
        let batch = batch_at(&train, s, bn, seq);
        let loss = tr.train_step(&batch).unwrap_or_else(|e| die("train_step", e));
        if s == 0 { first = loss; }
        last = loss;
        if s == 0 || (s + 1) % every == 0 {
            curve.push((s + 1, loss));
            println!("  step {:>4}/{steps}  loss {loss:.4}   {:.1} s elapsed",
                     s + 1, t0.elapsed().as_secs_f64());
        }
    }
    let train_s = t0.elapsed().as_secs_f64();
    let val_after = tr.eval_loss(&val).unwrap_or_else(|e| die("eval", e));

    // Bits per byte: the loss is per TOKEN, and a token is not a fixed
    // amount of text. Both sides share this token stream so their losses are
    // already comparable; bits/byte is the figure that stays meaningful
    // against a model tokenized any other way.
    let val_tok_per_byte = n_val as f64 / (flen - cut).max(1) as f64;
    let bpb = val_after as f64 / std::f64::consts::LN_2 * val_tok_per_byte;

    println!("\n  trained    {steps} steps in {train_s:.2} s  ({:.1} tokens/s, {:.1} ms/step)",
             (steps * bn * seq) as f64 / train_s, train_s * 1000.0 / steps as f64);
    println!("  train loss {first:.4} -> {last:.4}");
    println!("  HELD-OUT   {val_before:.4} -> {val_after:.4}   ({bpb:.4} bits/byte)");

    // ── save, reload, generate ──────────────────────────────────────────
    let model = tr.to_model().unwrap_or_else(|e| die("export", e));
    let dir = format!("{OUT}/model");
    model.save_dir(std::path::Path::new(&dir)).unwrap_or_else(|e| die("save", e));
    let size: u64 = std::fs::read_dir(&dir).map(|d| d.flatten()
        .filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
    let served = Model::load_dir(std::path::Path::new(&dir)).unwrap_or_else(|e| die("load", e));

    let prompt: String = std::env::var("R2_PROMPT").unwrap_or_else(|_| "Once upon a time".into());
    let gen_n: usize = env("R2_GEN", 48);
    let p: Vec<usize> = tok.encode(&prompt).unwrap_or_else(|e| die("encode prompt", e))
        .iter().map(|&i| i as usize).collect();
    // Greedy, so the continuation is a deterministic function of the weights
    // and the two sides' samples can be set side by side.
    let mut samp = Sampler::new(1, SamplerConfig { temperature: 0.0, ..Default::default() });
    let out = served.generate(&p, gen_n, &mut samp, None).unwrap_or_else(|e| die("generate", e));
    let text = tok.decode(&out.iter().map(|&i| i as u32).collect::<Vec<_>>());
    println!("\n  saved      {dir} ({:.1} MB)", size as f64 / 1e6);
    println!("  prompt     {prompt:?}");
    println!("  continues  {text:?}");

    // ── manifest ────────────────────────────────────────────────────────
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"")
                         .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
    let mut j = String::from("{\n");
    j.push_str(&format!("  \"dim\": {}, \"n_heads\": {}, \"n_kv_heads\": {},\n",
                        cfg.dim, cfg.n_heads, cfg.n_kv_heads));
    j.push_str(&format!("  \"n_layers\": {}, \"vocab\": {}, \"ffn_hidden\": {},\n",
                        cfg.n_layers, cfg.vocab, cfg.ffn_hidden));
    j.push_str(&format!("  \"eps\": {}, \"rope_base\": {},\n", cfg.eps, cfg.rope_base));
    j.push_str(&format!("  \"n_params\": {},\n", cfg.n_params()));
    j.push_str(&format!("  \"steps\": {steps}, \"batch\": {bn}, \"seq\": {seq},\n"));
    j.push_str(&format!("  \"lr\": {lr}, \"seed\": {seed},\n"));
    j.push_str(&format!("  \"val_batches\": {val_batches},\n"));
    j.push_str(&format!("  \"corpus_bytes\": {}, \"train_bytes\": {}, \"val_bytes\": {},\n",
                        flen, cut, flen - cut));
    j.push_str(&format!("  \"train_tokens\": {}, \"val_tokens\": {},\n",
                        n_train, n_val));
    j.push_str(&format!("  \"learn_s\": {learn_s:.4}, \"tokenize_s\": {tokenize_s:.4},\n"));
    // Stated explicitly rather than left to be inferred from a zero
    // timing: a reused run did not tokenize, so its pipeline total is not
    // comparable against a side that did, and the reader of this manifest
    // has to be able to tell those apart.
    j.push_str(&format!("  \"tokens_reused\": {reusable},\n"));
    j.push_str(&format!("  \"train_s\": {train_s:.4},\n"));
    j.push_str(&format!("  \"loss_first\": {first:.6}, \"loss_last\": {last:.6},\n"));
    j.push_str(&format!("  \"val_before\": {val_before:.6}, \"val_after\": {val_after:.6},\n"));
    j.push_str(&format!("  \"bits_per_byte\": {bpb:.6},\n"));
    j.push_str(&format!("  \"gen\": {gen_n},\n"));
    // The prompt as R2 tokenized it. The other side re-encodes the same
    // string with the exported `tokenizer.json` and checks it lands here —
    // which is what turns "we shared a tokenizer file" from a claim into a
    // verified fact, and would catch an export that HuggingFace parses but
    // interprets differently.
    j.push_str("  \"prompt_ids\": [");
    for (k, id) in p.iter().enumerate() {
        if k > 0 { j.push_str(", "); }
        j.push_str(&id.to_string());
    }
    j.push_str("],\n");
    j.push_str(&format!("  \"prompt\": \"{}\", \"sample\": \"{}\",\n",
                        esc(&prompt), esc(&text)));
    j.push_str("  \"curve\": [");
    for (k, (s, l)) in curve.iter().enumerate() {
        if k > 0 { j.push_str(", "); }
        j.push_str(&format!("[{s}, {l:.6}]"));
    }
    j.push_str("],\n  \"blocks\": [\n");
    for (i, p) in tr.params.iter().enumerate() {
        j.push_str(&format!("    {{\"i\": {i}, \"name\": \"{}\", \"len\": {}}}{}\n",
                            names[i], p.len(),
                            if i + 1 == tr.params.len() { "" } else { "," }));
    }
    j.push_str("  ]\n}\n");
    let man = format!("{OUT}/r2.json");
    std::fs::write(&man, j).unwrap_or_else(|e| die("writing manifest", e));
    println!("\nwrote {man} — now run:  python benchmarks/llm/tinystories_train.py");
}
