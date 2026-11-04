//! `llm.train` must learn from the whole text, and `llm.save` must keep what
//! `llm.generate` needs.
//!
//! Until 2026-11-04 `llm.train` built one batch from the first `batch + seq`
//! tokens and trained on it for every step: changing anything after those
//! tokens left the result bit-identical. And `llm.save` dropped the tokenizer,
//! so a reloaded model with a learned vocabulary decoded its ids against the
//! byte table and generated nonsense.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> RVal {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

fn scalar(v: &RVal) -> f64 {
    match v {
        RVal::Numeric(d, _) => d.iter().next().and_then(|x| *x).expect("scalar"),
        other => panic!("expected a number, got {:?}", other),
    }
}

fn text(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.iter().next().cloned().flatten().expect("a string").to_string(),
        other => panic!("expected text, got {:?}", other),
    }
}

/// Final loss of a small byte-level model on `txt`, `shuffle` as given.
fn loss_on(txt: &str, extra: &str) -> f64 {
    scalar(&run(&format!(
        "m <- llm.new(vocab = 256, dim = 16, layers = 1, heads = 2, ffn = 32, ctx = 16, seed = 42)\n\
         llm.train(m, \"{txt}\", steps = 30, seq = 8, batch = 2{extra})\n")))
}

#[test]
fn the_whole_text_is_trained_on() {
    let base = "the cat sat on the mat and the dog ran over the hill to the old red barn. ".repeat(3);
    let late = base.len() - 5;
    let mut changed = base.clone();
    changed.replace_range(late..late + 1, "Q");
    // 30 steps x 2 x 8 is 480 tokens, past the end of this 222-token text
    assert_ne!(loss_on(&base, ""), loss_on(&changed, ""), "a byte near the end must matter");
}

#[test]
fn shuffling_changes_the_order_and_only_the_seed_chooses_it() {
    let t = "ardon r2 is a statistical runtime written in pure rust. ".repeat(6);
    let (seq, shuf) = (loss_on(&t, ""), loss_on(&t, ", shuffle = TRUE"));
    assert_ne!(seq, shuf);
    assert_eq!(shuf, loss_on(&t, ", shuffle = TRUE"), "same seed, same run");
}

#[test]
fn a_second_call_continues_through_the_text() {
    let t = "abcdefghijklmnopqrstuvwxyz0123456789 ".repeat(6);
    let cfg = "vocab = 256, dim = 16, layers = 1, heads = 2, ffn = 32, ctx = 16, seed = 42";
    let joined = scalar(&run(&format!(
        "m <- llm.new({cfg})\nllm.train(m, \"{t}\", steps = 10, seq = 8, batch = 2)\n")));
    let parts = scalar(&run(&format!(
        "m <- llm.new({cfg})\ninvisible(llm.train(m, \"{t}\", steps = 4, seq = 8, batch = 2))\n\
         llm.train(m, \"{t}\", steps = 6, seq = 8, batch = 2)\n")));
    assert_eq!(joined, parts, "4 + 6 steps must be the same run as 10 steps");
}

#[test]
fn a_file_path_trains_on_the_file() {
    let t = "the rain in spain falls mainly on the plain and the sun in spain. ".repeat(4);
    let p = std::env::temp_dir().join(format!("r2_llm_corpus_{}.txt", std::process::id()));
    std::fs::write(&p, &t).unwrap();
    let path = p.to_string_lossy().replace('\\', "/");
    assert_eq!(loss_on(&path, ""), loss_on(&t, ""));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn a_saved_model_generates_the_same_text_after_loading() {
    let dir = std::env::temp_dir().join(format!("r2_llm_save_{}", std::process::id()));
    let d = dir.to_string_lossy().replace('\\', "/");
    let script = format!(
        "txt <- paste(rep(\"the cat sat on the mat and the dog ran over the hill. \", 40), collapse = \"\")\n\
         m <- llm.new(dim = 24, layers = 1, heads = 2, ffn = 48, ctx = 16, vocab = 400, seed = 42)\n\
         invisible(llm.train(m, txt, steps = 20, seq = 12, batch = 4))\n\
         a <- llm.generate(m, \"the cat\", n = 8)\n\
         invisible(llm.save(m, \"{d}\"))\n\
         b <- llm.generate(llm.load(\"{d}\"), \"the cat\", n = 8)\n\
         paste0(a, \"|\", b)\n");
    let out = text(&run(&script));
    let (a, b) = out.split_once('|').unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, b, "the reloaded model must decode with its own tokenizer");
    // Without the tokenizer a learned vocabulary is refused, not decoded wrongly.
    std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let mut e = Engine::new();
    let exprs = Parser::parse(&format!("llm.load(\"{d}\")")).unwrap();
    let err = e.eval(&exprs[0]).unwrap_err();
    assert!(err.msg.contains("tokenizer"), "{}", err.msg);
    let _ = std::fs::remove_dir_all(&dir);
}
