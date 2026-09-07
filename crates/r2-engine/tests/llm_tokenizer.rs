//! `llm.*` must use the tokenizer its handle was created with.
//!
//! Before this wiring, `llm.train` and `llm.generate` each constructed their
//! own `Tokenizer::byte_level()` — 256 ids, no merges — regardless of the
//! model's configured vocabulary. That was only safe because both happened
//! to pick the SAME tokenizer. The moment a model has a learned vocabulary,
//! a `generate` that rebuilt a byte-level table would decode the model's ids
//! against the wrong vocabulary and emit plausible nonsense, with nothing
//! erroring. These tests pin the pairing.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> RVal {
    let mut e = Engine::new();
    let exprs = Parser::parse(script).expect("parse ok");
    let mut last = RVal::Null;
    for ex in exprs {
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
        RVal::Character(c, _) => c.iter().next().cloned().flatten()
            .expect("a string").to_string(),
        other => panic!("expected text, got {:?}", other),
    }
}

/// Repetitive enough for BPE to find merges, long enough to fill a batch.
const CORPUS: &str = "txt <- paste(rep(\"the cat sat on the mat and the dog ran to the mat. \", 60), collapse=\"\")\n";

/// Default is byte-level: vocab 256, one token per byte. Unchanged, so
/// existing scripts keep working.
#[test]
fn default_is_byte_level_and_still_trains() {
    let v = run(&format!("{CORPUS}\
        m <- llm.new(dim=32, layers=1, ctx=16)\n\
        llm.train(m, txt, steps=3, seq=16, batch=4)\n"));
    let loss = scalar(&v);
    assert!(loss.is_finite() && loss > 0.0, "byte-level training gave loss {loss}");
    assert!(loss < 8.0, "loss {loss} is not plausible for vocab 256");

    let v = run(&format!("{CORPUS}\
        m <- llm.new(dim=32, layers=1, ctx=16)\n\
        llm.info(m)$vocab\n"));
    assert_eq!(scalar(&v), 256.0, "the byte-level default changed");
}

/// `vocab=` must reach the model configuration.
///
/// An earlier version of this test inferred the vocabulary from the LOSS —
/// "a vocab-600 model cannot get below ln(256) in three steps". That
/// reasoning is wrong on a corpus of ten distinct words, where the model
/// learns the marginal distribution almost immediately, and the test failed
/// on correct code. Ask the model what its vocabulary is instead.
#[test]
fn vocab_argument_reaches_the_config() {
    let v = run(&format!("{CORPUS}\
        m <- llm.new(dim=32, layers=1, ctx=16, vocab=600)\n\
        llm.info(m)$vocab\n"));
    assert_eq!(scalar(&v), 600.0, "llm.new ignored vocab=");
}

/// After training, the model's vocabulary must equal the tokenizer's.
///
/// A corpus only contains so many useful merges — this one has about ten
/// distinct words, so a request for 600 cannot be met. The model is resized
/// to what was actually learned. If it were not, the output layer would be
/// wider than the tokenizer has tokens: wasted `dim * unused` work on every
/// token, and ids the tokenizer cannot decode, which drops characters with
/// no error.
#[test]
fn model_is_resized_to_the_learned_vocabulary() {
    let v = run(&format!("{CORPUS}\
        m <- llm.new(dim=32, layers=1, ctx=16, vocab=600)\n\
        invisible(llm.train(m, txt, steps=1, seq=16, batch=4))\n\
        llm.info(m)$vocab\n"));
    let vocab = scalar(&v);
    assert!(vocab >= 256.0, "vocabulary {vocab} lost the byte tokens");
    assert!(vocab <= 600.0, "vocabulary {vocab} exceeds what was requested");
    assert!(vocab < 600.0,
            "a ten-word corpus cannot yield 600 merges — the model was not \
             resized to the learned vocabulary");
}

/// `llm.generate` must work on a BPE handle at all — it has to look the
/// tokenizer up by handle, and a model whose ids exceed 255 would have been
/// truncated or dropped by the byte-level table this used to rebuild.
///
/// WHAT THIS DOES NOT PROVE. An earlier version asserted the output
/// contained no U+FFFD, on the theory that a BPE model emits readable
/// word-pieces where a byte model emits arbitrary bytes. That is wrong:
/// BPE's *base* alphabet IS the 256 bytes, so an undertrained model emits
/// random single-byte tokens and therefore invalid UTF-8 under EITHER
/// tokenizer. The assertion failed on correct code. There is no cheap
/// observable at this level that separates the two decoders, so the pairing
/// rests on `tokenizer_for(id)` being the single lookup both call sites use,
/// plus the vocabulary tests above.
#[test]
fn bpe_generation_runs_on_its_own_handle() {
    let out = text(&run(&format!("{CORPUS}\
        m <- llm.new(dim=32, layers=1, ctx=16, vocab=600)\n\
        invisible(llm.train(m, txt, steps=2, seq=16, batch=4))\n\
        llm.generate(m, \"the cat\", n=8)\n")));
    assert!(!out.is_empty(), "generation returned nothing for a BPE model");
}

/// Two handles must not share a tokenizer. A single global one would make
/// the second model train on the first model's merges, and both would still
/// appear to work.
#[test]
fn two_models_keep_separate_tokenizers() {
    let v = run(&format!("{CORPUS}\
        a <- llm.new(dim=32, layers=1, ctx=16)\n\
        b <- llm.new(dim=32, layers=1, ctx=16, vocab=600)\n\
        invisible(llm.train(a, txt, steps=1, seq=16, batch=4))\n\
        invisible(llm.train(b, txt, steps=1, seq=16, batch=4))\n\
        llm.info(a)$vocab\n"));
    assert_eq!(scalar(&v), 256.0,
               "the byte-level handle's vocabulary changed after another \
                model trained — the handles share state");
}
