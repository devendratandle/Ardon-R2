//! GPT-2 pre-tokenization and BPE training — the two halves of a real
//! tokenizer that `tokenizer.rs` could not do.
//!
//! `tokenizer.rs` implements BPE *inference* correctly: the byte-level
//! alphabet, rank-ordered merges, HuggingFace `tokenizer.json` loading. What
//! it could not do is LEARN a merge table, so `Tokenizer::byte_level()`
//! returns 256 byte tokens and an empty merge list — one token per byte.
//!
//! That is not a small constant factor. English text is roughly **4 bytes
//! per GPT-2 token**, so a byte-level model reads ~4x more tokens for the
//! same text: 4x the sequence length for the same context, and attention is
//! O(seq²) in that length. It is the difference between a toy and a
//! tokenizer.
//!
//! # Pre-tokenization is not optional
//!
//! GPT-2 splits text into "words" BEFORE running BPE, with this pattern
//! (`gpt2/encoder.py`, and every HuggingFace ByteLevel tokenizer since):
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! Merges may then never cross a word boundary. Skipping it changes the
//! segmentation the model sees — " the cat" could merge into a single token
//! that GPT-2 would never produce — and it is also what keeps encoding
//! fast: BPE is quadratic in the length of the span it runs over, so
//! bounding each run to one short word makes encoding linear in the text
//! instead of quadratic.
//!
//! Implemented directly rather than with a regex engine. r2-tensor has no
//! regex dependency, the pattern is fixed, and a hand-written scanner over
//! `char` classes is both smaller and faster than compiling it.

use std::collections::HashMap;

/// Is `c` a Unicode letter? `\p{L}` in GPT-2's pattern.
///
/// `char::is_alphabetic` is Unicode's `Alphabetic` property, which is
/// `\p{L}` plus `\p{Nl}` and a few marks. The difference does not arise in
/// the byte-level alphabet the trainer sees, and using the standard library
/// keeps this free of a Unicode table dependency.
fn is_letter(c: char) -> bool { c.is_alphabetic() }

/// Is `c` a Unicode number? `\p{N}`.
fn is_number(c: char) -> bool { c.is_numeric() }

/// GPT-2's contraction pieces, matched before anything else so that
/// "don't" becomes "don" + "'t" exactly as the reference does.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// Split `text` into GPT-2 pre-tokens.
///
/// Returns borrowed slices in order; concatenating them reproduces `text`
/// exactly, which `pretokenize_is_lossless` pins.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut i = 0usize;
    while i < text.len() {
        // Only ever index at a char boundary.
        debug_assert!(text.is_char_boundary(i));
        let rest = &text[i..];

        // 1. 's 't 're 've 'm 'll 'd
        if let Some(m) = CONTRACTIONS.iter().find(|c| rest.starts_with(**c)) {
            out.push(&text[i..i + m.len()]);
            i += m.len();
            continue;
        }

        // A single optional leading space belongs to the token that
        // follows it — that is why GPT-2 vocabularies are full of "Ġthe"
        // rather than "the". `\s+(?!\S)` below is what stops a RUN of
        // spaces from being swallowed: only the last space of a run may
        // attach to the next word.
        let start = i;
        let mut j = i;
        if b[j] == b' ' {
            // Take the space only if exactly one, and something non-space
            // follows it. Otherwise fall through to the whitespace arms.
            let after = j + 1;
            let next = text[after..].chars().next();
            match next {
                Some(c) if !c.is_whitespace() => j = after,
                _ => {
                    // A whitespace run. `\s+(?!\S)` keeps all but the last
                    // character when a non-space follows, so the last one
                    // can join the next word; otherwise the whole run.
                    let ws_end = text[j..].find(|c: char| !c.is_whitespace())
                        .map(|k| j + k)
                        .unwrap_or(text.len());
                    let has_following_word = ws_end < text.len();
                    let mut end = ws_end;
                    if has_following_word {
                        // Give the final whitespace char back to the next token.
                        let last = text[j..ws_end].chars().next_back()
                            .map(|c| c.len_utf8()).unwrap_or(0);
                        if ws_end - last > j { end = ws_end - last; }
                    }
                    if end > j {
                        out.push(&text[j..end]);
                        i = end;
                        continue;
                    }
                    // A single space with a word after it: fall through so
                    // it attaches to that word below.
                    j = ws_end.min(j + 1);
                }
            }
        }

        let c = match text[j..].chars().next() {
            Some(c) => c,
            // The optional space ran to end-of-string.
            None => { out.push(&text[start..]); break; }
        };

        if is_letter(c) {
            // ` ?\p{L}+`
            let end = text[j..].find(|c: char| !is_letter(c))
                .map(|k| j + k).unwrap_or(text.len());
            out.push(&text[start..end]);
            i = end;
        } else if is_number(c) {
            // ` ?\p{N}+`
            let end = text[j..].find(|c: char| !is_number(c))
                .map(|k| j + k).unwrap_or(text.len());
            out.push(&text[start..end]);
            i = end;
        } else if !c.is_whitespace() {
            // ` ?[^\s\p{L}\p{N}]+`
            let end = text[j..]
                .find(|c: char| c.is_whitespace() || is_letter(c) || is_number(c))
                .map(|k| j + k).unwrap_or(text.len());
            out.push(&text[start..end]);
            i = end;
        } else {
            // Whitespace reached with a pending space: emit the run.
            let end = text[j..].find(|c: char| !c.is_whitespace())
                .map(|k| j + k).unwrap_or(text.len());
            out.push(&text[start..end]);
            i = end;
        }
    }
    out
}

/// A learned byte-level BPE: the ordered merge list plus the vocabulary it
/// implies. Feed straight into [`crate::tokenizer::Tokenizer::new`].
pub struct TrainedBpe {
    /// (token bytes, id), ids dense from 0. The 256 byte tokens come first
    /// so every input remains representable.
    pub vocab: Vec<(Vec<u8>, u32)>,
    /// Merges in priority order — first merges first, which is the order
    /// `Tokenizer` ranks them by.
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Train byte-level BPE on `corpus`, GPT-2 style.
///
/// The algorithm is Sennrich et al. 2016 as GPT-2 applies it:
///
/// 1. Pre-tokenize into words and count how often each word occurs. Every
///    later step works on the ~10⁵ DISTINCT words weighted by count, never
///    on the raw text again — that is what makes training a 700 MB corpus
///    feasible rather than quadratic in the bytes.
/// 2. Represent each word as a sequence of byte-level symbols.
/// 3. Repeatedly find the most frequent adjacent symbol pair across all
///    words (weighted by word count), emit it as the next merge, and apply
///    it inside every word that contains it.
/// 4. Stop at `vocab_size`.
///
/// `vocab_size` counts the 256 byte tokens, so GPT-2's 50257 is 50000
/// merges plus 256 bytes plus one special.
pub fn train(corpus: &str, vocab_size: usize) -> TrainedBpe {
    // The byte-level alphabet: every symbol is one printable stand-in char,
    // so a word is a Vec of those. Working in the mapped alphabet (not raw
    // bytes) is what makes the learned merges match what a GPT-2-format
    // file expects, and keeps every symbol a valid `str`.
    let (b2c, _) = crate::tokenizer::byte_level_alphabet_pub();

    // ── 1. word frequencies ────────────────────────────────────────────
    let mut freq: HashMap<String, u64> = HashMap::new();
    for w in pretokenize(corpus) {
        let mut mapped = String::with_capacity(w.len());
        for &byte in w.as_bytes() { mapped.push(b2c[byte as usize]); }
        *freq.entry(mapped).or_insert(0) += 1;
    }

    // ── 2. words as symbol sequences ───────────────────────────────────
    let mut words: Vec<(Vec<String>, u64)> = freq.into_iter()
        .map(|(w, n)| (w.chars().map(|c| c.to_string()).collect(), n))
        .collect();

    // ── 3+4. merge loop ────────────────────────────────────────────────
    let mut merges: Vec<(String, String)> = Vec::new();
    let target_merges = vocab_size.saturating_sub(256);
    // Pair counts are rebuilt only for words the last merge touched, so the
    // cost per merge is proportional to the affected words rather than to
    // the whole corpus.
    let mut pairs: HashMap<(String, String), i64> = HashMap::new();
    for (w, n) in &words {
        for p in w.windows(2) {
            *pairs.entry((p[0].clone(), p[1].clone())).or_insert(0) += *n as i64;
        }
    }

    while merges.len() < target_merges {
        // Most frequent pair; ties broken by the pair itself so training is
        // REPRODUCIBLE — a HashMap's iteration order is not, and a
        // tokenizer that differs run to run cannot be shipped with a model.
        let best = pairs.iter()
            .filter(|(_, &n)| n > 0)
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(p, _)| p.clone());
        let Some((l, r)) = best else { break };
        let joined = format!("{l}{r}");

        // Apply to every word containing the pair, updating pair counts
        // incrementally: the neighbours of a merged pair change, nothing else.
        for (w, n) in words.iter_mut() {
            if w.len() < 2 { continue; }
            let n = *n as i64;
            let mut k = 0usize;
            while k + 1 < w.len() {
                if w[k] == l && w[k + 1] == r {
                    // Left neighbour: (prev, l) becomes (prev, joined).
                    if k > 0 {
                        *pairs.entry((w[k - 1].clone(), l.clone())).or_insert(0) -= n;
                        *pairs.entry((w[k - 1].clone(), joined.clone())).or_insert(0) += n;
                    }
                    // Right neighbour: (r, next) becomes (joined, next).
                    if k + 2 < w.len() {
                        *pairs.entry((r.clone(), w[k + 2].clone())).or_insert(0) -= n;
                        *pairs.entry((joined.clone(), w[k + 2].clone())).or_insert(0) += n;
                    }
                    w[k] = joined.clone();
                    w.remove(k + 1);
                } else {
                    k += 1;
                }
            }
        }
        pairs.remove(&(l.clone(), r.clone()));
        merges.push((l, r));
    }

    // ── vocabulary ─────────────────────────────────────────────────────
    // The 256 byte symbols first (ids 0..255, so a byte token's id is
    // stable), then one token per merge in the order they were learned.
    let mut vocab: Vec<(Vec<u8>, u32)> = Vec::with_capacity(256 + merges.len());
    for b in 0u32..256 {
        vocab.push((b2c[b as usize].to_string().into_bytes(), b));
    }
    for (i, (l, r)) in merges.iter().enumerate() {
        vocab.push((format!("{l}{r}").into_bytes(), 256 + i as u32));
    }

    TrainedBpe {
        vocab,
        merges: merges.into_iter()
            .map(|(l, r)| (l.into_bytes(), r.into_bytes()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenating the pre-tokens must reproduce the input exactly. If it
    /// does not, some text is silently dropped or duplicated before the
    /// model ever sees it.
    #[test]
    fn pretokenize_is_lossless() {
        for s in [
            "hello world",
            "  leading and   multiple   spaces  ",
            "don't stop; it's 42 apples!",
            "tabs\tand\nnewlines\r\n",
            "CJK 日本語 and emoji 🙂 mixed",
            "",
            " ",
            "a",
        ] {
            let joined: String = pretokenize(s).concat();
            assert_eq!(joined, s, "pre-tokenization lost or duplicated text for {s:?}");
        }
    }

    /// The landmark GPT-2 splits. These are the cases the reference
    /// implementation is universally checked against.
    #[test]
    fn pretokenize_matches_gpt2_landmarks() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pretokenize("it's 42"), vec!["it", "'s", " 42"]);
        // A run of spaces: all but the last stay together, the last joins
        // the following word. This is `\s+(?!\S)`.
        assert_eq!(pretokenize("a   b"), vec!["a", "  ", " b"]);
        // Punctuation groups separately from letters and digits.
        assert_eq!(pretokenize("x=1;"), vec!["x", "=", "1", ";"]);
    }

    /// Training must be REPRODUCIBLE: the same corpus and size must give
    /// byte-identical merges, or a checkpoint cannot be paired with its
    /// tokenizer. HashMap iteration order is not deterministic, so the tie
    /// break is what makes this hold.
    #[test]
    fn training_is_reproducible() {
        let corpus = "the cat sat on the mat. the cat ate the rat. \
                      a cat and a rat and a mat.";
        let a = train(corpus, 300);
        let b = train(corpus, 300);
        assert_eq!(a.merges, b.merges, "training is not reproducible");
        assert_eq!(a.vocab, b.vocab);
    }

    /// The learned vocabulary must still contain all 256 byte tokens, or
    /// the byte-level guarantee — nothing is unrepresentable — is lost.
    #[test]
    fn training_keeps_every_byte() {
        let t = train("hello hello world world", 300);
        assert!(t.vocab.len() >= 256);
        for b in 0u32..256 {
            assert_eq!(t.vocab[b as usize].1, b, "byte {b} must keep id {b}");
        }
    }

    /// The point of the exercise: merges must actually compress. On
    /// repetitive text a trained tokenizer must emit far fewer tokens than
    /// one token per byte.
    #[test]
    fn training_compresses_below_one_token_per_byte() {
        let corpus = "the cat sat on the mat ".repeat(200);
        let t = train(&corpus, 400);
        let tok = crate::tokenizer::Tokenizer::from_trained(&t);
        let ids = tok.encode(&corpus).expect("encode");
        let bytes_per_token = corpus.len() as f64 / ids.len() as f64;
        assert!(bytes_per_token > 3.0,
                "expected >3 bytes/token on repetitive text, got {bytes_per_token:.2}");
    }

    /// A trained tokenizer must survive a HuggingFace round-trip: write
    /// `tokenizer.json`, read it back, and encode identically. Without this
    /// the tokenizer cannot leave R2, and a checkpoint without a portable
    /// tokenizer is not portable either.
    #[test]
    fn hf_json_round_trip_preserves_encoding() {
        let t = train("the cat sat on the mat ".repeat(50).as_str(), 400);
        let a = crate::tokenizer::Tokenizer::from_trained(&t);
        let json = a.to_tokenizer_json();
        let b = crate::tokenizer::Tokenizer::from_tokenizer_json(&json)
            .expect("our own tokenizer.json must parse");
        for s in ["the cat sat", "unseen text 42!", "日本語 🙂"] {
            assert_eq!(a.encode(s).expect("a"), b.encode(s).expect("b"),
                       "round-tripped tokenizer encodes {s:?} differently");
        }
        assert_eq!(a.vocab_size(), b.vocab_size());
    }

    /// Round-tripping must survive training: decode(encode(s)) == s for
    /// text the tokenizer was never trained on, including non-ASCII.
    #[test]
    fn trained_tokenizer_round_trips_unseen_text() {
        let t = train("the cat sat on the mat ".repeat(50).as_str(), 400);
        let tok = crate::tokenizer::Tokenizer::from_trained(&t);
        for s in ["the cat", "unseen WORDS 123!", "日本語 🙂", "\ttab\n"] {
            assert_eq!(tok.decode(&tok.encode(s).expect("encode")), s,
                       "round-trip failed for {s:?}");
        }
    }
}
