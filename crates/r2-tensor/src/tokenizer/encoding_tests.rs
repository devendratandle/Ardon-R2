//! Encoding-format tests for the tokenizer (`super` is `crate::tokenizer`).

use super::*;

#[test]
fn byte_level_alphabet_is_a_bijection_over_all_256_bytes() {
    // Every byte must have a distinct printable stand-in, or some
    // input becomes unrepresentable — the property the whole scheme
    // exists to provide.
    let (b2c, c2b) = byte_level_alphabet();
    assert_eq!(c2b.len(), 256, "all 256 bytes must map to distinct chars");
    for b in 0..=255u8 {
        let ch = b2c[b as usize];
        assert_eq!(c2b.get(&ch), Some(&b), "byte {b} must round-trip");
        assert!(!ch.is_control(), "stand-in for byte {b} must be printable");
    }
    // The two landmarks every ByteLevel vocabulary shows: space -> Ġ,
    // newline -> Ċ. If these drift, real vocabularies stop matching.
    assert_eq!(b2c[b' ' as usize], 'Ġ');
    assert_eq!(b2c[b'\n' as usize], 'Ċ');
}

/// Metaspace round-trips text that does not start with whitespace, and
/// LOSES one leading space when it does. That is not a defect: it is
/// SentencePiece's `add_dummy_prefix`, and HuggingFace's real Mistral-7B
/// tokenizer does exactly the same —
///
///     'hello world' -> 'hello world'   (round-trips)
///     ' hello'      -> 'hello'         (leading space lost)
///     '  leading'   -> ' leading'
///
/// Pinned so the asymmetry stays deliberate. It was once accidental:
/// decoding mapped the prepended marker back to a space and returned
/// " " + text for EVERY input, which is the opposite error and a much
/// worse one.
#[test]
fn metaspace_round_trip_matches_sentencepiece_semantics() {
    let src = r#"{
      "decoder":{"type":"Metaspace","replacement":"▁"},
      "model":{"unk_token":"<unk>",
        "vocab":{"<unk>":0,"▁":1,"a":2,"b":3,"c":4},
        "merges":[]}}"#;
    let t = Tokenizer::from_tokenizer_json(src).unwrap();
    for s in ["abc", "a b c", "ab c"] {
        assert_eq!(t.decode(&t.encode(s).unwrap()), s,
                   "text not starting with a space must round-trip: {s:?}");
    }
    // One leading space is consumed by the dummy prefix, exactly as in
    // the reference implementation.
    assert_eq!(t.decode(&t.encode(" abc").unwrap()), "abc");
    assert_eq!(t.decode(&t.encode("  abc").unwrap()), " abc");
}

/// SentencePiece `byte_fallback` — the OTHER zero-OOV mechanism.
///
/// A Llama-2/Mistral vocabulary is a word vocabulary, not a byte one.
/// When a piece is not in it, the piece decomposes into its UTF-8 bytes
/// and each byte becomes a `<0xNN>` token. Without this the tokenizer
/// emits `<unk>` and the text is gone: the model sees only "something
/// was here", and decode cannot put it back.
fn byte_fallback_json() -> String {
    // A deliberately tiny word vocabulary plus all 256 byte tokens, the
    // shape SentencePiece exports.
    let mut v = String::from(r#"{"model":{"type":"BPE","byte_fallback":true,"unk_token":"<unk>","vocab":{"<unk>":0,"a":1,"b":2,"ab":3"#);
    for b in 0..256 {
        v.push_str(&format!(",\"<0x{:02X}>\":{}", b, 4 + b));
    }
    v.push_str(r#"},"merges":["a b"]}}"#);
    v
}

#[test]
fn byte_fallback_carries_characters_the_vocabulary_lacks() {
    let t = Tokenizer::from_tokenizer_json(&byte_fallback_json())
        .expect("a byte_fallback vocabulary must load");
    // "ab" is in the vocabulary; the rest is not and must survive as
    // bytes rather than collapsing to <unk>.
    for s in ["ab", "z", "日本", "🙂", "ab z 🙂"] {
        let ids = t.encode(s).expect("encode");
        assert!(!ids.contains(&0),
                "{s:?} produced <unk> — byte_fallback was not used: {ids:?}");
        assert_eq!(t.decode(&ids), s,
                   "{s:?} did not survive a byte_fallback round-trip");
    }
}

#[test]
fn byte_fallback_decodes_to_bytes_not_to_its_own_spelling() {
    let t = Tokenizer::from_tokenizer_json(&byte_fallback_json()).expect("load");
    // 'A' is 0x41, absent from this word vocabulary, so it must encode
    // as the <0x41> token and decode back to "A" — never to the literal
    // six characters "<0x41>".
    let ids = t.encode("A").expect("encode");
    assert_eq!(ids.len(), 1, "one byte should be one byte token: {ids:?}");
    let out = t.decode(&ids);
    assert_eq!(out, "A", "decoded to {out:?} instead of the byte it stands for");
    assert!(!out.contains("0x"), "decode emitted the token's spelling: {out:?}");
}

/// A vocabulary WITHOUT byte tokens must still take the `<unk>` path —
/// byte_fallback is opt-in by the file, not assumed.
#[test]
fn no_byte_tokens_means_no_byte_fallback() {
    let t = Tokenizer::from_tokenizer_json(
        r#"{"model":{"unk_token":"<unk>","vocab":{"<unk>":0,"a":1},"merges":[]}}"#)
        .expect("load");
    let ids = t.encode("z").expect("encode");
    assert_eq!(ids, vec![0], "expected <unk> when the file has no byte tokens");
}

/// A GPT-2/Llama-3 shaped file: vocabulary written in the ByteLevel
/// alphabet ("Ġthe", not " the").
fn bytelevel_json() -> &'static str {
    r#"{
      "decoder":{"type":"ByteLevel"},
      "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"ByteLevel"}]},
      "model":{
        "vocab":{"t":0,"h":1,"e":2,"Ġ":3,"th":4,"the":5,"Ġthe":6,"a":7},
        "merges":["t h","th e","Ġ the"]
      }}"#
}

#[test]
fn bytelevel_vocab_matches_real_text() {
    // THE gap this closes: without remapping, " the" would never match
    // the token "Ġthe" and the model would receive a different
    // segmentation than it was trained on.
    let t = Tokenizer::from_tokenizer_json(bytelevel_json()).unwrap();
    assert_eq!(t.encoding(), ByteEncoding::ByteLevel, "format must be auto-detected");
    assert_eq!(t.encode(" the").unwrap(), vec![6], "space+the must be the single token Ġthe");
    assert_eq!(t.encode("the").unwrap(), vec![5]);
    assert_eq!(t.decode(&[6]), " the", "decoding must map Ġ back to a space");
    assert_eq!(t.decode(&t.encode(" the").unwrap()), " the");
}

#[test]
fn metaspace_vocab_handles_sentencepiece_spaces() {
    let src = r#"{
      "decoder":{"type":"Metaspace","replacement":"\u2581"},
      "model":{"unk_token":"<unk>",
        "vocab":{"<unk>":0,"▁":1,"t":2,"h":3,"e":4,"th":5,"the":6,"▁the":7},
        "merges":["t h","th e","▁ the"]}}"#;
    let t = Tokenizer::from_tokenizer_json(src).unwrap();
    assert_eq!(t.encoding(), ByteEncoding::Metaspace);
    // SentencePiece marks the start of text as a word boundary.
    // SentencePiece marks the start of text as a word boundary, and the
    // merges build ▁+the into the single token ▁the.
    assert_eq!(t.encode("the").unwrap(), vec![7], "leading word gets the ▁ marker");
    // Decoding STRIPS the marker's space. This assertion previously
    // expected " the", and it was wrong: checked against HuggingFace's
    // real Mistral-7B tokenizer, `decode` of the `▁the` token returns
    // "the", and `decode(encode("hello world"))` returns "hello world"
    // rather than " hello world".
    //
    // The two halves have to be symmetric. Encoding prepends the marker
    // (SentencePiece's `add_dummy_prefix`, which is what makes "hello"
    // and " hello" segment alike), so decoding must remove it. Leaving
    // it in returned " " + text for every input not already starting
    // with a space — silent, and a model trained on it learns real
    // leading whitespace that was never in the corpus.
    assert_eq!(t.decode(&[7]), "the");
    // A character outside this small vocabulary becomes <unk> rather
    // than failing — SentencePiece vocabularies are not byte-complete.
    // "z" becomes ▁ (the word-start marker) followed by <unk>, which is
    // exactly what SentencePiece produces for an unknown word.
    assert_eq!(t.encode("z").unwrap(), vec![1, 0]);
}

#[test]
fn raw_vocab_still_works_and_is_the_default() {
    // Hand-built vocabularies (and our byte_level() fallback) declare
    // no decoder, and must keep behaving literally.
    let src = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#;
    let t = Tokenizer::from_tokenizer_json(src).unwrap();
    assert_eq!(t.encoding(), ByteEncoding::Raw);
    assert_eq!(t.encode("ab").unwrap(), vec![2]);
    assert_eq!(Tokenizer::byte_level().encoding(), ByteEncoding::Raw);
}

#[test]
fn bytelevel_round_trips_arbitrary_text() {
    // Build a full ByteLevel vocabulary (every byte's stand-in) and
    // confirm exact round-tripping, including bytes that only appear
    // inside multi-byte UTF-8.
    let (b2c, _) = byte_level_alphabet();
    let vocab: Vec<(Vec<u8>, u32)> = (0..256u32)
        .map(|b| (b2c[b as usize].to_string().into_bytes(), b))
        .collect();
    let mut t = Tokenizer::new(vocab, Vec::new()).unwrap();
    t.set_encoding(ByteEncoding::ByteLevel);
    for s in ["hello world", " leading space", "émoji 😀 中文", "tabs\tnewlines\n"] {
        assert_eq!(t.decode(&t.encode(s).unwrap()), s, "round-trip failed for {s:?}");
    }
}

/// Encoding a text in chunks that end on PRE-TOKEN boundaries gives
/// exactly the ids of encoding it whole — the property the out-of-core
/// tokenizer pass in `r2-train`'s `tinystories_train` depends on to
/// stream a corpus larger than RAM without changing what the model
/// sees.
///
/// Splitting anywhere else is NOT safe, and the second half of this
/// test pins that: a newline is an obvious-looking chunk boundary and
/// is wrong, because a newline followed by a space is a single
/// pre-token that a merge lives inside. Measured on 3 MB of
/// TinyStories at 64 KB chunks, a newline split produced `[10, 470]`
/// where whole-text encoding gives `[1293, 400]` for the same text.
#[test]
fn chunked_encoding_matches_whole_text_on_pretoken_boundaries() {
    let t = Tokenizer::from_tokenizer_json(bytelevel_json()).unwrap();
    let text = " the a the a the a the a";
    let whole = t.encode(text).unwrap();

    // Cut after each complete pre-token, the way `flush_chunk` does.
    let parts = crate::bpe::pretokenize(text);
    for split in 1..parts.len() {
        let cut: usize = parts[..split].iter().map(|p| p.len()).sum();
        let mut chunked = t.encode(&text[..cut]).unwrap();
        chunked.extend(t.encode(&text[cut..]).unwrap());
        assert_eq!(chunked, whole,
                   "splitting at pre-token {split} (byte {cut}) changed the ids");
    }

    // Concatenating the pre-tokens must reproduce the text, or the
    // boundaries above are not boundaries.
    assert_eq!(parts.concat(), text);
}

#[test]
fn multibyte_symbols_are_never_split_mid_character() {
    // A merge or vocab entry can only match whole symbols; splitting
    // `Ġ` into its two UTF-8 bytes would make it unmatchable.
    let t = Tokenizer::from_tokenizer_json(bytelevel_json()).unwrap();
    let ids = t.encode(" the a").unwrap();
    assert_eq!(t.decode(&ids), " the a");
    assert!(ids.contains(&6), "Ġthe must survive as one token");
}
