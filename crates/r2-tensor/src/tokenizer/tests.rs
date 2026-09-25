//! Core tests for the BPE tokenizer (`super` is `crate::tokenizer`).

use super::*;

/// Toy tokenizer: all 256 bytes plus merges that build "ab" then "abc".
fn toy() -> Tokenizer {
    let mut vocab: Vec<(Vec<u8>, u32)> = (0u32..256).map(|b| (vec![b as u8], b)).collect();
    vocab.push((b"ab".to_vec(), 300));
    vocab.push((b"abc".to_vec(), 301));
    let merges = vec![
        (b"a".to_vec(), b"b".to_vec()),    // rank 0
        (b"ab".to_vec(), b"c".to_vec()),   // rank 1
    ];
    Tokenizer::new(vocab, merges).unwrap()
}

#[test]
fn merges_apply_in_rank_order() {
    let t = toy();
    // "abc" must become the single token 301 via ab -> abc, NOT
    // [ab][c] and not [a][bc]. Wrong order still decodes, but is a
    // different segmentation than the model was trained on.
    assert_eq!(t.encode("abc").unwrap(), vec![301]);
    assert_eq!(t.encode("ab").unwrap(), vec![300]);
    // 'x' has no merge, so it stays a byte token.
    assert_eq!(t.encode("abx").unwrap(), vec![300, b'x' as u32]);
}

#[test]
fn round_trips_arbitrary_text_exactly() {
    let t = toy();
    for s in ["", "abc", "hello world", "abcabc", "a b c",
              "émoji 😀 中文", "tabs\tand\nnewlines", "\u{0}\u{1}binary\u{7f}"] {
        let ids = t.encode(s).unwrap();
        assert_eq!(t.decode(&ids), s, "round-trip failed for {:?}", s);
    }
}

#[test]
fn every_byte_is_representable() {
    // The byte-level guarantee: no input can fail to encode, so there
    // is no <UNK> and no silent data loss.
    let t = Tokenizer::byte_level();
    let all: Vec<u8> = (0u8..=255).collect();
    let s = String::from_utf8_lossy(&all).into_owned();
    let ids = t.encode(&s).unwrap();
    assert_eq!(t.decode_bytes(&ids), s.as_bytes());
    assert_eq!(t.vocab_size(), 256);
}

#[test]
fn special_tokens_are_matched_verbatim_longest_first() {
    let mut t = toy();
    t.add_special("<|eos|>", 400);
    t.add_special("<|e|>", 401);
    let ids = t.encode("abc<|eos|>abc").unwrap();
    assert_eq!(ids, vec![301, 400, 301], "special must not be split by BPE");
    // The longer marker wins even though the shorter one is a prefix-ish match.
    assert_eq!(t.encode("<|e|>").unwrap(), vec![401]);
    assert_eq!(t.decode(&[301, 400]), "abc<|eos|>");
}

#[test]
fn loads_a_tokenizer_json() {
    // Both merge spellings: "a b" and ["ab","c"].
    let src = r#"{
      "added_tokens":[{"id":9,"content":"<|end|>"}],
      "model":{
        "vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},
        "merges":["a b",["ab","c"]]
      }}"#;
    let t = Tokenizer::from_tokenizer_json(src).unwrap();
    assert_eq!(t.token_to_id(b"abc"), Some(4));
    assert_eq!(t.encode("abc").unwrap(), vec![4]);
    assert_eq!(t.encode("a<|end|>").unwrap(), vec![0, 9]);
    assert_eq!(t.decode(&[3, 2]), "abc");
}

#[test]
fn malformed_tokenizer_json_is_rejected() {
    assert!(Tokenizer::from_tokenizer_json("{}").is_err());
    assert!(Tokenizer::from_tokenizer_json(r#"{"model":{}}"#).is_err());
    assert!(Tokenizer::from_tokenizer_json(
        r#"{"model":{"vocab":{"a":"x"}}}"#).is_err(), "non-integer id");
    assert!(Tokenizer::from_tokenizer_json(
        r#"{"model":{"vocab":{"a":0},"merges":["ab"]}}"#).is_err(), "merge needs a pair");
}

#[test]
fn duplicate_ids_are_rejected() {
    let v = vec![(b"a".to_vec(), 0u32), (b"b".to_vec(), 0u32)];
    assert!(Tokenizer::new(v, vec![]).unwrap_err().contains("duplicate id"));
}

#[test]
fn decode_survives_unknown_ids() {
    // A model can emit an out-of-range id; serving must not panic.
    let t = toy();
    assert_eq!(t.decode(&[b'a' as u32, 99999, b'b' as u32]), "ab");
    assert_eq!(t.decode(&[]), "");
}

#[test]
fn encode_is_deterministic() {
    let t = toy();
    let a = t.encode("abcabc abx").unwrap();
    for _ in 0..5 { assert_eq!(t.encode("abcabc abx").unwrap(), a); }
}
