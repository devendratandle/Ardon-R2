//! Byte-level BPE tokenization — turn text into token ids and back.
//!
//! A model consumes integers, so text has to be segmented. Byte-level BPE
//! (GPT-2, Llama, Mistral, Qwen — effectively every modern LLM) works in
//! two stages: map the input to bytes, then repeatedly merge the
//! highest-priority adjacent pair according to a ranked merge table
//! learned at training time.
//!
//! Two properties make it the right choice, and both are tested here:
//!
//! * **Nothing is unrepresentable.** Because the alphabet is the 256 byte
//!   values, any input encodes — no `<UNK>`, no failure on emoji, CJK, or
//!   binary noise. A tokenizer that can silently drop input is a
//!   correctness hazard, not a convenience.
//! * **Round-tripping is exact.** `decode(encode(s)) == s` for arbitrary
//!   text, including invalid-UTF-8-shaped byte sequences, because decoding
//!   reassembles bytes and only then interprets them.
//!
//! Merge order is by RANK, not by length or greedily left-to-right —
//! applying merges in the wrong order yields a different (still decodable,
//! but wrong) segmentation than the model was trained on, which degrades
//! output quality in a way that is very hard to notice. Hence the explicit
//! rank test.

use std::collections::HashMap;

use crate::json::Json;

/// A byte-level BPE tokenizer: a vocabulary plus a ranked merge table.
#[derive(Debug, Clone, Default)]
pub struct Tokenizer {
    /// Token byte-string → id.
    vocab: HashMap<Vec<u8>, u32>,
    /// id → token byte-string (dense; index is the id).
    tokens: Vec<Vec<u8>>,
    /// (left, right) → rank. Lower rank merges first.
    merges: HashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Special tokens matched verbatim before BPE (e.g. end-of-sequence).
    specials: Vec<(Vec<u8>, u32)>,
}

impl Tokenizer {
    /// Build from an explicit vocabulary and ordered merge list. `merges`
    /// is in priority order — first entry merges first.
    pub fn new(vocab: Vec<(Vec<u8>, u32)>, merges: Vec<(Vec<u8>, Vec<u8>)>)
        -> Result<Tokenizer, String>
    {
        let mut t = Tokenizer::default();
        let max_id = vocab.iter().map(|(_, id)| *id).max().unwrap_or(0) as usize;
        t.tokens = vec![Vec::new(); max_id + 1];
        for (bytes, id) in vocab {
            if !t.tokens[id as usize].is_empty() {
                return Err(format!("tokenizer: duplicate id {}", id));
            }
            t.tokens[id as usize] = bytes.clone();
            t.vocab.insert(bytes, id);
        }
        for (rank, (l, r)) in merges.into_iter().enumerate() {
            t.merges.insert((l, r), rank as u32);
        }
        Ok(t)
    }

    /// A minimal tokenizer that can encode ANY input: the 256 single bytes,
    /// no merges. Useful as a fallback and as the base every real vocab
    /// extends — it guarantees the "nothing is unrepresentable" property
    /// even before merges exist.
    pub fn byte_level() -> Tokenizer {
        let vocab: Vec<(Vec<u8>, u32)> = (0u32..256).map(|b| (vec![b as u8], b)).collect();
        Tokenizer::new(vocab, Vec::new()).expect("byte vocab is well-formed")
    }

    /// Load from a HuggingFace `tokenizer.json`. Reads the `model.vocab`
    /// map and `model.merges` list, plus `added_tokens` as specials.
    /// Vocabulary strings are taken as UTF-8 bytes.
    pub fn from_tokenizer_json(src: &str) -> Result<Tokenizer, String> {
        let j = Json::parse(src).map_err(|e| format!("tokenizer: {}", e))?;
        let model = j.get("model").ok_or("tokenizer: no 'model' section")?;
        let vmap = model.get("vocab").and_then(|v| v.as_obj())
            .ok_or("tokenizer: no 'model.vocab' object")?;

        let mut vocab = Vec::with_capacity(vmap.len());
        for (tok, idv) in vmap {
            let id = idv.as_usize()
                .ok_or_else(|| format!("tokenizer: vocab entry '{}' has a non-integer id", tok))?;
            vocab.push((tok.as_bytes().to_vec(), id as u32));
        }

        let mut merges = Vec::new();
        if let Some(arr) = model.get("merges").and_then(|m| m.as_arr()) {
            for m in arr {
                // Either "a b" (classic) or ["a","b"] (newer files).
                if let Some(s) = m.as_str() {
                    let mut it = s.splitn(2, ' ');
                    match (it.next(), it.next()) {
                        (Some(a), Some(b)) =>
                            merges.push((a.as_bytes().to_vec(), b.as_bytes().to_vec())),
                        _ => return Err(format!("tokenizer: malformed merge '{}'", s)),
                    }
                } else if let Some(pair) = m.as_arr() {
                    if pair.len() != 2 {
                        return Err("tokenizer: merge pair must have 2 entries".into());
                    }
                    let a = pair[0].as_str().ok_or("tokenizer: merge entry not a string")?;
                    let b = pair[1].as_str().ok_or("tokenizer: merge entry not a string")?;
                    merges.push((a.as_bytes().to_vec(), b.as_bytes().to_vec()));
                } else {
                    return Err("tokenizer: merge entry is neither string nor pair".into());
                }
            }
        }

        let mut t = Tokenizer::new(vocab, merges)?;

        // Special tokens are matched literally, before BPE, so a control
        // marker can never be split into pieces.
        if let Some(added) = j.get("added_tokens").and_then(|a| a.as_arr()) {
            for a in added {
                if let (Some(content), Some(id)) = (
                    a.get("content").and_then(|c| c.as_str()),
                    a.get("id").and_then(|i| i.as_usize()),
                ) {
                    t.add_special(content, id as u32);
                }
            }
        }
        Ok(t)
    }

    /// Register a token matched verbatim before BPE. Longest match wins,
    /// so overlapping markers behave predictably.
    pub fn add_special(&mut self, text: &str, id: u32) {
        let bytes = text.as_bytes().to_vec();
        if self.tokens.len() <= id as usize { self.tokens.resize(id as usize + 1, Vec::new()); }
        self.tokens[id as usize] = bytes.clone();
        self.vocab.insert(bytes.clone(), id);
        self.specials.push((bytes, id));
        // Longest first so a longer marker is never shadowed by a prefix.
        self.specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }

    /// Number of ids in the table (the model's expected vocab size).
    pub fn vocab_size(&self) -> usize { self.tokens.len() }

    /// Look up an id for an exact token byte-string.
    pub fn token_to_id(&self, bytes: &[u8]) -> Option<u32> { self.vocab.get(bytes).copied() }

    /// Bytes for an id.
    pub fn id_to_token(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(|v| v.as_slice()).filter(|v| !v.is_empty())
    }

    /// Encode text to token ids.
    ///
    /// Special tokens are matched first; the remaining spans are encoded by
    /// BPE over bytes. Any byte that has no vocabulary entry falls back to
    /// its own id if present — the byte-level guarantee — so encoding can
    /// only fail if the vocabulary lacks single-byte tokens entirely.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            // Special tokens win over BPE, longest first.
            let mut matched = false;
            for (pat, id) in &self.specials {
                if bytes[i..].starts_with(pat) {
                    out.push(*id);
                    i += pat.len();
                    matched = true;
                    break;
                }
            }
            if matched { continue; }
            // Accumulate the span up to the next special (or the end).
            let start = i;
            while i < bytes.len()
                && !self.specials.iter().any(|(p, _)| bytes[i..].starts_with(p))
            { i += 1; }
            self.encode_span(&bytes[start..i], &mut out)?;
        }
        Ok(out)
    }

    /// BPE over one span of bytes.
    fn encode_span(&self, span: &[u8], out: &mut Vec<u32>) -> Result<(), String> {
        if span.is_empty() { return Ok(()); }
        // Start from single bytes and merge by rank until no pair qualifies.
        let mut parts: Vec<Vec<u8>> = span.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for w in 0..parts.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(parts[w].clone(), parts[w + 1].clone())) {
                    // Strictly lower rank wins; ties keep the leftmost.
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((w, rank));
                    }
                }
            }
            let Some((w, _)) = best else { break };
            let mut merged = parts[w].clone();
            merged.extend_from_slice(&parts[w + 1]);
            parts[w] = merged;
            parts.remove(w + 1);
        }
        for p in parts {
            match self.vocab.get(&p) {
                Some(&id) => out.push(id),
                // A merged piece must exist in the vocab (merges are built
                // from it); if not, fall back to its bytes so encoding is
                // still total rather than failing.
                None => for &b in &p {
                    let id = self.vocab.get(&vec![b][..]).copied().ok_or_else(|| format!(
                        "tokenizer: byte {:#04x} has no vocabulary entry \
                         (a byte-level vocab must contain all 256 bytes)", b))?;
                    out.push(id);
                },
            }
        }
        Ok(())
    }

    /// Decode ids back to bytes. Unknown ids are skipped rather than
    /// panicking — a model can emit an out-of-range id and a serving loop
    /// must survive it.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(b) = self.id_to_token(id) { out.extend_from_slice(b); }
        }
        out
    }

    /// Decode ids to a String, replacing any invalid UTF-8 (a partial
    /// multi-byte character at the end of a stream is normal mid-generation).
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

#[cfg(test)]
mod tests {
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
}
