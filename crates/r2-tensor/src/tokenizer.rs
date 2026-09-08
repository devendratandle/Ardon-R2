//! BPE tokenization — turn text into token ids and back.
//!
//! A model consumes integers, so text has to be segmented. BPE (GPT-2,
//! Llama, Mistral, Qwen — effectively every modern LLM) works in two
//! stages: translate the input into the vocabulary's own alphabet, then
//! repeatedly merge the highest-priority adjacent pair according to a
//! ranked merge table learned at training time.
//!
//! THE ALPHABET IS NOT OPTIONAL. Vocabulary entries are JSON strings, so
//! a real tokenizer cannot store raw bytes: GPT-2/Llama-3 files remap
//! every byte to a printable stand-in (a space is `Ġ`), and SentencePiece
//! files write spaces as `▁`. Treating those strings as literal UTF-8
//! means " the" never matches the token "Ġthe" — the model still runs and
//! still emits text, but from a segmentation it was never trained on. The
//! format is therefore detected from the file and applied on both sides
//! (see [`ByteEncoding`]).
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

/// How raw bytes are represented inside the vocabulary.
///
/// This is the detail that decides whether a real tokenizer file works at
/// all. Vocabulary entries are JSON strings, so a tokenizer cannot store
/// arbitrary bytes directly — every family encodes them somehow:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ByteEncoding {
    /// Vocabulary strings are literal UTF-8. Used by hand-built vocabs.
    #[default]
    Raw,
    /// GPT-2 / Llama-3 / Qwen "ByteLevel": each of the 256 bytes maps to a
    /// distinct *printable* character, so a space appears as `Ġ` and a
    /// newline as `Ċ`. Encoding text means translating bytes into that
    /// alphabet BEFORE merging — otherwise " the" never matches the token
    /// "Ġthe" and the model receives a completely different segmentation.
    ByteLevel,
    /// SentencePiece "Metaspace" (Llama-2, Mistral): spaces are written as
    /// `▁` (U+2581) and a leading `▁` marks the start of a word.
    Metaspace,
}


/// GPT-2's byte↔character alphabet.
///
/// Returns `(byte → char, char → byte)`. The construction is fixed by the
/// original GPT-2 implementation and every ByteLevel tokenizer since:
/// printable ASCII and Latin-1 ranges map to themselves; the remaining
/// bytes are pushed into the unused U+0100.. range so that every byte has
/// a printable, single-character representation.
fn byte_level_alphabet() -> (Vec<char>, HashMap<char, u8>) {
    let mut bs: Vec<u16> = Vec::new();
    bs.extend(b'!' as u16..=b'~' as u16);
    bs.extend(0xA1u16..=0xACu16);
    bs.extend(0xAEu16..=0xFFu16);
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u16..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("alphabet code points are valid");
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

/// The GPT-2 alphabet, for `crate::bpe`'s trainer. Same table the encoder
/// uses, so learned merges are in the alphabet the vocabulary is written in.
pub(crate) fn byte_level_alphabet_pub() -> (Vec<char>, HashMap<char, u8>) {
    byte_level_alphabet()
}

/// Working buffers for one `encode` call, reused across every word in it.
#[derive(Default)]
pub(crate) struct BpeScratch {
    /// Pieces as (start, end) ranges into the word being merged.
    parts: Vec<(usize, usize)>,
    /// `ranks[i]` = rank of merging piece i with piece i+1, or MAX.
    ranks: Vec<u32>,
}

/// Everything one `encode` call needs to allocate, allocated once.
#[derive(Default)]
pub(crate) struct EncodeScratch {
    /// The word remapped into the vocabulary's alphabet.
    mapped: Vec<u8>,
    bpe: BpeScratch,
}

/// A byte-level BPE tokenizer: a vocabulary plus a ranked merge table.
#[derive(Debug, Clone, Default)]
pub struct Tokenizer {
    /// Token byte-string → id.
    vocab: HashMap<Vec<u8>, u32>,
    /// id → token byte-string (dense; index is the id).
    tokens: Vec<Vec<u8>>,
    /// left → (right → rank). Lower rank merges first.
    ///
    /// NESTED rather than keyed by `(Vec<u8>, Vec<u8>)`. A tuple key cannot
    /// be looked up from two borrowed slices, so the flat form forced
    /// `parts[w].clone()` on BOTH sides for every candidate pair on every
    /// merge iteration — an allocation per lookup, in the innermost loop of
    /// the tokenizer. Nested, `HashMap<Vec<u8>, _>::get` takes a `&[u8]`
    /// through `Borrow`, and the whole encode path stops allocating.
    /// Measured on 2 MB of mixed prose and source, with the two other
    /// allocation fixes in the same pass (ranges instead of owned pieces in
    /// `encode_span`, one reused scratch buffer instead of one per word):
    /// **1.1 -> 3.5 MB/s**. Each step was measured, not estimated — an
    /// earlier version of this comment claimed 12.8 MB/s from a figure that
    /// had never been run.
    merges: HashMap<Vec<u8>, HashMap<Vec<u8>, u32>>,
    /// The SAME merge table keyed by the CONCATENATED result instead of by
    /// the pair — `merge_rank["th"] = rank of merging "t" with "h"`.
    ///
    /// This is the shape tiktoken's inner loop needs, and it is what lets
    /// the rank of each position be CACHED. R2's original loop looked up
    /// every adjacent pair on every iteration: O(n) hash lookups per merge,
    /// O(n^2) per word. With ranks cached in a parallel vector, a merge
    /// invalidates only its two neighbours, so the scan compares plain u32s
    /// and the whole word costs O(n) lookups total.
    ///
    /// Both tables are kept: the nested one answers "can these two merge",
    /// which `new` needs while building, and this one answers "what does
    /// this span merge to", which encoding needs.
    merge_rank: HashMap<Vec<u8>, u32>,
    /// Special tokens matched verbatim before BPE (e.g. end-of-sequence).
    specials: Vec<(Vec<u8>, u32)>,
    /// How bytes are represented in `vocab` (see [`ByteEncoding`]).
    encoding: ByteEncoding,
    /// GPT-2 alphabet tables, built once when `encoding` is ByteLevel.
    byte_to_char: Vec<char>,
    char_to_byte: HashMap<char, u8>,
    /// SentencePiece `byte_fallback`: ids for the 256 `<0xNN>` tokens, when
    /// the vocabulary declares them.
    ///
    /// The OTHER zero-OOV mechanism, and not the one GPT-2 uses. GPT-2's
    /// ByteLevel converts all text to a byte alphabet BEFORE merging, so an
    /// unknown piece cannot arise. SentencePiece (Llama-2, Mistral) keeps a
    /// normal word vocabulary and, when a piece is not in it, decomposes
    /// that piece into its UTF-8 bytes and emits one `<0xNN>` token per
    /// byte. Both guarantee that nothing is unrepresentable; they get there
    /// differently, and a tokenizer that implements only one cannot load
    /// the other family's files faithfully.
    ///
    /// Without this, a Llama-2 vocabulary hits the `<unk>` path for any
    /// character it lacks — the model receives a token that says only
    /// "something was here", and the text is unrecoverable on decode.
    byte_fallback: Option<Box<[u32; 256]>>,
    /// Unknown-token id, if the vocabulary declares one. SentencePiece
    /// vocabularies do NOT contain all 256 bytes, so a piece with no
    /// entry has to become <unk> — erroring there would reject text the
    /// model itself can handle.
    unk: Option<u32>,
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
            let mut joined = l.clone();
            joined.extend_from_slice(&r);
            t.merge_rank.insert(joined, rank as u32);
            t.merges.entry(l).or_default().insert(r, rank as u32);
        }
        Ok(t)
    }

    /// Declare how bytes are represented in the vocabulary, building the
    /// GPT-2 alphabet tables when needed.
    pub fn set_encoding(&mut self, enc: ByteEncoding) {
        self.encoding = enc;
        if enc == ByteEncoding::ByteLevel && self.byte_to_char.is_empty() {
            let (b2c, c2b) = byte_level_alphabet();
            self.byte_to_char = b2c;
            self.char_to_byte = c2b;
        }
    }

    pub fn encoding(&self) -> ByteEncoding { self.encoding }

    /// Text → the vocabulary's own alphabet. ByteLevel remaps every byte
    /// to its printable stand-in; Metaspace rewrites spaces as `▁` and
    /// marks the start of the text as a word boundary, which is what the
    /// SentencePiece vocabularies were trained with.
    fn to_vocab_space(&self, text: &str) -> Vec<u8> {
        match self.encoding {
            ByteEncoding::Raw => text.as_bytes().to_vec(),
            ByteEncoding::ByteLevel => {
                let mut s = String::with_capacity(text.len());
                for &b in text.as_bytes() { s.push(self.byte_to_char[b as usize]); }
                s.into_bytes()
            }
            ByteEncoding::Metaspace => {
                let mut s = String::with_capacity(text.len() + 3);
                if !text.starts_with(' ') { s.push('▁'); }
                for ch in text.chars() {
                    if ch == ' ' { s.push('▁'); } else { s.push(ch); }
                }
                s.into_bytes()
            }
        }
    }

    /// Inverse of [`to_vocab_space`], applied after concatenating tokens.
    fn from_vocab_space(&self, raw: &[u8]) -> Vec<u8> {
        match self.encoding {
            ByteEncoding::Raw => raw.to_vec(),
            ByteEncoding::ByteLevel => {
                // Each character stands for exactly one byte. An unknown
                // character can only come from a special token, which is
                // literal text — pass those through unchanged.
                match std::str::from_utf8(raw) {
                    Ok(s) => {
                        let mut out = Vec::with_capacity(s.len());
                        for ch in s.chars() {
                            match self.char_to_byte.get(&ch) {
                                Some(&b) => out.push(b),
                                None => { let mut buf = [0u8; 4]; out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()); }
                            }
                        }
                        out
                    }
                    Err(_) => raw.to_vec(),
                }
            }
            ByteEncoding::Metaspace => {
                match std::str::from_utf8(raw) {
                    Ok(s) => {
                        let mut t = s.replace('▁', " ");
                        // STRIP the word marker `to_vocab_space` prepended.
                        //
                        // Encoding adds a leading `▁` when the text does not
                        // already start with a space — SentencePiece's
                        // `add_dummy_prefix`, which is what makes "hello"
                        // and " hello" segment alike. Decoding mapped every
                        // `▁` back to a space including that one, so
                        // `decode(encode(s))` returned " " + s for every
                        // input not already starting with a space. Silent:
                        // the text was all there, just shifted by one space,
                        // which a model then learns as real leading
                        // whitespace. The two halves must be symmetric —
                        // whatever encoding adds, decoding removes.
                        if t.starts_with(' ') { t.remove(0); }
                        t.into_bytes()
                    }
                    Err(_) => raw.to_vec(),
                }
            }
        }
    }

    /// A minimal tokenizer that can encode ANY input: the 256 single bytes,
    /// no merges. Useful as a fallback and as the base every real vocab
    /// extends — it guarantees the "nothing is unrepresentable" property
    /// even before merges exist.
    pub fn byte_level() -> Tokenizer {
        let vocab: Vec<(Vec<u8>, u32)> = (0u32..256).map(|b| (vec![b as u8], b)).collect();
        Tokenizer::new(vocab, Vec::new()).expect("byte vocab is well-formed")
    }

    /// Build from a merge table learned by [`crate::bpe::train`].
    ///
    /// Sets `ByteLevel` encoding, because the trainer works in the GPT-2
    /// alphabet — the vocabulary holds "Ġthe", not " the". Getting this
    /// wrong is silent: the model still runs, on a segmentation it was
    /// never trained on.
    pub fn from_trained(t: &crate::bpe::TrainedBpe) -> Tokenizer {
        let mut tok = Tokenizer::new(t.vocab.clone(), t.merges.clone())
            .expect("a trained vocabulary is well-formed by construction");
        tok.set_encoding(ByteEncoding::ByteLevel);
        tok
    }

    /// Serialise to a HuggingFace `tokenizer.json`.
    ///
    /// The counterpart to [`from_tokenizer_json`], so a tokenizer trained
    /// here can be loaded by `tokenizers`/`transformers` and one trained
    /// there can be loaded here. A tokenizer that cannot leave the tool
    /// that made it is not interoperable with anything, and a model
    /// checkpoint without a portable tokenizer is not portable either.
    ///
    /// Emits the ByteLevel shape GPT-2 uses: `model.type = "BPE"`, the
    /// vocabulary in the byte-level alphabet, merges as `"a b"` strings in
    /// rank order, and the `ByteLevel` pre-tokenizer/decoder declarations
    /// that tell the loader this vocabulary is written in stand-in
    /// characters rather than literal UTF-8.
    ///
    /// [`from_tokenizer_json`]: Tokenizer::from_tokenizer_json
    pub fn to_tokenizer_json(&self) -> String {
        fn esc(s: &str) -> String {
            let mut o = String::with_capacity(s.len() + 2);
            for c in s.chars() {
                match c {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                    c => o.push(c),
                }
            }
            o
        }
        let bytelevel = self.encoding == ByteEncoding::ByteLevel;
        let mut s = String::from("{\n  \"version\": \"1.0\",\n");
        s.push_str("  \"truncation\": null,\n  \"padding\": null,\n");
        // added_tokens carries the specials, so a round-trip keeps them.
        s.push_str("  \"added_tokens\": [");
        for (i, (bytes, id)) in self.specials.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!(
                "\n    {{\"id\": {}, \"content\": \"{}\", \"special\": true}}",
                id, esc(&String::from_utf8_lossy(bytes))));
        }
        s.push_str("\n  ],\n");
        if bytelevel {
            // `trim_offsets` and `use_regex` are not optional in the
            // HuggingFace schema: `tokenizers` deserialises ByteLevel into a
            // struct with all four fields and rejects the file outright if
            // any is missing ("missing field `trim_offsets`"), so omitting
            // them made this export unloadable by the very library it exists
            // to interoperate with. `use_regex` is true because R2 really
            // does apply GPT-2's pre-tokenization before merging
            // (`bpe::pre_tokens`) — writing false here would describe a
            // different segmentation from the one the merges were learned
            // on, and the two sides would silently encode the same text
            // differently.
            s.push_str("  \"pre_tokenizer\": {\"type\": \"ByteLevel\", \
                        \"add_prefix_space\": false, \"trim_offsets\": true, \
                        \"use_regex\": true},\n");
            s.push_str("  \"decoder\": {\"type\": \"ByteLevel\", \
                        \"add_prefix_space\": false, \"trim_offsets\": true, \
                        \"use_regex\": true},\n");
        } else {
            s.push_str("  \"pre_tokenizer\": null,\n  \"decoder\": null,\n");
        }
        s.push_str("  \"model\": {\n    \"type\": \"BPE\",\n");
        s.push_str("    \"unk_token\": null,\n    \"vocab\": {");
        // Emit in id order so the file is stable across runs — a tokenizer
        // that serialises differently each time cannot be diffed or hashed.
        let mut first = true;
        for (id, tok) in self.tokens.iter().enumerate() {
            if tok.is_empty() && id != 0 { continue; }
            if !first { s.push(','); }
            first = false;
            s.push_str(&format!("\n      \"{}\": {}",
                                esc(&String::from_utf8_lossy(tok)), id));
        }
        s.push_str("\n    },\n    \"merges\": [");
        // Merges must be written in RANK order; the map is unordered.
        let mut ranked: Vec<(&Vec<u8>, &Vec<u8>, u32)> = self.merges.iter()
            .flat_map(|(l, rs)| rs.iter().map(move |(r, &rank)| (l, r, rank)))
            .collect();
        ranked.sort_by_key(|(_, _, r)| *r);
        for (i, (l, r, _)) in ranked.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("\n      \"{} {}\"",
                                esc(&String::from_utf8_lossy(l)),
                                esc(&String::from_utf8_lossy(r))));
        }
        s.push_str("\n    ]\n  }\n}\n");
        s
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

        // Detect the byte representation from the file itself rather than
        // asking the caller — getting it wrong silently produces a valid
        // but WRONG segmentation, which is the hardest kind of bug to see.
        // `decoder`/`pre_tokenizer` may be a single object or a
        // {"type":"Sequence","...":[..]} wrapper, so scan for the type name.
        let mut kind = String::new();
        for section in ["decoder", "pre_tokenizer"] {
            if let Some(s) = j.get(section) {
                collect_types(s, &mut kind);
            }
        }
        // Unknown token: named in model.unk_token, else the conventional
        // "<unk>" entry if the vocabulary has one.
        // `<0xNN>` byte tokens. Present iff the file declares byte_fallback
        // AND carries all 256 — a partial set would silently drop the bytes
        // it lacks, so it is all or nothing.
        {
            // Accepted whenever all 256 `<0xNN>` tokens are present, whether
            // or not the file sets `byte_fallback: true`. Some exports omit
            // the flag while still shipping the tokens, and using them is
            // strictly better than emitting <unk> for a byte the vocabulary
            // demonstrably has a token for. A PARTIAL set is refused: it
            // would carry some bytes and silently drop others, which is
            // worse than a consistent <unk>.
            let mut ids = [0u32; 256];
            let mut complete = true;
            for b in 0usize..256 {
                match t.vocab.get(format!("<0x{:02X}>", b).as_bytes()) {
                    Some(&id) => ids[b] = id,
                    None => { complete = false; break; }
                }
            }
            if complete { t.byte_fallback = Some(Box::new(ids)); }
        }
        t.unk = model.get("unk_token").and_then(|u| u.as_str())
            .and_then(|s| t.vocab.get(s.as_bytes()).copied())
            .or_else(|| t.vocab.get(&b"<unk>"[..]).copied());

        t.set_encoding(if kind.contains("ByteLevel") {
            ByteEncoding::ByteLevel
        } else if kind.contains("Metaspace") {
            ByteEncoding::Metaspace
        } else {
            ByteEncoding::Raw
        });

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
        // Specials are matched on the RAW text, and the scan advances by
        // CHARACTER so every slice below lands on a boundary.
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        // Allocated ONCE for the whole call and reused for every word.
        let mut sb = EncodeScratch::default();
        let mut i = 0usize;
        while i < text.len() {
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
            let start = i;
            while i < text.len()
                && !self.specials.iter().any(|(p, _)| bytes[i..].starts_with(p))
            {
                i += text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            }
            self.encode_text(&text[start..i], &mut out, &mut sb)?;
        }
        Ok(out)
    }

    /// Encode one special-free span of text.
    ///
    /// For ByteLevel this PRE-TOKENIZES first, running BPE separately on
    /// each GPT-2 word. That is not a refinement, it is the definition:
    /// GPT-2 splits with its regex before merging, so a merge may never
    /// cross a word boundary. Without it " the cat" can merge into a single
    /// token the reference implementation would never emit — the model runs
    /// on a segmentation it was not trained on, and nothing errors.
    ///
    /// It is also what makes encoding fast. `encode_span` rescans all of
    /// its pieces per merge, so it is quadratic in the span; bounding each
    /// run to one short word makes encoding linear in the text.
    ///
    /// Metaspace pre-tokenizes too, at each `▁`. SentencePiece defines a
    /// piece as one word carrying its leading space marker, so this is its
    /// split, not an approximation of it.
    ///
    /// That half was missed when ByteLevel was fixed, and the consequence
    /// was not subtle: loading Mistral-7B's real 32,000-token vocabulary
    /// and encoding 570 KB did not finish in NINE MINUTES, because BPE was
    /// running across the whole document as one span. The ByteLevel fix had
    /// made the quadratic path invisible on the tokenizers being tested
    /// while leaving it in place for an entire model family.
    ///
    /// Raw keeps the whole-span path: it has no word concept to split on.
    fn encode_text(&self, span: &str, out: &mut Vec<u32>,
                   sb: &mut EncodeScratch) -> Result<(), String> {
        if span.is_empty() { return Ok(()); }
        match self.encoding {
            ByteEncoding::ByteLevel => {
                for word in crate::bpe::pretokenize(span) {
                    // Remap the word into the byte-stand-in alphabet using
                    // the SHARED buffer — `to_vocab_space` would return an
                    // owned Vec per word, and the mapped word never outlives
                    // this iteration.
                    sb.mapped.clear();
                    for &b in word.as_bytes() {
                        let mut buf = [0u8; 4];
                        sb.mapped.extend_from_slice(
                            self.byte_to_char[b as usize].encode_utf8(&mut buf).as_bytes());
                    }
                    // Split the borrow: `mapped` is read, `bpe` is written.
                    let EncodeScratch { mapped, bpe } = sb;
                    self.encode_span(mapped, out, bpe)?;
                }
                Ok(())
            }
            ByteEncoding::Metaspace => {
                // Map once — this is where spaces become the marker and the
                // leading one is added — then cut at each marker. Splitting
                // AFTER mapping keeps the rule in one place: whatever
                // `to_vocab_space` decides about the leading marker is what
                // the first piece gets.
                let mapped = self.to_vocab_space(span);
                let text = match std::str::from_utf8(&mapped) {
                    Ok(t) => t,
                    // Not valid UTF-8 after mapping: one span, correct but
                    // slow, rather than splitting mid-character.
                    Err(_) => return self.encode_span(&mapped, out, &mut sb.bpe),
                };
                let mark = '\u{2581}'; // ▁
                let mut start = 0usize;
                for (i, c) in text.char_indices() {
                    if c == mark && i > start {
                        self.encode_span(text[start..i].as_bytes(), out, &mut sb.bpe)?;
                        start = i;
                    }
                }
                if start < text.len() {
                    self.encode_span(text[start..].as_bytes(), out, &mut sb.bpe)?;
                }
                Ok(())
            }
            ByteEncoding::Raw => {
                let mapped = self.to_vocab_space(span);
                self.encode_span(&mapped, out, &mut sb.bpe)
            }
        }
    }

    /// BPE over one span of bytes.
    /// `scratch` carries the two working vectors so they are allocated ONCE
    /// per call to `encode`, not once per word.
    ///
    /// They were locals. At 128,001 words in a 570 KB body that is 256,002
    /// allocations, and the merge loop is 95.3% of encode time — the
    /// allocations, not the hashing, which was measured and found neutral.
    fn encode_span(&self, span: &[u8], out: &mut Vec<u32>,
                   scratch: &mut BpeScratch) -> Result<(), String> {
        if span.is_empty() { return Ok(()); }
        // Pieces are (start, end) RANGES into `span`, not owned Vecs.
        //
        // Merging only ever joins ADJACENT pieces, so every piece — merged
        // or not — is one contiguous slice of `span`. Representing them as
        // ranges makes a merge "extend the left range", and the whole
        // routine allocates nothing per symbol and nothing per merge. The
        // owned form allocated a `Vec<u8>` for EVERY CHARACTER of every
        // word, which on 2 MB of text is millions of allocations.
        let parts = &mut scratch.parts;
        parts.clear();
        parts.reserve(span.len());
        match self.encoding {
            // Raw: the alphabet IS the byte set, so one piece per byte.
            ByteEncoding::Raw => parts.extend((0..span.len()).map(|i| (i, i + 1))),
            // Mapped: each symbol is a CHARACTER that may span several UTF-8
            // bytes (`Ġ`, `▁`); splitting mid-character would produce pieces
            // no merge or vocabulary entry could ever match.
            _ => match std::str::from_utf8(span) {
                Ok(s) => {
                    let mut i = 0usize;
                    for c in s.chars() {
                        let n = c.len_utf8();
                        parts.push((i, i + n));
                        i += n;
                    }
                }
                Err(_) => parts.extend((0..span.len()).map(|i| (i, i + 1))),
            },
        }
        // CACHED-RANK merge loop, the structure tiktoken uses.
        //
        // `ranks[i]` is the rank of merging piece `i` with piece `i+1`, or
        // MAX for "these cannot merge". The naive form this replaces did a
        // hash lookup for every adjacent pair on every iteration — O(n)
        // lookups per merge. Merging at `i` can only change the rank at `i`
        // and at `i-1`, so recomputing those two leaves the scan comparing
        // plain integers and costs O(n) lookups for the whole word.
        //
        // The algorithm is tiktoken's, reimplemented rather than copied:
        // tiktoken is MIT and R2 is AGPL-3.0, so vendoring its source would
        // put third-party code with its own copyright inside this crate.
        // The published method carries no such condition.
        let rank_at = |parts: &[(usize, usize)], i: usize| -> u32 {
            if i + 1 < parts.len() {
                self.merge_rank.get(&span[parts[i].0..parts[i + 1].1])
                    .copied().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        };
        let ranks = &mut scratch.ranks;
        ranks.clear();
        for i in 0..parts.len() { let r = rank_at(parts, i); ranks.push(r); }
        loop {
            // Lowest rank wins; ties keep the LEFTMOST, which `<` gives
            // because the scan runs left to right. Merge order is the whole
            // correctness property here — a different order is still
            // decodable but is not the segmentation the model was trained on.
            let mut best = u32::MAX;
            let mut at = usize::MAX;
            for (i, &r) in ranks.iter().enumerate() {
                if r < best { best = r; at = i; }
            }
            if best == u32::MAX { break; }
            parts[at].1 = parts[at + 1].1;
            parts.remove(at + 1);
            ranks.remove(at + 1);
            ranks[at] = rank_at(parts, at);
            if at > 0 { ranks[at - 1] = rank_at(parts, at - 1); }
        }
        for &(a, b) in parts.iter() {
            let p = &span[a..b];
            match self.vocab.get(p) {
                Some(&id) => out.push(id),
                // Not in the vocabulary: fall back to single bytes, then to
                // <unk>. SentencePiece vocabularies are NOT byte-complete,
                // so erroring here would reject text the model handles fine.
                // Only a vocabulary offering neither is a real error.
                None => for &byte in p {
                    if let Some(&id) = self.vocab.get(&vec![byte][..]) { out.push(id); continue; }
                    // SentencePiece byte_fallback: emit <0xNN> rather than
                    // <unk>, so the byte survives and decode can recover it.
                    if let Some(bf) = &self.byte_fallback { out.push(bf[byte as usize]); continue; }
                    match self.unk {
                        Some(u) => out.push(u),
                        None => return Err(format!(
                            "tokenizer: byte {:#04x} has no vocabulary entry and the                              vocabulary declares no unknown token", byte)),
                    }
                },
            }
        }
        Ok(())
    }

    /// Decode ids back to bytes. Unknown ids are skipped rather than
    /// panicking — a model can emit an out-of-range id and a serving loop
    /// must survive it.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        // Reverse of the byte_fallback table. Built per call rather than
        // stored: 256 entries is nothing against the decode itself, and a
        // second copy of the mapping is a second thing to keep in sync.
        let rev: Option<HashMap<u32, u8>> = self.byte_fallback.as_ref().map(|bf| {
            bf.iter().enumerate().map(|(b, &id)| (id, b as u8)).collect()
        });
        let mut out = Vec::new();
        for &id in ids {
            // A `<0xNN>` token stands for the BYTE NN, not for the six
            // characters that spell it. Emitting its literal text would put
            // "<0x41>" into the output where "A" belongs — silently, and
            // only for the rare characters byte_fallback exists to carry.
            if let Some(r) = &rev {
                if let Some(&b) = r.get(&id) { out.push(b); continue; }
            }
            if let Some(b) = self.id_to_token(id) { out.extend_from_slice(b); }
        }
        // Concatenate first, THEN unmap: a multi-byte symbol can be split
        // across two tokens, so per-token unmapping would corrupt it.
        self.from_vocab_space(&out)
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

/// Walk a JSON subtree collecting every `"type"` string, so a decoder
/// wrapped in a `Sequence` is still recognized.
fn collect_types(j: &Json, out: &mut String) {
    match j {
        Json::Obj(m) => {
            if let Some(Json::Str(t)) = m.get("type") { out.push_str(t); out.push(' '); }
            for v in m.values() { collect_types(v, out); }
        }
        Json::Arr(a) => for v in a { collect_types(v, out); },
        _ => {}
    }
}

#[cfg(test)]
mod encoding_tests {
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
}
