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
mod tests;

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
mod encoding_tests;
