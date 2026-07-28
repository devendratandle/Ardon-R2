//! A minimal JSON reader — just enough for model metadata.
//!
//! Weight files and tokenizers are described in JSON (safetensors headers,
//! `tokenizer.json`), so reading a real checkpoint needs a parser. Ardon-R2
//! takes no non-Rust dependencies and currently pulls in no serde, so this
//! is a small, self-contained, spec-correct reader rather than a new
//! dependency: objects, arrays, strings (with escapes and surrogate
//! pairs), numbers, booleans and null.
//!
//! Scope is deliberate. It parses into an owned tree, which is right for
//! metadata measured in kilobytes — never for the tensor payload, which is
//! read as raw bytes and memory-mapped.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// All JSON numbers are f64; integer accessors below check exactness.
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Ordered so iteration is deterministic (helpful in error messages
    /// and tests).
    Obj(BTreeMap<String, Json>),
}

impl Json {
    /// Parse a complete JSON document. Trailing whitespace is allowed;
    /// trailing garbage is an error.
    pub fn parse(src: &str) -> Result<Json, String> {
        let b = src.as_bytes();
        let mut p = Parser { b, i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != b.len() {
            return Err(format!("json: trailing data at byte {}", p.i));
        }
        Ok(v)
    }

    // ── Typed accessors: `None` when absent OR the wrong type, which is
    //    what callers want (a mis-typed field is as unusable as a missing
    //    one, and both should produce the same clear error upstream).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self { Json::Obj(m) => m.get(key), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Json::Str(s) => Some(s), _ => None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Json::Num(n) => Some(*n), _ => None }
    }
    /// Exact non-negative integer, or `None` (rejects 1.5 and -1).
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 && *n <= usize::MAX as f64 =>
                Some(*n as usize),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Json::Bool(b) => Some(*b), _ => None }
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self { Json::Arr(a) => Some(a), _ => None }
    }
    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self { Json::Obj(m) => Some(m), _ => None }
    }
    /// Array of exact non-negative integers — the common shape case.
    pub fn as_usize_vec(&self) -> Option<Vec<usize>> {
        self.as_arr()?.iter().map(|v| v.as_usize()).collect()
    }
}

struct Parser<'a> { b: &'a [u8], i: usize }

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len()
            && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; }
    }

    fn peek(&self) -> Result<u8, String> {
        self.b.get(self.i).copied().ok_or_else(|| "json: unexpected end of input".to_string())
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek()? != c {
            return Err(format!("json: expected '{}' at byte {}, found '{}'",
                               c as char, self.i, self.b[self.i] as char));
        }
        self.i += 1;
        Ok(())
    }

    fn lit(&mut self, word: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(())
        } else {
            Err(format!("json: invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => { self.lit("true")?;  Ok(Json::Bool(true)) }
            b'f' => { self.lit("false")?; Ok(Json::Bool(false)) }
            b'n' => { self.lit("null")?;  Ok(Json::Null) }
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut m = BTreeMap::new();
        self.ws();
        if self.peek()? == b'}' { self.i += 1; return Ok(Json::Obj(m)); }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            m.insert(k, v);
            self.ws();
            match self.peek()? {
                b',' => { self.i += 1; }
                b'}' => { self.i += 1; break; }
                c => return Err(format!("json: expected ',' or '}}' at byte {}, found '{}'",
                                        self.i, c as char)),
            }
        }
        Ok(Json::Obj(m))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.peek()? == b']' { self.i += 1; return Ok(Json::Arr(a)); }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => { self.i += 1; }
                b']' => { self.i += 1; break; }
                c => return Err(format!("json: expected ',' or ']' at byte {}, found '{}'",
                                        self.i, c as char)),
            }
        }
        Ok(Json::Arr(a))
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut s = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"'  => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/'  => s.push('/'),
                        b'b'  => s.push('\u{8}'),
                        b'f'  => s.push('\u{c}'),
                        b'n'  => s.push('\n'),
                        b'r'  => s.push('\r'),
                        b't'  => s.push('\t'),
                        b'u'  => {
                            let hi = self.hex4()?;
                            // Surrogate pair: astral characters arrive as
                            // \uD8xx\uDCxx and must be recombined, or the
                            // text silently corrupts (emoji, CJK extensions).
                            let ch = if (0xD800..0xDC00).contains(&hi) {
                                if self.peek()? != b'\\' {
                                    return Err("json: lone high surrogate".into());
                                }
                                self.i += 1;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("json: invalid low surrogate".into());
                                }
                                let c = 0x10000
                                    + (((hi - 0xD800) as u32) << 10)
                                    + (lo - 0xDC00) as u32;
                                char::from_u32(c).ok_or("json: invalid code point")?
                            } else {
                                char::from_u32(hi as u32).ok_or("json: invalid code point")?
                            };
                            s.push(ch);
                        }
                        other => return Err(format!("json: bad escape '\\{}'", other as char)),
                    }
                }
                // Multi-byte UTF-8 passes through untouched.
                _ => {
                    let start = self.i - 1;
                    let len = utf8_len(c);
                    if start + len > self.b.len() {
                        return Err("json: truncated UTF-8 in string".into());
                    }
                    let piece = std::str::from_utf8(&self.b[start..start + len])
                        .map_err(|_| "json: invalid UTF-8 in string".to_string())?;
                    s.push_str(piece);
                    self.i = start + len;
                }
            }
        }
        Ok(s)
    }

    fn hex4(&mut self) -> Result<u16, String> {
        if self.i + 4 > self.b.len() { return Err("json: truncated \\u escape".into()); }
        let h = std::str::from_utf8(&self.b[self.i..self.i + 4])
            .map_err(|_| "json: bad \\u escape".to_string())?;
        let v = u16::from_str_radix(h, 16).map_err(|_| "json: bad \\u escape".to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek()? == b'-' { self.i += 1; }
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        { self.i += 1; }
        let s = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "json: bad number".to_string())?;
        s.parse::<f64>().map(Json::Num)
            .map_err(|_| format!("json: bad number '{}' at byte {}", s, start))
    }
}

#[inline]
fn utf8_len(first: u8) -> usize {
    if first < 0x80 { 1 } else if first >> 5 == 0b110 { 2 }
    else if first >> 4 == 0b1110 { 3 } else { 4 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_safetensors_style_header() {
        let src = r#"{"__metadata__":{"format":"pt"},
            "model.layers.0.self_attn.q_proj.weight":
              {"dtype":"F16","shape":[4096,4096],"data_offsets":[0,33554432]}}"#;
        let j = Json::parse(src).unwrap();
        let t = j.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(t.get("dtype").unwrap().as_str(), Some("F16"));
        assert_eq!(t.get("shape").unwrap().as_usize_vec(), Some(vec![4096, 4096]));
        assert_eq!(t.get("data_offsets").unwrap().as_usize_vec(), Some(vec![0, 33554432]));
        assert_eq!(j.get("__metadata__").unwrap().get("format").unwrap().as_str(), Some("pt"));
    }

    #[test]
    fn scalars_arrays_and_nesting() {
        let j = Json::parse(r#"{"a":1,"b":-2.5e3,"c":true,"d":null,"e":[1,2,[3]],"f":{}}"#).unwrap();
        assert_eq!(j.get("a").unwrap().as_usize(), Some(1));
        assert_eq!(j.get("b").unwrap().as_f64(), Some(-2500.0));
        assert_eq!(j.get("c").unwrap().as_bool(), Some(true));
        assert_eq!(j.get("d").unwrap(), &Json::Null);
        assert_eq!(j.get("e").unwrap().as_arr().unwrap().len(), 3);
        assert!(j.get("f").unwrap().as_obj().unwrap().is_empty());
        assert_eq!(Json::parse("[]").unwrap(), Json::Arr(vec![]));
    }

    #[test]
    fn string_escapes_and_unicode() {
        // Escapes, a tab, and raw multi-byte UTF-8 passing through.
        let j = Json::parse(r#"{"s":"a\"b\\c\n\táé"}"#).unwrap();
        assert_eq!(j.get("s").unwrap().as_str(), Some("a\"b\\c\n\táé"));
        // \u escape for a BMP character.
        let j2 = Json::parse(r#"{"s":"éA"}"#).unwrap();
        assert_eq!(j2.get("s").unwrap().as_str(), Some("éA"));
    }

    #[test]
    fn surrogate_pairs_recombine() {
        // Astral plane: "😀" is 😀 — must not corrupt.
        let j = Json::parse(r#"{"e":"😀"}"#).unwrap();
        assert_eq!(j.get("e").unwrap().as_str(), Some("😀"));
        assert!(Json::parse(r#"{"e":"\uD83D"}"#).is_err(), "lone surrogate must fail");
    }

    #[test]
    fn wrong_type_reads_as_none_not_a_panic() {
        let j = Json::parse(r#"{"n":"7","a":{"x":1}}"#).unwrap();
        assert_eq!(j.get("n").unwrap().as_usize(), None); // string, not number
        assert_eq!(j.get("a").unwrap().as_arr(), None);   // object, not array
        assert_eq!(j.get("missing"), None);
    }

    #[test]
    fn non_integer_and_negative_rejected_as_usize() {
        let j = Json::parse(r#"{"a":1.5,"b":-1,"c":3}"#).unwrap();
        assert_eq!(j.get("a").unwrap().as_usize(), None);
        assert_eq!(j.get("b").unwrap().as_usize(), None);
        assert_eq!(j.get("c").unwrap().as_usize(), Some(3));
    }

    #[test]
    fn malformed_input_errors_rather_than_panicking() {
        for bad in ["{", "}", "{\"a\"}", "{\"a\":}", "[1,]", "tru", "{\"a\":1}x", "\"unterminated"] {
            assert!(Json::parse(bad).is_err(), "should reject: {bad}");
        }
    }
}
