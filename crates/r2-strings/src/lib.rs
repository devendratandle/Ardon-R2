//! R2 Strings — domain crate for character-vector builtins (Phase R.6).
//!
//! Hosts: `toupper`, `tolower`, `substr`, `grep`, `grepl`, `gsub`, `sub`,
//! `regexpr`, `strsplit`, `paste`, `paste0`, `nchar`, `sprintf`, `trimws`.
//!
//! All builtins follow the locked pure pattern
//! `fn(&[EvalArg]) -> Result<RVal, R2Err>` — no engine dependency.
//!
//! **Regex (v0.1.0):** `grep`, `grepl`, `gsub`, `sub`, `regexpr` route
//! through `regex-lite` (pure-Rust POSIX-ERE subset) behind the default-on
//! `regex` feature. `fixed=TRUE` forces literal-substring mode. Patterns
//! that fail to compile fall back to literal substring silently.
//! Disabling the feature reverts the whole crate to literal semantics.
//! Limits (lookaround, backreferences, Unicode categories) documented
//! in `docs/KNOWN_LIMITATIONS.md`.
//!
//! **`sprintf` honest scoping (v0.1.x):** recognises `%d`, `%f`, `%s`,
//! `%e`, and `%%` only — no width/precision specifiers, no flags. Format
//! strings beyond that subset pass through unchanged. Tracked in
//! KNOWN_LIMITATIONS.

use r2_types::{Attrs, Character, ErrKind, EvalArg, Integer, Logical, R2Err, RVal};
use std::sync::Arc;

#[inline]
fn gv(args: &[EvalArg], i: usize) -> RVal {
    args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter()
        .find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

#[inline]
fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(Arc::from(s))], Attrs::default())
}

/// Mirrors `r2_engine::val_to_str` — flat scalar/vector renderer used by
/// the `paste*` and `sprintf` family.
fn val_to_str(v: &RVal) -> String {
    match v {
        RVal::Numeric(v, _) => v.iter().map(|x| match x {
            Some(n) => r2_types::fmt_num(*n),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Integer(v, _) => v.iter().map(|x| match x {
            Some(n) => format!("{}", n),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Character(v, _) => v.iter().map(|x| match x {
            Some(s) => s.to_string(),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Logical(v, _) => v.iter().map(|x| match x {
            Some(true) => "TRUE",
            Some(false) => "FALSE",
            None => "NA",
        }).collect::<Vec<_>>().join(" "),
        RVal::Null => "NULL".into(),
        _ => format!("<{}>", v.type_name()),
    }
}

#[inline]
fn type_err(msg: &str) -> R2Err {
    R2Err { msg: msg.into(), kind: ErrKind::Type }
}

#[inline]
fn runtime_err(msg: &str) -> R2Err {
    R2Err { msg: msg.into(), kind: ErrKind::Runtime }
}

// ─────────────────────────────────────────────────────────────────────
// Case + slice
// ─────────────────────────────────────────────────────────────────────

pub fn bi_toupper(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Character(v, _) => Ok(RVal::Character(
            v.iter().map(|x| x.as_ref().map(|s| Arc::from(s.to_uppercase().as_str()))).collect(),
            Attrs::default(),
        )),
        _ => Err(type_err("toupper needs character")),
    }
}

pub fn bi_tolower(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Character(v, _) => Ok(RVal::Character(
            v.iter().map(|x| x.as_ref().map(|s| Arc::from(s.to_lowercase().as_str()))).collect(),
            Attrs::default(),
        )),
        _ => Err(type_err("tolower needs character")),
    }
}

pub fn bi_substr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let s = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("substr needs character")),
    };
    let start = match &gv(a, 1) {
        RVal::Numeric(v, _) => v.first().and_then(|x| *x).unwrap_or(1.0) as usize,
        RVal::Integer(v, _) => v.first().and_then(|x| *x).unwrap_or(1) as usize,
        _ => 1,
    };
    let stop = match &gv(a, 2) {
        RVal::Numeric(v, _) => v.first().and_then(|x| *x).unwrap_or(1.0) as usize,
        RVal::Integer(v, _) => v.first().and_then(|x| *x).unwrap_or(1) as usize,
        _ => s.len(),
    };
    // R semantics: substr("abcdef", 2, 4) == "bcd" (positions 2..=4
    // inclusive, 3 chars). The pre-migration engine impl returned 4
    // chars ("bcde") — an off-by-one bug now corrected at the point
    // of migration.
    let take = stop.saturating_sub(start.saturating_sub(1));
    let result: String = s.chars()
        .skip(start.saturating_sub(1))
        .take(take)
        .collect();
    Ok(rstr(&result))
}

// ─────────────────────────────────────────────────────────────────────
// Pattern matching — Phase R.13.
//
// With `--features regex` (default ON) `grep`/`grepl`/`gsub`/`sub`/
// `regexpr` use `regex-lite` to compile the pattern. The POSIX-ERE
// subset matches R's default `extended=TRUE` mode: anchors, character
// classes, groups, repetitions, alternation, `\d`/`\w`/`\s`.
//
// Without the feature (or when the pattern fails to compile as a regex)
// they fall back to literal-substring matching. This preserves backward
// compatibility with callers that pass plain strings.
//
// A `fixed = TRUE` named arg forces literal mode regardless of feature.

/// Optional regex compilation. Returns `Some(compiled)` when feature is
/// on AND the pattern parses, `None` otherwise (caller falls back to
/// literal substring matching).
#[cfg(feature = "regex")]
fn compile_pattern(pattern: &str, fixed: bool) -> Option<regex_lite::Regex> {
    if fixed { return None; }
    regex_lite::Regex::new(pattern).ok()
}

#[cfg(not(feature = "regex"))]
fn compile_pattern(_pattern: &str, _fixed: bool) -> Option<()> { None }

fn fixed_arg(a: &[EvalArg]) -> bool {
    a.iter().find(|x| x.name.as_ref().map(|n| n.as_ref()) == Some("fixed"))
        .and_then(|x| match &x.value {
            RVal::Logical(v, _) => v.first().copied().flatten(),
            _ => None,
        }).unwrap_or(false)
}

pub fn bi_grep(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("grep needs pattern")),
    };
    let x = match &gv(a, 1) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("grep needs character vector")),
    };
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    let indices: Vec<Integer> = x.iter().enumerate().filter_map(|(i, s)| {
        s.as_ref().and_then(|s| {
            let hit = match &re {
                #[cfg(feature = "regex")]
                Some(re) => re.is_match(s),
                #[cfg(not(feature = "regex"))]
                Some(_) => unreachable!(),
                None => s.contains(pattern.as_str()),
            };
            if hit { Some(Some((i + 1) as i32)) } else { None }
        })
    }).collect();
    Ok(RVal::Integer(indices.into(), Attrs::default()))
}

pub fn bi_grepl(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = val_to_str(&gv(a, 0));
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    match &gv(a, 1) {
        RVal::Character(v, _) => {
            let result: Vec<Logical> = v.iter().map(|x| x.as_ref().map(|s| {
                match &re {
                    #[cfg(feature = "regex")]
                    Some(re) => re.is_match(s),
                    #[cfg(not(feature = "regex"))]
                    Some(_) => unreachable!(),
                    None => s.contains(pattern.as_str()),
                }
            })).collect();
            Ok(RVal::Logical(result.into(), Attrs::default()))
        }
        _ => Err(type_err("grepl() needs character input")),
    }
}

pub fn bi_gsub(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("gsub needs pattern")),
    };
    let replacement = match &gv(a, 1) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("gsub needs replacement")),
    };
    let x = match &gv(a, 2) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("gsub needs character")),
    };
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    let result: Vec<Character> = x.iter().map(|s| s.as_ref().map(|s| {
        let out = match &re {
            #[cfg(feature = "regex")]
            Some(re) => re.replace_all(s, replacement.as_str()).into_owned(),
            #[cfg(not(feature = "regex"))]
            Some(_) => unreachable!(),
            None => s.replace(&pattern, &replacement),
        };
        Arc::from(out.as_str())
    })).collect();
    Ok(RVal::Character(result, Attrs::default()))
}

pub fn bi_sub(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = val_to_str(&gv(a, 0));
    let replacement = val_to_str(&gv(a, 1));
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    match &gv(a, 2) {
        RVal::Character(v, _) => {
            let result: Vec<Character> = v.iter().map(|x| x.as_ref().map(|s| {
                let out = match &re {
                    #[cfg(feature = "regex")]
                    Some(re) => re.replace(s, replacement.as_str()).into_owned(),
                    #[cfg(not(feature = "regex"))]
                    Some(_) => unreachable!(),
                    None => {
                        if let Some(pos) = s.find(pattern.as_str()) {
                            format!("{}{}{}", &s[..pos], replacement, &s[pos + pattern.len()..])
                        } else {
                            s.to_string()
                        }
                    }
                };
                Arc::from(out.as_str())
            })).collect();
            Ok(RVal::Character(result, Attrs::default()))
        }
        _ => Err(type_err("sub() needs character input")),
    }
}

pub fn bi_regexpr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = val_to_str(&gv(a, 0));
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    match &gv(a, 1) {
        RVal::Character(v, _) => {
            let result: Vec<Integer> = v.iter().map(|x| x.as_ref().map(|s| {
                let pos = match &re {
                    #[cfg(feature = "regex")]
                    Some(re) => re.find(s).map(|m| m.start()),
                    #[cfg(not(feature = "regex"))]
                    Some(_) => unreachable!(),
                    None => s.find(pattern.as_str()),
                };
                pos.map(|p| (p + 1) as i32).unwrap_or(-1)
            })).collect();
            Ok(RVal::Integer(result.into(), Attrs::default()))
        }
        _ => Err(type_err("regexpr() needs character")),
    }
}

pub fn bi_strsplit(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let s = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("strsplit needs character")),
    };
    let split = match &gv(a, 1) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_else(|| " ".into()),
        _ => " ".into(),
    };
    let parts: Vec<Character> = s.split(&split).map(|p| Some(Arc::from(p))).collect();
    Ok(RVal::Character(parts, Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Construction + length + format
// ─────────────────────────────────────────────────────────────────────

pub fn bi_paste(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let sep = gn(a, "sep").map(|v| val_to_str(&v)).unwrap_or_else(|| " ".into());
    let s: Vec<String> = a.iter()
        .filter(|x| x.name.as_ref().map(|n| n.as_ref()) != Some("sep"))
        .map(|x| val_to_str(&x.value))
        .collect();
    Ok(rstr(&s.join(&sep)))
}

pub fn bi_paste0(a: &[EvalArg]) -> Result<RVal, R2Err> {
    Ok(rstr(&a.iter().map(|x| val_to_str(&x.value)).collect::<Vec<_>>().join("")))
}

pub fn bi_nchar(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Character(v, _) => Ok(RVal::Integer(
            v.iter().map(|x| x.as_ref().map(|s| s.len() as i32)).collect(),
            Attrs::default(),
        )),
        _ => Err(type_err("nchar needs character")),
    }
}

pub fn bi_trimws(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Character(v, _) => Ok(RVal::Character(
            v.iter().map(|x| x.as_ref().map(|s| Arc::from(s.trim()))).collect(),
            Attrs::default(),
        )),
        _ => Err(type_err("trimws needs character")),
    }
}

/// `sprintf(fmt, ...)` — subset implementation (see crate docstring).
/// Scalar f64 from an arg (first element), for numeric sprintf conversions.
fn sp_num(v: &RVal) -> f64 {
    match v {
        RVal::Numeric(n, _) => n.as_vec().first().copied().flatten().unwrap_or(0.0),
        RVal::Integer(n, _) => n.as_vec().first().copied().flatten().map(|i| i as f64).unwrap_or(0.0),
        RVal::Logical(l, _) => l.as_vec().first().copied().flatten().map(|b| if b { 1.0 } else { 0.0 }).unwrap_or(0.0),
        _ => 0.0,
    }
}
fn sp_str(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => val_to_str(v),
    }
}

/// C-style `sprintf` supporting `%[flags][width][.precision]conv` —
/// flags `-`/`0`/`+`/space, width, precision, and conversions
/// d/i/f/e/E/g/x/X/s/%. (Previously only bare `%d/%f/%s/%e` worked, so
/// `%-12s`, `%.3f`, `%05d` were emitted literally — see test_real_program.)
pub fn bi_sprintf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let fmt = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(runtime_err("sprintf needs format string")),
    };
    let ch: Vec<char> = fmt.chars().collect();
    let mut out = String::new();
    let mut arg_idx = 1usize;
    let mut i = 0usize;
    while i < ch.len() {
        if ch[i] != '%' { out.push(ch[i]); i += 1; continue; }
        let start = i;
        i += 1;
        if i < ch.len() && ch[i] == '%' { out.push('%'); i += 1; continue; }
        // flags
        let (mut left, mut zero, mut plus, mut space) = (false, false, false, false);
        while i < ch.len() {
            match ch[i] { '-' => left = true, '0' => zero = true, '+' => plus = true, ' ' => space = true, '#' => {}, _ => break }
            i += 1;
        }
        // width
        let (mut width, mut has_w) = (0usize, false);
        while i < ch.len() && ch[i].is_ascii_digit() { has_w = true; width = width * 10 + (ch[i] as usize - '0' as usize); i += 1; }
        // precision
        let mut prec: Option<usize> = None;
        if i < ch.len() && ch[i] == '.' {
            i += 1; let mut p = 0usize;
            while i < ch.len() && ch[i].is_ascii_digit() { p = p * 10 + (ch[i] as usize - '0' as usize); i += 1; }
            prec = Some(p);
        }
        if i >= ch.len() { out.extend(ch[start..].iter()); break; }
        let conv = ch[i]; i += 1;
        let argv = gv(a, arg_idx);
        let mut body = match conv {
            'd' | 'i' => {
                arg_idx += 1;
                let n = sp_num(&argv).round() as i64;
                let mut s = n.unsigned_abs().to_string();
                if let Some(p) = prec { while s.len() < p { s.insert(0, '0'); } }
                let sign = if n < 0 { "-" } else if plus { "+" } else if space { " " } else { "" };
                format!("{}{}", sign, s)
            }
            'f' => {
                arg_idx += 1;
                let n = sp_num(&argv);
                let s = format!("{:.*}", prec.unwrap_or(6), n.abs());
                let sign = if n.is_sign_negative() && n != 0.0 { "-" } else if plus { "+" } else if space { " " } else { "" };
                format!("{}{}", sign, s)
            }
            'e' | 'E' => { arg_idx += 1; let s = format!("{:.*e}", prec.unwrap_or(6), sp_num(&argv)); if conv == 'E' { s.to_uppercase() } else { s } }
            'g' | 'G' => { arg_idx += 1; format!("{}", sp_num(&argv)) }
            'x' => { arg_idx += 1; format!("{:x}", sp_num(&argv) as i64) }
            'X' => { arg_idx += 1; format!("{:X}", sp_num(&argv) as i64) }
            's' => { arg_idx += 1; let s = sp_str(&argv); match prec { Some(p) => s.chars().take(p).collect(), None => s } }
            _ => { out.extend(ch[start..i].iter()); continue; } // unknown spec: literal
        };
        // width padding
        let blen = body.chars().count();
        if has_w && blen < width {
            let pad = width - blen;
            if left {
                body.push_str(&" ".repeat(pad));
            } else if zero && matches!(conv, 'd' | 'i' | 'f' | 'e' | 'E' | 'g' | 'G' | 'x' | 'X') {
                let (sign, rest) = if body.starts_with(['-', '+', ' ']) { body.split_at(1) } else { ("", body.as_str()) };
                body = format!("{}{}{}", sign, "0".repeat(pad), rest);
            } else {
                body = format!("{}{}", " ".repeat(pad), body);
            }
        }
        out.push_str(&body);
    }
    Ok(rstr(&out))
}

// ─────────────────────────────────────────────────────────────────────
// Builtins registry (Phase R.6).
// ─────────────────────────────────────────────────────────────────────

pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("toupper",  bi_toupper),
        ("tolower",  bi_tolower),
        ("substr",   bi_substr),
        ("grep",     bi_grep),
        ("grepl",    bi_grepl),
        ("gsub",     bi_gsub),
        ("sub",      bi_sub),
        ("regexpr",  bi_regexpr),
        ("strsplit", bi_strsplit),
        ("paste",    bi_paste),
        ("paste0",   bi_paste0),
        ("nchar",    bi_nchar),
        ("trimws",   bi_trimws),
        ("sprintf",  bi_sprintf),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }
    fn chs(v: &[&str]) -> RVal {
        RVal::Character(v.iter().map(|s| Some(Arc::from(*s))).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }
    fn evarg_named(name: &str, v: RVal) -> EvalArg {
        EvalArg { name: Some(Arc::from(name)), value: v }
    }

    #[test]
    fn toupper_basic() {
        let r = bi_toupper(&[evarg(ch("hello"))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("HELLO")), _ => panic!() }
    }

    #[test]
    fn tolower_basic() {
        let r = bi_tolower(&[evarg(ch("HELLO"))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("hello")), _ => panic!() }
    }

    #[test]
    fn substr_extracts_range() {
        // substr("abcdef", 2, 4) == "bcd"
        let r = bi_substr(&[
            evarg(ch("abcdef")),
            evarg(RVal::Integer(vec![Some(2)].into(), Attrs::default())),
            evarg(RVal::Integer(vec![Some(4)].into(), Attrs::default())),
        ]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("bcd")), _ => panic!() }
    }

    #[test]
    #[cfg(feature = "regex")]
    #[test]
    fn grep_regex_anchors_match() {
        // `^foo` should match only strings beginning with "foo", not "barfoo".
        let r = bi_grep(&[evarg(ch("^foo")), evarg(chs(&["foobar", "barfoo", "foo"]))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1, 3]);
            }
            _ => panic!(),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn gsub_regex_character_class() {
        // `[aeiou]` replaces all vowels with `_`.
        let r = bi_gsub(&[evarg(ch("[aeiou]")), evarg(ch("_")), evarg(ch("regular"))]).unwrap();
        match r {
            RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("r_g_l_r")),
            _ => panic!(),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn fixed_arg_forces_literal_match() {
        // With fixed=TRUE, `.` matches a literal dot, not "any char".
        let r = bi_grep(&[
            evarg(ch(".")), evarg(chs(&["abc", "a.c", "xyz"])),
            EvalArg { name: Some(Arc::from("fixed")), value: RVal::Logical(vec![Some(true)].into(), Attrs::default()) },
        ]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![2]);  // only "a.c" contains a literal dot
            }
            _ => panic!(),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn regexpr_returns_first_match_position() {
        // `\d+` finds "123" at 1-based position 5 in "abcd123ef".
        let r = bi_regexpr(&[evarg(ch(r"\d+")), evarg(ch("abcd123ef"))]).unwrap();
        match r {
            RVal::Integer(v, _) => assert_eq!(v[0], Some(5)),
            _ => panic!(),
        }
    }

    #[test]
    fn grep_returns_matching_indices() {
        let r = bi_grep(&[evarg(ch("o")), evarg(chs(&["foo", "bar", "boop"]))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![1, 3]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn grepl_returns_logical_vec() {
        let r = bi_grepl(&[evarg(ch("o")), evarg(chs(&["foo", "bar", "boop"]))]).unwrap();
        match r {
            RVal::Logical(v, _) => {
                let got: Vec<bool> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![true, false, true]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn gsub_replaces_all() {
        let r = bi_gsub(&[evarg(ch("o")), evarg(ch("0")), evarg(ch("foobar"))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("f00bar")), _ => panic!() }
    }

    #[test]
    fn sub_replaces_first() {
        let r = bi_sub(&[evarg(ch("o")), evarg(ch("0")), evarg(ch("foobar"))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("f0obar")), _ => panic!() }
    }

    #[test]
    fn paste_joins_with_sep() {
        let r = bi_paste(&[
            evarg(ch("a")),
            evarg(ch("b")),
            evarg_named("sep", ch("-")),
        ]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("a-b")), _ => panic!() }
    }

    #[test]
    fn paste0_concatenates() {
        let r = bi_paste0(&[evarg(ch("a")), evarg(ch("b")), evarg(ch("c"))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("abc")), _ => panic!() }
    }

    #[test]
    fn nchar_counts_bytes() {
        let r = bi_nchar(&[evarg(chs(&["abc", "12345"]))]).unwrap();
        match r {
            RVal::Integer(v, _) => {
                let got: Vec<i32> = v.iter().filter_map(|x| *x).collect();
                assert_eq!(got, vec![3, 5]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn trimws_strips_whitespace() {
        let r = bi_trimws(&[evarg(ch("   hi  "))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("hi")), _ => panic!() }
    }

    #[test]
    fn strsplit_splits_on_separator() {
        let r = bi_strsplit(&[evarg(ch("a,b,c")), evarg(ch(","))]).unwrap();
        match r {
            RVal::Character(v, _) => {
                let got: Vec<&str> = v.iter().filter_map(|x| x.as_deref()).collect();
                assert_eq!(got, vec!["a", "b", "c"]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn sprintf_substitutes_args() {
        let r = bi_sprintf(&[evarg(ch("hi %s, %d!")), evarg(ch("world")), evarg(RVal::Integer(vec![Some(7)].into(), Attrs::default()))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("hi world, 7!")), _ => panic!() }
    }
}
