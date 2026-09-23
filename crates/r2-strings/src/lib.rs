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
//! **`sprintf`:** `%d %i %f %e %E %g %G %x %X %s %%` with flags (`-+ 0#`),
//! width and precision, vectorised over the format and every argument as
//! R is; compared with R in `tests/differential/cases/formatting.R`. Not
//! yet: `*` widths, `%o`, `%a`, and `%5$s`-style argument positions —
//! such specifiers pass through literally.

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
    // VECTORIZED over the input vector (was first-element-only, so
    // substr(c("apple","Banana"),1,1) wrongly returned just "a").
    let start = match &gv(a, 1) {
        RVal::Numeric(v, _) => v.first().and_then(|x| *x).unwrap_or(1.0) as usize,
        RVal::Integer(v, _) => v.first().and_then(|x| *x).unwrap_or(1) as usize,
        _ => 1,
    };
    let stop_opt: Option<usize> = match &gv(a, 2) {
        RVal::Numeric(v, _) => v.first().and_then(|x| *x).map(|n| n as usize),
        RVal::Integer(v, _) => v.first().and_then(|x| *x).map(|n| n as usize),
        _ => None,
    };
    let v = match &gv(a, 0) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("substr needs character")),
    };
    // R semantics: substr("abcdef", 2, 4) == "bcd" (positions 2..=4 inclusive).
    let out: Vec<Character> = v.iter().map(|x| x.as_ref().map(|s| {
        let stop = stop_opt.unwrap_or_else(|| s.chars().count());
        let take = stop.saturating_sub(start.saturating_sub(1));
        Arc::from(s.chars().skip(start.saturating_sub(1)).take(take).collect::<String>().as_str())
    })).collect();
    Ok(RVal::Character(out, Attrs::default()))
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

/// All literal (non-regex) matches of `pat` in `s` as (byte-start, byte-len).
fn literal_all(pat: &str, s: &str) -> Vec<(usize, usize)> {
    if pat.is_empty() { return Vec::new(); }
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(p) = s[start..].find(pat) {
        let abs = start + p;
        out.push((abs, pat.len()));
        start = abs + pat.len();
    }
    out
}

/// `gregexpr(pattern, text)` — ALL match positions per element. Returns a
/// list; each element is an integer vector of 1-based start positions with a
/// `match.length` attribute (or -1 when there is no match). Pairs with
/// `regmatches()` to extract the matched substrings.
pub fn bi_gregexpr(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pattern = match &gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => return Err(type_err("gregexpr() needs a pattern")),
    };
    let fixed = fixed_arg(a);
    let re = compile_pattern(&pattern, fixed);
    let texts: Vec<Character> = match &gv(a, 1) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("gregexpr() needs character")),
    };
    let items: Vec<(Option<Arc<str>>, RVal)> = texts.iter().map(|t| {
        let matches: Vec<(usize, usize)> = match t {
            Some(s) => {
                #[cfg(feature = "regex")]
                { match &re { Some(re) => re.find_iter(s).map(|m| (m.start(), m.end() - m.start())).collect(), None => literal_all(&pattern, s) } }
                #[cfg(not(feature = "regex"))]
                { let _ = &re; literal_all(&pattern, s) }
            }
            None => Vec::new(),
        };
        let (pos, len): (Vec<Integer>, Vec<Integer>) = if matches.is_empty() {
            (vec![Some(-1)], vec![Some(-1)])
        } else {
            (matches.iter().map(|(st, _)| Some((*st + 1) as i32)).collect(),
             matches.iter().map(|(_, l)| Some(*l as i32)).collect())
        };
        let mut at = Attrs::default();
        at.custom.insert(Arc::from("match.length"), RVal::Integer(len.into(), Attrs::default()));
        (None, RVal::Integer(pos.into(), at))
    }).collect();
    Ok(RVal::List(items))
}

/// `regmatches(text, m)` — extract the substrings matched by a `gregexpr`
/// result (a list of integer-position vectors with `match.length`).
pub fn bi_regmatches(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let texts: Vec<Character> = match &gv(a, 0) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("regmatches() needs character text")),
    };
    match &gv(a, 1) {
        RVal::List(mlist) => {
            let out: Vec<(Option<Arc<str>>, RVal)> = texts.iter().enumerate().map(|(i, t)| {
                let s = t.as_ref().map(|x| x.to_string()).unwrap_or_default();
                let parts: Vec<Character> = match mlist.get(i).map(|(_, v)| v) {
                    Some(RVal::Integer(pos, at)) => {
                        let lens: Vec<Integer> = match at.custom.get("match.length") {
                            Some(RVal::Integer(l, _)) => l.as_vec().clone(),
                            _ => Vec::new(),
                        };
                        pos.as_vec().iter().enumerate().filter_map(|(j, p)| {
                            let p = (*p)?;
                            if p < 1 { return None; }
                            let st = (p - 1) as usize;
                            let len = lens.get(j).and_then(|x| *x).unwrap_or(0).max(0) as usize;
                            Some(Some(Arc::from(s.get(st..st + len).unwrap_or(""))))
                        }).collect()
                    }
                    _ => Vec::new(),
                };
                (None, RVal::Character(parts, Attrs::default()))
            }).collect();
            Ok(RVal::List(out))
        }
        _ => Err(type_err("regmatches(): second argument must be a gregexpr result")),
    }
}

pub fn bi_strsplit(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // R's strsplit returns a LIST (one character vector per input string).
    // An empty separator splits into individual characters.
    let strings: Vec<Character> = match &gv(a, 0) {
        RVal::Character(v, _) => v.clone(),
        _ => return Err(type_err("strsplit needs character")),
    };
    let split = match &gv(a, 1) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_else(|| " ".into()),
        _ => " ".into(),
    };
    let items: Vec<(Option<Arc<str>>, RVal)> = strings.iter().map(|s| {
        let parts: Vec<Character> = match s {
            Some(st) if split.is_empty() => st.chars().map(|c| Some(Arc::from(c.to_string().as_str()))).collect(),
            Some(st) => st.split(split.as_str()).map(|p| Some(Arc::from(p))).collect(),
            None => vec![None],
        };
        (None, RVal::Character(parts, Attrs::default()))
    }).collect();
    Ok(RVal::List(items))
}

// ─────────────────────────────────────────────────────────────────────
// Construction + length + format
// ─────────────────────────────────────────────────────────────────────

/// Element-wise string view of a value, for `paste`'s vectorized model.
fn paste_elems(v: &RVal) -> Vec<String> {
    match v {
        RVal::Character(c, _) => c.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Numeric(n, _)   => n.as_vec().iter().map(|x| x.map(r2_types::fmt_num).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Integer(n, _)   => n.as_vec().iter().map(|x| x.map(|i| i.to_string()).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Logical(l, _)   => l.as_vec().iter().map(|x| x.map(|b| if b { "TRUE" } else { "FALSE" }.to_string()).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Factor(f)       => f.codes.iter().map(|c| c.and_then(|i| f.levels.get(i as usize).map(|s| s.to_string())).unwrap_or_else(|| "NA".into())).collect(),
        RVal::Null            => Vec::new(),
        other                 => vec![val_to_str(other)],
    }
}

/// `paste(..., sep=" ", collapse=NULL)` — R's VECTORIZED concatenation:
/// arguments are recycled to the longest length and joined element-wise by
/// `sep`; `collapse` (when not NULL) then joins the result into one string.
fn paste_impl(a: &[EvalArg], default_sep: &str) -> Result<RVal, R2Err> {
    let sep = gn(a, "sep").map(|v| val_to_str(&v)).unwrap_or_else(|| default_sep.to_string());
    let collapse = gn(a, "collapse").filter(|v| !matches!(v, RVal::Null));
    let cols: Vec<Vec<String>> = a.iter()
        .filter(|x| !matches!(x.name.as_deref(), Some("sep") | Some("collapse")))
        .map(|x| paste_elems(&x.value))
        .filter(|v| !v.is_empty())
        .collect();
    if cols.is_empty() {
        return Ok(match collapse { Some(_) => rstr(""), None => RVal::Character(Vec::new(), Attrs::default()) });
    }
    let maxlen = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let out: Vec<String> = (0..maxlen).map(|i| {
        cols.iter().map(|c| c[i % c.len()].as_str()).collect::<Vec<_>>().join(&sep)
    }).collect();
    match collapse {
        Some(cv) => Ok(rstr(&out.join(&val_to_str(&cv)))),
        None => Ok(RVal::Character(out.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default())),
    }
}

pub fn bi_paste(a: &[EvalArg]) -> Result<RVal, R2Err> { paste_impl(a, " ") }
pub fn bi_paste0(a: &[EvalArg]) -> Result<RVal, R2Err> { paste_impl(a, "") }

/// Flatten any value to element strings (lists/data.frames recurse).
fn to_string_elems(v: &RVal) -> Vec<String> {
    match v {
        RVal::List(items)   => items.iter().flat_map(|(_, x)| to_string_elems(x)).collect(),
        RVal::DataFrame(df) => df.columns.iter().flat_map(|(_, c)| to_string_elems(c)).collect(),
        other               => paste_elems(other),
    }
}

/// `toString(x, sep=", ")` — collapse x to one comma-separated string;
/// flattens vectors, lists, and data.frame rows (e.g. `toString(df[i,])`).
pub fn bi_to_string(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let sep = gn(a, "sep").map(|v| val_to_str(&v)).unwrap_or_else(|| ", ".to_string());
    Ok(rstr(&to_string_elems(&gv(a, 0)).join(&sep)))
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

/// One element of a `sprintf` argument.
enum SpVal { Num(Option<f64>), Lgl(Option<bool>), Str(Option<String>) }

fn sp_len(v: &RVal) -> usize {
    match v {
        RVal::Numeric(n, _) => n.as_vec().len(),
        RVal::Integer(n, _) => n.as_vec().len(),
        RVal::Logical(l, _) => l.as_vec().len(),
        RVal::Character(c, _) => c.len(),
        RVal::Null => 0,
        _ => 1,
    }
}

/// Element `i` of an argument, recycled — R's `sprintf` is vectorised over
/// the format and every argument, to the longest length.
fn sp_elem(v: &RVal, i: usize) -> SpVal {
    match v {
        RVal::Numeric(n, _) => { let n = n.as_vec(); SpVal::Num(n[i % n.len()]) }
        RVal::Integer(n, _) => { let n = n.as_vec(); SpVal::Num(n[i % n.len()].map(|x| x as f64)) }
        RVal::Logical(l, _) => { let l = l.as_vec(); SpVal::Lgl(l[i % l.len()]) }
        RVal::Character(c, _) => SpVal::Str(c[i % c.len()].as_ref().map(|s| s.to_string())),
        other => SpVal::Str(Some(val_to_str(other))),
    }
}

/// C's `%e` body for a non-negative finite value: the mantissa with `prec`
/// digits and an exponent of at least two digits with its sign
/// (`1.500000e+03`) — Rust's `{:e}` writes `1.5e3`.
fn c_exp(v: f64, prec: usize, upper: bool) -> String {
    let s = format!("{:.*e}", prec, v);
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = exp.parse().unwrap_or(0);
    let out = format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    if upper { out.to_uppercase() } else { out }
}

/// C's `%g` body for a non-negative finite value: `prec` significant
/// digits (6 by default, 0 meaning 1), `%e` style when the exponent is
/// below -4 or at least the precision, `%f` style otherwise, trailing
/// zeros removed unless `#`.
fn c_g(v: f64, prec: Option<usize>, alt: bool, upper: bool) -> String {
    let p = match prec { None => 6, Some(0) => 1, Some(p) => p };
    let x = if v == 0.0 { 0 } else {
        let s = format!("{:.*e}", p - 1, v);
        s.split_once('e').and_then(|(_, e)| e.parse::<i32>().ok()).unwrap_or(0)
    };
    let (mut body, exp_part) = if x < -4 || x >= p as i32 {
        let s = c_exp(v, p - 1, upper);
        let at = s.find(['e', 'E']).unwrap_or(s.len());
        (s[..at].to_string(), s[at..].to_string())
    } else {
        (format!("{:.*}", (p as i32 - 1 - x).max(0) as usize, v), String::new())
    };
    if !alt && body.contains('.') {
        while body.ends_with('0') { body.pop(); }
        if body.ends_with('.') { body.pop(); }
    }
    body + &exp_part
}

/// A double as R's `as.character` writes it: 15 significant digits, in
/// fixed or scientific notation, whichever is narrower (fixed on a tie) —
/// `100000` is `"1e+05"`, `123456` is `"123456"`, `0.1 + 0.2` is `"0.3"`.
fn num_as_character(n: f64) -> String {
    if n.is_nan() { return "NaN".into(); }
    if n.is_infinite() { return if n > 0.0 { "Inf".into() } else { "-Inf".into() }; }
    if n == 0.0 { return "0".into(); }
    let neg = if n < 0.0 { "-" } else { "" };
    let v = n.abs();
    // the significant digits R keeps: 15, trailing zeros dropped
    let s = format!("{:.14e}", v);
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.replace('.', "").trim_end_matches('0').to_string();
    let nsig = digits.len().max(1) as i32;
    let sci_w = nsig + if nsig > 1 { 1 } else { 0 } + if e.abs() >= 100 { 5 } else { 4 };
    let fix_w = if e >= 0 { e + 1 + if nsig > e + 1 { nsig - e } else { 0 } } else { 1 + (-e) + nsig };
    if fix_w <= sci_w {
        let dec = if e >= 0 { (nsig - e - 1).max(0) } else { nsig - e - 1 } as usize;
        format!("{neg}{:.*}", dec, v)
    } else {
        let m = if nsig > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{neg}{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

/// One output string: `fmt` with its conversions filled from element `k`
/// of each argument.
fn sprintf_one(fmt: &str, args: &[RVal], k: usize) -> Result<String, R2Err> {
    let ch: Vec<char> = fmt.chars().collect();
    let mut out = String::new();
    let mut arg_idx = 0usize;
    let mut i = 0usize;
    while i < ch.len() {
        if ch[i] != '%' { out.push(ch[i]); i += 1; continue; }
        let start = i;
        i += 1;
        if i < ch.len() && ch[i] == '%' { out.push('%'); i += 1; continue; }
        let (mut left, mut zero, mut plus, mut space, mut alt) = (false, false, false, false, false);
        while i < ch.len() {
            match ch[i] { '-' => left = true, '0' => zero = true, '+' => plus = true, ' ' => space = true, '#' => alt = true, _ => break }
            i += 1;
        }
        let (mut width, mut has_w) = (0usize, false);
        while i < ch.len() && ch[i].is_ascii_digit() { has_w = true; width = width * 10 + (ch[i] as usize - '0' as usize); i += 1; }
        let mut prec: Option<usize> = None;
        if i < ch.len() && ch[i] == '.' {
            i += 1; let mut p = 0usize;
            while i < ch.len() && ch[i].is_ascii_digit() { p = p * 10 + (ch[i] as usize - '0' as usize); i += 1; }
            prec = Some(p);
        }
        if i >= ch.len() { out.extend(ch[start..].iter()); break; }
        let conv = ch[i]; i += 1;
        if !matches!(conv, 'd' | 'i' | 'f' | 'e' | 'E' | 'g' | 'G' | 'x' | 'X' | 's') {
            out.extend(ch[start..i].iter()); continue;             // unknown spec: literal
        }
        let arg = args.get(arg_idx).ok_or_else(|| runtime_err("too few arguments"))?;
        arg_idx += 1;
        let el = sp_elem(arg, k);
        let sign = |neg: bool| if neg { "-" } else if plus { "+" } else if space { " " } else { "" };
        // NA and the non-finite values print as R prints them, for every
        // numeric conversion, and pad with spaces, never zeros
        let mut numeric_special = false;
        let mut body = match (conv, el) {
            ('s', SpVal::Str(s)) => { let s = s.unwrap_or_else(|| "NA".into()); match prec { Some(p) => s.chars().take(p).collect(), None => s } }
            ('s', SpVal::Lgl(b)) => match b { Some(true) => "TRUE".into(), Some(false) => "FALSE".into(), None => "NA".into() },
            // as.character(): 15 significant digits
            ('s', SpVal::Num(n)) => match n { Some(n) => { let s = num_as_character(n); match prec { Some(p) => s.chars().take(p).collect(), None => s } } None => "NA".into() },
            (_, SpVal::Str(_)) => return Err(runtime_err(&format!("invalid format '%{conv}'; use format %s for character objects"))),
            (c, SpVal::Lgl(b)) if matches!(c, 'd' | 'i') => match b { Some(b) => format!("{}{}", sign(false), b as i32), None => { numeric_special = true; "NA".into() } },
            (_, SpVal::Lgl(_)) => return Err(runtime_err(&format!("invalid format '%{conv}'; use format %d or %i for logical objects"))),
            (_, SpVal::Num(None)) => { numeric_special = true; "NA".into() }
            (_, SpVal::Num(Some(n))) if !n.is_finite() => {
                numeric_special = true;
                if n.is_nan() { "NaN".into() } else if n > 0.0 { format!("{}Inf", if plus { "+" } else if space { " " } else { "" }) } else { "-Inf".into() }
            }
            ('d' | 'i', SpVal::Num(Some(n))) => {
                if n != n.trunc() {
                    return Err(runtime_err(&format!("invalid format '%{conv}'; use format %f, %e, %g or %a for numeric objects")));
                }
                let mut s = format!("{}", n.abs() as i64);
                if let Some(p) = prec { while s.len() < p { s.insert(0, '0'); } }
                format!("{}{}", sign(n < 0.0), s)
            }
            ('f', SpVal::Num(Some(n))) => format!("{}{:.*}", sign(n.is_sign_negative() && n != 0.0), prec.unwrap_or(6), n.abs()),
            ('e' | 'E', SpVal::Num(Some(n))) => format!("{}{}", sign(n.is_sign_negative() && n != 0.0), c_exp(n.abs(), prec.unwrap_or(6), conv == 'E')),
            ('g' | 'G', SpVal::Num(Some(n))) => format!("{}{}", sign(n.is_sign_negative() && n != 0.0), c_g(n.abs(), prec, alt, conv == 'G')),
            ('x', SpVal::Num(Some(n))) => format!("{:x}", n as i64),
            ('X', SpVal::Num(Some(n))) => format!("{:X}", n as i64),
            _ => String::new(),
        };
        let blen = body.chars().count();
        if has_w && blen < width {
            let pad = width - blen;
            if left {
                body.push_str(&" ".repeat(pad));
            } else if zero && !numeric_special && conv != 's' {
                let (sg, rest) = if body.starts_with(['-', '+', ' ']) { body.split_at(1) } else { ("", body.as_str()) };
                body = format!("{}{}{}", sg, "0".repeat(pad), rest);
            } else {
                body = format!("{}{}", " ".repeat(pad), body);
            }
        }
        out.push_str(&body);
    }
    Ok(out)
}

/// `sprintf(fmt, ...)`, vectorised as R's is: the format and every
/// argument are recycled to the longest length, one string each; a zero-
/// length argument gives `character(0)`; an NA format gives NA. The
/// conversions follow C (`%e` writes `1.500000e+03`, `%g` honours its
/// precision), NA and non-finite values print as `NA`, `Inf`, `-Inf`,
/// `NaN`, and `%d` of a non-whole number is R's error, not a rounding.
pub fn bi_sprintf(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let fmts: Vec<Option<String>> = match &gv(a, 0) {
        RVal::Character(v, _) => v.iter().map(|x| x.as_ref().map(|s| s.to_string())).collect(),
        _ => return Err(runtime_err("'fmt' is not a character vector")),
    };
    let args: Vec<RVal> = a.iter().skip(1).map(|x| x.value.clone()).collect();
    let lens: Vec<usize> = std::iter::once(fmts.len()).chain(args.iter().map(sp_len)).collect();
    if lens.contains(&0) { return Ok(RVal::Character(Vec::new(), Attrs::default())); }
    let n = *lens.iter().max().unwrap_or(&1);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        out.push(match &fmts[k % fmts.len()] {
            Some(f) => Some(Arc::from(sprintf_one(f, &args, k)?.as_str())),
            None => None,
        });
    }
    Ok(RVal::Character(out, Attrs::default()))
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

    fn num(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default()) }
    fn sp(args: Vec<RVal>) -> Vec<String> {
        match bi_sprintf(&args.into_iter().map(evarg).collect::<Vec<_>>()).unwrap() {
            RVal::Character(v, _) => v.iter().map(|s| s.as_deref().unwrap_or("<NA>").to_string()).collect(),
            _ => panic!("sprintf must return character"),
        }
    }

    /// sprintf's contract, independent of any other program's output (the
    /// comparison with R is tests/differential/cases/formatting.R): one
    /// result per element of the longest argument, recycling, zero length,
    /// NA formats, and the argument errors.
    #[test]
    fn sprintf_vectorises_recycles_and_rejects() {
        assert_eq!(sp(vec![ch("%g"), num(&[1.0, 2.0, 3.0])]).len(), 3, "one result per element");
        assert_eq!(sp(vec![chs(&["%g", "<%g>"]), num(&[1.0])]).len(), 2, "the format vector counts too");
        assert_eq!(sp(vec![ch("%s%g"), chs(&["a", "b"]), num(&[1.0, 2.0, 3.0, 4.0])]).len(), 4, "recycled to the longest");
        assert!(sp(vec![ch("%g"), num(&[])]).is_empty(), "a zero-length argument gives character(0)");
        let na_fmt = RVal::Character(vec![None], Attrs::default());
        assert_eq!(sp(vec![na_fmt, num(&[1.0])]), ["<NA>"], "an NA format gives NA");
        assert!(bi_sprintf(&[evarg(ch("%d")), evarg(num(&[3.5]))]).is_err(), "%d of a fractional number");
        assert!(bi_sprintf(&[evarg(ch("%f")), evarg(ch("a"))]).is_err(), "%f of a string");
        assert!(bi_sprintf(&[evarg(ch("%d %d")), evarg(num(&[1.0]))]).is_err(), "too few arguments");
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
        // R's strsplit returns a LIST (one character vector per input string).
        let r = bi_strsplit(&[evarg(ch("a,b,c")), evarg(ch(","))]).unwrap();
        match r {
            RVal::List(items) => match &items[0].1 {
                RVal::Character(v, _) => {
                    let got: Vec<&str> = v.iter().filter_map(|x| x.as_deref()).collect();
                    assert_eq!(got, vec!["a", "b", "c"]);
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn sprintf_substitutes_args() {
        let r = bi_sprintf(&[evarg(ch("hi %s, %d!")), evarg(ch("world")), evarg(RVal::Integer(vec![Some(7)].into(), Attrs::default()))]).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("hi world, 7!")), _ => panic!() }
    }
}
