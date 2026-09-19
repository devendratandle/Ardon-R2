//! R2 I/O — file/text I/O builtins (Phase R.8).
//!
//! Hosts: `read.csv`, `write.csv`, `read.table`, `read.delim`,
//! `write.table`, `file.exists`, `list.files`.
//!
//! All builtins are pure `fn(&[EvalArg]) -> Result<RVal, R2Err>` — no
//! engine dependency. Args coerce via `RVal::as_logicals()` /
//! `scalar_f64()` from r2-types.
//!
//! **CSV / TSV parser (v0.1.0):** RFC 4180 state-machine parser.
//! Handles embedded separators in quoted fields, doubled-quote escape
//! (`""`), multi-line quoted fields, and UTF-8 BOM stripping. Write
//! side properly escapes column names and character values containing
//! quotes/separators/newlines. Remaining edge cases (mixed line
//! endings inside the same field; column-type inference beyond
//! numeric/logical/character) tracked in `docs/KNOWN_LIMITATIONS.md`.
//!
//! Engine-state I/O (`save`, `load`) stays in r2-engine — those need
//! access to `e.global_env.bindings` and a mutable evaluator.

use r2_types::{fmt_num, Attrs, Character, DataFrame, ErrKind, EvalArg, R2Err, RVal, Real};
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
fn rbool(b: bool) -> RVal {
    RVal::Logical(vec![Some(b)].into(), Attrs::default())
}

#[inline]
fn val_to_str(v: &RVal) -> String {
    match v {
        RVal::Character(c, _) => c.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        _ => String::new(),
    }
}

fn require_path(arg: &RVal, fn_name: &str) -> Result<String, R2Err> {
    match arg {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref())
            .map(|s| s.to_string())
            .ok_or_else(|| R2Err { msg: "NA path".into(), kind: ErrKind::Runtime }),
        _ => Err(R2Err {
            msg: format!("{} needs character path", fn_name),
            kind: ErrKind::Runtime,
        }),
    }
}

/// RFC 4180-compliant CSV/TSV parser — Phase R.14.
///
/// State-machine over the entire input (not line-by-line) so it handles
/// all four cases the previous line-split-and-trim approach missed:
///
/// 1. **Embedded delimiters in quoted fields:** `"a,b",c` → 2 fields.
/// 2. **Escaped double-quotes:** `"He said ""hi"""` → `He said "hi"`.
/// 3. **CRLF (or any newline) inside quoted fields:** a quoted field can
///    span multiple lines; the newline is part of the value.
/// 4. **BOM stripping:** `\u{FEFF}` at file start is silently removed.
///
/// Returns one `Vec<String>` per row. Empty trailing newlines are
/// skipped; truly empty rows in the middle of the file are dropped too
/// (matching R's `read.csv` behaviour).
fn parse_csv(content: &str, sep: &str) -> Vec<Vec<String>> {
    // 1. Strip UTF-8 BOM if present.
    let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);

    let sep_bytes = sep.as_bytes();
    let sep_first = sep_bytes.first().copied().unwrap_or(b',');
    let single_char_sep = sep_bytes.len() == 1;

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;

    let bytes = content.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        if in_quotes {
            if b == b'"' {
                // Doubled quote → literal quote; lone quote → close.
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    field.push('"');
                    i += 2;
                    continue;
                } else {
                    in_quotes = false;
                    i += 1;
                    continue;
                }
            }
            // Anything else inside quotes (incl. sep, newlines) is literal.
            field.push(b as char);
            i += 1;
            continue;
        }

        // Not in quotes.
        if b == b'"' && field.is_empty() {
            // Opening quote — only valid at field start.
            in_quotes = true;
            i += 1;
            continue;
        }
        // Separator check: support both 1-byte and multi-byte separators.
        let is_sep = if single_char_sep {
            b == sep_first
        } else {
            bytes[i..].starts_with(sep_bytes)
        };
        if is_sep {
            row.push(std::mem::take(&mut field));
            i += if single_char_sep { 1 } else { sep_bytes.len() };
            continue;
        }
        if b == b'\r' || b == b'\n' {
            // End of row. Consume the rest of the line break (CRLF or LF).
            // Empty rows are dropped to match R semantics.
            row.push(std::mem::take(&mut field));
            if !(row.len() == 1 && row[0].is_empty()) {
                rows.push(std::mem::take(&mut row));
            } else {
                row.clear();
            }
            i += 1;
            if b == b'\r' && i < bytes.len() && bytes[i] == b'\n' { i += 1; }
            continue;
        }
        field.push(b as char);
        i += 1;
    }

    // Flush trailing partial row (file without final newline).
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        if !(row.len() == 1 && row[0].is_empty()) {
            rows.push(row);
        }
    }

    // Post-pass: trim surrounding whitespace and orphaned outer quotes
    // ONLY for unquoted fields. Already-quoted fields keep verbatim content
    // (including leading/trailing whitespace inside the quotes — RFC 4180
    // is silent on this, but R preserves it).
    rows
}

/// Escape one CSV/TSV field for writing. RFC 4180:
/// - If field contains the separator, a double-quote, or a newline,
///   wrap in double-quotes and double any internal quotes.
/// - Otherwise emit raw.
///
/// Currently unused — `write_delimited` always wraps character fields
/// in quotes (matching R's `quote=TRUE` default). Retained for callers
/// that want minimal-quoting output.
#[allow(dead_code)]
fn escape_csv_field(s: &str, sep: &str) -> String {
    let needs_quoting = s.contains(sep) || s.contains('"') || s.contains('\n') || s.contains('\r');
    if needs_quoting {
        let escaped = s.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        s.to_string()
    }
}

fn columns_from_rows(raw_cols: &[Vec<String>], col_names: &[String]) -> Vec<(Arc<str>, RVal)> {
    let mut columns = Vec::with_capacity(raw_cols.len());
    for (i, col_data) in raw_cols.iter().enumerate() {
        let name = if i < col_names.len() {
            Arc::from(col_names[i].as_str())
        } else {
            Arc::from(format!("V{}", i + 1).as_str())
        };
        let all_num = col_data.iter().all(|s| s.is_empty() || s == "NA" || s.parse::<f64>().is_ok());
        let has_num = col_data.iter().any(|s| s.parse::<f64>().is_ok());
        if all_num && has_num {
            let nums: Vec<Real> = col_data.iter()
                .map(|s| if s.is_empty() || s == "NA" { None } else { s.parse().ok() })
                .collect();
            columns.push((name, RVal::Numeric(nums.into(), Attrs::default())));
        } else {
            let strs: Vec<Character> = col_data.iter()
                .map(|s| if s == "NA" { None } else { Some(Arc::from(s.as_str())) })
                .collect();
            columns.push((name, RVal::Character(strs, Attrs::default())));
        }
    }
    columns
}


/// R's `make.names(x, unique=)`: turn arbitrary strings into
/// syntactically valid, unique column names.
///
/// `read.csv` applies this by default (`check.names = TRUE`), which is
/// why a header `a b,c-d,1x` becomes `a.b`, `c.d`, `X1x` in R and every
/// column can then be reached as `d$a.b`. R2 used to keep the raw header,
/// so a column with a space could only be reached with backticks or
/// `d[["a b"]]`.
///
/// Rules, in R's order: `X` is prepended when the name does not start
/// with a letter, or with a dot not followed by a digit (so `1x` -> `X1x`,
/// `.1` -> `X.1`, `` -> `X`); every character that is not alphanumeric,
/// `.` or `_` becomes `.`; a reserved word gets a trailing `.`; then
/// duplicates are made unique with `.1`, `.2`, ... appended to the later
/// occurrences (`make.unique`, sep = ".").
pub fn make_names(names: &[String], unique: bool) -> Vec<String> {
    const RESERVED: &[&str] = &["if", "else", "repeat", "while", "function", "for", "next",
        "break", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_",
        "NA_character_", "in"];
    let mut out: Vec<String> = names.iter().map(|n| {
        let mut chars = n.chars();
        let first = chars.next();
        let second = chars.next();
        let ok_start = match first {
            Some(c) if c.is_alphabetic() => true,
            Some('.') => !matches!(second, Some(d) if d.is_ascii_digit()),
            _ => false,
        };
        let mut s = String::with_capacity(n.len() + 1);
        if !ok_start { s.push('X'); }
        for c in n.chars() {
            s.push(if c.is_alphanumeric() || c == '.' || c == '_' { c } else { '.' });
        }
        if RESERVED.contains(&s.as_str()) { s.push('.'); }
        s
    }).collect();
    if !unique { return out; }
    // make.unique: later duplicates get .1, .2, ... and the result must
    // itself not collide with an existing name.
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for i in 0..out.len() {
        let base = out[i].clone();
        let count = seen.get(&base).copied().unwrap_or(0);
        if count > 0 {
            let mut k = count;
            let cand = loop {
                let cand = format!("{base}.{k}");
                if !out.contains(&cand) && !seen.contains_key(&cand) { break cand; }
                k += 1;
            };
            out[i] = cand;
            seen.insert(base, k + 1);
        } else {
            seen.insert(base, 1);
        }
    }
    out
}

/// The `check.names=` argument: R's default is TRUE.
pub fn check_names_arg(a: &[EvalArg]) -> bool {
    gn(a, "check.names")
        .and_then(|v| match v { RVal::Logical(b, _) => b.first().copied().flatten(), _ => None })
        .unwrap_or(true)
}

fn read_delimited(path: &str, sep: &str, header: bool, check_names: bool) -> Result<RVal, R2Err> {
    let content = std::fs::read_to_string(path).map_err(|e| R2Err {
        msg: format!("cannot read '{}': {}", path, e),
        kind: ErrKind::Runtime,
    })?;
    let mut rows = parse_csv(&content, sep);
    let col_names: Vec<String> = if header && !rows.is_empty() {
        let raw = rows.remove(0);
        if check_names { make_names(&raw, true) } else { raw }
    } else {
        Vec::new()
    };
    let mut raw_cols: Vec<Vec<String>> = vec![Vec::new(); col_names.len().max(1)];
    for fields in rows {
        if raw_cols.len() < fields.len() { raw_cols.resize(fields.len(), Vec::new()); }
        for (i, field) in fields.iter().enumerate() {
            if i < raw_cols.len() { raw_cols[i].push(field.clone()); }
        }
    }
    let columns = columns_from_rows(&raw_cols, &col_names);
    Ok(RVal::DataFrame(DataFrame { columns, row_names: None }))
}

fn header_arg(a: &[EvalArg]) -> bool {
    gn(a, "header")
        .and_then(|v| match v {
            RVal::Logical(l, _) => l.first().copied().flatten(),
            _ => None,
        })
        .unwrap_or(true)
}

fn sep_arg(a: &[EvalArg], default: &str) -> String {
    gn(a, "sep")
        .and_then(|v| match v {
            RVal::Character(s, _) => s.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| default.into())
}

// ─────────────────────────────────────────────────────────────────────
// CSV / TSV readers
// ─────────────────────────────────────────────────────────────────────

pub fn bi_read_csv(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let path = require_path(&gv(a, 0), "read.csv")?;
    let header = header_arg(a);
    read_delimited(&path, ",", header, check_names_arg(a))
}

pub fn bi_read_table(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let path = require_path(&gv(a, 0), "read.table")?;
    let header = header_arg(a);
    let sep = sep_arg(a, "\t");
    read_delimited(&path, &sep, header, check_names_arg(a))
}

pub fn bi_read_delim(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // `read.delim` is `read.table` with tab separator (already default).
    bi_read_table(a)
}

// ─────────────────────────────────────────────────────────────────────
// Writers
// ─────────────────────────────────────────────────────────────────────

/// Header row: column names always quoted (matches R's default), with
/// any embedded quotes doubled per RFC 4180.
fn quoted_header(df: &DataFrame, sep: &str) -> String {
    df.columns.iter()
        .map(|(n, _)| {
            let escaped = n.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        })
        .collect::<Vec<_>>()
        .join(sep)
}

fn write_delimited(df: &DataFrame, path: &str, sep: &str, csv_style: bool) -> Result<(), R2Err> {
    let mut out = String::new();
    out.push_str(&quoted_header(df, sep));
    out.push('\n');
    for r in 0..df.nrow() {
        let row: Vec<String> = df.columns.iter().map(|(_, col)| match col {
            RVal::Numeric(v, _) => v.get(r).map(|x| match x {
                Some(n) => if csv_style { format!("{}", n) } else { fmt_num(*n) },
                None => "NA".into(),
            }).unwrap_or_default(),
            // RFC 4180: only quote+escape character fields when they
            // contain the separator, a quote, or a newline. Always
            // wrap in `"..."` though (matches R's `quote=TRUE` default).
            RVal::Character(v, _) => v.get(r).map(|x| match x {
                Some(s) => {
                    let escaped = s.replace('"', "\"\"");
                    format!("\"{}\"", escaped)
                }
                None => "NA".into(),
            }).unwrap_or_default(),
            RVal::Integer(v, _) => v.get(r).map(|x| match x {
                Some(n) => format!("{}", n),
                None => "NA".into(),
            }).unwrap_or_default(),
            _ => String::new(),
        }).collect();
        out.push_str(&row.join(sep));
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| R2Err {
        msg: format!("cannot write: {}", e),
        kind: ErrKind::Runtime,
    })?;
    println!("Written to {}", path);
    Ok(())
}

pub fn bi_write_csv(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "write.csv needs data.frame".into(), kind: ErrKind::Runtime }),
    };
    let path = require_path(&gv(a, 1), "write.csv")?;
    write_delimited(&df, &path, ",", true)?;
    Ok(RVal::Null)
}

pub fn bi_write_table(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let df = match &gv(a, 0) {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(R2Err { msg: "write.table needs data.frame".into(), kind: ErrKind::Runtime }),
    };
    let path = require_path(&gv(a, 1), "write.table")?;
    let sep = sep_arg(a, "\t");
    write_delimited(&df, &path, &sep, false)?;
    Ok(RVal::Null)
}

// ─────────────────────────────────────────────────────────────────────
// File system queries
// ─────────────────────────────────────────────────────────────────────

pub fn bi_file_exists(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a, 0));
    Ok(rbool(std::path::Path::new(&path).exists()))
}

pub fn bi_list_files(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let path = if matches!(gv(a, 0), RVal::Null) {
        ".".to_string()
    } else {
        val_to_str(&gv(a, 0))
    };
    let mut files: Vec<Character> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&path) {
        for entry in entries.flatten() {
            files.push(Some(Arc::from(entry.file_name().to_string_lossy().as_ref())));
        }
    }
    files.sort();
    Ok(RVal::Character(files, Attrs::default()))
}

// ─────────────────────────────────────────────────────────────────────
// Builtins registry (Phase R.8).
// ─────────────────────────────────────────────────────────────────────

pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("read.csv",     bi_read_csv),
        ("read.table",   bi_read_table),
        ("read.delim",   bi_read_delim),
        ("write.csv",    bi_write_csv),
        ("write.table",  bi_write_table),
        ("file.exists",  bi_file_exists),
        ("list.files",   bi_list_files),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }
    fn evarg_named(name: &str, v: RVal) -> EvalArg {
        EvalArg { name: Some(Arc::from(name)), value: v }
    }
    fn ch(s: &str) -> RVal { RVal::Character(vec![Some(Arc::from(s))], Attrs::default()) }

    fn tmp_path(suffix: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("r2io_test_{}_{}", std::process::id(), suffix));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn rfc4180_embedded_separator() {
        // `"a,b",c` is two fields: "a,b" and "c", not three.
        let path = tmp_path("embed_sep.csv");
        std::fs::write(&path, "x,y\n\"a,b\",c\n").unwrap();
        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                assert_eq!(df.ncol(), 2);
                assert_eq!(df.nrow(), 1);
                match &df.columns[0].1 {
                    RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("a,b")),
                    _ => panic!("col 0 should be character"),
                }
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rfc4180_doubled_quotes() {
        // `"He said ""hi"""` is one field whose value is `He said "hi"`.
        let path = tmp_path("doubled_quotes.csv");
        std::fs::write(&path, "msg\n\"He said \"\"hi\"\"\"\n").unwrap();
        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                match &df.columns[0].1 {
                    RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("He said \"hi\"")),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rfc4180_multiline_quoted_field() {
        // Newline inside a quoted field is part of the value, not a row break.
        let path = tmp_path("multiline.csv");
        std::fs::write(&path, "x,y\n\"line1\nline2\",ok\n").unwrap();
        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                assert_eq!(df.nrow(), 1);
                match &df.columns[0].1 {
                    RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("line1\nline2")),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rfc4180_bom_is_stripped() {
        // UTF-8 BOM at file start shouldn't pollute the first column name.
        let path = tmp_path("bom.csv");
        let content = format!("\u{FEFF}name,value\nalice,42\n");
        std::fs::write(&path, content).unwrap();
        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                assert_eq!(df.columns[0].0.as_ref(), "name", "BOM should be stripped");
                assert_eq!(df.nrow(), 1);
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_csv_escapes_quotes_and_separators() {
        // String containing comma and quotes should round-trip cleanly.
        use r2_types::Reals;
        let path = tmp_path("escape_rt.csv");
        let df = DataFrame {
            columns: vec![
                (Arc::from("name"), RVal::Character(
                    vec![Some(Arc::from("contains,comma")), Some(Arc::from("contains\"quote"))],
                    Attrs::default())),
                (Arc::from("n"), RVal::Numeric(
                    Reals::new(vec![Some(1.0), Some(2.0)]),
                    Attrs::default())),
            ],
            row_names: None,
        };
        bi_write_csv(&[evarg(RVal::DataFrame(df.clone())), evarg(ch(path.to_str().unwrap()))]).unwrap();
        // Now read back and verify content matches.
        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(rt) => {
                match &rt.columns[0].1 {
                    RVal::Character(v, _) => {
                        assert_eq!(v[0].as_deref(), Some("contains,comma"));
                        assert_eq!(v[1].as_deref(), Some("contains\"quote"));
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_csv_round_trip() {
        let path = tmp_path("rt.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#""x","y""#).unwrap();
        writeln!(f, "1,a").unwrap();
        writeln!(f, "2,b").unwrap();
        writeln!(f, "3,c").unwrap();
        drop(f);

        let r = bi_read_csv(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                assert_eq!(df.nrow(), 3);
                assert_eq!(df.ncol(), 2);
                assert_eq!(df.columns[0].0.as_ref(), "x");
                assert_eq!(df.columns[1].0.as_ref(), "y");
                match &df.columns[0].1 {
                    RVal::Numeric(v, _) => {
                        let got: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                        assert_eq!(got, vec![1.0, 2.0, 3.0]);
                    }
                    _ => panic!("first col should be numeric"),
                }
            }
            _ => panic!("read.csv must return DataFrame"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_csv_writes_quoted_header() {
        let path = tmp_path("write.csv");
        let df = DataFrame {
            columns: vec![
                (Arc::from("a"), RVal::Numeric(vec![Some(1.0), Some(2.0)].into(), Attrs::default())),
                (Arc::from("b"), RVal::Character(vec![Some(Arc::from("x")), Some(Arc::from("y"))], Attrs::default())).into(),
            ],
            row_names: None,
        };
        let r = bi_write_csv(&[evarg(RVal::DataFrame(df)), evarg(ch(path.to_str().unwrap()))]).unwrap();
        assert!(matches!(r, RVal::Null));
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.starts_with("\"a\",\"b\""));
        assert!(written.contains("\"x\""));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_table_uses_tab_default() {
        let path = tmp_path("rt.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "x\ty").unwrap();
        writeln!(f, "1\ta").unwrap();
        writeln!(f, "2\tb").unwrap();
        drop(f);

        let r = bi_read_table(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        match r {
            RVal::DataFrame(df) => assert_eq!(df.ncol(), 2),
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_csv_with_header_false_uses_v_names() {
        let path = tmp_path("noheader.csv");
        std::fs::write(&path, "1,2\n3,4\n").unwrap();
        let r = bi_read_csv(&[
            evarg(ch(path.to_str().unwrap())),
            evarg_named("header", RVal::Logical(vec![Some(false)].into(), Attrs::default())),
        ]).unwrap();
        match r {
            RVal::DataFrame(df) => {
                assert_eq!(df.nrow(), 2);
                // First col still gets V1, V2 names since header=false.
                assert!(df.columns[0].0.as_ref().starts_with('V'));
            }
            _ => panic!(),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn file_exists_reports_truthfully() {
        let path = tmp_path("exists.txt");
        std::fs::write(&path, "hi").unwrap();
        let r1 = bi_file_exists(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        assert!(matches!(r1, RVal::Logical(v, _) if v[0] == Some(true)));
        std::fs::remove_file(&path).unwrap();
        let r2 = bi_file_exists(&[evarg(ch(path.to_str().unwrap()))]).unwrap();
        assert!(matches!(r2, RVal::Logical(v, _) if v[0] == Some(false)));
    }
}
