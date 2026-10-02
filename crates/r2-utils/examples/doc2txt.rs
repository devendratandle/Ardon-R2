//! Turn the Markdown docs into plain text that reads well in Notepad, for
//! the Windows installer and zip (Windows has no default program for .md,
//! and renaming leaves `**`, `|` tables and HTML in place).
//!
//!     cargo run -p r2-utils --example doc2txt -- <out-dir> <file>...
//!
//! Each `NAME.md` becomes `<out-dir>/NAME.txt`; a file that is not Markdown
//! (LICENSE) is copied as `NAME.txt`. Output: headings underlined, Markdown
//! and HTML markup removed, links as "text (url)", images dropped, lists
//! and paragraphs wrapped at 78 columns, code blocks indented, and tables
//! restructured — aligned columns when they fit, otherwise one labelled
//! block per row ("Install size / R: 200+ MB / R2: ~9 MB installer").

use std::path::Path;

const WIDTH: usize = 78;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((out_dir, files)) = args.split_first() else {
        eprintln!("usage: doc2txt <out-dir> <file>..."); std::process::exit(2)
    };
    std::fs::create_dir_all(out_dir).expect("create output dir");
    for f in files {
        let src = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let stem = Path::new(f).file_stem().and_then(|s| s.to_str()).unwrap_or("doc");
        let text = if f.ends_with(".md") { md_to_txt(&src) } else { src };
        let out = Path::new(out_dir).join(format!("{stem}.txt"));
        // CRLF: Notepad shows LF-only files fine since Windows 10 1809, older
        // editors do not.
        std::fs::write(&out, text.replace("\r\n", "\n").replace('\n', "\r\n")).expect("write");
        println!("{} -> {}", f, out.display());
    }
}

/// The whole conversion, block by block.
fn md_to_txt(md: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let mut i = 0;
    let flush = |para: &mut Vec<String>, out: &mut Vec<String>| {
        if !para.is_empty() {
            let text = inline(&para.join(" "));
            if !text.trim().is_empty() { out.extend(wrap(text.trim(), "", "")); }
            para.clear();
        }
    };
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        if t.starts_with("```") {
            flush(&mut para, &mut out);
            blank(&mut out);
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("```") {
                out.push(format!("    {}", lines[i]));
                i += 1;
            }
            blank(&mut out);
        } else if t.is_empty() {
            flush(&mut para, &mut out);
            blank(&mut out);
        } else if let Some(h) = heading(t) {
            flush(&mut para, &mut out);
            blank(&mut out);
            let (level, text) = h;
            let text = inline(text);
            let n = text.chars().count();
            match level {
                1 => { out.push("=".repeat(n)); out.push(text); out.push("=".repeat(n)); }
                2 => { out.push(text); out.push("-".repeat(n)); }
                _ => { out.push(text); out.push("~".repeat(n)); }
            }
            blank(&mut out);
        } else if t.starts_with('|') && lines.get(i + 1).is_some_and(|l| is_table_rule(l)) {
            flush(&mut para, &mut out);
            let mut rows = vec![cells(t)];
            i += 2;
            while i < lines.len() && lines[i].trim().starts_with('|') { rows.push(cells(lines[i].trim())); i += 1; }
            // a table of images has nothing left in plain text
            if rows[1..].iter().all(|r| r.iter().all(|c| c.is_empty())) { continue; }
            blank(&mut out);
            out.extend(table(&rows));
            blank(&mut out);
            continue;
        } else if t.chars().all(|c| c == '-' || c == '*' || c == '_') && t.len() >= 3 {
            flush(&mut para, &mut out);
            blank(&mut out);
            out.push("-".repeat(WIDTH));
            blank(&mut out);
        } else if let Some((indent, bullet, rest)) = list_item(line) {
            flush(&mut para, &mut out);
            // continuation lines: indented, not a new item or block
            let mut item = vec![rest.to_string()];
            while i + 1 < lines.len() {
                let next = lines[i + 1];
                let nt = next.trim();
                if nt.is_empty() || list_item(next).is_some() || nt.starts_with('|') || nt.starts_with("```") || heading(nt).is_some() { break; }
                item.push(nt.to_string());
                i += 1;
            }
            let pad = " ".repeat(2 + indent.min(8));
            let first = format!("{pad}{bullet} ");
            let hang = " ".repeat(first.chars().count());
            out.extend(wrap(inline(&item.join(" ")).trim(), &first, &hang));
        } else if let Some(q) = t.strip_prefix('>') {
            flush(&mut para, &mut out);
            // a quote's lines are one paragraph
            let mut quote = vec![q.trim().to_string()];
            while let Some(next) = lines.get(i + 1).and_then(|l| l.trim().strip_prefix('>')) {
                quote.push(next.trim().to_string());
                i += 1;
            }
            out.extend(wrap(inline(&quote.join(" ")).trim(), "    ", "    "));
        } else {
            para.push(t.to_string());
        }
        i += 1;
    }
    flush(&mut para, &mut out);
    // collapse runs of blank lines; no leading/trailing blanks
    let mut res: Vec<String> = Vec::new();
    for l in out {
        let l = l.trim_end().to_string();
        if l.is_empty() && res.last().is_none_or(|p: &String| p.is_empty()) { continue; }
        res.push(l);
    }
    while res.last().is_some_and(|l| l.is_empty()) { res.pop(); }
    res.join("\n") + "\n"
}

fn blank(out: &mut Vec<String>) { if out.last().is_some_and(|l| !l.is_empty()) { out.push(String::new()); } }

fn heading(t: &str) -> Option<(usize, &str)> {
    let level = t.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&level) && t[level..].starts_with(' ') { Some((level, t[level..].trim())) } else { None }
}

fn is_table_rule(l: &str) -> bool {
    let t = l.trim();
    t.starts_with('|') && t.contains('-') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// `- x`, `* x`, `1. x` → (indent, bullet to print, text).
fn list_item(line: &str) -> Option<(usize, String, &str)> {
    let indent = line.len() - line.trim_start().len();
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).or_else(|| t.strip_prefix("+ ")) {
        return Some((indent, "-".into(), rest));
    }
    let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && t[digits..].starts_with(". ") { return Some((indent, t[..digits + 1].to_string(), &t[digits + 2..])); }
    None
}

/// Cells of a `| a | b |` row; a `|` inside backticks or escaped is not a separator.
fn cells(row: &str) -> Vec<String> {
    let inner = row.trim().trim_start_matches('|').trim_end_matches('|');
    let (mut cells, mut cur, mut in_code, mut prev) = (Vec::new(), String::new(), false, ' ');
    for c in inner.chars() {
        match c {
            '`' => { in_code = !in_code; cur.push(c); }
            '|' if !in_code && prev != '\\' => { cells.push(inline(cur.trim())); cur.clear(); }
            '|' => { if prev == '\\' { cur.pop(); } cur.push(c); }
            _ => cur.push(c),
        }
        prev = c;
    }
    cells.push(inline(cur.trim()));
    cells
}

/// A table as aligned columns when it fits in `WIDTH`, else one block per
/// row: the first cell as the row's title, then "Header: value" lines.
fn table(rows: &[Vec<String>]) -> Vec<String> {
    let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let cell = |r: &Vec<String>, j: usize| r.get(j).cloned().unwrap_or_default();
    let widths: Vec<usize> = (0..ncol).map(|j| rows.iter().map(|r| cell(r, j).chars().count()).max().unwrap_or(0)).collect();
    let total = 2 + widths.iter().sum::<usize>() + 3 * ncol.saturating_sub(1);
    let mut out = Vec::new();
    if total <= WIDTH {
        let line = |r: &Vec<String>| {
            let s = (0..ncol).map(|j| format!("{:<w$}", cell(r, j), w = widths[j])).collect::<Vec<_>>().join("   ");
            format!("  {}", s.trim_end())
        };
        out.push(line(&rows[0]));
        out.push(format!("  {}", widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("   ")));
        for r in &rows[1..] { out.push(line(r)); }
        return out;
    }
    let header = &rows[0];
    for r in &rows[1..] {
        let title = cell(r, 0);
        let title = if title.is_empty() { "-".to_string() } else { title };
        out.extend(wrap(&title, "  * ", "    "));
        for j in 1..ncol {
            let v = cell(r, j);
            if v.is_empty() { continue; }
            let h = cell(header, j);
            let lead = if h.is_empty() { "      ".to_string() } else { format!("      {h}: ") };
            out.extend(wrap(&v, &lead, "        "));
        }
        out.push(String::new());
    }
    out.pop();
    out
}

/// Inline Markdown and HTML to plain text: code spans kept verbatim,
/// emphasis markers and tags removed, links as "text (url)", images
/// dropped, entities decoded.
fn inline(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let rest: String = chars[i..].iter().collect();
        if c == '`' {
            let end = chars[i + 1..].iter().position(|&d| d == '`').map(|p| i + 1 + p);
            if let Some(e) = end { out.extend(&chars[i + 1..e]); i = e + 1; continue; }
        }
        if c == '!' && chars.get(i + 1) == Some(&'[') {
            if let Some((_, _, len)) = link_at(&chars, i + 1) { i += 1 + len; continue; }
        }
        if c == '[' {
            if let Some((text, url, len)) = link_at(&chars, i) {
                let text = inline(&text);
                if url.starts_with("http") && text != url { out.push_str(&format!("{text} ({url})")); } else { out.push_str(&text); }
                i += len;
                continue;
            }
        }
        if c == '<' && chars.get(i + 1).is_some_and(|d| d.is_ascii_alphabetic() || *d == '/') {
            if let Some(p) = chars[i..].iter().position(|&d| d == '>') {
                let tag: String = chars[i..i + p + 1].iter().collect();
                if tag.starts_with("<br") { out.push(' '); }
                i += p + 1;
                continue;
            }
        }
        if rest.starts_with("**") || rest.starts_with("__") { i += 2; continue; }
        if c == '*' && chars.get(i + 1).is_some_and(|d| !d.is_whitespace()) && i + 1 < chars.len() { i += 1; continue; }
        if c == '*' && i > 0 && !chars[i - 1].is_whitespace() { i += 1; continue; }
        if c == '&' {
            let ents = [("&amp;", '&'), ("&lt;", '<'), ("&gt;", '>'), ("&nbsp;", ' '), ("&quot;", '"')];
            if let Some((e, ch)) = ents.iter().find(|(e, _)| rest.starts_with(e)) { out.push(*ch); i += e.len(); continue; }
        }
        out.push(c);
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `[text](url)` at `start` → (text, url, characters consumed).
fn link_at(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    let mut depth = 0;
    let mut close = None;
    for (k, &c) in chars.iter().enumerate().skip(start) {
        if c == '[' { depth += 1; }
        if c == ']' { depth -= 1; if depth == 0 { close = Some(k); break; } }
    }
    let close = close?;
    if chars.get(close + 1) != Some(&'(') { return None; }
    let end = chars[close + 2..].iter().position(|&c| c == ')')? + close + 2;
    let text: String = chars[start + 1..close].iter().collect();
    let url: String = chars[close + 2..end].iter().collect();
    Some((text, url, end + 1 - start))
}

/// Wrap at `WIDTH`: `first` before the first line, `hang` before the rest.
fn wrap(text: &str, first: &str, hang: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = first.to_string();
    let mut empty = true;
    for word in text.split_whitespace() {
        if !empty && cur.chars().count() + 1 + word.chars().count() > WIDTH {
            lines.push(std::mem::take(&mut cur));
            cur = hang.to_string();
            empty = true;
        }
        if !empty { cur.push(' '); }
        cur.push_str(word);
        empty = false;
    }
    if !empty || lines.is_empty() { lines.push(cur); }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_and_links() {
        assert_eq!(inline("**R2** is `x <- 1` see [docs](https://a.b) and [map](FUNCTIONS.md)"),
                   "R2 is x <- 1 see docs (https://a.b) and map");
        assert_eq!(inline("<p align=\"center\"><strong>Inspired by R.</strong><br>"), "Inspired by R.");
        assert_eq!(inline("![logo](assets/logo.png) text"), "text");
    }

    #[test]
    fn narrow_table_aligns_and_wide_table_becomes_blocks() {
        let narrow = md_to_txt("| a | bb |\n|---|---|\n| 1 | 2 |\n");
        assert!(narrow.contains("  a   bb\n  -   --\n  1   2"));
        let wide = md_to_txt(&format!("| | R | R2 |\n|---|---|---|\n| Install size | {} | ~9 MB |\n", "x".repeat(80)));
        assert!(wide.contains("  * Install size\n      R: x"));
        assert!(wide.contains("      R2: ~9 MB"));
    }

    #[test]
    fn headings_and_lists() {
        let t = md_to_txt("# Title\n\n- **one** item that is long\n  continued\n");
        assert!(t.starts_with("=====\nTitle\n=====\n\n  - one item that is long continued"));
    }
}
