//! Engine-side file-I/O builtins. Most delegate one line
//! to `r2_io::bi_*`; `list.files` (which filters through `grepl`) and the
//! path helpers are here. The richer `bi_read_csv_v2` (with header /
//! sep / NA inference) stays in `lib.rs` because it uses Engine
//! helpers like `as_logicals`; only the thin pass-throughs live
//! here.

use std::sync::Arc;
use r2_types::{Attrs, EnvRef, EvalArg, R2Err, RVal};

use crate::{gn, gv, val_to_str, Engine};

pub(crate) fn bi_write_csv(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_write_csv(a)
}
pub(crate) fn bi_read_table(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_read_table(a)
}
pub(crate) fn bi_read_delim(_e: &mut Engine, a: &[EvalArg], _env: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_read_delim(a)
}
pub(crate) fn bi_write_table(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_write_table(a)
}
pub(crate) fn bi_file_exists(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_io::bi_file_exists(a)
}
/// R's `list.files(path = ".", pattern = NULL, all.files = FALSE,
/// full.names = FALSE, recursive = FALSE, ignore.case = FALSE,
/// include.dirs = FALSE)`, also as `dir()`. `pattern` is a regular
/// expression matched against each file NAME (as `grepl` matches);
/// names are relative to `path` unless `full.names`; sorted as `sort()`
/// sorts. (It read only the first argument, so `list.files(pattern = ...)`
/// looked for a folder named after the pattern and returned nothing.)
pub(crate) fn bi_list_files(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let positional: Vec<&RVal> = a.iter().filter(|x| x.name.is_none()).map(|x| &x.value).collect();
    let arg = |name: &str, pos: usize| gn(a, name).or_else(|| positional.get(pos).map(|v| (*v).clone()));
    let flag = |name: &str| gn(a, name).and_then(|v| v.as_logicals().ok())
        .and_then(|v| v.first().copied().flatten()).unwrap_or(false);
    let path = match arg("path", 0) { Some(RVal::Null) | None => ".".to_string(), Some(v) => val_to_str(&v) };
    let pattern = arg("pattern", 1).filter(|v| !matches!(v, RVal::Null));
    let (all, full, recursive, include_dirs) = (flag("all.files"), flag("full.names"), flag("recursive"), flag("include.dirs"));

    // (relative path, file name) of every entry, walking down if recursive
    let mut found: Vec<(String, String)> = Vec::new();
    let mut stack = vec![String::new()];
    while let Some(rel) = stack.pop() {
        let dir = if rel.is_empty() { path.clone() } else { format!("{}/{}", path, rel) };
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !all && name.starts_with('.') { continue; }
            let child = if rel.is_empty() { name.clone() } else { format!("{}/{}", rel, name) };
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir && recursive {
                stack.push(child.clone());
                if !include_dirs { continue; }
            }
            found.push((child, name));
        }
    }
    if let Some(p) = pattern {
        let names = RVal::Character(found.iter().map(|(_, n)| Some(Arc::from(n.as_str()))).collect(), Attrs::default());
        let mut ga = vec![EvalArg { name: None, value: p }, EvalArg { name: None, value: names }];
        if let Some(ic) = gn(a, "ignore.case") { ga.push(EvalArg { name: Some(Arc::from("ignore.case")), value: ic }); }
        let keep = match r2_strings::bi_grepl(&ga)? { RVal::Logical(v, _) => v.to_vec(), _ => Vec::new() };
        found = found.into_iter().zip(keep).filter(|(_, k)| *k == Some(true)).map(|(f, _)| f).collect();
    }
    let mut out: Vec<String> = found.into_iter()
        .map(|(rel, _)| if full { format!("{}/{}", path.trim_end_matches(['/', '\\']), rel) } else { rel }).collect();
    out.sort_by_cached_key(|s| r2_types::collation_key(s));
    Ok(RVal::Character(out.into_iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()))
}

/// `file.path(...)`: the parts joined with "/", element-wise with recycling.
pub(crate) fn bi_file_path(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let parts: Vec<Vec<String>> = a.iter().filter(|x| x.name.as_deref() != Some("fsep"))
        .map(|x| r2_types::factor_labels(&x.value).map(|(l, _)| l.into_iter().map(|s| s.map_or("NA".into(), |s| s.to_string())).collect()).unwrap_or_default())
        .collect();
    let sep = gn(a, "fsep").map(|v| val_to_str(&v)).unwrap_or_else(|| "/".into());
    if parts.iter().any(|p| p.is_empty()) { return Ok(RVal::Character(Vec::new(), Attrs::default())); }
    let n = parts.iter().map(|p| p.len()).max().unwrap_or(0);
    Ok(RVal::Character((0..n).map(|i| {
        let s = parts.iter().map(|p| p[i % p.len()].as_str()).collect::<Vec<_>>().join(&sep);
        Some(Arc::from(s.as_str()))
    }).collect(), Attrs::default()))
}

/// Split a path at its last separator ("/" or "\"), ignoring trailing ones.
fn split_path(p: &str) -> (String, String) {
    let t = p.trim_end_matches(['/', '\\']);
    match t.rfind(['/', '\\']) {
        Some(0) => ("/".into(), t[1..].into()),
        Some(i) => (t[..i].to_string(), t[i + 1..].to_string()),
        None => (".".into(), t.to_string()),
    }
}

fn map_paths(a: &[EvalArg], f: impl Fn(&str) -> String) -> Result<RVal, R2Err> {
    let v = gv(a, 0);
    let (labels, _) = r2_types::factor_labels(&v).unwrap_or_default();
    Ok(RVal::Character(labels.into_iter().map(|s| s.map(|s| Arc::from(f(&s).as_str()))).collect(), Attrs::default()))
}

/// `basename(path)`: the last component ("DASS.csv").
pub(crate) fn bi_basename(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    map_paths(a, |p| split_path(p).1)
}

/// `dirname(path)`: everything before the last component, "." if none.
pub(crate) fn bi_dirname(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    map_paths(a, |p| split_path(p).0)
}
