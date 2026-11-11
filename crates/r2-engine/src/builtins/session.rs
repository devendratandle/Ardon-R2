//! The session: working directory, waiting (`Sys.sleep`, `readline`),
//! saving and loading the workspace (`save` / `load`), and `version()`.

#![allow(clippy::all)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;

// ── working directory ────────────────────────────────────────────────

/// The working directory as R reports it: forward slashes on Windows too
/// (`C:/Users/...`), so it can be pasted back into a string unescaped.
fn working_dir() -> String {
    std::env::current_dir().map(|p| p.to_string_lossy().replace('\\', "/")).unwrap_or_default()
}

pub(crate) fn bi_getwd(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(rstr(&working_dir()))
}

/// `setwd(dir)` returns the PREVIOUS directory (invisibly), as R does, so
/// `old <- setwd(dir); ...; setwd(old)` restores it.
pub(crate) fn bi_setwd(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a,0));
    let old = working_dir();
    std::env::set_current_dir(&path).map_err(|_| R2Err{msg:format!("cannot change working directory to \"{}\"", path),kind:ErrKind::Runtime})?;
    Ok(rstr(&old))
}

// ── waiting for time or input ────────────────────────────────────────

pub(crate) fn bi_Sys_sleep(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let secs = match &gv(a,0) { RVal::Numeric(v,_) => v[0].unwrap_or(0.0), _ => 0.0 };
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    Ok(RVal::Null)
}

/// `readline(prompt="")` — blocks until the user types a line on stdin
/// and presses Enter. Returns the line as a character scalar (without
/// the trailing newline). The prompt, if provided, is printed first.
/// Used for interactive prompts in scripts ("press Enter to continue",
/// "type a filename:", etc.).
pub(crate) fn bi_readline(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::io::{BufRead, Write};
    let prompt = gv(a, 0);
    let prompt_str = match &prompt {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()).unwrap_or_default(),
        RVal::Null => String::new(),
        other => val_to_str(other),
    };
    if !prompt_str.is_empty() {
        sout!("{}", prompt_str);
        let _ = std::io::stdout().flush();
    }
    let mut line = String::new();
    let stdin = std::io::stdin();
    let _ = stdin.lock().read_line(&mut line);
    let trimmed = line.trim_end_matches(|c| c == '\n' || c == '\r').to_string();
    Ok(RVal::Character(
        vec![Some(std::sync::Arc::from(trimmed.as_str()))],
        Attrs::default(),
    ))
}

// ── save() / load() — session persistence ────────────────────────────

pub(crate) fn bi_save(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // save("file.r2s")          — save all session variables
    // save(object, "file.r2d")  — save single data object
    // save(model, "file.r2m")   — save model object
    let first = gv(a, 0);

    // Check if first arg is a string (session save) or an object (object save)
    let (obj_to_save, path) = match &first {
        RVal::Character(_, _) => {
            // save("session.r2s") — save all variables
            let path = val_to_str(&first);
            (None, path)
        }
        _ => {
            // save(object, "file.r2d") — save single object
            let path = gn(a, "file").or(Some(gv(a, 1))).map(|v| val_to_str(&v))
                .unwrap_or("object.r2d".into());
            (Some(first.clone()), path)
        }
    };

    let mut out = String::new();

    // Header with format version
    out.push_str("#R2 v0.1.1\n");

    if let Some(obj) = obj_to_save {
        // Single object save
        let serialized = serialize_rval(&obj);
        if serialized.is_empty() {
            return err!(Runtime, "cannot serialize {} objects", obj.type_name());
        }
        out.push_str(&format!("_obj={}\n", serialized));
        std::fs::write(&path, &out).map_err(|e| R2Err{msg:format!("cannot save to '{}': {}", path, e),kind:ErrKind::Runtime})?;
        let ext = path.rsplit('.').next().unwrap_or("");
        let kind = match ext { "r2m" => "model", "r2d" => "data", _ => "object" };
        soutln!("Saved {} ({}) to '{}'", kind, obj.type_name(), path);
    } else {
        // Session save — all variables
        let mut count = 0;
        for (name, val) in e.global_env.bindings.read().unwrap().iter() {
            let serialized = serialize_rval(val);
            if !serialized.is_empty() {
                out.push_str(&format!("{}={}\n", name, serialized));
                count += 1;
            }
        }
        std::fs::write(&path, &out).map_err(|e| R2Err{msg:format!("cannot save to '{}': {}", path, e),kind:ErrKind::Runtime})?;
        soutln!("Saved {} objects to '{}'", count, path);
    }
    Ok(RVal::Null)
}

/// `save(a, b, list = , file = )` as R: the objects under their names, and
/// nothing printed. The names come from the call (`special_forms.rs`).
pub(crate) fn save_named(items: &[(Arc<str>, RVal)], path: &str) -> Result<(), R2Err> {
    let mut out = String::from("#R2 v0.1.1\n");
    for (name, val) in items {
        let serialized = serialize_rval(val);
        if serialized.is_empty() { return err!(Runtime, "cannot save '{}': a {} cannot be saved", name, val.type_name()); }
        out.push_str(&format!("{}={}\n", name, serialized));
    }
    std::fs::write(path, &out).map_err(|e| R2Err{msg:format!("cannot open file '{}': {}", path, e),kind:ErrKind::Runtime})
}

/// `load(file)`: each saved object back under its name where `load` is
/// called, returning the names, as R does. A file written by the old
/// single-object `save(object, "file.r2d")` still returns its object.
pub(crate) fn bi_load(_e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let path = gn(a,"file").or(Some(gv(a,0))).map(|v| val_to_str(&v)).unwrap_or("session.r2s".into());
    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot open file '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let mut names = Vec::new();
    for line in content.lines() {
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some(eq_pos) = line.find('=') {
            let name = &line[..eq_pos];
            if let Some(val) = deserialize_rval(&line[eq_pos+1..]) {
                if name == "_obj" { return Ok(val); }
                env.set(Arc::from(name), val);
                names.push(Some(Arc::from(name)));
            }
        }
    }
    Ok(RVal::Character(names, Attrs::default()))
}

fn serialize_rval(val: &RVal) -> String {
    match val {
        RVal::Numeric(v, _) => {
            let nums: Vec<String> = v.iter().map(|x| match x { Some(n) => fmt_num(*n), None => "NA".into() }).collect();
            format!("N:{}", nums.join(","))
        }
        RVal::Integer(v, _) => {
            let nums: Vec<String> = v.iter().map(|x| match x { Some(n) => format!("{}", n), None => "NA".into() }).collect();
            format!("I:{}", nums.join(","))
        }
        RVal::Character(v, _) => {
            let strs: Vec<String> = v.iter().map(|x| match x { Some(s) => s.to_string(), None => "NA".into() }).collect();
            format!("C:{}", strs.join("\t"))
        }
        RVal::Logical(v, _) => {
            let vals: Vec<String> = v.iter().map(|x| match x { Some(true) => "T".into(), Some(false) => "F".into(), None => "NA".into() }).collect();
            format!("L:{}", vals.join(","))
        }
        RVal::Factor(f) => {
            // F:levels (tab-separated) \x1e codes (1-based, comma-separated) \x1e ordered
            let levels: Vec<&str> = f.levels.iter().map(|l| l.as_ref()).collect();
            let codes: Vec<String> = f.codes.iter().map(|c| c.map_or("NA".into(), |c| (c + 1).to_string())).collect();
            format!("F:{}\x1e{}\x1e{}", levels.join("\t"), codes.join(","), if f.ordered { "T" } else { "F" })
        }
        RVal::DataFrame(df) => {
            // Serialize DataFrame: D:ncol\tcol1_name\ttype:data\tcol2_name\ttype:data...
            let mut parts = vec![format!("{}", df.columns.len())];
            for (name, col) in &df.columns {
                let col_ser = serialize_rval(col);
                parts.push(format!("{}:{}", name, col_ser));
            }
            format!("D:{}", parts.join("\x1f")) // unit separator
        }
        RVal::Matrix(m) => {
            let nums: Vec<String> = m.data.iter().map(|n| fmt_num(*n)).collect();
            format!("M:{}:{}:{}", m.nrow, m.ncol, nums.join(","))
        }
        RVal::TypeInstance(inst) => {
            // Serialize model: T:classname\x1ffield1=ser\x1ffield2=ser...
            let mut parts = vec![inst.type_name.to_string()];
            for (k, v) in &inst.fields {
                let v_ser = serialize_rval(v);
                if !v_ser.is_empty() {
                    parts.push(format!("{}={}", k, v_ser));
                }
            }
            format!("T:{}", parts.join("\x1f"))
        }
        _ => String::new(),
    }
}

fn deserialize_rval(s: &str) -> Option<RVal> {
    if s.len() < 2 { return None; }
    let (typ, data) = (s.as_bytes()[0] as char, &s[2..]);
    match typ {
        'N' => {
            let vals: Vec<Real> = data.split(',').map(|s| if s == "NA" { None } else { s.parse().ok() }).collect();
            Some(RVal::Numeric(vals.into(), Attrs::default()))
        }
        'I' => {
            let vals: Vec<Integer> = data.split(',').map(|s| if s == "NA" { None } else { s.parse().ok() }).collect();
            Some(RVal::Integer(vals.into(), Attrs::default()))
        }
        'C' => {
            let vals: Vec<Character> = data.split('\t').map(|s| if s == "NA" { None } else { Some(Arc::from(s)) }).collect();
            Some(RVal::Character(vals, Attrs::default()))
        }
        'L' => {
            let vals: Vec<Logical> = data.split(',').map(|s| match s { "T" => Some(true), "F" => Some(false), _ => None }).collect();
            Some(RVal::Logical(vals.into(), Attrs::default()))
        }
        'F' => {
            let mut parts = data.split('\x1e');
            let levels: Vec<Arc<str>> = parts.next()?.split('\t').filter(|l| !l.is_empty()).map(Arc::from).collect();
            let codes = parts.next()?.split(',').filter(|c| !c.is_empty())
                .map(|c| c.parse::<u32>().ok().map(|k| k - 1)).collect();
            let ordered = parts.next() == Some("T");
            Some(RVal::Factor(Factor { codes, levels, ordered }))
        }
        'M' => {
            // Matrix: M:nrow:ncol:data
            let parts: Vec<&str> = data.splitn(3, ':').collect();
            if parts.len() != 3 { return None; }
            let nrow: usize = parts[0].parse().ok()?;
            let ncol: usize = parts[1].parse().ok()?;
            let vals: Vec<f64> = parts[2].split(',').filter_map(|s| s.parse().ok()).collect();
            Some(RVal::Matrix(Matrix::new(vals, nrow, ncol)))
        }
        'D' => {
            // DataFrame: D:ncol\x1fcol_name:type:data...
            let parts: Vec<&str> = data.split('\x1f').collect();
            if parts.is_empty() { return None; }
            let mut columns = Vec::new();
            for part in &parts[1..] {
                if let Some(colon) = part.find(':') {
                    let col_name = &part[..colon];
                    let col_data = &part[colon+1..];
                    if let Some(val) = deserialize_rval(col_data) {
                        columns.push((Arc::from(col_name), val));
                    }
                }
            }
            Some(RVal::DataFrame(DataFrame { columns, row_names: None }))
        }
        'T' => {
            // TypeInstance: T:classname\x1ffield=val...
            let parts: Vec<&str> = data.split('\x1f').collect();
            if parts.is_empty() { return None; }
            let type_name = Arc::from(parts[0]);
            let mut fields = HashMap::new();
            for part in &parts[1..] {
                if let Some(eq) = part.find('=') {
                    let key = Arc::from(&part[..eq]);
                    let val_str = &part[eq+1..];
                    if let Some(val) = deserialize_rval(val_str) {
                        fields.insert(key, val);
                    }
                }
            }
            Some(RVal::TypeInstance(TypeInstance { type_name, fields }))
        }
        _ => None,
    }
}

// ── version() ────────────────────────────────────────────────────────

pub(crate) fn bi_version(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Every number here is read, not typed: the version from the frontend,
    // the function count from the registry, the cores from rayon. The
    // typed version sat at 0.1.1 and the function count at "191+" for
    // four releases.
    soutln!("\nArdon-R2 — Statistics & AI Training, Reimagined");
    soutln!("Version: {}", crate::product_version());
    soutln!("Created by: Devendra Tandale");
    soutln!("An AI assisted project");
    soutln!("Platform: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
    soutln!("Kernel: pure Rust — no C, C++ or Fortran; no FFI");
    soutln!("Parallel cores: {}", rayon::current_num_threads());
    soutln!("Builtins: {} (functions and operators)", e.registry.n_functions());
    soutln!("License: AGPL v3");
    soutln!();
    Ok(RVal::Null)
}

// ── tempdir() / tempfile() ───────────────────────────────────────────

/// The session's own folder under the system temp folder, made on first use,
/// as R's `RtmpXXXX`.
fn session_tempdir() -> Result<std::path::PathBuf, R2Err> {
    let dir = std::env::temp_dir().join(format!("R2tmp{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| R2Err { msg: format!("cannot create the temporary folder: {}", e), kind: ErrKind::Runtime })?;
    Ok(dir)
}

fn path_text(p: &std::path::Path) -> RVal {
    RVal::Character(vec![Some(Arc::from(p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/").as_str()))], Attrs::default())
}

pub(crate) fn bi_tempdir(_: &mut Engine, _: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    Ok(path_text(&session_tempdir()?))
}

/// `tempfile(pattern = "file", tmpdir = tempdir(), fileext = "")`: a name for
/// a file that does not exist yet.
pub(crate) fn bi_tempfile(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = |name: &str, pos: usize, default: &str| -> String {
        gn(a, name).or_else(|| a.iter().filter(|x| x.name.is_none()).nth(pos).map(|x| x.value.clone()))
            .map(|v| val_to_str(&v)).unwrap_or_else(|| default.to_string())
    };
    let pattern = text("pattern", 0, "file");
    let dir = match gn(a, "tmpdir") { Some(v) => std::path::PathBuf::from(val_to_str(&v)), None => session_tempdir()? };
    let ext = text("fileext", 2, "");
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    loop {
        let tag = (nanos ^ (std::process::id() as u64) << 20).wrapping_add(NEXT.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9e37_79b9));
        let path = dir.join(format!("{}{:x}{}", pattern, tag & 0xffff_ffff_ffff, ext));
        if !path.exists() { return Ok(path_text(&path)); }
    }
}
