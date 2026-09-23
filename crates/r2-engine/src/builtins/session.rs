//! The session: working directory, waiting (`Sys.sleep`, `readline`),
//! saving and loading the workspace (`save` / `load`), and `version()`.

#![allow(clippy::all)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;
use crate::env_insert;

// ── working directory ────────────────────────────────────────────────

pub(crate) fn bi_getwd(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let cwd = std::env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    Ok(rstr(&cwd))
}

pub(crate) fn bi_setwd(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = val_to_str(&gv(a,0));
    std::env::set_current_dir(&path).map_err(|e| R2Err{msg:format!("cannot set working directory: {}", e),kind:ErrKind::Runtime})?;
    Ok(rstr(&path))
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
            if matches!(name.as_ref(), "iris" | "mtcars" | "airquality") { continue; }
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

pub(crate) fn bi_load(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = gn(a,"file").or(Some(gv(a,0))).map(|v| val_to_str(&v)).unwrap_or("session.r2s".into());
    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot load '{}': {}", path, e),kind:ErrKind::Runtime})?;

    let ext = path.rsplit('.').next().unwrap_or("");
    let mut count = 0;

    for line in content.lines() {
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some(eq_pos) = line.find('=') {
            let name = &line[..eq_pos];
            let val_str = &line[eq_pos+1..];
            if let Some(val) = deserialize_rval(val_str) {
                if name == "_obj" {
                    // Single-object file — return immediately with the value.
                    let kind = match ext { "r2m" => "model", "r2d" => "data", _ => "object" };
                    soutln!("Loaded {} ({}) from '{}'", kind, val.type_name(), path);
                    return Ok(val);
                }
                env_insert(&mut e.global_env, Arc::from(name), val);
                count += 1;
            }
        }
    }
    soutln!("Loaded {} objects from '{}'", count, path);
    Ok(RVal::Null)
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
    soutln!("\nArdon-R2 — Statistical Computing, Reimagined");
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
