//! Package machinery: `library()` / `require()` / `detach()`,
//! `installed.packages()`, `.libPaths()`, and the disk-loading helpers.
//!
//! Split out of `lib.rs` so the evaluator (the hot path) and the package
//! loader evolve independently — editing one no longer forces the other
//! to re-parse. None of this is on the per-expression hot path; it runs
//! only when a user (re)loads or detaches a package.
//!
//! `use super::*` pulls in the crate-root names (`Engine`, `gv`, `mkpkg`,
//! `rbool`, `registry_tables`, the `r2_types`/`std` imports, etc.) — a
//! child module can name its parent's private items. The `soutln!` macro
//! is in textual scope because this module is declared after it in `lib.rs`.

use super::*;

pub(crate) fn bi_library(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        // library(stats) without quotes — symbol
        _ => return err!(Runtime, "library() needs a package name (character string)"),
    };

    // 1. Check if already loaded and attached
    let already = e.registry.layers.iter().any(|l| l.name == name);
    if already {
        soutln!("package '{}' is already loaded", name);
        return Ok(RVal::Null);
    }

    // 2. Try to re-attach a known base package (compiled into binary)
    let base_result = try_reload_base(e, &name);
    if base_result {
        soutln!("Loading package: '{}'", name);
        // Print masking warnings
        for w in e.drain_warnings() { soutln!("{}", w); }
        return Ok(RVal::Null);
    }

    // 3. Try to load from disk (addon package)
    let loaded = try_load_from_disk(e, &name)?;
    if loaded {
        soutln!("Loading package: '{}'", name);
        for w in e.drain_warnings() { soutln!("{}", w); }
        return Ok(RVal::Null);
    }

    err!(Runtime, "there is no package called '{}'", name)
}

pub(crate) fn bi_require(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    match bi_library(e, a, env) {
        Ok(_) => Ok(rbool(true)),
        Err(e) => {
            soutln!("Warning: {}", e.msg);
            Ok(rbool(false))
        }
    }
}

pub(crate) fn bi_detach(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "detach() needs a package name"),
    };

    // Strip "package:" prefix if present (R compatibility)
    let name = name.strip_prefix("package:").unwrap_or(&name).to_string();

    match e.detach_package(&name) {
        Ok(restored) => {
            soutln!("Detached package: '{}'", name);
            if !restored.is_empty() {
                soutln!("Restored functions: {}", restored.join(", "));
            }
            Ok(RVal::Null)
        }
        Err(msg) => err!(Runtime, "{}", msg),
    }
}

pub(crate) fn bi_installed_packages(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    // Show base packages (always available)
    soutln!("{:<20} {:<10} {}", "Package", "Version", "Tier");


    // Built-in base packages
    let base_pkgs = vec![
        ("base", "0.1.0", "base"),
        ("stats", "0.1.0", "base"),
        ("graphics", "0.1.0", "base"),
        ("utils", "0.1.0", "base"),
        ("datasets", "0.1.0", "base"),
    ];
    for (name, ver, tier) in &base_pkgs {
        let status = if e.registry.layers.iter().any(|l| l.name == *name) { "loaded" } else { "available" };
        soutln!("{:<20} {:<10} {} [{}]", name, ver, tier, status);
    }

    // Installed addons from disk
    for (name, info) in &e.installed {
        let status = if e.registry.layers.iter().any(|l| l.name == *name) { "loaded" } else { "installed" };
        soutln!("{:<20} {:<10} addon [{}]", name, info.version, status);
    }

    // Scan disk for packages not yet discovered
    for lib_path in &e.lib_paths.clone() {
        let path = std::path::Path::new(lib_path);
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let pkg_name = entry.file_name().to_string_lossy().to_string();
                    if !e.installed.contains_key(&pkg_name)
                        && entry.path().join("MANIFEST.toml").exists()
                    {
                        soutln!("{:<20} {:<10} addon [installed]", pkg_name, "?");
                    }
                }
            }
        }
    }

    Ok(RVal::Null)
}

pub(crate) fn bi_lib_paths(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if a.is_empty() {
        // Get: show current paths
        let paths: Vec<Character> = e.lib_paths.iter().map(|p| Some(Arc::from(p.as_str()))).collect();
        Ok(RVal::Character(paths, Attrs::default()))
    } else {
        // Set: update paths
        match &gv(a, 0) {
            RVal::Character(v, _) => {
                e.lib_paths = v.iter().filter_map(|x| x.as_ref().map(|s| s.to_string())).collect();
                soutln!("Library paths updated");
                Ok(RVal::Null)
            }
            _ => err!(Runtime, ".libPaths() needs character vector"),
        }
    }
}

// ── Helper: re-attach a base package that was detached ───────────────

fn try_reload_base(e: &mut Engine, name: &str) -> bool {
    // Tables come from `registry_tables` — the same source `Engine::new()`
    // uses — so a detach+reload can never diverge from startup.
    match name {
        "base" => {
            e.registry.add_layer(mkpkg("base", PackageTier::Base, registry_tables::base_table()));
            true
        }
        "stats" => {
            e.registry.add_layer(mkpkg("stats", PackageTier::Base, registry_tables::stats_table()));
            true
        }
        "graphics" => {
            e.registry.add_layer(mkpkg("graphics", PackageTier::Base, registry_tables::graphics_table()));
            true
        }
        "utils" => {
            e.registry.add_layer(mkpkg("utils", PackageTier::Base, registry_tables::utils_table()));
            true
        }
        _ => false,
    }
}

// ── Helper: load addon package from disk ──────────────────────────────
//
// Package directory structure on disk:
//   ~/.r2/library/mypkg/
//   ├── MANIFEST.toml    # name, version, exports, depends
//   ├── R2/
//   │   ├── functions.r  # R2 source code defining functions
//   │   └── *.r
//   └── data/            # optional datasets
//
// This reads the .r files, parses them, evaluates them to extract
// function definitions, and registers them as a package layer.

fn try_load_from_disk(e: &mut Engine, name: &str) -> Result<bool, R2Err> {
    // Path A — r2-pkg standard layout: ~/.r2/packages/<name>/R/*.r2
    if let Ok(pkg_root) = r2_pkg::pkg_dir(name) {
        if pkg_root.is_dir() && pkg_root.join("package.r2").exists() {
            return load_r2pkg_layout(e, name, &pkg_root);
        }
    }
    // Path B — legacy layout: <lib_path>/<name>/R2/*.r
    for lib_path in &e.lib_paths.clone() {
        let pkg_dir = std::path::Path::new(lib_path).join(name);
        let r2_dir = pkg_dir.join("R2");
        if !r2_dir.is_dir() { continue; }

        // Read all .r files
        let mut all_source = String::new();
        if let Ok(entries) = std::fs::read_dir(&r2_dir) {
            let mut files: Vec<_> = entries.flatten()
                .filter(|e| e.path().extension().map(|ext| ext == "r").unwrap_or(false))
                .collect();
            files.sort_by_key(|e| e.file_name());
            for entry in files {
                match std::fs::read_to_string(entry.path()) {
                    Ok(content) => { all_source.push_str(&content); all_source.push('\n'); }
                    Err(err) => return err!(Runtime, "cannot read {}: {}", entry.path().display(), err),
                }
            }
        }
        if all_source.is_empty() { return err!(Runtime, "package '{}' has no R2 source files", name); }

        // Parse
        let stmts = r2_parser::Parser::parse(&all_source)
            .map_err(|pe| R2Err { msg: format!("error parsing package '{}': {}", name, pe), kind: ErrKind::Runtime })?;

        // Snapshot: record existing global names BEFORE eval
        let before: Vec<Arc<str>> = e.global_env.bindings.read().unwrap().keys().cloned().collect();

        // Evaluate all statements — assignments go directly into global_env
        let env = e.global_env.clone();
        for stmt in &stmts {
            match e.eval_in(stmt, &env) {
                Ok(_) => {}
                Err(err) => {
                    if err.kind != ErrKind::CtrlBreak && err.kind != ErrKind::CtrlNext {
                        eprintln!("Warning in package '{}': {}", name, err.msg);
                    }
                }
            }
        }

        // Diff: find NEW bindings that are closures
        let mut exports = Vec::new();
        for (fname, fval) in e.global_env.bindings.read().unwrap().iter() {
            if !before.contains(fname) && matches!(fval, RVal::Closure(_)) {
                if e.registry.is_core(fname) {
                    return err!(Runtime, "package '{}' cannot mask core function '{}'", name, fname);
                }
                exports.push(fname.to_string());
            }
        }

        if exports.is_empty() {
            return err!(Runtime, "package '{}' defines no functions", name);
        }

        // Register layer for search/detach tracking
        let layer = PackageLayer {
            name: name.to_string(),
            tier: PackageTier::Addon,
            functions: HashMap::new(),
            exports: exports.clone(),
        };
        let masks = e.registry.check_masks(&exports);
        for (func, from) in &masks {
            e.warnings.push(format!("Warning: package '{}' masks '{}' from '{}'", name, func, from));
        }
        e.registry.add_layer(layer);

        e.installed.insert(name.to_string(), InstalledPkgInfo {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            path: pkg_dir.to_string_lossy().to_string(),
            exports,
            depends: Vec::new(),
        });

        return Ok(true);
    }
    Ok(false)
}

// Load a package laid out per the r2-pkg convention:
//   <pkg_root>/package.r2      manifest (required)
//   <pkg_root>/R/*.r2          source files, sourced alphabetically
fn load_r2pkg_layout(e: &mut Engine, name: &str, pkg_root: &std::path::Path) -> Result<bool, R2Err> {
    let manifest = r2_pkg::read_manifest(pkg_root)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let files = r2_pkg::package_source_files(name)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;

    let mut all_source = String::new();
    for f in &files {
        match std::fs::read_to_string(f) {
            Ok(c) => { all_source.push_str(&c); all_source.push('\n'); }
            Err(err) => return err!(Runtime, "cannot read {}: {}", f.display(), err),
        }
    }
    // Empty R/ dir is allowed for metadata-only packages, but library() of one
    // would be a no-op — treat as an error so the user knows nothing happened.
    if all_source.trim().is_empty() {
        return err!(Runtime, "package '{}' has no .r2 source files under R/", name);
    }

    let stmts = r2_parser::Parser::parse(&all_source)
        .map_err(|pe| R2Err { msg: format!("error parsing package '{}': {}", name, pe), kind: ErrKind::Runtime })?;

    let before: Vec<Arc<str>> = e.global_env.bindings.read().unwrap().keys().cloned().collect();
    let methods_before: std::collections::HashSet<(Arc<str>, Arc<str>)> =
        e.methods.keys().cloned().collect();
    let env = e.global_env.clone();
    for stmt in &stmts {
        if let Err(err) = e.eval_in(stmt, &env) {
            if err.kind != ErrKind::CtrlBreak && err.kind != ErrKind::CtrlNext {
                eprintln!("Warning in package '{}': {}", name, err.msg);
            }
        }
    }

    // Determine exports: prefer the manifest's package_exports list if non-empty,
    // otherwise fall back to "every new closure" so authors don't have to maintain
    // the list while iterating.
    // Methods this package defined (name set) — methods live in `e.methods`
    // keyed by (method_name, type_name), not in `global_env`.
    let new_method_names: std::collections::HashSet<String> = e.methods.keys()
        .filter(|k| !methods_before.contains(*k))
        .map(|(m, _)| m.to_string())
        .collect();
    let mut exports: Vec<String> = Vec::new();
    if !manifest.exports.is_empty() {
        for ex in &manifest.exports {
            let key: Arc<str> = Arc::from(ex.as_str());
            // A package may export functions (Closure), types (TypeDef), or
            // methods (registered in e.methods).
            let is_fn_or_type = matches!(
                e.global_env.bindings.read().unwrap().get(&key),
                Some(RVal::Closure(_)) | Some(RVal::TypeDef(_))
            );
            if is_fn_or_type || new_method_names.contains(ex) {
                exports.push(ex.clone());
            } else {
                eprintln!("Warning: package '{}' exports '{}' but it was not defined in any R/ file", name, ex);
            }
        }
    } else {
        // Auto-export: every new function/type binding, plus every method.
        for (fname, fval) in e.global_env.bindings.read().unwrap().iter() {
            if !before.contains(fname) && matches!(fval, RVal::Closure(_) | RVal::TypeDef(_)) {
                exports.push(fname.to_string());
            }
        }
        for m in &new_method_names {
            if !exports.contains(m) { exports.push(m.clone()); }
        }
    }
    if exports.is_empty() {
        return err!(Runtime, "package '{}' defines no exported functions, types, or methods", name);
    }
    for ex in &exports {
        if e.registry.is_core(ex) {
            return err!(Runtime, "package '{}' cannot mask core function '{}'", name, ex);
        }
    }

    let layer = PackageLayer {
        name: name.to_string(),
        tier: PackageTier::Addon,
        functions: HashMap::new(),
        exports: exports.clone(),
    };
    let masks = e.registry.check_masks(&exports);
    for (func, from) in &masks {
        e.warnings.push(format!("Warning: package '{}' masks '{}' from '{}'", name, func, from));
    }
    e.registry.add_layer(layer);
    e.installed.insert(name.to_string(), InstalledPkgInfo {
        name: name.to_string(),
        version: manifest.version.clone(),
        path: pkg_root.to_string_lossy().to_string(),
        exports,
        depends: Vec::new(),
    });
    Ok(true)
}
