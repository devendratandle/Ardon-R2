//! System utilities, statistical tests, package installers, and
//! simple model accessors — extracted from lib.rs (engine-split,
//! opus-4.8 session, content-anchored).
//!
//! Covers: source(), system.time(), t.test(), chisq.test(), the
//! R.S.4 addon installers, the unified install.packages() dispatcher,
//! predict()/residuals() for lm objects, glm() IRLS, and confint().
//!
//! Module-private helpers `verify_name` and `find_pkg_root` are used
//! only here. Dead `#[cfg(any())]` legacy SVG bodies travel along but
//! never compile.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::all)]
#![allow(dead_code)]

use std::sync::Arc;

use r2_stats::dist::qnorm_approx;
use r2_types::*;

use crate::{gv, gn, val_to_str, Engine};
use crate::err;

pub(crate) fn bi_source(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).ok_or(R2Err{msg:"NA path".into(),kind:ErrKind::Runtime})?, _ => return err!(Runtime, "source() needs file path") };
    let content = std::fs::read_to_string(&path).map_err(|e| R2Err{msg:format!("cannot read '{}': {}", path, e),kind:ErrKind::Runtime})?;
    let stmts = r2_parser::Parser::parse(&content).map_err(|pe| R2Err{msg:format!("parse error in '{}': {}", path, pe),kind:ErrKind::Runtime})?;
    let mut last = RVal::Null;
    for stmt in &stmts {
        last = e.eval_in(stmt, env)?;
    }
    Ok(last)
}

// ═══════════════════════════════════════════════════════════════════════
// system.time() — measure execution time
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_system_time(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let func = gv(a,0);
    let start = std::time::Instant::now();
    let _ = e.call_fn(&func, &[], env)?;
    let elapsed = start.elapsed();
    println!("   user  system elapsed");
    println!("  {:.3}   0.000   {:.3}", elapsed.as_secs_f64(), elapsed.as_secs_f64());
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// t.test() — Student's t-test
// ═══════════════════════════════════════════════════════════════════════


// Phase R.10: t_cdf, incomplete_beta, gamma_approx live in r2_stats.
// All engine call sites migrated; imports retired.

// ═══════════════════════════════════════════════════════════════════════
// chisq.test() — Chi-squared test for independence
// ═══════════════════════════════════════════════════════════════════════


// Phase R.S.2 — Hotelling T² (one-sample / two-sample / paired)
pub(crate) fn bi_hotelling_test(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::multivariate::bi_hotelling_test(a)
}
// Phase R.S.2 — MANOVA (multivariate analysis of variance)
pub(crate) fn bi_manova(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::multivariate::bi_manova(a)
}
// Phase R.S.3 — linear mixed-effects model (random intercept)
pub(crate) fn bi_lmer(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::mixed::bi_lmer(a)
}

// ═══════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════
// Phase R.T.1 — Date / POSIXct builtins (thin engine adapters)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_as_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_as_date(a)
}
pub(crate) fn bi_as_posixct(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_as_posixct(a)
}
pub(crate) fn bi_format_time(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_format_time(a)
}
pub(crate) fn bi_sys_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_sys_date(a)
}
pub(crate) fn bi_sys_time(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_sys_time(a)
}
pub(crate) fn bi_difftime(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_stats::time::bi_difftime(a)
}
// ── Phase R.T.2 — ts() class adapters ─────────────────────────────────
pub(crate) fn bi_tsp(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_tsp(a) }
pub(crate) fn bi_ts_start(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_start(a) }
pub(crate) fn bi_ts_end(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_end(a) }
pub(crate) fn bi_frequency(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_frequency(a) }
pub(crate) fn bi_deltat(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_deltat(a) }
pub(crate) fn bi_time_idx(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_time(a) }
pub(crate) fn bi_cycle(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_cycle(a) }
pub(crate) fn bi_window(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_window(a) }
pub(crate) fn bi_is_ts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_ts(a) }
// ── Phase R.T.3 — xts adapters ────────────────────────────────────────
pub(crate) fn bi_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_xts(a) }
pub(crate) fn bi_index(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_index(a) }
pub(crate) fn bi_coredata(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_coredata(a) }
pub(crate) fn bi_is_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_xts(a) }
pub(crate) fn bi_xts_subset(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_xts_subset(a) }
pub(crate) fn bi_first(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_first(a) }
pub(crate) fn bi_last(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_last(a) }
pub(crate) fn bi_na_locf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_na_locf(a) }
pub(crate) fn bi_merge_xts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_merge_xts(a) }
// ── Phase R.T.4 — TS analytics ─────────────────────────────────────────
pub(crate) fn bi_acf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_acf(a) }
pub(crate) fn bi_pacf(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_pacf(a) }
pub(crate) fn bi_decompose(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_decompose(a) }
pub(crate) fn bi_is_regular(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_is_regular(a) }
pub(crate) fn bi_periodicity(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_periodicity(a) }
pub(crate) fn bi_lag(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_lag(a) }
pub(crate) fn bi_diff_ts(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_diff_ts(a) }
// ── Phase R.T.5 — period aggregation ────────────────────────────────
pub(crate) fn bi_to_daily(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_daily(a) }
pub(crate) fn bi_to_weekly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_weekly(a) }
pub(crate) fn bi_to_monthly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_monthly(a) }
pub(crate) fn bi_to_quarterly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_quarterly(a) }
pub(crate) fn bi_to_yearly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_to_yearly(a) }
pub(crate) fn bi_apply_daily(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_daily(a) }
pub(crate) fn bi_apply_weekly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_weekly(a) }
pub(crate) fn bi_apply_monthly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_monthly(a) }
pub(crate) fn bi_apply_quarterly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_quarterly(a) }
pub(crate) fn bi_apply_yearly(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_apply_yearly(a) }
// ── Phase R.T.5b — Hindu calendar ───────────────────────────────────
pub(crate) fn bi_tithi(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_tithi(a) }
pub(crate) fn bi_hindu_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_hindu_date(a) }
pub(crate) fn bi_hnc_date(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::time::bi_hnc_date(a) }

// ═══════════════════════════════════════════════════════════════════════
// Phase R.S.4 — Addon installers (r2-pkg backed)
//
// install.from.dir("path/to/pkg")
//   Validates package.r2 manifest and copies into ~/.r2/packages/<name>/
//
// install.from.zip("path/to/pkg.zip")
//   Extracts into a tmp dir then delegates to install_from_dir. Uses the
//   system `unzip` (or `tar -xf` on Windows 10+) — no Rust zip dep.
//
// install.from.github("user/repo" [, ref="main"])
//   Shells out to `git clone --depth 1` into a tmp dir, then installs.
//
// uninstall("name")
//   Removes the package directory; library() of it afterwards will fail.
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_install_from_dir(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let path = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA path".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.dir() needs a path (character string)"),
    };
    let m = r2_pkg::install_from_dir(std::path::Path::new(&path))
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, path);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

pub(crate) fn bi_install_from_zip(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let zip_path = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA path".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.zip() needs a zip path"),
    };
    let zip = std::path::Path::new(&zip_path);
    if !zip.exists() {
        return err!(Runtime, "zip file not found: {}", zip_path);
    }
    // Extract to a tmp directory using the platform's unzip/tar tool.
    let tmp = std::env::temp_dir().join(format!("r2-install-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)
        .map_err(|e| R2Err { msg: format!("cannot create tmp dir: {}", e), kind: ErrKind::Runtime })?;

    // Try `tar -xf` first (available on Windows 10+, macOS, Linux).
    let status = std::process::Command::new("tar")
        .arg("-xf").arg(zip)
        .arg("-C").arg(&tmp)
        .status();
    if !matches!(&status, Ok(s) if s.success()) {
        // Fall back to `unzip`.
        let status2 = std::process::Command::new("unzip")
            .arg("-q").arg(zip).arg("-d").arg(&tmp)
            .status();
        if !matches!(status2, Ok(s) if s.success()) {
            let _ = std::fs::remove_dir_all(&tmp);
            return err!(Runtime, "could not extract '{}' — neither `tar` nor `unzip` succeeded. Install one or extract manually then use install.from.dir().", zip_path);
        }
    }

    // Find the package root inside tmp (top-level dir containing package.r2,
    // OR tmp itself if the zip was extracted flat).
    let src_dir = find_pkg_root(&tmp)
        .ok_or_else(|| R2Err { msg: format!("no package.r2 found inside '{}'", zip_path), kind: ErrKind::Runtime })?;

    let m = r2_pkg::install_from_dir(&src_dir)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let _ = std::fs::remove_dir_all(&tmp);
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, zip_path);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

pub(crate) fn bi_install_from_github(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let spec = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA repo".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.from.github() needs a repo spec like \"user/repo\""),
    };
    // Optional ref="branch_or_tag" arg.
    let git_ref: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("ref"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let url = if spec.starts_with("http://") || spec.starts_with("https://") || spec.starts_with("git@") {
        spec.clone()
    } else if spec.contains('/') {
        format!("https://github.com/{}.git", spec)
    } else {
        return err!(Runtime, "github spec must be \"user/repo\" or a full git URL; got '{}'", spec);
    };

    let tmp = std::env::temp_dir().join(format!("r2-gh-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);

    let mut cmd = std::process::Command::new("git");
    cmd.arg("clone").arg("--depth").arg("1");
    if let Some(r) = &git_ref {
        cmd.arg("--branch").arg(r);
    }
    cmd.arg(&url).arg(&tmp);
    let status = cmd.status()
        .map_err(|e| R2Err { msg: format!("cannot run `git`: {}. Install git and put it on PATH.", e), kind: ErrKind::Runtime })?;
    if !status.success() {
        return err!(Runtime, "git clone failed for '{}'", url);
    }

    // Optional subdir = "r2pkg-foo" arg — for monorepos where many
    // packages live under one repo (e.g. Ardon-R2-libraries).
    let subdir: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("subdir"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let search_root = match &subdir {
        Some(s) => tmp.join(s),
        None    => tmp.clone(),
    };
    if !search_root.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
        return err!(Runtime, "subdir '{}' not found in cloned repo '{}'", subdir.unwrap_or_default(), spec);
    }

    let src_dir = find_pkg_root(&search_root)
        .ok_or_else(|| R2Err { msg: format!("no package.r2 found in cloned repo '{}'{}", spec, subdir.as_ref().map(|s| format!(" subdir '{}'", s)).unwrap_or_default()), kind: ErrKind::Runtime })?;

    let m = r2_pkg::install_from_dir(&src_dir)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    let _ = std::fs::remove_dir_all(&tmp);
    let from_str = match subdir { Some(s) => format!("github:{}/{}", spec, s), None => format!("github:{}", spec) };
    println!("* installing package '{}' (version {}) from {}", m.name, m.version, from_str);
    println!("* DONE — load with library(\"{}\")", m.name);
    Ok(RVal::Null)
}

pub(crate) fn bi_uninstall(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "uninstall() needs a package name"),
    };
    r2_pkg::uninstall(&name)
        .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
    println!("* removed package '{}'", name);
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// Unified `install.packages()` dispatcher
//
// One entry point that auto-detects the install source from the `path`
// argument. Mirrors R's `install.packages()` for muscle memory, and
// future-compatible with a website-based registry.
//
// Signature:
//   install.packages(name, path = NULL, subdir = NULL, ref = NULL)
//
// `path` dispatch rules (checked in order):
//
//   1. NULL / omitted        → registry lookup (not yet implemented;
//                              returns an informative error pointing
//                              at the future r2-packages.dev endpoint)
//   2. ends with ".zip"      → local zip extract → install
//                              (or download-then-extract if it's a URL)
//   3. "user/repo[/subdir]"  → github shorthand: clone, then install
//                              (a single `/`, no path separator, no
//                              backslash, no dot before the slash)
//   4. starts with "http://" or "https://" or "git@" or ends ".git"
//                            → git clone or URL download
//   5. an existing directory → install.from.dir
//
// `name` is the expected package name; after extracting/cloning we
// verify it matches the manifest's `package_name` and warn on mismatch.
// `subdir` (optional) — for monorepos like Ardon-R2-libraries.
// `ref`    (optional) — branch/tag for GitHub installs.
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_install_packages(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a, 0) {
        RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string())
            .ok_or(R2Err { msg: "NA package name".into(), kind: ErrKind::Runtime })?,
        _ => return err!(Runtime, "install.packages() needs a package name (character string)"),
    };
    let path: Option<String> = a.iter()
        .find(|x| x.name.as_deref() == Some("path"))
        .and_then(|x| match &x.value {
            RVal::Character(v, _) => v[0].as_ref().map(|s| s.to_string()),
            _ => None,
        });

    let path = match path {
        Some(p) => p,
        None => {
            return err!(Runtime,
                "install.packages('{}') without 'path' would consult the R2 registry, \
                 which is not yet online. Until then, supply path = ... pointing to: \
                 \n  - a local directory:        path = 'path/to/pkg-dir'\
                 \n  - a local zip file:         path = 'path/to/pkg.zip'\
                 \n  - a GitHub repo:            path = 'user/repo'\
                 \n  - a monorepo subdir:        path = 'user/repo', subdir = 'r2pkg-foo'\
                 \n  - a specific branch/tag:    path = 'user/repo', ref = 'v0.1.0'", name);
        }
    };

    // Classify the path. Order matters: zip before github-shorthand
    // (since "foo/bar.zip" contains a slash too).
    let is_url   = path.starts_with("http://") || path.starts_with("https://");
    let is_zip   = path.to_lowercase().ends_with(".zip");
    let is_git   = path.starts_with("git@") || path.ends_with(".git");
    let is_dir   = std::path::Path::new(&path).is_dir();
    // GitHub shorthand: one slash, no backslash, no path-separator
    // characters past the slash, and the part before the slash has no
    // dots (would otherwise look like a domain).
    let is_github_shorthand = {
        let p = path.trim_end_matches('/');
        let slashes: Vec<usize> = p.match_indices('/').map(|(i, _)| i).collect();
        !is_url && !is_git && !is_zip && !is_dir &&
            slashes.len() >= 1 && !p.contains('\\') &&
            !p[..slashes[0]].contains('.')
    };

    // Build a faux EvalArg list to call our existing helpers.
    let make_arg = |val: RVal| EvalArg { name: None, value: val };

    if is_dir {
        let m = r2_pkg::install_from_dir(std::path::Path::new(&path))
            .map_err(|e| R2Err { msg: format!("{}", e), kind: ErrKind::Runtime })?;
        verify_name(&m.name, &name);
        println!("* installed '{}' (version {}) from local dir {}", m.name, m.version, path);
        return Ok(RVal::Null);
    }

    if is_zip {
        // Existing handler accepts a local path. URL-zip support would
        // need a download step; for now we error if the user passed a URL.
        if is_url {
            return err!(Runtime, "install.packages(): downloading remote .zip is not yet supported. \
                Download manually then pass the local path.");
        }
        let zip_args = vec![make_arg(RVal::Character(vec![Some(std::sync::Arc::from(path.as_str()))], Attrs::default()))];
        return bi_install_from_zip(e, &zip_args, env);
    }

    if is_github_shorthand || is_git || is_url {
        // Reuse install.from.github for the heavy lifting. Pass through
        // subdir and ref if present.
        let mut gh_args = vec![make_arg(RVal::Character(vec![Some(std::sync::Arc::from(path.as_str()))], Attrs::default()))];
        for arg_name in &["ref", "subdir"] {
            if let Some(x) = a.iter().find(|x| x.name.as_deref() == Some(*arg_name)) {
                gh_args.push(EvalArg { name: Some(std::sync::Arc::from(*arg_name)), value: x.value.clone() });
            }
        }
        return bi_install_from_github(e, &gh_args, env);
    }

    err!(Runtime,
        "install.packages(): could not classify path '{}'. Expected one of:\
         \n  - existing local directory\
         \n  - .zip file\
         \n  - GitHub shorthand 'user/repo'\
         \n  - full URL (https://... or git@...)", path)
}

fn verify_name(manifest_name: &str, requested_name: &str) {
    if manifest_name != requested_name {
        eprintln!(
            "Warning: package name mismatch — you asked for '{}' but the manifest \
             says '{}'. Installed under '{}'.",
            requested_name, manifest_name, manifest_name);
    }
}

// Search up to 2 levels deep for a directory containing package.r2.
fn find_pkg_root(start: &std::path::Path) -> Option<std::path::PathBuf> {
    if start.join("package.r2").exists() { return Some(start.to_path_buf()); }
    if let Ok(entries) = std::fs::read_dir(start) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("package.r2").exists() {
                return Some(p);
            }
        }
    }
    None
}

// Phase R.10: chi_sq_cdf and ln_gamma moved to r2_stats. No engine callers remain.

// ═══════════════════════════════════════════════════════════════════════
// predict() and residuals() for lm objects
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_predict(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let model = gv(a, 0);
    let newdata = gn(a, "newdata").or(Some(gv(a, 1)));

    match &model {
        RVal::TypeInstance(inst) => {
            match inst.type_name.as_ref() {
                "lm" | "glm" => {
                    // If newdata provided, compute X*beta; else return fitted
                    if let Some(RVal::Matrix(xnew)) = &newdata {
                        let coeffs: Vec<f64> = e.as_reals(inst.fields.get("coefficients").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let p = coeffs.len();
                        let n = xnew.nrow;
                        let mut preds = vec![0.0; n];
                        for i in 0..n {
                            preds[i] = coeffs[0]; // intercept
                            for j in 1..p.min(xnew.ncol + 1) {
                                preds[i] += coeffs[j] * xnew.get(i, j - 1);
                            }
                        }
                        // Apply link function for glm
                        if inst.type_name.as_ref() == "glm" {
                            let family = inst.fields.get("family").map(|v| val_to_str(v)).unwrap_or("gaussian".into());
                            for p in preds.iter_mut() {
                                *p = match family.as_str() {
                                    "binomial" => 1.0 / (1.0 + (-*p).exp()),
                                    "poisson" => p.exp(),
                                    _ => *p,
                                };
                            }
                        }
                        Ok(rnums(&preds))
                    } else {
                        inst.fields.get("fitted.values").cloned().ok_or(R2Err{msg:"no fitted values".into(),kind:ErrKind::Runtime})
                    }
                }
                "rpart" => {
                    // Predict using serialized tree
                    if let Some(RVal::Matrix(xnew)) = &newdata {
                        let feat: Vec<f64> = e.as_reals(inst.fields.get("_tree_feat").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let thresh: Vec<f64> = e.as_reals(inst.fields.get("_tree_thresh").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let pred: Vec<f64> = e.as_reals(inst.fields.get("_tree_pred").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let leaf: Vec<f64> = e.as_reals(inst.fields.get("_tree_leaf").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let left: Vec<f64> = e.as_reals(inst.fields.get("_tree_left").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();
                        let right: Vec<f64> = e.as_reals(inst.fields.get("_tree_right").unwrap_or(&RVal::Null))?.into_iter().filter_map(|x| x).collect();

                        let mut preds = Vec::new();
                        for i in 0..xnew.nrow {
                            let mut node = 0usize;
                            loop {
                                if node >= leaf.len() || leaf[node] == 1.0 { break; }
                                let f = feat[node] as usize;
                                let t = thresh[node];
                                if xnew.get(i, f) <= t { node = left[node] as usize; }
                                else { node = right[node] as usize; }
                            }
                            preds.push(if node < pred.len() { pred[node] } else { 0.0 });
                        }
                        Ok(rnums(&preds))
                    } else {
                        inst.fields.get("predictions").cloned().ok_or(R2Err{msg:"no predictions".into(),kind:ErrKind::Runtime})
                    }
                }
                "rf" | "kmeans" | "naive.bayes" | "gbm" => {
                    inst.fields.get("predictions").or(inst.fields.get("cluster")).cloned()
                        .ok_or(R2Err{msg:"no predictions".into(),kind:ErrKind::Runtime})
                }
                _ => err!(Runtime, "predict() does not support {} objects", inst.type_name),
            }
        }
        _ => err!(Runtime, "predict() needs a model object (lm, glm, rpart, rf, kmeans)"),
    }
}

pub(crate) fn bi_residuals(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" => {
            inst.fields.get("residuals").cloned().ok_or(R2Err{msg:"no residuals".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "residuals() needs an lm object"),
    }
}

pub(crate) fn bi_fitted(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" => {
            inst.fields.get("fitted.values").cloned().ok_or(R2Err{msg:"no fitted values".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "fitted() needs an lm object"),
    }
}

pub(crate) fn bi_coef(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" || inst.type_name.as_ref() == "glm" => {
            inst.fields.get("coefficients").cloned().ok_or(R2Err{msg:"no coefficients".into(),kind:ErrKind::Runtime})
        }
        _ => err!(Runtime, "coef() needs an lm or glm object"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// boxplot() — SVG box-and-whisker plot
// ═══════════════════════════════════════════════════════════════════════

// (bi_boxplot moved to src/builtins/graphics.rs. The dead-body
// fallback below was #[cfg(any())] — never compiled — and is
// deleted as part of the migration cleanup.)

#[cfg(any())]
#[allow(dead_code, unused_variables)]
fn _deleted_legacy_bi_boxplot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let title = gn(a,"main").map(|v| val_to_str(&v)).unwrap_or("Boxplot".into());
    let (w, h) = (500.0, 400.0);
    let (ml, mr, mt, mb) = (60.0, 30.0, 30.0, 40.0);
    let pw = w - ml - mr; let ph = h - mt - mb;

    // Collect each argument as a data group
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for (gi, arg) in a.iter().enumerate() {
        if arg.name.as_ref().map(|n| n.as_ref()) == Some("main") { continue; }
        let data: Vec<f64> = e.as_reals(&arg.value)?.into_iter().filter_map(|x| x).collect();
        let name = arg.name.as_ref().map(|n| n.to_string()).unwrap_or(format!("V{}", gi + 1));
        groups.push((name, data));
    }

    if groups.is_empty() { return err!(Runtime, "boxplot needs data"); }

    // Find global range
    let all_min = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::INFINITY, f64::min);
    let all_max = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (all_max - all_min).abs() < 1e-10 { 1.0 } else { all_max - all_min };

    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#, w, h, w, h);
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(r#"<text x="{}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#, w/2.0, title));

    let ng = groups.len() as f64;
    let bw = pw / ng * 0.6;
    let gap = pw / ng;

    for (i, (name, data)) in groups.iter().enumerate() {
        if data.len() < 2 { continue; }
        let mut sorted = data.clone();
        sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q1 = sorted[n / 4]; let median = sorted[n / 2]; let q3 = sorted[3 * n / 4];
        let min_val = sorted[0]; let max_val = sorted[n - 1];
        let iqr = q3 - q1;
        let lower_fence = (q1 - 1.5 * iqr).max(min_val);
        let upper_fence = (q3 + 1.5 * iqr).min(max_val);

        let cx = ml + gap * i as f64 + gap / 2.0;
        let map_y = |v: f64| mt + ph - (v - all_min) / range * ph;

        // Whiskers
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(lower_fence), cx, map_y(q1)));
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx, map_y(q3), cx, map_y(upper_fence)));
        // Fence caps
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx-bw/4.0, map_y(lower_fence), cx+bw/4.0, map_y(lower_fence)));
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black"/>"#, cx-bw/4.0, map_y(upper_fence), cx+bw/4.0, map_y(upper_fence)));
        // Box
        let by = map_y(q3); let bh = map_y(q1) - by;
        svg.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="{:.0}" height="{:.0}" fill="{}" stroke="black"/>"#, cx-bw/2.0, by, bw, bh, "#93c5fd"));
        // Median line
        svg.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="black" stroke-width="2"/>"#, cx-bw/2.0, map_y(median), cx+bw/2.0, map_y(median)));
        // Label
        svg.push_str(&format!(r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10">{}</text>"#, cx, h-mb+15.0, name));
    }
    svg.push_str("</svg>");
    let _ = std::fs::write("boxplot.svg", &svg);
    println!("Boxplot saved to boxplot.svg");
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// barplot() — SVG bar chart
// ═══════════════════════════════════════════════════════════════════════


#[cfg(any())]
#[allow(dead_code, unused_variables)]
fn _legacy_bi_barplot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let heights: Vec<f64> = e.as_reals(&gv(a,0))?.into_iter().filter_map(|x| x).collect();
    let title = gn(a,"main").map(|v| val_to_str(&v)).unwrap_or("Barplot".into());
    let names = gn(a,"names.arg");

    let labels: Vec<String> = if let Some(RVal::Character(v, _)) = &names {
        v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()
    } else {
        (1..=heights.len()).map(|i| format!("{}", i)).collect()
    };

    let (w, h) = (600.0, 400.0);
    let (ml, mr, mt, mb) = (60.0, 20.0, 30.0, 50.0);
    let pw = w - ml - mr; let ph = h - mt - mb;
    let max_h = heights.iter().cloned().fold(0.0f64, f64::max);
    let bw = pw / heights.len() as f64 * 0.8;
    let gap = pw / heights.len() as f64;

    let colors = vec!["#3b82f6","#ef4444","#22c55e","#f59e0b","#8b5cf6","#ec4899","#06b6d4","#f97316"];

    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#, w, h, w, h);
    svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
    svg.push_str(&format!(r#"<text x="{}" y="18" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#, w/2.0, title));

    for (i, &val) in heights.iter().enumerate() {
        let bh = if max_h > 0.0 { val / max_h * ph } else { 0.0 };
        let bx = ml + gap * i as f64 + (gap - bw) / 2.0;
        let by = mt + ph - bh;
        let color = colors[i % colors.len()];
        svg.push_str(&format!(r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}"/>"#, bx, by, bw, bh, color));
        // Value on top
        svg.push_str(&format!(r#"<text x="{:.0}" y="{:.0}" text-anchor="middle" font-size="10">{:.1}</text>"#, bx+bw/2.0, by-5.0, val));
        // Label below
        let label = labels.get(i).map(|s| s.as_str()).unwrap_or("");
        svg.push_str(&format!(r#"<text x="{:.0}" y="{}" text-anchor="middle" font-size="10" transform="rotate(-30,{:.0},{})">{}</text>"#, bx+bw/2.0, h-mb+20.0, bx+bw/2.0, h-mb+20.0, label));
    }
    svg.push_str("</svg>");
    let _ = std::fs::write("barplot.svg", &svg);
    println!("Barplot saved to barplot.svg");
    Ok(RVal::Null)
}

// ═══════════════════════════════════════════════════════════════════════
// read.table(), write.table(), read.delim() — more file I/O
// ═══════════════════════════════════════════════════════════════════════




// ═══════════════════════════════════════════════════════════════════════
// which.min(), which.max(), range(), prod(), any(), all()
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_which_min(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_which_min(a) }

pub(crate) fn bi_which_max(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_which_max(a) }

pub(crate) fn bi_range(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::summary::bi_range(a) }


pub(crate) fn bi_any(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a,0))?;
    Ok(rbool(v.iter().any(|x| *x == Some(true))))
}

pub(crate) fn bi_all(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let v = e.as_logicals(&gv(a,0))?;
    Ok(rbool(v.iter().all(|x| *x == Some(true))))
}

// ═══════════════════════════════════════════════════════════════════════
// sprintf(), trimws(), startsWith(), endsWith(), nrow/ncol for matrix
// ═══════════════════════════════════════════════════════════════════════



pub(crate) fn bi_starts_with(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = match &gv(a,0) { RVal::Character(v,_) => v.clone(), _ => return err!(Type, "startsWith needs character") };
    let prefix = match &gv(a,1) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Type, "startsWith needs prefix") };
    let result: Vec<Logical> = x.iter().map(|s| s.as_ref().map(|s| s.starts_with(&prefix.as_str()))).collect();
    Ok(RVal::Logical(result.into(), Attrs::default()))
}

pub(crate) fn bi_ends_with(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let x = match &gv(a,0) { RVal::Character(v,_) => v.clone(), _ => return err!(Type, "endsWith needs character") };
    let suffix = match &gv(a,1) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Type, "endsWith needs suffix") };
    let result: Vec<Logical> = x.iter().map(|s| s.as_ref().map(|s| s.ends_with(&suffix.as_str()))).collect();
    Ok(RVal::Logical(result.into(), Attrs::default()))
}

pub(crate) fn bi_Sys_time(_: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    Ok(rnum(now))
}

pub(crate) fn bi_stop(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    err!(Runtime, "{}", msg)
}

pub(crate) fn bi_warning(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    e.warnings.push(format!("Warning: {}", msg));
    Ok(RVal::Null)
}

pub(crate) fn bi_message(_: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let msg = val_to_str(&gv(a,0));
    eprintln!("{}", msg);
    Ok(RVal::Null)
}

pub(crate) fn bi_ls(e: &mut Engine, _a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let names: Vec<Character> = e.global_env.bindings.keys()
        .map(|k| Some(k.clone()))
        .collect();
    Ok(RVal::Character(names, Attrs::default()))
}

pub(crate) fn bi_rm(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Runtime, "rm needs name") };
    let mut binding = e.global_env.clone();
    let g = Arc::make_mut(&mut binding);
    g.bindings.remove(name.as_str());
    e.global_env = Arc::new(g.clone());
    Ok(RVal::Null)
}

pub(crate) fn bi_exists(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = match &gv(a,0) { RVal::Character(v,_) => v[0].as_ref().map(|s| s.to_string()).unwrap_or_default(), _ => return err!(Runtime, "exists needs name") };
    Ok(rbool(e.global_env.lookup(&name).is_some() || e.registry.resolve(&name).is_some()))
}

// ═══════════════════════════════════════════════════════════════════════
// glm() — Generalized Linear Model (logistic regression via IRLS)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_glm(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> { r2_stats::models::bi_glm(a) }

// ═══════════════════════════════════════════════════════════════════════
// confint() — confidence intervals for model coefficients
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn bi_confint(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let level = gn(a, "level").and_then(|v| e.scalar_f64(&v).ok().flatten()).unwrap_or(0.95);
    match &gv(a,0) {
        RVal::TypeInstance(inst) if inst.type_name.as_ref() == "lm" || inst.type_name.as_ref() == "glm" => {
            let coeffs_val = inst.fields.get("coefficients").ok_or(R2Err{msg:"no coefficients".into(),kind:ErrKind::Runtime})?;
            let coeffs = e.as_reals(coeffs_val)?.into_iter().filter_map(|x| x).collect::<Vec<f64>>();
            let sigma = inst.fields.get("sigma").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(1.0);
            let df = inst.fields.get("df").and_then(|v| e.scalar_f64(v).ok().flatten()).unwrap_or(30.0);

            let alpha = 1.0 - level;
            let t_crit = qnorm_approx(1.0 - alpha / 2.0); // approximate

            let names: Vec<String> = match coeffs_val {
                RVal::Numeric(_, attrs) => attrs.names.as_ref().map(|n| n.iter().map(|s| s.to_string()).collect()).unwrap_or_else(|| (0..coeffs.len()).map(|i| format!("x{}", i)).collect()),
                _ => (0..coeffs.len()).map(|i| format!("x{}", i)).collect(),
            };

            // Standard errors (simplified — assumes diagonal of (X'X)^-1 * sigma^2)
            let se = sigma / (df.sqrt());

            println!("{:>15} {:>15} {:>15}", "", fmt_num(alpha/2.0*100.0).to_string() + " %", fmt_num((1.0-alpha/2.0)*100.0).to_string() + " %");
            for (i, name) in names.iter().enumerate() {
                let lo = coeffs[i] - t_crit * se;
                let hi = coeffs[i] + t_crit * se;
                println!("{:>15} {:>15} {:>15}", name, fmt_num(lo), fmt_num(hi));
            }
            Ok(RVal::Null)
        }
        _ => err!(Runtime, "confint() needs lm or glm object"),
    }
}
