//! The version is written once, in the root Cargo.toml, and the builtin
//! count lives in the registration tables. Prose cannot be derived from
//! either, so the documents that state them are checked here: a version
//! bump or a new builtin fails the gate until they say so. (llms.txt sat
//! at "0.4.0, 438 builtins" for a week after both had moved.)

fn doc(name: &str) -> String {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

/// llms.txt is what AI assistants read about the project; it must name
/// the version the code is at.
#[test]
fn llms_txt_names_this_version() {
    let v = format!("v{}", env!("CARGO_PKG_VERSION"));
    assert!(doc("llms.txt").contains(&v), "llms.txt does not mention {v} — update its version lines");
}

/// The live count (what `version()` prints) in the documents that state it.
#[test]
fn documents_state_the_live_builtin_count() {
    let n = r2_engine::Engine::new().registry.n_functions();
    let want = format!("{n} built-in functions");
    for name in ["llms.txt", "FUNCTIONS.md"] {
        assert!(doc(name).to_lowercase().contains(&want.to_lowercase()),
            "{name} does not say `{want}` — the registry now has {n}; update the document");
    }
}

#[test]
fn changelog_has_an_entry_for_this_version() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../CHANGELOG.md");
    let log = std::fs::read_to_string(path).expect("read CHANGELOG.md");
    let heading = format!("## v{} ", env!("CARGO_PKG_VERSION"));
    assert!(
        log.lines().any(|l| l.starts_with(&heading)),
        "CHANGELOG.md has no `{heading}(...)` heading — add the release notes"
    );
}
