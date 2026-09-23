//! The version is written once, in the root Cargo.toml. The one thing
//! that cannot be derived from it is the CHANGELOG entry, so a release
//! without one fails here instead of shipping.

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
