// build.rs — embed the same multi-resolution R2 icon and version
// metadata into R2Gui.exe that r2-repl's build.rs already embeds into
// r2.exe. Without this, the GUI binary appears unbranded in Explorer,
// the taskbar, and Alt-Tab.
//
// Pipeline matches r2-repl/build.rs: assets/logo.png is converted to
// installer/r2.ico (multi-resolution PNG-in-ICO) on every release
// build. If that file already exists (which it will because r2-repl's
// build.rs ran first), we reuse it instead of regenerating.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir.parent().unwrap().parent().unwrap();
    let ico_path = project_root.join("installer").join("r2.ico");
    println!("cargo:rerun-if-changed={}", ico_path.display());

    if !ico_path.exists() {
        println!(
            "cargo:warning=installer/r2.ico not found — build r2-repl first \
             (`cargo build --release -p r2-repl`) so it generates the icon, \
             then rebuild r2-gui. Continuing without an embedded icon."
        );
        return;
    }

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon(ico_path.to_str().unwrap());
        res.set("ProductName",     "Ardon-R2");
        res.set("FileDescription", "Ardon-R2 — desktop GUI");
        res.set("CompanyName",     "Devendra Tandale");
        res.set("LegalCopyright",  "AGPL-3.0");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed icon resource: {}", e);
        }
    }
    #[cfg(not(windows))]
    {
        let _ = ico_path;
    }
}
