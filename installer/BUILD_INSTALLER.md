# Building the Ardon-R2 Windows installer

Three-step process. The output is a single ~15 MB `.exe` users can double-click
to install Ardon-R2 anywhere on their machine.

---

## Step 1 — Build the release binary

```bash
cd E:\R2_Rust _opus4.6\r2\r2
cargo build --release -p r2-repl -p r2-gui
```

This produces `target/release/r2.exe` and `target/release/R2Gui.exe`
(~15 MB and ~21 MB with the optimized release profile in `Cargo.toml`).
Do NOT use the no-LTO override here — that is for the test gate; the
shipped binaries are the fat-LTO build.

The profile uses:

| Setting | Effect |
|---|---|
| `strip = "symbols"` | Drops debug symbols, saves ~3 MB |
| `lto = "fat"` | Cross-crate inlining, ~10–20% smaller and faster |
| `codegen-units = 1` | Single codegen unit unlocks full LTO |
| `panic = "abort"` | No unwinding tables, smaller + faster panics |
| `opt-level = 3` | Maximum performance |

---

## Step 2 — Install Inno Setup (one-time, ~3 MB)

Download and run the installer from **https://jrsoftware.org/isdl.php**.
It's free, open-source, and used by GitHub Desktop, OBS Studio, VLC,
and most Windows apps that aren't from giants. You only need this on your
build machine, not on the user's.

---

## Step 3 — Compile the installer

1. Launch **"Inno Setup Compiler"** from the Start menu.
2. **File → Open…** → choose `installer\R2.iss`.
3. **Build → Compile** (or press F9).
4. Done. The output appears at:

```
installer\Output\R2-Setup-0.4.0.exe
```

That single file is everything a user needs to install Ardon-R2.

---

## What the installer does

When the user runs `R2-Setup-0.4.0.exe`:

* Default install path: `%LOCALAPPDATA%\Programs\Ardon-R2` (no admin needed).
* Optional install path: `C:\Program Files\Ardon-R2` (admin needed).
* Copies `r2.exe` + `docs/` + `samples/` into the install dir.
* Creates a Start Menu entry "Ardon-R2 REPL".
* Three opt-in checkboxes:
  * Desktop icon.
  * Add to PATH (per-user) — lets users type `r2` in any terminal.
  * Associate `.r2` files — double-click a script to run it.
* Generates an uninstaller in Add/Remove Programs.

---

## Estimated final size

| Component | Approx |
|---|---|
| `r2.exe` (release-optimized) | 4 MB |
| Docs (`README`, `CHANGELOG`, `FUNCTIONS`, `ARCHITECTURE`, `KNOWN_LIMITATIONS`) | ~500 KB |
| Samples (~30 `.r2` files) | ~150 KB |
| Inno Setup overhead + LZMA2/ultra64 compression | ~500 KB |
| **Final installer (.exe)** | **~5 MB** |

For reference: R 4.4 ships at ~80 MB, RStudio at ~200 MB. R2 is ~16× smaller
than R and ~40× smaller than RStudio because Rust statically links the
standard library and there's no JVM, no GTK, no Python runtime.

---

## Alternative: portable .zip

If you don't want an installer at all, just zip the release directory:

```bash
cd target/release
7z a R2-portable-0.4.0.zip r2.exe ../../../samples ../../../docs
```

Users unzip and run `r2.exe` directly. No registry writes, no PATH changes,
no uninstaller — but also no Start Menu integration. ~4 MB compressed.

---

## Future considerations

* **Code-signing**: an unsigned `.exe` triggers SmartScreen warnings on
  Windows 10+. A code-signing certificate costs ~$70–250/year (Sectigo,
  DigiCert). Worth it if you have public release ambitions.
* **Linux/.deb**: use `cargo-deb` — produces a `.deb` of similar size.
* **macOS/.dmg**: use `cargo-bundle` — produces a `.app` and `.dmg`.
* **MSI for enterprise**: `cargo-wix` produces a Windows Installer
  package suitable for Group Policy mass-deploy.
