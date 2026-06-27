# Ardon-R2 Mobile Porting Notes (Android + iPad-OS)

**Status:** the substrate is ready; mobile shells are not yet built.
**Last revised:** Week 5 — after R2Gui refactor onto r2-ui.

This doc captures what's already prepared for mobile and what each
mobile session will need to add. The point: the v0.3 desktop GUI was
built so that ~80% of the code base ports unchanged.

---

## What's already mobile-ready

| Layer | Status | Notes |
|---|---|---|
| **r2-engine** | ✅ pure-Rust, no GUI deps | Compiles unchanged for `aarch64-linux-android` and `aarch64-apple-ios`. |
| **r2-parser, r2-types, r2-stats, r2-graphics** | ✅ pure-Rust | No platform-specific code paths. |
| **r2-console** | ✅ pure-Rust | OutputSink trait works the same on every host. |
| **r2-ui touch events** | ✅ wired (`event.rs`) | `WindowEvent::Touch` is converted into `MouseDown / MouseMoved / MouseUp` with the primary finger. Every widget (CellGridState, InputField, GraphPanel, MdiHost, MenuBarState) works under touch with **zero changes**. |
| **r2-ui rendering** | ✅ wgpu | wgpu uses **Vulkan / GLES** on Android and **Metal** on iOS automatically. No code changes. |
| **r2-ui font loading** | ✅ probes multiple paths | `load_system_font()` already includes Linux paths (DejaVu, Liberation). Android needs `/system/fonts/DroidSansMono.ttf` added; iOS needs `/System/Library/Fonts/Menlo.ttc` (already present). |
| **r2-ui clipboard** | ⚠️ desktop-only | `arboard` doesn't support Android/iOS. Need a thin platform shim — both OSes expose clipboard APIs via JNI / `UIPasteboard`. |

## What each mobile session will need to add

### 1. winit's mobile entry points

* **Android:** `winit::platform::android::EventLoopBuilderExtAndroid` —
  takes an `AndroidApp` from `android-activity`. R2-UI's `R2Ui::run()`
  will need a sibling `R2Ui::run_android(app)`.
* **iOS:** winit handles this transparently as long as `main` is the
  entry symbol; iOS app bundle structure is the bigger task.

### 2. Single-window shell (no MDI)

Phones / tablets don't get a windowing system. Replace the desktop's
`MdiHost` with the **Tabs** layout already declared in `layout.rs`:

```text
┌─────────────────────────────┐
│ ⌂ Console  ⌃ Graphics  ☰    │   tabs along top
├─────────────────────────────┤
│                             │
│  (active tab content)       │
│                             │
└─────────────────────────────┘
│ q  w  e  r  t  y  …         │   software keyboard
```

The widgets inside each tab are the same `CellGridState`,
`InputField`, `GraphPanel` used by the desktop — only the host shell
swaps.

### 3. Software keyboard (IME)

* **Android:** winit dispatches `ImeEvent::{Preedit, Commit}` once IME
  is enabled via `Window::set_ime_allowed(true)`. R2-UI's `event.rs`
  needs an `InputEvent::ImeCommit(String)` variant alongside the
  existing `Char`. `InputField::handle_events` then accepts that as a
  bulk insert.
* **iOS:** same winit IME pipeline.

### 4. Touch gestures beyond single-tap

The current touch shim only handles primary-finger drag (good enough
for selection + buttons). Pinch-zoom and two-finger scroll need:

* A `Touch` collector that tracks active finger IDs and their
  velocities each frame.
* New `InputEvent::Pinch { scale }` and `InputEvent::Scroll2 { dx, dy }`.
* `GraphPanel` reads `Pinch` to update its rasterization size.

### 5. Clipboard shim

```rust
// crates/r2-ui/src/event.rs
impl Clipboard {
    #[cfg(target_os = "android")]
    pub fn new() -> Self { /* JNI to ClipboardManager */ }
    #[cfg(target_os = "ios")]
    pub fn new() -> Self { /* objc to UIPasteboard.general */ }
    // existing arboard path for desktop targets
}
```

### 6. Filesystem + working directory

* Android: writable space is `Context.getFilesDir()` (per-app, no
  Documents folder). The CLI's `pick_user_home()` must learn to use
  this on Android.
* iOS: `~/Documents` inside the app bundle's sandbox.

### 7. Packaging

* **Android:** `cargo-apk` or `xbuild` produces a signed APK from the
  r2-gui crate (renamed `r2-mobile` for clarity). ~15–20 MB target.
* **iOS:** `xcrun` + an Xcode project shell, AppStore signing.

---

## Why this list is short

Because we built r2-ui on winit + wgpu from day 1 and kept widget
state separate from the windowing shell, mobile is a **shell swap +
two adapters (IME + clipboard)** — not a rewrite.

Concrete reuse estimate after a clean Android port:

```
crates/r2-ui            ~ 95% reused (touch + IME deltas only)
crates/r2-console       100% reused
crates/r2-engine        100% reused
crates/r2-parser        100% reused
crates/r2-graphics      100% reused
crates/r2-stats         100% reused

r2-gui main.rs         ~ 60% reused (MDI → Tabs is the main rewrite)
```

That ratio is the dividend of locking the framework scope and going
through the r2-ui abstraction in v0.3.

---

# Android port (merged from ANDROID_PORT.md)


**Status:** planning. Not started.
**Estimated effort:** 6–8 focused sessions, ~1500–2000 LoC of new/modified
code, ~150-250k assistant tokens. Final APK: 15–20 MB.

R2 is fundamentally portable: the engine, parser, stats, time series,
graphics (SVG), and addon-package crates are all pure-Rust without any
platform-specific code. Cross-compiling to `aarch64-linux-android` is
expected to work. The work is concentrated in three layers:

1. **Cross-compile plumbing** (Cargo target + NDK config)
2. **Replace shell-outs** (`git clone` / `unzip` are not available)
3. **Mobile UX** in R2Gui (single-pane layout, touch + soft keyboard)

This document is the gating reference. Pick it up in a dedicated
multi-session push when the desktop builds are stable.

---

## 1. Cross-compile plumbing  ·  ~0.5 session  ·  ~100 LoC

### Toolchain

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi
# Android Studio → SDK Manager → install NDK (Side by side)
# expose:
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.x.y
```

### `~/.cargo/config.toml`

```toml
[target.aarch64-linux-android]
ar       = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin/llvm-ar"
linker   = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin/aarch64-linux-android24-clang"

[target.armv7-linux-androideabi]
ar       = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin/llvm-ar"
linker   = "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin/armv7a-linux-androideabi24-clang"
```

### Validate

```bash
cargo build --release --target aarch64-linux-android -p r2-engine
cargo build --release --target aarch64-linux-android -p r2-console
```

Expected: both build. r2-engine pulls in r2-linalg / r2-stats / r2-arrow
which are all pure Rust — no system libraries required.

---

## 2. CLI port  ·  ~0.5 session  ·  ~50 LoC

```bash
cargo build --release --target aarch64-linux-android -p r2-repl
```

The result is an ARM ELF binary that runs in **Termux** (popular Android
terminal emulator) just like any other Linux CLI. For Termux users this
is the whole story.

Code changes needed:
* `cwd` defaulting: Android sandbox = `/data/data/dev.devendra.r2/files/`.
  Resolve `~/.r2/packages/` accordingly. Add a `#[cfg(target_os = "android")]`
  branch in `pick_user_home()`.

---

## 3. GUI port  ·  ~2 sessions  ·  ~500 LoC

eframe supports Android via the `android` feature on winit. The shape:

```rust
// crates/r2-gui/src/lib.rs (NEW, for Android only)
#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    let options = eframe::NativeOptions {
        android_app: Some(app),
        ..Default::default()
    };
    eframe::run_native(
        "Ardon-R2",
        options,
        Box::new(|cc| Box::new(crate::app::R2App::new(cc))),
    ).unwrap();
}
```

Build the GUI crate as `cdylib`:

```toml
# crates/r2-gui/Cargo.toml
[lib]
crate-type = ["cdylib"]
```

### Caveats discovered in advance

* **Soft keyboard**: when the user taps the input field, Android pops
  the on-screen keyboard which covers the bottom of the screen. The
  console widget needs to know its visible rect and scroll the prompt
  into view.
* **Touch UX**: scroll bars are too thin; need to enable swipe-scroll
  via `egui::ScrollArea::scroll_bar_visibility(AlwaysVisible)`.
* **Font sizes**: bump default to 16pt for readable touch targets.
* **MDI is wrong for phones**: don't show floating sub-windows on a
  4-inch screen. Switch to a tabbed layout (Console / Graphics / Help).

---

## 4. Replace shell-outs  ·  ~1 session  ·  ~300 LoC

Android has no `git`, no `unzip`, no `tar`. The current
`install.from.github` / `install.from.zip` shell out to those.
Need pure-Rust replacements:

| Function | Current | Android-friendly replacement |
|---|---|---|
| `install.from.github` | `std::process::Command::new("git").arg("clone")` | Fetch `https://github.com/<repo>/archive/<ref>.tar.gz` via `ureq`, extract via `tar` crate |
| `install.from.zip` | `std::process::Command::new("tar").arg("-xf")` | Use the `zip` crate (pure Rust) |

Wire these behind a `#[cfg(target_os = "android")]` and let the desktop
build keep using the system tools (faster, no auth confusion).

Dependencies added (Android-only):
```toml
[target.'cfg(target_os = "android")'.dependencies]
ureq = { version = "2", default-features = false, features = ["tls"] }
zip  = { version = "2", default-features = false, features = ["deflate"] }
tar  = "0.4"
flate2 = "1"
```

Increases the Android binary by ~2 MB after LTO.

---

## 5. APK packaging  ·  ~1–2 sessions  ·  ~150 LoC

Use **cargo-mobile2** (`cargo install cargo-mobile2`) or the simpler
**xbuild**:

```bash
cargo install xbuild
x build --release --target aarch64-linux-android --platform android
# produces target/x/release/android/ardon-r2.apk
```

Manifest essentials (`AndroidManifest.xml`):
* `android.permission.INTERNET` (for `install.packages` from GitHub)
* `android.permission.WRITE_EXTERNAL_STORAGE` (scoped storage opt-in
  for saving plots to user-visible folders)
* Minimum API 24 (matches eframe's expectation)
* Icon: reuse `assets/logo.png` → multiple density assets via
  `cargo-mobile2`'s asset pipeline

---

## 6. Mobile UX redesign  ·  ~1-2 sessions  ·  ~400 LoC

The desktop MDI layout (floating Console + Graphics sub-windows) doesn't
fit a phone. Replace with a tabbed layout:

```
┌─────────────────────────────────┐
│ ☰  Ardon-R2                     │ ← top bar with hamburger menu
├─────────────────────────────────┤
│ [Console]  [Plot]  [Help]       │ ← tab strip
├─────────────────────────────────┤
│                                 │
│   active tab content            │
│   (full screen)                 │
│                                 │
├─────────────────────────────────┤
│ R2>  ▌  [enter ⏎]               │ ← input + visible Enter button
└─────────────────────────────────┘
```

Detect the platform at runtime:

```rust
#[cfg(target_os = "android")]
fn layout() -> Layout { Layout::Tabbed }
#[cfg(not(target_os = "android"))]
fn layout() -> Layout { Layout::Mdi }
```

Both layouts drive the same `ConsoleBuffer` — only the rendering differs.
This is exactly why the `r2-console` refactor is a prerequisite for the
Android port: the layouts share zero rendering code but identical state.

---

## 7. Testing on real Android  ·  ongoing  ·  N/A

* **Emulator**: Android Studio → AVD Manager → Pixel 7 / API 34
* **Real device**: any modern Android phone, USB debugging on, `adb install r2.apk`
* **Termux**: side-load the CLI binary, drop into `$PREFIX/bin/r2`

---

## Risk / unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| eframe Android backend has rough edges (IME / text selection) | medium | Test early; have a "report bug" fallback to GitHub issues |
| wgpu Vulkan support varies on cheap Androids | medium | eframe falls back to glow (OpenGL) on Android automatically |
| AGPL licensing on Play Store | low | Play allows GPL-family with proper SOURCE link; AGPL section in README |
| Touch + multi-window not implemented | low | Tabbed layout sidesteps |

---

## Order of execution

1. r2-console refactor lands ← **prerequisite**
2. Cross-compile passes for r2-engine and r2-console
3. r2-repl Android binary builds + runs in Termux
4. r2-gui builds as cdylib, basic `android_main` shows a window
5. ConsoleBuffer wired into the Android render loop
6. Tabbed layout
7. Shell-out replacements (ureq + zip + tar)
8. Manifest + permissions + icon
9. APK packaging
10. Real-device validation

Each step is independently testable.

---

## Out of scope (deferred to v2 of Android port)

* iOS port (same eframe approach should work; Apple's review hurdles
  are a separate problem)
* Chromebook touch optimizations
* Background compute / long-running task isolation
* Multi-window Android tablet layouts
