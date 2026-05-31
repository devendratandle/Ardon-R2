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
