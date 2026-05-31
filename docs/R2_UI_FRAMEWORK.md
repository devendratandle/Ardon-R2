# R2-UI — Framework Design Document

**Status:** approved, not yet implemented.
**Owner:** Devendra Tandale + AI collaborator.
**Last revised:** 2026-05-27.

R2-UI is the dedicated GUI framework for Ardon-R2. It replaces the
project's current dependency on `eframe` / `egui` with a small,
purpose-built library that follows 1970s-terminal logic
(grid-based text, fixed cells, range-based selection) combined with
modern Rust + GPU primitives (24-bit color, mouse, smooth scroll,
hardware-accelerated rendering).

This document is the **architectural contract** for the project. Once
approved, scope is **locked** — feature additions need an explicit
amendment to this doc, not an ad-hoc PR. That discipline is what keeps
R Console small, stable, and useful after 20+ years; we follow the same
rule.

---

## 1. Goals

Listed in priority order. When in doubt, earlier goals override later
ones.

1. **Reliability over features.** Selection works perfectly in every
   direction. The cursor never drifts into read-only regions. Keyboard
   shortcuts behave like a 1980s VT-100 / xterm: predictable, complete,
   no surprises.
2. **Per-cell color preserved.** Red input, blue output, error red.
   Output text doesn't have to drop colors to gain selection.
3. **One-time engineering investment.** Locked scope means a stable
   framework. R Console didn't change between 2004 and 2024; R2-UI is
   intended to age similarly.
4. **Small, auditable surface.** Total framework ≤ 5,000 LoC core
   (excluding tests and docs). Anything not necessary for the REPL +
   plot pane MVP is rejected.
5. **DLL-friendly.** Build as `cdylib + rlib` from day 1. R2's
   distribution model is multi-DLL (mirroring R's split between
   `R.dll`, `Rblas.dll`, etc.); R2-UI is the first GUI-side DLL.
6. **Modern stack underneath.** GPU rendering via `wgpu`; native window
   via `winit`. We don't reinvent these.

## 2. Non-goals

What R2-UI explicitly will **not** do:

- General-purpose layout engine (no flexbox, no CSS-style positioning)
- Accessibility tree / screen-reader integration (defer to v2 via `accesskit`)
- Animations, transitions, motion design
- Drop shadows, rounded corners as a system feature (themes can paint
  whichever rectangle style they like, but the framework doesn't
  abstract it)
- Rich text inside text (no inline images, no clickable links — links
  are a `Hyperlink` widget, not embedded HTML)
- Right-to-left / bidirectional text shaping (defer to v2 via
  `cosmic-text`'s shaping pipeline)
- Touch / multi-touch gestures (defer to Android port doc)
- Hot-reloading of widgets at runtime
- Visual scripting / drag-and-drop GUI builder

If something on this list later turns out essential, **the doc gets
amended first**; code never gets ahead of the design.

## 3. The 1970s grid insight (with modern dynamic sizing)

The defining design choice. R2-UI represents the **transcript area**
(the part everyone struggles to make selectable in pixel-canvas
toolkits) as a 2D grid of `Cell`s. Each cell stores one Unicode
grapheme cluster plus its foreground color, optional background
color, and bold / italic flags. Layout is **fixed cell size**
(monospace font, ceil to integer pixel width). Selection is a
**range of grid coordinates** — not pixel coordinates.

### 3.1 Dynamic grid sizing (not fixed 80×24)

Unlike a 1970s VT100 which had a hard-coded 80×24 grid, R2-UI's grid
dimensions are **derived per-frame from the window's actual pixel
rect ÷ font cell size**. This matches what every modern terminal
(Alacritty, Kitty, Windows Terminal) has done since ~2015:

```
font           = Consolas 14pt
cell_w         = font.glyph_advance('M')   →  e.g.  9 px
cell_h         = font.line_height          →  e.g. 18 px

current rect   = 820 × 480
visible_cols   = 820 ÷ 9  = 91
visible_rows   = 480 ÷ 18 = 26
```

On every resize (or font-size change in Settings), the grid simply
re-derives `(visible_cols, visible_rows)`. Nothing is stored — the
numbers are recomputed cheaply each frame.

Consequence: there's no separate "grid resize" code path. The
transcript is an infinite-length list of variable-length lines; the
renderer paints whatever rows fit in the current rect, with overflow
handled by the surrounding ScrollArea. ~200 LoC of clipping and
"resize-grid-or-scroll?" logic that a fixed-grid design would need
simply doesn't exist.

Consequence: every problem that's hard in modern toolkits becomes
trivial:

| Problem | In a pixel-canvas toolkit | In R2-UI's grid |
|---|---|---|
| Hit-test mouse → character | Walk the glyph layout, compare bounding rects, deal with kerning | `((y - top) / cell_h, (x - left) / cell_w)` |
| Select multi-line range | Galley math, possibly broken on line wraps | Sort `(start_row, start_col)` and `(end_row, end_col)` |
| Repaint highlight | Custom rect-from-galley-range routine | A few `painter.rect_filled()` calls |
| Cross-line selection | One of the hardest UX problems in GUI | Trivially free, same data structure |
| Copy selection as text | Reconstruct from glyphs | Iterate cells in range, push their chars to a String |
| Word selection (double-click) | Locale-aware tokenization | Walk cells until non-alphanumeric |

This is why terminal emulators have shipped flawless text selection
since the 1970s while CSS-based browsers and pixel-canvas toolkits
still ship broken edge cases in 2025.

R2-UI confines pixel-canvas reasoning to **input fields** and **plot
panes**. Everything else is a grid.

## 4. Layer architecture

```
┌─────────────────────────────────────────────────┐
│  R2-UI public API           crates/r2-ui/lib.rs │  ← what R2Gui calls
├─────────────────────────────────────────────────┤
│  Widget layer:                                  │
│  • CellGrid   • InputField   • PlotPanel        │
│  • Window     • MenuBar      • Dialog           │
│  • Toolbar    • ScrollArea   • Hyperlink        │
├─────────────────────────────────────────────────┤
│  Layout layer:                                  │
│  • Mdi   • Tabs   • Split                       │
│  • Theme   • Keymap                             │
├─────────────────────────────────────────────────┤
│  Render layer:                                  │
│  • paint_rect, paint_glyph, paint_image         │
│  • Selection painter                            │
├─────────────────────────────────────────────────┤
│  Substrate (we depend on, don't ship):          │
│  • winit       — windowing + input              │
│  • wgpu        — GPU rendering                  │
│  • cosmic-text — glyph shaping + atlas          │
│  • arboard     — cross-platform clipboard       │
│  • image       — PNG/JPEG load (for icons)      │
│  • resvg       — SVG rasterize (for plots)      │
└─────────────────────────────────────────────────┘
```

**Stack rationale:** every substrate crate is either stable, has been
shipping for years, or is the de-facto Rust choice. We don't take a
risk on a crate <0.5 unless it's the only option in its space.

## 5. Module structure

```
crates/r2-ui/
├── Cargo.toml          crate-type = ["rlib", "cdylib"]
├── README.md
├── src/
│   ├── lib.rs          re-exports + App entry point (≤ 200 LoC)
│   ├── app.rs          App + event loop + redraw scheduling (~300)
│   ├── render.rs       wgpu pipeline + paint primitives (~500)
│   ├── grid.rs         CellGrid widget + selection state (~400)
│   │                      ← reduced from 600 LoC because dynamic
│   │                        sizing eliminates fixed-grid clip/resize logic
│   ├── input.rs        InputField widget (~350)
│   ├── layout.rs       Mdi / Tabs / Split layout enums (~300)
│   │                      ← reduced from 400 LoC for the same reason
│   ├── window.rs       Window (sub-window) primitive (~300)
│   ├── menu.rs         MenuBar + menu items (~250)
│   ├── dialog.rs       modal Dialog + standard buttons (~200)
│   ├── plot.rs         PlotPanel (SVG/PNG via resvg) (~250)
│   ├── theme.rs        Theme struct + named themes (~150)
│   ├── keymap.rs       Keybinding registry + standard set (~200)
│   ├── clipboard.rs    arboard wrapper (~80)
│   ├── font.rs         font loading, fallback chain (~150)
│   └── icon.rs         small image loader for title-bar icons (~80)
├── tests/
│   ├── grid_select.rs            selection math, all directions
│   ├── grid_keyboard.rs          shift+arrow, ctrl+arrow, home/end
│   ├── input_history.rs          Up/Down history navigation
│   ├── layout_mdi.rs             window resize, maximize, restore
│   └── theme_swap.rs             swapping themes at runtime
└── examples/
    ├── minimal_console.rs        50-line REPL using just CellGrid+InputField
    └── full_app.rs               complete R2Gui-style app (~150 LoC)
```

**Total core**: ~3,400 LoC (reduced from ~3,800 by adopting dynamic
grid sizing — see §3.1).
**With tests + docs + examples**: ~4,200 LoC.

## 6. Public API — the contract that defines R2-UI

The full API surface that R2Gui depends on. Anything outside this list
is implementation detail and can change between R2-UI versions; this
list is **semver-stable** once R2-UI 1.0 ships.

### 6.1 App entry

```rust
use r2_ui::*;

R2Ui::app("Window title")
    .theme(Theme::khaki())
    .font_family("Consolas", 14.0)
    .icon_png(include_bytes!("../../assets/logo.png"))
    .mdi(|mdi| { /* configure sub-windows */ })
    .menu(|m| { /* configure menu bar */ })
    .on_quit(|app| QuitDialog::default())
    .run()
```

### 6.2 CellGrid (the transcript widget)

```rust
pub struct CellGrid { /* private */ }

impl CellGrid {
    pub fn bind(buffer: &ConsoleBuffer) -> CellGridBuilder;
}

impl CellGridBuilder {
    pub fn font(self, font: FontId) -> Self;
    pub fn min_rows(self, rows: usize) -> Self;
    pub fn show(self, ui: &mut Ui) -> CellGridResponse;
}

pub struct CellGridResponse {
    pub selection_text: Option<String>,   // ← Some when Ctrl+C just fired
    // ...
}
```

### 6.3 InputField (the prompt widget)

```rust
pub struct InputField { /* private */ }

impl InputField {
    pub fn bind(buffer: &ConsoleBuffer) -> Self;
    pub fn prompt(self, prompt: &str) -> Self;
    pub fn show(self, ui: &mut Ui) -> InputFieldResponse;
}

pub struct InputFieldResponse {
    pub submitted: Option<String>,        // ← Some when Enter pressed
}
```

InputField's behavior is **terminal-style**:
- Cursor is locked to the current line — cannot wander into history
- Up/Down arrows navigate history (not text)
- Shift+Up/Down do nothing (no multi-line selection in input)
- Home / End jump to start / end of current line
- Ctrl+Left/Right are word-jumps within the current line
- Enter submits; Shift+Enter does nothing (no soft-break for REPL)
- Multi-line input is achieved via the `+` continuation prompt loop
  managed by `ConsoleBuffer` — input field stays single-line

### 6.4 Window (MDI sub-window)

```rust
mdi.window("R2 Console")
    .icon(&logo_handle)           // shown in title bar
    .min_max_close_buttons(true)  // [_][□][X]
    .titlebar_drag(true)
    .resize_corners(true)
    .body(/* widget */)
    .default_pos((60.0, 80.0))
    .default_size((820.0, 480.0));
```

Window state (`pos`, `size`, `maximized`, `minimized`) is tracked by
R2-UI; no app-side bookkeeping needed.

### 6.5 Menu

```rust
m.file()
 .item("Source script…", "Ctrl+Shift+S", |a| a.source_script_dialog())
 .separator()
 .item("Quit",            "Alt+F4",       |a| a.quit());

m.help()
 .item("About",        "",  about_dialog)
 .item("Documentation","F1", |_| open_url("https://github.com/..."));
```

Each menu method (`.file()`, `.edit()`, `.help()`, etc.) returns a
typed builder so the menu structure is checked at compile time.

### 6.6 Theme

```rust
pub struct Theme {
    pub mdi_background:    Color,
    pub window_background: Color,
    pub menu_background:   Color,
    pub menu_text:         Color,

    pub console_input:        Color,
    pub console_output:       Color,
    pub console_error:        Color,
    pub console_banner:       Color,
    pub console_selection_bg: Color,
    pub cursor:               Color,

    pub button_min:   Color,    // green —
    pub button_max:   Color,    // blue □
    pub button_close: Color,    // red ✕

    pub font_size:   f32,
    pub line_height: f32,
}

impl Theme {
    pub fn khaki()    -> Self;       // light parchment workspace
    pub fn rgui()     -> Self;       // classic R Console (white, black text)
    pub fn solarized_dark() -> Self;
    pub fn solarized_light()-> Self;
}
```

Themes are pure data — third parties can publish theme crates that
just provide a `pub fn my_theme() -> Theme`.

### 6.7 ConsoleBuffer integration

R2-UI doesn't own the transcript — `r2-console` does. R2-UI's
`CellGrid` and `InputField` are **views** over a `ConsoleBuffer`.
This is critical: the engine writes via `OutputSink`, the buffer
collects, the UI renders. Three layers, three responsibilities.

## 7. Selection model (the core of why we exist)

`CellGrid` holds an optional `Selection { start: GridPos, end: GridPos }`.
`GridPos = (row, col)`. Order is by row then column.

### 7.1 Mouse behavior

- `drag_started` at position P → `selection.start = selection.end = pos_to_grid(P)`
- `dragged` → `selection.end = pos_to_grid(current)`
- Direction-agnostic: rendering sorts `(start, end)` so visual
  highlight is always top-left → bottom-right regardless of drag
  direction
- Double-click → select word (walk cells until non-alphanumeric in
  both directions)
- Triple-click → select entire line
- Click outside selection → clear selection

### 7.2 Keyboard behavior

When `CellGrid` has focus (clicked into the transcript):
- Arrow keys → move cursor (highlighted, not blinking — it's a
  read-only widget)
- Shift+Arrow → extend selection toward arrow
- Ctrl+Arrow → jump word
- Shift+Ctrl+Arrow → extend selection by word
- Home → start of current line
- End → end of current line
- Ctrl+Home → top of transcript
- Ctrl+End → bottom of transcript
- Ctrl+A → select all
- Ctrl+C → copy selection to clipboard
- Escape → clear selection

### 7.3 Why this works flawlessly

Because `pos_to_grid(p) = ((p.y - origin.y) / cell.h, (p.x - origin.x) / cell.w)`
is exact integer arithmetic — there's no floating-point hit-test on
curved glyphs. The selection range is always well-defined. Painting
the highlight is `rect_filled` per row of the selection. No edge cases.

## 8. DLL boundary

```toml
# crates/r2-ui/Cargo.toml
[lib]
crate-type = ["rlib", "cdylib"]
```

`rlib` for static linking when building R2Gui dev/test builds.
`cdylib` for the shipped `r2_ui.dll` (Windows) / `libr2_ui.so` (Linux)
/ `libr2_ui.dylib` (macOS).

The DLL is loaded by R2Gui (and any other R2-UI consumer) at process
start. Update R2-UI by replacing the DLL — no recompile of R2Gui.

### 8.1 ABI choice

We expose a **Rust ABI** (not C ABI). Justification:

| ABI | Pros | Cons |
|---|---|---|
| Rust ABI (default) | Trivial; all Rust types work natively | Compiler-version-coupled — R2Gui and R2-UI must be built with the same Rust version |
| C ABI (`extern "C"`) | Stable across Rust versions; enables non-Rust consumers | Have to wrap every public type in C-compatible wrappers; significantly more code |

For R2 (Rust-only consumer), Rust ABI is fine. The R2 installer ships
both binaries built with the same compiler — no version mismatch
possible.

If we ever want third-party non-Rust plugins, we add a thin C ABI
layer in `src/c_abi.rs` then. Not part of v1.

### 8.2 What the user (R2 installer) sees

```
Ardon-R2/
├── R2Gui.exe              ~300 KB (just startup + dynamic loader)
├── bin/
│   ├── r2_console.dll
│   ├── r2_engine.dll
│   ├── r2_graphics.dll
│   ├── r2_linalg_avx2.dll   ← installer picks the right variant
│   ├── r2_stats.dll
│   ├── r2_time.dll
│   └── r2_ui.dll            ← THIS framework
└── packages/
    └── (lazy-loaded addons)
```

A future R2-UI update ships as a single new `r2_ui.dll` file (~2 MB
download) instead of an 80 MB full reinstall.

## 9. Keymap discipline

The keymap is **published as a table** in this doc and the user-facing
README. Every shortcut listed here is contract; adding or changing
one needs a doc revision.

| Key | Action |
|---|---|
| Enter | Submit current input line |
| Up Arrow | Previous history entry |
| Down Arrow | Next history entry |
| Home | Start of current input |
| End | End of current input |
| Ctrl+Left / Ctrl+Right | Word-jump within input |
| Backspace | Delete char before cursor (within input only — never past prompt) |
| Delete | Delete char after cursor |
| Tab | (reserved for future completion) |
| Ctrl+C (when transcript focused) | Copy selection |
| Ctrl+C (when input focused) | Cancel current input (R-style — clear input, fresh prompt) |
| Ctrl+L | Clear transcript |
| Ctrl+A (when transcript focused) | Select all transcript |
| Ctrl+A (when input focused) | Select all input |
| F1 | Open documentation |
| F5 / Ctrl+R | Run current input (alias for Enter; explicit visible button) |
| Esc | Cancel modal dialog / clear selection |

## 10. Testing strategy

R2-UI's tests target behavior, not pixels. Three layers:

1. **Pure-data unit tests** (the bulk): given a `CellGrid` with a
   known transcript and a programmatic selection, assert the
   selection range, the copied text, the paint-rect bounds. No window
   needs to open.

   ```rust
   #[test] fn select_diagonal_returns_correct_text() {
       let g = grid_with_lines(["abc", "def", "ghi"]);
       g.select((0, 1), (2, 1));   // 'b' through 'h'
       assert_eq!(g.selection_text(), "bc\ndef\ngh");
   }
   ```

2. **Snapshot tests** of paint commands: each frame, the renderer
   emits a list of `PaintCmd` (rect, glyph, image). We snapshot that
   list for known-input cases and diff on change.

3. **Manual + golden-image tests** for actual rendering on real GPU.
   Run on CI's Windows runner with a headless wgpu adapter. Compare
   produced PNGs to checked-in reference PNGs. Few of these — they're
   slow.

Target: ≥ 80% line coverage on `grid.rs`, `input.rs`, `layout.rs`.

## 11. Migration plan (from current eframe-based R2Gui)

Phased over ~5 weeks, with R2Gui-on-egui still working until R2-UI is
proven:

### Week 1 — substrate

- New `crates/r2-ui/` workspace member, rlib only at first
- `app.rs` boilerplate: open a window via winit, clear to khaki
- `render.rs`: wgpu pipeline that paints rectangles and a single test glyph
- Theme struct, two themes (khaki, rgui)

### Week 2 — grid

- `grid.rs`: CellGrid widget, builds from a `ConsoleBuffer`, paints
  cells, no selection yet
- Font loading from system (Consolas / Courier) via `cosmic-text` or
  fontdue
- First milestone: a window that shows scrollable colored transcript

### Week 3 — selection + input

- Selection state machine in `grid.rs`
- Mouse drag selection
- Keyboard selection (Shift+Arrow, Ctrl+A, etc.)
- Ctrl+C → clipboard via `arboard`
- `input.rs`: InputField with cursor, history, multi-line continuation
- Milestone: parity with current R2Gui's transcript + prompt

### Week 4 — windowing + menus + plot

- `window.rs`: MDI sub-window with custom title bar, traffic-light buttons
- `layout.rs`: Mdi layout enum, sub-window drag/resize/maximize
- `menu.rs`: MenuBar with declarative items
- `plot.rs`: PlotPanel rendering SVG via resvg → wgpu texture
- Milestone: feature parity with current R2Gui

### Week 5 — refactor R2Gui, retire eframe

- Rewrite `crates/r2-gui/src/main.rs` to use R2-UI's declarative API
  (≤ 300 LoC target as proven above)
- Remove `eframe` / `egui` / `egui_wgpu` dependencies
- Update installer to bundle `r2_ui.dll`
- Final milestone: ship R2 0.3 on R2-UI

If anything in week 3 (selection) takes longer than planned, we hold
the timeline — selection is THE feature, getting it wrong defeats
the purpose.

## 12. Versioning

R2-UI follows strict semver from 1.0 onward:

- `1.x.y` — bug fixes, no API surface change
- `1.X.0` — additive API (new method, new theme, new keymap entry); strictly backward-compatible
- `2.0.0` — removes or changes existing API; only allowed if a feature
  on the non-goals list (§2) is being added

We expect to live at `1.x` for a long time. The whole point.

## 13. Open questions

Items deliberately deferred so the doc can be finalized without
blocking progress:

1. **Soft-wrap policy in CellGrid.** Should very long output lines
   wrap visually, or scroll horizontally? R Console scrolls; many
   terminals wrap. Recommendation: wrap, with a guide column at 80.
2. **Plot pane interaction**: should plots support zoom / pan
   gestures, or remain static (re-plot to change view)? Recommend:
   static for v1; zoom in v2 via a separate `r2.viz` addon.
3. **Font fallback chain** for non-ASCII characters in output (e.g.
   Greek letters from `sigma <- 1`). Pick `cosmic-text` over `fontdue`
   if this matters.
4. **Plugin loading** at runtime via `libloading` — not part of v1;
   would need stable C ABI first.

## 14. Acceptance criteria for R2-UI 1.0

Closing the framework's first release requires ALL of:

- [ ] All public API in §6 implemented and stable
- [ ] All keymap entries in §9 working as specified
- [ ] R2Gui rewritten using R2-UI, ≤ 350 LoC main, all current
      features intact
- [ ] All 12 r2-console tests still passing through the GUI
- [ ] Selection works in every direction tested (drag, keyboard, mouse-then-keyboard, mouse-then-mouse-different-side)
- [ ] Khaki and RGui themes both shipped and switchable at runtime
- [ ] DLL build (`r2_ui.dll`) produced, R2Gui loads it dynamically
- [ ] Installer ships R2Gui.exe + r2_ui.dll + r2_engine.dll separately
- [ ] CI green on Windows + Linux (macOS optional for 1.0)
- [ ] At least one third-party theme demonstrated (sanity-check
      the public API)

## 15. Out-of-scope (will NOT be added without doc amendment)

To prevent feature creep, the following are explicitly excluded from
R2-UI's roadmap. If anyone wants them, they need a separate addon /
fork:

- Markdown rendering inside the transcript
- Inline plot rendering (plots stay in their own window)
- Code completion / autocomplete UI
- Multiple cursors / multi-cursor edits
- Find-and-replace within transcript
- Recording / replay of console sessions
- Network sync of console state
- Theming animations (color crossfades, etc.)
- Plugin marketplace UI
- Custom mouse cursors per widget
- Per-character animation
- WASM build target for browser embedding

These are explicitly fine for **addon** crates to implement on top of
R2-UI's public API. They are not framework concerns.
