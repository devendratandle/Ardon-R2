# r2-ui — Ardon-R2 GUI framework

Grid-first, terminal-philosophy GUI framework purpose-built for R2.
Replaces eframe / egui in R2Gui from v0.3 onward.

**See `docs/R2_UI_FRAMEWORK.md` for the design contract.** This README
is the quick orientation; the design doc is the authoritative spec.

## Why this crate exists

eframe / egui served R2 well through v0.2 but its pixel-canvas
selection model fights every keystroke we add. R Console (the
ancestor we admire) handles selection flawlessly because it treats
text as a 2D grid of cells, not as glyphs on a canvas. r2-ui adopts
that grid philosophy with modern Rust + GPU rendering on top.

## Status

**Phase 2 Week 1 — scaffolding.** The public API surface exists and
compiles. Actual rendering (winit + wgpu pipeline) is implemented in
the next session per the design doc's §11 roadmap.

| Module | Purpose | Phase 2 implementation status |
|---|---|---|
| `app.rs`   | `R2Ui` builder + event loop | scaffold |
| `theme.rs` | `Theme` struct, named themes | ✅ done |
| `grid.rs`  | `CellGrid` + `Selection` math | data + tests done; rendering scaffold |
| `input.rs` | `InputField` (prompt + cursor + history) | scaffold |
| `layout.rs`| `Mdi` / `Tabs` / `Split` layouts | scaffold |
| `menu.rs`  | `MenuBar` declarative menus | scaffold |
| `window.rs`| Sub-window builder | scaffold |

Tests for selection math (`grid.rs`) pass. Rendering tests come
when the wgpu pipeline lands.

## Public API quick reference

```rust
use r2_ui::*;

R2Ui::app("Ardon-R2")
    .theme(Theme::khaki())
    .font_family("Consolas", 14.0)
    .icon_png(include_bytes!("../../assets/logo.png"))
    .mdi(|mdi| { /* configure sub-windows */ })
    .menu(|m|   { /* configure menus */ })
    .run();
```

That's the WHOLE public surface a typical R2-UI consumer touches.
~3,400 LoC of framework code, ~70 LoC of consumer code per app.

## Dependencies

| Crate | Why |
|---|---|
| `winit` | window + raw input (industry default) |
| `wgpu`  | GPU rendering (modern, cross-platform) |
| `fontdue` | font shaping (smaller / more stable than cosmic-text for our Latin-only needs) |
| `arboard` | cross-platform clipboard |
| `image` | PNG decoders for icons |
| `resvg` + `usvg` + `tiny-skia` | SVG → raster for plot panels |
| `bytemuck` | typed bytes for wgpu uniform/vertex buffers |

## Build output

Cargo produces both forms — static rlib (compile-time link) and
cdylib (runtime DLL). The installer's launcher loads the DLL form
so r2_ui.dll can be hot-swapped for patch releases.

```
target/release/
├── libr2_ui.rlib   ← static link target
└── r2_ui.dll       ← shipped DLL (loaded by R2Gui.exe at startup)
```

## License

AGPL-3.0, matching Ardon-R2's overall license.
