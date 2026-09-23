// On Windows, suppress the console window that would otherwise flash
// when R2Gui.exe is launched from Explorer or the Start Menu. Debug
// builds keep the console so println!/eprintln! still surface during
// development.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

//! Ardon-R2 desktop GUI — built on the `r2-ui` framework.
//!
//! This is the v0.3 rewrite that retires eframe / egui. All UI work
//! happens through `r2-ui`'s public API: `MdiHost` for sub-windows,
//! `CellGridState` for the transcript, `InputField` for the prompt,
//! `GraphPanel` for SVG plot output, `MenuBarState` for the menu bar.
//!
//! Architecture:
//!
//!   ┌────────────────────────────────────────┐
//!   │ winit window (one OS window)           │
//!   │  ┌──────────────────────────────────┐  │
//!   │  │ menu bar (File/Edit/Windows/…)   │  │
//!   │  ├──────────────────────────────────┤  │
//!   │  │ MDI workspace                    │  │
//!   │  │  ┌──────────┐  ┌──────────────┐  │  │
//!   │  │  │ R2       │  │ R2 Graphics  │  │  │
//!   │  │  │ Console  │  │ (GraphPanel) │  │  │
//!   │  │  └──────────┘  └──────────────┘  │  │
//!   │  └──────────────────────────────────┘  │
//!   └────────────────────────────────────────┘
//!
//! Files: this one sets the process up and owns the state (`Gui`);
//! `frame.rs` runs one frame in order (devices, menus, input, paint);
//! `actions.rs` carries out menu and context-menu commands;
//! `console_view.rs` is the prompt and the transcript; `dialogs.rs` the
//! quit and settings modals; `menus.rs` builds the menus; `support.rs`
//! holds the engine glue (output sink, `run_source`, row wrapping).
//!
//! On mobile (Android / iPad-OS) the same widgets will run inside a
//! single tabbed layout instead of MDI — that's a swap of the host
//! shell, not the widgets.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use r2_console::ConsoleBuffer;
use r2_engine::Engine;
use r2_graphics::device::DeviceId;
use r2_ui::{
    CellGridState, ContextMenu, Dialog, GraphPanel, ImageHandle, InputField,
    MdiHost, MenuBarState, R2Ui, Rect, ScrollOrientation, Scrollbar, Theme, WindowId,
};

mod actions;
mod console_view;
mod dialogs;
mod frame;
mod menus;
mod support;
use support::*;

const LOGO_PNG: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/logo.png"));

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
    r2_engine::set_product_version(env!("CARGO_PKG_VERSION"));
    // Working directory: launched from the Start Menu, the GUI's cwd is
    // the (read-only) install dir, so file writes — write.csv, save,
    // mmap.write, plot-save — fail with "Access is denied". Match the
    // CLI: move to the user's Documents (or $HOME). Relative paths then
    // land somewhere writable and visible in Explorer.
    if let Some(home) = pick_user_home() {
        let _ = std::env::set_current_dir(&home);
    }

    // Warm the SVG font database off the critical path. The first plot
    // otherwise scans the whole system font directory (hundreds of files)
    // before the Graphics window can show anything — the "plot opens late,
    // fast the second time" lag. Loading it once on a background thread
    // while the GUI/engine starts up means the first plot is already warm.
    std::thread::spawn(|| {
        r2_ui::graph::warm_fonts();
        r2_graphics::device::warm_fonts();
    });

    // The engine emits a `dev.view()`-style browser plot by default —
    // we have a native Graphics window, so disable that side-channel.
    r2_graphics::device::disable_autoview();
    std::env::set_var("R2_NO_AUTOVIEW", "1");
    // We ARE a live display: plots render into the Graphics window, so
    // they should NOT auto-write .svg files. Saving stays explicit
    // (save_plot() / the Save menu).
    r2_graphics::device::set_display_present(true);

    let mut gui = Gui::new()?;
    // Dark is the default (see Theme::default); khaki/rgui remain
    // selectable for the classic R look.
    R2Ui::app("Ardon-R2")
        .theme(Theme::default())
        .initial_size(1280, 800)
        .icon_png(LOGO_PNG)
        .on_frame(move |ctx, renderer, frame, theme| gui.frame(ctx, renderer, frame, theme))
        .run()
}

// ─── State ────────────────────────────────────────────────────────

/// Everything the GUI keeps between frames. The frame closure owns the
/// one value; each concern's methods live in their own file (see the
/// module docs above).
struct Gui {
    /// Shared with the engine's output sink, hence the lock.
    buffer: Arc<Mutex<ConsoleBuffer>>,
    engine: Engine,
    mdi: MdiHost,
    console_id: WindowId,
    /// Graphics windows are created lazily — one per `dev.new()` (or
    /// the auto-created device-1 on the first plot), keyed by the
    /// engine-side DeviceId so events round-trip cleanly.
    devices: HashMap<DeviceId, (WindowId, GraphPanel)>,
    grid: CellGridState,
    /// Two scrollbars on the Console transcript. Created hidden; each
    /// frame computes visible_fraction from the current content vs
    /// viewport sizes and shows the bar only when the content overflows.
    vscroll: Scrollbar,
    hscroll: Scrollbar,
    /// Previous frame's input text — lets us tell "user typed" from "user
    /// scrolled". Horizontal cursor-follow only runs on a typing change, so
    /// a manual drag of the bottom scrollbar isn't snapped back every frame.
    last_input: String,
    /// Transcript row count last frame — detects "new output arrived" so the
    /// console can snap back to the prompt (R-console behaviour).
    last_total_rows: usize,
    input: InputField,
    /// Set by q()/quit() and the File ▸ Quit menu; the next frame turns
    /// it into the quit dialog.
    quit_requested: bool,
    /// Modal dialogs (R-style): quit confirmation ("Save workspace image?")
    /// and the Settings panel (font resize).
    quit_dialog: Dialog,
    settings_dialog: Dialog,
    menu_console: MenuBarState,
    menu_graphics: MenuBarState,
    ctx_console: ContextMenu,
    ctx_graphics: ContextMenu,
    logo: TitleLogo,
    frame_counter: u64,
    /// One-time adaptive window layout on the first frame (workspace known).
    did_layout: bool,
}

impl Gui {
    fn new() -> Result<Gui, String> {
        let buffer = Arc::new(Mutex::new(ConsoleBuffer::new()));
        {
            // Canonical banner (shared with the CLI via r2-console) + the
            // GUI's one host-specific hint line.
            let mut b = buffer.lock().unwrap();
            for line in r2_console::banner_lines(env!("CARGO_PKG_VERSION")) {
                b.push_banner(&line);
            }
            b.push_banner("plot(x, y) opens the Graphics window. q() quits.");
            b.push_banner("");
        }

        // Engine + install the single output sink. set_output_sink wires
        // the ONE process-wide console channel (r2_types::out): engine
        // print/cat output AND every compute crate's formatted output
        // (t.test / aov / manova / summary / …) converge on this GuiSink →
        // ConsoleBuffer. No separate hook needed — install once, like R's
        // R_WriteConsole.
        let mut engine = Engine::new();
        engine.set_output_sink(Box::new(GuiSink { buf: buffer.clone() }));
        // clear() / cls() from the console empties this buffer (GUI has no
        // terminal to send an ANSI clear to).
        {
            let buf = buffer.clone();
            r2_types::out::set_clear_hook(Some(Box::new(move || {
                if let Ok(mut b) = buf.lock() { b.clear(); }
            })));
        }

        let mut mdi = MdiHost::new();
        // Default sizes chosen to read at the same visual proportion R's
        // RGui ships with — Console slightly wider than tall. The first
        // frame re-lays it out against the real workspace.
        let console_id = mdi.add_window("R2 Console",
            Rect { x: 24.0, y: 36.0, w: 640.0, h: 440.0 });

        Ok(Gui {
            buffer,
            engine,
            mdi,
            console_id,
            devices: HashMap::new(),
            grid: CellGridState::new(),
            vscroll: Scrollbar::new(ScrollOrientation::Vertical),
            hscroll: Scrollbar::new(ScrollOrientation::Horizontal),
            last_input: String::new(),
            last_total_rows: 0,
            input: InputField::new(),
            quit_requested: false,
            quit_dialog: Dialog::new(),
            settings_dialog: Dialog::new(),
            menu_console: menus::console_menu(),
            menu_graphics: menus::graphics_menu(),
            ctx_console: menus::console_context(),
            ctx_graphics: menus::graphics_context(),
            logo: TitleLogo::load()?,
            frame_counter: 0,
            did_layout: false,
        })
    }
}

/// Title-bar logo — decoded + resampled at startup; the atlas upload
/// happens on the first frame (that needs a Renderer, and we only get
/// one inside the frame closure).
struct TitleLogo {
    rgba: Vec<u8>,
    w: u32,
    h: u32,
    handle: Option<ImageHandle>,
    uploaded: bool,
}

impl TitleLogo {
    fn load() -> Result<TitleLogo, String> {
        let full = image::load_from_memory(LOGO_PNG)
            .map_err(|e| format!("logo decode: {}", e))?
            .into_rgba8();
        // The full logo is a WIDE composite: the "R2" monogram on top, the
        // "Ardon" wordmark beneath, plus a faint corner watermark. Squeezed
        // into the ~18 px title-bar square the whole thing collapses into an
        // unreadable smudge. Crop to just the colorful "R2" mark (top ~60%,
        // central ~78%) so the small icon reads as R2. The taskbar / Alt-Tab
        // icon (set via .icon_png in main) still uses the full logo.
        let (fw, fh) = (full.width(), full.height());
        let cx = (fw as f32 * 0.11) as u32;
        let cy = (fh as f32 * 0.03) as u32;
        let cw = (fw as f32 * 0.78) as u32;
        let ch = (fh as f32 * 0.60) as u32;
        let img = image::imageops::crop_imm(&full, cx, cy, cw, ch).to_image();
        let (sw, sh) = (img.width(), img.height());
        // Upload the R2 mark at its NATURAL aspect (no square letterbox) so
        // the title bar can draw it filling the full bar height as a wide
        // icon. A square canvas would pad the short (vertical) axis with
        // transparent bands — exactly what made the mark look tiny. ~128 px
        // on the long edge gives the GPU bilinear filter headroom for a crisp
        // downscale to the ~20 px title-bar height. Triangle (bilinear) keeps
        // edges sharper than Lanczos3 at extreme downscale ratios.
        let target: u32 = 128;
        let scale = target as f32 / sw.max(sh) as f32;
        let nw = ((sw as f32 * scale).round() as u32).max(1);
        let nh = ((sh as f32 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(
            &img, nw, nh, image::imageops::FilterType::Triangle);
        Ok(TitleLogo { rgba: resized.into_raw(), w: nw, h: nh, handle: None, uploaded: false })
    }
}
