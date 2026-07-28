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
//! On mobile (Android / iPad-OS) the same widgets will run inside a
//! single tabbed layout instead of MDI — that's a swap of the host
//! shell, not the widgets.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use r2_console::{ConsoleBuffer, SubmitAction};
use r2_engine::Engine;
use r2_ui::{
    auto_scroll_offset, Cell, CellGridState, Color,
    Dialog, DialogButton,
    menu_bar_height, GraphPanel, GridPos, InputField, MdiHost, R2Ui,
    Rect, Selection, Theme, WindowId,
};

mod menus;
mod support;
use support::*;

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
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

    // Dark is the default (see Theme::default); khaki/rgui remain
    // selectable for the classic R look.
    let theme = Theme::default();

    // ── Shared state ───────────────────────────────────────────────
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

    // Engine + install the single output sink. set_output_sink now wires
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
    let engine = Rc::new(RefCell::new(engine));

    let mdi = Rc::new(RefCell::new(MdiHost::new()));
    // Default sizes chosen to read at the same visual proportion R's
    // RGui ships with — Console slightly wider than tall, Graphics
    // close to square.
    let console_id  = mdi.borrow_mut().add_window("R2 Console",
        Rect { x: 24.0, y: 36.0, w: 640.0, h: 440.0 });
    // Graphics windows are created lazily — one per `dev.new()` (or
    // the auto-created device-1 on the first plot). Map keyed by
    // engine-side DeviceId so events round-trip cleanly.
    let active_devices: Rc<RefCell<std::collections::HashMap<
        r2_graphics::device::DeviceId, (WindowId, GraphPanel)>>> =
        Rc::new(RefCell::new(std::collections::HashMap::new()));

    let grid_state = Rc::new(RefCell::new(CellGridState::new()));
    // Two scrollbars on the Console transcript. Created hidden;
    // each frame computes visible_fraction from the current
    // content vs viewport sizes and shows the bar only when the
    // content overflows.
    let vscroll = Rc::new(RefCell::new(
        r2_ui::Scrollbar::new(r2_ui::ScrollOrientation::Vertical)));
    let hscroll = Rc::new(RefCell::new(
        r2_ui::Scrollbar::new(r2_ui::ScrollOrientation::Horizontal)));
    // Previous frame's input text — lets us tell "user typed" from "user
    // scrolled". Horizontal cursor-follow only runs on a typing change, so
    // a manual drag of the bottom scrollbar isn't snapped back every frame.
    let last_input = Rc::new(RefCell::new(String::new()));
    // Transcript row count last frame — detects "new output arrived" so the
    // console can snap back to the prompt (R-console behaviour).
    let last_total_rows = Rc::new(RefCell::new(0usize));
    let input      = Rc::new(RefCell::new(InputField::new()));
    let quit_requested = Rc::new(RefCell::new(false));
    // Modal dialogs (R-style): quit confirmation ("Save workspace image?")
    // and the Settings panel (font resize). Owned here, driven + painted
    // last in the frame closure.
    let quit_dialog     = Rc::new(RefCell::new(Dialog::new()));
    let settings_dialog = Rc::new(RefCell::new(Dialog::new()));

    // ── Menus (built in menus.rs) — action strings share one namespace
    // with the central dispatch in the frame closure below.
    let menu_console  = menus::console_menu();
    let menu_graphics = menus::graphics_menu();
    let ctx_console   = menus::console_context();
    let ctx_graphics  = menus::graphics_context();

    // Title-bar logo — decoded + resampled to a small square at startup.
    // The actual atlas upload happens on the first frame (we need a
    // Renderer for that, and we only get one inside on_frame).
    let logo_rgba: Vec<u8>; let logo_w: u32; let logo_h: u32;
    {
        const LOGO_BYTES: &[u8] = include_bytes!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/logo.png"));
        let full = image::load_from_memory(LOGO_BYTES)
            .map_err(|e| format!("logo decode: {}", e))?
            .into_rgba8();
        // The full logo is a WIDE composite: the "R2" monogram on top, the
        // "Ardon" wordmark beneath, plus a faint corner watermark. Squeezed
        // into the ~18 px title-bar square the whole thing collapses into an
        // unreadable smudge. Crop to just the colorful "R2" mark (top ~60%,
        // central ~78%) so the small icon reads as R2. The taskbar / Alt-Tab
        // icon (set via .icon_png below) still uses the full logo.
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
        logo_rgba = resized.into_raw();
        logo_w = nw;
        logo_h = nh;
    }
    let logo_uploaded = Rc::new(RefCell::new(false));
    let logo_handle: Rc<RefCell<Option<r2_ui::ImageHandle>>> = Rc::new(RefCell::new(None));

    let frame_counter = Rc::new(RefCell::new(0u64));
    // One-time adaptive window layout on the first frame (workspace known).
    let did_layout = Rc::new(RefCell::new(false));
    // SVG cache key — re-rasterize the GraphPanel only when the engine
    // produces new SVG content. Comparing string length is cheap and
    // catches every plot-mutation we currently emit.

    R2Ui::app("Ardon-R2")
        .theme(theme.clone())
        .initial_size(1280, 800)
        .icon_png(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/logo.png")))
        .on_frame({
            let buffer       = buffer.clone();
            let engine       = engine.clone();
            let mdi          = mdi.clone();
            let grid_state   = grid_state.clone();
            let input        = input.clone();
            let active_devices = active_devices.clone();
            let menu_console = menu_console.clone();
            let menu_graphics = menu_graphics.clone();
            let ctx_console   = ctx_console.clone();
            let ctx_graphics  = ctx_graphics.clone();
            let vscroll       = vscroll.clone();
            let hscroll       = hscroll.clone();
            let last_input    = last_input.clone();
            let last_total_rows = last_total_rows.clone();
            let frame_counter = frame_counter.clone();
            let did_layout    = did_layout.clone();
            let quit_requested = quit_requested.clone();
            let quit_dialog    = quit_dialog.clone();
            let settings_dialog = settings_dialog.clone();
            let logo_uploaded  = logo_uploaded.clone();
            let logo_handle    = logo_handle.clone();
            move |ctx, renderer, frame, theme| {
                *frame_counter.borrow_mut() += 1;

                // First-frame: upload the title-bar logo and attach to
                // each sub-window. Atlas alloc happens once; the
                // ImageHandle is cheap to copy after.
                if !*logo_uploaded.borrow() {
                    if let Some(handle) = renderer.upload_image(
                        logo_w, logo_h, &logo_rgba)
                    {
                        *logo_handle.borrow_mut() = Some(handle);
                        if let Some(w) = mdi.borrow_mut().window_mut(console_id) {
                            w.icon = Some(handle);
                        }
                    }
                    *logo_uploaded.borrow_mut() = true;
                }

                // ── Sync engine device events → MDI sub-windows.
                //     Each `dev.new()` produces a Created event; we
                //     spawn a fresh sub-window + GraphPanel. Plotted
                //     events refresh the matching panel. Closed
                //     events hide + drop the window.
                {
                    use r2_graphics::device::{DeviceEvent, drain_events,
                                              device_full_svg};
                    for ev in drain_events() {
                        match ev {
                            DeviceEvent::Created(id) => {
                                // R-style near-square device window,
                                // sized PROPORTIONALLY to the actual
                                // window (adapts 720p → 4K). Cascade
                                // subsequent devices so multiple windows
                                // don't overlap identically.
                                let ww = renderer.size.width  as f32;
                                let wh = renderer.size.height as f32;
                                let n = id.0 as f32;
                                let casc = theme.px(32.0);
                                let bounds = Rect {
                                    x: ww * 0.55 + (n - 1.0) * casc,
                                    y: menu_bar_height(theme) + wh * 0.03 + (n - 1.0) * casc * 0.8,
                                    w: ww * 0.42,
                                    h: wh * 0.72,
                                };
                                let wid = mdi.borrow_mut()
                                    .add_window(format!("R2 Graphics — Dev {}", id.0), bounds);
                                if let Some(handle) = *logo_handle.borrow() {
                                    if let Some(w) = mdi.borrow_mut().window_mut(wid) {
                                        w.icon = Some(handle);
                                    }
                                }
                                let panel = GraphPanel::new();
                                active_devices.borrow_mut().insert(id, (wid, panel));
                            }
                            DeviceEvent::Plotted(id) => {
                                if let Some(svg) = device_full_svg(id) {
                                    if let Some((wid, panel)) = active_devices.borrow_mut().get_mut(&id) {
                                        panel.set_svg(svg.into_bytes());
                                        if let Some(w) = mdi.borrow_mut().window_mut(*wid) {
                                            w.visible = true;
                                        }
                                    }
                                }
                            }
                            DeviceEvent::Closed(id) => {
                                if let Some((wid, _)) = active_devices.borrow_mut().remove(&id) {
                                    if let Some(w) = mdi.borrow_mut().window_mut(wid) {
                                        w.visible = false;
                                    }
                                }
                            }
                            DeviceEvent::CurrentChanged(_) => { /* z-order shift handled on click */ }
                        }
                    }
                }
                // Compute "current graphics window id" once per frame —
                // any per-window menu / save-dialog / paint dispatcher
                // below uses this when it needs "the graphics window
                // the user is currently working with".
                let graphics_id: Option<WindowId> = (|| {
                    let cur = r2_graphics::device::current_device()?;
                    active_devices.borrow().get(&cur).map(|(w, _)| *w)
                })();
                let win_w = renderer.size.width  as f32;
                let win_h = renderer.size.height as f32;

                // Quit path (R-style): q()/quit() set `quit_requested`, and
                // the OS close button (✕ / Alt-F4) arrives as
                // `ctx.close_requested`. Instead of exiting outright, pop a
                // modal "Save workspace image? [Yes/No/Cancel]" confirmation.
                // Consume the engine flag so it fires once.
                let quit_flag = {
                    let mut b = quit_requested.borrow_mut();
                    std::mem::replace(&mut *b, false)
                };
                if (quit_flag || ctx.close_requested)
                    && !quit_dialog.borrow().is_open()
                    && !settings_dialog.borrow().is_open()
                {
                    let mut d = quit_dialog.borrow_mut();
                    d.title = "Quit Ardon-R2".into();
                    d.lines = vec!["Save workspace image?".into()];
                    d.buttons = vec![
                        DialogButton::new("Yes",    "quit.yes"),
                        DialogButton::new("No",     "quit.no"),
                        DialogButton::new("Cancel", "quit.cancel"),
                    ];
                    d.default_action = "quit.yes".into();
                    d.cancel_action  = "quit.cancel".into();
                    d.open();
                }

                // While any modal is open it OWNS input: feed the widgets
                // underneath an empty event slice so clicks/keys don't leak
                // through to the console, menus, or window chrome. The dialog
                // itself (handled + painted last) reads the real `ctx.events`.
                let modal_open = quit_dialog.borrow().is_open()
                              || settings_dialog.borrow().is_open();
                let ui_events: &[r2_ui::InputEvent] =
                    if modal_open { &[] } else { ctx.events };

                // ── Workspace
                let menu_h = menu_bar_height(theme);
                let menu_rect = Rect { x: 0.0, y: 0.0, w: win_w, h: menu_h };
                let workspace = Rect { x: 0.0, y: menu_h,
                                       w: win_w, h: win_h - menu_h };
                mdi.borrow_mut().set_workspace(workspace);

                // ── One-time adaptive layout: the OS window is maximized
                //    (see r2-ui WindowBuilder), but the console deliberately
                //    takes only the LEFT half of the workspace. Graphics
                //    devices open at x = 55% (see DeviceEvent::Created), so
                //    console and plots sit SIDE BY SIDE — no window
                //    switching, no overlap, both readable at a glance.
                //    Full height, since the vertical space is free: long
                //    transcripts stay readable. Done once; the user can then
                //    drag / resize / maximize any window freely.
                if !*did_layout.borrow() && workspace.w > 0.0 {
                    *did_layout.borrow_mut() = true;
                    if let Some(w) = mdi.borrow_mut().window_mut(console_id) {
                        w.bounds = Rect {
                            x: workspace.x + workspace.w * 0.015,
                            y: workspace.y + workspace.h * 0.03,
                            // Right edge lands at ~53.5%, clear of the 55%
                            // where graphics windows start.
                            w: (workspace.w * 0.52).max(200.0),
                            h: (workspace.h * 0.90).max(120.0),
                        };
                    }
                }

                // ── Pick the menu bar belonging to the active window.
                //     The OTHER menu's open-popup state is closed each
                //     frame so it doesn't linger when focus switches.
                // graphics_id is now Option<WindowId>. A frame with no
                // open device → graphics_id is None → always console menu.
                let active_menu = if graphics_id.is_some()
                    && mdi.borrow().z_order().last() == graphics_id {
                    menu_graphics.clone()
                } else {
                    menu_console.clone()
                };
                // Close any popups on the inactive menu.
                if Rc::ptr_eq(&active_menu, &menu_console) {
                    menu_graphics.borrow_mut().open = None;
                } else {
                    menu_console.borrow_mut().open = None;
                }

                // ── Menu bar + right-click context menu events.
                //     Both funnel into the SAME dispatch below — one
                //     place to add a feature, two ways for the user to
                //     reach it.
                let topmost_now = mdi.borrow().z_order().last();
                // Snapshot whether any context menu was already open
                // BEFORE we process this frame's events. If it was, a
                // left-click that just landed on the popup item must
                // not also reach the grid (which would collapse the
                // user's selection before Copy can read it).
                let ctx_was_open = ctx_console.borrow().is_open()
                                || ctx_graphics.borrow().is_open();
                let mb_action = active_menu.borrow_mut()
                    .handle_events(ui_events, menu_rect, renderer, theme);
                let cm_action = match topmost_now {
                    Some(id) if id == console_id => {
                        let content = mdi.borrow().window(console_id)
                            .map(|w| w.content_rect(theme));
                        content.and_then(|c| ctx_console.borrow_mut()
                            .handle_events(ui_events, c, renderer, theme))
                    }
                    Some(id) if graphics_id == Some(id) => {
                        let content = graphics_id.and_then(|gid|
                            mdi.borrow().window(gid).map(|w| w.content_rect(theme)));
                        content.and_then(|c| ctx_graphics.borrow_mut()
                            .handle_events(ui_events, c, renderer, theme))
                    }
                    _ => None,
                };
                if let Some(action) = mb_action.or(cm_action) {
                    match action.as_str() {
                        "file.quit"  => { *quit_requested.borrow_mut() = true; }
                        "edit.settings" => {
                            let mut d = settings_dialog.borrow_mut();
                            d.title = "Settings".into();
                            d.buttons = vec![
                                DialogButton::new("A \u{2013}", "settings.font_dec"),
                                DialogButton::new("A +",        "settings.font_inc"),
                                DialogButton::new("Reset",      "settings.font_reset"),
                                DialogButton::new("Close",      "settings.close"),
                            ];
                            d.default_action = String::new();
                            d.cancel_action  = "settings.close".into();
                            d.open();
                        }
                        "file.clear" => { buffer.lock().unwrap().clear(); }
                        "file.save_plot" => {
                            if take_engine_svg().is_some() {
                                // Resolution-aware: read the Graphics window's
                                // current panel rect × DPI and rasterize at
                                // exactly those pixel dimensions. On 4K /
                                // 200% scaling this naturally gives a 4K
                                // PNG; on 100% it gives a panel-sized PNG.
                                // SVG ignores width/height (vector format).
                                let (sw, sh) = graphics_id
                                    .and_then(|gid| mdi.borrow().window(gid).map(|w| {
                                        let r = w.content_rect(theme);
                                        (((r.w * theme.dpi).round() as u32).max(320),
                                         ((r.h * theme.dpi).round() as u32).max(240))
                                    }))
                                    .unwrap_or((1024, 768));
                                let pick = rfd::FileDialog::new()
                                    .set_title("Save R2 plot")
                                    .set_file_name("plot.svg")
                                    .add_filter("SVG vector",     &["svg"])
                                    .add_filter("PNG image",      &["png"])
                                    .add_filter("All supported",  &["svg", "png"])
                                    .save_file();
                                if let Some(path) = pick {
                                    let path_str = path.to_string_lossy().into_owned();
                                    let result = r2_graphics::device::save_plot(
                                        &path_str, sw, sh)
                                        .map(|_| ())
                                        .map_err(|e| e.msg);
                                    match result {
                                        Ok(_)  => buffer.lock().unwrap()
                                                    .push_output(&format!("Saved plot to {} ({}×{})",
                                                                          path_str, sw, sh)),
                                        Err(e) => buffer.lock().unwrap()
                                                    .push_error(&format!("Save failed: {}", e)),
                                    }
                                }
                            } else {
                                buffer.lock().unwrap().push_output("No plot to save.");
                            }
                        }
                        "file.save_plot_png" => {
                            if r2_graphics::device::current_has_plot() {
                                // Same window-aware sizing as
                                // file.save_plot — exactly what the
                                // panel shows, scaled by DPI.
                                let (sw, sh) = graphics_id
                                    .and_then(|gid| mdi.borrow().window(gid).map(|w| {
                                        let r = w.content_rect(theme);
                                        (((r.w * theme.dpi).round() as u32).max(320),
                                         ((r.h * theme.dpi).round() as u32).max(240))
                                    }))
                                    .unwrap_or((1024, 768));
                                let pick = rfd::FileDialog::new()
                                    .set_title("Save R2 plot as PNG")
                                    .set_file_name("plot.png")
                                    .add_filter("PNG image", &["png"])
                                    .save_file();
                                if let Some(path) = pick {
                                    let path_str = path.to_string_lossy().into_owned();
                                    match r2_graphics::device::save_plot(&path_str, sw, sh) {
                                        Ok(_)  => buffer.lock().unwrap()
                                                    .push_output(&format!("Saved PNG to {} ({}×{})",
                                                                          path_str, sw, sh)),
                                        Err(e) => buffer.lock().unwrap()
                                                    .push_error(&format!("Save failed: {}", e.msg)),
                                    }
                                }
                            } else {
                                buffer.lock().unwrap().push_output("No plot to save.");
                            }
                        }
                        "file.copy_plot" => {
                            // Copy the raw SVG source to the clipboard so the
                            // user can paste into an editor or vector tool.
                            if let Some(svg) = take_engine_svg() {
                                ctx.clipboard.set_text(&svg);
                                buffer.lock().unwrap()
                                    .push_output("Plot SVG copied to clipboard.");
                            } else {
                                buffer.lock().unwrap().push_output("No plot to copy.");
                            }
                        }
                        "file.copy_plot_image" => {
                            // Rasterise the current plot at the active
                            // Graphics window's pixel size (× DPI) and
                            // put the bitmap on the clipboard. Pastes
                            // into Word / Excel / Outlook / any image
                            // editor that accepts a clipboard bitmap.
                            if r2_graphics::device::current_has_plot() {
                                let (sw, sh) = graphics_id
                                    .and_then(|gid| mdi.borrow().window(gid).map(|w| {
                                        let r = w.content_rect(theme);
                                        (((r.w * theme.dpi).round() as u32).max(320),
                                         ((r.h * theme.dpi).round() as u32).max(240))
                                    }))
                                    .unwrap_or((1024, 768));
                                match r2_graphics::device::render_to_rgba(sw, sh) {
                                    Ok((rgba, w, h)) => {
                                        if ctx.clipboard.set_image(w, h, &rgba) {
                                            buffer.lock().unwrap().push_output(
                                                &format!("Plot copied to clipboard as {}×{} image.", w, h));
                                        } else {
                                            buffer.lock().unwrap().push_error(
                                                "Clipboard image copy failed (OS rejected).");
                                        }
                                    }
                                    Err(e) => buffer.lock().unwrap()
                                        .push_error(&format!("Rasterise failed: {}", e.msg)),
                                }
                            } else {
                                buffer.lock().unwrap().push_output("No plot to copy.");
                            }
                        }
                        "edit.copy" => {
                            // Copy current selection. We must include
                            // the LIVE prompt row in `rows` because
                            // paint also appends it — selection rows
                            // are indexed against that combined list.
                            // Without the prompt row, selections that
                            // touched the last visible line fell off
                            // the end and returned empty.
                            let mut rows = rows_from_buffer(&buffer.lock().unwrap(), theme);
                            let inp = input.borrow();
                            let prompt_row: Vec<Cell> = {
                                let prefix = format!("{} ", inp.prompt);
                                let full = format!("{}{}", prefix, inp.current);
                                full.chars().map(|c| Cell::plain(c, theme.console_input)).collect()
                            };
                            rows.push(prompt_row);
                            // Wrap to the same width paint/selection use, so the
                            // selection's (row,col) indices map to the same rows.
                            let (cell_w, _) = renderer.cell_metrics(theme.fs());
                            let ww = mdi.borrow().window(console_id)
                                .map(|w| console_wrap_width(w.content_rect(theme).w, cell_w))
                                .unwrap_or(0);
                            let rows = wrap_rows(rows, ww);
                            if let Some(sel) = grid_state.borrow().selection {
                                let text = r2_ui::grid::selection_to_text(&rows, sel);
                                if !text.is_empty() {
                                    ctx.clipboard.set_text(&text);
                                }
                            }
                        }
                        "edit.paste" => {
                            // Paste through the same multi-line path
                            // InputField's Ctrl+V uses: first chunk
                            // completes the line being typed, each
                            // intermediate line auto-submits as if
                            // Enter-pressed, the final chunk stays in
                            // the editor. Identical behavior whether
                            // the user typed Ctrl+V, picked Edit ▸
                            // Paste, or right-clicked → Paste.
                            if let Some(s) = ctx.clipboard.get_text() {
                                let s = s.replace('\r', "");
                                if !s.contains('\n') {
                                    let mut f = input.borrow_mut();
                                    let pos = f.cursor;
                                    f.current.insert_str(pos, &s);
                                    f.cursor = pos + s.len();
                                } else {
                                    let mut parts: Vec<String> =
                                        s.split('\n').map(String::from).collect();
                                    let head = parts.remove(0);
                                    let tail = parts.pop().unwrap_or_default();
                                    // Insert head into the current line, then
                                    // take its full content as the first
                                    // submission, plus any middle lines.
                                    let first_submission: String = {
                                        let mut f = input.borrow_mut();
                                        let pos = f.cursor;
                                        f.current.insert_str(pos, &head);
                                        std::mem::take(&mut f.current)
                                    };
                                    let to_submit: Vec<String> =
                                        std::iter::once(first_submission)
                                            .chain(parts.into_iter())
                                            .collect();
                                    for line in to_submit {
                                        let action = buffer.lock().unwrap().submit_line(line);
                                        if let SubmitAction::Submit(src) = action {
                                            run_source(&src, &mut engine.borrow_mut(),
                                                       &buffer, &quit_requested);
                                            // DeviceEvent::Plotted (drained
                                            // at frame top) refreshes any
                                            // graphics window for us.
                                        }
                                    }
                                    let mut f = input.borrow_mut();
                                    f.current = tail;
                                    f.cursor  = f.current.len();
                                }
                            }
                        }
                        "edit.select_all" => {
                            let rows = rows_from_buffer(&buffer.lock().unwrap(), theme);
                            // Wrap to match paint/selection row structure.
                            let (cell_w, _) = renderer.cell_metrics(theme.fs());
                            let ww = mdi.borrow().window(console_id)
                                .map(|w| console_wrap_width(w.content_rect(theme).w, cell_w))
                                .unwrap_or(0);
                            let rows = wrap_rows(rows, ww);
                            if !rows.is_empty() {
                                let last = rows.len() - 1;
                                let last_col = rows[last].len().saturating_sub(1);
                                grid_state.borrow_mut().selection = Some(Selection {
                                    start: GridPos { row: 0, col: 0 },
                                    end:   GridPos { row: last, col: last_col },
                                });
                            }
                        }
                        "win.console" => {
                            if let Some(w) = mdi.borrow_mut().window_mut(console_id) { w.visible = true; }
                            // bring to front
                            if let Some(w) = mdi.borrow_mut().window_mut(console_id) {
                                let b = w.bounds; let _ = b;
                            }
                        }
                        "win.graphics" => {
                            // Reveal every device's window. Cheap when
                            // no devices are open.
                            let ids: Vec<WindowId> = active_devices.borrow()
                                .values().map(|(w, _)| *w).collect();
                            for wid in ids {
                                if let Some(w) = mdi.borrow_mut().window_mut(wid) {
                                    w.visible = true;
                                }
                            }
                        }
                        "help.about" => {
                            let mut b = buffer.lock().unwrap();
                            b.push_banner("Ardon-R2 — pure-Rust reimplementation of R, AGPL-3.0.");
                            b.push_banner("GUI built on the r2-ui framework (winit + wgpu + fontdue).");
                        }
                        _ => {}
                    }
                }

                // ── MDI chrome events (drag / resize / close / max)
                mdi.borrow_mut().handle_events(ui_events, theme);

                // Resize/move cursor affordance: turn the pointer into the
                // familiar ↔ ↕ ⤡ ⤢ arrows over a window's edges/corners so
                // users see windows are resizable (R/desktop behaviour).
                ctx.set_cursor(mdi.borrow().hover_cursor(theme));

                // ── Console keyboard input — ALWAYS active so the console
                //    stays typeable regardless of which MDI window is
                //    topmost (RGui keeps the console interactive; a plot no
                //    longer "steals" the keyboard). Clicking a window still
                //    raises it via the MDI handler above.
                let topmost = mdi.borrow().z_order().last();
                {
                    let mut input_mut = input.borrow_mut();
                    let resp = input_mut.handle_events(ui_events, ctx.clipboard);

                    // Multi-line paste: each pasted line goes through
                    // ConsoleBuffer::submit_line exactly as if typed
                    // and Enter-pressed. ConsoleBuffer handles the
                    // continuation logic (open braces / parens span
                    // multiple lines until balanced).
                    for line in resp.auto_submit_lines {
                        let action = buffer.lock().unwrap().submit_line(line);
                        if let SubmitAction::Submit(src) = action {
                            run_source(&src, &mut engine.borrow_mut(),
                                       &buffer, &quit_requested);
                            // DeviceEvent::Plotted (drained at frame
                            // top) refreshes any graphics window.
                        }
                    }

                    if let Some(line) = resp.submitted {
                        let action = buffer.lock().unwrap().submit_line(line);
                        if let SubmitAction::Submit(src) = action {
                            run_source(&src, &mut engine.borrow_mut(),
                                       &buffer, &quit_requested);
                            // DeviceEvent::Plotted (drained at the top
                            // of the next frame) auto-refreshes the
                            // matching graphics window.
                        }
                    }
                    if resp.history_up {
                        if let Some(s) = buffer.lock().unwrap().history_up()   { input_mut.set_line(s); }
                    }
                    if resp.history_down {
                        if let Some(s) = buffer.lock().unwrap().history_down() { input_mut.set_line(s); }
                    }
                    input_mut.set_prompt(buffer.lock().unwrap().current_prompt());

                    // Drag-select / Ctrl+A / Ctrl+C — only when the console
                    // is the focused (topmost) window, so mouse selection
                    // targets the window the user is actually working in.
                    if topmost == Some(console_id) {
                    let mut rows = rows_from_buffer(&buffer.lock().unwrap(), theme);
                    let prompt_row: Vec<Cell> = {
                        let prefix = format!("{} ", input_mut.prompt);
                        let full = format!("{}{}", prefix, input_mut.current);
                        full.chars().map(|c| Cell::plain(c, theme.console_input)).collect()
                    };
                    rows.push(prompt_row);
                    let (cell_w, line_h) = renderer.cell_metrics(theme.fs());
                    let content = mdi.borrow().window(console_id).map(|w| w.content_rect(theme));
                    if let Some(content) = content {
                        // Must match the PAINT grid_rect below: reserve the
                        // scrollbar strips on the right/bottom. Otherwise the
                        // selection hit-area overlaps the scrollbar and
                        // dragging the scrollbar starts a text selection.
                        let sbt = r2_ui::SCROLLBAR_THICKNESS;
                        let grid_rect = Rect {
                            x: content.x + 8.0, y: content.y + 8.0,
                            w: content.w - 16.0 - sbt,
                            h: content.h - 16.0 - sbt,
                        };
                        // Wrap to the console width so hit-testing indexes the
                        // same physical rows paint draws (otherwise a click on
                        // a wrapped line would select the wrong text).
                        let rows = wrap_rows(rows, console_wrap_width(content.w, cell_w));
                        // Skip selection events on the frame a context
                        // menu was open / fired — the click that picked
                        // a menu item would otherwise also reach the
                        // grid and collapse the selection.
                        if !ctx_was_open {
                            let _copied = grid_state.borrow_mut().handle_events(
                                ui_events, &rows, grid_rect,
                                cell_w, line_h, ctx.clipboard,
                            );
                        }
                    }
                    } // end: grid selection (console topmost)
                }

                // ── Paint ─────────────────────────────────────────
                frame.paint_rect(workspace.x, workspace.y, workspace.w, workspace.h,
                                 theme.mdi_background);
                active_menu.borrow().paint_strip(frame, renderer, menu_rect, theme);

                // Pure z-order: for each window from bottom to top,
                // paint its BODY → CONTENT → TITLE BAR as one unit.
                // The next-higher window's body then cleanly covers
                // everything below it, including the previous title
                // strip. No leaking title bars between windows.
                let order: Vec<WindowId> = mdi.borrow().z_order().collect();
                for id in order {
                    if !mdi.borrow().should_paint_content(id) { continue; }
                    mdi.borrow().paint_body(id, frame, theme);
                    let content = mdi.borrow()
                        .window(id)
                        .filter(|w| w.visible)
                        .map(|w| w.content_rect(theme));
                    let content = match content { Some(r) => r, None => continue };

                    if id == console_id {
                        let (cell_w, line_h) = renderer.cell_metrics(theme.fs());
                        // Console body follows the THEME (was hardcoded
                        // white, which made the console the one surface a
                        // dark theme couldn't reach).
                        frame.paint_rect(content.x, content.y, content.w, content.h,
                                         theme.window_background);

                        // Reserve a strip on the right edge (vertical
                        // scrollbar) and the bottom edge (horizontal
                        // scrollbar). The grid_rect shrinks by that
                        // thickness so transcript content never lands
                        // under the bars.
                        let sbt = r2_ui::SCROLLBAR_THICKNESS;
                        let grid_rect = Rect {
                            x: content.x + 8.0,
                            y: content.y + 8.0,
                            w: content.w - 16.0 - sbt,
                            h: content.h - 16.0 - sbt,
                        };
                        let vtrack = Rect {
                            x: grid_rect.x + grid_rect.w,
                            y: grid_rect.y,
                            w: sbt,
                            h: grid_rect.h,
                        };
                        let htrack = Rect {
                            x: grid_rect.x,
                            y: grid_rect.y + grid_rect.h,
                            w: grid_rect.w,
                            h: sbt,
                        };

                        let transcript_rows = rows_from_buffer(&buffer.lock().unwrap(), theme);
                        let input_ref = input.borrow();
                        // Build the live prompt row: "<prompt> <typed text>"
                        // in console-input color, appended to the transcript.
                        let prompt_row: Vec<Cell> = {
                            let prefix = format!("{} ", input_ref.prompt);
                            let full = format!("{}{}", prefix, input_ref.current);
                            full.chars().map(|c| Cell::plain(c, theme.console_input)).collect()
                        };
                        // Cursor position within the (unwrapped) prompt line.
                        let cursor_col_logical = input_ref.prompt.chars().count() + 1
                            + input_ref.current[..input_ref.cursor].chars().count();

                        // ── Wrap long lines to the console width so output
                        //    folds instead of running off the right edge. Wrap
                        //    the transcript, then the prompt separately, so we
                        //    can map the cursor into the wrapped grid: the
                        //    prompt starts at `prompt_base`, and a caret that
                        //    is `cursor_col_logical` chars in lands `/ wrap`
                        //    rows down and `% wrap` cols across.
                        let wrap = (grid_rect.w / cell_w).floor() as usize;
                        let mut rows = wrap_rows(transcript_rows, wrap);
                        let prompt_base = rows.len();
                        let (prompt_row_index, cursor_col_in_row) = if wrap > 0 {
                            (prompt_base + cursor_col_logical / wrap, cursor_col_logical % wrap)
                        } else {
                            (prompt_base, cursor_col_logical)
                        };
                        rows.extend(wrap_rows(vec![prompt_row], wrap));

                        // ── Drive the scrollbars from current content
                        //     vs viewport sizes (in cell units). With wrapping
                        //     on, no row exceeds `wrap`, so the horizontal bar
                        //     stays inert (full thumb) — kept only as a no-op.
                        let total_rows  = rows.len();
                        let max_cols    = rows.iter().map(|r| r.len()).max().unwrap_or(0)
                                          .max(cursor_col_in_row + 1);
                        let visible_rows = (grid_rect.h / line_h).floor() as usize;
                        let visible_cols = (grid_rect.w / cell_w).floor() as usize;

                        // ── Snap to the prompt on new output (R-console).
                        // If the user had scrolled up (pinned override) or
                        // right (long line) and a command returns, the fresh
                        // prompt would otherwise sit below / left-of the view
                        // — i.e. hidden. Any transcript growth hands control
                        // back to auto-scroll and returns the horizontal bar
                        // to its default left position. Typing doesn't change
                        // the row count, so this never fights the cursor-
                        // follow logic below.
                        {
                            let mut lt = last_total_rows.borrow_mut();
                            if *lt != total_rows {
                                *lt = total_rows;
                                let mut gs = grid_state.borrow_mut();
                                gs.scroll_y_override = None;
                                gs.scroll_x = 0;
                            }
                        }
                        if total_rows > 0 {
                            vscroll.borrow_mut().visible_fraction =
                                (visible_rows as f32 / total_rows as f32).min(1.0);
                        }
                        if max_cols > 0 {
                            hscroll.borrow_mut().visible_fraction =
                                (visible_cols as f32 / max_cols as f32).min(1.0);
                        }
                        if let Some(p) = vscroll.borrow_mut().handle_events(ui_events, vtrack) {
                            let off = r2_ui::scroll_pos_to_row(p, total_rows, visible_rows);
                            // Pin to manual offset; if user dragged to
                            // the bottom, hand control back to
                            // auto-scroll so new lines keep showing.
                            grid_state.borrow_mut().scroll_y_override =
                                if off + visible_rows >= total_rows { None } else { Some(off) };
                        }
                        if let Some(p) = hscroll.borrow_mut().handle_events(ui_events, htrack) {
                            grid_state.borrow_mut().scroll_x =
                                r2_ui::scroll_pos_to_col(p, max_cols, visible_cols);
                        }

                        // ── Horizontal cursor-follow (R-console). Keep the
                        //     typing cursor visible without dragging the prompt
                        //     off the left margin:
                        //   • cursor within the first screen-width ("default
                        //     frame") → snap scroll_x to 0 so the prompt is
                        //     shown at the left and the down-bar returns to its
                        //     default left position;
                        //   • cursor past the right edge → slide right just
                        //     enough to reveal it (text scrolls under the bar);
                        //   • cursor still left of the view but beyond the
                        //     default frame → follow it so it stays on screen.
                        // Only follow the cursor when the user actually TYPED
                        // this frame (input text changed). Otherwise leave
                        // scroll_x alone so a manual drag / wheel of the bottom
                        // bar sticks instead of snapping back to 0 every frame.
                        let typed = {
                            let mut li = last_input.borrow_mut();
                            let changed = *li != input_ref.current;
                            if changed { *li = input_ref.current.clone(); }
                            changed
                        };
                        if typed && visible_cols > 0 {
                            let mut gs = grid_state.borrow_mut();
                            if cursor_col_in_row < visible_cols {
                                gs.scroll_x = 0;
                            } else if cursor_col_in_row >= gs.scroll_x + visible_cols {
                                gs.scroll_x = cursor_col_in_row + 1 - visible_cols;
                            } else if cursor_col_in_row < gs.scroll_x {
                                gs.scroll_x = cursor_col_in_row;
                            }
                        }

                        // ── Keep the thumbs in sync with the ACTUAL scroll
                        //     offset every frame. Covers wheel / touchpad
                        //     scroll, auto-scroll-to-bottom, and keyboard
                        //     (Shift+Arrow) selection — not just dragging the
                        //     thumb. R-console behaviour: the bar always
                        //     reflects where the transcript is.
                        {
                            let gs = grid_state.borrow();
                            let eff_y = gs.scroll_y_override
                                .unwrap_or_else(|| auto_scroll_offset(total_rows, grid_rect.h, line_h));
                            vscroll.borrow_mut().position =
                                r2_ui::row_offset_to_scroll_pos(eff_y, total_rows, visible_rows);
                            hscroll.borrow_mut().position =
                                r2_ui::col_offset_to_scroll_pos(gs.scroll_x, max_cols, visible_cols);
                        }

                        // ── Transcript paint — uses the scroll state
                        //     CellGridState now owns.
                        grid_state.borrow().paint(frame, renderer, &rows, grid_rect,
                                                  cell_w, line_h, theme.fs(), theme);

                        // ── Cursor — must follow the SAME effective
                        //     vertical scroll the painter used.
                        let scroll = match grid_state.borrow().scroll_y_override {
                            Some(s) => s,
                            None    => auto_scroll_offset(rows.len(), grid_rect.h, line_h),
                        };
                        let scroll_x = grid_state.borrow().scroll_x;
                        let cursor_on = (*frame_counter.borrow() / 30) % 2 == 0;
                        if cursor_on && prompt_row_index >= scroll && cursor_col_in_row >= scroll_x {
                            let visible_row = prompt_row_index - scroll;
                            let cx = grid_rect.x + (cursor_col_in_row - scroll_x) as f32 * cell_w;
                            let cy = grid_rect.y + visible_row as f32 * line_h;
                            if cy + line_h <= grid_rect.y + grid_rect.h
                                && cx + 2.0 <= grid_rect.x + grid_rect.w
                            {
                                frame.paint_rect(cx, cy + line_h * 0.1,
                                                 2.0, line_h * 0.8, theme.cursor);
                            }
                        }

                        // ── Scrollbars on top of the transcript.
                        vscroll.borrow().paint(frame, vtrack, theme);
                        hscroll.borrow().paint(frame, htrack, theme);
                    } else {
                        // Any other window is a graphics device. Find
                        // the matching GraphPanel in active_devices
                        // and paint it. Pure window-id lookup so the
                        // user can have any number of dev.new()
                        // windows open simultaneously.
                        let device_for_window: Option<r2_graphics::device::DeviceId> =
                            active_devices.borrow().iter()
                                .find(|(_, (w, _))| *w == id)
                                .map(|(dev_id, _)| *dev_id);
                        if let Some(dev_id) = device_for_window {
                            // Plot canvases stay WHITE in every theme: a
                            // plot is a document (it gets saved, printed
                            // and published), not chrome. R does the same.
                            frame.paint_rect(content.x, content.y, content.w, content.h,
                                             Color::WHITE);
                            let inner = Rect {
                                x: content.x + 8.0,  y: content.y + 8.0,
                                w: (content.w - 16.0).max(0.0),
                                h: (content.h - 16.0).max(0.0),
                            };
                            if let Some((_, panel)) = active_devices.borrow_mut()
                                .get_mut(&dev_id)
                            {
                                panel.paint(frame, renderer, inner, theme);
                            }
                        }
                    }

                    // Paint this window's title bar BEFORE moving on
                    // to the next higher window. Pure z-order = the
                    // next window's body covers this title strip if
                    // they overlap, which is what users expect.
                    mdi.borrow().paint_titlebar(id, frame, renderer, theme);
                }

                // ── Close-button handling
                if mdi.borrow_mut().take_close_requested(console_id) {
                    if let Some(w) = mdi.borrow_mut().window_mut(console_id) { w.visible = false; }
                }
                // Each graphics device's close button routes back to
                // the engine via `close_device`, which emits a
                // DeviceEvent::Closed picked up next frame.
                let device_ids: Vec<(r2_graphics::device::DeviceId, WindowId)> =
                    active_devices.borrow().iter()
                        .map(|(d, (w, _))| (*d, *w))
                        .collect();
                for (dev_id, wid) in device_ids {
                    if mdi.borrow_mut().take_close_requested(wid) {
                        r2_graphics::device::close_device(Some(dev_id));
                    }
                }

                // ── Popup + context menus — painted LAST. Drop-down
                //    floats above every sub-window; right-click
                //    context menu floats above everything including
                //    the popup. No window can cover any open menu.
                active_menu.borrow().paint_popup(frame, renderer, menu_rect, theme);
                ctx_console.borrow().paint(frame, renderer, theme);
                ctx_graphics.borrow().paint(frame, renderer, theme);

                // ── Modal dialogs — handled + painted LAST so they dim and
                //    float above everything and own the keyboard/mouse.
                if settings_dialog.borrow().is_open() {
                    // Live body: show the current base (pre-DPI) font size.
                    {
                        let mut d = settings_dialog.borrow_mut();
                        d.lines = vec![
                            format!("UI font size:  {} pt", theme.font_size_base() as i32),
                            String::new(),
                            "Resizes the console + window text.".into(),
                            "(DPI scaling applies on top automatically.)".into(),
                        ];
                    }
                    let action = settings_dialog.borrow_mut()
                        .handle_events(ctx.events, renderer, theme, win_w, win_h);
                    if let Some(a) = action {
                        match a.as_str() {
                            "settings.font_dec"   =>
                                ctx.set_base_font_size(theme.font_size_base() - 1.0),
                            "settings.font_inc"   =>
                                ctx.set_base_font_size(theme.font_size_base() + 1.0),
                            "settings.font_reset" => ctx.set_base_font_size(14.0),
                            "settings.close"      => settings_dialog.borrow_mut().close(),
                            _ => {}
                        }
                    }
                    settings_dialog.borrow().paint(frame, renderer, theme, win_w, win_h);
                }

                if quit_dialog.borrow().is_open() {
                    let action = quit_dialog.borrow_mut()
                        .handle_events(ctx.events, renderer, theme, win_w, win_h);
                    if let Some(a) = action {
                        match a.as_str() {
                            "quit.yes" => {
                                // Save the session (all variables) to a default
                                // workspace file, then quit — R writes .RData;
                                // Ardon-R2 writes a .r2s session.
                                run_source("save(\".r2session.r2s\")",
                                           &mut engine.borrow_mut(), &buffer, &quit_requested);
                                ctx.request_exit();
                            }
                            "quit.no"     => { ctx.request_exit(); }
                            "quit.cancel" => { quit_dialog.borrow_mut().close(); }
                            _ => {}
                        }
                    }
                    quit_dialog.borrow().paint(frame, renderer, theme, win_w, win_h);
                }
            }
        })
        .run()
}
