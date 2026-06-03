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

use r2_console::{ConsoleBuffer, LineKind, OutputSink, SubmitAction};
use r2_engine::Engine;
use r2_parser::Parser;
use r2_ui::{
    auto_scroll_offset, Cell, CellGridState, Color, ContextItem, ContextMenu,
    GraphPanel, GridPos, InputField, MdiHost, MenuBarState, MenuBuilder, R2Ui,
    Rect, Selection, Theme, WindowId, MENU_BAR_HEIGHT,
};

// ─── Output sink — engine writes through this, lands in ConsoleBuffer
//
// Matches R's internal architecture (Rinterface.h ptr_R_WriteConsole):
// the engine talks to an abstraction, the frontend installs the
// concrete impl. The CLI installs StdoutSink; we install this.

struct GuiSink {
    buf: Arc<Mutex<ConsoleBuffer>>,
}

impl OutputSink for GuiSink {
    fn write_output(&mut self, text: &str) {
        if let Ok(mut b) = self.buf.lock() { b.push_output(text); }
    }
    fn write_error(&mut self, text: &str) {
        if let Ok(mut b) = self.buf.lock() { b.push_error(text); }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────

fn line_color(theme: &Theme, kind: LineKind) -> Color {
    match kind {
        LineKind::Input | LineKind::Continuation => theme.console_input,
        LineKind::Output                          => theme.console_output,
        LineKind::Error                           => theme.console_error,
        LineKind::Banner                          => theme.console_banner,
    }
}

fn rows_from_buffer(buf: &ConsoleBuffer, theme: &Theme) -> Vec<Vec<Cell>> {
    buf.transcript().iter()
        .map(|cl| {
            let col = line_color(theme, cl.kind);
            cl.text.chars().map(|c| Cell::plain(c, col)).collect()
        })
        .collect()
}

/// Capture the engine's current SVG plot, if any. Returns `None` when
/// no plot has been produced.
fn take_engine_svg() -> Option<String> {
    if !r2_graphics::device::current_has_plot() { return None; }
    let svg = r2_graphics::device::with_device(|d| d.full_svg());
    if svg.is_empty() { None } else { Some(svg) }
}

/// Drive one user-submitted source string through the engine — parse
/// each top-level statement, evaluate it, apply R's auto-print rule
/// (silent for assignments / control flow / side-effect calls), short-
/// circuit on q() / quit() by setting `quit_requested`.
fn run_source(
    src: &str,
    engine: &mut Engine,
    buffer: &Arc<Mutex<ConsoleBuffer>>,
    quit_requested: &Rc<RefCell<bool>>,
) {
    let stmts = match Parser::parse(src) {
        Ok(v)  => v,
        Err(e) => {
            buffer.lock().unwrap().push_error(&format!("Parse error: {}", e));
            return;
        }
    };
    for stmt in stmts {
        if r2_console::is_quit_call(&stmt) {
            *quit_requested.borrow_mut() = true;
            return;
        }
        let silent = r2_console::is_silent(&stmt);
        match engine.eval(&stmt) {
            Ok(val) => {
                if !silent {
                    buffer.lock().unwrap().push_output(&format!("{}", val));
                }
            }
            Err(err) => {
                buffer.lock().unwrap().push_error(&format!("Error: {:?}", err));
            }
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
    // The engine emits a `dev.view()`-style browser plot by default —
    // we have a native Graphics window, so disable that side-channel.
    r2_graphics::device::disable_autoview();
    std::env::set_var("R2_NO_AUTOVIEW", "1");

    let theme = Theme::khaki();

    // ── Shared state ───────────────────────────────────────────────
    let buffer = Arc::new(Mutex::new(ConsoleBuffer::new()));
    {
        let mut b = buffer.lock().unwrap();
        b.push_banner(&format!("Ardon-R2 {} — pure-Rust R", env!("CARGO_PKG_VERSION")));
        b.push_banner("Type expressions at the R2> prompt. Up/Down recalls history.");
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
    let input      = Rc::new(RefCell::new(InputField::new()));
    let quit_requested = Rc::new(RefCell::new(false));

    // ── Menu bars ──────────────────────────────────────────────────
    // Each sub-window owns its own menu set. The one currently
    // displayed depends on which window is topmost — paint / event
    // dispatch picks the right state every frame. Action strings
    // share a namespace so the central match in `on_frame` doesn't
    // care which menu fired the event.

    // Console menu — focused on REPL workflow.
    let mut mb_con = MenuBuilder::new();
    mb_con.top("File")
        .item("Clear console", "",       "file.clear")
        .item("Quit",          "Ctrl+Q", "file.quit");
    mb_con.top("Edit")
        .item("Copy",          "Ctrl+C", "edit.copy")
        .item("Paste",         "Ctrl+V", "edit.paste")
        .item("Select all",    "Ctrl+A", "edit.select_all");
    mb_con.top("Windows")
        .item("Show Console",  "", "win.console")
        .item("Show Graphics", "", "win.graphics");
    mb_con.top("Help")
        .item("About Ardon-R2", "", "help.about");
    let menu_console = Rc::new(RefCell::new(MenuBarState::new(mb_con.bar)));

    // Graphics menu — viewer-only. No Paste (a plot pane is output,
    // not an editor). Save/Copy are the meaningful actions.
    let mut mb_grf = MenuBuilder::new();
    mb_grf.top("File")
        .item("Save plot as SVG…", "Ctrl+S", "file.save_plot")
        .item("Save plot as PNG…", "",       "file.save_plot_png")
        .item("Copy plot as image","",       "file.copy_plot_image")
        .item("Copy plot SVG",     "",       "file.copy_plot")
        .item("Quit",              "Ctrl+Q", "file.quit");
    mb_grf.top("Windows")
        .item("Show Console",      "",       "win.console")
        .item("Show Graphics",     "",       "win.graphics");
    mb_grf.top("Help")
        .item("About Ardon-R2",    "",       "help.about");
    let menu_graphics = Rc::new(RefCell::new(MenuBarState::new(mb_grf.bar)));

    // ── Right-click context menus ──────────────────────────────────
    // Each sub-window owns one. Triggered on right-click inside the
    // window's content rect; paints LAST so it floats above
    // everything. Actions reuse the same dispatch table as the
    // top menu bar — one place to add a feature, two ways to reach it.
    let ctx_console = Rc::new(RefCell::new(ContextMenu::new(vec![
        ContextItem::new("Copy",       "edit.copy"),
        ContextItem::new("Paste",      "edit.paste"),
        ContextItem::new("Select all", "edit.select_all"),
        ContextItem::separator(),
        ContextItem::new("Clear console", "file.clear"),
    ])));
    let ctx_graphics = Rc::new(RefCell::new(ContextMenu::new(vec![
        ContextItem::new("Save plot as SVG…",   "file.save_plot"),
        ContextItem::new("Save plot as PNG…",   "file.save_plot_png"),
        ContextItem::separator(),
        ContextItem::new("Copy plot as image",  "file.copy_plot_image"),
        ContextItem::new("Copy plot SVG",       "file.copy_plot"),
    ])));

    // Title-bar logo — decoded + resampled to a small square at startup.
    // The actual atlas upload happens on the first frame (we need a
    // Renderer for that, and we only get one inside on_frame).
    let logo_square: Vec<u8>; let logo_side: u32;
    {
        const LOGO_BYTES: &[u8] = include_bytes!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/logo.png"));
        // Upload at 64 px (~3.5× the displayed 18-px title-bar icon).
        // The GPU bilinear filter handles the final downscale at draw
        // time, which keeps edges crisp without rasterising at every
        // possible target size.
        let side: u32 = 64;
        let img = image::load_from_memory(LOGO_BYTES)
            .map_err(|e| format!("logo decode: {}", e))?
            .into_rgba8();
        let (sw, sh) = (img.width(), img.height());
        let scale = side as f32 / sw.max(sh) as f32;
        let nw = (sw as f32 * scale).round() as u32;
        let nh = (sh as f32 * scale).round() as u32;
        // Triangle (bilinear) gives sharper results than Lanczos3 when
        // the source is much larger than the target — Lanczos's lobes
        // create soft halos at extreme downscale ratios.
        let resized = image::imageops::resize(
            &img, nw, nh, image::imageops::FilterType::Triangle);
        let mut canvas = image::RgbaImage::from_pixel(side, side,
            image::Rgba([255, 255, 255, 0]));
        let ox = (side - nw) / 2;
        let oy = (side - nh) / 2;
        image::imageops::overlay(&mut canvas, &resized, ox as i64, oy as i64);
        logo_square = canvas.into_raw();
        logo_side   = side;
    }
    let logo_uploaded = Rc::new(RefCell::new(false));
    let logo_handle: Rc<RefCell<Option<r2_ui::ImageHandle>>> = Rc::new(RefCell::new(None));

    let frame_counter = Rc::new(RefCell::new(0u64));
    // SVG cache key — re-rasterize the GraphPanel only when the engine
    // produces new SVG content. Comparing string length is cheap and
    // catches every plot-mutation we currently emit.
    let last_svg_len = Rc::new(RefCell::new(0usize));

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
            let frame_counter = frame_counter.clone();
            let last_svg_len  = last_svg_len.clone();
            let quit_requested = quit_requested.clone();
            let logo_uploaded  = logo_uploaded.clone();
            let logo_handle    = logo_handle.clone();
            move |ctx, renderer, frame, theme| {
                *frame_counter.borrow_mut() += 1;

                // First-frame: upload the title-bar logo and attach to
                // each sub-window. Atlas alloc happens once; the
                // ImageHandle is cheap to copy after.
                if !*logo_uploaded.borrow() {
                    if let Some(handle) = renderer.upload_image(
                        logo_side, logo_side, &logo_square)
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
                                // R-style: near-square default
                                // (~680×620). Cascade subsequent
                                // devices so multiple windows don't
                                // overlap identically.
                                let n = id.0 as f32;
                                let bounds = Rect {
                                    x: 700.0 + (n - 1.0) * 36.0,
                                    y:  36.0 + (n - 1.0) * 28.0,
                                    w: 680.0, h: 620.0,
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

                // Quit if engine asked.
                if *quit_requested.borrow() {
                    std::process::exit(0);
                }

                // ── Workspace
                let menu_rect = Rect { x: 0.0, y: 0.0, w: win_w, h: MENU_BAR_HEIGHT };
                let workspace = Rect { x: 0.0, y: MENU_BAR_HEIGHT,
                                       w: win_w, h: win_h - MENU_BAR_HEIGHT };
                mdi.borrow_mut().set_workspace(workspace);

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
                    .handle_events(ctx.events, menu_rect, renderer, theme);
                let cm_action = match topmost_now {
                    Some(id) if id == console_id => {
                        let content = mdi.borrow().window(console_id)
                            .map(|w| w.content_rect(theme));
                        content.and_then(|c| ctx_console.borrow_mut()
                            .handle_events(ctx.events, c, renderer, theme))
                    }
                    Some(id) if graphics_id == Some(id) => {
                        let content = graphics_id.and_then(|gid|
                            mdi.borrow().window(gid).map(|w| w.content_rect(theme)));
                        content.and_then(|c| ctx_graphics.borrow_mut()
                            .handle_events(ctx.events, c, renderer, theme))
                    }
                    _ => None,
                };
                if let Some(action) = mb_action.or(cm_action) {
                    match action.as_str() {
                        "file.quit"  => { *quit_requested.borrow_mut() = true; }
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
                mdi.borrow_mut().handle_events(ctx.events, theme);

                // ── Console keyboard input — ALWAYS active so the console
                //    stays typeable regardless of which MDI window is
                //    topmost (RGui keeps the console interactive; a plot no
                //    longer "steals" the keyboard). Clicking a window still
                //    raises it via the MDI handler above.
                let topmost = mdi.borrow().z_order().last();
                {
                    let mut input_mut = input.borrow_mut();
                    let resp = input_mut.handle_events(ctx.events, ctx.clipboard);

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
                    let (cell_w, line_h) = renderer.cell_metrics(theme.font_size);
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
                        // Skip selection events on the frame a context
                        // menu was open / fired — the click that picked
                        // a menu item would otherwise also reach the
                        // grid and collapse the selection.
                        if !ctx_was_open {
                            let _copied = grid_state.borrow_mut().handle_events(
                                ctx.events, &rows, grid_rect,
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
                        let (cell_w, line_h) = renderer.cell_metrics(theme.font_size);
                        // R Console convention: white body.
                        frame.paint_rect(content.x, content.y, content.w, content.h,
                                         Color::WHITE);

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

                        let mut rows = rows_from_buffer(&buffer.lock().unwrap(), theme);
                        let input_ref = input.borrow();
                        // Build the live prompt row: "<prompt> <typed text>"
                        // in console-input color, appended to the transcript.
                        let prompt_row: Vec<Cell> = {
                            let prefix = format!("{} ", input_ref.prompt);
                            let full = format!("{}{}", prefix, input_ref.current);
                            full.chars().map(|c| Cell::plain(c, theme.console_input)).collect()
                        };
                        let prompt_row_index = rows.len();
                        let cursor_col_in_row = input_ref.prompt.chars().count() + 1
                            + input_ref.current[..input_ref.cursor].chars().count();
                        rows.push(prompt_row);

                        // ── Drive the scrollbars from current content
                        //     vs viewport sizes (in cell units).
                        let total_rows  = rows.len();
                        let max_cols    = rows.iter().map(|r| r.len()).max().unwrap_or(0)
                                          .max(cursor_col_in_row + 1);
                        let visible_rows = (grid_rect.h / line_h).floor() as usize;
                        let visible_cols = (grid_rect.w / cell_w).floor() as usize;
                        if total_rows > 0 {
                            vscroll.borrow_mut().visible_fraction =
                                (visible_rows as f32 / total_rows as f32).min(1.0);
                        }
                        if max_cols > 0 {
                            hscroll.borrow_mut().visible_fraction =
                                (visible_cols as f32 / max_cols as f32).min(1.0);
                        }
                        if let Some(p) = vscroll.borrow_mut().handle_events(ctx.events, vtrack) {
                            let off = r2_ui::scroll_pos_to_row(p, total_rows, visible_rows);
                            // Pin to manual offset; if user dragged to
                            // the bottom, hand control back to
                            // auto-scroll so new lines keep showing.
                            grid_state.borrow_mut().scroll_y_override =
                                if off + visible_rows >= total_rows { None } else { Some(off) };
                        }
                        if let Some(p) = hscroll.borrow_mut().handle_events(ctx.events, htrack) {
                            grid_state.borrow_mut().scroll_x =
                                r2_ui::scroll_pos_to_col(p, max_cols, visible_cols);
                        }

                        // ── Transcript paint — uses the scroll state
                        //     CellGridState now owns.
                        grid_state.borrow().paint(frame, renderer, &rows, grid_rect,
                                                  cell_w, line_h, theme.font_size, theme);

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
            }
        })
        .run()
}
