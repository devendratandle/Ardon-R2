//! In-memory plot device for the graphics crate.
//!
//! Replaces the previous file-based plot-state model (which detected
//! "is a plot open" by reading `plot.svg` from cwd). The file model
//! caused test-isolation races on Windows when `cargo test` runs
//! sibling tests in parallel — one test's `plot.svg` would leak into
//! another test's "no plot open" precondition.
//!
//! The new model:
//!   1. A `PlotDevice` holds the SVG body, the canvas size, the
//!      `PlotParams` (everything `par()` can set), and a panel cursor
//!      for multi-panel `mfrow`/`mfcol` layouts.
//!   2. The device lives in `thread_local!` storage so concurrent
//!      tests do not collide, and the production REPL still has a
//!      single per-thread device.
//!   3. Plot functions call `begin_plot()` to obtain the rectangle
//!      they should draw into (respecting multi-panel layout) and
//!      `append_svg()` to write fragments. Overlays
//!      (`lines`/`points`/`abline`/`legend`) use `append_svg()`
//!      directly; the function errors if no plot is open.
//!   4. The full SVG is materialized on demand via `full_svg()` and
//!      flushed to disk by `save_to_file()` — either auto-saved by
//!      the plot function to preserve existing UX, or explicitly via
//!      `dev.off()` / `save_plot()`.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use r2_types::{ErrKind, R2Err};

/// Whether the browser plot viewer may auto-open on the first plot.
/// **Opt-IN** (default `false`) — mirroring R, where no graphics device
/// opens unless the frontend installs one. Only the *interactive* CLI
/// enables it (`enable_autoview()`); scripts, the test suite, and the
/// GUI (which has its own plot window) never trigger a browser.
static AUTOVIEW_ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether a live plot display is present (CLI browser viewer or the GUI
/// window). When true, `plot()` only *displays* — it does NOT auto-write
/// a file (the user saves explicitly via `save_plot()` / a filename).
/// When false (headless script), plots auto-save a default `.svg` so the
/// output isn't lost. Set by `enable_autoview()` (CLI) and
/// `set_display_present()` (GUI).
static DISPLAY_PRESENT: AtomicBool = AtomicBool::new(false);

/// Opt in to the browser plot viewer for the lifetime of this process.
/// Called by the interactive REPL so `plot()` pops a live viewer, the
/// way RGui/RStudio open a device. Idempotent. Also marks a display as
/// present so plots stop auto-writing files.
pub fn enable_autoview() {
    AUTOVIEW_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    DISPLAY_PRESENT.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Force the browser viewer off (default state). Kept for hosts that
/// previously called it; with opt-in semantics the default is already
/// off, so this is only needed to override a prior `enable_autoview()`.
pub fn disable_autoview() {
    AUTOVIEW_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Declare that a live plot display exists (the GUI calls this — it has
/// its own Graphics window). Plots then display only; saving is explicit.
pub fn set_display_present(present: bool) {
    DISPLAY_PRESENT.store(present, std::sync::atomic::Ordering::Relaxed);
}

/// True if a live display (browser viewer or GUI window) is present.
pub fn display_present() -> bool {
    DISPLAY_PRESENT.load(std::sync::atomic::Ordering::Relaxed)
}

/// True when the interactive-CLI browser viewer is the active display.
pub fn autoview_enabled() -> bool {
    AUTOVIEW_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The current SCREEN-device plot, published here so the browser server
/// (a SEPARATE thread) can serve the *live* plot from memory. The device
/// itself is thread-local to the eval thread, so without this shared
/// buffer the server could only scan the working directory for stray
/// `.svg` files — which showed stale/leftover plots. Updated after every
/// high-level plot in interactive-CLI mode; replaced (not accumulated),
/// so a new `plot()`/`hist()` shows the latest, exactly like R's device.
static SCREEN_SVG: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Publish the current plot SVG to the shared screen buffer.
pub fn set_screen_svg(svg: String) {
    if let Ok(mut g) = SCREEN_SVG.lock() { *g = Some(svg); }
}

/// The current screen plot SVG, if any (read by the browser server).
pub fn screen_svg() -> Option<String> {
    SCREEN_SVG.lock().ok().and_then(|g| g.clone())
}

/// Called by every high-level plot builtin when it finishes drawing.
/// Routes the result to the ACTIVE output — the one place that knows
/// CLI-vs-GUI-vs-script, so graphics behave like R:
///   * interactive CLI (browser): publish to the shared screen buffer →
///     the browser shows the current plot (replacing the previous), no
///     file written;
///   * GUI window: nothing — the Graphics window paints the device;
///   * headless script: auto-save `default_name` to cwd (R's Rplots
///     convention) and report the path.
/// (When a file device — `pdf()`/`png()`/`svg()` — is active, that path
/// is handled separately and takes precedence.)
/// Returns `Some(path)` only when it actually wrote a file (headless
/// script mode), so callers like `plot()` can echo the filename.
pub fn finish_plot(default_name: &str) -> Option<String> {
    // A file device (pdf/png/svg) accumulates until dev.off() — never
    // auto-display or auto-save its content mid-stream.
    if with_device(|d| d.file_target.is_some()) {
        return None;
    }
    if autoview_enabled() {
        let svg = with_device(|d| d.full_svg());
        set_screen_svg(svg);
        return None;
    }
    if display_present() {
        return None;
    }
    if save_to_file(default_name).is_ok() {
        if let Ok(abs) = std::fs::canonicalize(default_name) {
            let d = abs.to_string_lossy();
            soutln!("Plot saved to {}", d.strip_prefix(r"\\?\").unwrap_or(&d));
        } else {
            soutln!("Plot saved to {}", default_name);
        }
        Some(default_name.to_string())
    } else {
        None
    }
}

/// Everything `par()` can set. Defaults mirror R's `par()` baseline so
/// scripts that do not call `par()` get R-compatible output.
#[derive(Debug, Clone)]
pub struct PlotParams {
    /// Multi-panel grid filled row-by-row. Mutually exclusive with `mfcol`.
    pub mfrow: Option<(u32, u32)>,
    /// Multi-panel grid filled column-by-column. Mutually exclusive with `mfrow`.
    pub mfcol: Option<(u32, u32)>,

    /// Inner margins in "lines" (bottom, left, top, right). R default `5.1, 4.1, 4.1, 2.1`.
    pub mar: [f64; 4],
    /// Outer margins. R default all zero.
    pub oma: [f64; 4],

    /// Text scale. R default 1.0.
    pub cex: f64,
    pub cex_axis: f64,
    pub cex_lab: f64,
    pub cex_main: f64,

    pub col: String,
    pub bg:  String,
    pub fg:  String,

    pub lty: String,
    pub lwd: f64,
    pub pch: i32,
    pub las: i32,

    /// If true, the next `plot()` overlays on the current panel instead of advancing.
    pub new: bool,
}

impl Default for PlotParams {
    fn default() -> Self {
        Self {
            mfrow: None,
            mfcol: None,
            mar: [5.1, 4.1, 4.1, 2.1],
            oma: [0.0; 4],
            cex: 1.0,
            cex_axis: 1.0,
            cex_lab: 1.0,
            cex_main: 1.2,
            col: "black".into(),
            bg:  "white".into(),
            fg:  "black".into(),
            lty: "solid".into(),
            lwd: 1.0,
            pch: 1,
            // R2-default: both axis labels horizontal (las=1) so they
            // read left-to-right without head-tilting.
            las: 1,
            new: false,
        }
    }
}

/// The active plot's coordinate system — the connective tissue that lets
/// overlay functions (`points`/`lines`/`abline`/`text`/`rect`/…) draw in
/// *data* coordinates that align with the base plot. A high-level plot
/// (`plot`/`hist`/`curve`/…) records this after laying out its panel; the
/// overlays map data → pixels through [`PlotCoords::to_px`]. This is what
/// makes R's incremental "draw, then add" graphics model work.
#[derive(Debug, Clone, Copy)]
pub struct PlotCoords {
    /// Inner plotting region in canvas pixels (top-left + size).
    pub px0: f64, pub py0: f64, pub pw: f64, pub ph: f64,
    /// Data-space ranges mapped onto that region.
    pub xmin: f64, pub xmax: f64, pub ymin: f64, pub ymax: f64,
}

impl PlotCoords {
    /// Map a data-space `(x, y)` to canvas pixels (y inverted: data up =
    /// pixels down). Degenerate ranges collapse to a unit span so a single
    /// point still lands inside the region.
    pub fn to_px(&self, x: f64, y: f64) -> (f64, f64) {
        let xr = if (self.xmax - self.xmin).abs() < 1e-12 { 1.0 } else { self.xmax - self.xmin };
        let yr = if (self.ymax - self.ymin).abs() < 1e-12 { 1.0 } else { self.ymax - self.ymin };
        let sx = self.px0 + (x - self.xmin) / xr * self.pw;
        let sy = self.py0 + self.ph - (y - self.ymin) / yr * self.ph;
        (sx, sy)
    }
}

/// In-memory canvas. Holds the accumulated SVG body and the panel cursor.
#[derive(Debug, Clone)]
pub struct PlotDevice {
    /// Concatenated SVG fragments — placed between `<svg ...>` and `</svg>` at render time.
    pub svg_body: String,
    pub has_plot: bool,
    pub width: f64,
    pub height: f64,
    pub params: PlotParams,
    /// Index of the next panel to fill (0-indexed, wraps on `mfrow`/`mfcol` overflow).
    pub panel_cursor: u32,
    /// Coordinate system of the most recent high-level plot, so overlays
    /// align with it. `None` until the first plot of a figure.
    pub coords: Option<PlotCoords>,
    /// For a *file device* (`pdf()`/`png()`/`svg()`): where `dev.off()`
    /// writes. `None` for an interactive/screen device.
    pub file_target: Option<FileTarget>,
    /// Completed pages for a multi-page file device (pdf): each `plot.new()`
    /// / new high-level plot finishes the current page and starts a new one.
    /// `dev.off()` writes them all (+ the in-progress page) as one PDF.
    pub pages: Vec<String>,
}

/// Destination of a file device — set by `pdf()`/`png()`/`svg()`, written
/// by `dev.off()`. Format is inferred from the path extension.
#[derive(Debug, Clone)]
pub struct FileTarget {
    pub path: String,
    pub width: u32,
    pub height: u32,
}

impl Default for PlotDevice {
    fn default() -> Self {
        Self {
            svg_body: String::new(),
            has_plot: false,
            width: 600.0,
            height: 400.0,
            params: PlotParams::default(),
            panel_cursor: 0,
            coords: None,
            file_target: None,
            pages: Vec::new(),
        }
    }
}

impl PlotDevice {
    /// Compute the rectangle the next plot should draw into.
    /// Returns `(x, y, panel_width, panel_height)` in canvas coordinates,
    /// and advances `panel_cursor` for the subsequent plot call.
    pub fn next_panel_rect(&mut self) -> (f64, f64, f64, f64) {
        let (rows, cols, col_major) = match (self.params.mfrow, self.params.mfcol) {
            (Some((r, c)), _) => (r as usize, c as usize, false),
            (_, Some((r, c))) => (r as usize, c as usize, true),
            (None, None) => return (0.0, 0.0, self.width, self.height),
        };
        let total = (rows * cols).max(1);
        let idx = (self.panel_cursor as usize) % total;
        let (row, col) = if col_major {
            (idx % rows, idx / rows)
        } else {
            (idx / cols, idx % cols)
        };
        let pw = self.width / cols as f64;
        let ph = self.height / rows as f64;
        let x = col as f64 * pw;
        let y = row as f64 * ph;
        self.panel_cursor = self.panel_cursor.wrapping_add(1);
        (x, y, pw, ph)
    }

    /// Materialize the full SVG document.
    pub fn full_svg(&self) -> String {
        let mut s = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );
        s.push_str(&format!(r#"<rect width="100%" height="100%" fill="{}"/>"#, self.params.bg));
        s.push_str(&self.svg_body);
        s.push_str("</svg>");
        s
    }
}

// ─── Multi-device support — session B ──────────────────────────────
//
// The legacy `DEVICE` thread-local is replaced by a `DeviceTable` that
// holds a `BTreeMap<DeviceId, PlotDevice>` plus a `current` pointer.
// Existing callers reach the "currently active" device via the
// unchanged `with_device(...)` helper — that now lazily creates device
// id 1 on first use so a plain `plot()` (no preceding `dev.new()`)
// still works identically to before.

/// Unique handle for an open plot device. Engine returns this as an
/// integer scalar from `dev.new()`, `dev.set()`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub u32);

/// Events the GUI host polls each frame so it can spawn / hide /
/// repaint sub-windows in lock-step with engine commands.
#[derive(Debug, Clone)]
pub enum DeviceEvent {
    Created(DeviceId),
    Closed(DeviceId),
    Plotted(DeviceId),
    CurrentChanged(DeviceId),
}

pub struct DeviceTable {
    pub devices:        std::collections::BTreeMap<DeviceId, PlotDevice>,
    pub current:        Option<DeviceId>,
    pub next_id:        u32,
    pub pending_events: Vec<DeviceEvent>,
}

impl DeviceTable {
    fn new() -> Self {
        Self {
            devices: std::collections::BTreeMap::new(),
            current: None,
            next_id: 1,
            pending_events: Vec::new(),
        }
    }

    /// Lazily ensure at least one device exists and return its id.
    /// Used by the `with_device` shim for code that pre-dates the
    /// multi-device API.
    fn ensure_current(&mut self) -> DeviceId {
        if let Some(id) = self.current { return id; }
        self.spawn()
    }

    fn spawn(&mut self) -> DeviceId {
        let id = DeviceId(self.next_id);
        self.next_id += 1;
        self.devices.insert(id, PlotDevice::default());
        self.current = Some(id);
        self.pending_events.push(DeviceEvent::Created(id));
        self.pending_events.push(DeviceEvent::CurrentChanged(id));
        id
    }
}

thread_local! {
    pub(crate) static DEVICE_TABLE: RefCell<DeviceTable> = RefCell::new(DeviceTable::new());
}

// ─── Public access surface (used by plots.rs, overlays.rs, params.rs) ──

pub fn with_device<R, F: FnOnce(&mut PlotDevice) -> R>(f: F) -> R {
    DEVICE_TABLE.with(|t| {
        let mut tbl = t.borrow_mut();
        let id = tbl.ensure_current();
        f(tbl.devices.get_mut(&id).expect("device just ensured"))
    })
}

/// Open a fresh device and make it current. Returns its id.
pub fn new_device() -> DeviceId {
    DEVICE_TABLE.with(|t| t.borrow_mut().spawn())
}

/// Open a *file device* (`pdf()`/`png()`/`svg()`): spawn a device, size
/// it, and record where `dev.off()` should write. Returns its id.
pub fn open_file_device(path: &str, width: u32, height: u32) -> DeviceId {
    let id = new_device();
    with_device(|d| {
        d.width = width as f64;
        d.height = height as f64;
        d.file_target = Some(FileTarget { path: path.to_string(), width, height });
    });
    id
}

/// The current device's file target, if it is a file device.
pub fn current_file_target() -> Option<FileTarget> {
    with_device(|d| d.file_target.clone())
}

/// Set the active device. Returns the *previous* current id, if any.
/// `None` return means the requested id was not open.
pub fn set_device(id: DeviceId) -> Option<DeviceId> {
    DEVICE_TABLE.with(|t| {
        let mut tbl = t.borrow_mut();
        if !tbl.devices.contains_key(&id) { return None; }
        let prev = tbl.current;
        tbl.current = Some(id);
        tbl.pending_events.push(DeviceEvent::CurrentChanged(id));
        prev
    })
}

/// Close the device with `id` (or the current device when `id` is
/// `None`). If the closed device was current, current shifts to the
/// next open device (highest remaining id) or `None` if none remain.
/// Returns the new current id.
pub fn close_device(id: Option<DeviceId>) -> Option<DeviceId> {
    DEVICE_TABLE.with(|t| {
        let mut tbl = t.borrow_mut();
        let target = id.or(tbl.current)?;
        if tbl.devices.remove(&target).is_some() {
            tbl.pending_events.push(DeviceEvent::Closed(target));
        }
        if tbl.current == Some(target) {
            tbl.current = tbl.devices.keys().next_back().copied();
            if let Some(new_cur) = tbl.current {
                tbl.pending_events.push(DeviceEvent::CurrentChanged(new_cur));
            }
        }
        tbl.current
    })
}

pub fn list_devices() -> Vec<DeviceId> {
    DEVICE_TABLE.with(|t| t.borrow().devices.keys().copied().collect())
}

pub fn current_device() -> Option<DeviceId> {
    DEVICE_TABLE.with(|t| t.borrow().current)
}

/// Drain pending device events. The GUI host calls this each frame.
pub fn drain_events() -> Vec<DeviceEvent> {
    DEVICE_TABLE.with(|t| {
        let mut tbl = t.borrow_mut();
        std::mem::take(&mut tbl.pending_events)
    })
}

/// Mark the current device as having received fresh plot output —
/// emit a `Plotted` event so the host can refresh its window.
pub fn notify_plotted() {
    DEVICE_TABLE.with(|t| {
        let mut tbl = t.borrow_mut();
        if let Some(id) = tbl.current {
            tbl.pending_events.push(DeviceEvent::Plotted(id));
        }
    });
}

/// Get the full SVG of a specific device. Used by the GUI when it
/// gets a `Plotted(id)` event to fetch that device's content.
pub fn device_full_svg(id: DeviceId) -> Option<String> {
    DEVICE_TABLE.with(|t| t.borrow().devices.get(&id).map(|d| d.full_svg()))
}

/// Has any plot been opened in this device? Source of truth for overlay
/// preconditions — replaces the previous file-existence check.
pub fn current_has_plot() -> bool {
    DEVICE_TABLE.with(|t| {
        let tbl = t.borrow();
        tbl.current.and_then(|id| tbl.devices.get(&id)).map(|d| d.has_plot).unwrap_or(false)
    })
}

/// Begin a new plot. Returns the canvas-coordinate rectangle the plot
/// should draw into. Honors `par(mfrow=...)` / `par(mfcol=...)` multi-panel
/// layout: when the panel cursor is at 0 (or no multi-panel is set), the
/// SVG body is cleared. When in the middle of a panel cycle, the previous
/// panels' content is preserved and the new plot is placed in the next slot.
pub fn begin_plot() -> (f64, f64, f64, f64) {
    // Phase R.G.4 — auto-launch the live browser plot viewer on first
    // plot of the session. Without this, users see SVG/PNG files written
    // to disk but no graphical window — confusing if they expected
    // RStudio/Rgui behavior. The browser stays open across the session
    // and live-refreshes after every plot.
    //
    // Opt-out: set R2_NO_AUTOVIEW=1 in the environment.
    static AUTOVIEW_LAUNCHED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let env_disabled = std::env::var("R2_NO_AUTOVIEW").is_ok();
    let enabled = AUTOVIEW_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    // Opt-in: only auto-open when a frontend enabled it (interactive CLI).
    // Default off → scripts, tests, and the GUI never spawn a browser.
    if AUTOVIEW_LAUNCHED.get().is_none() && enabled && !env_disabled {
        let _ = AUTOVIEW_LAUNCHED.set(());
        if let Some(port) = crate::server::ensure_started() {
            soutln!("Plot viewer opened in browser: http://127.0.0.1:{}/", port);
            soutln!("  (set R2_NO_AUTOVIEW=1 to disable, or close the tab any time.)");
            crate::server::open_browser(port);
        }
    }

    let rect = with_device(|dev| {
        // Number of panels in the current layout (1 when no mfrow/mfcol).
        let panels = match (dev.params.mfrow, dev.params.mfcol) {
            (Some((r, c)), _) | (_, Some((r, c))) => (r.max(1)) * (c.max(1)),
            _ => 1,
        };
        // Start a fresh page (clear the canvas) at every page boundary:
        // ALWAYS for a single panel, and when the cursor wraps for a grid.
        // (The old `panel_cursor == 0` check left mfrow=(1,1) — and any
        // plot after a grid filled — drawing on top of the previous one,
        // so plots piled up and persisted across runs.)
        if panels <= 1 || dev.panel_cursor % panels == 0 {
            // On a file device, a cleared frame is a FINISHED page — keep it
            // so dev.off() can emit a multi-page PDF.
            if dev.file_target.is_some() && dev.has_plot && !dev.svg_body.is_empty() {
                let page = dev.full_svg();
                dev.pages.push(page);
            }
            dev.svg_body.clear();
        }
        dev.has_plot = true;
        dev.next_panel_rect()
    });
    // Tell the GUI host (if any) that this device just got fresh
    // plot content — it can fetch the SVG and refresh its window.
    notify_plotted();
    rect
}

/// Append a raw SVG fragment to the device. Errors if no plot is open.
/// Used by overlay builtins (`lines`, `points`, `abline`, `legend`).
pub fn append_svg(fragment: &str) -> Result<(), R2Err> {
    with_device(|dev| {
        if !dev.has_plot {
            return Err(R2Err {
                msg: "no plot open — call plot() first".into(),
                kind: ErrKind::Runtime,
            });
        }
        dev.svg_body.push_str(fragment);
        Ok(())
    })
}

/// Flush the current device contents to a file. Does not modify device state.
pub fn save_to_file(path: &str) -> Result<(), std::io::Error> {
    let svg = with_device(|d| d.full_svg());
    std::fs::write(path, svg)
}

/// Process-wide font database, loaded once. `load_system_fonts()` scans
/// the whole system font directory (hundreds of files on Windows); doing
/// it on every render is the graphics cold-start. Cache + share it.
fn shared_fontdb() -> std::sync::Arc<usvg::fontdb::Database> {
    static DB: std::sync::OnceLock<std::sync::Arc<usvg::fontdb::Database>> =
        std::sync::OnceLock::new();
    DB.get_or_init(|| {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        std::sync::Arc::new(db)
    })
    .clone()
}

/// Pre-load the shared font database (call once at startup, ideally on a
/// background thread) so the first PNG/RGBA render skips the font scan.
pub fn warm_fonts() {
    let _ = shared_fontdb();
}

/// Rasterize the current plot to a fresh RGBA buffer at the given
/// pixel dimensions. Used by "Copy plot as image" — same render
/// path as `save_to_png` but returns the bytes instead of writing
/// to disk. Returns `(rgba, width, height)`.
pub fn render_to_rgba(target_w: u32, target_h: u32) -> Result<(Vec<u8>, u32, u32), R2Err> {
    let svg = with_device(|d| d.full_svg());
    let mut opt = usvg::Options::default();
    opt.fontdb = shared_fontdb();
    let tree = usvg::Tree::from_str(&svg, &opt)
        .map_err(|e| R2Err { msg: format!("svg→rgba: parse failed: {}", e), kind: ErrKind::Runtime })?;
    let svg_size = tree.size();
    let sw = svg_size.width().max(1.0);
    let sh = svg_size.height().max(1.0);
    // Fit-with-aspect inside the requested bounding box. Output
    // pixmap matches the SVG's aspect ratio — no transparent
    // borders, no whitespace when the clipboard image is pasted
    // into Word / Outlook / Excel. The earlier code created a
    // pixmap of exactly (target_w, target_h) and rendered into a
    // sub-region, which left ~30 % blank space when the Graphics
    // window aspect didn't match the SVG's 1.5:1.
    let scale = (target_w as f32 / sw).min(target_h as f32 / sh);
    let out_w = (sw * scale).round().max(1.0) as u32;
    let out_h = (sh * scale).round().max(1.0) as u32;
    let mut pixmap = tiny_skia::Pixmap::new(out_w, out_h)
        .ok_or_else(|| R2Err { msg: format!("svg→rgba: cannot allocate {}×{} pixmap", out_w, out_h), kind: ErrKind::Runtime })?;
    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    Ok((pixmap.data().to_vec(), out_w, out_h))
}

/// Rasterize the current SVG plot to a PNG file. Uses resvg under the
/// hood — pure Rust, no external dependencies. Renders at the device's
/// own pixel dimensions; the user can scale by passing different
/// width/height via the `png()` device builtin.
pub fn save_to_png(path: &str, target_w: u32, target_h: u32) -> Result<(), R2Err> {
    let svg = with_device(|d| d.full_svg());
    let mut opt = usvg::Options::default();
    // Shared font database (loaded once) so axis labels, titles, and
    // legend text render without re-scanning the system fonts each call.
    opt.fontdb = shared_fontdb();
    let tree = usvg::Tree::from_str(&svg, &opt)
        .map_err(|e| R2Err { msg: format!("svg→png: parse failed: {}", e), kind: ErrKind::Runtime })?;
    let mut pixmap = tiny_skia::Pixmap::new(target_w, target_h)
        .ok_or_else(|| R2Err { msg: format!("svg→png: cannot allocate {}×{} pixmap", target_w, target_h), kind: ErrKind::Runtime })?;
    // Compute the scale that fits the SVG into the target box.
    let svg_size = tree.size();
    let sx = target_w as f32 / svg_size.width();
    let sy = target_h as f32 / svg_size.height();
    let scale = sx.min(sy);
    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    pixmap.save_png(path)
        .map_err(|e| R2Err { msg: format!("svg→png: write failed: {}", e), kind: ErrKind::Runtime })
}

/// Rasterize-free vector PDF via `svg2pdf` (pure Rust). Matches the
/// SVG exactly — resolution-independent, ideal for papers / LaTeX.
///
/// Emits a **multi-page** PDF: every `plot.new()` / new high-level plot
/// on a file device finishes the current page (see `begin_plot`), and
/// `dev.pages` holds the completed pages; the in-progress canvas is the
/// final page. Each page's SVG is converted to a PDF form XObject via
/// `svg2pdf::to_chunk`, then `pdf_writer` stitches one Page per XObject.
pub fn save_to_pdf(path: &str) -> Result<(), R2Err> {
    // Completed pages + the in-progress canvas as the last page.
    let pages: Vec<String> = with_device(|d| {
        let mut p = d.pages.clone();
        if d.has_plot && !d.svg_body.is_empty() {
            p.push(d.full_svg());
        }
        p
    });
    let pages = if pages.is_empty() {
        // No plotting happened — emit the (blank) canvas as one page.
        vec![with_device(|d| d.full_svg())]
    } else {
        pages
    };

    use pdf_writer::{Content, Name, Pdf, Rect, Ref};
    use std::collections::HashMap;

    let mut opt = usvg::Options::default();
    opt.fontdb = shared_fontdb();

    let mut alloc = Ref::new(1);
    let catalog_id = alloc.bump();
    let page_tree_id = alloc.bump();

    // First pass: convert each page SVG → renumbered chunk + per-page refs.
    struct PageRefs {
        page_id: Ref,
        content_id: Ref,
        svg_id: Ref,
        chunk: pdf_writer::Chunk,
        w: f32,
        h: f32,
    }
    let mut built: Vec<PageRefs> = Vec::with_capacity(pages.len());
    for svg in &pages {
        let tree = usvg::Tree::from_str(svg, &opt)
            .map_err(|e| R2Err { msg: format!("svg→pdf: parse failed: {}", e), kind: ErrKind::Runtime })?;
        let size = tree.size();
        // svg px are at 96 dpi; PDF points are 72 dpi → ×0.75 keeps the
        // physical size faithful (a 6in pdf() page stays 6in = 432pt).
        let (w, h) = (size.width() * 0.75, size.height() * 0.75);
        let (chunk, svg_ref) = svg2pdf::to_chunk(&tree, svg2pdf::ConversionOptions::default())
            .map_err(|e| R2Err { msg: format!("svg→pdf: {}", e), kind: ErrKind::Runtime })?;
        // Renumber the chunk's refs into our global allocator's space.
        let mut map: HashMap<Ref, Ref> = HashMap::new();
        let chunk = chunk.renumber(|old| *map.entry(old).or_insert_with(|| alloc.bump()));
        let svg_id = *map.get(&svg_ref).expect("svg root ref present in chunk");
        built.push(PageRefs {
            page_id: alloc.bump(),
            content_id: alloc.bump(),
            svg_id,
            chunk,
            w,
            h,
        });
    }

    let mut pdf = Pdf::new();
    pdf.catalog(catalog_id).pages(page_tree_id);
    pdf.pages(page_tree_id)
        .kids(built.iter().map(|p| p.page_id))
        .count(built.len() as i32);

    let svg_name = Name(b"S1");
    for p in &built {
        {
            let mut page = pdf.page(p.page_id);
            page.media_box(Rect::new(0.0, 0.0, p.w, p.h));
            page.parent(page_tree_id);
            page.contents(p.content_id);
            page.resources().x_objects().pair(svg_name, p.svg_id);
        } // page dropped → finished
        // svg2pdf XObjects are drawn in the unit square; scale to the page.
        let mut content = Content::new();
        content.transform([p.w, 0.0, 0.0, p.h, 0.0, 0.0]).x_object(svg_name);
        pdf.stream(p.content_id, &content.finish());
        pdf.extend(&p.chunk);
    }

    std::fs::write(path, pdf.finish())
        .map_err(|e| R2Err { msg: format!("svg→pdf: write failed: {}", e), kind: ErrKind::Runtime })
}

/// Dispatch on file extension: `.svg` → save_to_file, `.png` → save_to_png,
/// `.pdf` → save_to_pdf. Returns the absolute (canonicalized) path.
pub fn save_plot(path: &str, width: u32, height: u32) -> Result<std::path::PathBuf, R2Err> {
    let lower = path.to_lowercase();
    if lower.ends_with(".svg") {
        save_to_file(path).map_err(|e| R2Err {
            msg: format!("could not write {}: {}", path, e),
            kind: ErrKind::Runtime,
        })?;
    } else if lower.ends_with(".png") {
        save_to_png(path, width, height)?;
    } else if lower.ends_with(".pdf") {
        save_to_pdf(path)?;
    } else {
        return Err(R2Err {
            msg: format!("save_plot(): unsupported extension in '{}'. Use .svg, .png, or .pdf.", path),
            kind: ErrKind::Runtime,
        });
    }
    Ok(std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path)))
}

/// Legacy `dev.off()` equivalent kept for backward compatibility —
/// closes the *current* device. New code should call
/// [`close_device(None)`] directly.
pub fn dev_off() { let _ = close_device(None); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_device_has_no_plot() {
        // Tests share thread-local state — explicitly reset first.
        dev_off();
        assert!(!current_has_plot());
    }

    #[test]
    fn begin_plot_sets_has_plot_true_and_returns_full_canvas_by_default() {
        dev_off();
        let (x, y, w, h) = begin_plot();
        assert!(current_has_plot());
        assert_eq!((x, y), (0.0, 0.0));
        assert!(w > 0.0 && h > 0.0);
    }

    #[test]
    fn append_errors_when_no_plot_open() {
        dev_off();
        let r = append_svg("<circle cx=\"1\" cy=\"2\" r=\"3\"/>");
        assert!(r.is_err());
    }

    #[test]
    fn mfrow_2x2_advances_through_four_panels_then_wraps() {
        dev_off();
        with_device(|d| d.params.mfrow = Some((2, 2)));
        let r0 = begin_plot();
        let r1 = begin_plot();
        let r2 = begin_plot();
        let r3 = begin_plot();
        let r4 = begin_plot();
        // Row-major fill: (0,0), (0,c), (r,0), (r,c), then back to (0,0).
        assert_eq!((r0.0, r0.1), (0.0, 0.0));
        assert_eq!((r1.0, r1.1), (300.0, 0.0));
        assert_eq!((r2.0, r2.1), (0.0, 200.0));
        assert_eq!((r3.0, r3.1), (300.0, 200.0));
        assert_eq!((r4.0, r4.1), (0.0, 0.0));
        dev_off();
    }
}
