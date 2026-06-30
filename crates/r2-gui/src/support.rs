//! Free helpers for the GUI binary: the engine output sink,
//! console->cell rendering, plot capture, the REPL run loop, and
//! home-directory selection. Pure moves out of main.rs (no behaviour
//! change) so main.rs is just window setup + the event/paint loop.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use r2_console::{ConsoleBuffer, LineKind, OutputSink};
use r2_engine::Engine;
use r2_parser::Parser;
use r2_ui::{Cell, Color, Theme};

pub(crate) struct GuiSink {
    pub(crate) buf: Arc<Mutex<ConsoleBuffer>>,
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

pub(crate) fn line_color(theme: &Theme, kind: LineKind) -> Color {
    match kind {
        LineKind::Input | LineKind::Continuation => theme.console_input,
        LineKind::Output                          => theme.console_output,
        LineKind::Error                           => theme.console_error,
        LineKind::Banner                          => theme.console_banner,
    }
}

pub(crate) fn rows_from_buffer(buf: &ConsoleBuffer, theme: &Theme) -> Vec<Vec<Cell>> {
    buf.transcript().iter()
        .map(|cl| {
            let col = line_color(theme, cl.kind);
            cl.text.chars().map(|c| Cell::plain(c, col)).collect()
        })
        .collect()
}

/// Capture the engine's current SVG plot, if any. Returns `None` when
/// no plot has been produced.
pub(crate) fn take_engine_svg() -> Option<String> {
    if !r2_graphics::device::current_has_plot() { return None; }
    let svg = r2_graphics::device::with_device(|d| d.full_svg());
    if svg.is_empty() { None } else { Some(svg) }
}

/// Drive one user-submitted source string through the engine — parse
/// each top-level statement, evaluate it, apply R's auto-print rule
/// (silent for assignments / control flow / side-effect calls), short-
/// circuit on q() / quit() by setting `quit_requested`.
pub(crate) fn run_source(
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
        match engine.eval(&stmt) {
            Ok(val) => {
                // Unified auto-print rule (shared with the CLI) — silent
                // set + NULL-invisibility, so both consoles behave identically.
                if r2_console::should_autoprint(&stmt, &val) {
                    buffer.lock().unwrap().push_output(&format!("{}", val));
                }
            }
            Err(err) => {
                buffer.lock().unwrap().push_error(&format!("Error: {:?}", err));
            }
        }
    }
}

/// Pick a writable default working directory (R convention): R2_HOME
/// override, else OneDrive/Windows Documents, else $HOME. Mirrors the
/// CLI's `pick_user_home`.
pub(crate) fn pick_user_home() -> Option<std::path::PathBuf> {
    if let Ok(custom) = std::env::var("R2_HOME") {
        let p = std::path::PathBuf::from(custom);
        if p.is_dir() { return Some(p); }
    }
    if let Ok(od) = std::env::var("OneDrive") {
        let p = std::path::PathBuf::from(&od).join("Documents");
        if p.is_dir() { return Some(p); }
    }
    if let Ok(user) = std::env::var("USERPROFILE") {
        let od = std::path::PathBuf::from(&user).join("OneDrive").join("Documents");
        if od.is_dir() { return Some(od); }
        let docs = std::path::PathBuf::from(user).join("Documents");
        if docs.is_dir() { return Some(docs); }
    }
    if let Ok(home) = std::env::var("HOME") {
        let docs = std::path::PathBuf::from(&home).join("Documents");
        if docs.is_dir() { return Some(docs); }
        let h = std::path::PathBuf::from(home);
        if h.is_dir() { return Some(h); }
    }
    None
}
