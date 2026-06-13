//! Modal dialog — a centered panel over a dimmed backdrop that captures
//! input until the user picks a button (or presses Enter / Esc).
//!
//! Used by R2Gui for the quit confirmation ("Save workspace image?
//! [Yes / No / Cancel]") and the Settings panel (font resize). It is a
//! plain state object like [`crate::context_menu::ContextMenu`]: the host
//! owns it, feeds it the frame's events, acts on the returned action, and
//! paints it LAST so it floats above every window.
//!
//! The dialog does NOT close itself on a button press — the host decides
//! (Settings keeps it open while the user nudges the font; a quit prompt
//! closes on Cancel and exits on Yes/No). Call [`Dialog::close`] when done.

use crate::event::{InputEvent, KeyCode, MouseButton, MousePos};
use crate::grid::Rect;
use crate::render::{Frame, Renderer};
use crate::theme::{Color, Theme};

/// One button along the dialog's bottom row. `action` is an opaque
/// string the host matches on (e.g. "quit.yes").
#[derive(Debug, Clone)]
pub struct DialogButton {
    pub label: String,
    pub action: String,
}

impl DialogButton {
    pub fn new(label: impl Into<String>, action: impl Into<String>) -> Self {
        Self { label: label.into(), action: action.into() }
    }
}

/// A modal dialog. Configure `title` / `lines` / `buttons` (and
/// optionally `default_action` for Enter and `cancel_action` for Esc),
/// then `open()`.
pub struct Dialog {
    pub title: String,
    /// Message body, one entry per line.
    pub lines: Vec<String>,
    /// Bottom-row buttons, laid out right-aligned (last = rightmost).
    pub buttons: Vec<DialogButton>,
    /// Action fired when Enter is pressed. Empty = ignore Enter.
    pub default_action: String,
    /// Action fired when Esc is pressed. Empty = ignore Esc.
    pub cancel_action: String,
    open: bool,
    last_mouse: MousePos,
}

impl Default for Dialog {
    fn default() -> Self { Self::new() }
}

impl Dialog {
    pub fn new() -> Self {
        Self {
            title: String::new(),
            lines: Vec::new(),
            buttons: Vec::new(),
            default_action: String::new(),
            cancel_action: String::new(),
            open: false,
            last_mouse: MousePos { x: 0.0, y: 0.0 },
        }
    }

    pub fn is_open(&self) -> bool { self.open }
    pub fn open(&mut self) { self.open = true; }
    pub fn close(&mut self) { self.open = false; }

    /// Walk this frame's events. Returns `Some(action)` when a button is
    /// clicked, or when Enter/Esc map to a configured action. Does not
    /// close — the host calls [`close`] when appropriate.
    pub fn handle_events(&mut self,
                         events: &[InputEvent],
                         renderer: &mut Renderer,
                         theme: &Theme,
                         screen_w: f32, screen_h: f32) -> Option<String> {
        if !self.open { return None; }
        let panel = self.panel_rect(renderer, theme, screen_w, screen_h);
        let btns = self.button_rects(panel, renderer, theme);
        let mut fired: Option<String> = None;
        for ev in events {
            match *ev {
                InputEvent::MouseMoved(p) => { self.last_mouse = p; }
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    for (i, r) in btns.iter().enumerate() {
                        if point_in(*r, pos) {
                            fired = Some(self.buttons[i].action.clone());
                        }
                    }
                    // Clicks anywhere else are swallowed (modal) — no action.
                }
                InputEvent::Key { code: KeyCode::Enter, pressed: true, .. } => {
                    if !self.default_action.is_empty() {
                        fired = Some(self.default_action.clone());
                    }
                }
                InputEvent::Key { code: KeyCode::Escape, pressed: true, .. } => {
                    if !self.cancel_action.is_empty() {
                        fired = Some(self.cancel_action.clone());
                    }
                }
                _ => {}
            }
        }
        fired
    }

    // ── Layout ──────────────────────────────────────────────────────
    // panel_rect and button_rects MUST agree between handle_events and
    // paint, so both derive from these two helpers.

    fn panel_rect(&self, renderer: &mut Renderer, theme: &Theme,
                  screen_w: f32, screen_h: f32) -> Rect {
        let (cw, lh) = renderer.cell_metrics(theme.font_size);
        let pad = theme.px(16.0);

        // Widest of: title, each message line.
        let mut text_cols = self.title.chars().count();
        for l in &self.lines { text_cols = text_cols.max(l.chars().count()); }
        let text_w = text_cols as f32 * cw;

        // Button row width.
        let (bw_each, _bh, gap) = self.button_metrics(cw, lh, theme);
        let buttons_w = if self.buttons.is_empty() {
            0.0
        } else {
            bw_each.iter().sum::<f32>() + gap * (self.buttons.len() as f32 - 1.0)
        };

        let inner_w = text_w.max(buttons_w);
        let mut w = inner_w + pad * 2.0;
        w = w.max(theme.px(300.0)).min(screen_w * 0.85);

        let title_h = lh + theme.px(12.0);
        let (_b, bh, _g) = self.button_metrics(cw, lh, theme);
        let lines_h = self.lines.len() as f32 * lh;
        let h = title_h + pad + lines_h + pad + bh + pad;

        let x = ((screen_w - w) * 0.5).round();
        let y = ((screen_h - h) * 0.5).round();
        Rect { x, y, w, h }
    }

    /// (per-button widths, uniform height, inter-button gap).
    fn button_metrics(&self, cw: f32, lh: f32, theme: &Theme)
        -> (Vec<f32>, f32, f32) {
        let bh = lh + theme.px(12.0);
        let gap = theme.px(10.0);
        let widths = self.buttons.iter().map(|b| {
            (b.label.chars().count() as f32 * cw + theme.px(28.0))
                .max(theme.px(72.0))
        }).collect();
        (widths, bh, gap)
    }

    fn button_rects(&self, panel: Rect, renderer: &mut Renderer, theme: &Theme)
        -> Vec<Rect> {
        let (cw, lh) = renderer.cell_metrics(theme.font_size);
        let pad = theme.px(16.0);
        let (widths, bh, gap) = self.button_metrics(cw, lh, theme);
        let by = panel.y + panel.h - pad - bh;
        // Right-aligned: last button hugs the right edge.
        let total: f32 = widths.iter().sum::<f32>()
            + gap * (self.buttons.len().max(1) as f32 - 1.0);
        let mut x = panel.x + panel.w - pad - total;
        let mut out = Vec::with_capacity(self.buttons.len());
        for w in &widths {
            out.push(Rect { x, y: by, w: *w, h: bh });
            x += w + gap;
        }
        out
    }

    /// Paint the dimmed backdrop + centered panel. Call LAST in the
    /// frame so it floats above everything.
    pub fn paint(&self, frame: &mut Frame, renderer: &mut Renderer, theme: &Theme,
                 screen_w: f32, screen_h: f32) {
        if !self.open { return; }
        // Dim the whole screen so the modal reads as blocking.
        frame.paint_rect(0.0, 0.0, screen_w, screen_h, Color::rgba(0, 0, 0, 120));

        let panel = self.panel_rect(renderer, theme, screen_w, screen_h);
        let (_cw, lh) = renderer.cell_metrics(theme.font_size);
        let pad = theme.px(16.0);

        // Drop shadow.
        frame.paint_rect(panel.x + 3.0, panel.y + 3.0, panel.w, panel.h,
                         Color::rgba(0, 0, 0, 80));
        // Body.
        frame.paint_rect(panel.x, panel.y, panel.w, panel.h, theme.window_background);
        // 1-px border.
        let border = theme.border_active;
        frame.paint_rect(panel.x, panel.y,                  panel.w, 1.0, border);
        frame.paint_rect(panel.x, panel.y + panel.h - 1.0,  panel.w, 1.0, border);
        frame.paint_rect(panel.x, panel.y, 1.0, panel.h, border);
        frame.paint_rect(panel.x + panel.w - 1.0, panel.y, 1.0, panel.h, border);

        // Title strip.
        let title_h = lh + theme.px(12.0);
        frame.paint_rect(panel.x, panel.y, panel.w, title_h, theme.titlebar_active);
        frame.paint_rect(panel.x, panel.y + title_h - 1.0, panel.w, 1.0, border);
        if !self.title.is_empty() {
            let baseline = panel.y + title_h * 0.70;
            frame.paint_text(renderer, panel.x + pad, baseline,
                             &self.title, theme.font_size, theme.menu_text);
        }

        // Message lines.
        let mut y = panel.y + title_h + pad;
        for line in &self.lines {
            let baseline = y + lh * 0.78;
            frame.paint_text(renderer, panel.x + pad, baseline,
                             line, theme.font_size, theme.menu_text);
            y += lh;
        }

        // Buttons.
        let btns = self.button_rects(panel, renderer, theme);
        for (i, r) in btns.iter().enumerate() {
            let hovered = point_in(*r, self.last_mouse);
            // Default (Enter) button gets an accent fill; others a faint
            // raised fill that brightens on hover.
            let is_default = !self.default_action.is_empty()
                && self.buttons[i].action == self.default_action;
            let fill = if is_default {
                if hovered { Color::rgba(70, 130, 180, 235) } else { Color::rgba(70, 130, 180, 200) }
            } else if hovered {
                Color::rgba(70, 130, 180, 70)
            } else {
                Color::rgba(0, 0, 0, 18)
            };
            frame.paint_rect(r.x, r.y, r.w, r.h, fill);
            // Border.
            frame.paint_rect(r.x, r.y,             r.w, 1.0, border);
            frame.paint_rect(r.x, r.y + r.h - 1.0, r.w, 1.0, border);
            frame.paint_rect(r.x, r.y, 1.0, r.h, border);
            frame.paint_rect(r.x + r.w - 1.0, r.y, 1.0, r.h, border);
            // Label, horizontally centered.
            let (cw, _) = renderer.cell_metrics(theme.font_size);
            let label = &self.buttons[i].label;
            let tw = label.chars().count() as f32 * cw;
            let tx = r.x + (r.w - tw) * 0.5;
            let baseline = r.y + r.h * 0.70;
            let col = if is_default { Color::WHITE } else { theme.menu_text };
            frame.paint_text(renderer, tx, baseline, label, theme.font_size, col);
        }
    }
}

#[inline]
fn point_in(r: Rect, p: MousePos) -> bool {
    p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h
}
