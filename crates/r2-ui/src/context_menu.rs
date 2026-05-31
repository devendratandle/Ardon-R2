//! Right-click context menu — small popup that appears at the mouse
//! position when the user right-clicks a registered region.
//!
//! Behavioural cloning note: matches the same per-widget right-click
//! menus that desktop R Console exposes (Copy / Paste / Select all
//! / Clear for the console; Save / Copy for graphics). The
//! implementation is entirely fresh Rust — same idea, no shared code.
//!
//! ## Lifecycle
//!
//! 1. Host owns one `ContextMenu` per region that should have a
//!    right-click menu (e.g. one for the Console body, one for each
//!    Graphics body).
//! 2. Every frame the host calls
//!    `cm.handle_events(events, region_rect)` — that watches for a
//!    right-click inside the rect to open, and for outside-clicks /
//!    item-clicks / Escape to close. Returns `Some(action)` on the
//!    frame an item is clicked.
//! 3. Host calls `cm.paint(...)` LAST in the frame so the popup
//!    floats above any window underneath.

use crate::event::{InputEvent, KeyCode, MouseButton, MousePos};
use crate::grid::Rect;
use crate::render::{Frame, Renderer};
use crate::theme::{Color, Theme};

/// One entry in a context menu. `action` is an opaque string the
/// host matches on; the widget never interprets it. An entry with
/// empty label paints as a thin separator.
#[derive(Debug, Clone)]
pub struct ContextItem {
    pub label: String,
    pub action: String,
}

impl ContextItem {
    pub fn new(label: impl Into<String>, action: impl Into<String>) -> Self {
        Self { label: label.into(), action: action.into() }
    }
    pub fn separator() -> Self {
        Self { label: String::new(), action: String::new() }
    }
}

const ITEM_HEIGHT: f32 = 22.0;
const SEP_HEIGHT:  f32 = 7.0;
const PADDING:     f32 = 4.0;
const MIN_WIDTH:   f32 = 160.0;

pub struct ContextMenu {
    pub items: Vec<ContextItem>,
    /// Origin (top-left) of the open popup, or None if closed.
    open_at: Option<MousePos>,
    last_mouse: MousePos,
}

impl ContextMenu {
    pub fn new(items: Vec<ContextItem>) -> Self {
        Self { items, open_at: None, last_mouse: MousePos { x: 0.0, y: 0.0 } }
    }

    pub fn is_open(&self) -> bool { self.open_at.is_some() }
    pub fn close(&mut self) { self.open_at = None; }

    /// Walk this frame's events. Opens on right-click inside `region`,
    /// closes on outside-click / Escape. Returns Some(action) on the
    /// frame the user clicks an item.
    pub fn handle_events(&mut self,
                         events: &[InputEvent],
                         region: Rect,
                         renderer: &mut Renderer,
                         theme: &Theme) -> Option<String> {
        let mut fired: Option<String> = None;
        for ev in events {
            match *ev {
                InputEvent::MouseMoved(p) => { self.last_mouse = p; }
                InputEvent::MouseDown { button: MouseButton::Right, pos } => {
                    if point_in(region, pos) {
                        self.open_at = Some(pos);
                    } else {
                        self.open_at = None;
                    }
                }
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    if let Some(action) = self.item_at(pos, renderer, theme) {
                        fired = Some(action);
                        self.open_at = None;
                    } else if self.open_at.is_some() {
                        // Left-click outside the popup → close.
                        self.open_at = None;
                    }
                }
                InputEvent::Key { code: KeyCode::Escape, pressed: true, .. } => {
                    self.open_at = None;
                }
                _ => {}
            }
        }
        fired
    }

    fn popup_rect(&self, renderer: &mut Renderer, theme: &Theme) -> Option<Rect> {
        let origin = self.open_at?;
        let (cell_w, _) = renderer.cell_metrics(theme.font_size);
        let label_w = self.items.iter()
            .map(|i| i.label.chars().count())
            .max().unwrap_or(0) as f32 * cell_w + 24.0;
        let w = label_w.max(MIN_WIDTH);
        let h = self.items.iter()
            .map(|i| if i.label.is_empty() { SEP_HEIGHT } else { ITEM_HEIGHT })
            .sum::<f32>() + PADDING * 2.0;
        Some(Rect { x: origin.x, y: origin.y, w, h })
    }

    fn item_at(&self, pos: MousePos, renderer: &mut Renderer, theme: &Theme)
        -> Option<String>
    {
        let r = self.popup_rect(renderer, theme)?;
        if !point_in(r, pos) { return None; }
        let mut y = r.y + PADDING;
        for item in &self.items {
            let ih = if item.label.is_empty() { SEP_HEIGHT } else { ITEM_HEIGHT };
            if pos.y >= y && pos.y < y + ih {
                if item.label.is_empty() || item.action.is_empty() {
                    return None;            // clicked a separator — no-op
                }
                return Some(item.action.clone());
            }
            y += ih;
        }
        None
    }

    /// Paint the popup if open. Call LAST in the frame so it floats
    /// above any sub-window underneath.
    pub fn paint(&self, frame: &mut Frame, renderer: &mut Renderer, theme: &Theme) {
        let rect = match self.popup_rect(renderer, theme) {
            Some(r) => r, None => return,
        };
        // Drop shadow.
        frame.paint_rect(rect.x + 2.0, rect.y + 2.0, rect.w, rect.h,
                         Color::rgba(0, 0, 0, 50));
        // Body + 1-px border.
        frame.paint_rect(rect.x, rect.y, rect.w, rect.h, theme.window_background);
        let border = Color::rgba(0, 0, 0, 90);
        frame.paint_rect(rect.x, rect.y,                rect.w, 1.0, border);
        frame.paint_rect(rect.x, rect.y + rect.h - 1.0, rect.w, 1.0, border);
        frame.paint_rect(rect.x, rect.y, 1.0, rect.h, border);
        frame.paint_rect(rect.x + rect.w - 1.0, rect.y, 1.0, rect.h, border);

        let mut y = rect.y + PADDING;
        for item in &self.items {
            if item.label.is_empty() {
                // Separator — thin horizontal line, indented from edges.
                let line_y = y + SEP_HEIGHT * 0.5;
                frame.paint_rect(rect.x + 8.0, line_y, rect.w - 16.0, 1.0,
                                 Color::rgba(0, 0, 0, 50));
                y += SEP_HEIGHT;
                continue;
            }
            let row_r = Rect { x: rect.x + 1.0, y, w: rect.w - 2.0, h: ITEM_HEIGHT };
            if point_in(row_r, self.last_mouse) {
                frame.paint_rect(row_r.x, row_r.y, row_r.w, row_r.h,
                                 Color::rgba(70, 130, 180, 70));
            }
            let baseline = y + ITEM_HEIGHT * 0.72;
            frame.paint_text(renderer, rect.x + 12.0, baseline,
                             &item.label, theme.font_size, theme.menu_text);
            y += ITEM_HEIGHT;
        }
    }
}

#[inline]
fn point_in(r: Rect, p: MousePos) -> bool {
    p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h
}
