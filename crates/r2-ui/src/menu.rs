//! MenuBar — top-of-window menu with click-to-open drop-downs.
//!
//! Each top-level entry shows a label in a horizontal strip at the top
//! of the MDI workspace. Clicking opens a drop-down with its items;
//! clicking an item fires the item's `action` string (the host wires
//! that to a real handler). Clicking outside any menu closes it.

use crate::event::{InputEvent, MouseButton, MousePos};
use crate::grid::Rect;
use crate::render::{Frame, Renderer};
use crate::theme::{Color, Theme};

#[derive(Debug, Clone)]
pub struct MenuBar {
    pub items: Vec<MenuTopLevel>,
}

#[derive(Debug, Clone)]
pub struct MenuTopLevel {
    pub label: String,
    pub items: Vec<MenuItem>,
}

#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: String,
    pub shortcut: String,
    /// Action label — host wires this to a real callback. Using a
    /// label rather than a Boxed closure keeps the menu data
    /// `Send + Sync` and serializable for theme/keymap import-export.
    pub action: String,
}

impl MenuBar {
    pub fn new() -> Self { Self { items: Vec::new() } }
    pub fn add_top(&mut self, label: impl Into<String>) -> &mut MenuTopLevel {
        self.items.push(MenuTopLevel { label: label.into(), items: Vec::new() });
        self.items.last_mut().unwrap()
    }
}

impl MenuTopLevel {
    pub fn item(&mut self, label: impl Into<String>, shortcut: impl Into<String>, action: impl Into<String>) -> &mut Self {
        self.items.push(MenuItem {
            label: label.into(),
            shortcut: shortcut.into(),
            action: action.into(),
        });
        self
    }
}

impl Default for MenuBar { fn default() -> Self { Self::new() } }

/// Builder used inside `R2Ui::app(...).menu(|m| { ... })`.
pub struct MenuBuilder {
    pub bar: MenuBar,
}

impl MenuBuilder {
    pub fn new() -> Self { Self { bar: MenuBar::new() } }
    pub fn top(&mut self, label: impl Into<String>) -> &mut MenuTopLevel {
        self.bar.add_top(label)
    }
}

impl Default for MenuBuilder { fn default() -> Self { Self::new() } }

// ─── MenuBarState — interaction + painting ──────────────────────────

pub const MENU_BAR_HEIGHT: f32 = 22.0;
const ITEM_HEIGHT: f32 = 20.0;
const POPUP_MIN_W: f32 = 160.0;

/// Stateful menu bar — owns the open-popup state and dispatches
/// click → action string. Lives across frames.
pub struct MenuBarState {
    pub bar: MenuBar,
    pub open: Option<usize>,
    last_mouse: MousePos,
}

impl MenuBarState {
    pub fn new(bar: MenuBar) -> Self {
        Self { bar, open: None, last_mouse: MousePos { x: 0.0, y: 0.0 } }
    }

    /// Walk events; returns Some(action) when an item was clicked.
    pub fn handle_events(&mut self,
                         events: &[InputEvent],
                         workspace: Rect,
                         renderer: &mut Renderer,
                         theme: &Theme) -> Option<String> {
        let (cell_w, _) = renderer.cell_metrics(theme.font_size);
        let mut fired: Option<String> = None;
        for ev in events {
            match *ev {
                InputEvent::MouseMoved(p) => { self.last_mouse = p; }
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    // Hit-test top-level labels.
                    let mut pen_x = workspace.x + 8.0;
                    let mut clicked_top: Option<usize> = None;
                    for (i, top) in self.bar.items.iter().enumerate() {
                        let w = top.label.chars().count() as f32 * cell_w + 16.0;
                        let r = Rect { x: pen_x, y: workspace.y, w, h: MENU_BAR_HEIGHT };
                        if point_in(r, pos) { clicked_top = Some(i); break; }
                        pen_x += w;
                    }
                    if let Some(i) = clicked_top {
                        self.open = if self.open == Some(i) { None } else { Some(i) };
                        continue;
                    }
                    // If a popup is open and the click is inside one of
                    // its items, fire.
                    if let Some(oi) = self.open {
                        if let Some(action) = self.item_at(oi, pos, renderer, theme, workspace) {
                            fired = Some(action);
                            self.open = None;
                        } else {
                            // Click outside the popup → close.
                            self.open = None;
                        }
                    }
                }
                _ => {}
            }
        }
        fired
    }

    fn item_at(&self, top_idx: usize, pos: MousePos,
               renderer: &mut Renderer, theme: &Theme,
               workspace: Rect) -> Option<String> {
        let top = self.bar.items.get(top_idx)?;
        let (cell_w, _) = renderer.cell_metrics(theme.font_size);
        // Find this top-level's x position (sum widths of preceding entries).
        let mut popup_x = workspace.x + 8.0;
        for (i, t) in self.bar.items.iter().enumerate() {
            if i == top_idx { break; }
            popup_x += t.label.chars().count() as f32 * cell_w + 16.0;
        }
        let popup_y = workspace.y + MENU_BAR_HEIGHT;
        let popup_w = popup_width(top, cell_w);
        for (j, item) in top.items.iter().enumerate() {
            let r = Rect { x: popup_x, y: popup_y + j as f32 * ITEM_HEIGHT, w: popup_w, h: ITEM_HEIGHT };
            if point_in(r, pos) { return Some(item.action.clone()); }
        }
        None
    }

    /// Paint the menu STRIP only (no popups). Call early in the
    /// frame; sub-windows below will not be obstructed by it.
    pub fn paint_strip(&self, frame: &mut Frame, renderer: &mut Renderer,
                       workspace: Rect, theme: &Theme) {
        // Strip background.
        frame.paint_rect(workspace.x, workspace.y, workspace.w, MENU_BAR_HEIGHT, theme.menu_background);
        // Bottom 1px shadow.
        frame.paint_rect(workspace.x, workspace.y + MENU_BAR_HEIGHT - 1.0,
                         workspace.w, 1.0, Color::rgba(0, 0, 0, 40));

        let (cell_w, _) = renderer.cell_metrics(theme.font_size);
        let baseline = workspace.y + MENU_BAR_HEIGHT * 0.74;

        let mut pen_x = workspace.x + 8.0;
        for (i, top) in self.bar.items.iter().enumerate() {
            let w = top.label.chars().count() as f32 * cell_w + 16.0;
            if self.open == Some(i) {
                frame.paint_rect(pen_x, workspace.y, w, MENU_BAR_HEIGHT,
                                 Color::rgba(70, 130, 180, 70));
            }
            frame.paint_text(renderer, pen_x + 8.0, baseline,
                             &top.label, theme.font_size, theme.menu_text);
            pen_x += w;
        }
    }

    /// Paint the OPEN POPUP only (nothing if no popup is open). Call
    /// LAST in the frame — after every sub-window and its content —
    /// so the popup floats on top of whichever windows it overlaps.
    pub fn paint_popup(&self, frame: &mut Frame, renderer: &mut Renderer,
                       workspace: Rect, theme: &Theme) {
        let oi = match self.open { Some(i) => i, None => return };
        let top = match self.bar.items.get(oi) { Some(t) => t, None => return };
        let (cell_w, _) = renderer.cell_metrics(theme.font_size);

        let mut popup_x = workspace.x + 8.0;
        for (i, t) in self.bar.items.iter().enumerate() {
            if i == oi { break; }
            popup_x += t.label.chars().count() as f32 * cell_w + 16.0;
        }
        let popup_y = workspace.y + MENU_BAR_HEIGHT;
        let popup_w = popup_width(top, cell_w);
        let popup_h = top.items.len() as f32 * ITEM_HEIGHT + 4.0;

        // Background + 1-px border, drop a subtle shadow rectangle so
        // the popup reads as a separate plane above whichever window
        // it overlaps.
        frame.paint_rect(popup_x + 2.0, popup_y + 2.0, popup_w, popup_h, Color::rgba(0, 0, 0, 50));
        frame.paint_rect(popup_x, popup_y, popup_w, popup_h, theme.window_background);
        frame.paint_rect(popup_x, popup_y, popup_w, 1.0, Color::rgba(0,0,0,80));
        frame.paint_rect(popup_x, popup_y + popup_h - 1.0, popup_w, 1.0, Color::rgba(0,0,0,80));
        frame.paint_rect(popup_x, popup_y, 1.0, popup_h, Color::rgba(0,0,0,80));
        frame.paint_rect(popup_x + popup_w - 1.0, popup_y, 1.0, popup_h, Color::rgba(0,0,0,80));

        for (j, item) in top.items.iter().enumerate() {
            let row_y = popup_y + 2.0 + j as f32 * ITEM_HEIGHT;
            let row_r = Rect { x: popup_x, y: row_y, w: popup_w, h: ITEM_HEIGHT };
            if point_in(row_r, self.last_mouse) {
                frame.paint_rect(popup_x + 1.0, row_y, popup_w - 2.0, ITEM_HEIGHT,
                                 Color::rgba(70, 130, 180, 70));
            }
            let baseline = row_y + ITEM_HEIGHT * 0.72;
            frame.paint_text(renderer, popup_x + 12.0, baseline,
                             &item.label, theme.font_size, theme.menu_text);
            if !item.shortcut.is_empty() {
                let sx = popup_x + popup_w - 12.0 - item.shortcut.chars().count() as f32 * cell_w;
                frame.paint_text(renderer, sx, baseline,
                                 &item.shortcut, theme.font_size,
                                 Color::rgba(80, 80, 90, 220));
            }
        }
    }

    /// Backward-compatible: paint strip + popup together. New callers
    /// prefer the split (`paint_strip` early, `paint_popup` last) so
    /// popups float above sub-windows.
    pub fn paint(&self, frame: &mut Frame, renderer: &mut Renderer,
                 workspace: Rect, theme: &Theme) {
        self.paint_strip(frame, renderer, workspace, theme);
        self.paint_popup(frame, renderer, workspace, theme);
    }
}

fn popup_width(top: &MenuTopLevel, cell_w: f32) -> f32 {
    let max_label = top.items.iter().map(|i| i.label.chars().count()).max().unwrap_or(0);
    let max_sc    = top.items.iter().map(|i| i.shortcut.chars().count()).max().unwrap_or(0);
    let needed = (max_label + max_sc) as f32 * cell_w + 36.0;
    needed.max(POPUP_MIN_W)
}

#[inline]
fn point_in(r: Rect, p: MousePos) -> bool {
    p.x >= r.x && p.x < r.x + r.w && p.y >= r.y && p.y < r.y + r.h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_collects_items() {
        let mut mb = MenuBuilder::new();
        mb.top("File")
            .item("New",  "Ctrl+N", "file.new")
            .item("Open", "Ctrl+O", "file.open")
            .item("Quit", "Ctrl+Q", "file.quit");
        mb.top("Edit")
            .item("Copy", "Ctrl+C", "edit.copy");
        assert_eq!(mb.bar.items.len(), 2);
        assert_eq!(mb.bar.items[0].items.len(), 3);
        assert_eq!(mb.bar.items[1].items[0].action, "edit.copy");
    }
}
