//! CellGrid widget — the transcript renderer.
//!
//! The defining widget of R2-UI. A 2D grid of `Cell`s with O(1)
//! mouse-to-grid hit testing and range-based selection. See
//! `docs/R2_UI_FRAMEWORK.md` §3 for the philosophy.
//!
//! Implementation notes:
//! - Grid dimensions are DYNAMIC — derived from `rect ÷ cell_size`
//!   each frame. No stored `cols × rows` state.
//! - The data source is a `&ConsoleBuffer`; the widget is a view,
//!   never an owner. Multiple CellGrids can render the same buffer.

use crate::event::{Clipboard, InputEvent, KeyCode, MouseButton};
use crate::render::{Frame, Renderer};
use crate::theme::{Color, Theme};

/// Pixel-space axis-aligned rectangle. Top-left origin.
#[derive(Debug, Clone, Copy)]
pub struct Rect { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }

/// One screen cell: a single Unicode grapheme + its style.
#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub ch: char,
    pub fg: Color,
    pub bg: Option<Color>,
    pub bold: bool,
    pub italic: bool,
}

impl Cell {
    pub fn plain(ch: char, fg: Color) -> Self {
        Self { ch, fg, bg: None, bold: false, italic: false }
    }
}

/// (row, col) coordinate within the transcript. Row 0 = top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GridPos {
    pub row: usize,
    pub col: usize,
}

/// Inclusive range of grid cells. `start` and `end` are ordered such
/// that `start <= end` after canonicalization, regardless of which
/// end the user dragged from.
#[derive(Debug, Clone, Copy)]
pub struct Selection {
    pub start: GridPos,
    pub end:   GridPos,
}

impl Selection {
    /// Canonical (top-left → bottom-right) form.
    pub fn normalized(self) -> (GridPos, GridPos) {
        if self.start <= self.end { (self.start, self.end) }
        else                       { (self.end, self.start) }
    }
}

/// Per-frame builder that renders a CellGrid bound to a ConsoleBuffer.
/// The actual `show()` integration with winit + wgpu lives in the
/// `app` module; this struct is the user-facing handle.
pub struct CellGrid {
    /// Optional render-time selection. Restored from app state each
    /// frame; mutations go through the app's event loop.
    pub selection: Option<Selection>,
    /// Whether to wrap long lines at the right edge (true) or
    /// scroll horizontally (false). Default: true.
    pub wrap_long_lines: bool,
    /// Width in cells where the soft-wrap guide is drawn (0 = off).
    /// Conventional: 80.
    pub wrap_guide_at: usize,
}

impl CellGrid {
    pub fn new() -> Self {
        Self {
            selection: None,
            wrap_long_lines: true,
            wrap_guide_at: 80,
        }
    }
}

impl Default for CellGrid {
    fn default() -> Self { Self::new() }
}

/// Returned each frame from the CellGrid's `show()` call (defined in
/// the `app` module). Tells the host what just happened.
pub struct CellGridResponse {
    /// When the user pressed Ctrl+C / right-click → Copy this frame,
    /// the selected text is set here for the host to clipboard.
    pub selection_text: Option<String>,
    /// Whether the user dragged a fresh selection this frame.
    pub selection_changed: bool,
}

// ─── Selection math — pure functions, easy to test ───────────────

/// Walk the cells of a row range and stitch their `ch`'s into a
/// String, newline-separating rows. Pure function — testable
/// without any rendering.
pub fn selection_to_text(rows: &[Vec<Cell>], sel: Selection) -> String {
    let (a, b) = sel.normalized();
    let mut out = String::new();
    for r in a.row..=b.row {
        let row = match rows.get(r) { Some(r) => r, None => continue };
        let start_col = if r == a.row { a.col } else { 0 };
        let end_col   = if r == b.row { b.col } else { row.len().saturating_sub(1) };
        let end_col   = end_col.min(row.len().saturating_sub(1));
        if start_col <= end_col {
            for c in start_col..=end_col {
                if let Some(cell) = row.get(c) { out.push(cell.ch); }
            }
        }
        if r < b.row { out.push('\n'); }
    }
    out
}

/// Map a pixel coordinate inside `rect` to the cell `(row, col)` it
/// covers. Returns `None` if the point is outside the rect. Cells
/// are clamped to `row_count × col_count` so the result is always a
/// valid index into `rows`.
pub fn hit_test(
    rect: Rect,
    cell_w: f32, line_h: f32,
    mouse_x: f32, mouse_y: f32,
    row_count: usize, col_count: usize,
) -> Option<GridPos> {
    if mouse_x < rect.x || mouse_y < rect.y
        || mouse_x >= rect.x + rect.w || mouse_y >= rect.y + rect.h
    { return None; }
    if cell_w <= 0.0 || line_h <= 0.0 { return None; }
    let col = (((mouse_x - rect.x) / cell_w) as usize).min(col_count.saturating_sub(1));
    let row = (((mouse_y - rect.y) / line_h) as usize).min(row_count.saturating_sub(1));
    Some(GridPos { row, col })
}

/// How many rows of `rows` would have been scrolled off the top so
/// that the last row fits inside `rect`. Pure function — call this to
/// keep the cursor position in sync with the painter.
pub fn auto_scroll_offset(total_rows: usize, rect_h: f32, line_h: f32) -> usize {
    if line_h <= 0.0 { return 0; }
    let max_visible = (rect_h / line_h).floor() as usize;
    total_rows.saturating_sub(max_visible)
}

/// Paint a slice of cell-rows into `frame`, clipped to `rect`. The
/// renderer **auto-scrolls to the bottom**: when there are more rows
/// than fit, the OLDEST rows are scrolled off the top so the prompt
/// (the last row) always stays visible. The selection rectangle is
/// painted in the same scrolled coordinate space.
///
/// `size_pt` is the font size used to rasterize the glyphs; it should
/// match what was used to derive `cell_w` / `line_h` via
/// [`Renderer::cell_metrics`].
pub fn paint_cells(
    frame: &mut Frame,
    renderer: &mut Renderer,
    rows: &[Vec<Cell>],
    rect: Rect,
    cell_w: f32, line_h: f32,
    size_pt: f32,
    selection: Option<Selection>,
    theme: &Theme,
) {
    // Auto-scroll-to-bottom + no horizontal scroll. Same behavior as
    // before; this preserves the old call sites.
    paint_cells_scrolled(frame, renderer, rows, rect, cell_w, line_h,
                         size_pt, selection,
                         /* scroll_y_override */ None,
                         /* scroll_x          */ 0,
                         theme);
}

/// Core paint with explicit vertical / horizontal scroll offsets.
/// `scroll_y_override = None` keeps the original auto-scroll-to-
/// bottom behavior; `Some(n)` pins the viewport's top row to `n`.
/// `scroll_x` shifts the column-0 origin so the first visible
/// column is `scroll_x`. Both selection bands and glyphs use the
/// scrolled coordinate space.
pub fn paint_cells_scrolled(
    frame: &mut Frame,
    renderer: &mut Renderer,
    rows: &[Vec<Cell>],
    rect: Rect,
    cell_w: f32, line_h: f32,
    size_pt: f32,
    selection: Option<Selection>,
    scroll_y_override: Option<usize>,
    scroll_x: usize,
    theme: &Theme,
) {
    let scroll = match scroll_y_override {
        Some(s) => s.min(rows.len().saturating_sub(1)),
        None    => auto_scroll_offset(rows.len(), rect.h, line_h),
    };
    let max_visible = (rect.h / line_h).floor() as usize;
    let last_visible = (scroll + max_visible).min(rows.len());

    // Selection background — rows in the scrolled window only.
    if let Some(sel) = selection {
        let (a, b) = sel.normalized();
        let lo = a.row.max(scroll);
        let hi = b.row.min(last_visible.saturating_sub(1));
        if lo <= hi {
            for r in lo..=hi {
                let row = match rows.get(r) { Some(r) => r, None => continue };
                if row.is_empty() { continue; }
                let start_col = if r == a.row { a.col } else { 0 };
                let end_col   = if r == b.row { b.col } else { row.len().saturating_sub(1) };
                let end_col   = end_col.min(row.len().saturating_sub(1));
                if start_col > end_col { continue; }
                // Shift by horizontal scroll. Selection band clips
                // to the visible left edge so the highlight doesn't
                // bleed past the rect.
                let vis_start = start_col.saturating_sub(scroll_x);
                let vis_end   = end_col.saturating_sub(scroll_x);
                if start_col >= scroll_x || end_col >= scroll_x {
                    let sx = rect.x + vis_start as f32 * cell_w;
                    let sy = rect.y + (r - scroll) as f32 * line_h;
                    let sw = (vis_end - vis_start + 1) as f32 * cell_w;
                    frame.paint_rect(sx, sy, sw, line_h, theme.console_selection_bg);
                }
            }
        }
    }

    // Glyphs — only the rows in the scrolled window.
    let baseline_offset = line_h * 0.8;
    let max_x = rect.x + rect.w;
    for r in scroll..last_visible {
        let row = match rows.get(r) { Some(r) => r, None => continue };
        let y_baseline = rect.y + (r - scroll) as f32 * line_h + baseline_offset;
        for (c, cell) in row.iter().enumerate() {
            if c < scroll_x { continue; }   // skip columns scrolled off left
            let x = rect.x + (c - scroll_x) as f32 * cell_w;
            if x >= max_x { break; }        // hard clip on the right
            if cell.ch == ' ' { continue; }
            frame.paint_glyph(renderer, x, y_baseline, cell.ch, size_pt, cell.fg);
        }
    }
}

/// Convert a 0..1 scrollbar position to a row offset, given the
/// viewport's row capacity. Used by the host to translate a
/// `Scrollbar::position` into a `CellGridState::scroll_y_override`
/// value.
pub fn scroll_pos_to_row(position: f32, total_rows: usize, visible_rows: usize) -> usize {
    if total_rows <= visible_rows { return 0; }
    let max_off = (total_rows - visible_rows) as f32;
    (position.clamp(0.0, 1.0) * max_off).round() as usize
}

/// Convert a 0..1 scrollbar position to a column offset.
pub fn scroll_pos_to_col(position: f32, max_cols: usize, visible_cols: usize) -> usize {
    if max_cols <= visible_cols { return 0; }
    let max_off = (max_cols - visible_cols) as f32;
    (position.clamp(0.0, 1.0) * max_off).round() as usize
}

// ─── CellGridState — selection state machine ────────────────────────
//
// Owned by the host (typically inside a closure capture or app struct).
// Each frame the host calls `handle_events`, then `paint`. The
// selection rectangle is drawn for free as part of paint.

/// Stateful CellGrid widget. Tracks the current selection, the
/// in-progress mouse drag, and the user-driven scroll offsets. Lives
/// across frames.
#[derive(Debug, Default)]
pub struct CellGridState {
    pub selection: Option<Selection>,
    /// Drag anchor — set on MouseDown inside the grid, cleared on Up.
    dragging: Option<GridPos>,
    /// Horizontal column offset. `0` = leftmost column shown.
    /// Bumped by the horizontal scrollbar.
    pub scroll_x: usize,
    /// Vertical scroll override. `None` (default) means
    /// "auto-scroll to keep the bottom row visible" — the same
    /// behavior the console has had all along. `Some(n)` means the
    /// user grabbed the vertical scrollbar and pinned the top of
    /// the viewport at row `n`.
    pub scroll_y_override: Option<usize>,
}

impl CellGridState {
    pub fn new() -> Self { Self::default() }

    /// Walk this frame's events, updating selection state and copying
    /// to clipboard on Ctrl+C. Returns the text just copied to the
    /// clipboard (or `None`) so the host can show feedback.
    pub fn handle_events(
        &mut self,
        events: &[InputEvent],
        rows: &[Vec<Cell>],
        rect: Rect,
        cell_w: f32, line_h: f32,
        clipboard: &mut Clipboard,
    ) -> Option<String> {
        let row_count = rows.len();
        let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut copied: Option<String> = None;

        // Effective scroll offset, computed exactly as the painter does
        // (`paint_cells_scrolled`): an explicit override, else
        // auto-scroll-to-bottom. `hit_test` returns a VIEWPORT-relative
        // cell, so we add the scroll offsets to get the ABSOLUTE row/col
        // in `rows`. Without this, every selection in a scrolled console
        // (i.e. whenever there's more output than fits) lands on the
        // wrong rows.
        let scroll_y = self.scroll_y_override
            .unwrap_or_else(|| auto_scroll_offset(row_count, rect.h, line_h));
        let scroll_x = self.scroll_x;
        let to_abs = |p: GridPos| GridPos {
            row: (p.row + scroll_y).min(row_count.saturating_sub(1)),
            col: (p.col + scroll_x).min(col_count.saturating_sub(1)),
        };

        for ev in events {
            match *ev {
                InputEvent::MouseDown { button: MouseButton::Left, pos } => {
                    if let Some(p) = hit_test(rect, cell_w, line_h, pos.x, pos.y, row_count, col_count) {
                        let a = to_abs(p);
                        self.dragging = Some(a);
                        self.selection = Some(Selection { start: a, end: a });
                    } else {
                        // click outside → clear
                        self.dragging  = None;
                        self.selection = None;
                    }
                }
                InputEvent::MouseMoved(pos) => {
                    if let Some(anchor) = self.dragging {
                        // Auto-scroll when the drag passes the top/bottom
                        // edge, so a selection can extend beyond the visible
                        // rows (one row per move event). Recompute the
                        // effective scroll afterwards for the hit-test.
                        let max_visible = if line_h > 0.0 { (rect.h / line_h).floor() as usize } else { 0 };
                        let max_scroll = row_count.saturating_sub(max_visible);
                        let mut cur_scroll = self.scroll_y_override.unwrap_or(scroll_y);
                        if pos.y >= rect.y + rect.h && cur_scroll < max_scroll {
                            cur_scroll = (cur_scroll + 1).min(max_scroll);
                            self.scroll_y_override = Some(cur_scroll);
                        } else if pos.y <= rect.y && cur_scroll > 0 {
                            cur_scroll -= 1;
                            self.scroll_y_override = Some(cur_scroll);
                        }
                        // Clamp to rect even when cursor wanders out so
                        // the selection keeps tracking visually.
                        let mx = pos.x.clamp(rect.x, rect.x + rect.w - 1.0);
                        let my = pos.y.clamp(rect.y, rect.y + rect.h - 1.0);
                        if let Some(p) = hit_test(rect, cell_w, line_h, mx, my, row_count, col_count) {
                            let end = GridPos {
                                row: (p.row + cur_scroll).min(row_count.saturating_sub(1)),
                                col: (p.col + scroll_x).min(col_count.saturating_sub(1)),
                            };
                            self.selection = Some(Selection { start: anchor, end });
                        }
                    }
                }
                InputEvent::MouseUp { button: MouseButton::Left, .. } => {
                    self.dragging = None;
                    // Collapse zero-width selection (just a click) to None
                    // so a single click doesn't leave a stale highlight.
                    if let Some(sel) = self.selection {
                        let (a, b) = sel.normalized();
                        if a == b { self.selection = None; }
                    }
                }
                InputEvent::Key { code: KeyCode::KeyA, mods, pressed: true } if mods.ctrl => {
                    if row_count > 0 && col_count > 0 {
                        let last_row = row_count - 1;
                        let last_col = rows[last_row].len().saturating_sub(1);
                        self.selection = Some(Selection {
                            start: GridPos { row: 0, col: 0 },
                            end:   GridPos { row: last_row, col: last_col },
                        });
                    }
                }
                InputEvent::Key { code: KeyCode::KeyC, mods, pressed: true } if mods.ctrl => {
                    if let Some(sel) = self.selection {
                        let text = selection_to_text(rows, sel);
                        if !text.is_empty() {
                            clipboard.set_text(&text);
                            copied = Some(text);
                        }
                    }
                }
                InputEvent::Key { code: KeyCode::Escape, pressed: true, .. } => {
                    self.selection = None;
                    self.dragging  = None;
                }
                _ => {}
            }
        }
        copied
    }

    /// Paint the rows + selection band. Convenience wrapper around
    /// the free `paint_cells` function.
    pub fn paint(
        &self,
        frame: &mut Frame,
        renderer: &mut Renderer,
        rows: &[Vec<Cell>],
        rect: Rect,
        cell_w: f32, line_h: f32,
        size_pt: f32,
        theme: &Theme,
    ) {
        paint_cells_scrolled(frame, renderer, rows, rect, cell_w, line_h,
                             size_pt, self.selection,
                             self.scroll_y_override, self.scroll_x, theme);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Color;

    fn line(s: &str) -> Vec<Cell> {
        s.chars().map(|c| Cell::plain(c, Color::BLACK)).collect()
    }

    #[test]
    fn selection_normalizes_regardless_of_drag_direction() {
        let s1 = Selection { start: GridPos { row: 0, col: 5 }, end: GridPos { row: 2, col: 1 } };
        let s2 = Selection { start: GridPos { row: 2, col: 1 }, end: GridPos { row: 0, col: 5 } };
        assert_eq!(s1.normalized(), s2.normalized());
    }

    #[test]
    fn selection_within_one_row() {
        let rows = vec![line("hello world")];
        let sel = Selection {
            start: GridPos { row: 0, col: 6 },
            end:   GridPos { row: 0, col: 10 },
        };
        assert_eq!(selection_to_text(&rows, sel), "world");
    }

    #[test]
    fn selection_spans_multiple_rows() {
        let rows = vec![line("abc"), line("def"), line("ghi")];
        let sel = Selection {
            start: GridPos { row: 0, col: 1 },
            end:   GridPos { row: 2, col: 1 },
        };
        assert_eq!(selection_to_text(&rows, sel), "bc\ndef\ngh");
    }

    #[test]
    fn hit_test_outside_rect_returns_none() {
        let rect = Rect { x: 10.0, y: 10.0, w: 80.0, h: 60.0 };
        assert!(hit_test(rect, 8.0, 12.0, 5.0,  20.0, 10, 10).is_none());
        assert!(hit_test(rect, 8.0, 12.0, 95.0, 20.0, 10, 10).is_none());
        assert!(hit_test(rect, 8.0, 12.0, 20.0, 75.0, 10, 10).is_none());
    }

    #[test]
    fn hit_test_inside_quantizes_to_cell() {
        let rect = Rect { x: 0.0, y: 0.0, w: 80.0, h: 60.0 };
        // cell 8x12, click at (17, 25): col=2, row=2
        let p = hit_test(rect, 8.0, 12.0, 17.0, 25.0, 100, 100).unwrap();
        assert_eq!(p, GridPos { row: 2, col: 2 });
    }

    fn ev_down(x: f32, y: f32) -> InputEvent {
        InputEvent::MouseDown { button: MouseButton::Left, pos: crate::event::MousePos { x, y } }
    }
    fn ev_move(x: f32, y: f32) -> InputEvent {
        InputEvent::MouseMoved(crate::event::MousePos { x, y })
    }
    fn ev_up(x: f32, y: f32) -> InputEvent {
        InputEvent::MouseUp { button: MouseButton::Left, pos: crate::event::MousePos { x, y } }
    }
    fn ev_ctrl_a() -> InputEvent {
        InputEvent::Key { code: KeyCode::KeyA, mods: crate::event::Mods { ctrl: true, shift: false, alt: false }, pressed: true }
    }
    fn ev_ctrl_c() -> InputEvent {
        InputEvent::Key { code: KeyCode::KeyC, mods: crate::event::Mods { ctrl: true, shift: false, alt: false }, pressed: true }
    }

    #[test]
    fn drag_creates_selection() {
        let rows = vec![line("hello world"), line("second row "), line("third row  ")];
        let rect = Rect { x: 0.0, y: 0.0, w: 200.0, h: 60.0 };
        let mut state = CellGridState::new();
        let mut clip = Clipboard::new();
        state.handle_events(&[ev_down(0.0, 0.0), ev_move(40.0, 25.0), ev_up(40.0, 25.0)],
                            &rows, rect, 10.0, 20.0, &mut clip);
        let sel = state.selection.expect("selection should exist after drag");
        let (a, b) = sel.normalized();
        assert_eq!(a, GridPos { row: 0, col: 0 });
        assert_eq!(b, GridPos { row: 1, col: 4 });
    }

    #[test]
    fn single_click_clears_selection() {
        let rows = vec![line("hello")];
        let rect = Rect { x: 0.0, y: 0.0, w: 200.0, h: 60.0 };
        let mut state = CellGridState::new();
        let mut clip = Clipboard::new();
        state.handle_events(&[ev_down(15.0, 5.0), ev_up(15.0, 5.0)],
                            &rows, rect, 10.0, 20.0, &mut clip);
        assert!(state.selection.is_none(), "zero-width selection should collapse");
    }

    #[test]
    fn ctrl_a_selects_everything() {
        let rows = vec![line("abc"), line("defgh")];
        let rect = Rect { x: 0.0, y: 0.0, w: 200.0, h: 60.0 };
        let mut state = CellGridState::new();
        let mut clip = Clipboard::new();
        state.handle_events(&[ev_ctrl_a()], &rows, rect, 10.0, 20.0, &mut clip);
        let sel = state.selection.expect("Ctrl+A should select all");
        assert_eq!(sel.start, GridPos { row: 0, col: 0 });
        assert_eq!(sel.end,   GridPos { row: 1, col: 4 });
    }

    #[test]
    fn ctrl_c_copies_selection_text() {
        let rows = vec![line("hello world")];
        let rect = Rect { x: 0.0, y: 0.0, w: 200.0, h: 60.0 };
        let mut state = CellGridState::new();
        let mut clip = Clipboard::new();
        // Select first 5 chars then Ctrl+C.
        state.handle_events(&[ev_down(0.0, 0.0), ev_move(45.0, 5.0), ev_up(45.0, 5.0)],
                            &rows, rect, 10.0, 20.0, &mut clip);
        let copied = state.handle_events(&[ev_ctrl_c()], &rows, rect, 10.0, 20.0, &mut clip);
        assert_eq!(copied.as_deref(), Some("hello"));
    }

    #[test]
    fn hit_test_clamps_to_grid_bounds() {
        let rect = Rect { x: 0.0, y: 0.0, w: 800.0, h: 600.0 };
        // grid is 5 cols × 3 rows; click at (790, 590) clamps to (2, 4)
        let p = hit_test(rect, 8.0, 12.0, 790.0, 590.0, 3, 5).unwrap();
        assert_eq!(p, GridPos { row: 2, col: 4 });
    }

    #[test]
    fn selection_accounts_for_auto_scroll() {
        // 10 rows; viewport fits 3 (line_h=20, rect.h=60) → auto-scroll
        // offset 7. A drag from the TOP visible row must select from
        // ABSOLUTE row 7 (not row 0). The pre-fix hit_test ignored the
        // scroll offset, so selections in a scrolled console (i.e. once
        // there's more output than fits) landed on the wrong rows.
        let rows: Vec<Vec<Cell>> = (0..10).map(|i| line(&format!("row{}", i))).collect();
        let rect = Rect { x: 0.0, y: 0.0, w: 200.0, h: 60.0 };
        let mut state = CellGridState::new();
        let mut clip = Clipboard::new();
        state.handle_events(
            &[ev_down(5.0, 5.0), ev_move(15.0, 45.0), ev_up(15.0, 45.0)],
            &rows, rect, 10.0, 20.0, &mut clip);
        let sel = state.selection.expect("drag selection should exist");
        let (a, b) = sel.normalized();
        assert_eq!(a.row, 7, "top visible row maps to absolute 7 when scrolled");
        assert_eq!(b.row, 9, "bottom visible row maps to absolute 9");
    }

    #[test]
    fn selection_reverse_drag_gives_same_text() {
        let rows = vec![line("abc"), line("def"), line("ghi")];
        let s1 = Selection { start: GridPos { row: 0, col: 1 }, end: GridPos { row: 2, col: 1 } };
        let s2 = Selection { start: GridPos { row: 2, col: 1 }, end: GridPos { row: 0, col: 1 } };
        assert_eq!(selection_to_text(&rows, s1), selection_to_text(&rows, s2));
    }
}
