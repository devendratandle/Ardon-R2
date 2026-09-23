//! The console window: keyboard input at the prompt, drag-selection, and
//! painting the transcript with its scrollbars and cursor.
//!
//! Selection, copy and paint must agree on ONE row structure — the
//! transcript plus the live prompt row, wrapped to the console width — or
//! a (row, col) picked by the mouse copies the wrong text. The helpers
//! here (`rows_with_prompt`, `transcript_rect`, `console_wrap`) are that
//! agreement.

use r2_console::SubmitAction;
use r2_ui::{auto_scroll_offset, Cell, Frame, FrameCtx, InputEvent, InputField,
            Rect, Renderer, Theme, SCROLLBAR_THICKNESS};

use crate::support::{console_wrap_width, rows_from_buffer, run_source, wrap_rows};
use crate::Gui;

/// "<prompt> <typed text>" in the console-input colour.
fn prompt_row(input: &InputField, theme: &Theme) -> Vec<Cell> {
    let full = format!("{} {}", input.prompt, input.current);
    full.chars().map(|c| Cell::plain(c, theme.console_input)).collect()
}

/// Where the transcript grid sits in the console's content rect: an 8 px
/// margin, minus the scrollbar strips on the right and bottom. Hit-testing
/// uses this too, so dragging a scrollbar never starts a text selection.
fn transcript_rect(content: Rect) -> Rect {
    Rect {
        x: content.x + 8.0,
        y: content.y + 8.0,
        w: content.w - 16.0 - SCROLLBAR_THICKNESS,
        h: content.h - 16.0 - SCROLLBAR_THICKNESS,
    }
}

impl Gui {
    /// The transcript plus the live prompt row, unwrapped.
    pub(crate) fn rows_with_prompt(&self, theme: &Theme) -> Vec<Vec<Cell>> {
        let mut rows = rows_from_buffer(&self.buffer.lock().unwrap(), theme);
        rows.push(prompt_row(&self.input, theme));
        rows
    }

    /// The console's wrap width in cells (0 = don't wrap).
    pub(crate) fn console_wrap(&self, renderer: &mut Renderer, theme: &Theme) -> usize {
        let (cell_w, _) = renderer.cell_metrics(theme.fs());
        self.mdi.window(self.console_id)
            .map(|w| console_wrap_width(w.content_rect(theme).w, cell_w))
            .unwrap_or(0)
    }

    /// Hand one line to the console buffer as if typed and Enter-pressed;
    /// a complete statement (the buffer handles `+` continuations) runs.
    /// DeviceEvent::Plotted, drained at the top of the next frame,
    /// refreshes any graphics window it drew into.
    pub(crate) fn submit(&mut self, line: String) {
        let action = self.buffer.lock().unwrap().submit_line(line);
        if let SubmitAction::Submit(src) = action {
            run_source(&src, &mut self.engine, &self.buffer, &mut self.quit_requested);
        }
    }

    /// Console keyboard input — ALWAYS active so the console stays
    /// typeable regardless of which MDI window is topmost (RGui keeps the
    /// console interactive; a plot no longer "steals" the keyboard).
    /// Clicking a window still raises it via the MDI handler.
    pub(crate) fn console_input(&mut self, ui_events: &[InputEvent], ctx: &mut FrameCtx,
                                renderer: &mut Renderer, theme: &Theme, ctx_was_open: bool) {
        let topmost = self.mdi.z_order().last();
        let resp = self.input.handle_events(ui_events, ctx.clipboard);

        // Multi-line paste: each pasted line goes through submit_line
        // exactly as if typed, before the line Enter submitted.
        for line in resp.auto_submit_lines {
            self.submit(line);
        }
        if let Some(line) = resp.submitted {
            self.submit(line);
        }
        if resp.escaped {
            // Esc at the prompt: the field already cleared its line; a
            // pending `+` continuation goes with it.
            self.buffer.lock().unwrap().cancel_continuation();
        }
        if resp.history_up {
            if let Some(s) = self.buffer.lock().unwrap().history_up()   { self.input.set_line(s); }
        }
        if resp.history_down {
            if let Some(s) = self.buffer.lock().unwrap().history_down() { self.input.set_line(s); }
        }
        self.input.set_prompt(self.buffer.lock().unwrap().current_prompt());

        // Drag-select / Ctrl+A / Ctrl+C — only when the console is the
        // focused (topmost) window, so mouse selection targets the window
        // the user is actually working in.
        if topmost != Some(self.console_id) { return; }
        let rows = self.rows_with_prompt(theme);
        let (cell_w, line_h) = renderer.cell_metrics(theme.fs());
        let Some(content) = self.mdi.window(self.console_id).map(|w| w.content_rect(theme))
            else { return };
        // Wrap to the console width so hit-testing indexes the same
        // physical rows paint draws (otherwise a click on a wrapped line
        // would select the wrong text).
        let rows = wrap_rows(rows, console_wrap_width(content.w, cell_w));
        // Skip selection events on the frame a context menu was open /
        // fired — the click that picked a menu item would otherwise also
        // reach the grid and collapse the selection.
        if !ctx_was_open {
            let _copied = self.grid.handle_events(
                ui_events, &rows, transcript_rect(content), cell_w, line_h, ctx.clipboard);
        }
    }

    pub(crate) fn paint_console(&mut self, content: Rect, ui_events: &[InputEvent],
                                frame: &mut Frame, renderer: &mut Renderer, theme: &Theme) {
        let (cell_w, line_h) = renderer.cell_metrics(theme.fs());
        // Console body follows the THEME (was hardcoded white, which made
        // the console the one surface a dark theme couldn't reach).
        frame.paint_rect(content.x, content.y, content.w, content.h, theme.window_background);

        // The grid shrinks by the scrollbar thickness on the right and
        // bottom so transcript content never lands under the bars.
        let grid_rect = transcript_rect(content);
        let sbt = SCROLLBAR_THICKNESS;
        let vtrack = Rect { x: grid_rect.x + grid_rect.w, y: grid_rect.y, w: sbt, h: grid_rect.h };
        let htrack = Rect { x: grid_rect.x, y: grid_rect.y + grid_rect.h, w: grid_rect.w, h: sbt };

        // ── Wrap long lines to the console width so output folds instead
        //    of running off the right edge. Wrap the transcript, then the
        //    prompt separately, so we can map the cursor into the wrapped
        //    grid: the prompt starts at `prompt_base`, and a caret that is
        //    `cursor_col` chars in lands `/ wrap` rows down and `% wrap`
        //    cols across.
        let cursor_col = self.input.prompt.chars().count() + 1
            + self.input.current[..self.input.cursor].chars().count();
        let wrap = (grid_rect.w / cell_w).floor() as usize;
        let mut rows = wrap_rows(rows_from_buffer(&self.buffer.lock().unwrap(), theme), wrap);
        let prompt_base = rows.len();
        let (prompt_row_index, cursor_col_in_row) = if wrap > 0 {
            (prompt_base + cursor_col / wrap, cursor_col % wrap)
        } else {
            (prompt_base, cursor_col)
        };
        rows.extend(wrap_rows(vec![prompt_row(&self.input, theme)], wrap));

        let view = View {
            grid_rect, vtrack, htrack, line_h,
            total_rows: rows.len(),
            // With wrapping on, no row exceeds `wrap`, so the horizontal
            // bar stays inert (full thumb) — kept only as a no-op.
            max_cols: rows.iter().map(|r| r.len()).max().unwrap_or(0)
                .max(cursor_col_in_row + 1),
            visible_rows: (grid_rect.h / line_h).floor() as usize,
            visible_cols: (grid_rect.w / cell_w).floor() as usize,
        };
        self.drive_scrollbars(ui_events, &view, cursor_col_in_row);

        // ── Transcript paint — uses the scroll state CellGridState owns.
        self.grid.paint(frame, renderer, &rows, grid_rect, cell_w, line_h, theme.fs(), theme);

        // ── Cursor — must follow the SAME effective vertical scroll the
        //    painter used.
        let scroll = self.grid.scroll_y_override
            .unwrap_or_else(|| auto_scroll_offset(rows.len(), grid_rect.h, line_h));
        let scroll_x = self.grid.scroll_x;
        let cursor_on = (self.frame_counter / 30).is_multiple_of(2);
        if cursor_on && prompt_row_index >= scroll && cursor_col_in_row >= scroll_x {
            let cx = grid_rect.x + (cursor_col_in_row - scroll_x) as f32 * cell_w;
            let cy = grid_rect.y + (prompt_row_index - scroll) as f32 * line_h;
            if cy + line_h <= grid_rect.y + grid_rect.h && cx + 2.0 <= grid_rect.x + grid_rect.w {
                frame.paint_rect(cx, cy + line_h * 0.1, 2.0, line_h * 0.8, theme.cursor);
            }
        }

        // ── Scrollbars on top of the transcript.
        self.vscroll.paint(frame, vtrack, theme);
        self.hscroll.paint(frame, htrack, theme);
    }

    /// Scroll state from the current content vs viewport sizes (in cell
    /// units), the user's scrollbar drags, and the typing cursor.
    fn drive_scrollbars(&mut self, ui_events: &[InputEvent], v: &View, cursor_col_in_row: usize) {
        // ── Snap to the prompt on new output (R-console). If the user had
        //    scrolled up (pinned override) or right (long line) and a
        //    command returns, the fresh prompt would otherwise sit below /
        //    left-of the view — i.e. hidden. Any transcript growth hands
        //    control back to auto-scroll and returns the horizontal bar to
        //    its default left position. Typing doesn't change the row
        //    count, so this never fights the cursor-follow below.
        if self.last_total_rows != v.total_rows {
            self.last_total_rows = v.total_rows;
            self.grid.scroll_y_override = None;
            self.grid.scroll_x = 0;
        }
        if v.total_rows > 0 {
            self.vscroll.visible_fraction = (v.visible_rows as f32 / v.total_rows as f32).min(1.0);
        }
        if v.max_cols > 0 {
            self.hscroll.visible_fraction = (v.visible_cols as f32 / v.max_cols as f32).min(1.0);
        }
        if let Some(p) = self.vscroll.handle_events(ui_events, v.vtrack) {
            let off = r2_ui::scroll_pos_to_row(p, v.total_rows, v.visible_rows);
            // Pin to the manual offset; dragged to the bottom hands control
            // back to auto-scroll so new lines keep showing.
            self.grid.scroll_y_override =
                if off + v.visible_rows >= v.total_rows { None } else { Some(off) };
        }
        if let Some(p) = self.hscroll.handle_events(ui_events, v.htrack) {
            self.grid.scroll_x = r2_ui::scroll_pos_to_col(p, v.max_cols, v.visible_cols);
        }

        // ── Horizontal cursor-follow (R-console). Keep the typing cursor
        //    visible without dragging the prompt off the left margin:
        //   • cursor within the first screen-width ("default frame") →
        //     snap scroll_x to 0 so the prompt shows at the left;
        //   • cursor past the right edge → slide right just enough;
        //   • cursor left of the view but beyond the default frame →
        //     follow it so it stays on screen.
        // Only when the user actually TYPED this frame (input text
        // changed); otherwise a manual drag / wheel of the bottom bar
        // would snap back to 0 every frame.
        let typed = self.last_input != self.input.current;
        if typed {
            self.last_input = self.input.current.clone();
        }
        if typed && v.visible_cols > 0 {
            let gs = &mut self.grid;
            if cursor_col_in_row < v.visible_cols {
                gs.scroll_x = 0;
            } else if cursor_col_in_row >= gs.scroll_x + v.visible_cols {
                gs.scroll_x = cursor_col_in_row + 1 - v.visible_cols;
            } else if cursor_col_in_row < gs.scroll_x {
                gs.scroll_x = cursor_col_in_row;
            }
        }

        // ── Keep the thumbs in sync with the ACTUAL scroll offset every
        //    frame: wheel / touchpad, auto-scroll-to-bottom and keyboard
        //    (Shift+Arrow) selection, not just thumb drags. R-console
        //    behaviour: the bar always reflects where the transcript is.
        let eff_y = self.grid.scroll_y_override
            .unwrap_or_else(|| auto_scroll_offset(v.total_rows, v.grid_rect.h, v.line_h));
        self.vscroll.position = r2_ui::row_offset_to_scroll_pos(eff_y, v.total_rows, v.visible_rows);
        self.hscroll.position =
            r2_ui::col_offset_to_scroll_pos(self.grid.scroll_x, v.max_cols, v.visible_cols);
    }
}

/// The console transcript's geometry for one frame.
struct View {
    grid_rect: Rect,
    vtrack: Rect,
    htrack: Rect,
    line_h: f32,
    total_rows: usize,
    max_cols: usize,
    visible_rows: usize,
    visible_cols: usize,
}
