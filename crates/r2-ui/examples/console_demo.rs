//! R2-UI · Phase 2 Week 3 integration demo.
//!
//! A real interactive console — no engine yet, but the full
//! interaction model:
//!   * Typing characters → appears at the prompt
//!   * Enter → submits to ConsoleBuffer, echoes a fake "ok" response
//!   * Up / Down → walk history
//!   * Left / Right / Home / End → cursor navigation in the prompt
//!   * Backspace / Delete → edit
//!   * Mouse drag in the transcript → selection band
//!   * Ctrl+A → select all transcript
//!   * Ctrl+C → copy selection to clipboard
//!   * Ctrl+V → paste at cursor
//!
//! This is the end-to-end smoke test for Phase 2 Week 3.
//! Run with:
//!   cargo run -p r2-ui --example console_demo

use std::cell::RefCell;
use std::rc::Rc;

use r2_console::{ConsoleBuffer, LineKind, SubmitAction};
use r2_ui::{
    Cell, CellGridState, Color, InputField, R2Ui, Rect, Theme,
};

fn line_color(theme: &Theme, kind: LineKind) -> Color {
    match kind {
        LineKind::Input        => theme.console_input,
        LineKind::Continuation => theme.console_input,
        LineKind::Output       => theme.console_output,
        LineKind::Error        => theme.console_error,
        LineKind::Banner       => theme.console_banner,
    }
}

/// Convert a `ConsoleBuffer`'s transcript into rows of `Cell`s using
/// the theme's per-kind colors.
fn rows_from_buffer(buf: &ConsoleBuffer, theme: &Theme) -> Vec<Vec<Cell>> {
    buf.transcript().iter()
        .map(|cl| {
            let col = line_color(theme, cl.kind);
            cl.text.chars().map(|c| Cell::plain(c, col)).collect()
        })
        .collect()
}

fn main() -> Result<(), String> {
    // Shared state — Rc<RefCell> because the closure is FnMut and
    // captures need to be moved in once.
    let theme = Theme::khaki();
    let buffer = Rc::new(RefCell::new(ConsoleBuffer::new()));
    buffer.borrow_mut().push_banner("Ardon-R2 · R2-UI Week 3 console demo");
    buffer.borrow_mut().push_banner("Type any text and press Enter. Up/Down recalls history.");
    buffer.borrow_mut().push_banner("Drag to select. Ctrl+C copies. Ctrl+V pastes. Ctrl+A selects all.");
    buffer.borrow_mut().push_banner("");

    let grid_state = Rc::new(RefCell::new(CellGridState::new()));
    let input      = Rc::new(RefCell::new(InputField::new()));

    // Cursor blink — toggle every ~16 frames at 60 FPS ≈ every 0.27s.
    let frame_counter = Rc::new(RefCell::new(0u64));

    R2Ui::app("R2-UI · Phase 2 Week 3 — Console Demo")
        .theme(theme.clone())
        .initial_size(960, 640)
        .on_frame({
            let buffer       = buffer.clone();
            let grid_state   = grid_state.clone();
            let input        = input.clone();
            let frame_counter = frame_counter.clone();
            move |ctx, renderer, frame, theme| {
                *frame_counter.borrow_mut() += 1;

                let size_pt = theme.font_size;
                let (cell_w, line_h) = renderer.cell_metrics(size_pt);

                // ── Layout: transcript fills top, prompt sits at bottom.
                let pad = 12.0;
                let win_w = renderer.size.width  as f32;
                let win_h = renderer.size.height as f32;
                let prompt_h = line_h + 6.0;
                let transcript_rect = Rect {
                    x: pad, y: pad,
                    w: win_w - 2.0 * pad,
                    h: win_h - 2.0 * pad - prompt_h - 6.0,
                };
                let prompt_rect = Rect {
                    x: pad,
                    y: transcript_rect.y + transcript_rect.h + 6.0,
                    w: win_w - 2.0 * pad,
                    h: prompt_h,
                };

                // ── InputField: process events + maybe submit.
                let mut input_mut = input.borrow_mut();
                let resp = input_mut.handle_events(ctx.events, ctx.clipboard);
                if let Some(line) = resp.submitted {
                    // Mirror what the GUI's REPL loop will eventually do:
                    // submit to buffer, then for any non-Submit result
                    // do nothing else; for Submit, fake an output line
                    // so users can see end-to-end roundtrip.
                    let action = buffer.borrow_mut().submit_line(line.clone());
                    match action {
                        SubmitAction::Submit(src) => {
                            // No engine yet — fake an echo.
                            buffer.borrow_mut().push_output(&format!("(would eval) {}", src.trim()));
                        }
                        SubmitAction::Continue | SubmitAction::Empty => {}
                    }
                }
                if resp.history_up {
                    if let Some(s) = buffer.borrow_mut().history_up() {
                        input_mut.set_line(s);
                    }
                }
                if resp.history_down {
                    if let Some(s) = buffer.borrow_mut().history_down() {
                        input_mut.set_line(s);
                    }
                }

                // ── Reflect ConsoleBuffer's current prompt (R2> vs +).
                input_mut.set_prompt(buffer.borrow().current_prompt());

                // ── CellGridState: events + selection band.
                let rows = rows_from_buffer(&buffer.borrow(), theme);
                let _copied = grid_state.borrow_mut().handle_events(
                    ctx.events, &rows, transcript_rect,
                    cell_w, line_h, ctx.clipboard,
                );

                // ── Paint transcript.
                grid_state.borrow().paint(frame, renderer, &rows, transcript_rect,
                                          cell_w, line_h, size_pt, theme);

                // ── Paint prompt area with a faint top divider.
                frame.paint_rect(prompt_rect.x, prompt_rect.y - 1.0,
                                 prompt_rect.w, 1.0,
                                 Color::rgba(40, 40, 40, 60));
                let cursor_on = (*frame_counter.borrow() / 30) % 2 == 0;
                input_mut.paint(frame, renderer, prompt_rect,
                                cell_w, line_h, size_pt, theme, cursor_on);
            }
        })
        .run()
}
