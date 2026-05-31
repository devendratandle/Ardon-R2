//! InputField — the prompt + currently-typed-line widget.
//!
//! Single-line edit field with cursor locked to the line, Up/Down
//! arrow history (driven by `ConsoleBuffer`), Home/End/Left/Right for
//! intra-line nav, Backspace/Delete for edits, Ctrl+V for paste.
//!
//! Multi-line constructs (for-loops, function defs) use the `+`
//! continuation prompt managed by `ConsoleBuffer` — the user types
//! one physical line at a time, the buffer accumulates them.

use crate::event::{Clipboard, InputEvent, KeyCode};
use crate::grid::Rect;
use crate::render::{Frame, Renderer};
use crate::theme::{Color, Theme};

/// Stateful prompt-line editor.
///
/// Holds the *currently being typed* line plus an insertion cursor.
/// History navigation populates `current` from a `ConsoleBuffer`; the
/// host wires that up because the buffer owns the history list.
pub struct InputField {
    pub prompt: String,
    pub current: String,
    /// Byte-index cursor inside `current`. Always at a UTF-8 boundary.
    pub cursor: usize,
}

impl InputField {
    pub fn new() -> Self {
        Self { prompt: "R2>".into(), current: String::new(), cursor: 0 }
    }

    pub fn prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    pub fn set_prompt(&mut self, p: impl Into<String>) {
        self.prompt = p.into();
    }

    /// Set the line content (used by history recall). Cursor moves to
    /// end-of-line, which matches every shell.
    pub fn set_line(&mut self, s: String) {
        self.current = s;
        self.cursor  = self.current.len();
    }

    pub fn clear(&mut self) {
        self.current.clear();
        self.cursor = 0;
    }

    /// Walk one frame's events, mutating the line + cursor. Returns:
    /// * `Submit(line)` — user pressed Enter; line is the trimmed value.
    /// * `HistoryUp` / `HistoryDown` — host should ask its
    ///   `ConsoleBuffer` for the next history entry and call
    ///   [`set_line`].
    /// * `None` — nothing actionable this frame.
    pub fn handle_events(
        &mut self,
        events: &[InputEvent],
        clipboard: &mut Clipboard,
    ) -> InputFieldResponse {
        let mut submitted: Option<String> = None;
        let mut history_up   = false;
        let mut history_down = false;
        let mut auto_submit_lines: Vec<String> = Vec::new();

        for ev in events {
            match *ev {
                InputEvent::Char(ch) => {
                    self.current.insert(self.cursor, ch);
                    self.cursor += ch.len_utf8();
                }
                InputEvent::Key { code, mods, pressed: true } => match code {
                    KeyCode::Enter => {
                        let line = std::mem::take(&mut self.current);
                        self.cursor = 0;
                        submitted = Some(line);
                    }
                    KeyCode::Backspace => {
                        if self.cursor > 0 {
                            // Walk back to the previous char boundary.
                            let mut i = self.cursor - 1;
                            while !self.current.is_char_boundary(i) && i > 0 { i -= 1; }
                            self.current.replace_range(i..self.cursor, "");
                            self.cursor = i;
                        }
                    }
                    KeyCode::Delete => {
                        if self.cursor < self.current.len() {
                            let mut j = self.cursor + 1;
                            while j < self.current.len() && !self.current.is_char_boundary(j) { j += 1; }
                            self.current.replace_range(self.cursor..j, "");
                        }
                    }
                    KeyCode::Left => {
                        if self.cursor > 0 {
                            let mut i = self.cursor - 1;
                            while !self.current.is_char_boundary(i) && i > 0 { i -= 1; }
                            self.cursor = i;
                        }
                    }
                    KeyCode::Right => {
                        if self.cursor < self.current.len() {
                            let mut j = self.cursor + 1;
                            while j < self.current.len() && !self.current.is_char_boundary(j) { j += 1; }
                            self.cursor = j;
                        }
                    }
                    KeyCode::Home => self.cursor = 0,
                    KeyCode::End  => self.cursor = self.current.len(),
                    KeyCode::Up   => history_up   = true,
                    KeyCode::Down => history_down = true,
                    KeyCode::KeyV if mods.ctrl => {
                        if let Some(s) = clipboard.get_text() {
                            let s = s.replace('\r', "");
                            if !s.contains('\n') {
                                // Single-line paste: just insert at cursor.
                                self.current.insert_str(self.cursor, &s);
                                self.cursor += s.len();
                            } else {
                                // Multi-line paste — R Console semantics:
                                //  * first paste-line completes the line
                                //    the user was already typing → submit it
                                //  * each intermediate line submits as if
                                //    Enter-pressed
                                //  * the trailing line (whatever follows
                                //    the last newline, often empty) is
                                //    left in the editor so the user can
                                //    keep typing.
                                let mut parts: Vec<String> =
                                    s.split('\n').map(|p| p.to_string()).collect();
                                // First chunk goes into current at the cursor.
                                let head = parts.remove(0);
                                self.current.insert_str(self.cursor, &head);
                                // The (now-complete) current line auto-submits.
                                auto_submit_lines.push(std::mem::take(&mut self.current));
                                self.cursor = 0;
                                // Take everything except the final piece —
                                // those are intermediate complete lines.
                                let tail = parts.pop().unwrap_or_default();
                                for line in parts {
                                    auto_submit_lines.push(line);
                                }
                                // The final piece stays in the editor.
                                self.current = tail;
                                self.cursor  = self.current.len();
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        InputFieldResponse {
            submitted,
            history_up,
            history_down,
            auto_submit_lines,
        }
    }

    /// Paint the prompt + current line + blinking cursor at the given
    /// baseline (top-left rect). Returns the prompt's pixel width so
    /// callers know where the editable area begins.
    pub fn paint(
        &self,
        frame: &mut Frame,
        renderer: &mut Renderer,
        rect: Rect,
        cell_w: f32, line_h: f32,
        size_pt: f32,
        theme: &Theme,
        cursor_visible: bool,
    ) {
        let baseline = rect.y + line_h * 0.8;
        let prompt_color = theme.console_input;
        // Prompt + trailing space.
        let after_prompt = frame.paint_text(renderer, rect.x, baseline,
                                            &self.prompt, size_pt, prompt_color);
        let line_x0 = after_prompt + cell_w * 0.5;
        // Editable line — same color as prompt for visual cohesion.
        let _ = frame.paint_text(renderer, line_x0, baseline,
                                 &self.current, size_pt, theme.console_input);

        // Cursor — solid rectangle at cursor byte position. We use
        // mono-cell width × cursor's char-count for x position.
        if cursor_visible {
            let chars_before = self.current[..self.cursor].chars().count();
            let cx = line_x0 + chars_before as f32 * cell_w;
            // Thin I-beam: 2 px wide, line height tall.
            frame.paint_rect(cx, rect.y + line_h * 0.1,
                             2.0, line_h * 0.8, theme.cursor);
        }

        // Silence the dead-code warning on Color until callers want
        // a separate prompt color.
        let _ = Color::WHITE;
    }
}

impl Default for InputField {
    fn default() -> Self { Self::new() }
}

/// Returned each frame from `InputField::handle_events`.
#[derive(Debug, Default)]
pub struct InputFieldResponse {
    /// Set on the frame the user pressed Enter; contains the line text
    /// (may be empty for blank submissions, which the buffer treats as
    /// a fresh prompt).
    pub submitted: Option<String>,
    /// User pressed Up — host should call `buffer.history_up()` and
    /// `field.set_line(...)`.
    pub history_up: bool,
    /// User pressed Down — `buffer.history_down()` + `set_line`.
    pub history_down: bool,
    /// Lines that arrived from a MULTI-LINE clipboard paste, in order.
    /// The host should feed each one to `ConsoleBuffer::submit_line`
    /// (and dispatch the resulting `SubmitAction`) before processing
    /// `submitted` — same as if the user had typed them and pressed
    /// Enter between each. The final, partial line of the paste is
    /// left in `current` for further editing.
    pub auto_submit_lines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{KeyCode, Mods};

    fn ch(c: char) -> InputEvent { InputEvent::Char(c) }
    fn key(c: KeyCode) -> InputEvent {
        InputEvent::Key { code: c, mods: Mods::default(), pressed: true }
    }

    #[test]
    fn typing_chars_advances_cursor() {
        let mut f = InputField::new();
        let mut clip = Clipboard::new();
        let r = f.handle_events(&[ch('x'), ch(' '), ch('<'), ch('-'), ch(' '), ch('1')], &mut clip);
        assert!(r.submitted.is_none());
        assert_eq!(f.current, "x <- 1");
        assert_eq!(f.cursor, 6);
    }

    #[test]
    fn enter_submits_and_clears() {
        let mut f = InputField::new();
        let mut clip = Clipboard::new();
        f.handle_events(&[ch('1'), ch('+'), ch('2')], &mut clip);
        let r = f.handle_events(&[key(KeyCode::Enter)], &mut clip);
        assert_eq!(r.submitted.as_deref(), Some("1+2"));
        assert_eq!(f.current, "");
        assert_eq!(f.cursor, 0);
    }

    #[test]
    fn backspace_removes_prev_char() {
        let mut f = InputField::new();
        let mut clip = Clipboard::new();
        f.handle_events(&[ch('a'), ch('b'), ch('c')], &mut clip);
        f.handle_events(&[key(KeyCode::Backspace)], &mut clip);
        assert_eq!(f.current, "ab");
        assert_eq!(f.cursor, 2);
    }

    #[test]
    fn left_right_home_end_move_cursor() {
        let mut f = InputField::new();
        let mut clip = Clipboard::new();
        f.handle_events(&[ch('a'), ch('b'), ch('c'), ch('d')], &mut clip);
        f.handle_events(&[key(KeyCode::Home)], &mut clip);
        assert_eq!(f.cursor, 0);
        f.handle_events(&[key(KeyCode::Right), key(KeyCode::Right)], &mut clip);
        assert_eq!(f.cursor, 2);
        f.handle_events(&[key(KeyCode::End)], &mut clip);
        assert_eq!(f.cursor, 4);
        f.handle_events(&[key(KeyCode::Left)], &mut clip);
        assert_eq!(f.cursor, 3);
    }

    #[test]
    fn up_down_set_history_flags() {
        let mut f = InputField::new();
        let mut clip = Clipboard::new();
        let r = f.handle_events(&[key(KeyCode::Up)], &mut clip);
        assert!(r.history_up && !r.history_down);
        let r = f.handle_events(&[key(KeyCode::Down)], &mut clip);
        assert!(r.history_down && !r.history_up);
    }

    #[test]
    fn set_line_moves_cursor_to_end() {
        let mut f = InputField::new();
        f.set_line("for (i in 1:5) print(i)".into());
        assert_eq!(f.cursor, f.current.len());
    }
}
