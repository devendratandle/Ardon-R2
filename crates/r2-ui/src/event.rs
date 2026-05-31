//! Input events — host-agnostic shape that widgets consume.
//!
//! `R2Ui::run` accumulates one frame's worth of winit events, converts
//! them into [`InputEvent`]s, and hands the slice to the user-provided
//! `on_frame` closure. Widgets like `CellGridState` and `InputField`
//! walk the slice to update their state.
//!
//! Keeping our own enum means widget code does not link winit directly
//! and the same widget would work under a different backend later.

/// Pixel-space mouse position (top-left origin).
#[derive(Debug, Clone, Copy)]
pub struct MousePos { pub x: f32, pub y: f32 }

/// Mouse button. Only the three useful ones — no horizontal-wheel,
/// no thumb buttons (yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton { Left, Right, Middle }

/// Logical keys we actually react to. Kept tiny so widgets only
/// pattern-match on what matters; printable characters arrive via
/// [`InputEvent::Char`] instead of this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Enter, Backspace, Delete, Tab, Escape,
    Left, Right, Up, Down, Home, End, PageUp, PageDown,
    /// Letter keys we care about for hotkeys (A, C, V, X). Other
    /// letters arrive as printable Chars.
    KeyA, KeyC, KeyV, KeyX,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mods {
    pub shift: bool,
    pub ctrl:  bool,
    pub alt:   bool,
}

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    MouseMoved(MousePos),
    MouseDown { button: MouseButton, pos: MousePos },
    MouseUp   { button: MouseButton, pos: MousePos },
    Scroll    { dy: f32 },
    /// Printable character typed (Enter / Backspace etc. do NOT come
    /// through here — they come through [`InputEvent::Key`]).
    Char(char),
    /// Key press (pressed=true) or release (pressed=false).
    Key { code: KeyCode, mods: Mods, pressed: bool },
}

// ─── Clipboard wrapper ───────────────────────────────────────────────

/// Thin handle around `arboard::Clipboard`. Construction can fail on
/// headless / locked-down systems; in that case the handle is silently
/// inert and `set`/`get` become no-ops. Widgets do not need to special-case.
pub struct Clipboard {
    inner: Option<arboard::Clipboard>,
}

impl Clipboard {
    pub fn new() -> Self {
        Self { inner: arboard::Clipboard::new().ok() }
    }
    pub fn set_text(&mut self, s: &str) {
        if let Some(c) = self.inner.as_mut() { let _ = c.set_text(s); }
    }
    /// Raw clipboard text, unprocessed. Most callers want
    /// [`get_text`] which returns the sanitized form.
    pub fn get_raw_text(&mut self) -> Option<String> {
        self.inner.as_mut()?.get_text().ok()
    }
    /// Sanitized clipboard text — see [`normalize_paste`]. Use this
    /// for any code path that feeds the clipboard into an R2 console
    /// prompt or transcript: smart quotes from Word, em-dashes from
    /// browsers, mixed `\r\n` line endings, BOMs from PowerShell
    /// output all get folded down to plain ASCII-friendly text. The
    /// only structure we preserve is line breaks (`\n`).
    pub fn get_text(&mut self) -> Option<String> {
        let raw = self.get_raw_text()?;
        Some(normalize_paste(&raw))
    }

    /// Put an RGBA image on the clipboard. `rgba.len()` must equal
    /// `width * height * 4` bytes (R, G, B, A per pixel). On every
    /// desktop OS the receiving app pastes this as a bitmap — Word,
    /// Excel, Outlook, image editors, etc. Returns `true` on success.
    pub fn set_image(&mut self, width: u32, height: u32, rgba: &[u8]) -> bool {
        let inner = match self.inner.as_mut() { Some(c) => c, None => return false };
        if rgba.len() != (width as usize) * (height as usize) * 4 { return false; }
        let img = arboard::ImageData {
            width: width as usize,
            height: height as usize,
            bytes: std::borrow::Cow::Borrowed(rgba),
        };
        inner.set_image(img).is_ok()
    }
}

/// Universal paste sanitizer. Converts any clipboard payload into
/// clean plain text suitable for an R2 prompt:
///
/// * `\r\n` and `\r` → `\n` (Unix line endings, single representation).
/// * BOM (`\u{FEFF}`) stripped.
/// * Curly quotes (`"`, `"`, `'`, `'`) → straight ASCII quotes.
/// * En-dash / em-dash / minus-sign / hyphen-bullet (`–`, `—`, `−`,
///   `‐`) → ASCII hyphen-minus `-`.
/// * Non-breaking space, narrow NBSP, figure-space → regular space.
/// * Ellipsis `…` → `...`.
/// * Other control chars (except `\n` and `\t`) stripped.
///
/// Trailing whitespace on each line is preserved (some R syntax
/// depends on it). Empty lines are kept (they're R-meaningful too).
///
/// This is a pure function so it's trivially testable.
pub fn normalize_paste(input: &str) -> String {
    // Pre-pass: line-ending normalization.
    let s = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\u{FEFF}' => {}                             // BOM — drop
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}'  // ‘ ’ ‚ ‛
                       => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}'  // “ ” „ ‟
                       => out.push('"'),
            '\u{2013}' | '\u{2014}' | '\u{2212}' | '\u{2010}'  // – — − ‐
                       => out.push('-'),
            '\u{00A0}' | '\u{202F}' | '\u{2007}'                // NBSP, NNBSP, FIGURE SP
                       => out.push(' '),
            '\u{2026}' => out.push_str("..."),                  // …
            c if c == '\n' || c == '\t' => out.push(c),
            c if (c as u32) < 0x20 || (c as u32) == 0x7F => {}  // C0 / DEL — drop
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod paste_tests {
    use super::normalize_paste;

    #[test]
    fn smart_quotes_become_ascii() {
        assert_eq!(normalize_paste("\u{201C}hi\u{201D} \u{2018}ok\u{2019}"),
                   "\"hi\" 'ok'");
    }
    #[test]
    fn em_and_en_dashes_become_hyphen() {
        assert_eq!(normalize_paste("a\u{2013}b\u{2014}c\u{2212}d"), "a-b-c-d");
    }
    #[test]
    fn crlf_normalized() {
        assert_eq!(normalize_paste("one\r\ntwo\rthree"), "one\ntwo\nthree");
    }
    #[test]
    fn bom_and_nbsp_handled() {
        assert_eq!(normalize_paste("\u{FEFF}a\u{00A0}b"), "a b");
    }
    #[test]
    fn ellipsis_expanded() {
        assert_eq!(normalize_paste("wait\u{2026}done"), "wait...done");
    }
    #[test]
    fn newlines_and_tabs_preserved() {
        assert_eq!(normalize_paste("a\nb\tc"), "a\nb\tc");
    }
    #[test]
    fn bell_and_other_control_chars_stripped() {
        assert_eq!(normalize_paste("x\u{0007}y"), "xy");
    }
}

impl Default for Clipboard {
    fn default() -> Self { Self::new() }
}

// ─── winit → InputEvent ──────────────────────────────────────────────
//
// Kept in this module so the conversion lives next to the enum. The
// app loop calls `from_winit` from inside its match-arm; widgets
// never see winit types.

/// Convert a winit `WindowEvent` into zero or more [`InputEvent`]s,
/// using the supplied `mouse_pos` (latched by the caller from the
/// previous `CursorMoved`) and `mods` (latched from `ModifiersChanged`).
pub fn from_winit(
    we: &winit::event::WindowEvent,
    mouse_pos: MousePos,
    mods: Mods,
    scale_factor: f64,
) -> Vec<InputEvent> {
    use winit::event::{ElementState, MouseButton as WB, MouseScrollDelta, WindowEvent};
    use winit::keyboard::{Key, NamedKey};

    let mut out = Vec::new();
    match we {
        WindowEvent::CursorMoved { position, .. } => {
            let p = position.to_logical::<f64>(scale_factor);
            out.push(InputEvent::MouseMoved(MousePos { x: p.x as f32, y: p.y as f32 }));
        }
        // Touch events (Android, iOS, Windows tablets). We synthesize
        // left-mouse events from primary-finger touches so every widget
        // built against MouseDown / MouseMoved / MouseUp keeps working
        // on mobile without modification. Multi-touch gestures will
        // need their own pass — for now, single-touch only.
        WindowEvent::Touch(t) => {
            use winit::event::TouchPhase;
            let p = t.location.to_logical::<f64>(scale_factor);
            let pos = MousePos { x: p.x as f32, y: p.y as f32 };
            out.push(InputEvent::MouseMoved(pos));
            match t.phase {
                TouchPhase::Started   => out.push(InputEvent::MouseDown { button: MouseButton::Left, pos }),
                TouchPhase::Moved     => { /* MouseMoved already pushed */ }
                TouchPhase::Ended |
                TouchPhase::Cancelled => out.push(InputEvent::MouseUp   { button: MouseButton::Left, pos }),
            }
        }
        WindowEvent::MouseInput { state, button, .. } => {
            let b = match button {
                WB::Left   => MouseButton::Left,
                WB::Right  => MouseButton::Right,
                WB::Middle => MouseButton::Middle,
                _ => return out,
            };
            let ev = match state {
                ElementState::Pressed  => InputEvent::MouseDown { button: b, pos: mouse_pos },
                ElementState::Released => InputEvent::MouseUp   { button: b, pos: mouse_pos },
            };
            out.push(ev);
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let dy = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y * 20.0,
                MouseScrollDelta::PixelDelta(p)   => p.y as f32,
            };
            out.push(InputEvent::Scroll { dy });
        }
        WindowEvent::KeyboardInput { event, .. } => {
            // OS-level key-repeat handling: ALLOW repeats for the
            // editing keys (Backspace / Delete / arrows / Home / End)
            // so holding them does what users expect; BLOCK repeats
            // for everything else (Enter would spam empty-submit;
            // printable characters would auto-fill the buffer).
            let pressed = event.state == ElementState::Pressed;
            if event.repeat {
                use winit::keyboard::{Key, NamedKey};
                let allow_repeat = matches!(&event.logical_key,
                    Key::Named(NamedKey::Backspace)  |
                    Key::Named(NamedKey::Delete)     |
                    Key::Named(NamedKey::ArrowLeft)  |
                    Key::Named(NamedKey::ArrowRight) |
                    Key::Named(NamedKey::ArrowUp)    |
                    Key::Named(NamedKey::ArrowDown)  |
                    Key::Named(NamedKey::Home)       |
                    Key::Named(NamedKey::End)        |
                    Key::Named(NamedKey::PageUp)     |
                    Key::Named(NamedKey::PageDown)
                );
                if !allow_repeat { return out; }
            }

            // Logical-key → KeyCode mapping (named keys only).
            let code = match &event.logical_key {
                Key::Named(NamedKey::Enter)      => Some(KeyCode::Enter),
                Key::Named(NamedKey::Backspace)  => Some(KeyCode::Backspace),
                Key::Named(NamedKey::Delete)     => Some(KeyCode::Delete),
                Key::Named(NamedKey::Tab)        => Some(KeyCode::Tab),
                Key::Named(NamedKey::Escape)     => Some(KeyCode::Escape),
                Key::Named(NamedKey::ArrowLeft)  => Some(KeyCode::Left),
                Key::Named(NamedKey::ArrowRight) => Some(KeyCode::Right),
                Key::Named(NamedKey::ArrowUp)    => Some(KeyCode::Up),
                Key::Named(NamedKey::ArrowDown)  => Some(KeyCode::Down),
                Key::Named(NamedKey::Home)       => Some(KeyCode::Home),
                Key::Named(NamedKey::End)        => Some(KeyCode::End),
                Key::Named(NamedKey::PageUp)     => Some(KeyCode::PageUp),
                Key::Named(NamedKey::PageDown)   => Some(KeyCode::PageDown),
                Key::Character(s) => {
                    let c = s.chars().next().unwrap_or(' ').to_ascii_lowercase();
                    match c {
                        'a' => Some(KeyCode::KeyA),
                        'c' => Some(KeyCode::KeyC),
                        'v' => Some(KeyCode::KeyV),
                        'x' => Some(KeyCode::KeyX),
                        _   => None,
                    }
                }
                _ => None,
            };
            if let Some(code) = code {
                out.push(InputEvent::Key { code, mods, pressed });
            }

            // Printable Char events on press only, and only when Ctrl
            // / Alt are not held (those are reserved for hotkeys).
            if pressed && !mods.ctrl && !mods.alt {
                if let Some(text) = event.text.as_ref() {
                    for ch in text.chars() {
                        // Reject control chars; Enter/Backspace handled above.
                        if !ch.is_control() {
                            out.push(InputEvent::Char(ch));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}
