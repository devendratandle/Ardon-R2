//! Ardon-R2 console core — shared REPL state machine.
//!
//! This crate is the **single source of truth** for how the R2 REPL
//! behaves. Both the CLI (`r2.exe`) and the GUI (`R2Gui.exe`) drive
//! the same [`ConsoleBuffer`] type — only the rendering and input
//! capture layers differ. The engine writes its output through the
//! same [`OutputSink`] trait regardless of host.
//!
//! Design mirrors R's internal architecture (see `Rinterface.h`'s
//! `ptr_R_WriteConsole` / `ptr_R_ReadConsole` callbacks): the engine
//! talks to abstractions, the frontend installs concrete implementations.
//!
//! ## Three-point REPL state machine
//!
//! Every interactive shell — bash, zsh, R, Python `-i`, IRB — follows
//! the same loop:
//!
//! 1. **Show prompt.** `R2>` for a new expression, `+` mid-continuation.
//! 2. **Read one line.** Cursor stays inside that line; Enter ends it.
//! 3. **Check completeness.** If parens / braces / brackets balance,
//!    submit to the engine and wait for all output. If not, accumulate
//!    and loop back to step 1 with the `+` prompt.
//!
//! The state machine for steps 2-3 lives in [`ConsoleBuffer`].
//!
//! ## Used by
//!
//! * `r2-repl` (CLI): drives `ConsoleBuffer` with raw stdin lines,
//!   installs an `OutputSink` that prints to stdout with ANSI colors.
//! * `r2-gui` (GUI): drives `ConsoleBuffer` from an egui `TextEdit`
//!   widget, installs an `OutputSink` that pushes into the transcript.
//! * `r2-engine`: holds `output_sink: Box<dyn OutputSink>`. Every
//!   `bi_print`, `bi_cat`, `bi_message`, etc. writes through it
//!   instead of `println!`.

use r2_types::Expr;

// ─── Transcript lines ─────────────────────────────────────────────────

/// Semantic role of a transcript line. The renderer maps this to a
/// concrete color: input → red, output → blue, error → bright red,
/// banner → dim. The CLI uses ANSI escapes; the GUI uses egui colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    /// First line of an expression — typically rendered as the
    /// "R2> ..." prompt.
    Input,
    /// Continuation of a multi-line expression — typically the
    /// "+ ..." prompt.
    Continuation,
    /// Engine output / value auto-print.
    Output,
    /// Error / warning messages.
    Error,
    /// Welcome banner, info text, etc.
    Banner,
}

#[derive(Debug, Clone)]
pub struct ConsoleLine {
    pub kind: LineKind,
    pub text: String,
}

/// The action the host should take after `submit_line()`.
#[derive(Debug, Clone)]
pub enum SubmitAction {
    /// Expression complete; evaluate this source (multi-line OK).
    Submit(String),
    /// Expression still open (unbalanced braces); next prompt is `+`.
    Continue,
    /// Blank submission — emit a fresh prompt, do nothing else.
    Empty,
}

// ─── ConsoleBuffer ────────────────────────────────────────────────────

/// All REPL state: scrollback, accumulator, history.
///
/// Public API is host-agnostic. The host's render loop reads
/// `transcript()` for the visible scrollback and `current_prompt()`
/// for the next prompt to show. On user Enter, host calls
/// `submit_line(typed)` which returns a [`SubmitAction`].
pub struct ConsoleBuffer {
    transcript: Vec<ConsoleLine>,
    /// Lines accumulated for an in-progress multi-line expression.
    /// Empty iff we're at a fresh `R2>` prompt.
    continuation: String,
    history: Vec<String>,
    /// Position in history when the user is browsing with ↑/↓.
    /// `None` means not browsing (current command is fresh input).
    history_cursor: Option<usize>,
    /// Soft cap so heavy plotting / data-dump sessions don't grow
    /// the transcript without bound. The oldest lines are dropped
    /// when exceeded. Tunable via [`with_max_lines`].
    max_lines: usize,
}

impl Default for ConsoleBuffer {
    fn default() -> Self { Self::new() }
}

impl ConsoleBuffer {
    pub fn new() -> Self {
        Self {
            transcript: Vec::new(),
            continuation: String::new(),
            history: Vec::new(),
            history_cursor: None,
            max_lines: 10_000,
        }
    }

    pub fn with_max_lines(mut self, n: usize) -> Self {
        self.max_lines = n.max(100);
        self
    }

    // ── Inspection ────────────────────────────────────────────────────

    pub fn transcript(&self) -> &[ConsoleLine] { &self.transcript }
    pub fn history(&self) -> &[String] { &self.history }

    /// Prompt the renderer should display NOW.
    /// `R2>` for fresh input, `+` mid-continuation.
    pub fn current_prompt(&self) -> &'static str {
        if self.continuation.is_empty() { "R2>" } else { "+" }
    }

    pub fn in_continuation(&self) -> bool {
        !self.continuation.is_empty()
    }

    // ── Submission ────────────────────────────────────────────────────

    /// User pressed Enter with `line` as their typed text. Returns
    /// what the host should do next:
    /// * [`SubmitAction::Submit`] — call `engine.eval_string(&src)`,
    ///   wait for output to land via the OutputSink, then continue.
    /// * [`SubmitAction::Continue`] — next call to `current_prompt()`
    ///   will return `"+"`. Host shows that prompt and reads another line.
    /// * [`SubmitAction::Empty`] — blank submission, just show a new
    ///   `R2>` prompt.
    pub fn submit_line(&mut self, line: String) -> SubmitAction {
        // Echo the typed line into the transcript with the right
        // prompt prefix BEFORE updating state, so renderers see it.
        let kind = if self.continuation.is_empty() {
            LineKind::Input
        } else {
            LineKind::Continuation
        };
        let prompt = self.current_prompt();
        self.push_line(ConsoleLine {
            kind,
            text: format!("{} {}", prompt, line.trim_end()),
        });

        // Empty line: blank submission. R behaves the same: just
        // re-show a fresh prompt unless we're mid-continuation
        // (in which case treat as cancel — clear the accumulator).
        if line.trim().is_empty() {
            if !self.continuation.is_empty() {
                self.continuation.clear();
            }
            self.history_cursor = None;
            return SubmitAction::Empty;
        }

        // Append into the continuation accumulator.
        if !self.continuation.is_empty() {
            self.continuation.push('\n');
        }
        self.continuation.push_str(&line);

        // If accumulated expression is balanced, fire it.
        if !is_incomplete(&self.continuation) {
            let src = std::mem::take(&mut self.continuation);
            if !src.trim().is_empty() {
                self.push_history(src.clone());
            }
            self.history_cursor = None;
            return SubmitAction::Submit(src);
        }

        SubmitAction::Continue
    }

    // ── Output (engine → user) ────────────────────────────────────────

    /// Append engine output. May contain embedded newlines — they're
    /// split into separate transcript lines so the renderer can color
    /// each one.
    pub fn push_output(&mut self, text: &str) {
        if text.is_empty() {
            self.push_line(ConsoleLine { kind: LineKind::Output, text: String::new() });
            return;
        }
        for line in text.split('\n') {
            self.push_line(ConsoleLine { kind: LineKind::Output, text: line.to_string() });
        }
    }

    pub fn push_error(&mut self, text: &str) {
        for line in text.split('\n') {
            self.push_line(ConsoleLine { kind: LineKind::Error, text: line.to_string() });
        }
    }

    pub fn push_banner(&mut self, text: &str) {
        for line in text.split('\n') {
            self.push_line(ConsoleLine { kind: LineKind::Banner, text: line.to_string() });
        }
    }

    /// Wipe the transcript (Edit ▸ Clear console). History is kept.
    pub fn clear(&mut self) {
        self.transcript.clear();
    }

    fn push_line(&mut self, line: ConsoleLine) {
        self.transcript.push(line);
        if self.transcript.len() > self.max_lines {
            let drop = self.transcript.len() - self.max_lines;
            self.transcript.drain(0..drop);
        }
    }

    fn push_history(&mut self, src: String) {
        // Dedupe consecutive identical entries (bash behavior).
        if self.history.last().map(|h| h.as_str()) == Some(src.as_str()) {
            return;
        }
        self.history.push(src);
        // Cap history at 1000 entries (R default).
        if self.history.len() > 1000 {
            let drop = self.history.len() - 1000;
            self.history.drain(0..drop);
        }
    }

    // ── History navigation (↑ / ↓) ────────────────────────────────────

    /// Step backwards through history. Returns the recalled command.
    pub fn history_up(&mut self) -> Option<String> {
        if self.history.is_empty() { return None; }
        let next = match self.history_cursor {
            None    => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_cursor = Some(next);
        Some(self.history[next].clone())
    }

    /// Step forwards through history. Returns the recalled command,
    /// or `Some(String::new())` when stepping off the end (= clear input).
    pub fn history_down(&mut self) -> Option<String> {
        let i = self.history_cursor?;
        if i + 1 < self.history.len() {
            self.history_cursor = Some(i + 1);
            Some(self.history[i + 1].clone())
        } else {
            self.history_cursor = None;
            Some(String::new())
        }
    }
}

// ─── OutputSink trait ─────────────────────────────────────────────────
//
// The engine writes via this. CLI installs a StdoutSink. GUI installs
// a sink backed by Arc<Mutex<ConsoleBuffer>>. The engine never knows
// or cares which host it's running in.

pub trait OutputSink: Send + 'static {
    fn write_output(&mut self, text: &str);
    fn write_error(&mut self, text: &str);
}

/// Default sink: writes through to the process's real stdout/stderr.
/// Used by the CLI; useful as a fallback when no host has installed
/// a custom sink yet.
pub struct StdoutSink;

impl OutputSink for StdoutSink {
    fn write_output(&mut self, text: &str) {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        let _ = h.write_all(text.as_bytes());
        if !text.ends_with('\n') { let _ = h.write_all(b"\n"); }
        let _ = h.flush();
    }
    fn write_error(&mut self, text: &str) {
        use std::io::Write;
        let stderr = std::io::stderr();
        let mut h = stderr.lock();
        let _ = h.write_all(text.as_bytes());
        if !text.ends_with('\n') { let _ = h.write_all(b"\n"); }
        let _ = h.flush();
    }
}

// ─── Shared semantic helpers ──────────────────────────────────────────

/// Brace / paren / bracket balance check for multi-line input.
/// Ignores content inside string literals and line comments.
pub fn is_incomplete(s: &str) -> bool {
    let (mut p, mut b, mut k) = (0i32, 0i32, 0i32);
    let mut in_str = false;
    let mut q = ' ';
    for ch in s.chars() {
        if in_str { if ch == q { in_str = false; } continue; }
        match ch {
            '"' | '\'' => { in_str = true; q = ch; }
            '(' => p += 1, ')' => p -= 1,
            '{' => b += 1, '}' => b -= 1,
            '[' => k += 1, ']' => k -= 1,
            '#' => break,
            _ => {}
        }
    }
    p > 0 || b > 0 || k > 0
}

/// R's auto-print rule: a top-level expression's return value is
/// auto-printed UNLESS the expression is an assignment, control
/// flow, type definition, or a function whose side effect IS the
/// print.
pub fn is_silent(e: &Expr) -> bool {
    if matches!(
        e,
        Expr::Assign { .. } | Expr::TypeDef { .. } | Expr::MethodDef(_)
            | Expr::For { .. } | Expr::While { .. }
    ) {
        return true;
    }
    if let Expr::Call { func, .. } = e {
        if let Expr::Symbol(s) = func.as_ref() {
            return matches!(s.as_ref(),
                "print" | "cat" | "message" | "warning" | "writeLines" | "invisible" |
                "plot"  | "hist" | "boxplot" | "barplot" |
                "lines" | "points" | "abline" | "legend" |
                "library" | "detach" | "require" | "save.plot" | "dev.view" | "dev.off" |
                "install.packages" | "uninstall" | "set.seed" | "Sys.sleep");
        }
    }
    false
}

/// Detect a top-level call to `q()` / `quit()`. Returns true if the
/// given expression is a 0-argument call to either symbol.
pub fn is_quit_call(e: &Expr) -> bool {
    if let Expr::Call { func, args } = e {
        if args.is_empty() {
            if let Expr::Symbol(s) = func.as_ref() {
                return matches!(s.as_ref(), "q" | "quit");
            }
        }
    }
    false
}

/// Convenience: did any parsed top-level statement quit?
pub fn any_quit_call(stmts: &[Expr]) -> bool {
    stmts.iter().any(is_quit_call)
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_buffer_has_R2_prompt() {
        let b = ConsoleBuffer::new();
        assert_eq!(b.current_prompt(), "R2>");
        assert!(!b.in_continuation());
    }

    #[test]
    fn simple_one_liner_submits() {
        let mut b = ConsoleBuffer::new();
        match b.submit_line("2 + 2".into()) {
            SubmitAction::Submit(s) => assert_eq!(s, "2 + 2"),
            _ => panic!("should submit"),
        }
        assert_eq!(b.current_prompt(), "R2>");
    }

    #[test]
    fn open_brace_triggers_continuation() {
        let mut b = ConsoleBuffer::new();
        match b.submit_line("for (i in 1:3) {".into()) {
            SubmitAction::Continue => {}
            _ => panic!("should continue"),
        }
        assert_eq!(b.current_prompt(), "+");
        match b.submit_line("print(i)".into()) {
            SubmitAction::Continue => {}
            _ => panic!("still continuing"),
        }
        assert_eq!(b.current_prompt(), "+");
        match b.submit_line("}".into()) {
            SubmitAction::Submit(s) => assert!(s.starts_with("for (i in 1:3) {")),
            _ => panic!("should submit on closing brace"),
        }
        assert_eq!(b.current_prompt(), "R2>");
    }

    #[test]
    fn empty_submission_yields_empty_action() {
        let mut b = ConsoleBuffer::new();
        match b.submit_line("".into()) {
            SubmitAction::Empty => {}
            _ => panic!("should be Empty"),
        }
    }

    #[test]
    fn empty_during_continuation_cancels() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("for (i in 1:3) {".into());
        assert!(b.in_continuation());
        match b.submit_line("".into()) {
            SubmitAction::Empty => {}
            _ => panic!("should reset"),
        }
        assert!(!b.in_continuation());
        assert_eq!(b.current_prompt(), "R2>");
    }

    #[test]
    fn history_appends_completed_submissions_only() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("x <- 1".into());
        b.submit_line("y <- 2".into());
        b.submit_line("for (i in 1:3) {".into());     // continuation, no history yet
        b.submit_line("print(i)".into());
        b.submit_line("}".into());                     // now history+1
        assert_eq!(b.history().len(), 3);
        assert_eq!(b.history()[0], "x <- 1");
        assert_eq!(b.history()[1], "y <- 2");
        assert!(b.history()[2].starts_with("for (i in 1:3) {"));
    }

    #[test]
    fn history_dedupes_consecutive_duplicates() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("ls()".into());
        b.submit_line("ls()".into());
        b.submit_line("ls()".into());
        assert_eq!(b.history().len(), 1);
    }

    #[test]
    fn history_up_walks_backwards() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("a".into());
        b.submit_line("b".into());
        b.submit_line("c".into());
        assert_eq!(b.history_up(), Some("c".into()));
        assert_eq!(b.history_up(), Some("b".into()));
        assert_eq!(b.history_up(), Some("a".into()));
        // sticks at oldest
        assert_eq!(b.history_up(), Some("a".into()));
    }

    #[test]
    fn history_down_walks_forwards_and_clears() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("a".into());
        b.submit_line("b".into());
        b.history_up(); // b
        b.history_up(); // a
        assert_eq!(b.history_down(), Some("b".into()));
        // step past last → empty
        assert_eq!(b.history_down(), Some(String::new()));
    }

    #[test]
    fn transcript_contains_input_and_continuation_kinds() {
        let mut b = ConsoleBuffer::new();
        b.submit_line("for (i in 1:3) {".into());
        b.submit_line("}".into());
        let lines = b.transcript();
        assert_eq!(lines[0].kind, LineKind::Input);
        assert_eq!(lines[1].kind, LineKind::Continuation);
        assert!(lines[0].text.starts_with("R2> for"));
        assert!(lines[1].text.starts_with("+ }"));
    }

    #[test]
    fn is_incomplete_balanced_pairs() {
        assert!( is_incomplete("{"));
        assert!(!is_incomplete("{}"));
        assert!( is_incomplete("("));
        assert!(!is_incomplete("()"));
        assert!( is_incomplete("for (i in 1:3) {"));
        assert!(!is_incomplete("for (i in 1:3) {}"));
    }

    #[test]
    fn is_incomplete_ignores_strings_and_comments() {
        // The `{` inside the string and comment shouldn't count.
        assert!(!is_incomplete("'{'"));
        assert!(!is_incomplete("x <- 1  # {"));
        assert!( is_incomplete("x <- '{' + ("));   // still has open paren
    }
}
