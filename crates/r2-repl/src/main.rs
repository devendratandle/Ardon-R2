use r2_parser::Parser;
use r2_engine::Engine;
use r2_types::Expr;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    // Stack size set to 64MB via .cargo/config.toml linker flags
    // This avoids issues with _getch() FFI on spawned threads.
    //
    // Batch mode: `r2 <script.r2>` runs the script non-interactively, prints
    // results of non-silent top-level expressions to stdout, exits 1 on the
    // first eval error. Used for benchmarking and CI. Without arguments,
    // launches the interactive REPL.
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && !args[1].starts_with('-') {
        std::process::exit(run_script(&args[1]));
    }
    repl_main();
}

fn run_script(path: &str) -> i32 {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("r2: cannot read {}: {}", path, e); return 1; }
    };
    let exprs = match Parser::parse(&source) {
        Ok(e) => e,
        Err(e) => { eprintln!("{}", e.display_with_source(&source)); return 1; }
    };
    let mut engine = Engine::new();
    for stmt in &exprs {
        match engine.eval(stmt) {
            Ok(val) => {
                if !is_silent(stmt) && !matches!(&val, r2_types::RVal::Null) {
                    println!("{}", val);
                }
            }
            Err(err) => { eprintln!("{}", err); return 1; }
        }
    }
    for w in engine.drain_warnings() { eprintln!("{}", w); }
    0
}

fn repl_main() {
    // ── Default working directory ────────────────────────────────────
    // Following R's convention: launch in the user's Documents folder
    // (or $HOME on Unix) rather than wherever the .exe lives. Without
    // this, users who launched via the Start Menu would see `getwd()`
    // return `C:\Users\…\AppData\Local\Programs\Ardon-R2` — confusing
    // and not writable on Program Files installs.
    //
    // Resolution order:
    //   1. `R2_HOME` env var (explicit user override).
    //   2. `%USERPROFILE%\Documents` on Windows.
    //   3. `$HOME` on Unix.
    //   4. Fall back to current cwd (no change) if none of the above.
    //
    // We only change cwd in *interactive* mode. Scripts run via
    // `r2 script.r2` keep their invocation cwd so relative paths in
    // user scripts work as expected.
    if let Some(home) = pick_user_home() {
        let _ = std::env::set_current_dir(&home);
    }

    // Phase R.M.2 — install Ctrl+C handler. SIGINT sets the engine's
    // global interrupt flag; the eval loop polls it at every Expr and
    // raises ErrKind::Interrupt, which we catch below and treat as a
    // "return to prompt" event instead of letting it kill the process.
    // The handler is idempotent — set_handler errors only if a handler
    // is already installed, which we silently ignore for safety.
    let _ = ctrlc::set_handler(|| {
        r2_types::request_interrupt();
        // Print on a new line so the next prompt is clean.
        eprintln!();
    });

    println!("\nArdon-R2 — Statistical Computing, Reimagined");
    println!("Version 0.1.1 (2026) | Inspired by R. Built on Rust.");
    println!("Created by Devendra Tandale | An AI-Assisted Project");
    println!("Assignment: both <- and = work. Mode: strict.");
    println!("Type q() to quit.\n");

    let mut engine = Engine::new();
    // Interactive session only: opt in to the browser plot viewer so
    // `plot()` opens a live view (RGui-style). Script mode (run_script)
    // and the test suite leave it off, so they never spawn a browser.
    engine.enable_plot_autoview();
    let mut history: Vec<String> = Vec::new();
    let mut buffer = String::new();
    let mut continuation = false;

    loop {
        let prompt = if continuation { "R2+ " } else { "R2> " };
        let line = match read_line_with_history(prompt, &history) {
            Some(l) => l,
            None => break,
        };

        let trimmed = line.trim();
        if !continuation && (trimmed == "q()" || trimmed == "quit()") {
            // Phase R.M.3 — R-style workspace save prompt.
            // y → save all globals to session.r2s, then exit.
            // n → exit without saving (default if user just hits Enter).
            // c → cancel quit, return to prompt with state intact.
            print!("Save workspace image? [y/n/c]: ");
            io::stdout().flush().ok();
            let mut answer = String::new();
            io::stdin().lock().read_line(&mut answer).ok();
            let a = answer.trim().to_lowercase();
            match a.as_str() {
                "y" | "yes" => {
                    // Second prompt: let the user pick a filename, or
                    // accept the R-style default by hitting Enter.
                    print!("Filename [session.r2s]: ");
                    io::stdout().flush().ok();
                    let mut name = String::new();
                    io::stdin().lock().read_line(&mut name).ok();
                    let filename = {
                        let t = name.trim();
                        if t.is_empty() { "session.r2s".to_string() } else { t.to_string() }
                    };

                    // Reuse the existing save() builtin via a synthetic
                    // parse → eval call. One serialization code path
                    // covers explicit save("path") and the q() prompt.
                    let saved = match Parser::parse(&format!("save(\"{}\")", filename.replace('\\', "\\\\").replace('"', "\\\""))) {
                        Ok(stmts) => {
                            let mut ok = true;
                            for s in &stmts {
                                if let Err(e) = engine.eval(s) {
                                    eprintln!("Save failed: {}", e);
                                    ok = false;
                                }
                            }
                            ok
                        }
                        Err(_) => { eprintln!("Save failed: internal parser error"); false }
                    };

                    if saved {
                        // Print the absolute path so the user knows where
                        // their workspace went — equivalent to R's printout.
                        let abs = std::fs::canonicalize(&filename)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| filename.clone());
                        println!("Workspace saved to: {}", abs);
                    }
                    println!("Goodbye.");
                    break;
                }
                "c" | "cancel" => {
                    println!("(quit cancelled — back to prompt)");
                    buffer.clear();
                    continue;
                }
                _ => {
                    // "n", "no", empty, or anything else → exit without saving.
                    println!("Goodbye.");
                    break;
                }
            }
        }

        // R-style help: ?topic or ??topic → help("topic")
        let line = if !continuation && trimmed.starts_with("??") {
            let topic = trimmed[2..].trim();
            format!("help(\"{}\")", topic)
        } else if !continuation && trimmed.starts_with('?') && trimmed.len() > 1 {
            let topic = trimmed[1..].trim();
            format!("help(\"{}\")", topic)
        } else {
            line
        };
        let trimmed = line.trim();

        if !trimmed.is_empty() {
            if history.last().map(|s| s.as_str()) != Some(trimmed) {
                history.push(trimmed.to_string());
            }
        }

        buffer.push_str(&line);
        buffer.push('\n');

        match Parser::parse(&buffer) {
            Ok(stmts) => {
                continuation = false;
                // Clear any stale interrupt flag set while the user was at
                // the idle prompt (Esc/Ctrl+C at the prompt should not
                // interrupt the very next command).
                r2_types::clear_interrupt();
                for stmt in &stmts {
                    // Phase R.M.2 — start the Esc-polling thread for the
                    // duration of this single statement's evaluation.
                    // Stopped after the eval call completes regardless of
                    // success or interrupt.
                    let poller = EscPoller::start();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        engine.eval(stmt)
                    }));
                    poller.stop();
                    match result {
                        Ok(Ok(val)) => {
                            if !is_silent(stmt) && !matches!(&val, r2_types::RVal::Null) {
                                println!("{}", val);
                            }
                        }
                        // Phase R.M.2 — Ctrl+C caught here: print a brief
                        // notice, clear the global flag, break out of the
                        // current statement batch and loop back to the prompt.
                        // The engine state is left intact (variables defined
                        // before the interrupt are still bound).
                        Ok(Err(e)) if e.kind == r2_types::ErrKind::Interrupt => {
                            eprintln!("interrupted — returning to prompt");
                            r2_types::clear_interrupt();
                            break;
                        }
                        Ok(Err(e)) => eprintln!("{}", e),
                        Err(_) => eprintln!("Error: internal error (please report this bug)"),
                    }
                }
                for w in engine.drain_warnings() { eprintln!("{}", w); }
                buffer.clear();
            }
            Err(_) => {
                if incomplete(&buffer) {
                    continuation = true;
                } else {
                    if let Err(e) = Parser::parse(&buffer) {
                        // Rich format with source-line + caret underline.
                        eprintln!("{}", e.display_with_source(&buffer));
                    }
                    buffer.clear();
                    continuation = false;
                }
            }
        }
    }
}

// Locate a writable default working directory for the interactive REPL.
// Returns None if no candidate exists.
//
// Critical Windows nuance: when OneDrive is configured to back up the
// Documents library, Windows Explorer's "Documents" shortcut points at
// `%USERPROFILE%\OneDrive\Documents\`, NOT at `%USERPROFILE%\Documents\`.
// Both folders physically exist as separate trees. If we save plots to
// the literal `%USERPROFILE%\Documents\`, the user clicks "Documents"
// in Explorer, doesn't find their plot, and reasonably thinks R2 is
// broken. So we prefer the OneDrive path when it exists.
fn pick_user_home() -> Option<std::path::PathBuf> {
    // 1. Explicit user override always wins.
    if let Ok(custom) = std::env::var("R2_HOME") {
        let p = std::path::PathBuf::from(custom);
        if p.is_dir() { return Some(p); }
    }
    // 2. OneDrive-redirected Documents — what Explorer shows.
    //    OneDrive sets %OneDrive% when its client is running; we also
    //    look at the canonical %USERPROFILE%\OneDrive\Documents path
    //    in case the env var isn't propagated.
    if let Ok(od) = std::env::var("OneDrive") {
        let p = std::path::PathBuf::from(&od).join("Documents");
        if p.is_dir() { return Some(p); }
    }
    if let Ok(user) = std::env::var("USERPROFILE") {
        let od = std::path::PathBuf::from(&user).join("OneDrive").join("Documents");
        if od.is_dir() { return Some(od); }
        // 3. Plain Windows Documents.
        let docs = std::path::PathBuf::from(user).join("Documents");
        if docs.is_dir() { return Some(docs); }
    }
    // 4. Unix: $HOME/Documents if it exists, else $HOME.
    if let Ok(home) = std::env::var("HOME") {
        let docs = std::path::PathBuf::from(&home).join("Documents");
        if docs.is_dir() { return Some(docs); }
        let h = std::path::PathBuf::from(home);
        if h.is_dir() { return Some(h); }
    }
    None
}

fn is_silent(e: &Expr) -> bool {
    if matches!(e, Expr::Assign{..}|Expr::TypeDef{..}|Expr::MethodDef(_)|Expr::For{..}|Expr::While{..}) {
        return true;
    }
    // Calls to functions whose side effect IS the print (and whose return
    // value is uninteresting) shouldn't trigger auto-print of the return.
    // R does this via invisible(); we mark a small set by name.
    if let Expr::Call { func, .. } = e {
        if let Expr::Symbol(s) = func.as_ref() {
            return matches!(s.as_ref(),
                "print" | "cat" | "message" | "warning" | "writeLines" | "invisible" |
                "clear" | "cls" | "clr" |
                // Hypothesis tests / ANOVA print their own formatted output
                // as a side effect and return an htest/model object whose
                // Display is just "<… model>" — don't auto-print that.
                "t.test" | "chisq.test" | "wilcox.test" | "var.test" | "ks.test" |
                "fisher.test" | "cor.test" | "prop.test" | "binom.test" |
                "oneway.test" | "kruskal.test" | "shapiro.test" | "bartlett.test" |
                "poisson.test" | "anova" | "aov" | "manova" | "hotelling.test" |
                // Inspectors that print their report and return NULL.
                "summary" | "str");
        }
    }
    false
}

fn incomplete(s: &str) -> bool {
    let (mut p, mut b, mut k) = (0i32, 0i32, 0i32);
    let mut in_str = false; let mut in_comment = false; let mut q = ' ';
    for ch in s.chars() {
        // '#' comment runs to end of line only — must not abort the scan
        // (else an inline comment in a multi-line c(...) hides the ')').
        if in_comment { if ch == '\n' { in_comment = false; } continue; }
        if in_str { if ch == q { in_str = false; } continue; }
        match ch {
            '"'|'\'' => { in_str = true; q = ch; }
            '#' => in_comment = true,
            '(' => p+=1, ')' => p-=1, '{' => b+=1, '}' => b-=1, '[' => k+=1, ']' => k-=1,
            _ => {}
        }
    }
    p > 0 || b > 0 || k > 0
}

// ═══════════════════════════════════════════════════════════════════════
// Windows line editor with arrow key history
// ═══════════════════════════════════════════════════════════════════════

#[cfg(windows)]
fn read_line_with_history(prompt: &str, history: &[String]) -> Option<String> {
    print!("{}", prompt);
    io::stdout().flush().unwrap();

    let mut line = String::new();
    let mut cursor = 0usize;
    let mut hist_idx: usize = history.len();
    let mut saved_line = String::new();

    loop {
        let ch = win_getch();
        match ch {
            13 => { println!(); return Some(line); }           // Enter
            3 => { println!("^C"); return Some(String::new()); } // Ctrl+C
            4 if line.is_empty() => { println!(); return None; } // Ctrl+D
            8 | 127 => {                                        // Backspace
                if cursor > 0 {
                    // Move back to previous char boundary
                    cursor -= 1;
                    while cursor > 0 && !line.is_char_boundary(cursor) { cursor -= 1; }
                    if line.is_char_boundary(cursor) { line.remove(cursor); }
                    redraw_line(prompt, &line, cursor);
                }
            }
            0 | 224 => {                                        // Special key prefix
                let key = win_getch();
                match key {
                    72 => {                                     // Up arrow
                        if !history.is_empty() && hist_idx > 0 {
                            if hist_idx == history.len() { saved_line = line.clone(); }
                            hist_idx -= 1;
                            line = history[hist_idx].clone();
                            cursor = line.len();
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    80 => {                                     // Down arrow
                        if hist_idx < history.len() {
                            hist_idx += 1;
                            line = if hist_idx == history.len() { saved_line.clone() } else { history[hist_idx].clone() };
                            cursor = line.len();
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    75 => {                                     // Left arrow
                        if cursor > 0 {
                            // Move back one character (could be multi-byte)
                            cursor -= 1;
                            while cursor > 0 && !line.is_char_boundary(cursor) { cursor -= 1; }
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    77 => {                                     // Right arrow
                        if cursor < line.len() {
                            cursor += 1;
                            while cursor < line.len() && !line.is_char_boundary(cursor) { cursor += 1; }
                            redraw_line(prompt, &line, cursor);
                        }
                    }
                    71 => { cursor = 0; redraw_line(prompt, &line, cursor); }       // Home
                    79 => { cursor = line.len(); redraw_line(prompt, &line, cursor); } // End
                    83 => {                                     // Delete
                        if cursor < line.len() && line.is_char_boundary(cursor) { line.remove(cursor); redraw_line(prompt, &line, cursor); }
                    }
                    _ => {}
                }
            }
            ch if ch >= 32 => {                                 // Printable
                let c = ch as u8 as char;
                if cursor <= line.len() && line.is_char_boundary(cursor) {
                    line.insert(cursor, c);
                    cursor += c.len_utf8();
                } else {
                    line.push(c);
                    cursor = line.len();
                }
                if cursor == line.len() { print!("{}", c); io::stdout().flush().unwrap(); }
                else { redraw_line(prompt, &line, cursor); }
            }
            _ => {}
        }
    }
}

#[cfg(windows)]
extern "C" {
    fn _getch() -> i32;
    fn _kbhit() -> i32;
}

#[cfg(windows)]
fn win_getch() -> i32 { unsafe { _getch() } }

#[cfg(windows)]
fn win_kbhit() -> bool { unsafe { _kbhit() != 0 } }

#[cfg(not(windows))]
fn win_kbhit() -> bool {
    // Unix: rely on Ctrl+C only. A proper poll-for-Esc on Unix needs
    // termios raw mode toggling, which interferes with the line editor
    // above. Acceptable: r/rust + r/rstats users on Linux/Mac are
    // comfortable with Ctrl+C, and ctrlc::set_handler covers them.
    false
}

// ─────────────────────────────────────────────────────────────────────
// Phase R.M.2 — Esc-as-interrupt polling thread.
//
// Spawned just before each user-driven evaluation, joined after. Polls
// the keyboard non-blocking every 50 ms; if it sees byte 27 (Esc),
// it sets the engine's global INTERRUPT flag, which the eval loop
// observes at the next Expr boundary and unwinds with ErrKind::Interrupt.
//
// The polling thread shuts itself down when the `active` flag flips to
// false (signaled by the REPL after eval completes). On Windows, _kbhit
// + _getch are non-blocking and OS-level; on Unix we currently fall back
// to Ctrl+C only (see comment above on termios).
// ─────────────────────────────────────────────────────────────────────

struct EscPoller {
    active: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl EscPoller {
    fn start() -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let active_clone = active.clone();
        let handle = std::thread::Builder::new()
            .name("r2-esc-poll".into())
            .spawn(move || {
                while active_clone.load(Ordering::Relaxed) {
                    if win_kbhit() {
                        #[cfg(windows)]
                        {
                            let ch = win_getch();
                            if ch == 27 {
                                // Escape pressed — raise interrupt and exit.
                                r2_types::request_interrupt();
                                break;
                            }
                            // Other keystrokes during eval are discarded
                            // (acceptable tradeoff: typing-ahead during a
                            // long compute is rare; Ctrl+C remains as
                            // signal-level fallback).
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            })
            .ok();
        EscPoller { active, handle }
    }

    fn stop(mut self) {
        self.active.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(windows)]
fn redraw_line(prompt: &str, line: &str, cursor: usize) {
    print!("\r{}{}\x1b[K", prompt, line);
    let back = line.len() - cursor;
    if back > 0 { print!("\x1b[{}D", back); }
    io::stdout().flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Unix fallback
// ═══════════════════════════════════════════════════════════════════════

#[cfg(not(windows))]
fn read_line_with_history(prompt: &str, _history: &[String]) -> Option<String> {
    use std::io::BufRead;
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line.trim_end().to_string()),
        Err(_) => None,
    }
}
