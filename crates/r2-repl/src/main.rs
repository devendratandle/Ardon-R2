use r2_parser::Parser;
use r2_engine::Engine;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

mod serve;

fn main() {
    // Stack size set to 64MB via .cargo/config.toml linker flags
    // This avoids issues with _getch() FFI on spawned threads.
    //
    // Batch mode: `r2 <script.r2>` runs the script non-interactively, prints
    // results of non-silent top-level expressions to stdout, exits 1 on the
    // first eval error. Used for benchmarking and CI. Without arguments,
    // launches the interactive REPL.
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--serve" {
        let addr = args.get(2).cloned().unwrap_or_else(|| "127.0.0.1:8787".into());
        std::process::exit(serve::serve_main(&addr));
    }
    if args.len() >= 2 && args[1] == "--json" {
        std::process::exit(json_main());
    }
    if args.len() >= 2 && (args[1] == "--self-check" || args[1] == "--selfcheck") {
        std::process::exit(self_check());
    }
    if args.len() >= 2 && !args[1].starts_with('-') {
        std::process::exit(run_script(&args[1]));
    }
    repl_main();
}

// ── Agent mode: `r2 --json` ─────────────────────────────────────────────
// Newline-delimited JSON protocol so agent frameworks and other programs
// can drive the engine without screen-scraping a REPL. One request per
// line on stdin: {"expr": "<R code>"}. One response per line on stdout:
//   {"ok":true,"class":"numeric","length":3,"result":"[1] 1 2 3","output":"…"}
//   {"ok":false,"error":"object 'x' not found"}
// `result` is the value's display form; `output` is everything the code
// printed (cat/print). State persists across lines (one engine session).
fn json_main() -> i32 {
    use std::io::{BufRead, Write};

    // Minimal JSON string escaping for output (we emit only strings/nums).
    fn esc(s: &str) -> String {
        let mut o = String::with_capacity(s.len() + 8);
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
                '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"),
                '\t' => o.push_str("\\t"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o
    }
    // Extract the "expr" string from a {"expr": "..."} request line —
    // a tiny scanner (find the key, decode the JSON string after it), so
    // the protocol needs no serde dependency.
    fn parse_expr(line: &str) -> Option<String> {
        let key = line.find("\"expr\"")?;
        let colon = line[key + 6..].find(':')? + key + 6;
        let rest = line[colon + 1..].trim_start();
        let mut chars = rest.strip_prefix('"')?.chars();
        let mut out = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(out),
                '\\' => match chars.next()? {
                    'n' => out.push('\n'), 't' => out.push('\t'), 'r' => out.push('\r'),
                    '"' => out.push('"'), '\\' => out.push('\\'), '/' => out.push('/'),
                    'u' => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) { out.push(ch); }
                    }
                    other => out.push(other),
                },
                c => out.push(c),
            }
        }
        None
    }

    // Capture routed output (cat/print/summary…) per request.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    {
        let cap = captured.clone();
        r2_types::out::set_output_hook(Some(Box::new(move |s, _is_err| {
            cap.lock().unwrap().push_str(s);
        })));
    }

    let mut engine = Engine::new();
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        let mut o = stdout.lock();
        let Some(expr_src) = parse_expr(&line) else {
            let _ = writeln!(o, "{{\"ok\":false,\"error\":\"malformed request: expected {{\\\"expr\\\": \\\"...\\\"}}\"}}");
            let _ = o.flush();
            continue;
        };
        captured.lock().unwrap().clear();
        let reply = match Parser::parse(&expr_src) {
            Err(e) => format!("{{\"ok\":false,\"error\":\"parse error: {}\"}}", esc(&e.to_string())),
            Ok(stmts) => {
                let mut last = r2_types::RVal::Null;
                let mut err: Option<String> = None;
                for st in &stmts {
                    match engine.eval(st) {
                        Ok(v) => last = v,
                        Err(e) => { err = Some(e.msg.clone()); break; }
                    }
                }
                match err {
                    Some(m) => format!("{{\"ok\":false,\"error\":\"{}\",\"output\":\"{}\"}}",
                                       esc(&m), esc(&captured.lock().unwrap())),
                    None => {
                        let class = last.type_name();
                        let len = r2_types::rval_length(&last);
                        let display = format!("{}", last);
                        format!("{{\"ok\":true,\"class\":\"{}\",\"length\":{},\"result\":\"{}\",\"output\":\"{}\"}}",
                                esc(class), len, esc(&display), esc(&captured.lock().unwrap()))
                    }
                }
            }
        };
        let _ = writeln!(o, "{}", reply);
        let _ = o.flush();
    }
    0
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
                if r2_console::should_autoprint(stmt, &val) {
                    println!("{}", val);
                }
            }
            Err(err) => { eprintln!("{}", err); return 1; }
        }
    }
    for w in engine.drain_warnings() { eprintln!("{}", w); }
    0
}

// ── Trust command: `r2 --self-check` ────────────────────────────────────
// A self-contained accuracy battery. It proves — on the USER'S OWN
// hardware, with no R install and no network — that this build computes
// the statistical surface correctly. Every expected value is an
// authoritative mathematical constant (not "whatever R2 happened to
// print"), so a PASS means the engine agrees with mathematics to the
// IEEE-754 floor on this CPU. Exit 0 iff every check passes, so it doubles
// as a CI / deployment gate ("did the binary survive the install?").
fn self_check() -> i32 {
    // (label, R2 expression, expected value, relative tolerance)
    // Expected==0.0 is compared with ABSOLUTE tolerance (rel is undefined).
    // Tolerances are the tight IEEE band (1e-12) except where an algorithm's
    // own convergence floor is looser (noted inline).
    let checks: &[(&str, &str, f64, f64)] = &[
        // ── descriptive statistics ──
        ("mean(2,4,6,8)",        "mean(c(2,4,6,8))",                 5.0,                   1e-12),
        ("var(2,4,6,8)",         "var(c(2,4,6,8))",                  6.666666666666667,     1e-12),
        ("sd(2,4,6,8)",          "sd(c(2,4,6,8))",                   2.581988897471611,     1e-12),
        ("median(1,2,3,4)",      "median(c(1,2,3,4))",              2.5,                   1e-12),
        ("quantile p50 t7",      "quantile(1:100, 0.5)",           50.5,                   1e-12),
        // ── elementary functions (bit-exact math) ──
        ("sqrt(2)",              "sqrt(2)",                         1.4142135623730951,    1e-15),
        ("exp(1)",               "exp(1)",                          2.718281828459045,     1e-15),
        ("log(2)",               "log(2)",                          0.6931471805599453,    1e-15),
        // ── distributions (the full-precision pnorm/qnorm surface) ──
        ("dnorm(0)",             "dnorm(0)",                        0.3989422804014327,    1e-14),
        ("pnorm(0)",             "pnorm(0)",                        0.5,                   1e-15),
        ("pnorm(1.96)",          "pnorm(1.96)",                     0.9750021048517795,    1e-12),
        ("pnorm(-8) far tail",   "pnorm(-8)",                       6.220960574271782e-16, 1e-9),
        ("qnorm(0.975)",         "qnorm(0.975)",                    1.959963984540054,     1e-12),
        ("qt(0.975, df=10)",     "qt(0.975, 10)",                   2.228138851986273,     1e-9),
        ("pchisq(3.841, df=1)",  "pchisq(3.841458820694124, 1)",   0.95,                   1e-9),
        // ── combinatorics ──
        ("factorial(10)",        "factorial(10)",             3628800.0,                   1e-12),
        ("choose(10,3)",         "choose(10,3)",                  120.0,                   1e-12),
        // ── linear algebra + regression ──
        ("matrix %*% I",         "(matrix(c(1,2,3,4),2) %*% matrix(c(1,0,0,1),2))[2,2]", 4.0, 1e-12),
        ("solve(A) roundtrip",   "(solve(matrix(c(2,0,0,4),2)) %*% matrix(c(2,0,0,4),2))[1,1]", 1.0, 1e-12),
        ("cor(x, 2x) = 1",       "cor(c(1,2,3,4), c(2,4,6,8))",     1.0,                   1e-12),
        ("lm slope (y=2x)",      "coef(lm(c(2,4,6,8) ~ c(1,2,3,4)))[[2]]", 2.0,            1e-9),
    ];

    println!("Ardon-R2 self-check");
    println!("  version : {}", env!("CARGO_PKG_VERSION"));
    println!("  target  : {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("  build   : {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!();

    // Route engine output (warnings etc.) away from the report.
    r2_types::out::set_output_hook(Some(Box::new(|_s, _e| {})));

    let mut engine = Engine::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    println!("  {:<22} {:>24} {:>24}   result", "check", "expected", "got");
    println!("  {}", "-".repeat(78));
    for (label, expr, expected, tol) in checks {
        let got = eval_scalar(&mut engine, expr);
        let (ok, got_str) = match got {
            Ok(v) => {
                let err = if *expected == 0.0 { v.abs() } else { ((v - expected) / expected).abs() };
                (err <= *tol, format!("{:.17e}", v))
            }
            Err(e) => (false, format!("ERROR: {}", e)),
        };
        if ok { passed += 1; } else { failed += 1; }
        println!("  {:<22} {:>24.17e} {:>24}   {}",
                 label, expected, got_str, if ok { "ok" } else { "FAIL" });
    }
    println!("  {}", "-".repeat(78));
    println!();
    println!("  {} passed, {} failed, {} total", passed, failed, checks.len());
    if failed == 0 {
        println!("  PASS — this build matches mathematics to the IEEE-754 floor on this machine.");
        0
    } else {
        println!("  FAIL — {} check(s) diverged. This binary should not be trusted for production.", failed);
        1
    }
}

// Evaluate a single R2 expression to a scalar f64 (last statement's value).
fn eval_scalar(engine: &mut Engine, src: &str) -> Result<f64, String> {
    let stmts = Parser::parse(src).map_err(|e| e.to_string())?;
    let mut last = r2_types::RVal::Null;
    for st in &stmts {
        last = engine.eval(st).map_err(|e| e.msg.clone())?;
    }
    match last.scalar_f64().map_err(|e| e.msg.clone())? {
        Some(v) => Ok(v),
        None => Err("NA".into()),
    }
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

    // Canonical banner (shared with the GUI via r2-console) + the CLI's
    // one host-specific hint line.
    println!();
    for line in r2_console::banner_lines(env!("CARGO_PKG_VERSION")) {
        println!("{}", line);
    }
    println!("Type q() to quit.\n");

    // Interactive session: allow dev.view() to launch the browser plot
    // viewer (its daemon HTTP server stays alive with the REPL). Scripts
    // never reach this path, so batch runs keep dev.view() a no-op.
    r2_graphics::device::enable_autoview();

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
                            if r2_console::should_autoprint(stmt, &val) {
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
            Err(e) => {
                // Continue on the next line when the input is unfinished:
                // either unbalanced brackets, OR the parser ran out of input
                // expecting more (e.g. `repeat`, `if (x)`, `while (x)`,
                // `function(...)` with the body on the following line).
                if incomplete(&buffer) || e.msg.contains("Eof") {
                    continuation = true;
                } else {
                    // Rich format with source-line + caret underline.
                    eprintln!("{}", e.display_with_source(&buffer));
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

// The auto-print rule (silent set + NULL-invisibility) is unified in
// `r2_console::should_autoprint` so the CLI and GUI consoles behave
// identically — see the call sites above. (The old local `is_silent`
// copy was removed; it had drifted from the GUI's.)

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
            13 | 10 => { println!(); return Some(line); }        // Enter (CR or LF)
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
