//! `r2 --ide` — R2 as the engine of a separate IDE (docs/IDE_PROTOCOL.md).
//!
//! The IDE is its own program; it starts `r2 --ide` and talks to it over
//! stdin/stdout, one JSON object per line, the way RStudio drives an R
//! session. R2 does all statistics, formatting and plotting; the IDE only
//! displays. Requests carry an `"id"`; every reply echoes it.
//!
//! | op          | does                                                          |
//! |-------------|---------------------------------------------------------------|
//! | `hello`     | R2 version and protocol version                                |
//! | `eval`      | console input: output streams as it is printed, then a reply  |
//! | `value`     | evaluate quietly, the value as JSON (Sakha's statistics)      |
//! | `env`       | the Environment pane's rows                                    |
//! | `view`      | one page of the data viewer's formatted cells                  |
//! | `interrupt` | stop the running evaluation (no reply; read on its own thread) |
//! | `quit`      | reply, then exit                                               |
//!
//! `--json` (one expression per line, display text back) is unchanged.

mod env;
mod grid;
pub mod json;

use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::{ErrKind, RVal};

use crate::serve::{json_num, json_str};
use json::string;

/// Version of this protocol; `hello` reports it. Bump on breaking changes.
pub const PROTOCOL: u32 = 1;
/// Rows a `view` page holds when the request does not say.
const PAGE: usize = 200;

/// Where printed output goes while a request runs.
#[derive(Clone, Copy)]
enum Sink { Stream(u64), Discard }

/// Requests read so far, counted by the reading thread.
static RECEIVED: AtomicU64 = AtomicU64::new(0);
/// The count when the last `interrupt` came: it stops that request, even one
/// still waiting its turn, and not a later one.
static INTERRUPT_AFTER: AtomicU64 = AtomicU64::new(0);

/// Before request `n` runs: an interrupt sent after it stays set and stops it
/// at once; an older one, left over from a request already done, is dropped.
fn take_stale_interrupt(n: u64) {
    if INTERRUPT_AFTER.load(Ordering::SeqCst) != n { r2_types::clear_interrupt(); }
}

pub fn ide_main() -> i32 {
    // stdin on its own thread, so `interrupt` arrives while code runs
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if json_str(&line, "op").as_deref() == Some("interrupt") {
                INTERRUPT_AFTER.store(RECEIVED.load(Ordering::SeqCst), Ordering::SeqCst);
                r2_types::request_interrupt();
            } else {
                RECEIVED.fetch_add(1, Ordering::SeqCst);
                if tx.send(line).is_err() { break; }
            }
        }
    });
    // plots stay in memory as SVG and replies carry the new one; the IDE
    // shows them, so no browser viewer and no .svg files (as R2Gui does)
    std::env::set_var("R2_NO_AUTOVIEW", "1");
    r2_graphics::device::enable_autoview();
    let sink = Rc::new(RefCell::new(Sink::Discard));
    {
        let sink = sink.clone();
        r2_types::out::set_output_hook(Some(Box::new(move |text, is_err| {
            // the hook gets whole lines without their '\n'; put it back so
            // every stream text is printed output exactly
            if let Sink::Stream(id) = *sink.borrow() { stream(id, is_err, &format!("{text}\n")); }
        })));
    }
    let mut engine = Engine::new();
    let mut last_plot = r2_graphics::device::current_svg();
    let mut handled = 0u64;
    for line in rx {
        handled += 1;
        take_stale_interrupt(handled);
        if line.trim().is_empty() { continue; }
        let id = json_num(&line, "id").unwrap_or(0.0) as u64;
        let op = json_str(&line, "op").unwrap_or_default();
        let code = json_str(&line, "code").unwrap_or_default();
        let reply = match op.as_str() {
            "hello" => hello(),
            "eval" => {
                *sink.borrow_mut() = Sink::Stream(id);
                let r = eval(&mut engine, id, &code, &mut last_plot);
                *sink.borrow_mut() = Sink::Discard;
                r
            }
            "value" => match quiet(&mut engine, &code) {
                Ok(v) => format!("\"ok\":true,\"value\":{}", json::value(&v)),
                Err(e) => format!("\"ok\":false,\"error\":{}", string(&e)),
            },
            "env" => {
                let rows = env::env_rows(&engine.global_env.bindings.read().unwrap());
                let rows: Vec<String> = rows.iter().map(|r| format!(
                    "{{\"name\":{},\"group\":{},\"summary\":{},\"viewable\":{}}}",
                    string(&r.name), string(r.group.label()), string(&r.summary), r.viewable)).collect();
                format!("\"ok\":true,\"rows\":[{}]", rows.join(","))
            }
            "view" => view(&mut engine, &line, &code),
            "quit" => { send(id, "\"ok\":true"); return 0; }
            "" => "\"ok\":false,\"error\":\"missing \\\"op\\\"\"".into(),
            other => format!("\"ok\":false,\"error\":{}", string(&format!("unknown op '{other}'"))),
        };
        send(id, &reply);
    }
    0
}

/// What `hello` adds to protocol 1; an IDE uses one only when it is listed,
/// so R2 and the IDE can each be updated without the other.
const FEATURES: &[&str] = &["banner", "prompts"];

/// `hello`: the versions, then the console as r2-console defines it (the
/// banner every frontend shows unchanged, and the prompts), so an IDE's
/// console is R2's own and not a copy of it.
fn hello() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let list = |items: &[String]| items.iter().map(|s| string(s)).collect::<Vec<_>>().join(",");
    let features: Vec<String> = FEATURES.iter().map(|f| f.to_string()).collect();
    format!("\"ok\":true,\"r2\":{},\"protocol\":{PROTOCOL},\"banner\":[{}],\"prompts\":{{\"main\":{},\"continue\":{}}},\"features\":[{}]",
            string(version), list(&r2_console::banner_lines(version)),
            string(r2_console::PROMPT), string(r2_console::PROMPT_MORE), list(&features))
}

/// One reply line: `{"id":N,` + body + `}`.
fn send(id: u64, body: &str) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{{\"id\":{id},{body}}}");
    let _ = o.flush();
}

/// Printed output, as it happens: `{"id":N,"stream":"out"|"err","text":…}`.
fn stream(id: u64, is_err: bool, text: &str) {
    send(id, &format!("\"stream\":\"{}\",\"text\":{}", if is_err { "err" } else { "out" }, string(text)));
}

/// Console input. Unfinished input (an open bracket or string) is not run:
/// the reply says `"incomplete":true` and the IDE asks for more, as the
/// console's `+` prompt does.
fn eval(engine: &mut Engine, id: u64, code: &str, last_plot: &mut Option<String>) -> String {
    let stmts = match Parser::parse(code) {
        Ok(s) => s,
        // an open string is unfinished too, as in R: the next line continues it
        Err(e) if crate::incomplete(code) || e.to_string().contains("Eof") || e.to_string().contains("unterminated string") => return "\"ok\":true,\"incomplete\":true".into(),
        Err(e) => { stream(id, true, &format!("Error: {e}\n")); return "\"ok\":true,\"error\":true".into(); }
    };
    let (mut failed, mut interrupted) = (false, false);
    for stmt in &stmts {
        match engine.eval(stmt) {
            Ok(v) => if r2_console::should_autoprint(engine.visible, &v) && !engine.print_by_method(&v).unwrap_or(false) { stream(id, false, &format!("{v}\n")); },
            Err(e) if matches!(e.kind, ErrKind::Interrupt) => {
                stream(id, true, "interrupted\n");
                r2_types::clear_interrupt();
                interrupted = true;
                break;
            }
            Err(e) => { stream(id, true, &format!("{e}\n")); failed = true; break; }
        }
    }
    for w in engine.drain_warnings() { stream(id, true, &format!("{w}\n")); }
    let mut body = format!("\"ok\":true,\"error\":{failed},\"interrupted\":{interrupted}");
    // the whole picture, overlays included, not only what the last plot() drew
    let plot = r2_graphics::device::current_svg();
    if plot != *last_plot {
        if let Some(svg) = &plot { body.push_str(&format!(",\"plot\":{}", string(svg))); }
        *last_plot = plot;
    }
    body
}

/// Evaluate without printing anything; the last statement's value.
fn quiet(engine: &mut Engine, code: &str) -> Result<RVal, String> {
    let stmts = Parser::parse(code).map_err(|e| format!("parse error: {e}"))?;
    let mut last = RVal::Null;
    for stmt in &stmts {
        last = engine.eval(stmt).map_err(|e| e.msg.clone())?;
    }
    engine.drain_warnings();
    Ok(last)
}

/// One page of the data viewer: `code` names the value; `offset` and
/// `limit` pick the rows.
fn view(engine: &mut Engine, line: &str, code: &str) -> String {
    let v = match quiet(engine, code) { Ok(v) => v, Err(e) => return format!("\"ok\":false,\"error\":{}", string(&e)) };
    let Some(g) = grid::DataGrid::from_value(&v) else {
        return format!("\"ok\":false,\"error\":{}", string(&format!("cannot view a {}", v.type_name())));
    };
    let offset = json_num(line, "offset").unwrap_or(0.0).max(0.0) as usize;
    let limit = json_num(line, "limit").unwrap_or(PAGE as f64).max(0.0) as usize;
    let list = |f: &dyn Fn(usize) -> String| (0..g.ncol()).map(f).collect::<Vec<_>>().join(",");
    let rows: Vec<String> = (offset..(offset + limit).min(g.nrow())).map(|r| format!(
        "{{\"label\":{},\"cells\":[{}]}}", string(&g.row_label(r)), list(&|c| string(&g.cell(r, c))))).collect();
    format!("\"ok\":true,\"shape\":{},\"nrow\":{},\"ncol\":{},\"headers\":[{}],\"types\":[{}],\"widths\":[{}],\"right\":[{}],\"offset\":{offset},\"rows\":[{}]",
            string(&g.shape()), g.nrow(), g.ncol(),
            list(&|c| string(g.header(c))), list(&|c| string(g.types[c])),
            list(&|c| g.width(c).to_string()), list(&|c| g.right_aligned(c).to_string()), rows.join(","))
}
