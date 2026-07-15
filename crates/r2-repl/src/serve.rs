//! `r2 --serve [addr]` — the AI-server foundation (Pillar 3).
//!
//! A TCP server speaking the same newline-delimited JSON protocol as
//! `r2 --json`, extended with sessions and a security model designed for
//! LLM-agent callers (every request is untrusted generated code):
//!
//!   auth      every request carries "token"; compared constant-time
//!             against R2_TOKEN (required — the server refuses to start
//!             without it).
//!   sessions  {"token":T,"op":"session.new","caps":["fs_read",…]} → id.
//!             Sessions get an isolated engine with a DEFAULT-DENY
//!             capability policy; the operator grants caps explicitly.
//!   eval      {"token":T,"op":"eval","session":id,"expr":"…",
//!              "timeout_ms":5000} — wall-clock watchdog raises the
//!             engine interrupt; output captured and size-capped.
//!   admin     {"op":"sessions"} lists; {"op":"session.close","session":id}.
//!   audit     every request is appended to the audit log (path from
//!             R2_AUDIT_LOG, default "r2-serve-audit.log"): timestamp,
//!             peer, session, op, ok/error. Append-only file.
//!
//! v1 handles requests SERIALLY (one at a time): the engine interrupt
//! flag is process-global, so a per-eval watchdog is race-free only with
//! serial execution. Parallel sessions are the documented next step
//! (per-engine interrupt plumbing).

use r2_engine::{Engine, Policy};
use r2_parser::Parser;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

// ── tiny JSON helpers (same conventions as json_main) ──────────────────

pub(crate) fn esc(s: &str) -> String {
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

/// Decode the JSON string value following `"key":` in `line`.
pub(crate) fn json_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let k = line.find(&pat)?;
    let colon = line[k + pat.len()..].find(':')? + k + pat.len();
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

/// Decode a JSON number value following `"key":`.
fn json_num(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\"", key);
    let k = line.find(&pat)?;
    let colon = line[k + pat.len()..].find(':')? + k + pat.len();
    let rest = line[colon + 1..].trim_start();
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+')).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Constant-time byte comparison — token checks must not leak length
/// prefixes through timing.
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = (a.len() ^ b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        let x = *a.get(i % a.len().max(1)).unwrap_or(&0);
        let y = *b.get(i % b.len().max(1)).unwrap_or(&0);
        diff |= x ^ y;
    }
    diff == 0
}

struct Session {
    engine: Engine,
    caps: String, // display form for sessions() listing
}

const MAX_LINE: usize = 4 * 1024 * 1024;     // request size cap (4 MB)
const MAX_OUTPUT: usize = 1024 * 1024;       // captured-output cap (1 MB)
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

pub fn serve_main(addr: &str) -> i32 {
    // Refuse to start without a token — an unauthenticated eval server
    // is remote code execution for anyone who can reach the port.
    let Ok(token) = std::env::var("R2_TOKEN") else {
        eprintln!("r2 --serve: set R2_TOKEN (bearer token required on every request).");
        eprintln!("            refusing to start an unauthenticated eval server.");
        return 2;
    };
    if token.len() < 16 {
        eprintln!("r2 --serve: R2_TOKEN must be at least 16 characters.");
        return 2;
    }
    let audit_path = std::env::var("R2_AUDIT_LOG").unwrap_or_else(|_| "r2-serve-audit.log".into());
    let mut audit = std::fs::OpenOptions::new().create(true).append(true).open(&audit_path)
        .map_err(|e| eprintln!("r2 --serve: cannot open audit log '{}': {}", audit_path, e)).ok();

    let listener = match TcpListener::bind(addr) {
        Ok(l) => l,
        Err(e) => { eprintln!("r2 --serve: cannot bind {}: {}", addr, e); return 2; }
    };
    eprintln!("r2 --serve listening on {} (audit: {})", addr, audit_path);

    // Capture routed output per request.
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    {
        let cap = captured.clone();
        r2_types::out::set_output_hook(Some(Box::new(move |s, _| {
            let mut c = cap.lock().unwrap();
            if c.len() < MAX_OUTPUT { c.push_str(s); }
        })));
    }

    let mut sessions: HashMap<String, Session> = HashMap::new();
    let mut next_id: u64 = 1;

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let peer = stream.peer_addr().map(|p| p.to_string()).unwrap_or_default();
        let mut reader = BufReader::new(match stream.try_clone() { Ok(s) => s, Err(_) => continue });
        let mut w = stream;
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,           // connection closed
                Ok(n) if n > MAX_LINE => {
                    let _ = writeln!(w, "{{\"ok\":false,\"error\":\"request too large\"}}");
                    break;
                }
                Ok(_) => {}
            }
            if line.trim().is_empty() { continue; }

            // ── auth first, before touching anything else ──────────────
            let authed = json_str(&line, "token").map(|t| ct_eq(&t, &token)).unwrap_or(false);
            let op = json_str(&line, "op").unwrap_or_else(|| "eval".into());
            let sid = json_str(&line, "session").unwrap_or_default();
            if !authed {
                let _ = writeln!(w, "{{\"ok\":false,\"error\":\"unauthorized\"}}");
                let _ = w.flush();
                if let Some(a) = audit.as_mut() {
                    let _ = writeln!(a, "{}\t{}\t-\t{}\tDENIED-auth", now(), peer, esc(&op));
                }
                continue;
            }

            let reply = match op.as_str() {
                "session.new" => {
                    let id = format!("s{}", next_id); next_id += 1;
                    let caps_raw = json_str(&line, "caps").unwrap_or_default();
                    // caps may also arrive as a JSON array; accept both by
                    // substring scan of the raw line's caps region.
                    let grant = |name: &str| caps_raw.contains(name) || line.contains(&format!("\"{}\"", name));
                    let mut engine = Engine::new();
                    engine.policy = Policy {
                        fs_read:    grant("fs_read"),
                        fs_write:   grant("fs_write"),
                        env_access: grant("env_access"),
                        install:    grant("install"),
                    };
                    let caps = format!("fs_read={},fs_write={},env_access={},install={}",
                        engine.policy.fs_read, engine.policy.fs_write,
                        engine.policy.env_access, engine.policy.install);
                    sessions.insert(id.clone(), Session { engine, caps: caps.clone() });
                    format!("{{\"ok\":true,\"session\":\"{}\",\"caps\":\"{}\"}}", id, caps)
                }
                "session.close" => {
                    match sessions.remove(&sid) {
                        Some(_) => format!("{{\"ok\":true,\"closed\":\"{}\"}}", sid),
                        None => format!("{{\"ok\":false,\"error\":\"no such session '{}'\"}}", esc(&sid)),
                    }
                }
                "sessions" => {
                    let list: Vec<String> = sessions.iter()
                        .map(|(id, s)| format!("{{\"id\":\"{}\",\"caps\":\"{}\"}}", id, s.caps))
                        .collect();
                    format!("{{\"ok\":true,\"sessions\":[{}]}}", list.join(","))
                }
                "eval" => {
                    match sessions.get_mut(&sid) {
                        None => format!("{{\"ok\":false,\"error\":\"no such session '{}' (create one with op session.new)\"}}", esc(&sid)),
                        Some(sess) => {
                            let Some(expr_src) = json_str(&line, "expr") else {
                                let _ = writeln!(w, "{{\"ok\":false,\"error\":\"missing expr\"}}");
                                let _ = w.flush();
                                continue;
                            };
                            let timeout_ms = json_num(&line, "timeout_ms")
                                .map(|t| t as u64).unwrap_or(DEFAULT_TIMEOUT_MS)
                                .clamp(100, 600_000);
                            captured.lock().unwrap().clear();
                            // Wall-clock watchdog: raise the interrupt flag if
                            // the eval overruns. Serial server ⇒ race-free.
                            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                            let watchdog = {
                                let done = done.clone();
                                std::thread::spawn(move || {
                                    let step = std::time::Duration::from_millis(25);
                                    let mut waited = 0u64;
                                    while waited < timeout_ms {
                                        if done.load(std::sync::atomic::Ordering::Relaxed) { return; }
                                        std::thread::sleep(step);
                                        waited += 25;
                                    }
                                    if !done.load(std::sync::atomic::Ordering::Relaxed) {
                                        r2_types::request_interrupt();
                                    }
                                })
                            };
                            let result = eval_all(&mut sess.engine, &expr_src);
                            done.store(true, std::sync::atomic::Ordering::Relaxed);
                            let _ = watchdog.join();
                            // Clear any interrupt the watchdog latched, so a
                            // timed-out eval can't poison the next request.
                            let _ = r2_types::take_interrupt();
                            let out = {
                                let mut c = captured.lock().unwrap();
                                let s = c.clone(); c.clear(); s
                            };
                            match result {
                                Ok(v) => format!(
                                    "{{\"ok\":true,\"class\":\"{}\",\"length\":{},\"result\":\"{}\",\"output\":\"{}\"}}",
                                    esc(v.type_name()), r2_types::rval_length(&v),
                                    esc(&format!("{}", v)), esc(&out)),
                                Err(m) => format!("{{\"ok\":false,\"error\":\"{}\",\"output\":\"{}\"}}", esc(&m), esc(&out)),
                            }
                        }
                    }
                }
                other => format!("{{\"ok\":false,\"error\":\"unknown op '{}'\"}}", esc(other)),
            };
            if let Some(a) = audit.as_mut() {
                let ok = reply.starts_with("{\"ok\":true");
                let _ = writeln!(a, "{}\t{}\t{}\t{}\t{}", now(), peer, sid, esc(&op),
                                 if ok { "ok" } else { "err" });
            }
            let _ = writeln!(w, "{}", reply);
            let _ = w.flush();
        }
    }
    0
}

fn eval_all(engine: &mut Engine, src: &str) -> Result<r2_types::RVal, String> {
    let stmts = Parser::parse(src).map_err(|e| format!("parse error: {}", e))?;
    let mut last = r2_types::RVal::Null;
    for st in &stmts {
        last = engine.eval(st).map_err(|e| e.msg.clone())?;
    }
    Ok(last)
}

fn now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}", d.as_secs())
}
