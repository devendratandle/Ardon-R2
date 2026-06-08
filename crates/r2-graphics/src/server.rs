//! Built-in HTTP plot server (Phase R.G.2).
//!
//! Serves the current plot SVG over `http://127.0.0.1:<port>/` so that
//! plots appear in the user's browser instead of requiring a manual
//! "open file in browser" step. Implemented with `std::net` only — no
//! external HTTP crate, no async runtime, no dependencies.
//!
//! Lifecycle:
//!   1. The user calls `dev.view()` once at the start of their session.
//!      That starts a small server thread on `127.0.0.1:8765` (or the
//!      first free port up to 8775), opens the user's default browser
//!      to the index page, and returns.
//!   2. Subsequent `plot()` / `hist()` / `boxplot()` / `barplot()` calls
//!      write their SVG to disk as usual. The HTML page polls the
//!      `/current.svg` endpoint every 1.5 seconds and swaps the image
//!      element's `src` to refresh.
//!   3. `dev.close()` shuts the viewer page link down by abandoning the
//!      server thread (it stays alive in the background until process
//!      exit — daemon thread, costs nothing while idle).
//!
//! The endpoint set is intentionally tiny:
//!   GET /             → HTML page with embedded auto-refresh script
//!   GET /current.svg  → the most-recently-modified plot SVG in cwd
//!   GET /plot.svg     → explicitly the scatter SVG
//!   GET /hist.svg     → explicitly the histogram SVG
//!   GET /boxplot.svg  → explicitly the boxplot SVG
//!   GET /barplot.svg  → explicitly the barplot SVG
//!   anything else     → 404

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::thread;

const DEFAULT_PORT_START: u16 = 8765;
const PORT_SCAN_RANGE: u16 = 10;

/// `Some(port)` once the server is running, `None` until first call.
static SERVER_PORT: OnceLock<u16> = OnceLock::new();

/// Start the server if not already running. Returns the port it listens on,
/// or `None` if every candidate port was in use.
pub fn ensure_started() -> Option<u16> {
    if let Some(p) = SERVER_PORT.get() {
        return Some(*p);
    }
    for port in DEFAULT_PORT_START..(DEFAULT_PORT_START + PORT_SCAN_RANGE) {
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let _ = thread::Builder::new()
                .name("r2-plot-server".into())
                .spawn(move || run_server(listener, cwd));
            let _ = SERVER_PORT.set(port);
            return Some(port);
        }
    }
    None
}

/// Open the user's default browser at the server's index page. Best-effort:
/// errors are swallowed so a missing browser does not fail the REPL.
pub fn open_browser(port: u16) {
    let url = format!("http://127.0.0.1:{}/", port);
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/c", "start", "", &url]).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }
}

// ─── Server internals ─────────────────────────────────────────────────

fn run_server(listener: TcpListener, root: PathBuf) {
    for stream in listener.incoming().flatten() {
        let root = root.clone();
        // One thread per request — fine for localhost low-volume traffic.
        let _ = thread::Builder::new()
            .name("r2-plot-conn".into())
            .spawn(move || {
                let _ = handle_request(stream, &root);
            });
    }
}

fn handle_request(mut stream: std::net::TcpStream, root: &Path) -> std::io::Result<()> {
    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]);

    // Extract the request path (first line, second whitespace token).
    let path = req
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();

    let (status, content_type, body): (&str, &str, Vec<u8>) = match path.as_str() {
        "/" => ("200 OK", "text/html; charset=utf-8", index_html().into_bytes()),
        // Serve the CURRENT plot from the shared in-memory screen buffer —
        // the live device plot, not a scan of stray .svg files in cwd
        // (which showed stale/leftover plots from earlier sessions).
        "/current.svg" => match crate::device::screen_svg() {
            Some(svg) => ("200 OK", "image/svg+xml", svg.into_bytes()),
            None => ("404 Not Found", "text/plain", b"no plot yet".to_vec()),
        },
        // No file gallery: R's screen device shows the current plot only.
        "/list" => ("200 OK", "application/json; charset=utf-8", b"[]".to_vec()),
        // Phase R.G.2 — serve any .svg file from cwd by name. Used by
        // the gallery to display every plot saved during the session
        // (user-chosen names plus the four type-default names).
        p if p.ends_with(".svg") => {
            // Strip the leading slash and prevent path traversal.
            let name = p.trim_start_matches('/');
            if name.contains('/') || name.contains('\\') || name.contains("..") {
                ("400 Bad Request", "text/plain", b"invalid path".to_vec())
            } else {
                serve_file(&root.join(name))
            }
        }
        _ => ("404 Not Found", "text/plain", b"not found".to_vec()),
    };

    let header = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status,
        content_type,
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn serve_file(path: &Path) -> (&'static str, &'static str, Vec<u8>) {
    match std::fs::read(path) {
        Ok(data) => ("200 OK", "image/svg+xml", data),
        Err(_) => ("404 Not Found", "text/plain", b"file not found".to_vec()),
    }
}

fn index_html() -> String {
    // Two-pane page:
    //   - Top: current plot, auto-refreshes from /current.svg every 1.5s
    //   - Bottom: gallery of every .svg in cwd, rebuilt every 2s from /list
    // Clicking a gallery thumbnail loads that file into the top pane so
    // the user can step back through any earlier plot in the session.
    r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ardon-R2 plot viewer</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0f172a; --fg: #e2e8f0; --muted: #94a3b8;
    --card: #ffffff; --accent: #2563eb; --border: rgba(255,255,255,0.08);
  }
  html, body { margin: 0; padding: 0; min-height: 100%; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  .topbar { display: flex; align-items: baseline; gap: 12px; padding: 14px 20px;
    border-bottom: 1px solid var(--border); }
  .topbar h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: 0.2px; }
  .topbar .sub { color: var(--muted); font-size: 12px; }
  .topbar .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 8px rgba(34,197,94,0.6); display: inline-block; vertical-align: middle; margin-right: 6px; }
  .stage { padding: 20px; display: flex; justify-content: center; }
  .card { background: var(--card); border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    padding: 16px; max-width: 95vw; position: relative; }
  .card img { display: block; max-width: 100%; height: auto; }
  .card .label { position: absolute; top: -22px; left: 4px; font-size: 12px; color: var(--muted); }
  .live-btn { font-size: 11px; color: var(--accent); cursor: pointer; margin-left: 8px;
    text-decoration: underline; background: none; border: none; padding: 0; font-family: inherit; }
  .empty { color: var(--muted); font-size: 13px; padding: 40px; text-align: center; }

  .gallery-head { padding: 18px 20px 6px; border-top: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 10px; }
  .gallery-head h2 { font-size: 14px; font-weight: 600; margin: 0; }
  .gallery-head .count { color: var(--muted); font-size: 12px; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px; padding: 12px 20px 24px; }
  .thumb { background: var(--card); border-radius: 6px; padding: 8px; cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.1s ease; }
  .thumb:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.4); }
  .thumb.active { outline: 2px solid var(--accent); }
  .thumb img { width: 100%; height: 110px; object-fit: contain; display: block; background: white; }
  .thumb .name { font-size: 11px; color: #1e293b; padding: 6px 4px 0; word-break: break-all; text-align: center; }
</style>
</head>
<body>

<div class="topbar">
  <h1><span class="dot"></span>Ardon-R2 plot viewer</h1>
  <span class="sub">Inspired by R. Built on Rust. — auto-refreshes when you call plot()</span>
</div>

<div class="stage">
  <div class="card">
    <div class="label">
      <span id="current-label">current</span>
      <button class="live-btn" id="live-btn" onclick="resumeLive()">return to live</button>
    </div>
    <img id="plot" alt="current plot" src="/current.svg?t=0"
         onerror="this.style.display='none'; document.getElementById('empty').style.display='block';">
    <div id="empty" class="empty" style="display:none">
      No plot rendered yet. Call <code>plot(...)</code> in the REPL.
    </div>
  </div>
</div>

<div class="gallery-head">
  <h2>Session gallery</h2>
  <span class="count" id="gallery-count">scanning…</span>
</div>
<div class="gallery" id="gallery"></div>

<script>
  // ── Live current-plot polling ────────────────────────────────────
  // When the user is in "live" mode (default), poll /current.svg.
  // When the user clicks a gallery thumbnail, pin to that file
  // until they click "return to live".
  let live = true;
  let pinnedName = null;
  let liveTick = 0;

  const img = document.getElementById('plot');
  const empty = document.getElementById('empty');
  const liveBtn = document.getElementById('live-btn');
  const currentLabel = document.getElementById('current-label');

  function refreshLive() {
    if (!live) return;
    liveTick++;
    const probe = new Image();
    probe.onload = function () {
      img.style.display = 'block';
      empty.style.display = 'none';
      img.src = probe.src;
    };
    probe.onerror = function () { /* keep current view */ };
    probe.src = '/current.svg?t=' + liveTick;
    currentLabel.textContent = 'current (live)';
    liveBtn.style.display = 'none';
  }
  setInterval(refreshLive, 1500);

  function pinTo(name) {
    live = false;
    pinnedName = name;
    img.src = '/' + name + '?pin=' + Date.now();
    img.style.display = 'block';
    empty.style.display = 'none';
    currentLabel.textContent = 'viewing: ' + name;
    liveBtn.style.display = 'inline-block';
    markActive(name);
  }
  function resumeLive() {
    live = true;
    pinnedName = null;
    refreshLive();
    markActive(null);
  }

  function markActive(name) {
    document.querySelectorAll('.thumb').forEach(t => {
      if (t.dataset.name === name) t.classList.add('active');
      else t.classList.remove('active');
    });
  }

  // ── Gallery polling ───────────────────────────────────────────────
  let lastSig = '';
  async function refreshGallery() {
    try {
      const r = await fetch('/list');
      if (!r.ok) return;
      const items = await r.json();
      const head = document.querySelector('.gallery-head');
      const galleryEl = document.getElementById('gallery');
      if (items.length === 0) {
        if (head) head.style.display = 'none';
        galleryEl.style.display = 'none';
        return;
      }
      if (head) head.style.display = '';
      galleryEl.style.display = '';
      const sig = items.map(x => x.name + ':' + x.mtime).join('|');
      if (sig === lastSig) return;
      lastSig = sig;
      const g = document.getElementById('gallery');
      g.innerHTML = '';
      items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'thumb';
        div.dataset.name = item.name;
        div.onclick = () => pinTo(item.name);
        const im = document.createElement('img');
        im.src = '/' + item.name + '?t=' + item.mtime;
        const lbl = document.createElement('div');
        lbl.className = 'name';
        lbl.textContent = item.name;
        div.appendChild(im);
        div.appendChild(lbl);
        g.appendChild(div);
      });
      document.getElementById('gallery-count').textContent =
        items.length + (items.length === 1 ? ' file' : ' files') + ' (newest first)';
      if (!live && pinnedName) markActive(pinnedName);
    } catch (e) { /* server might be reloading */ }
  }
  setInterval(refreshGallery, 2000);
  refreshGallery();
</script>
</body>
</html>
"#.into()
}
