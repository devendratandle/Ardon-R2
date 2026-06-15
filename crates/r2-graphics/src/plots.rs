//! Primary data-shape plot builtins — Phase R.3 / Phase R.G.
//!
//! `plot`, `hist`, `boxplot`, `barplot` — each draws into the
//! thread-local `PlotDevice` defined in `device.rs`. They no longer
//! write a file directly; the device exposes `save_to_file()` for
//! explicit flushes via `dev.off()` / `save_plot()`. To preserve the
//! historical UX where `plot()` produced `plot.svg` on disk, each
//! function also calls `save_to_file()` with its default path at the
//! end. The in-memory state is the source of truth for the
//! "is a plot open" predicate that overlays consult.
//!
//! Multi-panel `par(mfrow=...)` / `par(mfcol=...)` is honored: the
//! `begin_plot()` call returns the rectangle to draw into, which
//! shifts and scales the SVG coordinates without changing the rest
//! of the drawing code.

use crate::device::{begin_plot, with_device};
use crate::{gn, gv, val_to_str};
use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal};

// ─── Label / axis chrome — shared by every plot function ────────────
//
// `LabelOpts` snapshots the device's per-element font/color/scale
// defaults and applies any per-call overrides (main, sub, xlab, ylab,
// axis ticks). This lets a single `render_chrome()` call do the right
// thing for `plot()`, `hist()`, `boxplot()`, `barplot()` etc.

#[derive(Debug, Clone)]
pub(crate) struct LabelOpts {
    pub cex_main: f64, pub cex_sub: f64, pub cex_lab: f64, pub cex_axis: f64,
    pub font_main: i32, pub font_sub: i32, pub font_lab: i32, pub font_axis: i32,
    pub col_main: String, pub col_sub: String, pub col_lab: String, pub col_axis: String,
    /// las: 0 = parallel to axis (default), 1 = always horizontal,
    ///      2 = perpendicular to axis,        3 = always vertical.
    pub las: i32,
}

impl LabelOpts {
    pub(crate) fn from_args(a: &[EvalArg]) -> Self {
        // Defaults follow R's documented `par()` settings:
        //   font.main = 2 (bold)   — title visually darker than body
        //   font.sub  = 1 (plain)
        //   font.lab  = 1 (plain)
        //   font.axis = 1 (plain)
        // The user can override per-call via `font.main = 1`, etc.
        let (cm, cs, cl, cx, fm, fl, fa, com, cosu, col, coax, las) =
            with_device(|d| {
                let p = &d.params;
                (p.cex_main, p.cex.max(0.9), p.cex_lab, p.cex_axis,
                 2, 1, 1,
                 p.fg.clone(), p.fg.clone(), p.fg.clone(), p.fg.clone(),
                 p.las)
            });
        Self {
            cex_main:  gn(a, "cex.main") .and_then(num).unwrap_or(cm),
            cex_sub:   gn(a, "cex.sub")  .and_then(num).unwrap_or(cs),
            cex_lab:   gn(a, "cex.lab")  .and_then(num).unwrap_or(cl),
            cex_axis:  gn(a, "cex.axis") .and_then(num).unwrap_or(cx),
            font_main: gn(a, "font.main").and_then(int).unwrap_or(fm),
            font_sub:  gn(a, "font.sub") .and_then(int).unwrap_or(1),
            font_lab:  gn(a, "font.lab") .and_then(int).unwrap_or(fl),
            font_axis: gn(a, "font.axis").and_then(int).unwrap_or(fa),
            col_main:  gn(a, "col.main") .map(|v| val_to_str(&v)).unwrap_or(com),
            col_sub:   gn(a, "col.sub")  .map(|v| val_to_str(&v)).unwrap_or(cosu),
            col_lab:   gn(a, "col.lab")  .map(|v| val_to_str(&v)).unwrap_or(col),
            col_axis:  gn(a, "col.axis") .map(|v| val_to_str(&v)).unwrap_or(coax),
            las:       gn(a, "las")      .and_then(int).unwrap_or(las),
        }
    }
}

fn num(v: RVal) -> Option<f64> { v.as_reals().ok()?.into_iter().next()? }
fn int(v: RVal) -> Option<i32> { v.as_reals().ok()?.into_iter().next()?.map(|x| x as i32) }

/// R's default colour palette (1-based). `col=2` → red, etc. Index 0 /
/// out-of-range wraps like R's `palette()` recycling.
pub(crate) fn r_palette(i: i32) -> &'static str {
    const PAL: [&str; 8] = [
        "black", "red", "green3", "blue", "cyan", "magenta", "yellow", "gray",
    ];
    if i <= 0 { return "black"; }
    PAL[((i - 1) as usize) % PAL.len()]
}

/// Resolve a `col=` argument into a per-element colour vector (recycled
/// by the caller). Accepts a character vector of names/hex, or an integer
/// vector mapped through [`r_palette`]. Falls back to `default`.
pub(crate) fn color_vec(arg: Option<RVal>, default: &str) -> Vec<String> {
    match arg {
        Some(RVal::Character(v, _)) if !v.is_empty() => v
            .iter()
            .map(|c| c.as_ref().map(|s| s.to_string()).unwrap_or_else(|| default.to_string()))
            .collect(),
        Some(RVal::Integer(v, _)) if !v.is_empty() => v
            .iter()
            .map(|n| n.map(|n| r_palette(n).to_string()).unwrap_or_else(|| default.to_string()))
            .collect(),
        Some(RVal::Numeric(v, _)) if !v.is_empty() => v
            .iter()
            .map(|n| n.map(|n| r_palette(n as i32).to_string()).unwrap_or_else(|| default.to_string()))
            .collect(),
        _ => vec![default.to_string()],
    }
}

/// Render one R plotting symbol (`pch`) centred at `(px,py)` with radius
/// `r` in colour `col`. Covers the commonly-used codes; unknown codes
/// fall back to an open circle (R's default `pch=1`).
pub(crate) fn point_symbol(px: f64, py: f64, r: f64, pch: i32, col: &str) -> String {
    let open = format!(r#"fill="none" stroke="{}" stroke-width="1""#, col);
    let fill = format!(r#"fill="{}" stroke="{}" stroke-width="0.5""#, col, col);
    match pch {
        16 | 19 => format!(r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" {}/>"#, px, py, r, fill),
        20      => format!(r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" {}/>"#, px, py, r * 0.66, fill),
        21      => format!(r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" {}/>"#, px, py, r, fill),
        1       => format!(r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" {}/>"#, px, py, r, open),
        0 | 15 | 22 => {
            let s = r * 1.8;
            let style = if pch == 0 { &open } else { &fill };
            format!(r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" {}/>"#,
                    px - s / 2.0, py - s / 2.0, s, s, style)
        }
        2 | 17 | 24 => {
            let s = r * 2.0;
            let style = if pch == 2 { &open } else { &fill };
            format!(r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" {}/>"#,
                    px, py - s * 0.6, px - s * 0.55, py + s * 0.4, px + s * 0.55, py + s * 0.4, style)
        }
        5 | 18 | 23 => {
            let s = r * 1.6;
            let style = if pch == 5 { &open } else { &fill };
            format!(r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" {}/>"#,
                    px, py - s, px + s, py, px, py + s, px - s, py, style)
        }
        3 => format!(
            r#"<path d="M{:.1},{:.1} L{:.1},{:.1} M{:.1},{:.1} L{:.1},{:.1}" stroke="{}" stroke-width="1"/>"#,
            px - r, py, px + r, py, px, py - r, px, py + r, col),
        4 => format!(
            r#"<path d="M{:.1},{:.1} L{:.1},{:.1} M{:.1},{:.1} L{:.1},{:.1}" stroke="{}" stroke-width="1"/>"#,
            px - r, py - r, px + r, py + r, px - r, py + r, px + r, py - r, col),
        _ => format!(r#"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" {}/>"#, px, py, r, open),
    }
}

/// R's `font` codes: 1 plain, 2 bold, 3 italic, 4 bold+italic.
fn font_attrs(code: i32) -> &'static str {
    match code {
        2 => r#" font-weight="bold""#,
        3 => r#" font-style="italic""#,
        4 => r#" font-weight="bold" font-style="italic""#,
        _ => "",
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PanelRect {
    pub ox: f64, pub oy: f64, pub w: f64, pub h: f64,
    pub ml: f64, pub mt: f64, pub pw: f64, pub ph: f64,
}

pub(crate) fn render_chrome(p: &PanelRect, title: &str, sub: &str,
                            xlab: &str, ylab: &str, o: &LabelOpts) -> String {
    let mut s = String::new();
    if !title.is_empty() {
        let fs = 14.0 * o.cex_main;
        s.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}"{}>{}</text>"#,
            p.ox + p.w / 2.0, p.oy + 18.0, fs, o.col_main, font_attrs(o.font_main), escape_xml(title)
        ));
    }
    if !sub.is_empty() {
        // Subtitle sits BELOW the panel (below xlab).
        let fs = 11.0 * o.cex_sub;
        s.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}"{}>{}</text>"#,
            p.ox + p.ml + p.pw / 2.0,
            p.oy + p.h - 4.0,
            fs, o.col_sub, font_attrs(o.font_sub), escape_xml(sub)
        ));
    }
    if !xlab.is_empty() {
        let fs = 12.0 * o.cex_lab;
        // xlab sits 18 px below the tick labels (which are at
        // p.oy + p.mt + p.ph + 14). Keeps the bottom margin tight —
        // earlier "p.h - 6" left a visible ~30-px empty band when
        // mb was 50.
        let tick_y = p.oy + p.mt + p.ph + 14.0;
        let y = if sub.is_empty() { tick_y + 18.0 } else { tick_y + 30.0 };
        s.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}"{}>{}</text>"#,
            p.ox + p.ml + p.pw / 2.0, y, fs, o.col_lab, font_attrs(o.font_lab), escape_xml(xlab)
        ));
    }
    if !ylab.is_empty() {
        let fs = 12.0 * o.cex_lab;
        let yx = p.ox + 15.0;
        let yy = p.oy + p.mt + p.ph / 2.0;
        s.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}"{} transform="rotate(-90,{},{})">{}</text>"#,
            yx, yy, fs, o.col_lab, font_attrs(o.font_lab), yx, yy, escape_xml(ylab)
        ));
    }
    s
}

/// Format an axis tick value compactly, adapting precision to the
/// step size — so count axes show `32058`, not `32058.00`, and data
/// axes still get the decimals they need. Mirrors R's tidy tick labels.
fn fmt_tick(v: f64, range: f64) -> String {
    let step = (range / 4.0).abs();
    if v.abs() < 1e-9 { return "0".to_string(); }
    if step >= 1.0 {
        if (v - v.round()).abs() < 1e-6 { format!("{}", v.round() as i64) }
        else { format!("{:.1}", v) }
    } else if step >= 0.01 {
        format!("{:.2}", v)
    } else {
        format!("{:.4}", v)
    }
}

pub(crate) fn render_axis_ticks(p: &PanelRect,
                                xmin: f64, _xmax: f64, ymin: f64, _ymax: f64,
                                xrange: f64, yrange: f64,
                                o: &LabelOpts) -> String {
    let mut s = String::new();
    let fs = 9.0 * o.cex_axis;
    // x-axis tick rotation: las=0 or 1 horizontal, las=2 or 3 vertical.
    let x_rot = matches!(o.las, 2 | 3);
    let y_rot = matches!(o.las, 0 | 3);
    for i in 0..=4 {
        let frac = i as f64 / 4.0;
        let xv = xmin + frac * xrange;
        let yv = ymin + frac * yrange;
        let px = p.ox + p.ml + frac * p.pw;
        let py = p.oy + p.mt + p.ph - frac * p.ph;
        // x-axis tick label below the panel.
        let tx = px;
        let ty = p.oy + p.mt + p.ph + 14.0;
        if x_rot {
            s.push_str(&format!(
                r#"<text x="{:.0}" y="{}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}" transform="rotate(-90,{:.0},{})">{}</text>"#,
                tx, ty, fs, o.col_axis, tx, ty, fmt_tick(xv, xrange)
            ));
        } else {
            s.push_str(&format!(
                r#"<text x="{:.0}" y="{}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}">{}</text>"#,
                tx, ty, fs, o.col_axis, fmt_tick(xv, xrange)
            ));
        }
        // y-axis tick label to the left of the panel.
        let lx = p.ox + p.ml - 5.0;
        let ly = py + 3.0;
        if y_rot {
            s.push_str(&format!(
                r#"<text x="{}" y="{:.0}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}" transform="rotate(-90,{},{:.0})">{}</text>"#,
                lx, ly, fs, o.col_axis, lx, ly, fmt_tick(yv, yrange)
            ));
        } else {
            s.push_str(&format!(
                r#"<text x="{}" y="{:.0}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px" fill="{}">{}</text>"#,
                lx, ly, fs, o.col_axis, fmt_tick(yv, yrange)
            ));
        }
    }
    s
}

pub(crate) fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
     .replace('"', "&quot;").replace('\'', "&apos;")
}

#[inline]
fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(std::sync::Arc::from(s))], Attrs::default())
}

/// `plot(x [, y], main=, xlab=, ylab=)` — scatter into the device.
///
/// Model-aware dispatch (`plot(lm)`, `plot(gbm)`, `plot(kmeans)`)
/// lives in r2-engine via the split-handler pattern.
pub fn bi_plot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = if a.len() > 1 && a[1].name.is_none() {
        gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect()
    } else {
        (1..=x.len()).map(|i| i as f64).collect()
    };
    // R-faithful default labels: empty when not supplied. The engine
    // will eventually pass the symbol names of the input args via a
    // deparse hook; for now empty defaults match R when those names
    // are not deducible.
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or_default();
    let sub   = gn(a, "sub" ).map(|v| val_to_str(&v)).unwrap_or_default();
    let xlab  = gn(a, "xlab").map(|v| val_to_str(&v)).unwrap_or_else(|| "x".into());
    let ylab  = gn(a, "ylab").map(|v| val_to_str(&v)).unwrap_or_else(|| "y".into());

    // Per-element scale, color, font overrides — any of these can
    // arrive as a named arg to plot(...) and shadow the par() default
    // for THIS plot only.
    let opts = LabelOpts::from_args(a);

    // Per-plot params: inline args (col=, cex=, pch=, lwd=, type=) shadow
    // the par() device defaults for THIS call only — R semantics.
    let (dev_col, dev_cex, dev_pch, dev_lwd) =
        with_device(|d| (d.params.col.clone(), d.params.cex, d.params.pch, d.params.lwd));
    let cex_pt = gn(a, "cex").and_then(num).unwrap_or(dev_cex);
    let pch    = gn(a, "pch").and_then(int).unwrap_or(dev_pch);
    let lwd    = gn(a, "lwd").and_then(num).unwrap_or(dev_lwd);
    let cols   = color_vec(gn(a, "col"), &dev_col);
    let ptype  = gn(a, "type").map(|v| val_to_str(&v)).unwrap_or_else(|| "p".into());

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 20.0, 36.0, 40.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let xmin_raw = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax_raw = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin_raw = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax_raw = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // 4 % padding on each side so the extreme data points don't sit
    // exactly on the axis frame — matches R's pretty-axis behavior in
    // spirit without re-implementing the full pretty() algorithm.
    let pad = 0.04;
    let xrange_raw = if (xmax_raw - xmin_raw).abs() < 1e-10 { 1.0 } else { xmax_raw - xmin_raw };
    let yrange_raw = if (ymax_raw - ymin_raw).abs() < 1e-10 { 1.0 } else { ymax_raw - ymin_raw };
    let xmin = xmin_raw - xrange_raw * pad;
    let xmax = xmax_raw + xrange_raw * pad;
    let ymin = ymin_raw - yrange_raw * pad;
    let ymax = ymax_raw + yrange_raw * pad;
    let xrange = xmax - xmin;
    let yrange = ymax - ymin;

    // Record the coordinate system so overlays (points/lines/abline/text/
    // rect/…) align with this plot — R's incremental graphics model.
    with_device(|d| d.coords = Some(crate::device::PlotCoords {
        px0: ox + ml, py0: oy + mt, pw, ph, xmin, xmax, ymin, ymax,
    }));

    let mut frag = String::new();
    // Plotting region border.
    frag.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1" shape-rendering="crispEdges"/>"#,
        ox + ml, oy + mt, pw, ph
    ));

    // Title + subtitle + axis labels.
    let panel = PanelRect { ox, oy, w, h, ml, mt, pw, ph };
    frag.push_str(&render_chrome(&panel, &title, &sub, &xlab, &ylab, &opts));

    // Axis tick labels with `las` rotation support.
    frag.push_str(&render_axis_ticks(&panel, xmin, xmax, ymin, ymax, xrange, yrange, &opts));

    // Pixel coordinates for each (x,y).
    let n = x.len().min(y.len());
    let pts: Vec<(f64, f64)> = (0..n).map(|i| {
        let px = ox + ml + (x[i] - xmin) / xrange * pw;
        let py = oy + mt + ph - (y[i] - ymin) / yrange * ph;
        (px, py)
    }).collect();

    // type=: "p" points (default), "l" lines, "b"/"o" both, "h" needles
    // to the x-axis baseline, "n" nothing (axes only). The line colour
    // uses the first recycled colour, matching R.
    let draw_lines  = matches!(ptype.as_str(), "l" | "b" | "o");
    let draw_points = matches!(ptype.as_str(), "p" | "b" | "o");
    let line_col = cols.first().map(|s| s.as_str()).unwrap_or("black");
    if ptype == "h" {
        let baseline = oy + mt + ph - (0.0_f64.max(ymin) - ymin) / yrange * ph;
        for (i, &(px, py)) in pts.iter().enumerate() {
            let c = &cols[i % cols.len()];
            frag.push_str(&format!(
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.1}"/>"#,
                px, baseline, px, py, c, lwd));
        }
    }
    if draw_lines && pts.len() > 1 {
        let path: String = pts.iter().enumerate()
            .map(|(i, &(px, py))| format!("{}{:.1},{:.1}", if i == 0 { "M" } else { " L" }, px, py))
            .collect();
        frag.push_str(&format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="{:.1}"/>"#, path, line_col, lwd));
    }
    if draw_points {
        for (i, &(px, py)) in pts.iter().enumerate() {
            let c = &cols[i % cols.len()];
            frag.push_str(&point_symbol(px, py, 3.0 * cex_pt, pch, c));
        }
    }

    // Commit to the device. Cannot use append_svg() because the device's
    // has_plot is already true (begin_plot set it), but append_svg requires
    // the precondition we just satisfied — so the direct push is correct here.
    with_device(|d| d.svg_body.push_str(&frag));

    // Auto-flush to preserve the legacy "plot()  produces plot.svg" UX.
    // Print the absolute path so the user knows where to find it — the
    // default cwd is the user's Documents folder (R-style), not the
    // .exe's install dir, so the file lands somewhere they can actually
    // see in Explorer.
    // Display-aware: with a live display (GUI window / CLI browser) the
    // plot is shown by the device — don't write a file. Only a headless
    // script auto-saves (so output isn't lost); explicit save is
    // save_plot() / a filename.
    // Route to the active output (browser / GUI window / file) via the
    // single device hook. Returns the filename only when it wrote one.
    match crate::device::finish_plot("plot.svg") {
        Some(p) => Ok(rstr(&p)),
        None => Ok(RVal::Null),
    }
}

/// Min/max of a slice, ignoring non-finite values.
fn minmax(v: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in v {
        if x.is_finite() {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    if !lo.is_finite() { (0.0, 1.0) } else { (lo, hi) }
}

/// Extract numeric columns + their names from a `pairs()` argument list.
/// Accepts a data.frame / list (named numeric columns) as the first arg,
/// or two-plus bare numeric-vector args. Non-numeric columns are skipped.
fn extract_columns(a: &[EvalArg]) -> (Vec<Vec<f64>>, Vec<String>) {
    let first = a.iter().find(|e| e.name.is_none()).map(|e| &e.value);
    // data.frame — the common pairs() input.
    if let Some(RVal::DataFrame(df)) = first {
        let mut cols = Vec::new();
        let mut names = Vec::new();
        for (nm, v) in &df.columns {
            if let Ok(reals) = v.as_reals() {
                let col: Vec<f64> = reals.into_iter().flatten().collect();
                if !col.is_empty() {
                    cols.push(col);
                    names.push(nm.to_string());
                }
            }
        }
        return (cols, names);
    }
    // Plain list of named numeric columns.
    if let Some(RVal::List(items)) = first {
        let mut cols = Vec::new();
        let mut names = Vec::new();
        for (idx, (nm, v)) in items.iter().enumerate() {
            if let Ok(reals) = v.as_reals() {
                let col: Vec<f64> = reals.into_iter().flatten().collect();
                if !col.is_empty() {
                    cols.push(col);
                    names.push(nm.as_ref().map(|s| s.to_string())
                        .unwrap_or_else(|| format!("V{}", idx + 1)));
                }
            }
        }
        return (cols, names);
    }
    // Fallback: multiple bare numeric vectors → columns V1, V2, …
    let mut cols = Vec::new();
    let mut names = Vec::new();
    for (i, e) in a.iter().filter(|e| e.name.is_none()).enumerate() {
        if let Ok(reals) = e.value.as_reals() {
            let col: Vec<f64> = reals.into_iter().flatten().collect();
            if !col.is_empty() {
                cols.push(col);
                names.push(format!("V{}", i + 1));
            }
        }
    }
    (cols, names)
}

/// `pairs(x)` — scatterplot matrix. `x` is a data.frame / list of numeric
/// columns (or several numeric vectors). Draws an n×n grid: the diagonal
/// shows each variable's name; cell (row i, col j) scatters column j on x
/// against column i on y (R convention). Honors `col=`/`pch=`/`cex=`.
pub fn bi_pairs(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let (cols, names) = extract_columns(a);
    let n = cols.len();
    if n < 2 {
        return Err(R2Err {
            msg: "pairs(): need at least 2 numeric columns".into(),
            kind: ErrKind::Runtime,
        });
    }

    let main = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or_default();
    let pch = gn(a, "pch").and_then(int).unwrap_or_else(|| with_device(|d| d.params.pch));
    let cex = gn(a, "cex").and_then(num).unwrap_or(1.0);
    let dev_col = with_device(|d| d.params.col.clone());
    let point_cols = color_vec(gn(a, "col"), &dev_col);

    // pairs() owns the whole figure — begin_plot opens a fresh panel and
    // returns its (x, y, w, h); with no mfrow active that's the full canvas.
    let (ox, oy, w, h) = begin_plot();

    let top = if main.is_empty() { 8.0 } else { 28.0 };
    let cell_w = w / n as f64;
    let cell_h = (h - top) / n as f64;
    let pad = 6.0;

    let mut frag = String::new();
    if !main.is_empty() {
        frag.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15px" font-weight="bold" fill="black">{}</text>"#,
            ox + w / 2.0, oy + 18.0, escape_xml(&main)));
    }

    for i in 0..n {            // row → y variable
        for j in 0..n {        // col → x variable
            let cx0 = ox + j as f64 * cell_w + pad;
            let cy0 = oy + top + i as f64 * cell_h + pad;
            let cwd = cell_w - 2.0 * pad;
            let chd = cell_h - 2.0 * pad;
            // Cell border.
            frag.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="none" stroke="black" stroke-width="1" shape-rendering="crispEdges"/>"#,
                cx0, cy0, cwd, chd));
            if i == j {
                // Diagonal — variable name.
                frag.push_str(&format!(
                    r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13px" font-weight="bold" fill="black">{}</text>"#,
                    cx0 + cwd / 2.0, cy0 + chd / 2.0 + 4.0, escape_xml(&names[i])));
            } else {
                let xs = &cols[j];
                let ys = &cols[i];
                let (xmin, xmax) = minmax(xs);
                let (ymin, ymax) = minmax(ys);
                let xr = if (xmax - xmin).abs() < 1e-10 { 1.0 } else { xmax - xmin };
                let yr = if (ymax - ymin).abs() < 1e-10 { 1.0 } else { ymax - ymin };
                let (inx, iny) = (cx0 + 4.0, cy0 + 4.0);
                let (inw, inh) = (cwd - 8.0, chd - 8.0);
                let m = xs.len().min(ys.len());
                for k in 0..m {
                    let px = inx + (xs[k] - xmin) / xr * inw;
                    let py = iny + inh - (ys[k] - ymin) / yr * inh;
                    let c = &point_cols[k % point_cols.len()];
                    frag.push_str(&point_symbol(px, py, 2.0 * cex, pch, c));
                }
            }
        }
    }

    with_device(|d| d.svg_body.push_str(&frag));
    match crate::device::finish_plot("pairs.svg") {
        Some(p) => Ok(rstr(&p)),
        None => Ok(RVal::Null),
    }
}

/// `hist(x, breaks=, main=)` — text + SVG histogram into the device.
pub fn bi_hist(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let nbins = gn(a, "breaks").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(10.0) as usize;
    let nbins = nbins.max(1);
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or_else(|| "Histogram".into());
    let sub   = gn(a, "sub" ).map(|v| val_to_str(&v)).unwrap_or_default();
    let xlab  = gn(a, "xlab").map(|v| val_to_str(&v)).unwrap_or_else(|| "x".into());
    let ylab  = gn(a, "ylab").map(|v| val_to_str(&v)).unwrap_or_else(|| "Frequency".into());

    let opts = LabelOpts::from_args(a);
    // R-style named color args. `col` = fill of each bar. `border` =
    // outline stroke. Defaults match R's hist(): light gray fill
    // with a black outline. User-supplied colors pass through
    // unchanged so any CSS name or "#rrggbb" works.
    let col_fill   = gn(a, "col"   ).map(|v| val_to_str(&v)).unwrap_or_else(|| "lightgray".into());
    let col_border = gn(a, "border").map(|v| val_to_str(&v)).unwrap_or_else(|| "black".into());

    // Raw data range, then add a 4 % cushion on each side so the
    // first / last bars don't sit flush against the axis frame.
    // Y-axis stays anchored at 0 (histogram convention).
    let xmin_raw = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax_raw = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xrange_raw = if (xmax_raw - xmin_raw).abs() < 1e-10 { 1.0 } else { xmax_raw - xmin_raw };
    let pad = 0.04;
    let xmin = xmin_raw - xrange_raw * pad;
    let xmax = xmax_raw + xrange_raw * pad;
    let bin_width = (xmax_raw - xmin_raw) / nbins as f64;
    let bin_width = if bin_width.is_finite() && bin_width > 0.0 { bin_width } else { 1.0 };

    let mut counts = vec![0usize; nbins];
    for &val in &x {
        let bin = ((val - xmin_raw) / bin_width).floor() as usize;
        let bin = bin.min(nbins - 1);
        counts[bin] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1);

    let (ox, oy, w, h) = begin_plot();
    // mb tightened from 50 to 40: the new xlab placement (right
    // below tick labels) no longer needs the extra 10 px margin
    // and the plot region gets that height back.
    let (ml, mr, mt, mb) = (60.0, 20.0, 36.0, 40.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut frag = String::new();

    // Plot area frame.
    frag.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1" shape-rendering="crispEdges"/>"#,
        ox + ml, oy + mt, pw, ph
    ));

    // Bars positioned over the padded x-axis so the cushion shows
    // on both sides. Each bar paints with `col` fill and `border`
    // stroke; R's hist() defaults are lightgray / black.
    let xrange_padded = xmax - xmin;
    for (i, &count) in counts.iter().enumerate() {
        let bh = if max_count > 0 { count as f64 / max_count as f64 * ph } else { 0.0 };
        let bin_lo = xmin_raw + i as f64 * bin_width;
        let bin_hi = bin_lo + bin_width;
        let bx = ox + ml + (bin_lo - xmin) / xrange_padded * pw;
        let bx_hi = ox + ml + (bin_hi - xmin) / xrange_padded * pw;
        let bw_px = bx_hi - bx;
        let by = oy + mt + ph - bh;
        frag.push_str(&format!(
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}" stroke="{}" stroke-width="0.5"/>"#,
            bx, by, bw_px, bh, col_fill, col_border
        ));
    }

    // Shared chrome — title / sub / xlab / ylab with R-style options.
    let panel = PanelRect { ox, oy, w, h, ml, mt, pw, ph };
    frag.push_str(&render_chrome(&panel, &title, &sub, &xlab, &ylab, &opts));

    // Axis ticks: x-axis along data range, y-axis 0..max_count.
    let yrange = if max_count > 0 { max_count as f64 } else { 1.0 };
    frag.push_str(&render_axis_ticks(&panel, xmin, xmax, 0.0, yrange,
                                     xmax - xmin, yrange, &opts));

    with_device(|d| d.svg_body.push_str(&frag));
    crate::device::finish_plot("hist.svg");
    Ok(RVal::Null)
}

/// `boxplot(g1=, g2=, ..., main=)` — multi-group box-and-whisker into the device.
pub fn bi_boxplot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or("Boxplot".into());

    // R-style named color args.
    //   col       = fill of each box     (default: light blue like R)
    //   border    = outline + whiskers   (default: black)
    //   medcol    = median bar           (default: black)
    let col_box    = gn(a, "col"    ).map(|v| val_to_str(&v)).unwrap_or_else(|| "lightblue".into());
    let col_border = gn(a, "border" ).map(|v| val_to_str(&v)).unwrap_or_else(|| "black".into());
    let col_med    = gn(a, "medcol" ).map(|v| val_to_str(&v)).unwrap_or_else(|| col_border.clone());

    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for (gi, arg) in a.iter().enumerate() {
        let nm = arg.name.as_ref().map(|n| n.as_ref());
        // Skip args that name themselves as known scalar opts.
        if matches!(nm, Some("main") | Some("col") | Some("border") | Some("medcol")
                       | Some("sub") | Some("xlab") | Some("ylab")
                       | Some("cex.main") | Some("cex.sub") | Some("cex.lab") | Some("cex.axis")
                       | Some("font.main") | Some("font.sub") | Some("font.lab") | Some("font.axis")
                       | Some("col.main") | Some("col.sub") | Some("col.lab") | Some("col.axis")
                       | Some("las"))
        { continue; }
        let data: Vec<f64> = arg.value.as_reals()?.into_iter().filter_map(|x| x).collect();
        let name = arg.name.as_ref().map(|n| n.to_string()).unwrap_or(format!("V{}", gi + 1));
        groups.push((name, data));
    }
    if groups.is_empty() {
        return Err(R2Err { msg: "boxplot needs data".into(), kind: ErrKind::Runtime });
    }

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 30.0, 36.0, 50.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    // 4 % cushion on the y-axis so whiskers aren't flush with the
    // frame. Same convention as bi_plot / bi_hist.
    let raw_min = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::INFINITY, f64::min);
    let raw_max = groups.iter().flat_map(|(_, d)| d.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let raw_range = if (raw_max - raw_min).abs() < 1e-10 { 1.0 } else { raw_max - raw_min };
    let all_min = raw_min - raw_range * 0.04;
    let all_max = raw_max + raw_range * 0.04;
    let range = all_max - all_min;

    let opts = LabelOpts::from_args(a);
    let sub  = gn(a, "sub" ).map(|v| val_to_str(&v)).unwrap_or_default();
    let xlab = gn(a, "xlab").map(|v| val_to_str(&v)).unwrap_or_default();
    let ylab = gn(a, "ylab").map(|v| val_to_str(&v)).unwrap_or_default();

    let mut frag = String::new();

    // Plot frame.
    frag.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1" shape-rendering="crispEdges"/>"#,
        ox + ml, oy + mt, pw, ph
    ));

    let ng = groups.len() as f64;
    let bw = pw / ng * 0.6;
    let gap = pw / ng;

    for (i, (name, data)) in groups.iter().enumerate() {
        if data.len() < 2 { continue; }
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q1 = sorted[n / 4]; let median = sorted[n / 2]; let q3 = sorted[3 * n / 4];
        let min_val = sorted[0]; let max_val = sorted[n - 1];
        let iqr = q3 - q1;
        let lower_fence = (q1 - 1.5 * iqr).max(min_val);
        let upper_fence = (q3 + 1.5 * iqr).min(max_val);

        let cx = ox + ml + gap * i as f64 + gap / 2.0;
        let map_y = |v: f64| oy + mt + ph - (v - all_min) / range * ph;

        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}"/>"#, cx, map_y(lower_fence), cx, map_y(q1), col_border));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}"/>"#, cx, map_y(q3), cx, map_y(upper_fence), col_border));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}"/>"#, cx - bw / 4.0, map_y(lower_fence), cx + bw / 4.0, map_y(lower_fence), col_border));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}"/>"#, cx - bw / 4.0, map_y(upper_fence), cx + bw / 4.0, map_y(upper_fence), col_border));
        let by = map_y(q3); let bh = map_y(q1) - by;
        frag.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="{:.0}" height="{:.0}" fill="{}" stroke="{}"/>"#, cx - bw / 2.0, by, bw, bh, col_box, col_border));
        frag.push_str(&format!(r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}" stroke-width="2"/>"#, cx - bw / 2.0, map_y(median), cx + bw / 2.0, map_y(median), col_med));
        // Group name beneath the box.
        let label_y = oy + h - 8.0;
        frag.push_str(&format!(
            r#"<text x="{:.0}" y="{:.0}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10px" fill="{}">{}</text>"#,
            cx, label_y, opts.col_axis, escape_xml(name)
        ));
    }

    // Shared chrome — title / subtitle / xlab / ylab + y-axis ticks.
    // No x-axis ticks (group names already drawn above).
    let panel = PanelRect { ox, oy, w, h, ml, mt, pw, ph };
    frag.push_str(&render_chrome(&panel, &title, &sub, &xlab, &ylab, &opts));
    frag.push_str(&render_axis_ticks(&panel, 0.0, ng, all_min, all_max, ng.max(1.0), range, &opts));

    with_device(|d| d.svg_body.push_str(&frag));
    crate::device::finish_plot("boxplot.svg");
    Ok(RVal::Null)
}

/// `barplot(heights, names.arg=, main=)` — colour-cycled bars into the device.
pub fn bi_barplot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let heights: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let title = gn(a, "main").map(|v| val_to_str(&v)).unwrap_or("Barplot".into());
    let sub   = gn(a, "sub" ).map(|v| val_to_str(&v)).unwrap_or_default();
    let xlab  = gn(a, "xlab").map(|v| val_to_str(&v)).unwrap_or_default();
    let ylab  = gn(a, "ylab").map(|v| val_to_str(&v)).unwrap_or_default();
    let names = gn(a, "names.arg");

    let labels: Vec<String> = if let Some(RVal::Character(v, _)) = &names {
        v.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect()
    } else {
        (1..=heights.len()).map(|i| format!("{}", i)).collect()
    };

    let opts = LabelOpts::from_args(a);
    // `col` can be a single color OR a character vector that cycles
    // across bars. Default palette is similar to R's `palette()`.
    let col_arg = gn(a, "col");
    let palette: Vec<String> = match col_arg {
        Some(RVal::Character(v, _)) if !v.is_empty() => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "gray".into()))
            .collect(),
        Some(other) => vec![val_to_str(&other)],
        None => vec!["#3b82f6".into(), "#ef4444".into(), "#22c55e".into(),
                     "#f59e0b".into(), "#8b5cf6".into(), "#ec4899".into(),
                     "#06b6d4".into(), "#f97316".into()],
    };
    let col_border = gn(a, "border").map(|v| val_to_str(&v)).unwrap_or_else(|| "black".into());

    let (ox, oy, w, h) = begin_plot();
    let (ml, mr, mt, mb) = (60.0, 20.0, 36.0, 60.0);
    let pw = w - ml - mr;
    let ph = h - mt - mb;
    let raw_max = heights.iter().cloned().fold(0.0f64, f64::max);
    // 4 % y cushion so the tallest bar doesn't kiss the frame top.
    let max_h = raw_max * 1.04;
    let bw = pw / heights.len().max(1) as f64 * 0.8;
    let gap = pw / heights.len().max(1) as f64;

    let mut frag = String::new();

    // Plot frame.
    frag.push_str(&format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1" shape-rendering="crispEdges"/>"#,
        ox + ml, oy + mt, pw, ph
    ));

    for (i, &val) in heights.iter().enumerate() {
        let bh = if max_h > 0.0 { val / max_h * ph } else { 0.0 };
        let bx = ox + ml + gap * i as f64 + (gap - bw) / 2.0;
        let by = oy + mt + ph - bh;
        let color = &palette[i % palette.len()];
        frag.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="{}" stroke-width="0.5"/>"#,
            bx, by, bw, bh, color, col_border
        ));
        let label = labels.get(i).map(|s| s.as_str()).unwrap_or("");
        if !label.is_empty() {
            frag.push_str(&format!(
                r#"<text x="{:.0}" y="{:.0}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10px" fill="{}" transform="rotate(-30,{:.0},{:.0})">{}</text>"#,
                bx + bw / 2.0, oy + h - mb + 16.0, opts.col_axis,
                bx + bw / 2.0, oy + h - mb + 16.0, escape_xml(label)
            ));
        }
    }

    // Shared chrome — title, subtitle, xlab, ylab, y-axis ticks.
    let panel = PanelRect { ox, oy, w, h, ml, mt, pw, ph };
    frag.push_str(&render_chrome(&panel, &title, &sub, &xlab, &ylab, &opts));
    frag.push_str(&render_axis_ticks(&panel, 0.0, heights.len() as f64,
                                     0.0, max_h, heights.len().max(1) as f64,
                                     max_h.max(1.0), &opts));

    with_device(|d| d.svg_body.push_str(&frag));
    crate::device::finish_plot("barplot.svg");
    Ok(RVal::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::dev_off;
    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }

    #[test]
    fn plot_writes_svg() {
        dev_off();
        let a = vec![evarg(nums(&[1.0, 2.0, 3.0])), evarg(nums(&[4.0, 5.0, 6.0]))];
        let r = bi_plot(&a).unwrap();
        match r { RVal::Character(v, _) => assert_eq!(v[0].as_deref(), Some("plot.svg")), _ => panic!("plot must return path") }
        dev_off();
    }

    #[test]
    fn hist_returns_null_and_writes() {
        dev_off();
        let a = vec![evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))];
        let r = bi_hist(&a).unwrap();
        assert!(matches!(r, RVal::Null));
        dev_off();
    }

    #[test]
    fn boxplot_errors_with_no_groups() {
        dev_off();
        let r = bi_boxplot(&[]);
        assert!(r.is_err());
        dev_off();
    }

    #[test]
    fn barplot_returns_null() {
        dev_off();
        let a = vec![evarg(nums(&[3.0, 1.0, 4.0, 1.0, 5.0]))];
        let r = bi_barplot(&a).unwrap();
        assert!(matches!(r, RVal::Null));
        dev_off();
    }
}
