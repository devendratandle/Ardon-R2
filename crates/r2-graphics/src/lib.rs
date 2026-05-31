//! R2 Graphics — domain crate for SVG-based plotting builtins (Phase R.3).
//!
//! Per docs/ARCHITECTURE.md §5 Phase R.3, this crate hosts:
//!   - plot, hist, boxplot, barplot — primary data-shape plot builders
//!   - lines, points, abline, legend — overlays that mutate the existing
//!     `plot.svg` file
//!
//! All builtins follow the locked pure pattern:
//! `fn(&[EvalArg]) -> Result<RVal, R2Err>`. r2-engine wraps via 1-line
//! adapter to its `BuiltinFn` shape.
//!
//! The legacy `plot_scatter_svg` / `hist_svg` SVG primitives remain
//! available for direct callers (e.g. r2-pkg fixtures).
//!
//! Model-aware dispatch (`plot(lm)`, `plot(gbm)`, ...) stays in r2-engine
//! using the split-handler pattern: engine inspects for `RVal::TypeInstance`
//! first; if not a known model, delegates to `plots::bi_plot`.

use r2_types::{EvalArg, R2Err, RVal};
use std::fmt::Write;

pub mod colors;
pub mod device;
pub mod overlays;
pub mod params;
pub mod plots;
pub mod server;

// ─────────────────────────────────────────────────────────────────────
// Shared internal helpers (scope: this crate only).
// ─────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn gv(args: &[EvalArg], i: usize) -> RVal {
    args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
pub(crate) fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter()
        .find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

/// Mirrors `r2_engine::val_to_str` — flat scalar/vector renderer.
pub(crate) fn val_to_str(v: &RVal) -> String {
    match v {
        RVal::Numeric(v, _) => v.iter().map(|x| match x {
            Some(n) => r2_types::fmt_num(*n),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Integer(v, _) => v.iter().map(|x| match x {
            Some(n) => format!("{}", n),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Character(v, _) => v.iter().map(|x| match x {
            Some(s) => s.to_string(),
            None => "NA".into(),
        }).collect::<Vec<_>>().join(" "),
        RVal::Logical(v, _) => v.iter().map(|x| match x {
            Some(true) => "TRUE",
            Some(false) => "FALSE",
            None => "NA",
        }).collect::<Vec<_>>().join(" "),
        RVal::Null => "NULL".into(),
        _ => format!("<{}>", v.type_name()),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Legacy SVG helpers (retained — no engine dependency).
// ─────────────────────────────────────────────────────────────────────

pub struct PlotConfig {
    pub width: u32, pub height: u32, pub title: String,
    pub xlab: String, pub ylab: String, pub col: String,
}
impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig { width: 800, height: 600, title: String::new(),
            xlab: String::new(), ylab: String::new(), col: "#2C3E50".into() }
    }
}

pub fn plot_scatter_svg(x: &[f64], y: &[f64], cfg: &PlotConfig) -> String {
    let m = 60.0; let w = cfg.width as f64 - 2.0*m; let h = cfg.height as f64 - 2.0*m;
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xr = if (xmax-xmin).abs() < 1e-10 { 1.0 } else { xmax-xmin };
    let yr = if (ymax-ymin).abs() < 1e-10 { 1.0 } else { ymax-ymin };
    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" style="background:white">"#, cfg.width, cfg.height);
    svg.push('\n');
    if !cfg.title.is_empty() { let _ = writeln!(svg, r#"<text x="{}" y="30" text-anchor="middle" font-size="16">{}</text>"#, cfg.width as f64/2.0, cfg.title); }
    for (xi, yi) in x.iter().zip(y) {
        let px = m + (xi - xmin)/xr * w; let py = cfg.height as f64 - m - (yi - ymin)/yr * h;
        let _ = writeln!(svg, r#"<circle cx="{:.1}" cy="{:.1}" r="4" fill="{}" opacity="0.7"/>"#, px, py, cfg.col);
    }
    svg.push_str("</svg>"); svg
}

pub fn hist_svg(values: &[f64], bins: usize, cfg: &PlotConfig) -> String {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max-min).abs() < 1e-10 { 1.0 } else { max-min };
    let bw = range / bins as f64;
    let mut counts = vec![0usize; bins];
    for v in values { let i = ((v-min)/bw).floor() as usize; counts[i.min(bins-1)] += 1; }
    let mc = *counts.iter().max().unwrap_or(&1);
    let m = 60.0; let w = cfg.width as f64 - 2.0*m; let h = cfg.height as f64 - 2.0*m;
    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" style="background:white">"#, cfg.width, cfg.height);
    svg.push('\n');
    let bar_w = w / bins as f64;
    for (i, c) in counts.iter().enumerate() {
        let bh = (*c as f64 / mc as f64) * h;
        let x = m + i as f64 * bar_w; let y = cfg.height as f64 - m - bh;
        let _ = writeln!(svg, r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="white"/>"#, x, y, bar_w, bh, cfg.col);
    }
    svg.push_str("</svg>"); svg
}

// ─────────────────────────────────────────────────────────────────────
// Builtins registry (Phase R.3).
// ─────────────────────────────────────────────────────────────────────

/// Returns this crate's exported builtins as `(name, fn-pointer)` pairs.
/// r2-engine adapts the signature at registration time. Note:
/// r2-engine retains its own `bi_plot` wrapper for model-aware dispatch
/// (split-handler pattern), then falls through to `plots::bi_plot`.
pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("plot",      plots::bi_plot),
        ("hist",      plots::bi_hist),
        ("boxplot",   plots::bi_boxplot),
        ("barplot",   plots::bi_barplot),
        ("lines",     overlays::bi_lines),
        ("points",    overlays::bi_points),
        ("abline",    overlays::bi_abline),
        ("legend",    overlays::bi_legend),
        // Phase R.G — explicit device controls.
        ("par",       params::bi_par),
        ("dev.off",   params::bi_dev_off),
        ("save_plot", params::bi_save_plot),
        // Phase R.G.2 — built-in HTTP plot viewer.
        ("dev.view",  params::bi_dev_view),
    ]
}
