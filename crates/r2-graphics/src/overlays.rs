//! Overlay builtins — Phase R.3 / Phase R.G.
//!
//! `lines`, `points`, `abline`, `legend` — each appends an SVG fragment
//! to the thread-local `PlotDevice`. The "is a plot open" precondition
//! is checked against the device's in-memory `has_plot` flag, not by
//! reading `plot.svg` from the filesystem. The old file-state model
//! was racy under cargo's parallel test execution; the device-state
//! model is not.

use crate::device::{append_svg, save_to_file, with_device, PlotCoords};
use crate::plots::{color_vec, escape_xml, point_symbol};
use crate::{gn, gv, val_to_str};
use r2_types::{EvalArg, R2Err, RVal};

/// The active plot's coordinate system, or a sensible fallback if no plot
/// has established one yet. Overlays map data → pixels through this so they
/// align with the base plot (R's incremental graphics model).
fn coords() -> PlotCoords {
    with_device(|d| d.coords).unwrap_or(PlotCoords {
        px0: 60.0, py0: 36.0, pw: 520.0, ph: 324.0,
        xmin: 0.0, xmax: 1.0, ymin: 0.0, ymax: 1.0,
    })
}

#[inline]
fn argf(a: &[EvalArg], name: &str) -> Option<f64> {
    gn(a, name).and_then(|v| v.scalar_f64().ok().flatten())
}
#[inline]
fn argi(a: &[EvalArg], name: &str) -> Option<i32> {
    argf(a, name).map(|x| x as i32)
}
/// Default colour for an overlay: explicit `col=` else the device `par(col)`.
fn overlay_col(a: &[EvalArg]) -> String {
    gn(a, "col").map(|v| val_to_str(&v))
        .unwrap_or_else(|| with_device(|d| d.params.col.clone()))
}

/// `lines(x, y, col=, lwd=, lty=)` — connect (x,y) as a polyline in the
/// base plot's data coordinates.
pub fn bi_lines(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let col = overlay_col(a);
    let lwd = argf(a, "lwd").unwrap_or(1.0);
    let dash = if argi(a, "lty") == Some(2) { r#" stroke-dasharray="5,5""# } else { "" };
    let c = coords();

    let n = x.len().min(y.len());
    let mut elems = String::new();
    if n >= 2 {
        let path: String = (0..n).map(|i| {
            let (px, py) = c.to_px(x[i], y[i]);
            format!("{}{:.1},{:.1}", if i == 0 { "M" } else { " L" }, px, py)
        }).collect();
        elems = format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="{:.1}"{}/>"#,
            path, col, lwd, dash);
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `points(x, y, col=, pch=, cex=)` — markers in the base plot's data
/// coordinates. `col` recycles over the points; `pch` selects the symbol.
pub fn bi_points(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let dev_col = with_device(|d| d.params.col.clone());
    let cols = color_vec(gn(a, "col"), &dev_col);
    let pch = argi(a, "pch").unwrap_or_else(|| with_device(|d| d.params.pch));
    let cex = argf(a, "cex").unwrap_or(1.0);
    let c = coords();

    let mut elems = String::new();
    for i in 0..x.len().min(y.len()) {
        let (px, py) = c.to_px(x[i], y[i]);
        let col = &cols[i % cols.len()];
        elems.push_str(&point_symbol(px, py, 3.0 * cex, pch, col));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `abline(a=intercept, b=slope)`, `abline(h=)`, or `abline(v=)` — a
/// reference line in the base plot's data coordinates.
pub fn bi_abline(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let col = overlay_col(a);
    let lwd = argf(a, "lwd").unwrap_or(1.0);
    let dash = if argi(a, "lty") == Some(2) { r#" stroke-dasharray="5,5""# } else { "" };
    let c = coords();

    let elem = if let Some(h) = argf(a, "h") {
        let (_, py) = c.to_px(c.xmin, h);
        let (x1, _) = c.to_px(c.xmin, h);
        let (x2, _) = c.to_px(c.xmax, h);
        format!(r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.1}"{}/>"#,
                x1, py, x2, py, col, lwd, dash)
    } else if let Some(v) = argf(a, "v") {
        let (px, y1) = c.to_px(v, c.ymin);
        let (_, y2) = c.to_px(v, c.ymax);
        format!(r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.1}"{}/>"#,
                px, y1, px, y2, col, lwd, dash)
    } else {
        let intercept = argf(a, "a").or_else(|| gv(a, 0).scalar_f64().ok().flatten()).unwrap_or(0.0);
        let slope     = argf(a, "b").or_else(|| gv(a, 1).scalar_f64().ok().flatten()).unwrap_or(1.0);
        // Draw across the visible x-range at the data y = a + b*x.
        let (x1, y1) = c.to_px(c.xmin, intercept + slope * c.xmin);
        let (x2, y2) = c.to_px(c.xmax, intercept + slope * c.xmax);
        format!(r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.1}"{}/>"#,
                x1, y1, x2, y2, col, lwd, dash)
    };
    append_svg(&elem)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `text(x, y, labels, col=, cex=, pos=)` — data-coordinate text. `pos`
/// 1/2/3/4 nudges below/left/above/right of the point (else centred).
pub fn bi_text(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let labels: Vec<String> = match gn(a, "labels").unwrap_or_else(|| gv(a, 2)) {
        RVal::Character(v, _) => v.iter().map(|s| s.as_ref().map(|x| x.to_string()).unwrap_or_default()).collect(),
        other => other.as_reals().map(|r| r.into_iter().map(|o| o.map(|n| fmt_num(n)).unwrap_or_default()).collect()).unwrap_or_default(),
    };
    let col = overlay_col(a);
    let cex = argf(a, "cex").unwrap_or(1.0);
    let pos = argi(a, "pos");
    // adj: horizontal justification (0 = left, 0.5 = centre, 1 = right).
    let adj = argf(a, "adj");
    // font: 1 plain, 2 bold, 3 italic, 4 bold+italic.
    let font_attr = match argi(a, "font") {
        Some(2) => r#" font-weight="bold""#,
        Some(3) => r#" font-style="italic""#,
        Some(4) => r#" font-weight="bold" font-style="italic""#,
        _ => "",
    };
    let fs = 12.0 * cex;
    let c = coords();

    let mut elems = String::new();
    for i in 0..x.len().min(y.len()) {
        let label = labels.get(i % labels.len().max(1)).cloned().unwrap_or_default();
        if label.is_empty() && labels.is_empty() { continue; }
        let (mut px, mut py) = c.to_px(x[i], y[i]);
        // `adj` (if given) sets horizontal anchor; else `pos` nudges + anchors.
        let mut anchor = match adj {
            Some(a) if a <= 0.25 => "start",
            Some(a) if a >= 0.75 => "end",
            Some(_) => "middle",
            None => "middle",
        };
        let off = fs * 0.9;
        match pos {
            Some(1) => { py += off; }
            Some(2) => { px -= off; anchor = "end"; }
            Some(3) => { py -= off; }
            Some(4) => { px += off; anchor = "start"; }
            _ => { py += fs * 0.35; } // vertical-centre the glyph
        }
        elems.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="{}" font-family="Arial, Helvetica, sans-serif" font-size="{:.1}px"{} fill="{}">{}</text>"#,
            px, py, anchor, fs, font_attr, col, escape_xml(&label)));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `title(main=, sub=, xlab=, ylab=)` — add titles to the current plot,
/// positioned relative to its plotting region.
pub fn bi_title(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let c = coords();
    let col = gn(a, "col.main").map(|v| val_to_str(&v))
        .unwrap_or_else(|| with_device(|d| d.params.fg.clone()));
    let mut elems = String::new();
    let cx = c.px0 + c.pw / 2.0;
    if let Some(main) = gn(a, "main").map(|v| val_to_str(&v)).filter(|s| !s.is_empty()) {
        elems.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15px" font-weight="bold" fill="{}">{}</text>"#,
            cx, c.py0 - 12.0, col, escape_xml(&main)));
    }
    if let Some(xlab) = gn(a, "xlab").map(|v| val_to_str(&v)).filter(|s| !s.is_empty()) {
        elems.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12px" fill="{}">{}</text>"#,
            cx, c.py0 + c.ph + 30.0, col, escape_xml(&xlab)));
    }
    if let Some(ylab) = gn(a, "ylab").map(|v| val_to_str(&v)).filter(|s| !s.is_empty()) {
        let yx = c.px0 - 40.0;
        let yy = c.py0 + c.ph / 2.0;
        elems.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" transform="rotate(-90 {:.1} {:.1})" font-family="Arial, Helvetica, sans-serif" font-size="12px" fill="{}">{}</text>"#,
            yx, yy, yx, yy, col, escape_xml(&ylab)));
    }
    if let Some(sub) = gn(a, "sub").map(|v| val_to_str(&v)).filter(|s| !s.is_empty()) {
        elems.push_str(&format!(
            r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11px" fill="{}">{}</text>"#,
            cx, c.py0 + c.ph + 48.0, col, escape_xml(&sub)));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `rect(xleft, ybottom, xright, ytop, col=, border=)` — data-coordinate
/// rectangle(s); coordinate vectors recycle.
pub fn bi_rect(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let xl: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let yb: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let xr: Vec<f64> = gv(a, 2).as_reals()?.into_iter().filter_map(|x| x).collect();
    let yt: Vec<f64> = gv(a, 3).as_reals()?.into_iter().filter_map(|x| x).collect();
    if xl.is_empty() || yb.is_empty() || xr.is_empty() || yt.is_empty() {
        return Err(R2Err { msg: "rect(): need xleft, ybottom, xright, ytop".into(),
                           kind: r2_types::ErrKind::Runtime });
    }
    let fill = gn(a, "col").map(|v| val_to_str(&v)).unwrap_or_else(|| "none".into());
    let border = gn(a, "border").map(|v| val_to_str(&v))
        .unwrap_or_else(|| with_device(|d| d.params.fg.clone()));
    let lwd = argf(a, "lwd").unwrap_or(1.0);
    let c = coords();
    let n = xl.len().max(yb.len()).max(xr.len()).max(yt.len());
    let mut elems = String::new();
    for i in 0..n {
        let (x1, y1) = c.to_px(xl[i % xl.len()], yb[i % yb.len()]);
        let (x2, y2) = c.to_px(xr[i % xr.len()], yt[i % yt.len()]);
        let (rx, ry) = (x1.min(x2), y1.min(y2));
        let (rw, rh) = ((x1 - x2).abs(), (y1 - y2).abs());
        elems.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="{}" stroke-width="{:.1}"/>"#,
            rx, ry, rw, rh, fill, border, lwd));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// `axis(side, at=, labels=)` — draw an axis on side 1=bottom, 2=left,
/// 3=top, 4=right with tick marks + labels at the given data positions.
pub fn bi_axis(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let side = gv(a, 0).scalar_f64().ok().flatten().unwrap_or(1.0) as i32;
    let at: Vec<f64> = gn(a, "at").map(|v| v.as_reals().ok())
        .flatten().map(|r| r.into_iter().flatten().collect()).unwrap_or_default();
    let labels: Vec<String> = match gn(a, "labels") {
        Some(RVal::Character(v, _)) => v.iter().map(|s| s.as_ref().map(|x| x.to_string()).unwrap_or_default()).collect(),
        _ => at.iter().map(|v| fmt_num(*v)).collect(),
    };
    let col = overlay_col(a);
    let c = coords();
    let mut elems = String::new();
    let tick = 5.0;
    for (i, &v) in at.iter().enumerate() {
        let label = labels.get(i).cloned().unwrap_or_default();
        match side {
            2 | 4 => {
                let (_, py) = c.to_px(c.xmin, v);
                let x_edge = if side == 2 { c.px0 } else { c.px0 + c.pw };
                let dir = if side == 2 { -1.0 } else { 1.0 };
                elems.push_str(&format!(r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="1"/>"#,
                    x_edge, py, x_edge + dir * tick, py, col));
                let (tx, anchor) = if side == 2 { (x_edge - tick - 3.0, "end") } else { (x_edge + tick + 3.0, "start") };
                elems.push_str(&format!(r#"<text x="{:.1}" y="{:.1}" text-anchor="{}" font-family="Arial" font-size="9px" fill="{}">{}</text>"#,
                    tx, py + 3.0, anchor, col, escape_xml(&label)));
            }
            _ => {
                let (px, _) = c.to_px(v, c.ymin);
                let y_edge = if side == 3 { c.py0 } else { c.py0 + c.ph };
                let dir = if side == 3 { -1.0 } else { 1.0 };
                elems.push_str(&format!(r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="1"/>"#,
                    px, y_edge, px, y_edge + dir * tick, col));
                let ty = if side == 3 { y_edge - tick - 3.0 } else { y_edge + tick + 10.0 };
                elems.push_str(&format!(r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-family="Arial" font-size="9px" fill="{}">{}</text>"#,
                    px, ty, col, escape_xml(&label)));
            }
        }
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    Ok(RVal::Null)
}

/// Compact numeric formatting for auto tick/text labels.
fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 { format!("{}", v as i64) } else { format!("{:.2}", v) }
}

/// `legend("topleft"|..., legend=, col=)` — coloured swatches + labels.
pub fn bi_legend(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let pos = val_to_str(&gv(a, 0));
    let legend_items = gn(a, "legend").unwrap_or(RVal::Null);
    let col = gn(a, "col");

    let labels: Vec<String> = match &legend_items {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
        _ => vec!["Series 1".into()],
    };
    let colors: Vec<String> = match &col {
        Some(RVal::Character(v, _)) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or("black".into())).collect(),
        _ => vec!["black".into(), "red".into(), "blue".into(), "green".into()],
    };

    // Size the box to its content and sit it FLUSH in the requested corner
    // of the device (relative to the actual canvas), so it doesn't float over
    // the middle of a panel. Opaque white background keeps it readable.
    let (w, h) = with_device(|d| (d.width, d.height));
    let longest = labels.iter().map(|l| l.chars().count()).max().unwrap_or(6);
    let row_h = 18.0;
    let box_w = 28.0 + longest as f64 * 7.0;
    let box_h = labels.len() as f64 * row_h + 12.0;
    let m = 8.0; // margin from the canvas edge
    let (bx, by) = match pos.as_str() {
        "topleft"     => (m, m),
        "bottomleft"  => (m, (h - box_h - m).max(m)),
        "bottomright" => ((w - box_w - m).max(m), (h - box_h - m).max(m)),
        _             => ((w - box_w - m).max(m), m), // topright (default)
    };

    let mut elems = format!(
        r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="white" stroke="black" stroke-width="1"/>"#,
        bx, by, box_w, box_h
    );
    for (i, label) in labels.iter().enumerate() {
        let c = colors.get(i).map(|s| s.as_str()).unwrap_or("black");
        let row_y = by + 6.0 + i as f64 * row_h;
        elems.push_str(&format!(r#"<rect x="{:.1}" y="{:.1}" width="12" height="12" fill="{}"/>"#, bx + 6.0, row_y + 2.0, c));
        elems.push_str(&format!(r#"<text x="{:.1}" y="{:.1}" font-family="Arial, Helvetica, sans-serif" font-size="11px" fill="black">{}</text>"#, bx + 24.0, row_y + 12.0, escape_xml(label)));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    soutln!("Legend added to plot.svg");
    Ok(RVal::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::dev_off;
    use r2_types::Attrs;

    #[test]
    fn lines_errs_when_no_plot_open() {
        // No filesystem state involved — device-only precondition.
        dev_off();
        let a = vec![
            EvalArg { name: None, value: RVal::Numeric(vec![Some(1.0)].into(), Attrs::default()) },
            EvalArg { name: None, value: RVal::Numeric(vec![Some(2.0)].into(), Attrs::default()) },
        ];
        let r = bi_lines(&a);
        assert!(r.is_err(), "lines() must error when no plot is open");
        dev_off();
    }
}
