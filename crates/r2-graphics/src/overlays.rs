//! Overlay builtins — Phase R.3 / Phase R.G.
//!
//! `lines`, `points`, `abline`, `legend` — each appends an SVG fragment
//! to the thread-local `PlotDevice`. The "is a plot open" precondition
//! is checked against the device's in-memory `has_plot` flag, not by
//! reading `plot.svg` from the filesystem. The old file-state model
//! was racy under cargo's parallel test execution; the device-state
//! model is not.

use crate::device::{append_svg, save_to_file};
use crate::{gn, gv, val_to_str};
use r2_types::{EvalArg, R2Err, RVal};

/// `lines(x, y, col=)` — connects (x,y) sequence as a polyline.
pub fn bi_lines(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let col = gn(a, "col").map(|v| val_to_str(&v)).unwrap_or("red".into());

    let mut elems = String::new();
    for i in 0..x.len().saturating_sub(1) {
        elems.push_str(&format!(
            r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="2"/>"#,
            60.0 + x[i] * 10.0,
            370.0 - y[i] * 10.0,
            60.0 + x[i + 1] * 10.0,
            370.0 - y[i + 1] * 10.0,
            col
        ));
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    soutln!("Lines added to plot.svg");
    Ok(RVal::Null)
}

/// `points(x, y, col=, pch=)` — discrete point markers.
pub fn bi_points(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let x: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let y: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let col = gn(a, "col").map(|v| val_to_str(&v)).unwrap_or("red".into());
    let pch = gn(a, "pch").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0) as i32;

    let mut elems = String::new();
    for i in 0..x.len().min(y.len()) {
        let px = 60.0 + x[i] * 10.0;
        let py = 370.0 - y[i] * 10.0;
        match pch {
            0 => elems.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="6" height="6" fill="none" stroke="{}"/>"#, px - 3.0, py - 3.0, col)),
            2 => elems.push_str(&format!(r#"<polygon points="{:.0},{:.0} {:.0},{:.0} {:.0},{:.0}" fill="none" stroke="{}"/>"#, px, py - 4.0, px - 4.0, py + 3.0, px + 4.0, py + 3.0, col)),
            _ => elems.push_str(&format!(r#"<circle cx="{:.0}" cy="{:.0}" r="3" fill="{}"/>"#, px, py, col)),
        }
    }
    append_svg(&elems)?;
    let _ = save_to_file("plot.svg");
    soutln!("Points added to plot.svg");
    Ok(RVal::Null)
}

/// `abline(a=intercept, b=slope)` or `abline(h=)` or `abline(v=)`.
pub fn bi_abline(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let col = gn(a, "col").map(|v| val_to_str(&v)).unwrap_or("red".into());
    let lty = gn(a, "lty").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0) as i32;
    let dash = if lty == 2 { r#" stroke-dasharray="5,5""# } else { "" };

    let elem = if let (Some(h_val), _) = (gn(a, "h"), gn(a, "v")) {
        let h = h_val.scalar_f64()?.unwrap_or(0.0);
        let py = 370.0 - h * 10.0;
        format!(r#"<line x1="60" y1="{:.0}" x2="580" y2="{:.0}" stroke="{}"{}/>"#, py, py, col, dash)
    } else if let Some(v_val) = gn(a, "v") {
        let v = v_val.scalar_f64()?.unwrap_or(0.0);
        let px = 60.0 + v * 10.0;
        format!(r#"<line x1="{:.0}" y1="30" x2="{:.0}" y2="370" stroke="{}"{}/>"#, px, px, col, dash)
    } else {
        let intercept = gn(a, "a").or_else(|| Some(gv(a, 0)))
            .and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);
        let slope = gn(a, "b").or_else(|| Some(gv(a, 1)))
            .and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(1.0);
        let x1 = 0.0; let x2 = 50.0;
        let y1 = intercept + slope * x1;
        let y2 = intercept + slope * x2;
        format!(
            r#"<line x1="{:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="{}"{} stroke-width="2"/>"#,
            60.0 + x1 * 10.0, 370.0 - y1 * 10.0,
            60.0 + x2 * 10.0, 370.0 - y2 * 10.0,
            col, dash
        )
    };
    append_svg(&elem)?;
    let _ = save_to_file("plot.svg");
    soutln!("Line added to plot.svg");
    Ok(RVal::Null)
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

    let (lx, ly) = match pos.as_str() {
        "topleft" => (70.0, 45.0),
        "topright" => (420.0, 45.0),
        "bottomleft" => (70.0, 330.0),
        "bottomright" => (420.0, 330.0),
        _ => (420.0, 45.0),
    };

    let mut elems = format!(
        r#"<rect x="{:.0}" y="{:.0}" width="140" height="{}" fill="white" stroke="black" stroke-width="0.5"/>"#,
        lx - 5.0, ly - 15.0, labels.len() * 20 + 10
    );
    for (i, label) in labels.iter().enumerate() {
        let c = colors.get(i).map(|s| s.as_str()).unwrap_or("black");
        let yp = ly + i as f64 * 20.0;
        elems.push_str(&format!(r#"<rect x="{:.0}" y="{:.0}" width="12" height="12" fill="{}"/>"#, lx, yp - 9.0, c));
        elems.push_str(&format!(r#"<text x="{:.0}" y="{:.0}" font-size="11">{}</text>"#, lx + 18.0, yp, label));
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
