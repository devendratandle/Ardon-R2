//! R-style color helpers — `rgb()`, `gray()` / `grey()`, `hsv()`,
//! palette generators (`rainbow`, `heat.colors`, `terrain.colors`,
//! `cm.colors`, `topo.colors`), and `adjustcolor()`.
//!
//! Each returns a character vector of `"#RRGGBB"` or `"#RRGGBBAA"`
//! strings ready to drop into any plot function's `col=` /
//! `border=` argument. Same signatures and defaults as R for the
//! cases that come up in everyday plotting.

use std::sync::Arc;

use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal};

use crate::{gn, gv};

#[inline]
fn rchar(v: Vec<String>) -> RVal {
    let inner: Vec<Option<Arc<str>>> = v.into_iter().map(|s| Some(Arc::from(s.as_str()))).collect();
    RVal::Character(inner, Attrs::default())
}

#[inline]
fn hex2(b: u8) -> String { format!("{:02X}", b) }

fn clamp_u8_from_unit(x: f64) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// `rgb(r, g, b, alpha=1, maxColorValue=1)` — build hex strings from
/// numeric vectors. Vectors recycle to the longest, matching R.
/// Returns `"#RRGGBB"` if alpha is 1 throughout, `"#RRGGBBAA"`
/// otherwise.
pub fn bi_rgb(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let r: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let g: Vec<f64> = gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect();
    let b: Vec<f64> = gv(a, 2).as_reals()?.into_iter().filter_map(|x| x).collect();
    let alpha: Vec<f64> = gn(a, "alpha")
        .map(|v| v.as_reals().ok().unwrap_or_default().into_iter().filter_map(|x| x).collect())
        .unwrap_or_else(|| vec![1.0]);
    let max_val: f64 = gn(a, "maxColorValue")
        .and_then(|v| v.as_reals().ok()?.into_iter().next()?)
        .unwrap_or(1.0);
    if r.is_empty() || g.is_empty() || b.is_empty() {
        return Err(R2Err { msg: "rgb(): empty r/g/b vector".into(), kind: ErrKind::Runtime });
    }
    let n = r.len().max(g.len()).max(b.len()).max(alpha.len());
    let mut out = Vec::with_capacity(n);
    let any_alpha = alpha.iter().any(|&a| a < 1.0);
    for i in 0..n {
        let ri = clamp_u8_from_unit(r[i % r.len()] / max_val);
        let gi = clamp_u8_from_unit(g[i % g.len()] / max_val);
        let bi = clamp_u8_from_unit(b[i % b.len()] / max_val);
        if any_alpha {
            // alpha is always taken as a 0..1 fraction regardless of
            // maxColorValue (R's behaviour).
            let ai = clamp_u8_from_unit(alpha[i % alpha.len()]);
            out.push(format!("#{}{}{}{}", hex2(ri), hex2(gi), hex2(bi), hex2(ai)));
        } else {
            out.push(format!("#{}{}{}", hex2(ri), hex2(gi), hex2(bi)));
        }
    }
    Ok(rchar(out))
}

/// `gray(level, alpha=1)` / `grey(...)` — grayscale 0..1.
pub fn bi_gray(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let levels: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let alpha: Option<f64> = gn(a, "alpha").and_then(|v| v.as_reals().ok()?.into_iter().next()?);
    let out: Vec<String> = levels.iter().map(|&lv| {
        let g = clamp_u8_from_unit(lv);
        match alpha {
            Some(a) if a < 1.0 => format!("#{}{}{}{}", hex2(g), hex2(g), hex2(g),
                                          hex2(clamp_u8_from_unit(a))),
            _ => format!("#{}{}{}", hex2(g), hex2(g), hex2(g)),
        }
    }).collect();
    Ok(rchar(out))
}

// ─── HSV → RGB helper ─────────────────────────────────────────────

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = (h % 1.0 + 1.0) % 1.0;    // wrap into 0..1
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    (clamp_u8_from_unit(r), clamp_u8_from_unit(g), clamp_u8_from_unit(b))
}

/// `hsv(h, s=1, v=1, alpha=1)` — vectorised hue / saturation / value.
pub fn bi_hsv(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let h: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let s: Vec<f64> = if a.len() > 1 && a[1].name.is_none() {
        gv(a, 1).as_reals()?.into_iter().filter_map(|x| x).collect()
    } else { vec![1.0] };
    let v: Vec<f64> = if a.len() > 2 && a[2].name.is_none() {
        gv(a, 2).as_reals()?.into_iter().filter_map(|x| x).collect()
    } else { vec![1.0] };
    let alpha: Vec<f64> = gn(a, "alpha")
        .map(|v| v.as_reals().ok().unwrap_or_default().into_iter().filter_map(|x| x).collect())
        .unwrap_or_else(|| vec![1.0]);
    let n = h.len().max(s.len()).max(v.len()).max(alpha.len());
    let any_alpha = alpha.iter().any(|&a| a < 1.0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (r, g, b) = hsv_to_rgb(h[i % h.len()], s[i % s.len()], v[i % v.len()]);
        if any_alpha {
            let ai = clamp_u8_from_unit(alpha[i % alpha.len()]);
            out.push(format!("#{}{}{}{}", hex2(r), hex2(g), hex2(b), hex2(ai)));
        } else {
            out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
        }
    }
    Ok(rchar(out))
}

// ─── Palette generators ───────────────────────────────────────────

fn palette_n(a: &[EvalArg]) -> Result<usize, R2Err> {
    gv(a, 0).as_reals()?.into_iter().next().flatten()
        .map(|x| (x as usize).max(1))
        .ok_or_else(|| R2Err { msg: "palette(): need an integer n".into(), kind: ErrKind::Runtime })
}

/// `rainbow(n, s=1, v=1, start=0, end=max(1,n-1)/n)` — n equally-
/// spaced hues. Matches R's defaults.
pub fn bi_rainbow(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = palette_n(a)?;
    let s = gn(a, "s")    .and_then(|v| v.as_reals().ok()?.into_iter().next()?).unwrap_or(1.0);
    let v = gn(a, "v")    .and_then(|v| v.as_reals().ok()?.into_iter().next()?).unwrap_or(1.0);
    let start = gn(a, "start").and_then(|v| v.as_reals().ok()?.into_iter().next()?).unwrap_or(0.0);
    let end   = gn(a, "end")  .and_then(|v| v.as_reals().ok()?.into_iter().next()?)
                .unwrap_or_else(|| (n as f64 - 1.0).max(1.0) / n as f64);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n == 1 { start } else { start + (end - start) * i as f64 / (n as f64 - 1.0) };
        let (r, g, b) = hsv_to_rgb(t, s, v);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    Ok(rchar(out))
}

/// `heat.colors(n)` — red → orange → yellow → white. Matches R's
/// HSV ramp: hue 0 → 1/6, saturation falling to 0 near the top end.
pub fn bi_heat_colors(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = palette_n(a)?;
    let n_main = (n * 3 / 4).max(1);
    let n_tail = n - n_main;
    let mut out = Vec::with_capacity(n);
    for i in 0..n_main {
        let h = (i as f64) / (n_main as f64) * (1.0 / 6.0);
        let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    for i in 0..n_tail {
        // Hue stays at 1/6 (yellow), saturation falls to 0 (white).
        let s = 1.0 - (i + 1) as f64 / (n_tail as f64 + 1.0);
        let (r, g, b) = hsv_to_rgb(1.0 / 6.0, s, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    Ok(rchar(out))
}

/// `terrain.colors(n)` — green → yellow → tan → white.
pub fn bi_terrain_colors(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = palette_n(a)?;
    let n_main = (n / 2).max(1);
    let n_tail = n - n_main;
    let mut out = Vec::with_capacity(n);
    for i in 0..n_main {
        // Hue 4/12 (green) → 2/12 (yellow).
        let h = 4.0 / 12.0 - (i as f64 / (n_main as f64).max(1.0)) * (2.0 / 12.0);
        let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    for i in 0..n_tail {
        // Hue 2/12 (yellow) → 0 (red), saturation falling to 0 (white).
        let h = 2.0 / 12.0 - (i as f64 / (n_tail as f64).max(1.0)) * (2.0 / 12.0);
        let s = 1.0 - (i + 1) as f64 / (n_tail as f64 + 1.0);
        let (r, g, b) = hsv_to_rgb(h.max(0.0), s, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    Ok(rchar(out))
}

/// `topo.colors(n)` — blue → green → yellow → tan.
pub fn bi_topo_colors(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = palette_n(a)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Hue ramps 2/3 (blue) down to 1/6 (yellow).
        let t = i as f64 / (n as f64 - 1.0).max(1.0);
        let h = (2.0 / 3.0) - t * (2.0 / 3.0 - 1.0 / 6.0);
        let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    Ok(rchar(out))
}

/// `cm.colors(n)` — cyan → magenta. Matches R's diverging palette.
pub fn bi_cm_colors(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = palette_n(a)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0).max(1.0);
        // Cyan hsv ≈ 0.5; magenta hsv ≈ 0.833. Saturation full.
        let h = 0.5 + t * (0.833 - 0.5);
        let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
        out.push(format!("#{}{}{}", hex2(r), hex2(g), hex2(b)));
    }
    Ok(rchar(out))
}

/// `adjustcolor(col, alpha.f=NA)` — apply an alpha factor (0..1) to a
/// vector of hex colors. Names ("red", "blue", ...) pass through
/// untouched if they don't start with `#`.
pub fn bi_adjustcolor(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let cols: Vec<String> = match gv(a, 0) {
        RVal::Character(v, _) => v.iter()
            .map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default())
            .collect(),
        other => vec![crate::val_to_str(&other)],
    };
    let alpha_f: Option<f64> = gn(a, "alpha.f")
        .and_then(|v| v.as_reals().ok()?.into_iter().next()?);
    let out: Vec<String> = cols.into_iter().map(|c| {
        if !c.starts_with('#') { return c; }
        match (c.len(), alpha_f) {
            (7, Some(a)) => format!("{}{}", &c[..7], hex2(clamp_u8_from_unit(a))),
            (9, Some(a)) => format!("#{}{}{}{}", &c[1..3], &c[3..5], &c[5..7],
                                    hex2(clamp_u8_from_unit(a))),
            _ => c,
        }
    }).collect();
    Ok(rchar(out))
}
