//! `par()` — graphical-parameter getter and setter.
//!
//! Mirrors R's `par()` API:
//!   - `par()`            → returns current params as a named list
//!   - `par("col")`       → returns just that single param's value
//!   - `par(col="red")`   → sets `col`; returns the *previous* values
//!     of all changed params so the user can restore via `par(oldpar)`
//!   - `par(mfrow=c(2,2))` → enables multi-panel layout
//!
//! The split-handler pattern keeps this function pure (no engine
//! reference) — the device thread-local in `device.rs` is the state.

use std::sync::Arc;

use r2_types::{Attrs, ErrKind, EvalArg, R2Err, RVal};

use crate::device::{with_device, PlotParams};
use crate::{gn, gv, val_to_str};

#[inline]
fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(Arc::from(s))], Attrs::default())
}
#[inline]
fn rnum(x: f64) -> RVal {
    RVal::Numeric(vec![Some(x)].into(), Attrs::default())
}
#[inline]
fn rint(n: i32) -> RVal {
    RVal::Integer(vec![Some(n)].into(), Attrs::default())
}
#[inline]
fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default())
}

/// Render a single parameter as an `RVal` (for `par("name")` queries
/// and for the "previous values" return of `par(name=value)` calls).
fn param_value(p: &PlotParams, name: &str) -> Option<RVal> {
    Some(match name {
        "mfrow" => match p.mfrow {
            Some((r, c)) => rnums(&[r as f64, c as f64]),
            None => RVal::Null,
        },
        "mfcol" => match p.mfcol {
            Some((r, c)) => rnums(&[r as f64, c as f64]),
            None => RVal::Null,
        },
        "mar"      => rnums(&p.mar),
        "oma"      => rnums(&p.oma),
        "cex"      => rnum(p.cex),
        "cex.axis" => rnum(p.cex_axis),
        "cex.lab"  => rnum(p.cex_lab),
        "cex.main" => rnum(p.cex_main),
        "col"      => rstr(&p.col),
        "bg"       => rstr(&p.bg),
        "fg"       => rstr(&p.fg),
        "lty"      => rstr(&p.lty),
        "lwd"      => rnum(p.lwd),
        "pch"      => rint(p.pch),
        "las"      => rint(p.las),
        "new"      => RVal::Logical(vec![Some(p.new)].into(), Attrs::default()),
        _ => return None,
    })
}

/// Parse a 2-element numeric vector into `(u32, u32)` — used by `mfrow`/`mfcol`.
fn parse_pair(v: &RVal, name: &str) -> Result<(u32, u32), R2Err> {
    let reals = v.as_reals()?;
    if reals.len() < 2 {
        return Err(R2Err {
            msg: format!("par({}=...) expects a length-2 numeric vector", name),
            kind: ErrKind::Runtime,
        });
    }
    let r = reals[0].unwrap_or(1.0).max(1.0) as u32;
    let c = reals[1].unwrap_or(1.0).max(1.0) as u32;
    Ok((r, c))
}

/// Parse a length-4 margin vector.
fn parse_quad(v: &RVal, name: &str) -> Result<[f64; 4], R2Err> {
    let reals = v.as_reals()?;
    if reals.len() < 4 {
        return Err(R2Err {
            msg: format!("par({}=...) expects a length-4 numeric vector", name),
            kind: ErrKind::Runtime,
        });
    }
    Ok([
        reals[0].unwrap_or(0.0),
        reals[1].unwrap_or(0.0),
        reals[2].unwrap_or(0.0),
        reals[3].unwrap_or(0.0),
    ])
}

/// Apply one named setting to `p`. Returns the parameter name (so we
/// can build the "previous values" return list) on success.
fn apply_one(p: &mut PlotParams, name: &str, val: &RVal) -> Result<(), R2Err> {
    // RVal::Null is a legitimate value for mfrow/mfcol (means "no multi-panel
    // layout") — used during par(oldpar) restore when the captured snapshot
    // had no panel grid set. Treat NULL as a clearing assignment.
    if matches!(val, RVal::Null) {
        match name {
            "mfrow" => { p.mfrow = None; return Ok(()); }
            "mfcol" => { p.mfcol = None; return Ok(()); }
            _ => {} // other params don't accept NULL — fall through to the error below
        }
    }

    match name {
        "mfrow" => {
            p.mfrow = Some(parse_pair(val, "mfrow")?);
            p.mfcol = None;
        }
        "mfcol" => {
            p.mfcol = Some(parse_pair(val, "mfcol")?);
            p.mfrow = None;
        }
        "mar" => p.mar = parse_quad(val, "mar")?,
        "oma" => p.oma = parse_quad(val, "oma")?,
        "cex"      => p.cex      = val.scalar_f64()?.unwrap_or(p.cex),
        "cex.axis" => p.cex_axis = val.scalar_f64()?.unwrap_or(p.cex_axis),
        "cex.lab"  => p.cex_lab  = val.scalar_f64()?.unwrap_or(p.cex_lab),
        "cex.main" => p.cex_main = val.scalar_f64()?.unwrap_or(p.cex_main),
        "col" => p.col = val_to_str(val),
        "bg"  => p.bg  = val_to_str(val),
        "fg"  => p.fg  = val_to_str(val),
        "lty" => p.lty = val_to_str(val),
        "lwd" => p.lwd = val.scalar_f64()?.unwrap_or(p.lwd),
        "pch" => p.pch = val.scalar_f64()?.unwrap_or(p.pch as f64) as i32,
        "las" => p.las = val.scalar_f64()?.unwrap_or(p.las as f64) as i32,
        "new" => match val {
            RVal::Logical(v, _) => p.new = v.first().and_then(|x| *x).unwrap_or(false),
            _ => p.new = val.scalar_f64()?.unwrap_or(0.0) != 0.0,
        },
        other => {
            return Err(R2Err {
                msg: format!("par(): unknown parameter '{}'", other),
                kind: ErrKind::Runtime,
            });
        }
    }
    Ok(())
}

/// Returns every parameter as a named list. Used by `par()` with no
/// arguments and by `par(<setters>)` to capture "old" values.
fn snapshot(p: &PlotParams) -> RVal {
    let names = [
        "mfrow", "mfcol", "mar", "oma", "cex", "cex.axis", "cex.lab", "cex.main",
        "col", "bg", "fg", "lty", "lwd", "pch", "las", "new",
    ];
    let items: Vec<(Option<Arc<str>>, RVal)> = names
        .iter()
        .map(|n| (Some(Arc::from(*n)), param_value(p, n).unwrap_or(RVal::Null)))
        .collect();
    RVal::List(items)
}

/// `par(...)` builtin entry point.
pub fn bi_par(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // Case 1: par() with no args → return a snapshot.
    if a.is_empty() {
        return Ok(with_device(|d| snapshot(&d.params)));
    }

    // Case 2: par("col") — single unnamed character arg → return that one.
    if a.len() == 1 && a[0].name.is_none() {
        if let RVal::Character(v, _) = &a[0].value {
            if let Some(Some(name)) = v.first() {
                let val = with_device(|d| param_value(&d.params, name.as_ref()));
                return val.ok_or_else(|| R2Err {
                    msg: format!("par(): unknown parameter '{}'", name),
                    kind: ErrKind::Runtime,
                });
            }
        }
        // Single unnamed arg that's a list → bulk setter (par(oldpar) restore form).
        if let RVal::List(items) = &a[0].value {
            let mut old: Vec<(Option<Arc<str>>, RVal)> = Vec::new();
            with_device(|d| -> Result<(), R2Err> {
                for (name_opt, val) in items {
                    if let Some(name) = name_opt {
                        if let Some(prev) = param_value(&d.params, name.as_ref()) {
                            old.push((Some(name.clone()), prev));
                        }
                        apply_one(&mut d.params, name.as_ref(), val)?;
                    }
                }
                Ok(())
            })?;
            return Ok(RVal::List(old));
        }
    }

    // Case 3: par(name=value, ...) → set, return list of previous values.
    let mut old: Vec<(Option<Arc<str>>, RVal)> = Vec::new();
    with_device(|d| -> Result<(), R2Err> {
        for arg in a {
            if let Some(name) = &arg.name {
                if let Some(prev) = param_value(&d.params, name.as_ref()) {
                    old.push((Some(name.clone()), prev));
                }
                apply_one(&mut d.params, name.as_ref(), &arg.value)?;
            }
        }
        Ok(())
    })?;
    Ok(RVal::List(old))
}

/// `dev.off(n = current)` — close one open device. Returns the new
/// current device's id as an integer scalar (or NULL if no devices
/// remain). Matches R: `dev.off()` closes the current device,
/// `dev.off(2)` closes device 2 explicitly.
pub fn bi_dev_off(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let target: Option<crate::device::DeviceId> = a.first()
        .and_then(|arg| arg.value.as_reals().ok()?.into_iter().next()?)
        .map(|x| crate::device::DeviceId(x as u32));
    match crate::device::close_device(target) {
        Some(crate::device::DeviceId(n)) => Ok(rint(n as i32)),
        None => Ok(RVal::Null),
    }
}

/// `dev.new()` — open a fresh device and make it current. Returns
/// the new device's id.
pub fn bi_dev_new(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    let crate::device::DeviceId(n) = crate::device::new_device();
    Ok(rint(n as i32))
}

/// `dev.set(n)` — make device `n` the current one. Returns the
/// previous current id, or NULL if `n` is not open / no previous.
pub fn bi_dev_set(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let id: u32 = a.first()
        .and_then(|arg| arg.value.as_reals().ok()?.into_iter().next()?)
        .map(|x| x as u32)
        .ok_or_else(|| R2Err {
            msg: "dev.set(n): integer id required".into(),
            kind: ErrKind::Runtime,
        })?;
    match crate::device::set_device(crate::device::DeviceId(id)) {
        Some(crate::device::DeviceId(n)) => Ok(rint(n as i32)),
        None => Ok(RVal::Null),
    }
}

/// `dev.list()` — integer vector of open device ids. Empty if no
/// devices are open.
pub fn bi_dev_list(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    let ids: Vec<Option<i32>> = crate::device::list_devices()
        .into_iter()
        .map(|crate::device::DeviceId(n)| Some(n as i32))
        .collect();
    Ok(RVal::Integer(ids.into(), Attrs::default()))
}

/// `dev.cur()` — current device id, or `0L` if none open. Matches
/// R's convention of returning 0 when no device exists.
pub fn bi_dev_cur(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    let n = crate::device::current_device()
        .map(|crate::device::DeviceId(n)| n as i32)
        .unwrap_or(0);
    Ok(rint(n))
}

/// `dev.view()` builtin — start the built-in HTTP plot server (if not
/// already running) and open the user's default browser at the index
/// page. Subsequent plot calls render into the device, which the page
/// polls and displays. Idempotent: calling it more than once just
/// re-opens the browser tab to the same server.
pub fn bi_dev_view(_a: &[EvalArg]) -> Result<RVal, R2Err> {
    match crate::server::ensure_started() {
        Some(port) => {
            crate::server::open_browser(port);
            soutln!(
                "Ardon-R2 plot viewer at http://127.0.0.1:{}/  (auto-refreshes when you call plot())",
                port
            );
            // The HTTP server runs on a daemon thread; the process must
            // stay alive for the browser to reach it. In REPL mode this
            // happens naturally. In script mode, end your script with
            // Sys.sleep(N) or the server dies when the script exits.
            soutln!("  (REPL: stays alive automatically. Script: end with Sys.sleep(N) or the server exits.)");
            Ok(rstr(&format!("http://127.0.0.1:{}/", port)))
        }
        None => Err(R2Err {
            msg: "dev.view(): could not bind to any port in 8765..8775".into(),
            kind: ErrKind::Runtime,
        }),
    }
}

/// `save.plot(file, width=800, height=600)` — flush current device
/// contents to a file. Extension drives the format:
///
///   * `*.svg` → vector SVG (resolution-independent, best for editing).
///   * `*.png` → rasterized PNG (best for embedding in docs / chat /
///               slides — most apps don't render SVG).
///
/// `width` and `height` are pixels and only affect PNG; SVG ignores them.
pub fn bi_save_plot(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let path = match gv(a, 0) {
        RVal::Character(v, _) => v.first().and_then(|x| x.as_ref()).map(|s| s.to_string()),
        _ => None,
    }
    .or_else(|| gn(a, "file").map(|v| val_to_str(&v)))
    .ok_or_else(|| R2Err {
        msg: "save.plot(): path argument required".into(),
        kind: ErrKind::Runtime,
    })?;
    let width: u32 = gn(a, "width").and_then(|v| match v {
        RVal::Numeric(vs, _) => vs.first().and_then(|x| *x).map(|x| x as u32),
        RVal::Integer(vs, _) => vs.first().and_then(|x| *x).map(|x| x as u32),
        _ => None,
    }).unwrap_or(800);
    let height: u32 = gn(a, "height").and_then(|v| match v {
        RVal::Numeric(vs, _) => vs.first().and_then(|x| *x).map(|x| x as u32),
        RVal::Integer(vs, _) => vs.first().and_then(|x| *x).map(|x| x as u32),
        _ => None,
    }).unwrap_or(600);
    let abs = crate::device::save_plot(&path, width, height)?;
    let display = abs.to_string_lossy();
    let clean = display.strip_prefix(r"\\?\").unwrap_or(&display);
    soutln!("Plot saved to {}", clean);
    Ok(RVal::Character(vec![Some(Arc::from(clean))], Attrs::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::dev_off;

    fn arg(name: Option<&str>, value: RVal) -> EvalArg {
        EvalArg {
            name: name.map(Arc::from),
            value,
        }
    }

    #[test]
    fn par_no_args_returns_snapshot_list() {
        dev_off();
        let r = bi_par(&[]).unwrap();
        match r {
            RVal::List(items) => assert!(items.iter().any(|(n, _)| n.as_deref() == Some("mfrow"))),
            _ => panic!("expected list"),
        }
        dev_off();
    }

    #[test]
    fn par_query_single_name_returns_that_value() {
        dev_off();
        let r = bi_par(&[arg(None, rstr("cex"))]).unwrap();
        // cex default is 1.0.
        assert_eq!(r.scalar_f64().unwrap(), Some(1.0));
        dev_off();
    }

    #[test]
    fn par_set_returns_old_values_and_mutates_device() {
        dev_off();
        let old = bi_par(&[arg(Some("cex"), rnum(2.5))]).unwrap();
        // Returned "old" should contain cex=1.0.
        match old {
            RVal::List(items) => {
                let cex_old = items.iter().find(|(n, _)| n.as_deref() == Some("cex"));
                assert!(cex_old.is_some());
            }
            _ => panic!("expected list"),
        }
        // Device should now have cex=2.5.
        with_device(|d| assert!((d.params.cex - 2.5).abs() < 1e-9));
        dev_off();
    }

    #[test]
    fn par_mfrow_sets_panel_grid() {
        dev_off();
        let v = rnums(&[2.0, 3.0]);
        let _ = bi_par(&[arg(Some("mfrow"), v)]).unwrap();
        with_device(|d| assert_eq!(d.params.mfrow, Some((2, 3))));
        dev_off();
    }

    #[test]
    fn par_unknown_param_errors() {
        dev_off();
        let r = bi_par(&[arg(Some("nope"), rnum(1.0))]);
        assert!(r.is_err());
        dev_off();
    }
}
