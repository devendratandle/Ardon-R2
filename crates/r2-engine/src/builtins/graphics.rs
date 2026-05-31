//! Engine-side shims for graphics builtins. Every function here is
//! a thin delegator into `r2_graphics::{plots, overlays, params,
//! colors}` — keeping the Engine free of plot logic but preserving
//! the (`&mut Engine, &[EvalArg], &EnvRef`) signature the function
//! registry expects.
//!
//! `bi_plot` is the only function with non-trivial body: it carries
//! the split-handler dispatch for model-aware plotting
//! (`plot(lm)`, `plot(gbm)`, `plot(kmeans)`) before delegating to
//! `r2_graphics::plots::bi_plot` for the data-path case.

use std::sync::Arc;

use r2_types::{Attrs, EnvRef, EvalArg, R2Err, RVal};

use crate::Engine;

#[inline] fn rstr(s: &str) -> RVal {
    RVal::Character(vec![Some(Arc::from(s))], Attrs::default())
}
#[inline] fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|x| Some(*x)).collect::<Vec<_>>().into(), Attrs::default())
}
#[inline] fn gv(a: &[EvalArg], i: usize) -> RVal {
    a.iter().filter(|x| x.name.is_none()).nth(i).map(|x| x.value.clone()).unwrap_or(RVal::Null)
}

// ─── Plot / hist / boxplot / barplot ───────────────────────────────

/// `plot(x, y, ...)` with model-aware dispatch for `lm`, `glm`,
/// `gbm`, `kmeans`. Falls through to the data-path implementation
/// in `r2_graphics::plots::bi_plot` for plain numeric vectors.
pub(crate) fn bi_plot(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    if let RVal::TypeInstance(inst) = &gv(a, 0) {
        match inst.type_name.as_ref() {
            "gbm" => {
                if let Some(tl) = inst.fields.get("train.loss") {
                    let losses: Vec<f64> = e.as_reals(tl)?.into_iter().filter_map(|x| x).collect();
                    let iters:  Vec<f64> = (1..=losses.len()).map(|i| i as f64).collect();
                    let fake_args = vec![
                        EvalArg { name: None, value: rnums(&iters) },
                        EvalArg { name: None, value: rnums(&losses) },
                        EvalArg { name: Some(Arc::from("main")), value: rstr("GBM Training Loss") },
                        EvalArg { name: Some(Arc::from("xlab")), value: rstr("Iteration") },
                        EvalArg { name: Some(Arc::from("ylab")), value: rstr("Loss") },
                    ];
                    let env = e.global_env.clone();
                    return bi_plot(e, &fake_args, &env);
                }
            }
            "lm" | "glm" => {
                if let (Some(fitted), Some(resid)) =
                    (inst.fields.get("fitted.values"), inst.fields.get("residuals"))
                {
                    let fake_args = vec![
                        EvalArg { name: None, value: fitted.clone() },
                        EvalArg { name: None, value: resid.clone() },
                        EvalArg { name: Some(Arc::from("main")), value: rstr("Residuals vs Fitted") },
                        EvalArg { name: Some(Arc::from("xlab")), value: rstr("Fitted values") },
                        EvalArg { name: Some(Arc::from("ylab")), value: rstr("Residuals") },
                    ];
                    let env = e.global_env.clone();
                    return bi_plot(e, &fake_args, &env);
                }
            }
            "kmeans" => {
                if let Some(ws) = inst.fields.get("withinss") {
                    let wss: Vec<f64> = e.as_reals(ws)?.into_iter().filter_map(|x| x).collect();
                    let clusters: Vec<f64> = (1..=wss.len()).map(|i| i as f64).collect();
                    let fake_args = vec![
                        EvalArg { name: None, value: rnums(&clusters) },
                        EvalArg { name: None, value: rnums(&wss) },
                        EvalArg { name: Some(Arc::from("main")), value: rstr("K-means Within-SS") },
                        EvalArg { name: Some(Arc::from("xlab")), value: rstr("Cluster") },
                        EvalArg { name: Some(Arc::from("ylab")), value: rstr("Within SS") },
                    ];
                    let env = e.global_env.clone();
                    return bi_plot(e, &fake_args, &env);
                }
            }
            _ => {}
        }
    }
    let _ = e;
    r2_graphics::plots::bi_plot(a)
}

pub(crate) fn bi_hist(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::plots::bi_hist(a)
}
pub(crate) fn bi_boxplot(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::plots::bi_boxplot(a)
}
pub(crate) fn bi_barplot(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::plots::bi_barplot(a)
}

// ─── Overlays ──────────────────────────────────────────────────────

pub(crate) fn bi_lines(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::overlays::bi_lines(a)
}
pub(crate) fn bi_points(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::overlays::bi_points(a)
}
pub(crate) fn bi_abline(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::overlays::bi_abline(a)
}
pub(crate) fn bi_legend(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::overlays::bi_legend(a)
}

// ─── par() + dev.*  ────────────────────────────────────────────────

pub(crate) fn bi_par(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_par(a)
}
pub(crate) fn bi_dev_off(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_off(a)
}
pub(crate) fn bi_dev_new(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_new(a)
}
pub(crate) fn bi_dev_set(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_set(a)
}
pub(crate) fn bi_dev_list(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_list(a)
}
pub(crate) fn bi_dev_cur(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_cur(a)
}
pub(crate) fn bi_save_plot(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_save_plot(a)
}
pub(crate) fn bi_dev_view(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::params::bi_dev_view(a)
}

// ─── Color helpers (R-style) ───────────────────────────────────────

pub(crate) fn bi_rgb(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_rgb(a)
}
pub(crate) fn bi_gray(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_gray(a)
}
pub(crate) fn bi_hsv(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_hsv(a)
}
pub(crate) fn bi_rainbow(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_rainbow(a)
}
pub(crate) fn bi_heat_colors(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_heat_colors(a)
}
pub(crate) fn bi_terrain_colors(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_terrain_colors(a)
}
pub(crate) fn bi_topo_colors(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_topo_colors(a)
}
pub(crate) fn bi_cm_colors(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_cm_colors(a)
}
pub(crate) fn bi_adjustcolor(_e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    r2_graphics::colors::bi_adjustcolor(a)
}
