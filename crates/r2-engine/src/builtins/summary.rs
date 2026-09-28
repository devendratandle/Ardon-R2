//! `summary(x)` returns a value, as R's does, and prints only when that
//! value is printed or auto-printed — `s <- summary(fit)` is silent and
//! `summary(fit)$coefficients` shows just the table.
//!
//! A numeric vector gives R's six-number `summaryDefault` (plus `NA's`);
//! a fitted `lm`/`glm` gives `summary.lm`/`summary.glm` carrying the fit's
//! fields and R's four-column `coefficients` matrix; a data frame gives a
//! `table`. The report text is still produced by `coerce::summary_report`
//! — it is captured and kept on the result as `.report`, the way the
//! hypothesis tests keep theirs.

use std::collections::HashMap;
use std::sync::Arc;
use r2_types::*;
use crate::{gv, Engine};
use super::coerce::summary_report;

pub(crate) fn bi_summary(e: &mut Engine, a: &[EvalArg], env: &EnvRef) -> Result<RVal, R2Err> {
    let x = gv(a, 0);
    if let RVal::Numeric(..) | RVal::Integer(..) = &x {
        return Ok(summary_default(&x.as_reals()?));
    }
    let (r, report) = r2_types::out::capture(|| summary_report(e, a, env));
    let r = r?;
    let (type_name, mut fields): (Arc<str>, HashMap<Arc<str>, RVal>) = match &x {
        RVal::TypeInstance(inst) => {
            let mut fields = inst.fields.clone();
            if let Some(m) = coef_table(inst) { fields.insert(Arc::from("coefficients"), m); }
            (Arc::from(format!("summary.{}", inst.type_name).as_str()), fields)
        }
        RVal::DataFrame(_) => (Arc::from("table"), HashMap::new()),
        _ if report.is_empty() => return Ok(r),
        _ => (Arc::from("summary"), HashMap::new()),
    };
    fields.insert(Arc::from(".report"), RVal::Character(vec![Some(Arc::from(report.as_str()))], Attrs::default()));
    Ok(RVal::TypeInstance(TypeInstance { type_name, fields }))
}

/// R's `summary.default` of numbers: Min., 1st Qu., Median, Mean, 3rd Qu.,
/// Max. (quantile type 7), then the NA count if there are any. Printed
/// with 4 significant digits (see `summaryDefault` in r2-types' Display).
fn summary_default(v: &[Real]) -> RVal {
    let mut x: Vec<f64> = v.iter().flatten().copied().filter(|x| !x.is_nan()).collect();
    let nas = v.len() - x.len();
    x.sort_by(|p, q| p.total_cmp(q));
    let quantile = |p: f64| -> Real {
        if x.is_empty() { return None; }
        let h = (x.len() - 1) as f64 * p;
        let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
        Some(x[lo] + (h - lo as f64) * (x[hi] - x[lo]))
    };
    let mean = if x.is_empty() { None } else { Some(x.iter().sum::<f64>() / x.len() as f64) };
    let mut vals = vec![quantile(0.0), quantile(0.25), quantile(0.5), mean, quantile(0.75), quantile(1.0)];
    let mut names = vec!["Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."];
    if nas > 0 { vals.push(Some(nas as f64)); names.push("NA's"); }
    RVal::Numeric(vals.into(), Attrs {
        names: Some(names.into_iter().map(Arc::from).collect()),
        class: Some(Arc::from("summaryDefault")),
        ..Default::default()
    })
}

/// R's `coef(summary(fit))`: Estimate, Std. Error, t (or z) value and its
/// p-value per coefficient, rows named by the coefficients.
fn coef_table(inst: &TypeInstance) -> Option<RVal> {
    let glm = inst.type_name.as_ref() == "glm";
    let col = |k: &str| inst.fields.get(k).and_then(|v| v.as_reals().ok());
    let cols = [col("coefficients")?, col("std.errors")?, col(if glm { "z.values" } else { "t.values" })?, col("p.values")?];
    let n = cols[0].len();
    if n == 0 || cols.iter().any(|c| c.len() != n) { return None; }
    let data: Vec<f64> = cols.iter().flat_map(|c| c.iter().map(|x| x.unwrap_or(f64::NAN))).collect();
    let mut m = Matrix::new(data, n, 4);
    m.row_names = match inst.fields.get("coefficients") {
        Some(RVal::Numeric(_, at)) => at.names.clone(),
        _ => None,
    };
    let (stat, p) = if glm { ("z value", "Pr(>|z|)") } else { ("t value", "Pr(>|t|)") };
    m.col_names = Some(["Estimate", "Std. Error", stat, p].into_iter().map(Arc::from).collect());
    Some(RVal::Matrix(m))
}
