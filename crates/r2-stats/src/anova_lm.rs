//! `anova(fit)` for a linear model as R gives it: one row per term in the
//! order of the formula (sequential, "Type I" sums of squares) and a row for
//! the residuals, with columns `Df`, `Sum Sq`, `Mean Sq`, `F value`, `Pr(>F)`.
//!
//! A term's sum of squares is the squared length of the part of `Q'y` that
//! belongs to its columns, `Q` from the QR of the design in formula order.

use std::collections::HashMap;
use std::sync::Arc;

use r2_linalg::linpack;
use r2_types::{fmt_num, Attrs, RVal, TypeInstance};

fn reals(v: Option<&RVal>) -> Option<Vec<f64>> {
    v?.as_reals().ok().map(|x| x.into_iter().map(|y| y.unwrap_or(f64::NAN)).collect())
}

fn texts(v: Option<&RVal>) -> Vec<String> {
    match v { Some(RVal::Character(s, _)) => s.iter().map(|x| x.as_deref().unwrap_or("").to_string()).collect(), _ => Vec::new() }
}

/// The table, or `None` for a fit that does not carry its design.
pub fn sequential(inst: &TypeInstance) -> Option<RVal> {
    let RVal::Matrix(x) = inst.fields.get(".x")? else { return None };
    let y = reals(inst.fields.get(".y"))?;
    let assign: Vec<usize> = reals(inst.fields.get(".assign"))?.iter().map(|&k| k as usize).collect();
    let terms = texts(inst.fields.get(".terms"));
    let (n, p) = (x.nrow, x.ncol);
    if y.len() != n || assign.len() != p || n <= p { return None; }

    let mut qr = x.data.clone();
    let d = linpack::dqrdc2(&mut qr, n, p, 1e-7);
    let mut effects = y.clone();
    linpack::qty(&qr, n, d.rank, &d.qraux, &mut effects);

    // sums of squares per term, in formula order; terms whose columns were all
    // aliased away have none and get no row
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for (t, label) in terms.iter().enumerate() {
        let cols: Vec<usize> = (0..d.rank).filter(|&j| assign.get(d.pivot[j] as usize - 1) == Some(&(t + 1))).collect();
        if cols.is_empty() { continue; }
        let ss: f64 = cols.iter().map(|&j| effects[j] * effects[j]).sum();
        rows.push((label.clone(), cols.len() as f64, ss));
    }
    let df_res = (n - d.rank) as f64;
    let ss_res: f64 = effects[d.rank..].iter().map(|e| e * e).sum();
    let ms_res = ss_res / df_res;

    let mut df = Vec::new();
    let mut ss = Vec::new();
    let mut ms = Vec::new();
    let mut fv = Vec::new();
    let mut pv = Vec::new();
    for (_, k, s) in &rows {
        let m = s / k;
        let f = m / ms_res;
        df.push(Some(*k as i32));
        ss.push(Some(*s));
        ms.push(Some(m));
        fv.push(Some(f));
        pv.push(Some(crate::htest::f_sf(f, *k, df_res)));
    }
    df.push(Some(df_res as i32));
    ss.push(Some(ss_res));
    ms.push(Some(ms_res));
    fv.push(None);
    pv.push(None);
    let mut names: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    names.push("Residuals".into());

    let response = texts(inst.fields.get(".response")).into_iter().next().unwrap_or_default();
    let report = report(&response, &names, &df, &ss, &ms, &fv, &pv);
    let num = |v: Vec<Option<f64>>| RVal::Numeric(v.into(), Attrs::default());
    let mut fields = HashMap::new();
    fields.insert(Arc::from("Df"), RVal::Integer(df.into(), Attrs::default()));
    fields.insert(Arc::from("Sum Sq"), num(ss));
    fields.insert(Arc::from("Mean Sq"), num(ms));
    fields.insert(Arc::from("F value"), num(fv));
    fields.insert(Arc::from("Pr(>F)"), num(pv));
    fields.insert(Arc::from(".rownames"), RVal::Character(names.iter().map(|n| Some(Arc::from(n.as_str()))).collect(), Attrs::default()));
    fields.insert(Arc::from(".report"), RVal::Character(vec![Some(Arc::from(report.as_str()))], Attrs::default()));
    Some(RVal::TypeInstance(TypeInstance { type_name: Arc::from("anova"), fields }))
}

fn stars(p: f64) -> &'static str {
    if p < 0.001 { "***" } else if p < 0.01 { "**" } else if p < 0.05 { "*" } else if p < 0.1 { "." } else { " " }
}

/// R's printed table, columns right-aligned under their headers.
fn report(response: &str, names: &[String], df: &[Option<i32>], ss: &[Option<f64>], ms: &[Option<f64>],
          fv: &[Option<f64>], pv: &[Option<f64>]) -> String {
    let cell = |x: &Option<f64>, f: &dyn Fn(f64) -> String| x.map(f).unwrap_or_default();
    let p_text = |p: f64| if p < 2.2e-16 { "< 2.2e-16".to_string() } else { crate::htest::fmt_pval(p) };
    let cols: Vec<(&str, Vec<String>)> = vec![
        ("Df", df.iter().map(|d| d.map(|d| d.to_string()).unwrap_or_default()).collect()),
        ("Sum Sq", ss.iter().map(|x| cell(x, &|v| fmt_num(v))).collect()),
        ("Mean Sq", ms.iter().map(|x| cell(x, &|v| fmt_num(v))).collect()),
        ("F value", fv.iter().map(|x| cell(x, &|v| fmt_num(v))).collect()),
        ("Pr(>F)", pv.iter().map(|x| cell(x, &p_text)).collect()),
    ];
    let label_w = names.iter().map(|n| n.len()).max().unwrap_or(0);
    let mut out = String::from("Analysis of Variance Table\n\n");
    if !response.is_empty() { out.push_str(&format!("Response: {}\n", response)); }
    out.push_str(&format!("{:w$}", "", w = label_w));
    let widths: Vec<usize> = cols.iter().map(|(h, v)| v.iter().map(|s| s.len()).max().unwrap_or(0).max(h.len())).collect();
    for ((h, _), w) in cols.iter().zip(&widths) { out.push_str(&format!(" {:>w$}", h, w = w)); }
    out.push('\n');
    for (i, name) in names.iter().enumerate() {
        out.push_str(&format!("{:<w$}", name, w = label_w));
        for ((_, v), w) in cols.iter().zip(&widths) { out.push_str(&format!(" {:>w$}", v[i], w = w)); }
        if let Some(p) = pv[i] { out.push_str(&format!(" {}", stars(p))); }
        out.push('\n');
    }
    out.push_str("---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");
    out
}
