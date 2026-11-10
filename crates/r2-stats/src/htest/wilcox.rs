//! `wilcox.test` as R computes it: the rank sum test of two samples (`W`) and
//! the signed rank test of one sample or of pairs (`V`). With fewer than 50
//! values a side and no ties (and, signed, no zero differences) the p-value is
//! exact, from the statistic's distribution counted out; otherwise it is the
//! normal approximation with R's tie correction and continuity correction.

use super::*;
use r2_types::{EvalArg, R2Err, RVal, TypeInstance};
use std::collections::HashMap;
use std::sync::Arc;
use crate::dist::phi_upper;

fn text(a: &[EvalArg], name: &str, default: &str) -> String {
    match arg_named(a, name) {
        Some(RVal::Character(s, _)) => s.first().cloned().flatten().map(|x| x.to_string()).unwrap_or_else(|| default.into()),
        _ => default.into(),
    }
}

fn flag(a: &[EvalArg], name: &str) -> Option<bool> {
    match arg_named(a, name) {
        Some(RVal::Logical(l, _)) => l.first().copied().flatten(),
        _ => None,
    }
}

/// Ranks with ties given their mean rank, and the sizes of the tied groups.
fn midranks(v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    let mut ranks = vec![0.0; v.len()];
    let mut ties = Vec::new();
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] { j += 1; }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j { ranks[idx[k]] = r; }
        ties.push((j - i + 1) as f64);
        i = j + 1;
    }
    (ranks, ties)
}

/// P(W <= q) for the rank sum statistic of samples of sizes n and m, the
/// number of n-subsets of 1..n+m by their rank sum.
fn pwilcox(q: f64, n: usize, m: usize) -> f64 {
    let max = n * m;
    if q < 0.0 { return 0.0; }
    if q >= max as f64 { return 1.0; }
    // ways[k][u]: k ranks chosen, u = their sum minus k(k+1)/2
    let mut ways = vec![vec![0.0f64; max + 1]; n + 1];
    ways[0][0] = 1.0;
    for item in 0..(n + m) {
        for k in (1..=n.min(item + 1)).rev() {
            // the new rank is item + 1; as the k-th chosen it adds item + 1 - k
            let add = item + 1 - k;
            if add > m { continue; }
            for u in (add..=max).rev() {
                let w = ways[k - 1][u - add];
                if w != 0.0 { ways[k][u] += w; }
            }
        }
    }
    let total: f64 = ways[n].iter().sum();
    ways[n][..=q.floor() as usize].iter().sum::<f64>() / total
}

/// P(V <= q) for the signed rank statistic of n differences.
fn psignrank(q: f64, n: usize) -> f64 {
    let max = n * (n + 1) / 2;
    if q < 0.0 { return 0.0; }
    if q >= max as f64 { return 1.0; }
    let mut ways = vec![0.0f64; max + 1];
    ways[0] = 1.0;
    for r in 1..=n {
        for s in (r..=max).rev() { ways[s] += ways[s - r]; }
    }
    let total: f64 = ways.iter().sum();
    ways[..=q.floor() as usize].iter().sum::<f64>() / total
}

/// The exact p-value from a lower-tail distribution function, as R takes it.
fn exact_p(stat: f64, centre: f64, alternative: &str, lower: &dyn Fn(f64) -> f64) -> f64 {
    let upper = |q: f64| 1.0 - lower(q);
    match alternative {
        "less" => lower(stat),
        "greater" => upper(stat - 1.0),
        _ => {
            let p = if stat > centre { upper(stat - 1.0) } else { lower(stat) };
            (2.0 * p).min(1.0)
        }
    }
}

/// The normal approximation: `z` the centred statistic, `sigma` its sd.
fn normal_p(z: f64, sigma: f64, alternative: &str, correct: bool) -> f64 {
    let correction = if !correct { 0.0 } else {
        match alternative { "less" => -0.5, "greater" => 0.5, _ => z.signum() * 0.5 }
    };
    let z = (z - correction) / sigma;
    match alternative {
        "less" => phi_upper(-z),
        "greater" => phi_upper(z),
        _ => (2.0 * phi_upper(z.abs())).min(1.0),
    }
}

fn result(stat_name: &str, stat: f64, p: f64, null_name: &str, mu: f64, alternative: &str, method: String) -> RVal {
    let mut fields = HashMap::new();
    fields.insert(Arc::from("statistic"), named(&[stat], &[stat_name]));
    fields.insert(Arc::from("parameter"), RVal::Null);
    fields.insert(Arc::from("p.value"), rnum(p));
    fields.insert(Arc::from("null.value"), named(&[mu], &[null_name]));
    fields.insert(Arc::from("alternative"), rstr(alternative));
    fields.insert(Arc::from("method"), rstr(&method));
    fields.insert(Arc::from("data.name"), rstr("x and y"));
    let order = ["statistic", "parameter", "p.value", "null.value", "alternative", "method", "data.name"];
    fields.insert(Arc::from(".order"), RVal::Character(order.iter().map(|s| Some(Arc::from(*s))).collect(), r2_types::Attrs::default()));
    RVal::TypeInstance(TypeInstance { type_name: Arc::from("htest"), fields })
}

pub fn bi_wilcox_test(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let finite = |v: RVal| -> Result<Vec<f64>, R2Err> { Ok(v.as_reals()?.into_iter().flatten().filter(|x| x.is_finite()).collect()) };
    let x = finite(first(a))?;
    let y = match arg_named(a, "y").or_else(|| a.iter().filter(|e| e.name.is_none()).nth(1).map(|e| e.value.clone())) {
        Some(v) if !matches!(v, RVal::Null) => Some(finite(v)?),
        _ => None,
    };
    let alternative = text(a, "alternative", "two.sided");
    let alternative = ["two.sided", "less", "greater"].into_iter().find(|s| s.starts_with(alternative.as_str())).unwrap_or("two.sided");
    let mu = arg_named(a, "mu").and_then(|v| v.scalar_f64().ok().flatten()).unwrap_or(0.0);
    let paired = flag(a, "paired").unwrap_or(false);
    let correct = flag(a, "correct").unwrap_or(true);
    let exact_arg = flag(a, "exact");
    let cc = if correct { " with continuity correction" } else { "" };

    match (y, paired) {
        (Some(y), false) => {
            let (n, m) = (x.len(), y.len());
            if n == 0 { return Err(runtime_err("not enough (non-missing) 'x' observations".into())); }
            if m == 0 { return Err(runtime_err("not enough 'y' observations".into())); }
            let mut all: Vec<f64> = x.iter().map(|v| v - mu).collect();
            all.extend(&y);
            let (ranks, ties) = midranks(&all);
            let w = ranks[..n].iter().sum::<f64>() - (n * (n + 1)) as f64 / 2.0;
            let tied = ties.iter().any(|&t| t > 1.0);
            let exact = exact_arg.unwrap_or(n < 50 && m < 50);
            let (p, method) = if exact && !tied {
                (exact_p(w, (n * m) as f64 / 2.0, alternative, &|q| pwilcox(q, n, m)), "Wilcoxon rank sum exact test".to_string())
            } else {
                let nm = (n + m) as f64;
                let tie_term: f64 = ties.iter().map(|t| t * t * t - t).sum::<f64>() / (nm * (nm - 1.0));
                let sigma = ((n * m) as f64 / 12.0 * ((nm + 1.0) - tie_term)).sqrt();
                (normal_p(w - (n * m) as f64 / 2.0, sigma, alternative, correct), format!("Wilcoxon rank sum test{}", cc))
            };
            Ok(result("W", w, p, "location shift", mu, alternative, method))
        }
        (y, _) => {
            let d: Vec<f64> = match y {
                Some(y) => {
                    if y.len() != x.len() { return Err(runtime_err("'x' and 'y' must have the same length".into())); }
                    x.iter().zip(&y).map(|(a, b)| a - b - mu).collect()
                }
                None => x.iter().map(|v| v - mu).collect(),
            };
            let zeroes = d.iter().any(|v| *v == 0.0);
            let d: Vec<f64> = d.into_iter().filter(|v| *v != 0.0).collect();
            let n = d.len();
            if n == 0 { return Err(runtime_err("not enough (non-missing) 'x' observations".into())); }
            let (ranks, ties) = midranks(&d.iter().map(|v| v.abs()).collect::<Vec<_>>());
            let v: f64 = ranks.iter().zip(&d).filter(|(_, x)| **x > 0.0).map(|(r, _)| r).sum();
            let tied = ties.iter().any(|&t| t > 1.0);
            let exact = exact_arg.unwrap_or(n < 50);
            let (p, method) = if exact && !tied && !zeroes {
                (exact_p(v, (n * (n + 1)) as f64 / 4.0, alternative, &|q| psignrank(q, n)), "Wilcoxon signed rank exact test".to_string())
            } else {
                let nf = n as f64;
                let tie_term: f64 = ties.iter().map(|t| t * t * t - t).sum::<f64>() / 48.0;
                let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_term).sqrt();
                (normal_p(v - nf * (nf + 1.0) / 4.0, sigma, alternative, correct), format!("Wilcoxon signed rank test{}", cc))
            };
            Ok(result("V", v, p, "location", mu, alternative, method))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R: pwilcox(3, 4, 4) = 0.1, psignrank(5, 10) = 0.009765625
    #[test]
    fn the_exact_distributions_are_r_s() {
        assert!((pwilcox(3.0, 4, 4) - 0.1).abs() < 1e-15);
        assert!((psignrank(5.0, 10) - 0.009765625).abs() < 1e-15);
        assert!((pwilcox(16.0, 4, 4) - 1.0).abs() < 1e-15);
    }
}
