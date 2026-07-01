//! Partial Least Squares Path Modeling (PLS-SEM / composite SEM).
//!
//! A pure-Rust, compiled implementation of the PLS-PM algorithm (the same
//! method `cSEM`/`SEMinR` run in interpreted R), plus a parallel bootstrap.
//! Reflective (Mode A) measurement, path-weighting inner scheme — cSEM's
//! defaults — so results are directly comparable.
//!
//! The engine builtin (`plssem`) wires argument/formula parsing and the
//! result object on top of [`fit_pls`] and [`bootstrap_paths`] here.

/// A fitted PLS path model.
#[derive(Debug, Clone)]
pub struct PlsFit {
    /// Construct scores, one Vec<f64> (length n) per construct.
    pub scores: Vec<Vec<f64>>,
    /// Outer weights, per construct, aligned to that construct's indicators.
    pub weights: Vec<Vec<f64>>,
    /// Loadings (indicator ↔ own construct correlation), per construct.
    pub loadings: Vec<Vec<f64>>,
    /// Path coefficients: `paths[j]` aligns with `structural[j]` predecessors.
    pub paths: Vec<Vec<f64>>,
    /// R² per construct (0 for exogenous).
    pub r2: Vec<f64>,
    /// Iterations to convergence.
    pub iters: usize,
}

/// Model layout shared by fit + bootstrap.
pub struct PlsModel {
    /// `blocks[j]` = column indices (into the data matrix) of construct j's
    /// indicators.
    pub blocks: Vec<Vec<usize>>,
    /// `structural[j]` = construct indices that point INTO construct j
    /// (its direct predecessors). Empty for exogenous constructs.
    pub structural: Vec<Vec<usize>>,
}

impl PlsModel {
    fn n_constructs(&self) -> usize { self.blocks.len() }
    /// Constructs that j points to (successors), derived from `structural`.
    fn successors(&self, j: usize) -> Vec<usize> {
        (0..self.n_constructs())
            .filter(|&k| self.structural[k].contains(&j))
            .collect()
    }
}

#[inline]
fn mean(x: &[f64]) -> f64 { x.iter().sum::<f64>() / x.len() as f64 }

/// Population standard deviation (÷ n), matching PLS score standardization.
fn sd_pop(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Standardize in place to mean 0, unit (population) variance.
fn standardize(x: &mut [f64]) {
    let m = mean(x);
    let s = sd_pop(x);
    let s = if s < 1e-12 { 1.0 } else { s };
    for v in x.iter_mut() { *v = (*v - m) / s; }
}

/// Pearson correlation of two equal-length vectors.
fn cor(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let mut num = 0.0; let mut da = 0.0; let mut db = 0.0;
    for i in 0..a.len() {
        let (xa, xb) = (a[i] - ma, b[i] - mb);
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

/// OLS coefficients of `y` on `preds` (no intercept — inputs are centered,
/// standardized construct scores) via normal equations + Gaussian
/// elimination. `preds` is a slice of predictor columns (each length n).
fn ols(preds: &[&[f64]], y: &[f64]) -> Vec<f64> {
    let k = preds.len();
    if k == 0 { return Vec::new(); }
    let n = y.len();
    // Normal equations: (X'X) b = X'y
    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    for a in 0..k {
        for b in a..k {
            let mut s = 0.0;
            for i in 0..n { s += preds[a][i] * preds[b][i]; }
            xtx[a * k + b] = s; xtx[b * k + a] = s;
        }
        let mut s = 0.0;
        for i in 0..n { s += preds[a][i] * y[i]; }
        xty[a] = s;
    }
    gauss_solve(&mut xtx, &mut xty, k)
}

/// Solve A x = b in place (A row-major k×k), returning x. Partial pivoting.
fn gauss_solve(a: &mut [f64], b: &mut [f64], k: usize) -> Vec<f64> {
    for col in 0..k {
        // pivot
        let mut piv = col;
        for r in col + 1..k {
            if a[r * k + col].abs() > a[piv * k + col].abs() { piv = r; }
        }
        if piv != col {
            for c in 0..k { a.swap(col * k + c, piv * k + c); }
            b.swap(col, piv);
        }
        let d = a[col * k + col];
        let d = if d.abs() < 1e-12 { 1e-12 } else { d };
        for r in 0..k {
            if r == col { continue; }
            let f = a[r * k + col] / d;
            if f == 0.0 { continue; }
            for c in col..k { a[r * k + c] -= f * a[col * k + c]; }
            b[r] -= f * b[col];
        }
    }
    (0..k).map(|i| b[i] / a[i * k + i]).collect()
}

/// Fit PLS-PM on a data matrix given as `n` rows × columns, column-major in
/// `data` (data[col*n + row]) — matches R's matrix storage. Mode A outer
/// estimation, path-weighting inner scheme.
pub fn fit_pls(data: &[f64], n: usize, model: &PlsModel, max_iter: usize, tol: f64) -> PlsFit {
    let c = model.n_constructs();
    // Standardized indicator columns, indexed by original data column.
    let ncol = data.len() / n;
    let mut z: Vec<Vec<f64>> = (0..ncol)
        .map(|col| {
            let mut v = data[col * n..col * n + n].to_vec();
            standardize(&mut v);
            v
        })
        .collect();

    // Initialize outer weights = 1 for each indicator; initial scores.
    let mut weights: Vec<Vec<f64>> =
        model.blocks.iter().map(|b| vec![1.0; b.len()]).collect();
    let score = |z: &[Vec<f64>], block: &[usize], w: &[f64]| -> Vec<f64> {
        let mut s = vec![0.0; n];
        for (wi, &col) in w.iter().zip(block) {
            for i in 0..n { s[i] += wi * z[col][i]; }
        }
        standardize(&mut s);
        s
    };
    let mut scores: Vec<Vec<f64>> = (0..c)
        .map(|j| score(&z, &model.blocks[j], &weights[j]))
        .collect();

    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        // ── Inner approximation (path weighting scheme) ──
        let mut inner: Vec<Vec<f64>> = vec![vec![0.0; n]; c];
        for j in 0..c {
            let mut zj = vec![0.0; n];
            // predecessors → multiple-regression coefficients
            let preds = &model.structural[j];
            if !preds.is_empty() {
                let cols: Vec<&[f64]> = preds.iter().map(|&m| scores[m].as_slice()).collect();
                let b = ols(&cols, &scores[j]);
                for (idx, &m) in preds.iter().enumerate() {
                    for i in 0..n { zj[i] += b[idx] * scores[m][i]; }
                }
            }
            // successors → correlations
            for k in model.successors(j) {
                let e = cor(&scores[j], &scores[k]);
                for i in 0..n { zj[i] += e * scores[k][i]; }
            }
            standardize(&mut zj);
            inner[j] = zj;
        }
        // ── Outer approximation (Mode A: weight = cor(indicator, inner)) ──
        let mut new_weights: Vec<Vec<f64>> = Vec::with_capacity(c);
        for j in 0..c {
            let w: Vec<f64> = model.blocks[j].iter()
                .map(|&col| cor(&z[col], &inner[j]))
                .collect();
            new_weights.push(w);
        }
        // new scores
        let new_scores: Vec<Vec<f64>> = (0..c)
            .map(|j| score(&z, &model.blocks[j], &new_weights[j]))
            .collect();
        // convergence on outer weights
        let mut max_d = 0.0f64;
        for j in 0..c {
            for i in 0..new_weights[j].len() {
                max_d = max_d.max((new_weights[j][i].abs() - weights[j][i].abs()).abs());
            }
        }
        weights = new_weights;
        scores = new_scores;
        if max_d < tol { break; }
    }

    // ── Structural paths, R², loadings ──
    let mut paths = vec![Vec::new(); c];
    let mut r2 = vec![0.0; c];
    for j in 0..c {
        let preds = &model.structural[j];
        if preds.is_empty() { continue; }
        let cols: Vec<&[f64]> = preds.iter().map(|&m| scores[m].as_slice()).collect();
        let b = ols(&cols, &scores[j]);
        // R² = variance of fitted / variance of y (y standardized → var 1)
        let mut fitted = vec![0.0; n];
        for (idx, col) in cols.iter().enumerate() {
            for i in 0..n { fitted[i] += b[idx] * col[i]; }
        }
        r2[j] = sd_pop(&fitted).powi(2);
        paths[j] = b;
    }
    let loadings: Vec<Vec<f64>> = (0..c)
        .map(|j| model.blocks[j].iter().map(|&col| cor(&z[col], &scores[j])).collect())
        .collect();

    // (recompute standardized loadings uses z already standardized)
    let _ = &mut z;
    PlsFit { scores, weights, loadings, paths, r2, iters }
}

/// Bootstrap the structural path coefficients: `reps` resamples with
/// replacement, refit, collect each path coefficient. Returns, per
/// endogenous construct, a Vec over predecessors of the bootstrap
/// distribution (`Vec<f64>` of length `reps`).
///
/// Runs the reps **in parallel** across cores (`r2_kernel::par_for_rayon`):
/// each resample is independent, and each gets its own RNG seeded from the
/// rep index, so there is no shared mutable state — the embarrassingly
/// parallel structure that lets this beat interpreted-R bootstrapping.
pub fn bootstrap_paths(
    data: &[f64], n: usize, model: &PlsModel, reps: usize, seed: u64,
) -> Vec<Vec<Vec<f64>>> {
    let c = model.n_constructs();
    let ncol = data.len() / n;

    // Parallel map: rep → that resample's path coefficients (per construct).
    let per_rep: Vec<Vec<Vec<f64>>> = r2_kernel::par_for_rayon(reps, |rep| {
        let mut rng = seed
            .wrapping_add((rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(1);
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as usize
        };
        let mut resampled = vec![0.0; ncol * n];
        for i in 0..n {
            let src = next() % n;
            for col in 0..ncol { resampled[col * n + i] = data[col * n + src]; }
        }
        fit_pls(&resampled, n, model, 100, 1e-7).paths
    });

    // Transpose [rep][construct][path] → [construct][path][rep].
    let mut out: Vec<Vec<Vec<f64>>> = model.structural.iter()
        .map(|p| vec![Vec::with_capacity(reps); p.len()])
        .collect();
    for rep_paths in &per_rep {
        for j in 0..c {
            for (idx, &b) in rep_paths[j].iter().enumerate() {
                out[j][idx].push(b);
            }
        }
    }
    out
}

// ── Engine builtin: plssem(data, .model=, .R=, .seed=) ────────────────
//
// cSEM-style model syntax:
//   Construct =~ Ind1 + Ind2 + ...     (reflective measurement, Mode A)
//   Endo      ~  Pred1 + Pred2 + ...   (structural path)
// `csem()` is registered as an alias.

use r2_types::{Attrs, ErrKind, EvalArg, RVal, R2Err, TypeInstance};
use std::sync::Arc;

fn e_gv(a: &[EvalArg], i: usize) -> RVal { a.get(i).map(|x| x.value.clone()).unwrap_or(RVal::Null) }
fn e_gn(a: &[EvalArg], name: &str) -> Option<RVal> {
    a.iter().find(|x| x.name.as_deref() == Some(name)).map(|x| x.value.clone())
}
fn e_str(v: &RVal) -> Option<String> {
    match v { RVal::Character(c, _) => c.first().and_then(|x| x.as_ref().map(|s| s.to_string())), _ => None }
}
fn e_num(v: &RVal) -> Option<f64> { v.as_reals().ok().and_then(|r| r.into_iter().flatten().next()) }
fn rnums(v: &[f64]) -> RVal { RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default()) }
fn rchars(v: &[String]) -> RVal { RVal::Character(v.iter().map(|s| Some(Arc::from(s.as_str()))).collect(), Attrs::default()) }
fn err(m: &str) -> R2Err { R2Err { msg: m.to_string(), kind: ErrKind::Runtime } }

/// Parse cSEM-style syntax → (construct names in measurement order,
/// indicators per construct, structural (endogenous, predecessors) pairs).
fn parse_model(model: &str) -> Result<(Vec<String>, Vec<Vec<String>>, Vec<(String, Vec<String>)>), R2Err> {
    let mut cnames = Vec::new();
    let mut inds = Vec::new();
    let mut structural = Vec::new();
    for raw in model.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let split = |rhs: &str| -> Vec<String> {
            rhs.split('+').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
        };
        if let Some(p) = line.find("=~") {
            cnames.push(line[..p].trim().to_string());
            inds.push(split(&line[p + 2..]));
        } else if let Some(p) = line.find('~') {
            structural.push((line[..p].trim().to_string(), split(&line[p + 1..])));
        }
    }
    if cnames.is_empty() { return Err(err("plssem: model has no measurement (=~) equations")); }
    Ok((cnames, inds, structural))
}

fn sample_sd(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn bi_plssem(a: &[EvalArg]) -> Result<RVal, R2Err> {
    // Arguments (accept cSEM-style dotted names and plain positional).
    let data_rv = e_gn(a, ".data").or_else(|| e_gn(a, "data")).unwrap_or_else(|| e_gv(a, 0));
    let df = match &data_rv {
        RVal::DataFrame(df) => df.clone(),
        _ => return Err(err("plssem: .data must be a data frame")),
    };
    let model_rv = e_gn(a, ".model").or_else(|| e_gn(a, "model")).unwrap_or_else(|| e_gv(a, 1));
    let model_str = e_str(&model_rv).ok_or_else(|| err("plssem: .model must be a string"))?;
    let reps = e_gn(a, ".R").or_else(|| e_gn(a, "R")).and_then(|v| e_num(&v)).map(|x| x as usize).unwrap_or(200);
    let seed = e_gn(a, ".seed").and_then(|v| e_num(&v)).map(|x| x as u64).unwrap_or(1);

    let (cnames, inds, struct_pairs) = parse_model(&model_str)?;
    let cpos = |name: &str| cnames.iter().position(|c| c == name);

    // Column name → Option<f64> data.
    let coldata = |name: &str| -> Option<Vec<Option<f64>>> {
        df.columns.iter().find(|(n, _)| n.as_ref() == name)
            .and_then(|(_, v)| v.as_reals().ok())
    };

    // Assemble used indicator columns (block order) + complete-case filter.
    let mut used_cols: Vec<Vec<Option<f64>>> = Vec::new();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut ind_names: Vec<Vec<String>> = Vec::new();
    for block in &inds {
        let mut idxs = Vec::new();
        for name in block {
            let col = coldata(name).ok_or_else(|| err(&format!("plssem: indicator '{name}' not found in data")))?;
            idxs.push(used_cols.len());
            used_cols.push(col);
        }
        blocks.push(idxs);
        ind_names.push(block.clone());
    }
    let n_raw = used_cols.first().map(|c| c.len()).unwrap_or(0);
    let keep: Vec<usize> = (0..n_raw)
        .filter(|&i| used_cols.iter().all(|c| c.get(i).map(|o| o.is_some()).unwrap_or(false)))
        .collect();
    let n = keep.len();
    if n < 10 { return Err(err("plssem: fewer than 10 complete cases")); }

    // Flat column-major matrix over complete cases.
    let ncol = used_cols.len();
    let mut data = vec![0.0f64; ncol * n];
    for (c, col) in used_cols.iter().enumerate() {
        for (r, &row) in keep.iter().enumerate() {
            data[c * n + r] = col[row].unwrap();
        }
    }

    // Structural predecessors per construct.
    let mut structural = vec![Vec::new(); cnames.len()];
    for (endo, preds) in &struct_pairs {
        let j = cpos(endo).ok_or_else(|| err(&format!("plssem: construct '{endo}' has no measurement model")))?;
        for p in preds {
            let k = cpos(p).ok_or_else(|| err(&format!("plssem: predictor '{p}' has no measurement model")))?;
            structural[j].push(k);
        }
    }
    let model = PlsModel { blocks, structural };

    // Fit + parallel bootstrap.
    let t0 = std::time::Instant::now();
    let fit = fit_pls(&data, n, &model, 300, 1e-8);
    let boot = if reps > 0 { bootstrap_paths(&data, n, &model, reps, seed) } else { Vec::new() };
    let elapsed = t0.elapsed().as_secs_f64();

    // Report + collect result vectors.
    soutln!("");
    soutln!("PLS-SEM (plssem) — Mode A, path-weighting");
    soutln!("  n = {n} complete cases | bootstrap reps = {reps} | iterations = {} | {:.3}s",
        fit.iters, elapsed);
    soutln!("");
    soutln!("Structural paths:");
    soutln!("  {:<26} {:>9} {:>9} {:>8} {:>20}", "Path", "Estimate", "Std.Err", "t", "95% CI (percentile)");
    let mut labels = Vec::new();
    let mut est = Vec::new();
    let mut se = Vec::new();
    let mut tval = Vec::new();
    let mut ci_lo = Vec::new();
    let mut ci_hi = Vec::new();
    for j in 0..cnames.len() {
        for (k, &pred) in model.structural[j].iter().enumerate() {
            let coef = fit.paths[j][k];
            let (s, lo, hi) = if !boot.is_empty() {
                let mut d = boot[j][k].clone();
                d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (sample_sd(&d), percentile(&d, 0.025), percentile(&d, 0.975))
            } else { (f64::NAN, f64::NAN, f64::NAN) };
            let t = if s > 1e-12 { coef / s } else { f64::NAN };
            let label = format!("{} -> {}", cnames[pred], cnames[j]);
            soutln!("  {:<26} {:>9.3} {:>9.3} {:>8.2}   [{:>6.3}, {:>6.3}]",
                label, coef, s, t, lo, hi);
            labels.push(label); est.push(coef); se.push(s); tval.push(t); ci_lo.push(lo); ci_hi.push(hi);
        }
    }
    soutln!("");
    let mut r2_lab = Vec::new();
    let mut r2_val = Vec::new();
    for j in 0..cnames.len() {
        if !model.structural[j].is_empty() { r2_lab.push(cnames[j].clone()); r2_val.push(fit.r2[j]); }
    }
    let r2_line: Vec<String> = r2_lab.iter().zip(&r2_val).map(|(l, v)| format!("{l} = {v:.3}")).collect();
    soutln!("R² (endogenous): {}", r2_line.join("   "));
    soutln!("");

    let mut fields = std::collections::HashMap::new();
    fields.insert(Arc::from("path_labels"), rchars(&labels));
    fields.insert(Arc::from("path_coef"), rnums(&est));
    fields.insert(Arc::from("path_se"), rnums(&se));
    fields.insert(Arc::from("path_t"), rnums(&tval));
    fields.insert(Arc::from("ci_lower"), rnums(&ci_lo));
    fields.insert(Arc::from("ci_upper"), rnums(&ci_hi));
    fields.insert(Arc::from("r2_labels"), rchars(&r2_lab));
    fields.insert(Arc::from("r2"), rnums(&r2_val));
    fields.insert(Arc::from("n"), RVal::Integer(vec![Some(n as i32)].into(), Attrs::default()));
    fields.insert(Arc::from("reps"), RVal::Integer(vec![Some(reps as i32)].into(), Attrs::default()));
    fields.insert(Arc::from("iterations"), RVal::Integer(vec![Some(fit.iters as i32)].into(), Attrs::default()));
    fields.insert(Arc::from("elapsed"), RVal::Numeric(vec![Some(elapsed)].into(), Attrs::default()));
    let _ = ind_names;
    Ok(RVal::TypeInstance(TypeInstance { type_name: Arc::from("plssem"), fields }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic normal RNG (Box–Muller over an LCG) for reproducible
    // synthetic data.
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.u().max(1e-12);
            let u2 = self.u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Build synthetic data with a known structure:
    ///   Stress → Anxiety (0.50), Stress → Depression (0.30),
    ///   Anxiety → Depression (0.40). 7 reflective indicators/construct,
    ///   loading ≈ 0.75. Column order: Stress(0..7), Anxiety(7..14),
    ///   Depression(14..21). Data returned column-major (col*n + row).
    fn synth(n: usize) -> (Vec<f64>, usize) {
        let mut rng = Rng(42);
        let load = 0.75f64;
        let noise = (1.0 - load * load).sqrt();
        let ncol = 21;
        let mut data = vec![0.0; ncol * n];
        for i in 0..n {
            let stress = rng.normal();
            let anx = 0.50 * stress + (1.0 - 0.25f64).sqrt() * rng.normal();
            // depression variance normalized approx
            let dep = 0.30 * stress + 0.40 * anx + 0.70 * rng.normal();
            let latents = [stress, anx, dep];
            for c in 0..3 {
                for k in 0..7 {
                    let col = c * 7 + k;
                    data[col * n + i] = load * latents[c] + noise * rng.normal();
                }
            }
        }
        (data, n)
    }

    fn model() -> PlsModel {
        PlsModel {
            blocks: vec![(0..7).collect(), (7..14).collect(), (14..21).collect()],
            // 0=Stress (exo), 1=Anxiety (<-Stress), 2=Depression (<-Stress,Anxiety)
            structural: vec![vec![], vec![0], vec![0, 1]],
        }
    }

    #[test]
    fn recovers_known_paths() {
        let (data, n) = synth(1500);
        let fit = fit_pls(&data, n, &model(), 300, 1e-8);
        // Anxiety ~ Stress
        let a_s = fit.paths[1][0];
        // Depression ~ Stress, ~ Anxiety
        let d_s = fit.paths[2][0];
        let d_a = fit.paths[2][1];
        eprintln!("paths: A~S={a_s:.3} D~S={d_s:.3} D~A={d_a:.3} iters={}", fit.iters);
        eprintln!("R²: Anx={:.3} Dep={:.3}", fit.r2[1], fit.r2[2]);
        assert!((0.40..0.60).contains(&a_s), "A~S={a_s}");
        assert!((0.18..0.42).contains(&d_s), "D~S={d_s}");
        assert!((0.30..0.52).contains(&d_a), "D~A={d_a}");
        // loadings should be high (well-measured constructs)
        let mean_load = fit.loadings[0].iter().sum::<f64>() / 7.0;
        assert!(mean_load > 0.6, "mean loading {mean_load}");
    }

    #[test]
    fn bootstrap_produces_distributions() {
        let (data, n) = synth(600);
        let boot = bootstrap_paths(&data, n, &model(), 50, 7);
        // Depression has 2 path coefficients, each with 50 bootstrap draws.
        assert_eq!(boot[2].len(), 2);
        assert_eq!(boot[2][0].len(), 50);
        let m: f64 = boot[2][1].iter().sum::<f64>() / 50.0;
        assert!((0.25..0.55).contains(&m), "mean D~A bootstrap {m}");
    }
}
