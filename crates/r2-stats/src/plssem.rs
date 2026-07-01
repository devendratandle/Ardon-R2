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
/// distribution (Vec<f64> of length = admissible reps). Serial here;
/// the engine layer parallelizes across reps with Rayon.
pub fn bootstrap_paths(
    data: &[f64], n: usize, model: &PlsModel, reps: usize, seed: u64,
) -> Vec<Vec<Vec<f64>>> {
    let c = model.n_constructs();
    let ncol = data.len() / n;
    let mut out: Vec<Vec<Vec<f64>>> = model.structural.iter()
        .map(|p| vec![Vec::with_capacity(reps); p.len()])
        .collect();
    let mut rng = seed;
    let mut next = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (rng >> 33) as usize };
    let mut resampled = vec![0.0; ncol * n];
    for _ in 0..reps {
        // resample rows with replacement
        for i in 0..n {
            let src = next() % n;
            for col in 0..ncol { resampled[col * n + i] = data[col * n + src]; }
        }
        let fit = fit_pls(&resampled, n, model, 100, 1e-7);
        for j in 0..c {
            for (idx, &b) in fit.paths[j].iter().enumerate() {
                out[j][idx].push(b);
            }
        }
    }
    out
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
