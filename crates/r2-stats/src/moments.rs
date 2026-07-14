//! Shared centred-moment primitives — THE single implementation of
//! `Σ(x-mx)(y-my)`-family math used by cov/cor/cor-matrix/cor.test (and
//! any future consumer). One implementation ⇒ one set of numerics: every
//! function that reports a correlation or covariance agrees to the last
//! bit, and the two-pass (centred) form avoids the catastrophic
//! cancellation of the naive `E[x²]−E[x]²` shortcut.

use r2_types::Real;

/// Two-pass centred moments of one DENSE f64 slice: `(n, mean, sxx)`
/// where `sxx = Σ(x-mean)²` — the shared core of sd / total-SS / R².
pub fn centred1_dense(x: &[f64]) -> (usize, f64, f64) {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let sxx = x.iter().map(|v| { let d = v - mean; d * d }).sum();
    (n, mean, sxx)
}

/// Two-pass centred moments over a pair of DENSE f64 slices of equal
/// length: `(n, mx, my, sxx, syy, sxy)`.
pub fn centred2_dense(x: &[f64], y: &[f64]) -> (usize, f64, f64, f64, f64, f64) {
    let n = x.len().min(y.len());
    let nf = n as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for i in 0..n { sx += x[i]; sy += y[i]; }
    let (mx, my) = (sx / nf, sy / nf);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    (n, mx, my, sxx, syy, sxy)
}

/// Pairwise-complete centred moments over NA-aware inputs: rows where
/// either side is NA are dropped (R's `use = "complete.obs"` for the
/// two-vector case). Returns None when fewer than 2 complete pairs.
pub fn centred2_pairwise(x: &[Real], y: &[Real]) -> Option<(usize, f64, f64, f64, f64, f64)> {
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter_map(|(a, b)| match (a, b) { (Some(a), Some(b)) => Some((*a, *b)), _ => None })
        .collect();
    if pairs.len() < 2 { return None; }
    let (xs, ys): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    Some(centred2_dense(&xs, &ys))
}

/// Pearson correlation from moments (NaN when either variance is zero —
/// callers map that to NA, matching R's warning case).
pub fn pearson_from(sxx: f64, syy: f64, sxy: f64) -> f64 {
    if sxx > 0.0 && syy > 0.0 { sxy / (sxx * syy).sqrt() } else { f64::NAN }
}

/// Sample covariance from moments.
pub fn cov_from(n: usize, sxy: f64) -> f64 { sxy / (n as f64 - 1.0) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments_match_hand_computation() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let (n, mx, my, sxx, syy, sxy) = centred2_dense(&x, &y);
        assert_eq!(n, 4);
        assert!((mx - 2.5).abs() < 1e-12 && (my - 5.0).abs() < 1e-12);
        assert!((sxx - 5.0).abs() < 1e-12 && (syy - 20.0).abs() < 1e-12 && (sxy - 10.0).abs() < 1e-12);
        assert!((pearson_from(sxx, syy, sxy) - 1.0).abs() < 1e-12);
        assert!((cov_from(n, sxy) - 10.0 / 3.0).abs() < 1e-12);
    }
    #[test]
    fn pairwise_drops_na_rows() {
        let x = [Some(1.0), None, Some(3.0), Some(4.0)];
        let y = [Some(2.0), Some(9.0), None, Some(8.0)];
        let (n, ..) = centred2_pairwise(&x, &y).unwrap();
        assert_eq!(n, 2); // only rows 0 and 3 are complete
    }
}
