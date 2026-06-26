//! Distance kernels — Phase K.11.

use crate::par_for_rayon;

// ════════════════════════════════════════════════════════════════════
// Phase K.11 — Distance kernels
// ════════════════════════════════════════════════════════════════════
//
// Pairwise distance primitives shared by k-means, knn, hierarchical
// clustering, and similar pattern-detection workloads. Before K.11,
// each builtin rolled its own distance loop with the same shape but
// no parallel dispatch and no NA handling discipline. K.11 gives
// them one kernel.
//
// Distance operates on TWO same-length f64 slices (point coordinates).
// `pairwise_distance` operates on a row-major or column-major matrix
// + two row indices.
//
//   Euclidean: sqrt(sum((a_i - b_i)²))
//   Manhattan: sum(|a_i - b_i|)
//   Cosine:    1 - (a·b) / (||a|| · ||b||)
//
// NA semantics: any None in either operand at the same position
// causes that pair to be skipped (na.rm=TRUE behavior; matches R's
// `dist(..., upper=TRUE, diag=TRUE)`). If all positions have a NA on
// at least one side, returns None.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceOp {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Distance between two same-length f64 slices. `None` if both slices
/// have NA at every position (degenerate) or are empty.
pub fn distance(op: DistanceOp, a: &[Option<f64>], b: &[Option<f64>]) -> Option<f64> {
    assert_eq!(a.len(), b.len(), "distance: length mismatch");
    if a.is_empty() { return None; }
    let mut count = 0usize;
    match op {
        DistanceOp::Euclidean => {
            let mut ss = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    let d = xv - yv;
                    ss += d * d;
                    count += 1;
                }
            }
            if count == 0 { None } else { Some(ss.sqrt()) }
        }
        DistanceOp::Manhattan => {
            let mut s = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    s += (xv - yv).abs();
                    count += 1;
                }
            }
            if count == 0 { None } else { Some(s) }
        }
        DistanceOp::Cosine => {
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    dot += xv * yv;
                    norm_a += xv * xv;
                    norm_b += yv * yv;
                    count += 1;
                }
            }
            if count == 0 || norm_a == 0.0 || norm_b == 0.0 { return None; }
            Some(1.0 - dot / (norm_a.sqrt() * norm_b.sqrt()))
        }
    }
}

/// Compute pairwise distances for all pairs in a flat column-major
/// matrix (n rows × p cols, stored as `&[Option<f64>]` of length n*p).
/// Returns an n×n distance matrix in row-major order (Vec of length n²).
/// Uses Rayon when `n >= threshold` (n² scales fast; we go parallel early).
pub fn pairwise_distance(
    op: DistanceOp,
    data: &[Option<f64>],
    nrow: usize,
    ncol: usize,
) -> Vec<Option<f64>> {
    assert_eq!(data.len(), nrow * ncol, "pairwise_distance: shape mismatch");
    // Row i of the matrix = elements data[i*ncol..(i+1)*ncol] when
    // stored row-major. For our convention (matching how k-means + knn
    // builtins typically pass data), we assume row-major.
    let row = |i: usize| -> Vec<Option<f64>> {
        data[i * ncol..(i + 1) * ncol].to_vec()
    };
    let total_work = nrow.saturating_mul(nrow).saturating_mul(ncol);
    let go_parallel = matches!(
        r2_oracle::dispatch(r2_oracle::Op::PerPointDistance, r2_oracle::Shape::nmk(nrow, nrow, ncol)),
        r2_oracle::Backend::Rayon
    );
    if go_parallel && nrow >= 16 {
        let _ = total_work;
        par_for_rayon(nrow * nrow, |idx| {
            let i = idx / nrow;
            let j = idx % nrow;
            if i == j { Some(0.0) }
            else if j < i {
                // Diagonal symmetry — we'll fill from the (i<j) cell.
                Some(0.0) // placeholder; overwritten below
            } else {
                let ri = row(i); let rj = row(j);
                distance(op, &ri, &rj)
            }
        }).into_iter()
            .enumerate()
            .map(|(idx, v)| {
                let i = idx / nrow;
                let j = idx % nrow;
                if j < i {
                    // Mirror from (j, i): same distance.
                    let rj = row(j); let ri = row(i);
                    distance(op, &rj, &ri)
                } else { v }
            })
            .collect()
    } else {
        let mut out = vec![None; nrow * nrow];
        for i in 0..nrow {
            let ri = row(i);
            out[i * nrow + i] = Some(0.0);
            for j in (i + 1)..nrow {
                let rj = row(j);
                let d = distance(op, &ri, &rj);
                out[i * nrow + j] = d;
                out[j * nrow + i] = d;
            }
        }
        out
    }
}
