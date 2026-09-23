//! `.Internal(name, ...)` — the bridge from R2-language functions to Rust
//! primitives, e.g. `.Internal("solve_lstsq", x_matrix, y_vector)`: users
//! write statistics in R2 syntax and only the heavy math runs in Rust.

#![allow(clippy::all)]

use std::collections::HashMap;
use std::sync::Arc;

use r2_stats::dist::{phi, qnorm_approx};
use r2_types::*;

use crate::{gv, val_to_str, Engine};
use crate::err;

/// The RNG draw behind `.Internal("rnorm_vec", …)`, which R-language
/// helper code still calls (the bi_* RNG builtins use r2_stats::rng).
fn r2_next_random() -> f64 { r2_stats::rng::next_random() }

pub(crate) fn bi_internal(e: &mut Engine, a: &[EvalArg], _: &EnvRef) -> Result<RVal, R2Err> {
    let name = val_to_str(&gv(a, 0));

    match name.as_str() {
        // Matrix operations
        "matmul" => {
            let a_mat = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal matmul: need matrix") };
            let b_mat = match &gv(a,2) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal matmul: need matrix") };
            Ok(RVal::Matrix(a_mat.matmul(&b_mat).map_err(|e| R2Err{msg:e,kind:ErrKind::Runtime})?))
        }
        "crossprod" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal crossprod: need matrix") };
            Ok(RVal::Matrix(m.crossprod()))
        }
        "crossprod_vec" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal crossprod_vec: need matrix") };
            let v: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = m.crossprod_vec(&v);
            Ok(rnums(&result))
        }
        // Linear algebra
        "solve" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal solve: need matrix") };
            let b: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = m.solve(&b).map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "solve_lstsq" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal solve_lstsq: need matrix") };
            let y: Vec<f64> = e.as_reals(&gv(a,2))?.into_iter().filter_map(|x| x).collect();
            let result = r2_linalg::dlsq_fused(m.nrow, m.ncol, &m.data, &y)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "inverse" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal inverse: need matrix") };
            let result = r2_linalg::dgetri(m.nrow, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(RVal::Matrix(Matrix::new(result, m.nrow, m.ncol)))
        }
        "cholesky" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal cholesky: need matrix") };
            let mut data = m.data.clone();
            r2_linalg::dpotrf(m.nrow, &mut data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(RVal::Matrix(Matrix::new(data, m.nrow, m.ncol)))
        }
        "eigenvalues" => {
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal eigenvalues: need matrix") };
            let result = r2_linalg::dsyev(m.nrow, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            Ok(rnums(&result))
        }
        "svd" => {
            // Full thin SVD: A = U · diag(d) · Vᵀ.
            let m = match &gv(a,1) { RVal::Matrix(m) => m.clone(), _ => return err!(Runtime, ".Internal svd: need matrix") };
            let (sigma, u_data, vt_data) = r2_linalg::dgesvd_full(m.nrow, m.ncol, &m.data)
                .map_err(|e| R2Err{msg:format!("{}", e),kind:ErrKind::Runtime})?;
            let n = m.ncol;
            // Transpose Vᵀ → V (R convention: $v holds V, not Vᵀ).
            let mut v_data = vec![0.0_f64; n * n];
            for i in 0..n { for j in 0..n { v_data[j * n + i] = vt_data[i * n + j]; } }
            let mut fields = HashMap::new();
            fields.insert(Arc::from("d"), rnums(&sigma));
            fields.insert(Arc::from("u"), RVal::Matrix(Matrix::new(u_data, m.nrow, n)));
            fields.insert(Arc::from("v"), RVal::Matrix(Matrix::new(v_data, n, n)));
            Ok(RVal::List(fields.into_iter().map(|(k,v)| (Some(k), v)).collect()))
        }
        // Random numbers
        "rnorm_vec" => {
            let n = e.scalar_f64(&gv(a,1))?.unwrap_or(1.0) as usize;
            let mu = e.scalar_f64(&gv(a,2))?.unwrap_or(0.0);
            let sigma = e.scalar_f64(&gv(a,3))?.unwrap_or(1.0);
            let vals: Vec<Real> = (0..n).map(|_| {
                let u1 = r2_next_random().max(1e-15);
                let u2 = r2_next_random();
                Some(mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos())
            }).collect();
            Ok(RVal::Numeric(vals.into(), Attrs::default()))
        }
        // Phi (normal CDF) for p-values
        "pnorm" => {
            let x = e.scalar_f64(&gv(a,1))?.unwrap_or(0.0);
            Ok(rnum(phi(x)))
        }
        "qnorm" => {
            let p = e.scalar_f64(&gv(a,1))?.unwrap_or(0.5);
            Ok(rnum(qnorm_approx(p)))
        }

        _ => err!(Runtime, ".Internal: unknown function '{}'", name),
    }
}
