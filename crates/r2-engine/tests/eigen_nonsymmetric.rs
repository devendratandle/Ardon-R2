//! `eigen()` on non-symmetric matrices — the general real eigenvalue path
//! (`dgeev`: Hessenberg + Francis double-shift QR). Before this, asymmetric
//! matrices were wrongly run through the symmetric solver. Symmetric input must
//! keep its existing behaviour; real spectra return correct value + vector
//! pairs; complex spectra expose real parts in `$values` and imaginary parts in
//! `$imaginary`.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn eval_scalar(script: &str) -> f64 {
    let mut e = Engine::new();
    let exprs = Parser::parse(script).expect("parse ok");
    let mut last = RVal::Null;
    for ex in exprs {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    match last {
        RVal::Numeric(d, _) => d.iter().next().and_then(|x| *x).expect("scalar"),
        other => panic!("expected numeric scalar, got {:?}", other),
    }
}

#[test]
fn symmetric_path_unchanged() {
    // eigen(matrix(c(2,1,1,2),2,2))$values == c(3,1).
    let d = eval_scalar("sum(abs(eigen(matrix(c(2,1,1,2),2,2))$values - c(3,1)))");
    assert!(d < 1e-9, "symmetric values off by {d}");
}

#[test]
fn nonsymmetric_real_values_match() {
    // Lower-triangular non-symmetric → eigenvalues are the diagonal 6,4,2.
    let d = eval_scalar(
        r#"
R1 <- matrix(c(6,2,1, 0,4,3, 0,0,2), 3, 3)
sum(abs(sort(eigen(R1)$values) - c(2,4,6)))
"#,
    );
    assert!(d < 1e-6, "non-symmetric real values off by {d}");
}

#[test]
fn nonsymmetric_eigenpair_residual_is_zero() {
    // The returned vectors must satisfy A v = lambda v.
    let resid = eval_scalar(
        r#"
R1 <- matrix(c(6,2,1, 0,4,3, 0,0,2), 3, 3)
e <- eigen(R1)
v <- e$vectors[,1]
max(abs(R1 %*% v - e$values[1] * v))
"#,
    );
    assert!(resid < 1e-6, "eigenpair residual = {resid}");
}

#[test]
fn complex_spectrum_exposes_imaginary_parts() {
    // Column-major of R's byrow c(4,1,2,0, 3,5,1,2, 0,1,6,3, 1,0,2,7) 4x4.
    // Complex pair 3.247928 ± 1.219032i.
    let max_im = eval_scalar(
        r#"
C1 <- matrix(c(4,3,0,1, 1,5,1,0, 2,1,6,2, 0,2,3,7), 4, 4)
max(abs(eigen(C1)$imaginary))
"#,
    );
    assert!((max_im - 1.219032).abs() < 1e-4, "max imaginary = {max_im}");
}
