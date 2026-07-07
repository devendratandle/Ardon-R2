//! `backsolve` / `forwardsolve` / `rcond` / `kappa` — triangular solves and
//! condition numbers, verified against R 4.5.3 reference values.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn eval_last(script: &str) -> RVal {
    let mut e = Engine::new();
    let exprs = Parser::parse(script).expect("parse ok");
    let mut last = RVal::Null;
    for ex in exprs {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

fn scalar(v: RVal) -> f64 {
    match v {
        RVal::Numeric(d, _) => d.iter().next().and_then(|x| *x).expect("scalar"),
        other => panic!("expected numeric scalar, got {:?}", other),
    }
}

#[test]
fn backsolve_upper_matches_r() {
    // U = [[2,1,4],[0,3,5],[0,0,6]]; U y = c(5,8,6) → y = c(0,1,1).
    let d = scalar(eval_last(
        r#"
U <- matrix(c(2,0,0, 1,3,0, 4,5,6), 3, 3)
sum(abs(backsolve(U, c(5,8,6)) - c(0,1,1)))
"#,
    ));
    assert!(d < 1e-9, "backsolve off by {d}");
}

#[test]
fn backsolve_transpose_matches_r() {
    // Uᵀ y = c(5,8,6) → R: (2.5, 1.833333, -2.194444).
    let d = scalar(eval_last(
        r#"
U <- matrix(c(2,0,0, 1,3,0, 4,5,6), 3, 3)
sum(abs(backsolve(U, c(5,8,6), transpose=TRUE) - c(2.5, 1.8333333333, -2.1944444444)))
"#,
    ));
    assert!(d < 1e-6, "backsolve transpose off by {d}");
}

#[test]
fn backsolve_solves_system() {
    // The returned y must satisfy U %*% y == x.
    let resid = scalar(eval_last(
        r#"
U <- matrix(c(2,0,0, 1,3,0, 4,5,6), 3, 3)
x <- c(5, 8, 6)
y <- backsolve(U, x)
max(abs(U %*% y - x))
"#,
    ));
    assert!(resid < 1e-9, "U y - x residual = {resid}");
}

#[test]
fn forwardsolve_lower_matches_r() {
    // L lower-triangular = [[2,0,0],[1,3,0],[4,5,6]]; L y = c(4,11,32).
    // y1=2, y2=(11-2)/3=3, y3=(32-8-15)/6=1.5.
    let resid = scalar(eval_last(
        r#"
L <- matrix(c(2,1,4, 0,3,5, 0,0,6), 3, 3)
y <- forwardsolve(L, c(4,11,32))
max(abs(L %*% y - c(4,11,32)))
"#,
    ));
    assert!(resid < 1e-9, "forwardsolve residual = {resid}");
}

#[test]
fn rcond_matches_r() {
    // A = [[4,1,0],[1,3,1],[0,1,2]]; R: rcond = 0.225.
    let d = scalar(eval_last(
        "abs(rcond(matrix(c(4,1,0, 1,3,1, 0,1,2), 3, 3)) - 0.225)",
    ));
    assert!(d < 1e-6, "rcond off by {d}");
}

#[test]
fn kappa_matches_r() {
    // R: kappa(A, exact=TRUE) = 3.732051 (symmetric), 4.054454 (non-symmetric).
    let d1 = scalar(eval_last("abs(kappa(matrix(c(4,1,0, 1,3,1, 0,1,2), 3, 3)) - 3.7320508)"));
    assert!(d1 < 1e-5, "kappa symmetric off by {d1}");
    let d2 = scalar(eval_last("abs(kappa(matrix(c(3,0,0, 1,2,0, 1,1,1), 3, 3)) - 4.0544544)"));
    assert!(d2 < 1e-5, "kappa non-symmetric off by {d2}");
}
