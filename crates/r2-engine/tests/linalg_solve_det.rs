//! Regression tests for the newly-exposed `solve()` / `det()` builtins
//! and `svd` singular-value accuracy. A = [[4,1],[1,3]] (column-major
//! c(4,1,1,3)): det = 11, inverse = [3,-1,-1,4]/11, singular values
//! (symmetric PD ⇒ eigenvalues) = (5±√5)/2 = 4.618034, 2.381966.

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
        RVal::Numeric(xs, _) => xs.iter().next().and_then(|x| *x).expect("scalar"),
        other => panic!("expected numeric scalar, got {:?}", other),
    }
}

#[test]
fn det_of_2x2() {
    let v = eval_last("det(matrix(c(4,1,1,3), nrow=2))");
    assert!((scalar(v) - 11.0).abs() < 1e-10);
}

#[test]
fn solve_linear_system() {
    // A x = c(1,2) → x = A⁻¹·[1,2] = [1,7]/11 = [0.0909.., 0.6363..]
    let x1 = scalar(eval_last("solve(matrix(c(4,1,1,3), nrow=2), c(1,2))[1]"));
    let x2 = scalar(eval_last("solve(matrix(c(4,1,1,3), nrow=2), c(1,2))[2]"));
    assert!((x1 - 1.0 / 11.0).abs() < 1e-10, "x1={}", x1);
    assert!((x2 - 7.0 / 11.0).abs() < 1e-10, "x2={}", x2);
}

#[test]
fn solve_inverse_element() {
    // solve(A)[1,1] = 3/11
    let v = scalar(eval_last("solve(matrix(c(4,1,1,3), nrow=2))[1,1]"));
    assert!((v - 3.0 / 11.0).abs() < 1e-10, "inv[1,1]={}", v);
}

#[test]
fn svd_singular_values_exact() {
    let d1 = scalar(eval_last("svd(matrix(c(4,1,1,3), nrow=2))$d[1]"));
    let d2 = scalar(eval_last("svd(matrix(c(4,1,1,3), nrow=2))$d[2]"));
    assert!((d1 - 4.618033988749895).abs() < 1e-9, "d1={}", d1);
    assert!((d2 - 2.381966011250105).abs() < 1e-9, "d2={}", d2);
}
