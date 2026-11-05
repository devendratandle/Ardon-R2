//! What `qr()` and its accessors require and refuse. Their values are compared
//! with R itself in tests/differential/cases/qr.R.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> Vec<f64> {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("{script}: {}", err.msg));
    }
    match last {
        RVal::Matrix(m) => m.data,
        v => v.as_reals().unwrap().iter().map(|x| x.unwrap_or(f64::NAN)).collect(),
    }
}

fn err(script: &str) -> String {
    let mut e = Engine::new();
    for ex in Parser::parse(script).expect("parse ok") {
        if let Err(x) = e.eval(&ex) { return x.msg; }
    }
    panic!("expected an error from {script}");
}

const X: &str = "x <- cbind(c(1, 2, 3, 5), c(1, 0, 1, 1)); q <- qr(x); ";

#[test]
fn the_first_argument_must_be_a_decomposition() {
    for f in ["qr.Q(1)", "qr.R(1)", "qr.X(1)", "qr.qy(1, 1)", "qr.qty(1, 1)", "qr.resid(1, 1)", "qr.fitted(1, 1)"] {
        assert!(err(f).contains("not a QR decomposition"), "{f}");
    }
    assert!(err("qr.coef(1, 1)").contains("first argument must be a QR decomposition"));
}

#[test]
fn y_must_have_the_rows_of_the_matrix() {
    for f in ["qr.qy(q, 1:3)", "qr.qty(q, 1:3)", "qr.coef(q, 1:3)", "qr.resid(q, 1:3)", "qr.fitted(q, 1:3)"] {
        assert!(err(&format!("{X}{f}")).contains("must have the same number of rows"), "{f}");
    }
}

#[test]
fn missing_values_are_refused_where_r_refuses_them() {
    assert!(err("qr(c(1, NA, 3))").contains("NA/NaN/Inf in foreign function call (arg 1)"));
    assert!(err("qr(matrix(c(1, Inf, 3, 4), 2))").contains("NA/NaN/Inf"));
    assert!(err(&format!("{X}qr.qy(q, c(1, NA, 3, 4))")).contains("(arg 5)"));
    assert!(err(&format!("{X}qr.fitted(q, c(1, NA, 3, 4))")).contains("(arg 5)"));
    assert!(run(&format!("{X}qr.coef(q, c(1, NA, 3, 4))")).iter().all(|v| v.is_nan()), "qr.coef lets them through");
}

#[test]
fn a_singular_system_says_so() {
    let singular = "s <- cbind(1:3, 2 * (1:3), c(1, 0, 1)); ";
    assert!(err(&format!("{singular}qr.solve(s, 1:3)")).contains("singular matrix 'a' in solve"));
    assert!(err(&format!("{singular}solve(qr(s), 1:3)")).contains("singular matrix 'a' in 'solve'"));
    assert!(err("qr.solve(cbind(1:3, c(1, 0, 1)))").contains("only square matrices can be inverted"));
}

#[test]
fn k_and_ncol_are_checked() {
    assert!(err(&format!("{X}qr.fitted(q, 1:4, k = 3)")).contains("'k' is too large"));
    let pivoted = "q <- qr(cbind(1:3, 2 * (1:3), c(1, 0, 1))); ";
    assert!(err(&format!("{pivoted}qr.X(q, ncol = 2)")).contains("need larger value of 'ncol' as pivoting occurred"));
}

#[test]
fn lapack_says_it_is_not_implemented() {
    assert!(err("qr(diag(2), LAPACK = TRUE)").contains("not implemented"));
}

#[test]
fn the_fields_can_be_read() {
    assert_eq!(run(&format!("{X}q$rank")), vec![2.0]);
    assert_eq!(run(&format!("{X}q$pivot")), vec![1.0, 2.0]);
    assert_eq!(run(&format!("{X}length(q$qraux)")), vec![2.0]);
}

/// An ill-conditioned least-squares problem that the normal equations cannot
/// solve: the condition number of x'x is about 1e16.
#[test]
fn the_condition_number_is_not_squared() {
    let b = run("t <- 1:12 / 12; x <- cbind(1, t, t^2, t^3, t^4, t^5); y <- x %*% c(1, 1, 1, 1, 1, 1); qr.coef(qr(x), y)");
    for v in b { assert!((v - 1.0).abs() < 1e-8, "coefficient {v}"); }
}
