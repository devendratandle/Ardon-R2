//! Phase J.4 brick 2 — multi-reduction scalar kernels. Functions that *combine*
//! whole-vector reductions (ratios of sums, two-pass mean/variance) compile to
//! native code — several fused loops with scalar locals threaded between them,
//! no intermediate vector materialised. Verified against the interpreter and R.

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

fn string(v: RVal) -> String {
    match v {
        RVal::Character(d, _) => d.iter().next().cloned().flatten().map(|s| s.to_string()).unwrap_or_default(),
        other => panic!("expected character, got {:?}", other),
    }
}

#[test]
fn regression_coefficient_jits_and_matches() {
    let ex = string(eval_last(r#"explain(function(x, y) sum(x*y) / sum(x*x))"#));
    assert!(ex.contains("JIT-compiled"), "expected JIT, got: {ex}");
    let d = scalar(eval_last(
        r#"
set.seed(1); x <- rnorm(500); y <- rnorm(500)
reg <- function(x, y) sum(x*y) / sum(x*x)
reg(x, y) - (sum(x*y) / sum(x*x))
"#,
    ));
    assert!(d.abs() < 1e-12, "regression-coef diff = {d}");
}

#[test]
fn variance_two_pass_matches_r() {
    let d = scalar(eval_last(
        r#"
set.seed(2); x <- rnorm(500)
va <- function(x) { m <- mean(x); sum((x-m)*(x-m)) / (length(x)-1) }
va(x) - var(x)
"#,
    ));
    assert!(d.abs() < 1e-9, "variance diff = {d}");
}

#[test]
fn covariance_matches_r() {
    let d = scalar(eval_last(
        r#"
set.seed(3); x <- rnorm(500); y <- rnorm(500)
co <- function(x, y) { mx <- mean(x); my <- mean(y); sum((x-mx)*(y-my)) / (length(x)-1) }
co(x, y) - cov(x, y)
"#,
    ));
    assert!(d.abs() < 1e-9, "covariance diff = {d}");
}

#[test]
fn pearson_correlation_matches_r() {
    let d = scalar(eval_last(
        r#"
set.seed(4); x <- rnorm(500); y <- rnorm(500)
pr <- function(x, y) {
  mx <- mean(x); my <- mean(y)
  sxy <- sum((x-mx)*(y-my)); sxx <- sum((x-mx)*(x-mx)); syy <- sum((y-my)*(y-my))
  sxy / sqrt(sxx*syy)
}
pr(x, y) - cor(x, y)
"#,
    ));
    assert!(d.abs() < 1e-9, "correlation diff = {d}");
}

#[test]
fn single_reduction_paths_unaffected() {
    // Existing specialised shapes must keep their own (non-kernel) codegen.
    for (body, kind) in [
        ("function(v) sum(v)", "Vector1ToScalar"),
        ("function(a, b) sum(a*b)", "Vector2ToScalar"),
        ("function(v) v*v", "VectorMap"),
    ] {
        let ex = string(eval_last(&format!("explain({body})")));
        assert!(ex.contains("JIT-compiled") && ex.contains(kind), "{body} -> {ex}");
    }
}
