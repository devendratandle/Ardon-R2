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
fn nested_reduction_oneliner_jits() {
    // Brick 3: the textbook one-liner (mean nested inside sum) must JIT and match.
    let ex = string(eval_last(r#"explain(function(x) sum((x-mean(x))^2)/(length(x)-1))"#));
    assert!(ex.contains("JIT-compiled"), "expected JIT, got: {ex}");
    let d = scalar(eval_last(
        r#"
set.seed(5); x <- rnorm(400)
va <- function(x) sum((x-mean(x))^2)/(length(x)-1)
va(x) - var(x)
"#,
    ));
    assert!(d.abs() < 1e-9, "one-liner variance diff = {d}");
}

#[test]
fn vector_local_fused_away() {
    // Brick 3: a vector intermediate (`e <- pred-obs`) is fused, no buffer.
    let ex = string(eval_last(r#"explain(function(pred, obs) { e <- pred - obs; sqrt(mean(e*e)) })"#));
    assert!(ex.contains("JIT-compiled"), "expected JIT, got: {ex}");
    let d = scalar(eval_last(
        r#"
set.seed(6); a <- rnorm(400); b <- rnorm(400)
rmse <- function(pred, obs) { e <- pred - obs; sqrt(mean(e*e)) }
rmse(a, b) - sqrt(mean((a-b)^2))
"#,
    ));
    assert!(d.abs() < 1e-12, "rmse diff = {d}");
}

#[test]
fn correlation_oneliner_matches_r() {
    let d = scalar(eval_last(
        r#"
set.seed(7); x <- rnorm(400); y <- rnorm(400)
cr <- function(x,y) sum((x-mean(x))*(y-mean(y)))/sqrt(sum((x-mean(x))^2)*sum((y-mean(y))^2))
cr(x, y) - cor(x, y)
"#,
    ));
    assert!(d.abs() < 1e-9, "one-liner correlation diff = {d}");
}

#[test]
fn shared_primitive_reused_correctly() {
    // Brick 4: `d(x)=x-mean(x)` defined once and reused in var/cov/cor; CSE +
    // wave fusion must not change results. sd(x) reuses the same centred sum.
    for (formula, r) in [
        ("va <- function(x) { d <- function(v) v-mean(v); sum(d(x)*d(x))/(length(x)-1) }; va(x) - var(x)", ()),
        ("sdf <- function(x) { d <- function(v) v-mean(v); sqrt(sum(d(x)*d(x))/(length(x)-1)) }; sdf(x) - sd(x)", ()),
        ("cr <- function(x,y) { d <- function(v) v-mean(v); sum(d(x)*d(y))/sqrt(sum(d(x)*d(x))*sum(d(y)*d(y))) }; cr(x,y) - cor(x,y)", ()),
    ] {
        let _ = r;
        let d = scalar(eval_last(&format!("set.seed(9); x <- rnorm(300); y <- rnorm(300)\n{formula}")));
        assert!(d.abs() < 1e-9, "{formula} -> diff {d}");
    }
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
