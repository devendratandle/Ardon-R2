//! Phase J.4 (whole-function slice 1) — iterative kernels. A whole numeric
//! function with a counted loop carrying scalar state across per-iteration
//! vector reductions (gradient descent / Newton / fixed-point / EM updates)
//! compiles to ONE native unit — no per-iteration RVal allocation. Must match
//! the interpreter, including R's descending `1:0` loop edge, and must fall
//! back (never 0-init) on read-before-define.

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
fn gradient_descent_jits_and_matches_interpreter() {
    let ex = string(eval_last(
        r#"explain(function(x) { b <- 0; for(it in 1:50) b <- b + 0.5*mean(x - b); b })"#,
    ));
    assert!(ex.contains("JIT-compiled"), "expected JIT, got: {ex}");

    let d = scalar(eval_last(
        r#"
set.seed(1); x <- rnorm(500)
gd <- function(x) { b <- 0; for(it in 1:50) b <- b + 0.5*mean(x - b); b }
b <- 0; for(it in 1:50) b <- b + 0.5*mean(x - b)
abs(gd(x) - b)
"#,
    ));
    assert!(d < 1e-12, "gradient-descent JIT vs interpreter diff = {d}");
}

#[test]
fn two_vector_gd_converges_to_ols() {
    let d = scalar(eval_last(
        r#"
set.seed(2); x <- rnorm(400); y <- 0.7*x + rnorm(400)
gd2 <- function(x, y) { b <- 0; for(it in 1:200) { g <- mean((y - b*x)*x); b <- b + 0.5*g }; b }
abs(gd2(x, y) - sum(x*y)/sum(x*x))
"#,
    ));
    assert!(d < 1e-9, "GD regression vs OLS diff = {d}");
}

#[test]
fn descending_one_to_zero_matches_r() {
    // R's `1:0` iterates 1 THEN 0 — the compiled loop must do the same.
    let d = scalar(eval_last(
        r#"
set.seed(3); x <- rnorm(100)
e0 <- function(x) { b <- 5; for(it in 1:0) b <- b + mean(x); b }
b <- 5; for(it in 1:0) b <- b + mean(x)
abs(e0(x) - b)
"#,
    ));
    assert!(d == 0.0, "1:0 edge diff = {d}");
}

#[test]
fn scalar_only_loop_with_outer_reduction() {
    // Newton iteration: reduction hoisted outside, pure-scalar loop inside.
    let d = scalar(eval_last(
        r#"
set.seed(4); x <- rnorm(300)
nsq <- function(x) { m <- mean(x*x); r <- m; for(it in 1:30) r <- 0.5*(r + m/r); r }
abs(nsq(x) - sqrt(mean(x*x)))
"#,
    ));
    assert!(d < 1e-12, "Newton sqrt diff = {d}");
}

#[test]
fn while_convergence_matches_interpreter() {
    // Convergence loop: reduction in the body, scalar tolerance condition.
    let d = scalar(eval_last(
        r#"
set.seed(5); x <- rnorm(400)
wgd <- function(x) { b <- 0; g <- 1; while (abs(g) > 1e-10) { g <- mean(x - b); b <- b + 0.5*g }; b }
b <- 0; g <- 1; while (abs(g) > 1e-10) { g <- mean(x - b); b <- b + 0.5*g }
abs(wgd(x) - b)
"#,
    ));
    assert!(d < 1e-12, "while-convergence diff = {d}");
}

#[test]
fn adaptive_ifelse_step_matches() {
    // Scalar if/else value inside the loop (adaptive step size).
    let d = scalar(eval_last(
        r#"
set.seed(6); x <- rnorm(300)
ad <- function(x) { b <- 0; for(it in 1:60) { g <- mean(x - b); s <- if (abs(g) > 0.5) 0.9 else 0.3; b <- b + s*g }; b }
b <- 0; for(it in 1:60) { g <- mean(x - b); s <- if (abs(g) > 0.5) 0.9 else 0.3; b <- b + s*g }
abs(ad(x) - b)
"#,
    ));
    assert!(d < 1e-12, "adaptive if/else diff = {d}");
}

#[test]
fn while_false_at_entry_runs_zero_iterations() {
    let v = scalar(eval_last(
        r#"
x <- as.numeric(1:10)
z0 <- function(x) { b <- 7; while (b > 100) { b <- b - mean(x) }; b }
z0(x)
"#,
    ));
    assert!((v - 7.0).abs() < 1e-12, "zero-iteration while = {v}");
}

#[test]
fn read_before_define_falls_back() {
    // `g` is read before it is assigned inside the loop — R errors; the JIT
    // must NOT compile this (a 0.0 phi-init would silently produce a value).
    let ex = string(eval_last(
        r#"explain(function(x) { b <- 0; for(it in 1:3) { b <- b + g; g <- mean(x) }; b })"#,
    ));
    assert!(!ex.contains("JIT-compiled"), "read-before-define must not JIT: {ex}");
}
