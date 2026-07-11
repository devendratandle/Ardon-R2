//! Phase J.4 (final piece) — matrix-state iterative kernels. A whole
//! `function(X, y)` with `X %*% b` / `t(X) %*% r` statements and two length
//! classes (n- and p-vectors) compiles to one native unit calling the shared
//! matvec externs. Must match the interpreter exactly and fall back on any
//! unsupported shape.

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

const GD: &str = "gd <- function(X, y) {\n  b <- rep(0, ncol(X))\n  for (it in 1:200) {\n    r <- y - X %*% b\n    g <- t(X) %*% r\n    b <- b + (0.5/nrow(X)) * g\n  }\n  b\n}";

#[test]
fn matrix_gd_jits_and_matches_interpreter() {
    let ex = string(eval_last(&format!("{GD}\nexplain(gd)")));
    // JIT is x86_64-only (Cranelift PLT gate); on aarch64 these run interpreted.
    if cfg!(target_arch = "x86_64") { assert!(ex.contains("MatVecIterOut"), "expected matrix kernel, got: {ex}"); }

    let d = scalar(eval_last(&format!(
        "set.seed(1); X <- matrix(rnorm(300), 100, 3); y <- as.numeric(X %*% c(1,-2,0.5)) + 0.1*rnorm(100)\n\
         {GD}\n\
         b <- rep(0, ncol(X))\n\
         for (it in 1:200) {{ r <- y - X %*% b; g <- t(X) %*% r; b <- b + (0.5/nrow(X)) * g }}\n\
         max(abs(gd(X, y) - b))"
    )));
    assert!(d < 1e-9, "matrix GD JIT vs interpreter diff = {d}");
}

#[test]
fn matrix_gd_converges_near_truth() {
    let d = scalar(eval_last(&format!(
        "set.seed(2); X <- matrix(rnorm(2000), 500, 4); beta <- c(0.5, -1, 2, 0.3)\n\
         y <- as.numeric(X %*% beta) + 0.05*rnorm(500)\n\
         {GD}\n\
         max(abs(gd(X, y) - beta))"
    )));
    assert!(d < 0.05, "GD estimate off truth by {d}");
}

#[test]
fn output_length_is_ncol() {
    let v = scalar(eval_last(&format!(
        "X <- matrix(as.numeric(1:35), 5, 7); y <- as.numeric(1:5)\n\
         {GD}\n\
         0 + length(gd(X, y))"
    )));
    assert!((v - 7.0).abs() < 1e-9, "output length = {v}, expected ncol=7");
}

#[test]
fn unsupported_shape_falls_back() {
    // A `while` inside a matrix kernel isn't supported (v1) — must not JIT
    // as MatVecIterOut, and the interpreter must still produce the answer.
    let ex = string(eval_last(
        "f <- function(X, y) { b <- rep(0, ncol(X)); k <- 0; while (k < 3) { g <- t(X) %*% (y - X %*% b); b <- b + 0.1*g; k <- k + 1 }; b }\nexplain(f)",
    ));
    assert!(!ex.contains("MatVecIterOut"), "while-shape must not take the matrix kernel: {ex}");
}
