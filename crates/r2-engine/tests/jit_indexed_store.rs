//! Phase J.3 — indexed-STORE map JIT. An imperative loop that writes `y[i]`
//! from `x[i]`/`w[i]` (bare loop var, in-bounds) compiles to native code with
//! real `Store` instructions. Covers the cases the single-input VectorMap
//! recogniser does not: two inputs, multi-statement bodies, if/else values.

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
fn two_input_store_jits_and_matches() {
    let ex = string(eval_last(
        r#"explain(function(x, w){ y <- numeric(length(x)); for(i in 1:length(x)) y[i] <- x[i] + w[i]; y })"#,
    ));
    // JIT is x86_64-only (Cranelift PLT gate); on aarch64 these run interpreted.
    if cfg!(target_arch = "x86_64") { assert!(ex.contains("IndexedStoreMap2") || ex.contains("VectorBinaryMap"), "expected store-map or vectorized JIT, got: {ex}"); }

    let d = scalar(eval_last(
        r#"
x <- as.numeric(1:50); w <- as.numeric(50:1)
f <- function(x, w){ y <- numeric(length(x)); for(i in 1:length(x)) y[i] <- x[i]*w[i] + 1; y }
max(abs(f(x, w) - (x*w + 1)))
"#,
    ));
    assert!(d < 1e-9, "two-input store diff = {d}");
}

#[test]
fn multi_statement_store_matches() {
    let d = scalar(eval_last(
        r#"
x <- as.numeric(1:30)
f <- function(x){ y <- numeric(length(x)); for(i in 1:length(x)) { t <- x[i]; y[i] <- t*t + 1 }; y }
max(abs(f(x) - (x*x + 1)))
"#,
    ));
    assert!(d < 1e-9, "multi-statement store diff = {d}");
}

#[test]
fn ifelse_valued_store_matches() {
    // y[i] <- if(x[i] > 0) x[i] else -x[i]  == abs(x)
    let d = scalar(eval_last(
        r#"
x <- c(-2, 3, -4, 5, -6)
f <- function(x){ y <- numeric(length(x)); for(i in 1:length(x)) y[i] <- if(x[i] > 0) x[i] else 0 - x[i]; y }
max(abs(f(x) - abs(x)))
"#,
    ));
    assert!(d < 1e-9, "if/else store diff = {d}");
}

#[test]
fn store_map_propagates_na() {
    // NA in either input → NA in the corresponding output element.
    let n_na = scalar(eval_last(
        r#"
f <- function(x, w){ y <- numeric(length(x)); for(i in 1:length(x)) y[i] <- x[i] + w[i]; y }
sum(is.na(f(c(1, NA, 3, 4), c(10, 20, NA, 40))))
"#,
    ));
    assert!((n_na - 2.0).abs() < 1e-9, "expected 2 NA, got {n_na}");
}
