//! Phase J.4 (matrix/vector lowering, step 1) — vector-RETURNING kernels with
//! embedded reductions. `x - mean(x)`, standardise, normalise compile to a
//! reduction pass (fused waves) + a SIMD map pass writing the output vector.
//! The centring primitive `d` written once is both a standalone vector kernel
//! and inlined into the scalar var/cov/cor formulas.

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
fn centering_primitive_jits_to_vector_map() {
    let ex = string(eval_last(r#"explain(function(v) v - mean(v))"#));
    assert!(ex.contains("JIT-compiled") && ex.contains("VectorMap"), "expected VectorMap, got: {ex}");
}

#[test]
fn centering_matches_interpreter_even_and_odd() {
    for n in [64usize, 65, 100, 101, 1000, 1001] {
        let d = scalar(eval_last(&format!(
            "set.seed({n}); x <- rnorm({n})\n\
             d <- function(v) v - mean(v)\n\
             max(abs(d(x) - (x - mean(x))))"
        )));
        assert!(d < 1e-12, "n={n} centering err {d}");
    }
}

#[test]
fn zscore_standardise_matches() {
    // (x-mean)/sd has mean ~0 and unit sample sd.
    let s = scalar(eval_last(
        r#"
set.seed(3); x <- rnorm(500)
z <- function(x) (x - mean(x)) / sqrt(sum((x-mean(x))^2)/(length(x)-1))
r <- z(x)
sqrt(sum(r*r)/(length(r)-1))
"#,
    ));
    assert!((s - 1.0).abs() < 1e-9, "z-score sd = {s}");
}

#[test]
fn two_input_vector_output_matches() {
    let d = scalar(eval_last(
        r#"
set.seed(4); x <- rnorm(300); y <- rnorm(300)
cen2 <- function(x, y) (x - mean(x)) * (y - mean(y))
max(abs(cen2(x, y) - (x - mean(x)) * (y - mean(y))))
"#,
    ));
    assert!(d < 1e-12, "two-input vector-output err {d}");
}
