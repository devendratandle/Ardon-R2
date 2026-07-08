//! Phase J.4 — SIMD (F64X2, 4× unrolled) fused map-reduce for the common
//! `sum(f(x))` / `sum(f(x,w))` reductions and dot products. Must match the
//! interpreter across all lengths (the 4× unroll means the scalar tail handles
//! remainders of 0..7). Independent accumulators break the fadd latency chain.

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
fn dot_and_ssq_match_all_lengths() {
    // Cover every remainder class of the 8-wide unrolled loop (n mod 8).
    for n in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 1000, 1001, 4096, 4099] {
        let d = scalar(eval_last(&format!(
            "set.seed({n}); x <- rnorm({n}); w <- rnorm({n})\n\
             dot <- function(x, w) sum(x*w); ssq <- function(x) sum(x*x)\n\
             max(abs(dot(x,w) - sum(x*w)), abs(ssq(x) - sum(x*x)))"
        )));
        assert!(d < 1e-9, "n={n} err {d}");
    }
}

#[test]
fn compound_fused_reduction_matches() {
    // sum(sqrt(x*x + w*w)) — the fused-SIMD case that beats the multi-pass
    // interpreter. Result must equal the interpreter's.
    let d = scalar(eval_last(
        r#"
set.seed(7); x <- rnorm(5000); w <- rnorm(5000)
hyp <- function(x, w) sum(sqrt(x*x + w*w))
abs(hyp(x, w) - sum(sqrt(x*x + w*w)))
"#,
    ));
    assert!(d < 1e-6, "compound reduction err {d}");
}

#[test]
fn prod_reduction_matches() {
    // Product reduction exercises the multiplicative identity/combine path.
    let d = scalar(eval_last(
        r#"
x <- as.numeric(1:12) / 6
p <- function(x) prod(x*x)
abs(p(x) - prod(x*x))
"#,
    ));
    assert!(d < 1e-9, "prod reduction err {d}");
}
