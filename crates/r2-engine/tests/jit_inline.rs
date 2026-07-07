//! Phase J.4 (brick 1) — user-function call inlining. A numeric function
//! composed of pure JIT-lowerable helpers must compile as one native unit
//! (instead of bailing on the user-function `Call`), with results identical to
//! the interpreter. Recursion must fall back safely, not hang.

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
fn composed_helpers_jit_and_match() {
    let ex = string(eval_last(r#"sq <- function(x) x*x; explain(function(a, b) sq(a) + sq(b))"#));
    assert!(ex.contains("JIT-compiled"), "expected JIT, got: {ex}");
    // (a+1)^2 over a vector, arg expression duplicated by inlining — must match.
    let d = scalar(eval_last(
        r#"
sq <- function(x) x*x
f <- function(a) sq(a + 1)
v <- as.numeric(1:50)
max(abs(f(v) - (v + 1)^2))
"#,
    ));
    assert!(d < 1e-9, "inlined arg-dup diff = {d}");
}

#[test]
fn nested_helpers_and_reduction() {
    // helper calling a helper, then used inside sum().
    let d = scalar(eval_last(
        r#"
sq <- function(x) x*x
cube <- function(x) x*sq(x)
ss <- function(v) sum(cube(v))
v <- as.numeric(1:10)
ss(v) - sum(v^3)
"#,
    ));
    assert!(d.abs() < 1e-6, "nested-helper reduction diff = {d}");
}

#[test]
fn captured_scalar_in_helper() {
    // The inlined helper references a captured free var, baked after inlining.
    let v = scalar(eval_last(
        r#"
k <- 10
scal <- function(x) x*k
g <- function(a) scal(a) + 1
g(3)
"#,
    ));
    assert!((v - 31.0).abs() < 1e-9, "captured-scalar inline = {v}");
}

#[test]
fn recursion_falls_back_and_is_correct() {
    // A self-recursive body is pure-inlinable, so the inliner must terminate
    // (depth-bounded) and leave a residual call that falls back to the
    // interpreter — never hang. Shallow depth keeps within the small test-thread
    // stack (the interpreter's per-call recursion is deep; the CLI's 8 MB main
    // stack handles far deeper — this only checks compile-termination + result).
    let v = scalar(eval_last(
        r#"
fact <- function(n) if (n <= 1) 1 else n * fact(n - 1)
fact(4)
"#,
    ));
    assert!((v - 24.0).abs() < 1e-9, "fact(4) = {v}");
}
