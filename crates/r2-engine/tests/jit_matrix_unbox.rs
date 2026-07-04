//! Phase J.3 — JIT matrix unboxing. A closure that JIT-compiles to a vector
//! kernel (map / reduce / binary) must accept a matrix argument by treating its
//! column-major buffer as the dense f64 vector, producing results identical to
//! the interpreter and preserving `dim` on element-wise maps.

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
fn jit_matrix_reduction_matches_interp() {
    // Vector1ToScalar over a matrix: sum(m*m) reduces the whole buffer.
    let v = eval_last(
        r#"
m <- matrix(as.numeric(1:6), nrow = 2)
ss <- function(x) sum(x * x)
ss(m) - sum(m * m)
"#,
    );
    assert!(scalar(v).abs() < 1e-9);
}

#[test]
fn jit_matrix_binary_reduction() {
    // Vector2ToScalar: Frobenius inner product sum(A*B) over two same-shape mats.
    let v = eval_last(
        r#"
m <- matrix(as.numeric(1:6), nrow = 2)
fip <- function(a, b) sum(a * b)
fip(m, m)
"#,
    );
    assert!((scalar(v) - 91.0).abs() < 1e-9); // 1+4+9+16+25+36
}

#[test]
fn jit_matrix_map_preserves_dim() {
    // VectorMap over a matrix returns a matrix with dim preserved.
    let v = eval_last(
        r#"
m <- matrix(as.numeric(1:6), nrow = 2)
sq <- function(x) x * x
r <- sq(m)
c(is.matrix(r), nrow(r), ncol(r), r[2, 3])
"#,
    );
    match v {
        RVal::Numeric(d, _) => {
            let g = |i: usize| d.iter().nth(i).and_then(|x| *x).unwrap();
            assert_eq!(g(0), 1.0); // is.matrix -> TRUE
            assert_eq!(g(1), 2.0); // nrow
            assert_eq!(g(2), 3.0); // ncol
            assert_eq!(g(3), 36.0); // (2,3) = 6^2
        }
        other => panic!("expected numeric vector, got {:?}", other),
    }
}

#[test]
fn jit_matrix_binary_map_preserves_dim() {
    // VectorBinaryMap over two matrices returns a matrix.
    let v = eval_last(
        r#"
m <- matrix(as.numeric(1:6), nrow = 2)
ad <- function(a, b) a + b
r <- ad(m, m)
c(is.matrix(r), r[1, 2])
"#,
    );
    match v {
        RVal::Numeric(d, _) => {
            let g = |i: usize| d.iter().nth(i).and_then(|x| *x).unwrap();
            assert_eq!(g(0), 1.0); // is.matrix
            assert_eq!(g(1), 6.0); // (1,2) = 3+3
        }
        other => panic!("expected numeric vector, got {:?}", other),
    }
}
