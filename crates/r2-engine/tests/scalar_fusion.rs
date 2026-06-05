//! Regression tests for Phase-1 vector⊗scalar chain fusion. A fused
//! chain (e.g. `v*2+1`) must produce exactly the same result as the
//! un-fused two-step computation, and must preserve NA.

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

#[test]
fn fused_chain_matches_unfused() {
    // v*2+1 (fused, length >= 64) == (v*2) then +1 (two single-op passes).
    let script = r#"
v <- as.numeric(1:100)
fused  <- v*2+1+3
t <- v*2; t <- t+1; manual <- t+3
max(abs(fused - manual))
"#;
    match eval_last(script) {
        RVal::Numeric(d, _) => {
            let diff = d.iter().next().and_then(|x| *x).expect("scalar");
            assert!(diff < 1e-12, "fused vs unfused diff = {}", diff);
        }
        other => panic!("expected numeric, got {:?}", other),
    }
}

#[test]
fn fusion_preserves_na() {
    // A NA anywhere makes the column non-dense → fusion must bail to the
    // normal path so NA still propagates.
    let script = r#"
v <- c(as.numeric(1:99), NA)
r <- v*2+1
r[100]
"#;
    match eval_last(script) {
        RVal::Numeric(d, _) => {
            assert!(d.iter().next().map(|x| x.is_none()).unwrap_or(false),
                "element 100 should be NA");
        }
        other => panic!("expected numeric NA, got {:?}", other),
    }
}

#[test]
fn fusion_small_vector_correct() {
    // Below the length threshold → normal path, still correct.
    let script = "(c(1,2,3,4) * 2 + 1)";
    match eval_last(script) {
        RVal::Numeric(d, _) => {
            let got: Vec<Option<f64>> = d.iter().copied().collect();
            assert_eq!(got, vec![Some(3.0), Some(5.0), Some(7.0), Some(9.0)]);
        }
        other => panic!("expected numeric, got {:?}", other),
    }
}
