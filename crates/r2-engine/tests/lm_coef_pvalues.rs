//! Regression test: `lm` coefficient p-values use the t-distribution
//! (Pr(>|t|)), not the normal approximation.
//!
//! At small residual df the difference is large. For this fixture
//! (n=10, p=2 → df=8) the x-coefficient has t≈3.545; R reports
//! Pr(>|t|)=0.00755. The old `2*(1-Φ(|t|))` normal approximation gave
//! ≈3.9e-4 — about 19× too small. This pins the t-distribution path
//! (which now routes through the exact Lentz incomplete beta).

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
fn lm_coefficient_pvalue_uses_t_distribution() {
    let script = r#"
df <- data.frame(
  val = c(5.1,4.9,4.7,4.6,5.0,6.2,5.9,6.1,6.3,5.8),
  x   = c(1,2,3,4,5,6,7,8,9,10)
)
lm(val ~ x, data = df)$p.values[2]
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let p = v.iter().next().and_then(|x| *x).expect("scalar p-value");
            // R: 0.007552 (t-distribution, df=8).
            assert!((p - 0.007552).abs() < 5e-4, "x coef Pr(>|t|): got {}", p);
            // Guard against a regression to the normal approximation
            // (~3.9e-4 here): the t-distribution value is ~19× larger.
            assert!(p > 1e-3, "p={} looks like the normal approximation, not t", p);
        }
        other => panic!("expected numeric scalar, got {:?}", other),
    }
}
