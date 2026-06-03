//! Integration tests for formula-based NSE dispatch through the
//! engine's centralized formula preprocessor (r2-engine/src/lib.rs,
//! the `matches!(fname, "lm" | ... | "aggregate")` block).
//!
//! Two things are pinned here:
//!
//!  1. `aggregate(value ~ group, data = df, FUN = ...)` — the formula
//!     form must resolve `value`/`group` against the data frame and
//!     feed aggregate's existing (x, by =, FUN =) core. Before the
//!     Phase-1 fix this failed with "object 'value' not found" because
//!     `aggregate` was missing from the formula-function list.
//!
//!  2. t.test formula↔vector equivalence — `t.test(y ~ g, data = df)`
//!     must produce the *same* statistics as the two-vector call on the
//!     split groups. This guarantees the formula path is purely an
//!     input adapter and never alters the math.

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
fn aggregate_formula_resolves_and_computes_group_means() {
    // grp "a" has mean 4.86, grp "b" has mean 6.06 (first-seen order).
    let script = r#"
df <- data.frame(
  val = c(5.1, 4.9, 4.7, 4.6, 5.0, 6.2, 5.9, 6.1, 6.3, 5.8),
  grp = c("a","a","a","a","a","b","b","b","b","b")
)
result <- aggregate(val ~ grp, data = df, FUN = mean)
result$val
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got.len(), 2, "two groups expected");
            assert!((got[0].unwrap() - 4.86).abs() < 1e-9, "group a mean: {:?}", got[0]);
            assert!((got[1].unwrap() - 6.06).abs() < 1e-9, "group b mean: {:?}", got[1]);
        }
        other => panic!("expected $val numeric, got {:?}", other),
    }
}

#[test]
fn aggregate_formula_positional_fun_sum() {
    let script = r#"
df <- data.frame(
  val = c(1, 2, 3, 4, 5, 6),
  grp = c("a","a","a","b","b","b")
)
result <- aggregate(val ~ grp, data = df, sum)
result$val
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got[0], Some(6.0), "sum of a (1+2+3)");
            assert_eq!(got[1], Some(15.0), "sum of b (4+5+6)");
        }
        other => panic!("expected $val numeric, got {:?}", other),
    }
}

#[test]
fn aggregate_formula_multi_group() {
    // y1 ~ g1 + g2 — composite grouping, sorted by (g1, g2).
    // combos: (a,x)=rows{1,5}->mean(1,5)=3; (a,y)=rows{2,6}->4;
    //         (b,x)=rows{3,7}->5; (b,y)=rows{4,8}->6.
    let script = r#"
df <- data.frame(
  y1 = c(1,2,3,4,5,6,7,8),
  g1 = c("a","a","b","b","a","a","b","b"),
  g2 = c("x","y","x","y","x","y","x","y")
)
result <- aggregate(y1 ~ g1 + g2, data = df, FUN = mean)
result$y1
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got, vec![Some(3.0), Some(4.0), Some(5.0), Some(6.0)]);
        }
        other => panic!("expected $y1 numeric, got {:?}", other),
    }
}

#[test]
fn aggregate_formula_multi_response_cbind() {
    // cbind(y1, y2) ~ g1 — two response columns, summed per group.
    // g1=a: y1=1+2+5+6=14, y2=10+20+50+60=140
    // g1=b: y1=3+4+7+8=22, y2=30+40+70+80=220
    let script = r#"
df <- data.frame(
  y1 = c(1,2,3,4,5,6,7,8),
  y2 = c(10,20,30,40,50,60,70,80),
  g1 = c("a","a","b","b","a","a","b","b")
)
result <- aggregate(cbind(y1, y2) ~ g1, data = df, FUN = sum)
c(result$y1[1], result$y2[1], result$y1[2], result$y2[2])
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let got: Vec<Option<f64>> = v.iter().copied().collect();
            assert_eq!(got, vec![Some(14.0), Some(140.0), Some(22.0), Some(220.0)]);
        }
        other => panic!("expected numeric vector, got {:?}", other),
    }
}

#[test]
fn ttest_formula_matches_vector_form() {
    // The formula path is only an input adapter — its p-value must be
    // identical to the two-vector call on the same split groups.
    let script = r#"
df <- data.frame(
  val = c(5.1, 4.9, 4.7, 4.6, 5.0, 6.2, 5.9, 6.1, 6.3, 5.8),
  grp = c("a","a","a","a","a","b","b","b","b","b")
)
x <- c(5.1, 4.9, 4.7, 4.6, 5.0)
y <- c(6.2, 5.9, 6.1, 6.3, 5.8)
abs(t.test(val ~ grp, data = df)$p.value - t.test(x, y)$p.value)
"#;
    match eval_last(script) {
        RVal::Numeric(v, _) => {
            let d = v.iter().next().and_then(|x| *x).expect("scalar");
            assert!(d < 1e-12, "formula vs vector p-value differ by {}", d);
        }
        other => panic!("expected numeric scalar, got {:?}", other),
    }
}
