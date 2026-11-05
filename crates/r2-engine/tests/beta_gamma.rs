//! The beta and gamma builtins' arguments: what R requires and refuses. Their
//! values are compared with R itself in tests/differential/cases/beta_gamma.R.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> Vec<f64> {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("{script}: {}", err.msg));
    }
    last.as_reals().unwrap().iter().map(|x| x.unwrap_or(f64::NAN)).collect()
}

fn err(script: &str) -> String {
    let mut e = Engine::new();
    for ex in Parser::parse(script).expect("parse ok") {
        if let Err(x) = e.eval(&ex) { return x.msg; }
    }
    panic!("expected an error from {script}");
}

#[test]
fn rate_and_scale_name_the_same_thing() {
    let by_rate = run("pgamma(3, 2, rate = 4)");
    assert_eq!(by_rate, run("pgamma(3, 2, 4)"), "rate is the third position");
    assert_eq!(by_rate, run("pgamma(3, 2, scale = 0.25)"));
    assert_eq!(by_rate, run("pgamma(3, shape = 2, rate = 4)"));
    assert_eq!(by_rate, run("pgamma(3, 2, rate = 4, scale = 0.25)"), "both may be given when they agree");
    assert!(err("pgamma(3, 2, rate = 4, scale = 2)").contains("specify 'rate' or 'scale' but not both"));
    assert!(err("rgamma(3, 2, rate = 4, scale = 2)").contains("specify 'rate' or 'scale' but not both"));
}

#[test]
fn a_missing_shape_is_an_error_not_a_default() {
    assert!(err("pgamma(3)").contains("\"shape\""));
    assert!(err("qgamma(0.5)").contains("\"shape\""));
    assert!(err("rgamma(3)").contains("\"shape\""));
    assert!(err("pbeta(0.5)").contains("\"shape1\""));
    assert!(err("pbeta(0.5, 2)").contains("\"shape2\""));
    assert!(err("dbeta(0.5, 2)").contains("\"shape2\""));
    assert!(err("rbeta(3, 2)").contains("\"shape2\""));
}

#[test]
fn a_non_central_beta_says_it_is_not_implemented() {
    assert!(err("pbeta(0.5, 2, 3, ncp = 1)").contains("not implemented"));
    assert_eq!(run("pbeta(0.5, 2, 3, ncp = 0)"), run("pbeta(0.5, 2, 3)"));
}

#[test]
fn flags_work_by_name_and_invalid_input_is_nan() {
    let lower = run("pbeta(0.3, 2, 3)")[0];
    assert_eq!(run("pbeta(0.3, 2, 3, log.p = TRUE)")[0], lower.ln());
    assert!((run("pbeta(0.3, 2, 3, lower.tail = FALSE)")[0] - (1.0 - lower)).abs() < 1e-15);
    for bad in ["pbeta(0.5, -1, 2)", "dbeta(0.5, 2, -1)", "qbeta(2, 2, 3)", "pgamma(1, -1)", "dgamma(1, 2, scale = -1)", "qgamma(-1, 2)"] {
        assert!(run(bad)[0].is_nan(), "{bad}");
    }
}

#[test]
fn the_generators_take_r_arguments() {
    assert_eq!(run("rgamma(7, 2)").len(), 7);
    assert_eq!(run("rgamma(n = 3, shape = 2, scale = 2)").len(), 3);
    assert_eq!(run("rbeta(5, 2, 3)").len(), 5);
    assert!(run("rbeta(100, 0.01, 0.01)").iter().all(|x| (0.0..=1.0).contains(x)), "small shapes must not be NaN");
    assert_eq!(run("rgamma(3, 0)"), vec![0.0; 3], "shape 0 is a point mass at 0");
    let m = run("set.seed(1); x <- rgamma(20000, 3, scale = 2); mean(x)")[0];
    assert!((m - 6.0).abs() < 0.2, "mean {m}");
    let a = run("set.seed(9); rgamma(4, 2)");
    assert_eq!(a, run("set.seed(9); rgamma(4, 2)"), "the seed decides the draws");
}
