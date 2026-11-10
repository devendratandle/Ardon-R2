//! Names through `[`, and classes on lists and vectors: what R's comparison in
//! tests/differential/output/names_and_classes.R cannot say, because it is about
//! R2's own types or about what R2 refuses.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn eval(script: &str) -> RVal {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("{script}: {}", err.msg));
    }
    last
}

fn num(script: &str) -> Vec<f64> {
    eval(script).as_reals().unwrap().iter().map(|x| x.unwrap_or(f64::NAN)).collect()
}

fn text(script: &str) -> String { eval(script).to_string() }

fn err(script: &str) -> String {
    let mut e = Engine::new();
    for ex in Parser::parse(script).expect("parse ok") {
        if let Err(x) = e.eval(&ex) { return x.msg; }
    }
    panic!("expected an error from {script}");
}

const POINT: &str = "type Point { x: numeric, y: numeric }\np <- Point(x = 1, y = 2)\n";

#[test]
fn a_declared_type_is_not_a_list_but_a_result_is() {
    assert_eq!(text(&format!("{POINT}is.list(p)")), "[1] FALSE");
    assert_eq!(num(&format!("{POINT}length(p)")), vec![1.0]);
    assert!(matches!(eval(&format!("{POINT}names(p)")), RVal::Null));
    let t = "t <- t.test(c(1, 2, 3, 4))\n";
    assert_eq!(text(&format!("{t}is.list(t)")), "[1] TRUE");
    assert!(num(&format!("{t}length(t)"))[0] > 1.0);
    assert!(text(&format!("{t}names(t)")).contains("statistic"));
    assert!(!text(&format!("{t}names(t)")).contains(".report"), "what R2 keeps for itself stays out");
}

#[test]
fn a_result_can_be_unclassed_into_its_fields() {
    let q = "q <- unclass(qr(matrix(c(2, 1, 1, 3), 2)))\n";
    assert_eq!(text(&format!("{q}class(q)")), "[1] \"list\"");
    assert_eq!(text(&format!("{q}names(q)")), "[1] \"qr\"    \"rank\"  \"qraux\" \"pivot\"");
    assert_eq!(num(&format!("{q}q$rank")), vec![2.0]);
}

#[test]
fn class_set_says_what_it_cannot_do() {
    assert!(err("m <- matrix(1:4, 2); class(m) <- \"foo\"").contains("not implemented"));
    assert!(err("x <- list(1); class(x) <- \"data.frame\"").contains("not implemented"));
    assert!(err("x <- 1; attr(x, 2) <- 3").contains("'name' must be non-null character string"));
    assert!(err("row(1:3)").contains("a matrix-like object is required as argument to 'row'"));
    assert!(err("col(1:3)").contains("a matrix-like object is required as argument to 'col'"));
}

#[test]
fn the_class_vector_stays_whole_through_attr_and_attributes() {
    let k = "k <- list(a = 1)\nclass(k) <- c(\"foo\", \"bar\")\n";
    assert_eq!(text(&format!("{k}attr(k, \"class\")")), "[1] \"foo\" \"bar\"");
    assert!(!text(&format!("{k}attributes(k)")).contains(".class"), "the storage does not show");
    assert_eq!(text(&format!("{k}attr(k, \"class\") <- NULL\nclass(k)")), "[1] \"list\"");
}

#[test]
fn indexing_a_vector_without_names_is_unchanged() {
    assert_eq!(num("c(1, 2, 3)[2]"), vec![2.0]);
    assert_eq!(num("c(1, 2, 3)[c(TRUE, FALSE, TRUE)]"), vec![1.0, 3.0]);
    assert_eq!(num("c(1, 2, 3)[-1]"), vec![2.0, 3.0]);
    assert!(matches!(eval("c(1, 2, 3)[2]"), RVal::Numeric(_, at) if at.names.is_none()));
}

#[test]
fn a_name_that_appears_twice_gives_its_first_element() {
    assert_eq!(num("x <- c(a = 1, a = 2, b = 3); x[\"a\"]"), vec![1.0]);
}

#[test]
fn time_classes_survive_the_subscript() {
    assert_eq!(text("d <- as.Date(c(\"2024-03-15\", \"2024-03-16\")); class(d[2])"), "[1] \"Date\"");
    assert_eq!(text("v <- c(1, 2); class(v) <- \"myclass\"; class(v[1])"), "[1] \"numeric\"");
}
