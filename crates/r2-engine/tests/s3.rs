//! S3 dispatch: what the comparison with R in
//! tests/differential/output/s3_dispatch.R cannot show, the errors R2 gives
//! and that dispatch costs nothing until a method exists.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn eval(script: &str) -> Result<RVal, String> {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).map_err(|x| x.msg)?;
    }
    Ok(last)
}

#[test]
fn use_method_and_next_method_outside_a_dispatch_say_so() {
    assert!(eval("UseMethod(\"f\")").unwrap_err().contains("UseMethod called from outside a function"));
    assert!(eval("f <- function() NextMethod(); f()").unwrap_err().contains("NextMethod called from outside a method dispatch"));
    assert!(eval("g <- function(x) UseMethod(1); g(1)").unwrap_err().contains("'generic' argument must be a character string"));
}

#[test]
fn a_generic_with_a_builtin_of_its_name_falls_back_to_the_builtin() {
    let v = eval("length2 <- function(x) UseMethod(\"length\"); length2(c(1, 2, 3))").unwrap();
    assert_eq!(v.as_reals().unwrap(), vec![Some(3.0)]);
}

#[test]
fn the_generic_body_after_use_method_never_runs() {
    let v = eval("f <- function(x) { UseMethod(\"f\"); 99 }; f.default <- function(x) 1; f(0)").unwrap();
    assert_eq!(v.as_reals().unwrap(), vec![Some(1.0)]);
}

#[test]
fn a_name_with_a_dot_is_not_a_method_for_another_class() {
    // print.data is not a print method for class "frame"
    let v = eval("print.data <- function(x) 0; x <- structure(list(a = 1), class = \"frame\"); length(x)").unwrap();
    assert_eq!(v.as_reals().unwrap(), vec![Some(1.0)]);
}

#[test]
fn operators_on_unclassed_values_are_untouched_by_methods() {
    let v = eval("\"+.money\" <- function(e1, e2) 0; 1 + 2").unwrap();
    assert_eq!(v.as_reals().unwrap(), vec![Some(3.0)]);
}
