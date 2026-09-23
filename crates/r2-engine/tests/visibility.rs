//! R's visibility rule, as `Engine::visible` after each top-level statement
//! — the flag both consoles read to decide what auto-prints. The whole-
//! output comparison with R is tests/differential/output/visibility.R.

use r2_engine::Engine;
use r2_parser::Parser;

/// `(statement, visible after it)` pairs, run in order in one engine.
fn check(cases: &[(&str, bool)]) {
    let mut e = Engine::new();
    for (src, want) in cases {
        for ex in Parser::parse(src).expect("parse ok") {
            e.eval(&ex).unwrap_or_else(|err| panic!("{src}: {}", err.msg));
        }
        assert_eq!(e.visible, *want, "after `{src}`");
    }
}

#[test]
fn values_are_visible_and_assignments_are_not() {
    check(&[
        ("x <- 5", false), ("x", true), ("(x <- 6)", true), ("x = 7", false),
        ("1 + 2", true), ("c(1, 2)", true), ("function(a) a", true),
    ]);
}

#[test]
fn loops_and_a_false_if_without_else_are_invisible() {
    check(&[
        ("for (i in 1:3) i", false), ("w <- 0; while (w < 2) w <- w + 1", false),
        ("if (FALSE) 1", false), ("if (TRUE) 2", true), ("if (FALSE) 1 else 3", true),
    ]);
}

#[test]
fn invisible_and_print_are_invisible_and_calls_pass_it_on() {
    check(&[
        ("invisible(1)", false), ("print(1)", false), ("1 + invisible(2)", true),
        // a user function returns the visibility of its last expression
        ("f <- function(v) print(v)", false), ("f(1)", false),
        ("g <- function() invisible(7)", false), ("g()", false),
        ("h <- function() { y <- 3 }", false), ("h()", false),
        ("k <- function() 42", false), ("k()", true),
        ("p <- function(v) return(invisible(v))", false), ("p(1)", false),
        ("m <- function() { print('a'); 10 }", false), ("m()", true),
    ]);
}

#[test]
fn builtins_that_run_user_code_pass_its_visibility_through() {
    check(&[
        ("tryCatch(invisible(1), error = function(e) 0)", false),
        ("tryCatch(8, error = function(e) 0)", true),
        ("switch('a', a = invisible(1), b = 2)", false),
        ("switch('b', a = 1, b = 2)", true),
        ("local({ z <- 1 })", false), ("local({ 14 })", true),
    ]);
}
