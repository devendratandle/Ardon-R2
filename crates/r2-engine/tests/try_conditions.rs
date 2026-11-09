//! R's `try()` beside R2's `try { } catch` syntax, and the condition object.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::RVal;

fn run(script: &str) -> RVal {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex).unwrap_or_else(|err| panic!("eval error: {}", err.msg));
    }
    last
}

fn text(v: &RVal) -> Vec<String> {
    match v {
        RVal::Character(c, _) => c.iter().map(|x| x.as_ref().map(|s| s.to_string()).unwrap_or_default()).collect(),
        RVal::Logical(l, _) => l.iter().map(|x| format!("{:?}", x)).collect(),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn try_returns_a_try_error_with_its_condition() {
    let v = run("r <- try(stop(\"boom\"), silent = TRUE)\nc(class(r), conditionMessage(attr(r, \"condition\")))\n");
    assert_eq!(text(&v), ["try-error", "boom"]);
}

#[test]
fn try_passes_a_value_through_and_prints_the_error_unless_silent() {
    assert!(matches!(run("try(2 + 3)\n"), RVal::Numeric(..)));
    // not silent: the message goes to stderr, and the value is the same
    let v = run("r <- try(stop(\"loud\"))\nc(class(r), r)\n");
    assert_eq!(text(&v), ["try-error", "Error : loud\n"]);
}

#[test]
fn the_syntax_form_still_works() {
    let v = run("x <- try { stop(\"s\") } catch (e) { \"caught\" }\nx\n");
    assert_eq!(text(&v), ["caught"]);
}

#[test]
fn a_handler_gets_a_condition() {
    let v = run("e <- tryCatch(stop(\"bad\"), error = function(e) e)\nc(conditionMessage(e), e$message)\n");
    assert_eq!(text(&v), ["bad", "bad"]);
    assert!(matches!(run("conditionCall(tryCatch(stop(\"x\"), error = function(e) e))\n"), RVal::Null));
}
