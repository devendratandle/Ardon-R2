//! `for (i in a:b)` iterates a counter, never a materialised range — and
//! so it can be interrupted.
//!
//! `x <- 0; for (i in 1:1e9) x <- x + i` used to build `1:1e9` (4 GB)
//! and then box every element into a `Vec<RVal>` before the first
//! iteration ran; the GUI froze and Esc did nothing, because the
//! interrupt flag is polled per expression and no expression boundary
//! is reached while a billion elements are being allocated. Now the
//! loop reaches its first iteration at once, and the flag stops it.

use r2_engine::Engine;
use r2_parser::Parser;
use r2_types::{ErrKind, RVal};

fn eval_last(script: &str) -> Result<RVal, r2_types::R2Err> {
    let mut e = Engine::new();
    let mut last = RVal::Null;
    for ex in Parser::parse(script).expect("parse ok") {
        last = e.eval(&ex)?;
    }
    Ok(last)
}

fn scalar(v: &RVal) -> f64 {
    match v {
        RVal::Numeric(d, _) => d.iter().next().and_then(|x| *x).expect("scalar"),
        RVal::Integer(d, _) => d.iter().next().and_then(|x| *x).expect("scalar") as f64,
        other => panic!("expected a number, got {:?}", other),
    }
}

#[test]
fn counted_ranges_keep_their_semantics() {
    assert_eq!(scalar(&eval_last("s <- 0; for (i in 1:10) s <- s + i; s").unwrap()), 55.0);
    assert_eq!(scalar(&eval_last("s <- 0; for (i in 5:1) s <- s * 10 + i; s").unwrap()), 54321.0);
    assert_eq!(scalar(&eval_last("n <- 0; for (i in 3:3) n <- n + 1; n").unwrap()), 1.0);
    assert_eq!(scalar(&eval_last(
        "k <- 0; for (i in 1:100) { if (i %% 2 == 0) next; if (i > 9) break; k <- k + i }; k").unwrap()), 25.0);
    // the loop variable is an integer, as in R
    assert!(matches!(eval_last("for (i in 1:3) NULL; i").unwrap(), RVal::Integer(..)));
    assert_eq!(scalar(&eval_last("for (i in 1:3) NULL; i").unwrap()), 3.0);
}

#[test]
fn a_billion_iterations_start_at_once_and_stop_on_interrupt() {
    r2_types::clear_interrupt();
    let t = std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_millis(300));
        r2_types::request_interrupt();
    });
    let started = std::time::Instant::now();
    let res = eval_last("x <- 0; for (i in 1:1e9) x <- x + i");
    let took = started.elapsed();
    t.join().unwrap();
    r2_types::clear_interrupt();
    match res {
        Err(e) if matches!(e.kind, ErrKind::Interrupt) => {}
        other => panic!("expected an interrupt, got {:?}", other.map(|_| ()).map_err(|e| e.msg)),
    }
    // Materialising the range took minutes (and tens of GB); the loop must
    // now be running, and interruptible, within the interrupt's own delay.
    assert!(took.as_secs() < 5, "took {:?} to honour the interrupt", took);
}
