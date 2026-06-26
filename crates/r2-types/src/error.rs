//! Error types (`R2Err`/`ErrKind`) and the global interrupt flag.

use crate::RVal;

#[derive(Debug)]
pub struct R2Err { pub msg: String, pub kind: ErrKind }

#[derive(Debug)]
pub enum ErrKind {
    Runtime,
    Type,
    Index,
    CtrlBreak,
    CtrlNext,
    CtrlReturn(Box<RVal>),
    /// Phase R.M.2 — user interrupt (Ctrl+C). The engine raises this when
    /// it observes the global INTERRUPT flag set by the SIGINT handler;
    /// the REPL catches it, prints a brief notice, and returns to the
    /// prompt instead of exiting. Non-interactive (script) mode treats
    /// it as a normal error and exits with a non-zero status.
    Interrupt,
}

impl PartialEq for ErrKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ErrKind::Runtime, ErrKind::Runtime) => true,
            (ErrKind::Type, ErrKind::Type) => true,
            (ErrKind::Index, ErrKind::Index) => true,
            (ErrKind::CtrlBreak, ErrKind::CtrlBreak) => true,
            (ErrKind::CtrlNext, ErrKind::CtrlNext) => true,
            (ErrKind::CtrlReturn(_), ErrKind::CtrlReturn(_)) => true,
            (ErrKind::Interrupt, ErrKind::Interrupt) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for R2Err {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error: {}", self.msg)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Phase R.M.2 — global interrupt flag.
//
// Set by the SIGINT (Ctrl+C) handler installed in the REPL binary. Read
// by the engine's evaluation loop at safe interruption points (top of
// each expression, top of each loop iteration, top of each function
// call). When set, the engine returns Err(R2Err { kind: Interrupt, .. }),
// which unwinds the call stack cleanly. The REPL catches it, prints a
// notice, clears the flag, and returns to the prompt — the process
// stays alive. Script mode does not catch it and exits non-zero.
// ─────────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicBool, Ordering};

pub static INTERRUPT: AtomicBool = AtomicBool::new(false);

/// Set by the SIGINT handler. The engine polls `is_interrupted()` at
/// safe points and raises `ErrKind::Interrupt` when it observes `true`.
#[inline]
pub fn request_interrupt() {
    INTERRUPT.store(true, Ordering::Relaxed);
}

/// Check and consume the interrupt flag in one step. Returns `true` if
/// the flag was set (and clears it).
#[inline]
pub fn take_interrupt() -> bool {
    INTERRUPT.swap(false, Ordering::Relaxed)
}

/// Non-consuming check. Used by the engine eval loop to decide whether
/// to raise `ErrKind::Interrupt` without clearing the flag (the REPL
/// driver clears it after catching).
#[inline]
pub fn is_interrupted() -> bool {
    INTERRUPT.load(Ordering::Relaxed)
}

/// Clear the flag without consuming it. Called by the REPL after it has
/// finished handling an Interrupt error, before returning to the prompt.
#[inline]
pub fn clear_interrupt() {
    INTERRUPT.store(false, Ordering::Relaxed);
}
