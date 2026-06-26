//! JIT error type and result alias.

use r2_ir::{BlockId, VReg};

// ── Errors ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum JitError {
    Unsupported(String),
    CraneliftError(String),
    UndefinedVReg(VReg),
    UndefinedBlock(BlockId),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            JitError::Unsupported(s) => write!(f, "JIT: unsupported in current phase: {}", s),
            JitError::CraneliftError(s) => write!(f, "JIT: Cranelift error: {}", s),
            JitError::UndefinedVReg(v) => write!(f, "JIT: undefined VReg {}", v),
            JitError::UndefinedBlock(b) => write!(f, "JIT: undefined block {}", b),
        }
    }
}

impl std::error::Error for JitError {}

pub type JitResult<T> = Result<T, JitError>;
