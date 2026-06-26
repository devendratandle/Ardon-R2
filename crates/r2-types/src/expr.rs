//! `Expr` AST nodes (parser output) + call/arg/operator enums.

use std::sync::Arc;
use crate::*;

// ═══════════════════════════════════════════════════════════════════════
// Expr — AST nodes (what the parser produces, separate from RVal)
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    NumLit(f64),
    IntLit(i32),
    StrLit(String),
    BoolLit(bool),
    FStringLit(Vec<FStringPart>),
    NaLit,
    NullLit,

    // Identifiers
    Symbol(Arc<str>),

    // Operations
    Unary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },

    // Assignment. `superassign` = true for `<<-` (assign in an enclosing /
    // global scope instead of the current one).
    Assign { target: Box<Expr>, value: Box<Expr>, superassign: bool },

    // Function call
    Call { func: Box<Expr>, args: Vec<CallArg> },

    // Indexing
    Index { object: Box<Expr>, indices: Vec<Option<Expr>> },
    DblIndex { object: Box<Expr>, index: Box<Expr> },
    Dollar { object: Box<Expr>, field: Arc<str> },

    // Namespace
    Namespace { pkg: Arc<str>, name: Arc<str> },

    // Pipe
    Pipe { lhs: Box<Expr>, rhs: Box<Expr> },

    // Control flow
    If { cond: Box<Expr>, then: Box<Expr>, else_: Option<Box<Expr>> },
    For { var: Arc<str>, iter: Box<Expr>, body: Box<Expr> },
    While { cond: Box<Expr>, body: Box<Expr> },
    Repeat { body: Box<Expr> },
    Match { expr: Box<Expr>, arms: Vec<MatchArm> },
    Block(Vec<Expr>),

    // Functions
    FuncDef { params: Vec<Param>, body: Box<Expr> },
    Lambda { params: Vec<Param>, body: Box<Expr> },
    Return(Box<Expr>),

    // R2 type system
    TypeDef { name: Arc<str>, fields: Vec<FieldDef>, parent: Option<Arc<str>> },
    MethodDef(Method),

    // Try-catch
    TryCatch { body: Box<Expr>, var: Arc<str>, catch: Box<Expr> },

    // Control
    Break,
    Next,
    Dots,
}

#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<Arc<str>>,
    pub value: Expr,
}

/// Evaluated argument — used at runtime after expressions are evaluated to values
#[derive(Debug, Clone)]
pub struct EvalArg {
    pub name: Option<Arc<str>>,
    pub value: RVal,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub patterns: Vec<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum FStringPart {
    Literal(String),
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp { Neg, Pos, Not }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Pow, Mod, IntDiv,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or, AndShort, OrShort,
    Colon, Tilde, MatMul,
}

// ── Error mode ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorMode {
    Strict,
    Lenient,
}

// ── Helpers ──────────────────────────────────────────────────────────

