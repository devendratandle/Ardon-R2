//! R2 Type Inferencer — Phase A of the new architecture.
//!
//! Pure annotation pass over the AST. Walks an `Expr` tree and produces a
//! columnar `IrType` (element type + shape) for every sub-expression.
//!
//! Per docs/ARCHITECTURE.md §5 Phase A:
//!   - Pure annotation pass over AST
//!   - Output: every AST node tagged with shape + element type (or Unknown)
//!   - Single file, ~300 LoC
//!   - Engine ignores it for now — runs and validates only
//!
//! Locked design (§4):
//!   - Values are columnar: scalar = vector of length 1
//!   - Shape carries known length where statically derivable, else `None`
//!   - Element type defaults to `Unknown`; the inferencer never panics on
//!     missing information, it just widens to `Unknown`.
//!
//! This module deliberately does not mutate `Expr`. The IR builder (Phase B)
//! will consume this inferencer's results to construct typed IR nodes.

use crate::*;
use std::collections::HashMap;

// ── Type lattice ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrElem {
    Real,    // f64
    Int,     // i32
    Bool,    // logical
    Char,    // string
    Any,     // heterogeneous (List, etc.)
    Unknown, // not yet inferred / failed inference
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrShape {
    Scalar,                                   // length 1, dimensionless
    Vec(Option<usize>),                       // 1-D, length known when Some
    Mat(Option<usize>, Option<usize>),        // 2-D, (nrow, ncol)
    DataFrame,                                // heterogeneous columns
    Function,                                 // closure / builtin
    Null,                                     // R's NULL
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IrType {
    pub elem: IrElem,
    pub shape: IrShape,
}

impl IrType {
    pub const fn unknown() -> Self { IrType { elem: IrElem::Unknown, shape: IrShape::Unknown } }
    pub const fn null() -> Self { IrType { elem: IrElem::Any, shape: IrShape::Null } }
    pub const fn scalar(e: IrElem) -> Self { IrType { elem: e, shape: IrShape::Scalar } }
    pub const fn vector(e: IrElem, n: Option<usize>) -> Self { IrType { elem: e, shape: IrShape::Vec(n) } }
    pub const fn matrix(e: IrElem, r: Option<usize>, c: Option<usize>) -> Self { IrType { elem: e, shape: IrShape::Mat(r, c) } }
    pub fn is_unknown(&self) -> bool { self.elem == IrElem::Unknown || self.shape == IrShape::Unknown }
}

// ── Type-promotion rules (R's standard widening) ─────────────────────

pub fn promote_elem(a: IrElem, b: IrElem) -> IrElem {
    use IrElem::*;
    match (a, b) {
        (Unknown, _) | (_, Unknown) => Unknown,
        (Real, _) | (_, Real) => Real,
        (Int, _) | (_, Int) => Int,
        (Bool, Bool) => Bool,
        (Char, Char) => Char,
        (Any, _) | (_, Any) => Any,
        _ => Unknown,
    }
}

pub fn promote_shape(a: &IrShape, b: &IrShape) -> IrShape {
    use IrShape::*;
    match (a, b) {
        (Unknown, _) | (_, Unknown) => Unknown,
        (Null, x) | (x, Null) => x.clone(),
        (Scalar, x) | (x, Scalar) => x.clone(),
        (Vec(Some(n1)), Vec(Some(n2))) if n1 == n2 => Vec(Some(*n1)),
        (Vec(_), Vec(_)) => Vec(None),
        (Mat(r1, c1), Mat(r2, c2)) => Mat(
            match (r1, r2) { (Some(a), Some(b)) if a == b => Some(*a), _ => None },
            match (c1, c2) { (Some(a), Some(b)) if a == b => Some(*a), _ => None },
        ),
        (Mat(r, c), Vec(_)) | (Vec(_), Mat(r, c)) => Mat(*r, *c),
        _ => Unknown,
    }
}

// ── Inference context ────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct TypeCtx {
    pub bindings: HashMap<Arc<str>, IrType>,
}

impl TypeCtx {
    pub fn new() -> Self { Self::default() }
    pub fn lookup(&self, name: &str) -> IrType {
        self.bindings.get(name).cloned().unwrap_or_else(IrType::unknown)
    }
    pub fn bind(&mut self, name: Arc<str>, t: IrType) { self.bindings.insert(name, t); }
}

// ── Builtin signatures (small, hand-written, expanded later) ─────────

/// Public alias for the IR builder; the private `builtin_return_type`
/// is kept stable inside this module for clarity.
pub fn builtin_return_type_pub(name: &str, arg_types: &[IrType]) -> IrType {
    builtin_return_type(name, arg_types)
}

fn builtin_return_type(name: &str, arg_types: &[IrType]) -> IrType {
    match name {
        // Reductions: vector → scalar real
        "sum" | "mean" | "median" | "sd" | "var" | "min" | "max" | "prod" | "length"
            => IrType::scalar(IrElem::Real),
        // Predicates: scalar bool
        "any" | "all" | "is.numeric" | "is.character" | "is.logical" | "is.null" | "is.na"
            => IrType::scalar(IrElem::Bool),
        // Constructors with known shape
        "c" => {
            // c() promotes element types and concatenates lengths when known.
            let mut elem = IrElem::Bool; // identity for promotion
            let mut total: Option<usize> = Some(0);
            for t in arg_types {
                elem = promote_elem(elem, t.elem);
                total = match (total, &t.shape) {
                    (Some(n), IrShape::Scalar) => Some(n + 1),
                    (Some(n), IrShape::Vec(Some(k))) => Some(n + k),
                    _ => None,
                };
            }
            IrType::vector(elem, total)
        }
        "rep" => IrType::vector(arg_types.first().map(|t| t.elem).unwrap_or(IrElem::Unknown), None),
        "seq" | "seq_len" | ":" => IrType::vector(IrElem::Int, None),
        // Matrix-producing
        "matrix" | "cbind" | "rbind" | "t" => IrType::matrix(IrElem::Real, None, None),
        "%*%" => IrType::matrix(IrElem::Real, None, None),
        "solve" | "chol" | "qr" => IrType::matrix(IrElem::Real, None, None),
        "diag" => IrType::matrix(IrElem::Real, None, None),
        // Element-wise math (preserves shape of first arg)
        "sqrt" | "exp" | "log" | "log2" | "log10" | "abs" | "sin" | "cos" | "tan"
            => arg_types.first().cloned().unwrap_or_else(IrType::unknown),
        // Frame ops
        "data.frame" | "read.csv" => IrType { elem: IrElem::Any, shape: IrShape::DataFrame },
        "nrow" | "ncol" => IrType::scalar(IrElem::Int),
        // Side-effect / void
        "print" | "cat" | "plot" | "hist" | "boxplot" | "barplot" | "save" | "message"
            => IrType::null(),
        _ => IrType::unknown(),
    }
}

// ── Main inference entry point ───────────────────────────────────────

pub fn infer_expr(e: &Expr, ctx: &mut TypeCtx) -> IrType {
    match e {
        Expr::NumLit(_)  => IrType::scalar(IrElem::Real),
        Expr::IntLit(_)  => IrType::scalar(IrElem::Int),
        Expr::BoolLit(_) => IrType::scalar(IrElem::Bool),
        Expr::StrLit(_)  => IrType::scalar(IrElem::Char),
        Expr::FStringLit(_) => IrType::scalar(IrElem::Char),
        Expr::NaLit      => IrType::scalar(IrElem::Unknown),
        Expr::NullLit    => IrType::null(),

        Expr::Symbol(s) => ctx.lookup(s),

        Expr::Unary { expr, .. } => infer_expr(expr, ctx),

        Expr::Binary { op, lhs, rhs } => {
            let lt = infer_expr(lhs, ctx);
            let rt = infer_expr(rhs, ctx);
            use BinOp::*;
            match op {
                Eq | Ne | Lt | Gt | Le | Ge | And | Or | AndShort | OrShort => {
                    IrType { elem: IrElem::Bool, shape: promote_shape(&lt.shape, &rt.shape) }
                }
                MatMul => IrType::matrix(IrElem::Real, None, None),
                Tilde => IrType { elem: IrElem::Any, shape: IrShape::Unknown }, // formula
                _ => IrType { elem: promote_elem(lt.elem, rt.elem), shape: promote_shape(&lt.shape, &rt.shape) },
            }
        }

        Expr::Assign { target, value } => {
            let vt = infer_expr(value, ctx);
            if let Expr::Symbol(name) = target.as_ref() {
                ctx.bind(name.clone(), vt.clone());
            }
            vt
        }

        Expr::Call { func, args } => {
            let arg_types: Vec<IrType> = args.iter().map(|a| infer_expr(&a.value, ctx)).collect();
            match func.as_ref() {
                Expr::Symbol(name) => builtin_return_type(name, &arg_types),
                _ => IrType::unknown(),
            }
        }

        Expr::Index { object, .. } | Expr::DblIndex { object, .. } => {
            // Indexing widens shape: known length is lost, element type preserved.
            let ot = infer_expr(object, ctx);
            IrType { elem: ot.elem, shape: IrShape::Vec(None) }
        }

        Expr::Dollar { object, .. } => {
            let ot = infer_expr(object, ctx);
            // df$col — element type unknown without column info, shape is a vector.
            match ot.shape { IrShape::DataFrame => IrType::vector(IrElem::Unknown, None), _ => IrType::unknown() }
        }

        Expr::Block(stmts) => stmts.last().map(|s| infer_expr(s, ctx)).unwrap_or_else(IrType::null),

        Expr::If { then, else_, .. } => {
            let tt = infer_expr(then, ctx);
            match else_ { Some(e) => { let et = infer_expr(e, ctx); IrType { elem: promote_elem(tt.elem, et.elem), shape: promote_shape(&tt.shape, &et.shape) } } None => tt }
        }

        Expr::For { body, .. } | Expr::While { body, .. } | Expr::Repeat { body } => { infer_expr(body, ctx); IrType::null() }

        Expr::FuncDef { .. } | Expr::Lambda { .. } => IrType { elem: IrElem::Any, shape: IrShape::Function },

        Expr::Return(v) => infer_expr(v, ctx),

        Expr::Pipe { lhs, rhs } => {
            let _ = infer_expr(lhs, ctx);
            infer_expr(rhs, ctx)
        }

        Expr::Match { arms, .. } => {
            arms.first().map(|a| infer_expr(&a.body, ctx)).unwrap_or_else(IrType::unknown)
        }

        Expr::TryCatch { body, .. } => infer_expr(body, ctx),

        // Things we don't model yet
        Expr::Namespace { .. } | Expr::TypeDef { .. } | Expr::MethodDef(_)
        | Expr::Break | Expr::Next | Expr::Dots => IrType::null(),
    }
}

// ── Convenience: infer a sequence (top-level program) ────────────────

/// Infer types over a flat sequence of top-level expressions, updating
/// the context as assignments are seen. Returns the type of the last
/// expression (matches R's REPL value semantics).
pub fn infer_program(prog: &[Expr], ctx: &mut TypeCtx) -> IrType {
    let mut last = IrType::null();
    for e in prog { last = infer_expr(e, ctx); }
    last
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t_lit(e: Expr) -> IrType { let mut c = TypeCtx::new(); infer_expr(&e, &mut c) }

    #[test]
    fn literals() {
        assert_eq!(t_lit(Expr::NumLit(1.5)),  IrType::scalar(IrElem::Real));
        assert_eq!(t_lit(Expr::IntLit(3)),    IrType::scalar(IrElem::Int));
        assert_eq!(t_lit(Expr::BoolLit(true)),IrType::scalar(IrElem::Bool));
        assert_eq!(t_lit(Expr::StrLit("x".into())), IrType::scalar(IrElem::Char));
        assert_eq!(t_lit(Expr::NullLit),      IrType::null());
    }

    #[test]
    fn arithmetic_promotes_element() {
        let e = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(Expr::NumLit(1.0)),
            rhs: Box::new(Expr::IntLit(2)),
        };
        let t = t_lit(e);
        assert_eq!(t.elem, IrElem::Real);
        assert_eq!(t.shape, IrShape::Scalar);
    }

    #[test]
    fn comparison_returns_bool() {
        let e = Expr::Binary {
            op: BinOp::Lt,
            lhs: Box::new(Expr::NumLit(1.0)),
            rhs: Box::new(Expr::NumLit(2.0)),
        };
        assert_eq!(t_lit(e).elem, IrElem::Bool);
    }

    #[test]
    fn assignment_updates_context() {
        let mut ctx = TypeCtx::new();
        let prog = vec![
            Expr::Assign {
                target: Box::new(Expr::Symbol(Arc::from("x"))),
                value:  Box::new(Expr::NumLit(3.14)),
            },
            Expr::Symbol(Arc::from("x")),
        ];
        let t = infer_program(&prog, &mut ctx);
        assert_eq!(t, IrType::scalar(IrElem::Real));
    }

    #[test]
    fn c_concatenates_lengths() {
        let e = Expr::Call {
            func: Box::new(Expr::Symbol(Arc::from("c"))),
            args: vec![
                CallArg { name: None, value: Expr::NumLit(1.0) },
                CallArg { name: None, value: Expr::NumLit(2.0) },
                CallArg { name: None, value: Expr::NumLit(3.0) },
            ],
        };
        let t = t_lit(e);
        assert_eq!(t, IrType::vector(IrElem::Real, Some(3)));
    }

    #[test]
    fn unknown_symbol_is_unknown() {
        let e = Expr::Symbol(Arc::from("nowhere"));
        assert!(t_lit(e).is_unknown());
    }

    #[test]
    fn matrix_multiply_returns_matrix() {
        let e = Expr::Binary {
            op: BinOp::MatMul,
            lhs: Box::new(Expr::Symbol(Arc::from("a"))),
            rhs: Box::new(Expr::Symbol(Arc::from("b"))),
        };
        let t = t_lit(e);
        assert!(matches!(t.shape, IrShape::Mat(_, _)));
        assert_eq!(t.elem, IrElem::Real);
    }

    #[test]
    fn block_returns_last_value_type() {
        let e = Expr::Block(vec![
            Expr::NumLit(1.0),
            Expr::StrLit("hello".into()),
        ]);
        assert_eq!(t_lit(e), IrType::scalar(IrElem::Char));
    }

    #[test]
    fn lambda_has_function_shape() {
        let e = Expr::Lambda { params: vec![], body: Box::new(Expr::NumLit(1.0)) };
        assert_eq!(t_lit(e).shape, IrShape::Function);
    }
}
