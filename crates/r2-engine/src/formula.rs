//! Formula-walking helpers — pure functions over `r2_types::Expr`
//! that the engine's lm / glm / aov / lmer NSE preprocessor calls
//! before dispatching to `r2_stats::models`.
//!
//! Two domains live here:
//!
//! * **`Error(...)` splitting** for repeated-measures ANOVA. R's
//!   `aov(y ~ x + Error(subject))` syntax declares `Error(...)` as the
//!   random-effect stratum; we lift it out of the predictor expansion
//!   so the predictor side resolves cleanly and the stratum side
//!   passes to the stats backend.
//! * **Random-intercept splitting** for `lmer` formulas like
//!   `y ~ x + (1 | group)`. The parser produces `Or(NumLit(1), Symbol)`
//!   for `(1 | group)`; this module recognises the shape and lifts
//!   the grouping factor out.
//!
//! Plus `fmt_expr` — a small deparser that turns an `Expr` back into
//! a source-like string for `$call` fields on fitted-model results.

use r2_types::{BinOp, Expr, fmt_num};

// ─── Error(...) term splitter ───────────────────────────────────────

pub(crate) fn is_error_call(func: &Expr) -> bool {
    matches!(func, Expr::Symbol(s) if s.as_ref() == "Error")
}

/// Unwrap `subject/treatment` style nested Error specifications to
/// the outermost grouping factor (the wholeplot stratum). For Phase
/// R.S.1 this is the only stratum we use — the full split-plot
/// decomposition across multiple within-subject factors lands in
/// v0.2.1. Accepting the syntax with one-way semantics matches R's
/// behaviour in the common case where each subject sees every
/// treatment level.
pub(crate) fn unwrap_nested_error(stratum: Expr) -> Expr {
    match stratum {
        Expr::Binary { op: BinOp::Div, lhs, .. } => unwrap_nested_error(*lhs),
        other => other,
    }
}

pub(crate) fn split_error_term(rhs: &Expr) -> (Expr, Option<Expr>) {
    match rhs {
        // Degenerate: the entire RHS is Error(...). Fixed part
        // becomes NullLit. Nested `Error(subject/treatment)`
        // collapses to `Error(subject)` for the one-way RM case.
        Expr::Call { func, args } if is_error_call(func) => {
            let raw = args
                .first()
                .map(|a| a.value.clone())
                .unwrap_or(Expr::NullLit);
            let stratum = unwrap_nested_error(raw);
            (Expr::NullLit, Some(stratum))
        }
        // Compound: walk both sides of `+` recursively, combine.
        Expr::Binary { op: BinOp::Add, lhs, rhs: r } => {
            let (l_fixed, l_err) = split_error_term(lhs);
            let (r_fixed, r_err) = split_error_term(r);
            let l_null = matches!(l_fixed, Expr::NullLit);
            let r_null = matches!(r_fixed, Expr::NullLit);
            let combined = match (l_null, r_null) {
                (true, true) => Expr::NullLit,
                (true, false) => r_fixed,
                (false, true) => l_fixed,
                (false, false) => Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(l_fixed),
                    rhs: Box::new(r_fixed),
                },
            };
            (combined, l_err.or(r_err))
        }
        // Leaf or non-Add expression: no Error possible here.
        other => (other.clone(), None),
    }
}

// ─── Random-intercept splitter for `(1 | group)` ────────────────────

/// Returns true if `expr` matches the `(1|group)` random-intercept
/// shape: `Binary { Or, lhs = NumLit(1.0), rhs = <grouping> }`.
pub(crate) fn is_random_intercept(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Binary { op: BinOp::Or, lhs, .. }
            if matches!(lhs.as_ref(), Expr::NumLit(n) if (n - 1.0).abs() < 1e-12)
    )
}

/// Extract the grouping-factor expression from a `(1|group)`
/// random-effect spec. Returns `None` if `expr` is not a
/// random-intercept shape.
pub(crate) fn random_intercept_grouping(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::Binary { op: BinOp::Or, rhs, .. } => Some((**rhs).clone()),
        _ => None,
    }
}

/// Split a formula RHS into `(fixed_part, random_intercept_groupings)`.
/// Each entry in the returned Vec is the grouping-factor expression
/// of one random-intercept term. The fixed part is the RHS with all
/// `(1|g)` subexpressions removed.
pub(crate) fn split_random_effects(rhs: &Expr) -> (Expr, Vec<Expr>) {
    match rhs {
        // Direct: the entire RHS is a single `(1|group)` term.
        expr if is_random_intercept(expr) => {
            let group = random_intercept_grouping(expr).unwrap_or(Expr::NullLit);
            (Expr::NullLit, vec![group])
        }
        Expr::Binary { op: BinOp::Add, lhs, rhs: r } => {
            let (l_fixed, mut l_re) = split_random_effects(lhs);
            let (r_fixed, mut r_re) = split_random_effects(r);
            l_re.append(&mut r_re);
            let l_null = matches!(l_fixed, Expr::NullLit);
            let r_null = matches!(r_fixed, Expr::NullLit);
            let combined = match (l_null, r_null) {
                (true, true) => Expr::NullLit,
                (true, false) => r_fixed,
                (false, true) => l_fixed,
                (false, false) => Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(l_fixed),
                    rhs: Box::new(r_fixed),
                },
            };
            (combined, l_re)
        }
        // Leaf or non-Add expression: not a random-effect spec.
        other => (other.clone(), Vec::new()),
    }
}

// ─── Expr → source-like string ──────────────────────────────────────

/// Stringify a parser `Expr` back to source-like text. Used by the
/// lm / glm / aov NSE preprocessor to capture the original call shape
/// as a `$call` field on the fitted-model TypeInstance — so
/// `summary(fit)` can print `Call: lm(formula = y ~ x, data = df)`
/// instead of the placeholder `Call: lm(formula)`. Covers symbols,
/// numeric literals, binary/unary operators, function calls, and
/// indexing — the subset needed for typical model formulas.
pub(crate) fn fmt_expr(e: &Expr) -> String {
    match e {
        Expr::Symbol(s) => s.to_string(),
        Expr::NumLit(n) => fmt_num(*n),
        Expr::IntLit(n) => format!("{}", n),
        Expr::StrLit(s) => format!("\"{}\"", s),
        Expr::BoolLit(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        Expr::NaLit => "NA".into(),
        Expr::NullLit => "NULL".into(),
        Expr::Binary { op, lhs, rhs } => {
            let opstr = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
                BinOp::Pow => "^", BinOp::Mod => "%%", BinOp::IntDiv => "%/%",
                BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::Lt => "<", BinOp::Gt => ">",
                BinOp::Le => "<=", BinOp::Ge => ">=",
                // Lexer convention: `&` → And, `&&` → AndShort.
                BinOp::And => "&", BinOp::Or => "|",
                BinOp::AndShort => "&&", BinOp::OrShort => "||",
                BinOp::Tilde => "~", BinOp::MatMul => "%*%",
                BinOp::Colon => ":",
            };
            format!("{} {} {}", fmt_expr(lhs), opstr, fmt_expr(rhs))
        }
        Expr::Call { func, args } => {
            let fname = fmt_expr(func);
            let parts: Vec<String> = args.iter().map(|a| match &a.name {
                Some(n) => format!("{} = {}", n, fmt_expr(&a.value)),
                None => fmt_expr(&a.value),
            }).collect();
            format!("{}({})", fname, parts.join(", "))
        }
        Expr::Dollar { object, field } => format!("{}${}", fmt_expr(object), field),
        Expr::Index { object, indices } => {
            let parts: Vec<String> = indices.iter().map(|i| match i {
                Some(e) => fmt_expr(e),
                None => String::new(),
            }).collect();
            format!("{}[{}]", fmt_expr(object), parts.join(", "))
        }
        _ => "<expr>".into(),
    }
}
