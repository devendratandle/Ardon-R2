//! Rewrites that bring a closure body into the canonical forms the kernel
//! code generators consume: indexed accumulation loops as vector
//! reductions (J.5), integer powers expanded, reductions hoisted to scalar
//! locals with common subexpressions shared, vector-valued locals fused
//! away, and `%*%` products hoisted (J.4).

use crate::*;

/// Canonical string key for an expression (the reduction-kernel node subset).
/// Used for common-subexpression elimination: identical reduction sub-trees
/// (e.g. `mean(x)`, `x-mean(x)` occurring many times) map to the same key and
/// are computed once. Written verbatim, so `a+b` and `b+a` are distinct — fine,
/// because the reuse philosophy writes each primitive (`d(x)`) identically.
pub(crate) fn expr_key(e: &r2_types::Expr) -> String {
    use r2_types::Expr::*;
    match e {
        NumLit(x) => format!("#{}", x),
        IntLit(x) => format!("i{}", x),
        BoolLit(b) => format!("b{}", b),
        Symbol(s) => format!("${}", s),
        Unary { op, expr } => format!("u{:?}({})", op, expr_key(expr)),
        Binary { op, lhs, rhs } => format!("({}{:?}{})", expr_key(lhs), op, expr_key(rhs)),
        Call { func, args } => format!("{}[{}]", expr_key(func),
            args.iter().map(|a| expr_key(&a.value)).collect::<Vec<_>>().join(",")),
        If { cond, then, else_ } => format!("if({},{},{})", expr_key(cond), expr_key(then),
            else_.as_ref().map(|x| expr_key(x)).unwrap_or_else(|| "_".into())),
        other => format!("?{:p}", other),
    }
}

/// Phase J.4 brick 3 — hoist every reduction sub-expression (`sum`/`prod`/
/// `mean`/`length`) to a fresh scalar local, so what remains inside each fused
/// loop is a pure element expression over vector params + (now hoisted) scalar
/// locals. `sum((x-mean(x))^2)` → `__hr0 <- mean(x); __hr1 <- sum((x-__hr0)^2); __hr1`.
/// Nested reductions are hoisted innermost-first.
///
/// Phase J.4 brick 4 — **CSE**: a reduction whose (fully-hoisted) form was
/// already emitted reuses that local instead of recomputing. So `mean(x)`
/// appearing in variance, sd, covariance and correlation is computed *once* —
/// the "compute the shared primitive `d(x)`/`mean(x)` once, reuse everywhere"
/// design made real at the machine level.
/// Phase J.5 — loop-to-vector normalization ("same math, one kernel").
/// Rewrites elementwise accumulation loops into the vector reductions the
/// existing kernels already compile:
///
///   for (i in 1:length(x)) s <- s + EXPR(x[i], w[i], scalars, mean(x)…)
///     ⇒  s <- s + sum(EXPR(x, w, …))
///
/// Guards (all must hold, else the loop is left untouched):
///  - iterator is `1:length(v)`, or `1:n` where `n` was bound to
///    `length(v)` earlier in the same block;
///  - the loop body is exactly one statement `s <- s + EXPR` (Add only —
///    order-independent, so vectorising cannot change results beyond the
///    FP reassociation already inherent to the SIMD waves);
///  - the index var appears ONLY as a whole-symbol subscript `vec[i]`;
///  - the accumulator `s` is not read inside EXPR (no recurrence).
/// Same-length requirements between subscripted vectors are enforced at
/// call time by the kernel handles (mismatch → interpreter fallback).
pub(crate) fn vectorize_indexed_loops(e: &r2_types::Expr) -> r2_types::Expr {
    use r2_types::Expr::*;
    fn is_one(e: &r2_types::Expr) -> bool {
        matches!(e, NumLit(n) if *n == 1.0) || matches!(e, IntLit(1))
    }
    fn length_of(e: &r2_types::Expr) -> Option<std::sync::Arc<str>> {
        if let Call { func, args } = e {
            if matches!(func.as_ref(), Symbol(s) if s.as_ref() == "length") && args.len() == 1 {
                if let Symbol(v) = &args[0].value { return Some(v.clone()); }
            }
        }
        None
    }
    // Replace `vec[i]` with `vec` (counting replacements); fail (None) if
    // `i` or the accumulator appears any other way.
    fn strip_index(e: &r2_types::Expr, i: &str, acc: &str, hits: &mut u32) -> Option<r2_types::Expr> {
        match e {
            Index { object, indices } => {
                if let (Symbol(_), [Some(Symbol(idx))]) = (object.as_ref(), indices.as_slice()) {
                    if idx.as_ref() == i { *hits += 1; return Some((**object).clone()); }
                }
                None // any other indexing shape: bail conservatively
            }
            Symbol(s) if s.as_ref() == i || s.as_ref() == acc => None,
            Binary { op, lhs, rhs } => Some(Binary { op: *op,
                lhs: Box::new(strip_index(lhs, i, acc, hits)?), rhs: Box::new(strip_index(rhs, i, acc, hits)?) }),
            Unary { op, expr } => Some(Unary { op: *op, expr: Box::new(strip_index(expr, i, acc, hits)?) }),
            Call { func, args } => {
                let mut na = Vec::with_capacity(args.len());
                for a in args { na.push(r2_types::CallArg { name: a.name.clone(), value: strip_index(&a.value, i, acc, hits)? }); }
                Some(Call { func: func.clone(), args: na })
            }
            If { cond, then, else_ } => Some(If {
                cond: Box::new(strip_index(cond, i, acc, hits)?),
                then: Box::new(strip_index(then, i, acc, hits)?),
                else_: match else_ { Some(x) => Some(Box::new(strip_index(x, i, acc, hits)?)), None => None } }),
            other => Some(other.clone()),
        }
    }
    fn rewrite_for(var: &std::sync::Arc<str>, iter: &r2_types::Expr, body: &r2_types::Expr,
                   len_aliases: &std::collections::HashMap<std::sync::Arc<str>, std::sync::Arc<str>>,
                   buf_aliases: &std::collections::HashMap<std::sync::Arc<str>, std::sync::Arc<str>>) -> Option<r2_types::Expr> {
        // The vector the iterator spans: `1:length(v)` or `1:n`, n ≡ length(v).
        let iter_vec: std::sync::Arc<str> = if let Binary { op: r2_types::BinOp::Colon, lhs, rhs } = iter {
            if !is_one(lhs) { return None; }
            match length_of(rhs) {
                Some(v) => v,
                None => match rhs.as_ref() {
                    Symbol(n) => len_aliases.get(n.as_ref())?.clone(),
                    _ => return None,
                },
            }
        } else { return None; };
        let stmt = match body { Block(v) if v.len() == 1 => &v[0], b @ Assign { .. } => b, _ => return None };
        if let Assign { target, value, superassign: false } = stmt {
            // Form 1 — accumulation: `s <- s + EXPR`  ⇒  `s <- s + sum(EXPR')`.
            if let (Symbol(s), Binary { op: r2_types::BinOp::Add, lhs, rhs }) = (target.as_ref(), value.as_ref()) {
                if matches!(lhs.as_ref(), Symbol(l) if l == s) {
                    let mut hits = 0u32;
                    let vecexpr = strip_index(rhs, var.as_ref(), s.as_ref(), &mut hits)?;
                    // The index must actually be used (else it's not a map).
                    if hits == 0 { return None; }
                    let sum = Call {
                        func: Box::new(Symbol(std::sync::Arc::from("sum"))),
                        args: vec![r2_types::CallArg { name: None, value: vecexpr }] };
                    return Some(Assign { target: target.clone(),
                        value: Box::new(Binary { op: r2_types::BinOp::Add, lhs: lhs.clone(), rhs: Box::new(sum) }),
                        superassign: false });
                }
            }
            // Form 2 — store map: `y[i] <- EXPR`  ⇒  `y <- EXPR'`, only when
            // `y` was allocated as `numeric(length(v))` over the SAME vector
            // the iterator spans (so every element is written exactly once
            // and whole-vector replacement is semantics-preserving).
            if let Index { object, indices } = target.as_ref() {
                if let (Symbol(y), [Some(Symbol(ix))]) = (object.as_ref(), indices.as_slice()) {
                    if ix.as_ref() == var.as_ref()
                        && buf_aliases.get(y.as_ref()).is_some_and(|v| v.as_ref() == iter_vec.as_ref()) {
                        let mut hits = 0u32;
                        let vecexpr = strip_index(value, var.as_ref(), y.as_ref(), &mut hits)?;
                        if hits == 0 { return None; }
                        return Some(Assign { target: Box::new(Symbol(y.clone())),
                            value: Box::new(vecexpr), superassign: false });
                    }
                }
            }
        }
        None
    }
    match e {
        Block(stmts) => {
            let mut aliases: std::collections::HashMap<std::sync::Arc<str>, std::sync::Arc<str>> = Default::default();
            let mut bufs: std::collections::HashMap<std::sync::Arc<str>, std::sync::Arc<str>> = Default::default();
            let out = stmts.iter().map(|st| {
                if let Assign { target, value, superassign: false } = st {
                    if let Symbol(n) = target.as_ref() {
                        if let Some(v) = length_of(value) { aliases.insert(n.clone(), v); }
                        // `y <- numeric(length(v))` (or numeric(n), n ≡ length(v))
                        if let Call { func, args } = value.as_ref() {
                            if matches!(func.as_ref(), Symbol(s) if s.as_ref() == "numeric") && args.len() == 1 {
                                let vs = match length_of(&args[0].value) {
                                    Some(v) => Some(v),
                                    None => match &args[0].value {
                                        Symbol(a) => aliases.get(a.as_ref()).cloned(),
                                        _ => None,
                                    },
                                };
                                if let Some(v) = vs { bufs.insert(n.clone(), v); }
                            }
                        }
                    }
                }
                if let For { var, iter, body } = st {
                    if let Some(r) = rewrite_for(var, iter, body, &aliases, &bufs) { return r; }
                }
                vectorize_indexed_loops(st)
            }).collect::<Vec<_>>();
            // A vectorized store-map fully overwrites its buffer, so the
            // `y <- numeric(length(v))` allocation is dead — drop it (the
            // kernels reject vector-valued alloc calls they can't lower).
            let rewritten: std::collections::HashSet<std::sync::Arc<str>> = out.iter().filter_map(|st| {
                if let Assign { target, value, superassign: false } = st {
                    if let Symbol(n) = target.as_ref() {
                        if bufs.contains_key(n.as_ref())
                            && !matches!(value.as_ref(), Call { func, .. } if matches!(func.as_ref(), Symbol(s) if s.as_ref() == "numeric")) {
                            return Some(n.clone());
                        }
                    }
                }
                None
            }).collect();
            let out = out.into_iter().filter(|st| {
                if let Assign { target, value, superassign: false } = st {
                    if let (Symbol(n), Call { func, .. }) = (target.as_ref(), value.as_ref()) {
                        if matches!(func.as_ref(), Symbol(s) if s.as_ref() == "numeric") && rewritten.contains(n.as_ref()) {
                            return false;
                        }
                    }
                }
                true
            }).collect();
            Block(out)
        }
        For { var, iter, body } => {
            if let Some(r) = rewrite_for(var, iter, body, &Default::default(), &Default::default()) { return r; }
            For { var: var.clone(), iter: iter.clone(), body: Box::new(vectorize_indexed_loops(body)) }
        }
        While { cond, body } => While { cond: cond.clone(), body: Box::new(vectorize_indexed_loops(body)) },
        other => other.clone(),
    }
}

pub(crate) fn hoist_reductions(e: &r2_types::Expr, stmts: &mut Vec<r2_types::Expr>, ctr: &mut u32, seen: &mut std::collections::HashMap<String, std::sync::Arc<str>>) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Call { func, args } => {
            let is_red = matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean" | "length"));
            let new_args: Vec<r2_types::CallArg> = args.iter()
                .map(|a| r2_types::CallArg { name: a.name.clone(), value: hoist_reductions(&a.value, stmts, ctr, seen) })
                .collect();
            let call = Call { func: func.clone(), args: new_args };
            if is_red {
                let key = expr_key(&call);
                if let Some(existing) = seen.get(&key) { return Symbol(existing.clone()); } // CSE hit
                let name: std::sync::Arc<str> = std::sync::Arc::from(format!(".__hr{}", *ctr));
                *ctr += 1;
                seen.insert(key, name.clone());
                stmts.push(Assign { target: Box::new(Symbol(name.clone())), value: Box::new(call), superassign: false });
                Symbol(name)
            } else {
                call
            }
        }
        Unary { op, expr } => Unary { op: *op, expr: Box::new(hoist_reductions(expr, stmts, ctr, seen)) },
        Binary { op, lhs, rhs } => Binary { op: *op,
            lhs: Box::new(hoist_reductions(lhs, stmts, ctr, seen)), rhs: Box::new(hoist_reductions(rhs, stmts, ctr, seen)) },
        If { cond, then, else_ } => If { cond: Box::new(hoist_reductions(cond, stmts, ctr, seen)),
            then: Box::new(hoist_reductions(then, stmts, ctr, seen)),
            else_: else_.as_ref().map(|x| Box::new(hoist_reductions(x, stmts, ctr, seen))) },
        other => other.clone(),
    }
}

/// Normalize a reduction-kernel body: fuse vector locals, then hoist all
/// reductions to scalar locals → a `Block` of `local <- <reduction|scalar>`
/// followed by a final scalar expression (the shape `compile_reduction_kernel`
/// consumes). Non-Block scalar bodies are handled too.
/// Rewrite small integer powers `b^k` (k = 2..=4) into repeated multiplication,
/// e.g. `(x-mean(x))^2` → `(x-mean(x))*(x-mean(x))`. Exact for real `b`, and —
/// unlike the `pow` extern call — SIMD-vectorisable, so variance/correlation
/// element expressions become F64X2-clean. Walks the whole tree.
pub(crate) fn expand_int_powers(e: &r2_types::Expr) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Binary { op: r2_types::BinOp::Pow, lhs, rhs } => {
            let base = expand_int_powers(lhs);
            let k = match rhs.as_ref() {
                NumLit(x) if *x >= 2.0 && *x <= 4.0 && x.fract() == 0.0 => Some(*x as u32),
                IntLit(x) if (2..=4).contains(x) => Some(*x as u32),
                _ => None,
            };
            match k {
                Some(k) => {
                    let mut acc = base.clone();
                    for _ in 1..k { acc = Binary { op: r2_types::BinOp::Mul, lhs: Box::new(acc), rhs: Box::new(base.clone()) }; }
                    acc
                }
                None => Binary { op: r2_types::BinOp::Pow, lhs: Box::new(base), rhs: Box::new(expand_int_powers(rhs)) },
            }
        }
        Binary { op, lhs, rhs } => Binary { op: *op, lhs: Box::new(expand_int_powers(lhs)), rhs: Box::new(expand_int_powers(rhs)) },
        Unary { op, expr } => Unary { op: *op, expr: Box::new(expand_int_powers(expr)) },
        Call { func, args } => Call { func: func.clone(),
            args: args.iter().map(|a| r2_types::CallArg { name: a.name.clone(), value: expand_int_powers(&a.value) }).collect() },
        If { cond, then, else_ } => If { cond: Box::new(expand_int_powers(cond)),
            then: Box::new(expand_int_powers(then)), else_: else_.as_ref().map(|x| Box::new(expand_int_powers(x))) },
        Assign { target, value, superassign } => Assign { target: target.clone(), value: Box::new(expand_int_powers(value)), superassign: *superassign },
        Block(s) => Block(s.iter().map(expand_int_powers).collect()),
        Return(v) => Return(Box::new(expand_int_powers(v))),
        Pipe { lhs, rhs } => Pipe { lhs: Box::new(expand_int_powers(lhs)), rhs: Box::new(expand_int_powers(rhs)) },
        other => other.clone(),
    }
}

pub(crate) fn normalize_reduction_kernel(body: &r2_types::Expr, vec_params: &[std::sync::Arc<str>]) -> (r2_types::Expr, Vec<std::sync::Arc<str>>) {
    use r2_types::Expr::*;
    let (inlined_raw, kept) = inline_vector_locals(body, vec_params);
    let inlined = expand_int_powers(&inlined_raw);
    let mut out: Vec<r2_types::Expr> = Vec::new();
    let mut ctr = 0u32;
    let mut seen: std::collections::HashMap<String, std::sync::Arc<str>> = std::collections::HashMap::new();
    let final_expr = match &inlined {
        Block(ss) => {
            let (last, init) = match ss.split_last() { Some(x) => x, None => return (inlined.clone(), kept) };
            for st in init {
                match st {
                    Assign { target, value, superassign } => {
                        let v = hoist_reductions(value, &mut out, &mut ctr, &mut seen);
                        out.push(Assign { target: target.clone(), value: Box::new(v), superassign: *superassign });
                    }
                    // J.4 iterative kernels — normalize a counted loop's body:
                    // hoist reductions PER STATEMENT with a fresh CSE scope
                    // (loop-carried scalars change between statements and
                    // iterations, so no cross-statement reuse), keeping the
                    // hoisted temps inside the loop body. The bound is loop-
                    // invariant → hoisted into the outer scope.
                    For { var, iter, body } => {
                        let iter_n = hoist_reductions(iter, &mut out, &mut ctr, &mut seen);
                        let body_stmts: Vec<r2_types::Expr> = match body.as_ref() {
                            Block(b) => b.clone(),
                            single => vec![single.clone()],
                        };
                        let mut new_body: Vec<r2_types::Expr> = Vec::new();
                        for bs in &body_stmts {
                            if let Assign { target, value, superassign } = bs {
                                let mut fresh: std::collections::HashMap<String, std::sync::Arc<str>> = std::collections::HashMap::new();
                                let v = hoist_reductions(value, &mut new_body, &mut ctr, &mut fresh);
                                new_body.push(Assign { target: target.clone(), value: Box::new(v), superassign: *superassign });
                            } else {
                                new_body.push(bs.clone()); // codegen will reject → fallback
                            }
                        }
                        out.push(For { var: var.clone(), iter: Box::new(iter_n), body: Box::new(Block(new_body)) });
                    }
                    // J.4 while-convergence loops: hoist body reductions per
                    // statement (fresh CSE scope); the condition stays intact —
                    // `emit_scalar` evaluates embedded reductions per iteration.
                    While { cond, body } => {
                        let body_stmts: Vec<r2_types::Expr> = match body.as_ref() {
                            Block(b) => b.clone(),
                            single => vec![single.clone()],
                        };
                        let mut new_body: Vec<r2_types::Expr> = Vec::new();
                        for bs in &body_stmts {
                            if let Assign { target, value, superassign } = bs {
                                let mut fresh: std::collections::HashMap<String, std::sync::Arc<str>> = std::collections::HashMap::new();
                                let v = hoist_reductions(value, &mut new_body, &mut ctr, &mut fresh);
                                new_body.push(Assign { target: target.clone(), value: Box::new(v), superassign: *superassign });
                            } else {
                                new_body.push(bs.clone());
                            }
                        }
                        out.push(While { cond: cond.clone(), body: Box::new(Block(new_body)) });
                    }
                    _ => out.push(st.clone()),
                }
            }
            hoist_reductions(last, &mut out, &mut ctr, &mut seen)
        }
        other => hoist_reductions(other, &mut out, &mut ctr, &mut seen),
    };
    out.push(final_expr);
    (Block(out), kept)
}

/// Does `e` reference a vector name in a *vector position* — i.e. bare, or under
/// element-wise ops, but NOT enclosed in a reduction (`sum`/`prod`/`mean`/
/// `length`, which collapse a vector to a scalar)? Determines whether a local's
/// rhs evaluates to a vector (fuse it) or a scalar (keep it).
pub(crate) fn refs_vector_bare(e: &r2_types::Expr, vec_names: &[std::sync::Arc<str>]) -> bool {
    use r2_types::Expr::*;
    match e {
        Symbol(s) => vec_names.iter().any(|v| v.as_ref() == s.as_ref()),
        Unary { expr, .. } => refs_vector_bare(expr, vec_names),
        Binary { lhs, rhs, .. } => refs_vector_bare(lhs, vec_names) || refs_vector_bare(rhs, vec_names),
        If { cond, then, else_ } => refs_vector_bare(cond, vec_names) || refs_vector_bare(then, vec_names)
            || else_.as_ref().map_or(false, |x| refs_vector_bare(x, vec_names)),
        Call { func, args } => {
            // A reduction collapses its argument to a scalar → not a vector position.
            if matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean" | "length")) {
                return false;
            }
            args.iter().any(|a| refs_vector_bare(&a.value, vec_names))
        }
        _ => false,
    }
}

/// Names assigned anywhere inside `For`/`While` bodies of `e` — such vector
/// locals are LOOP-CARRIED STATE and must stay as real (buffered) statements,
/// never fused away by substitution.
pub(crate) fn loop_assigned_names(e: &r2_types::Expr, out: &mut std::collections::HashSet<std::sync::Arc<str>>) {
    use r2_types::Expr::*;
    fn collect_assigns(e: &r2_types::Expr, out: &mut std::collections::HashSet<std::sync::Arc<str>>) {
        match e {
            Assign { target, value, .. } => {
                if let Symbol(n) = target.as_ref() { out.insert(n.clone()); }
                collect_assigns(value, out);
            }
            Block(s) => for x in s { collect_assigns(x, out); },
            If { cond, then, else_ } => { collect_assigns(cond, out); collect_assigns(then, out);
                if let Some(x) = else_ { collect_assigns(x, out); } }
            For { body, .. } | While { body, .. } => collect_assigns(body, out),
            _ => {}
        }
    }
    match e {
        For { body, .. } | While { body, .. } => collect_assigns(body, out),
        Block(s) => for x in s { loop_assigned_names(x, out); },
        Assign { value, .. } => loop_assigned_names(value, out),
        If { cond, then, else_ } => { loop_assigned_names(cond, out); loop_assigned_names(then, out);
            if let Some(x) = else_ { loop_assigned_names(x, out); } }
        _ => {}
    }
}

/// Phase J.4 brick 3 — inline vector-valued locals into a reduction-kernel body
/// by substitution, so composed formulas like `{ e <- pred-obs; sqrt(mean(e*e)) }`
/// or `{ d <- x-mean(x); sum(d*d) }` compile with the intermediate vector *fused*
/// away (no buffer allocated). A leading `local <- rhs` is a **vector** local iff
/// `rhs` has no top-level reduction and references a vector (param or earlier
/// vector-local); such statements are dropped and their definition substituted
/// into later statements. Scalar locals (those whose rhs reduces to a scalar) are
/// kept. Non-Block bodies pass through unchanged.
///
/// Returns the transformed body plus the vector locals that were KEPT as real
/// statements because a loop reassigns them (loop-carried vector state — the
/// codegen buffers those; everything else fuses by substitution as before).
pub(crate) fn inline_vector_locals(body: &r2_types::Expr, vec_params: &[std::sync::Arc<str>]) -> (r2_types::Expr, Vec<std::sync::Arc<str>>) {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s, _ => return (body.clone(), Vec::new()) };
    if stmts.len() < 2 { return (body.clone(), Vec::new()); }
    let (last, init) = stmts.split_last().unwrap();

    let mut in_loops: std::collections::HashSet<std::sync::Arc<str>> = std::collections::HashSet::new();
    loop_assigned_names(body, &mut in_loops);

    let mut vecdefs: std::collections::HashMap<std::sync::Arc<str>, r2_types::Expr> = std::collections::HashMap::new();
    let mut vec_names: Vec<std::sync::Arc<str>> = vec_params.to_vec();
    let mut kept: Vec<std::sync::Arc<str>> = Vec::new();
    let mut out: Vec<r2_types::Expr> = Vec::new();

    for st in init {
        if let Assign { target, value, superassign } = st {
            if let Symbol(nm) = target.as_ref() {
                // Inline already-known vector-locals into this rhs first.
                let rhs = substitute_symbols(value, &vecdefs);
                let kept_names: Vec<std::sync::Arc<str>> =
                    vec_names.iter().chain(kept.iter()).cloned().collect();
                if refs_vector_bare(&rhs, &kept_names) {
                    if in_loops.contains(nm) {
                        // Loop-carried vector: keep as a real statement.
                        if !kept.iter().any(|k| k.as_ref() == nm.as_ref()) { kept.push(nm.clone()); }
                        out.push(Assign { target: target.clone(), value: Box::new(rhs), superassign: *superassign });
                    } else {
                        // Pure vector temp: fuse away by substitution.
                        vecdefs.insert(nm.clone(), rhs);
                        vec_names.push(nm.clone());
                    }
                    continue;
                }
                // Scalar local: keep, with vector-locals substituted in.
                out.push(Assign { target: target.clone(), value: Box::new(rhs), superassign: *superassign });
                continue;
            }
        }
        out.push(st.clone());
    }
    out.push(substitute_symbols(last, &vecdefs));
    (Block(out), kept)
}

/// J.4 matrix state — hoist every `%*%` sub-expression to its own statement
/// (`.__mvN <- X %*% v`), and any non-symbol `%*%` right operand to an
/// element-wise temp, so the matrix kernel sees matvecs only as whole
/// statements with named operands. Mirrors `hoist_reductions`.
pub(crate) fn hoist_matmuls_expr(e: &r2_types::Expr, pre: &mut Vec<r2_types::Expr>, ctr: &mut u32) -> r2_types::Expr {
    use r2_types::Expr::*;
    match e {
        Binary { op: r2_types::BinOp::MatMul, lhs, rhs } => {
            // Named right operand, else hoist it first.
            let rname = match rhs.as_ref() {
                Symbol(_) => rhs.as_ref().clone(),
                other => {
                    let inner = hoist_matmuls_expr(other, pre, ctr);
                    let t: std::sync::Arc<str> = std::sync::Arc::from(format!(".__mvarg{}", *ctr));
                    *ctr += 1;
                    pre.push(Assign { target: Box::new(Symbol(t.clone())), value: Box::new(inner), superassign: false });
                    Symbol(t)
                }
            };
            let call = Binary { op: r2_types::BinOp::MatMul, lhs: lhs.clone(), rhs: Box::new(rname) };
            let t: std::sync::Arc<str> = std::sync::Arc::from(format!(".__mv{}", *ctr));
            *ctr += 1;
            pre.push(Assign { target: Box::new(Symbol(t.clone())), value: Box::new(call), superassign: false });
            Symbol(t)
        }
        Binary { op, lhs, rhs } => Binary { op: *op,
            lhs: Box::new(hoist_matmuls_expr(lhs, pre, ctr)), rhs: Box::new(hoist_matmuls_expr(rhs, pre, ctr)) },
        Unary { op, expr } => Unary { op: *op, expr: Box::new(hoist_matmuls_expr(expr, pre, ctr)) },
        If { cond, then, else_ } => If { cond: Box::new(hoist_matmuls_expr(cond, pre, ctr)),
            then: Box::new(hoist_matmuls_expr(then, pre, ctr)),
            else_: else_.as_ref().map(|x| Box::new(hoist_matmuls_expr(x, pre, ctr))) },
        Call { func, args } => Call { func: func.clone(),
            args: args.iter().map(|a| r2_types::CallArg { name: a.name.clone(), value: hoist_matmuls_expr(&a.value, pre, ctr) }).collect() },
        other => other.clone(),
    }
}

/// Statement-level matmul hoisting (recurses into loop bodies).
pub(crate) fn hoist_matmuls(body: &r2_types::Expr, ctr: &mut u32) -> r2_types::Expr {
    use r2_types::Expr::*;
    let stmts = match body { Block(s) => s.clone(), other => vec![other.clone()] };
    let mut out: Vec<r2_types::Expr> = Vec::new();
    let n = stmts.len();
    for (i, st) in stmts.iter().enumerate() {
        let is_last = i + 1 == n;
        match st {
            Assign { target, value, superassign } => {
                // Whole-rhs matvec stays a statement; only its rhs-arg may hoist.
                if let Binary { op: r2_types::BinOp::MatMul, lhs, rhs } = value.as_ref() {
                    let rname = match rhs.as_ref() {
                        Symbol(_) => rhs.as_ref().clone(),
                        other => {
                            let mut pre = Vec::new();
                            let inner = hoist_matmuls_expr(other, &mut pre, ctr);
                            out.extend(pre);
                            let t: std::sync::Arc<str> = std::sync::Arc::from(format!(".__mvarg{}", *ctr));
                            *ctr += 1;
                            out.push(Assign { target: Box::new(Symbol(t.clone())), value: Box::new(inner), superassign: false });
                            Symbol(t)
                        }
                    };
                    out.push(Assign { target: target.clone(),
                        value: Box::new(Binary { op: r2_types::BinOp::MatMul, lhs: lhs.clone(), rhs: Box::new(rname) }),
                        superassign: *superassign });
                } else {
                    let mut pre = Vec::new();
                    let v = hoist_matmuls_expr(value, &mut pre, ctr);
                    out.extend(pre);
                    out.push(Assign { target: target.clone(), value: Box::new(v), superassign: *superassign });
                }
            }
            For { var, iter, body } => {
                let mut pre = Vec::new();
                let it = hoist_matmuls_expr(iter, &mut pre, ctr);
                out.extend(pre);
                let nb = hoist_matmuls(body, ctr);
                out.push(For { var: var.clone(), iter: Box::new(it), body: Box::new(nb) });
            }
            other if is_last => {
                let mut pre = Vec::new();
                let v = hoist_matmuls_expr(other, &mut pre, ctr);
                out.extend(pre);
                out.push(v);
            }
            other => out.push(other.clone()),
        }
    }
    Block(out)
}

/// Which of the two params is used as a `%*%` LEFT operand (bare or wrapped in
/// `t(...)`)? That param is the matrix of a matrix-state kernel.
pub(crate) fn matmul_matrix_param(e: &r2_types::Expr, p0: &std::sync::Arc<str>, p1: &std::sync::Arc<str>) -> Option<std::sync::Arc<str>> {
    use r2_types::Expr::*;
    let as_param = |s: &str| -> Option<std::sync::Arc<str>> {
        if s == p0.as_ref() { Some(p0.clone()) } else if s == p1.as_ref() { Some(p1.clone()) } else { None }
    };
    match e {
        Binary { op: r2_types::BinOp::MatMul, lhs, rhs } => {
            let hit = match lhs.as_ref() {
                Symbol(s) => as_param(s.as_ref()),
                Call { func, args } if matches!(func.as_ref(), Symbol(f) if f.as_ref() == "t") && args.len() == 1 =>
                    match &args[0].value { Symbol(s) => as_param(s.as_ref()), _ => None },
                _ => None,
            };
            hit.or_else(|| matmul_matrix_param(rhs, p0, p1))
        }
        Binary { lhs, rhs, .. } => matmul_matrix_param(lhs, p0, p1).or_else(|| matmul_matrix_param(rhs, p0, p1)),
        Unary { expr, .. } => matmul_matrix_param(expr, p0, p1),
        Assign { value, .. } => matmul_matrix_param(value, p0, p1),
        Call { args, .. } => args.iter().find_map(|a| matmul_matrix_param(&a.value, p0, p1)),
        If { cond, then, else_ } => matmul_matrix_param(cond, p0, p1)
            .or_else(|| matmul_matrix_param(then, p0, p1))
            .or_else(|| else_.as_ref().and_then(|x| matmul_matrix_param(x, p0, p1))),
        For { iter, body, .. } => matmul_matrix_param(iter, p0, p1).or_else(|| matmul_matrix_param(body, p0, p1)),
        While { cond, body } => matmul_matrix_param(cond, p0, p1).or_else(|| matmul_matrix_param(body, p0, p1)),
        Block(s) => s.iter().find_map(|x| matmul_matrix_param(x, p0, p1)),
        Return(v) => matmul_matrix_param(v, p0, p1),
        _ => None,
    }
}

/// Does `e` contain a `sum`/`prod`/`mean` reduction call? Cheap gate so the
/// multi-reduction kernel is only attempted on plausibly-scalar bodies.
pub(crate) fn mentions_reduction(e: &r2_types::Expr) -> bool {
    use r2_types::Expr::*;
    match e {
        Call { func, args } => matches!(func.as_ref(), Symbol(s) if matches!(s.as_ref(), "sum" | "prod" | "mean"))
            || args.iter().any(|a| mentions_reduction(&a.value)),
        Binary { lhs, rhs, .. } => mentions_reduction(lhs) || mentions_reduction(rhs),
        Unary { expr, .. } => mentions_reduction(expr),
        Assign { value, .. } => mentions_reduction(value),
        If { cond, then, else_ } => mentions_reduction(cond) || mentions_reduction(then)
            || else_.as_ref().map_or(false, |x| mentions_reduction(x)),
        Block(s) => s.iter().any(mentions_reduction),
        Return(v) => mentions_reduction(v),
        // J.4 iterative kernels: reductions inside a counted loop body count.
        For { iter, body, .. } => mentions_reduction(iter) || mentions_reduction(body),
        While { cond, body } => mentions_reduction(cond) || mentions_reduction(body),
        _ => false,
    }
}

