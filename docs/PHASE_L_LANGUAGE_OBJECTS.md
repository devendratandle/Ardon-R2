# Phase L — First-class language objects (the `language.c` equivalent)

> Planning doc. Nothing here is implemented yet. Bootstrap a fresh session
> from this file — do not re-read old chat. Locked git policy still applies:
> local commits free; **no push without explicit per-action permission**.

## Why

R's defining trait is "code is data": a `LANGSXP` is both runnable and
inspectable. GNU R spends `language.c` (~4.1k C lines) on this. R2 has the
`eval.c` equivalent (`eval_in`/`call_fn`/JIT in `r2-engine/src/lib.rs`) and
the `Rinternals.h` equivalent (`RVal`/`Expr`/`Attrs`/`EvalArg`/`EngineCtx`
in `r2-types/src/lib.rs`), but **no first-class language objects** —
`quote`/`substitute`/`eval`/`deparse`/`match.call`/`body`/`formals` are all
missing. Only NSE special-cases (`with`/`switch`/`tryCatch`/`curve`),
formulas (`formula.rs`), and the internal `fmt_expr` deparse helper exist.

## Keystone

`Expr` already lives in **r2-types** (next to `RVal`; `Closure.body` is
`Arc<Expr>`). So adding the language-object type is a same-crate change,
no new dependency, AST already `Arc`-shareable:

```rust
// r2-types/src/lib.rs, RVal enum
Lang(Arc<Expr>),   // R's LANGSXP/EXPRSXP — a quoted, unevaluated expression
```

`quote(e)` wraps an `Expr` in `Lang`; `eval(l)` unwraps and runs it through
the existing `eval_in`. Everything else is builtins on top.

## Slices, builtins, and how each is done

### L.1 — round-trip (easy, NO hot-path changes) ✅ DONE
Shipped: `RVal::Lang(Arc<Expr>)` added; `fmt_expr` deparser moved to
r2-types as `pub fn deparse` (shared by `RVal::Lang` Display + the builtin);
`type_name` → "call". Builtins `eval`/`parse`/`deparse`/`call`/`as.call` in
`builtins/lang.rs`; `quote` is an NSE intercept in `eval_in`. Verified:
`deparse(quote(x+1))`→"x + 1", `eval(parse(text="1+1"))`→2, multi-stmt
parse, `eval(call("sum",1,2,3))`→6, Lang autoprints as deparsed source.
The new variant broke NO exhaustive matches beyond type_name (the proxy
estimate was generous). FUNCTIONS.md 301→307.
- `quote(e)` — NSE intercept in `eval_in` (like existing `with`/`curve`):
  take arg-0 UNEVALUATED, wrap its `Expr` in `Lang`.
- `eval(l, env)` — plain builtin: `Lang(e) => self.eval_in(&e, env)`.
- `deparse(x)` — expose existing `fmt_expr` (formula.rs:143) as a builtin.
- `parse(text=)` — `r2_parser::Parser::parse(text)` → `Vec<Expr>` → `Lang`/list.
- `call(name, …)`, `as.call(list)` — build `Lang(Expr::Call{…})`.

### L.2 — function introspection (easy) ~0.5–1 session, low risk
- `body(f)` — `RVal::Closure(cl) => Lang(cl.body.clone())` (one Arc clone).
- `formals(f)`, `args(f)` — return `cl.params` as a list/pairlist.
- `body<-`, `formals<-` — rebuild the `Closure` with a new body/params.

### L.3 — NSE + call stack (the hard part) ~1.5–2 sessions, medium-high risk
R relies on **promises** (lazy unevaluated arg exprs); R2 is **eager**, so the
original arg expressions are gone by the time a closure runs. Fix = **arg-
expression capture**: in `call_fn` closure binding, also record each arg's
unevaluated `Expr` in the call frame:
```rust
struct CallFrame { call: Arc<Expr>, arg_exprs: Vec<(Option<Arc<str>>, Arc<Expr>)>, … }
```
This one mechanism unlocks all of:
- `substitute(e)` — walk `e`, replace symbols with captured arg-exprs.
- `match.call()` — reconstruct the call with names matched.
- `sys.call()` — return `frame.call`.
- `bquote()` — quasiquotation, layered on substitute.

**PERFORMANCE GATE (mandatory):** capturing arg-exprs on every call is waste
(99% never use substitute/match.call). Gate it with a one-time static flag
set at closure creation if the body textually references
`substitute`/`match.call`/`sys.call`. Closures that don't use them pay ZERO
cost. Do NOT capture unconditionally on the hot path.

**Semantic caveat:** R2's `substitute` is promise-free; with arg-expr capture
it matches R for the common case (parameter passed as an expression).
Document divergence rather than chase bug-for-bug parity.

## Companion functions (recommend including, small)
`is.call`, `is.symbol`, `is.name`, `is.language`, `as.name`, `expression`.

## Counts (for doc updates)
- **New builtins: ~15 core** (L.1 = 6, L.2 = 5, L.3 = 4), **~21 with companions**.
- FUNCTIONS.md: **301 → ~316–321**.

## Code to update / reassess (the `RVal::Lang` ripple)
`RVal::Closure` is matched in **9 sites**; ~**5–7** need a real `Lang` arm
(rest fall through `_`):
1. `type_name()`/class → `"call"`/`"name"`
2. autoprint/`print` → deparse (`fmt_expr`)
3. `val_to_str`/`format`
4. `to_items()` — BOTH copies (r2-types AND r2-engine)
5. coercions: `as.character` (→deparse), `as.list` (→call parts)
6. `str()`
Plus: `call_fn` closure-binding (L.3, gated) ×1; `registry_tables.rs` ×1.

## Docs to update (6)
FUNCTIONS.md, docs/MISSING_FUNCTIONS.md (remove `match.call`, line 62),
CHANGELOG.md, README.md, llms.txt, CLAUDE.md §8.

## Recommended order
L.1 (+L.2) first — high ROI, low risk, no hot-path exposure (gives
`eval(parse())`, readable deparsed errors, function introspection). Treat L.3
as a separate deliberate decision — it touches the evaluator and carries the
promise-semantics caveat; do it when targeting NSE-heavy packages
(dplyr/rlang/testthat-style). Pairs with the `...` dots work already done.

## First concrete step next session
1. `grep -rn "RVal::Closure" crates/*/src --include=*.rs` → triage which of the
   9 sites need a `Lang` arm vs. a `_` wildcard.
2. Add `RVal::Lang(Arc<Expr>)` to r2-types; fix the must-update arms.
3. Implement L.1 builtins; register in `registry_tables.rs`; build debug
   (`cargo build -p r2-repl`, kill r2.exe first) and verify
   `eval(parse(text="1+1"))` and `deparse(quote(x+1))`.
