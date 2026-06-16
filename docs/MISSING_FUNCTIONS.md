# Missing base-R functions — roadmap to a "basic usable runtime"

R2 implements ~220+ builtins (see FUNCTIONS.md) — strong on stats/ML/
graphics/data-frames. This lists **common base-R functions still missing**,
found by auditing the engine + parser registries against everyday R
workflows. Tiered by how often a typical user hits them.

## Tier 1 — Essential everyday idioms — ✅ DONE
Implemented + verified: `seq_len`, `seq_along`, `%in%`, `setdiff`, `union`,
`intersect`, `unlist`, `setNames`, `append`, `split`, `pmin`, `pmax`,
`cut`, `Reduce`, `Filter`, `Map`, `switch`, `with`, `invisible`,
`stopifnot`, `attr`, `attributes`, `structure`, `inherits`, `format`
(+ `signif`, and a JIT closure-cache correctness fix). The list below is
kept for reference.

These show up in almost every R script; their absence makes common code
fail outright.

| Function | Why it matters |
|---|---|
| `seq_len(n)`, `seq_along(x)` | the canonical safe loop/index idioms |
| `%in%` | membership test — extremely common in filtering |
| `setdiff`, `union`, `intersect` | set algebra on vectors |
| `unlist(x)` | flatten a list — pervasive after `lapply`/`Map` |
| `setNames(x, nm)` | one-liner naming idiom |
| `pmin`, `pmax` | parallel (element-wise) min/max |
| `cut(x, breaks)` | bin a numeric into factor intervals |
| `split(x, f)` | group a vector/df by a factor |
| `with(data, expr)` | evaluate an expression in a df's scope |
| `Reduce`, `Filter`, `Map` | functional-programming staples |
| `append(x, values, after)` | insert into a vector |
| `invisible(x)` | return without auto-printing |
| `stopifnot(...)` | assertion guard |
| `switch(expr, ...)` | multi-branch selection |
| `format(x, ...)` | generic value formatting (only Date/POSIXct exist) |
| `attr`, `attributes`, `structure` | generic attribute access / construction |
| `inherits(x, class)` | class predicate used everywhere in S3 code |

## Tier 2 — Common math / stats — mostly DONE
Implemented + verified: `factorial`, `choose`, `gamma`, `lgamma`, `beta`,
`signif`, `outer`, `combn`, `mad`, `fivenum`; distributions `dexp`/`pexp`/
`qexp`, `dbinom`/`pbinom`, `dpois`/`ppois`, `dt`/`pt`, `dchisq`/`pchisq`,
`pf` (verified against R to ~1e-6).

Still missing (Tier 2 remainder):
| Function | Why |
|---|---|
| `density(x)`, `ecdf(x)` | distribution estimation + plotting |
| `rexp(n, rate)` | exponential RNG (have d/p/q; needs the rng module) |
| `qbinom`, `qpois`, `qt`, `qchisq`, `qf` | quantile (inverse-CDF) forms |
| `optimize`, `uniroot`, `integrate` | 1-D numerical methods (closure args) |

## Tier 3 — I/O & misc convenience — mostly DONE
Implemented + verified: `readLines`, `writeLines`, `substring`, `tryCatch`,
`as.matrix`/`as.vector`/`as.list`, `is.function`/`is.list`/`is.vector`/
`is.element`, plus the `numeric`/`integer`/`character`/`logical`
constructors. **Also fixed (engine):** functional `...` (dots) forwarding,
variadic `sum`/`min`/`max`/`prod`, `[[`-read, `df[cols]` column select,
iterate-over-df-columns.

Still missing (Tier 3 remainder):
| Function | Why |
|---|---|
| `formatC`, `prettyNum`, `strrep`, `chartr` | string formatting |
| `by`, `ave`, `within`, `stack` | split-apply on data frames |
| `Sys.setenv` | (have `Sys.getenv`) |
| `match.arg`, `match.call`, `nargs` | argument helpers (`..N` lexer-split also pending) |

## Suggested build order
Tier 1 first (most are small, pure-Rust builtins — biggest usability win
per line), then Tier 2 (distributions extend the existing d/p/q/r family),
then Tier 3. `with`/`switch`/`Reduce`/`Filter`/`Map`/`outer` need a small
amount of engine support (closure application / NSE) like `curve()` already
uses; the rest are straightforward vector builtins.
