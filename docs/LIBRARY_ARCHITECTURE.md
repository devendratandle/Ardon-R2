# R2 Library Architecture

## Three Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3: ADDON LIBRARIES (installed by user)                    │
│  install.packages("r2ml")  →  library(r2ml)                    │
│  Location: ~/.r2/library/                                       │
│  Examples: r2ml, r2plot, r2tidyverse, r2shiny                   │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2: BASE LIBRARIES (ship with R2, auto-loaded)            │
│  Always on search path, detachable                              │
│  Location: /usr/lib/r2/base/                                    │
│                                                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │  base    │  stats   │ graphics │  utils   │ datasets │      │
│  │ c()      │ lm()     │ plot()   │read.csv()│ iris     │      │
│  │ length() │ glm()    │ hist()   │write.csv()│mtcars   │      │
│  │ paste()  │ t.test() │ barplot()│ str()    │airquality│      │
│  │ which()  │ cor()    │ boxplot()│ head()   │          │      │
│  │ seq()    │ sd()     │ lines()  │ tail()   │          │      │
│  │ sort()   │ rnorm()  │ points() │ summary()│          │      │
│  │ ...      │ ...      │ ...      │ ...      │          │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│  TIER 1: CORE (compiled into r2 binary, always present)        │
│  Cannot be detached or unloaded                                 │
│                                                                 │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │  types   │  parser  │  engine  │  memory  │  repl    │      │
│  │ RVal     │ Lexer    │ Eval     │ Tiered   │ Console  │      │
│  │ Expr     │ Parser   │ Builtins │ Chunked  │ I/O      │      │
│  │ Matrix   │ AST      │ Scoping  │ Parallel │          │      │
│  │ Tensor   │          │ Types    │          │          │      │
│  │ DataFrame│          │ Methods  │          │          │      │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## What lives where

### CORE (r2 binary) — ~5 MB compiled
Only the absolute minimum to run the language:
- Type system (RVal, Expr, Matrix, Tensor, DataFrame, etc.)
- Parser (lexer + parser)
- Evaluation engine (variable lookup, function calls, control flow)
- Memory manager
- Package loading mechanism
- REPL
- Primitive operators: +, -, *, /, <-, |>, $, [, [[, ::
- Primitive builtins ONLY: c(), length(), print(), cat(), typeof(), is.na(), 
  as.numeric(), as.character(), if/for/while, function(), type, method

### BASE LIBRARIES (ship with installer) — ~10 MB total
These use the SAME addon mechanism as third-party packages,
but ship pre-installed and auto-load on startup.

**r2-base** (~3 MB): seq, rep, which, sort, rev, unique, paste, paste0,
  toupper, tolower, substr, grep, gsub, strsplit, nchar, table, sapply,
  lapply, apply, tapply, merge, rbind, cbind, match, factor, names,
  nrow, ncol, complete.cases, na.omit, Sys.time, system.time

**r2-stats** (~3 MB): mean, sd, var, cor, lm, glm, t.test, chisq.test,
  anova, kmeans, prcomp, optim, dnorm, pnorm, qnorm, rnorm, dunif,
  punif, qunif, runif, dbinom, pbinom, rbinom, dpois, rpois, sample,
  median, quantile, predict, residuals, fitted, confint, AIC, BIC

**r2-graphics** (~2 MB): plot, hist, barplot, boxplot, lines, points,
  abline, legend, par, title, axis, text, polygon, segments, arrows,
  svg.open, svg.close  (SVG backend, no system deps)

**r2-utils** (~1 MB): read.csv, write.csv, read.table, write.table,
  str, head, tail, summary, View, search, ls, rm, getwd, setwd,
  Sys.time, proc.time, format, sprintf, cat, message, warning, stop

**r2-datasets** (~1 MB): iris, mtcars, airquality, ToothGrowth,
  PlantGrowth, USArrests, ChickWeight, CO2, faithful, sleep

### ADDON LIBRARIES (user-installed) — varies
Use install.packages() and library(). Same mechanism as base libraries
but stored in user's library path.

**Example: r2-ml** (addon):
  - Depends on: r2-stats (for matrix ops, distributions)
  - Provides: nn.train(), nn.predict(), rf.train(), rf.predict(),
    boost.train(), svm(), tensor.train(), onnx.load()
  - Uses Tensor type from CORE (already in the binary)

## How a library registers functions

Every library (base or addon) provides a registration function:

```rust
// In r2-stats/src/lib.rs
pub fn register(engine: &mut Engine) {
    engine.register_fn("lm", bi_lm);
    engine.register_fn("glm", bi_glm);
    engine.register_fn("t.test", bi_t_test);
    engine.register_fn("cor", bi_cor);
    engine.register_fn("sd", bi_sd);
    engine.register_fn("rnorm", bi_rnorm);
    // ...
}
```

The engine provides `register_fn()` which addon libraries call.
Base libraries call it on startup. Addon libraries call it when `library()` runs.

## Search path on startup

```
> search()
[1] ".GlobalEnv"        "package:stats"     "package:graphics"
[4] "package:utils"     "package:datasets"  "package:base"
```

## detach() works for base libraries too

```
> detach(stats)    # lm() is now gone
> lm(y ~ x)
Error: object 'lm' not found
> library(stats)   # bring it back
```

## Addon library lifecycle

```
> install.packages("r2ml")     # downloads from r2cran, compiles
> library(r2ml)                # loads, registers functions
> nn.train(data, target)       # now available
> detach(r2ml)                 # unloaded, functions gone
```

## Estimated installer sizes

| Component | Size |
|---|---|
| r2 binary (core) | ~5 MB |
| r2-base library | ~3 MB |
| r2-stats library | ~3 MB |
| r2-graphics library | ~2 MB |
| r2-utils library | ~1 MB |
| r2-datasets | ~1 MB |
| **Total base installer** | **~15 MB** |

Compare: R base installer is ~80 MB. Python is ~30 MB.
