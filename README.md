<p align="center">
  <img src="assets/logo.png" alt="Ardon-R2 logo" width="320">
</p>

<h1 align="center">Ardon-R2</h1>

<p align="center"><strong>Inspired by R. Built on Rust.</strong><br>
<em>An AI-Assisted Project. v0.3.4.</em></p>

---

**A ground-up reimplementation of statistical computing in Rust.**

> The dominant inefficiencies in today's AI stack are not the math — they're
> the glue between languages, the copies between memory regions, the GIL,
> the dependency mess, and the silent-failure culture. R2 is built to remove
> them at the foundation, not paper over them with another wrapper.

R2 takes R's best ideas — vectorized operations, formula syntax, data frames — and rebuilds them from scratch with modern performance, Rust-only dependencies, and built-in machine learning.

```
R2 — Statistical Computing, Reimagined
Version 0.3.4 (2026) | Inspired by R, Built on Rust
Created by Devendra Tandale | An AI assisted project
```

## What's new in v0.3.4 (June 2026)

A JIT correctness fix for `for`-loop accumulators inside functions, a Linux
installer for the CLI and GUI, and an internal refactor that breaks the
largest source files into focused modules (no behaviour change). The
previous release (v0.3.3) added first-class language objects (the
`language.c` equivalent: `quote`/`eval`/`substitute`/`bquote`/…).

**Full per-release history lives in [CHANGELOG.md](CHANGELOG.md)** — this
README stays version-light on purpose.

## Why R2?

| | R | Python (scikit-learn) | R2 |
|---|---|---|---|
| Install size | 200+ MB | 5-8 GB | **5 MB** |
| Setup time | minutes | hours (pip conflicts) | **0 seconds** |
| ML packages needed | 5-10 installs | 3-5 installs | **0 (built-in)** |
| Matrix multiply speed | 1x (ref BLAS) | 1x (NumPy) | **2.2x faster** |
| Cloud required? | No | Often yes | **No** |

## Who is this for

**R users** who love the syntax but are tired of `install.packages()` failing on a fresh machine, 200+ MB toolchain installs, and Imports/Suggests dependency cascades. R2 gives you `lm()`, `gbm()`, `rpart()`, `kmeans()` out of the box from a single 5 MB binary — same R-style syntax, no package install for the 320 built-in functions.

**Rust developers** who want to work on a real numerical-computing project that isn't a thin wrapper around someone else's BLAS. Every line — micro-kernel, decompositions, distributions, ML algorithms, parser, REPL — is hand-written Rust you can read and modify. No C, no C++, no Fortran underneath.

**Researchers and teams** who care about reproducibility and energy cost. The 5 MB binary builds the same on Linux, Windows, macOS, x86 and ARM. No conda envs to break, no CUDA versions to match, no cloud bills for interpreter overhead.

## Quick Start

```bash
# Build (requires Rust 1.70+)
cargo build --release

# Run
.\target\release\r2.exe      # Windows
./target/release/r2           # Linux/Mac
```

## Features at a Glance

- **320 built-in functions** — no packages to install
- **Repeated-measures ANOVA** — `aov(y ~ x + Error(subject), data=df)`, R-bit-identical
- **Hotelling's T²** — one-sample, two-sample, and paired/multivariate variants
- **MANOVA** — `manova(cbind(y1, y2) ~ group, data=df)` with all four classical statistics
- **12 ML algorithms** — decision tree, random forest, gradient boosting, KNN, PCA, K-means, Naive Bayes
- **Formula syntax with factor expansion** — `lm(mpg ~ factor(cyl) + wt, data = mtcars)`
- **In-memory graphics device + full `par()`** — `par(mfrow=c(2,2))` multi-panel layouts, `dev.off()`, `save_plot(path)`
- **Built-in browser-based plot viewer** — `dev.view()` opens a live, auto-refreshing page with a session gallery of every plot you've made
- **2.2x faster** matrix multiply than R (Windows default BLAS)
- **6.6MB binary** — runs on any laptop, no cloud needed
- **R-compatible syntax** — `<-` assignment, 1-based indexing, `$` column access
- **Pure Rust math kernel** — Rust-only dependencies, no C/C++
- **Cross-platform** — Linux, Windows, macOS (Intel and Apple Silicon)

## Statistics

```r
# Linear and generalized linear models
model <- lm(mpg ~ wt + hp, data = mtcars)
coef(model)
summary(model)
glm(y ~ x, data = df, family = "binomial")

# Hypothesis tests
t.test(x, mu = 0)
t.test(y ~ group, data = df)            # Welch two-sample
cor(x, y)

# Repeated-measures ANOVA (R-bit-identical when R uses factor(subject))
aov(response ~ treatment + Error(subject), data = df)

# Paired t-test through formula + Error syntax (extension over R)
t.test(response ~ treatment + Error(subject), paired = TRUE, data = df)

# Multivariate hypothesis testing — Hotelling's T²
hotelling.test(X)                       # one-sample,  H0: mu = 0
hotelling.test(X, mu = c(0, 0))         # one-sample,  H0: mu = mu0
hotelling.test(A, B)                    # two-sample,  H0: mu_A = mu_B
hotelling.test(X, Y, paired = TRUE)     # paired/multivariate paired

# MANOVA — multivariate ANOVA
manova(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) ~ Species,
       data = iris)
# Reports Wilks' Lambda, Pillai, Hotelling-Lawley, Roy's largest root.
```

## Machine Learning

```r
rpart(Petal.Length ~ ., data = iris)           # Decision tree
rf(Petal.Length ~ ., data = iris, ntrees = 50) # Random forest
gbm(y ~ ., data = df, ntrees = 100)           # Gradient boosting
kmeans(x, centers = 3)                         # Clustering
prcomp(x)                                      # PCA
knn(train, test, labels, k = 5)                # KNN
cv(x, y, model = "lm", k = 5)                 # Cross-validation
confusion.matrix(predicted, actual)            # Evaluation
```

## Plotting

```r
# Open the browser-based live viewer (auto-refreshes as you plot)
dev.view()

# Single plot
plot(iris$Sepal.Length, iris$Sepal.Width, main = "Iris", col = "blue")
abline(h = mean(iris$Sepal.Width), col = "red", lty = 2)

# Multi-panel layout via par(mfrow=...)
par(mfrow = c(2, 2))
plot(iris$Sepal.Length, iris$Sepal.Width)
hist(iris$Petal.Length)
boxplot(iris$Petal.Width)
barplot(table(iris$Species))
save_plot("iris-overview.svg")   # explicit flush
dev.off()                        # reset device
```

Plots draw into a thread-local in-memory graphics device. `par()`
supports `mfrow`, `mfcol`, `mar`, `cex`, `col`, `lty`, `lwd`, `pch`,
and the rest of the common CRAN R parameters. Use `oldpar <- par(...)`
and `par(oldpar)` for save-and-restore semantics.

`dev.view()` starts a tiny built-in HTTP server (zero external
dependencies) and opens your default browser at
`http://127.0.0.1:8765/`. The page shows the current plot at the top
(auto-refreshes every 1.5 s) and a session gallery of every `.svg`
file in your working directory underneath. Click any thumbnail to pin
the top pane to that file; click "return to live" to resume polling.

Try `samples/demo_graphics.r` for a walk-through:

```bash
./target/release/r2 samples/demo_graphics.r
```

The demo is interactive — each plot waits for you to inspect it,
prompts for a save filename (or default), and pauses for Enter before
moving on.

## Data Handling

```r
df <- read.csv("data.csv")
filter(iris, iris$Sepal.Length > 7)
select(iris, "Sepal.Length", "Species")
mutate(iris, ratio = iris$Sepal.Length / iris$Sepal.Width)
iris[1:10, ]
summary(iris)
```

## File Types

| Extension | Purpose | Example |
|---|---|---|
| `.r` | R2 script (source code) | `source("analysis.r")` |
| `.r2s` | Session save (all variables) | `save("session.r2s")` |
| `.r2d` | Data object (DataFrame, Matrix) | `save(iris, "data.r2d")` |
| `.r2m` | Model object (lm, gbm, rf...) | `save(model, "model.r2m")` |

## Performance

R2 is fast where it counts — **matrix multiply 22×** faster than R's
reference BLAS (1024²), fused **math-JIT** loops, and the **apply family**
(`sapply` ~25×) — while matching R to **~7 significant figures** on
statistical accuracy. R trades back on single memory-bandwidth-bound passes.

Full benchmark tables, the accuracy comparison, the math-JIT breakdown, and
reproducibility notes live in **[PERFORMANCE.md](PERFORMANCE.md)**. Reproduce
with `pwsh benchmarksmparison\run.ps1` (or `bash benchmarks/comparison/run.sh`).

## Project Structure

```
r2/
├── Cargo.toml                  # Workspace configuration
├── LICENSE                     # AGPL v3
├── README.md                   # This file
├── VISION.md                   # Green AI roadmap
├── FUNCTIONS.md                # All 192 functions documented
├── CHANGELOG.md                # Release history
├── CLA.md                      # Contributor License Agreement
├── CONTRIBUTING.md             # How to contribute
├── crates/
│   ├── r2-engine/              # evaluator, JIT call path, builtin registry
│   │   └── src/lib.rs
│   ├── r2-linalg/              # Pure Rust math kernel (1,278 lines)
│   │   └── src/
│   │       ├── lib.rs          # BLAS L1-L3, dgemm 8×4 micro-kernel
│   │       ├── decomp.rs       # LU, Cholesky, QR, SVD, Eigenvalues
│   │       └── solve.rs        # Linear solvers, least-squares
│   ├── r2-types/               # Core types (1,023 lines)
│   │   └── src/lib.rs          # RVal, DataFrame, Matrix, Tensor
│   ├── r2-parser/              # Lexer + parser (485 lines)
│   │   └── src/
│   │       ├── lexer.rs
│   │       └── parser.rs
│   ├── r2-repl/                # Interactive console (195 lines)
│   │   └── src/main.rs         # Arrow key history, ? help
│   ├── r2-base/                # Embedded datasets (126 lines)
│   │   └── src/lib.rs          # iris, mtcars, airquality
│   ├── r2-graphics/            # SVG plot generation
│   ├── r2-stats/               # Statistical functions
│   ├── r2-utils/               # Utility functions
│   ├── r2-memory/              # Memory management
│   └── r2-pkg/                 # Package system
└── samples/
    ├── demo_basics.r           # Statistics demo script
    ├── demo_ml.r               # ML algorithms demo
    ├── demo_benchmark.r        # Speed comparison vs R
    └── mymath/                 # Sample addon package
        └── R2/
            └── mymath.r        # factorial, fibonacci, gcd
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  R2 REPL — Interactive Console                  │
│  Arrow keys, ?help, tab completion              │
├─────────────────────────────────────────────────┤
│  R2 Parser — Lexer + Recursive Descent          │
│  Tokenize → AST → Expression tree               │
├─────────────────────────────────────────────────┤
│  R2 Engine — 192 Builtin Functions              │
│  Stats │ ML │ Data │ Graphics │ I/O │ System    │
│  Class-based dispatch: summary(), plot(),       │
│  predict() auto-detect object type              │
├─────────────────────────────────────────────────┤
│  R2 Types — RVal, DataFrame, Matrix, Tensor     │
│  Factor, Formula, TypeInstance, Env              │
├─────────────────────────────────────────────────┤
│  r2-linalg — Pure Rust Math Kernel              │
│  BLAS L1-L3 │ 8×4 micro-kernel │ cache blocking │
│  LU │ Cholesky │ QR │ SVD │ Jacobi eigenvalues  │
│  Fused least-squares │ Cramer 2×2/3×3           │
│  Rust-only dependencies, no C/C++                     │
└─────────────────────────────────────────────────┘
```

## Documentation

- `?topic` or `??topic` — Quick help in REPL
- `help()` — List all help topics
- `FUNCTIONS.md` — Complete function reference (400+ functions)
- `PERFORMANCE.md` — R-vs-R2 benchmarks + accuracy comparison
- `CHANGELOG.md` — Release history
- `VISION.md` — Project roadmap and Green AI vision
- `docs/ARCHITECTURE.md` — compiler / runtime / IR / JIT design
- `INSTALL_LINUX.md` — Linux install guide (CLI + GUI)
- `benchmarks/`, `benchmarks/comparison/` — runnable benchmark harnesses
- `samples/` — example programs you can run with `r2 <file>`

## ~25 crates | 400+ builtins | JIT-compiled user functions | Pure-Rust dependencies — no C/C++ libraries

## Roadmap

Shipped releases are recorded in **[CHANGELOG.md](CHANGELOG.md)**; the
forward plan and design direction live in **[VISION.md](VISION.md)** and
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

In brief: the v0.x line has built out the core language, statistics, ML,
the columnar storage substrate, the Cranelift JIT for user functions, and
first-class language objects. Next on the horizon are richer graphics
back-ends, more statistical tests, and hardware-aware scheduling (Phase G:
CPU-feature detection, GPU dispatch). This is an evolving project — the
living docs above are the source of truth, not a frozen checklist.

## How to contribute

Good first issues, grouped by background:

**If you know R well** — port a missing statistical test (Mann-Whitney U, Levene's, Kruskal-Wallis, Friedman); port a CRAN dataset (just data plus a help topic); write a help topic for an existing builtin; find R-vs-R2 behavior mismatches by running scripts from `COMPARISON_TESTS.md` and file them as issues.

**If you know Rust well** — extend the JIT to handle a new pattern (e.g. `function(v) sort(v)`); add a pure builtin to `pure_apply()` so the apply family parallelizes it; add a new SVG plot type (violin, Q-Q, density); add QR with column pivoting or Lanczos iteration for large eigenvalues to `r2-linalg`; help with the Phase F.3 destructive storage migration (mechanical, well-scoped); profile a builtin and submit a PR with speedup numbers.

**Either way** — open the issue first so we can scope it together. See `CONTRIBUTING.md` for the workflow and `CLA.md` for the contributor agreement.

## License

AGPL v3 — Created by Devendra Tandale
