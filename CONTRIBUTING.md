# Contributing to Ardon-R2

Thank you for considering a contribution to Ardon-R2. This document is what you read before writing code, and what reviewers point at during PRs. Following it makes review faster, prevents breakage, and keeps the codebase coherent as it grows.

By submitting code you agree to the [Contributor License Agreement](CLA.md) and accept that your contribution is licensed under [AGPL-3.0](LICENSE).

---

## Quick start

```bash
# Clone and build
git clone https://github.com/{owner}/Ardon-R2.git
cd Ardon-R2
cargo build --release --workspace

# Run the full test suite (must be green before submitting any PR)
cargo test --release --workspace

# Format check (CI enforces this)
cargo fmt --all --check

# Lint (CI tracks this; aim for zero new warnings)
cargo clippy --workspace --release --all-targets

# Launch the REPL
./target/release/r2
```

Build prerequisites: a recent stable Rust toolchain (1.75+) installed via `rustup`. No system libraries are required — Ardon-R2 is intentionally free of C/C++ dependencies.

---

## Rules that prevent breakage

These exist because we have all paid for breaking them at least once. Reviewers will block PRs that violate them.

### 1. Statistical numerics never silently regress

Any change touching `r2-stats`, `r2-linalg`, `r2-engine` arithmetic paths, or the kernel layer must include a test asserting numerical output to at least **1e-9 of CRAN R 4.5.3's output** for the same input. If your change moves a number, R has to agree it should move. The `bench/r_vs_r2/` harness is the easiest way to verify this; the `dataset_integrity` tests in `r2-base` are non-negotiable.

### 2. Public APIs do not change without a deprecation cycle

If a function, builtin, or builtin signature is renamed, removed, or reshaped, the old form must continue to work with a deprecation warning for at least one minor version. Breaking changes that bypass this rule require explicit sign-off in the PR description and a corresponding `BREAKING:` entry in `CHANGELOG.md`.

### 3. No new `unsafe` in hot paths

`unsafe` blocks already in the JIT and kernel layer are accepted because they were audited. New `unsafe` introductions need a justification in the PR description, a comment block in the code naming the safety invariants, and reviewer agreement that no safe alternative is feasible. "It's faster" is not sufficient justification on its own.

### 4. Tests live next to the code they test

Unit tests inside `#[cfg(test)] mod tests` blocks at the bottom of the file. Integration tests in `crates/<crate>/tests/`. Cross-crate scenario tests in the workspace's `tests/` directory. A PR adding a public function without at least one test will be sent back.

### 5. Cross-crate changes require a design note in the PR

Touching three or more crates in a single PR? Open the PR with a short description of the architectural change you are making and why it spans crates. Trying to refactor across the whole workspace without that context makes review impossible.

### 6. Performance changes are measured, not asserted

If you claim a change is faster, include before/after numbers from `cargo bench` or the `bench/r_vs_r2/` harness. If you claim a change is equivalent, prove it didn't regress. Subjective performance claims without data will be questioned.

---

## Rules for snippet quality

These exist because we want the codebase to remain readable after the tenth contributor.

### Formatting and lints

- Run `cargo fmt --all` before every commit. CI verifies this.
- Run `cargo clippy --workspace --all-targets`. Aim for zero new warnings introduced by your PR. Existing warnings are tracked; do not add to the pile.
- A `rustfmt.toml` is committed at the repo root and locks the project style. Do not override it locally.

### Function size and shape

- Default to **functions under 100 lines**. If a function grows past that, ask whether two functions would be clearer.
- Default to **one concept per function**. A function that parses, validates, computes, and formats is doing four things.
- Default to **named arguments over boolean parameters**. `compute(x, fast=true)` is better than `compute(x, true)`.
- Avoid early returns mixed with deep nesting. Prefer the flat shape with `?` for error propagation.

### Naming

- `snake_case` for functions, variables, modules.
- `UpperCamelCase` for types, traits, enum variants.
- `SCREAMING_SNAKE_CASE` for constants.
- Crate names: `r2-{domain}` (lowercase, hyphenated). Module paths inside use underscores.
- Avoid abbreviations except for ubiquitous ones (`fn`, `mut`, `ref`, `df` for data frame, `mat` for matrix). When in doubt, spell it out.

### Comments and documentation

- Public items (`pub fn`, `pub struct`, `pub enum`) require a `///` doc comment.
- The doc comment answers two questions: *what does this do?* and *when should I reach for it?* — not *how is it implemented?*
- For statistical functions, link to the R documentation behavior you are matching: `/// Equivalent to R's lm() with treatment contrasts.`
- Comments inside function bodies explain *why*, not *what*. The code already tells the reader what.

### Imports

- Group imports: `std`, then external crates, then workspace crates, then `super`/`crate` locals. Blank line between groups.
- Avoid `use foo::*` glob imports outside of test modules and prelude files.

### Errors

- All fallible operations return `Result<T, R2Err>`, never `panic!()` in library code. Panics are acceptable in tests and in the REPL binary for unrecoverable startup errors.
- Use the `err!(Kind, "message")` macro for new errors, not raw `R2Err { ... }` literal construction.
- Error messages start with a lowercase verb and read as a continuation of the user's expectation: `"cannot convert character to numeric"`, not `"Error: Conversion failed."`.

### Tests

- Test function names are full sentences describing the behavior: `fn lm_two_vector_legacy_path_returns_intercept_and_slope()`. Underscores are free; use them.
- One `assert!` per logical claim. A test that mixes five unrelated assertions is five tests pretending to be one.
- Floating-point comparisons use a named tolerance, not magic numbers: `let tol = 1e-9;`.

---

## Commit messages

We use a relaxed form of [Conventional Commits](https://www.conventionalcommits.org). The first line is the only thing reviewers care about; everything else is for future archaeology.

**Format:**
```
<type>(<scope>): <one-line summary in lowercase, no period>

[Optional body explaining what changed and why.]
```

**Types we use:**
- `feat` — new feature visible to the user
- `fix` — bug fix
- `perf` — performance improvement (must include measurement)
- `refactor` — code change with no behavior change
- `test` — adding or correcting tests
- `docs` — documentation only
- `chore` — build, CI, tooling
- `breaking` — anything that breaks public API (rare; requires sign-off)

**Scope** is usually a crate or domain: `r2-stats`, `r2-jit`, `formula`, `kernel`, `oracle`, etc.

**Examples:**
```
feat(r2-stats): expand factor predictors in lm() via treatment contrasts
fix(r2-engine): resolve formula bare names against data argument first
perf(r2-jit): fuse map-reduce into single Cranelift loop (11x speedup)
test(r2-base): verify iris row 143 matches canonical R values
```

---

## Pull request process

1. Fork the repo and create a topic branch off `main`. Branch naming: `feat/short-description`, `fix/short-description`, etc.
2. Make your change. Commit in logical chunks (one logical change per commit if practical).
3. Push the branch and open a PR against `main`. Fill in the PR template completely.
4. CI will run automatically on Linux, Windows, and macOS. Wait for green.
5. A maintainer reviews. Expect at least one round of feedback for non-trivial PRs.
6. Once approved, the maintainer squashes-and-merges (default) or rebases-and-merges (for clean histories). You don't need to squash yourself.

### What blocks a merge

- Failing CI on any platform
- Reduced test coverage on changed lines
- New `unsafe` without justification
- Numerical regression vs CRAN R
- Public API change without deprecation path
- Cross-crate change without design note

### What does not block a merge

- Clippy warnings that already existed before your PR
- Unrelated improvements you noticed but did not change ("noticed-but-didn't-fix" is a fine comment to leave on the PR)
- Personal style preferences not codified in this document

---

## What we need help with most

Sorted roughly by current impact:

1. **Apple Silicon testing and Cranelift NEON dispatch** — Phase G prep work
2. **Out-of-core training data** — `r2-train/src/tokens.rs` (memmap) and `r2-arrow/src/parquet_io.rs` both exist, but nothing in the training path uses them, so a corpus must fit in RAM. Wiring them up is self-contained and high value
3. **Coverage of remaining R idioms** — S4 dispatch, R5 reference classes, more of the long tail of CRAN-style helpers
4. **Documentation and examples** — every `pub fn` deserves a doc comment with a runnable snippet
5. **Performance benchmarks across more hardware classes** — Raspberry Pi, AWS Graviton, M2/M3, EPYC, Threadripper
6. **`r2-pkg` online package registry** — script packages already install from a local dir, a `.zip`, or a GitHub `user/repo`, and export functions, types and methods; `install.packages(name)` with no path is what's missing
7. **The `r2-calculus` and `r2-symbolic` libraries on the roadmap** — green-field, designed to be modular, ideal for a small focused contributor team

> Deep learning is **not** a bindings project here. R2 has its own stack in
> pure Rust — `r2-tensor`, `r2-autograd`, `r2-train` — and trains a
> transformer at PyTorch's speed (`benchmarks/llm/REPORT.md`). An earlier
> version of this list proposed binding to `candle`; that would reintroduce
> exactly the glue layer R2 exists to remove.

---

## Reporting bugs

Open an issue using the **Bug report** template. Include:

- Ardon-R2 version (`version()` from the REPL)
- OS and architecture
- Minimal reproducing code
- Expected output (what R 4.5.3 would say, if relevant)
- Actual output

Bugs that affect numerical accuracy vs R are the highest priority and will be looked at within 48 hours.

---

## Asking questions

If a question is too open-ended for an issue, open a **Discussion** on GitHub instead. Examples:

- "Is there a planned migration path from CRAN package X to Ardon-R2?"
- "What's the architectural reasoning behind the four-layer split?"
- "How should I structure my new domain crate?"

Discussions are the right venue for design conversations. Issues are the right venue for "this is broken" or "this should exist."

---

Thank you for reading this far. Now go write something good.
