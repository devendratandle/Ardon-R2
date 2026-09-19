# Changelog

Version-by-version record of what became **available** or got **fixed** for
users. It describes capabilities, not internal implementation — algorithm
choices and refactors live in the code and `docs/ARCHITECTURE.md`.

---

## v0.4.0 (September 2026)

**Train a real model and compare it, in two commands.** `cargo run
--release -p r2-train --example tinystories_train` trains a 7.24M-parameter
model on 19.4 MB of TinyStories, saves it, and generates from it;
`python benchmarks/llm/tinystories_train.py` trains the same model in
PyTorch and prints one joined table of speed and accuracy. The two halves
hold three things identical — the token stream, the initial weights, and
the batch order — so the only thing left free is the arithmetic and how
fast each library does it. It gates itself: both sides evaluate held-out
loss from the same weights before training, and the comparison aborts if
those disagree by more than f32 rounding.

Result, machine at full clock, two interleaved pairs at 300 steps /
614,400 tokens (2026-09-19): **R2 trains 1.15-1.16x FASTER than PyTorch
2.13+MKL and learns identically** — the training loss matches to four
decimals at every one of eleven checkpoints, held-out loss 4.3113 vs
4.3114, and both models generate the same sentence. The pipeline
(tokenize + train) is 1.25x faster; tokenizing alone is 10x. Forward is
1.48x faster and backward 1.41x, measured phase by phase. Full numbers in
`benchmarks/llm/REPORT.md`. (v0.3.9 was 1.05x behind; the session that
produced this release moved it by removing work — see below.)

- **Corrected: the "R2 1.71x faster end to end" claim was a short-run
  artefact.** It came from a 30-step arm, where training is ~21 s against
  ~3 s of tokenizing and an 11x tokenizer advantage carries the total. At
  500 steps tokenizing is 0.5% of the pipeline and the total is level
  (1.02x, one pair each way). Same code, different run length. The
  training ratio is the number to quote.
- **Fixed: `Tokenizer::to_tokenizer_json` wrote files HuggingFace refuses
  to load.** The ByteLevel sections omitted `trim_offsets` and
  `use_regex`, which are not optional in that schema — `tokenizers`
  rejects the file outright with "missing field `trim_offsets`". R2's
  export was unusable by the ecosystem it exists to interoperate with.
  Both fields are now written, and a shared vocabulary round-trips: R2 and
  HuggingFace encode the same prompt to the same ids and the two models
  generate byte-identical text.
- **New: `Trainer::eval_loss`** — one forward pass, no gradients, no
  optimizer. `train_step`'s loss is what the model had *before* that
  step's update on the batch it was about to fit, which measures
  memorisation of the training stream; held-out loss is the number that
  compares two implementations.
- **Fixed: `installer/R2.iss` still declared 0.3.9.** It ships both
  `r2.exe` and `R2Gui.exe`, so a release cut from it would have packaged
  binaries stamped with two different versions.

**Three pieces of work a training step was doing for no reason.** Found by
re-running `--example step_census` rather than reasoning about it: the top
three items after the GEMM were not arithmetic at all, and none of them is
something PyTorch does. Measured separately, in interleaved pairs at 60
steps, with the loss trajectory bit-identical in every run.

- **`backward()` no longer memsets gradient buffers that are already
  zero** — **8.0% of a training step.** `Tape::push` allocates every
  gradient buffer with `vec![0.0; n]` and `train_step` builds a fresh tape
  each step, so the blanket reset at the top of `backward()` was writing
  zeros over zeros: 71.86M elements, 287 MB, 27.7 ms, every step. Worse
  than the write, `fill` TOUCHES every page, forcing resident what the
  allocator had handed out as untouched — including the buffers of
  `requires = false` nodes that no backward ever writes. A second backward
  on the same tape still resets, because that one has real gradients to
  clear. (Threading this was measured *worse* in v0.3.9; that was a correct
  measurement of the wrong fix.)
- **Adam updates the parameter blocks in place, with an AVX2 kernel** —
  **9.6% together with the item below.** Reaching the flat `Adam::step`
  meant building a 29 MB gradient buffer with a division per element,
  copying all 7.24M parameters into a second 29 MB buffer, and copying them
  back: ~116 MB of traffic per step to satisfy an API shape. The new
  `Adam::step_blocks` walks the blocks against a running offset into
  `m`/`v` instead, and folds the gradient scaling into the update. The
  kernel is bit-identical to the scalar form — `div` and `sqrt`, not
  reciprocals; separate multiply-add, not FMA — and a test asserts that
  against the original `Adam::step` over five steps and six block shapes.
  It earned its place immediately: the first version associated
  `(1-b2)*g*g` as `(1-b2)*(g*g)` where the scalar form evaluates
  `((1-b2)*g)*g`, and the last bit differed.
- **The tape borrows the weights instead of cloning them.** Parameters are
  moved onto the tape and returned by a new `Tape::take_value`, which
  leaves the node's gradient intact. A forward pass only reads them, so the
  29 MB copy in and 29 MB copy out were pure overhead — 11.4 ms, 2.0% of a
  step. PyTorch does not copy parameters into its graph either.
- **Fixed: the benchmark crashed on a reused token file.** With tokens
  cached, R2's `tokenize_s` is zero and the Python half divided by it. It
  now reports "reused" and declines to print a pipeline ratio at all —
  scoring R2's zero against PyTorch's full tokenizing pass would credit R2
  for work it had simply already done.

**Out-of-core training data — a corpus no longer has to fit in RAM.**
`tinystories_train` now tokenizes straight to a file and trains from a
memory map, so neither the corpus text nor the id stream is ever fully
resident: 96 bytes of mapping against 37.7 MB as a `Vec<usize>`. The old
path cost roughly four times the corpus before a step ran (text + `u32`
ids + `usize` ids), which is ~76 MB at 19.4 MB and ~2.7 GB at 700 MB — the
point where a run simply does not start.

- **Tokenizing is streamed and chunk boundaries are correct.** Chunks are
  cut on a GPT-2 **pre-token** boundary, because merges never cross one.
  Cutting on newlines instead looks safe and is not: measured over 3 MB at
  64 KB chunks it produced different ids from whole-text encoding
  (`[1293, 400]` versus `[10, 470]` for identical text), since a newline
  plus the following space is a single pre-token. The streamed pass now
  reproduces the whole-corpus token count exactly — 4,615,591 either way —
  and a new test,
  `chunked_encoding_matches_whole_text_on_pretoken_boundaries`, pins the
  property.
- **Tokenizing happens once per corpus.** A stamp records the corpus path,
  size, split point and tokenizer settings; a later run maps the existing
  token files and skips both BPE training and encoding (3.15 s -> 0 s).
- **Rejected after measurement: split-K in `sgemm`.** Parallelising the
  depth loop with per-worker accumulators, to cut the eight fork-joins a
  `grad_B` multiply makes down to one. Four interleaved pairs put it inside
  noise (mean 2.8%, and the three pairs after a first-run outlier were
  +2.4%, +1.0%, -0.7%). The premise was also wrong: the block count is
  already 6-84, so `grad_B` was never starved of tasks, and removing seven
  of its eight barriers changed nothing measurable — the shortfall is
  memory bandwidth. Reverted; recorded in `benchmarks/llm/REPORT.md` so it
  is not retried.

**Data frames and joins — three silent-wrong-answer bugs.** All found by
re-testing `docs/KNOWN_LIMITATIONS.md` against the build instead of
trusting it, and all now covered by
`tests/differential/cases/merge_joins.R`, which diffs R2 against GNU R.

- **`merge()` is now a real join.** It supported one key column and
  **silently ignored `all.x` / `all.y` / `all`** — asking for an outer join
  returned the inner join, with no warning and no error, just fewer rows.
  It also read only the first element of `by`, so `by = c("k1","k2")`
  joined on `k1` and duplicated `k2` as a `.y` column; suffixed only the
  right-hand side of a name collision where R suffixes both (`v.x`/`v.y`);
  returned rows in match order where R sorts by the key; and rebuilt every
  column by formatting it to a string and re-parsing, which turned a
  character column of digits (`"007"`) into a number. Now: composite keys,
  `by.x`/`by.y`, all four join types with NA fill, `.x`/`.y` suffixes,
  key-ordered output (`sort=`), and type-preserving columns. The right
  frame is indexed once rather than scanned per left row — O(n+m) instead
  of O(n x m).
- **`data.frame(stringsAsFactors = FALSE)` added a column called
  `stringsAsFactors`.** The construction flags were treated as data, so a
  frame written by any pre-R-4.0 script silently carried one extra logical
  column into every `ncol`, `names`, column loop and join downstream.
  `stringsAsFactors`, `check.names`, `check.rows` and `fix.empty.names` are
  now consumed, and `row.names =` sets the row names instead of becoming a
  column.
- **`data.frame()` now recycles short columns**, as R does.
  `data.frame(k = 1:3, g = "x")` built a frame whose `nrow()` said 3 while
  `g` held one element; every row-wise read past the end quietly produced
  NA. A length that does not divide evenly is still left alone rather than
  half-filled, so a genuinely ragged column stays visible.
- **`lm(y ~ x1 + x2)` without `data=` fitted ONE predictor.** A bare
  formula evaluated its right-hand side as arithmetic, so `x1 + x2` became
  the elementwise sum and the model was `y ~ (x1 + x2)` — two coefficients
  printed where three were asked for, with no warning. The `data=` path
  split terms correctly; the bare path now goes through the same resolver.
- **Standard errors in `lm` and `glm` now come from the QR factor.**
  Coefficients were solved by Householder QR (condition number of X), but
  the standard errors were then computed by forming X'X and inverting it
  (condition number of X, squared) — the accuracy the QR was paid for was
  thrown away in the one column of `summary()` people read. `(X'X)^-1` is
  now `R^-1 R^-T` from the same factorisation, R's `chol2inv(qr.R)`; the
  IRLS steps in `glm` solve `W^(1/2) X` by QR as `glm.fit` does. New
  differential case `lm_ill_conditioned`: on a design with kappa(X'X) of
  1.6e11 the old route was off by 1.2e-6 relative; the new one agrees with
  GNU R to 1e-9.
- **`confint()` returned NULL and used one standard error for every
  coefficient.** It printed `coef +/- z * sigma/sqrt(df)` — the same width
  on every row, with a normal quantile — and returned nothing, so
  `confint(fit)[2, 1]` failed. For `lm(mpg ~ wt + hp, mtcars)` the `hp`
  row read `[-0.98, 0.91]` against R's `[-0.050, -0.013]`. It now returns
  the p x 2 matrix from each coefficient's own standard error with the
  Student-t quantile on the residual df (`lm`) or the normal quantile
  (`glm`, R's `confint.default`).
- **`cor.test()` misaligned pairs after an NA and used the wrong
  distribution.** NAs were dropped from `x` and `y` independently, so one
  NA in `x` shifted every later pair and correlated `x[i]` with `y[i+1]`;
  the p-value came from the normal distribution rather than Student-t on
  n-2 df (at n=10 that halves the p-value). Rows are now dropped pairwise
  and the p-value matches R to seven digits.
- **`cor(x, y)` and `cov(x, y)` silently truncated to the shorter
  vector.** `cor(1:4, 1:3)` returned 1 where R errors "incompatible
  dimensions". Both now raise that error.
- **`Sys.time()` returned a bare number.** A second registration in the
  `utils` layer masked the POSIXct one in `base`, so `class(Sys.time())`
  was `"numeric"` and `format(Sys.time())` printed epoch seconds. The
  duplicate is removed and a test now fails the build if any builtin is
  registered in two layers (it also caught `clear`/`cls`, harmless
  duplicates in `core` and `utils`).
- **The binary contained C.** The Parquet reader's default codec set
  pulled `zstd-sys`, compiled from bundled C, while the README promised
  none anywhere in the stack. The zstd codec is dropped — snappy, gzip,
  lz4 and brotli are all pure Rust and remain — and `cargo tree
  --workspace -i cc` now prints nothing. A zstd-compressed Parquet file
  fails to read with the codec named; recompress with snappy.
- **The tape recycles its value buffers between steps.** Returning
  ~3,000 value and gradient buffers (~287 MB) to the allocator at the end
  of every step measured 38-43 ms — 7.7% of a step, more than Adam
  (10.7 ms) and rmsnorm together — and the next step page-faulted the
  same sizes back in. No census had a line for either, because neither
  is an op. Every forward op writes its whole output, so value buffers
  now pass from one step's tape to the next by exact length and are
  reused without zeroing (`sgemm_assign_into` lets the output-head GEMM
  write into a recycled 64 MB buffer directly); gradient buffers stay
  freshly allocated and are freed off the training thread. Tape drop
  43 -> 0.1 ms, forward 153 -> 146 ms; 300-step training 147.8 -> 141.5 s
  and 150.1 -> 140.6 s in two interleaved pairs, loss curve identical.
  The comparison moved from 1.10x to 1.19-1.23x across this and the
  earlier off-thread free. Adam itself is 2.7x faster than PyTorch's.
- **`read.csv` header names are now valid names, as in R.** A header
  `a b,c-d,1x` produced columns named `a b`, `c-d`, `1x`, reachable only
  with backticks or `d[["a b"]]`. R applies `make.names` unless
  `check.names = FALSE`: the same file now gives `a.b`, `c.d`, `X1x`, and
  `d$a.b` works. Duplicate headers get `.1`, `.2`. `check.names = FALSE`
  keeps the raw header; `make.names(x, unique=)` is callable directly.
  Differential case `csv_names` matches GNU R on every rule.
- **`?help` covers every builtin: 438 of 438**, up from 37 at v0.3.9.
  `FUNCTIONS.md` is the single source (embedded at build time and parsed
  on first use), and 153 functions that were registered but never
  documented — `sin`, `sort`, `seq`, `rep`, `which`, `nrow`, `stop`,
  every `d/p/q` distribution function, the `llm.*`, `mem.*`, `gpu.*`,
  `roll*`, `apply.*` and Hindu-calendar families, the package installers
  — now have entries. The parser reads column layouts, alias chains of
  any length, nested parentheses in signatures, and a description on the
  line below a long signature.
- **`docs/MISSING_FUNCTIONS.md` and `docs/R2_SESSION_B_MULTI_DEVICE.md`
  are retired** — 15 of the 16 functions on the first roadmap shipped
  (`ecdf` is now listed in `KNOWN_LIMITATIONS.md`) and the multi-device
  session is done.
- **`llms.txt` described v0.3.3** — 320 functions, ~25 crates, a ~5 MB
  binary. Now 438 (read from the registration table), 29 crates, ~15 MB,
  and the LLM stack is listed. `FUNCTIONS.md`'s header count corrected
  the same way.
- **The `R2_BLAS` DLL mechanism is removed.** v0.2.1 planned per-CPU
  builds of the kernel (`r2_linalg_avx2.dll` ...) chosen by the installer
  and loaded at runtime. Runtime `is_x86_feature_detected!` dispatch
  inside one binary — which every hot kernel now uses — measured faster
  than a `-C target-cpu=native` build, needs no loader, and is the only
  form a certification review can qualify. With it go `libloading`, the
  `cdylib` crate type and the workspace's only `dlopen`; `r2.exe` now has
  no FFI of any kind. `docs/BLAS_DISPATCH.md` records the reasoning.
- **`unsafe` is down from 85 sites to 54 in non-test code, and out of the engine
  entirely.** The JIT handle's entry points take slices and check every
  length and the compiled kind before the call, so the twelve `unsafe`
  blocks in `r2-engine` that existed only to pass raw pointers are gone
  (12 → 0); the externs compiled code calls back into go through two
  slice helpers instead of five `from_raw_parts`; and the seven SIMD
  wrapper kernels whose bodies are ordinary Rust are now safe functions
  (the gate that calls them keeps its `unsafe` and its proof). Machine
  code and every bit-identity test unchanged.

## v0.3.9 (September 2026)

**LLM training performance — the LMO queue.** Numbers are the 30-step BPE arm
(19.4 MB TinyStories, 30 x 32 x 64, vocab 8,000, dim 256 x 4 layers) unless
stated; every comparison was taken against PyTorch 2.13.0+cpu and JAX 0.11.1
in the same window as the R2 run.

- **LMO-14 `grad_B` was serial and is now threaded.** It was the one gradient
  `Op::MatMul`'s backward never parallelised, and the output head's alone
  measured 1,999 ms of a 5,540 ms step. Partitioned over grad_B's own rows —
  every `i2` writes every row, so splitting `i2` would race — keeping ascending
  accumulation order, hence bit-identical results.
- **LMO-15 / LMO-5: `Op::Attention`, a fused batched attention op.** Attention
  was built per sequence per head: slice, transpose, matmul, mask, softmax,
  matmul, concat, 128 times at the shipping shape — 1,156 tape nodes and 2,312
  Vec allocations per block. Measured, the two matmuls were 4.4% of the block
  and slicing alone 47%; it ran at 5.7 GFLOP/s against R2's own GEMM at 66-76.
  The fused op is one node: no slices, no transposes, no materialised score
  matrix, softmax over `j <= i` only, and the backward recomputes the
  probabilities rather than storing a quadratic buffer.
- **`Tape::backward_from(var, seed)`** — PyTorch's `.backward(g)`. Lets a
  benchmark differentiate an op from a supplied gradient instead of inventing a
  scalar loss, which is what had made LMO-1's fwd+bwd comparison invalid.
- **`requires` guards on MatMul/Mul/Add backward.** The one-hot embedding form
  was computing an 8.4 GFLOP `grad_A` into a leaf that needs no gradient; the
  one-hot fwd+bwd went 658 ms -> 47 ms at vocab 8,000. `Op::Mul`'s backward no
  longer clones both operands.
- **Embedding gather and scatter-add threaded** (`Op::Embed`), and
  `transformer.rs` switched off the one-hot form onto the gather.
- **Fixed: `continue` in the GPU matmul-backward branch** skipped the
  gradient-buffer hand-back, leaving that node's gradient an empty `Vec` — a
  second `backward()` read zero out of it, silently.
- **Fixed: the float64 accuracy gate was red on a clean tree.**
  `benchmarks/llm/accuracy_check.py` implemented SPLIT-HALF RoPE while
  `ops::rope_inplace` is INTERLEAVED, and read the output head as
  `view(vocab,dim).T` when it is stored row-major `(dim, vocab)`. Uniform ~1.35
  relative error on all 21 blocks; now ~7e-7.
- **Benchmarks now print one joint table with a verdict.** The R2 half writes
  `lmoN_r2.json`; the Python half joins it and reports R2 against PyTorch and
  JAX per row. An R2-versus-R2 speedup is not a result.

## v0.3.8 (July 2026)

**New — `r2 --self-check`, verify accuracy on your own machine.** A
self-contained battery of 21 checks (descriptive statistics, elementary
functions, the distribution surface including the far normal tail,
combinatorics, linear algebra, regression) compares this build against
authoritative mathematical constants — no R install and no network
needed. It prints a one-page report and exits non-zero if anything
diverges, so it also works as a post-install or deployment gate.

**Fixed — element-wise math kept matrix and array shape.** `sqrt()`,
`exp()`, `log()`, the trig family, `floor`/`ceiling`/`round`/`signif`,
`gamma`/`lgamma` and friends returned a flat vector when given a matrix
(so `sqrt(t(x) %*% x)` lost its dimensions). As in R's Math group, they
now preserve the input's structure: a matrix in yields a matrix out, an
array keeps its `dim`, and a named vector keeps its names.

**GUI — rebuilt for long sessions.**

*Colour and contrast.* Colours were being displayed lighter than the
theme specified, which is what made the interface look washed out and the
text low-contrast. Colour handling is now correct end to end, so what a
theme specifies is what appears on screen — and text is markedly crisper
at every size and DPI.

*A dark interface, now the default.* Ardon-R2 ships a dark theme built
for hours in front of the screen: body text sits around 12:1 contrast
(past the WCAG AAA bar), with the familiar scarlet prompt kept and
distinct colours for input, output, errors and the banner. A refined
light theme is included, and the classic khaki and RGui looks remain
available. Glyph rendering adapts to whichever theme is active, so text
never looks thin on a light background or bloomed on a dark one. Plots
stay on a white canvas in every theme — a plot is a document, not
chrome.

*Layout that needs no window juggling.* The window opens maximized with
the console on the left and graphics windows opening beside it, so a plot
and the command that produced it are visible together — no switching, no
overlap. Long output lines wrap to the console width instead of running
past the right edge, so reading a wide result no longer means resizing
the window or scrolling sideways.

*Console behaviour matches the CLI.* Engine errors are shown in their
normal form (no internal debug text), warnings appear after a command
instead of being silently dropped, and both frontends share one startup
banner.

**Improved — `explain()` closes the loop.** When a function falls back to
the interpreter, `explain()` now names the remedy, not just the blocker.
For data it reports where the serial-to-parallel crossover lies on the
current machine (for example "serial — parallel at n>=524288"), and
`explain(1e8)` answers for a hypothetical vector length without
allocating it.

**Accuracy — normal CDF upgraded to full double precision.** `pnorm`,
`erf`, and `erfc` now use Cody's rational-Chebyshev algorithm (~1e-16, the
method R and Boost use) instead of the Abramowitz-Stegun polynomial
(~1.5e-7). The normal CDF and every p-value built on it (t/z tests, `lm`,
mixed models) now match CRAN R to 14-16 significant figures, up from ~7.
`erfc` is used on the tails to avoid the `1 - erf` cancellation.

**New — mutable environments (R's environment semantics).** Environments
are now live, shared objects rather than snapshots. Everything built on
them works as in R: stateful closures (`make_counter()` factories,
accumulators), `<<-` from any nesting depth using R's lexical rule
(rebind in the nearest enclosing frame that has the name, else global),
independent per-call factory environments, and the new `local(expr)`.
As a direct consequence the interpreter got dramatically faster — the old
copy-on-write model cloned the global binding table on every top-level
assignment: top-level loops run ~31× faster, function calls ~18× faster
(release, measured on the same machine).

**New — agent JSON mode.** `r2 --json` runs a newline-delimited JSON
protocol on stdin/stdout ({"expr": "<R code>"} in; ok/class/length/
result/output or error out), with session state persisting across
requests — agent frameworks and other programs drive the engine without
screen-scraping a REPL.

**New — the interactive plot viewer works from the CLI.** dev.view() in
the REPL launches the browser viewer (daemon HTTP server; auto-refreshes
on plot(); prompt stays responsive). Scripts can opt in with
R2_AUTOVIEW=1 (keep the process alive with Sys.sleep).

**New — JIT compiles indexed-loop math (J.5).** Textbook loops like
`for (i in 1:n) s <- s + (x[i] - mean(x))^2` now JIT-compile by
normalizing to their vector form, reusing the same SIMD kernels as the
one-line formulas — nested loops inside iterative kernels and two-vector
dot/weighted-sum loops included. Scalar recurrences and anything outside
the guarded shape still fall back safely.

**New — R-faithful default arguments + `missing()`.** Defaults evaluate
in the function's own environment, so they can reference other arguments
and chain in declaration order (`function(n, m = n + 1, p = m * 2)`), as
in R. `missing(arg)` reports whether a parameter was supplied in the
call.

**New — differential-vs-R correctness harness.** `tests/differential/run.sh`
executes a battery of R scripts under both Ardon-R2 and GNU R and compares
their outputs numerically. Any semantic divergence from R now fails a test
before it can ship. Nine case files cover matrix arithmetic and metadata,
indexing, statistics, lm/glm, data frames, strings, control flow, and
numeric edge semantics.

**Fixed — R-compatibility batch (found by the harness on its first run):**
- `v[["name"]]` extracts from a named vector (previously an error).
- `is.nan()`, `is.infinite()`, `is.finite()` added; `is.na()` is TRUE for
  NaN and works elementwise on every atomic type.
- `summary(fit)` returns the summary object, so `s <- summary(fit);
  s$r.squared` works (printing unchanged).
- `scale(x)` accepts a plain numeric vector (promoted to n×1, as in R).
- Division by zero follows IEEE/R semantics: `1/0` is `Inf`, `0/0` is
  `NaN` — no longer a runtime error.
- `%%` is floored modulo like R (`-7 %% 3` is `2`, was `-1`), across the
  scalar, fused, and columnar paths.
- `round()` rounds half to even (banker's rounding): `round(2.5)` is `2`.
- Recycling is silent when the longer length is a clean multiple of the
  shorter (`c(1,2,3,4) + c(10,20)` works).
- `glm(..., family = binomial)` accepts the bare family function.
- `deviance(model)` added for lm/glm.
- Numbers print with 7 significant digits like R (values below 1 with
  leading zeros previously lost a digit).
- `colnames()` / `rownames()` getters work on matrices (previously NULL
  even when names were set).

## v0.3.7 (July 2026)

**New — the JIT compiles whole iterative algorithms.** Plain R2 source
functions carrying scalar, vector, and matrix state across `for`/`while`
loops compile to one native unit — gradient descent, Newton, EM,
fixed-point, shrinkage, and multi-parameter GD/IRLS with `X %*% b` /
`t(X) %*% r`. Measured vs interpreted (release): matrix-state GD **32×**,
vector-state iteration up to ~64×, scalar-state training loops ~3×.
Anything outside the compiled subset (mis-typed calls, read-before-define,
unsupported shapes) falls back to the interpreter — never a wrong answer.

**New — statistics formulas reach/beat native.** `var`/`cov`/`cor`/`sd`/
z-score/RMSE/R² written as one-line R2 formulas JIT with common-
subexpression elimination, single-pass SIMD wave fusion, and user-helper
inlining; `cor` written in R2 source outruns the native builtin.
`explain(f)` reports whether a function compiled and, if not, exactly why;
`explain(x)` reports data size, architecture, and the serial-vs-parallel
plan.

**New — parallel apply for library code.** `mclapply` / `par.lapply` /
`par.sapply` run pure closures across cores with isolated workers and
per-worker reproducible RNG (~4.5× on 6 cores; the r2sem PLS-SEM source
library runs ~13× faster than R's cSEM on identical estimates). Composes
with the JIT.

**New — linear algebra.** Non-symmetric `eigen()` (complex spectra reported
honestly via `$imaginary`), `backsolve`/`forwardsolve`, `rcond`, `kappa`.
Matrix multiply routes small/thin shapes to a fast path (5×5 `%*%` ~34×
faster; thin regression shapes like `X %*% w` benefit throughout).

**GUI (verified live on screen).** Console selection is a solid blue band
with white inverted text; warm paper background, deeper ink, and scarlet
input for long-session readability; the console snaps back to the prompt
when output arrives (previously the returning prompt could stay hidden
after scrolling up or right); platform-appropriate monospace fonts on
Windows/macOS/Linux with a directory-scan fallback; plot y-axis labels and
frame no longer clip at the default device size.

## v0.3.6 (July 2026)

**Fixed**
- `factor(x, levels = c(...))` now honours the given levels (set and order);
  previously the levels argument was ignored and values were sorted
  alphabetically, producing wrong codes.
- Random generators take positional parameters: `rnorm(n, 10, 3)`,
  `runif(n, 5, 15)`, `rexp`, `rbinom`, `rpois` — these silently dropped the
  parameters before and returned defaults.
- JIT no longer mis-runs a `for`-loop accumulator inside a function
  (`function(n){ s<-0; for(k in 1:n) s<-s+k; s }` now returns the sum, not 0).
- `abline(lm(y ~ x))` draws the actual fitted line (intercept/slope read from
  the model) instead of a 45° `y = x` line.
- `seq(from, to, length.out = n)` returns exactly `n` evenly spaced points;
  previously `length.out` was ignored and it fell back to `by = 1`
  (`seq(0, 2*pi, length.out = 100)` gave 7 points, which broke `matplot` curves).
- `pairs()` draws axis tick labels on the matrix's outer edges (alternating
  sides); `barplot`/`boxplot` no longer overlay numeric ticks on the
  categorical x-axis.

**Graphics**
- Shared graphical parameters now behave like R's `par()` across **every**
  plot type (`plot`/`hist`/`matplot`/`barplot`/`boxplot`/`pairs`): `las`
  (axis-label rotation 0/1/2/3), `col.axis`/`cex.axis`, and per-call `mar=`
  or `par(mar=)` margins — each with a tuned default. Category labels on
  bar/box plots rotate with `las` (horizontal default, vertical on `las=2/3`).

**Performance**
- Element-wise arithmetic (`a + b`, `-`, `*`, `/`) on numeric vectors is now
  zero-copy on the columnar path — at parity with, and in repeated use
  slightly faster than, R. This was R2's one remaining slower-than-R workload.

**Added**
- **Type methods & inheritance.** `method name(x: Type) …` now actually
  dispatches — `name(obj)` runs the method registered for `obj`'s type,
  including methods inherited via `type B extends A`. Inheriting types also
  inherit their parent's fields. (Previously `method` parsed but calling it
  gave "object not found".)
- **Addon packages can export types and methods**, not just functions —
  `library()` of a `type`+`method` package works, and `detach()` cleans them
  up.
- **Date rendering.** `format(d, fmt)` / `strftime` / `as.character()` print
  `Date`/`POSIXct` as calendar strings (not the raw day/second count),
  `c()` keeps the date class, and `class()` reports `"Date"`.
- Linux install guide and one-shot installer for the CLI and GUI
  (`INSTALL_LINUX.md`, `scripts/install-linux.sh`).

---

## v0.3.3 (June 2026)

**Available**
- Metaprogramming: `quote`, `eval`, `parse`, `deparse`, `call`/`as.call`,
  `body`/`formals`/`args`, `substitute`, `match.call`/`sys.call`, `bquote` —
  code is data and back. (Arguments are still eagerly evaluated; no lazy
  promises yet.)
- `isTRUE`/`isFALSE`/`identical`/`all.equal`/`diag`/`toString`; operators
  usable as functions (`Reduce(\`+\`, x)`); `repeat { }` loops.

**Fixed**
- Factors: `factor[i]`, `factor == "level"`, factor columns under
  `df[mask, ]`, `as.numeric(factor)`, and `tapply`/`aggregate` by a factor.
- `sprintf` full `%[flags][width][.precision]` specs; `ifelse` keeps the
  branch type; `pdf()` writes one page per plot.
- Negative (exclusion) indexing on vectors, matrices, and data frames;
  `strsplit()` returns a list; vectorized `paste`/`substr`; `as.numeric("…")`.
- Replacement functions `names<-`/`colnames<-`/`rownames<-`; `rm()` of
  multiple names; `T`/`F` reassignable; variables may be named `c`/`t`/`df`.
- Top-level `for`/`while` no longer read a loop variable one iteration stale.

---

## v0.3.2 (June 2026)

**Available**
- Graphics: `pairs()`, `pie()`, `matplot()`, `curve()`; overlays `text()`,
  `title()`, `axis()`, `rect()`; plot params `col`/`cex`/`pch`/`type`/`lwd`;
  `pdf()`/`png()`/`svg()` file devices.
- ~85 more base-R functions: `seq_len`/`seq_along`/`%in%`/`setdiff`/`union`/
  `intersect`/`unlist`/`split`/`cut`/`pmin`/`pmax`; `Reduce`/`Filter`/`Map`;
  `switch`/`with`/`tryCatch`/`stopifnot`; `attr`/`attributes`/`structure`/
  `format`/`inherits`; the `dexp/pexp/qexp`, `dbinom/…`, `dpois/…`, `dt/…`,
  `dchisq/…`, `pf/qf` distribution families; `uniroot`/`integrate`/`optimize`.

**Fixed**
- `...` (dots) are captured and forwarded into inner calls; variadic
  `sum`/`min`/`max`/`prod`; `[[ ]]` read indexing; data-frame column
  iteration in `lapply`/`sapply`/`for`.

---

## v0.3.1 (June 2026)

**GUI**
- Resolution-adaptive UI (720p → 4K) with edge/corner resize cursors and a
  legible title-bar logo.

---

## v0.3.0 (June 2026)

**Available**
- Out-of-core compute on larger-than-RAM data: `mmap.csv()` streams a CSV to
  per-column sidecars; `mmap.lm()` fits least squares in one streaming pass;
  `mmap.map()` transforms out-of-core; streaming `sd`/`var`/`prod`/`range`
  and approximate `median`/`quantile` over mmapped columns.
- `read.parquet()` — pure-Rust Parquet import.
- Hardware-aware matrix multiply: multi-core + runtime AVX2/AVX-512 — about
  14× faster than default R on `%*%`, results identical.

**Fixed**
- `na.rm = TRUE` honoured across `sum`/`mean`/`min`/`max`/`prod`/`var`/`sd`/
  `median`; `names()` works on lists; `lm`/`glm`/`aov` accept the positional
  `data` argument (`lm(y ~ x, df)`).

---

## v0.2.2 (June 2026)

**Available**
- `solve()` (inverse / linear solve) and `det()` exposed as functions.

**Fixed**
- Statistical output (`t.test`, `chisq.test`, `aov`, `manova`, `wilcox.test`,
  model summaries, ML output) now appears in the desktop GUI console, not
  only the CLI.
- Numerical accuracy now matches R to ~1e-9: `lm` (stable least squares),
  `qnorm`, and every t / F / ANOVA / MANOVA / Hotelling p-value and CI.
- `aggregate()` accepts the formula interface, including multi-term and
  `cbind(...)` multi-response formulas, with real source column names.

---

## v0.2.1 (June 2026)

**Available**
- Runtime-swappable BLAS: matrix multiply can dispatch at runtime to a
  CPU-specialised build of the same pure-Rust kernel via `R2_BLAS`, falling
  back to the built-in kernel. Stays strictly pure-Rust.

---

## v0.2.0 (June 2026)

**Available**
- Native desktop GUI (`R2Gui`): an MDI workspace with floating console and
  graphics windows (drag/resize), replacing the CLI-only build.
- Multivariate statistics: `manova()` (Wilks / Pillai / Hotelling-Lawley /
  Roy), `hotelling.test()` (one-/two-sample/paired), and repeated-measures
  `aov(y ~ x + Error(subject))`.

---

## v0.1.1 (May 2026)

**Available**
- In-memory graphics device with full `par()` (multi-panel `mfrow`/`mfcol`,
  margins, `pch`/`lty`/`lwd`/`col`) and a built-in browser plot viewer
  (`dev.view()`) with a session gallery.
- `.r2d` native binary dataset format; formula data scope + factor expansion
  in `lm()`/`glm()`; opt-in single-precision (f32) storage.
- Wider JIT coverage for user closures (math calls, 2-argument and
  branchy bodies, fused map-reduce).

---

## v0.1.0 (May 2026) — first stable release

**Available**
- A Cranelift JIT that compiles pure-arithmetic user functions to native
  code (scalar, vector maps, reductions, branchy and composed bodies), with
  a central scheduler choosing serial vs. parallel execution.
- Columnar numeric storage with dense fast-path reductions; `cumsum`/
  `cumprod`/`cummax`/`cummin`, `which.max`/`which.min`, rolling
  `sum`/`mean`/`max`/`min`/`sd`, hash-based group aggregation, and
  Euclidean/Manhattan/cosine distances.
- Real `svd()`, `eigen()` (eigenvalues **and** eigenvectors), and QR;
  `prcomp()$rotation` is genuine.
- R-faithful hypothesis tests (Welch t-test, formula and paired forms, exact
  Fisher); RFC-4180 CSV parsing; regular expressions.

---

## v0.0.9 (April 2026) — initial release

**Available**
- Core language: vectors, data frames, formulas, R-style assignment and
  1-based indexing.
- Statistics: `lm`, `glm`, `t.test`, `aov`, `shapiro.test`, `cor.test`.
- Machine learning, all built in: decision tree, random forest, gradient
  boosting, KNN, PCA, K-means, naive Bayes (12 algorithms).
- Math kernel: BLAS-style operations, matrix decompositions, SVD,
  eigenvalues.
- Data handling: CSV read, filter/select/mutate/arrange.
