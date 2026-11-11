# Changelog

Version-by-version record of what became **available** or got **fixed** for
users. It describes capabilities, not internal implementation — algorithm
choices and refactors live in the code and `docs/ARCHITECTURE.md`.

---

## v0.4.1 (November 2026)

Everything since v0.4.0. (The code carried the number 0.4.2 for a while during
development; that number was never released.)

### All of R's datasets, built in — 2026-11-11

- **The whole `datasets` package: 108 objects**, up from 5 — `CO2`,
  `PlantGrowth`, `ChickWeight`, `cars`, `Titanic`, `AirPassengers`,
  `volcano`, `state.x77`, `penguins` and the rest. They are inside the
  executable, so a bare `r2` binary has them as well as an installed one.
- **The values are R's own.** R writes them out
  (`crates/r2-base/datasets/export_datasets.R`), nothing is copied by hand,
  and a comparison case checks every dataset against R: sizes, the sum and
  NA count of every numeric column and vector, factor levels.
- **Two built-in datasets were wrong and are fixed:** `airquality` had only
  its first 30 of 153 rows, and `ToothGrowth$supp` was character, not a
  factor.
- **`data()` works as in R.** `data()` lists every dataset with its title,
  `data(CO2)` puts a fresh copy in the workspace, and an unknown name warns.
  `help("CO2")` gives the title and shape.
- **`letters`, `LETTERS`, `month.name` and `month.abb`** exist.
- **`head()` and `tail()` fixed.** `head()` of a data frame did not cut
  factor columns: `head(iris, 3)$Species` was still 150 long, and
  `head(CO2)` crashed. It also dropped row names (`head(mtcars)` lost the
  car names). `tail()` handled only numeric vectors. Both now cut every
  column, keep row names, dimnames, a `Date` column's class and a time
  series' `tsp`, and take a negative `n`, as R does.
- **`is.numeric()`** is now TRUE for a numeric matrix and FALSE for a
  `Date`, as in R.

### Plots an editor can read: whole pictures, marked parts, R's geometry — 2026-11-11

- **`r2 --ide` replies carry the whole plot.** After `plot()` then `lines()`,
  `points()`, `abline()`, `text()` or `title()`, the reply's `plot` is the complete
  picture; it held only the first drawing, so overlays (and `title(sub =)` and
  `text()`, which are overlays) seemed to be ignored.
- **The SVG's parts are marked**: `class="main"`, `"sub"`, `"xlab"`, `"ylab"` on the
  titles and axis names, and `class="plot-region"` with `data-usr="x1 x2 y1 y2"` (R's
  `par("usr")`) on the plotting area, so an editor can pick and edit them.
- **`xlim` and `ylim`** set the axes of `plot()` and `hist()`, padded 4 % as R pads
  them; both were ignored.
- **`barplot` draws as R does**: bars of width 1 at 0.7, 1.9, ..., returned as R
  returns them; `horiz = TRUE` (it drew vertical bars); names from the vector; no
  "Barplot" title R does not draw; negative bars. **`hist`, `boxplot` and
  `barplot` record their coordinates**, so `lines(density(x))` over a histogram lands
  where it should. `boxplot` uses Tukey's hinges and whiskers and draws outliers.
- **`density()`** has R's bandwidth (`bw.nrd0`: the smaller of the sd and IQR/1.34; it
  used the sd alone), takes `bw` and `n`, and returns `bw`, `n`, `call`, `data.name`
  and `has.na` with class "density". `plot(density(x))` draws R's line, title and
  "N = 5   Bandwidth = 0.9736" label; `lines(density(x))` adds the curve.
- **The datasets are R's.** `iris` was a variant with two rows differing from R's,
  and `faithful` was not R's data; both are now R's, and `ToothGrowth$supp` is a factor.
- **Also:** `round()` is R 4's (`round(3.5965625, 6)` is 3.596562; it gave 3.596563);
  `sapply()` names its result after a data frame's columns or a list's names;
  `pdf(NULL)` draws without writing; `pdf()`, `png()` and `svg()` are invisible; `str()`
  shows ten values and "...".
- **`interrupt` in `r2 --ide` is no longer lost.** One sent while its request was still
  waiting to run (R2 starting, or busy) was cleared when the request began, so a
  runaway loop could not be stopped. An interrupt now stops the request it followed,
  even one that has not started yet, and only that one.
- Checked in the `--ide` reply itself (`crates/r2-repl/tests/ide_protocol.rs`), and the
  values compared with R in `tests/differential/output/studio_graphics.R`.

### Two-way tables, `wilcox.test` and `airquality` as in R — 2026-11-11

- **`table(a, b)` counts both variables.** It ignored `b` and gave a one-way table.
  The result is an integer matrix of class "table" with named dimnames, printed in
  R's layout, readable as `t["6", "4"]`; `table(am, vs)` names its dimensions from
  the call, as R does.
- **`wilcox.test` gives R's p-values.** With fewer than 50 values a side and no ties
  (and no zero differences when paired) the p-value is exact; otherwise it is the
  normal approximation with R's tie and continuity corrections. It was always an
  uncorrected normal approximation (0.149 where R gives 0.2). `paired`,
  `alternative`, `exact`, `correct`, the formula form `wilcox.test(mpg ~ am, data =
  mtcars)`, and the statistic named `W` or `V` all work.
- **`airquality` has all 153 rows of R's**, with integer columns; it had the first 30.
- **Also found and fixed:** a matrix indexed by row or column names
  (`m["r1", "c2"]`) stopped with an error, and a row or column taken from a named
  matrix lost its names. `x$method`, `x$type` and other R2 keywords after `$` did not
  parse. `numeric(0) + 1:3` crashed R2; it is `numeric(0)`. `df[153, ]` keeps its row
  name. `colSums`, `colMeans`, `rowSums` and `rowMeans` give NA for a missing value
  unless `na.rm = TRUE` (they skipped it, and `colMeans` still divided by every row),
  and name their result as R does.
- Compared with R line by line in `tests/differential/output/studio_checks.R`.

### Model tables as R gives them: coefficient names, `anova`, `cor.test` — 2026-11-10

- **Coefficients of expression terms are named as written.** `lm(mpg ~ wt +
  factor(cyl))` gives `factor(cyl)6` and `factor(cyl)8`, and `I(wt^2)` and `log(hp)`
  keep their names; they were `x26`, `x28` and `x3`. **New: `I()`.** `deparse`
  writes `x^2` and `1:3` without spaces, as R does.
- **`anova(fit)` is R's table**: one row per term in formula order (sequential sums
  of squares from the QR of the design), then Residuals, with `Df`, `Sum Sq`,
  `Mean Sq`, `F value` and `Pr(>F)`, readable as `a$Df`, `a[["F value"]]` and
  `rownames(a)`. It was one overall F test. A fit keeps its design for this, out
  of `names(fit)`.
- **`cor.test`** gives `parameter` (df), `conf.int` with its `conf.level` (Fisher's
  z), `null.value`, `alternative` and `method`, takes `alternative` and `conf.level`,
  and lists them in R's order; `df` is now `parameter`, as in R.
- **New: `dimnames()`, `nobs()`.** `matrix(..., dimnames = list(...))` keeps the
  names it was ignoring. `[[` on a result reads its fields by name or position, and
  `print()` of a list ends with R's blank line. 543 built-in functions.
- Compared with R line by line in `tests/differential/output/studio_tables.R`.

### Editing data frames as Ardon Studio's Data tab does — 2026-11-10

- **Every `d[i, j] <- value` works.** Text into a factor or text column
  (`d[1, 2] <- "b"`) stopped with "cannot convert character to numeric", and
  `d[1, ] <- list(5, "b", "z")` gave each column the whole list.
- **`rbind` of data frames with factor columns** adds rows: a new value becomes a new
  level, two factors join their levels, numbers and integers mix, and columns are
  matched by name as R matches them. It stopped with "incompatible column types".
- **`cut()` returns a factor**, as in R; it returned text. A single `breaks` is a
  number of intervals over the range, and `labels`, `labels = FALSE`, `right`,
  `include.lowest` and `ordered_result` work, with R's labels ("(0.992,3.67]").
- **`save(d, file = f)` and `load(f)` work as R's.** Objects are saved under their
  names, factors included, and `load()` puts them back where it is called and
  returns their names, silently. Factor columns were dropped and `load()` printed a
  message. R2's older `save("session.r2s")` and `save(obj, "file.r2d")` still work.
- **`read.csv` honours `na.strings`, `dec` and `sep`** (`read.csv(f, sep = ";", dec =
  ",", na.strings = "-")`); all three were ignored. `write.csv` no longer prints
  "Written to ...".
- **New: `tempfile()`, `tempdir()`.** `str()` of a data frame lays out names and
  factor levels as R does. 540 built-in functions.
- Compared with R line by line in `tests/differential/output/studio_data.R`.

### Numbers as text, as R writes them — 2026-11-09

- **`as.character()` and `paste()` write a number with up to 15 significant digits**, as R
  does, where they used 7: `paste(1/3)` is "0.333333333333333", `paste(0.1 + 0.2)` "0.3".
  Scientific notation is used where it is no wider, so `as.character(1e5)` is "1e+05",
  `123456` stays "123456". Text written this way reads back with `as.numeric()` and
  `read.csv()`. Printing is unchanged (7 digits, as in R).
- **`sprintf()` takes a numeric matrix**, formatting its numbers column by column.
- The CUDA example programs are marked as needing `--features cuda`, so
  `cargo test --workspace` builds again without it. Differential case `as_character.R`.

### `try()`, `conditionMessage()`; `prcomp()` and `kappa()` on the SVD — 2026-11-09

- **`try(expr, silent = FALSE)`** works as in R: the value, or on an error an invisible
  string of class `"try-error"` with the condition in `attr(, "condition")`. R2's own
  `try { } catch (e) { }` syntax still works beside it.
- **A `tryCatch` handler gets a condition**, so `e$message` and `conditionMessage(e)` work;
  `conditionCall(e)` is `NULL`, as R2 records no call. 538 built-in functions with the S3 work merged.
- **`prcomp()` is computed as R computes it**, from the SVD of the centred (and scaled)
  data rather than the eigenvectors of its covariance matrix, which squares the condition
  number: a component near zero now keeps its digits. It returns the scores `$x`, and
  `$rotation` carries the variable names. `scale = TRUE` without centring now scales.
  `prcomp()` and `scale()` share one centring routine.
- **`kappa(exact = TRUE)`** comes from the singular values too, not `XᵀX`.
- New differential cases `conditions.R` and `prcomp.R`: 43 of 43 agree with R 4.5.3.

### S3 methods: `UseMethod`, `NextMethod`, `print.foo` — 2026-11-09

- **`UseMethod("g")`** in a generic calls `g.<class>` for the first class of the
  object (its class attribute, else R's implicit class: `numeric`, `character`,
  `function`, ...), else `g.default`, else the builtin of that name, and returns
  its value from the generic. `UseMethod("g", obj)` dispatches on another argument.
  With no method it stops with R's "no applicable method" message.
- **`NextMethod()`** calls the next class's method, then the default, then the
  builtin, so a `print.foo` can add a line and hand over to the ordinary print.
- **A builtin dispatches to a user's method**: `print.money`, `format.temp`,
  `length.stack`, `summary.stack`, `toString.money`, and auto-print uses
  `print.<class>`. **Operators too**: `"+.money" <- function(e1, e2)`, or one
  `Ops.temp` for all of them, reading `.Generic`.
- **`get("name")`** returns a variable, a builtin or an operator (`get("+")(1, 2)`).
  **`conditionMessage(e)`**, and `tryCatch` hands its error handler R's condition
  object, so `e$message` works; it was a bare string. A quoted name can be assigned
  (`"+.money" <- ...`). 536 built-in functions.
- Nothing is looked up until a function with a dot in its name is defined, so code
  without methods pays nothing. Compared with R line by line in
  `tests/differential/output/s3_dispatch.R`; what is not covered is in
  `docs/KNOWN_LIMITATIONS.md`.

### Dates that stay dates; operators keep names and shape — 2026-11-09

- **Date arithmetic keeps the class.** `d + 1` and `d - 1` are dates, `d2 - d` is a
  `difftime` in days ("Time difference of 5 days"), and a date-time difference picks
  R's units (secs, mins, hours, days). `difftime` keeps its units through `+ - * /`.
  All of these were plain numbers. `Date + Date` stops with R's error.
- **Dates and date-times print as dates everywhere**, auto-printed, inside lists and
  through `print`. Before, only `print(d)` showed a date and `d` alone showed 19797.
  Date-times print their time zone, and their clock only when it is not midnight.
- **`seq` of dates** (`by = "day"`, `"week"`, `"2 weeks"`, `"month"`, `"quarter"`,
  `"year"`, a number of days, `to` or `length.out`) and **`diff` of dates** (a
  `difftime`). `as.POSIXct("2024-03-15")` reads a date alone, as R's does, and keeps
  `tz` and the class `c("POSIXct", "POSIXt")`.
- **New: `as.difftime`, `weekdays`, `months`.**
- **Operators keep names and shape.** `x * 2` keeps `x`'s names; arithmetic keeps a
  class (`class(y * 2)`); `-m` stays a matrix. `m > 2`, `m == 3` and `!g` give logical
  matrices that print as R prints them and work in `which(g, arr.ind = TRUE)`,
  `apply` and `ifelse`. `dim()` and `is.matrix()` read a `dim` attribute.
- **`apply` over a matrix without column names gives an unnamed result**; it named
  the results `V1`, `V2`, ...
- **`qr()` picks its routine.** A large tall matrix that needs no pivoting goes to
  the blocked, parallel QR that `lm` uses, and the result is converted to R's
  `$qr`/`$qraux` form, so the object is the same: about 2x faster at 2000 x 500,
  20% at 1000 x 200, and not used below n·p² = 2^24, where it measured slower.
  `LAPACK = TRUE` is accepted and changes nothing.
- Compared with R line by line in `tests/differential/output/dates_and_shapes.R`.

### Seeing into LLM training from a script — 2026-11-07

- **`llm.train` and `llm.finetune` take `callback = function(step, loss)`**, called
  on the steps `report =` names, in place of the printed line. It can read the model
  (`llm.inspect`, `llm.generate`, `llm.save`) but not change it, and `stop()` in it
  ends the run with the weights it has reached. With `device = "gpu"` each call first
  reads the weights and optimizer state back to the CPU.
- **`llm.inspect(m, text)`** gives the loss on one batch, a data frame of weight and
  gradient norms per block, and a data frame of activation mean, sd, largest value
  and non-finite count per stage. It costs one forward and backward pass and that
  pass's memory per call, changes no weight, and always runs on the CPU.
- **`llm.new(preset = "tiny")`**: `"tiny"` 1.3M, `"small"` 7.2M, `"medium"` 40M and
  `"large"` 125M parameters, the shapes R2 has been benchmarked at. A preset sets the
  shape only, and a named argument overrides it.
- A run using none of these is unchanged: on a CPU training and a fine-tuning script,
  the saved weights match the previous build's byte for byte.
- `samples/llm/inspect.r2` shows all three.

### `svd()` and `eigen()`: bidiagonal QR, divide and conquer, blocked reductions — 2026-11-07

- **`svd()` is more accurate.** Singular vectors went through `BᵀB`, which squares the
  condition number: a 1000 x 1000 product reconstructed to 1.8e-11, and a low-rank one to
  1.4e-9. They now reconstruct to 9e-14 and 1.1e-13 (R: 1.6e-13 on the first). Hilbert(12), whose
  smallest values are below rounding, matches R's residuals. Values alone use the zero-shift QR
  of Demmel and Kahan, for high relative accuracy.
- **Faster at n = 1000, six threads:** `svd()` 1.02 s to 0.59 s, values alone 0.46 s to 0.33 s,
  symmetric `eigen()` 0.34 s to 0.29 s. R 4.5.3 takes 2.20, 0.88 and 1.25 s on the same machine.
  From n = 64 vectors come from divide and conquer (Gu and Eisenstat), and the reductions and
  `Q` are blocked onto GEMM.
- Matrices of extreme scale are divided down first, so their squares no longer overflow.

### Training on NVIDIA GPUs through CUDA — 2026-11-06

- **`--features cuda`**: `llm.train` and `llm.finetune` with `device = "gpu"`, and
  `gpu.map`, run on an NVIDIA GPU. Nothing is linked and nothing new is depended
  on: the driver's library is opened at run time, so the build works on any
  machine, and without a usable GPU `gpu.map` runs on the CPU while training with
  `device = "gpu"` says there is no device. Linux for now; f32 storage only.
- **On a Tesla T4 against PyTorch+CUDA:** level at 7M parameters, PyTorch 1.14x
  faster at 40M and 1.55x at 125M, with the same losses to four decimals
  (`benchmarks/llm/REPORT.md` §4c).
- `gpu.status()` now reports the device and where the last `gpu.map` really ran.
- **Fixed:** `gpu.enable()` never reached the part of R2 that routes the training
  tape's f32 matrix products, so they stayed on the CPU. They now go to the GPU
  above 64M operations once it is enabled. `%*%` works in f64 and stays on the CPU.
### Named vectors, list subscripts, and classes on lists — 2026-11-05

- **`x["a"]` works.** A character subscript on a named vector stopped with an error.
  It now gives the first element of that name, and `NA` named `<NA>` for a name that
  is not there.
- **`[` keeps names.** `x[2]`, `x[2:3]`, `x[-1]` and `x[c(TRUE, FALSE)]` dropped them.
  An empty selection prints `named numeric(0)`.
- **`[` works on lists**, by position, name, mask and negative position. It stopped
  with an error for all of them; only `[[` and `$` worked.
- **Subscripts that R allows no longer stop the script.** `c(1, 2)[5]` is `NA` and
  `c(1, 2)[0]` is empty; both were errors for numeric vectors in the default mode. A
  logical mask recycles and an `NA` in it gives `NA`.
- **`m[m > 2]`, `m[-1]` and `m[0]` work on a matrix.** A logical mask was read as the
  positions 0 and 1 and stopped with an error at the first 0.
- **Lists and vectors carry classes and attributes.** `RVal::List` now has the same
  attributes as the atomic vectors. New: `class<-`, `oldClass`, `unclass`, `attr<-`.
  `structure(list(...), class = "foo")` used to drop the class silently, and a class
  vector like `c("foo", "bar")` was flattened into one string; both are kept now.
  `inherits` reads the whole vector, and `attr(,"class")` prints as in R.
- **A classed result reads as a list**: `is.list(fit)` is TRUE, and `names` and
  `length` report its fields. A type declared with `type` is unchanged.
- **Lists print as R prints them**, with a blank line between elements, `$a$b` tags
  for nested lists, and `list()` when empty.
- **New: `is.integer`, `row`, `col`.** 531 built-in functions.
- Compared with R line by line in `tests/differential/output/names_and_classes.R`.
  Dispatch on class (`UseMethod`, `print.foo`) does not exist yet
  (`docs/KNOWN_LIMITATIONS.md`).

### `qr()` and its accessors; `cbind` and `rbind` recycle — 2026-11-05

- **`qr(x, tol, LAPACK)`, `qr.Q`, `qr.R`, `qr.X`, `qr.coef`, `qr.qy`, `qr.qty`,
  `qr.resid`, `qr.fitted`, `qr.solve`, `is.qr`, and `solve()` of a `qr` object.**
  524 built-in functions. The object is R's own compact form, because R documents
  it: `$qr`, `$rank`, `$qraux`, `$pivot`, from LINPACK's `dqrdc2` with its limited
  pivoting, so a negligible column moves right and the others keep their order.
  `dgeqrf`, which `lm` uses, stays as it was.
- **Compared with R on 3,038 values** (`tests/differential/cases/qr.R`): full-rank,
  wide, rank-deficient and zero matrices, the Hilbert matrix of order 6, `tol`,
  matrix responses and names. All agree to 1e-9 and 3,035 to 1e-13; the other three
  are Hilbert values at about 4e-12. A 2000 x 500 `qr()` took 0.34 s against R's
  0.32 s, one run each. Differences from R are in `docs/KNOWN_LIMITATIONS.md`.
- **`cbind(1, x)` and `rbind(1, x)` work.** Both refused any vector shorter than
  the others. They now follow R: matrices fix the shared side, otherwise the longest
  vector does, shorter vectors recycle with R's warning when they do not divide
  evenly, and empty vectors and `NULL` drop out.

### The standard distribution families, and R's arguments everywhere — 2026-11-05

- **New: uniform, Weibull, lognormal, Cauchy, logistic, geometric, negative
  binomial (`size` with `prob` or `mu`) and hypergeometric**, each with `d`, `p`,
  `q` and `r`; the F density `df`; the generators `rchisq`, `rt` and `rf`.
  513 built-in functions.
- **`rf` is both things now.** `rf(n, df1, df2)` with single numbers draws from F;
  with a formula or vectors it is still the random forest.
- **The older functions take R's arguments.** `dnorm`, `pnorm`, `qnorm`, `dexp`,
  `pexp`, `qexp`, `dbinom`, `pbinom`, `qbinom`, `dpois`, `ppois`, `qpois`, `dt`,
  `pt`, `qt` and `pf` read `log`, recycle every parameter and accept names in any
  order. Before this, `log = TRUE` was ignored, so `dnorm(1, log = TRUE)` returned
  the density itself, and a vector parameter kept its first value, both silently.
  `pnorm(1, 2, 3)` ignored its positional mean and sd.
- **Log tails no longer underflow.** `pt`, `pf`, `pbinom` and `ppois` with
  `log.p = TRUE` continue on the log scale where they used to reach -Inf.
- **`qf` goes through the beta quantile instead of a bisection.** The 555-value
  `distribution_grid` went from 543 values within 1e-13 of R to 551. The
  four left are all `qf`, with `qf(0.01, 1, 200)` the worst, and a 60-digit reference
  sides with R2 on them.
- **The generators recycle their parameters.** `rpois` is right at large means
  (PTRS above 30), and `rbinom` no longer loops over `size` trials above size 1000.
- **Compared with R on 4,462 more values** (`tests/differential/cases/families.R`).
  Sixteen differ by more than 1e-11, and a high-precision reference or exact
  fractions side with R2 on all of them apart from one difference that should be
  zero. `qt` at `df = 1e5` is the one place R is measurably closer: 2.3e-13 here
  against 4e-17. Non-central forms still stop with an error
  (`docs/KNOWN_LIMITATIONS.md`).
- **`c()` of lists returns a list**, flattening nested data with `recursive =` and
  honouring `use.names =`. **`do.call("name", args)`** takes a function name.

### The beta and gamma distributions — 2026-11-05

- **`dbeta`, `pbeta`, `qbeta`, `rbeta`, `dgamma`, `pgamma`, `qgamma`, `rgamma`**,
  with R's arguments: `rate` or `scale`, `lower.tail`, `log.p` and `log`, and
  every `d/p/q` parameter recycled.
- **Compared with R on 16,730 values**, shapes from 1e-3 to 1e5, tails down to
  1e-300. Where the two differ by more than 1e-11, a 340-digit reference sides
  with R2, apart from two quantiles below the smallest double. R itself returns
  -Inf for `pbeta(0.75, 1e4, 25, log.p = TRUE)` and 1 for
  `qbeta(1e-300, 10, 1e5, lower.tail = FALSE)`
  (`docs/KNOWN_LIMITATIONS.md`). `tests/differential/cases/beta_gamma.R` keeps
  about 12,000 of them where R is right.
- **`pt` and `pf` now share the new incomplete beta**, whose prefactor is in
  Loader's saddle-point form rather than from `lgamma`, and whose smaller tail
  is the one computed. They still pass the 555-value `distribution_grid` case.
- **`qchisq` stopped early on very small answers**: `qchisq(0.5, 0.002)` came out
  near 1e-61, where the answer is about 1e-301. A bracket that started at 0 was
  being halved, and 200 halvings cannot get there.

### LLM: `llm.train` learns from the whole text; `llm.save` keeps the tokenizer — 2026-11-04

- **`llm.train` only ever saw the first `batch + seq` tokens**: it built one
  batch and trained on it for every step, so a byte after those tokens could
  not change the result. It now walks through the whole text, and a second
  call continues where the first stopped. **Results on any text longer than
  one batch change, because the data does**; the forward, backward and Adam
  arithmetic did not (`grad_fingerprint` and the trained model's SHA-256 are
  unchanged). `shuffle = TRUE` visits windows in a new seeded order per epoch.
- **`llm.train(m, "file.txt")` reads the file**, as `?llm.train` always said;
  it trained on the path's own characters.
- **`llm.save` writes `tokenizer.json` and `llm.load` reads it.** A reloaded
  model with a learned vocabulary decoded its ids against the byte table and
  generated nonsense. A directory without the tokenizer is now refused, unless
  the model is pure byte-level.
- New `r2_train::data` (`Batcher`, `Order`): a batch is a function of the step
  index, so a resumed run sees the same data and shuffling stores nothing.
- **Runs can stop and continue.** `llm.train(..., checkpoint = dir, every =,
  keep =)` and the new `llm.resume(dir)`; `R2_CKPT_DIR`, `R2_CKPT_EVERY` and
  `R2_CKPT_KEEP` in `tinystories_train`. Nothing used `r2_train::checkpoint`
  before. Stopping the 7.24M model at step 50 and resuming to 100 writes the
  same file as 100 uninterrupted steps (same SHA-256).
- **`llm.load(dir, train = TRUE, lr =)`** starts a new run from saved weights
  (`Trainer::from_model`): the starting point of fine-tuning.
- **Fewer copies, same bytes.** A checkpoint is written from where the tensors
  live and read through a memory map; it held the model three times over while
  saving and twice while loading. Tensors are written and read in bulk.
  `llm.generate` copied the whole model per call and `llm.save` twice; neither
  copies now (`Trainer::with_model` lends the weights). Files are unchanged.
- **Fine-tuning from script.** `llm.finetune(m, prompts, completions, ...)`:
  only completion tokens count in the loss. New in the tape: targets marked
  `r2_autograd::IGNORE` take no part, as PyTorch's `ignore_index` (checked
  against it). Batches are right-padded to their longest example and grouped by
  length; attention is causal, so padding changes neither loss nor gradient (a
  test holds this to the bit) and needs no attention mask.
- **`llm.freeze(m, blocks)`**: frozen blocks keep their bits, take no gradient
  and hold no Adam state; checkpoints leave them out and `llm.resume(dir, base =)`
  takes them from the model they came from, refusing different weights.
- **Training settings in script**: `lr`, `wd`, `clip`, `schedule = "cosine"`,
  `warmup`, `min.lr` and `accum` (gradient accumulation, on
  `Trainer::train_step_parts`) in `llm.train` and `llm.finetune`. `llm.train`
  on a BPE model also stopped ignoring `llm.new(lr =)` on its first call.
- **LoRA, written natively** (`r2_train::lora`; no PEFT, no Python).
  `llm.lora(m, rank =, alpha =, targets =)` freezes the model and adds low-rank
  adapters; `B` starts at zero, so training begins exactly at the base. Only the
  adapters take gradients and Adam state, and checkpoints hold only them.
  `llm.save.adapter` writes a small file that applies to its own base
  (`llm.load.adapter` refuses another by fingerprint); `llm.save` writes a whole
  model with them merged, which the serving code runs unchanged. Adapter
  gradients are checked against finite differences; served logits against the
  tape's. Fine-tuning starts from R2's own models: there is no importer for
  other projects' weights. `samples/llm/finetune.r2` is the whole recipe.
  At 7.24M, rank 8: 57,344 values train instead of 7.24M, a checkpoint is
  0.66 MB instead of 82.9, an adapter 0.22 MB against 27.64 of weights, and a
  step about 15% faster. Peak memory is about equal (658 vs 641 MB): at this
  size activations dominate RAM, and the saving is disk and optimizer state.
- **On the GPU:** AdamW (bit-identical to the CPU, one fused multiply-add per
  weight), gradient clipping (a norm summed on the device in short chunks,
  within 2e-8 of f64 on any driver, reading back 226 KB instead of 29 MB), and
  ignored targets with padded
  batches. `device = "gpu"` in `llm.train` and `llm.finetune`, in a build with
  `--features gpu` (`r2-engine`, `r2-repl`); the default build compiles no wgpu.

### `r2 --ide`: R2 as the engine of a separate IDE — 2026-11-01

- **`r2 --ide`** runs R2 as the engine of a separate IDE program, the way
  RStudio drives an R session: one JSON object per line on stdin/stdout.
  The IDE links no R2 code. Operations: `hello` (versions), `eval`
  (console input; output streams as it is printed; unfinished input
  reported as `incomplete`; a changed plot returned as SVG), `value` (any
  value as JSON at full precision, nothing printed), `env` (Environment
  pane rows), `view` (a page of the data viewer, formatted as `print()`
  formats), `interrupt` (stops running code), `quit`. Specification:
  `docs/IDE_PROTOCOL.md`. `r2 --json` is unchanged.
- **A factor column in a printed data frame shows its levels** (an NA as
  `<NA>`), as R does; it printed `?` in every cell. Found by the `--ide`
  smoke test; `tests/differential/output/df_factor_print.R`.

### `View()`, and editor keys for apps built on R2 — 2026-11-01

- **`View(x)`**, as in R: a GUI built on R2 that installs a viewer (the
  separate IDE project does) opens `x` in its data viewer, titled with
  the expression (`View(trial)` → "trial"); `View(x, title = "...")` sets
  the title. Without a viewer — the `r2` command line and R2Gui — it
  prints `x`. Returns NULL invisibly.
- **r2-ui**: five more keys (S, Z, Y, O, N) reach applications while Ctrl
  is held, so an editor built on the framework can offer Ctrl+S / Ctrl+Z /
  Ctrl+Y / Ctrl+O / Ctrl+N. R2Gui's console keys are unchanged.

### `hist()` computes what R's does; `pretty()` and `nclass.Sturges()` — 2026-11-01

- **`hist(x, plot = FALSE)` returns R's `"histogram"` object** —
  `$breaks`, `$counts`, `$density`, `$mids`, `$equidist` — and draws
  nothing. It used to ignore `plot = FALSE`, write `hist.svg` into the
  working folder and return NULL, so R code that reads a histogram's
  numbers could not run. With `plot = TRUE` it draws and returns the same
  object.
- **R's breaks and counts**: Sturges' rule through `pretty()` by default
  (it was 10 equal-width bins over the data range), `breaks = n` as a
  suggested number of cells, `breaks = c(...)` as given; right-closed bins
  with the lowest break closed, as R's `C_BinCount` counts them. The bars
  drawn are those counts.
- **New builtins `pretty(x, n = 5)`** (R's `R_pretty`, line for line) and
  **`nclass.Sturges(x)`**. 464 builtins. Differential case `hist_pretty`:
  every break, count, density and mid, and ten `pretty()` ranges, equal to
  R 4.5.3.

### Statistics fixed against R: chisq.test, t.test, glm, lm, quantile, cor — 2026-10-30

Found by four new differential cases (221 values, compared at 15
significant digits — `sprintf("%.15g")`, so the harness's 1e-9 tolerance
is really applied). Each was a mismatch with R first; 32/32 cases pass.

- **`chisq.test(correct = FALSE)` and every table larger than 2x2 gave the
  wrong X²**: cells below their expectation were counted as zero, about
  half the statistic — a 2x3 table read 5.52 (p 0.063) where R has 10.27
  (p 0.0059), so the conclusion flipped. Yates' correction now applies to
  2x2 tables only, as in R.
- **`t.test(var.equal = TRUE)`** now runs the pooled-variance test (it was
  always Welch), and the two-sample `mu =` is honoured.
- **`glm`**: the Poisson AIC was missing a term (300.19 where R has 46.50);
  log(y!) for large counts is exact (`lgamma`); gaussian p-values come from
  t on the residual df, not the normal (3.9e-26 where R has 5.6e-6); the
  Poisson fit starts where R's does (μ = y + 0.1) and no longer diverges
  on counts in the hundreds.
- **`lm`/`glm` with an NA no longer crash the session**: rows with a missing
  response or predictor are dropped (R's `na.omit`). An exactly collinear
  column is reported `NA` and the fit uses the rest, as R does (it returned
  coefficients of 2e13).
- **`t.test` with missing values drops the whole row, as R does**: the
  formula form (`t.test(score ~ grp, data = d)` with an NA score failed
  with "LHS length (7) != RHS length (8)"), its `id =` / `Error()` forms,
  and `t.test(x, y, paired = TRUE)`, which used to pair every observation
  after the first NA with the wrong partner.
- **`aov(y ~ a + b)`** stops with a clear error instead of silently fitting
  `a` alone (multi-factor ANOVA: `docs/KNOWN_LIMITATIONS.md`).
- **`quantile(type = 1..9)`** and **`cor(method = "spearman" / "kendall")`**
  were ignored (always type 7 / Pearson); both now follow R, ties included.
- The silu backward's scalar tail now rounds as the AVX2 lanes do (it
  differed on 0.5% of inputs; training bits unchanged — FFN widths are
  multiples of 8). `--example numerics_exhaustive` checks all 2^32 floats.

### LLM training recipe: AdamW, warmup + cosine schedule, gradient clipping — 2026-10-30

- `r2_train::llm::Trainer` gains the three pieces every modern LLM run
  uses, all **off by default** (a run that sets none of them is bit-for-bit
  the run it was): AdamW decoupled weight decay (`opt.weight_decay`; every
  matrix decays, RMSNorm gains never do), a linear-warmup + cosine-decay
  learning rate (`sched: Some(LrSchedule { .. })`), and global-norm
  gradient clipping as `torch.nn.utils.clip_grad_norm_` (`clip`), with the
  norm reported per step. Checkpoints store the decay; older ones load as
  plain Adam. The TinyStories harness takes `R2_SCHED=cosine`,
  `R2_WARMUP`, `R2_MIN_LR`, `R2_CLIP`, `R2_WD` and the PyTorch half applies
  the same recipe: with all three on, 100 steps, the two learn identically
  (every checkpoint to four decimals, held-out 5.8068 vs 5.8068). The
  decay is applied as `p - (lr*wd)*p` in one fused multiply-add rather
  than `p * (1 - lr*wd)`, whose rounded factor misstates a small decay the
  same way every step: against an exact f64 AdamW over 3,000 steps
  (`--example adam_audit`) the weights now stay within 9e-6 of exact
  (relative to how far they moved), down from 7.2e-5; plain Adam 5.7e-6. The GPU
  trainer follows the schedule and refuses decay or clipping with a clear
  error until the device has them.
- GPU: an adapter whose `fma` is not fused (Mesa's software llvmpipe) is
  named as such in the device line, and the CPU bit-identity tests skip
  there with the reason; on real GPUs they run (they pass on the Vega).
  CI gains a debug-build job (overflow checks on) that also runs the GPU
  path on llvmpipe — the class of bug `f16_to_f32` was.

### LLM numerics audited; RoPE exact at any context length — 2026-10-29

- **RoPE positions are now correctly rounded at every context length.** The
  rotation angles were computed in f32, which rounds the angle itself, and
  that error grows with the position: cos/sin were off by 9e-6 below
  position 256, 3e-4 by 8,192 and 1.2e-3 by 32,768. They are now computed
  in f64 and rounded once, right to 3e-8 (half an f32 ulp) at every
  position — on the CPU tape, in inference and on the GPU alike. The
  PyTorch reference in `benchmarks/llm/` computes its table the same way,
  and a 300-step TinyStories pair still trains to identical losses
  (held-out 4.3113 vs 4.3114, the same generated sentence).
- **Every other LLM primitive was measured against f64** and is within a
  few ulps: `exp` 1.01 ulp and `log` 0.83 ulp over every float in range,
  silu 2.4, softmax cross-entropy 3.2, RMSNorm 3.9. `cargo run --release
  -p r2-tensor --example numerics_audit` reproduces the table.
- **Training step: two serial passes made parallel, bit for bit.** The
  cross-entropy forward (14.9 -> 3.8 ms a step on the 7M model) and the
  RMSNorm weight gradient (10.2 -> 7.1 ms) now use every core and give the
  same bits as before — the loss and a hash over all 7.24M gradients are
  unchanged (`--example grad_fingerprint`).

### Installer documentation opens in Notepad — 2026-10-02

- The Windows installer and the Windows zip ship the documentation as real
  plain text (`README.txt`, `CHANGELOG.txt`, `FUNCTIONS.txt`, ...): a
  double-click opens it in Notepad (Windows has no default program for
  `.md`), and it reads as text, not Markdown source — headings underlined,
  `**`/HTML markup removed, links as "text (url)", lines wrapped at 78
  columns, and tables restructured (aligned columns when narrow, one
  labelled block per row when wide). Generated at build time from the
  Markdown by `cargo run -p r2-utils --example doc2txt`, so they never go
  stale. The Start menu gains an "Ardon-R2 Read Me" shortcut.

### Windows paths can be pasted as they are — 2026-10-02

- **On Windows, a path copied from Explorer works in a string literal**:
  `setwd("C:\Users\deven\Desktop\DAS paper")` and
  `read.csv("C:\Users\...\DASS.csv")` read the backslashes literally. In R
  these fail (`\U` is an escape) or silently go wrong (`\n` in
  `...\new folder` becomes a newline). The rule applies only to a string
  that starts with a drive path (`C:\` followed by a non-backslash) or a
  network path (`\\server\share`), so correctly escaped R code
  (`"C:\\Users"`) keeps R's meaning. Forward slashes and `\\` work as in R.
  On Linux and macOS strings follow R exactly.
- Paths R2 reports stay R's style: `getwd()` and `list.files(full.names =
  TRUE)` use forward slashes on every OS.

### Files, folders and strings behave as in R — 2026-09-30

- **`list.files()` honours its arguments**: `pattern` (a regular
  expression on the file name), `full.names`, `recursive`, `all.files`,
  `ignore.case`, `include.dirs`; `list.files(pattern = "\\.csv$")` returned
  nothing (only the first argument was read, as a folder). New: `dir()`
  (same function), `file.path()`, `basename()`, `dirname()`.
- **`getwd()` uses forward slashes** on Windows (`C:/Users/...`), as R does;
  **`setwd()` returns the previous folder**, so `old <- setwd(d); ...;
  setwd(old)` restores it (it returned the new one).
- **`ls()` lists only your own objects**, sorted: `pi`, `Inf`, the typed
  NAs and the built-in datasets (`iris`, `mtcars`, ...) now live in a base
  environment that is the workspace's parent, as in R, instead of
  appearing in every `ls()`.
- **String escapes**: `"say \"hi\""` kept its backslashes (10 characters,
  not 8) — only `\n`, `\t`, `\\` were read. Now also `\"`, `\'`, `\r`, `\a`,
  `\b`, `\f`, `\v`, octal, `\xhh`, `\uXXXX`, `\U{...}`. Printed strings are
  escaped as R prints them (`"C:\\Users"`, `"a\tb\nc"` on one line).
- `grepl`, `grep`, `sub`, `gsub` and `regexpr` accept `ignore.case = TRUE`.
- `read.csv(DASS.csv)` (no quotes) adds to R's "object not found" that a
  file of that name is in the working directory and needs quotes.
- `read.csv()` no longer prints "Read n rows × m columns" (R is silent).
- Whole-output differential case `output/files_and_strings`.

### Reports re-timed; `sort()` speed restored — 2026-09-28

- **`sort()` of 1e6 numbers was 6.6x slower than R** (0.45 s vs 0.068 s)
  since the 2026-09-23 rewrite that gave it R's ordering rules — every
  comparison went through the general NA-aware path. A fast path for a
  single numeric key sorts value/position pairs: 0.070 s (R 0.065 s), same
  order as R including NA/NaN placement and `-0`/`0` ties.
- `set.seed()` no longer prints "Random seed set to ..." (R prints nothing).
- `PERFORMANCE.md` sections 1-4 re-timed on the AMD laptop (two
  interleaved passes; they were from v0.3.8): R2 ahead 1.1-1.6x on vector
  add and sum, 11x on 500x500 `%*%`, 6-10x on loops and ~40,000x on the
  variance loop; behind on `cor` (2.9x), `sd` and the composed
  standardize helper (1.2x). The linear-algebra table now uses a
  full-rank matrix (the old `sin(1:n^2)` one had rank 2, which made `qr`
  and `svd` atypical): R2 20-25x on `%*%`, 2.0-6.2x on the decompositions.
  README, `llms.txt`, `benchmarks/README.md` and REPORT.md §5 (dgemm, now
  at MKL parity) updated to match.

### `svd()` small singular values and vectors accurate — 2026-09-28

- **Small singular values lost digits**: they came from the square roots
  of the eigenvalues of BᵀB, so they carried the rounding of the largest
  (Hilbert(10)'s smallest, 1.09e-13, was 4% off; absolute error ~4e-15).
  They now come from the Golub–Kahan form (eigenvalues ±σ, nothing
  squared): the error is ~3e-18, well under R's own rounding floor.
- **`svd()$u` was not orthonormal for small or zero singular values**:
  its columns were `B·v / σ` (rounding amplified by 1/σ) and exactly-zero
  σ left a zero column. Those columns are now re-orthonormalised
  (block Gram–Schmidt on the GEMM kernel), as LAPACK's are; a
  rank-deficient matrix gets a proper orthonormal completion.
- Cost: `svd(x, nu = 0, nv = 0)` at 1000x1000 ~425 ms (was 323), a full
  SVD of a rank-deficient 1000x1000 ~1.0 s (R 4.5.3: 2.0 s); a
  well-conditioned matrix skips the re-orthonormalisation.
- Differential case `svd_accuracy` (singular values compared as 4-digit
  strings so the harness's absolute floor cannot hide an error at 1e-13).

### Small p-values are no longer rounded to zero; distributions checked against R — 2026-09-28

- **t-test and regression p-values below ~1e-16 were reported as 0**:
  they were computed as `2 * (1 - pt(|t|))`, which cancels. `lm(mpg ~ wt,
  mtcars)` gave the intercept p = 0; it is now 8.241799e-19, R's value.
  Same fix in `t.test` (all forms) and `cor.test`.
- **`qt()` was accurate to only ~1e-8** (`qt(0.5, df)` gave -1e-8, not 0)
  and clipped the far tail (the t-test's quantile searched only [-50, 50];
  `qt(1e-10, 1)` is about -3.2e9). Now: exact at 0.5, an expanding
  bracket and Newton refinement to full precision.
- `pt()` lost precision near 0 and in the far tails (either identity alone
  cancels on one side); it now picks the identity that gives the small
  probability directly.
- **`ppois()` / `pbinom()` upper tails** were `1 - lower`: `ppois(150, 0.5,
  lower.tail = FALSE, log.p = TRUE)` was `-Inf` (R: -715.2) and
  `ppois(150, 4, ...)` -34 (R: -405). The small tail is now summed directly
  in log space.
- `log.p = TRUE` of a probability near 1 (`log(1 - 1e-10)`) keeps all its
  digits (it kept six).
- New differential case `distribution_grid`: 555 values of `pt`, `qt`,
  `dt`, `pf`, `qf`, `pnorm`, `qnorm`, `dnorm`, `pbinom`, `qbinom`,
  `dbinom`, `ppois`, `qpois`, `dpois`, `pexp`, `qexp`, `dexp` including far
  tails and `log.p`; 543 agree with R to 1e-13, the worst (`qf`) to 4e-11.

### `summary()`, `anova()`, `aov()` return values and print only when shown — 2026-09-28

- **`summary()` returns an object and prints nothing while computing**, as
  in R: `s <- summary(fit)` is silent, `s` prints the summary, and
  `summary(fit)$coefficients` is R's matrix (Estimate, Std. Error,
  t value, Pr(>|t|), rows named by the coefficients) — it printed the
  whole summary and returned the model itself. `class(s)` is
  `summary.lm` / `summary.glm`.
- `summary(x)` of numbers returns R's `summaryDefault` (Min., 1st Qu.,
  Median, Mean, 3rd Qu., Max., and NA's when present), printed at 4
  significant digits like R; it returned NULL. `summary(df)` returns a
  value that prints the table.
- `anova()`, `aov()`, `manova()` and `system.time()` results no longer
  print on assignment, and print when shown.
- Matrices with row names print them left-aligned, as R does.
- Differential case `output/summary_values`.

### Numbers print with R's common decimals — 2026-09-28

- **A numeric vector prints in one format, as R's does**: `c(1, 2.5)` is
  `[1] 1.0 2.5` (it was `1 2.5`), `c(1.123456789, 100)` is
  `1.123457 100.000000`, `c(0.5, 100000)` is `5e-01 1e+05` — the most
  decimals any element needs, fixed unless scientific is narrower (R's
  `formatReal`). Named vectors, matrix columns and data-frame columns
  print the same way; `print(x, digits = n)` applies.
- **`format()`** gives the whole vector one format and one width
  (`format(c(1, 10, 100))` is `"  1" " 10" "100"`), treats `nsmall` as a
  minimum (`format(3.14159, nsmall = 2)` is `"3.14159"`), pads text on the
  right and keeps names.
- A data frame's header is no longer indented one space too far, its row
  names print when set, and matrices and data frames no longer print a
  blank line after themselves.
- Whole-output differential case `output/print_numbers`.

### Documents state the live version and counts — 2026-09-28

- `llms.txt` said "v0.4.0, 438 builtins" after the code had moved to
  0.4.1 and 458; it now states both correctly and adds the Intel, GPU
  and linear-algebra results. `FUNCTIONS.md`'s title says 458.
- **`?help` now covers the operators and replacement functions** (`+`,
  `==`, `%in%`, `(`, `names<-`, ...): 17 of the 458 had no entry.
- Tests now enforce this: every registered builtin must have `?help`,
  and `llms.txt` / `FUNCTIONS.md` must state the registry's count and
  `llms.txt` the current version — a new builtin or a version bump fails
  the test gate until the documents are updated.

### `solve()` 11x faster, `chol()` added, `svd()` values-only — 2026-09-25

- **`solve()` is 11x faster** and was slower than R: a 1000x1000 inverse
  took 1.27 s (R 4.5.3: 0.72 s) and now 0.11 s. The factorisation was
  already fast; the n column solves after it ran one at a time, and
  `solve(a, B)` re-factored `a` for every column of `B`. Now one LU and
  blocked, multi-core triangular solves for all right-hand sides.
- **`chol(x)`** is available (it was internal only): R's upper factor,
  `t(R) %*% R == x`; 1000x1000 in 23 ms (R: 120 ms).
- **`svd(x, nu = 0, nv = 0)`** returns only `$d` and skips building `u`
  and `v` (it ignored `nu`/`nv`): 1000x1000 761 -> 323 ms (R: 850 ms).
  `svd()` lists `d`, `u`, `v` in R's order (it was random).
- **`system.time()`** returns R's `proc_time` value, so
  `system.time(f())[["elapsed"]]` works; it printed the timing while
  running and returned NULL.
- **On the first machine (AMD Ryzen 5 4500U, 6 threads) vs Intel MKL**
  (PyTorch 2.13 float64): `%*%` median 1.04-1.21x over 11 shapes (two
  passes; 2000x2000 1.05x, 128x128 0.32-0.41x, tall/deep 0.8-0.9x); at
  n = 1000 LU 0.43x, Cholesky 0.18x, QR 0.31x, symmetric eigen 0.27x,
  SVD values 0.27x, SVD with vectors 0.30x. Against R 4.5.3 on the same
  machine every routine is 2.6-27x faster (`PERFORMANCE.md`).
- LLM training on the same machine after the Intel work: R2 1.54-1.69x
  faster than PyTorch (two 100-step pairs, identical losses); an
  interleaved A/B against the build before it shows no change (35-38 s
  per 100 steps either way).
- Differential case `linalg_user` checks `solve`, `chol` and `svd`
  against R.

### `lm()`'s QR 5x faster — 2026-09-25

- **The QR decomposition behind `lm()` and least squares is ~6x
  faster** (R2 has no user-level `qr()` yet): 1000x1000 225 ms -> 38 ms
  on an Intel i5-12500, about 7x faster than R 4.6.1 (285 ms) and ~0.25x
  of Intel MKL (was 0.04x).
  It now works in blocks of 64 columns applied as matrix multiplies (as
  LAPACK does). Same factor, same results. Reproduce:
  `python benchmarks/lapack_cases.py`.

### `svd()` 17-34x faster — 2026-09-25

- **`svd()` is 17-34x faster.** On an Intel i5-12500 at 1000x1000:
  singular values alone 6.7 s -> 0.2 s, the full thin SVD (with `u` and
  `v`, R's default) ~0.39 s — about 2.7x faster than R 4.6.1's reference
  LAPACK (0.52 s for values) and 0.29-0.44x of Intel MKL (was 0.01x).
  Same method, same results; it no longer builds `u` and `v` when only
  the values are wanted. Reproduce: `python benchmarks/lapack_cases.py`.

### `eigen()` on symmetric matrices 3-16x faster — 2026-09-25

- **Symmetric `eigen()` (values and vectors) is 3-16x faster**: 1000x1000
  3.2 s -> 0.2 s on an Intel i5-12500 — now ~3x faster than R 4.6.1's
  reference LAPACK (0.6 s) where it was 5x slower, and 0.19-0.29x of
  Intel MKL (was 0.02-0.09x). Same algorithm and same results; it now
  walks memory contiguously and uses every core. Reproduce:
  `python benchmarks/lapack_cases.py`.

### `solve()`, `det()`, `chol()` faster — 2026-09-25

- **LU (`solve()`, `det()`, inverse) is 2.7-8x faster** and **Cholesky
  (`chol()`, MANOVA, mixed models) 1.4-8x**: their O(n³) updates now
  run on the multi-core matrix-multiply kernel, and LU recurses the way
  LAPACK's does. On an Intel i5-12500 at 1000x1000: LU 160 -> 19 ms,
  Cholesky 88 -> 11 ms (2000x2000: 141 -> 65 ms). Still behind Intel MKL
  (0.18x and ~0.18x of its speed, from 0.02x). Reproduce:
  `python benchmarks/lapack_cases.py`.

### `solve()`, `det()` and symmetric `eigen()` vectors fixed — 2026-09-25

- **`solve()`, `det()` and matrix inversion were wrong for matrices
  larger than 32x32** (a residual of ~0.01 where it should be ~1e-15):
  the blocked LU factorisation skipped a triangular solve for every block
  after the first. Also affected: `lm()`'s standard-error fallback and
  the multivariate quadratic forms built on the inverse. Fixed; residuals
  are ~1e-15 from 4x4 to 200x200.
- **`eigen()$vectors` for symmetric matrices were wrong from 4x4 up.**
  The eigenvalues were always right; the vectors were not eigenvectors
  (`A %*% v` was not `lambda * v`). Fixed; `A %*% v - lambda * v` is now
  ~1e-14 at every size tested.
- A new test suite checks every decomposition — LU, inverse, Cholesky,
  QR least squares, determinant, symmetric eigen, SVD — by what it
  promises (residuals, `A` rebuilt from its factors) at 4x4 to 200x200.

### Cholesky fixed above 4x4 — 2026-09-25

- **The Cholesky factorisation was wrong for every matrix larger than
  4x4** (at 200x200, `L %*% t(L)` missed the input by up to 12.9), and so
  was everything built on it: MANOVA with more than 4 response variables,
  mixed models with more than 4 fixed effects, the normal-equations
  least-squares solver (the ML `lm` dispatch with more than 3
  predictors, and `lm()`'s fallback when QR does not apply), and the
  internal Cholesky routine. `lm()` itself uses QR and was not affected.
  Fixed; `L %*% t(L)` now reproduces the input from 5x5 to 200x200.

### AVX-512 matrix kernel for LLM training — 2026-09-25

- **On CPUs with AVX-512** (Intel Xeon Skylake-SP and later, AMD Zen 4/5)
  LLM training's matrix multiplies use a new 512-bit kernel; other CPUs
  are unchanged. It gives bit-identical results to the AVX2 kernel
  (verified under Intel's emulator, including a training run). Its speed
  has **not been measured yet** — no AVX-512 machine was available.
  `R2_SIMD=avx2` switches it off.

### Matrix multiply (`%*%`) at MKL speed — 2026-09-25

- **`%*%` on doubles is 3-16x faster** and now matches Intel MKL's
  `dgemm` (PyTorch float64, 6 threads, Intel i5-12500): median 0.97-1.02x
  over 11 shapes, ahead on large squares (2000x2000: 1.6x) — it was 0.23x.
  It runs on the same packed AVX2 kernel as LLM training, and goes
  multi-core from much smaller sizes (256x256: 2.95 ms -> 0.18 ms).
  Tall-and-thin products (`X %*% W` with 100 columns) are still ~0.45x
  of MKL. Reproduce: `python benchmarks/dgemm_cases.py`.

### `NA`, assignment, sorting and printing match R — 2026-09-23

- **`sort()` no longer crashes** on NaN (`sort(c(3, NaN, 1))` panicked:
  the GUI's error popup). It keeps the type (integer, character, logical,
  factor), honours `decreasing =` (ignored before) and `na.last =`.
  `order()` takes several keys and puts NA last; `unique()` and
  `duplicated()` compare exactly and keep NA once (NA and NaN distinct);
  `rank()` averages ties and has `ties.method = "min"/"max"/"first"`.
- **Strings sort as R does on Windows and in English locales**:
  case-insensitive first, lowercase first on a tie (`"a" "A" "b" "B"`),
  accents after the base letter (`"naive" "Naive" "naïve"`), punctuation
  before digits before letters in R's order. `sort`, `order`, factor
  levels and every grouping function share it.
- **`NA` is logical**, as in R: `c(TRUE, FALSE, NA)` stays logical,
  `c(1L, NA)` integer, `class(NA)` is `"logical"`. `NA_real_`,
  `NA_integer_` and `NA_character_` exist. `c()` keeps element names
  (`c(a = 1, b = 2)`), and combining with strings no longer drops
  logical/integer values (`c("a", TRUE)` lost the TRUE) or writes doubles
  to 17 digits.
- **Element assignment works like R's.** `x[x > 2] <- 0`, `x[-1] <- 9`,
  `x[is.na(x)] <- 0`, `x["name"] <- v` all failed with an error. Also now:
  assigning past the end grows the vector with NA; the vector widens to
  hold the value (`x[2] <- 2.5` on integers, `x[2] <- "z"` on numbers);
  `NA` into a character vector is NA, not the string "NA"; assigning
  into logical vectors and factors; nested targets (`df$a[2] <- v`,
  `l$v[2] <- v`, `names(x)[2] <- "b"`); `df[i, "col"] <- v`;
  `df$col <- 0` recycles to every row; `l[["name"]] <- v` and
  `l$x <- NULL` / `df$x <- NULL` remove.
- **`x <<- v` inside a function whose body is only that assignment
  changed nothing**: the JIT compiled it as a local assignment. It and
  `x[i] <<- v` now update the enclosing variable. The JIT also compiled
  the `NA` literal as 0; such bodies now run in the interpreter.
- **`matrix()` recycles its data and keeps NA**: `matrix(1, 2, 2)` was
  `1 0 0 0` and `matrix(c(1, NA, 3, 4), 2)` dropped the NA, shifting every
  later cell. `is.na()` works on a matrix; `cat()` and `paste()` print its
  cells.
- `cat()` puts `sep` between elements (`cat(c(1, 2), sep = ",")` is
  `1,2`) and no longer adds a newline of its own. `pmin`/`pmax` return NA
  for NA input unless `na.rm = TRUE` (they always dropped it).
- `sapply()` / `vapply()` simplify integer, logical and character results
  (they returned a list unless every result was a double) and name them
  by a character input. `tapply()` returns a named vector.
- `factor(labels =)` renames levels (one label per level, a single
  prefix numbered `L1 L2 ...`, or duplicates that merge levels);
  `factor(ordered = TRUE)` and new `is.ordered()`; an ordered factor
  prints `Levels: s < m < l`. `data.frame(stringsAsFactors = TRUE)` makes
  factors. `t.test`, `cor.test` and `fisher.test` name their estimate
  (`mean of x`, `mean in group a`, `cor`, `odds ratio`).
- **Printing matches R's layout**: vectors pad to a common width (text
  left-aligned, numbers right), `[k]` labels align, empty vectors print
  `numeric(0)` / `integer(0)` / `logical(0)` (all said `character(0)`);
  named vectors (now also logical and character ones) use one common
  width with R's spacing; a factor prints NA as `<NA>`. `table()` returns
  a table that prints like R's, with the variable name as its header,
  instead of printing 12-wide columns while being computed.
- Differential cases `vector_semantics` and `output/print_vectors`.

### Factor levels are sorted, as in R — 2026-09-23

- **`as.factor()` sorts its levels.** `levels(as.factor(c("b", "a", "b")))`
  is `"a" "b"` (it was `"b" "a"`, first-appearance order). **`factor()`
  sorts numbers numerically**: `factor(c(10, 2))` has levels `"2" "10"`
  (it sorted them as strings). Integer and logical vectors work in both.
  Number labels are `as.character`'s (`1e+05`, `0.333333333333333`).
- `levels = c(...)` keeps its order; values outside it, and NA, get NA
  codes; a duplicated level is an error. `factor(f)` drops unused levels
  and keeps their order; `as.factor(f)` returns `f` unchanged.
  `is.na()` works on a factor.
- **Grouping follows the level order and never makes an NA group:**
  `split()` (including empty levels of a factor), `tapply()` (an empty
  level is NA), `aggregate()`, `table()` (numbers sorted numerically, NA
  not counted), `boxplot(y ~ g)`. A factor's explicit level order is now
  honoured by all of them (`split` and `boxplot` re-sorted by string).
- **`lm()` with a character predictor uses the alphabetically first level
  as the reference**, like R (it used the first observed value, so the
  coefficients differed from R's). `t.test(y ~ g)` orders its two groups
  the same way, so the estimate order and the sign of `t` match R.
- Strings sort in R's collation order (see the entry above).
- Differential case `factor_levels` covers character and numeric input,
  NA, explicit `levels=` order and the grouping functions.

### What prints, and when, is R's — 2026-09-23

- **Visibility works like R's.** A value now carries whether it should
  auto-print, instead of the console guessing from the text of the line.
  `f <- function(v) print(v); f(1)` prints once (it printed twice);
  `h <- function() { x <- 3 }; h()` prints nothing; `(x <- 5)` prints 5;
  `invisible()`, `tryCatch`, `switch` and `local` behave as in R.
- **Hypothesis tests return their result and print nothing while
  computing.** `res <- chisq.test(m)` is silent, `res` or `print(res)`
  shows the report, and `chisq.test(m)$p.value` shows just the number (it
  printed the whole report first). Same for `t.test`, `cor.test`,
  `wilcox.test`, `fisher.test` and the other tests.
- **`sprintf` matches R**: vectorised, C's `%g`/`%e` rules, `Inf`/`NA`;
  **`print(x, digits = n)`** and **`format(x, digits = n)`** are honoured,
  `format(pi)` is `3.141593`, and scientific notation prints as R's
  (`3.333333e-09`).
- The differential harness compares numbers within its documented
  tolerance (it had been comparing any value with more than 6 digits as an
  exact string), and gains whole-output cases for what a script prints.

### Chi-squared matches R; `lower.tail` / `log.p` work — 2026-09-23

- **`pchisq`, `qchisq`, `dchisq` now agree with R 4.5.3 to 1.5e-13** on a
  492-value grid (261 values bit-identical). They were up to **10%** off:
  `pchisq(1101, 1000)` returned 1 (R: 0.98615), `qchisq(0.95, 10000)`
  returned 10100 (R: 10233.7), and df within 0.01 of 1 or 2 was treated
  as exactly 1 or 2. Rebuilt the way R's nmath does it.
- **`lower.tail = FALSE` and `log.p = TRUE` were silently ignored by every
  p-function** — `pnorm(10, lower.tail = FALSE)` returned the lower tail,
  1. Now honoured by `pnorm`, `pt`, `pf`, `pexp`, `pbinom`, `ppois`,
  `pchisq` and their q-functions, with each tail computed directly so a
  tiny p-value keeps its digits (`pchisq(300, 100, lower.tail = FALSE)` =
  7.41e-22; `pnorm(-40, log.p = TRUE)` = -804.61).
- `chisq.test` p-values come from the upper tail directly: a table whose
  p-value is 1.24e-279 reported 0.
- `df` is recycled like R's (`pchisq(3, c(1, 2, 5))` is three values);
  a missing `df` is an error; `ncp != 0` is an explicit "not implemented"
  error rather than a silently central answer.

### GPU training step (`r2-gpu`, `r2-train --features gpu`) — 2026-09-21/22

- `GpuTrainer` (`crates/r2-train/src/gpu_llm.rs`): the whole training step
  on the device — weights, activations, gradients and Adam state resident,
  tokens in, a loss out, one launch per op, no allocator in the loop. Built
  from a `Trainer` (`from_trainer`), hands its state back
  (`download_into`) for evaluation and checkpoints. Matches the tape's loss
  and every gradient block to f32 rounding; trains TinyStories to
  identical losses. `R2_GPU=1` in `tinystories_train`; `R2_GPU_STATS=<n>`
  prints a per-kernel census of step n; `R2_GPU_CKPT=1` activation
  checkpointing (one layer's work set, recomputed in the backward, same
  bits, +15% time).
- **1.34-1.37x faster than PyTorch-DirectML at 7M parameters and
  1.5-2.0x at 40M on the same integrated GPU**, learning identically —
  the reference run at its best (its `repeat_interleave` backward into
  SDPA measured 3.4 s per layer here and was replaced by an equal `cat`).
  `TS_DEVICE=dml` in `benchmarks/llm/tinystories_train.py`;
  `benchmarks/llm/dml_phase_split.py`. REPORT.md §4b.
- **The tiled attention backward** (`attn_tiled::dkv_wgsl`): dK and dV in
  one workgroup per key block, the score and dP tiles as register-blocked
  GEMMs, the result tiles held in registers for the whole query walk —
  **twice as fast as the two kernels it replaces** (6.55 vs 13.35 ms at
  32 x 64 tokens, 39.4 vs 77.4 at 8 x 256), 6-10% of a whole GPU step,
  same losses, still bit-reproducible. `R2_GPU_ATTN_TILED_BWD=0` selects
  the reference kernels, which remain correct for any head dimension.
- **Mixed precision** (`R2_GPU_PREC=mixed`): `Tensor` carries a storage
  type, every kernel compiles per type signature and accumulates in f32;
  f32 master weights and Adam, f16 weight copies and activations, dynamic
  loss scaling. Same losses; no speedup on an integrated GPU with no f16
  arithmetic units, as expected — built for hardware that has them. Its
  overflow test also catches a *saturated* f16 store (this adapter clamps
  to 65,504 rather than producing an infinity, which the textbook
  inf/nan check would miss).
- `r2-gpu`: deterministic split-K in `sgemm` (partial slabs summed in
  order) for outputs under 24 tiles — the short-output `grad_B` shapes
  53-110 -> 150-210 GFLOP/s; rmsnorm kernels re-laid out (8 rows per
  workgroup, dW reduced in row blocks), 50 -> 19 ms a step; the
  one-thread-per-row attention forward selected over the tiled one
  (measured faster here; the tiled kernel stays under test). Closed by
  measurement: a 64x64 tile, a register-prefetching slab loop, splitting
  attention's reduction axis.

- **GPU `sgemm`** (`r2-gpu`, `--features gpu`, wgpu 30): one WGSL kernel
  for the NN / NT / TN cases a training step runs — 128x128 workgroup
  tile, 8x8 register tile per thread, transposes handled where the slab
  is staged so the inner loop never strides — checked against the CPU
  `sgemm` at ragged shapes in every case and bit-reproducible run to run
  (every element summed in one thread, in one order). On this laptop's
  integrated Radeon it runs 330-425 GFLOP/s, 1.05-1.35x the six-core CPU
  kernel; `--example gemm_bench`. Device, queue and pipelines are created
  once per process; `Tensor` keeps buffers resident across calls.
- **Esc works like R's, in the GUI and the CLI.** At the prompt it
  abandons the line and any pending `+` continuation without evaluating
  anything; while a command runs it interrupts it — the statement
  unwinds, the rest of the submission is dropped, bindings made before
  the interrupt stay, and the console prints "interrupted — returning to
  prompt". The GUI is busy during evaluation, so Esc is read straight
  from the OS and only while an R2 window is in front.
- **`read.csv` reads the files Excel writes.** A CSV in the Windows code
  page (cp1252 — an accented letter as one byte) failed with "stream did
  not contain valid UTF-8". `read.csv`, `read.table`/`read.delim`,
  `readLines`, `source` and `r2 script.r2` now strip BOMs, recognise
  UTF-16, honour `fileEncoding=`, and take non-UTF-8 bytes as cp1252
  when unasked, as R on Windows does. No byte sequence is an error.
- **Fixed: the vectorised `exp` test failed on Apple Silicon** (the
  macOS CI job). The scalar `exp` fallback — the only path on aarch64 —
  produced subnormals where the AVX2 kernel flushes to zero; the two
  paths and the 8-wide loops' tails now agree.
- **Attention no longer slows down as the sequence gets longer or the
  batch gets smaller.** At dim 768 / seq 256 / batch 8 the medium model
  trained at parity with PyTorch (1.05x) while the small one led by
  1.38x; a full in-situ census of both sides found the whole gap in one
  kernel, attention backward (286 ms a step vs SDPA's 78). Two causes:
  the work was split across cores by sequence, so a batch of 8 on 6 cores
  ran at 67% utilisation and a batch of 1 ran serial; and the gradient
  accumulation read-modify-wrote three rows per (query, key) pair. Work
  units are now (sequence, head), and the accumulations are register
  blocked. On the way: the "AVX2+FMA" kernels had been emitting no FMA
  instructions at all (Rust never contracts `a*b + c`; they use
  `mul_add` now). Attention backward 286 -> 89 ms in situ, forward
  66 -> 34; medium model **1.15-1.27x faster than PyTorch** (48.3/43.4 vs
  55.5/55.2 s), small model **1.49-1.50x** at 100 steps. Held-out loss
  unchanged to four decimals on every pair. `benchmarks/llm/REPORT.md`.
- **`grad_B` no longer trails MKL on any shape from dim 768 up.** Kernel
  to kernel against PyTorch's own three GEMM calls
  (`benchmarks/llm/gemm_cases.py`, 23 shapes, both sides per shape in
  one window) the transposed-A case ran at 0.50-0.84x of MKL for every
  M <= 768 — the shape, not the transpose: the shared B panel per depth
  slab was re-fetched cold by every core when there were only eight
  row-blocks per fork. `sgemm` now has a second partition for that
  regime: A packed once for the whole depth, tasks that own column
  groups of C and pack their own B strips, three forks per call instead
  of three per slab, bit-identical results. TN median 0.87x -> 1.32x of
  MKL, NN 1.19x -> 1.37x, NT 1.16x -> 1.26x. Medium model **1.40-1.63x
  faster than PyTorch** (41.3/34.2 vs 57.8/55.9 s), small **1.66x**
  (36.3/35.9 vs 60.4/59.4 s at 100 steps); held-out loss unchanged.
- **`%*%` on doubles gets real fused multiply-adds.** The f64 GEMM's
  "AVX2+FMA" build emitted separate multiplies and adds (Rust never
  contracts `a*b+c`); `mul_add` under a const flag, 500x500 dgemm
  16-20 -> 19-27 GFLOP/s. Results agree with the SSE2 build to f64
  rounding rather than bit for bit.
- **In-situ censuses for any model size.** `R2_TAPE_STATS=1` now times
  the forward too (between tape pushes), `--example phase_split` prints
  GEMM vs non-GEMM milliseconds inside each phase, and
  `benchmarks/llm/op_profile.py` gives PyTorch's per-op self time forward
  and backward separately — the like-for-like table that found the
  attention deficit.

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
  Gradient buffers are recycled as well and are never zeroed: the
  backward tracks which gradients have been written, the first consumer
  assigns and later ones accumulate, so no buffer is memset or
  page-faulted. 300-step training 140.0 -> 131.6 s and 140.2 -> 131.8 s.
  The comparison moved from 1.10x to 1.23-1.29x across the off-thread
  free, the value pool and the gradient pool. Adam itself is 2.7x faster
  than PyTorch's.
- **Attention is tiled like a GEMM.** Both directions computed every
  score as a 64-long dot product with a horizontal reduction per
  (query, key) pair; they now pack Kᵀ/Vᵀ once per sequence and form 4x16
  score tiles lane-parallel, with the backward's dK/dV accumulation
  key-outer so each row is touched once per query block. At 2,048
  tokens the forward is now faster than PyTorch's fused
  `scaled_dot_product_attention` (1,136 vs 1,344 us) and the backward
  went 8.6 -> 6.4 ms. 300-step training 131.2 -> 128.1 s and 132.5 ->
  126.7 s; 1.29-1.38x vs PyTorch across five pairs.
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
