# R2 Graphics — Pending Work

**Last revised:** v0.3.4 (graph-gallery review).
**Status:** what shipped vs. what's deferred to the next session(s).
**Note (v0.3.4):** item **A is DONE** — `bi_hist` / `bi_boxplot` / `bi_barplot`
now all call `render_chrome` + `render_axis_ticks` with the full `LabelOpts`,
so `sub` / `cex.*` / `font.*` / `col.*` / `las` are honoured on every plot
type (verified via `samples/graph_gallery.r2`). Remaining open: boxplot /
barplot **per-bar fill `col=`** (boxes render black even when `col=c(...)`
is given), plus items B–G below.

This file exists so we don't lose track of the graphics features the
user explicitly asked for. Don't ship v0.2 / v0.3 without crossing
these off (or at least re-prioritising them with the user).

---

## ✅ Done in this session (v0.1.9)

Applied to `bi_plot` in `crates/r2-graphics/src/plots.rs`:

| Argument | Effect |
|---|---|
| `main`        | Plot title (existing — kept) |
| `sub`         | **NEW.** Subtitle below the panel (under xlab) |
| `xlab`, `ylab`| Axis labels (existing — kept) |
| `cex.main`    | **NEW.** Title font scale |
| `cex.sub`     | **NEW.** Subtitle font scale |
| `cex.lab`     | **NEW.** Axis-label font scale |
| `cex.axis`    | **NEW.** Tick-label font scale |
| `font.main` / `font.sub` / `font.lab` / `font.axis` | **NEW.** 1 = plain, 2 = bold, 3 = italic, 4 = bold-italic (R-faithful encoding) |
| `col.main` / `col.sub` / `col.lab` / `col.axis`     | **NEW.** Per-element text color |
| `las` (0/1/2/3) | **NEW.** Axis-tick rotation: 0 parallel (default), 1 always horizontal, 2 perpendicular, 3 always vertical |
| `cex` (point scale) | **NEW.** Now scales point radius proportionally |

Shared helpers in the same file:
- `LabelOpts::from_args` — pulls `par()` defaults from the device,
  applies per-call overrides.
- `PanelRect`, `render_chrome`, `render_axis_ticks` — reusable so the
  other plot functions can be migrated without duplicating SVG code.
- `font_attrs` and `escape_xml` helpers.

---

## ⏳ Deferred to next sessions

### A. Wire the same chrome into the other plot functions ✅ DONE (v0.3.4)

- [x] `bi_hist`     — uses `LabelOpts::from_args` + `render_chrome` + `render_axis_ticks`
- [x] `bi_boxplot`  — same
- [x] `bi_barplot`  — same
- [x] `bi_lines` / `bi_points` / `bi_abline` — respect device `col`/`lty`/`lwd`/`cex`

So `sub` / `cex.*` / `font.*` / `col.*` / `las` are now honoured on every
plot type. **Residual gap:** `bi_boxplot` / `bi_barplot` ignore a per-element
**fill `col=`** — boxes/bars render black even when `col=c("…","…")` is
supplied (only the chrome text colours use `col.*`). Fixing this means
threading the fill colour vector into the box/bar SVG rect emit. Small, but
open.

### B. Default labels from variable names (R-faithful)

R's `plot(x, y)` uses `xlab = "x"` and `ylab = "y"` not because they're
the variable names but because of `deparse(substitute(x))`. To match,
the engine needs to thread the un-evaluated AST node of each argument
into the builtin call so the builtin can render `deparse(node)` as the
default. Currently we use the string `"x"` and `"y"` as a poor proxy.

Files to touch:
- `crates/r2-engine/src/lib.rs` — `EvalArg` could grow an
  `Option<Expr>` field for the original unevaluated AST.
- `bi_plot` etc. — when `xlab`/`ylab` is not supplied, deparse the
  positional arg.

### C. Additional axis-control args

Not yet wired:

- [ ] `xlim` / `ylim` — explicit axis ranges (override the data min/max).
- [ ] `xaxt` / `yaxt` (`"n"` to suppress axis) — common idiom for
       building custom axes with `axis()`.
- [ ] `log = "x"`, `log = "y"`, `log = "xy"` — log-scale axes.
- [ ] `tck` / `tcl` — tick mark length.
- [ ] `mgp` — three-component vector for label / axis / line margins.
- [ ] `axis()` builtin — explicit tick placement (currently only the 5
       auto-ticks emitted by `render_axis_ticks` exist).

### D. Text utilities

- [ ] `text(x, y, labels, cex=, srt=, adj=, col=, font=)` — arbitrary
       text at data coords with rotation (`srt`) and alignment (`adj`).
- [ ] `mtext(text, side=, line=, ...)` — text in the margins.
- [ ] `title(main=, sub=, xlab=, ylab=)` — add chrome to an existing
       plot. Trivial wrapper around `render_chrome` once the device
       remembers the last panel rect (currently it doesn't).

### E. Legend

- [ ] `legend("topright", legend=c("A","B"), lty=, col=, pch=, cex=, bty=, title=)`
       Most-requested missing feature for scatter / line plots.

### F. Colors

- [ ] Named-color table (`"red"`, `"steelblue"`, …) — currently only
       hex / `rgb()` works in some paths. R has ~657 named colors.
- [ ] `rgb()`, `hsv()`, `col2rgb()`, `adjustcolor()` — already partial,
       finalize.

### G. Plot types not yet implemented

- [x] `pairs()` — scatter matrix (v0.3.2; outer-edge axis tick labels +
       shared `las`/`col.axis`/`cex.axis` params added in v0.3.6).
- [ ] `image()` — heatmap.
- [ ] `contour()` — contour plot.
- [ ] `persp()` — 3D surface.

These are larger pieces; postpone until A–E are done.

---

## Recommended next-session order

1. **A** — propagate `render_chrome` to `bi_hist`/`bi_boxplot`/`bi_barplot`. ~30 min.
2. **D** + **B** — `title()` + default-from-deparse. ~1 hour.
3. **E** — `legend()`. ~1 hour.
4. **C** — `xlim`/`ylim`/`log`/`xaxt`. ~1 hour.
5. **F** — named-color table (paste from R's source).
6. **G** — case-by-case.

Whole list is ~1–2 sessions of focused work depending on token budget.

---

## v0.1.9 Snapshot

What works **right now** for the user:

```r
plot(1:10, (1:10)^2,
     main = "Squares", sub = "demo",
     xlab = "n", ylab = "n²",
     cex.main = 1.5, font.main = 2,    # bold title
     cex.lab = 1.2, col.lab = "navy",
     col.axis = "gray40", cex.axis = 0.9,
     las = 1)                          # horizontal y-axis labels
```

Everything in that call respected. As of v0.3.6 the same graphical params
(`las`, `col.axis`/`cex.axis`, `mar`/`par(mar)`, and the title/label family)
are honored on `hist()` / `boxplot()` / `barplot()` / `matplot()` / `pairs()`
too, not just `plot()` — item **A** has landed.
