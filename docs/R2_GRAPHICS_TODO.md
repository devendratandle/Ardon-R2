# R2 Graphics — Pending Work

**Last revised:** during v0.1.9 build (post-R2-UI refactor).
**Status:** what shipped vs. what's deferred to the next session(s).

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

### A. Wire the same chrome into the other plot functions

`bi_plot` was migrated to `render_chrome` / `render_axis_ticks`. Still
emitting the **old** label code (no `sub` / `cex.*` / `font.*` /
`col.*` / `las`):

- [ ] `bi_hist`     (`crates/r2-graphics/src/plots.rs`)
- [ ] `bi_boxplot`  (same file)
- [ ] `bi_barplot`  (same file)
- [ ] `bi_lines`, `bi_points`, `bi_abline` (overlays — `crates/r2-graphics/src/overlays.rs`) — these don't draw labels but should respect the device-level `col`/`lty`/`lwd`/`cex` updates already in `PlotParams`.

Pattern to follow for each function: read args, call `LabelOpts::from_args(a)`,
build a `PanelRect`, call `render_chrome(...)` + `render_axis_ticks(...)`.

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

- [ ] `pairs()` — scatter matrix.
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

Everything in that call respected. The same args on `hist()` /
`boxplot()` / `barplot()` are accepted (R passes-through) but only
`main`/`xlab`/`ylab` are honored until item **A** above lands.
