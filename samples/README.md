# Ardon-R2 — sample programs

Runnable programs you can use to **test Ardon-R2 on your own machine** and
cross-check it against R. If any program errors or any number disagrees
with R, please [open an issue](https://github.com/devendratandle/Ardon-R2/issues) —
that feedback is exactly what hardens the project.

| File | What it does | How to run |
|------|--------------|-----------|
| **`smoke.r`** | Core smoke test — descriptive stats, data frames, `lm`/`glm`/`t.test`, linear algebra, and timing. **Written in the common R/R2 subset so the *same file* runs on both engines** — run it on each and compare the output side by side. | `r2 samples/smoke.r` and `Rscript samples/smoke.r` |
| **`capabilities.r2`** | A tour of the R2-specific surface — dates, time series (`ts`/`acf`/`diff`), language objects (`quote`/`eval`/`deparse`), built-in ML (`kmeans`/`rf`/`rpart`), and factors. Self-checking (`[PASS]`/`[FAIL]`). | `r2 samples/capabilities.r2` |
| **`graph_gallery.r2`** | Renders one colourful, labelled example of every plot type (scatter+fit, line, bar, histogram, boxplot, pie) and saves a PNG per plot into [`../screenshots/`](../screenshots). | `r2 samples/graph_gallery.r2` |
| **`packages/`** | Example of the R2 package system — a small add-on package (`mymath`) and an install demo. | see `packages/` |

**Performance & accuracy in depth:** the rigorous R-vs-R2 comparison
harness (separate, tuned `.R`/`.r2` files for each workload) lives in
[`../benchmarks/`](../benchmarks); the measured results are summarised in
[`../PERFORMANCE.md`](../PERFORMANCE.md).
