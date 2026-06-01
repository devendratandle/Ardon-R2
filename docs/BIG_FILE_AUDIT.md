# Big-File Audit — files > 800 LoC

**Policy:** ≤ 800 LoC *where practical*. A single cohesive algorithm
(one JIT pipeline, one decomposition family, one wgpu pipeline) is
**justified large** — splitting it mid-logic hurts readability more
than it helps. The real targets are **multi-domain grab-bags**.

Generated during the opus-4.8 engine-split session (Monday). Method
for any SPLIT below: **marker-first** — read once, drop unique
sentinel comments at every cut boundary + pre-audit each new module's
`use` header and any `pub(crate)` promotions into this doc, then
execute mechanically with one build-verify per file. (Content-anchored
perl-slurp; greedy `.*` with `/s`, never explicit `\n` — the tree is
CRLF.)

| File | LoC | Verdict | Rationale |
|---|---|---|---|
| `r2-jit/src/lib.rs` | 2604 | **KEEP** | One Cranelift codegen pipeline (errors→handle→compiler→lowering→externs). Interconnected; reviewers expect a JIT to be one big focused file. |
| `r2-engine/src/lib.rs` | 2402 | **DEFER** | Now the genuine eval core after this session's split. Further reduction = dividing the `impl Engine` eval dispatcher across files — structural, delicate (arms share private state). Own focused pass. |
| `r2-kernel/src/lib.rs` | 1943 | **KEEP** | Parallel-compute kernel: serial backend + Rayon backend + dispatcher for the same op set. One cohesive subsystem. |
| `r2-types/src/lib.rs` | 1844 | **DEFER** | RVal enum + element types + Reals/Singles storage + error types + format/coerce. Foundational; every crate depends on it. Splittable into `types/{val,format,coerce,error}.rs` but high blast radius — do carefully, not under time pressure. |
| `r2-stats/src/time.rs` | 1826 | **SPLIT** ✅ | **True two-domain grab-bag:** Gregorian date/time-series machinery (Hinnant day↔ymd, strftime, ts/xts builtins) + the Hindu calendar (tithi, hnc.date, Saka era, sankranti). Clean seam. **This session's target.** |
| `r2-arrow/src/lib.rs` | 1364 | KEEP | Arrow columnar memory layer — one data-format subsystem. |
| `r2-stats/src/htest.rs` | 1242 | KEEP | Hypothesis-test family (t/chisq/shapiro/wilcox/fisher/cor) sharing fmt_pval/signif_stars. Cohesive. Light split possible later if it keeps growing. |
| `r2-linalg/src/decomp.rs` | 1143 | KEEP | Matrix decomposition family (LU/QR/SVD/Cholesky/eigen). One numerical-methods file; reviewers expect this. |
| `r2-stats/src/models.rs` | 1128 | KEEP | lm/glm/aov model fitting — one regression subsystem. |
| `r2-stats/src/multivariate.rs` | 1038 | LIGHT-SPLIT (low pri) | Hotelling (1/2/paired) + MANOVA + dispatcher. Could become `multivariate/{hotelling,manova}.rs` but all share the small helpers; modest gain. |
| `r2-gui/src/main.rs` | 922 | DEFER | One big `on_frame` closure driving the MDI desktop. Hard to split meaningfully without inventing artificial seams; the closure is the app. |
| `r2-ui/src/render.rs` | 899 | KEEP | wgpu pipeline (atlas + shader + frame builder). One GPU subsystem. |
| `r2-ml/src/dispatch.rs` | 888 | LIGHT-SPLIT (low pri) | Per-model dispatch (rpart/rf/gbm/kmeans/knn/nb/prcomp/cv). Could split per model-family; modest gain. |
| `r2-engine/src/builtins/misc.rs` | 832 | ACCEPT | Just created this session as the trailing grab-bag. At the 800 line; could sub-split (save-load / aov / .Internal) but low value — it's a leaf module nobody navigates deeply. |

## Verdict summary
- **SPLIT now:** `time.rs` (1 file, clean two-domain seam).
- **DEFER (structural, own pass):** `engine/lib.rs` eval dispatcher, `types/lib.rs`.
- **LIGHT-SPLIT (optional, low priority):** `multivariate.rs`, `ml/dispatch.rs`.
- **KEEP (justified large):** jit, kernel, arrow, htest, decomp, models, render — cohesive single-subsystem files. Splitting hurts.

The headline number for a Foundation reviewer: after the engine split,
**no file is a multi-domain grab-bag except `time.rs`** (being fixed),
and the remaining large files are each a single, named, cohesive
subsystem — which is exactly what a well-factored numerical stack
looks like.

## time.rs split — DONE

time.rs (1826) -> time/mod.rs (1447, single-domain date/ts/xts) +
time/hindu.rs (398, tithi/hnc.date/saka). Child module reaches
parent helpers via super::; zero promotions needed. Builtins still
resolve at r2_stats::time::* via `pub use hindu::*` in mod.rs.
Verified: tithi()/hnc.date() runtime-correct.

## (original plan kept for reference)
Target: `r2-stats/src/time/` directory module
- `time/mod.rs` — re-exports, shared day↔ymd (Hinnant), strftime helpers
- `time/series.rs` — ts() / xts() / period aggregation builtins
- `time/hindu.rs` — tithi(), hnc.date(), Saka era, sankranti, Adhik Maas

Pre-audit (imports/promotions) to be filled in as the split runs.
