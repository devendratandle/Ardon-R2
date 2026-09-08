# R2 Vision — The Path to Green AI

## What R2 Is

R2 is a statistical computing language inspired by R, built from scratch in Rust.
It is NOT an R package, NOT a wrapper, NOT a binding. It is a new language that
speaks R's syntax but runs at Rust's speed.

## The Problem

Modern data science wastes enormous computational resources:

- Python calls C calls Fortran calls CUDA — 4 language boundaries, each with overhead
- R's interpreter executes 50-100x more instructions than necessary
- Installing a typical ML stack: 2-8 GB of dependencies
- Cloud computing burns electricity on interpreter overhead

## R2's Answer

**One language. Minimal glue. Direct to hardware.**

### V0.4.0 — Current Release (2026-09)
- **438 built-in functions** across 29 crates — no packages to install
- Accuracy **matches CRAN R on 13/13 differential cases**, and R2's LLM
  gradients match an independent float64 implementation to f32 rounding
  across all 21 parameter blocks
- **LLM training at PyTorch's speed**: a 7.24M-parameter model on
  TinyStories trains **1.05x behind PyTorch 2.13+MKL** and learns
  *identically* — same loss to four decimals at every checkpoint over 500
  steps from shared initial weights. See `benchmarks/llm/REPORT.md`
- **BPE tokenizer ~11x faster than HuggingFace `tokenizers`** — which is
  itself Rust, so that is an algorithmic win, not a language one
- Pure-Rust `sgemm`: blocked/packed Goto-BLIS with a hand-written AVX2
  micro-kernel selected at *runtime*, so one binary ships everywhere
- 12 ML algorithms built in; `lm`/`glm`/`aov`, MANOVA, Hotelling's T²
- Full thin SVD with U and Vᵀ; Householder + Wilkinson-shift QR eigendecomp
- Cranelift JIT with branchy multi-block IR + 3-arg ternary ABI
- RFC 4180 CSV parser; `regex-lite` regex engine; NA-aware `&`/`|`
- Columnar memory layer with mmap-backed reader
- Native desktop GUI (`R2Gui`), graphics to SVG/PNG/PDF
- Runs on x86_64 and ARM (Windows, Linux, macOS)
- **638 tests passing** across 71 binaries, clean build
- Rust-only dependencies, no C/C++/Fortran anywhere
- AGPL v3

### V1.0 — Stability Release
- Bug fixes from community feedback
- More statistical tests and distributions
- Expanded help system and documentation
- Smart auto-parallelism based on data size

### V1.5 — Performance Release
- Rayon parallelism expanded to gbm, cv, kmeans
- Memory-mapped CSV for large file reading
- Expected 4-8x speedup on multi-core workloads

### V2.0 — Bytecode VM (Game-Changer)
- R2 bytecode compiler for user-written functions
- User functions run 10-20x faster than R
- Write in R2 syntax, execute at compiled speed
- .Internal() calls Rust for heavy math — zero overhead
- Community can build statistical packages in R2 — no Rust needed
- This is the release that makes R2 a true language, not just a tool

### V2.5 — Big Data
- Columnar storage engine (inspired by Arrow, built in Rust)
- Process datasets larger than RAM
- Chunked filter/select/aggregate on disk
- Memory-efficient data pipelines

### V3.0 — Universal Compute
- Hardware-accelerated compute for supported platforms
- Distributed computing for cluster deployments
- R2 as a complete data science runtime

## Green AI Impact

| Metric | Traditional Stack | R2 Target |
|---|---|---|
| Install size | 2-8 GB | ~9 MB |
| Language boundaries | 3-5 | 1 |
| Interpreter overhead | 50-100x | 0x (compiled) |
| User function speed | baseline (R) | 10-20x faster (bytecode VM) |
| Dependency downloads | hundreds of packages | Rust-only |
| Energy per computation | baseline | 50-70% less |

## Technical Principles

1. **Rust-only dependencies** — no C, C++, or Fortran libraries
2. **No glue code** — one language from script to hardware
3. **Correct first, fast second** — numerical accuracy is non-negotiable
4. **Open source (AGPL v3)** — community-driven development
5. **Green by design** — less overhead = less energy = less carbon

## R2 Roadmap Summary

```
DONE       V0.1-0.3 →  Full SVD, branchy JIT, RFC 4180 CSV, regex, columnar
                       storage, graphics backends, native GUI, self-check.
NOW        V0.4.0  →  LLM training at PyTorch's speed on CPU: packed
                       AVX2 sgemm, fused attention, vectorised exp.
                       438 builtins, 638 tests.
Next       V0.5.0  →  Out-of-core training data (Arrow/Parquet + memmap),
                       so a corpus larger than RAM trains without a
                       preprocessing step. Closing the remaining sgemm
                       parallel efficiency (2.7-3.8x against a measured
                       5.53x machine ceiling) and the memory-traffic tail.
Later      V1.0    →  Stability release. Community feedback baked in.
           V2.0    →  Hardware awareness (cores/ISA/cache), Oracle
                       calibration, GPU dispatcher (WGPU).
           V2.5    →  Bytecode VM. User functions JITed at built-in speed.
           V3.0    →  Universal compute. Distributed processing.
```

Dated milestones ("Month 1", "Month 3") were removed rather than
recalculated: they were written against a 2026-05 start and every one of
them had passed while the file still called V0.1.0 the current release.

## Created By

Devendra Tandale
An AI assisted project
License: AGPL v3
