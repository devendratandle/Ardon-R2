"""R2's f64 `dgemm` against MKL's, kernel to kernel, per shape in one window.

`dgemm` is the column-major BLAS entry point behind `%*%` and the blocked
LAPACK routines. PyTorch's float64 `mm` is MKL `dgemm`; for every shape
this times MKL, then runs `--example dgemm_cases` for R2 right after, so
the two numbers share a thermal state. Ratios above 1 are R2 ahead.

    cargo build --release -p r2-linalg --example dgemm_cases
    python benchmarks/dgemm_cases.py
"""
import os, statistics, subprocess, time
import torch

EXE = os.path.join("target", "release", "examples", "dgemm_cases" + (".exe" if os.name == "nt" else ""))
torch.set_num_threads(int(os.environ.get("TS_THREADS", "6")))

# (m, k, n, label): C (m x n) = A (m x k) · B (k x n)
SHAPES = [
    (128, 128, 128, "square 128"), (256, 256, 256, "square 256"), (512, 512, 512, "square 512"),
    (1024, 1024, 1024, "square 1024"), (2000, 2000, 2000, "square 2000"),
    (10000, 100, 100, "X %*% W, tall"), (100, 10000, 100, "t(X) %*% Y, deep"),
    (5000, 500, 500, "X %*% W, wide"), (500, 5000, 500, "t(X) %*% Y 500"),
    (1000, 1000, 50, "A %*% few cols"), (200, 200, 20000, "short-wide"),
]

def med_us(flops, fn):
    fn()
    reps = int(max(2, min(2000, (0.25 * 50e9) / flops)))
    for _ in range(min(reps, 3)): fn()
    out = []
    for _ in range(7):
        s = time.perf_counter()
        for _ in range(reps): fn()
        out.append((time.perf_counter() - s) / reps * 1e6)
    return statistics.median(out)

print(f"dgemm, R2 vs PyTorch {torch.__version__} float64 (MKL), {torch.get_num_threads()} threads\n")
hdr = f"{'shape':<18} {'m x k x n':>18} {'R2 us':>10} {'MKL us':>10} {'R2 GF/s':>8} {'MKL GF/s':>8} {'ratio':>6}"
print(hdr); print("-" * len(hdr))
ratios = []
for m, k, n, label in SHAPES:
    flops = 2.0 * m * k * n
    # column-major A (m x k) is a row-major (k x m) buffer; any layout does for timing
    a = torch.randn(m, k, dtype=torch.float64); b = torch.randn(k, n, dtype=torch.float64)
    c = torch.empty(m, n, dtype=torch.float64)
    with torch.no_grad():
        tu = med_us(flops, lambda: torch.mm(a, b, out=c))
    ru = float(subprocess.run([EXE, str(m), str(k), str(n)], capture_output=True, text=True, check=True).stdout)
    ratios.append(tu / ru)
    print(f"{label:<18} {f'{m}x{k}x{n}':>18} {ru:>10.0f} {tu:>10.0f} {flops / ru / 1e3:>8.1f} {flops / tu / 1e3:>8.1f} {tu / ru:>6.2f}", flush=True)

print(f"\nratio (>1 = R2 ahead): min {min(ratios):.2f}  median {statistics.median(ratios):.2f}  max {max(ratios):.2f}")
