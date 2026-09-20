"""The three GEMM cases of a layer's forward and backward, R2 against
PyTorch, kernel to kernel, over a sweep of shapes and sizes.

PyTorch's `MmBackward0` computes `grad_A = grad.mm(B.t())` and
`grad_B = A.t().mm(grad)`: two more calls to `aten::mm` on stride-only
transposed views, which MKL's `sgemm` takes as `transa`/`transb`. R2's
tape does the same through `sgemm(.., Trans::No, .., Trans::Yes)` (NT) and
`(Trans::Yes, Trans::No)` (TN). So the like-for-like comparison is those
three calls, at the same shape, same thread count, same window — which is
what this runs: for every shape, PyTorch's three then R2's three
(`--example gemm_cases`, one process per shape, started right after), so
neither side's numbers come from a different thermal state than the
other's.

Ratios above 1 are R2 ahead. The last table is the point: how the ratio
MOVES with shape — a kernel that is 1.3x ahead at one size and 0.8x at
another has a structural problem at the second, whatever the average.

    cargo build --release -p r2-linalg --example gemm_cases
    python benchmarks/llm/gemm_cases.py
"""
import os, statistics, subprocess, sys, time
import torch

EXE = os.path.join("target", "release", "examples", "gemm_cases" + (".exe" if os.name == "nt" else ""))
torch.set_num_threads(6)

# (tokens, k, n, label). k is the layer's input width, n its output width.
SHAPES = [
    # the small model (dim 256) at 2,048 tokens
    (2048, 256, 256, "dim256 q/o"), (2048, 256, 128, "dim256 k/v"), (2048, 256, 768, "dim256 w1/w3"),
    (2048, 768, 256, "dim256 w2"), (2048, 256, 8000, "dim256 head"),
    # the medium model (dim 768)
    (2048, 768, 768, "dim768 q/o"), (2048, 768, 256, "dim768 k/v"), (2048, 768, 2304, "dim768 w1/w3"),
    (2048, 2304, 768, "dim768 w2"), (2048, 768, 8000, "dim768 head"),
    # larger widths, same token count
    (2048, 1024, 4096, "dim1024 w1/w3"), (2048, 4096, 1024, "dim1024 w2"), (2048, 2048, 2048, "dim2048 q/o"),
    (2048, 2048, 8192, "dim2048 w1/w3"), (2048, 8192, 2048, "dim2048 w2"),
    # fewer and more tokens
    (512, 768, 768, "dim768 q/o @512"), (512, 768, 2304, "dim768 w1/w3 @512"), (512, 2304, 768, "dim768 w2 @512"),
    (8192, 768, 768, "dim768 q/o @8192"), (8192, 768, 2304, "dim768 w1/w3 @8192"), (8192, 2304, 768, "dim768 w2 @8192"),
    (8192, 256, 768, "dim256 w1/w3 @8192"), (8192, 768, 256, "dim256 w2 @8192"),
]
if os.environ.get("R2_QUICK"):
    SHAPES = SHAPES[:5]

def med_us(flops, fn):
    fn()
    reps = int(max(2, min(200, (0.25 * 100e9) / flops)))
    for _ in range(min(reps, 3)): fn()
    out = []
    for _ in range(7):
        s = time.perf_counter()
        for _ in range(reps): fn()
        out.append((time.perf_counter() - s) / reps * 1e6)
    return statistics.median(out)

def r2_cases(t, k, n):
    out = subprocess.run([EXE, str(t), str(k), str(n)], capture_output=True, text=True, check=True).stdout
    return [float(x) for x in out.split()]

print(f"GEMM cases, R2 vs PyTorch {torch.__version__} (MKL), {torch.get_num_threads()} threads — us per call, GFLOP/s, ratio (>1 = R2 ahead)\n")
hdr = f"{'shape':<22} {'t x k x n':>16} {'case':>7} {'R2 us':>9} {'torch us':>9} {'R2 GF/s':>8} {'MKL GF/s':>8} {'ratio':>6}"
print(hdr); print("-" * len(hdr))
ratios = {"NN": [], "NT": [], "TN": []}
rows = []
for t, k, n, label in SHAPES:
    flops = 2.0 * t * k * n
    a = torch.randn(t, k); b = torch.randn(k, n); g = torch.randn(t, n)
    c = torch.empty(t, n); da = torch.empty(t, k); db = torch.empty(k, n)
    with torch.no_grad():
        torch_us = [
            med_us(flops, lambda: torch.mm(a, b, out=c)),          # forward   NN
            med_us(flops, lambda: torch.mm(g, b.t(), out=da)),     # grad_A    NT
            med_us(flops, lambda: torch.mm(a.t(), g, out=db)),     # grad_B    TN
        ]
    r2_us = r2_cases(t, k, n)
    for case, ru, tu in zip(("NN", "NT", "TN"), r2_us, torch_us):
        ratio = tu / ru
        ratios[case].append((ratio, label))
        rows.append((label, t, k, n, case, ru, tu, ratio))
        print(f"{label if case == 'NN' else '':<22} {f'{t}x{k}x{n}' if case == 'NN' else '':>16} {case:>7} "
              f"{ru:>9.0f} {tu:>9.0f} {flops / ru / 1e3:>8.1f} {flops / tu / 1e3:>8.1f} {ratio:>6.2f}")
    sys.stdout.flush()

print("\nconsistency — ratio by case over the sweep (>1 = R2 ahead):")
print(f"{'case':>7} {'min':>6} {'median':>7} {'max':>6}   worst shape")
for case, v in ratios.items():
    v.sort()
    print(f"{case:>7} {v[0][0]:>6.2f} {statistics.median(x for x, _ in v):>7.2f} {v[-1][0]:>6.2f}   {v[0][1]}")
