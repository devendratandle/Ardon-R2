"""R2's LAPACK-level routines against MKL's (PyTorch float64), per routine
and size, in one window: MKL first, then `--example lapack_cases` right
after. Ratios above 1 are R2 ahead.

    cargo build --release -p r2-linalg --example lapack_cases
    python benchmarks/lapack_cases.py            # LAPACK_SIZES=200,500,1000

Like for like: getrf <-> torch.linalg.lu_factor, potrf <-> cholesky,
geqrf <-> torch.geqrf, syev (with vectors) <-> eigh, gesvd (values only)
<-> svdvals.
"""
import os, statistics, subprocess, time
import torch

EXE = os.path.join("target", "release", "examples", "lapack_cases" + (".exe" if os.name == "nt" else ""))
torch.set_num_threads(int(os.environ.get("TS_THREADS", "6")))
SIZES = [int(x) for x in os.environ.get("LAPACK_SIZES", "200,500,1000").split(",")]
ROUTINES = os.environ.get("LAPACK_ROUTINES", "getrf,potrf,geqrf,syev,gesvd").split(",")

def med_ms(fn):
    fn()
    t, start = [], time.perf_counter()
    while len(t) < 3 or (time.perf_counter() - start < 1.0 and len(t) < 50):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1e3)
    return statistics.median(t)

print(f"LAPACK, R2 vs PyTorch {torch.__version__} float64 (MKL), {torch.get_num_threads()} threads — ms per call\n")
hdr = f"{'routine':<8} {'n':>6} {'R2 ms':>10} {'MKL ms':>10} {'ratio':>7}"
print(hdr); print("-" * len(hdr))
for r in ROUTINES:
    for n in SIZES:
        g = torch.rand(n, n, dtype=torch.float64) * 2 - 1
        gen = g + torch.eye(n, dtype=torch.float64) * n * 0.5
        sym = 0.5 * (g + g.T)
        spd = g.T @ g + torch.eye(n, dtype=torch.float64) * n
        fn = {"getrf": lambda: torch.linalg.lu_factor(gen),
              "potrf": lambda: torch.linalg.cholesky(spd),
              "geqrf": lambda: torch.geqrf(gen),
              "syev":  lambda: torch.linalg.eigh(sym),
              "gesvd": lambda: torch.linalg.svdvals(gen)}[r]
        with torch.no_grad():
            tm = med_ms(fn)
        rm = float(subprocess.run([EXE, r, str(n)], capture_output=True, text=True, check=True).stdout)
        print(f"{r:<8} {n:>6} {rm:>10.2f} {tm:>10.2f} {tm / rm:>7.3f}", flush=True)
