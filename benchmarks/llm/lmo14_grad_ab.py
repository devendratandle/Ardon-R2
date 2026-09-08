"""LMO-13/14 — grad_A and grad_B, R2 against PyTorch AND JAX.

These two ARE the backward: of forward + grad_A + grad_B, they are about
80% between them. Both were once judged only against R2's own past — one
was marked "DONE" on that basis and had never been compared to anything.
They are now BLAS sgemm's NT and TN cases and 2.6x behind the best
reference, weighted by calls per step. See `REPORT.md` section 3.

METHOD, identical on all three sides: one operand requires grad and the
other does not, so exactly one of the two gradients is computed, and each
is reported as (fwd+bwd) − (fwd) measured separately.

    cargo run --release -p r2-train --example lmo14_grad_ab
    python benchmarks/llm/lmo14_grad_ab.py
"""

import json
import statistics
import time

import torch

try:
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except ImportError:                                    # pragma: no cover
    HAVE_JAX = False


def med_us(reps, fn):
    for _ in range(min(reps, 3)):
        fn()
    out = []
    for _ in range(9):
        s = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - s) / reps * 1e6)
    return statistics.median(out)


def main():
    torch.set_num_threads(6)
    try:
        r2 = json.load(open("lmo14_r2.json"))
    except FileNotFoundError:
        print("lmo14_r2.json not found — run the R2 half first:")
        print("  cargo run --release -p r2-train --example lmo14_grad_ab")
        return

    print("LMO-13/14 — grad_A and grad_B, PyTorch and JAX references")
    print(f"  torch {torch.__version__}, {torch.get_num_threads()} threads", end="")
    print(f", jax {jax.__version__} on {jax.default_backend()}" if HAVE_JAX else ", no jax")
    print()

    hdr = (f"{'block':>14} {'m x k x n':>18} {'grad':>7} {'R2 us':>10} {'torch':>10}"
           f" {'jax':>10} {'best':>10} {'R2/best':>9}")
    print(hdr)
    print("-" * len(hdr))

    worst = 0.0
    tot = {"r2": 0.0, "best": 0.0}
    for row in r2["rows"]:
        m, k, n, cnt = row["m"], row["k"], row["n"], row["count"]
        reps = max(2, min(20, int(4e9 / (m * k * n))))
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        g = torch.randn(m, n)

        with torch.no_grad():
            t_fwd = med_us(reps, lambda: a @ b)

        ag = a.clone().requires_grad_(True)

        def fb_a():
            ag.grad = None
            (ag @ b).backward(g)

        bg = b.clone().requires_grad_(True)

        def fb_b():
            bg.grad = None
            (a @ bg).backward(g)

        pt_a = max(0.0, med_us(reps, fb_a) - t_fwd)
        pt_b = max(0.0, med_us(reps, fb_b) - t_fwd)

        jx_a = jx_b = None
        if HAVE_JAX:
            ja, jb, jg = (jnp.asarray(x.numpy()) for x in (a, b, g))
            # NO forward subtraction here. `jax.vjp(...)[1](g)` is the
            # pullback ALONE — under jit the forward is dead code for a
            # matmul (the cotangent needs only the other operand), so it
            # is never run. Subtracting a forward that did not happen is
            # what made an earlier version of this file print 0.0 us for
            # a 268 MFLOP gradient. torch keeps its subtraction because
            # `(a @ b).backward(g)` really does re-run the forward.
            va = jax.jit(lambda x, y, gg: jax.vjp(lambda p: p @ y, x)[1](gg))
            vb = jax.jit(lambda x, y, gg: jax.vjp(lambda p: x @ p, y)[1](gg))
            jax.block_until_ready(va(ja, jb, jg))
            jax.block_until_ready(vb(ja, jb, jg))
            jx_a = med_us(reps, lambda: jax.block_until_ready(va(ja, jb, jg)))
            jx_b = med_us(reps, lambda: jax.block_until_ready(vb(ja, jb, jg)))

        # Sanity floor: each gradient is 2*m*k*n FLOPs. Anything faster
        # than 400 GFLOP/s on a 6-core CPU is a measurement error, not a
        # result, and must not silently become the bar R2 is judged by.
        flop = 2.0 * m * k * n
        for nm, val in (("torch grad_A", pt_a), ("torch grad_B", pt_b),
                        ("jax grad_A", jx_a), ("jax grad_B", jx_b)):
            if val is not None and val > 0 and flop / (val * 1e-6) / 1e9 > 400:
                print(f"  !! {nm} at {m}x{k}x{n} implies "
                      f"{flop / (val * 1e-6) / 1e9:.0f} GFLOP/s — not physical, "
                      f"treating as unmeasured")
                if nm.startswith("torch"):
                    pt_a, pt_b = (None, pt_b) if "A" in nm else (pt_a, None)
                else:
                    jx_a, jx_b = (None, jx_b) if "A" in nm else (jx_a, None)

        for which, r2_us, pt, jx in (("grad_A", row["grad_a"], pt_a, jx_a),
                                     ("grad_B", row["grad_b"], pt_b, jx_b)):
            cands = [x for x in (pt, jx) if x is not None and x > 0]
            best = min(cands) if cands else float("nan")
            ratio = r2_us / best if best > 0 else float("nan")
            worst = max(worst, ratio)
            tot["r2"] += r2_us * cnt
            tot["best"] += best * cnt
            js = f"{jx:>10.1f}" if jx is not None else f"{'-':>10}"
            print(f"{row['label']:>14} {m:>6}x{k}x{n:<6} {which:>7} {r2_us:>10.1f}"
                  f" {pt:>10.1f} {js} {best:>10.1f} {ratio:>8.1f}x")

    print("-" * len(hdr))
    print(f"  weighted by calls per step: R2 {tot['r2']/1000:.1f} ms vs best reference "
          f"{tot['best']/1000:.1f} ms  ->  {tot['r2']/tot['best']:.2f}x")
    if worst <= 1.0:
        print("LMO-13/14 PASS: R2 is at or above the best reference on every row.")
    else:
        print(f"LMO-13/14 OPEN: worst row is {worst:.1f}x behind the best reference.")


if __name__ == "__main__":
    main()
