"""LMO-15 — the attention block, R2 against PyTorch AND JAX.

Attention was once the largest open component with no reference number at
all. It has one now, from two independent implementations, and this file
is how it is kept: R2 1.5-2.3x behind on the forward and 2.1-2.7x on
forward+backward, ahead of torch-explicit and JAX at 4,096 tokens, behind
only `scaled_dot_product_attention`. See `REPORT.md` section 3.

WHAT IS COMPARED. The same function R2 computes in
`llm.rs::forward_fused` between the q/k/v projections and the output
projection: grouped-query causal attention, scale 1/sqrt(head_dim), with
q/k/v as ALREADY-COMPUTED activations. The backward is seeded with the
supplied upstream gradient — `.backward(g)` in torch, a `vjp` applied to
`g` in JAX, `backward_from(ctx, g)` in R2 — so all three differentiate the
same function from the same seed, with no scalar loss invented in between.

TWO REFERENCES PER FRAMEWORK, deliberately:

  explicit  matmul / mask / softmax / matmul, written out. This is R2's
            algorithm, so it isolates KERNEL QUALITY.
  fused     `F.scaled_dot_product_attention` / `jax.nn.dot_product_attention`
            — what these frameworks actually run in a real model. R2 has no
            equivalent, and that ABSENCE is a finding, not an unfair
            comparison. Confirm by dispatch, not by name.

    cargo run --release -p r2-train --example lmo15_attention
    python benchmarks/llm/lmo15_attention.py
"""

import json
import math
import os
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


# ── the function under test, written once per framework ────────────────

def torch_attn(q, k, v, nh, nkv, hd, seq, causal_mask):
    """Explicit form — R2's algorithm, in torch ops."""
    bn = q.shape[0]
    q = q.view(bn, seq, nh, hd).transpose(1, 2)          # [bn, nh, seq, hd]
    k = k.view(bn, seq, nkv, hd).transpose(1, 2)
    v = v.view(bn, seq, nkv, hd).transpose(1, 2)
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + causal_mask
    return (sc.softmax(-1) @ v).transpose(1, 2).reshape(bn, seq, nh * hd)


def torch_sdpa(q, k, v, nh, nkv, hd, seq):
    """What PyTorch actually runs in a model."""
    bn = q.shape[0]
    q = q.view(bn, seq, nh, hd).transpose(1, 2)
    k = k.view(bn, seq, nkv, hd).transpose(1, 2)
    v = v.view(bn, seq, nkv, hd).transpose(1, 2)
    rep = nh // nkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(bn, seq, nh * hd)


def jax_attn(q, k, v, nh, nkv, hd, seq, mask):
    bn = q.shape[0]
    q = jnp.transpose(q.reshape(bn, seq, nh, hd), (0, 2, 1, 3))
    k = jnp.transpose(k.reshape(bn, seq, nkv, hd), (0, 2, 1, 3))
    v = jnp.transpose(v.reshape(bn, seq, nkv, hd), (0, 2, 1, 3))
    rep = nh // nkv
    k = jnp.repeat(k, rep, axis=1)
    v = jnp.repeat(v, rep, axis=1)
    sc = (q @ jnp.swapaxes(k, -2, -1)) / math.sqrt(hd) + mask
    o = jax.nn.softmax(sc, axis=-1) @ v
    return jnp.transpose(o, (0, 2, 1, 3)).reshape(bn, seq, nh * hd)


def main():
    torch.set_num_threads(6)
    torch.manual_seed(1)
    try:
        r2 = json.load(open("lmo15_r2.json"))
    except FileNotFoundError:
        print("lmo15_r2.json not found — run the R2 half first:")
        print("  cargo run --release -p r2-train --example lmo15_attention")
        print("Without it there is no verdict, only reference numbers.")
        r2 = None

    seq = int(os.environ.get("R2_LMO_SEQ", r2["seq"] if r2 else 64))
    nh = int(os.environ.get("R2_LMO_HEADS", r2["heads"] if r2 else 4))
    nkv = int(os.environ.get("R2_LMO_KV", r2["kv"] if r2 else 2))
    hd = int(os.environ.get("R2_LMO_HD", r2["hd"] if r2 else 64))
    batches = [row["batch"] for row in r2["rows"]] if r2 else [8, 32, 64]

    print("LMO-15 — attention block only, PyTorch and JAX references")
    print(f"  seq {seq}, {nh} query heads, {nkv} kv heads, head_dim {hd}")
    print(f"  torch {torch.__version__}, {torch.get_num_threads()} threads", end="")
    print(f", jax {jax.__version__} on {jax.default_backend()}" if HAVE_JAX else ", no jax")
    print()
    hdr = (f"{'batch':>7} {'tokens':>7} {'pt expl f':>10} {'pt expl fb':>11}"
           f" {'pt sdpa f':>10} {'pt sdpa fb':>11}")
    if HAVE_JAX:
        hdr += f" {'jax f':>9} {'jax fb':>9}"
    print(hdr)
    print("-" * len(hdr))

    ref = {}
    for bn in batches:
        reps = max(2, min(30, int(6e7 / (bn * seq * seq))))
        q = torch.randn(bn, seq * nh * hd)
        k = torch.randn(bn, seq * nkv * hd)
        v = torch.randn(bn, seq * nkv * hd)
        g = torch.randn(bn, seq, nh * hd)
        mask = torch.full((seq, seq), float("-inf")).triu(1)

        with torch.no_grad():
            f_expl = med_us(reps, lambda: torch_attn(q, k, v, nh, nkv, hd, seq, mask))
            f_sdpa = med_us(reps, lambda: torch_sdpa(q, k, v, nh, nkv, hd, seq))

        qg, kg, vg = (x.clone().requires_grad_(True) for x in (q, k, v))

        def fb_expl():
            for x in (qg, kg, vg):
                x.grad = None
            torch_attn(qg, kg, vg, nh, nkv, hd, seq, mask).backward(g)

        def fb_sdpa():
            for x in (qg, kg, vg):
                x.grad = None
            torch_sdpa(qg, kg, vg, nh, nkv, hd, seq).backward(g)

        fb_e = med_us(reps, fb_expl)
        fb_s = med_us(reps, fb_sdpa)

        row = [bn, bn * seq, f_expl, fb_e, f_sdpa, fb_s]
        line = (f"{bn:>7} {bn*seq:>7} {f_expl:>10.1f} {fb_e:>11.1f}"
                f" {f_sdpa:>10.1f} {fb_s:>11.1f}")

        if HAVE_JAX:
            jq, jk, jv = (jnp.asarray(x.numpy()) for x in (q, k, v))
            jg = jnp.asarray(g.numpy())
            jmask = jnp.asarray(mask.numpy())
            f = jax.jit(lambda a, b, c: jax_attn(a, b, c, nh, nkv, hd, seq, jmask))
            f(jq, jk, jv).block_until_ready()

            def jf():
                f(jq, jk, jv).block_until_ready()

            def _vjp(a, b, c, gg):
                _, pull = jax.vjp(lambda x, y, z: jax_attn(x, y, z, nh, nkv, hd, seq, jmask),
                                  a, b, c)
                return pull(gg)
            jfb_fn = jax.jit(_vjp)
            jax.block_until_ready(jfb_fn(jq, jk, jv, jg))

            def jfb():
                jax.block_until_ready(jfb_fn(jq, jk, jv, jg))

            jf_us = med_us(reps, jf)
            jfb_us = med_us(reps, jfb)
            row += [jf_us, jfb_us]
            line += f" {jf_us:>9.1f} {jfb_us:>9.1f}"

        print(line)
        ref[bn] = row

    if r2 is None:
        return
    verdict(r2, ref)


def verdict(r2, ref):
    """One table, one answer per row. R2 improving against itself is not a
    result; only a ratio at or under 1.00x is."""
    print("\n" + "=" * 86)
    print("LMO-15 VERDICT — R2 against PyTorch and JAX, same shape, same window")
    print("  'best' is the fastest reference on that row: the bar R2 has to clear.")
    print("=" * 86)
    hdr = (f"{'tokens':>7} {'phase':>8} {'R2 us':>11} {'pt expl':>10} {'pt sdpa':>10}"
           f" {'jax':>10} {'best':>10} {'R2/best':>9}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for row in r2["rows"]:
        bn = row["batch"]
        if bn not in ref:
            continue
        _, tokens, f_expl, fb_e, f_sdpa, fb_s, *rest = ref[bn]
        jf, jfb = (rest + [None, None])[:2]
        for phase, r2_us, pt_e, pt_s, jx in (
            ("fwd", row["fwd"], f_expl, f_sdpa, jf),
            ("fwd+bwd", row["fb"], fb_e, fb_s, jfb),
        ):
            cands = [x for x in (pt_e, pt_s, jx) if x is not None]
            best = min(cands)
            ratio = r2_us / best
            worst = max(worst, ratio)
            js = f"{jx:>10.1f}" if jx is not None else f"{'-':>10}"
            print(f"{tokens:>7} {phase:>8} {r2_us:>11.1f} {pt_e:>10.1f} {pt_s:>10.1f}"
                  f" {js} {best:>10.1f} {ratio:>8.1f}x")
    print("-" * len(hdr))
    if worst <= 1.0:
        print("LMO-15 PASSES: R2 is at or above the best reference on every row.")
    else:
        print(f"LMO-15 OPEN: worst row is {worst:.1f}x behind the best reference.")
        print("This is the largest open component in the queue. It now has a")
        print("number, which it never had before.")


if __name__ == "__main__":
    main()
