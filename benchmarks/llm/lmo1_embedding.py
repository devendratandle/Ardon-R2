"""LMO-1 — PyTorch's embedding step, to compare against R2's.

Pairs with `crates/r2-train/examples/lmo1_embedding.rs`. Same tokens, same
dim, same vocabularies.

PyTorch's `nn.Embedding` is a GATHER: `aten::embedding` dispatches to
`aten::index_select`, which copies row tok_i of the table into row i of the
output; its backward is `aten::embedding_dense_backward`, a scatter-add
into the touched rows under `at::parallel_for`. Confirmed by dispatch with
torch.profiler, not assumed from the Python name.

R2 computes the same result as a one-hot matmul, so the comparison is
between two algorithms for one operation rather than between two
implementations of one algorithm. The FLOP counts are printed for both so
the difference is visible rather than inferred.

    R2_LMO_TOKENS=2048 R2_LMO_VOCABS=256,8000,32000 \
      python benchmarks/llm/lmo1_embedding.py
"""

import json
import os
import statistics
import time

import torch


def env_list(key, default):
    raw = os.environ.get(key)
    return [int(x) for x in raw.split(",")] if raw else default


def med_us(reps, fn):
    for _ in range(min(reps, 5)):
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
    torch.manual_seed(1)
    t = int(os.environ.get("R2_LMO_TOKENS", 2048))
    d = int(os.environ.get("R2_LMO_DIM", 256))
    vocabs = env_list("R2_LMO_VOCABS", [256, 8000, 32000])

    print("LMO-1 — embedding step only, PyTorch's gather")
    print(f"  {t} tokens, dim {d}, microseconds, median of 9")
    print(f"  torch {torch.__version__}, {torch.get_num_threads()} threads\n")
    print(f"{'vocab':>7} {'onehot MB':>12} {'build':>11} {'forward':>11}"
          f" {'fwd+bwd':>11} {'MFLOP fwd':>13}")
    print("-" * 70)

    torch_rows = {}
    for vocab in vocabs:
        reps = max(3, min(200, int(2e9 / (t * vocab * d)) + 1))
        idx = torch.tensor([(i * 7919) % vocab for i in range(t)], dtype=torch.long)
        table = torch.randn(vocab, d)
        g = torch.randn(t, d)

        # PyTorch builds NO one-hot and needs no such buffer. Shown as 0 so
        # the column lines up with R2's, where it is the dominant cost.
        def fwd():
            with torch.no_grad():
                return torch.nn.functional.embedding(idx, table)

        w = table.clone().requires_grad_(True)

        def fwd_bwd():
            if w.grad is not None:
                w.grad = None
            torch.nn.functional.embedding(idx, w).backward(g)

        f_us = med_us(reps, fwd)
        fb_us = med_us(reps, fwd_bwd)
        # A gather does t*d element copies; there is no multiply-add at all.
        print(f"{vocab:>7} {0.0:>12.1f} {0.0:>11.1f} {f_us:>11.1f}"
              f" {fb_us:>11.1f} {0.0:>13.1f}")
        torch_rows[vocab] = (f_us, fb_us)

    print(f"\nPyTorch's gather does t*d = {t*d} element copies for ANY")
    print("vocabulary and zero multiply-adds, so its cost is flat in vocab")
    print("while the one-hot form's grows linearly.")

    verdict(torch_rows, t, d)


def verdict(torch_rows, t, d):
    """Join R2's numbers with PyTorch's and say who is ahead, per row.

    THE POINT OF THE WHOLE FILE. R2 improving against its own past is not
    a result — the only question is whether R2 is at or above PyTorch on
    this stage. Two programs printing two tables leaves that join to a
    human, and it got skipped. It does not get skipped now.

    Reads `lmo1_r2.json`, written by
    `cargo run --release -p r2-train --example lmo1_embedding`.
    """
    try:
        r2 = json.load(open("lmo1_r2.json"))
    except FileNotFoundError:
        print("\nlmo1_r2.json not found — run the R2 half first:")
        print("  cargo run --release -p r2-train --example lmo1_embedding")
        print("without it there is no verdict, only PyTorch's numbers.")
        return
    if (r2["tokens"], r2["dim"]) != (t, d):
        print(f"\nlmo1_r2.json is for {r2['tokens']} tokens x dim {r2['dim']},")
        print(f"this run is {t} x {d}. Re-run the R2 half at the same shape.")
        return

    print("\n" + "=" * 74)
    print("LMO-1 VERDICT — R2 against PyTorch, same shape, same window")
    print("  GA = the gather both sides ship. fwd+bwd is `.backward(g)` on")
    print("  both: R2 seeds the gradient with `backward_from`, not a")
    print("  manufactured scalar loss.")
    print("=" * 74)
    print(f"{'vocab':>7} {'phase':>9} {'R2 us':>10} {'torch us':>10}"
          f" {'ratio':>8}  verdict")
    print("-" * 74)
    worst = 0.0
    for row in r2["rows"]:
        v = row["vocab"]
        if v not in torch_rows:
            continue
        for phase, r2_us, pt_us in (
            ("fwd", row["ga_fwd"], torch_rows[v][0]),
            ("fwd+bwd", row["ga_fb"], torch_rows[v][1]),
        ):
            ratio = r2_us / pt_us if pt_us > 0 else float("inf")
            worst = max(worst, ratio)
            mark = "R2 AHEAD" if ratio <= 1.0 else f"R2 BEHIND {ratio:.2f}x"
            print(f"{v:>7} {phase:>9} {r2_us:>10.1f} {pt_us:>10.1f}"
                  f" {ratio:>7.2f}x  {mark}")
    print("-" * 74)
    if worst <= 1.0:
        print("LMO-1 PASSES: R2 is at or above PyTorch on every row.")
    else:
        print(f"LMO-1 NOT DONE: worst row is {worst:.2f}x behind PyTorch.")
        print("An R2-versus-R2 speedup does not close this. Only a row at")
        print("or under 1.00x does.")


if __name__ == "__main__":
    main()
