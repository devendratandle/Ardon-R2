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

    print(f"\nPyTorch's gather does t*d = {t*d} element copies for ANY")
    print("vocabulary and zero multiply-adds, so its cost is flat in vocab")
    print("while the one-hot form's grows linearly.")


if __name__ == "__main__":
    main()
