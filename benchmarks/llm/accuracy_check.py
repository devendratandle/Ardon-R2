"""Do Ardon-R2 and an independent float64 implementation compute the SAME
function?

R2's own tests check R2 against itself — finite differences, its own
explicit forms. Those catch a coding slip but not a shared
misunderstanding: if a forward and its backward agree on a wrong
definition, finite differences agree with both. This rebuilds the model in
torch.float64 from the published equations and compares R2's loss, logits
and EVERY parameter gradient.

float64 matters. In float32 a real defect and ordinary rounding look alike,
so the reference is computed in double and R2's f32 result is judged
against it — the expected disagreement is then f32 epsilon (~1e-7
relative), and anything larger is R2's.

Block by block, not one summary number: "max error 3e-4" says something is
wrong without saying where, and the block name is the whole diagnostic.

    cargo run --release -p r2-train --example dump_reference
    python benchmarks/llm/accuracy_check.py
"""

import json
import math
import sys

import numpy as np
import torch

DT = torch.float64


def load(name, n):
    a = np.fromfile(name, dtype="<f4")
    assert a.size == n, f"{name}: {a.size} floats, manifest says {n}"
    return torch.tensor(a.astype(np.float64), dtype=DT)


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, base):
    """Rotate ADJACENT pairs (2i, 2i+1) by position*freq.

    This is the GPT-NeoX/interleaved convention, which is what
    `r2_tensor::ops::rope_inplace` implements:

        x[2i], x[2i+1]  <-  a*cos - b*sin, a*sin + b*cos
        freq            =   1 / base^(2i/hd)

    NOT the Llama/HuggingFace SPLIT-HALF convention that pairs (i, i+half).
    The two use the same frequencies and both are valid RoPE; they are not
    the same function, and this checker used to implement the split-half
    one while claiming in its docstring to implement R2's. That alone put
    the logits 1.4 (relative) away from R2's and made every one of the 21
    gradient blocks fail at ~1.35 — a uniform failure that looks like a
    broken model and is actually a broken reference.
    """
    t, h, hd = x.shape
    half = hd // 2
    freq = base ** (-2.0 * torch.arange(half, dtype=DT) / hd)
    ang = torch.arange(t, dtype=DT)[:, None] * freq[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos),
                       dim=-1).reshape(t, h, hd)


def main():
    d = json.load(open("reference_dump.json"))
    dim, nh, nkv = d["dim"], d["n_heads"], d["n_kv_heads"]
    nl, vocab, ffn = d["n_layers"], d["vocab"], d["ffn_hidden"]
    eps, base = d["eps"], d["rope_base"]
    bn, seq = d["batch"], d["seq"]
    hd, kvd = dim // nh, nkv * (dim // nh)

    blocks = d["blocks"]
    P = [load(f"ref_param_{b['i']}.bin", b["len"]).requires_grad_(True) for b in blocks]
    name = {b["i"]: b["name"] for b in blocks}

    toks = torch.tensor(d["tokens"], dtype=torch.long)      # [bn, seq]
    tgts = torch.tensor(d["targets"], dtype=torch.long)

    # R2 fuses the batch into one (bn*seq) row block and rotates by
    # position % seq, so each sequence sees positions 0..seq exactly as it
    # would alone. Reproduce that, not a per-sequence loop.
    emb = P[0].view(vocab, dim)
    x = emb[toks.reshape(-1)]                                # [bn*seq, dim]

    for l in range(nl):
        b = 3 + l * 9
        an, wq, wk, wv, wo, fn, w1, w2, w3 = (P[b + k] for k in range(9))
        h = rms_norm(x, an, eps)
        q = (h @ wq.view(dim, dim)).view(bn, seq, nh, hd)
        k = (h @ wk.view(dim, kvd)).view(bn, seq, nkv, hd)
        v = (h @ wv.view(dim, kvd)).view(bn, seq, nkv, hd)
        q = torch.stack([rope(q[i], base) for i in range(bn)])
        k = torch.stack([rope(k[i], base) for i in range(bn)])
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        qt, kt, vt = (z.transpose(1, 2) for z in (q, k, v))   # [bn, nh, seq, hd]
        mask = torch.full((seq, seq), float("-inf"), dtype=DT).triu(1)
        att = (qt @ kt.transpose(-2, -1)) / math.sqrt(hd) + mask
        ctx = (att.softmax(-1) @ vt).transpose(1, 2).reshape(bn * seq, dim)
        x = x + ctx @ wo.view(dim, dim)
        h = rms_norm(x, fn, eps)
        gate = torch.nn.functional.silu(h @ w1.view(dim, ffn))
        x = x + (gate * (h @ w3.view(dim, ffn))) @ w2.view(ffn, dim)

    x = rms_norm(x, P[1], eps)
    # The output head is stored (dim x vocab) ROW-MAJOR — R2 computes it as
    # `matmul(xn, leaves[2], t, d, vocab)`, whose B operand is k x n = d x
    # vocab (`Trainer::block_shapes` writes it `d * cfg.vocab`, against
    # `cfg.vocab * d` for the embedding). Reading it as `view(vocab, dim).T`
    # is a DIFFERENT matrix, not a transposed view of the same one — the
    # embedding and the head have identical element counts, so nothing in
    # the manifest catches the mix-up.
    logits = x @ P[2].view(dim, vocab)
    loss = torch.nn.functional.cross_entropy(logits, tgts.reshape(-1))
    loss.backward()

    r2_loss = d["loss"]
    r2_logits = load("ref_logits.bin", bn * seq * vocab).view(bn * seq, vocab)

    print(f"model {sum(b['len'] for b in blocks):,} params, "
          f"{nl} layers, dim {dim}, vocab {vocab}")
    print(f"reference: torch {torch.__version__}, dtype {DT}\n")

    ok = True
    dl = abs(loss.item() - r2_loss)
    print(f"{'loss':<22} R2 {r2_loss:.10f}  ref {loss.item():.10f}  diff {dl:.3e}")
    ok &= dl < 1e-5

    lg = (logits - r2_logits).abs().max().item()
    scale = logits.abs().max().item()
    print(f"{'logits':<22} max abs diff {lg:.3e}  "
          f"(relative to |max| {scale:.3f}: {lg/scale:.3e})")
    ok &= lg / scale < 1e-5

    print(f"\n{'block':<22}{'rel err':>12}{'max |grad|':>14}")
    print("-" * 50)
    rows = []
    for b in blocks:
        i, n = b["i"], b["len"]
        got = load(f"ref_grad_{i}.bin", n)
        want = P[i].grad.reshape(-1)
        # Scale by the LARGEST gradient in the block, not per element:
        # elementwise relative error explodes wherever a gradient is near
        # zero through cancellation and makes both sides look broken.
        denom = want.abs().max().item()
        rel = (got - want).abs().max().item() / denom if denom > 0 else 0.0
        rows.append((rel, name[i], denom))
    for rel, nm, denom in sorted(rows, reverse=True):
        print(f"{nm:<22}{rel:>12.3e}{denom:>14.4f}")
        ok &= rel < 1e-4

    worst, med = max(r[0] for r in rows), sorted(r[0] for r in rows)[len(rows) // 2]
    print("-" * 50)
    print(f"median relative error {med:.3e}, worst {worst:.3e}")
    print("worst/median near 1 means uniform f32 noise; one block far above")
    print("the rest is that operator's defect.\n")

    print("PASS — R2 and an independent float64 implementation agree on the"
          if ok else "FAIL — R2 disagrees with the float64 reference on the")
    print("loss, the logits and every parameter gradient."
          if ok else "quantities marked above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
