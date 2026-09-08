"""PyTorch half of the TinyStories training comparison — and the only half
that prints a verdict.

Pairs with `crates/r2-train/examples/tinystories_train.rs`. That side trains
first and writes `ts_run/r2.json` plus the token stream and the initial
weights; this side trains the SAME model on the SAME tokens from the SAME
starting point, then joins the two and prints one table.

    cargo run --release -p r2-train --example tinystories_train
    python benchmarks/llm/tinystories_train.py

Nothing here reports an R2 number it did not read out of that manifest, and
nothing reports a comparison across two windows: `benchmarks/llm/REPORT.md`
records that hand-pairing two halves produced a wrong claim once already.

# The validity gate

Both sides evaluate the held-out set BEFORE any training, from identical
weights on identical tokens. Those two numbers are the same forward pass in
two languages, so they must agree to f32 rounding. If they do not, the
comparison below is between two different models and every row of it is
meaningless — so that check runs first and fails loudly.

# Why the architecture is spelled out rather than assembled from nn.Linear

The parameter layout has to match R2's blocks exactly, and R2 stores each
weight as (in, out) row-major while `nn.Linear` stores (out, in). Writing
`x @ W` against the raw blocks makes the mapping visible instead of hiding a
transpose inside a module. It dispatches to the same `aten::mm`.

RoPE here is INTERLEAVED — pairs (2i, 2i+1), the GPT-NeoX convention — which
is what `ops::rope_inplace` implements. The split-half Llama/HF convention
is a different function on the same frequencies, and using it silently makes
this a comparison against a model R2 is not training.
"""

import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT = os.environ.get("TS_OUT", "ts_run")
# SDPA is what a real PyTorch training script uses, so it is the honest bar
# for speed. It computes the same causal-softmax attention; set
# TS_ATTN=explicit to run the written-out form instead.
ATTN = os.environ.get("TS_ATTN", "sdpa")


def die(msg):
    print(f"\n{msg}", file=sys.stderr)
    sys.exit(1)


def load_f32(name, n):
    a = np.fromfile(os.path.join(OUT, name), dtype="<f4")
    if a.size != n:
        die(f"{name}: {a.size} floats, manifest says {n}")
    return torch.from_numpy(a.astype(np.float32))


def load_ids(name):
    return np.fromfile(os.path.join(OUT, name), dtype="<u4")


def batch_at(ids, s, bn, seq):
    """Byte-for-byte the same windows `batch_at` in the Rust half produces."""
    span = len(ids) - seq - 1
    off = [((s * bn + b) * seq) % span for b in range(bn)]
    inp = np.stack([ids[o:o + seq] for o in off]).astype(np.int64)
    tgt = np.stack([ids[o + 1:o + seq + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(inp), torch.from_numpy(tgt)


class R2Model(torch.nn.Module):
    """R2's architecture, parameter block for parameter block."""

    def __init__(self, d, seq):
        super().__init__()
        self.dim, self.nh, self.nkv = d["dim"], d["n_heads"], d["n_kv_heads"]
        self.nl, self.vocab, self.ffn = d["n_layers"], d["vocab"], d["ffn_hidden"]
        self.eps, self.base = d["eps"], d["rope_base"]
        self.hd = self.dim // self.nh
        self.kvd = self.nkv * self.hd

        # Load every block at its R2 index, so `P[3 + l*9 + k]` here names the
        # same matrix it names there.
        self.P = torch.nn.ParameterList(
            torch.nn.Parameter(load_f32(f"init_{b['i']}.bin", b["len"]))
            for b in d["blocks"])

        half = self.hd // 2
        freq = self.base ** (-2.0 * torch.arange(half, dtype=torch.float32) / self.hd)
        ang = torch.arange(seq, dtype=torch.float32)[:, None] * freq[None, :]
        # Registered so a longer generation prompt cannot silently index past
        # the table — it is rebuilt for the length actually needed.
        self.register_buffer("rope_cos", ang.cos(), persistent=False)
        self.register_buffer("rope_sin", ang.sin(), persistent=False)

    def rms_norm(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, t):
        """Rotate ADJACENT pairs (2i, 2i+1) — R2's interleaved convention."""
        if self.rope_cos.shape[0] < t:
            half = self.hd // 2
            freq = self.base ** (-2.0 * torch.arange(half, dtype=torch.float32) / self.hd)
            ang = torch.arange(t, dtype=torch.float32)[:, None] * freq[None, :]
            self.rope_cos, self.rope_sin = ang.cos(), ang.sin()
        c = self.rope_cos[:t][None, :, None, :]
        s = self.rope_sin[:t][None, :, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)

    def forward(self, idx):
        bn, t = idx.shape
        P, dim, hd = self.P, self.dim, self.hd
        x = P[0].view(self.vocab, dim)[idx.reshape(-1)].view(bn, t, dim)

        for l in range(self.nl):
            b = 3 + l * 9
            an, wq, wk, wv, wo, fn, w1, w2, w3 = (P[b + k] for k in range(9))
            h = self.rms_norm(x, an)
            q = self.rope((h @ wq.view(dim, dim)).view(bn, t, self.nh, hd), t)
            k = self.rope((h @ wk.view(dim, self.kvd)).view(bn, t, self.nkv, hd), t)
            v = (h @ wv.view(dim, self.kvd)).view(bn, t, self.nkv, hd)
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))
            if ATTN == "sdpa":
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = torch.full((t, t), float("-inf")).triu(1)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + mask
                ctx = att.softmax(-1) @ v
            ctx = ctx.transpose(1, 2).reshape(bn, t, dim)
            x = x + ctx @ wo.view(dim, dim)

            h = self.rms_norm(x, fn)
            gate = F.silu(h @ w1.view(dim, self.ffn))
            x = x + (gate * (h @ w3.view(dim, self.ffn))) @ w2.view(self.ffn, dim)

        x = self.rms_norm(x, P[1])
        return x @ P[2].view(dim, self.vocab)


def loss_on(model, idx, tgt):
    return F.cross_entropy(model(idx).reshape(-1, model.vocab), tgt.reshape(-1))


@torch.no_grad()
def eval_loss(model, ids, bn, seq, nbatch):
    """Mean over the same fixed held-out batches R2 evaluates."""
    tot = 0.0
    for s in range(nbatch):
        idx, tgt = batch_at(ids, s, bn, seq)
        tot += loss_on(model, idx, tgt).item()
    return tot / nbatch


@torch.no_grad()
def generate(model, prompt_ids, n):
    """Greedy decode, matching R2's temperature-0 sampler."""
    ids = list(prompt_ids)
    for _ in range(n):
        idx = torch.tensor([ids], dtype=torch.long)
        nxt = int(model(idx)[0, -1].argmax().item())
        ids.append(nxt)
    return ids


def measure_tokenize(d):
    """Time HuggingFace tokenizing the same training text, over R2's vocab.

    Called AFTER training, and that ordering is not cosmetic. Run before it,
    this holds a ~19 MB string and a 4.6M-element id list alive for the whole
    training loop, and PyTorch's measured training time moved 20.7% between
    two runs fifteen minutes apart (302.85 s -> 365.46 s) while R2's moved
    5.6%. A measurement that perturbs the thing it is measured beside is not
    a measurement. Doing it afterwards costs nothing and cannot reach back.

    Each side runs ITS OWN implementation over the SAME vocabulary — R2's
    `Tokenizer::encode`, HuggingFace's Rust `tokenizers` — because that is
    the choice a user actually faces. Loading R2's exported file is what
    keeps the two token streams identical.
    """
    try:
        from tokenizers import Tokenizer
    except Exception as e:                                # noqa: BLE001
        print(f"  tokenize   skipped: {e}")
        return None
    enc = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    corpus = open(os.environ.get("R2_CORPUS", "corpus.txt"),
                  encoding="utf-8", errors="replace").read()
    train_txt = corpus.encode("utf-8")[:d["train_bytes"]].decode("utf-8", "ignore")
    del corpus
    t0 = time.perf_counter()
    ids = enc.encode(train_txt).ids
    secs = time.perf_counter() - t0
    print(f"\n  tokenize   {len(ids)} ids from {d['train_bytes']/1e6:.1f} MB in "
          f"{secs:.2f} s (HuggingFace, R2's vocabulary; R2 got "
          f"{d['train_tokens']} in {d['tokenize_s']:.2f} s)")
    return secs


def main():
    torch.set_num_threads(int(os.environ.get("TS_THREADS", "6")))
    man = os.path.join(OUT, "r2.json")
    if not os.path.exists(man):
        die(f"{man} not found — run the R2 half first:\n"
            f"  cargo run --release -p r2-train --example tinystories_train")
    d = json.load(open(man))

    steps, bn, seq = d["steps"], d["batch"], d["seq"]
    train_ids, val_ids = load_ids("train_ids.bin"), load_ids("val_ids.bin")
    if len(train_ids) != d["train_tokens"] or len(val_ids) != d["val_tokens"]:
        die("token files disagree with the manifest — stale ts_run/")

    print("PyTorch — same model, same tokens, same initial weights")
    print(f"  torch      {torch.__version__}, {torch.get_num_threads()} threads, "
          f"attention: {ATTN}")
    print(f"  model      {d['n_params']/1e6:.2f}M parameters, dim {d['dim']} x "
          f"{d['n_layers']} layers, ffn {d['ffn_hidden']}, vocab {d['vocab']}")
    print(f"  schedule   {steps} steps x {bn} x {seq} = {steps*bn*seq} tokens, "
          f"Adam lr {d['lr']}")

    model = R2Model(d, max(seq, 64))
    n = sum(p.numel() for p in model.parameters())
    if n != d["n_params"]:
        die(f"parameter count {n} != R2's {d['n_params']} — different models")

    # (Tokenization is measured AFTER training — see `measure_tokenize`.)

    # ── validity gate ───────────────────────────────────────────────────
    val_before = eval_loss(model, val_ids, bn, seq, d["val_batches"])
    gap = abs(val_before - d["val_before"])
    print(f"\n  held-out loss at init:  R2 {d['val_before']:.6f}   "
          f"torch {val_before:.6f}   diff {gap:.2e}")
    if gap > 2e-3:
        die(f"FAIL: the two forward passes disagree at init by {gap:.2e}.\n"
            "Identical weights and identical tokens must give the same loss;\n"
            "this is a comparison between two different models. Stop here.")
    print("  same function, same starting point — the comparison is valid\n")

    # ── train ───────────────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=d["lr"],
                           betas=(0.9, 0.999), eps=1e-8)
    every = max(steps // 10, 1)
    curve = []
    t0 = time.perf_counter()
    first = last = 0.0
    for s in range(steps):
        idx, tgt = batch_at(train_ids, s, bn, seq)
        loss = loss_on(model, idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
        if s == 0:
            first = last
        if s == 0 or (s + 1) % every == 0:
            curve.append((s + 1, last))
            print(f"  step {s+1:>4}/{steps}  loss {last:.4f}   "
                  f"{time.perf_counter()-t0:.1f} s elapsed")
    train_s = time.perf_counter() - t0
    val_after = eval_loss(model, val_ids, bn, seq, d["val_batches"])
    bpb = val_after / math.log(2) * (d["val_tokens"] / d["val_bytes"])
    torch_tokenize_s = measure_tokenize(d)

    # ── sample ──────────────────────────────────────────────────────────
    sample = "(tokenizers not installed)"
    try:
        from tokenizers import Tokenizer
        enc = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
        pid = enc.encode(d["prompt"]).ids
        # Verify the shared tokenizer rather than assuming it: HuggingFace
        # must encode the prompt to the same ids R2 did. A file that parses
        # but segments differently would make every loss above incomparable,
        # and nothing else here would notice.
        if list(pid) != d["prompt_ids"]:
            sample = (f"(tokenizer MISMATCH: torch {list(pid)} vs "
                      f"R2 {d['prompt_ids']} — sample not comparable)")
        else:
            # R2's `generate` returns the CONTINUATION only, so drop the
            # prompt here too, or the two are not the same thing.
            sample = enc.decode(generate(model, pid, d["gen"])[len(pid):])
    except Exception as e:                                # noqa: BLE001
        sample = f"(sample skipped: {e})"

    # ── the joined table ────────────────────────────────────────────────
    r2_train = d["train_s"]
    tok_total = steps * bn * seq
    w = 26
    print(f"\n{'='*72}\nTinyStories — Ardon-R2 vs PyTorch, one window, one model\n{'='*72}")
    print(f"  corpus     {d['corpus_bytes']/1e6:.1f} MB, "
          f"train {d['train_bytes']/1e6:.1f} MB / held-out {d['val_bytes']/1e6:.1f} MB")
    print(f"  tokens     {d['train_tokens']} train + {d['val_tokens']} held-out, "
          f"vocab {d['vocab']} (R2's BPE, shared)")

    print(f"\n{'SPEED':<{w}}{'R2':>14}{'PyTorch':>14}{'':>4}")
    print("-" * 72)
    rows = [
        ("train, seconds", r2_train, train_s, "lower"),
        ("tokens/s", tok_total / r2_train, tok_total / train_s, "higher"),
        ("ms/step", r2_train * 1000 / steps, train_s * 1000 / steps, "lower"),
    ]
    for name, a, b, better in rows:
        ratio = (b / a) if better == "lower" else (a / b)
        verdict = (f"R2 {ratio:.2f}x faster" if ratio > 1
                   else f"PyTorch {1/ratio:.2f}x faster")
        print(f"{name:<{w}}{a:>14.2f}{b:>14.2f}   {verdict}")
    r2_tok = d["tokenize_s"]
    if torch_tokenize_s is None:
        pass
    elif d.get("tokens_reused") or r2_tok <= 0.0:
        # R2 mapped a token file an earlier run built, so it did no
        # tokenizing THIS run. Reporting 0 s against PyTorch's full pass
        # would flatter R2 for work it had simply already done, and
        # dividing by it is how this crashed the first time. Delete
        # `ts_run/` and re-run for a comparable pipeline number.
        print(f"{'tokenize corpus, seconds':<{w}}{'reused':>14}"
              f"{torch_tokenize_s:>14.2f}   R2 reused a cached token file")
        print(f"{'PIPELINE TOTAL, seconds':<{w}}{'—':>14}{'—':>14}"
              "   not comparable: delete ts_run/ to re-tokenize")
    else:
        print(f"{'tokenize corpus, seconds':<{w}}{r2_tok:>14.2f}"
              f"{torch_tokenize_s:>14.2f}   R2 {torch_tokenize_s/r2_tok:.2f}x faster")
        # What it actually costs to get a trained model from raw text.
        a, b = r2_tok + r2_train, torch_tokenize_s + train_s
        print(f"{'PIPELINE TOTAL, seconds':<{w}}{a:>14.2f}{b:>14.2f}   "
              + (f"R2 {b/a:.2f}x faster" if b > a else f"PyTorch {a/b:.2f}x faster"))

    print(f"\n{'ACCURACY (held-out)':<{w}}{'R2':>14}{'PyTorch':>14}{'':>4}")
    print("-" * 72)
    print(f"{'loss at init':<{w}}{d['val_before']:>14.4f}{val_before:>14.4f}"
          f"   identical start, by construction")
    print(f"{'loss after training':<{w}}{d['val_after']:>14.4f}{val_after:>14.4f}"
          f"   {'R2' if d['val_after'] < val_after else 'PyTorch'} lower by "
          f"{abs(d['val_after']-val_after):.4f}")
    print(f"{'bits per byte':<{w}}{d['bits_per_byte']:>14.4f}{bpb:>14.4f}")
    print(f"{'perplexity':<{w}}{math.exp(d['val_after']):>14.2f}"
          f"{math.exp(val_after):>14.2f}")
    print(f"{'train loss, first -> last':<{w}}"
          f"{d['loss_first']:>7.3f}->{d['loss_last']:<6.3f}"
          f"{first:>7.3f}->{last:<6.3f}")

    print("\n  loss curve (training loss at each checkpoint)")
    print(f"    {'step':>6}{'R2':>10}{'PyTorch':>10}{'diff':>10}")
    for (s1, l1), (s2, l2) in zip(d["curve"], curve):
        print(f"    {s1:>6}{l1:>10.4f}{l2:>10.4f}{abs(l1-l2):>10.4f}")

    print(f"\n  prompt   {d['prompt']!r}")
    print(f"  R2       {d['sample']!r}")
    print(f"  PyTorch  {sample!r}")

    rel = abs(d["val_after"] - val_after) / val_after
    print(f"\n{'='*72}")
    speed = train_s / r2_train
    print(f"  TRAINING  " + (f"R2 {speed:.2f}x faster" if speed > 1
                             else f"PyTorch {1/speed:.2f}x faster")
          + f"  ({r2_train:.1f} s vs {train_s:.1f} s on {tok_total} tokens)")
    if torch_tokenize_s is not None and not (d.get("tokens_reused") or r2_tok <= 0.0):
        a, b = r2_tok + r2_train, torch_tokenize_s + train_s
        print(f"  PIPELINE  " + (f"R2 {b/a:.2f}x faster" if b > a
                                 else f"PyTorch {a/b:.2f}x faster")
              + f"  ({a:.1f} s vs {b:.1f} s, tokenize + train)")
    print(f"  ACCURACY  held-out loss within {rel*100:.2f}% "
          f"({d['val_after']:.4f} vs {val_after:.4f})"
          + ("  — parity" if rel < 0.02 else "  — NOT parity, investigate"))
    print("=" * 72)


if __name__ == "__main__":
    main()
