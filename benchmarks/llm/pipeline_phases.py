"""PyTorch half of the pipeline-phase measurement. Same corpus, same phases.

Pairs with `crates/r2-train/examples/pipeline_phases.rs`. Architecture is
matched to R2's `Config` and the parameter count is ASSERTED against it
before anything is reported — a speed comparison between two different
models is worthless, and the assert is the only thing that stops it
happening silently.

    R2_CORPUS=corpus.txt R2_STEPS=40 R2_SEQ=64 R2_BATCH=32 \
      R2_VOCABS=256,8000 python benchmarks/llm/pipeline_phases.py

Each side uses ITS OWN ecosystem's tokenizer — R2 its BPE trainer, PyTorch
the HuggingFace one — which is the comparison a user actually faces.
Neither is asked to load the other's file.

Because the two vocabularies then differ, raw cross-entropy is NOT
comparable between them: it is per token, and guessing 1 of 8,000 tokens
is a harder question than 1 of 8,143, so the lower loss can belong to the
worse model. BITS PER BYTE is reported alongside and is the comparable
quantity — how much information the model needs per byte of the original
text, independent of how that text was chopped up.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM, HEADS, KV_HEADS, LAYERS, FFN = 256, 4, 2, 4, 768


def env(k, d, cast=int):
    return cast(os.environ.get(k, d))


def rope(x, base=10000.0):
    """Rotary embedding over the head dimension, matching R2's rope_inplace."""
    b, t, h, hd = x.shape
    half = hd // 2
    freq = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    pos = torch.arange(t, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    a, bb = x[..., :half], x[..., half:]
    return torch.cat([a * cos - bb * sin, a * sin + bb * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.hd = DIM // HEADS
        kv = KV_HEADS * self.hd
        self.an = nn.Parameter(torch.ones(DIM))
        self.fn = nn.Parameter(torch.ones(DIM))
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, kv, bias=False)
        self.wv = nn.Linear(DIM, kv, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.w1 = nn.Linear(DIM, FFN, bias=False)
        self.w2 = nn.Linear(FFN, DIM, bias=False)
        self.w3 = nn.Linear(DIM, FFN, bias=False)

    @staticmethod
    def norm(x, w, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def forward(self, x, mask):
        b, t, _ = x.shape
        h = self.norm(x, self.an)
        q = rope(self.wq(h).view(b, t, HEADS, self.hd))
        k = rope(self.wk(h).view(b, t, KV_HEADS, self.hd))
        v = self.wv(h).view(b, t, KV_HEADS, self.hd)
        rep = HEADS // KV_HEADS
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd) + mask
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, DIM)
        x = x + self.wo(out)
        h = self.norm(x, self.fn)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, DIM)
        self.blocks = nn.ModuleList(Block(vocab) for _ in range(LAYERS))
        self.fnorm = nn.Parameter(torch.ones(DIM))
        self.out = nn.Linear(DIM, vocab, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        mask = torch.full((t, t), float("-inf")).triu(1)
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.out(Block.norm(x, self.fnorm))


def r2_params(vocab):
    kv = KV_HEADS * (DIM // HEADS)
    per_layer = 2 * DIM + 2 * DIM * DIM + 2 * DIM * kv + 3 * DIM * FFN
    return vocab * DIM + LAYERS * per_layer + DIM + vocab * DIM


def main():
    torch.set_num_threads(6)
    torch.manual_seed(1)
    path = os.environ.get("R2_CORPUS", "corpus.txt")
    steps, seq, bn = env("R2_STEPS", 40), env("R2_SEQ", 64), env("R2_BATCH", 32)
    vocabs = [int(x) for x in os.environ.get("R2_VOCABS", "256,8000").split(",")]
    learn_cap = env("R2_LEARN_BYTES", 4_000_000)

    t0 = time.perf_counter()
    corpus = open(path, encoding="utf-8", errors="replace").read()
    read_s = time.perf_counter() - t0

    print("Pipeline phases — PyTorch")
    print(f"  corpus   {path}  ({len(corpus.encode())/1e6:.1f} MB)")
    print(f"  schedule {steps} steps x {bn} x {seq} = {steps*bn*seq} tokens/arm\n")
    print(f"{'arm':<12}{'read':>8}{'learn':>8}{'tokenize':>10}{'train':>10}"
          f"{'TOTAL':>10}{'tok/byte':>10}{'tokens/s':>12}{'text B/s':>11}")
    print("-" * 92)

    for v in vocabs:
        t0 = time.perf_counter()
        shared = os.environ.get("R2_TOKENIZER_JSON")
        if shared:
            # The SAME vocabulary R2 used. Without this the two sides have
            # different token streams and their losses are not comparable —
            # only the speed numbers would mean anything.
            from tokenizers import Tokenizer
            enc = Tokenizer.from_file(shared)
        elif v <= 256:
            enc = None                      # raw bytes are the ids
        else:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
            enc = Tokenizer(models.BPE())
            enc.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            enc.decoder = decoders.ByteLevel()
            tr = trainers.BpeTrainer(vocab_size=v,
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                     show_progress=False)
            enc.train_from_iterator([corpus[:learn_cap]], tr)
        learn_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        ids = list(corpus.encode("utf-8")) if enc is None else enc.encode(corpus).ids
        tokenize_s = time.perf_counter() - t0

        vocab_actual = 256 if enc is None else enc.get_vocab_size()
        need = steps * bn * (seq + 1)
        if len(ids) < need:
            print(f"vocab {vocab_actual}: only {len(ids)} tokens, needs {need}")
            continue

        model = Model(vocab_actual)
        n = sum(p.numel() for p in model.parameters())
        assert n == r2_params(vocab_actual), (
            f"ARCHITECTURE MISMATCH at vocab {vocab_actual}: "
            f"PyTorch {n:,} vs R2 {r2_params(vocab_actual):,}")
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        cursor, loss_v = 0, 0.0
        first, step, traj = float("nan"), 0, []
        t0 = time.perf_counter()
        for _ in range(steps):
            xs, ys = [], []
            for _ in range(bn):
                if cursor + seq + 1 >= len(ids):
                    cursor = 0
                xs.append(ids[cursor:cursor + seq])
                ys.append(ids[cursor + 1:cursor + seq + 1])
                cursor += seq
            xb, yb = torch.tensor(xs), torch.tensor(ys)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).view(-1, vocab_actual), yb.view(-1))
            loss.backward()
            opt.step()
            loss_v = float(loss.detach())
            step += 1
            if step == 1:
                first = loss_v
            if step % max(steps // 5, 1) == 0:
                traj.append(f"{step}:{loss_v:.4f}")
        train_s = time.perf_counter() - t0

        total = read_s + learn_s + tokenize_s + train_s
        tpb = len(ids) / len(corpus.encode("utf-8"))
        toks = steps * bn * seq
        print(f"{'vocab '+str(vocab_actual):<12}{read_s:>8.2f}{learn_s:>8.2f}"
              f"{tokenize_s:>10.2f}{train_s:>10.2f}{total:>10.2f}{tpb:>10.3f}"
              f"{toks/train_s:>12.0f}{toks/tpb/total:>11.0f}")
        print(f"{'  share':<12}{'':>8}{learn_s/total*100:>7.1f}%"
              f"{tokenize_s/total*100:>9.1f}%{train_s/total*100:>9.1f}%"
              f"{'':>10}{'':>10}{'':>12}{'loss '+format(loss_v,'.3f'):>11}")
        # bits/byte: comparable across vocabularies where raw loss is not.
        bpb = loss_v * tpb / math.log(2)
        print(f"  loss {first:.4f} -> {loss_v:.4f}  [{'  '.join(traj)}]"
              f"  bits/byte {bpb:.4f}")


if __name__ == "__main__":
    main()
