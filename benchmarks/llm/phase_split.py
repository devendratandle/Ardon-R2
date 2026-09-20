"""One PyTorch training step split into forward / backward / optimizer,
each timed on its own — the mirror of `--example phase_split` in R2.

Same architecture (dim 256 x 4 layers, ffn 768, vocab 8000, GQA 4/2),
same 32 x 64 = 2,048 tokens per step, same warm-up to the sustained
clock, median over the same number of steps. Weights are random here:
this measures time, not learning, and the two harnesses' parameter
counts are checked to agree so the shapes are the same.

Uses the model class from `tinystories_train.py` unchanged, so the
forward timed is exactly the one the 500-step comparison ran.

    python benchmarks/llm/phase_split.py
"""
import os, sys, time, statistics
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinystories_train as ts

E = lambda k, d: int(os.environ.get(k, d))
dim, nl, ffn, vocab, nh, nkv = E('R2_DIM', 256), E('R2_LAYERS', 4), E('R2_FFN', 768), E('R2_VOCAB', 8000), E('R2_HEADS', 4), E('R2_KV', 2)
hd = dim // nh
kvd = nkv * hd
bn, seq = E('R2_BATCH', 32), E('R2_SEQ', 64)
steps = int(os.environ.get("R2_STEPS", "20"))

# R2's block layout: embed, final norm, head, then 9 blocks per layer.
lens = [vocab * dim, dim, dim * vocab]
for _ in range(nl):
    lens += [dim, dim * dim, dim * kvd, dim * kvd, dim * dim, dim, dim * ffn, ffn * dim, dim * ffn]
d = {"dim": dim, "n_layers": nl, "ffn_hidden": ffn, "vocab": vocab, "n_heads": nh,
     "n_kv_heads": nkv, "eps": 1e-5, "rope_base": 10000.0,
     "blocks": [{"i": i, "len": n} for i, n in enumerate(lens)]}
g = torch.Generator().manual_seed(7)
ts.load_f32 = lambda name, n: torch.randn(n, generator=g) * 0.02   # random init for timing

model = ts.R2Model(d, seq)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
idx = torch.tensor([[(b * 131 + i * 7919 + 1) % vocab for i in range(seq)] for b in range(bn)])
tgt = torch.tensor([[(b * 131 + i * 7919 + 2) % vocab for i in range(seq)] for b in range(bn)])

print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads, attention: {ts.ATTN}")
print(f"dim {dim} x {nl} layers, ffn {ffn}, vocab {vocab}, {bn} x {seq} = {bn*seq} tokens/step, {n_params} params")

def one_step():
    loss = ts.loss_on(model, idx, tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

t0 = time.perf_counter()
while time.perf_counter() - t0 < 8.0:
    one_step()

fw, bw, op, whole = [], [], [], []
for _ in range(steps):
    s0 = time.perf_counter()
    loss = ts.loss_on(model, idx, tgt)
    loss.item()
    t_f = (time.perf_counter() - s0) * 1e3

    s1 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    t_b = (time.perf_counter() - s1) * 1e3

    s2 = time.perf_counter()
    opt.step()
    t_o = (time.perf_counter() - s2) * 1e3
    fw.append(t_f); bw.append(t_b); op.append(t_o)

    s3 = time.perf_counter()
    one_step()
    whole.append((time.perf_counter() - s3) * 1e3)

f, b, o, w = (statistics.median(x) for x in (fw, bw, op, whole))
print(f"\n{'phase':<12} {'ms/step':>10} {'share':>8}")
print("-" * 32)
for l, v in (("forward", f), ("backward", b), ("optimizer", o)):
    print(f"{l:<12} {v:>10.1f} {v/(f+b+o)*100:>7.1f}%")
print("-" * 32)
print(f"{'phase sum':<12} {f+b+o:>10.1f}")
print(f"{'whole step':<12} {w:>10.1f}   (one_step timed whole; should match the sum)")
print(f"\nTORCH_PHASES forward={f:.2f} backward={b:.2f} optimizer={o:.2f} whole={w:.2f}")
