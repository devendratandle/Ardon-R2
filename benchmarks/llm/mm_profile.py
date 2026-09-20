"""PyTorch's per-GEMM times INSIDE a training step, by shape - the
like-for-like baseline for R2's `team_diag`.

Every earlier "R2 vs MKL" figure on the small shapes compared R2 measured
inside a real step (cold operands) against MKL in an isolated loop (hot
operands). The pack-free experiment showed those differ by up to 1.5x.
This profiles the same model, same 2,048 tokens/step, warm to the
sustained clock, and groups `aten::mm` by input shape: calls per step and
mean microseconds, as PyTorch actually pays them.

    python benchmarks/llm/mm_profile.py
"""
import os, sys, time, collections
import torch
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinystories_train as ts

E = lambda k, d: int(os.environ.get(k, d))
dim, nl, ffn, vocab, nh, nkv = E('R2_DIM', 256), E('R2_LAYERS', 4), E('R2_FFN', 768), E('R2_VOCAB', 8000), E('R2_HEADS', 4), E('R2_KV', 2)
hd = dim // nh; kvd = nkv * hd
bn, seq = E('R2_BATCH', 32), E('R2_SEQ', 64)
steps = int(os.environ.get("R2_STEPS", "20"))
lens = [vocab * dim, dim, dim * vocab]
for _ in range(nl):
    lens += [dim, dim * dim, dim * kvd, dim * kvd, dim * dim, dim, dim * ffn, ffn * dim, dim * ffn]
d = {"dim": dim, "n_layers": nl, "ffn_hidden": ffn, "vocab": vocab, "n_heads": nh,
     "n_kv_heads": nkv, "eps": 1e-5, "rope_base": 10000.0,
     "blocks": [{"i": i, "len": n} for i, n in enumerate(lens)]}
g = torch.Generator().manual_seed(7)
ts.load_f32 = lambda name, n: torch.randn(n, generator=g) * 0.02
model = ts.R2Model(d, seq)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
idx = torch.tensor([[(b * 131 + i * 7919 + 1) % vocab for i in range(seq)] for b in range(bn)])
tgt = torch.tensor([[(b * 131 + i * 7919 + 2) % vocab for i in range(seq)] for b in range(bn)])

def step():
    loss = ts.loss_on(model, idx, tgt)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

t0 = time.perf_counter()
while time.perf_counter() - t0 < 8.0: step()

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(steps): step()

by = collections.defaultdict(list)
for e in prof.events():
    if e.name in ("aten::mm", "aten::addmm", "aten::bmm") and e.input_shapes:
        key = (e.name, tuple(tuple(s) for s in e.input_shapes[:2]))
        by[key].append(e.time_range.elapsed_us())
print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads - GEMMs inside {steps} training steps\n")
print(f"{'op':<11} {'A shape':<14} {'B shape':<14} {'calls/step':>10} {'mean us':>9} {'p50 us':>8} {'max us':>8} {'ms/step':>8}")
print("-" * 90)
tot = 0.0
for key, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
    v.sort()
    per = len(v) / steps
    ms = sum(v) / steps / 1e3
    tot += ms
    print(f"{key[0]:<11} {str(list(key[1][0])):<14} {str(list(key[1][1])):<14} {per:>10.1f} {sum(v)/len(v):>9.1f} {v[len(v)//2]:>8.1f} {v[-1]:>8.1f} {ms:>8.2f}")
print("-" * 90)
print(f"all GEMMs: {tot:.1f} ms/step")
