"""PyTorch's step by op, INSIDE training steps, forward and backward
profiled separately - the mirror of `R2_TAPE_STATS=1 R2_GEMM_STATS=1
--example phase_split` (its forward/backward census and GEMM-per-phase
lines). Self CPU time per aten op, grouped into GEMM / attention /
elementwise / other, so the non-GEMM part of each phase can be read off
on both sides at any model size.

    python benchmarks/llm/op_profile.py
    R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8 \
        python benchmarks/llm/op_profile.py
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

GEMM = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm"}
def group(name):
    if name in GEMM: return "gemm"
    if "scaled_dot_product" in name or "flash_attention" in name: return "attention"
    return "other"

phase = {"forward": collections.defaultdict(float), "backward": collections.defaultdict(float),
         "optimizer": collections.defaultdict(float)}
wall = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
for _ in range(steps):
    s = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as pf:
        loss = ts.loss_on(model, idx, tgt); loss.item()
    wall["forward"] += time.perf_counter() - s
    opt.zero_grad(set_to_none=True)
    s = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as pb:
        loss.backward()
    wall["backward"] += time.perf_counter() - s
    s = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as po:
        opt.step()
    wall["optimizer"] += time.perf_counter() - s
    for ph, pr in (("forward", pf), ("backward", pb), ("optimizer", po)):
        for e in pr.key_averages():
            phase[ph][e.key] += e.self_cpu_time_total / 1e3   # ms, whole run

print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads, attention: {ts.ATTN} - "
      f"dim {dim} x {nl} layers, ffn {ffn}, {nh}/{nkv} heads, {bn} x {seq} tokens, {steps} steps\n")
for ph in ("forward", "backward", "optimizer"):
    rows = sorted(phase[ph].items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in rows) / steps
    grp = collections.defaultdict(float)
    for k, v in rows: grp[group(k)] += v / steps
    print(f"{ph}: wall {wall[ph] / steps * 1e3:.1f} ms/step, self-time sum {tot:.1f};  "
          f"gemm {grp['gemm']:.1f}  attention {grp['attention']:.1f}  other {grp['other']:.1f}")
    print(f"  {'op':<52} {'ms/step':>8} {'share':>7}")
    for k, v in rows[:14]:
        print(f"  {k:<52} {v / steps:>8.1f} {v / steps / tot * 100:>6.1f}%")
    print()
