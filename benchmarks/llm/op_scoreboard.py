"""PyTorch's side of `crates/r2-gpu/examples/op_scoreboard.rs`: each operation written
as `tinystories_train.R2Model` writes it, on the same shapes and data. Backward runs
through autograd on a retained graph, so it pays what a step pays and no forward.

    SCORE_LAYER=cpu  python benchmarks/llm/op_scoreboard.py
    SCORE_LAYER=vega TS_DEVICE=dml  python benchmarks/llm/op_scoreboard.py
    SCORE_LAYER=t4   TS_DEVICE=cuda python benchmarks/llm/op_scoreboard.py

The model comes from R2_DIM .. R2_BATCH, its name from SCORE_SIZE. With TS_WORLD=2
it times only the gradient exchange, block by block over NCCL. Scoreboard: scoreboard.py.
"""
import math, os, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinystories_train as ts

E = lambda k, d: int(os.environ.get(k, d))
dim, nl, ffn, vocab, nh, nkv = E("R2_DIM", 256), E("R2_LAYERS", 4), E("R2_FFN", 768), E("R2_VOCAB", 8000), E("R2_HEADS", 4), E("R2_KV", 2)
nseq, seq = E("R2_BATCH", 32), E("R2_SEQ", 64)
hd, t = dim // nh, nseq * seq
kv = nkv * hd
LAYER, SIZE = os.environ.get("SCORE_LAYER", "cpu"), os.environ.get("SCORE_SIZE", "small")
BUDGET = float(os.environ.get("SCORE_SECS", "0.15"))
DEV, DEVICE = ts.DEV, ts.DEVICE
torch.set_num_threads(E("TS_THREADS", 6))


def fence(x=None):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "dml" and x is not None:
        x.reshape(-1)[0].item()   # DirectML has no fence: one element read back


def fill(n, ph, grad=False):
    x = ((torch.arange(n, dtype=torch.float32) * 0.0007 + ph).sin()).to(DEV)
    return x.requires_grad_() if grad else x


def line(op, phase, count, ms, check=float("nan")):
    print(f"SCORE torch {LAYER} {SIZE} {op} {phase} {count} {ms:.4f} {check:.6e}", flush=True)


def timed(f):
    """Milliseconds per call: enough calls for BUDGET closed by one fence, median of 5."""
    fence(f())
    s = time.perf_counter()
    fence(f())
    reps = max(1, min(200, math.ceil(BUDGET / max(time.perf_counter() - s, 1e-6))))
    v = []
    for _ in range(5):
        s = time.perf_counter()
        for _ in range(reps):
            r = f()
        fence(r)
        v.append((time.perf_counter() - s) / reps * 1e3)
    return sorted(v)[2]


def abs_sum(*ts_):
    return sum(x.detach().double().abs().sum().item() if DEVICE != "dml" else x.detach().cpu().double().abs().sum().item() for x in ts_)


def row(op, count, leaves, build, seed=None):
    """Forward without keeping a graph between calls; backward on one retained graph."""
    def fwd():
        return build(*leaves)
    ms = timed(fwd)
    y = fwd()
    line(op, "fwd", count, ms, abs_sum(y))

    y = fwd()
    def bwd():
        y.backward(seed, retain_graph=True)
        g = leaves[0].grad
        for x in leaves:
            x.grad = None
        return g
    ms = timed(bwd)
    y.backward(seed, retain_graph=True)
    check = abs_sum(*[x.grad for x in leaves])
    for x in leaves:
        x.grad = None
    line(op, "bwd", count, ms, check)


def tokens(shift):
    return torch.tensor([[(b * 131 + i * 7919 + shift) % vocab for i in range(seq)] for b in range(nseq)])


stub = SimpleNamespace(eps=1e-5, base=10000.0, hd=hd)
cos, sin = ts.R2Model.rope_tables(stub, seq)
stub.rope_cos, stub.rope_sin = cos.to(DEV), sin.to(DEV)
sizes = [vocab * dim, dim, dim * vocab]
for _ in range(nl):
    sizes += [dim, dim * dim, dim * kv, dim * kv, dim * dim, dim, dim * ffn, ffn * dim, dim * ffn]


def exchange():
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=E("TS_RANK", 0), world_size=2)
    gs = [fill(n, 1.0) for n in sizes]
    def once():
        for g in gs:
            dist.all_reduce(g)
            g.div_(2)
    once()
    v = []
    for i in range(8):
        s = time.perf_counter()
        once()
        fence()
        v.append((time.perf_counter() - s) * 1e3)
    line("exchange", "step", 1, sorted(v[3:])[2])
    dist.destroy_process_group()


def main():
    print(f"device: {DEVICE} torch {torch.__version__}", flush=True)
    idx, tgt = tokens(1).to(DEV), tokens(2).to(DEV)

    # the model's own order of the calls
    row("embed", 1, [fill(vocab * dim, 3.0, True)],
        lambda p: p.view(vocab, dim)[idx.reshape(-1)].view(nseq, seq, dim), fill(t * dim, 2.0).view(nseq, seq, dim))
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + stub.eps) * w
    row("rmsnorm", 2 * nl + 1, [fill(t * dim, 0.0, True), fill(dim, 1.0, True)],
        lambda x, w: rms(x.view(nseq, seq, dim), w), fill(t * dim, 2.0).view(nseq, seq, dim))
    for name, k, n, c in [("gemm_wq", dim, dim, 2 * nl), ("gemm_wk", dim, kv, 2 * nl), ("gemm_w1", dim, ffn, 2 * nl),
                          ("gemm_w2", ffn, dim, nl), ("gemm_head", dim, vocab, 1)]:
        row(name, c, [fill(t * k, 0.0, True), fill(k * n, 1.0, True)],
            lambda a, b, k=k, n=n: a.view(nseq, seq, k) @ b.view(k, n), fill(t * n, 2.0).view(nseq, seq, n))
    for name, h, w in [("rope_q", nh, dim), ("rope_k", nkv, kv)]:
        row(name, nl, [fill(t * w, 0.0, True)],
            lambda x, h=h: ts.R2Model.rope(stub, x.view(nseq, seq, h, hd), seq), fill(t * w, 2.0).view(nseq, seq, h, hd))

    rep = nh // nkv
    def attention(q, k, v):
        q, k, v = q.view(nseq, seq, nh, hd), k.view(nseq, seq, nkv, hd), v.view(nseq, seq, nkv, hd)
        if DEVICE == "dml":
            k = torch.cat([k[:, :, i:i + 1] for i in range(nkv) for _ in range(rep)], dim=2)
            v = torch.cat([v[:, :, i:i + 1] for i in range(nkv) for _ in range(rep)], dim=2)
        else:
            k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return ctx.transpose(1, 2).reshape(nseq, seq, dim)
    row("attention", nl, [fill(t * dim, 0.0, True), fill(t * kv, 1.0, True), fill(t * kv, 2.0, True)],
        attention, fill(t * dim, 3.0).view(nseq, seq, dim))

    row("add", 2 * nl, [fill(t * dim, 0.0, True), fill(t * dim, 1.0, True)], lambda a, b: a + b, fill(t * dim, 2.0))
    row("silu", nl, [fill(t * ffn, 0.0, True)], lambda a: F.silu(a), fill(t * ffn, 2.0))
    row("mul", nl, [fill(t * ffn, 0.0, True), fill(t * ffn, 1.0, True)], lambda a, b: a * b, fill(t * ffn, 2.0))
    row("softmax_ce", 1, [fill(t * vocab, 0.0, True)], lambda x: F.cross_entropy(x.view(t, vocab), tgt.reshape(-1)))

    if DEVICE != "cpu":
        s = time.perf_counter()
        loss = F.cross_entropy(fill(t * vocab, 0.0).view(t, vocab), tgt.reshape(-1))
        fence(loss)
        s = time.perf_counter()
        loss.item()
        line("read_loss", "step", 1, (time.perf_counter() - s) * 1e3)
        np_i, np_t = tokens(1).numpy().astype(np.int64), tokens(2).numpy().astype(np.int64)
        line("batch_setup", "step", 1, timed(lambda: (torch.from_numpy(np_i).to(DEV), torch.from_numpy(np_t).to(DEV))[0]))

    params = [torch.nn.Parameter(fill(n, 0.0)) for n in sizes]
    for p, n in zip(params, sizes):
        p.grad = fill(n, 1.0)
    opt = torch.optim.Adam(params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, foreach=False if DEVICE == "dml" else None)
    def adam():
        opt.step()
        return params[-1].detach()
    line("adam", "step", 1, timed(adam))
    del params, opt

    if DEVICE != "cpu":
        a, b = fill(1, 1.0), fill(1, 2.0)
        line("launch_one", "info", 0, timed(lambda: a + b))
        big = fill(16 << 20, 0.0).cpu()
        dst = big.to(DEV)
        def copy(f):
            v = []
            for _ in range(5):
                s = time.perf_counter()
                f()
                v.append((time.perf_counter() - s) * 1e3)
            return sorted(v)[2]
        def up():
            fence(dst.copy_(big))
        line("upload_64MB", "info", 0, copy(up))
        line("download_64MB", "info", 0, copy(lambda: dst.cpu()))


if os.environ.get("TS_WORLD"):
    exchange()
else:
    main()
