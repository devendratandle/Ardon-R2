"""PyTorch's GEMM and attention on the T4, the shapes `cuda_bench` times."""
import time

import torch
import torch.nn.functional as F

dev = torch.device("cuda")
print("torch", torch.__version__, torch.cuda.get_device_name(0))


def timed(fn, reps=50):
    fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(5):
        s = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        v.append((time.perf_counter() - s) / reps)
    return sorted(v)[2]


print(f"{'t x k x n':>18} {'NN':>9} {'NT':>9} {'TN':>9}   GFLOP/s")
for m, k, n in [(2048, 256, 256), (2048, 256, 768), (2048, 768, 256), (2048, 256, 8000), (2048, 768, 768),
                (2048, 768, 2304), (2048, 2304, 768), (2048, 768, 8000), (2048, 2048, 2048)]:
    a = torch.randn(m, k, device=dev)
    b = torch.randn(k, n, device=dev)
    g = torch.randn(m, n, device=dev)
    fl = 2 * m * k * n / 1e9
    reps = max(2, min(50, int(100 / fl)))
    print(f"{f'{m}x{k}x{n}':>18} {fl / timed(lambda: a @ b, reps):>9.0f} "
          f"{fl / timed(lambda: g @ b.T, reps):>9.0f} {fl / timed(lambda: a.T @ g, reps):>9.0f}")

print(f"\n{'shape':<26} {'fwd ms':>9} {'bwd ms':>9}   SDPA")
for nseq, seq, nh, nkv, hd in [(32, 64, 4, 2, 64), (8, 256, 12, 4, 64)]:
    group = nh // nkv
    q = torch.randn(nseq, nh, seq, hd, device=dev, requires_grad=True)
    k = torch.randn(nseq, nkv, seq, hd, device=dev, requires_grad=True)
    v = torch.randn(nseq, nkv, seq, hd, device=dev, requires_grad=True)

    def fwd():
        return F.scaled_dot_product_attention(q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                                              is_causal=True)

    def both():
        fwd().sum().backward()

    f = timed(lambda: fwd().detach(), 20)
    fb = timed(both, 10)
    print(f"{f'{nseq}x{seq} {nh}/{nkv} hd{hd}':<26} {f * 1e3:>9.2f} {(fb - f) * 1e3:>9.2f}")
