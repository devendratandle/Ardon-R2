"""PyTorch's own matrix product held for 30 s on the shape cuda_ceiling sustains, random
operands, in f32 and in f16: what the card gives the reference, under the same sampler.

    python3 kaggle/torch_sustain.py f16|f32
"""
import sys, time
import torch

dtype = torch.float16 if sys.argv[1] == "f16" else torch.float32
t, k, m = 4096, 768, 3072
g = torch.Generator(device="cuda").manual_seed(1)
a = (torch.rand(t, k, device="cuda", generator=g) - 0.5).to(dtype)
b = (torch.rand(k, m, device="cuda", generator=g) - 0.5).to(dtype)
c = a @ b
torch.cuda.synchronize()
flops, start = 0.0, time.perf_counter()
while time.perf_counter() - start < 30.0:
    for _ in range(20):
        torch.matmul(a, b, out=c)
        flops += 2.0 * t * k * m
    torch.cuda.synchronize()
print(f"CEILING sustained_torch_{sys.argv[1]} {flops / (time.perf_counter() - start) / 1e9:.1f} GFLOP/s")
# error against f32 of the same operands, relative to the product's own size
ref = a.float() @ b.float()
err = (c.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
print(f"CEILING accuracy_torch_{sys.argv[1]} rms {err.item():.3e}")
