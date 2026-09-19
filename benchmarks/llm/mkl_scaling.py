import torch, time, statistics
shapes = [(2048,256,8000,"output head"),(2048,256,768,"ffn w1/w3"),(2048,768,256,"ffn w2"),(2048,256,256,"q/o proj"),(2048,256,128,"k/v proj")]
def rate(f, flop):
    f(); ts=[]
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(3): f()
        ts.append((time.perf_counter()-t0)/3)
    return flop/statistics.median(ts)/1e9
print(f"torch {torch.__version__} MKL sgemm: serial (1 thread) vs 6 threads, GF/s")
print(f"{'block':>14} {'case':>5} {'1 thr':>8} {'6 thr':>8} {'scale':>6}")
for m,k,n,label in shapes:
    a=torch.randn(m,k); b=torch.randn(k,n); g=torch.randn(m,n)
    flop=2*m*k*n
    for case,f in [("NN",lambda: a@b),("NT",lambda: g@b.T),("TN",lambda: a.T@g)]:
        torch.set_num_threads(1); r1=rate(f,flop)
        torch.set_num_threads(6); r6=rate(f,flop)
        print(f"{label:>14} {case:>5} {r1:>8.1f} {r6:>8.1f} {r6/r1:>6.2f}x")
