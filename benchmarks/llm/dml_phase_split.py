"""Where PyTorch-DirectML spends a training step on this machine's GPU:
forward, backward and the optimizer, timed apart, on the manifest's model.

Companion to `tinystories_train.py` with TS_DEVICE=dml; run after the
R2 half so `ts_run/` exists.

    G:/r2-target/venv-dml/Scripts/python.exe benchmarks/llm/dml_phase_split.py
"""
import json
import os
import sys
import time

import torch

sys.argv = [sys.argv[0]]
os.environ.setdefault("TS_DEVICE", "dml")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinystories_train as ts  # noqa: E402


def sync(x):
    """DirectML has no explicit sync; reading one element back waits for
    everything queued before it."""
    return float(x.reshape(-1)[0].item())


def main():
    d = json.load(open(os.path.join(ts.OUT, "r2.json")))
    bn, seq = d["batch"], d["seq"]
    ids = ts.load_ids("train_ids.bin")
    model = ts.R2Model(d, max(seq, 64)).to(ts.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=d["lr"], betas=(0.9, 0.999), eps=1e-8, foreach=False)
    idx, tgt = ts.batch_at(ids, 0, bn, seq)
    print(f"device {ts.DEV_NAME}; model {d['n_params']/1e6:.2f}M, batch {bn} x {seq}")

    def timed(label, fn, reps=3):
        fn()                      # warm: shader compiles, allocations
        ts_ = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts_.append(time.perf_counter() - t0)
        ts_.sort()
        print(f"  {label:<28} {ts_[len(ts_)//2]*1e3:>10.1f} ms")

    def fwd():
        with torch.no_grad():
            sync(model(idx))

    def fwd_bwd():
        loss = ts.loss_on(model, idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync(next(model.parameters()).grad)

    def step():
        loss = ts.loss_on(model, idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(next(model.parameters()))

    timed("forward (no grad)", fwd)
    timed("forward + backward", fwd_bwd)
    timed("full step (with Adam)", step)


if __name__ == "__main__":
    main()
