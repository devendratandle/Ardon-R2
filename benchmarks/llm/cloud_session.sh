#!/usr/bin/env bash
# One rented GPU, one command: R2's GPU training step against PyTorch+CUDA
# on the SAME card, interleaved, at three model sizes, plus the kernel
# benches and the r2-gpu tests on a second driver.
#
# Written for a vast.ai / RunPod pod running a `pytorch/pytorch` image
# (Ubuntu, CUDA torch preinstalled, root). Idempotent: re-running skips
# what is already installed or built.
#
#   bash benchmarks/llm/cloud_session.sh            # everything
#   bash benchmarks/llm/cloud_session.sh setup      # tools, Rust, Vulkan check
#   bash benchmarks/llm/cloud_session.sh build      # build + r2-gpu tests
#   bash benchmarks/llm/cloud_session.sh pairs      # the interleaved pairs
#   bash benchmarks/llm/cloud_session.sh kernels    # gemm_bench, attn_bench, probe
#
# Needs corpus.txt in the repo root (scp it up: 19 MB of TinyStories).
# Every number lands in cloud_results/<timestamp>/ as plain text, one file
# per run, so nothing has to be read off a scrolling terminal.

set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
OUT="cloud_results/$(date +%Y%m%d-%H%M%S)"
PAIRS="${PAIRS:-3}"
STEPS="${STEPS:-100}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# python may be python3; keep whichever exists
PY="${PY:-$(command -v python || command -v python3)}"

setup() {
    log "tools"
    if [ -f /etc/debian_version ] && ! command -v vulkaninfo >/dev/null; then
        apt-get update -qq && apt-get install -y -qq build-essential curl git vulkan-tools libvulkan1 >/dev/null
    fi
    log "Vulkan sees the card? (R2 talks to the GPU through wgpu/Vulkan, not CUDA)"
    if command -v vulkaninfo >/dev/null; then
        if ! vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName"; then
            echo "!! vulkaninfo found no device. The container lacks the Vulkan ICD."
            echo "   On vast.ai pick a host that lists 'vulkan' / has NVIDIA_DRIVER_CAPABILITIES=all;"
            echo "   on RunPod choose a different template. PyTorch will run; R2's GPU path cannot."
            exit 1
        fi
    else
        echo "   (no vulkaninfo; the adapter probe below is the real test)"
    fi
    log "Rust"
    if ! command -v cargo >/dev/null; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
    fi
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    rustc --version
    log "Python side"
    "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    "$PY" -c "import numpy, tokenizers" 2>/dev/null || "$PY" -m pip install -q numpy tokenizers
    nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
    [ -f corpus.txt ] || { echo "!! corpus.txt missing in $(pwd) — copy it up (19 MB of TinyStories)"; exit 1; }
}

build() {
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    log "build (first time ~10 min)"
    cargo build --release -p r2-train --features gpu --example tinystories_train
    cargo build --release -p r2-gpu --features gpu --examples
    log "r2-gpu tests on this driver's WGSL compiler"
    cargo test --release -p r2-gpu --features gpu 2>&1 | tee "$OUT/r2-gpu-tests.txt" | grep -E "^test |test result"
    log "adapter as wgpu sees it"
    ./target/release/examples/probe | tee "$OUT/probe.txt"
}

# one interleaved pair set at one model size: R2 GPU, then PyTorch+CUDA,
# PAIRS times, from the same manifest, joined table per pair
pairs_at() {
    local name="$1"; shift
    log "pairs: $name  ($* ; $STEPS steps x $PAIRS pairs)"
    for i in $(seq 1 "$PAIRS"); do
        env "$@" R2_STEPS="$STEPS" R2_GPU=1 ./target/release/examples/tinystories_train \
            2>&1 | tee "$OUT/$name-pair$i-r2.txt" | grep -E "device|model |trained|HELD-OUT"
        env "$@" TS_DEVICE=cuda "$PY" benchmarks/llm/tinystories_train.py \
            2>&1 | tee "$OUT/$name-pair$i-torch.txt" | grep -E "torch |^ms/step|held-out loss|FAIL"
    done
}

pairs() {
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    mkdir -p "$OUT"
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee "$OUT/clock-before.txt"
    # Each harness prints what it will ask for before it asks
    # ("memory ~X GB host, ~Y GB on the device"); the figures below are
    # what that estimate gives, so a card can be matched to a size.
    pairs_at small  R2_DIM=256 R2_LAYERS=4 R2_FFN=768  R2_HEADS=4  R2_KV=2 R2_SEQ=64  R2_BATCH=32   # 7.2M,  0.4 GB
    pairs_at medium R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8    # 39.8M, 1.2 GB
    # ~125M, GPT-2 small's shape: 4.7 GB on the device and 2 GB of host
    # RAM, which is why the development laptop (7.4 GB shared with its
    # iGPU) cannot run it at all and a rented card is the point.
    pairs_at large  R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8  # 125.5M, 4.7 GB
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | tee "$OUT/clock-after.txt"
    log "summary"
    grep -H "^ms/step" "$OUT"/*-torch.txt | sed 's|.*/||'
}

kernels() {
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    mkdir -p "$OUT"
    log "GPU sgemm vs CPU sgemm, per shape"
    ./target/release/examples/gemm_bench | tee "$OUT/gemm_bench.txt"
    log "attention kernels vs the tape's"
    ./target/release/examples/attn_bench | tee "$OUT/attn_bench.txt"
    log "cuBLAS and SDPA on the same shapes, for the kernel-to-kernel row"
    "$PY" - <<'EOF' | tee "$OUT/torch_kernels.txt"
import time, torch, torch.nn.functional as F
dev = torch.device("cuda")
def t(fn, reps=50):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / reps
print(f"{'t x k x n':>18} {'NN':>9} {'NT':>9} {'TN':>9}   cuBLAS GFLOP/s")
for (m, k, n) in [(2048,256,256),(2048,256,768),(2048,768,256),(2048,256,8000),(2048,768,768),(2048,768,2304),(2048,2304,768),(2048,768,8000),(4096,768,3072),(4096,3072,768)]:
    a = torch.randn(m, k, device=dev); b = torch.randn(k, n, device=dev); g = torch.randn(m, n, device=dev)
    fl = 2*m*k*n/1e9
    nn = fl / t(lambda: a @ b); nt = fl / t(lambda: g @ b.T); tn = fl / t(lambda: a.T @ g)
    print(f"{f'{m}x{k}x{n}':>18} {nn:>9.0f} {nt:>9.0f} {tn:>9.0f}")
print()
print(f"{'shape':<26} {'fwd ms':>9} {'fwd+bwd ms':>11}   SDPA (flash)")
for (nseq, seq, nh, hd) in [(32,64,4,64),(8,256,12,64),(8,512,12,64),(4,2048,12,64)]:
    q = torch.randn(nseq, nh, seq, hd, device=dev, requires_grad=True)
    k = torch.randn(nseq, nh, seq, hd, device=dev, requires_grad=True)
    v = torch.randn(nseq, nh, seq, hd, device=dev, requires_grad=True)
    f = t(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    def fb():
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True); o.sum().backward()
    fbt = t(fb, 20)
    print(f"{f'{nseq}x{seq} {nh} heads hd{hd}':<26} {f*1e3:>9.3f} {fbt*1e3:>11.3f}")
EOF
}

mkdir -p "$OUT"
case "${1:-all}" in
    setup)   setup ;;
    build)   build ;;
    pairs)   pairs ;;
    kernels) kernels ;;
    all)     setup; build; kernels; pairs ;;
    *)       echo "usage: $0 [setup|build|pairs|kernels|all]"; exit 2 ;;
esac
log "results in $OUT"
