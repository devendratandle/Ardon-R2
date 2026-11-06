import subprocess, os

def sh(cmd, tail=8000):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"$ {cmd}\n{r.stdout[-tail:]}\n[exit {r.returncode}]", flush=True)
    return r.returncode

KERNELS = "crates/r2-gpu/src/cuda/kernels"
# The workspace profile is fat LTO; a test run would link for hours without this.
CARGO = "CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo"
# `| tail` hides cargo's exit code, so every step greps its own results.
RESULTS = "grep -E '^test |test result|panicked|^error|Running' | tail -40"
def test(args, env="R2_REQUIRE_CUDA=1"):
    sh(f"cd /tmp/r2 && {env} {CARGO} test --release {args} 2>&1 | {RESULTS}")

sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
os.environ["PATH"] += ":/root/.cargo/bin"
sh("rm -rf /tmp/r2 && git clone --depth 1 --branch cuda-backend https://github.com/devendratandle/Ardon-R2.git /tmp/r2")
sh("cd /tmp/r2 && git log --oneline -1")
sh("nvidia-smi -L; nvcc --version | tail -2")

# Regenerate the PTX and hand it back through the kernel's output (only these
# few files go there; the build stays in /tmp). If the committed .ptx differs
# from what nvcc makes, the log says STALE and the new ones are what to commit.
sh(f"cd /tmp/r2/{KERNELS} && sh build.sh && mkdir -p /kaggle/working/ptx && cp *.ptx /kaggle/working/ptx/")
sh(f"cd /tmp/r2 && git add -N {KERNELS} && if git diff --quiet -- {KERNELS}; then echo PTX up to date; else echo STALE; git diff --stat -- {KERNELS}; fi")

test("-p r2-gpu --features cuda")
test("-p r2-train --features cuda --lib gpu_llm")
test("-p r2-engine --features cuda --test cuda_wiring")
test("-p r2-engine --features cuda --test llm_finetune")
test("-p r2-engine --features cuda --test cuda_wiring", env="CUDA_VISIBLE_DEVICES=")

# ── R2 on CUDA against PyTorch on CUDA, same card, interleaved ──────────────
# Both sides read the same token files and start from the same weights. The
# corpus is not in the repository; these bytes of TinyStories are the corpus.txt
# behind benchmarks/llm/REPORT.md (sha256 94e43181...).
sh("cd /tmp/r2 && curl -sSfL -r 0-19447281 -o corpus.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt; ls -la corpus.txt; sha256sum corpus.txt")
sh("python3 -c 'import numpy, tokenizers, torch; print(torch.__version__, torch.cuda.get_device_name(0))' || pip install -q numpy tokenizers")
sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-train --features cuda --example tinystories_train 2>&1 | tail -1")
sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_bench 2>&1 | grep -vE '^ +(Compiling|Finished|Running)'", tail=12000)
sh("cd /tmp/r2 && python3 kaggle/torch_kernels.py 2>&1", tail=6000)

PICK = "grep -E '^(train, seconds|tokens/s|ms/step|loss after training|train loss|ACCURACY)|device  |FAIL|panicked|rror'"
def pairs(name, env, steps, n):
    for i in range(n):
        print(f"\n=== {name} pair {i + 1} of {n}: {env}, {steps} steps", flush=True)
        sh(f"cd /tmp/r2 && nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0; {env} R2_STEPS={steps} R2_GPU=1 ./target/release/examples/tinystories_train 2>&1 | grep -E 'device  |model |trained|HELD-OUT|rror|panicked'")
        sh(f"cd /tmp/r2 && {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")

pairs("small 7.2M", "R2_DIM=256 R2_LAYERS=4 R2_FFN=768 R2_HEADS=4 R2_KV=2 R2_SEQ=64 R2_BATCH=32", 300, 3)
pairs("medium 39.8M", "R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8", 100, 3)
# one held-out batch: the harness evaluates on the CPU tape, and the default eight
# in one tape (32,768 tokens through 12 layers) get the process killed for memory
sh("free -g | head -2")
pairs("large 125M", "R2_VAL_BATCHES=1 R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8", 30, 2)
sh("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0")

# Where the step goes now, per kernel (the queue is drained after each launch, so
# the distribution is the information, not the total).
for name, env in [("medium", "R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8"),
                  ("large", "R2_VAL_BATCHES=1 R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8")]:
    print(f"\n=== census {name}", flush=True)
    sh(f"cd /tmp/r2 && {env} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 ./target/release/examples/tinystories_train 2>&1 | grep -A16 'GPU step census'; true")
