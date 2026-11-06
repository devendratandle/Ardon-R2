import subprocess, os

def sh(cmd, tail=8000):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"$ {cmd}\n{r.stdout[-tail:]}\n[exit {r.returncode}]", flush=True)
    return r.returncode

# What this run does. All of them take about 20 minutes.
STAGES = {"probe"}   # also: "tests", "two", "audit", "bench", "pairs", "census"

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

if "tests" in STAGES: test("-p r2-gpu --features cuda")
if "tests" in STAGES: test("-p r2-train --features cuda --lib gpu_llm")
if "tests" in STAGES: test("-p r2-engine --features cuda --test cuda_wiring")
if "tests" in STAGES: test("-p r2-engine --features cuda --test llm_finetune")
if "tests" in STAGES: test("-p r2-engine --features cuda --test cuda_wiring", env="CUDA_VISIBLE_DEVICES=")

# ── R2 on CUDA against PyTorch on CUDA, same card, interleaved ──────────────
# Both sides read the same token files and start from the same weights. The
# corpus is not in the repository; these bytes of TinyStories are the corpus.txt
# behind benchmarks/llm/REPORT.md (sha256 94e43181...).
sh("cd /tmp/r2 && curl -sSfL -r 0-19447281 -o corpus.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt; ls -la corpus.txt; sha256sum corpus.txt")
sh("python3 -c 'import numpy, tokenizers, torch; print(torch.__version__, torch.cuda.get_device_name(0))' || pip install -q numpy tokenizers")
sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-train --features cuda --example tinystories_train 2>&1 | tail -1")
if "bench" in STAGES:
    sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_bench 2>&1 | grep -vE '^ +(Compiling|Finished|Running)'", tail=12000)
    sh("cd /tmp/r2 && python3 kaggle/torch_kernels.py 2>&1", tail=6000)

PICK = "grep -E '^(train, seconds|tokens/s|ms/step|loss after training|train loss|ACCURACY)|device  |weights  |FAIL|panicked|rror'"
TRAIN = "/tmp/r2/target/release/examples/tinystories_train"
R2PICK = "grep -E 'device  |model |trained|HELD-OUT|cluster  |exchange  |weights  |rror|panicked'"

# ── Two GPUs, one process each ──────────────────────────────────────────────
# Rank 1 has its own directory, so the two do not write one token file. One GPU
# on each side first, in the same window: what two GPUs are worth is the ratio.
# R2 twice, because its bits repeating is one of the claims; PyTorch twice to see.
def r2_two(env, steps):
    sh(f"mkdir -p /tmp/rank1 && ln -sf /tmp/r2/corpus.txt /tmp/rank1/corpus.txt; cd /tmp/rank1 && "
       f"(CUDA_VISIBLE_DEVICES=1 R2_WORLD=2 R2_RANK=1 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} > /tmp/rank1.log 2>&1 &); "
       f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 R2_WORLD=2 R2_RANK=0 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}; "
       f"sleep 2; echo rank 1:; {R2PICK} /tmp/rank1.log")

def torch_two(env):
    py = f"{env} TS_DEVICE=cuda TS_WORLD=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 python3 benchmarks/llm/tinystories_train.py"
    sh(f"cd /tmp/r2 && (TS_RANK=1 {py} > /tmp/torch1.log 2>&1 &); TS_RANK=0 {py} 2>&1 | {PICK}; "
       f"sleep 2; echo rank 1:; grep -E 'weights  |rror|Traceback' /tmp/torch1.log")

def two_gpus(name, env, steps, n):
    for i in range(n):
        print(f"\n=== {name} round {i + 1} of {n}, one GPU: {env}, {steps} steps", flush=True)
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}")
        sh(f"cd /tmp/r2 && {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")
        print(f"\n=== {name} round {i + 1} of {n}, two GPUs", flush=True)
        r2_two(env, steps)
        torch_two(env)

def audit(name, env, steps, n):
    for i in range(n):
        print(f"\n=== audit {name} round {i + 1} of {n}", flush=True)
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 R2_PROFILE=1 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | grep -E 'phases  |trained|rror|panicked'")
        sh(f"cd /tmp/r2 && TS_PROFILE=1 {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | grep -E 'phases  |^ms/step|rror|Traceback'")

def pairs(name, env, steps, n):
    for i in range(n):
        print(f"\n=== {name} pair {i + 1} of {n}: {env}, {steps} steps", flush=True)
        sh(f"cd /tmp/r2 && nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0; {env} R2_STEPS={steps} R2_GPU=1 ./target/release/examples/tinystories_train 2>&1 | grep -E 'device  |model |trained|HELD-OUT|rror|panicked'")
        sh(f"cd /tmp/r2 && {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")

SMALL = "R2_DIM=256 R2_LAYERS=4 R2_FFN=768 R2_HEADS=4 R2_KV=2 R2_SEQ=64 R2_BATCH=32"
MEDIUM = "R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8"
# one held-out batch: the harness evaluates on the CPU tape, and the default eight
# in one tape (32,768 tokens through 12 layers) get the process killed for memory
LARGE = "R2_VAL_BATCHES=1 R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8"
if "pairs" in STAGES:
    pairs("small 7.2M", SMALL, 300, 3)
    pairs("medium 39.8M", MEDIUM, 100, 3)
    sh("free -g | head -2")
    pairs("large 125M", LARGE, 30, 2)
if "probe" in STAGES:
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_peer_probe 2>&1 | tail -1")
    P = "/tmp/r2/target/release/examples/cuda_peer_probe"
    sh(f"rm -rf /tmp/probe; (R2_CUDA_DEVICE=0 {P} 0 /tmp/probe > /tmp/probe0.log 2>&1 &); R2_CUDA_DEVICE=1 {P} 1 /tmp/probe 2>&1; sleep 2; cat /tmp/probe0.log")
    # the host-memory road, for comparison and in case the direct one is closed
    sh("nvidia-smi topo -m; lspci | grep -i nvidia; df -h /dev/shm | tail -1")
if "audit" in STAGES:
    sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_bench -- layout 2>&1 | grep -vE '^ +(Compiling|Finished|Running)'")
    # what PyTorch really launches, at the sizes where it leads: R2 first, for its manifest
    for name, env, steps in [("medium", MEDIUM, 12), ("large", LARGE, 12)]:
        print(f"\n=== kineto {name}", flush=True)
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | tail -1")
        sh(f"cd /tmp/r2 && TS_KINETO=1 {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | grep -A85 'kineto:'", tail=16000)
    audit("small 7.2M", SMALL, 300, 2)
    audit("medium 39.8M", MEDIUM, 100, 2)
    audit("large 125M", LARGE, 30, 2)
if "two" in STAGES:
    sh("nvidia-smi -L; nproc; free -g | head -2")
    two_gpus("small 7.2M", SMALL, 300, 2)
    two_gpus("medium 39.8M", MEDIUM, 100, 2)
    two_gpus("large 125M", LARGE, 30, 2)
sh("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0")

# Where the step goes now, per kernel (the queue is drained after each launch, so
# the distribution is the information, not the total).
for name, env in [] if "census" not in STAGES else [("medium", "R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8"),
                  ("large", "R2_VAL_BATCHES=1 R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8")]:
    print(f"\n=== census {name}", flush=True)
    sh(f"cd /tmp/r2 && {env} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 ./target/release/examples/tinystories_train 2>&1 | grep -A16 'GPU step census'; true")
