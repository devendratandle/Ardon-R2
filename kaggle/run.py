import subprocess, os

def sh(cmd, tail=8000):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"$ {cmd}\n{r.stdout[-tail:]}\n[exit {r.returncode}]", flush=True)
    return r.returncode

# What this run does. All of them take about 20 minutes.
STAGES = {"tests", "fp16"}   # also: "leads", "settle", "lossgap", "energy", "two", "audit", "bench", "pairs", "census", "score"

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
# Work not yet on GitHub, carried in this script by kaggle/push_unpushed.py: a kernel can
# then test a branch without anything being pushed. Empty in the repository.
PATCH = ""
if PATCH:
    import base64, gzip
    with open("/tmp/unpushed.patch", "wb") as f:
        f.write(gzip.decompress(base64.b64decode(PATCH)))
    sh("cd /tmp/r2 && git apply --whitespace=nowarn /tmp/unpushed.patch && git status --short | wc -l")
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

PICK = "grep -E '^(train, seconds|tokens/s|ms/step|loss after training|train loss|ACCURACY)|device  |weights  |steptime|FAIL|panicked|rror'"
TRAIN = "/tmp/r2/target/release/examples/tinystories_train"
R2PICK = "grep -E 'device  |model |trained|steptime|HELD-OUT|cluster  |exchange  |weights  |rror|panicked'"

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
if "score" in STAGES:
    # Operation by operation, R2 and PyTorch alternating: benchmarks/llm/scoreboard.py reads these.
    SCORE = "grep -E '^(SCORE|device:)|rror|panicked|Traceback'"
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example op_scoreboard 2>&1 | tail -1; cp target/release/examples/op_scoreboard /tmp/score_cuda")
    for round_ in range(3):
        for name, env in [("small", SMALL), ("medium", MEDIUM), ("large", LARGE)]:
            print(f"\n=== score {name} round {round_ + 1}", flush=True)
            sh(f"cd /tmp/r2 && {env} SCORE_LAYER=t4 SCORE_SIZE={name} CUDA_VISIBLE_DEVICES=0 /tmp/score_cuda 2>&1 | {SCORE}", tail=12000)
            sh(f"cd /tmp/r2 && {env} SCORE_LAYER=t4 SCORE_SIZE={name} TS_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python3 benchmarks/llm/op_scoreboard.py 2>&1 | {SCORE}", tail=12000)
            # the gradient exchange, two processes on two GPUs
            sh(f"cd /tmp/r2 && (CUDA_VISIBLE_DEVICES=1 R2_WORLD=2 R2_RANK=1 {env} SCORE_LAYER=t4x2 SCORE_SIZE={name} /tmp/score_cuda > /tmp/x1.log 2>&1 &); "
               f"CUDA_VISIBLE_DEVICES=0 R2_WORLD=2 R2_RANK=0 {env} SCORE_LAYER=t4x2 SCORE_SIZE={name} /tmp/score_cuda 2>&1 | {SCORE}; sleep 2; grep -E 'rror|panicked' /tmp/x1.log")
            py = f"{env} SCORE_LAYER=t4x2 SCORE_SIZE={name} TS_DEVICE=cuda TS_WORLD=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29601 python3 benchmarks/llm/op_scoreboard.py"
            sh(f"cd /tmp/r2 && (TS_RANK=1 {py} > /tmp/x1.log 2>&1 &); TS_RANK=0 {py} 2>&1 | {SCORE}; sleep 2; grep -E 'rror|Traceback' /tmp/x1.log")
SCORE = "grep -E '^(SCORE|device:)|rror|panicked|Traceback'"
QUERY = "index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.active"
def sampled(cmd):
    """Runs cmd with both cards sampled each second; prints each card's readings under load."""
    s = subprocess.Popen(f"nvidia-smi --query-gpu={QUERY} --format=csv,noheader,nounits -l 1 > /tmp/gpu.csv", shell=True)
    sh(cmd)
    s.terminate()
    subprocess.run("pkill -f 'nvidia-smi --query-gpu' ; true", shell=True)
    rows = [[x.strip() for x in l.split(",")] for l in open("/tmp/gpu.csv") if l.count(",") == 6]
    for card in sorted({r[0] for r in rows}):
        busy = [r for r in rows if r[0] == card and r[5].isdigit() and int(r[5]) > 50]
        if not busy:
            continue
        med = lambda i: sorted(float(r[i]) for r in busy)[len(busy) // 2]
        print(f"card {card}: {len(busy)} s busy, median clock {med(1):.0f} MHz, memory {med(2):.0f} MHz, "
              f"{med(3):.0f} C, {med(4):.0f} W, throttle {sorted({r[6] for r in busy})}", flush=True)

if "energy" in STAGES:
    # The T4 holds a 70 W cap, so what a kernel costs in power decides its clock.
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_ceiling --example op_scoreboard 2>&1 | tail -1; "
       "cp target/release/examples/cuda_ceiling /tmp/ceiling; cp target/release/examples/op_scoreboard /tmp/score_cuda")
    sh("CUDA_VISIBLE_DEVICES=0 /tmp/ceiling 2>&1 | grep -E '^(CEILING|device:)|rror|panicked'", tail=6000)
    for card in (0, 1):
        for kind in ("fma", "cublas", "read", "copy"):
            print(f"\n=== sustain {kind} card {card}", flush=True)
            sh("sleep 45")
            sampled(f"CUDA_VISIBLE_DEVICES={card} /tmp/ceiling sustain {kind} 2>&1 | grep -E '^CEILING|rror|panicked'")
    for round_ in range(3):
        for name, env in [("small", SMALL), ("medium", MEDIUM), ("large", LARGE)]:
            sh(f"cd /tmp/r2 && {env} SCORE_LAYER=t4 SCORE_SIZE={name} CUDA_VISIBLE_DEVICES=0 /tmp/score_cuda 2>&1 | {SCORE}", tail=12000)
            sh(f"cd /tmp/r2 && {env} SCORE_LAYER=t4 SCORE_SIZE={name} TS_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python3 benchmarks/llm/op_scoreboard.py 2>&1 | {SCORE}", tail=12000)
    for round_ in range(2):
        peak = f"{LARGE} SCORE_AMP=10 SCORE_STEP=0.37 SCORE_LAYER=t4peak SCORE_SIZE=large CUDA_VISIBLE_DEVICES=0"
        sh(f"cd /tmp/r2 && {peak} /tmp/score_cuda 2>&1 | {SCORE}", tail=12000)
        sh(f"cd /tmp/r2 && {peak} TS_DEVICE=cuda python3 benchmarks/llm/op_scoreboard.py 2>&1 | {SCORE}", tail=12000)
    # Fair order: R2, PyTorch, PyTorch, R2, cooling before each, so neither always
    # runs on the card the other just heated.
    def r2_run(env, steps, card):
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES={card} {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}")
    def torch_run(env, card):
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES={card} {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")
    for name, env, steps, card in [("small", SMALL, 300, 0), ("medium", MEDIUM, 100, 0), ("large", LARGE, 30, 0), ("large", LARGE, 30, 1), ("long", LARGE, 100, 0)]:
        print(f"\n=== {name} pair fair order, card {card}, {steps} steps", flush=True)
        r2_run(env, steps, card)
        torch_run(env, card)
        torch_run(env, card)
        r2_run(env, steps, card)
    print("\n=== census large", flush=True)
    sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {LARGE} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 {TRAIN} 2>&1 | grep -A60 'GPU step census'; true")
if "settle" in STAGES:
    # Paired starts: each nudge moves one weight by one ulp, and both sides train from it.
    # The spread is what rounding alone does, so the means can be held against it.
    wide = LARGE.replace("R2_VAL_BATCHES=1", "R2_VAL_BATCHES=3")
    for k in range(1, 7):
        print(f"\n=== nudge {k} of 6", flush=True)
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {wide} R2_NUDGE={k} R2_STEPS=100 R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}")
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {wide} R2_NUDGE={k} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")
    # Whether fewer significant bits in the operands cost less energy, and so raise the clock.
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_ceiling 2>&1 | tail -1; cp target/release/examples/cuda_ceiling /tmp/ceiling")
    for card in (0, 1):
        for round_ in range(2):
            for bits in (24, 16, 12, 8):
                print(f"\n=== sustain bits {bits} card {card} round {round_ + 1}", flush=True)
                sh("sleep 45")
                sampled(f"CUDA_VISIBLE_DEVICES={card} /tmp/ceiling sustain cublas_bits_{bits} 2>&1 | grep -E '^CEILING|rror|panicked'")
    # PyTorch at its fastest, in fair order: R2, plain, compiled, compiled, plain, R2.
    def r2_turn(env, steps):
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}")
    def torch_turn(env, compiled):
        sh("sleep 45")
        print(f"--- PyTorch {'compiled' if compiled else 'plain'}", flush=True)
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} TS_COMPILE={int(compiled)} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")
    for name, env, steps in [("small", SMALL, 300), ("medium", MEDIUM, 100), ("large", LARGE, 30)]:
        print(f"\n=== compile pair {name}, {steps} steps", flush=True)
        r2_turn(env, steps)
        torch_turn(env, False)
        torch_turn(env, True)
        torch_turn(env, True)
        torch_turn(env, False)
        r2_turn(env, steps)
if "fp16" in STAGES:
    # Run 33's cublasGemmEx asked for algorithm 160 and got status 7 (invalid value). Now
    # the default algorithm, cuBLAS's own log for the accuracy call, and PyTorch's f16 and
    # f32 products held the same way on the same card, alternating with R2's.
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_ceiling 2>&1 | tail -1; "
       "cp target/release/examples/cuda_ceiling /tmp/ceiling")
    PICKC = "grep -E '^CEILING|refused|no cuBLAS|rror|panicked|Traceback'"
    sh(f"CUDA_VISIBLE_DEVICES=0 /tmp/ceiling accuracy 2>&1 | grep -E '^CEILING|refused|rror|panicked'")
    sh(f"CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=stdout CUDA_VISIBLE_DEVICES=0 /tmp/ceiling accuracy 2>&1 | grep -E 'refused|Atype|computeType' | tail -12")
    for round_ in range(2):
        for kind in ("gemm_ex_f32", "torch_f32", "gemm_ex_fast16", "gemm_ex_f16", "torch_f16"):
            print(f"\n=== sustain {kind} card 0 round {round_ + 1}", flush=True)
            sh("sleep 45")
            if kind.startswith("torch"):
                sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 python3 kaggle/torch_sustain.py {kind[6:]} 2>&1 | {PICKC}")
            else:
                sampled(f"CUDA_VISIBLE_DEVICES=0 /tmp/ceiling sustain {kind} 2>&1 | {PICKC}")
if "leads" in STAGES:
    # Tensor cores under the 70 W cap: f32 through plain cuBLAS and through cublasGemmEx, then
    # f32 storage with f16 arithmetic, then f16 throughout, on the same random operands.
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_ceiling --example op_scoreboard 2>&1 | tail -1; "
       "cp target/release/examples/cuda_ceiling /tmp/ceiling; cp target/release/examples/op_scoreboard /tmp/score_cuda")
    PICKC = "grep -E '^CEILING|refused|no cuBLAS|rror|panicked'"
    sh(f"CUDA_VISIBLE_DEVICES=0 /tmp/ceiling accuracy 2>&1 | {PICKC}")
    sh(f"CUDA_VISIBLE_DEVICES=0 /tmp/ceiling launch 2>&1 | {PICKC}")
    for card in (0, 1):
        for round_ in range(2):
            for kind in ("cublas_random", "gemm_ex_f32", "gemm_ex_fast16", "gemm_ex_f16"):
                print(f"\n=== sustain {kind} card {card} round {round_ + 1}", flush=True)
                sh("sleep 45")
                sampled(f"CUDA_VISIBLE_DEVICES={card} /tmp/ceiling sustain {kind} 2>&1 | {PICKC}")
    # Launches per step at 7.2M: this census's calls times the launch cost above.
    print("\n=== census small", flush=True)
    sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {SMALL} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 {TRAIN} 2>&1 | grep -A60 'GPU step census'; true")
    # The fused backward scores against the stored form, and PyTorch, in fair order.
    def turn(label, env, steps, extra=""):
        print(f"--- {label}", flush=True)
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} {extra} R2_STEPS={steps} R2_GPU=1 {TRAIN} 2>&1 | {R2PICK}")
    def torch_turn(env):
        print("--- PyTorch", flush=True)
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PICK}")
    STORED = "R2_CUDA_ATTN_SCORES=0"
    for name, env, steps in [("small", SMALL, 300), ("medium", MEDIUM, 100), ("large", LARGE, 30), ("large", LARGE, 30)]:
        print(f"\n=== scores pair {name}, {steps} steps: fused, stored, PyTorch, PyTorch, stored, fused", flush=True)
        turn("fused", env, steps)
        turn("stored", env, steps, STORED)
        torch_turn(env)
        torch_turn(env)
        turn("stored", env, steps, STORED)
        turn("fused", env, steps)
    # Attention alone, per call, each side's own row.
    for round_ in range(3):
        for label, extra in [("fused", ""), ("stored", STORED)]:
            print(f"\n=== score large {label} round {round_ + 1}", flush=True)
            sh(f"cd /tmp/r2 && {LARGE} {extra} SCORE_LAYER=t4 SCORE_SIZE=large CUDA_VISIBLE_DEVICES=0 /tmp/score_cuda 2>&1 | {SCORE}", tail=12000)
        sh(f"cd /tmp/r2 && {LARGE} SCORE_LAYER=t4 SCORE_SIZE=large TS_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python3 benchmarks/llm/op_scoreboard.py 2>&1 | {SCORE}", tail=12000)
if "lossgap" in STAGES:
    # Tiled products are the same maths as cuBLAS with other rounding, so how far they
    # move the loss says how far rounding alone can. Held-out every 10 steps shows the
    # gap's growth; three held-out batches show whether one batch overstates it (eight are killed for memory).
    R2LOSS = "grep -iE 'device  |trained|steptime|held-out|weights  |rror|panicked'"
    PTLOSS = "grep -iE 'steptime|ms/step|loss after training|held-out|weights  |FAIL|rror'"
    def r2_loss(env, extra=""):
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} {extra} R2_STEPS=100 R2_GPU=1 {TRAIN} 2>&1 | {R2LOSS}")
    def torch_loss(env):
        sh("sleep 45")
        sampled(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {env} TS_DEVICE=cuda python3 benchmarks/llm/tinystories_train.py 2>&1 | {PTLOSS}")
    curve = f"{LARGE} R2_EVAL_EVERY=10"
    print("\n=== loss curve, one held-out batch: cuBLAS, PyTorch, PyTorch, tiled", flush=True)
    r2_loss(curve)
    torch_loss(curve)
    torch_loss(curve)
    r2_loss(curve, "R2_CUDA_GEMM=tiled")
    wide = LARGE.replace("R2_VAL_BATCHES=1", "R2_VAL_BATCHES=3")
    print("\n=== three held-out batches: cuBLAS, PyTorch, tiled", flush=True)
    r2_loss(wide)
    torch_loss(wide)
    r2_loss(wide, "R2_CUDA_GEMM=tiled")
    # What operand data costs in clock, on the shape the sustained test already holds.
    sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_ceiling 2>&1 | tail -1; cp target/release/examples/cuda_ceiling /tmp/ceiling")
    for card in (0, 1):
        for kind in ("cublas", "cublas_random", "cublas", "cublas_random"):
            print(f"\n=== sustain {kind} card {card}", flush=True)
            sh("sleep 45")
            sampled(f"CUDA_VISIBLE_DEVICES={card} /tmp/ceiling sustain {kind} 2>&1 | grep -E '^CEILING|rror|panicked'")
    # Forward attention on real activations, then again on the same operands.
    for i in range(2):
        print(f"\n=== census large {i + 1} of 2", flush=True)
        sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES=0 {LARGE} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 {TRAIN} 2>&1 | grep -E 'GPU step census|attention'; true")
sh("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader -i 0")

# Where the step goes now, per kernel (the queue is drained after each launch, so
# the distribution is the information, not the total).
for name, env in [] if "census" not in STAGES else [("medium", "R2_DIM=768 R2_LAYERS=4 R2_FFN=2304 R2_HEADS=12 R2_KV=4 R2_SEQ=256 R2_BATCH=8"),
                  ("large", "R2_VAL_BATCHES=1 R2_DIM=768 R2_LAYERS=12 R2_FFN=3072 R2_HEADS=12 R2_KV=12 R2_SEQ=512 R2_BATCH=8")]:
    print(f"\n=== census {name}", flush=True)
    sh(f"cd /tmp/r2 && {env} R2_STEPS=12 R2_GPU_STATS=10 R2_GPU=1 ./target/release/examples/tinystories_train 2>&1 | grep -A60 'GPU step census'; true")
