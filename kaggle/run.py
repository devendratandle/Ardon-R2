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

sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_probe 2>&1 | tail -12")

test("-p r2-gpu --features cuda")
test("-p r2-train --features cuda --lib gpu_llm")
test("-p r2-engine --features cuda --test cuda_wiring")
test("-p r2-engine --features cuda --test llm_finetune")
# No visible device: the same wiring falls back, and says so.
test("-p r2-engine --features cuda --test cuda_wiring", env="CUDA_VISIBLE_DEVICES=")
sh(f"cd /tmp/r2 && {CARGO} check --release -p r2-repl --features cuda 2>&1 | tail -2")

# A real training step, at both report models, device against CPU.
sh(f"cd /tmp/r2 && R2_GPU_STATS=10 {CARGO} run --release -p r2-train --features cuda --example device_step 2>&1 | grep -vE '^ +(Compiling|Downloaded|Downloading|Finished|Running)'", tail=12000)
sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES= {CARGO} run --release -p r2-train --features cuda --example device_step 2>&1 | tail -3")
