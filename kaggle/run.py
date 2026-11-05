import subprocess, os

def sh(cmd, tail=8000):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"$ {cmd}\n{r.stdout[-tail:]}\n[exit {r.returncode}]", flush=True)
    return r.returncode

KERNELS = "crates/r2-gpu/src/cuda/kernels"
# The workspace profile is fat LTO; a test run would link for hours without this.
CARGO = "CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo"
TEST = f"cd /tmp/r2 && R2_REQUIRE_CUDA=1 {CARGO} test --release -p r2-gpu --features cuda"

sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
os.environ["PATH"] += ":/root/.cargo/bin"
sh("rm -rf /tmp/r2 && git clone --depth 1 --branch cuda-backend https://github.com/devendratandle/Ardon-R2.git /tmp/r2")
sh("cd /tmp/r2 && git log --oneline -1")
sh("nvidia-smi -L; nvcc --version | tail -2; echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH")

# Regenerate the PTX and hand it back through the kernel's output (only these
# few files go there; the build stays in /tmp). If the committed .ptx differs
# from what nvcc makes, the log says STALE and the new ones are what to commit.
sh(f"cd /tmp/r2/{KERNELS} && sh build.sh && ls -la && mkdir -p /kaggle/working/ptx && cp *.ptx /kaggle/working/ptx/")
sh(f"cd /tmp/r2 && git add -N {KERNELS} && if git diff --quiet -- {KERNELS}; then echo PTX up to date; else echo STALE; git diff --stat -- {KERNELS}; fi")

sh(f"cd /tmp/r2 && {CARGO} build --release -p r2-gpu --features cuda --example cuda_probe 2>&1 | tail -30")
sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_probe 2>&1 | tail -20")
# No visible device: the library loads, finds nothing, and everything falls back.
sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES= {CARGO} run --release -p r2-gpu --features cuda --example cuda_probe 2>&1 | tail -12")

# Step 1 of the port first, so its result reads on its own.
sh(f"{TEST} --lib cuda::tests -- --nocapture 2>&1 | grep -vE '^ +(Compiling|Downloaded|Downloading)' | tail -30")
sh(f"{TEST} --test cuda_dispatch 2>&1 | tail -12")
sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES= {CARGO} test --release -p r2-gpu --features cuda --test cuda_dispatch 2>&1 | tail -12")
sh(f"{TEST} --lib cuda::gemm 2>&1 | tail -30")
sh(f"{TEST} --lib cuda::attention 2>&1 | tail -30")
sh(f"{TEST} --lib cuda::optim 2>&1 | tail -30")
# The whole crate, CPU reference tests included, then the numbers.
sh(f"{TEST} 2>&1 | grep -E '^test |test result|panicked|error' | tail -40")
sh(f"cd /tmp/r2 && {CARGO} run --release -p r2-gpu --features cuda --example cuda_bench 2>&1 | grep -vE '^ +(Compiling|Finished|Running)'", tail=12000)
sh(f"cd /tmp/r2 && CUDA_VISIBLE_DEVICES= {CARGO} run --release -p r2-gpu --features cuda --example cuda_bench 2>&1 | tail -3")
