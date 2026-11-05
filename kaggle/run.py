import subprocess, os

def sh(cmd, tail=6000):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"$ {cmd}\n{r.stdout[-tail:]}", flush=True)

sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
os.environ["PATH"] += ":/root/.cargo/bin"
sh("rm -rf /tmp/r2 && git clone --depth 1 --branch cuda-backend https://github.com/devendratandle/Ardon-R2.git /tmp/r2")
sh("nvidia-smi | head -15")
sh("nvcc --version | tail -2")
sh("cd /tmp/r2 && cargo test --release -p r2-gpu --features cuda 2>&1 | tail -40")
sh("cd /tmp/r2 && cargo run --release -p r2-gpu --features cuda --example cuda_probe 2>&1 | tail -20")