"""Run this branch on Kaggle without pushing it: what differs from the branch run.py clones
goes inside a copy of run.py, as a patch, and that copy is pushed as the private kernel.

    python kaggle/push_unpushed.py [base, default origin/cuda-backend]

Raw logs are left out, as they only make the patch large.
"""
import base64, gzip, json, pathlib, shutil, subprocess, sys, tempfile

base = sys.argv[1] if len(sys.argv) > 1 else "origin/cuda-backend"
here = pathlib.Path(__file__).parent
patch = subprocess.run(["git", "diff", "--binary", base, "HEAD", "--", ".", ":(exclude)benchmarks/llm/kaggle-logs"],
                       stdout=subprocess.PIPE, check=True).stdout
source = (here / "run.py").read_text(encoding="utf8")
assert source.count('PATCH = ""') == 1, "run.py has no empty PATCH"
blob = base64.b64encode(gzip.compress(patch, 9)).decode()
with tempfile.TemporaryDirectory() as out:
    out = pathlib.Path(out)
    (out / "run.py").write_text(source.replace('PATCH = ""', f'PATCH = "{blob}"'), encoding="utf8", newline="\n")
    shutil.copy(here / "kernel-metadata.json", out)
    assert json.loads((out / "kernel-metadata.json").read_text())["is_private"], "the kernel must stay private"
    print(f"patch {len(patch)} bytes, {len(blob)} in the script")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)
