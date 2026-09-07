"""Diff R2's GPT-2 token ids against HuggingFace's, on the same vocabulary.

Run the Rust half first:

    R2_GPT2_JSON=<path to gpt2_tokenizer.json> \
      cargo run --release -p r2-tensor --example gpt2_compare > r2_ids.txt
    python benchmarks/llm/gpt2_compare.py r2_ids.txt <path to gpt2_tokenizer.json>

Every other claim about matching GPT-2 is an argument from the published
algorithm. This is the only evidence: one vocabulary, one input, two
implementations, compared id for id.
"""

import sys
import time

from tokenizers import Tokenizer

CASES = [
    "hello world", " hello", "hello", "don't stop",
    "It's 42 apples, isn't it?", "a   b", "  leading", "trailing  ",
    "x=1;y=2", "The quick brown fox jumps over the lazy dog.",
    "1234567890", "CamelCaseIdentifier", "snake_case_name",
    "tabs\tand\nnewlines", "日本語のテキスト", "emoji 🙂 and 🎉 mixed",
    "café naïve résumé", "  ", "a", "",
]


def main():
    r2_path = sys.argv[1] if len(sys.argv) > 1 else "r2_ids.txt"
    json_path = sys.argv[2] if len(sys.argv) > 2 else "gpt2_tokenizer.json"

    hf = Tokenizer.from_file(json_path)
    r2 = {}
    r2_throughput = None
    for line in open(r2_path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("CASE "):
            _, idx, rest = line.split(" ", 2)
            r2[int(idx)] = ([] if not rest or rest == "" else
                            [int(x) for x in rest.split(",") if x != ""])
        elif line.startswith("THROUGHPUT"):
            r2_throughput = dict(
                kv.split("=") for kv in line.split()[1:])

    print(f"{'#':>3} {'case':<34} {'HF':>6} {'R2':>6}  verdict")
    print("-" * 72)
    agree = 0
    for i, case in enumerate(CASES):
        want = hf.encode(case).ids
        got = r2.get(i)
        if got is None:
            print(f"{i:>3} {case!r:<34} {len(want):>6} {'-':>6}  MISSING")
            continue
        ok = want == got
        agree += ok
        label = case.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{i:>3} {label!r:<34} {len(want):>6} {len(got):>6}  "
              f"{'match' if ok else 'DIFFER'}")
        if not ok:
            print(f"      HF: {want}")
            print(f"      R2: {got}")
            print(f"      HF pieces: {[hf.decode([t]) for t in want]}")

    print("-" * 72)
    print(f"{agree}/{len(CASES)} cases identical to HuggingFace")

    # Throughput on the same body the Rust side timed.
    body = " ".join(CASES) * 2000
    t0 = time.perf_counter()
    ids = hf.encode(body).ids
    secs = time.perf_counter() - t0
    hf_mbps = len(body.encode()) / secs / 1e6
    print(f"\nthroughput on {len(body.encode())/1e6:.1f} MB")
    print(f"  HuggingFace {hf_mbps:>8.2f} MB/s  ({len(ids)} tokens)")
    if r2_throughput:
        r2_mbps = float(r2_throughput["mbps"])
        print(f"  Ardon-R2    {r2_mbps:>8.2f} MB/s  "
              f"({r2_throughput['tokens']} tokens)")
        print(f"  HF is {hf_mbps / r2_mbps:.1f}x faster")


if __name__ == "__main__":
    main()
