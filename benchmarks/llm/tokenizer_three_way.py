"""tiktoken vs HuggingFace vs Ardon-R2, on ONE vocabulary.

All three implement byte-level BPE and all three have a Rust core; the
Python two reach it through PyO3. `tiktoken.get_encoding("gpt2")` is the
same 50,257-token vocabulary as the `tokenizer.json` R2 loads, so this
compares implementations rather than dictionaries.

WHAT IS AND IS NOT BEING MEASURED. Both Python libraries are timed as a
user calls them, which includes the PyO3 boundary, and HuggingFace's
`encode` additionally builds offsets, type ids and an attention mask that
neither tiktoken nor R2 produces. So HF's number is not a like-for-like
kernel comparison and is not presented as one. tiktoken's `encode_ordinary`
is the closest analogue to what R2 does: text in, token ids out.

    python benchmarks/llm/tokenizer_three_way.py <gpt2_tokenizer.json> [r2_ids.txt]
"""

import statistics
import sys
import time

CASES = [
    "hello world", " hello", "hello", "don't stop",
    "It's 42 apples, isn't it?", "a   b", "  leading", "trailing  ",
    "x=1;y=2", "The quick brown fox jumps over the lazy dog.",
    "1234567890", "CamelCaseIdentifier", "snake_case_name",
    "tabs\tand\nnewlines", "日本語のテキスト", "emoji 🙂 and 🎉 mixed",
    "café naïve résumé", "  ", "a", "",
]


def med(fn, reps=5):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        n = fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), n


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "gpt2_tokenizer.json"
    r2_path = sys.argv[2] if len(sys.argv) > 2 else None

    body = " ".join(CASES) * 2000
    nbytes = len(body.encode("utf-8"))

    rows = []

    import tiktoken
    tk = tiktoken.get_encoding("gpt2")
    secs, n = med(lambda: len(tk.encode_ordinary(body)))
    rows.append(("tiktoken (Rust core, PyO3)", secs, n))

    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(json_path)
    secs, n = med(lambda: len(hf.encode(body).ids))
    rows.append(("HuggingFace (Rust core, PyO3)", secs, n))

    # R2's figure comes from its own run — it is a compiled binary and has
    # no Python entry point, which is the whole structural difference.
    if r2_path:
        for line in open(r2_path, encoding="utf-8"):
            if line.startswith("THROUGHPUT"):
                kv = dict(x.split("=") for x in line.split()[1:])
                rows.append(("Ardon-R2 (native, no FFI)",
                             float(kv["secs"]), int(kv["tokens"])))

    print(f"encoding {nbytes/1e6:.2f} MB with GPT-2's 50,257-token vocabulary\n")
    print(f"{'implementation':<32}{'secs':>9}{'MB/s':>9}{'tokens':>10}")
    print("-" * 60)
    for name, secs, n in rows:
        print(f"{name:<32}{secs:>9.4f}{nbytes/secs/1e6:>9.2f}{n:>10}")

    counts = {n for _, _, n in rows}
    print("-" * 60)
    if len(counts) == 1:
        print(f"all implementations produced the SAME token count ({counts.pop()})")
    else:
        print(f"TOKEN COUNTS DIFFER: {counts} — not the same segmentation")

    # Do tiktoken and HuggingFace agree id for id? If they do, matching
    # either one is matching the standard.
    diff = [c for c in CASES if tk.encode_ordinary(c) != hf.encode(c).ids]
    print(f"tiktoken vs HuggingFace: {len(CASES)-len(diff)}/{len(CASES)} cases identical"
          + (f", differing on {diff}" if diff else ""))


if __name__ == "__main__":
    main()
