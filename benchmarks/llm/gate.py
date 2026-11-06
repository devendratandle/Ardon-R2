"""The gate every optimization passes, read from a Kaggle run's log: accuracy, validity,
reliability and consistency. Speed is scoreboard.py's and the pairs' business; a change
that is faster and fails here does not stay.

    python benchmarks/llm/gate.py run.log

Accuracy      held-out loss within 0.0002 of PyTorch, every pair
Validity      the kernel tests, which check outputs and gradients against the tape
Reliability   every test binary ran and passed; no panic; no faulted kernel
Consistency   R2's final weights identical across rounds of a size, and across ranks
Same work     no scoreboard row whose two sides produced different output
"""
import collections, json, math, re, sys

raw = open(sys.argv[1], encoding="utf8", errors="replace").read()
try:
    text = "".join(e.get("data", "") for e in json.loads(raw))
except ValueError:
    text = raw

failures, notes = [], []
weights = collections.defaultdict(set)
tests = passed = losses = 0
section = ""
for line in text.splitlines():
    if line.startswith("=== "):
        m = re.match(r"=== (\w+).*?(one GPU|two GPUs|pair|kineto|audit|census|score)", line)
        section = f"{m.group(1)} {m.group(2)}" if m else line
    if line.startswith("test result:"):
        tests += 1
        m = re.search(r"(\d+) passed; (\d+) failed", line)
        passed += int(m.group(1))
        if int(m.group(2)):
            failures.append(line.strip())
    if re.search(r"panicked at|GPU kernel failed|Traceback \(most", line):
        failures.append(line.strip()[:120])
    m = re.match(r"\s*weights\s+fnv (\w+)", line)
    if m:
        weights[section].add(m.group(1))
    m = re.match(r"loss after training\s+([\d.]+)\s+([\d.]+)", line)
    if m:
        losses += 1
        if abs(float(m.group(1)) - float(m.group(2))) > 2e-4:
            failures.append(f"loss differs in {section}: {line.strip()}")
    if line.startswith("STALE"):
        notes.append("committed PTX is stale: commit the generated files before calling this final")

for key, seen in weights.items():
    if len(seen) > 1:
        failures.append(f"weights differ across rounds or ranks in {key}: {sorted(seen)}")

score = collections.defaultdict(dict)
for side, layer, size, op, phase, check in re.findall(r"SCORE (r2|torch) (\S+) (\S+) (\S+) (\S+) \d+ \S+ (\S+)", text):
    if not math.isnan(float(check)):
        score[(layer, size, op, phase)][side] = float(check)
rows = differ = 0
for key, c in score.items():
    if len(c) == 2:
        rows += 1
        if abs(c["r2"] - c["torch"]) > 1e-3 * max(abs(c["r2"]), abs(c["torch"]), 1e-9):
            differ += 1
            failures.append(f"different work in {key}: {c}")

print(f"tests: {passed} passed in {tests} binaries; losses compared: {losses}; "
      f"weight groups: {len(weights)}; scoreboard rows checked: {rows}, differing: {differ}")
for n in sorted(set(notes)):
    print("note:", n)
for f in failures:
    print("FAIL:", f)
print("GATE", "FAILED" if failures else "PASSED")
sys.exit(1 if failures else 0)
