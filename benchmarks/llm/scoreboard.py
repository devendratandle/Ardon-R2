"""The scoreboard: joins the SCORE lines of op_scoreboard.rs and op_scoreboard.py from
one or more logs. Every round on each side is one set of lines, in the order run, so
rounds pair up in time. A side wins a row only if it leads by 10% in EVERY round;
anything else is level (under 10% is not a result here). A row whose two sides'
outputs differ by over 0.1% is marked WORK DIFFERS and counts for nobody.

    python benchmarks/llm/scoreboard.py run.log [more.log ...]

Rows sort by what they cost R2 per step (calls per step x the gap per call), so the
top of a table is the next thing to work on.
"""
import collections, math, re, sys

rows = collections.defaultdict(lambda: {"r2": [], "torch": [], "count": 0, "check": {}})
# Kaggle's log is JSON with escaped newlines; a plain log has real ones
line = re.compile(r"SCORE (r2|torch) (\S+) (\S+) (\S+) (\S+) (\d+) (\S+) (\S+)")
for path in sys.argv[1:]:
    text = open(path, encoding="utf8", errors="replace").read()
    for side, layer, size, op, phase, count, ms, check in line.findall(text):
        r = rows[(layer, size, op, phase)]
        r["count"] = int(count)
        if ms == "failed":
            r[side].append(math.nan)
            continue
        r[side].append(float(ms))
        try:
            c = float(check.rstrip("\\n"))
            if not math.isnan(c):
                r["check"][side] = c
        except ValueError:
            pass


def median(v):
    v = sorted(v)
    return v[len(v) // 2]


def verdict(r):
    c = r["check"]
    if len(c) == 2 and abs(c["r2"] - c["torch"]) > 1e-3 * max(abs(c["r2"]), abs(c["torch"]), 1e-9):
        return "WORK DIFFERS"
    pairs = list(zip(r["r2"], r["torch"]))
    if any(math.isnan(a) or math.isnan(b) for a, b in pairs):
        return "FAILED"
    if all(b / a >= 1.10 for a, b in pairs):
        return "R2"
    if all(a / b >= 1.10 for a, b in pairs):
        return "PyTorch"
    return "level"


for key in sorted({k[:2] for k in rows}):
    layer, size = key
    table = []
    for (l, s, op, phase), r in rows.items():
        if (l, s) != key or not r["r2"] or not r["torch"]:
            continue
        v = verdict(r)
        a, b = median(r["r2"]), median(r["torch"])
        table.append(((a - b) * r["count"], op, phase, r["count"], a, b, v, len(r["r2"])))
    table.sort(key=lambda x: -x[0] if not math.isnan(x[0]) else 0)
    tally = collections.Counter(x[6] for x in table if x[3] > 0)
    print(f"\n## {layer} {size}: R2 {tally['R2']}, PyTorch {tally['PyTorch']}, level {tally['level']}"
          + "".join(f", {k} {tally[k]}" for k in ("WORK DIFFERS", "FAILED") if tally[k]))
    print("| op | phase | per step | R2 ms | PyTorch ms | faster | R2 minus PyTorch, ms/step | who |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for gap, op, phase, count, a, b, v, n in table:
        ratio = f"R2 {b / a:.2f}x" if b >= a else f"PyTorch {a / b:.2f}x"
        print(f"| {op} | {phase} | {count} | {a:.3f} | {b:.3f} | {ratio} | {gap:+.1f} | {v} |")
    for phase in ("fwd", "bwd", "step"):
        pick = [x for x in table if x[2] == phase and x[3] > 0]
        print(f"sum {phase}: R2 {sum(x[3] * x[4] for x in pick):.1f} ms/step, PyTorch {sum(x[3] * x[5] for x in pick):.1f} ms/step")
    alone = [f"{k[2]}/{k[3]}" for k, r in rows.items() if k[:2] == key and bool(r["r2"]) != bool(r["torch"])]
    if alone:
        print("one side only:", ", ".join(alone))
