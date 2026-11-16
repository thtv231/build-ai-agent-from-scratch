"""Lesson 3.2 - Stop conditions: answer, max steps, budget.

Before = the baseline v0 runs (nightwatch/runs/baseline_gpt-oss-20b.jsonl, same incidents, same samples).
After  = the same incidents and sample numbers with guards=True: an identical repeated write is blocked,
         and a green health check is followed by "call finish now"; plus a token budget.
Same sample number -> same cached model replies until the first point where a guard changes the conversation.
    uv run python lessons/l3_02_stops/examples/stops_experiment.py > lessons/l3_02_stops/examples/results.txt
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NW = ROOT / "nightwatch"
sys.path.insert(0, str(NW))
import agent_v0  # noqa: E402
from brightcart.incidents import generate  # noqa: E402

rows = [json.loads(line) for line in open(NW / "runs" / "baseline_gpt-oss-20b.jsonl", encoding="utf-8")]


def why(r):
    if r["pass"]:
        return "pass"
    if not r["final"]:
        return "no finish"
    if not r["healthy"]:
        return "unhealthy"
    return "wrong service" if not r["right_service"] else "extra writes"


print("== 1. Baseline v0: why runs failed (180 runs) ==")
print(dict(Counter(why(r) for r in rows)))

fails = defaultdict(int)
for r in rows:
    if why(r) == "no finish":
        fails[r["incident"]] += 1
worst = [i for i, _ in sorted(fails.items(), key=lambda x: (-x[1], x[0]))[:10]]
print("10 incidents with the most no-finish runs:", worst)

print("\n== 2. Steps used by no-finish runs ==")
print("steps:", sorted(r["steps"] for r in rows if why(r) == "no finish"))

incs = {x.id: x for x in generate()}
before, after = [], []
print("\n== 3. Same incidents, same samples: v0 vs v0 + stop conditions ==")
for iid in worst:
    for s in range(3):
        b = next(r for r in rows if r["incident"] == iid and r["sample"] == s)
        a = agent_v0.run(incs[iid], sample=s, guards=True, token_budget=60000)
        a.update(agent_v0.grade(a))
        before.append(b); after.append(a)
        blocked = sum(1 for t in a["trace"] if str(t.get("result", "")).startswith("ALREADY DONE"))
        print(f"{iid:20} s{s}  before {why(b):13} steps {b['steps']:2}  |  after {why(a):13} steps {a['steps']:2}"
              f"  stop={a['stop_reason']:11} blocked repeats {blocked}")


def rate(xs):
    return sum(x["pass"] for x in xs) / len(xs)


tok_b = sum(r["usage"]["prompt"] + r["usage"]["completion"] for r in before) / len(before)
tok_a = sum(r["usage"]["prompt"] + r["usage"]["completion"] for r in after) / len(after)
print(f"\npass: before {rate(before):.0%} -> after {rate(after):.0%}  (30 runs each)")
print(f"failure kinds after: {dict(Counter(why(r) for r in after))}")
print(f"mean tokens per run: before {tok_b:,.0f} -> after {tok_a:,.0f}")
print(f"mean steps: before {sum(r['steps'] for r in before) / 30:.1f} -> after {sum(r['steps'] for r in after) / 30:.1f}")
