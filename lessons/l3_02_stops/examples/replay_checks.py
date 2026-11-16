"""Lesson 3.2 - extra facts for the slides, from the cache only (mode="replay": no API call).
    py -3.12 lessons/l3_02_stops/examples/replay_checks.py > lessons/l3_02_stops/examples/replay_checks.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NW = ROOT / "nightwatch"
sys.path.insert(0, str(NW))
import agent_v0  # noqa: E402
from brightcart.incidents import generate  # noqa: E402

I = {x.id: x for x in generate()}
rows = [json.loads(line) for line in open(NW / "runs" / "baseline_gpt-oss-20b.jsonl", encoding="utf-8")]

print("== A. Baseline no-finish runs that ended healthy ==")
nf = [r for r in rows if not r["final"]]
print(f"{sum(r['healthy'] for r in nf)} of {len(nf)}")

print("\n== B. pool_config-3 sample 1 with guards (trace) ==")
r = agent_v0.run(I["pool_config-3"], mode="replay", sample=1, guards=True, token_budget=60000)
for t in r["trace"]:
    if "error" in t:
        print(f"{t['step'] + 1:2}  (malformed tool call)")
    else:
        print(f"{t['step'] + 1:2}  {t['tool']}({json.dumps(t['args'])[:50]}) -> {t['result'][:60]}")
print("stop:", r["stop_reason"], "| steps:", r["steps"], "| final:", r["final"])

print("\n== C. Incidents that stayed 'no finish' after the guards ==")
for iid in ["stuck_consumer-1", "traffic_spike-1"]:
    for s in range(3):
        a = agent_v0.run(I[iid], mode="replay", sample=s, guards=True, token_budget=60000)
        g = sum(1 for t in a["trace"] if t.get("tool") == "check_health" and "'healthy': True" in t.get("result", ""))
        print(f"{iid:17} s{s}  stop={a['stop_reason']:10}  green health checks {g}")

print("\n== D. Same run, token_budget=5000 ==")
r = agent_v0.run(I["pool_config-3"], mode="replay", sample=1, guards=True, token_budget=5000)
print("stop:", r["stop_reason"], "| steps:", r["steps"], "| tokens:", r["usage"]["prompt"] + r["usage"]["completion"])
