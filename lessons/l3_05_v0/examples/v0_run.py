"""Lesson 3.5 - Nightwatch v0 finds the failing service.

One full run of nightwatch/agent_v0.py (17 tools, guards=True from lesson 3.2) on a hard incident:
expired_secret-1 (the pager names checkout, the cause is the payments provider key).
Record mode: the first run calls Groq and caches every reply; reruns replay for free.
    uv run python lessons/l3_05_v0/examples/v0_run.py > lessons/l3_05_v0/examples/results.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
import agent_v0  # noqa: E402
from brightcart.incidents import generate  # noqa: E402

inc = next(x for x in generate() if x.id == "expired_secret-1")

print(f"== 1. The tools Nightwatch v0 can call ({len(agent_v0.TOOLS)}) ==")
for t in agent_v0.TOOLS:
    f = t["function"]
    kind = "WRITE" if f["name"] in agent_v0.WRITE_TOOLS else ("end" if f["name"] == "finish" else "read")
    print(f"  {kind:5} {f['name']}({', '.join(f['parameters']['properties'])})")

print("\n== 2. The incident ==")
print(f"{inc.id} | difficulty {inc.difficulty} | split {inc.split} | PAGER: {inc.alert}")
print("recent deploys:", inc.deploys)

print("\n== 3. The trace (guards=True, max 14 steps, gpt-oss-20b, temperature 1.0) ==")
r = agent_v0.run(inc, mode="record", sample=0, guards=True, verbose=True)

print("\n== 4. Result ==")
print("finish():", json.dumps(r["final"], indent=1))
print(f"stop {r['stop_reason']} | steps {r['steps']} | healthy {r['healthy']} | writes {r['writes']}")
print(f"tokens prompt {r['usage']['prompt']} + completion {r['usage']['completion']}")
print("hidden label:", r["label"])
print("grade():", agent_v0.grade(r))
