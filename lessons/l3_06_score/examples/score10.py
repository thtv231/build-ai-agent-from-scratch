"""Lesson 3.6 - Score v0 on 10 incidents: exact and rule graders.

10 incidents = the first train incident of each of the 10 templates, k=1 (sample 0 of the baseline,
nightwatch/runs/baseline_gpt-oss-20b.jsonl). No new model calls: we only grade saved runs.
  exact grader  : finish() named the right service (needs the hidden label)
  rule grader   : finished + system healthy at the end + at most one write that was not the fix
                  (needs no label: "the fix" = the last write when the end state is healthy)
  combined      : exact AND rule
    uv run python lessons/l3_06_score/examples/score10.py > lessons/l3_06_score/examples/results.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NW = ROOT / "nightwatch"
sys.path.insert(0, str(NW))
import agent_v0  # noqa: E402
from brightcart.incidents import TEMPLATES, generate  # noqa: E402

rows = {(r["incident"], r["sample"]): r for r in map(json.loads, open(NW / "runs" / "baseline_gpt-oss-20b.jsonl", encoding="utf-8"))}
incs = generate()
TEN = [next(x for x in incs if x.template == t and x.split == "train") for t in TEMPLATES]


def exact_grader(run, label):
    return (run["final"] or {}).get("root_cause_service") == label["cause_service"]


def rule_grader(run):
    wrong_writes = len(run["writes"]) - (1 if run["healthy"] and run["writes"] else 0)
    return run["final"] is not None and run["healthy"] and wrong_writes <= 1


print("== 1. The 10 incidents (train split, one per template, sample 0) ==")
print(f"{'incident':18} {'diff':6} {'label service':13} {'finish service':14} {'fin':>3} {'healthy':>7} {'writes':>6}"
      f"  {'exact':>5} {'rule':>5} {'both':>5} {'grade()':>7}")
res = []
for inc in TEN:
    r = rows[(inc.id, 0)] | {"label": inc.label}
    e, ru = exact_grader(r, inc.label), rule_grader(r)
    g = agent_v0.grade(r)["pass"]
    res.append((inc, r, e, ru, g))
    print(f"{inc.id:18} {inc.difficulty:6} {inc.label['cause_service']:13} {str((r['final'] or {}).get('root_cause_service')):14}"
          f" {'y' if r['final'] else 'n':>3} {str(r['healthy']):>7} {len(r['writes']):>6}  {str(e):>5} {str(ru):>5} {str(e and ru):>5} {str(g):>7}")

n = len(res)
print("\n== 2. Rates (n = 10, k = 1) ==")
print(f"exact grader {sum(x[2] for x in res)}/{n} | rule grader {sum(x[3] for x in res)}/{n} | "
      f"combined {sum(x[2] and x[3] for x in res)}/{n} | agent_v0.grade() pass {sum(x[4] for x in res)}/{n}")

print("\n== 3. Where the graders disagree ==")
dis = [x for x in res if x[2] != x[3] or (x[2] and x[3]) != x[4]]
for inc, r, e, ru, g in dis:
    print(f"{inc.id}: exact={e} rule={ru} grade()={g}")
    print(f"  final  : {r['final']}")
    print(f"  writes : {r['writes']} | healthy {r['healthy']} | steps {r['steps']} | hidden fix {inc.label['fix']}")
if not dis:
    print("none on these 10 runs")

print("\n== 4. The same graders on all 180 baseline runs (to find disagreements the 10 do not show) ==")
allr = [r | {"label": next(x for x in incs if x.id == r["incident"]).label} for r in rows.values()]
E = [exact_grader(r, r["label"]) for r in allr]
R = [rule_grader(r) for r in allr]
print(f"exact {sum(E)}/180 | rule {sum(R)}/180 | both {sum(a and b for a, b in zip(E, R))}/180 | "
      f"exact only {sum(a and not b for a, b in zip(E, R))} | rule only {sum(b and not a for a, b in zip(E, R))}")
for r, a, b in zip(allr, E, R):
    if a != b:
        print(f"  rule={b} exact={a}: {r['incident']} s{r['sample']} final={r['final']} writes={r['writes']}")
print(f"agent_v0.grade() pass on all 180: {sum(agent_v0.grade(r)['pass'] for r in allr)}/180")
