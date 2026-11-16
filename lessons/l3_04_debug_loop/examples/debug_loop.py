"""Lesson 3.4 - Debug a looping agent from its trace.

1. Which baseline runs can we still replay exactly from the cache? (the cache only keeps what was recorded)
2. Replay one real no-finish run (stuck_consumer-0, sample 2) and read its trace.
3. A detector for three loop patterns: (a) identical repeated write, or identical read with no write in between, (b) health green but no finish,
   (c) steps after the fix.
4. Count the patterns over all 180 baseline runs, as far as the jsonl allows (writes, final, healthy).
    uv run python lessons/l3_04_debug_loop/examples/debug_loop.py > lessons/l3_04_debug_loop/examples/results.txt
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NW = ROOT / "nightwatch"
sys.path.insert(0, str(NW))
import agent_v0  # noqa: E402
from brightcart.incidents import generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

incs = {x.id: x for x in generate()}
rows = [json.loads(line) for line in open(NW / "runs" / "baseline_gpt-oss-20b.jsonl", encoding="utf-8")]

print("== 1. Replay coverage: 180 baseline runs, cache only, no API calls ==")
exact, traces = 0, {}
for r in rows:
    try:
        a = agent_v0.run(incs[r["incident"]], mode="replay", sample=r["sample"])
    except LookupError:
        continue
    if a["steps"] == r["steps"] and a["writes"] == [tuple(w) for w in r["writes"]]:
        exact += 1
        traces[(r["incident"], r["sample"])] = a
print(f"replay reproduces {exact} of {len(rows)} runs exactly; the rest hit a model call that is not in the cache")
print("(the cache does not hold every reply of those runs; a replay stops at the first missing one)")
print("bad_deploy-4 s1 (rollback twice + restart) replayable:", ("bad_deploy-4", 1) in traces)


def fix_step(trace, label):
    """index of the trace entry where the labelled fix was applied (None if never)."""
    if not label["fix"]:
        return None
    for i, t in enumerate(trace):
        args = t.get("args") or {}
        target = args.get("service") or args.get("flag")
        arg = args.get("replicas") or args.get("value") or (f"{args['key']}={args['value']}" if "key" in args else None)
        if t.get("tool") in agent_v0.WRITE_TOOLS and World._matches(label["fix"], (t["tool"], target, arg)):
            return i
    return None


# ---------------------------------------------------------------- the detector
def detect(run):
    trace, flags = run["trace"], []
    seen, writes = Counter(), 0          # a read repeated with no write in between can only return the same answer
    for t in trace:
        if t.get("tool"):
            is_write = t["tool"] in agent_v0.WRITE_TOOLS
            seen[(t["tool"], json.dumps(t.get("args"), sort_keys=True), None if is_write else writes)] += 1
            writes += is_write
    for (tool, args, _), n in seen.items():
        if n > 1:
            kind = "WRITE" if tool in agent_v0.WRITE_TOOLS else "read"
            flags.append(f"(a) identical {kind} x{n}: {tool}{args}")
    green = [i for i, t in enumerate(trace) if t.get("tool") == "check_health" and "'healthy': True" in t.get("result", "")]
    if green and run["final"] is None:
        flags.append(f"(b) health green at entry {green[0]} but no finish()")
    fx = fix_step(trace, run["label"])
    if fx is not None and len(trace) - 1 - fx > 2:     # fix, check_health, finish is the ideal tail
        flags.append(f"(c) {len(trace) - 1 - fx} entries after the fix at entry {fx}")
    return flags
# ----------------------------------------------------------------


print("\n== 2. The trace of a real no-finish run: stuck_consumer-0, sample 2 (exact replay) ==")
run = traces[("stuck_consumer-0", 2)]
print("alert:", incs["stuck_consumer-0"].alert, "| hidden fix:", run["label"]["fix"])
for i, t in enumerate(run["trace"]):
    what = t.get("tool") or t.get("error") or "text"
    print(f"{i:2d} step {t['step']:2d} {what:16} {json.dumps(t.get('args'))[:52]:52} -> {str(t.get('result', ''))[:55]}")
print(f"stop: {run['stop_reason']} | final {run['final']} | healthy {run['healthy']} | writes {run['writes']}")

print("\n== 3. Detector on that trace ==")
for f in detect(run) or ["(nothing flagged)"]:
    print(" ", f)

print("\n== 4. Detector on every exact-replay trace ==")
cnt = Counter()
for (iid, s), a in sorted(traces.items()):
    fl = detect(a)
    for f in fl:
        cnt[f[:3] + (" WRITE" if "WRITE" in f else " read" if f.startswith("(a)") else "")] += 1
    if fl:
        print(f"{iid:18} s{s} {'PASS' if agent_v0.grade(a)['pass'] else 'fail'}  " + " | ".join(fl))
print(f"runs with full traces: {len(traces)}; flags: {dict(cnt)}")

print("\n== 5. Counts over all 180 baseline runs from the jsonl (no traces needed) ==")
rep_write = [r for r in rows if len({json.dumps(w) for w in r["writes"]}) < len(r["writes"])]
nofin = [r for r in rows if not r["final"]]
green_nofin = [r for r in nofin if r["healthy"]]
after_fix = []
for r in rows:
    lab = incs[r["incident"]].label
    if lab["fix"]:
        idx = [i for i, w in enumerate(r["writes"]) if World._matches(lab["fix"], tuple(w))]
        if idx and idx[0] < len(r["writes"]) - 1:
            after_fix.append(r)
print(f"(a) runs with an identical repeated write: {len(rep_write)}  e.g. "
      + ", ".join(f"{r['incident']} s{r['sample']} {r['writes']}" for r in rep_write[:3]))
print(f"no-finish runs: {len(nofin)}, all at the 14-step limit: {all(r['steps'] == 14 for r in nofin)}")
print(f"(b) no finish but the system ends healthy: {len(green_nofin)} of {len(nofin)}  "
      f"(the jsonl stores only the END health, not when it first turned green)")
print(f"(c) runs that wrote again after the right fix: {len(after_fix)}  e.g. "
      + ", ".join(f"{r['incident']} s{r['sample']} {r['writes']}" for r in after_fix[:3]))
print(f"pass rate of those (c) runs: {sum(r['pass'] for r in after_fix)}/{len(after_fix)}; "
      f"of (b) runs: {sum(r['pass'] for r in green_nofin)}/{len(green_nofin)}")
print("by template, (b) runs:", dict(Counter(r["template"] for r in green_nofin)))

print("\n== 6. Before/after: the same 30 runs as lesson 3.2, v0 vs v0 + stop conditions (guards) ==")


def counts(rs):
    out = Counter()
    for r in rs:
        lab = incs[r["incident"]].label
        ws = [tuple(w) for w in r["writes"]]
        out["(a) repeated write"] += len(set(ws)) < len(ws)
        out["(b) healthy, no finish"] += (not r["final"]) and r["healthy"]
        idx = [i for i, w in enumerate(ws) if lab["fix"] and World._matches(lab["fix"], w)]
        out["(c) write after fix"] += bool(idx) and idx[0] < len(ws) - 1
        out["no finish"] += not r["final"]
    return dict(out)
worst = ["pool_config-3", "stuck_consumer-1", "stuck_consumer-5", "traffic_spike-1", "memory_leak-0",
         "memory_leak-1", "memory_leak-2", "memory_leak-4", "memory_leak-5", "stuck_consumer-0"]
before = [r for r in rows if r["incident"] in worst]
after = [agent_v0.run(incs[r["incident"]], mode="replay", sample=r["sample"], guards=True, token_budget=60000) for r in before]
print("before (jsonl):", counts(before))
print("after  (replayed lesson 3.2 runs, guards=True):", counts(after))
