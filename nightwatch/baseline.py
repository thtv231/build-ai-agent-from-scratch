"""Spike: baseline success and its variance for Nightwatch v0.

    py -3.12 baseline.py --split all --k 3              # 60 incidents x 3 samples
    py -3.12 baseline.py --split train --k 3 --limit 12 # smaller, for a free key
    py -3.12 baseline.py --replay                       # re-score from cache, no calls

Writes runs/baseline_<model>.jsonl and prints pass rate, pass@k, pass^k, a 95%
bootstrap interval, per-template rates and token use. Go/no-go for Ch13:
baseline pass rate between 35% and 60% (room to improve, not hopeless).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import agent_v0
from brightcart.incidents import generate

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="all")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="openai/gpt-oss-20b")
ap.add_argument("--replay", action="store_true")
a = ap.parse_args()

incs = [x for x in generate() if a.split == "all" or x.split == a.split]
if a.limit:   # keep templates balanced when limiting
    by = defaultdict(list)
    for x in incs:
        by[x.template].append(x)
    incs, i = [], 0
    while len(incs) < a.limit and any(by.values()):
        for t in list(by):
            if by[t] and len(incs) < a.limit:
                incs.append(by[t].pop(0))

out = Path(__file__).parent / "runs" / f"baseline_{a.model.split('/')[-1]}.jsonl"
rows = []
with out.open("w", encoding="utf-8") as f:
    for inc in incs:
        for s in range(a.k):
            try:
                r = agent_v0.run(inc, model=a.model, mode="replay" if a.replay else "record", sample=s)
            except (agent_v0.DailyCap, LookupError) as e:
                print(f"STOP at {inc.id} s{s}: {e}"); break
            g = agent_v0.grade(r)
            row = {k: r[k] for k in ("incident", "split", "template", "difficulty", "sample", "final",
                                     "writes", "healthy", "steps", "usage")} | g
            rows.append(row)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{inc.id:22s} s{s} {'PASS' if g['pass'] else 'fail'} steps={r['steps']:2d} "
                  f"tok={r['usage']['prompt'] + r['usage']['completion']}")
        else:
            continue
        break

per = defaultdict(list)
for r in rows:
    per[r["incident"]].append(r["pass"])
rate = st.mean(r["pass"] for r in rows)
pass_at_k = st.mean(any(v) for v in per.values())
pass_hat_k = st.mean(all(v) for v in per.values())
ids = list(per)
boot = sorted(st.mean(st.mean(per[random.choice(ids)]) for _ in ids) for _ in range(2000))
tpl = defaultdict(list)
for r in rows:
    tpl[r["template"]].append(r["pass"])
tok = [r["usage"]["prompt"] + r["usage"]["completion"] for r in rows]
# spread between samples of the same incident = how noisy one run is
flips = st.mean(0 < sum(v) < len(v) for v in per.values())
print(f"\nruns={len(rows)} incidents={len(per)} model={a.model}")
print(f"pass rate {rate:.0%}  95% CI [{boot[50]:.0%}, {boot[1949]:.0%}]  pass@{a.k} {pass_at_k:.0%}  "
      f"pass^{a.k} {pass_hat_k:.0%}  incidents that flip between samples {flips:.0%}")
print("per template:", {t: f"{st.mean(v):.0%}" for t, v in sorted(tpl.items())})
print(f"tokens/run mean {st.mean(tok):.0f}  total {sum(tok)}")
print("GO for Ch13" if 0.35 <= rate <= 0.60 else "outside 35-60%: adjust difficulty before Ch13")
