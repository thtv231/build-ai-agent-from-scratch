"""Lesson 4.5 - Prove it: same trajectory, same score.

1. Same model replies -> same trajectory? Feed the recorded replies of every baseline run we can replay
   exactly (lesson 3.4) into the create_agent version through a scripted model. No API calls.
2. Live, k=1, temperature 1.0: the 10 incidents of lesson 3.6, Nightwatch v0 vs create_agent.
   v0 = agent_v0.run(record, sample 0): cached baseline replies replay, missing ones are called live.
   create_agent = nw_agent.run(sample 0): its own record/replay cache, first run calls Groq.
    uv run python lessons/l4_05_prove/examples/prove.py > lessons/l4_05_prove/examples/results.txt
"""
import json
import sys
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "l4_04_create_agent" / "examples"))
import nw_agent  # noqa: E402
from nw_agent import agent_v0  # noqa: E402
from brightcart.incidents import TEMPLATES, generate  # noqa: E402

incs = {x.id: x for x in generate()}
rows = [json.loads(line) for line in open(nw_agent.ROOT / "nightwatch" / "runs" / "baseline_gpt-oss-20b.jsonl", encoding="utf-8")]


class Scripted(BaseChatModel):
    """A chat model that plays back a fixed list of replies (or a Malformed error), in order."""
    script: list

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if not self.script:
            raise IndexError("script ran out: create_agent asked for more model calls than v0 made")
        item = self.script.pop(0)
        if item == "MALFORMED":
            raise nw_agent.Malformed("scripted")
        return ChatResult(generations=[ChatGeneration(message=item)])

    def bind_tools(self, tools, **kw):
        return self

    @property
    def _llm_type(self):
        return "scripted"


def v0_with_replies(inc, sample, mode="replay"):
    """Run agent_v0 and keep every raw model reply it got."""
    replies, real = [], agent_v0.chat

    def spy(body, mode, sample):
        out = real(body, mode, sample)
        replies.append(out)
        return out
    agent_v0.chat = spy
    try:
        r = agent_v0.run(inc, mode=mode, sample=sample)
    finally:
        agent_v0.chat = real
    return r, replies


def to_message(out):
    if "error" in out:
        return "MALFORMED"
    m = out["choices"][0]["message"]
    calls = []
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": c["function"]["name"], "args": args, "id": c["id"], "type": "tool_call"})
    return AIMessage(content=m.get("content") or "", tool_calls=calls)


def seq(r):
    return [(t.get("tool"), json.dumps(t.get("args"), sort_keys=True)) for t in r["trace"] if t.get("tool")]


print("== 1. Same replies in, same trajectory out? (scripted model, no API calls) ==")
same_traj = same_grade = n = 0
for row in rows:
    inc = incs[row["incident"]]
    try:
        v0, replies = v0_with_replies(inc, row["sample"])
    except LookupError:
        continue
    if v0["steps"] != row["steps"] or v0["writes"] != [tuple(w) for w in row["writes"]]:
        continue                                              # only runs that replay exactly
    n += 1
    try:
        ca = nw_agent.run(inc, model=Scripted(script=[to_message(o) for o in replies]))
        err = ""
    except IndexError as e:
        ca, err = None, str(e)
    t_ok = ca is not None and seq(ca) == seq(v0) and ca["writes"] == v0["writes"] and ca["final"] == v0["final"]
    g_ok = ca is not None and agent_v0.grade(ca) == agent_v0.grade(v0)
    same_traj += t_ok
    same_grade += g_ok
    if not t_ok:
        print(f"  differs: {inc.id} s{row['sample']} v0 steps {v0['steps']} stop {v0['stop_reason']} "
              f"malformed replies {sum('error' in t for t in v0['trace'])} | "
              f"create_agent: {err or (ca['steps'], ca['stop_reason'], ca['writes'])}")
print("(v0 counts a malformed reply as one of its 14 steps; retry_malformed retries inside one model call,\n"
      " so create_agent gets more model calls before ModelCallLimitMiddleware stops it)")
print(f"exact-replay runs: {n} | identical trajectory (tool+args, writes, finish) {same_traj}/{n} | same grade {same_grade}/{n}")

print("\n== 2. Live, k=1, temperature 1.0: 10 incidents (lesson 3.6), v0 vs create_agent ==")
TEN = [next(x for x in incs.values() if x.template == t and x.split == "train") for t in TEMPLATES]
jsonl = {(r["incident"], r["sample"]): r for r in rows}
res = []
for inc in TEN:
    v0, _ = v0_with_replies(inc, 0, mode="record")
    ca = nw_agent.run(inc, sample=0)
    res.append((inc, v0, ca))
    gv, gc = agent_v0.grade(v0)["pass"], agent_v0.grade(ca)["pass"]
    print(f"{inc.id:18} jsonl {'PASS' if jsonl[(inc.id, 0)]['pass'] else 'fail'} | v0 {'PASS' if gv else 'fail'} "
          f"steps {v0['steps']:2} | create_agent {'PASS' if gc else 'fail'} steps {ca['steps']:2} "
          f"retries {len(ca['errors'])} | same sequence {seq(v0) == seq(ca)}")
    print(f"   v0: {[t for t, _ in seq(v0)]}")
    print(f"   ca: {[t for t, _ in seq(ca)]}")

pv = sum(agent_v0.grade(v)["pass"] for _, v, _ in res)
pc = sum(agent_v0.grade(c)["pass"] for _, _, c in res)
pj = sum(jsonl[(i.id, 0)]["pass"] for i, _, _ in res)
tv = sum(v["usage"]["prompt"] + v["usage"]["completion"] for _, v, _ in res)
tc = sum(c["usage"]["prompt"] + c["usage"]["completion"] for _, _, c in res)
print(f"\npass: baseline jsonl {pj}/10 | v0 now {pv}/10 | create_agent {pc}/10")
print(f"identical tool sequences {sum(seq(v) == seq(c) for _, v, c in res)}/10 | "
      f"same first tool {sum(seq(v)[:1] == seq(c)[:1] for _, v, c in res)}/10 | "
      f"same writes {sum(v['writes'] == c['writes'] for _, v, c in res)}/10")
print(f"tokens: v0 {tv:,} | create_agent {tc:,}  (v0 count includes replies replayed from cache)")
