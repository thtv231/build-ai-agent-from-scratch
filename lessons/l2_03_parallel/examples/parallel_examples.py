"""Lesson 2.3 - Run calls, return results, parallel calls.

    uv run python lessons/l2_03_parallel/examples/parallel_examples.py > lessons/l2_03_parallel/examples/results.txt
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
world = World(fresh(next(x for x in generate() if x.id == "upstream_deploy-0")))


def fn(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": req or []}}}


TOOLS = [fn("service_status", "Error %, p95 latency, CPU and memory of one service.",
            {"service": {"type": "string"}}, ["service"]),
         fn("read_logs", "Last log lines of one service.", {"service": {"type": "string"}}, ["service"]),
         fn("recent_deploys", "Deploys in the last hour.")]
SLOW = 0.8                                  # pretend each tool talks to a remote system


def run_tool(name, args):
    time.sleep(SLOW)
    if name == "service_status":
        return world.status(args["service"])
    if name == "read_logs":
        return world.logs(args["service"], 6)
    if name == "recent_deploys":
        return world.inc.deploys
    raise KeyError(name)


SYSTEM = "You are Nightwatch. Gather everything you need in as few turns as possible."
ASK = "Checkout error rate is 23%. Check checkout and payments status, payments logs and recent deploys."
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]

QWEN = "qwen/qwen3.8-27b"

ERRORS_400 = []


def create(**kw):
    """One model call; a 400 from a malformed tool call is logged and retried (lesson 2.4 explains why)."""
    for _ in range(4):
        try:
            return client.chat.completions.create(**kw)
        except BadRequestError as e:
            ERRORS_400.append(f"{kw['model']}: {str(e).split(' - ', 1)[-1][:110]}")
    raise RuntimeError("4 malformed replies in a row")



def loop(model, max_turns=8):
    """Ask, run every requested tool, send results back, until the model stops asking."""
    m_ = list(msgs)
    turns, n_calls, t0, bad = 0, 0, time.time(), 0
    while turns < max_turns:
        try:
            r = client.chat.completions.create(model=model, temperature=0, messages=m_, tools=TOOLS)
        except BadRequestError as e:          # malformed tool call from the model (lesson 2.4): count it, try again
            bad += 1
            print(f"   [{model}] 400 on turn {turns + 1}: {str(e).split(' - ', 1)[-1][:70]} -> nudge and retry")
            if bad > 3:
                return turns, n_calls, time.time() - t0, "(gave up after 400s)"
            m_.append({"role": "user", "content": "Your last tool call had invalid JSON arguments. "
                                                  "Call one tool again with valid JSON."})
            continue
        msg = r.choices[0].message
        turns += 1
        if not msg.tool_calls:
            return turns, n_calls, time.time() - t0, msg.content
        m_.append({"role": "assistant", "content": msg.content or "", "tool_calls": [c.model_dump() for c in msg.tool_calls]})
        with ThreadPoolExecutor(len(msg.tool_calls)) as ex:
            res = list(ex.map(lambda c: run_tool(c.function.name, json.loads(c.function.arguments)), msg.tool_calls))
        n_calls += len(msg.tool_calls)
        for c, x in zip(msg.tool_calls, res):
            m_.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(x, default=str)})
    return turns, n_calls, time.time() - t0, "(gave up)"


print("== 1. How many tool calls in one reply? (3 tries each) ==")
for model in (MODEL, QWEN):
    counts = []
    for _ in range(3):
        r = create(model=model, temperature=0, messages=msgs, tools=TOOLS)
        counts.append(len(r.choices[0].message.tool_calls or []))
    print(f"{model:22} -> {counts}")

r = create(model=QWEN, temperature=0, messages=msgs, tools=TOOLS)
m = r.choices[0].message
calls = m.tool_calls or []
print("\nqwen's reply asks for:")
for c in calls:
    print(f"  {c.id[-8:]}  {c.function.name}({c.function.arguments})")

print("\n== 2. Run them one by one vs at the same time ==")
t0 = time.time()
seq = [run_tool(c.function.name, json.loads(c.function.arguments)) for c in calls]
t_seq = time.time() - t0
t0 = time.time()
with ThreadPoolExecutor(len(calls) or 1) as ex:
    par = list(ex.map(lambda c: run_tool(c.function.name, json.loads(c.function.arguments)), calls))
t_par = time.time() - t0
print(f"each tool takes {SLOW} s | one by one: {t_seq:.1f} s | at once: {t_par:.1f} s | same results: {seq == par}")

print("\n== 3. Every call gets exactly one result, matched by id ==")
assistant = {"role": "assistant", "content": m.content or "", "tool_calls": [c.model_dump() for c in calls]}
results = [{"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, default=str)} for c, res in zip(calls, par)]
for t in results:
    print(f"  tool result for {t['tool_call_id'][-8:]}: {t['content'][:70]}")

print("\n== 4. Forget one result: what happens? ==")
try:
    r4 = client.chat.completions.create(model=QWEN, temperature=0, tools=TOOLS, messages=msgs + [assistant] + results[:-1])
    m4 = r4.choices[0].message
    print("accepted, no error. Model then:", "asks again for " + str([c.function.name for c in m4.tool_calls])
          if m4.tool_calls else "answers: " + (m4.content or "")[:150])
except BadRequestError as e:
    print("400:", str(e).split(" - ", 1)[-1][:200])

print("\n== 5. Whole loop until the answer: one call per turn vs several ==")
for model in (MODEL, QWEN):
    turns, n, secs, ans = loop(model)
    print(f"{model:22} turns {turns}  tool calls {n}  time {secs:.1f} s")
    print(f"   answer: {(ans or '')[:220]}")

print("\n== 6. Malformed tool calls the server rejected during this run (retried) ==")
print(len(ERRORS_400), "x")
for e in ERRORS_400:
    print("  ", e)
