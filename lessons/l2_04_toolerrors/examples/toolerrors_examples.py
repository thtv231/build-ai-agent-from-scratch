"""Lesson 2.4 (E) - When tool calls go wrong: bad arguments, malformed calls, recovery.

    uv run python lessons/l2_04_toolerrors/examples/toolerrors_examples.py > lessons/l2_04_toolerrors/examples/results.txt
Evidence of a malformed tool call + blind retry: evidence_blind_retry_from_2_3.txt (recorded earlier the same day).
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
world = World(fresh(next(x for x in generate() if x.id == "upstream_deploy-0")))

print("== 1. Read the wire: a malformed tool call (recorded earlier today, lesson 2.3 run) ==")
ev = (HERE / "evidence_blind_retry_from_2_3.txt").read_text(encoding="utf-8")
for line in ev.splitlines():
    if "400 on turn" in line:
        print(" ", line.strip())
print("  -> same request, temperature 0, four retries, four identical failures")

TOOLS = [{"type": "function", "function": {
    "name": "get_metrics", "description": "Error %, p95 latency, CPU and memory of one service.",
    "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}}]
QUESTION = "How is the payment service doing? Use get_metrics, then answer in one sentence."


def get_metrics(service):
    return world.status(service)        # KeyError for an unknown name, listing the known ones


def run(policy, max_turns=5):
    """policy: crash | vague | helpful. Returns (outcome, args tried)."""
    msgs = [{"role": "user", "content": QUESTION}]
    tried = []
    for _ in range(max_turns):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=1, messages=msgs, tools=TOOLS)
        except BadRequestError:
            msgs.append({"role": "user", "content": "Your tool call was malformed JSON. Try again."})
            continue
        m = r.choices[0].message
        if not m.tool_calls:
            return ("answered" if any(t == "payments" for t in tried) else "answered without data"), tried
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [c.model_dump() for c in m.tool_calls]})
        for c in m.tool_calls:
            args = json.loads(c.function.arguments or "{}")
            tried.append(args.get("service"))
            try:
                out = json.dumps(get_metrics(**args))
            except Exception as e:
                if policy == "crash":
                    return f"crashed: {type(e).__name__}", tried
                out = "Error" if policy == "vague" else f"ERROR: {e}"
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return "gave up", tried


print("\n== 2. What the model guesses for 'the payment service' ==")
print("real names:", ", ".join(world.inc.services))

N = 8
for policy in ("crash", "vague", "helpful"):
    outcomes = [run(policy) for _ in range(N)]
    ok = sum(o == "answered" for o, _ in outcomes)
    print(f"\npolicy {policy:8}: correct data reached {ok}/{N}")
    for o, t in outcomes:
        print(f"   {o:24} tried {t}")

print("\n== 3. What the helpful error says ==")
try:
    get_metrics("payment-service")
except KeyError as e:
    print("ERROR:", e)
