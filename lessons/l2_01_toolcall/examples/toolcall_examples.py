"""Lesson 2.1 - What a tool call is: the model asks, your code acts.

    uv run python lessons/l2_01_toolcall/examples/toolcall_examples.py > lessons/l2_01_toolcall/examples/results.txt
Uses the Brightcart simulator from nightwatch/ (incident upstream_deploy-0: checkout errors, cause = payments deploy).
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
inc = next(x for x in generate() if x.id == "upstream_deploy-0")
world = World(fresh(inc))
calls_run = 0


def read_logs(service, n=5):
    global calls_run
    calls_run += 1
    return world.logs(service, n)


TOOLS = [{"type": "function", "function": {
    "name": "read_logs",
    "description": "Return the last n log lines of one Brightcart service.",
    "parameters": {"type": "object",
                   "properties": {"service": {"type": "string", "description": "e.g. checkout, payments"},
                                  "n": {"type": "integer", "description": "how many lines"}},
                   "required": ["service"]}}}]
SYSTEM = "You are Nightwatch, the on-call assistant for the shop Brightcart."
ASK = "Checkout errors spiked. What do the checkout logs say? Answer in one sentence."

print("== 1. No tools: ask the model to read the logs ==")
r = client.chat.completions.create(model=MODEL, temperature=0,
                                   messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}])
print("reply:", r.choices[0].message.content)
print("times our code ran read_logs:", calls_run)

print("\n== 2. With a tool: the model asks ==")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]
r = client.chat.completions.create(model=MODEL, temperature=0, messages=msgs, tools=TOOLS)
m = r.choices[0].message
print("finish_reason:", r.choices[0].finish_reason)
print("content:", repr(m.content))
for c in m.tool_calls:
    print("tool_call:", json.dumps({"id": c.id, "name": c.function.name, "arguments": c.function.arguments}))
print("type of arguments:", type(m.tool_calls[0].function.arguments).__name__)
print("times our code ran read_logs so far:", calls_run)

print("\n== 3. Your code acts, then sends the result back ==")
msgs.append({"role": "assistant", "content": m.content or "",
             "tool_calls": [c.model_dump() for c in m.tool_calls]})
for c in m.tool_calls:
    args = json.loads(c.function.arguments)
    result = read_logs(**args)
    print(f"ran read_logs({args}) ->")
    for line in result:
        print("   ", line)
    msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
print("times our code ran read_logs:", calls_run)
print("messages now:", [x["role"] for x in msgs])
r = client.chat.completions.create(model=MODEL, temperature=0, messages=msgs, tools=TOOLS)
print("finish_reason:", r.choices[0].finish_reason)
print("final answer:", r.choices[0].message.content)

print("\n== 4. Usage of the two calls ==")
print("call 2 (ask) and call 3 (answer) together re-sent", len(msgs), "messages in the last call")
print("last call usage:", r.usage.prompt_tokens, "prompt +", r.usage.completion_tokens, "completion")
