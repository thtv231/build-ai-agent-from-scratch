"""Lesson 2.2 - A tool schema by hand, then @tool, and how to design tools.

    uv run python lessons/l2_02_schema/examples/schema_examples.py > lessons/l2_02_schema/examples/results.txt
"""
import inspect
import json
import os
import sys
import typing
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
enc = tiktoken.get_encoding("o200k_harmony")
world = World(fresh(next(x for x in generate() if x.id == "upstream_deploy-0")))

print("== 1. A schema by hand ==")
by_hand = {"type": "function", "function": {
    "name": "get_metrics",
    "description": "Current CPU %, memory %, error % and p95 latency (ms) of one Brightcart service.",
    "parameters": {"type": "object",
                   "properties": {"service": {"type": "string", "enum": list(world.inc.services)}},
                   "required": ["service"]}}}
print(json.dumps(by_hand, indent=1))


print("\n== 2. Our own @tool: build the schema from the function ==")
PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(fn):
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, p in sig.parameters.items():
        props[name] = {"type": PY2JSON[hints[name]]}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    fn.schema = {"type": "function", "function": {
        "name": fn.__name__, "description": inspect.getdoc(fn),
        "parameters": {"type": "object", "properties": props, "required": required}}}
    return fn


@tool
def read_logs(service: str, n: int = 20) -> list:
    """Return the last n log lines of one Brightcart service."""
    return world.logs(service, n)


print(json.dumps(read_logs.schema, indent=1))


print("\n== 3. The library version: LangChain's @tool ==")
from langchain_core.tools import tool as lc_tool  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402


@lc_tool
def read_logs_lc(service: str, n: int = 20) -> list:
    """Return the last n log lines of one Brightcart service."""
    return world.logs(service, n)


lc_schema = convert_to_openai_tool(read_logs_lc)
print(json.dumps(lc_schema, indent=1))
ours, theirs = read_logs.schema["function"]["parameters"], lc_schema["function"]["parameters"]
print("same property names:", list(ours["properties"]) == list(theirs["properties"]),
      "| same required:", ours["required"] == theirs.get("required"),
      "| library adds:", sorted(set(json.dumps(theirs)) - set()) and
      [k for k in theirs["properties"]["n"] if k not in ours["properties"]["n"]])


print("\n== 4. Names and descriptions decide which tool is picked (8 runs each) ==")
QUESTION = "Is the payments service using too much memory right now?"


def make(names_good):
    a = ("get_metrics", "Current CPU %, memory %, error % and p95 latency of one service.") if names_good \
        else ("tool_a", "Gets data.")
    b = ("read_logs", "Last log lines of one service.") if names_good else ("tool_b", "Gets data.")
    return [{"type": "function", "function": {"name": n, "description": d,
             "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}}
            for n, d in (b, a)]      # the right tool is SECOND, so position can't explain a pick


for good in (False, True):
    tools = make(good)
    right = tools[1]["function"]["name"]
    picks = []
    for _ in range(8):
        r = client.chat.completions.create(model=MODEL, temperature=1, tools=tools,
                                           messages=[{"role": "user", "content": QUESTION}])
        tc = r.choices[0].message.tool_calls
        picks.append(tc[0].function.name if tc else "(no call)")
    print(f"{'clear names' if good else 'vague names'}: right tool {picks.count(right)}/8 -> {picks}")


print("\n== 5. Truncate big results ==")
many = [f"04:{i % 60:02d} INFO checkout handled request id={10000 + i} in {50 + i % 150}ms" for i in range(2000)]
full = json.dumps(many)
short = json.dumps(many[-20:] + [f"... {len(many) - 20} earlier lines cut; ask for a time range to see more"])
print("2000 lines:", len(enc.encode(full)), "tokens | last 20 + a note:", len(enc.encode(short)), "tokens")
print(f"input cost per call at $0.075/M: ${len(enc.encode(full)) * 0.075 / 1e6:.6f} vs ${len(enc.encode(short)) * 0.075 / 1e6:.6f}")


print("\n== 6. Least privilege: what each tool can touch ==")
for name, kind in [("read_logs", "read"), ("get_metrics", "read"), ("rollback_deploy", "WRITE"),
                   ("restart_service", "WRITE"), ("run_sql(any query)", "WRITE + READ ALL")]:
    print(f"{name:20} {kind}")
