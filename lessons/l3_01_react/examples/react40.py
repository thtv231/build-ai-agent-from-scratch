"""Lesson 3.1 - The ReAct loop in 40 lines (the loop itself is the function `react`).

    uv run python lessons/l3_01_react/examples/react40.py > lessons/l3_01_react/examples/results.txt
"""
import json, os, sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
world = World(fresh(next(x for x in generate() if x.id == "upstream_deploy-0")))


def tool(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}}}


TOOLS = [tool("read_logs", "Last log lines of a service.", service=1),
         tool("recent_deploys", "Deploys in the last hour."),
         tool("rollback_deploy", "Roll back the latest deploy of a service.", service=1),
         tool("check_health", "Is every service healthy?")]
RUN = {"read_logs": lambda a: world.logs(a["service"], 8), "recent_deploys": lambda a: world.inc.deploys,
       "rollback_deploy": lambda a: world.act("rollback_deploy", a["service"]), "check_health": lambda a: world.health()}


# ---------------------------------------------------------------- the loop: 20 lines
def react(goal, max_steps=8):
    msgs = [{"role": "system", "content": "You are Nightwatch, Brightcart's on-call agent. Fix the problem, "
                                          "check health, then answer."},
            {"role": "user", "content": goal}]
    for step in range(1, max_steps + 1):
        m = client.chat.completions.create(model="openai/gpt-oss-20b", messages=msgs, tools=TOOLS,
                                           temperature=0).choices[0].message
        if not m.tool_calls:                                   # no action asked -> this is the answer
            print(f"step {step}: ANSWER  {m.content}")
            return m.content
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [c.model_dump() for c in m.tool_calls]})
        for c in m.tool_calls:                                 # act, observe, feed the result back
            args = json.loads(c.function.arguments or "{}")
            try:
                out = RUN[c.function.name](args)
            except Exception as e:
                out = f"ERROR: {e}"
            print(f"step {step}: {c.function.name}({args}) -> {str(out)[:90]}")
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out, default=str)})
    return "stopped: step limit"
# ----------------------------------------------------------------


print("== One incident, one loop ==")
answer = react("PAGER: checkout error rate above 5%")
print("\nhealthy at the end:", world.health()["healthy"], "| writes made:", world.actions)
