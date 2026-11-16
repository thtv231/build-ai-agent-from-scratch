"""Lesson 4.4 - create_agent + middleware = the same loop, line by line.

nw_agent.py rebuilds Nightwatch v0 with langchain 1.5 create_agent: same model, same system prompt,
same 17 tools, same 14-call limit, and three small middlewares that copy what agent_v0.run() does by hand.
One incident: expired_secret-1 (lesson 3.5). LLM replies are cached on disk, reruns are free.
    uv run python lessons/l4_04_create_agent/examples/create_agent_demo.py > lessons/l4_04_create_agent/examples/results.txt
"""
import json
import sys
from importlib.metadata import version
from pathlib import Path

import nw_agent
from langchain.agents.middleware import AgentMiddleware
from nw_agent import agent_v0

sys.path.insert(0, str(Path(nw_agent.ROOT) / "nightwatch"))
from brightcart.incidents import generate  # noqa: E402

inc = next(x for x in generate() if x.id == "expired_secret-1")
print(f"langchain {version('langchain')} | langgraph {version('langgraph')} | langchain-groq {version('langchain-groq')}")

print("\n== 1. Your loop, line by line, in create_agent ==")
for line in nw_agent.__doc__.split("Line-by-line map to agent_v0.run():")[1].strip().splitlines():
    print("  " + line.strip())

print("\n== 2. The middleware stack ==")
box = {"final": None, "trace": [], "errors": []}
for m in nw_agent.make_middleware(box):
    hooks = [h for h in ("before_model", "after_model", "wrap_model_call", "wrap_tool_call")
             if getattr(type(m), h, None) is not getattr(AgentMiddleware, h, None)]
    print(f"  {m.name:28} hooks: {hooks}")

print("\n== 3. The graph create_agent built (draw_mermaid) ==")
world_tools = nw_agent.make_tools(nw_agent.World(nw_agent.fresh(inc)), box)
agent = nw_agent.create_agent(nw_agent.make_model(), world_tools, system_prompt=agent_v0.SYSTEM,
                              middleware=nw_agent.make_middleware(box))
print(agent.get_graph().draw_mermaid())

print(f"== 4. One run: {inc.id} | PAGER: {inc.alert} ==")
r = nw_agent.run(inc, sample=0, verbose=True)
print("finish():", json.dumps(r["final"]))
print(f"stop {r['stop_reason']} | model calls {r['model_calls']} | tool calls {r['steps']} | healthy {r['healthy']} | "
      f"writes {r['writes']} | malformed-call retries {len(r['errors'])}")
print(f"tokens prompt {r['usage']['prompt']} + completion {r['usage']['completion']}")
print("grade():", agent_v0.grade(r))

print("\n== 5. Same incident, Nightwatch v0 (baseline sample 0, replayed from cache) ==")
v0 = agent_v0.run(inc, mode="replay", sample=0)
print("v0            :", [t.get("tool") for t in v0["trace"]], "grade", agent_v0.grade(v0)["pass"])
print("create_agent  :", [t.get("tool") for t in r["trace"]], "grade", agent_v0.grade(r)["pass"])
print("same tool sequence:", [t.get("tool") for t in v0["trace"]] == [t.get("tool") for t in r["trace"]])
