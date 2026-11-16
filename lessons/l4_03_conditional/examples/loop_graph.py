"""Lesson 4.3 - Conditional edges: your loop as a graph.

The 40-line ReAct loop of lesson 3.1 (same 4 tools, same incident upstream_deploy-0, temperature 0)
as a LangGraph: a model node, a tools node, and one conditional edge (tool calls? -> tools : END).
Model calls go through agent_v0.chat (record/replay cache), so reruns are free.
    uv run python lessons/l4_03_conditional/examples/loop_graph.py > lessons/l4_03_conditional/examples/results.txt
"""
import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
import agent_v0  # noqa: E402
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

INC = next(x for x in generate() if x.id == "upstream_deploy-0")
SYSTEM = "You are Nightwatch, Brightcart's on-call agent. Fix the problem, check health, then answer."


def tool(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}}}


TOOLS = [tool("read_logs", "Last log lines of a service.", service=1),
         tool("recent_deploys", "Deploys in the last hour."),
         tool("rollback_deploy", "Roll back the latest deploy of a service.", service=1),
         tool("check_health", "Is every service healthy?")]


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]     # OpenAI-style dicts, appended


def make_graph(world):
    run = {"read_logs": lambda a: world.logs(a["service"], 8), "recent_deploys": lambda a: world.inc.deploys,
           "rollback_deploy": lambda a: world.act("rollback_deploy", a["service"]), "check_health": lambda a: world.health()}

    def model(state):                                           # = one turn of the while loop
        body = {"model": "openai/gpt-oss-20b", "messages": state["messages"], "tools": TOOLS, "temperature": 0}
        out = agent_v0.chat(body, mode="record")
        if "error" in out:                                      # Groq rejected a malformed tool call
            return {"messages": [{"role": "user", "content": "Your last tool call was malformed. Try again."}]}
        m = out["choices"][0]["message"]
        msg = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        return {"messages": [msg]}

    def tools(state):                                           # = the for-loop over tool calls
        outs = []
        for c in state["messages"][-1]["tool_calls"]:
            args = json.loads(c["function"]["arguments"] or "{}")
            try:
                res = run[c["function"]["name"]](args)
            except Exception as e:
                res = f"ERROR: {e}"
            outs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res, default=str)})
        return {"messages": outs}

    def route(state):                                           # = "if not m.tool_calls: return"
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") else ("model" if last["role"] == "user" else END)

    g = StateGraph(AgentState)
    g.add_node("model", model)
    g.add_node("tools", tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, ["tools", "model", END])
    g.add_edge("tools", "model")
    return g.compile()


start = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"PAGER: {INC.alert}"}]}

world = World(fresh(INC))
graph = make_graph(world)
print("== 1. The graph (draw_mermaid) ==")
print(graph.get_graph().draw_mermaid())

print("== 2. Node by node (stream_mode='updates') ==")
for i, chunk in enumerate(graph.stream(start, stream_mode="updates", config={"recursion_limit": 25}), 1):
    (node, upd), = chunk.items()
    for m in upd["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            what = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m["tool_calls"])
            print(f"{i:2d} {node:6} asks {what}")
        elif m["role"] == "tool":
            print(f"{i:2d} {node:6} -> {m['content'][:95]}")
        else:
            print(f"{i:2d} {node:6} {m['role']}: {m['content'][:300]!r}")
print(f"\nhealthy {world.health()['healthy']} | writes {world.actions}")

print("\n== 3. recursion_limit = 4 on a fresh world ==")
w2 = World(fresh(INC))
try:
    make_graph(w2).invoke(start, config={"recursion_limit": 4})
except GraphRecursionError as e:
    print("GraphRecursionError:", str(e).splitlines()[0])
    print(f"world after the stop: healthy {w2.health()['healthy']} | writes {w2.actions}")
