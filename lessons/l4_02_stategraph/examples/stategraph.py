"""Lesson 4.2 - StateGraph and reducers vs your dict.

The SAME node functions as lesson 4.1 (triage_nodes.py), now wired with LangGraph 1.2.
The state is a TypedDict; 'notes' has a reducer (operator.add) = the hand-written merge() of 4.1.
    uv run python lessons/l4_02_stategraph/examples/stategraph.py > lessons/l4_02_stategraph/examples/results.txt
"""
import json
import operator
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "l4_01_state_by_hand" / "examples"))
import by_hand  # noqa: E402  (lesson 4.1's runner, for the comparison)
from triage_nodes import NODES, INCIDENTS, after_health, after_logs, start_state  # noqa: E402


class Triage(TypedDict):
    incident: str
    alert: str
    suspect: str | None
    unhealthy: list
    evidence: list
    hops: int
    cause: str | None
    notes: Annotated[list, operator.add]        # reducer: new notes are APPENDED


class TriageNoReducer(TypedDict):               # same keys, no reducer on notes
    incident: str
    alert: str
    suspect: str | None
    unhealthy: list
    evidence: list
    hops: int
    cause: str | None
    notes: list                                 # default: the last write wins (overwrite)


def build(schema):
    g = StateGraph(schema)
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "read_alert")
    g.add_edge("read_alert", "check_health")
    g.add_conditional_edges("check_health", after_health, ["read_logs", "false_alarm"])
    g.add_conditional_edges("read_logs", after_logs, ["follow_upstream", "classify"])
    g.add_edge("follow_upstream", "read_logs")
    g.add_edge("classify", END)
    g.add_edge("false_alarm", END)
    return g.compile()


graph = build(Triage)
print(f"langgraph {version('langgraph')}")

print("\n== 1. upstream_deploy-0 through the StateGraph, node by node (stream_mode='updates') ==")
for i, chunk in enumerate(graph.stream(start_state("upstream_deploy-0"), stream_mode="updates"), 1):
    (node, update), = chunk.items()
    print(f"step {i} {node:16} update {json.dumps({k: v for k, v in update.items() if k != 'notes'})}")
final_graph = graph.invoke(start_state("upstream_deploy-0"))
print("final  ", json.dumps(final_graph))

print("\n== 2. Same final state as the hand runner of lesson 4.1? ==")
same = 0
for iid in INCIDENTS:
    a = by_hand.run(start_state(iid), show=False)
    b = graph.invoke(start_state(iid))
    same += a == b
print(f"upstream_deploy-0: {by_hand.run(start_state('upstream_deploy-0'), show=False) == final_graph}")
print(f"all 60 incidents: {same}/60 final states identical")

print("\n== 3. Without a reducer: notes are overwritten, not appended ==")
plain = build(TriageNoReducer).invoke(start_state("upstream_deploy-0"))
print("with operator.add :", final_graph["notes"])
print("no reducer        :", plain["notes"])

print("\n== 4. The reducer for chat messages: add_messages ==")


class Chat(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state):
    return {"messages": [AIMessage("I will read the checkout logs.", id="a1")]}


def edit(state):          # same id -> add_messages REPLACES that message instead of appending
    return {"messages": [AIMessage("I will read the payments logs.", id="a1")]}


cg = StateGraph(Chat)
cg.add_node("agent", agent)
cg.add_node("edit", edit)
cg.add_edge(START, "agent")
cg.add_edge("agent", "edit")
cg.add_edge("edit", END)
out = cg.compile().invoke({"messages": [HumanMessage("PAGER: checkout error rate above 5%", id="h1")]})
for m in out["messages"]:
    print(f"  {type(m).__name__:12} id={m.id}  {m.content}")
print("(2 messages, not 3: a message with a new id is appended, one with the same id replaces the old one)")
