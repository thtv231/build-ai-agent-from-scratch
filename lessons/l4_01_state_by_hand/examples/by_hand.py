"""Lesson 4.1 - State, nodes and edges, by hand (no LLM, no framework).

state = a dict. node = function(state) -> partial update. edge = "after this node, go there".
conditional edge = a function(state) -> name of the next node. The runner is ~15 lines.
    uv run python lessons/l4_01_state_by_hand/examples/by_hand.py > lessons/l4_01_state_by_hand/examples/results.txt
"""
import json

from triage_nodes import NODES, after_health, after_logs, start_state

END = "END"
EDGES = {"read_alert": "check_health",          # plain edge: always the same next node
         "check_health": after_health,          # conditional edge: a function picks the next node
         "read_logs": after_logs,
         "follow_upstream": "read_logs",        # a cycle: back to read the callee's logs
         "classify": END,
         "false_alarm": END}


# ---------------------------------------------------------------- the runner
def merge(state, update):
    """How an update joins the state: 'notes' is appended, every other key is overwritten."""
    new = dict(state)
    for k, v in update.items():
        new[k] = state[k] + v if k == "notes" else v
    return new


def run(state, entry="read_alert", max_steps=10, show=True):
    node = entry
    for step in range(1, max_steps + 1):
        update = NODES[node](state)
        state = merge(state, update)
        if show:
            print(f"step {step} {node:16} update {json.dumps({k: v for k, v in update.items() if k != 'notes'})}")
        nxt = EDGES[node]
        node = nxt(state) if callable(nxt) else nxt
        if node == END:
            return state
    raise RuntimeError("step limit")
# ----------------------------------------------------------------


if __name__ == "__main__":
    for i, iid in enumerate(["upstream_deploy-0", "false_alarm-0", "disk_full-2"], 1):
        print(f"== {i}. {iid} ==")
        s = start_state(iid)
        print("start  ", json.dumps(s))
        final = run(s)
        print("final  ", json.dumps(final))
        print("notes  ", final["notes"], "\n")

    print("== 4. The same hand-written flow on all 60 incidents (rules only, no LLM) ==")
    from collections import Counter
    from triage_nodes import INCIDENTS
    ok_svc = ok_cause = 0
    miss = Counter()
    for iid, inc in INCIDENTS.items():
        f = run(start_state(iid), show=False)
        ok_svc += f["suspect"] == inc.label["cause_service"]
        ok_cause += f["cause"] == inc.label["cause"]
        if f["cause"] != inc.label["cause"]:
            miss[f"{inc.template} -> {f['cause']}"] += 1
    print(f"right service {ok_svc}/60 | right cause {ok_cause}/60 | misses: {dict(miss)}")
