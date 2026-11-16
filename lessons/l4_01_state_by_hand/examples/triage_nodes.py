"""A toy triage flow over Brightcart data, no LLM. Shared by lesson 4.1 (hand runner) and 4.2 (StateGraph).

Every node takes the whole state and returns ONLY the keys it changes (a partial update).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
from brightcart.incidents import fresh, generate  # noqa: E402
from brightcart.sim import World  # noqa: E402

INCIDENTS = {x.id: x for x in generate()}
SIGNS = [("401 Unauthorized", "expired_secret"), ("OOMKilled", "memory_leak"), ("No space left", "disk_full"),
         ("guarded by flag", "bad_flag"), ("db connection", "bad_config"), ("shedding load", "traffic_spike"),
         ("Error:", "bad_deploy")]


def world(state):
    return World(fresh(INCIDENTS[state["incident"]]))


def read_alert(state):
    return {"suspect": state["alert"].split()[0], "notes": ["alert names " + state["alert"].split()[0]]}


def check_health(state):
    h = world(state).health()
    return {"unhealthy": h["unhealthy_services"], "notes": [f"unhealthy: {h['unhealthy_services']}"]}


def read_logs(state):
    lines = [x for x in world(state).logs(state["suspect"], 20) if ("ERROR" in x or "WARN" in x) and "GC pause" not in x]
    return {"evidence": lines[-2:], "hops": state["hops"] + 1, "notes": [f"read {state['suspect']} logs: {len(lines)} bad lines"]}


def follow_upstream(state):
    line = next(e for e in state["evidence"] if "upstream" in e).split()
    callee = line[line.index("upstream") + 1]          # "... upstream payments returned 500 ..."
    return {"suspect": callee, "notes": [f"errors come from upstream {callee}"]}


def classify(state):
    text = " ".join(state["evidence"])
    cause = next((c for sign, c in SIGNS if sign in text), "unknown")
    return {"cause": cause, "notes": [f"cause = {cause}"]}


def false_alarm(state):
    return {"cause": "false_alarm", "notes": ["all green: false alarm, change nothing"]}


# the two decisions of the flow (conditional edges)
def after_health(state):
    return "read_logs" if state["unhealthy"] else "false_alarm"


def after_logs(state):
    upstream = any("upstream" in e for e in state["evidence"])
    return "follow_upstream" if upstream and state["hops"] < 3 else "classify"


NODES = {f.__name__: f for f in (read_alert, check_health, read_logs, follow_upstream, classify, false_alarm)}


def start_state(incident_id):
    inc = INCIDENTS[incident_id]
    return {"incident": incident_id, "alert": inc.alert, "suspect": None, "unhealthy": [], "evidence": [],
            "hops": 0, "cause": None, "notes": []}
