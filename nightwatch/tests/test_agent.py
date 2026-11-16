"""The loop, tools and grader work end to end with a scripted fake model (no network)."""
import json

import agent_v0
from brightcart.incidents import generate

INCS = {x.id: x for x in generate()}


def scripted(calls):
    """Return a fake chat() that plays the given tool calls one per turn."""
    it = iter(calls)

    def fake(body, mode="record", sample=0):
        name, args = next(it)
        return {"usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"content": "", "tool_calls": [
                    {"id": f"c{name}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}]}}]}
    return fake


def test_oracle_passes(monkeypatch):
    inc = INCS["upstream_deploy-0"]
    svc = inc.label["cause_service"]
    monkeypatch.setattr(agent_v0, "chat", scripted([
        ("read_logs", {"service": "checkout"}),
        ("recent_deploys", {}),
        ("rollback_deploy", {"service": svc}),
        ("check_health", {}),
        ("finish", {"root_cause_service": svc, "cause": "bad_deploy", "summary": "rolled back"}),
    ]))
    r = agent_v0.run(inc)
    g = agent_v0.grade(r)
    assert r["healthy"] and g["pass"] and g["clean"]


def test_blaming_the_caller_fails(monkeypatch):
    inc = INCS["upstream_deploy-0"]
    caller = "checkout"
    monkeypatch.setattr(agent_v0, "chat", scripted([
        ("rollback_deploy", {"service": caller}),
        ("restart_service", {"service": caller}),
        ("finish", {"root_cause_service": caller, "cause": "bad_deploy", "summary": "x"}),
    ]))
    r = agent_v0.run(inc)
    assert not agent_v0.grade(r)["pass"]


def test_false_alarm_needs_no_writes(monkeypatch):
    inc = next(x for x in INCS.values() if x.template == "false_alarm")
    svc = inc.label["cause_service"]
    monkeypatch.setattr(agent_v0, "chat", scripted([
        ("check_health", {}),
        ("finish", {"root_cause_service": svc, "cause": "false_alarm", "summary": "cleared"}),
    ]))
    assert agent_v0.grade(agent_v0.run(inc))["pass"]


def test_tool_errors_go_back_to_model(monkeypatch):
    inc = INCS["bad_deploy-0"]
    monkeypatch.setattr(agent_v0, "chat", scripted([
        ("read_logs", {"service": "nope"}),
        ("finish", {"root_cause_service": "web", "cause": "unknown", "summary": "x"}),
    ]))
    r = agent_v0.run(inc)
    assert r["trace"][0]["result"].startswith("ERROR")
