"""Nightwatch v0: a ReAct loop from scratch (no framework).

The model sees the pager alert and a set of tools over one Brightcart World. It
reads, acts, checks health, and ends with finish(). Every model call goes through
a record/replay cache so a run can be replayed for free.

    py -3.12 agent_v0.py bad_deploy-0            # one incident, live (needs GROQ_API_KEY)
    py -3.12 agent_v0.py bad_deploy-0 --replay   # from cache only
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from brightcart.incidents import fresh, generate
from brightcart.sim import RUNBOOKS, World

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "runs" / "cache"
URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = """You are Nightwatch, the on-call agent for the online shop Brightcart.
A pager alert woke you. Find the root cause, fix it with the smallest safe change,
confirm with check_health, then call finish.
Rules: read before you write. Never change something you have no evidence for.
If everything is healthy and the alert cleared, it is a false alarm: change nothing."""


def _fn(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": req or []}}}


S = {"type": "string"}
TOOLS = [
    _fn("service_status", "CPU, memory, error %, p95 latency, replicas, restarts of one or all services.",
        {"service": S}),
    _fn("read_logs", "Last n log lines of a service.", {"service": S, "n": {"type": "integer"}}, ["service"]),
    _fn("recent_deploys", "Deploys in the last hour."),
    _fn("list_flags", "Feature flags with owner service and when they changed."),
    _fn("config_changes", "Config changes in the last hour."),
    _fn("queue_depth", "Depth of each message queue."),
    _fn("query_db", "Read-only SQL (SELECT) on tables services, deploys, flags, config_changes, orders.",
        {"sql": S}, ["sql"]),
    _fn("read_runbook", f"Read a runbook. Topics: {', '.join(RUNBOOKS)}.", {"topic": S}, ["topic"]),
    _fn("check_health", "Run the health check. Use it after any change."),
    _fn("restart_service", "Restart a service.", {"service": S}, ["service"]),
    _fn("rollback_deploy", "Roll back the latest deploy of a service.", {"service": S}, ["service"]),
    _fn("scale_service", "Set the replica count.", {"service": S, "replicas": {"type": "integer"}},
        ["service", "replicas"]),
    _fn("set_flag", "Turn a feature flag on or off.", {"flag": S, "value": {"type": "string", "enum": ["on", "off"]}},
        ["flag", "value"]),
    _fn("set_config", "Set a config key of a service.", {"service": S, "key": S, "value": S},
        ["service", "key", "value"]),
    _fn("rotate_secret", "Rotate the provider secret of a service.", {"service": S}, ["service"]),
    _fn("clear_cache", "Clear the disk cache of a service.", {"service": S}, ["service"]),
    _fn("finish", "End the incident with your conclusion.",
        {"root_cause_service": S,
         "cause": {"type": "string", "enum": ["bad_deploy", "memory_leak", "traffic_spike", "bad_flag",
                                               "stuck_consumer", "expired_secret", "bad_config", "disk_full",
                                               "false_alarm", "unknown"]},
         "summary": S}, ["root_cause_service", "cause", "summary"]),
]


WRITE_TOOLS = {"restart_service", "rollback_deploy", "scale_service", "set_flag", "set_config",
               "rotate_secret", "clear_cache"}


def run_tool(w: World, name: str, a: dict):
    if name == "service_status":
        return w.status(a.get("service") or None)
    if name == "read_logs":
        return w.logs(a["service"], int(a.get("n") or 20))
    if name == "recent_deploys":
        return w.inc.deploys
    if name == "list_flags":
        return w.inc.flags
    if name == "config_changes":
        return w.inc.config_history
    if name == "queue_depth":
        return w.inc.queue or {"note": "no queues over 0"}
    if name == "query_db":
        return w.query(a["sql"])
    if name == "read_runbook":
        return w.runbook(a["topic"])
    if name == "check_health":
        return w.health()
    if name == "scale_service":
        return w.act(name, a["service"], a["replicas"])
    if name == "set_flag":
        return w.act(name, a["flag"], a["value"])
    if name == "set_config":
        return w.act(name, a["service"], f"{a['key']}={a['value']}")
    if name in ("restart_service", "rollback_deploy", "rotate_secret", "clear_cache"):
        return w.act(name, a["service"])
    raise KeyError(f"unknown tool {name}")


# ---------------- model client with record/replay ----------------
def _key():
    k = os.environ.get("GROQ_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip() == "GROQ_API_KEY":
                k = val.strip().strip('"')
    return k


class DailyCap(RuntimeError):
    pass


def chat(body: dict, mode: str = "record", sample: int = 0) -> dict:
    """mode: record (cache hit or call+save), replay (cache only), live (always call, save)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256((json.dumps(body, sort_keys=True) + f"#{sample}").encode()).hexdigest()[:24]
    f = CACHE / f"{h}.json"
    if mode != "live" and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if mode == "replay":
        raise LookupError("not in cache: " + h)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
                                          "User-Agent": "nightwatch-course/0.1"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            f.write_text(json.dumps(out), encoding="utf-8")
            return out
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 2 ** attempt * 3)
                if wait > 120:      # daily cap: stop cleanly, the cache keeps what is done
                    raise DailyCap(f"rate limit, retry after {wait:.0f}s: {msg[-200:]}")
                print(f"  [{e.code}] waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(min(wait, 90))
                continue
            if e.code == 400 and ("tool_use_failed" in msg or "output_parse_failed" in msg):
                out = {"error": "tool_use_failed" if "tool_use_failed" in msg else "output_parse_failed",
                       "detail": msg}
                f.write_text(json.dumps(out), encoding="utf-8")   # cache failures too, so replay is complete
                return out
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:   # network blips
            print(f"  [net] {e}; retrying", file=sys.stderr)
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("gave up after retries")


def run(inc, model="openai/gpt-oss-20b", mode="record", sample=0, max_steps=14, verbose=False,
        guards=False, token_budget=None) -> dict:
    """guards=True adds the lesson 3.2 stop conditions: block an identical repeated write, and
    once a health check is green after a change (or on a read-only false alarm), ask for finish()."""
    w = World(fresh(inc))
    done_writes, stop_reason = {}, None
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"PAGER: {inc.alert}"}]
    trace, usage, final = [], {"prompt": 0, "completion": 0}, None
    for step in range(max_steps):
        body = {"model": model, "messages": msgs, "tools": TOOLS, "temperature": 1.0,
                "reasoning_effort": "medium", "max_completion_tokens": 2048}
        if token_budget and usage["prompt"] + usage["completion"] > token_budget:
            stop_reason = "token budget"
            break
        out = chat(body, mode, sample)
        if "error" in out:
            msgs.append({"role": "user", "content": "Your last tool call was malformed. Call one tool with valid JSON."})
            trace.append({"step": step, "error": out["error"]})
            continue
        u = out.get("usage", {})
        usage["prompt"] += u.get("prompt_tokens", 0); usage["completion"] += u.get("completion_tokens", 0)
        m = out["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            msgs.append({"role": "user", "content": "Use the tools. End with finish()."})
            trace.append({"step": step, "text": (m.get("content") or "")[:200]})
            continue
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "finish":
                final = args
                result = "ok"
            elif guards and name in WRITE_TOOLS and (name, json.dumps(args, sort_keys=True)) in done_writes:
                result = (f"ALREADY DONE at step {done_writes[(name, json.dumps(args, sort_keys=True))]}: "
                          "this exact change was applied. Do not repeat it; run check_health.")
            else:
                try:
                    result = run_tool(w, name, args)
                except Exception as e:      # the error goes back to the model
                    result = f"ERROR: {e}"
                if guards and name in WRITE_TOOLS:
                    done_writes[(name, json.dumps(args, sort_keys=True))] = step
            trace.append({"step": step, "tool": name, "args": args, "result": str(result)[:300]})
            if verbose:
                print(f"{step:2d} {name}({json.dumps(args)[:80]}) -> {str(result)[:100]}")
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)[:4000]})
        if final is not None:
            break
        if guards and any(t.get("tool") == "check_health" and "'healthy': True" in t.get("result", "")
                          for t in trace if t.get("step") == step):
            msgs.append({"role": "user", "content": "Health is green. If you have found the cause, call finish now."})
    if final is None and stop_reason is None:
        stop_reason = "step limit"
    return {"stop_reason": stop_reason or "finish", "incident": inc.id, "split": inc.split, "template": inc.template, "difficulty": inc.difficulty,
            "sample": sample, "final": final, "writes": w.actions, "healthy": w.health()["healthy"],
            "steps": len(trace), "usage": usage, "trace": trace, "label": inc.label}


def grade(r: dict) -> dict:
    """Hidden-label grader. pass = healthy end + right cause service + at most one wrong write."""
    lab, fin = r["label"], r["final"] or {}
    right_service = fin.get("root_cause_service") == lab["cause_service"]
    if lab["fix"] is None:
        ok = right_service and fin.get("cause") == "false_alarm" and not r["writes"]
        return {"pass": ok, "clean": ok, "right_service": right_service}
    wrong = len(r["writes"]) - (1 if r["healthy"] else 0)
    return {"pass": r["healthy"] and right_service and wrong <= 1,
            "clean": r["healthy"] and right_service and wrong == 0,
            "right_service": right_service}


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "bad_deploy-0"
    mode = "replay" if "--replay" in sys.argv else "record"
    model = "openai/gpt-oss-120b" if "--120b" in sys.argv else "openai/gpt-oss-20b"
    inc = next(x for x in generate() if x.id == want)
    r = run(inc, model=model, mode=mode, verbose=True)
    print(json.dumps({"final": r["final"], "writes": r["writes"], "healthy": r["healthy"],
                      "usage": r["usage"], **grade(r)}, indent=1))
