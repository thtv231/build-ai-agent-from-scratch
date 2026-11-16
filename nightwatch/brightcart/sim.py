"""Brightcart: a made-up online shop that breaks on purpose.

One World = one incident. It holds a small SQLite database (services, deploys,
flags, config, metrics), per-service log lines, and the runbooks. Agents see it
only through tools. The health check is the signal an agent may learn from; the
hidden label (incident.label) is what the grader uses.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

SERVICES = ["web", "checkout", "payments", "inventory", "search", "email-worker"]

RUNBOOKS = {
    "high-error-rate": (
        "# High error rate\n"
        "1. Find which service the errors START in. Errors in a caller often come from a callee.\n"
        "2. Check recent deploys and recent flag or config changes for that service.\n"
        "3. A bad deploy is fixed by rolling back that deploy. Restarting does not help.\n"
        "4. A bad flag is fixed by turning the flag off.\n"
        "5. A bad config change is fixed by setting the old value back."
    ),
    "high-latency": (
        "# High latency\n"
        "1. Check CPU and replicas. CPU above 85% with normal errors means not enough capacity: scale out.\n"
        "2. Check memory. Memory climbing with restarts (OOMKilled) means a leak: restart buys time.\n"
        "3. Check connection pool waits in the logs: a pool that is too small looks like latency."
    ),
    "queue-backlog": (
        "# Queue backlog\n"
        "1. If the queue grows and the consumer logs show no progress, the consumer is stuck: restart it.\n"
        "2. Do not scale a stuck consumer; new replicas get stuck on the same message."
    ),
    "auth-failures": (
        "# Upstream 401/403\n"
        "1. 401 from a provider after a date boundary usually means an expired secret: rotate it.\n"
        "2. Rolling back does not fix an expired secret."
    ),
    "disk-full": (
        "# Disk full\n"
        "1. 'No space left on device' in logs: clear the cache of that service.\n"
        "2. Never delete data tables to free space."
    ),
    "flapping-alert": (
        "# Alert that clears by itself\n"
        "1. If every health metric is normal and the alert fired once and cleared, it is a false alarm.\n"
        "2. Change nothing. Report it as a false alarm."
    ),
}

WRITE_ACTIONS = {"restart_service", "rollback_deploy", "scale_service", "set_flag",
                 "set_config", "rotate_secret", "clear_cache"}


@dataclass
class Incident:
    id: str
    template: str
    alert: str                      # what the pager says
    split: str                      # train / val / test
    difficulty: str                 # easy / medium / hard
    label: dict                     # hidden: {"cause_service", "cause", "fix": [action, target, arg]}
    services: dict                  # name -> {version, replicas, cpu, mem, err, p95, restarts}
    deploys: list                   # [{t, service, version, by}]
    flags: dict                     # name -> {on, service, changed_t}
    config: dict                    # service -> {key: value}
    config_history: list            # [{t, service, key, old, new}]
    logs: dict                      # service -> [lines]
    queue: dict = field(default_factory=dict)   # name -> depth


class World:
    """One live incident. Tools call methods here; the grader reads .actions."""

    def __init__(self, inc: Incident):
        self.inc = inc
        self.actions: list[tuple] = []          # every write: (name, target, arg)
        self.fixed = False
        self.db = sqlite3.connect(":memory:")
        self._load_db()

    # ---------- database (read-only to the agent) ----------
    def _load_db(self):
        c = self.db
        c.execute("create table services(name text primary key, version text, replicas int,"
                  " cpu_pct int, mem_pct int, error_pct real, p95_ms int, restarts_1h int)")
        c.execute("create table deploys(t text, service text, version text, deployed_by text)")
        c.execute("create table flags(name text, is_on int, service text, changed_t text)")
        c.execute("create table config_changes(t text, service text, key text, old text, new text)")
        c.execute("create table orders(id int, status text, created_t text)")
        for n, s in self.inc.services.items():
            c.execute("insert into services values(?,?,?,?,?,?,?,?)",
                      (n, s["version"], s["replicas"], s["cpu"], s["mem"], s["err"], s["p95"], s["restarts"]))
        for d in self.inc.deploys:
            c.execute("insert into deploys values(?,?,?,?)", (d["t"], d["service"], d["version"], d["by"]))
        for n, f in self.inc.flags.items():
            c.execute("insert into flags values(?,?,?,?)", (n, int(f["on"]), f["service"], f["changed_t"]))
        for h in self.inc.config_history:
            c.execute("insert into config_changes values(?,?,?,?,?)",
                      (h["t"], h["service"], h["key"], str(h["old"]), str(h["new"])))
        failed = 40 if self.inc.label.get("cause") not in (None, "false_alarm") else 1
        for i in range(60):
            c.execute("insert into orders values(?,?,?)",
                      (1000 + i, "failed" if i < failed // 2 else "paid", f"02:{i:02d}"))
        c.commit()

    def query(self, sql: str) -> list:
        if not sql.strip().lower().startswith("select"):
            raise ValueError("read-only: only SELECT is allowed")
        return self.db.execute(sql).fetchall()[:50]

    # ---------- reads ----------
    def status(self, service: str | None = None) -> dict:
        s = self.inc.services
        if service:
            if service not in s:
                raise KeyError(f"unknown service {service!r}; known: {', '.join(s)}")
            return {service: s[service]}
        return s

    def logs(self, service: str, n: int = 20) -> list[str]:
        if service not in self.inc.services:
            raise KeyError(f"unknown service {service!r}")
        return self.inc.logs.get(service, [])[-n:]

    def runbook(self, topic: str) -> str:
        if topic not in RUNBOOKS:
            raise KeyError(f"no runbook {topic!r}; topics: {', '.join(RUNBOOKS)}")
        return RUNBOOKS[topic]

    # ---------- writes ----------
    def act(self, name: str, target: str, arg=None) -> str:
        if name not in WRITE_ACTIONS:
            raise ValueError(f"unknown action {name}")
        known = set(self.inc.services) | set(self.inc.flags)
        if target not in known:
            raise KeyError(f"unknown target {target!r}")
        self.actions.append((name, target, arg))
        want = self.inc.label["fix"]
        if want and self._matches(want, (name, target, arg)):
            self.fixed = True
            self._heal()
            return f"{name}({target}) done."
        return f"{name}({target}) done."   # same message: the agent must check health itself

    @staticmethod
    def _matches(want, got) -> bool:
        wn, wt, wa = want
        gn, gt, ga = got
        if (wn, wt) != (gn, gt):
            return False
        if wn == "scale_service":
            try:
                return int(ga) >= int(wa)
            except (TypeError, ValueError):
                return False
        if wn in ("set_flag", "set_config"):
            return str(ga).lower() == str(wa).lower()
        return True

    def _heal(self):
        for s in self.inc.services.values():
            s["err"] = min(s["err"], 0.3)
            s["p95"] = min(s["p95"], 220)
            s["cpu"] = min(s["cpu"], 55)
            s["mem"] = min(s["mem"], 60)
        for q in self.inc.queue:
            self.inc.queue[q] = 0

    # ---------- health check (the learning signal) ----------
    def health(self) -> dict:
        bad = {n: s for n, s in self.inc.services.items()
               if s["err"] > 1.0 or s["p95"] > 800 or s["cpu"] > 85}
        stuck = {q: d for q, d in self.inc.queue.items() if d > 100}
        ok = not bad and not stuck
        return {"healthy": ok,
                "unhealthy_services": sorted(bad),
                "queues_over_limit": stuck}

    def snapshot(self) -> str:
        return json.dumps({"alert": self.inc.alert, "health": self.health()})
