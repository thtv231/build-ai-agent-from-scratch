"""Generate Brightcart incidents from templates.

60 incidents = 10 templates x 6 variants. Each variant changes the broken
service, the clock, the log noise and the red herrings (a harmless deploy or flag
on another service). Splits are fixed by a seeded shuffle that is stratified by
template: train 24 / val 12 / test 24.

    python -m brightcart.incidents          # writes incidents.json and prints a summary
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

from .sim import SERVICES, Incident

# who calls whom: errors show up in the caller
CALLERS = {"payments": "checkout", "inventory": "checkout", "checkout": "web", "search": "web"}


def _base(rng: random.Random) -> dict:
    svc = {}
    for n in SERVICES:
        svc[n] = {"version": f"v{rng.randint(1, 3)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}",
                  "replicas": rng.choice([2, 3]), "cpu": rng.randint(20, 50), "mem": rng.randint(30, 55),
                  "err": round(rng.uniform(0.0, 0.4), 2), "p95": rng.randint(80, 200), "restarts": 0}
    return svc


def _noise(rng, service, t0, n=6):
    lines = [f"{t0}:{rng.randint(0, 59):02d} INFO {service} handled request id={rng.randint(10000, 99999)} in {rng.randint(20, 180)}ms"
             for _ in range(n)]
    lines.append(f"{t0}:{rng.randint(0, 59):02d} WARN {service} slow GC pause 120ms")
    rng.shuffle(lines)
    return lines


def _bump(v: str) -> str:
    a, b, c = v[1:].split(".")
    return f"v{a}.{b}.{int(c) + 1}"


def _make(tpl: str, i: int, rng: random.Random) -> Incident:
    svc = _base(rng)
    hh = f"0{rng.randint(1, 4)}"
    logs = {n: _noise(rng, n, hh) for n in SERVICES}
    deploys, flags, config, history, queue = [], {}, {n: {"db_pool_size": 20} for n in SERVICES}, [], {}
    # red herring: a harmless deploy on a random other service
    def herring(exclude):
        other = rng.choice([s for s in SERVICES if s not in exclude])
        newv = _bump(svc[other]["version"])
        deploys.append({"t": f"{hh}:0{rng.randint(0, 5)}", "service": other, "version": newv, "by": "ci"})
        svc[other]["version"] = newv
        logs[other].append(f"{hh}:06 INFO {other} started version {newv}")
        return other

    diff = "easy"
    if tpl == "bad_deploy":
        x = rng.choice(["checkout", "search", "inventory", "web"])
        newv = _bump(svc[x]["version"])
        deploys.append({"t": f"{hh}:10", "service": x, "version": newv, "by": "ci"})
        svc[x].update(version=newv, err=round(rng.uniform(8, 30), 1))
        logs[x] += [f"{hh}:11 ERROR {x} TypeError: 'NoneType' object is not subscriptable (handler v{newv[1:]})"] * 3
        if i % 2:
            herring({x}); diff = "medium"
        label = {"cause_service": x, "cause": "bad_deploy", "fix": ["rollback_deploy", x, None]}
        alert = f"{x} error rate above 5%"
    elif tpl == "upstream_deploy":           # errors show in the caller, cause is the callee
        x = rng.choice(["payments", "inventory"])
        caller = CALLERS[x]
        newv = _bump(svc[x]["version"])
        deploys.append({"t": f"{hh}:12", "service": x, "version": newv, "by": "ci"})
        svc[x].update(version=newv, err=round(rng.uniform(5, 12), 1))
        svc[caller]["err"] = round(rng.uniform(15, 35), 1)
        logs[caller] += [f"{hh}:13 ERROR {caller} upstream {x} returned 500 for POST /{x}/v2/charge"] * 4
        logs[x] += [f"{hh}:13 ERROR {x} KeyError: 'currency' in serializer (v{newv[1:]})"]
        herring({x, caller})
        diff = "hard"
        label = {"cause_service": x, "cause": "bad_deploy", "fix": ["rollback_deploy", x, None]}
        alert = f"{caller} error rate above 5%"
    elif tpl == "memory_leak":
        x = rng.choice(SERVICES[:5])
        svc[x].update(mem=rng.randint(92, 99), restarts=rng.randint(4, 9), p95=rng.randint(900, 2500))
        logs[x] += [f"{hh}:2{k} WARN {x} container OOMKilled, restarting" for k in range(3)]
        if i % 2:
            herring({x}); diff = "medium"
        label = {"cause_service": x, "cause": "memory_leak", "fix": ["restart_service", x, None]}
        alert = f"{x} p95 latency above 800ms"
    elif tpl == "traffic_spike":
        x = rng.choice(["web", "search", "checkout"])
        svc[x].update(cpu=rng.randint(92, 99), p95=rng.randint(1200, 3000), replicas=2)
        logs[x] += [f"{hh}:3{k} WARN {x} request queue 512/512, shedding load" for k in range(3)]
        logs["web"].append(f"{hh}:30 INFO web campaign 'BLACKFRIDAY-EARLY' traffic x4")
        need = 4 + (i % 2) * 2
        rps = need * 250 - rng.randint(50, 200)
        logs[x].append(f"{hh}:35 INFO {x} load {rps} rps; capacity is 250 rps per replica")
        label = {"cause_service": x, "cause": "traffic_spike", "fix": ["scale_service", x, need]}
        diff = "medium"
        alert = f"{x} p95 latency above 800ms; CPU above 90%"
    elif tpl == "bad_flag":
        x = rng.choice(["checkout", "search", "web"])
        fname = rng.choice(["new_pricing_engine", "fast_cart", "ranker_v2", "one_click_pay"]) + f"_{i}"
        flags[fname] = {"on": True, "service": x, "changed_t": f"{hh}:15"}
        flags[f"dark_mode_{i}"] = {"on": True, "service": "web", "changed_t": f"{hh}:14"}
        svc[x]["err"] = round(rng.uniform(6, 20), 1)
        logs[x] += [f"{hh}:16 ERROR {x} ValueError in code path guarded by flag {fname}"] * 2
        if i % 2:
            herring({x}); diff = "medium"
        label = {"cause_service": x, "cause": "bad_flag", "fix": ["set_flag", fname, "off"]}
        alert = f"{x} error rate above 5%"
    elif tpl == "stuck_consumer":
        queue["email_outbox"] = rng.randint(4000, 20000)
        logs["email-worker"] += [f"{hh}:4{k} INFO email-worker processing msg id=88123 (attempt {k + 1})" for k in range(4)]
        label = {"cause_service": "email-worker", "cause": "stuck_consumer", "fix": ["restart_service", "email-worker", None]}
        alert = "queue email_outbox depth above 1000"
        if i % 2:
            herring({"email-worker"}); diff = "medium"
    elif tpl == "expired_secret":
        svc["payments"]["err"] = round(rng.uniform(20, 60), 1)
        svc["checkout"]["err"] = round(rng.uniform(10, 30), 1)
        logs["payments"] += [f"{hh}:00 ERROR payments provider returned 401 Unauthorized: key expired"] * 3
        logs["checkout"] += [f"{hh}:01 ERROR checkout upstream payments returned 502"] * 2
        herring({"payments", "checkout"})
        diff = "hard" if i % 2 else "medium"
        label = {"cause_service": "payments", "cause": "expired_secret", "fix": ["rotate_secret", "payments", None]}
        alert = "checkout error rate above 5%" if i % 2 else "payments error rate above 5%"
    elif tpl == "pool_config":
        x = rng.choice(["checkout", "inventory", "search"])
        config[x]["db_pool_size"] = 2
        history.append({"t": f"{hh}:20", "service": x, "key": "db_pool_size", "old": 20, "new": 2})
        svc[x]["p95"] = rng.randint(1500, 4000)
        logs[x] += [f"{hh}:2{k} WARN {x} waited {rng.randint(900, 3000)}ms for a db connection (pool 2/2 busy)" for k in range(3)]
        diff = "hard" if i % 2 else "medium"
        if i % 2:
            herring({x})
        label = {"cause_service": x, "cause": "bad_config", "fix": ["set_config", x, "db_pool_size=20"]}
        alert = f"{x} p95 latency above 800ms"
    elif tpl == "disk_full":
        x = rng.choice(["search", "inventory"])
        svc[x]["err"] = round(rng.uniform(5, 15), 1)
        logs[x] += [f"{hh}:5{k} ERROR {x} OSError: [Errno 28] No space left on device: '/var/cache/{x}'" for k in range(2)]
        if i % 2:
            herring({x}); diff = "medium"
        label = {"cause_service": x, "cause": "disk_full", "fix": ["clear_cache", x, None]}
        alert = f"{x} error rate above 5%"
    elif tpl == "false_alarm":
        x = rng.choice(SERVICES)
        logs[x] += [f"{hh}:33 WARN {x} health probe timed out once", f"{hh}:34 INFO {x} health probe ok"]
        if i % 2:
            herring({x}); diff = "medium"
        label = {"cause_service": x, "cause": "false_alarm", "fix": None}
        alert = f"{x} health probe failed (1 of 3), auto-resolved"
    else:
        raise ValueError(tpl)

    for n in logs:                       # real logs are in time order
        logs[n].sort(key=lambda line: line[:5])
    return Incident(id=f"{tpl}-{i}", template=tpl, alert=alert, split="", difficulty=diff,
                    label=label, services=svc, deploys=deploys, flags=flags, config=config,
                    config_history=history, logs=logs, queue=queue)


TEMPLATES = ["bad_deploy", "upstream_deploy", "memory_leak", "traffic_spike", "bad_flag",
             "stuck_consumer", "expired_secret", "pool_config", "disk_full", "false_alarm"]


def generate(seed: int = 7) -> list[Incident]:
    rng = random.Random(seed)
    out = []
    for tpl in TEMPLATES:
        group = [_make(tpl, i, rng) for i in range(6)]
        order = list(range(6))
        random.Random(f"{seed}-{tpl}").shuffle(order)
        # stratified 24/12/24: each template gives 2 train, 1 val, 2 test; the 6th goes to
        # val (templates 0-1), train (2-5) or test (6-9)
        t = TEMPLATES.index(tpl)
        sixth = "val" if t < 2 else ("train" if t < 6 else "test")
        for pos, idx in enumerate(order):
            group[idx].split = ["train", "train", "val", "test", "test", sixth][pos]
        out += group
    return out


def load(path: str | Path) -> list[Incident]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Incident(**d) for d in data]


def fresh(inc: Incident) -> Incident:
    return copy.deepcopy(inc)


if __name__ == "__main__":
    incs = generate()
    p = Path(__file__).resolve().parent.parent / "incidents.json"
    p.write_text(json.dumps([asdict(x) for x in incs], indent=1), encoding="utf-8")
    from collections import Counter
    print(len(incs), dict(Counter(x.split for x in incs)), dict(Counter(x.difficulty for x in incs)))
