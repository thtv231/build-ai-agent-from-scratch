"""Lesson 3.3 - Your own tracer and waterfall.

A span = name, start, end, tokens, a short args/result summary. We wrap the two things Nightwatch v0
does in its loop (call the model, run a tool) and replay one cached baseline run: upstream_deploy-0, sample 0.
Replay is free, so model durations are the server times Groq recorded in each cached response
(queue_time + total_time); tool durations are measured now with perf_counter.
    uv run python lessons/l3_03_tracer/examples/tracer.py > lessons/l3_03_tracer/examples/results.txt
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
import agent_v0  # noqa: E402
from brightcart.incidents import generate  # noqa: E402


# ---------------------------------------------------------------- the tracer: ~25 lines
class Tracer:
    def __init__(self):
        self.spans, self.clock = [], 0.0          # clock in seconds since the run started

    def add(self, name, seconds, **info):
        span = {"name": name, "start": self.clock, "end": self.clock + seconds, **info}
        self.spans.append(span)
        self.clock = span["end"]
        return span


tracer = Tracer()
real_chat, real_tool = agent_v0.chat, agent_v0.run_tool


def traced_chat(body, mode="record", sample=0):
    t0 = time.perf_counter()
    out = real_chat(body, mode, sample)
    u = out.get("usage", {})
    secs = (u.get("queue_time", 0) + u.get("total_time", 0)) or (time.perf_counter() - t0)
    calls = [c["function"]["name"] for c in (out.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []]
    tracer.add("model", secs, prompt=u.get("prompt_tokens", 0), completion=u.get("completion_tokens", 0),
               summary=("asks " + ", ".join(calls)) if calls else out.get("error", "text answer"))
    return out


def traced_tool(world, name, args):
    t0 = time.perf_counter()
    result = "(raised)"
    try:
        result = real_tool(world, name, args)
        return result
    finally:                                   # the span is recorded even when the tool raises
        tracer.add(f"tool:{name}", time.perf_counter() - t0, prompt=0, completion=0,
                   summary=f"{json.dumps(args)[:40]} -> {str(result)[:40]}")


agent_v0.chat, agent_v0.run_tool = traced_chat, traced_tool       # wrap, don't edit the loop
# ----------------------------------------------------------------

inc = next(x for x in generate() if x.id == "upstream_deploy-0")
r = agent_v0.run(inc, mode="replay", sample=0)

print("== 1. The run (replayed from cache, no API call) ==")
print(f"incident {inc.id} | alert: {inc.alert} | stop: {r['stop_reason']} | healthy {r['healthy']} | "
      f"writes {r['writes']} | finish: {r['final']['root_cause_service']}/{r['final']['cause']}")

print("\n== 2. Span table ==")
print(f"{'#':>2} {'span':24} {'start ms':>9} {'end ms':>9} {'dur ms':>8} {'prompt':>7} {'compl':>6}  summary")
for i, s in enumerate(tracer.spans):
    print(f"{i:2d} {s['name']:24} {s['start'] * 1000:9.1f} {s['end'] * 1000:9.1f} {(s['end'] - s['start']) * 1000:8.2f}"
          f" {s['prompt']:7d} {s['completion']:6d}  {s['summary'][:70]}")

print("\n== 3. Waterfall (each # = 1/60 of the run) ==")
total = tracer.clock
for s in tracer.spans:
    a, b = int(s["start"] / total * 60), max(int(s["end"] / total * 60), int(s["start"] / total * 60) + 1)
    bar = " " * a + ("#" if s["name"] == "model" else "|") * (b - a)
    print(f"{s['name'][:20]:20} {bar:60} {(s['end'] - s['start']) * 1000:8.2f} ms")
print("(model spans = #, tool spans = | ; a tool bar is drawn 1 char wide even when it is far shorter)")

print("\n== 4. Totals ==")
model = [s for s in tracer.spans if s["name"] == "model"]
tools = [s for s in tracer.spans if s["name"] != "model"]
mt, tt = sum(s["end"] - s["start"] for s in model), sum(s["end"] - s["start"] for s in tools)
print(f"model calls {len(model)} | tool calls {len(tools)} | trace steps (agent_v0) {r['steps']}")
print(f"model time {mt * 1000:.0f} ms ({mt / total:.1%}) | tool time {tt * 1000:.2f} ms ({tt / total:.2%}) | total {total * 1000:.0f} ms")
print("prompt tokens per model call:", [s["prompt"] for s in model])
print("completion tokens per model call:", [s["completion"] for s in model])
print(f"tokens: prompt {sum(s['prompt'] for s in model)} + completion {sum(s['completion'] for s in model)}"
      f" = {sum(s['prompt'] + s['completion'] for s in model)}  (agent_v0 usage: {r['usage']})")
