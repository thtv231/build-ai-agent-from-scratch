"""Nightwatch v0 rebuilt with langchain 1.5 create_agent. Shared by lesson 4.4 and 4.5.

Line-by-line map to agent_v0.run():
  msgs = [system, PAGER]               -> system_prompt=SYSTEM, {"messages": [PAGER]}
  for step in range(14)                -> ModelCallLimitMiddleware(run_limit=14)
  out = chat(body)                     -> ChatGroq (same model, temperature 1.0, reasoning medium, 2048 tokens)
  chat()'s record/replay cache         -> @wrap_model_call  record_replay
  if "error" in out: nudge, continue   -> @wrap_model_call  retry_malformed
  if not calls: "Use the tools..."     -> @after_model      nudge_to_tools (jump back to the model)
  for c in calls: run_tool(...)        -> the 17 tools as StructuredTools (ToolNode runs them)
  trace.append(...)                    -> @wrap_tool_call   log_tool
  if name == "finish": break           -> finish tool with return_direct=True
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware, ModelResponse, after_model, wrap_model_call,
                                         wrap_tool_call)
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain_core.tools import StructuredTool
import os
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "nightwatch"))
import agent_v0  # noqa: E402
from brightcart.incidents import fresh  # noqa: E402
from brightcart.sim import World  # noqa: E402

load_dotenv(ROOT / ".env")


CACHE = ROOT / "nightwatch" / "runs" / "lc_cache"


class Malformed(Exception):
    """Groq answered HTTP 400 tool_use_failed: the model wrote a tool call that is not valid JSON."""


def make_model(temperature=1.0):
    # no key yet? replies come from the record/replay cache; the client just needs *some* key to start
    key = os.environ.get("GROQ_API_KEY") or "replay-only-no-key"
    return ChatGroq(model="openai/gpt-oss-20b", temperature=temperature, reasoning_effort="medium",
                    max_tokens=2048, max_retries=6, api_key=key)


def make_tools(world, box):
    def make(spec):
        name = spec["function"]["name"]

        def fn(**args):
            if name == "finish":
                box["final"] = args
                return "ok"
            try:
                result = agent_v0.run_tool(world, name, args)
            except Exception as e:                       # the error goes back to the model, as in v0
                result = f"ERROR: {e}"
            return json.dumps(result, default=str)[:4000]
        return StructuredTool.from_function(fn, name=name, description=spec["function"]["description"],
                                            args_schema=spec["function"]["parameters"],
                                            return_direct=(name == "finish"))
    return [make(t) for t in agent_v0.TOOLS]


def make_middleware(box, verbose=False, sample=0, temperature=1.0, cache=True):
    @wrap_model_call
    def retry_malformed(request, handler):
        """A broken tool call: tell the model and call it again (v0 does the same, as one more step)."""
        for attempt in range(3):
            try:
                return handler(request)
            except Malformed as e:
                box["errors"].append(str(e)[:80])
                request = request.override(messages=request.messages + [
                    HumanMessage("Your last tool call was malformed. Call one tool with valid JSON.")])
        return handler(request)

    @wrap_model_call
    def record_replay(request, handler):
        """Same idea as agent_v0.chat(): key = the conversation so far; a hit skips the API call."""
        convo = [(m.type, m.content, getattr(m, "tool_calls", None), getattr(m, "tool_call_id", None))
                 for m in request.messages]
        key = json.dumps([str(request.system_prompt), convo, sample, temperature], sort_keys=True, default=str)
        f = CACHE / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
        else:
            try:
                rec = {"messages": messages_to_dict(handler(request).result)}
            except BadRequestError as e:
                rec = {"error": str(e)[:300]}
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(rec), encoding="utf-8")
        if "error" in rec:
            raise Malformed(rec["error"])
        return ModelResponse(result=messages_from_dict(rec["messages"]))

    @after_model(can_jump_to=["model"])
    def nudge_to_tools(state, runtime):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            box["trace"].append({"tool": None, "text": (last.content or "")[:120]})
            return {"messages": [HumanMessage("Use the tools. End with finish().")], "jump_to": "model"}
        return None

    @wrap_tool_call
    def log_tool(request, handler):
        out = handler(request)
        call = request.tool_call
        box["trace"].append({"tool": call["name"], "args": call["args"], "result": str(out.content)[:120]})
        if verbose:
            print(f"{len(box['trace']) - 1:2d} {call['name']}({json.dumps(call['args'])[:70]}) -> {str(out.content)[:80]}")
        return out

    # wrap hooks: first in the list = outermost (retry around the cache around the real call).
    # after_model hooks run in REVERSE list order: nudge_to_tools is listed before the limit so the
    # limit counts a text-only reply before the nudge jumps back to the model (v0 counts it as a step).
    return [nudge_to_tools, ModelCallLimitMiddleware(run_limit=14, exit_behavior="end"), retry_malformed,
            *([record_replay] if cache else []), log_tool]


def run(inc, sample=0, temperature=1.0, verbose=False, model=None):
    """Same return shape as agent_v0.run(), so agent_v0.grade() works unchanged."""
    world = World(fresh(inc))
    # ToolNode runs sync tools in a thread pool; v0's sqlite connection belongs to the main thread, so
    # query_db failed with "SQLite objects created in a thread ..." until we reopen it thread-safe:
    world.db = sqlite3.connect(":memory:", check_same_thread=False)
    world._load_db()
    box = {"final": None, "trace": [], "errors": []}
    agent = create_agent(model or make_model(temperature), make_tools(world, box), system_prompt=agent_v0.SYSTEM,
                         middleware=make_middleware(box, verbose, sample, temperature, cache=model is None))
    out = agent.invoke({"messages": [HumanMessage(f"PAGER: {inc.alert}")]}, config={"recursion_limit": 100})
    usage = {"prompt": 0, "completion": 0}
    for m in out["messages"]:
        if isinstance(m, AIMessage) and m.usage_metadata:
            usage["prompt"] += m.usage_metadata["input_tokens"]
            usage["completion"] += m.usage_metadata["output_tokens"]
    return {"incident": inc.id, "final": box["final"], "writes": world.actions, "healthy": world.health()["healthy"],
            "steps": len(box["trace"]), "model_calls": sum(isinstance(m, AIMessage) for m in out["messages"]), "usage": usage,
            "trace": box["trace"], "errors": box["errors"], "label": inc.label,
            "stop_reason": "finish" if box["final"] else "model call limit"}
