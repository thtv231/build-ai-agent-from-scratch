# Build an AI Agent from Scratch with LangGraph — course code

Code for the course: **Nightwatch**, an on-call agent for the made-up online shop **Brightcart**.
You build the agent by hand first (raw HTTP, your own tool calls, your own loop, tracer and graders),
then rebuild it with LangGraph and prove both behave the same.

## What is in here

| Folder | What |
|---|---|
| `nightwatch/brightcart/` | The Brightcart simulator: services, logs, deploys, flags, runbooks, a health check, and 60 incidents (train 24 / val 12 / test 24) |
| `nightwatch/agent_v0.py` | Nightwatch v0: a ReAct loop from scratch with 17 tools, a record/replay cache and a grader |
| `nightwatch/baseline.py` | Score v0 on all incidents (pass rate, pass@k, pass^k, confidence interval) |
| `nightwatch/runs/` | The recorded baseline runs and their cache, so you can replay them for free |
| `lessons/<lesson>/notebook/` | One notebook per lesson: the lesson's code, traps, exercises with check cells, solutions at the end |
| `lessons/<lesson>/examples/` | The exact scripts and outputs shown in the videos |

## Setup (about 10 minutes, see lesson 0.2)

1. Install [uv](https://docs.astral.sh/uv/) (it brings the right Python for you).
2. Clone and install:
   ```bash
   git clone https://github.com/thtv231/build-ai-agent-from-scratch.git
   cd build-ai-agent-from-scratch
   uv sync
   ```
3. Get a free Groq API key at console.groq.com, then copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY=your-key-here
   ```
   Never commit `.env` (it is in `.gitignore`).
4. Check it works:
   ```bash
   uv run pytest nightwatch/tests -q
   uv run jupyter lab
   ```

**No key yet?** Most notebooks replay recorded model replies from the caches, so they run without a key.
Cells that need a live call are marked "calls the API".

**Free tier limits:** Groq's free tier allows a limited number of tokens per minute and per day. The client in
lesson 1.4 retries on 429 errors; if you hit the daily cap, use replay mode or wait for the reset.

## Versions

Python 3.11+ (3.13 recommended), LangGraph 1.2, LangChain 1.5 — pinned in `uv.lock`.

## License

MIT
