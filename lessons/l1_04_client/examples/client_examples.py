"""Lesson 1.4 (E) - Your own client: retries, backoff, record/replay cache.

Read the wire: what a 429 looks like (status, headers, body). Before/after numbers: failures without and with retry,
time and tokens for a live call vs a replayed call.
    uv run python lessons/l1_04_client/examples/client_examples.py > lessons/l1_04_client/examples/results.txt
"""
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["GROQ_API_KEY"]
URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
CACHE = Path(__file__).parent / "cache"
FILLER = "Log line: checkout handled request in 120 ms. " * 60          # ~600 tokens, to fill the per-minute budget


def body(i):
    return {"model": MODEL, "temperature": 0, "max_completion_tokens": 64,
            "messages": [{"role": "user", "content": FILLER + f"\nRequest {i}: reply with the word OK."}]}


print("== 1. Read the wire: the rate-limit headers on a normal reply ==")
r = httpx.post(URL, json=body(0), headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
print("status:", r.status_code)
for k in ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-limit-tokens",
          "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"):
    print(f"  {k}: {r.headers.get(k)}")

print("\n== 2. Twenty calls at once, no retry ==")


def bare(i):
    try:
        return httpx.post(URL, json=body(i), headers={"Authorization": f"Bearer {KEY}"}, timeout=60).status_code
    except httpx.TransportError as e:          # the connection itself failed: no status code at all
        return type(e).__name__


with ThreadPoolExecutor(20) as ex:
    codes = list(ex.map(bare, range(1, 21)))
print("results:", codes)
fails = [c for c in codes if c != 200]
print("ok:", codes.count(200), "| failed:", len(fails), "| kinds:", sorted({str(c) for c in fails}))


print("\n== 3. The same twenty, with retry + exponential backoff + jitter ==")


def post_retry(b, tries=8, log=None):
    for attempt in range(tries):
        try:
            r = httpx.post(URL, json=b, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
        except httpx.TransportError:               # dropped connection: retry like a 503
            r = None
        if r is not None and r.status_code not in (429, 500, 502, 503):
            return r, attempt
        ra = r.headers.get("retry-after") if r is not None else None
        wait = float(ra or 0) or min(30, 2 ** attempt)
        wait += random.uniform(0, 0.5)                           # jitter: don't all wake up at once
        if log is not None:
            log.append(round(wait, 1))
        time.sleep(wait)
    return r, tries


waits = []
t0 = time.time()
with ThreadPoolExecutor(20) as ex:
    res = list(ex.map(lambda i: post_retry(body(100 + i), log=waits), range(20)))
codes = [x.status_code for x, _ in res]
print("ok:", codes.count(200), "of 20 | retries used:", sum(a for _, a in res),
      f"| total time {time.time() - t0:.1f} s | longest wait {max(waits) if waits else 0} s")


print("\n== 4. Record / replay cache ==")


def cached_post(b):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()[:16]
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")), "replay"
    r, _ = post_retry(b)
    f.write_text(r.text, encoding="utf-8")
    return r.json(), "record"


for f in CACHE.glob("*.json"):
    f.unlink()
q = {"model": MODEL, "temperature": 0,
     "messages": [{"role": "user", "content": "Alert: checkout error rate is 23%. One sentence: what does it mean?"}]}
for run in (1, 2):
    t0 = time.time()
    data, how = cached_post(q)
    ms = (time.time() - t0) * 1000
    tok = data["usage"]["total_tokens"] if how == "record" else 0
    print(f"run {run}: {how:6} {ms:8.1f} ms  tokens billed {tok:4}  -> {data['choices'][0]['message']['content'][:60]}...")
print("files in cache:", len(list(CACHE.glob('*.json'))), "| key = sha256 of the request body")
