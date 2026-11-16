"""Lesson 1.1 - Raw HTTP, then the SDK: messages and roles.

Every output shown in the video comes from running this file:
    uv run python lessons/l1_01_messages/examples/messages_examples.py > .../results.txt
"""
import json
import os
import time

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
KEY = os.environ["GROQ_API_KEY"]
URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def post(body):
    for attempt in range(6):          # the free tier sometimes says 429: wait and retry
        r = httpx.post(URL, json=body, timeout=60,
                       headers={"Authorization": f"Bearer {KEY}"})
        if r.status_code != 429:
            return r
        time.sleep(float(r.headers.get("retry-after", 5)))
    return r


print("== 1. The request is just JSON ==")
body = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are Nightwatch, the on-call assistant for the shop Brightcart. Answer in one sentence."},
        {"role": "user", "content": "Alert: checkout error rate is 23%. What does that mean?"},
    ],
    "temperature": 0,
}
print(json.dumps(body, indent=2)[:600])

print("\n== 2. Send it with plain HTTP ==")
r = post(body)
print("status:", r.status_code)
data = r.json()
print("top-level keys:", list(data))
msg = data["choices"][0]["message"]
print("role:", msg["role"])
print("content:", msg["content"])
print("finish_reason:", data["choices"][0]["finish_reason"])
u = data["usage"]
print("usage:", {k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")})

print("\n== 3. The same call with the SDK ==")
client = OpenAI(api_key=KEY, base_url="https://api.groq.com/openai/v1", max_retries=6)
resp = client.chat.completions.create(model=MODEL, messages=body["messages"], temperature=0)
print("type:", type(resp).__name__)
print("role:", resp.choices[0].message.role)
print("content:", resp.choices[0].message.content)
print("total_tokens:", resp.usage.total_tokens)

print("\n== 4. The API remembers nothing ==")
first = [{"role": "user", "content": "The failing service is called payments. Remember that."}]
a1 = client.chat.completions.create(model=MODEL, messages=first, temperature=0)
print("call 1 reply:", a1.choices[0].message.content)
second = [{"role": "user", "content": "Which service did I say is failing? Answer with one word, or 'unknown'."}]
a2 = client.chat.completions.create(model=MODEL, messages=second, temperature=0)
print("call 2, new list only:", a2.choices[0].message.content)
history = first + [{"role": "assistant", "content": a1.choices[0].message.content}] + second
a3 = client.chat.completions.create(model=MODEL, messages=history, temperature=0)
print("call 3, whole history:", a3.choices[0].message.content)
print("messages sent in call 3:", len(history))

print("\n== 5. Roles change behaviour ==")
for system in ["Answer in one sentence.", "Answer with exactly one word: CRITICAL, HIGH or LOW."]:
    m = [{"role": "system", "content": system},
         {"role": "user", "content": "Alert: checkout error rate is 23%. How bad is it?"}]
    out = client.chat.completions.create(model=MODEL, messages=m, temperature=0)
    print(f"system={system[:30]!r:34} -> {out.choices[0].message.content}")
