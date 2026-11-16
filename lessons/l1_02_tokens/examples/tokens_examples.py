"""Lesson 1.2 - Tokens, cost, temperature and reasoning effort, by hand.

Every output shown in the video comes from running this file:
    uv run python lessons/l1_02_tokens/examples/tokens_examples.py > lessons/l1_02_tokens/examples/results.txt
Prices: Groq model page, checked 2026-11-16: gpt-oss-20b $0.075 / 1M input, $0.30 / 1M output.
"""
import os
from collections import Counter

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
PRICE_IN, PRICE_OUT = 0.075, 0.30          # dollars per million tokens
SYSTEM = "You are Nightwatch, the on-call assistant for the shop Brightcart. Answer in one sentence."
ALERT = "Alert: checkout error rate is 23%. What does that mean?"

print("== 1. Text becomes tokens ==")
enc = tiktoken.get_encoding("o200k_harmony")
for text in ["checkout", "Brightcart", "error rate is 23%", "Nightwatch"]:
    ids = enc.encode(text)
    print(f"{text!r:22} -> {len(ids)} tokens: {[enc.decode([i]) for i in ids]}")
n_sys, n_user = len(enc.encode(SYSTEM)), len(enc.encode(ALERT))
print("system text:", n_sys, "tokens | user text:", n_user, "tokens | sum:", n_sys + n_user)

print("\n== 2. What the API counts ==")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ALERT}]
r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0)
u = r.usage
print("prompt_tokens:", u.prompt_tokens, "| our count of the two texts:", n_sys + n_user,
      "| extra (roles, wrappers, default instructions):", u.prompt_tokens - (n_sys + n_user))
reasoning = getattr(u, "completion_tokens_details", None)
rt = getattr(reasoning, "reasoning_tokens", None) if reasoning else None
answer_tokens = len(enc.encode(r.choices[0].message.content))
print("completion_tokens:", u.completion_tokens, "| reasoning_tokens:", rt, "| tokens in the visible answer:", answer_tokens)
think = getattr(r.choices[0].message, "reasoning", None)
print("hidden reasoning (first 160 chars):", (think or "")[:160].replace("\n", " "))

print("\n== 3. Cost by hand ==")
cost = u.prompt_tokens / 1e6 * PRICE_IN + u.completion_tokens / 1e6 * PRICE_OUT
print(f"{u.prompt_tokens} x $0.075/M + {u.completion_tokens} x $0.30/M = ${cost:.8f}")
print(f"one million such calls: ${cost * 1e6:,.2f}")
print(f"an agent run of 11,000 tokens (90% input): ${(9900 * PRICE_IN + 1100 * PRICE_OUT) / 1e6:.6f}")

print("\n== 4. Temperature: same question, five times ==")
q = [{"role": "user", "content": "Name one likely cause of a sudden checkout error spike. Answer in at most five words."}]
for t in (0, 1):
    outs = [client.chat.completions.create(model=MODEL, messages=q, temperature=t).choices[0].message.content.strip()
            for _ in range(5)]
    c = Counter(outs)
    print(f"temperature {t}: {len(c)} different answers out of 5")
    for text, k in c.most_common():
        print(f"   {k}x {text}")

print("\n== 5. Reasoning effort ==")
hard = [{"role": "user", "content": "A service has 3 replicas, each handles 250 requests per second. Traffic is 1,400 "
                                    "requests per second. How many replicas are needed? Answer with the number only."}]
for effort in ("low", "medium", "high"):
    r = client.chat.completions.create(model=MODEL, messages=hard, temperature=0, reasoning_effort=effort)
    d = r.usage.completion_tokens_details
    print(f"{effort:6} -> answer {r.choices[0].message.content.strip()!r:6} completion_tokens {r.usage.completion_tokens:4}"
          f"  reasoning_tokens {getattr(d, 'reasoning_tokens', None)}")
