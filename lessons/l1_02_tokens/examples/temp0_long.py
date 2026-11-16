"""Section 6: temperature 0 with a longer answer - is it really the same every time?"""
import os
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
msgs = [{"role": "system", "content": "You are Nightwatch, the on-call assistant for the shop Brightcart. Answer in one sentence."},
        {"role": "user", "content": "Alert: checkout error rate is 23%. What does that mean?"}]
outs = [client.chat.completions.create(model="openai/gpt-oss-20b", messages=msgs, temperature=0) for _ in range(8)]
c = Counter(o.choices[0].message.content.strip() for o in outs)
print("\n== 6. Temperature 0, one-sentence answer, eight times ==")
print(f"{len(c)} different answers out of 8; completion tokens: {[o.usage.completion_tokens for o in outs]}")
for text, k in c.most_common():
    print(f"   {k}x {text[:110]}")
