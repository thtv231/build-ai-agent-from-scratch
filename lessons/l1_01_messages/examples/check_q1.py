"""Check-yourself Q1: the one-word rule lives only in call 1's list. Does call 2 still obey it?"""
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=6)
M = "openai/gpt-oss-20b"
rule = {"role": "system", "content": "Answer with exactly one word: CRITICAL, HIGH or LOW."}
q1 = {"role": "user", "content": "Alert: checkout error rate is 23%. How bad is it?"}
q2 = {"role": "user", "content": "Alert: search latency is 900 ms. How bad is it?"}
r1 = client.chat.completions.create(model=M, messages=[rule, q1], temperature=0).choices[0].message.content
print("== 6. Check-yourself: does the rule carry over? ==")
print("call 1 (rule + question):", r1)
r2 = client.chat.completions.create(model=M, messages=[q2], temperature=0).choices[0].message.content
print("call 2 (new question only):", r2)
print("words in call 2 reply:", len(r2.split()))
