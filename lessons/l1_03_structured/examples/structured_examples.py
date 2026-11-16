"""Lesson 1.3 - Structured output: JSON schema, Pydantic, validate and retry.

Every output shown in the video comes from running this file:
    uv run python lessons/l1_03_structured/examples/structured_examples.py > lessons/l1_03_structured/examples/results.txt
"""
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=8)
MODEL = "openai/gpt-oss-20b"
ALERT = ("Alert: checkout error rate is 23%. Logs: 'upstream payments returned 500 for POST /payments/v2/charge'. "
         "Deploy at 02:12: payments v1.3.5.")


class Triage(BaseModel):
    service: str = Field(description="the service that is most likely broken")
    severity: Literal["critical", "high", "low"]
    summary: str = Field(max_length=120)


print("== 1. Asking nicely for JSON ==")
ask = [{"role": "system", "content": "Reply with JSON with keys service, severity, summary."},
       {"role": "user", "content": ALERT}]
r = client.chat.completions.create(model=MODEL, messages=ask, temperature=0)
text = r.choices[0].message.content
print("raw reply:", text[:300])
try:
    data = json.loads(text)
    print("json.loads: ok ->", data)
except json.JSONDecodeError as e:
    print("json.loads failed:", e)

print("\n== 2. Validate with Pydantic ==")
samples = {
    "good": '{"service": "payments", "severity": "high", "summary": "payments v1.3.5 breaks checkout"}',
    "wrong severity": '{"service": "payments", "severity": "SEV1", "summary": "payments is down"}',
    "missing key": '{"service": "payments", "summary": "payments is down"}',
    "markdown fence": '```json\n{"service": "payments", "severity": "high", "summary": "x"}\n```',
}
for name, s in samples.items():
    try:
        t = Triage.model_validate_json(s)
        print(f"{name:15} -> OK {t}")
    except ValidationError as e:
        err = e.errors()[0]
        print(f"{name:15} -> {err['type']}: {err['loc']} {err['msg'][:70]}")

print("\n== 3. The schema the model will get ==")
schema = Triage.model_json_schema()
print(json.dumps(schema, indent=1)[:700])

print("\n== 4. JSON schema mode (strict) ==")
def strict_call(schema):
    return client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": "Triage the alert."}, {"role": "user", "content": ALERT}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "triage", "strict": True, "schema": schema}})


try:
    strict_call(schema)
except BadRequestError as e:
    print("try 1, plain schema -> 400:", str(e).split(" - ", 1)[-1][:150])


class StrictTriage(Triage):
    model_config = ConfigDict(extra="forbid")      # -> "additionalProperties": false


schema2 = StrictTriage.model_json_schema()
print("additionalProperties now:", schema2.get("additionalProperties"))
try:
    raw = strict_call(schema2).choices[0].message.content
    print("try 2 raw:", raw)
    print("validated:", StrictTriage.model_validate_json(raw))
except BadRequestError as e:
    print("try 2 -> 400:", str(e).split(" - ", 1)[-1][:200])
except ValidationError as e:
    print("server accepted, Pydantic rejected:", e.errors()[0]["msg"])

print("\n== 5. Validate, and on failure send the error back ==")


def triage(alert, attempts=3, force_bad_first=True):
    messages = [{"role": "system", "content": "Reply with JSON only: service, severity (critical|high|low), summary."},
                {"role": "user", "content": alert}]
    for i in range(1, attempts + 1):
        if force_bad_first and i == 1:      # make attempt 1 fail on purpose, so we can watch the repair
            reply = '{"service": "payments", "severity": "SEV1", "summary": "payments is down"}'
        else:
            reply = client.chat.completions.create(model=MODEL, messages=messages, temperature=0).choices[0].message.content
        try:
            t = Triage.model_validate_json(reply)
            print(f"attempt {i}: valid -> {t}")
            return t, i
        except ValidationError as e:
            err = e.errors()[0]
            msg = f"Your JSON was invalid: {err['loc']} {err['msg']}. Reply again with JSON only."
            print(f"attempt {i}: invalid -> {err['msg'][:70]}")
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": msg}]
    raise RuntimeError("no valid answer")


result, tries = triage(ALERT)
print("calls needed:", tries)
