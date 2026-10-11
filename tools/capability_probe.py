#!/usr/bin/env python3
"""Capability probe for an OpenAI-compatible TQLLM server (stdlib only).

A serving config has capabilities as well as speed. Throughput benchmarks do not send a `tools` field, a long generation or a
sampled request, so they miss rejected tool calls, truncated reasoning and broken sampling. Run this against every config you
publish or benchmark; it exits 1 if any check fails.

  python3 tools/capability_probe.py --url http://127.0.0.1:8086            # quick
  python3 tools/capability_probe.py --url http://127.0.0.1:8086 --full --concurrency 4

Checks: health, tool call, sampled coherence (T=0.6), sequential greedy determinism, long generation, optional concurrent greedy
determinism (identical requests in parallel must give identical text).
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NO_THINK = {"enable_thinking": False}
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def http(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except Exception:
            return e.code, None


def chat(base, messages, timeout=600, **kw):
    body = {"model": "x", "messages": messages, "chat_template_kwargs": NO_THINK, **kw}
    return http(base.rstrip("/") + "/v1/chat/completions", body, timeout)


def text_of(resp):
    m = resp["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or "")


def degenerate(t):
    """True for empty text, a handful of distinct characters, or a long run of one character or one short phrase."""
    t = t.strip()
    if len(t) < 10 or len(set(t[:200])) < 8:
        return True
    if re.search(r"(.)\1{29,}", t):
        return True
    return re.search(r"(.{3,64}?)\1{7,}", t) is not None


def check_health(base, a):
    code, body = http(base.rstrip("/") + "/health", timeout=30)
    ok = code == 200 and "ok" in json.dumps(body)
    return ok, f"/health -> {code} {json.dumps(body)[:60]}"


def check_tools(base, a):
    code, r = chat(base, [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                   tools=TOOLS, temperature=0, max_tokens=512)
    if code != 200 or not r:
        return False, f"HTTP {code}: a request with `tools` was rejected"
    calls = r["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return False, "no tool call returned"
    fn = calls[0]["function"]
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except Exception:
        return False, f"tool arguments are not JSON: {str(fn.get('arguments'))[:60]}"
    return fn["name"] == "get_weather" and "city" in args, f"{fn['name']}({json.dumps(args)})"


def check_sampling(base, a):
    code, r = chat(base, [{"role": "user", "content": "In one short sentence, what is the capital of France?"}],
                   temperature=0.6, top_p=0.95, top_k=20, max_tokens=300)
    if code != 200 or not r:
        return False, f"HTTP {code}"
    t = text_of(r)
    if degenerate(t):
        return False, f"degenerate sampled output: {t[:80]!r}"
    return "paris" in t.lower(), t.strip()[:80]


def greedy_text(base, prompt="List the first five prime numbers, then explain what a prime is.", max_tokens=200):
    code, r = chat(base, [{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
    return text_of(r) if code == 200 and r else f"HTTP {code}"


def check_determinism(base, a):
    texts = [greedy_text(base) for _ in range(2)]
    shas = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in texts]
    ok = texts[0] == texts[1] and not texts[0].startswith("HTTP ")
    return ok, f"greedy sha {shas[0]} / {shas[1]}"


def check_concurrent(base, a):
    n = a.concurrency
    with ThreadPoolExecutor(max_workers=n) as ex:
        outs = list(ex.map(lambda _: greedy_text(base), range(n)))
    distinct = len(set(outs))
    return distinct == 1, f"{n} concurrent identical greedy requests -> {distinct} distinct output(s)"


def validate_long(response, requested_tokens, min_chars, min_tokens):
    text = text_of(response)
    count = response.get("usage", {}).get("completion_tokens")
    finish = response["choices"][0].get("finish_reason")
    valid_count = type(count) is int and min_tokens <= count <= requested_tokens
    valid_finish = finish == "stop" or (finish == "length" and count == requested_tokens)
    ok = valid_count and valid_finish and len(text) >= min_chars and not degenerate(text)
    coverage = "budget reached" if finish == "length" and count == requested_tokens else "natural stop; full budget not exercised"
    return ok, (f"{len(text)} chars (need {min_chars}), {count} tokens (need {min_tokens}, budget {requested_tokens}), "
                f"finish={finish}, {coverage}")


def check_long(base, a):
    toks, chars = (12000 if a.full else a.long_tokens), a.long_chars
    min_tokens = 3000 if a.full else min(1000, toks)
    t0 = time.time()
    code, r = chat(base, [{"role": "user", "content": "Write a long, detailed essay on how a transformer language model "
                           "generates text, with many sections and examples. Do not stop early."}],
                   temperature=0.6, top_p=0.95, top_k=20, max_tokens=toks, timeout=3600)
    if code != 200 or not r:
        return False, f"HTTP {code}"
    ok, detail = validate_long(r, toks, chars, min_tokens)
    return ok, f"{detail}, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8086")
    ap.add_argument("--full", action="store_true", help="12000-token allowance; require >=3000 sustained tokens, >=long-chars, valid EOS or exact-budget length finish")
    ap.add_argument("--long-tokens", type=int, default=3000)
    ap.add_argument("--long-chars", type=int, default=6000)
    ap.add_argument("--concurrency", type=int, default=1, help=">1 adds the concurrent greedy determinism check")
    ap.add_argument("--skip", default="", help="comma list of: health,tools,sampling,determinism,long,concurrent")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    checks = [("health", check_health), ("tools", check_tools), ("sampling", check_sampling),
              ("determinism", check_determinism), ("long", check_long)]
    if a.concurrency > 1:
        checks.append(("concurrent", check_concurrent))
    results = []
    for name, fn in checks:
        if name in skip:
            continue
        try:
            ok, detail = fn(a.url, a)
        except Exception as e:  # a crashed check is a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name:12s} {detail}", flush=True)
    failed = [r["check"] for r in results if not r["pass"]]
    print(json.dumps({"url": a.url, "passed": len(results) - len(failed), "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
