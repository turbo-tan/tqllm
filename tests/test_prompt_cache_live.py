"""Opt-in real-server regression for prompt reuse (--prompt-cache on, or --kv-compact; --np 1).

TQSERVE_TEST_URL names the running server. A follow-up turn is answered twice: right after the first turn (the shared
prefix is reused) and after an unrelated request has replaced the cache (full prefill). Greedy answers must agree.
"""
import json
import os
import time
import unittest
import urllib.request

URL = os.environ.get("TQSERVE_TEST_URL", "")
FILLER = "The archive lists shipments by date, port and weight; most entries repeat the same three carriers. " * 40


def ask(msgs, max_tokens=64):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        m = json.load(r)
    return m["choices"][0]["message"]["content"] or "", time.time() - t0


@unittest.skipUnless(URL, "requires a running tqserve with prompt reuse (TQSERVE_TEST_URL)")
class PromptCache(unittest.TestCase):
    def test_follow_up_matches_full_prefill(self):
        m1 = [{"role": "user", "content": FILLER + "\nThe pallet code is 5172.\nQuestion: which three things does the "
                                                   "archive list for each shipment?"}]
        a1, _ = ask(m1)
        m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What is the pallet code?"}]
        warm, t_warm = ask(m2)
        ask([{"role": "user", "content": "Say hello."}])  # replaces the cached sequence
        cold, t_cold = ask(m2)
        self.assertEqual(warm, cold, "reused-prefix answer differs from the full-prefill answer")
        self.assertIn("5172", warm)
        print("follow-up %.2f s with reuse, %.2f s without" % (t_warm, t_cold))


if __name__ == "__main__":
    unittest.main()
