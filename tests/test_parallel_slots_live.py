"""Opt-in real-server regression for concurrent slots (--np N, N >= 3).

TQSERVE_TEST_URL names a running server, TQSERVE_TEST_SLOTS its --np. More requests than slots are sent at once (short
and long prompts mixed, greedy); each answer must complete and equal the same prompt run alone, and two concurrent rounds
must agree. Run once with all KV on the device and once with --kv-host N (the host tail) to cover both stores.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import unittest
import urllib.request

URL = os.environ.get("TQSERVE_TEST_URL", "")
SLOTS = int(os.environ.get("TQSERVE_TEST_SLOTS", "3"))
# TQSERVE_TEST_LONG: repeats of the filler sentence in the long prompt (~25 tokens each); raise it so the long prompt
# reaches past the device part of a --kv-host slot
FILLER = ("The archive lists shipments by date, port and weight; most entries repeat the same three carriers. " *
          int(os.environ.get("TQSERVE_TEST_LONG", "120")))
PROMPTS = [
    "Name the capital of France and one river that runs through it.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List three prime numbers between 50 and 80 and say why each is prime.",
    FILLER + "\nQuestion: which three things does the archive list for each shipment?",
    "Explain in two sentences what a hash map is.",
    "Translate 'good morning, how are you?' into Spanish and French.",
]


def ask(prompt, max_tokens=96):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        m = json.load(r)
    choice = m["choices"][0]
    text = choice["message"]["content"] or ""
    return hashlib.sha256(text.encode()).hexdigest()[:16], choice.get("finish_reason"), m["usage"]["completion_tokens"]


@unittest.skipUnless(URL, "requires a running tqserve (TQSERVE_TEST_URL, TQSERVE_TEST_SLOTS)")
class ParallelSlots(unittest.TestCase):
    def test_more_requests_than_slots_match_serial(self):
        self.assertGreaterEqual(SLOTS, 3, "this test is for --np 3 and above")
        serial = [ask(p) for p in PROMPTS]
        for sha, finish, n in serial:
            self.assertIn(finish, ("stop", "length"))
            self.assertGreater(n, 0)
        rounds = []
        for _ in range(2):
            with cf.ThreadPoolExecutor(max_workers=SLOTS + 2) as ex:  # more in flight than slots: queueing too
                rounds.append(list(ex.map(ask, PROMPTS)))
        for i, prompt in enumerate(PROMPTS):
            with self.subTest(prompt=i, long=len(prompt) > 1000):
                self.assertEqual(rounds[0][i][0], rounds[1][i][0], "two concurrent rounds disagree")
                self.assertEqual(rounds[0][i][0], serial[i][0], "concurrent answer differs from the serial one")


if __name__ == "__main__":
    unittest.main()
