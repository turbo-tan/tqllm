"""Unit test for tools/capability_probe.py against a mock OpenAI-compatible server (no GPU). Each mode seeds one failure the probe
must catch: rejected `tools`, a repeated-token sampled output (the C2 failure), non-deterministic greedy output, a short generation."""
import json
import importlib.util
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROBE = Path(__file__).resolve().parent.parent / "tools" / "capability_probe.py"
LONG = "A transformer predicts the next token from the previous ones using attention. " * 120


def make_handler(mode):
    counter = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"status": "ok"})

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msg = {"role": "assistant", "content": ""}
            if "tools" in b:
                if mode == "no_tools":
                    return self.reply(400, {"error": "tools not supported"})
                msg["tool_calls"] = [{"id": "1", "type": "function",
                                      "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]
            elif b.get("temperature") == 0:
                counter["n"] += 1
                msg["content"] = f"2, 3, 5, 7, 11. A prime has exactly two divisors. {counter['n'] if mode == 'nondeterministic' else ''}"
            elif b.get("max_tokens", 0) >= 1000:
                msg["content"] = "short" if mode == "short" else LONG
            else:
                msg["content"] = "C " * 100 if mode == "degenerate" else "The capital of France is Paris."
            self.reply(200, {"choices": [{"message": msg, "finish_reason": "stop"}],
                             "usage": {"completion_tokens": len(msg["content"]) // 4}})

    return H


def run_probe(mode, *extra):
    srv = HTTPServer(("127.0.0.1", 0), make_handler(mode))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = subprocess.run([sys.executable, str(PROBE), "--url", f"http://127.0.0.1:{srv.server_address[1]}", *extra],
                           capture_output=True, text=True, timeout=120)
    finally:
        srv.shutdown()
        srv.server_close()
    return p.returncode, json.loads(p.stdout.strip().splitlines()[-1])


class CapabilityProbe(unittest.TestCase):
    def test_good_server_passes(self):
        rc, s = run_probe("good", "--concurrency", "3")
        self.assertEqual((rc, s["failed"]), (0, []))
        self.assertEqual(s["passed"], 6)

    def test_rejected_tools_is_caught(self):
        rc, s = run_probe("no_tools")
        self.assertEqual((rc, s["failed"]), (1, ["tools"]))

    def test_repeated_token_sampling_is_caught(self):
        rc, s = run_probe("degenerate")
        self.assertEqual((rc, s["failed"]), (1, ["sampling"]))

    def test_nondeterministic_greedy_is_caught(self):
        rc, s = run_probe("nondeterministic")
        self.assertIn("determinism", s["failed"])

    def test_concurrent_nondeterminism_is_caught(self):
        rc, s = run_probe("nondeterministic", "--concurrency", "3", "--skip", "determinism")
        self.assertEqual(s["failed"], ["concurrent"])

    def test_short_generation_is_caught(self):
        rc, s = run_probe("short")
        self.assertEqual((rc, s["failed"]), (1, ["long"]))


class LongResponseContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("probe_contract", PROBE)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def response(self, count=4479, finish="stop", text=LONG):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"completion_tokens": count}}

    def check(self, response):
        return self.probe.validate_long(response, 12000, 6000, 3000)[0]

    def test_sustained_eos_passes_without_claiming_full_budget(self):
        ok, detail = self.probe.validate_long(self.response(), 12000, 6000, 3000)
        self.assertTrue(ok)
        self.assertIn("full budget not exercised", detail)

    def test_length_must_reach_request_not_server_cap(self):
        self.assertFalse(self.check(self.response(3000, "length")))
        self.assertTrue(self.check(self.response(12000, "length")))

    def test_short_eos_is_not_sustained_generation(self):
        self.assertFalse(self.check(self.response(2999)))
        self.assertFalse(self.check(self.response(text="Too short.")))

    def test_missing_or_invalid_count_is_rejected(self):
        for value in (None, True, "4479", -1, 12001):
            with self.subTest(value=value):
                self.assertFalse(self.check(self.response(value)))

    def test_error_finish_and_degenerate_text_are_rejected(self):
        self.assertFalse(self.check(self.response(finish="error")))
        self.assertFalse(self.check(self.response(text="loop " * 2000)))


if __name__ == "__main__":
    unittest.main()
