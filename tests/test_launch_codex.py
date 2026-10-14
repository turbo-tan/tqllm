import json, os, sys, threading, types, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "launch"))
import tqllm  # noqa: E402


def args(agent, extra=()):
    return types.SimpleNamespace(agent=agent, agent_args=list(extra), model="/models/m.gguf", context=65536)


class AgentCommands(unittest.TestCase):
    def test_codex_uses_the_responses_api(self):
        cmd = tqllm.agent_command(args("codex", ["exec", "hi"]), "http://127.0.0.1:8086", {})
        self.assertIn("model_providers.tqllm.wire_api=responses", cmd)
        self.assertFalse(any("wire_api=chat" in c for c in cmd))  # rejected by Codex 0.16x
        self.assertIn("model_providers.tqllm.base_url=http://127.0.0.1:8086/v1", cmd)
        self.assertEqual(cmd[-2:], ["exec", "hi"])

    def test_claude_talks_to_the_server_directly(self):
        env = {}
        cmd = tqllm.agent_command(args("claude"), "http://127.0.0.1:8086", env)
        self.assertEqual(cmd, ["claude"])
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8086")


class ApiProbe(unittest.TestCase):
    def test_api_ok(self):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                code = 400 if self.path == "/v1/responses" else 404
                b = json.dumps({"error": {"message": "x"}}).encode()
                self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            self.assertTrue(tqllm.api_ok(base, "/v1/responses"))
            self.assertFalse(tqllm.api_ok(base, "/v1/messages"))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
