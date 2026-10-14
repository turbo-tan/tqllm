import json, os, re, signal, socket, subprocess, sys, tempfile, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(ROOT, "tools", "launch")


class LaunchCatalog(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(LAUNCH, "models.json")) as f:
            self.cat = json.load(f)

    def test_entries_are_complete(self):
        for name, m in self.cat["models"].items():
            self.assertIn("min_vram_gb", m, name)
            for key in ("file", "drafter", "draft_vocab"):
                e = m[key]
                self.assertTrue(e["repo"] and e["path"] and e["size"] > 0, f"{name}.{key}")
                if "sha256" in e:
                    self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$", f"{name}.{key}")
            self.assertIn("sha256", m["file"], name)

    def test_defaults_name_catalog_models(self):
        for tier, name in self.cat["default"].items():
            self.assertIn(name, self.cat["models"], tier)

    def test_list_runs(self):
        out = subprocess.run([sys.executable, os.path.join(LAUNCH, "tqllm.py"), "--list"], capture_output=True, text=True,
                             timeout=60, env=dict(os.environ, TQLLM_MODELS=os.path.join(ROOT, "nonexistent")))
        self.assertEqual(out.returncode, 0, out.stderr)
        for name in self.cat["models"]:
            self.assertTrue(re.search(r"^" + re.escape(name) + r"\s", out.stdout, re.M), name)


FAKE_SERVER = """import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"status": "ok"} if self.path == "/health" else {"data": [{"id": "tqllm"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


class LaunchServe(unittest.TestCase):
    def test_sigterm_stops_the_server_it_started(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "tqserve"); model = os.path.join(d, "m.gguf")
            with open(fake, "w") as f: f.write("#!" + sys.executable + "\n" + FAKE_SERVER)
            os.chmod(fake, 0o755); open(model, "w").close()
            p = subprocess.Popen([sys.executable, os.path.join(LAUNCH, "tqllm.py"), "serve", "--native", fake, "--model", model,
                                  "--port", str(port), "--log", os.path.join(d, "log"), "--start-timeout", "30"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            line = p.stderr.readline()
            for _ in range(100):
                if "serving" in line or p.poll() is not None: break
                line = p.stderr.readline()
            self.assertIn("serving", line)
            p.send_signal(signal.SIGTERM)
            p.wait(30)
            for _ in range(50):
                try: socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                except OSError: break
                time.sleep(0.2)
            else:
                self.fail("the server is still listening after the launcher got SIGTERM")


class LaunchDrafter(unittest.TestCase):
    """A model given as a file that is a catalog model's file gets that model's drafter; downloaded when missing."""
    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("tqllm_launch", os.path.join(LAUNCH, "tqllm.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

    def _args(self, model, **kw):
        import argparse
        return argparse.Namespace(model=model, drafter=None, draft_vocab=None, no_drafter=False, yes=False, models_dir=None, **kw)

    def test_file_model_gets_catalog_drafter_downloaded(self):
        mod = self._load()
        cat = json.load(open(os.path.join(LAUNCH, "models.json")))
        name = next(k for k, v in cat["models"].items() if "drafter" in v)
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, os.path.basename(cat["models"][name]["file"]["path"])); open(model, "w").close()
            got = []
            def fake_download(entry, dest):
                p = os.path.join(dest, os.path.basename(entry["path"])); open(p, "w").close(); got.append(entry["path"]); return p
            mod.download = fake_download
            os.environ["TQLLM_MODELS"] = os.path.join(d, "dl"); os.makedirs(os.environ["TQLLM_MODELS"])
            mod.model_dirs = lambda: [os.environ["TQLLM_MODELS"]]
            a = self._args(model); mod.resolve_model(a)
            self.assertEqual(a.model, model)
            self.assertTrue(a.drafter and a.drafter.endswith(os.path.basename(cat["models"][name]["drafter"]["path"])))
            self.assertTrue(a.draft_vocab)
            self.assertEqual(len(got), 2)  # drafter + vocab, no prompt (stdin is not a terminal here)
            b = self._args(model); b.no_drafter = True; mod.resolve_model(b)
            self.assertIsNone(b.drafter)

    def test_unknown_file_runs_without_drafter(self):
        mod = self._load()
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "my-own-model.gguf"); open(model, "w").close()
            a = self._args(model); mod.resolve_model(a)
            self.assertEqual(a.model, model); self.assertIsNone(a.drafter)


if __name__ == "__main__":
    unittest.main()
