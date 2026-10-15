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


class LaunchPiAndDsh(unittest.TestCase):
    """pi and dsh get their provider through a temp dir/file the launcher creates and deletes."""
    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("tqllm_launch_pd", os.path.join(LAUNCH, "tqllm.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

    def _args(self, agent, extra=()):
        import types
        return types.SimpleNamespace(agent=agent, agent_args=list(extra), model="/models/m.gguf",
                                     context=65536, max_tokens=32768)

    def test_pi_command_and_models_json(self):
        mod = self._load()
        a = self._args("pi"); env = {}
        cmd = mod.agent_command(a, "http://127.0.0.1:8086", env)
        self.assertEqual(cmd, ["pi", "--provider", "tqllm", "--model", "tqllm"])
        d = env["PI_CODING_AGENT_DIR"]
        try:
            cfg = json.load(open(os.path.join(d, "models.json")))
            p = cfg["providers"]["tqllm"]
            self.assertEqual(p["baseUrl"], "http://127.0.0.1:8086/v1")
            self.assertEqual(p["models"][0]["id"], "tqllm")
            self.assertEqual(p["models"][0]["contextWindow"], 65536)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_dsh_web_command_and_patch(self):
        mod = self._load()
        a = self._args("dsh"); env = {}
        cmd = mod.agent_command(a, "http://127.0.0.1:8086", env)
        f = a.temp
        try:
            self.assertEqual(cmd, ["dsh", "web", "--patch", f])
            self.assertEqual(env["TQLLM_API_KEY"], "tqllm-local")
            text = open(f).read()
            self.assertIn("baseURL: http://127.0.0.1:8086/v1", text)
            self.assertIn("agent-default-model", text)
            self.assertIn("provider: tqllm", text)
        finally:
            if os.path.exists(f): os.remove(f)

    def test_dsh_headless_profile_is_consumed(self):
        mod = self._load()
        a = self._args("dsh", ["headless", "fix it"]); env = {}
        cmd = mod.agent_command(a, "http://127.0.0.1:8086", env)
        f = a.temp
        try:
            self.assertEqual(cmd, ["dsh", "headless", "--patch", f, "fix it"])
            self.assertEqual(env["TQLLM_API_KEY"], "tqllm-local")
        finally:
            if os.path.exists(f): os.remove(f)


if __name__ == "__main__":
    unittest.main()


# argv to a file next to the binary; a Ctrl+C takes 3 s to shut down (a real tqserve frees GPU memory first)
FAKE_SLOW_SERVER = """import json, os, signal, sys, time
open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "argv.json"), "w").write(json.dumps(sys.argv[1:]))
signal.signal(signal.SIGINT, lambda *a: (time.sleep(3), os._exit(0)))
""" + FAKE_SERVER


class LaunchFirstRun(unittest.TestCase):
    """What a first `tqllm launch` on a machine with existing model folders and a native build runs into."""
    def _load(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("tqllm_launch_fr", os.path.join(LAUNCH, "tqllm.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

    def _serve(self, d, extra=()):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        fake = os.path.join(d, "tqserve"); model = os.path.join(d, "m.gguf")
        with open(fake, "w") as f: f.write("#!" + sys.executable + "\n" + FAKE_SLOW_SERVER)
        os.chmod(fake, 0o755); open(model, "w").close()
        p = subprocess.Popen([sys.executable, os.path.join(LAUNCH, "tqllm.py"), "serve", "--native", fake, "--model", model,
                              "--port", str(port), "--log", os.path.join(d, "log"), "--start-timeout", "30", *extra],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        err = ""
        for _ in range(100):
            line = p.stderr.readline(); err += line
            if "serving" in line or p.poll() is not None: break
        self.assertIn("serving", err)
        return p, port

    def test_drafter_found_one_folder_down(self):
        mod = self._load()
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "dflash2"))
            want = os.path.join(d, "dflash2", "Qwen3.8-27B-DFlash2-TQ3.gguf"); open(want, "w").close()
            mod.model_dirs = lambda: [d]
            self.assertEqual(mod.find_local("drafters/Qwen3.8-27B-DFlash2-TQ3.gguf"), want)
            self.assertIsNone(mod.find_local("drafters/missing.gguf"))

    def test_agents_get_sampling_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self._serve(d)
            p.send_signal(signal.SIGTERM); p.wait(30)
            argv = json.load(open(os.path.join(d, "argv.json")))
            for flag, value in (("--temperature", "0.6"), ("--top-p", "0.95"), ("--top-k", "20"), ("--max-tokens", "32768")):
                self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        with tempfile.TemporaryDirectory() as d:
            p, _ = self._serve(d, ("--temperature", "0"))
            p.send_signal(signal.SIGTERM); p.wait(30)
            argv = json.load(open(os.path.join(d, "argv.json")))
            self.assertEqual(argv[argv.index("--temperature") + 1], "0.0")

    def test_second_interrupt_during_shutdown_is_quiet(self):
        with tempfile.TemporaryDirectory() as d:
            p, port = self._serve(d)
            p.send_signal(signal.SIGINT); time.sleep(0.5); p.send_signal(signal.SIGINT)
            _, err = p.communicate(timeout=60)
            self.assertNotIn("Traceback", err)
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
