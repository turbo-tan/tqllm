import json
import io
"""The `tqllm` command (python/tqllm/cli.py) without a server binary or network: defaults, file choice, agent env."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tqllm import cli  # noqa: E402


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["TQLLM_HOME"] = self.tmp.name
        self.asked = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, files):
        self.asked += files

    @staticmethod
    def published(repo, f):
        return f != cli.TQ_L_QSA

    def test_large_mac_gets_mlx_model_and_drafter(self):
        a = cli.serve_args(["--port", "8080"], ram=48, download=self.fake, exists=self.published)
        self.assertEqual(self.asked, [(cli.MLX_REPO, cli.MODEL), (cli.MLX_REPO, cli.DRAFTER)])
        self.assertTrue(a[a.index("--model") + 1].endswith(cli.MODEL))
        self.assertTrue(a[a.index("-md") + 1].endswith(cli.DRAFTER))
        self.assertIn("--port", a)
        self.assertEqual(a[a.index("--context") + 1], "65536")

    def test_small_mac_gets_tq_l(self):
        cli.serve_args([], ram=32, download=self.fake, exists=self.published)
        self.assertEqual(self.asked[0], (cli.TQ_REPO, cli.TQ_L))

    def test_qsa_flag_picks_qsa_file_and_is_not_passed_on(self):
        a = cli.serve_args(["--qsa"], ram=64, download=self.fake, exists=self.published)
        self.assertEqual(self.asked[0], (cli.MLX_REPO, cli.MODEL_QSA))
        self.assertNotIn("--qsa", a)

    def test_qsa_from_is_passed_on(self):
        a = cli.serve_args(["--qsa-from", "off"], ram=64, download=self.fake, exists=self.published)
        self.assertEqual(a[a.index("--qsa-from") + 1], "off")

    def test_given_model_and_context_are_kept(self):
        a = cli.serve_args(["--model", "/x.gguf", "--context", "8192"], ram=48, download=self.fake, exists=self.published)
        self.assertEqual(self.asked, [(cli.MLX_REPO, cli.DRAFTER)])
        self.assertEqual(a.count("--model"), 1)
        self.assertEqual(a[a.index("--context") + 1], "8192")

    def test_present_files_are_not_downloaded(self):
        d = cli.models_dir(); d.mkdir(parents=True)
        for f in (cli.MODEL, cli.DRAFTER): (d / f).write_bytes(b"x")
        cli.fetch([(cli.MLX_REPO, cli.MODEL), (cli.MLX_REPO, cli.DRAFTER)])  # no network, no import needed

    def test_claude_gets_the_served_model(self):
        # a model name left in the shell (e.g. ANTHROPIC_MODEL=qwen3.5-4b) must not reach tqLLM's server
        e = cli.agent_env("claude", 9000, env={"ANTHROPIC_MODEL": "qwen3.5-4b-instruct"}, model="tqllm")
        for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            self.assertEqual(e[k], "tqllm")
        # ~/.claude/settings.json's env block outranks the process env: tqllm passes its values with --settings
        args = cli.agent_args("claude", ["-p", "hi"], "tqllm", 8086)
        self.assertEqual(args[0], "--settings")
        st = json.loads(args[1])
        self.assertEqual(st["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8086")
        self.assertEqual(st["env"]["ANTHROPIC_MODEL"], "tqllm")
        self.assertEqual(st["model"], "tqllm")
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", st["env"])
        st = json.loads(cli.agent_args("claude", [], "tqllm", 8086, 65536)[1])
        self.assertEqual(st["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "65536")
        self.assertEqual(args[2:], ["--model", "tqllm", "-p", "hi"])
        self.assertEqual(cli.agent_args("claude", ["--model", "x", "--settings", "s.json"], "tqllm", 8086), ["--model", "x", "--settings", "s.json"])

    def test_foreign_server_on_the_port(self):
        self.assertEqual(cli.tqllm_model(8086, "claude", served=lambda p: ("tqllm", "tqllm")), "tqllm")
        with self.assertRaises(SystemExit) as c:
            cli.tqllm_model(8086, "claude", served=lambda p: ("qwen3.5-4b-instruct", "llamacpp"))
        self.assertIn("not tqLLM's server", str(c.exception))
        self.assertIn("qwen3.5-4b-instruct", str(c.exception))
        with self.assertRaises(SystemExit):
            cli.tqllm_model(8086, "claude", served=lambda p: None)

    def test_agent_env(self):
        e = cli.agent_env("claude", 9000, env={})
        self.assertEqual(e["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9000")
        e = cli.agent_env("codex", 9000, env={})
        self.assertEqual(e["OPENAI_BASE_URL"], "http://127.0.0.1:9000/v1")
        with self.assertRaises(SystemExit):
            cli.agent_env("vim", 9000, env={})

    def test_unknown_command(self):
        with self.assertRaises(SystemExit):
            cli.main(["frobnicate"])

    def test_no_command_serves_in_the_background(self):  # owner 11-09: plain `tqllm` serves, detached; --foreground stays
        started, execs = [], []
        orig = cli.ensure_server, cli.server_command, cli.os.execvp
        cli.ensure_server = lambda port, start: started.append(port) or True
        cli.server_command = lambda rest, mac, native: execs.append(list(rest)) or ["true"]
        class Exec(Exception): pass
        def execvp(f, a): raise Exec()  # the real one never returns
        cli.os.execvp = execvp
        try:
            self.assertEqual(cli.main([]), 0)
            self.assertEqual(cli.main(["--qsa", "--port", "9001"]), 0)
            with self.assertRaises(Exec): cli.main(["serve", "--foreground", "--qsa"])
        finally:
            cli.ensure_server, cli.server_command, cli.os.execvp = orig
        self.assertEqual(started, [cli.PORT, 9001])
        self.assertEqual(execs, [["--qsa"]])  # only the foreground run built its line here; background ones build it in start
        with self.assertRaises(SystemExit):
            cli.main(["frobnicate"])


    # Linux / NVIDIA (11-06): the model by the card's memory, Flash-Next on request, Docker or a native server
    def test_linux_24gb_gets_tq3_4s_drafter_and_vocab(self):
        c = cli.linux_command([], vram=24, download=self.fake)
        self.assertEqual(self.asked, [(cli.TQ34S_REPO, cli.TQ34S), (cli.TQ_REPO, cli.TQ_DRAFTER), (cli.TQ_REPO, cli.TQ_VOCAB)])
        self.assertEqual(c[:2], ["docker", "run"])
        self.assertIn(cli.IMAGE, c)
        self.assertEqual(c[c.index("--model") + 1], "/models/" + cli.TQ34S)
        self.assertEqual(c[c.index("-md") + 1], "/models/" + cli.TQ_DRAFTER)
        self.assertEqual(c[c.index("--context") + 1], "65536")

    def test_linux_16gb_gets_tq_s(self):
        cli.linux_command([], vram=16, download=self.fake)
        self.assertEqual(self.asked[0], (cli.TQ_REPO, cli.TQ_S))

    def test_linux_flash_next_downloads_its_files_and_serves_the_manifest(self):
        c = cli.linux_command(["--flash-next"], vram=24, download=self.fake)
        self.assertEqual([f for _, f in self.asked], cli.FN_FILES)
        self.assertEqual(c[c.index("--model") + 1], "/models/flashnext-tqllm.tqfn")
        self.assertEqual(c[c.index("--tokenizer") + 1], "/models/flashnext-tokenizer.gguf")
        self.assertNotIn("--flash-next", c)
        self.assertEqual(c[c.index("--context") + 1], "32768")

    def test_linux_native_uses_absolute_paths(self):
        c = cli.linux_command(["--port", "9000"], vram=24, native="/opt/tqserve", download=self.fake)
        self.assertEqual(c[0], "/opt/tqserve")
        self.assertTrue(c[c.index("--model") + 1].startswith(str(cli.models_dir())))
        self.assertEqual(c[c.index("--port") + 1], "9000")

    def test_linux_own_model_downloads_nothing(self):
        c = cli.linux_command(["--model", "/models/x.gguf"], vram=24, download=self.fake)
        self.assertEqual(self.asked, [])
        self.assertEqual(c.count("--model"), 1)

    def test_opencode_gets_a_generated_config_and_the_tqllm_model(self):
        e = cli.agent_env("opencode", 8099, env={})
        cfg = json.loads(Path(e["OPENCODE_CONFIG"]).read_text())
        self.assertEqual(cfg["provider"]["tqllm"]["options"]["baseURL"], "http://127.0.0.1:8099/v1")
        self.assertEqual(cli.agent_args("opencode", ["run", "hi"]), ["-m", cli.OPENCODE_MODEL, "run", "hi"])
        self.assertEqual(cli.agent_args("opencode", ["-m", "x/y"]), ["-m", "x/y"])

    def test_background_start_passes_the_options(self):
        seen = []
        orig = cli.server_command, cli.subprocess.Popen
        cli.server_command = lambda rest, mac, native: seen.append(list(rest)) or ["true"]
        cli.subprocess.Popen = lambda *a, **k: type("P", (), {"pid": 1})()
        try:
            cli.start_background(9001, True, None, ["--qsa", "--context", "131072"])
        finally:
            cli.server_command, cli.subprocess.Popen = orig
        self.assertEqual(seen, [["--qsa", "--context", "131072", "--port", "9001"]])

    def test_running_server_is_reused(self):
        started = []
        self.assertFalse(cli.ensure_server(8086, started.append, is_up=lambda p: True, out=io.StringIO()))
        self.assertEqual(started, [])

    def test_missing_server_is_started_and_awaited(self):
        state = {"calls": 0}
        def up(p):
            state["calls"] += 1
            return state["calls"] > 2
        started = []
        self.assertTrue(cli.ensure_server(8090, lambda p: started.append(p), is_up=up, poll=0.0, out=io.StringIO()))
        self.assertEqual(started, [8090])

    def test_server_that_exits_is_reported(self):
        class Dead:
            returncode = 3
            def poll(self): return 3
        with self.assertRaises(SystemExit):
            cli.ensure_server(8091, lambda p: Dead(), is_up=lambda p: False, poll=0.0, out=io.StringIO())

    def test_stop_without_a_started_server(self):
        self.assertEqual(cli.stop_server(), 1)

    def test_tq_l_qsa_preferred_once_published(self):
        cli.serve_args([], ram=48, download=self.fake, exists=lambda r, f: True)
        self.assertEqual(self.asked[0], (cli.TQ_REPO, cli.TQ_L_QSA))

    def test_downloaded_file_wins_without_network(self):
        d = cli.models_dir(); d.mkdir(parents=True, exist_ok=True); (d / cli.TQ_L).write_bytes(b"x")
        self.assertEqual(cli.pick(cli.model_candidates(ram=48), exists=lambda r, f: False), (cli.TQ_REPO, cli.TQ_L))

    def test_nothing_reachable_is_reported(self):
        with self.assertRaises(SystemExit):
            cli.pick(cli.model_candidates(ram=48), exists=lambda r, f: False)

    def test_hub_model_with_file(self):
        a = cli.serve_args(["--model", "someone/Some-27B-GGUF/some-Q4.gguf"], ram=48, download=self.fake, exists=self.published)
        self.assertIn(("someone/Some-27B-GGUF", "some-Q4.gguf"), self.asked)
        self.assertTrue(a[a.index("--model") + 1].endswith("some-Q4.gguf"))

    def test_hub_repo_with_one_model_file(self):
        p = cli.hub_model("someone/Repo", self.fake, list_files=lambda r: ["README.md", "m.gguf", "m-draft.gguf"])
        self.assertTrue(p.endswith("m.gguf"))
        with self.assertRaises(SystemExit):
            cli.hub_model("someone/Repo", self.fake, list_files=lambda r: ["a.gguf", "b.gguf"])

if __name__ == "__main__":
    unittest.main()
