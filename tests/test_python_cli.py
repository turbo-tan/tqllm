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

    def test_large_mac_gets_mlx_model_and_drafter(self):
        a = cli.serve_args(["--port", "8080"], ram=48, download=self.fake)
        self.assertEqual(self.asked, [(cli.MLX_REPO, cli.MODEL), (cli.MLX_REPO, cli.DRAFTER)])
        self.assertTrue(a[a.index("--model") + 1].endswith(cli.MODEL))
        self.assertTrue(a[a.index("-md") + 1].endswith(cli.DRAFTER))
        self.assertIn("--port", a)
        self.assertEqual(a[a.index("--context") + 1], "65536")

    def test_small_mac_gets_tq_l(self):
        cli.serve_args([], ram=32, download=self.fake)
        self.assertEqual(self.asked[0], (cli.TQ_REPO, cli.TQ_L))

    def test_qsa_flag_picks_qsa_file_and_is_not_passed_on(self):
        a = cli.serve_args(["--qsa"], ram=64, download=self.fake)
        self.assertEqual(self.asked[0], (cli.MLX_REPO, cli.MODEL_QSA))
        self.assertNotIn("--qsa", a)

    def test_given_model_and_context_are_kept(self):
        a = cli.serve_args(["--model", "/x.gguf", "--context", "8192"], ram=48, download=self.fake)
        self.assertEqual(self.asked, [(cli.MLX_REPO, cli.DRAFTER)])
        self.assertEqual(a.count("--model"), 1)
        self.assertEqual(a[a.index("--context") + 1], "8192")

    def test_present_files_are_not_downloaded(self):
        d = cli.models_dir(); d.mkdir(parents=True)
        for f in (cli.MODEL, cli.DRAFTER): (d / f).write_bytes(b"x")
        cli.fetch([(cli.MLX_REPO, cli.MODEL), (cli.MLX_REPO, cli.DRAFTER)])  # no network, no import needed

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


if __name__ == "__main__":
    unittest.main()
