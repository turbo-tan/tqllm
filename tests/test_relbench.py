"""Exercise the release runner's local server lifecycle without a GPU."""
import io
import json
import os
from pathlib import Path
import runpy
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "relbench.py"


class ReleaseBench(unittest.TestCase):
    def test_starts_and_stops_local_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            wiki, output = Path(tmp) / "wiki.txt", Path(tmp) / "results.jsonl"
            wiki.write_text(" " * 1000 + "café", encoding="utf-8")
            argv = [str(SCRIPT), "--bin", "tqserve", "--gpu", "test", "--wiki", str(wiki),
                    "--model", "xs=xs.gguf", "--out", str(output), "--lengths", "2000", "--repeats", "1"]
            server = Mock(pid=123)
            server.poll.return_value = None
            responses = [io.StringIO('{"engine": "test"}'),
                         io.StringIO('{"timings": {"prompt_n": 16, "prompt_per_second": 100, "predicted_per_second": 50}}')]
            with (
                patch("sys.argv", argv), patch("tempfile.gettempdir", return_value=tmp),
                patch("subprocess.Popen", return_value=server) as spawn,
                patch("subprocess.run", return_value=Mock(stdout="test GPU\n")),
                patch("urllib.request.urlopen", side_effect=responses) as request,
                patch("atexit.register"), patch("time.sleep"), patch("os.killpg", create=True) as killpg,
            ):
                runpy.run_path(str(SCRIPT))["out"].close()
            self.assertTrue((Path(tmp) / "relbench_xs_nospec.log").is_file())
            self.assertEqual(json.loads(output.read_text())["decode_tps"], 50)
            self.assertIn("café", json.loads(request.call_args.args[0].data)["messages"][0]["content"])
            self.assertEqual(spawn.call_args.kwargs["start_new_session"], os.name != "nt")
            server.wait.assert_called_once_with(timeout=60)
            if os.name == "nt":
                server.terminate.assert_called_once_with()
                killpg.assert_not_called()
            else:
                killpg.assert_called_once_with(server.pid, signal.SIGTERM)
                server.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
