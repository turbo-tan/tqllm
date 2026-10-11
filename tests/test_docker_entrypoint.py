import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ENTRY = Path(__file__).resolve().parents[1] / "docker/entrypoint.sh"


class DockerEntrypointTests(unittest.TestCase):
    def launch(self, options, arguments=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model with spaces.gguf"
            tokenizer = root / "tokenizer.tqtok"
            model.touch()
            tokenizer.touch()
            server = root / "tqserve"
            server.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            server.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("TQLLM_")}
            env.update(PATH=str(root) + os.pathsep + env["PATH"],
                       MODEL_FILE=str(model), TOKENIZER_FILE=str(tokenizer))
            for key, value in options.items():
                if value == "EXISTING":
                    value = str(root / (key + " with spaces"))
                    Path(value).touch()
                env[key] = value
            return subprocess.run(["sh", str(ENTRY), *arguments], env=env,
                                  text=True, capture_output=True)

    def test_defaults_and_arguments(self):
        result = self.launch({}, ["--lookup", "16"])
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("--context") + 1], "16384")
        self.assertEqual(args[-2:], ["--lookup", "16"])
        self.assertTrue(args[1].endswith("model with spaces.gguf"))

    def test_cli_settings_and_path_quoting(self):
        result = self.launch({"DRAFT_MODEL_FILE": "EXISTING", "DRAFT_VOCAB_FILE": "EXISTING",
                              "KV_FORMAT": "tq3", "CONTEXT": "32768"})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        self.assertEqual(args[args.index("--kv") + 1], "tq3")
        self.assertTrue(args[args.index("-md") + 1].endswith("DRAFT_MODEL_FILE with spaces"))
        self.assertTrue(args[args.index("--draft-vocab") + 1].endswith("DRAFT_VOCAB_FILE with spaces"))

    def test_missing_optional_file_is_an_error(self):
        result = self.launch({"DRAFT_MODEL_FILE": "/nonexistent/draft.gguf"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("Draft model missing", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
