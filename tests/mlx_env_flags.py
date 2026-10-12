import os
import subprocess
import sys

env = {k: v for k, v in os.environ.items() if not k.startswith("TQLLM_")}
for additions, code, text in (
    ({}, 0, ""),
    ({"TQLLM_UNKNOWN_TEST_ONLY": "1"}, 0, "unknown variable TQLLM_UNKNOWN_TEST_ONLY"),
    ({"TQLLM_BATCH_COLS": "1"}, 2, "TQLLM_BATCH_COLS was removed"),
):
    p = subprocess.run([sys.argv[1]], env=env | additions, capture_output=True, text=True, timeout=10)
    if p.returncode != code or text not in p.stderr:
        raise RuntimeError(f"environment scan failed: {p.returncode}, {p.stderr!r}")
print("mlx_env_flags: PASS")
