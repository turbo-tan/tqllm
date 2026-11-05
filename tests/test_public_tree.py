"""The public tree holds the product only: no benchmark dumps, agent handovers, private experiment notes, logs, secrets
or internal labels. Experiment records and handovers belong in the private workspace, not in this repository.

A new top-level directory, a dated file name or a large file fails here until it is added to the allow lists below in
the same pull request, so each one is a reviewed decision. Markers are assembled from fragments so this file does not
contain the strings it forbids.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_files import repo_files  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SELF = "tests/test_public_tree.py"

TOP_LEVEL = {
    ".dockerignore", ".gitattributes", ".github", ".gitignore", "AGENTS.md", "CHANGELOG.md", "CMakeLists.txt",
    "CONTRIBUTING.md", "Dockerfile", "LICENSE", "NOTICE", "README.md", "archive", "bench", "cmake", "compose.yaml",
    "config", "docker", "docs", "include", "licenses", "quant", "src", "tests", "third_party", "tools",
}
# path parts that mark working material, not product
FORBIDDEN_PARTS = {"results", "scratch", "tmp", "logs", "handover", "handovers", "inbox", "wip"}
FORBIDDEN_SUFFIX = (".log", ".bak", ".orig", ".rej", ".swp", ".tmp", ".pid")
FORBIDDEN_NAME = re.compile(r"(^|[-_.])(handover|nohup|scratch)([-_.]|$)", re.I)
DATED = re.compile(r"20[0-9]{2}[01][0-9][0-3][0-9]")
DATED_ALLOWED = {"docs/dev/layout-map-20261017.tsv"}  # read by tools/dev/relayout.py
MAX_BYTES = 5 << 20
LARGE_ALLOWED = set()

# text markers of internal material (case-insensitive)
MARKERS = [
    "private " + "experiment",
    "hand" + "over:",
    "for the " + "agent on",
    "ai-work" + "space-",
    "claude-" + "session:",
    "hc" + "q8",
    "owner " + "go",
    "not a release " + "feature",
]
SECRETS = [
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-(ant-)?[A-Za-z0-9_-]{32,}"),
    re.compile(r"\bdckr_pat_[A-Za-z0-9_-]{20,}"),
    re.compile("-----BEGIN [A-Z ]*PRIV" + "ATE KEY-----"),
]
BINARY = {".npz", ".gguf", ".bin", ".safetensors", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf",
          ".pdf", ".zip", ".gz", ".so", ".a", ".o", ".pyc", ".webm", ".mp4"}


def files():
    return [f for f in repo_files(REPO) if f != SELF]


class PublicTreeTests(unittest.TestCase):
    def test_top_level_is_known(self):
        extra = sorted({f.split("/")[0] for f in files()} - TOP_LEVEL)
        self.assertEqual(extra, [], "new top-level entries (add to TOP_LEVEL if they are product): " + ", ".join(extra))

    def test_no_working_material_paths(self):
        bad = []
        for f in files():
            parts = f.lower().split("/")
            name = parts[-1]
            if set(parts[:-1]) & FORBIDDEN_PARTS or name.endswith(FORBIDDEN_SUFFIX) or FORBIDDEN_NAME.search(name):
                bad.append(f)
            elif DATED.search(name) and f not in DATED_ALLOWED:
                bad.append(f + " (dated file name)")
        self.assertEqual(bad, [], "working material in the public tree:\n  " + "\n  ".join(bad))

    def test_no_large_files(self):
        big = [f"{f} ({(REPO / f).stat().st_size >> 20} MB)" for f in files()
               if (REPO / f).is_file() and (REPO / f).stat().st_size > MAX_BYTES and f not in LARGE_ALLOWED]
        self.assertEqual(big, [], "files over 5 MB (models and data go to Hugging Face):\n  " + "\n  ".join(big))

    def test_no_internal_markers_or_secrets(self):
        hits = []
        for f in files():
            p = REPO / f
            if p.suffix.lower() in BINARY or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            low = text.lower()
            for m in MARKERS:
                if m in low:
                    line = next(i for i, s in enumerate(text.splitlines(), 1) if m in s.lower())
                    hits.append(f"{f}:{line} [{m}]")
            for rx in SECRETS:
                mt = rx.search(text)
                if mt:
                    hits.append(f"{f}:{text.count(chr(10), 0, mt.start()) + 1} [secret-like {rx.pattern[:12]}...]")
        self.assertEqual(hits, [], "internal markers or secrets in the public tree:\n  " + "\n  ".join(hits))

    def test_rules_catch_known_cases(self):
        """The rules flag the files removed on 2026-11-05 (so a regression in the rules shows up here)."""
        for f in ["results/x_gb10/measure_speed.py", "docs/dev/handover-metal-tree-verify-20261101.md",
                  "docs/validation/run.log", "tools/foo_nohup.txt"]:
            parts = f.lower().split("/")
            name = parts[-1]
            caught = bool(set(parts[:-1]) & FORBIDDEN_PARTS or name.endswith(FORBIDDEN_SUFFIX)
                          or FORBIDDEN_NAME.search(name) or (DATED.search(name) and f not in DATED_ALLOWED))
            self.assertTrue(caught, f)
        self.assertTrue(any(m in "private experiment, not a release feature." for m in MARKERS))
        self.assertTrue(any(rx.search("hf_" + "a" * 34) for rx in SECRETS))


if __name__ == "__main__":
    unittest.main()
