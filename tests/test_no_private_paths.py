"""Shipped source must not name private repositories, personal paths or hostnames.

The patterns are assembled from fragments so this file does not contain the strings it forbids.
"""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Tokens that must not appear in shipped source. Case-insensitive.
FORBIDDEN = [
    "ai_" + "work" + "space",
    "tan" + "llm",
    "char" + "pdev",
    "/home/" + "awee",
    "/home/" + "ywtan",
    "DESKTOP-" + "RAINBOW",
]

# Binary/irrelevant extensions to skip (byte-noise false positives, per the .npz lesson).
SKIP_EXT = {
    ".npz", ".gguf", ".bin", ".safetensors", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".pdf", ".zip", ".gz", ".bz2", ".xz", ".so",
    ".a", ".o", ".pyc", ".pack", ".webm", ".mp4",
}

SELF = "tests/test_no_private_paths.py"


def tracked_text_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    files = []
    for line in out.splitlines():
        p = line.strip()
        if not p or p == SELF:
            continue
        if Path(p).suffix.lower() in SKIP_EXT:
            continue
        files.append(p)
    return files


class NoPrivateBreadcrumbTests(unittest.TestCase):
    def test_no_forbidden_tokens_in_tracked_source(self):
        offenders = {}
        for rel in tracked_text_files():
            fp = REPO / rel
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                continue
            for tok in FORBIDDEN:
                if tok.lower() in text:
                    # record file + first matching line number
                    for i, line in enumerate(fp.read_text(encoding="utf-8",
                                                          errors="replace").splitlines(), 1):
                        if tok.lower() in line.lower():
                            offenders.setdefault(rel, []).append((i, tok, line.strip()[:90]))
                            break
        pretty = "\n".join(
            f"  {f}:{ln} [{tok}] {snip}"
            for f, hits in offenders.items() for (ln, tok, snip) in hits
        )
        self.assertEqual(offenders, {}, "private breadcrumbs found in public source:\n" + pretty)

    def test_test_itself_is_the_only_allowed_carrier(self):
        """Sanity: the scan must actually be looking at files (not silently scanning none)."""
        files = tracked_text_files()
        self.assertGreater(len(files), 50, "tracked-file scan found suspiciously few files")
        self.assertNotIn(SELF, files, "this test must exclude itself from the scan")


if __name__ == "__main__":
    unittest.main()
