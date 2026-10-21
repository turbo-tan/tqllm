"""Source layering (AGENTS.md): includes point down the layers only, and a model never includes another model.

Layers, top to bottom: tools/ > src/engine/ > src/models/<family>/<arch>/ > components (src/formats, src/kv, src/vision,
src/experts, src/spec) > src/core/ and include/tqllm/ > third_party/. Includes name the file only (the folders are the
search path), so each include is resolved through the unique basenames of the tree. Known violations are listed in
tests/layering_allowlist.txt; the list may only shrink: a new violation fails, and so does an entry that no longer applies.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_files import repo_files  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ALLOW = REPO / "tests" / "layering_allowlist.txt"
SRC_EXT = (".cpp", ".cu", ".cuh", ".hpp", ".h", ".inc")
INTERFACE = {"src/engine/engine.hpp"}  # the interface every model implements: it sits with core


def layer(path):
    """(rank, model) of a tracked file; rank is None for files outside the layering (tests, bench)."""
    if path in INTERFACE or path.startswith(("src/core/", "include/")):
        return 1, None
    if path.startswith("third_party/"):
        return 0, None
    if path.startswith(("src/formats/", "src/kv/", "src/experts/", "src/spec/", "src/vision/")):
        return 2, None
    if path.startswith("src/models/"):
        parts = path.split("/")
        return 3, "/".join(parts[2:4]) if len(parts) > 4 else parts[2]
    if path.startswith("src/backends/"):  # an alternative implementation (e.g. Apple MLX / Metal): model level, its own group
        return 3, "backends/" + path.split("/")[2]
    if path.startswith("src/engine/"):
        return 4, None
    if path.startswith("tools/"):
        return 5, None
    return None, None


def violations():
    files = repo_files(REPO)
    by_base = {}
    for f in files:
        by_base.setdefault(Path(f).name, []).append(f)
    inc = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
    found = set()
    for f in files:
        if not f.endswith(SRC_EXT):
            continue
        rank, model = layer(f)
        if rank is None:
            continue
        text = (REPO / f).read_text(errors="replace")
        for name in inc.findall(text):
            target = (Path(f).parent / name).as_posix()
            if target not in by_base.get(Path(name).name, []):
                cands = [c for c in by_base.get(Path(name).name, []) if layer(c)[0] is not None]
                if len(cands) != 1:
                    continue
                target = cands[0]
            trank, tmodel = layer(target)
            if trank is None or target == f:
                continue
            upward = trank > rank
            cross = rank == 3 and trank == 3 and tmodel != model
            if upward or cross:
                found.add((f, target))
    return found


class Layering(unittest.TestCase):
    def test_archive_not_referenced(self):
        """archive/ holds retired sources (archive/README.md): no build file, source, test or script may name them."""
        files = repo_files(REPO)
        archived = [Path(f).name for f in files if f.startswith("archive/") and not f.endswith("README.md")]
        code = (".txt", ".cmake", ".cpp", ".cu", ".cuh", ".hpp", ".h", ".inc", ".py", ".sh", ".yml", ".yaml")
        build = [f for f in files if not f.startswith("archive/") and (f.endswith(code) or "Dockerfile" in f) and (REPO / f).is_file()]
        hits = [f"{f}: {name}" for f in build for name in archived if name in (REPO / f).read_text(errors="ignore")]
        self.assertEqual(hits, [], "archived sources referenced (move the needed code out of archive/ instead)")

    def test_includes_point_down(self):
        allowed = set()
        if ALLOW.exists():
            for line in ALLOW.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    a, b = line.split()[:2]
                    allowed.add((a, b))
        found = violations()
        new = sorted(found - allowed)
        stale = sorted(allowed - found)
        self.assertFalse(new, "includes that point up or across models (fix them, see AGENTS.md):\n" +
                         "\n".join("  %s -> %s" % p for p in new))
        self.assertFalse(stale, "allowlist entries that no longer apply (delete them from tests/layering_allowlist.txt):\n" +
                         "\n".join("  %s -> %s" % p for p in stale))


if __name__ == "__main__":
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--list":
        for a, b in sorted(violations()):
            print("%s\t%s" % (a, b))
    else:
        unittest.main()
