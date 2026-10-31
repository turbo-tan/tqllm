"""Docs match the code: every tqserve option is in docs/reference/options.md, and the AGENTS.md source map names every
top-level src/ folder, every platform under src/backends/ and every architecture folder under src/models/."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_files import repo_files  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OPTION_SOURCES = ("src/core/options.cpp", "tools/tqserve.cpp")
SRC_ROOT_FILES = {"src/README.md", "src/COMMIT"}  # COMMIT: the git-archive commit stamp (export-subst)


class OptionsDocumented(unittest.TestCase):
    def test_every_option_documented(self):
        doc = set(re.findall(r"(?<![\w-])(--?[a-z][a-z0-9-]*)", (REPO / "docs/reference/options.md").read_text()))
        missing = {}
        for f in OPTION_SOURCES:
            for o in re.findall(r'"(--?[a-z][a-z0-9-]*)"', (REPO / f).read_text()):
                if o not in doc:
                    missing.setdefault(o, f)
        self.assertEqual(missing, {}, "options missing from docs/reference/options.md")


class SourceMap(unittest.TestCase):
    def test_map_names_every_folder(self):
        text = (REPO / "AGENTS.md").read_text()
        files = [f for f in repo_files(REPO) if f.startswith("src/")]
        missing = sorted({f for f in files if f.count("/") == 1} - SRC_ROOT_FILES)  # loose files belong in a folder
        for f in files:
            parts = f.split("/")
            if len(parts) > 2 and f"src/{parts[1]}/" not in text:
                missing.append(f"src/{parts[1]}/")
            if parts[1] == "backends" and len(parts) > 3 and f"{parts[2]}/" not in text:
                missing.append(f"src/backends/{parts[2]}/")
            if parts[1] == "models":
                if len(parts) < 5:
                    missing.append(f"{f} (models live in src/models/<family>/<arch>/)")
                elif f"{parts[2]}/{parts[3]}" not in text:
                    missing.append(f"src/models/{parts[2]}/{parts[3]}/")
        self.assertEqual(sorted(set(missing)), [], "AGENTS.md source map is out of date")


if __name__ == "__main__":
    unittest.main()
