"""Every path the Dockerfile copies from the build context exists (a moved folder broke the image build silently)."""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class DockerfilePaths(unittest.TestCase):
    def test_copy_sources_exist(self):
        missing = []
        for line in (REPO / "Dockerfile").read_text().splitlines():
            m = re.match(r"\s*COPY\s+(?!--from)(.+)\s+\S+\s*$", line)
            if not m:
                continue
            for src in m.group(1).split():
                if not (REPO / src.rstrip("/")).exists():
                    missing.append(src)
        self.assertEqual(missing, [], "Dockerfile COPY sources missing from the repository")


if __name__ == "__main__":
    unittest.main()
