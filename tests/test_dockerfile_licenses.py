"""Fail-first test for release fix 1: the runtime image must ship the license files.

Apache-2.0 requires that a distributed binary carries the license text and the notices.
The Dockerfile's final (runtime) stage copied only tqserve / tqtokenize / entrypoint.sh,
so a `docker pull` image contained NO LICENSE, NO NOTICE and NO third-party notices --
a license-compliance blocker for the release.

This test asserts on the Dockerfile TEXT (no docker daemon, no GPU, no network), matching
the static-analysis style of tests/test_docker_entrypoint.py. It is registered with ctest
as `dockerfile-licenses` so `ctest` covers it alongside the C++ tests.

It deliberately FAILS on bd5b173 and passes once the runtime stage COPYs the license files.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"

# Files that exist in the repo and MUST end up inside the runtime image.
# third_party/stb/stb_image.h is public-domain and already ships inside src/ (copied by the build stage),
# so it needs no extra COPY; the rest are top-level or third_party notices that the runtime
# stage never received.
REQUIRED = [
    "LICENSE",
    "NOTICE",
    "licenses/ggml-MIT.txt",
    "third_party/jinja/LICENSE",
    "third_party/jinja/NOTICE",
]


def runtime_stage_text():
    """Return only the FINAL stage of the Dockerfile (after the last FROM).

    Asserting on the whole file would pass if the BUILD stage copied the licenses, which is
    not what ships. The runtime image is what `docker pull` gives a user.
    """
    text = DOCKERFILE.read_text(encoding="utf-8", errors="replace")
    # split on FROM at line start; the last chunk is the runtime stage
    parts = re.split(r"(?m)^\s*FROM\s+", text)
    if len(parts) < 2:
        raise AssertionError("Dockerfile has no FROM line")
    return parts[-1]


class DockerfileLicenseTests(unittest.TestCase):
    def test_repo_actually_has_the_license_files(self):
        """Guard: if these paths move, the test must fail loudly rather than pass vacuously."""
        missing = [p for p in REQUIRED if not (REPO / p).is_file()]
        self.assertEqual(missing, [], f"expected license files absent from repo: {missing}")

    def test_runtime_stage_copies_the_top_level_license_and_notice(self):
        stage = runtime_stage_text()
        for name in ("LICENSE", "NOTICE"):
            with self.subTest(file=name):
                # a COPY of that path into the image
                self.assertRegex(
                    stage,
                    rf"(?m)^\s*COPY\s+.*\b{re.escape(name)}\b",
                    f"runtime stage does not COPY {name} into the image",
                )

    def test_runtime_stage_copies_third_party_notices(self):
        stage = runtime_stage_text()
        # Accept either an explicit per-file COPY or a directory COPY that contains them.
        covers_licenses_dir = re.search(r"(?m)^\s*COPY\s+.*\blicenses\b", stage) is not None
        covers_third_party = re.search(r"(?m)^\s*COPY\s+.*\bthird_party\b", stage) is not None
        self.assertTrue(
            covers_licenses_dir,
            "runtime stage does not COPY licenses/ (ggml MIT text) into the image",
        )
        self.assertTrue(
            covers_third_party,
            "runtime stage does not COPY third_party/ (Jinja LICENSE+NOTICE) into the image",
        )

    def test_every_required_license_path_is_reachable_in_the_image(self):
        """Each required repo file must be named by some COPY in the runtime stage, or be
        inside a directory that stage COPYs. This is the assertion that actually failed on
        bd5b173, where the runtime stage copied nothing but binaries."""
        stage = runtime_stage_text()
        copied_sources = re.findall(r"(?m)^\s*COPY\s+(?:--from=\S+\s+)?(.+)$", stage)
        # last whitespace-separated token of each COPY is the destination; the rest are sources
        sources = []
        for line in copied_sources:
            toks = line.split()
            if len(toks) >= 2:
                sources.extend(toks[:-1])

        unresolved = []
        for path in REQUIRED:
            ok = any(path == s or path.startswith(s.rstrip("/") + "/")
                     or s.rstrip("/") in path.split("/")
                     for s in sources)
            if not ok:
                unresolved.append(path)
        self.assertEqual(
            unresolved, [],
            f"runtime image would ship without: {unresolved} "
            f"(COPY sources in runtime stage: {sources})",
        )


if __name__ == "__main__":
    unittest.main()
