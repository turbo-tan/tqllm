"""The repository's source files without git (tests run in archives, Docker builds and copies with no .git): every file
under the repository root except .git, hidden directories, build trees (any directory with a CMakeCache.txt, or named
build*) and caches."""
import os
from pathlib import Path


def repo_files(repo):
    repo = Path(repo)
    out = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not d.startswith("build") and d != "__pycache__"
                         and not (Path(root) / d / "CMakeCache.txt").exists())
        rel = Path(root).relative_to(repo)
        out += [(rel / f).as_posix() for f in sorted(files) if f != ".git"]  # a worktree's .git is a file
    return out
