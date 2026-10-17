#!/usr/bin/env python3
"""Move a tqLLM checkout (main or a branch) onto the component / architecture layout of AGENTS.md.

Run it at the root of a clean checkout that still has the old layout; it stages the result (git mv + edits) and
does not commit. Moving a branch: run it on the branch tip, commit ("relayout"), then rebase or merge onto the new
main; the branch's own changes then apply without path conflicts.

Driven by docs/dev/layout-map-20261017.tsv (old path -> new path; `branch:` rows map paths that only exist on
branches, `*` matches a file name). Steps: git mv every mapped file, rewrite includes of moved files to the bare file
name (CMake puts every source folder on the search path), replace old paths in text files, update CMakeLists.txt.
"""
import fnmatch
import os
import re
import subprocess
import sys

ROOT = os.getcwd()
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "dev", "layout-map-20261017.tsv")
TEXT_EXT = (".cpp", ".cu", ".cuh", ".hpp", ".h", ".inc", ".md", ".py", ".yaml", ".yml", ".txt", ".cmake", ".sh", ".json", ".toml")
SRC_EXT = (".cpp", ".cu", ".cuh", ".hpp", ".h", ".inc")
INCLUDE_DIRS = ["src/core", "src/engine", "src/formats/weight", "src/formats/kv", "src/kv", "src/experts", "src/spec",
                "src/models/qwen/qwen35", "src/models/qwen/qwen2", "src/models/qwen/qwen4exp", "src/vision",
                "src/formats/weight/x3", "src/models/nemotron", "src/models/glm", "third_party/nlohmann", "third_party/stb", "third_party"]
# literal edits outside plain path renames (applied only where the old text is present)
LITERAL = {
    "CONTRIBUTING.md": [("Put its CUDA kernels in `kernels/formats/<name>.cu`.",
                         "Put its CUDA kernels in `src/formats/weight/<name>.cu`, next to its host code.")],
    "NOTICE": [("* kernels/*_grid.inc,", "* src/formats/weight/*_grid.inc,")],
    "tools/gen_flags.py": [('list(ROOT.glob("kernels/**/*.cu*")) + \\\n             list(ROOT.glob("kernels/**/*.hpp")):',
                            'list(ROOT.glob("src/**/*.cu*")):')],
    ".gitignore": [("\nmodels/\n", "\n/models/\n")],
    "src/core/common.cuh": [("shared by kernels/formats/*.cu", "shared by the format kernels (src/formats/*/*.cu)")],
    "src/formats/weight/dequant.cpp": [("tables in kernels/*.inc", "tables in src/formats/weight/*.inc")],
    "src/formats/weight/weight_format.hpp": [("kernels/formats/) plus one line", "src/formats/weight/) plus one line")],
    "tests/int8_error_theory.py": [("from kernels/kvalues_iq4nl.inc", "from src/formats/weight/kvalues_iq4nl.inc")],
    "docs/ARCHITECTURE.md": [("-> kernels/formats/, src/formats/weight/quant.cu", "-> src/formats/weight/ (host + kernels, quant.cu)"),
                             ("kernels/formats/kv_*.", "src/formats/kv/kv_*.")],
}


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=ROOT, check=True, capture_output=True, text=True).stdout


def load_map(tracked):
    rows = []
    for line in open(MAP):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        branch = line.startswith("branch:")
        old, new = line[len("branch:"):].split("\t") if branch else line.split("\t")
        if "*" in old:
            for f in tracked:
                if fnmatch.fnmatchcase(f, old):
                    rows.append((f, new.replace("*", f[len(old.split("*")[0]):], 1) if new.endswith("*") else
                                 os.path.join(os.path.dirname(new), os.path.basename(f))))
        else:
            rows.append((old, new))
    return dict(rows)


def main():
    tracked = git("ls-files").split()
    full = load_map(tracked)
    present = {o: n for o, n in full.items() if o in tracked}
    for o, n in sorted(present.items()):
        os.makedirs(os.path.join(ROOT, os.path.dirname(n)), exist_ok=True)
        git("mv", o, n)
    print("moved %d files" % len(present))
    tracked = git("ls-files").split()
    new2old = {n: o for o, n in present.items()}
    base_new = {}
    for n in full.values():
        base_new.setdefault(os.path.basename(n), set()).add(n)
    quote = re.compile(r'(#\s*include\s*")([^"]+)(")')
    angle = re.compile(r'(#\s*include\s*<)((?:src|kernels|tools)/[^>]+)(>)')
    edited = 0
    for f in tracked:
        if not f.endswith(SRC_EXT):
            continue
        p = os.path.join(ROOT, f)
        s = open(p, errors="surrogateescape").read()
        old_loc = new2old.get(f, f)

        def fix_quote(m):
            path = m.group(2)
            b = os.path.basename(path)
            for d in [os.path.dirname(old_loc), "kernels", "src", ".", "tools", "kernels/formats", "src/formats"]:
                c = os.path.normpath(os.path.join(d, path))
                if c in full:
                    if full[c].startswith("third_party/"):  # vendored trees keep their own paths: <lib>/<path> (third_party/ is on the search path)
                        if os.path.dirname(full[c]) == os.path.dirname(full.get(old_loc, old_loc)):
                            return m.group(0)
                        return m.group(1) + full[c][len("third_party/"):] + m.group(3)
                    if len(base_new.get(b, ())) != 1:
                        return m.group(0)
                    return m.group(1) + b + m.group(3)
            return m.group(0)

        def fix_angle(m):
            path = m.group(2)
            return m.group(1) + full[path] + m.group(3) if path in full else m.group(0)

        n = angle.sub(fix_angle, quote.sub(fix_quote, s))
        if n != s:
            open(p, "w", errors="surrogateescape").write(n)
            edited += 1
    print("include edits in %d files" % edited)
    pats = sorted(full.items(), key=lambda r: -len(r[0]))
    # whole folders that move (wildcard rows `dir/*`): bare directory strings too (include dirs in CMakeLists.txt / cmake/*.cmake)
    dir_rows = []
    for line in open(MAP):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        old, new = (line[len("branch:"):] if line.startswith("branch:") else line).split("\t")
        if old.endswith("/*") and new.endswith("/*") and any(f.startswith(old[:-1]) for f in present):
            dir_rows.append((old[:-2], new[:-2]))
    texted = 0
    for f in tracked:
        if f.endswith(("layout-map-20261017.tsv", "relayout.py")) or not (f.endswith(TEXT_EXT) or f in ("NOTICE", "Dockerfile", "CMakeLists.txt", ".gitignore")):
            continue
        p = os.path.join(ROOT, f)
        try:
            s = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        n = s
        for o, nn in pats:
            n = re.sub(r'(?<![\w/])' + re.escape(o) + r'(?![\w.])', nn, n)
        for od, nd in dir_rows:
            n = re.sub(r'(?<![\w/])' + re.escape(od) + r'(?=[\s)"/;]|$)', nd, n, flags=re.M)
        for a, b in LITERAL.get(f, []):
            n = n.replace(a, b)
        if f == "CMakeLists.txt":
            n = n.replace("file(GLOB TQLLM_FORMAT_SOURCES CONFIGURE_DEPENDS src/formats/*.cpp kernels/formats/*.cu)",
                          "file(GLOB_RECURSE TQLLM_FORMAT_SOURCES CONFIGURE_DEPENDS src/formats/*.cpp src/formats/*.cu)")
            n = n.replace("${CMAKE_CURRENT_SOURCE_DIR}/kernels/third_party/", "${CMAKE_CURRENT_SOURCE_DIR}/third_party/")
            n = n.replace("(MIT, kernels/third_party/", "(MIT, third_party/")
            n = re.sub(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/kernels(?![\w/])", "${CMAKE_CURRENT_SOURCE_DIR}/src", n)
            if "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/core" not in n:
                dirs = [d for d in INCLUDE_DIRS if os.path.isdir(os.path.join(ROOT, d))]
                block = ("# Source layout (AGENTS.md): one folder per component and per model architecture, host and device code together.\n"
                         "# Includes name the file only; these folders are the search path.\n"
                         "include_directories(" + " ".join("${CMAKE_CURRENT_SOURCE_DIR}/" + d for d in dirs) + ")\n")
                n = n.replace("include(CTest)\n", "include(CTest)\n" + block, 1)
            # files the recursive format glob now compiles: drop their explicit entries
            for g in ["src/formats/weight/dequant.cpp", "src/formats/weight/quant.cu", "src/formats/weight/gsq_activation.cu", "src/formats/kv/kv_compact.cu"]:
                n = re.sub(r"(target_sources\(tqllm PRIVATE[^)]*?) " + re.escape(g) + r"(?=[\s)])", r"\1", n)
        if n != s:
            open(p, "w", encoding="utf-8").write(n)
            texted += 1
    print("path strings updated in %d files" % texted)
    if os.path.isdir(os.path.join(ROOT, "kernels")) and not any(f.startswith("kernels/") for f in tracked):
        os.system("rm -rf '%s'" % os.path.join(ROOT, "kernels"))
    git("add", "-A")
    left = [f for f in git("ls-files").split() if f.startswith("kernels/")]
    if left:
        print("not mapped (add them to the map):", " ".join(left))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
