#!/usr/bin/env python3
"""Engine headers shared by several translation units (src/models/qwen/qwen4exp/fn_*.hpp) must not declare file-scope
`static` variables: each .cpp would get its own copy without any error (a PersistentAlloc in one file would then be
invisible to an allocation in another). Shared state is `inline`."""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
bad = []
for p in sorted((ROOT / "src/models/qwen/qwen4exp").glob("fn_*.hpp")):
    depth = 0
    for n, line in enumerate(p.read_text().split("\n"), 1):
        code = re.sub(r'"(\\.|[^"\\])*"', '""', line).split("//")[0]
        if depth == 0:
            for stmt in code.split(";"):
                s = stmt.strip()
                if s.startswith("static ") and "(" not in s.split("=")[0]:
                    bad.append(f"{p.relative_to(ROOT)}:{n}: file-scope static variable: {s[:80]}")
        depth += code.count("{") - code.count("}")
for b in bad:
    print(b)
print("engine headers:", "FAIL" if bad else "ok")
sys.exit(1 if bad else 0)
