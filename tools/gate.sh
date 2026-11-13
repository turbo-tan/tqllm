#!/usr/bin/env bash
# Release gate on a GPU box: the full ctest with the model files set, then one record line, so "the model tests ran" is
# written down and not assumed.
#
#   tools/gate.sh [--build DIR] [--env-file FILE] [--model GGUF] [--drafter GGUF] [--draft-vocab TXT] [--env NAME=VALUE ...]
#                 [--record FILE]
#
# --env-file reads NAME=VALUE lines (# comments, ~ expanded; tools/gate.env.example lists every model variable), so
# a box's gate is one command; flags after it override its values.
#
# --model / --drafter / --draft-vocab set TQLLM_TEST_MODEL / _DRAFTER / _DRAFT_VOCAB (values already in the environment
# are kept when a flag is not given); --env sets any other test variable (embedder fixtures, TQLLM_TEST_TOKENS, ...).
# A model test is a registered test whose source reads one of the model-file variables below; each is listed as run,
# skipped, no-model (passed with none of its model variables set) or FAIL. Exit status: 0 when every test passed
# and every model test ran with its model, 1 otherwise.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build"
RECORD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) BUILD=$(cd "$2" && pwd); shift 2 ;;
    --model) export TQLLM_TEST_MODEL="$2"; shift 2 ;;
    --drafter) export TQLLM_TEST_DRAFTER="$2"; shift 2 ;;
    --draft-vocab) export TQLLM_TEST_DRAFT_VOCAB="$2"; shift 2 ;;
    --env) export "${2?--env NAME=VALUE}"; shift 2 ;;
    --env-file)
      while IFS= read -r line || [ -n "$line" ]; do
        line=${line%%#*}; line=${line%"${line##*[![:space:]]}"}
        [ -z "$line" ] && continue
        case "$line" in *=*) export "${line%%=*}=${line#*=}" ;; *) echo "gate.sh: bad line in $2: $line" >&2; exit 2 ;; esac
        v=${line#*=}; [ "${v:0:2}" = "~/" ] && export "${line%%=*}=$HOME/${v:2}"
      done < "$2"; shift 2 ;;
    --record) RECORD="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "gate.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -f "$BUILD/CTestTestfile.cmake" ] || { echo "gate.sh: no ctest tree in $BUILD (build first)" >&2; exit 2; }
# the source tree the build was configured from (test sources, commit), so a build of another checkout reports its own
SRC=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")
[ -d "$SRC" ] && ROOT="$SRC"

OUT="$BUILD/gate"
mkdir -p "$OUT"
( cd "$BUILD" && ctest -j1 --timeout 3600 --output-junit "$OUT/junit.xml" --output-on-failure > "$OUT/ctest.log" 2>&1 )
ctest_rc=$?
( cd "$BUILD" && ctest --show-only=json-v1 > "$OUT/tests.json" )

python3 - "$ROOT" "$OUT" "$ctest_rc" "$RECORD" <<'PY'
import json, os, re, socket, subprocess, sys, xml.etree.ElementTree as ET
root, out, ctest_rc, record = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
MODEL_VARS = re.compile(r"TQLLM_TEST_(?:MODEL|DRAFTER|DRAFT_VOCAB|TOKENS|HF_\w+|REF_GGUF)\b|TQLLM_TOK_(?:GGUF|TQTOK)\b|TQLLM_PPLX_\w+")

# test name -> executable -> its sources in tests/ (add_executable lines of every CMakeLists.txt)
cmake = ""
for d, _, files in os.walk(root):
    if "/build" in d or "/.git" in d: continue
    if "CMakeLists.txt" in files: cmake += open(os.path.join(d, "CMakeLists.txt"), errors="replace").read()
exe_src = {m.group(1): re.findall(r"tests/[\w./-]+", m.group(2)) for m in re.finditer(r"add_executable\(\s*([\w.-]+)\s+([^)]*)\)", cmake)}
model_tests = {}  # test name -> the model variables its sources read
for t in json.load(open(os.path.join(out, "tests.json")))["tests"]:
    cmd = t.get("command") or []
    if not cmd: continue
    exe = os.path.splitext(os.path.basename(cmd[0]))[0]
    srcs = exe_src.get(exe, []) + [a for a in cmd[1:] if re.match(r".*tests/[\w./-]+\.py$", a)]
    used = {v for s in srcs if os.path.isfile(os.path.join(root, s))
            for v in MODEL_VARS.findall(open(os.path.join(root, s), errors="replace").read())}
    if used: model_tests[t["name"]] = used

status, total, passed, failed = {}, 0, 0, []
for case in ET.parse(os.path.join(out, "junit.xml")).getroot().iter("testcase"):
    name, total = case.get("name"), total + 1
    text = "".join(e.text or "" for e in case.iter() if e.tag in ("system-out", "skipped"))
    if case.find("failure") is not None or case.find("error") is not None or case.get("status") == "fail":
        status[name] = "FAIL"; failed.append(name)
    elif case.find("skipped") is not None or case.get("status") in ("notrun", "disabled") or re.search(r"\bskipped\b", text, re.I):
        status[name] = "skipped"; passed += case.find("skipped") is None
    else:
        status[name] = "run"; passed += 1

commit = subprocess.run(["git", "-C", root, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
dirty = "+dirty" if subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() else ""
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
# a model test that passed while none of its model variables was set ran without a model
for n, used in model_tests.items():
    if status.get(n) == "run" and not any(os.environ.get(v) for v in used): status[n] = "no-model"
models = " ".join(f"{n}={status.get(n, 'missing')}" for n in sorted(model_tests))
line = (f"GATE {commit}{dirty} {socket.gethostname()} ({gpu[0] if gpu else 'no GPU'}) passed {passed}/{total}"
        f"{' failed ' + ','.join(failed) if failed else ''} | model tests {len(model_tests)}: {models}")
print(line)
if record:
    open(record, "a").write(line + "\n")
ok = ctest_rc == 0 and not failed and all(status.get(n) == "run" for n in model_tests)
sys.exit(0 if ok else 1)
PY
