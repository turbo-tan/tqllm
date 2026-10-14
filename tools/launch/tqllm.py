#!/usr/bin/env python3
"""tqllm launch: start (or reuse) a local tqserve and run a coding agent against it, like `ollama launch`.

  tqllm.py launch opencode --model FILE.gguf [--drafter FILE --draft-vocab FILE] [--context 65536] [-- agent args]
  tqllm.py launch codex    --model FILE.gguf ...
  tqllm.py launch claude   --model FILE.gguf ...   (Claude Code, through the server's Anthropic Messages API)
  tqllm.py serve           --model FILE.gguf ...   (just the server, until Ctrl+C)
  tqllm.py --list                                  (the model catalog: models.json, local copies marked)

--model takes a file, a catalog name (`--list`: qwen3.8-27b, -s, -xs, -l) or nothing (the catalog default for this GPU). Catalog
files are looked up in ./models, ~/models and $TQLLM_MODELS; missing ones are offered for download from Hugging Face
(size and sha256 checked), --yes downloads without asking.

The server comes from the Docker image (default) or a native build (--native PATH/tqserve). A server already answering
on --port with the same model name is reused and left running; one this command started is stopped when the agent exits.
The agents' own configuration files are never written: each gets its provider through environment / command line.
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

IMAGE = "ghcr.io/turbo-tan/tqllm:1.0.0"
NAME = "tqllm"


CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")


def model_dirs():
    dirs = [os.environ["TQLLM_MODELS"]] if os.environ.get("TQLLM_MODELS") else []
    return dirs + [os.path.abspath("models"), os.path.expanduser("~/models"), os.path.expanduser("~/.cache/tqllm/models")]


def find_local(rel):
    """A catalog file (repo path like drafters/x.gguf) in any model folder, by its path or by its file name."""
    for d in model_dirs():
        for cand in (os.path.join(d, rel), os.path.join(d, os.path.basename(rel))):
            if os.path.isfile(cand): return cand
    return None


def gpu_vram_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.split()[0]) / 1024
    except Exception:
        return None


def download(entry, dest_dir):
    """Hugging Face download with resume, size and sha256 check; returns the local path."""
    dest = os.path.join(dest_dir, entry["path"]); os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['path']}"
    part = dest + ".part"; have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        total, t0, done = entry.get("size", 0), time.time(), have
        while True:
            chunk = r.read(8 << 20)
            if not chunk: break
            f.write(chunk); done += len(chunk)
            if total: print(f"\rtqllm: {entry['path']}: {done / 1e9:.2f} / {total / 1e9:.2f} GB ({done / max(1e-9, time.time() - t0) / 1e6:.0f} MB/s)", end="", file=sys.stderr)
    print(file=sys.stderr)
    if entry.get("size") and os.path.getsize(part) != entry["size"]: sys.exit(f"tqllm: {entry['path']}: size mismatch, keep {part} and retry")
    if entry.get("sha256"):
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""): h.update(chunk)
        if h.hexdigest() != entry["sha256"]: os.remove(part); sys.exit(f"tqllm: {entry['path']}: sha256 mismatch, removed")
    os.replace(part, dest)
    return dest


def resolve_model(a):
    """--model as a file, a catalog name, or nothing (the default for this GPU); offers to download what is missing.
    A file that is a catalog model's file also gets that model's drafter (downloaded if missing) unless --no-drafter."""
    cat = json.load(open(CATALOG))
    if a.model and os.path.isfile(a.model):
        name = next((k for k, v in cat["models"].items() if os.path.basename(v["file"]["path"]) == os.path.basename(a.model)), None)
        if name is None or a.drafter or a.no_drafter: return
        m = dict(cat["models"][name]); m["file"] = None
    else:
        name, m = a.model, None
    if m is None and not name:
        vram = gpu_vram_gb() or 24
        name = cat["default"]["24"] if vram >= 20 else cat["default"]["16"]
    if m is None: m = cat["models"].get(name)
    if m is None:
        print(f"tqllm: model '{name}' not found as a file or catalog name. Available:", file=sys.stderr)
        for k, v in cat["models"].items(): print(f"  {k:16s} {v['file']['size'] / 1e9:5.1f} GB  {v['description']}", file=sys.stderr)
        sys.exit(1)
    keys = ("file",) + (() if a.no_drafter else ("drafter", "draft_vocab"))
    need = [(k, m[k]) for k in keys if m.get(k)]
    local = {k: find_local(e["path"]) for k, e in need}
    missing = [(k, e) for k, e in need if not local[k]]
    if missing:
        gb = sum(e.get("size", 0) for _, e in missing) / 1e9
        dest = a.models_dir or (os.environ.get("TQLLM_MODELS") or os.path.abspath("models"))
        print(f"tqllm: {name} ({m['description']}) needs {gb:.1f} GB from Hugging Face into {dest}:", file=sys.stderr)
        for _, e in missing: print(f"  {e['repo']}/{e['path']} ({e.get('size', 0) / 1e9:.2f} GB)", file=sys.stderr)
        if not a.yes and any(k == "file" for k, _ in missing):  # the drafter alone (< 1 GB) downloads without asking
            if not sys.stdin.isatty(): sys.exit("tqllm: run again with --yes to download (no terminal to ask)")
            if input("Download now? [y/N] ").strip().lower() not in ("y", "yes"): sys.exit(1)
        for k, e in missing: local[k] = download(e, dest)
    if local.get("file"): a.model = local["file"]
    if not a.drafter and local.get("drafter"): a.drafter, a.draft_vocab = local["drafter"], local.get("draft_vocab")
    print(f"tqllm: {name}: {a.model}" + (f" + drafter {os.path.basename(a.drafter)}" if a.drafter else ""), file=sys.stderr)


def get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def server_ok(port, name):
    try:
        get(f"http://127.0.0.1:{port}/health")
        models = get(f"http://127.0.0.1:{port}/v1/models").get("data", [])
        return any(m.get("id") == name for m in models)
    except Exception:
        return False


def start_server(a):
    """Returns (process, container name or None); waits for /health with a timeout and a liveness check."""
    args = ["--context", str(a.context), "--np", "1", "--reasoning", a.reasoning, "--max-tokens", str(a.max_tokens),
            "--name", NAME, "--prompt-cache", "on"] + a.server_arg
    if a.native:
        cmd = [a.native, "--model", a.model, "--port", str(a.port), "--host", "127.0.0.1"] + args
        if a.drafter: cmd += ["-md", a.drafter, "--draft-vocab", a.draft_vocab]
        proc, cname = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(a.log, "w")), None
    else:
        mdir, mfile = os.path.split(os.path.abspath(a.model))
        cname = f"tqllm-launch-{os.getpid()}"
        env = ["-e", f"MODEL_FILE=/models/{mfile}", "-e", f"CONTEXT={a.context}", "-e", f"REASONING={a.reasoning}",
               "-e", f"MAX_TOKENS={a.max_tokens}"]
        vols = ["-v", f"{mdir}:/models:ro"]
        if a.drafter:
            ddir, dfile = os.path.split(os.path.abspath(a.drafter)); vdir, vfile = os.path.split(os.path.abspath(a.draft_vocab))
            vols += ["-v", f"{ddir}:/drafter:ro", "-v", f"{vdir}:/vocab:ro"]
            env += ["-e", f"DRAFT_MODEL_FILE=/drafter/{dfile}", "-e", f"DRAFT_VOCAB_FILE=/vocab/{vfile}"]
        cmd = ["docker", "run", "--rm", "--name", cname, "--gpus", "all", "-p", f"127.0.0.1:{a.port}:8086"] + vols + env + \
              [a.image, "--name", NAME, "--prompt-cache", "on"] + a.server_arg
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(a.log, "w"))
    t0 = time.time()
    while time.time() - t0 < a.start_timeout:
        if proc.poll() is not None:
            sys.exit(f"tqllm: the server exited during start-up; see {a.log}")
        if server_ok(a.port, NAME):
            fit = get(f"http://127.0.0.1:{a.port}/health").get("fit")
            print(f"tqllm: server ready on 127.0.0.1:{a.port}" + (f" ({fit})" if fit else ""), file=sys.stderr)
            return proc, cname
        time.sleep(2)
    stop_server(proc, cname)
    sys.exit(f"tqllm: the server did not answer within {a.start_timeout} s; see {a.log}")


def stop_server(proc, cname):
    if cname: subprocess.run(["docker", "stop", "-t", "5", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(20)
        except subprocess.TimeoutExpired: proc.kill()


# ---- Anthropic Messages -> OpenAI chat bridge (for Claude Code) ----
def api_ok(base, path):
    """True when the server answers `path` (a POST with an empty body gets 400, an unknown path 404)."""
    req = urllib.request.Request(base + path, b"{}", {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        return e.code != 404
    except Exception:
        return False
    return True


def agent_command(a, base, env):
    """The agent's command line; its provider settings go into `env` or the command line (no config file is written)."""
    tool, extra = a.agent, a.agent_args
    if tool == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json", "model": f"{NAME}/{NAME}",
               "provider": {NAME: {"npm": "@ai-sdk/openai-compatible", "name": "tqLLM (local)",
                                   "options": {"baseURL": base + "/v1", "timeout": False, "chunkTimeout": 900000}, "models": {NAME: {"name": os.path.basename(a.model)}}}}}
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)
        cmd = ["opencode"] + extra
    elif tool == "codex":  # the server speaks the Responses API (wire_api = "responses"); no key needed
        cmd = ["codex", "-c", f"model_providers.{NAME}.name=tqLLM", "-c", f"model_providers.{NAME}.base_url={base}/v1",
               "-c", f"model_providers.{NAME}.wire_api=responses", "-c", f"model_provider={NAME}",
               "-c", f"model_context_window={a.context}", "-m", NAME] + extra
    elif tool == "claude":  # the server speaks the Anthropic Messages API (/v1/messages)
        env.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": "tqllm-local", "ANTHROPIC_MODEL": NAME,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": NAME, "ANTHROPIC_SMALL_FAST_MODEL": NAME,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
        cmd = ["claude"] + extra
    return cmd


def run_agent(a, base):
    tool, env = a.agent, dict(os.environ)
    cmd = agent_command(a, base, env)
    if not shutil.which(cmd[0]): sys.exit(f"tqllm: {cmd[0]} is not installed")
    if tool in ("codex", "claude") and not api_ok(base, "/v1/responses" if tool == "codex" else "/v1/messages"):
        sys.exit(f"tqllm: the server on {base} has no {'Responses' if tool == 'codex' else 'Messages'} API; use image 1.0.1 or newer")
    return subprocess.call(cmd, env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["launch", "serve"])
    p.add_argument("agent", nargs="?", choices=["opencode", "codex", "claude"])
    p.add_argument("--model", help="GGUF file or catalog name (models.json); default: the catalog default for this GPU")
    p.add_argument("--models-dir", help="where downloads go (default ./models, or $TQLLM_MODELS)")
    p.add_argument("--yes", action="store_true", help="download missing catalog files without asking")
    p.add_argument("--list", action="store_true", help="print the model catalog and exit")
    p.add_argument("--drafter"); p.add_argument("--draft-vocab")
    p.add_argument("--no-drafter", action="store_true", help="run without the catalog model's drafter")
    p.add_argument("--context", type=int, default=65536, help="tokens (default 65536: agent prompts are long)")
    p.add_argument("--max-tokens", type=int, default=8192); p.add_argument("--reasoning", default="on")
    p.add_argument("--port", type=int, default=8086); p.add_argument("--image", default=IMAGE)
    p.add_argument("--native", help="path to a tqserve binary instead of the Docker image")
    p.add_argument("--server-arg", action="append", default=[], help="extra tqserve option (repeatable)")
    p.add_argument("--start-timeout", type=int, default=600); p.add_argument("--log", default="/tmp/tqllm-launch.log")
    if "--list" in sys.argv:
        cat = json.load(open(CATALOG))
        for k, v in cat["models"].items():
            print(f"{k:16s} {v['file']['size'] / 1e9:5.1f} GB  {v['description']}  {'(local)' if find_local(v['file']['path']) else ''}")
        return 0
    a, rest = p.parse_known_args()
    a.agent_args = rest[1:] if rest[:1] == ["--"] else rest
    if a.command == "launch" and not a.agent: p.error("launch needs an agent: opencode, codex or claude")
    resolve_model(a)
    if a.drafter and not a.draft_vocab: p.error("--drafter needs --draft-vocab")
    def interrupt(*_): raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # background jobs start with SIGINT ignored
        signal.signal(sig, interrupt)
    proc = cname = None
    if server_ok(a.port, NAME):
        print(f"tqllm: reusing the server on 127.0.0.1:{a.port}", file=sys.stderr)
    else:
        proc, cname = start_server(a)
    base = f"http://127.0.0.1:{a.port}"
    try:
        if a.command == "serve":
            print("tqllm: serving; Ctrl+C to stop", file=sys.stderr); signal.pause(); return 0
        return run_agent(a, base)
    except KeyboardInterrupt:
        return 130
    finally:
        if proc: stop_server(proc, cname)


if __name__ == "__main__":
    sys.exit(main())
