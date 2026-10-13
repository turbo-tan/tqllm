#!/usr/bin/env python3
"""tqllm launch: start (or reuse) a local tqserve and run a coding agent against it, like `ollama launch`.

  tqllm.py launch opencode --model FILE.gguf [--drafter FILE --draft-vocab FILE] [--context 65536] [-- agent args]
  tqllm.py launch codex    --model FILE.gguf ...
  tqllm.py launch claude   --model FILE.gguf ...   (Claude Code: an in-process Anthropic Messages -> OpenAI bridge)
  tqllm.py serve           --model FILE.gguf ...   (just the server, until Ctrl+C)
  tqllm.py --list                                  (the model catalog: models.json, local copies marked)

--model takes a file, a catalog name (qwen3.8-27b, qwen3.8-27b-s) or nothing (the catalog default for this GPU). Catalog
files are looked up in ./models, ~/models and $TQLLM_MODELS; missing ones are offered for download from Hugging Face
(size and sha256 checked), --yes downloads without asking.

The server comes from the Docker image (default) or a native build (--native PATH/tqserve). A server already answering
on --port with the same model name is reused and left running; one this command started is stopped when the agent exits.
The agents' own configuration files are never written: each gets its provider through environment / command line.
"""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IMAGE = "ghcr.io/turbo-tan/tqllm:beta-015a14a-cu13"
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
    """--model as a file, a catalog name, or nothing (the default for this GPU); offers to download what is missing."""
    if a.model and os.path.isfile(a.model): return
    cat = json.load(open(CATALOG))
    name = a.model
    if not name:
        vram = gpu_vram_gb() or 24
        name = cat["default"]["24"] if vram >= 20 else cat["default"]["16"]
    m = cat["models"].get(name)
    if m is None:
        print(f"tqllm: model '{name}' not found as a file or catalog name. Available:", file=sys.stderr)
        for k, v in cat["models"].items(): print(f"  {k:16s} {v['file']['size'] / 1e9:5.1f} GB  {v['description']}", file=sys.stderr)
        sys.exit(1)
    need = [(k, m[k]) for k in ("file", "drafter", "draft_vocab") if k in m]
    local = {k: find_local(e["path"]) for k, e in need}
    missing = [(k, e) for k, e in need if not local[k]]
    if missing:
        gb = sum(e.get("size", 0) for _, e in missing) / 1e9
        dest = a.models_dir or (os.environ.get("TQLLM_MODELS") or os.path.abspath("models"))
        print(f"tqllm: {name} ({m['description']}) needs {gb:.1f} GB from Hugging Face into {dest}:", file=sys.stderr)
        for _, e in missing: print(f"  {e['repo']}/{e['path']} ({e.get('size', 0) / 1e9:.2f} GB)", file=sys.stderr)
        if not a.yes:
            if not sys.stdin.isatty(): sys.exit("tqllm: run again with --yes to download (no terminal to ask)")
            if input("Download now? [y/N] ").strip().lower() not in ("y", "yes"): sys.exit(1)
        for k, e in missing: local[k] = download(e, dest)
    a.model = local["file"]
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
def text_of(content):
    if isinstance(content, str): return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def to_openai(req):
    msgs = []
    if req.get("system"): msgs.append({"role": "system", "content": text_of(req["system"])})
    for m in req.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            msgs.append({"role": m["role"], "content": c}); continue
        if m["role"] == "assistant":
            calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                     for b in c if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text_of(c) or None}
            if calls: msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            for b in c:
                if b.get("type") == "tool_result":
                    rc = b.get("content")
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                 "content": rc if isinstance(rc, str) else text_of(rc)})
            t = text_of(c)
            if t: msgs.append({"role": "user", "content": t})
    out = {"model": NAME, "messages": msgs, "max_tokens": req.get("max_tokens", 4096), "stream": bool(req.get("stream"))}
    for k in ("temperature", "top_p"):
        if k in req: out[k] = req[k]
    if req.get("stop_sequences"): out["stop"] = req["stop_sequences"]
    if req.get("tools"):
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                          "parameters": t.get("input_schema", {})}} for t in req["tools"]]
        tc = req.get("tool_choice", {}).get("type")
        if tc == "any": out["tool_choice"] = "required"
        elif tc == "tool": out["tool_choice"] = {"type": "function", "function": {"name": req["tool_choice"]["name"]}}
    if out["stream"]: out["stream_options"] = {"include_usage": True}
    return out


STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def from_openai(r, model):
    ch = r["choices"][0]; msg = ch.get("message", {}); content = []
    if msg.get("content"): content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        try: inp = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError: inp = {"_raw": tc["function"].get("arguments")}
        content.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}", "name": tc["function"]["name"], "input": inp})
    u = r.get("usage", {})
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": "tool_use" if msg.get("tool_calls") else STOP.get(ch.get("finish_reason"), "end_turn"),
            "stop_sequence": None, "usage": {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}}


class Bridge(BaseHTTPRequestHandler):
    upstream = ""
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(200, {"status": "ok"})
    def do_HEAD(self): self.send_response(200); self.end_headers()
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.split("?")[0]
        if path.endswith("/count_tokens"):
            n = len(json.dumps(req.get("messages", []))) + len(json.dumps(req.get("system", ""))) + len(json.dumps(req.get("tools", [])))
            return self._send(200, {"input_tokens": n // 4})
        if not path.endswith("/messages"): return self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})
        oreq = to_openai(req); model = req.get("model", NAME)
        r = urllib.request.Request(self.upstream + "/v1/chat/completions", json.dumps(oreq).encode(), {"Content-Type": "application/json"})
        try:
            up = urllib.request.urlopen(r, timeout=3600)
        except urllib.error.HTTPError as e:
            return self._send(e.code, {"type": "error", "error": {"type": "api_error", "message": e.read().decode(errors="replace")[:500]}})
        if not oreq["stream"]:
            return self._send(200, from_openai(json.load(up), model))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
        def ev(name, data): self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()); self.wfile.flush()
        mid = f"msg_{uuid.uuid4().hex[:24]}"
        ev("message_start", {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                                                                 "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
        idx, open_kind, tools, finish, usage = -1, None, {}, None, {}
        def close():
            nonlocal open_kind
            if open_kind is not None: ev("content_block_stop", {"type": "content_block_stop", "index": idx}); open_kind = None
        for raw in up:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if dl.get("content"):
                    if open_kind != "text":
                        close(); idx += 1; open_kind = "text"
                        ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": dl["content"]}})
                for tc in dl.get("tool_calls") or []:
                    k = tc.get("index", 0)
                    if k not in tools:
                        close(); idx += 1; open_kind = ("tool", k); tools[k] = idx
                        ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {
                            "type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}", "name": tc.get("function", {}).get("name", ""), "input": {}}})
                    args = tc.get("function", {}).get("arguments")
                    if args: ev("content_block_delta", {"type": "content_block_delta", "index": tools[k], "delta": {"type": "input_json_delta", "partial_json": args}})
                if ch.get("finish_reason"): finish = ch["finish_reason"]
        close()
        stop = "tool_use" if tools else STOP.get(finish, "end_turn")
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                             "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
        ev("message_stop", {"type": "message_stop"})


def run_agent(a, base):
    tool, extra = a.agent, a.agent_args
    env = dict(os.environ)
    if tool == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json", "model": f"{NAME}/{NAME}",
               "provider": {NAME: {"npm": "@ai-sdk/openai-compatible", "name": "TQLLM (local)",
                                   "options": {"baseURL": base + "/v1"}, "models": {NAME: {"name": os.path.basename(a.model)}}}}}
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)
        cmd = ["opencode"] + extra
    elif tool == "codex":
        cmd = ["codex", "-c", f"model_providers.{NAME}.name=TQLLM", "-c", f"model_providers.{NAME}.base_url={base}/v1",
               "-c", f"model_providers.{NAME}.wire_api=chat", "-c", f"model_provider={NAME}", "-m", NAME] + extra
    elif tool == "claude":
        Bridge.upstream = base
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        env.update({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{srv.server_address[1]}", "ANTHROPIC_AUTH_TOKEN": "tqllm-local",
                    "ANTHROPIC_MODEL": NAME, "ANTHROPIC_DEFAULT_HAIKU_MODEL": NAME, "ANTHROPIC_SMALL_FAST_MODEL": NAME,
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
        cmd = ["claude"] + extra
    if not shutil.which(cmd[0]): sys.exit(f"tqllm: {cmd[0]} is not installed")
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
