"""tqllm [serve] [--qsa | --flash-next] [--native PATH] [--foreground] [tqserve options]
                                         start the server in the background (also with no command) and return once it
                                         answers; the first run downloads the model and drafter. --foreground: stay attached
    Apple Silicon: the packaged Metal server. Linux + NVIDIA: the Docker image (or --native PATH/tqserve), the model picked
    by the card's memory (Qwen3.8-27B TQ3_4S-v2 on 20 GB+, TQ_S below), or Qwen3.8 Flash-Next with --flash-next.
tqllm pull [--qsa | --flash-next]       download them only
tqllm claude | codex | opencode [--port N] [--native PATH] [args]
                                         start a coding agent pointed at the server; starts the server first when none
                                         answers on the port (it keeps running: tqllm stop)
tqllm stop                               stop a server tqllm started (its log: ~/.cache/tqllm/server.log)"""
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MLX_REPO, TQ_REPO = "YTan2000/Qwen3.8-27B-MLX-tqLLM", "YTan2000/Qwen3.8-27B-TQ"
DRAFTER = "Qwen3.8-27B-DFlash2-MLX-4bit.gguf"
MODEL, MODEL_QSA = "Qwen3.8-27B-MLX-4bit.gguf", "Qwen3.8-27B-MLX-4bit-QSA.gguf"  # mlx-community 4-bit (+ QSA indexers)
TQ_L = "Qwen3.8-27B-TQ_L.gguf"  # 14.7 GB: the file for Macs below LARGE_MAC_GB
TQ_L_QSA = "Qwen3.8-27B-TQ_L-QSA.gguf"  # 14.8 GB: TQ_L with the QSA indexers (preferred when published)
LARGE_MAC_GB = 40
PORT = 8086  # tqserve's default
IMAGE = "ghcr.io/turbo-tan/tqllm:1.5.0"
# Linux / NVIDIA: (repo, file) per model; files keep their repo subfolder under the models folder
TQ34S_REPO, TQ34S = "YTan2000/Qwen3.8-27B-TQ3_4S", "Qwen3.8-27B-TQ3_4S-v2.gguf"  # 24 GB cards
TQ_S = "Qwen3.8-27B-TQ_S.gguf"  # 16 GB cards
TQ_DRAFTER, TQ_VOCAB = "drafters/Qwen3.8-27B-DFlash2-TQ3.gguf", "mtp-vocab/atx_65536.txt"
LARGE_CARD_GB = 20
FN_REPO = "YTan2000/Qwen3.8-Flash-Next-tqLLM"
FN_FILES = ["mixed/Qwen3.8-Flash-Next-tqLLM-00001-of-00002.gguf", "mixed/Qwen3.8-Flash-Next-tqLLM-00002-of-00002.gguf",
            "flashnext-tqllm.tqfn", "mtp-blk48-iq4.gguf", "flashnext-tokenizer.gguf", "atx_65536.txt"]


def models_dir() -> Path:
    return Path(os.environ.get("TQLLM_HOME", Path.home() / ".cache" / "tqllm")) / "models"


def state_dir() -> Path:
    return Path(os.environ.get("TQLLM_HOME", Path.home() / ".cache" / "tqllm"))


def server() -> Path:
    return Path(__file__).resolve().parent / "bin" / "tqserve"


def has_option(args, *names):
    return any(a in names or any(a.startswith(n + "=") for n in names) for a in args)


def option(args, name, default=None):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def remote_exists(repo, f, timeout=10.0):
    """True when huggingface.co serves the file (HEAD, redirects followed); False offline or when it is missing."""
    req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{f}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def model_candidates(qsa=False, ram=None):
    """Mac model files in order of preference: TQ_L-QSA (smaller, fastest past 48K), then the mlx-community 4-bit file
    (its QSA file with --qsa) on Macs with LARGE_MAC_GB or more, then TQ_L."""
    ram = ram_gb() if ram is None else ram
    out = [(TQ_REPO, TQ_L_QSA)]
    if ram >= LARGE_MAC_GB or not ram:
        out.append((MLX_REPO, MODEL_QSA if qsa else MODEL))
    out.append((TQ_REPO, TQ_L))
    return out


def pick(candidates, exists=remote_exists):
    """The first candidate already in the models folder, else the first one huggingface.co serves."""
    d = models_dir()
    for c in candidates:
        if (d / c[1]).exists():
            return c
    for c in candidates:
        if exists(*c):
            return c
    raise SystemExit("tqllm: none of " + ", ".join(f for _, f in candidates) +
                     " is downloaded or reachable on huggingface.co; check the network or pass --model PATH")


def plan(qsa=False, ram=None, exists=remote_exists):
    """(repo, file) pairs a default Mac serve needs: the best model file that is local or published, and the drafter."""
    return [pick(model_candidates(qsa, ram), exists), (MLX_REPO, DRAFTER)]


def hub_model(value, download, list_files=None):
    """A --model value of the form owner/repo/file.gguf or owner/repo (Hugging Face) as a local path, downloaded on first
    use; None when it is not one (a local path)."""
    if Path(value).exists() or value.count("/") < 1 or value.startswith(("/", ".", "~")):
        return None
    parts = value.split("/")
    repo = "/".join(parts[:2])
    if len(parts) > 2:
        f = "/".join(parts[2:])
    else:
        if list_files is None:
            try:
                from huggingface_hub import list_repo_files
            except ImportError:
                raise SystemExit('huggingface_hub is missing: pip install "huggingface_hub>=0.24"')
            list_files = list_repo_files
        ggufs = [x for x in list_files(repo) if x.endswith(".gguf") and "draft" not in x.lower()]
        if len(ggufs) != 1:
            raise SystemExit(f"tqllm: {repo} has {len(ggufs)} model files; pick one: --model {repo}/FILE\n  " + "\n  ".join(ggufs))
        f = ggufs[0]
    download([(repo, f)])
    return str(models_dir() / f)


def vram_gb() -> float:
    """The first NVIDIA card's memory (0 without nvidia-smi)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return float(out[0]) / 1024 if out else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def linux_plan(vram, flash_next=False):
    """((repo, file) pairs, tqserve arguments with paths relative to the models folder) for a Linux / NVIDIA serve."""
    if flash_next:
        files = [(FN_REPO, f) for f in FN_FILES]
        return files, ["--model", "flashnext-tqllm.tqfn", "--tokenizer", "flashnext-tokenizer.gguf"]
    model = (TQ34S_REPO, TQ34S) if vram >= LARGE_CARD_GB or not vram else (TQ_REPO, TQ_S)
    files = [model, (TQ_REPO, TQ_DRAFTER), (TQ_REPO, TQ_VOCAB)]
    return files, ["--model", model[1], "-md", TQ_DRAFTER, "--draft-vocab", TQ_VOCAB]


def linux_command(args, vram=None, native=None, download=None):
    """The command that serves on Linux: the files of linux_plan downloaded (unless --model is given), then the Docker image
    with the models folder at /models, or a native tqserve with absolute paths."""
    download = download or fetch
    flash_next = "--flash-next" in args
    out = [a for a in args if a != "--flash-next"]
    files, model_args = linux_plan(vram_gb() if vram is None else vram, flash_next)
    if has_option(out, "--model", "-m"):
        files, model_args = [], []
    download(files)
    port = option(out, "--port", str(PORT))
    if not has_option(out, "--context", "-c"):
        out += ["--context", "32768" if flash_next else "65536"]
    d = models_dir()
    if native:
        rel = [str(d / a) if i % 2 and not a.startswith("-") else a for i, a in enumerate(model_args)]
        return [native] + rel + out
    rel = ["/models/" + a if i % 2 and not a.startswith("-") else a for i, a in enumerate(model_args)]
    if not has_option(out, "--host"):
        out += ["--host", "0.0.0.0"]
    if not has_option(out, "--port"):
        out += ["--port", port]
    return ["docker", "run", "--rm", "--gpus", "all", "-p", f"127.0.0.1:{port}:{port}", "-v", f"{d}:/models:ro",
            "--entrypoint", "tqserve", IMAGE] + rel + out


def fetch(files):
    d = models_dir()
    missing = [(r, f) for r, f in files if not (d / f).exists()]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('huggingface_hub is missing: pip install "huggingface_hub>=0.24"')
    d.mkdir(parents=True, exist_ok=True)
    for repo, f in missing:
        print(f"downloading {repo}/{f} -> {d} (first run only)", flush=True)
        hf_hub_download(repo, f, local_dir=d)


def serve_args(args, ram=None, download=fetch, exists=remote_exists):
    """tqserve arguments: the default model and drafter (downloaded if missing) unless --model / -md are given, and a
    context of 65,536 unless --context is given (tqserve's own default is 2,048)."""
    out = [a for a in args if a != "--qsa"]
    for name in ("--model", "-m"):
        if name in out and out.index(name) + 1 < len(out):
            i = out.index(name) + 1
            local = hub_model(out[i], download)
            if local:
                out[i] = local
    if has_option(out, "--model", "-m"):
        need = [(MLX_REPO, DRAFTER)]
    else:
        need = plan("--qsa" in args, ram, exists)
    if has_option(out, "-md", "--model-draft"):
        need = [x for x in need if x[1] != DRAFTER]
    download(need)
    d = models_dir()
    for repo, f in need:
        out = (["-md", str(d / f)] if f == DRAFTER else ["--model", str(d / f)]) + out
    if not has_option(out, "--context", "-c"):
        out += ["--context", "65536"]
    return out


AGENTS = ("claude", "codex", "opencode")
OPENCODE_MODEL = "tqllm/qwen3.8-27b"


def opencode_config(port):
    """OpenCode's provider entry for the local server (OpenAI-compatible chat completions)."""
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {"tqllm": {"npm": "@ai-sdk/openai-compatible", "name": "tqLLM",
                                   "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                                   "models": {"qwen3.8-27b": {"name": "Qwen3.8-27B (tqLLM)"}}}}}


def agent_env(name, port, env=None, model=None):
    """The environment that points a coding agent at the local server: Claude Code at /v1/messages, Codex at
    /v1/responses, OpenCode at /v1/chat/completions through a generated config (all served by tqserve)."""
    e = dict(os.environ if env is None else env)
    base = f"http://127.0.0.1:{port}"
    if name == "claude":
        e.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_AUTH_TOKEN="tqllm")
        if model:  # the served model for every Claude Code role, over a model name left in the shell or its settings
            e.update({k: model for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                                         "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")})
    elif name == "codex":
        e.update(OPENAI_BASE_URL=base + "/v1", OPENAI_API_KEY="tqllm")
    elif name == "opencode":
        cfg = state_dir() / "opencode.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(opencode_config(port), indent=2))
        e.update(OPENCODE_CONFIG=str(cfg))
    else:
        raise SystemExit(f"no launcher for {name!r}: tqllm " + " | tqllm ".join(AGENTS))
    return e


def server_context(port, timeout=3.0):
    """Positions per request that the server on the port reports at /health (tqserve 1.5.0+), or None."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            n = json.load(r).get("context")
        return int(n) if n else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def claude_settings(port, model, context=None):
    """Claude Code settings that point it at the local server. Passed with --settings: command-line settings rank above
    ~/.claude/settings.json, whose env block (e.g. another gateway's ANTHROPIC_BASE_URL / ANTHROPIC_MODEL) would otherwise
    override the environment tqllm sets."""
    env = {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}", "ANTHROPIC_AUTH_TOKEN": "tqllm", "ANTHROPIC_API_KEY": "tqllm"}
    if model:
        env.update({k: model for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                                       "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")})
    if context:  # Claude Code does not know the served model's window (it would assume 200K): compact at the server's
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(context)
    out = {"env": env}
    if model:
        out["model"] = model
    return json.dumps(out, separators=(",", ":"))


def agent_args(name, args, model=None, port=PORT, context=None):
    """OpenCode starts on the tqLLM model; Claude Code on the served model with settings that point it at the local
    server (unless the user passes their own --model / --settings)."""
    if name == "opencode" and not has_option(args, "-m", "--model"):
        return ["-m", OPENCODE_MODEL] + list(args)
    if name == "claude":
        out = list(args)
        if model and not has_option(args, "--model"):
            out = ["--model", model] + out
        if not has_option(args, "--settings"):
            out = ["--settings", claude_settings(port, model, context)] + out
        else:
            print("tqllm: your --settings is used as given; it must point Claude Code at "
                  f"http://127.0.0.1:{port}", file=sys.stderr)
        return out
    return list(args)


def served_model(port, timeout=3.0):
    """(model id, owner) that the server on the port reports at /v1/models, or None."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            first = json.load(r)["data"][0]
        return str(first.get("id", "")), str(first.get("owned_by", ""))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def tqllm_model(port, cmd, served=served_model):
    """The model id of tqLLM's server on the port; stops when another program answers there."""
    info = served(port)
    if info is None or info[1] != "tqllm":
        what = f"it serves {info[0]!r}" if info else "it does not answer /v1/models"
        raise SystemExit(f"tqllm: port {port} is not tqLLM's server ({what}); stop that server, or run tqllm {cmd} --port N")
    return info[0]


def healthy(port, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def server_command(args, mac, native=None):
    """The serve command line (tqllm serve's), for this machine."""
    if mac:
        exe = server()
        if not exe.exists():
            raise SystemExit(f"tqserve is missing from the package ({exe}): reinstall: curl -fsSL https://tqllm.com/install.sh | bash")
        return [str(exe)] + serve_args(args)
    if sys.platform != "linux":
        raise SystemExit("tqllm serve runs on Apple Silicon Macs and on Linux with an NVIDIA card")
    if not native and not shutil.which("docker"):
        raise SystemExit("Docker is not installed: install it with the NVIDIA container toolkit, or pass --native PATH/tqserve")
    return linux_command(args, native=native)


def ensure_server(port, start, is_up=healthy, wait=1800.0, poll=2.0, out=sys.stderr):
    """Nothing when a server answers on the port; otherwise start(port) launches one in the background and this waits for
    /health (the first run downloads the model first, in start). Returns True when it started one."""
    if is_up(port):
        return False
    proc = start(port)
    print(f"tqllm: starting the server on port {port} (log: {state_dir() / 'server.log'})", file=out, flush=True)
    t0 = time.monotonic()
    while not is_up(port):
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"tqllm: the server exited (code {proc.returncode}); see {state_dir() / 'server.log'}")
        if time.monotonic() - t0 > wait:
            raise SystemExit(f"tqllm: no server on port {port} after {wait:.0f} s; see {state_dir() / 'server.log'}")
        time.sleep(poll)
    print(f"tqllm: server ready in {time.monotonic() - t0:.0f} s (it keeps running: tqllm stop)", file=out, flush=True)
    return True


def start_background(port, mac, native=None, args=()):
    """Launch the server detached, output to state_dir()/server.log, pid in server.pid."""
    cmd = server_command(list(args) + ["--port", str(port)], mac, native)
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    log = open(d / "server.log", "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (d / "server.pid").write_text(str(proc.pid))
    return proc


def stop_server():
    f = state_dir() / "server.pid"
    if not f.exists():
        print("tqllm: no server started by tqllm")
        return 1
    pid = int(f.read_text())
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"tqllm: stopped the server (pid {pid})")
    except ProcessLookupError:
        print("tqllm: the server had already exited")
    f.unlink(missing_ok=True)
    return 0


def without(args, name):
    return [a for i, a in enumerate(args) if a != name and (i == 0 or args[i - 1] != name) and not a.startswith(name + "=")]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__ + f"\nmodels: {models_dir()} (TQLLM_HOME overrides)")
        return 0
    if not argv or argv[0].startswith("-"):  # no command: serve, options included (tqllm --qsa)
        argv = ["serve"] + argv
    cmd, rest = argv[0], argv[1:]
    mac = sys.platform == "darwin" and platform.machine() == "arm64"
    if cmd == "pull":
        fetch(plan("--qsa" in rest) if mac else linux_plan(vram_gb(), "--flash-next" in rest)[0])
        print("done: tqllm serve" + "".join(f" {x}" for x in ("--qsa", "--flash-next") if x in rest))
        return 0
    if cmd == "stop":
        return stop_server()
    if cmd == "serve":
        native = None if mac else option(rest, "--native")
        args = without(rest, "--native")
        if "--foreground" in args:  # attached: the server's own output, Ctrl+C stops it
            line = server_command([a for a in args if a != "--foreground"], mac, native)
            os.execvp(line[0], line)
        port = int(option(args, "--port", PORT))
        extra = without(args, "--port")
        if not ensure_server(port, lambda p: start_background(p, mac, native, extra)):
            print(f"tqllm: a server already answers on port {port} (tqllm stop ends one tqllm started)")
        print(f"tqllm: serving http://127.0.0.1:{port}/v1 (monitor: http://127.0.0.1:{port}/); stop: tqllm stop; log: {state_dir() / 'server.log'}")
        return 0
    if cmd in AGENTS:
        port = int(option(rest, "--port", PORT))
        native = None if mac else option(rest, "--native")
        args = without(without(rest, "--port"), "--native")
        exe = shutil.which(cmd)
        if not exe:
            raise SystemExit(f"{cmd} is not installed (it is a separate tool); install it, then run tqllm {cmd}")
        ensure_server(port, lambda p: start_background(p, mac, native))
        model = tqllm_model(port, cmd)
        context = server_context(port) if cmd == "claude" else None
        os.execve(exe, [exe] + agent_args(cmd, args, model, port, context), agent_env(cmd, port, model=model))
    raise SystemExit(f"unknown command {cmd!r}: tqllm serve | pull | stop | " + " | ".join(AGENTS))


if __name__ == "__main__":
    sys.exit(main())
