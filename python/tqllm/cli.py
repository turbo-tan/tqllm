"""tqllm serve [--qsa] [tqserve options]   serve on Apple Silicon; the first run downloads the model and its drafter
tqllm pull [--qsa]                       download them only
tqllm claude | codex [--port N] [args]   start a coding agent pointed at a running tqllm server"""
import os
import platform
import shutil
import sys
from pathlib import Path

MLX_REPO, TQ_REPO = "YTan2000/Qwen3.8-27B-MLX-tqLLM", "YTan2000/Qwen3.8-27B-TQ"
MODEL, MODEL_QSA, DRAFTER = "Qwen3.8-27B-MLX-4bit.gguf", "Qwen3.8-27B-MLX-4bit-QSA.gguf", "Qwen3.8-27B-DFlash2-MLX-4bit.gguf"
TQ_L = "Qwen3.8-27B-TQ_L.gguf"  # 14.7 GB: the file for Macs below LARGE_MAC_GB
LARGE_MAC_GB = 40
PORT = 8086  # tqserve's default
DOCKER = "docker run --gpus all -p 8080:8080 -v $HOME/models:/models ghcr.io/turbo-tan/tqllm:latest --model /models/<file>.gguf"


def models_dir() -> Path:
    return Path(os.environ.get("TQLLM_HOME", Path.home() / ".cache" / "tqllm")) / "models"


def server() -> Path:
    return Path(__file__).resolve().parent / "bin" / "tqserve"


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def has_option(args, *names):
    return any(a in names or any(a.startswith(n + "=") for n in names) for a in args)


def option(args, name, default=None):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def plan(qsa, ram):
    """(repo, file) pairs a default serve needs: the MLX 4-bit model on large Macs, TQ_L below; the drafter always."""
    model = (MLX_REPO, MODEL_QSA if qsa else MODEL) if ram >= LARGE_MAC_GB or not ram else (TQ_REPO, TQ_L)
    return [model, (MLX_REPO, DRAFTER)]


def fetch(files):
    d = models_dir()
    missing = [(r, f) for r, f in files if not (d / f).exists()]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit('huggingface_hub is missing: pip install "tqllm[mlx]"')
    d.mkdir(parents=True, exist_ok=True)
    for repo, f in missing:
        print(f"downloading {repo}/{f} -> {d} (first run only)", flush=True)
        hf_hub_download(repo, f, local_dir=d)


def serve_args(args, ram=None, download=fetch):
    """tqserve arguments: the default model and drafter (downloaded if missing) unless --model / -md are given, and a
    context of 65,536 unless --context is given (tqserve's own default is 2,048)."""
    out = [a for a in args if a != "--qsa"]
    ram = ram_gb() if ram is None else ram
    need = plan("--qsa" in args, ram)
    if has_option(out, "--model", "-m"):
        need = need[1:]
    if has_option(out, "-md", "--model-draft"):
        need = [x for x in need if x[1] != DRAFTER]
    download(need)
    d = models_dir()
    for repo, f in need:
        out = (["-md", str(d / f)] if f == DRAFTER else ["--model", str(d / f)]) + out
    if not has_option(out, "--context", "-c"):
        out += ["--context", "65536"]
    return out


def agent_env(name, port, env=None):
    """The environment that points a coding agent at the local server: Claude Code at /v1/messages, Codex at
    /v1/responses (both served by tqserve)."""
    e = dict(os.environ if env is None else env)
    base = f"http://127.0.0.1:{port}"
    if name == "claude":
        e.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_AUTH_TOKEN="tqllm")
    elif name == "codex":
        e.update(OPENAI_BASE_URL=base + "/v1", OPENAI_API_KEY="tqllm")
    else:
        raise SystemExit(f"no launcher for {name!r}: tqllm claude | tqllm codex")
    return e


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__ + f"\nmodels: {models_dir()} (TQLLM_HOME overrides)")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "pull":
        fetch(plan("--qsa" in rest, ram_gb()))
        print("done: tqllm serve" + (" --qsa" if "--qsa" in rest else ""))
        return 0
    if cmd == "serve":
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise SystemExit(f"this package carries the Apple Silicon server; on Linux use Docker:\n  {DOCKER}")
        exe = server()
        if not exe.exists():
            raise SystemExit(f"tqserve is missing from the package ({exe}): reinstall with pip install \"tqllm[mlx]\"")
        os.execv(str(exe), [str(exe)] + serve_args(rest))
    if cmd in ("claude", "codex"):
        port = int(option(rest, "--port", PORT))
        args = [a for i, a in enumerate(rest) if a != "--port" and (i == 0 or rest[i - 1] != "--port") and not a.startswith("--port=")]
        exe = shutil.which(cmd)
        if not exe:
            raise SystemExit(f"{cmd} is not installed (it is a separate tool); install it, then run tqllm {cmd}")
        os.execve(exe, [exe] + args, agent_env(cmd, port))
    raise SystemExit(f"unknown command {cmd!r}: tqllm serve | pull | claude | codex")


if __name__ == "__main__":
    sys.exit(main())
