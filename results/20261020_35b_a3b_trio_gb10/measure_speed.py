#!/usr/bin/env python3
"""One 35B-A3B model, one tqserve on the GB10 box, measured from the dev box.

  python3 measure_speed.py <name> <gguf-on-the-server> <port> [--reasoning on]

Speed rows come from tqserve's own llama.cpp-style timings (usage + timings are in every
non-streaming response and at the end of every SSE stream), so decode excludes the prompt and
first-token latency is measured on the stream, not inferred.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.environ.get("TQLLM_HOST", "127.0.0.1")
TQSERVE = "~/work/glm/build-cuda/tqserve"  # expanded by the remote shell
CONTEXT = int(os.environ.get("TQLLM_CONTEXT", "40960"))  # holds the 32768-token probe prompt + its answer
TARGETS = (4096, 16384, 32768)


def ssh(cmd, timeout=900):
    return subprocess.run(["ssh", SERVER, cmd], capture_output=True, text=True, timeout=timeout)


def ssh_detach(cmd):
    """Start a server on the box and return: ssh -fn -n would still hold the channel open."""
    subprocess.Popen(["ssh", "-n", SERVER, cmd], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tunnel(port):
    """Local -L tunnel to the server's port: the OpenAI calls then go to 127.0.0.1 on this box."""
    p = subprocess.Popen(["ssh", "-n", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:127.0.0.1:{port}",
                          SERVER, "-N"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(2)
    return p


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def post(port, payload, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())
    # SSE through an ssh tunnel: first content delta -> ttft; the stream closes with usage + timings.
    t0 = time.time()
    ttft = None
    out, text = {}, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            try:
                d = json.loads(body)
            except Exception:
                continue
            if d.get("usage"):
                out["usage"] = d["usage"]
            if d.get("timings"):
                out["timings"] = d["timings"]
            for ch in d.get("choices", []):
                delta = (ch.get("delta") or {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
    out["elapsed_s"] = time.time() - t0
    out["ttft_s"] = ttft
    out["text"] = "".join(text)[:400]
    return out


def wait_ready(port, log, timeout=600):
    for i in range(timeout):
        r = ssh(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health")
        if r.stdout.strip() == "200":
            log(f"server ready after {i + 1}s")
            return True
        time.sleep(1)
    return False


UNIT = "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon. "


def units(n):
    return UNIT * max(1, n)


def probe_ratio(port, name):
    """tokens per filler unit for this model's chat template + tokenizer."""
    d = post(port, {"model": name, "messages": [{"role": "user", "content": units(64)}],
                    "max_tokens": 1, "temperature": 0})
    pt = int(d.get("usage", {}).get("prompt_tokens", 0))
    return (pt / 64.0) if pt else 0.0


def filler(port, name, target, ratio):
    """the filler whose prompt lands nearest `target` tokens (the ratio was measured on this model)."""
    return units(max(1, int((target - 48) / max(ratio, 0.01))))


CHATS = ["What is the capital of France?", "Explain gravity in one sentence.", "What is 2+2?",
         "Name three primary colors.", "Write a haiku about the sea."]


def gguf_meta(path):
    """architecture / layer count / expert quant of a file on the server (header only)."""
    with open(os.path.join(HERE, "gguf_meta.py")) as f:
        src = f.read()
    r = subprocess.run(["ssh", SERVER, f"python3 - {path}"], input=src, capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout) if r.stdout.strip() else {"error": r.stderr.strip()[:200]}


def main():
    name, gguf, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    reasoning = "off"
    for a in sys.argv[4:]:
        if a.startswith("--reasoning="):
            reasoning = a.split("=", 1)[1]
    log = Log(os.path.join(HERE, "logs", f"{name}_speed.log"))
    args = " ".join(a for a in sys.argv[4:])
    log(f"start tqserve {name} port {port} reasoning={reasoning}: {gguf}")
    ssh_detach(f"cd ~/work/glm && setsid nohup {TQSERVE} --model {gguf} --host 127.0.0.1 --port {port} "
               f"--context {CONTEXT} --max-tokens 2048 --np 1 --reasoning {reasoning} --prompt-cache off "
               f"{args} > /tmp/tqs_{name}.log 2>&1 < /dev/null &")
    if not wait_ready(port, log):
        log("server did not come up: " + ssh("tail -30 /tmp/tqs_%s.log" % name).stdout)
        sys.exit(1)

    tun = tunnel(port)
    head = ssh(f"grep -m1 'engine turbo-tan/tqllm' /tmp/tqs_{name}.log").stdout.strip()
    res = {"model": name, "gguf": gguf, "port": int(port), "reasoning": reasoning,
           "context": CONTEXT, "server_log": f"/tmp/tqs_{name}.log",
           "engine": head.split("engine ", 1)[-1] if head else None,
           "source_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                            text=True, cwd=os.path.join(HERE, "..", "..")).stdout.strip(),
           "gguf_bytes": ssh(f"stat -c %s {gguf}").stdout.strip(),
           "sha256": ssh(f"sha256sum {gguf}").stdout.split()[0],
           "gguf_meta": gguf_meta(gguf),
           "host_free_gb_before": ssh("free -g | awk '/^Mem:/{print $7}'").stdout.strip(),
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    log(f"engine {res['engine']}, file sha256 {res['sha256'][:16]}...")

    # ---- decode: 5 short chats, engine timings (predicted_n / predicted_ms exclude the prompt) ----
    chats, pn, pm = [], 0, 0.0
    for c in CHATS:
        d = post(int(port), {"model": name, "messages": [{"role": "user", "content": c}],
                             "max_tokens": 128, "temperature": 0})
        u, t = d.get("usage", {}), d.get("timings", {})
        row = {"prompt": c, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "decode_tok_s": round(t.get("predicted_per_second", 0), 2),
               "content_head": (d.get("choices", [{}])[0].get("message", {}).get("content") or "")[:80]}
        chats.append(row)
        pn += max(0, int(t.get("predicted_n", 0)) - 1)
        pm += float(t.get("predicted_ms", 0))
        log(f"chat {row['prompt_tokens']}+{row['completion_tokens']} tok, decode {row['decode_tok_s']} tok/s")
    res["chats"] = chats
    res["decode"] = {"chats": len(chats), "predicted_n_minus_1": pn, "predicted_ms": round(pm, 1),
                     "tok_s": round(pn * 1000.0 / pm, 2) if pm else 0.0}
    log(f"decode over the 5 chats: {res['decode']['tok_s']} tok/s ({pn} tok / {pm:.0f} ms)")

    # ---- prefill + first token at 4K / 16K / 32K (streamed, engine timings for the rate) ----
    ratio = probe_ratio(int(port), name)
    res["tokens_per_filler_unit"] = round(ratio, 3)
    log(f"filler ratio {ratio:.2f} tok/unit")
    pre = {}
    for ctx in TARGETS:
        d = post(int(port), {"model": name, "stream": True,
                             "messages": [{"role": "user", "content": filler(int(port), name, ctx, ratio) +
                                          "\nAnswer with just OK."}],
                             "max_tokens": 8, "temperature": 0}, stream=True)
        u, t = d.get("usage", {}), d.get("timings", {})
        pre[ctx] = {"prompt_tokens": u.get("prompt_tokens"), "prefill_tok_s": round(t.get("prompt_per_second", 0), 1),
                    "prompt_ms": round(float(t.get("prompt_ms", 0)), 1),
                    "first_token_s": round(d["ttft_s"], 3) if d.get("ttft_s") else None}
        log(f"prefill {ctx}: {pre[ctx]['prompt_tokens']} tok in {pre[ctx]['prompt_ms']} ms = "
            f"{pre[ctx]['prefill_tok_s']} tok/s, first token {pre[ctx]['first_token_s']} s")
    res["prefill"] = pre
    res["host_free_gb_after"] = ssh("free -g | awk '/^Mem:/{print $7}'").stdout.strip()

    tun.terminate()
    log("stop tqserve")
    ssh(f"pkill -f '[t]qserve.*--port {port}' || true")
    time.sleep(5)
    with open(os.path.join(HERE, "json", f"{name}_speed.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"wrote json/{name}_speed.json")


if __name__ == "__main__":
    main()
