#!/usr/bin/env python3
"""Release performance matrix: one tqserve per (model, mode), wiki-summary prompts at fixed lengths, N repeats, median.
Every JSON line carries the engine commit read from /health, the GPU name, model file and size, so a row cannot be
published without its build. Greedy, reasoning on, 400 output tokens (decode clock = server timings).
Usage: relbench.py --bin TQSERVE --gpu LABEL --out RESULTS.jsonl --wiki wiki.test.raw
         --model LABEL=PATH [--model ...] [--drafter PATH --vocab PATH] [--lengths 2000,8000,32000] [--repeats 3]
         [--context 40960] [--port 18760] [--extra "--kv tq3"] [--modes nospec,dflash2]
   or, against a server that is already running (for example the Docker image):
       relbench.py --url http://localhost:8086 --gpu LABEL --out RESULTS.jsonl --wiki wiki.test.raw --model LABEL=PATH
         --mode dflash2 (or nospec: name what the server was started with)
"""
import argparse, atexit, json, os, re, shlex, signal, statistics, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--bin"); ap.add_argument("--url"); ap.add_argument("--mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--wiki", required=True); ap.add_argument("--model", action="append", required=True)
ap.add_argument("--drafter"); ap.add_argument("--vocab"); ap.add_argument("--lengths", default="2000,8000,32000")
ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--context", type=int, default=40960)
ap.add_argument("--port", type=int, default=18760); ap.add_argument("--extra", default="--kv tq3")
ap.add_argument("--max-tokens", type=int, default=400); ap.add_argument("--modes", default="nospec,dflash2")
a = ap.parse_args()
if not a.bin and not (a.url and a.mode):
    sys.exit("give --bin, or --url with --mode")
URL = a.url.rstrip("/") if a.url else "http://127.0.0.1:%d" % a.port
wiki = re.sub(r" @(.)@ ", r"\1", open(a.wiki, errors="ignore").read()).replace(" = = ", "\n## ").replace(" = ", "\n# ")


def get(path, timeout=5):
    return json.load(urllib.request.urlopen(URL + path, timeout=timeout))


def ask(n):
    body = {"model": "x", "max_tokens": a.max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": wiki[1000:1000 + int(n * 4.2)] + "\n\nSummarise the text above in a few paragraphs."}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


modes = [("nospec", [])]
if a.url:
    modes = [(a.mode, ["--draft"] if a.mode != "nospec" else [])]
elif a.drafter:
    modes.append(("dflash2", ["-md", a.drafter] + (["--draft-vocab", a.vocab] if a.vocab else [])))
modes = modes if a.url else [m for m in modes if m[0] in a.modes.split(",")]
lengths = [int(x) for x in a.lengths.split(",")]
out = open(a.out, "a")
for spec in a.model:
    label, path = spec.split("=", 1)
    for mode, margs in modes:
        log = "/tmp/relbench_%s_%s.log" % (label, mode)
        cmd = [a.bin, "--model", path, "--host", "127.0.0.1", "--port", str(a.port), "--context", str(a.context), "--np", "1",
               "--reasoning", "on", "--max-tokens", "4096"] + shlex.split(a.extra) + margs
        srv = None if a.url else subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        if a.url: open(log, "w").close()
        if srv: atexit.register(lambda p=srv: p.poll() is None and os.killpg(p.pid, signal.SIGTERM))  # never leave a server behind
        engine = None
        for _ in range(400):
            if srv and srv.poll() is not None:
                break
            try:
                h = get("/health"); engine = h.get("engine", "unstamped"); break
            except Exception:
                time.sleep(2)
        logtxt = open(log, errors="ignore").read()
        ctx = re.findall(r"context=(\d+)", logtxt)
        base = {"gpu": a.gpu, "gpu_name": gpu_name(), "model": label, "file": os.path.basename(path),
                "file_bytes": os.path.getsize(path) if os.path.exists(path) else None, "mode": mode,
                "drafter": (os.path.basename(a.drafter) if a.drafter else "server's") if margs else None,
                "engine": engine, "requested_context": a.context, "served_context": int(ctx[-1]) if ctx else None,
                "settings": ("server as started" if a.url else " ".join(shlex.split(a.extra))) + ", greedy, %d tokens" % a.max_tokens,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if engine is None:
            out.write(json.dumps(dict(base, error="server did not start: " + logtxt.strip().splitlines()[-1][:200])) + "\n"); out.flush()
            print("%s %s: does not start" % (label, mode), flush=True)
            continue
        for n in lengths:
            runs, err = [], None
            for r in range(a.repeats):
                try:
                    t = ask(n)["timings"]
                    runs.append(t)
                except Exception as e:
                    err = str(e)[:200]; break
            row = dict(base, target_prompt=n)
            if runs:
                row.update(prompt_tokens=runs[0]["prompt_n"], repeats=len(runs),
                           prefill_tps=round(statistics.median(x["prompt_per_second"] for x in runs), 1),
                           decode_tps=round(statistics.median(x["predicted_per_second"] for x in runs), 1),
                           decode_runs=[round(x["predicted_per_second"], 1) for x in runs],
                           accepted=[sum(x.get("draft_n_accepted") or 0 for x in runs), sum(x.get("draft_n") or 0 for x in runs)] if margs else None)
            if err:
                row["error"] = err
            out.write(json.dumps(row) + "\n"); out.flush()
            print("%s %s %6s: %s" % (label, mode, n, ("prefill %.1f decode %.1f" % (row["prefill_tps"], row["decode_tps"])) if runs else "FAILED " + err), flush=True)
        if srv: os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60); time.sleep(5)
print("RELBENCH_DONE", flush=True)
