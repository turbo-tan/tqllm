#!/usr/bin/env python3
"""Release-candidate gate: an old and a new tqserve build, same model and prompts, on one GPU.

For every --arm (a tqserve argument string) it starts each build in turn (ABAB with --rounds 2), sends every prompt twice
(greedy, the second pass is the measured one) plus an optional long prompt, and checks:
  - the new build's text equals the old build's, byte for byte (pass 2 and the long prompt);
  - the new build's mean chat decode is not below the old one's by more than --max-loss percent, nor its long-prompt
    decode by more than --max-long-loss percent;
--np-check N runs the new build at --np 1 and at --np N with N different prompts sent at once: each answer must equal its
--np 1 answer, and the N answers must differ from each other.
Exit status 1 if any check fails. Example:
  tools/rc_gate.py --old build-old/tqserve --new build/tqserve --prompts prompts/ \\
      --arm "27B nospec|--model Qwen3.8-27B-TQ4_L.gguf" --arm "27B dflash2|--model Qwen3.8-27B-TQ4_L.gguf -md DFlash2.gguf"
"""
import argparse, concurrent.futures, hashlib, json, os, shlex, signal, subprocess, sys, time, urllib.request

def verdict(same, long_same, loss, long_loss, max_loss, max_long_loss):
    """One arm passes when both texts are identical and neither the chats' mean decode nor the long prompt's decode
    lost more than its limit (a long decode of 67.1 -> 4.9 once passed on the chats alone)."""
    return same and long_same and loss <= max_loss and long_loss <= max_long_loss

def request(port, text, max_tokens, timeout=3600):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    t = d.get("timings", {})
    return d["choices"][0]["message"]["content"] or "", float(t.get("predicted_per_second", 0.0)), float(t.get("prompt_ms", 0.0)) / 1000

class Server:
    def __init__(self, binary, args, port, log):
        self.port = port
        self.p = subprocess.Popen([binary] + shlex.split(args) + ["--host", "127.0.0.1", "--port", str(port)],
                                  stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(1200):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2); return
            except Exception:
                if self.p.poll() is not None: raise RuntimeError("server exited at start, see " + log)
                time.sleep(1)
        raise RuntimeError("server did not start, see " + log)
    def stop(self):
        try: os.killpg(self.p.pid, signal.SIGINT); self.p.wait(30)
        except Exception: os.killpg(self.p.pid, signal.SIGKILL); self.p.wait()

def run_arm(binary, args, port, prompts, long_prompt, max_tokens, log):
    s = Server(binary, args, port, log)
    try:
        for t in prompts: request(port, t, max_tokens)  # pass 1 (warm-up)
        texts, speeds = [], []
        for t in prompts:
            text, tps, _ = request(port, t, max_tokens); texts.append(text); speeds.append(tps)
        lng = request(port, long_prompt, 128) if long_prompt else None
    finally:
        s.stop()
    return texts, speeds, lng

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--prompts", required=True, help="directory of *.txt prompts (sorted by name)")
    ap.add_argument("--long-prompt", help="file with one long prompt (decode 128 tokens after it)")
    ap.add_argument("--arm", action="append", default=[], help="'name|tqserve args'")
    ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-long-loss", type=float, default=8.0, help="percent the long-prompt decode may lose (one sample: noisier)")
    ap.add_argument("--max-loss", type=float, default=3.0, help="allowed decode loss, percent")
    ap.add_argument("--np-check", type=int, default=0); ap.add_argument("--np-args", default="", help="tqserve args for the np check")
    ap.add_argument("--port", type=int, default=18797); ap.add_argument("--out", default="rc_gate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.prompts) if f.endswith(".txt"))
    prompts = [open(os.path.join(a.prompts, f), encoding="utf-8").read() for f in files]
    long_prompt = open(a.long_prompt, encoding="utf-8").read() if a.long_prompt else None
    ok = True
    def say(line):
        print(line, flush=True); open(os.path.join(a.out, "rc_gate.result"), "a").write(line + "\n")
    for arm in a.arm:
        name, args = arm.split("|", 1)
        res = {"old": [], "new": []}
        for r in range(a.rounds):
            for which in ("old", "new"):
                tag = "%s_%s_r%d" % (name.replace(" ", "_"), which, r + 1)
                res[which].append(run_arm(getattr(a, which), args, a.port, prompts, long_prompt, a.max_tokens, os.path.join(a.out, tag + ".log")))
        mean = {w: sum(sum(x[1]) / len(x[1]) for x in res[w]) / len(res[w]) for w in res}
        lmean = {w: (sum(x[2][1] for x in res[w]) / len(res[w]) if long_prompt else 0) for w in res}
        same = all(o == n for o, n in zip(res["old"][0][0], res["new"][0][0]))
        lsame = not long_prompt or res["old"][0][2][0] == res["new"][0][2][0]
        loss = 100 * (1 - mean["new"] / mean["old"]) if mean["old"] else 0
        lloss = 100 * (1 - lmean["new"] / lmean["old"]) if lmean["old"] else 0
        good = verdict(same, lsame, loss, lloss, a.max_loss, a.max_long_loss)
        ok &= good
        say("%-24s chat old %.1f new %.1f tok/s (%+.1f %%)%s | text %s%s | %s" % (name, mean["old"], mean["new"], -loss,
            ("  long old %.1f new %.1f (%+.1f %%)" % (lmean["old"], lmean["new"], -lloss)) if long_prompt else "",
            "identical" if same else "DIFFERS", "" if lsame else ", long DIFFERS", "PASS" if good else "FAIL"))
    if a.np_check > 1:
        n = a.np_check
        if len(prompts) < n: sys.exit("np check needs %d prompts" % n)
        s = Server(a.new, a.np_args + " --np 1", a.port, os.path.join(a.out, "np1.log"))
        try: alone = [request(a.port, t, a.max_tokens)[0] for t in prompts[:n]]
        finally: s.stop()
        for k in sorted({2, 4, n} & set(range(2, n + 1))):
            s = Server(a.new, a.np_args + " --np %d" % k, a.port, os.path.join(a.out, "np%d.log" % k))
            try:
                with concurrent.futures.ThreadPoolExecutor(k) as ex: got = list(ex.map(lambda t: request(a.port, t, a.max_tokens)[0], prompts[:k]))
            finally: s.stop()
            exact = sum(g == x for g, x in zip(got, alone[:k])); distinct = len({hashlib.md5(g.encode()).hexdigest() for g in got})
            good = exact == k and distinct == k; ok &= good
            say("np %d: %d / %d == np 1, %d / %d distinct | %s" % (k, exact, k, distinct, k, "PASS" if good else "FAIL"))
    say("RC_GATE %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
