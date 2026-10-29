#!/data/data/com.termux/files/usr/bin/python3
"""Local chat front end for tqandroid-run --chat. Phone facts go into the system prompt.

Run from a normal Termux shell (not inside PRoot) so getprop/dumpsys are reachable:
    python3 tools/android/chat/server.py   # then open http://127.0.0.1:8765 in the phone browser
The runner and model default to the paths in $PREFIX/bin/tqandroid (installed by install-termux.sh).
"""
import argparse, codecs, http.server, json, os, re, shutil, subprocess, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEP = b"\x1e"


def run(*cmd, timeout=4):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def prop(name):
    return run("getprop", name)


def meminfo():
    info = {}
    try:
        for line in open("/proc/meminfo"):
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) // 1024
    except OSError:
        pass
    return info


def storage(path):
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize / 1e9, s.f_blocks * s.f_frsize / 1e9
    except OSError:
        return None


def battery():
    raw = run("termux-battery-status", timeout=8)
    if raw:
        try:
            b = json.loads(raw)
            return f"{b.get('percentage')}% ({b.get('status', '').lower()}, {b.get('temperature', '?')} C, health {b.get('health', '?').lower()})"
        except ValueError:
            pass
    dump = run("dumpsys", "battery")
    level = re.search(r"level: (\d+)", dump)
    if level:
        temp = re.search(r"temperature: (\d+)", dump)
        plugged = "charging" if re.search(r"(AC|USB|Wireless) powered: true", dump) else "on battery"
        return f"{level.group(1)}% ({plugged}" + (f", {int(temp.group(1)) / 10:.1f} C" if temp else "") + ")"
    return None


def cpu():
    freqs = []
    for f in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/cpuinfo_max_freq")):
        try:
            freqs.append(int(f.read_text()) / 1e6)
        except (OSError, ValueError):
            pass
    if not freqs:
        return f"{os.cpu_count()} cores"
    groups = {}
    for f in freqs:
        groups[f] = groups.get(f, 0) + 1
    return ", ".join(f"{n}x {f:.2f} GHz" for f, n in sorted(groups.items(), reverse=True))


def phone_facts():
    facts = []
    model = " ".join(x for x in (prop("ro.product.manufacturer").title(), prop("ro.product.marketname") or prop("ro.product.model")) if x)
    if model:
        facts.append(f"Phone: {model} (model number {prop('ro.product.model')})")
    android = prop("ro.build.version.release")
    if android:
        facts.append(f"Android {android}, One UI {prop('ro.build.version.oneui') or '?'}, security patch {prop('ro.build.version.security_patch') or '?'}, build {prop('ro.build.display.id') or '?'}")
    soc = prop("ro.soc.model") or prop("ro.board.platform")
    facts.append(f"Chip: {soc + ', ' if soc else ''}CPU {cpu()}")
    mem = meminfo()
    if mem:
        facts.append(f"RAM: {mem.get('MemTotal', 0) / 1024:.1f} GB total, {mem.get('MemAvailable', 0) / 1024:.1f} GB available now")
    for label, path in (("Internal storage", "/storage/emulated/0"), ("App storage", str(Path.home()))):
        s = storage(path)
        if s:
            facts.append(f"{label}: {s[0]:.1f} GB free of {s[1]:.0f} GB")
            break
    b = battery()
    if b:
        facts.append(f"Battery: {b}")
    facts.append(f"Checked at {time.strftime('%Y-%m-%d %H:%M')}")
    return facts


def system_prompt(facts):
    return ("You are a helpful assistant running fully offline on the user's own phone. "
            "Answer questions about the phone using these live facts; say so when a fact is not listed. "
            "Be concise. Give step-by-step Settings paths for how-to questions.\nPhone facts:\n- " + "\n- ".join(facts))


def escape(text):
    return text.replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n")


class Engine:
    def __init__(self, binary, model, threads, context, max_tokens):
        self.cmd = [binary, "--model", model, "--chat", "--threads", str(threads),
                    "--context", str(context), "--max-tokens", str(max_tokens)]
        self.lock = threading.Lock()
        self.proc = None
        self.facts = []

    def _send(self, line):
        self.proc.stdin.write(line.encode() + b"\n")
        self.proc.stdin.flush()

    def _read_reply(self):
        """Yield decoded text pieces; the final item is the metrics dict."""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = self.proc.stdout.read1(4096)
            if not chunk:
                raise RuntimeError("engine exited")
            if SEP in chunk:
                text, rest = chunk.split(SEP, 1)
                piece = decoder.decode(text, final=True)
                if piece:
                    yield piece
                while not rest.endswith(b"\n"):
                    more = self.proc.stdout.read1(4096)
                    if not more:
                        break
                    rest += more
                yield json.loads(rest.decode() or "{}")
                return
            piece = decoder.decode(chunk)
            if piece:
                yield piece

    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            self._new_conversation()

    def _new_conversation(self):
        self.facts = phone_facts()
        self._send("SYSTEM " + escape(system_prompt(self.facts)))
        for _ in self._read_reply():
            pass

    def reset(self):
        with self.lock:
            self._new_conversation()

    def ask(self, message):
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                self._new_conversation()
            self._send("USER " + escape(message))
            yield from self._read_reply()


class Handler(http.server.BaseHTTPRequestHandler):
    engine: Engine = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/phone":
            self._json({"facts": self.engine.facts})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        if self.path == "/api/reset":
            self.engine.reset()
            return self._json({"facts": self.engine.facts})
        if self.path != "/api/chat":
            return self._json({"error": "not found"}, 404)
        message = str(data.get("message", "")).strip()
        if not message:
            return self._json({"error": "empty message"}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def emit(obj):
            line = json.dumps(obj).encode() + b"\n"
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        try:
            for item in self.engine.ask(message):
                emit({"text": item} if isinstance(item, str) else {"done": item})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            emit({"done": {"error": str(e)}})
        self.wfile.write(b"0\r\n\r\n")


def main():
    launcher = Path(os.environ.get("PREFIX", "/usr")) / "bin" / "tqandroid"
    installed = {}
    if launcher.exists():
        for key in ("binary", "model"):
            m = re.search(rf"^{key}=(.+)$", launcher.read_text(), re.M)
            if m:
                installed[key] = m.group(1).strip("'\"")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", default=installed.get("binary") or shutil.which("tqandroid-run"))
    p.add_argument("--model", default=installed.get("model") or str(Path.home() / "tqllm-android/models/Qwen3.5-4B-Q4_K_M.gguf"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=512)
    a = p.parse_args()
    if not a.binary:
        p.error("no tqandroid-run binary found; pass --binary")
    Handler.engine = Engine(a.binary, a.model, a.threads, a.context, a.max_tokens)
    print("Loading model and reading phone facts...", flush=True)
    Handler.engine.start()
    print(f"Ready: open http://{a.host}:{a.port} in your browser (Ctrl+C to stop)", flush=True)
    server = http.server.ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
