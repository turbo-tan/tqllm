#!/usr/bin/env python3
"""tqllm tok/s dashboard: parses the pm2 log of tqllm-gpu2 and serves a live page.

Endpoints:
  /              -> HTML dashboard (auto-refreshes)
  /api/metrics   -> JSON: health, live prefill, latest + history of reqlog lines

Run under pm2:  pm2 start dash.py --name tqllm-dash
"""
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8090
TQLLM = "http://127.0.0.1:8086"
LOG = "/home/green-gpu/.pm2/logs/tqllm-gpu2-error.log"

REQLOG = re.compile(
    r"reqlog: prompt (\d+) tokens ([\d.]+) ms \(([\d.]+) tok/s\)"
    r" \| decode (\d+) tokens ([\d.]+) ms \(([\d.]+) tok/s\) \| draft (\d+)/(\d+)"
)
PREFILL = re.compile(r"prefill progress (\d+)/(\d+) tokens \(([\d.]+) tok/s\)")
FINISH = re.compile(r"finish prompt (\d+) out (\d+) steps (\d+) drafted (\d+) accepted (\d+)")


def read_log():
    """Return (entries, live) where entries are finished reqlog lines (newest last)
    and live is the newest prefill-progress line (a request currently running)."""
    entries, live = [], None
    try:
        with open(LOG, "r", errors="replace") as f:
            for line in f:
                m = REQLOG.search(line)
                if m:
                    entries.append({
                        "t": time.time(),  # replaced below with file mtime ordering
                        "prompt_n": int(m.group(1)),
                        "prompt_ms": float(m.group(2)),
                        "pp_tok_s": float(m.group(3)),
                        "decode_n": int(m.group(4)),
                        "decode_ms": float(m.group(5)),
                        "tg_tok_s": float(m.group(6)),
                        "draft": int(m.group(7)),
                        "draft_total": int(m.group(8)),
                    })
                    continue
                m = PREFILL.search(line)
                if m:
                    live = {"n": int(m.group(1)), "total": int(m.group(2)), "tok_s": float(m.group(3))}
    except FileNotFoundError:
        pass
    return entries, live


def health():
    try:
        with urllib.request.urlopen(TQLLM + "/health", timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"status": "down", "error": str(e)}


def metrics():
    entries, live = read_log()
    for i, e in enumerate(entries):
        e["t"] = i  # index order == log order
    return {
        "health": health(),
        "live": live,
        "latest": entries[-1] if entries else None,
        "history": entries[-64:],
        "updated": time.time(),
    }


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>tqllm tok/s</title>
<style>
:root { color-scheme: dark; }
body { background:#0d1117; color:#c9d1d9; font:14px/1.5 ui-monospace,monospace; margin:24px; }
h1 { font-size:16px; font-weight:600; margin:0 0 16px; color:#8b949e; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }
.card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:12px 16px; }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#8b949e; }
.card .v { font-size:22px; font-weight:600; margin-top:2px; }
.card .s { font-size:11px; color:#8b949e; margin-top:2px; }
.ok { color:#3fb950; } .bad { color:#f85149; } .live { color:#58a6ff; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th,td { text-align:right; padding:4px 8px; border-bottom:1px solid #21262d; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:#8b949e; font-weight:500; }
#live { margin:8px 0; font-size:13px; min-height:20px; }
#chart { width:100%; height:120px; background:#161b22; border:1px solid #21262d; border-radius:8px; }
#err { color:#f85149; font-size:12px; margin-top:8px; min-height:16px; }
</style></head><body>
<h1>tqllm-gpu2 &mdash; tok/s dashboard <span id="st" style="font-size:11px"></span></h1>
<div class="grid">
  <div class="card"><div class="k">server</div><div class="v" id="engine">&mdash;</div><div class="s" id="fit"></div></div>
  <div class="card"><div class="k">latest prefill (pp)</div><div class="v" id="pp">&mdash;</div><div class="s" id="pps"></div></div>
  <div class="card"><div class="k">latest decode (tg)</div><div class="v" id="tg">&mdash;</div><div class="s" id="tgs"></div></div>
  <div class="card"><div class="k">draft accept</div><div class="v" id="dr">&mdash;</div><div class="s" id="drs"></div></div>
</div>
<div id="live"></div>
<canvas id="chart" width="900" height="120"></canvas>
<table id="tbl"><thead><tr><th>time</th><th>prompt</th><th>pp tok/s</th><th>decode</th><th>tg tok/s</th><th>draft</th></tr></thead><tbody></tbody></table>
<div id="err"></div>
<script>
const $ = id => document.getElementById(id);
function fmt(n, d=1) { return n == null ? "&mdash;" : n.toFixed(d); }
async function tick() {
  try {
    const m = await (await fetch("/api/metrics")).json();
    const h = m.health || {};
    $("st").textContent = h.status === "ok" ? "online" : "DOWN";
    $("st").style.color = h.status === "ok" ? "#3fb950" : "#f85149";
    $("engine").textContent = h.engine || "&mdash;";
    $("fit").textContent = h.fit || "";
    const l = m.latest;
    if (l) {
      $("pp").innerHTML = fmt(l.pp_tok_s) + " <span style='font-size:12px'>tok/s</span>";
      $("pps").textContent = l.prompt_n + " tokens / " + (l.prompt_ms/1000).toFixed(1) + "s";
      $("tg").innerHTML = fmt(l.tg_tok_s) + " <span style='font-size:12px'>tok/s</span>";
      $("tgs").textContent = l.decode_n + " tokens / " + (l.decode_ms/1000).toFixed(1) + "s";
      $("dr").textContent = l.draft + "/" + l.draft_total + " (" + (100*l.draft/Math.max(1,l.draft_total)).toFixed(0) + "%)";
      $("drs").textContent = "drafted / accepted";
    }
    const lv = m.live;
    $("live").innerHTML = lv
      ? '<span class="live">prefill in progress: ' + lv.n + "/" + lv.total + " tokens @ " + lv.tok_s.toFixed(0) + " tok/s</span>"
      : "";
    const rows = m.history.slice().reverse().map(e =>
      "<tr><td>" + new Date(e.t*1000).toLocaleTimeString() + "</td><td>" + e.prompt_n +
      "</td><td>" + e.pp_tok_s.toFixed(1) + "</td><td>" + e.decode_n +
      "</td><td>" + e.tg_tok_s.toFixed(1) + "</td><td>" + e.draft + "/" + e.draft_total + "</td></tr>").join("");
    $("tbl").querySelector("tbody").innerHTML = rows;
    draw(m.history);
    $("err").textContent = "";
  } catch (e) { $("err").textContent = "fetch failed: " + e; }
}
function draw(h) {
  const c = $("chart"), x = c.getContext("2d");
  x.clearRect(0, 0, c.width, c.height);
  if (!h.length) return;
  const pp = h.map(e => e.pp_tok_s), tg = h.map(e => e.tg_tok_s);
  const max = Math.max(...pp, ...tg, 1);
  const line = (vals, color) => {
    x.beginPath();
    vals.forEach((v, i) => {
      const px = 4 + (c.width - 8) * (h.length > 1 ? i / (h.length - 1) : 0);
      const py = c.height - 6 - (c.height - 12) * v / max;
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    });
    x.strokeStyle = color; x.lineWidth = 1.5; x.stroke();
  };
  line(pp, "#58a6ff"); line(tg, "#3fb950");
  x.fillStyle = "#8b949e"; x.font = "10px monospace";
  x.fillText("pp " + Math.max(...pp).toFixed(0) + " / tg " + Math.max(...tg).toFixed(0) + " tok/s (last " + h.length + " reqs)", 8, 12);
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/metrics":
            self._send(200, json.dumps(metrics()).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
