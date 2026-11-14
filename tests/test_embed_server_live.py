"""tqserve with a contextual embedding GGUF, over HTTP: /health answers while a client stalls mid-request and while a
long embeddings request runs; base64 output decodes to the float output; a deeply nested body and an over-limit
request are 400s, not a crash. Needs TQLLM_PPLX_GGUF; exit 77 = skipped.

  python3 tests/test_embed_server_live.py PATH/TO/tqserve"""
import base64, json, os, socket, struct, subprocess, sys, threading, time, urllib.error, urllib.request


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(url, body, timeout=600):
    req = urllib.request.Request(url, body if isinstance(body, bytes) else json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def health_ms(base):
    t = time.time()
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        r.read()
    return (time.time() - t) * 1000


def main():
    gguf = os.environ.get("TQLLM_PPLX_GGUF")
    if not gguf:
        print("embed server live: skipped (set TQLLM_PPLX_GGUF)")
        return 77
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([sys.argv[1], "--model", gguf, "--host", "127.0.0.1", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    failures = []
    try:
        for _ in range(600):
            try:
                health_ms(base)
                break
            except OSError:
                time.sleep(0.5)
        # 1. a client that sends half a request and stops
        stall = socket.create_connection(("127.0.0.1", port))
        stall.sendall(b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{\"inp")
        time.sleep(0.5)
        ms = health_ms(base)
        print(f"/health with a stalled client: {ms:.0f} ms")
        if ms > 2000:
            failures.append(f"/health took {ms:.0f} ms behind a stalled client")
        # 2. a long embeddings request in flight
        long_doc = [["word " * 400] * 40] * 6
        done = {}
        th = threading.Thread(target=lambda: done.update(r=post(base + "/v1/embeddings", {"input": long_doc})))
        th.start()
        time.sleep(0.3)
        ms = health_ms(base)
        th.join()
        print(f"/health during a long request: {ms:.0f} ms (request status {done['r'][0]})")
        if ms > 2000:
            failures.append(f"/health took {ms:.0f} ms during a long request")
        if done["r"][0] != 200:
            failures.append(f"long request status {done['r'][0]}")
        stall.close()
        # 3. base64 (what the openai SDK asks for by default) = the float vector
        st, f = post(base + "/v1/embeddings", {"input": "hello world"})
        st2, b = post(base + "/v1/embeddings", {"input": "hello world", "encoding_format": "base64"})
        if st != 200 or st2 != 200:
            failures.append(f"float / base64 status {st} / {st2}")
        else:
            raw = base64.b64decode(b["data"][0]["embedding"])
            vec = struct.unpack(f"<{len(raw) // 4}f", raw)
            ref = f["data"][0]["embedding"]
            if len(vec) != len(ref) or max(abs(x - y) for x, y in zip(vec, ref)) > 1e-6:
                failures.append("base64 vector differs from the float one")
        st, d = post(base + "/v1/embeddings", {"input": "x", "dimensions": 7})
        if st != 400:
            failures.append(f"dimensions 7: status {st}, want 400")
        # 4. untrusted bodies
        st, d = post(base + "/v1/embeddings", b'{"input":' + b"[" * 100000 + b"]" * 100000 + b"}")
        if st != 400:
            failures.append(f"deeply nested body: status {st}, want 400")
        st, d = post(base + "/v1/embeddings", {"input": ["x"] * 5000})
        if st != 400 or "inputs" not in d.get("error", {}).get("message", json.dumps(d)):
            failures.append(f"5000 inputs: status {st} {d}")
        if srv.poll() is not None:
            failures.append("the server exited")
        else:
            health_ms(base)
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    for f in failures:
        print("FAIL:", f)
    print("embed server live:", "FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
