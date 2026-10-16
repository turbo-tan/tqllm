"""Optional browser gate against the real C++ monitor listener, with labelled fixture counters.

python tests/test_monitor_ui.py --binary build-cpu/test-monitor --output monitor-artifacts
Requires: pip install playwright; playwright install chromium firefox webkit
No application API is mocked. No model or GPU job is started.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class InferenceFixture(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.path == '/v1/chat/completions'
        assert self.headers.get('Authorization') == 'Bearer browser-test-key'
        assert body['messages'][0]['content'] in ('browser transport gate', 'slow transport gate')
        answer = b'data: {"choices":[{"delta":{"content":"fixture response"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        if body['messages'][0]['content'] != 'slow transport gate':
            self.send_header('Content-Length', str(len(answer)))
        self.end_headers()
        if body['messages'][0]['content'] == 'slow transport gate':
            try:
                for _ in range(20):
                    self.wfile.write(answer[:10])
                    self.wfile.flush()
                    time.sleep(.1)
            except (BrokenPipeError, ConnectionResetError):
                self.server.cancelled.set()
            return
        self.wfile.write(answer)

    def log_message(self, *args):
        pass


def main():
    from playwright.sync_api import sync_playwright, expect
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as temporary, sync_playwright() as pw:
        mode = Path(temporary) / 'mode'
        for browser_name in ('chromium', 'firefox', 'webkit'):
            mode.write_text('queued')
            upstream = ThreadingHTTPServer(('127.0.0.1', 0), InferenceFixture)
            upstream.cancelled = threading.Event()
            upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            upstream_thread.start()
            server = subprocess.Popen([str(args.binary.resolve()), '--fixture', str(mode), 'browser-test-key', str(upstream.server_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                port = int(server.stdout.readline().strip())
                browser = getattr(pw, browser_name).launch()
                page = browser.new_page(viewport={'width': 1440, 'height': 1080}, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{port}/')
                expect(page.locator('#auth')).to_be_visible()
                expect(page.locator('#connection')).to_have_text('Access required')
                page.locator('#apiKey').fill('browser-test-key')
                page.locator('#auth button').click()
                expect(page.locator('#connection')).to_have_text('Connected')
                expect(page.locator('#auth')).not_to_be_visible()
                expect(page.locator('#queue')).to_have_text('0 / 1')
                assert page.locator('#apiKey').input_value() == ''
                assert page.evaluate('localStorage.length + sessionStorage.length') == 0
                streamed = page.evaluate("""async()=>{
                    const response=await fetch('/api/monitor/v1/chat/completions',{
                        method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer browser-test-key'},
                        body:JSON.stringify({messages:[{role:'user',content:'browser transport gate'}],stream:true})});
                    return {status:response.status,body:await response.text()};
                }""")
                assert streamed['status'] == 200 and 'fixture response' in streamed['body'] and '[DONE]' in streamed['body']
                cancelled = page.evaluate("""async()=>{
                    const controller=new AbortController();
                    const response=await fetch('/api/monitor/v1/chat/completions',{
                        method:'POST',signal:controller.signal,
                        headers:{'Content-Type':'application/json',Authorization:'Bearer browser-test-key'},
                        body:JSON.stringify({messages:[{role:'user',content:'slow transport gate'}],stream:true})});
                    const reader=response.body.getReader();await reader.read();controller.abort();return true;
                }""")
                assert cancelled and upstream.cancelled.wait(3), 'Abort did not close upstream chat'


                mode.write_text('prefill')
                expect(page.locator('#phase')).to_have_text('Prefilling prompt')
                expect(page.locator('#speed')).to_have_text('--')
                expect(page.locator('#context')).to_contain_text('8,192')
                mode.write_text('decode')
                expect(page.locator('#phase')).to_have_text('Generating output')
                page.wait_for_function("document.querySelector('#speed').textContent!=='--'")
                page.wait_for_timeout(1200)
                assert page.locator('#chartLine').get_attribute('d')
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-desktop.png'), full_page=True)
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-mobile.png'), full_page=True)
                mode.write_text('failed')
                expect(page.locator('#requests tr').first).to_contain_text('failed')
                mode.write_text('idle')
                expect(page.locator('#phase')).to_have_text('Ready for your next request')
                expect(page.locator('#speed')).to_have_text('--')
                server.terminate()
                server.wait(timeout=5)
                expect(page.locator('#connection')).to_have_text('Disconnected', timeout=8000)
                expect(page.locator('#tokens')).to_have_text('--')
                expect(page.locator('#phase')).to_have_text('Telemetry unavailable')
                expect(page.locator('#notice')).to_be_visible()
                assert not errors, errors
                results.append({'browser': browser_name, 'passed': True, 'api_mocked': False, 'fixture': True, 'page_errors': errors})
                browser.close()
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=5)
                upstream.shutdown()
                upstream.server_close()
                upstream_thread.join(timeout=5)
    (args.output / 'browser-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
