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
        assert self.headers.get('Authorization') == ('Bearer ' + self.server.key if self.server.key else None)
        self.server.requests.append(body)
        prompt = body['messages'][-1]['content']
        if prompt in ('hello monitor', 'after clear', 'slow terminal', 'usage-less', 'scroll test', 'provider error'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            def emit(value):
                data=('data: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode()
                # Deliberately split SSE lines and UTF-8 across network writes.
                split = data.find(b'\xc3') + 1 if b'\xc3' in data else 9
                for part in [data[:split],data[split:-1],data[-1:]]:
                    self.wfile.write(part);self.wfile.flush()
            try:
                if prompt == 'provider error':
                    emit({'error':{'message':'fixture provider unavailable'}});self.close_connection=True;return
                emit({'choices':[{'delta':{'reasoning_content':'fixture thought'}}]})
                emit({'choices':[{'delta':{'content':'<b>safe text</b> cafe\u00e9' if prompt=='hello monitor' else 'partial' if prompt=='slow terminal' else 'clean followup'}}]})
                if prompt=='scroll test':
                    emit({'choices':[{'delta':{'content':'line\n'*200}}]})
                    for _ in range(30):
                        time.sleep(.1);emit({'choices':[{'delta':{'content':'new line\n'}}]})
                if prompt=='slow terminal':
                    for _ in range(50):
                        time.sleep(.1);emit({'choices':[{'delta':{'content':' more'}}]})
                final={'choices':[{'delta':{},'finish_reason':'stop'}]}
                if prompt!='usage-less':
                    final.update(usage={'prompt_tokens':20,'completion_tokens':8},timings={'predicted_per_second':123.4})
                emit(final);self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):
                self.server.cancelled.set()
            self.close_connection=True
            return
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
            upstream.requests = []
            upstream.key = 'browser-test-key'
            upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            upstream_thread.start()
            server = subprocess.Popen([str(args.binary.resolve()), '--fixture', str(mode), 'browser-test-key', str(upstream.server_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            page = None
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
                expect(page.locator('#ramMemory')).to_be_visible()
                expect(page.locator('#gpuMemory')).to_be_visible()
                expect(page.locator('#cpu')).to_be_visible()
                expect(page.locator('#temperature')).to_be_visible()
                assert page.locator('#themeButton').inner_text() == ''
                expect(page.locator('#themeButton')).to_have_accessible_name('Switch to light theme')
                assert page.locator('#apiKey').input_value() == ''
                assert page.evaluate('localStorage.length + sessionStorage.length') == 0

                assert page.evaluate("""() => {
                    const saved=samples,at=performance.now();
                    samples=[{at:at-4000,rate:1},{at:at-3000,rate:2},{at:at-2000,rate:null},{at:at-1000,rate:3},{at,rate:4}];
                    chart();const count=(document.querySelector('#chartFill').getAttribute('d').match(/M/g)||[]).length;
                    samples=saved;chart();return count===2;
                }"""), 'chart fill bridged an unmeasured interval'

                for control in ('themeButton','keyButton','chatClear','chatSend','chatStop'):
                    box = page.locator('#'+control).bounding_box()
                    assert box and box['width'] >= 44 and box['height'] >= 44, (control,box)
                    assert page.locator('#'+control).get_attribute('aria-label'), control
                page.locator('#keyButton').click()
                expect(page.locator('#keyButton')).to_have_attribute('aria-expanded','true')
                expect(page.locator('#apiKey')).to_be_focused()
                page.locator('#keyButton').click()
                expect(page.locator('#keyButton')).to_have_attribute('aria-expanded','false')
                expect(page.locator('#auth')).not_to_be_visible()
                page.locator('#keyButton').click()
                page.keyboard.press('Escape')
                expect(page.locator('#auth')).not_to_be_visible()
                expect(page.locator('#keyButton')).to_be_focused()
                page.locator('#chatInput').fill('IME in progress')
                page.locator('#chatInput').dispatch_event('keydown',{'key':'Enter','isComposing':True,'bubbles':True})
                expect(page.locator('#chatInput')).to_have_value('IME in progress')
                expect(page.locator('.msg')).to_have_count(0)
                page.locator('#chatInput').press('Shift+Enter')
                expect(page.locator('#chatInput')).to_have_value('IME in progress\n')
                page.locator('#chatInput').fill('   ')
                page.locator('#chatSend').click()
                expect(page.locator('.msg')).to_have_count(0)
                page.locator('#chatInput').fill('')

                # Inspect a retained request and pin a copy, rather than a live mutable row.
                page.locator('#requests .request-select').nth(1).click()
                expect(page.locator('#lensPhase')).to_contain_text('selected request')
                focused = page.locator('#requests .request-select').nth(1)
                focused.focus()
                page.wait_for_timeout(1300)
                expect(focused).to_be_focused()
                page.locator('#pinRequest').click()
                expect(page.locator('#comparison')).to_contain_text('Token counts match')
                page.locator('#requests .request-select').nth(2).click()
                expect(page.locator('#comparison')).to_contain_text('Different token counts')
                expect(page.locator('#lensNote')).to_contain_text('not isolated GPU time')
                page.locator('#followLatest').click()
                expect(page.locator('#lensPhase')).to_contain_text('following latest')
                expect(page.locator('#pinRequest')).to_be_disabled()
                page.locator('#clearReference').click()
                expect(page.locator('#comparison')).not_to_be_visible()
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


                page.locator('#chatInput').fill('hello monitor')
                page.locator('#chatSend').click()
                expect(page.locator('#chatStatus')).to_have_text('done')
                expect(page.locator('.msg.model pre:not(.thought)')).to_have_text('<b>safe text</b> cafe\u00e9')
                assert page.locator('.msg.model b').count() == 0, 'chat rendered model HTML'
                assert upstream.requests[-1]['chat_template_kwargs']['enable_thinking'] is False
                expect(page.locator('#chatStats')).to_contain_text('123.4 tok/s (server)')
                page.locator('#themeButton').click()
                assert page.evaluate('document.documentElement.dataset.theme') == 'light'
                expect(page.locator('#themeButton')).to_have_accessible_name('Switch to dark theme')
                page.locator('#themeButton').click()
                page.locator('#chatClear').click()
                page.locator('#chatInput').fill('slow terminal')
                page.locator('#chatSend').click()
                expect(page.locator('.msg.model pre:not(.thought)')).to_contain_text('partial')
                page.locator('#chatClear').click()
                expect(page.locator('#chatSend')).to_be_enabled()
                page.locator('#chatInput').fill('after clear')
                page.locator('#chatSend').click()
                expect(page.locator('#chatStatus')).to_have_text('done')
                assert upstream.requests[-1]['messages'] == [{'role':'user','content':'after clear'}], 'cleared reply leaked into next conversation'
                page.locator('#chatClear').click()
                page.locator('#chatInput').fill('slow terminal')
                page.locator('#chatSend').click()
                expect(page.locator('.msg.model pre:not(.thought)')).to_contain_text('partial')
                page.keyboard.press('Escape')
                expect(page.locator('#chatStatus')).to_have_text('stopped')
                expect(page.locator('#chatSend')).to_be_enabled()
                page.locator('#chatClear').click()
                page.locator('#chatInput').fill('usage-less')
                page.locator('#chatSend').click()
                expect(page.locator('#chatStatus')).to_have_text('done')
                expect(page.locator('#chatStats')).to_contain_text('chunks')
                expect(page.locator('#chatStats')).to_contain_text('no token count')
                assert page.evaluate('localStorage.length + sessionStorage.length') == 0

                page.locator('#chatClear').click()
                page.locator('#chatInput').fill('provider error')
                page.locator('#chatSend').click()
                expect(page.locator('#chatStatus')).to_contain_text('fixture provider unavailable')
                expect(page.locator('#chatInput')).to_have_value('provider error')
                expect(page.locator('#chatSend')).to_be_enabled()
                page.locator('#chatInput').fill('scroll test')
                page.locator('#chatSend').click()
                page.wait_for_function("() => document.querySelector('#transcript').scrollHeight > document.querySelector('#transcript').clientHeight + 100")
                page.locator('#transcript').evaluate('(el)=>{el.scrollTop=0;el.dispatchEvent(new Event("scroll"));}')
                page.wait_for_timeout(350)
                assert page.locator('#transcript').evaluate('(el)=>el.scrollTop') < 48, 'stream forced reader to bottom'
                page.locator('#chatStop').click()
                expect(page.locator('#chatStatus')).to_have_text('stopped')
                page.locator('#chatClear').click()

                # Trusted panel callbacks receive isolated telemetry and cannot break core updates.
                page.evaluate("""() => {
                    window.removePanel = monitorPanels.register({id:'test-count',title:'Fixture panel',render:({snapshot,element})=>{element.textContent=String(snapshot.schema_version);snapshot.schema_version=99;}});
                    window.panelFails=true;window.removeBroken = monitorPanels.register({id:'test-broken',title:'Failing panel',render:({element})=>{if(window.panelFails)throw new Error('fixture');element.textContent='Recovered';}});
                    window.removeWitness = monitorPanels.register({id:'test-witness',title:'Independent panel',render:({snapshot,element})=>{element.textContent=String(snapshot.schema_version)}});
                    let rejected=0;for(const id of ['test-count','bad id',null]){try{monitorPanels.register({id,title:'Bad',render:()=>{}})}catch(_){rejected++}}
                    if(rejected!==3)throw new Error('invalid panel accepted');
                }""")
                expect(page.locator('#panel-test-count')).to_contain_text('1')
                expect(page.locator('#panel-test-witness')).to_contain_text('1')
                expect(page.locator('#panel-test-broken')).to_contain_text('Panel unavailable.')
                expect(page.locator('#connection')).to_have_text('Connected')
                page.evaluate('window.panelFails=false')
                expect(page.locator('#panel-test-broken')).to_contain_text('Recovered')
                page.evaluate('removeBroken();removeBroken();removeWitness();removePanel()')
                page.evaluate("() => { window.removePanel=monitorPanels.register({id:'test-count',title:'Replacement panel',render:({element})=>{element.textContent='Replaced'}}); }")
                expect(page.locator('#panel-test-count')).to_contain_text('Replaced')
                page.evaluate('removePanel()')

                mode.write_text('prefill')
                expect(page.locator('#phase')).to_have_text('Prefilling prompt')
                expect(page.locator('#requests .progress')).to_have_count(1)
                expect(page.locator('#prefillStatus')).to_be_visible()
                page.wait_for_function("() => document.querySelector('#prefillSpeed').textContent !== '--'")
                expect(page.locator('#prefillCount')).to_contain_text('/ 8,192 tokens')
                expect(page.locator('#lensPrefillLabel')).to_have_text('PREFILL RATE')
                expect(page.locator('#lensRateLabel')).to_have_text('OUTPUT RATE')
                expect(page.locator('#lensPrefill')).to_contain_text('tok/s')
                expect(page.locator('#lensRate')).to_have_text('--')
                assert 'prompt tok/s' not in page.locator('#lensNote').inner_text()
                expect(page.locator('#speed')).to_have_text('--')
                expect(page.locator('#context')).to_contain_text('8,192')
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-prefill.png'), full_page=True)
                mode.write_text('decode')
                expect(page.locator('#phase')).to_have_text('Generating output')
                page.wait_for_function("() => document.querySelector('#speed').textContent!=='--'")
                expect(page.locator('#lensPrefill')).to_contain_text('tok/s')
                expect(page.locator('#lensRate')).to_contain_text('tok/s')
                page.wait_for_timeout(1200)
                assert page.locator('#chartLine').get_attribute('d')
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-desktop.png'), full_page=True)
                page.locator('#themeButton').click()
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-light.png'), full_page=True)
                page.locator('#themeButton').click()
                for width in (320,390,768):
                    page.set_viewport_size({'width':width,'height':844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), ('overflow',width)
                    expect(page.locator('#model')).to_be_visible()
                    rates=[page.locator('#'+id).bounding_box() for id in ('lensPrefill','lensRate')]
                    assert abs(rates[0]['y']-rates[1]['y'])<1, ('rates must share one row',width,rates)
                    controls=[page.locator('#'+id).bounding_box() for id in ('themeButton','keyButton')]
                    assert abs(controls[0]['y']-controls[1]['y']) < 1, ('header alignment',width,controls)
                    buttons=[page.locator('#'+id).bounding_box() for id in ('chatSend','chatStop')]
                    assert abs(buttons[0]['y']-buttons[1]['y']) < 1, ('composer alignment',width,buttons)
                page.set_viewport_size({'width': 390, 'height': 844})
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
                page.screenshot(path=str(args.output / f'monitor-{browser_name}-mobile.png'), full_page=True)
                mode.write_text('idle')
                expect(page.locator('#phase')).to_have_text('Ready for your next request')
                expect(page.locator('#lensPrefill')).to_contain_text('tok/s')
                expect(page.locator('#lensRate')).to_contain_text('tok/s')
                mode.write_text('failed')
                expect(page.locator('#requests tr').first).to_contain_text('failed')
                mode.write_text('idle')
                expect(page.locator('#phase')).to_have_text('Ready for your next request')
                expect(page.locator('#speed')).to_have_text('--')
                page.evaluate("() => { monitorPanels.register({id:'test-count',title:'Stale panel',render:({element})=>{element.textContent='Ready'}}); }")
                expect(page.locator('#panel-test-count')).to_contain_text('Ready')
                server.terminate()
                server.wait(timeout=5)
                expect(page.locator('#connection')).to_have_text('Disconnected', timeout=8000)
                expect(page.locator('#tokens')).to_have_text('--')
                expect(page.locator('#phase')).to_have_text('Telemetry unavailable')
                expect(page.locator('#lensPhase')).to_have_text('Snapshot unavailable')
                expect(page.locator('#pinRequest')).to_be_disabled()
                expect(page.locator('#notice')).to_be_visible()
                expect(page.locator('#panel-test-count')).to_contain_text('Waiting for fresh telemetry.')
                assert not errors, errors
                results.append({'browser': browser_name, 'passed': True, 'api_mocked': False, 'fixture': True, 'page_errors': errors})
                # A server without --api-key must not demand a made-up key in chat.
                upstream.key = ''
                mode.write_text('empty')
                no_key = subprocess.Popen([str(args.binary.resolve()), '--fixture', str(mode), '', str(upstream.server_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    free_port = int(no_key.stdout.readline().strip())
                    open_page = browser.new_page()
                    open_page.goto(f'http://127.0.0.1:{free_port}/')
                    expect(open_page.locator('#connection')).to_have_text('Connected')
                    expect(open_page.locator('#lensTitle')).to_have_text('Your next request starts here')
                    expect(open_page.locator('.journey')).not_to_be_visible()
                    expect(open_page.locator('#pinRequest')).to_be_disabled()
                    open_page.locator('#chatInput').fill('usage-less')
                    open_page.locator('#chatSend').click()
                    expect(open_page.locator('#chatStatus')).to_have_text('done')
                    open_page.close()
                finally:
                    no_key.terminate(); no_key.wait(timeout=5)
                browser.close()
            except Exception:
                if page is not None and not page.is_closed():
                    page.screenshot(path=str(args.output / f'monitor-{browser_name}-failure.png'), full_page=True)
                raise
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
