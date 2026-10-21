"""Record a running monitor with Playwright. Uses real requests; no API mocks.

Run from the repository root, after starting tqserve as described in README.md.
Requires the optional Playwright Python package and Chromium browser.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright


def prompt():
    rows = [
        f"Entry {i}: queue requests fairly, preserve the original task, record completed work, and keep the interface responsive."
        for i in range(1, 321)
    ]
    return "Read these engineering notes.\n" + "\n".join(rows) + "\nSummarize the three most useful ideas in three short bullet points."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8086')
    parser.add_argument('--output', type=Path, default=Path('monitor-recording'))
    args = parser.parse_args()
    origin = urlsplit(args.url)
    if origin.scheme not in ('http', 'https') or origin.username or origin.password:
        parser.error('use an HTTP(S) URL without embedded credentials')
    base = args.url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={'width': 1280, 'height': 900},
                                      record_video_dir=str(args.output),
                                      record_video_size={'width': 1280, 'height': 900},
                                      reduced_motion='reduce')
        # Apply privacy masking before the first page paint, including later navigations.
        context.add_init_script("""(() => {
            const install=() => {
                if (!document.documentElement) return false;
                const style=document.createElement('style');style.textContent='#host {filter:blur(8px)!important}';
                document.documentElement.append(style);return true;
            };
            if(!install()){const observer=new MutationObserver(()=>{if(install())observer.disconnect()});observer.observe(document,{childList:true,subtree:true})}
        })();""")
        page = context.new_page()
        video = page.video
        try:
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(base + '/monitor')
            page.wait_for_function("() => document.body.dataset.connection === 'live'")

            def caption(title, detail):
                page.evaluate('''([title,detail]) => {
                    document.querySelector('#recording-caption')?.remove();
                    const box=document.createElement('aside');box.id='recording-caption';
                    box.style.cssText='position:fixed;left:24px;bottom:22px;z-index:10000;padding:16px 22px;background:#111a2bf5;color:#f2f6ff;border:1px solid #659ad6;border-radius:12px;max-width:760px;box-shadow:0 8px 30px #0006;font:16px system-ui;pointer-events:none';
                    const heading=document.createElement('strong');heading.style.cssText='display:block;font-size:21px;margin-bottom:7px';heading.textContent=title;
                    const body=document.createElement('div');body.textContent=detail;box.append(heading,body);document.body.append(box);
                    document.querySelector('#host').style.filter='blur(8px)';
                }''', [title, detail])

            def clear_caption():
                page.evaluate("document.querySelector('#recording-caption')?.remove()")

            caption('1. Open the API address', './build/tqserve --model models/model.gguf --port 8086; open localhost:8086/monitor')
            page.wait_for_timeout(6500)
            page.goto(base + '/')
            page.wait_for_function("() => document.body.dataset.connection === 'live'")
            caption('The root address works too', 'The dashboard and OpenAI-compatible API share one port. No second service is needed.')
            page.wait_for_timeout(5000)
            caption('2. Send a long prompt', 'Watch completed prompt tokens, prefill tok/s and hardware readings before the first output.')
            page.locator('#chatInput').fill(prompt())
            page.wait_for_timeout(1800)
            page.locator('#chatSend').click()
            try:
                page.wait_for_function("() => document.querySelector('#phase').textContent === 'Prefilling prompt' || document.querySelector('#chatStatus').textContent.startsWith('failed')", timeout=90000)
                if page.locator('#phase').inner_text() != 'Prefilling prompt':
                    raise RuntimeError(page.locator('#chatStatus').inner_text())
            except Exception:
                page.screenshot(path=str(args.output/'failure.png'))
                (args.output/'failure.json').write_text(json.dumps({'phase':page.locator('#phase').inner_text(),'chat':page.locator('#chatStatus').inner_text(),'page_errors':errors,'messages':page.locator('#transcript').inner_text()[-2000:]},indent=2))
                raise
            page.wait_for_function("() => document.querySelector('#lensRateLabel').textContent === 'PREFILL RATE' && document.querySelector('#lensRate').textContent.includes('tok/s')", timeout=90000)
            clear_caption()
            page.screenshot(path=str(args.output/'prefill.png'))
            caption('Live prefill, before generation', 'The rate comes from completed GPU chunks. Cached tokens are excluded; no completion percentage is guessed.')
            page.wait_for_function("() => document.querySelector('#chatStatus').textContent === 'done'", timeout=120000)
            page.wait_for_timeout(4500)
            clear_caption()
            page.screenshot(path=str(args.output/'completed.png'))
            caption('3. Inspect a completed request', 'Select its row to keep the timings in view, then pin it as the reference for a later request.')
            page.locator('#requests .request-select').first.click()
            page.locator('#pinRequest').click()
            page.wait_for_timeout(6500)
            caption('4. Choose your view', 'Use the theme icon for light or dark mode. Hardware samples refresh every three seconds.')
            page.locator('#themeButton').click()
            page.wait_for_timeout(4000)
            clear_caption()
            page.screenshot(path=str(args.output/'light.png'))
            page.locator('#themeButton').click()
            caption('Ready for your own requests', 'Open / or /monitor on your API port. Use an API key when exposing the server. Full setup is linked in the README.')
            page.wait_for_timeout(12000)
        finally:
            context.close()
        video.save_as(str(args.output/'walkthrough.webm'))
        browser.close()
        if errors:
            raise RuntimeError(errors)
    (args.output/'recording.json').write_text(json.dumps({'real_requests':True,'viewport':[1280,900],
        'hostname_blurred':True,'page_errors':errors,'prompt_characters':len(prompt())},indent=2)+'\n')


if __name__ == '__main__':
    main()
