# Contributing to the monitor

The monitor ships inside `tqserve`. End users need no frontend build, browser
extension or monitoring service. Browser tooling below is for contributors only.

## Source map

| Responsibility | File |
| --- | --- |
| Page, styles, controls, request inspection and chat | `tools/serve/monitor.html` |
| Trusted panel registration and failure isolation | `tools/serve/monitor_panels.js` |
| Snapshot schema and thread-safe request counters | `tools/serve/monitor.hpp` |
| Optional NVML and Linux host measurements | `tools/serve/monitor_host.hpp` |
| Independent HTTP listener and same-instance chat bridge | `tools/serve/monitor_server.hpp` |
| Embedding both frontend files into the executable | `CMakeLists.txt`, `tools/serve/monitor_page.hpp.in` |
| Listener, authentication and cancellation tests | `tests/monitor.cpp` |
| Real-listener browser regression gate | `tests/test_monitor_ui.py` |

There are no runtime asset requests. CMake tracks the HTML and panel source and
embeds them in the executable. Rebuild after editing either file.

## Add a panel

`monitorPanels.version` is 1. Register a panel in `monitor_panels.js` after the
registry definition. This is a source-level extension point for reviewed, trusted
code. It is not a sandbox or a runtime third-party plugin installer. It is separate
from the engine's experimental kernel-plugin ABI.

```javascript
monitorPanels.register({
  id: 'finished-requests',
  title: 'Finished requests in retained history',
  render({snapshot, element}) {
    const count = snapshot.requests.filter(r => r.phase === 'complete').length;
    element.textContent = String(count);
  }
});
```

- IDs start with a lowercase letter, use lowercase letters/digits/hyphens, and
  contain at most 48 characters. Duplicate IDs and invalid callbacks are rejected.
- `render` is synchronous. It receives its own snapshot copy and a body element.
  Create DOM nodes or use `textContent`; never execute model-supplied markup.
- Snapshot copies contain telemetry only, with no prompts, output text or keys.
  Trusted extension code still shares the page's JavaScript environment: do not
  treat callback isolation as a security boundary.
- Reuse snapshot data. Do not start inference, poll another server, access the
  engine, load scripts, or persist credentials from a panel.
- Callbacks run once per successful snapshot. A throwing callback shows
  `Panel unavailable` and does not stop other panels or core monitoring. A later
  successful callback recovers the panel. Missing/stale telemetry clears it.
- `register` returns an idempotent removal function. Keep panel state bounded and
  reset state keyed by `snapshot.instance_id` when the engine restarts.

Add browser coverage for the panel's valid, missing-data, stale and error states.
The existing suite covers registration, snapshot-copy isolation, a throwing
callback and invalidation. Keep third-party dependencies and provenance explicit.

## Add hardware measurements

Reuse the existing host collector rather than spawning shell commands per HTTP
request. Hardware samples are cached for three seconds; request counters update
independently and the page fetches snapshots once per second. A hardware reading
must be optional, carry the sample age, and become unavailable when stale. Match
devices by engine identity, not a guessed device index. Whole-device GPU memory
is not the model's allocation. Additional vendor collectors should preserve this
contract and fail gracefully when a driver/library is absent.

Add fields to `Monitor::Hardware` and its snapshot serializer, document units and
scope in [MONITOR.md](MONITOR.md), then add display and schema tests. Use additive
fields for schema v1; incompatible changes need an explicit version decision.

## Test locally

```sh
cmake -S . -B build-cpu -DTQLLM_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
python3 -m venv .venv-monitor
.venv-monitor/bin/pip install playwright==1.64.0
.venv-monitor/bin/python -m playwright install --with-deps chromium firefox webkit
.venv-monitor/bin/python tests/test_monitor_ui.py \
  --binary build-cpu/test-monitor --output build-cpu/monitor-artifacts
```

The browser suite starts the real C++ listener and a disposable HTTP inference
fixture. It loads no model and needs no GPU or credentials. Its illustrative
counters are not engine performance measurements. CI runs the same three-browser
gate and retains screenshots. Never point fixtures at a user's live server.

## UI review checklist

- Check dark/light themes, 320/390/768-pixel widths and desktop. Page content must
  fit; the request table can scroll within its labelled region.
- Common icon actions need accessible names, tooltips and 44-pixel click targets.
  Keep unfamiliar actions labelled. Pair colours with text or values.
- Test keyboard focus through a telemetry refresh, IME composition, Shift+Enter,
  empty submission, Send, Stop, Escape, Clear, authentication and error recovery.
- Streaming must not pull a reader away from older output. Do not announce every
  token or telemetry sample to screen readers; use status messages for transitions.
- Test missing hardware, stale/disconnected data, retained references and engine
  restart. Never invent token rates, prefill progress or causal diagnoses.
- Preserve the same-origin bridge, key-in-memory policy and literal model-text
  rendering. A visual-only change does not establish real-model parity; engine or
  serving changes need a separately documented real-machine gate.

Use the repository PR checklist and record the exact source commit and checks.
Visual approval and merge/publication remain separate decisions.
