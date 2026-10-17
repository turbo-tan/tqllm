# Runtime monitor

A runtime dashboard built into `tqserve`, with read-only telemetry and a same-origin chat bridge. It runs on a separate, optional
listener so a long single-slot prefill cannot block monitoring requests. There
is no JavaScript build step, CDN, database, Python service or extra runtime
package. The page is embedded in the executable, including installed/container
builds.

![Runtime monitor desktop preview](assets/monitor-desktop.png)

The preview uses explicitly labelled fixture request counters, not a model
benchmark. [Mobile preview](assets/monitor-mobile.png).

## Request workbench

The page follows the latest request by default. Select a request ID in the history
to inspect it; selecting it again resumes following the latest request. Its journey
splits measured wall time before and after first output. The first span includes
queue time; the second is not isolated GPU time or a GPU bottleneck diagnosis.
Before first output is observed, no prefill completion percentage is inferred.

Pin a finished request as a reference to compare first-output latency and output
rate with another request. The reference is an in-memory copy, cleared on an engine
restart or page reload. Different token counts trigger a comparison warning; even
equal counts do not establish equal prompts or benchmark conditions. Hardware
readings live under **Engine and hardware details**. The compact chat dock remains
available beside the workbench, or below it on small screens.

## Start

Build from source as described in CONTRIBUTING.md, then:

```sh
./build/tqserve --model models/model.gguf --port 8086 --monitor-port 8087
```

Open `http://127.0.0.1:8087/`. Inference stays on port 8086. The monitor binds
to the same IPv4 address as `--host`. Omit `--monitor-port` to disable it;
its port must differ from the inference port. If using a container built from
this source, additionally publish the monitor port (`-p 8087:8087`) and append
`--monitor-port 8087` to the container arguments. Existing release images need
rebuilding to include this feature.

`--api-key` and `--api-key-file` protect monitor data with the same Bearer key.
The static page can load without a key, then asks for it. The key stays in page
memory: it is not written to storage or passed in URLs. In authenticated
setups, use an API client with a Bearer header to inspect JSON. No model-management or
administration endpoint is exposed on this listener. The chat bridge forwards only to this instance's configured inference port; it cannot target an arbitrary URL or execute shell commands. It has no CORS allowlist;
other applications should proxy the authenticated JSON from their own backend.

## Contract v1

- `GET /` or `/monitor`: the embedded dashboard.
- `GET /api/monitor/v1/snapshot`: cached engine/request counters and hardware.
- `GET /health`: health of the monitor listener, **not** a GPU readiness probe.
- `POST /api/monitor/v1/chat/completions`: normal OpenAI chat-completions JSON,
  including `stream: true`, proxied to this instance's inference endpoint. This
  route requires the same Bearer key, JSON content type and same-origin requests
  from browsers. There is one concurrent monitor chat, a 1 MiB request limit and
  a ten-minute connection limit. Telemetry remains responsive during this chat.
  Closing the stream propagates a disconnect upstream; inference observes that
  cancellation at its next stream callback, not midway through a blocked prefill.

The right-hand terminal-style chat uses this bridge. It is a chat, not an
operating-system shell.

The dashboard polls once per second while visible. Requests have a three-second
fetch timeout; failures clear live values, mark the connection disconnected and
retry. A five-second freshness guard clears live values after a stalled fetch.
Engine identity and dimmed request history remain visible as last-known data.

Snapshots include `schema_version`, per-process `instance_id`, uptime,
model alias/build/host/PCI identity, configured slots and per-slot context,
requests, optional hardware and explicit capability flags. The snapshot does
not include prompt text, output text, paths to model files, API keys or raw
error strings. Engine calls never occur in the monitoring thread.

| Field | Meaning |
| --- | --- |
| `phase` | queued, prefill, decode, complete, failed or cancelled (accepted requests only; not the operating system TCP backlog) |
| `prompt_tokens` | Tokenizer count for the current generation pass |
| `output_tokens` | Observed generated token IDs; cumulative across thinking-budget continuations |
| `context_tokens` | Current pass's prompt + generated IDs; logical sequence length, not resident KV bytes |
| `first_output_ms` | Time from admission to first observed output batch, including queue time; not a per-token device timestamp |
| `decode_tokens_per_second` | Mean decode rate; first prefill-produced token excluded per pass. Live uses host elapsed time; completed uses the engine's timings |
| `prefill_ms`, `decode_ms` | Engine timings after a pass completes; cumulative across continuation passes |
| `drafted`, `accepted` | Engine-reported draft counters, populated after a generation pass |
| `elapsed_ms` | Request lifetime, including queue time; frozen at termination |
| `hardware.sample_age_ms` | Age of the last host/device sample; unavailable readings are null |

The chart uses deltas of observed IDs between snapshots, not SSE chunk counts.
Its history is local to this browser page, up to 60 seconds. Engine restarts and
completed-history eviction produce a gap rather than a throughput spike.
Snapshots retain at most 32 completed requests and all currently active ones;
the dashboard shows the newest 14. Output/draft totals describe this retained
window, not all-time totals.

Linux CPU activity and host memory come from `/proc`. Device utilization,
used/total memory, temperature and power use optional, dynamically loaded NVML.
The collector matches the engine's CUDA PCI bus ID, including when
`CUDA_VISIBLE_DEVICES` remaps device indices. Device memory is whole-device
usage, not this model's allocation. No NVIDIA collector is needed for the
pure CPU monitor tests. Unsupported hardware is unavailable, never an invented
zero or another host's readings. This listener currently follows tqserve's
Linux/POSIX support; it does not add a Metal server.

## Limits

Intermediate prefill completion, cached-prefix counts, resident KV bytes and
expert-cache statistics are not yet exported. `capabilities.prefill_progress`,
`expert_cache` and `cancel` are false. The page shows prefill as a phase with
elapsed time, not an inaccurate completion percentage. Long prefills may have
no new token counters until the model returns from `spec_begin`.

The monitor starts after model load. It is an engine dashboard; it does not
include Arena's multi-provider chat, tools, rooms or fleet orchestration. A
future Arena adapter can consume the same versioned snapshot rather than
interpreting engine counters differently. No inference arithmetic, CUDA graph,
weight format or memory-fit policy is changed.

## Verify

```sh
cmake -S . -B build-cpu -DTQLLM_CUDA=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure

# Optional browser gate; install playwright and its three browsers first.
python3 tests/test_monitor_ui.py --binary build-cpu/test-monitor --output monitor-artifacts
```

The C++ gate covers lifecycle/counters, continuation passes, bounded history,
concurrent reads, ticket lifetime, authentication, read-only routing and actual
HTTP responsiveness while the test inference owner is blocked. Browser gates
use that same production C++ listener with fixture counters and no application
API mocks: authentication, live/idle/prefill/error states, storage hygiene,
mobile overflow and disconnect clearing on Chromium, Firefox and WebKit.
Real-model sampling, scheduling and output-parity checks remain part of the
engine release gate; a fixture screenshot is not such a gate.

## Terminal chat

The right-hand pane streams from this instance through the existing same-origin
bridge. Thinking is off by default; its toggle changes the chat template option.
Enter sends, Shift+Enter inserts a line, and Stop/Escape cancel. Clear discards
transcript and context, including any response still in flight. Interrupted or
failed responses are not silently reused as completed conversation history.

Text and reasoning are rendered as text, without executing model HTML. The
server key remains in tab memory and is omitted when the server requires no key.
Theme changes also remain in memory. No terminal commands are executed.

Chat token counts and throughput are shown only when reported by the server;
otherwise the footer labels events as chunks. Time to first output includes
queue time, so it is not used to invent prefill throughput. Allocated context is
not prefill progress; the progress bar requires an explicit processed-token
counter and capability from the backend.

The browser gate submits through the actual chat UI and C++ listener, including
fragmented SSE/UTF-8, reported timings, unknown token counts, abort/Escape,
Clear-follow-up isolation, authentication and keyless servers. No model or GPU
job is needed for this transport gate.

## Real-model regression gate

Commit `3095140` passed a monitor-off/on check on an RTX 5060 Ti 16 GB,
CUDA 12.9, with `Qwen3.5-9B-UD-Q4_K_XL.gguf`. The same binary, model, ordered
prompts, greedy settings and seed produced identical token IDs, text and finish
reasons on all five prompts (311 generated IDs per arm). Cases include arithmetic,
code, explanation, multi-turn recall and a longer repeated-text prompt.

There were 32 real snapshots during generation, with a maximum observed response
time of 20.5 ms. The actual chat bridge and browser UI returned output, usage and
engine timings; the mobile page had no horizontal overflow or JavaScript errors.
These are scoped runtime/parity checks, not a throughput benchmark, a general
quality evaluation, or proof across all models/platforms/speculative modes.
[Inputs, model/binary hashes and per-case output evidence](validation/monitor-3095140.json).
