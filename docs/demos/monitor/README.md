# Runtime monitor walkthrough

[Project home](../../../README.md) / [Monitor reference](../../MONITOR.md)

[Watch the full 50-second recording (MP4)](walkthrough.mp4)

[![Monitor walkthrough](walkthrough.gif)](walkthrough.mp4)

The animation is a 20-second preview; the MP4 shows the full walkthrough at its original speed.

![Completed request](completed.png)

[Live-prefill screenshot](../../assets/monitor-prefill.png) / [Light-theme screenshot](light.png).

## Try it

Build the source and start your supported model:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tqserve -j4
./build/tqserve --model models/model.gguf --port 8086
```

1. Open **http://localhost:8086/** or **http://localhost:8086/monitor**. No extra port is required.
2. Send a long prompt in the chat panel, or call `/v1/chat/completions` from your client. The dashboard observes both.
3. During prefill, watch completed/total prompt tokens and **PREFILL RATE**. The rate appears after a chunk finishes. Prefill and output rates stay visible together in the same card after generation starts and after completion. Cached-prefix tokens count toward progress, but not throughput.
4. Select a request in the history and use **Pin as reference** to compare it with another request.
5. Use the theme icon for light/dark mode. Hardware samples refresh every three seconds; request telemetry refreshes every second.

To keep a separate dashboard listener, add `--monitor-port 8087`. The API-port
page still works. API keys protect telemetry and chat with the same Bearer key
as inference; the static page itself can load before you enter the key.

## What the recording shows

This is a real Qwen3.8-27B-TQ4_L GGUF on an RTX 3090, with a 16K context,
512-token prefill chunks and greedy generation. It is a usage walkthrough,
not a performance comparison. The local hostname is blurred; telemetry and
model output are not replaced or accelerated. Captions explain each step.

## Record your own

Install the optional contributor tools (`pip install playwright` and
`playwright install chromium`), then run from the repository root against a
running server with a 16K context. The recorder submits a synthetic document
through the monitor's real chat form; it does not mock the API.

```sh
python docs/demos/monitor/record.py --url http://127.0.0.1:8086 --output monitor-recording
```

Use a private/local test server for recording. The script never reads API-key
files or injects credentials. Convert the generated WebM to MP4 for sharing;
the walkthrough above includes an animated preview and a downloadable MP4.
