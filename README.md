<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tqllm-logo-dark.png">
    <img src="docs/assets/tqllm-logo.png" alt="tqllm" width="360">
  </picture>
</p>

# tqLLM

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-davidytan2000-BD5FFF?logo=buymeacoffee&logoColor=white)](https://www.buymeacoffee.com/davidytan2000) [![GoFundMe](https://img.shields.io/badge/GoFundMe-support%20TQLLM-00B964?logo=gofundme&logoColor=white)](https://gofund.me/0f79d4274)

C++ inference for supported Qwen models and TurboQuant GGUF weights on NVIDIA GPUs (CUDA) and Apple Silicon (Metal),
with an OpenAI-compatible HTTP server.

[Quick start](docs/guides/quickstart.md) | [Documentation](docs/README.md) | [Benchmarks](docs/BENCHMARKS.md) | [Credits](docs/CREDITS.md)

**New in [1.6.0](https://github.com/turbo-tan/tqllm/releases/tag/v1.6.0)** ([all changes](CHANGELOG.md)):

![tqLLM 1.6.0](https://tqllm.com/assets/tqllm-1.6.0.png)

- **Long context, 3.2x faster on a 16 GB card:** QSA sparse attention for Qwen3.8-27B (`--qsa-from N`, opt-in,
  approximate). Past N keys each answer token reads only the key blocks the in-file indexer picks. RTX 5060 Ti 16 GB,
  TQ_XS-QSA, 200K prompt: 93.4 tok/s against 29.3. RTX 3090, TQ_L-QSA: 182.2 against 143.6 at 128K, 170.5 against
  119.5 at 256K.
- **Flash-Next over two linked boxes:** `--split experts` lets an idle second box compute half of each layer's experts;
  the output is the same as one box. Two DGX Sparks, one user: 68.4 tok/s against 64.4. With `--peer`, 16 users over
  both boxes: 238 tok/s.
- **Two GPUs in one PC:** `--split layer` for the dense 27B ([multi-GPU guide](docs/guides/multi-gpu.md)).
- **Mac, instant next turn:** `--prompt-cache on` resumes a chat from the state its last prompt kept; follow-up first
  token at 34K in 0.36-0.47 s.
- **int8 prefill for llama.cpp IQ4_XS GGUFs** (`--prefill-int8 on`), no extra VRAM.
- **Fixed:** garbage tokens on M1-M4 Macs (#38). **Known issue:** the 12 GB / 4-user copy-engine fault from 1.5.0 is
  not fixed; use `--np 2` on 12 GB cards.

Flash-Next setup (RTX 30 or newer with 24 GB and 64 GB RAM, or a GB10):
[YTan2000/Qwen3.8-Flash-Next-tqLLM](https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM).
[Benchmarks](docs/BENCHMARKS.md).

## Supported models

Checked on the release build: the model loads and answers fixed questions correctly (1.6.0: RTX 3090, DGX Spark / GB10, M5 Pro Mac).

| Model | Files tqLLM loads | Hardware | 1.6.0 check |
|---|---|---|---|
| Qwen3.8-27B and fine-tunes (e.g. Qwopus3.8) | TQ3_4S, TQ_L / TQ4, K-quant and IQ GGUFs; EXL3 folders; DFlash2 and MTP drafters | NVIDIA, 16 GB and up; DGX Spark / GB10 | passed |
| Qwen3.8-27B on a Mac | mlx-community 4-bit, TQ_L, QSA files ([Mac guide](docs/guides/mac.md)) | Apple Silicon (48 GB tested) | passed |
| Qwen3.8-Flash-Next (125B MoE, 6B active) | `.tqfn` manifest over the public IQ3_S GGUF, tqLLM's own file, Unsloth Dynamic v3.1 GGUFs, NVIDIA NVFP4 experts | 24 GB GPU + 64 GB RAM; DGX Spark / GB10 | passed |
| Qwen3.5 / Qwen3.6 27B | same formats as Qwen3.8-27B (same architecture) | NVIDIA | not re-run for 1.6.0 |
| GLM-5.3-Flash, MiniMax-M3.1-Flash | GSQ GGUF; EXL3 folder ([guide](docs/guides/gb10-big-moe.md)) | one GB10, one request | not re-run for 1.6.0 |

Not verified yet: Qwen3.5 / 3.6 mixture-of-experts GGUFs (`qwen35moe`, such as Qwen3.6-35B-A3B). The MoE models
(Flash-Next, GLM-5.3-Flash, MiniMax) do not run on native Windows yet; use WSL 2 there.

## See what your model is doing

Watch prefill, output rates and hardware use in one view. Both rates stay in the
same request card. Open `http://localhost:8086/` or `/monitor` on the API port.
No frontend build, extra service or second port is required.

![Prefill and output rates together in the tqLLM monitor](docs/demos/monitor/completed.png)

<details>
<summary>Watch the 50-second monitor walkthrough</summary>

[![Monitor usage recording](docs/demos/monitor/walkthrough.gif)](docs/demos/monitor/walkthrough.mp4)

20-second preview above; [full video and step-by-step instructions](docs/demos/monitor/README.md).
Recorded with a real model. The local hostname is hidden; telemetry is unmodified.

</details>

## Run locally

On a Mac (Apple Silicon, macOS 15 or newer):

```sh
curl -fsSL https://tqllm.com/install.sh | bash   # or: brew install turbo-tan/tap/tqllm
tqllm                                # starts the server in the background; the first run downloads the model and drafter
tqllm stop                           # stops it (log: ~/.cache/tqllm/server.log)
tqllm claude                         # Claude Code pointed at the local server (or: tqllm codex)
```

More in the [Mac guide](docs/guides/mac.md); on Windows see the [Windows guide](docs/guides/windows.md). The steps below are for Linux with an NVIDIA GPU. Place a supported Qwen GGUF at
`models/model.gguf`, then build from source:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tqserve -j4
./build/tqserve --model models/model.gguf --port 8086
```

Open **[localhost:8086](http://localhost:8086/)** for the monitor. Applications use
the same server at `http://localhost:8086/v1`.

Prefer Docker? The prebuilt image (x86-64 and arm64) runs the same server:

```sh
docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" --entrypoint tqserve \
  ghcr.io/turbo-tan/tqllm:latest --model /models/model.gguf --host 0.0.0.0 --port 8086
```

`latest` is the newest release; use a version tag such as `1.6.0` to pin one. Or build it locally with `docker compose up --build`. See [installation and requirements](docs/guides/quickstart.md).

Two or more machines with the same model? Start a normal server on each, then point one at the others; it becomes the
single address and sends each request to the box that will answer first (and a conversation back to the box that holds it):

```sh
./build/tqserve --model models/model.gguf --np 8 --port 8086 --host 0.0.0.0                 # box 2
./build/tqserve --model models/model.gguf --np 8 --port 8086 --peer http://box2:8086         # box 1: use this one
```

## Choose your next step

| You want to... | Read |
|---|---|
| Run on a Mac (Apple Silicon, Metal) | [Mac](docs/guides/mac.md) |
| Download a model and add a drafter | [DFlash2 and MTP](docs/guides/speculative-decoding.md) |
| Run MiniMax-M3.1-Flash or GLM-5.3-Flash on a DGX Spark / GB10 | [Big MoE models on one GB10](docs/guides/gb10-big-moe.md) |
| Run on an RTX 5090 (numbers, what is automatic) | [RTX 5090](docs/guides/rtx-5090.md) |
| Use a local coding agent | [Agent setup](docs/guides/coding-agents.md) |
| Serve several users or tune memory | [Server options](docs/reference/options.md) |
| Run a longer context | [TqCache](docs/guides/long-context.md) |
| Understand TQ3, TQ4 and compatible formats | [Weight formats](docs/reference/formats.md) |
| Inspect latency, compare requests and extend the dashboard | [Monitor guide](docs/MONITOR.md) |

## Contribute and credits

See [CONTRIBUTING.md](CONTRIBUTING.md) for building, testing and contribution
requirements, and [the architecture guide](docs/ARCHITECTURE.md) for the source map.

Built on work from Qwen, ggml/llama.cpp, Google Research, z-lab, Tom Turney,
spiritbuun, turboderp ([ExLlamaV3](https://github.com/turboderp-org/exllamav3): the trellis format and kernels behind
X3 and our EXL3 support) and the contributors listed in [Credits](docs/CREDITS.md).
Third-party notices remain in [NOTICE](NOTICE). Thank you to our community testers.

tqLLM is created by David Y. Tan and licensed under Apache-2.0: if you copy or redistribute it, keep the
[NOTICE](NOTICE) file's credit with your copy. Model weights and third-party components retain their own licences.
