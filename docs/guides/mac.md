# tqLLM on a Mac (Apple Silicon, Metal)

`tqserve` builds natively for Apple Silicon on MLX and Metal. It serves Qwen3.8-27B with the DFlash2 drafter, the same
OpenAI-compatible API and the same monitor as the CUDA build. The Docker image is CUDA only; on a Mac, install with the installer or Homebrew, or build from source.

Measured on an M5 Pro with 48 GB unified memory. M-series chips without the M5 matrix unit run, but more slowly.

## Install

Homebrew (macOS 15 or newer):

```sh
brew install turbo-tan/tap/tqllm
tqllm serve          # first run downloads the model and drafter (~17 GB), then serves on 127.0.0.1:8086
```

or the one-line installer (tries Homebrew, then pip in a venv under `~/.tqllm`, then a source build; `--check` reports
which route will work without installing; `--index-url URL` or `--wheels DIR` for networks that block PyPI):

```sh
curl -fsSL https://tqllm.com/install.sh | bash
tqllm serve
```

Both build `tqserve` against the stock MLX 0.32.4 wheel. `tqllm serve` downloads the best model file that is already
downloaded or published (TQ_L-QSA, else the mlx-community 4-bit file on Macs with 40 GB or more, else TQ_L) plus the
drafter; `--qsa` takes the QSA file for long contexts, `--context` sets the context (64K by default), and `--model` takes a
path or a Hugging Face `owner/repo` / `owner/repo/file.gguf`. `tqllm claude`, `tqllm codex` and `tqllm opencode` start
the agent pointed at the server, and start the server first when none is running (`tqllm stop` ends it). To build by hand instead,
follow the steps below.

## What you need

- macOS on Apple Silicon, Xcode command line tools, CMake 3.25 or newer.
- MLX 0.32.3 built from source with the patch in [`third_party/mlx`](../../third_party/mlx/README.md). The patch adds
  one call (`scaled_dot_product_attention_lse`) that tqLLM uses for attention over long contexts. The code also compiles
  against a stock MLX 0.32 or newer (it then uses a slower attention path at long context), but that build has not been
  tested yet, and the numbers below were measured with the patch.
- About 20 GB for the model files.

## 1. Build MLX with the patch

```sh
git clone --branch v0.32.3 https://github.com/ml-explore/mlx.git
cd mlx
patch -p1 < /path/to/tqllm/third_party/mlx/mlx-0.32.3-sdpa-lse.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local/mlx-tqllm"
cmake --build build -j 8 && cmake --install build
```

## 2. Build tqserve

```sh
cd /path/to/tqllm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTQLLM_CUDA=OFF -DTQLLM_MLX=ON \
      -DCMAKE_PREFIX_PATH="$HOME/.local/mlx-tqllm"
cmake --build build --target tqserve -j 8
```

## 3. Get the model files

Two choices of target weights. Both use the same drafter.

| Target | File | Size | Notes |
|---|---|---|---|
| mlx-community 4-bit, in tqLLM's Apple Silicon layout | `Qwen3.8-27B-MLX-4bit.gguf` | 15.8 GB | The weights behind the numbers below; the same codes as `mlx-community/Qwen3.8-27B-4bit`, repacked without requantising |
| tqLLM TQ_L | `Qwen3.8-27B-TQ_L.gguf` from [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) | 14.7 GB | Smaller and closer to the original model (KLD 0.022 vs 0.047), slower on the Mac today |
| DFlash2 drafter, 4-bit | `Qwen3.8-27B-DFlash2-MLX-4bit.gguf` | 1.3 GB | z-lab's DFlash2 drafter in the same layout |

The two Mac files are published at [YTan2000/Qwen3.8-27B-MLX-tqLLM](https://huggingface.co/YTan2000/Qwen3.8-27B-MLX-tqLLM).
The converters that made them are in [`quant/`](../../quant): `mlx_to_nax_gguf.py` and `dflash2_to_nax.py`.

## 4. Run

```sh
DYLD_LIBRARY_PATH="$HOME/.local/mlx-tqllm/lib" ./build/tqserve \
  --model Qwen3.8-27B-MLX-4bit.gguf -md Qwen3.8-27B-DFlash2-MLX-4bit.gguf \
  --context 262144 --port 8086
```

Open `http://localhost:8086/` for the monitor; applications use `http://localhost:8086/v1`.

- `--exact on` is the default: greedy output with the drafter is the same as plain decoding, token for token.
- A thinking budget (`--thinking-budget N`) closes the thinking after N tokens and continues the answer on the
  request's own state: only the closing tag is prefilled, not the whole prompt and thinking again.
- `--exact off` adds the measured approximations (int8 KV cache from 32K keys and an 8-bit query in attention).
- The server picks the KV cache format per request from free memory: BF16 while it fits, int8 when it does not
  (one `WARN` line under `--exact on`). On a 48 GB Mac a 256K-token prompt runs with an int8 cache.

## Memory (48 GB M5 Pro, mlx-community 4-bit + drafter)

| State | In use |
|---|---|
| Idle server, model and drafter loaded | 17.9 GB |
| Decoding after a 256K-token prompt (int8 KV) | 29.5 GB |
| Peak during that prompt's prefill | 34.2 GB |

The KV cache is freed when a request ends, so an idle server returns to 17.9 GB.

## Speed

Decode tokens per second, one request, greedy, 256 output tokens after long-document prompts, the same mlx-community
4-bit weights for all three engines, same client.

| Context | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|---|
| tqLLM 1.4.1 | 51.9 | 62.1 | **56.1** | **48.8** | **38.9** | **27.5** |
| tqLLM 1.4.1, QSA file | dense | dense | dense | 45.7 | **49.6** | |
| TensorFold 1.2.1 | **61.0** | 54.4 | 49.9 | 41.4 | 34.7 | does not fit |
| Splash 1.4.0 (int8 KV) | 60.8 | **62.4** | 48.4 | 43.9 | 35.9 | 24.5 |

Every tqLLM text equals plain one-token-at-a-time decoding (`--exact on`). The QSA file (`Qwen3.8-27B-MLX-4bit-QSA.gguf`)
reads only the 2,048 keys its indexers pick once a step is past 48K keys (`--qsa-from`), which changes the text.

Short prompts are mixed: tqLLM leads on some (99.6 vs TensorFold 91.1 on a 2K document) and trails on others (80.6 vs
97.0 on a chat prompt); see [Benchmarks](../BENCHMARKS.md).
