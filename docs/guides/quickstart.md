# Installation and quick start

[Documentation](../README.md) / [Project home](../../README.md)

## Get the image

The prebuilt image is public on the GitHub container registry and on Docker Hub. One multi-platform tag covers x86-64
(RTX 30 / 40 / 50) and arm64 (GB10 / DGX Spark); `docker pull` picks the right one.

```sh
docker pull ghcr.io/turbo-tan/tqllm:latest
# or
docker pull ytan2000/tqllm:latest
```

`latest` always points to the newest release; pin a version with its tag (for example `1.5.0`, see the
[changelog](../../CHANGELOG.md)).

## Docker quick start

Requires Linux, an NVIDIA GPU, Docker Compose, and NVIDIA Container Toolkit.
One image tag covers x86-64 PCs with RTX 30, 40 or 50 series GPUs (CUDA architectures 86, 89, 120) and arm64
GB10 / DGX Spark systems (121); `docker pull` picks the right platform and the engine picks its kernels for your GPU at
start. The image does not cover Apple Silicon: on a Mac, install with `brew install turbo-tan/tap/tqllm` ([Mac guide](mac.md)). On Windows, run the image
through WSL 2, or build `tqserve.exe` natively ([Windows guide](windows.md)).

Place a supported model at `models/model.gguf` (the tokenizer is read from the GGUF). Which models and files load, and
which were checked on the current release, is listed in [Supported models](../../README.md#supported-models).

Then run:

```sh
docker compose up            # uses the prebuilt image
docker compose up --build    # or build the image from this source
```

To use an existing models directory and different filenames:

```sh
MODEL_DIR=/path/to/models MODEL_NAME=your-model.gguf docker compose up --build
```

Models are mounted read-only. The API listens at `http://localhost:8086`.
Context defaults to 16384; increase CONTEXT only when the model fits your GPU.

```sh
curl http://localhost:8086/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain a hash map."}],"temperature":0,"max_tokens":128}'
```

## Native build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tqserve -j4
./build/tqserve --model models/model.gguf --port 8086
```

Needs CMake 3.25 or newer (CUDA C++20; `python3 -m pip install cmake` gets one on older distributions) and a recent
CUDA toolkit; CUDA 12.9 and 13.0 were tested on RTX 3090.
Model downloads are not included. Model weights and bundled third-party source
retain their own licenses. tqLLM itself is licensed under Apache-2.0 (see LICENSE and NOTICE).


## Tokenizer conversion

GGUF models need no tokenizer file: tqserve reads it from the GGUF metadata. Only safetensors checkpoints, or a
deliberate override (`--tokenizer FILE`), need a tokenizer.tqtok. Build it from tokenizer.json and
tokenizer_config.json of the SAME original model family/version. Conversion needs Python 3, not PyTorch:

```sh
python3 tools/tqconvert.py --source /path/to/hf-tokenizer --out-dir models --tokenizer-only
```
