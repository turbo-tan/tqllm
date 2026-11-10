# Native Windows

[Documentation](../README.md) / [Project home](../../README.md)

tqLLM builds and runs natively on Windows 10 / 11 with an NVIDIA GPU: no WSL, no Docker. The Docker image also works on
Windows through WSL 2 ([quick start](quickstart.md)).

What runs natively: `tqserve` with Qwen3.8 GGUFs (the 27B family, TQ and llama.cpp formats), the DFlash2 / DSpine
drafters and `--mtp`. The MoE engines (Flash-Next, GLM-5.3-Flash, MiniMax) are not ported yet; run them under WSL 2.

## Build

You need:
- Visual Studio 2022 (Community is fine) with the **Desktop development with C++** workload
- CUDA Toolkit 12.8 or newer (13.x works too)
- CMake 3.25+ and Ninja (`winget install Kitware.CMake Ninja-build.Ninja`)
- Git

From an **x64 Native Tools Command Prompt for VS 2022**:

```bat
git clone https://github.com/turbo-tan/tqllm.git
cd tqllm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=OFF
cmake --build build --target tqserve
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU: `86` RTX 30, `89` RTX 40, `120` RTX 50. The CUDA `bin` folder must be on
`PATH` when `tqserve.exe` runs (the CUDA installer adds it).

## Run

```bat
build\tqserve.exe --model C:\models\Qwen3.8-27B-TQ3_XS.gguf -md C:\models\Qwen3.8-27B-DFlash2-TQ3.gguf ^
  --draft-vocab C:\models\atx_65536.txt --host 127.0.0.1 --port 8086
```

The API is the same as on Linux (`/v1/chat/completions`, `/v1/models`, `/health`, the monitor at `/monitor`).

## Measured

RTX 5060 Ti 16 GB, Windows 11, CUDA 12.8, MSVC 19.30, `Qwen3.8-27B-TQ3_XS.gguf` (11.85 GB), greedy, thinking off,
three runs after a warm-up:

| prompt | prompt tok/s | decode tok/s, no drafter | decode tok/s, DFlash2-TQ3 |
|---|---|---|---|
| chat (30 tokens) | 270 | 33.8 | 160 |
| 2K | 1,190-1,228 | 33.5 | 116 |
| 6K | 1,170-1,186 | 32.9 | 102 |
