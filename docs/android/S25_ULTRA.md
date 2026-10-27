# Samsung Galaxy S25 Ultra: native Android backend

Status: experimental native CPU text inference. No llama.cpp/ggml, CUDA, MLX,
QNN or Python runtime is linked into the Android executable. This is a CLI,
not an APK. Phone throughput, thermal behavior and battery use are not yet measured.

## Model choice

Start with **Qwen3.5-4B, standard Q4_K_M GGUF**, text only. It balances capability
and memory headroom on the S25 Ultra's Snapdragon 8 Elite and typical 12 GB RAM.
The selected file is 2.74 GB; that is the file size, not total inference RAM.
The CPU backend keeps weights packed in a read-only mapping and decodes only
one row per worker, rather than expanding the model to FP32.

- Model: https://huggingface.co/Qwen/Qwen3.5-4B
- Quant: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- Pinned revision: `e87f176479d0855a907a41277aca2f8ee7a09523`
- File: `Qwen3.5-4B-Q4_K_M.gguf`
- Bytes: `2740937888`
- SHA256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- License: Apache-2.0; preserve the model license when distributing it.

Download and verify on the build host:

```sh
mkdir -p models
curl --fail --location --output models/Qwen3.5-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf
printf '%s  %s\n' \
  00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4 \
  models/Qwen3.5-4B-Q4_K_M.gguf | sha256sum --check
```

Qwen3.5-2B Q4_K_M is the smaller candidate for lower RAM use and faster replies;
it needs its own measured quality/speed comparison. Neither model is declared
best across every workload. This first choice favors useful answer quality.
No new quantization format is required. tqLLM-specific rotated TQ formats are
not implemented by this backend and fail explicitly.

## Build

Install CMake 3.22+, Python 3 for the build helper, and Android NDK r27c using
Android Studio or the official NDK archive. No Android Studio project is needed.

```sh
python3 tools/android/build.py --ndk /path/to/android-ndk-r27c
```

This builds `build-android-arm64/tqandroid-run` and `test-android-native`, targeting
Android API 28+, arm64-v8a, with a statically linked C++ runtime. The phone runs
Android 15 or newer, depending on updates. There are no non-system shared-library
dependencies. Do not try to run the ARM64 binary directly on an x86 build host.

For a native Termux build, install its compiler and CMake, then:

```sh
cmake -S . -B build-phone -DTQLLM_CUDA=OFF -DTQLLM_ANDROID=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-phone --target tqandroid-run -j 4
```

## Run on the phone

Enable Developer options and USB debugging, connect a USB cable and authorize
the host. Install Android platform-tools on the host. No root or bootloader
unlock is required. Select the exact serial printed by `adb devices`:

```sh
adb devices
python3 tools/android/run.py --serial PHONE_SERIAL \
  --binary build-android-arm64/tqandroid-run \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt 'Explain why the sky is blue in three sentences.'
```

The helper deploys only to `/data/local/tmp/tqllm`, runs the model, and pulls
`android-metrics.json`. Subsequent runs can use `--skip-model-push` if that
same model is already deployed. Do not use that option after changing models.
Keep the phone awake during initial validation. Android background and thermal
policies can affect a sustained run.

To run without the helper:

```sh
adb -s PHONE_SERIAL shell '/data/local/tmp/tqllm/tqandroid-run --model /data/local/tmp/tqllm/model.gguf --prompt "Say hello." --context 4096 --threads 4 --max-tokens 128'
adb -s PHONE_SERIAL shell /data/local/tmp/tqllm/test-android-native
```

Push `test-android-native` to that directory first if running the second command.
The executable emits text to stdout and JSON metrics to stderr. `--metrics FILE`
writes the metrics separately. Ctrl-C requests cancellation between tokens.
The initial Qwen3.5 prompt explicitly disables thinking; `--thinking` enables it.
`--raw` accepts an already formatted prompt. Chat history must be rendered into
that raw prompt by the caller; this first CLI does not maintain a conversation.

## Correctness and measurement

CPU host verification, including an independent NumPy hybrid-model oracle:

```sh
cmake -S . -B build-host -DTQLLM_CUDA=OFF -DTQLLM_ANDROID=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j 4
ctest --test-dir build-host --output-on-failure
```

The `android_reference` test needs NumPy in CMake's selected Python interpreter.
Pass `-DPython3_EXECUTABLE=/path/to/python` if needed. It checks five causal
prefixes across GDN and attention, reset after another sequence, and invalid
inputs. The low-level test checks Q4_K packed dot products, worker exceptions,
KV bounds, unsupported formats and mapping bounds.

For a reference comparison, `--token-ids-file FILE` reads whitespace-separated
IDs and bypasses prompt templating. `--logits FILE --max-tokens 0` writes final
prompt logits as native little-endian FP32. `--output-ids FILE` saves emitted
IDs. Compare the same GGUF and input IDs to a separate reference executable;
llama.cpp may be used as a validation tool, but is not a backend dependency.

Metrics distinguish loading, prefill, TTFT, decode and generation wall time.
The first predicted token belongs to prefill. `decode_tps` counts completed
single-token forwards after prefill, including a forward that predicts EOS;
`generated_tokens` excludes EOS. Wall time includes output streaming. Do not
report host CPU measurements as S25 Ultra performance.

Phone gates before adopting this as a daily driver:

1. Real-model logits and greedy IDs vs a same-file reference, English and Chinese.
2. Native vs scalar Arm numerical checks; explicit tolerance for NEON reduction order.
3. 512/2048-token prefill and 128/512-token generation, interleaved thread counts 2/4/6/8.
4. Ten-minute sustained run, recording thermal state, memory, battery and throttling.

## Native architecture and next work

`src/backends/android` implements dense qwen35 inference directly in C++. It
reuses the shared GGUF reader, Qwen tokenizer and host weight codecs. Packed
matrices live in `src/formats/weight/host_matrix.hpp`, persistent workers and
mapping in `src/core`, and full-attention KV storage in `src/kv`. GDN convolution
history and recurrence belong to each model instance. Reset makes prior KV
inaccessible by restarting positions and clears recurrent state.

Supported weights: F32, F16, BF16, Q4_K, Q5_K, Q6_K and Q8_0. FP32 activations
and state are the numerical baseline; do not claim bit parity with the CUDA or
MLX paths. Explicitly unsupported: MoE, MTP/DFlash, vision, RoPE scaling and
speculative rollback. One caller per model instance.

This initial implementation processes prompt tokens sequentially. It is a
correctness baseline, not the final mobile speed path. Next optimize packed
Arm matvec and batched prefill, then add native Adreno compute kernels behind
the same model operations. Neither GPU nor NPU acceleration is implemented yet.
The existing CUDA serving binary does not acquire Android support from this flag.

Reference reading: llama.cpp's Android build guide and Qwen3.5 graph/layout,
plus tqLLM's existing CUDA and MLX implementations. No upstream contribution
or runtime dependency is introduced.

## Install directly from Termux over SSH

A deployment host can stage the release ZIP, the model and
`tools/android/install-termux.sh` in its SSH user home under
`.cache/tqllm/android-s25/`. The installer verifies pinned model and ZIP hashes,
resumes interrupted transfers with rsync, runs the native unit test on the
phone, and installs the `tqandroid` command. It needs about 2.9 GB of free
Termux app storage. It installs Termux packages `openssh`, `rsync`, `unzip`
and `coreutils`. No Android SDK, ADB, NDK or compiler is needed on the phone.

```sh
pkg install -y openssh
ssh USER@HOST 'cat .cache/tqllm/android-s25/install-termux.sh' > "$HOME/install-tqandroid.sh"
bash "$HOME/install-tqandroid.sh" USER@HOST
tqandroid "Say hello in English and Chinese."
```

Stop if the SSH copy fails; do not execute an incomplete installer. Re-running
the installer resumes a partial model transfer. The launcher uses 4K context,
four CPU threads and a 256-token output limit; pass explicit CLI options to
override these defaults. The first runner does not remember previous requests.
