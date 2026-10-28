#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${1:-"$root/build-termux"}
compiler=${CXX:-clang++}
[[ $(uname -m) == aarch64 && -n ${PREFIX:-} ]] || {
    echo 'Run this helper in aarch64 Termux; use build.py for an NDK cross-build.' >&2
    exit 1
}
command -v "$compiler" >/dev/null
mkdir -p "$build"
build=$(cd -- "$build" && pwd)
flags=(-O3 -DNDEBUG -march=armv8.2-a+dotprod+fp16+i8mm -std=c++20 -pthread -Wall -Wextra -Wpedantic
       -I"$root/include" -I"$root/src/core" -I"$root/src/core/hf" -I"$root/src/kv"
       -I"$root/src/formats/weight" -I"$root/src/backends/android"
       -I"$root/third_party/nlohmann" -I"$root/third_party")
sources=(src/backends/android/android_model.cpp src/core/mapped_file.cpp
         src/core/gguf.cpp src/core/paw_gguf.cpp src/core/hf/hf_folder.cpp src/core/hf/hf_arch.cpp
         src/core/hf/arch_minimax_m3.cpp src/core/hf/arch_minimax_m3_dspark.cpp src/core/safetensors.cpp
         src/core/tokenizer.cpp src/formats/weight/dequant.cpp)
objects=()
"$compiler" --version > "$build/compiler.txt"
for source in "${sources[@]}"; do
    object="$build/${source//\//_}.o"
    "$compiler" "${flags[@]}" -c "$root/$source" -o "$object"
    objects+=("$object")
done
for entry in tools/tqandroid_run.cpp tests/android_native.cpp tests/android_packed.cpp tests/android_batch.cpp; do
    case "$entry" in
        tools/*) output=tqandroid-run ;;
        tests/android_native.cpp) output=test-android-native ;;
        tests/android_batch.cpp) output=test-android-batch ;;
        *) output=test-android-packed ;;
    esac
    "$compiler" "${flags[@]}" "$root/$entry" "${objects[@]}" -o "$build/$output"
done
"$build/test-android-native"
"$build/test-android-packed"
printf 'Built %s\n' "$build/tqandroid-run"

# tqserve (OpenAI-compatible HTTP server, the same tools/tqserve.cpp as the CUDA and MLX builds) over this backend
mkdir -p "$build/generated"
python3 - "$root" "$build/generated/monitor_page.hpp" <<'PY'
import sys
root, out = sys.argv[1], sys.argv[2]
html = open(f"{root}/tools/serve/monitor.html").read().replace("@TQLLM_MONITOR_PANELS@", open(f"{root}/tools/serve/monitor_panels.js").read())
page = open(f"{root}/tools/serve/monitor_page.hpp.in").read().replace("@TQLLM_MONITOR_HTML@", html)
open(out, "w").write(page)
PY
commit=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)
serve_flags=("${flags[@]}" -Wno-unused-parameter -DTQLLM_COMMIT="\"$commit\"" -I"$root" -I"$root/src" -I"$root/tools"
             -I"$root/tools/serve" -I"$root/third_party/jinja" -I"$root/src/engine" -I"$root/src/formats/kv"
             -I"$root/src/experts" -I"$root/src/spec" -I"$root/src/vision" -I"$build/generated")
serve_sources=(tools/tqserve.cpp src/backends/android/model_android.cpp src/core/options.cpp src/kv/kv_store.cpp
               tools/serve/jinja_template.cpp third_party/jinja/jinja/lexer.cpp third_party/jinja/jinja/parser.cpp
               third_party/jinja/jinja/runtime.cpp third_party/jinja/jinja/string.cpp third_party/jinja/jinja/value.cpp)
serve_objects=() pids=()
for source in "${serve_sources[@]}"; do
    object="$build/${source//\//_}.o"
    "$compiler" "${serve_flags[@]}" -c "$root/$source" -o "$object" & pids+=($!)
    serve_objects+=("$object")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$compiler" "${flags[@]}" "${serve_objects[@]}" "${objects[@]}" -o "$build/tqserve"
printf 'Built %s\n' "$build/tqserve"
