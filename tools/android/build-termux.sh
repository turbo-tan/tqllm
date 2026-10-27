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
flags=(-O3 -DNDEBUG -march=armv8.2-a+dotprod+fp16 -std=c++20 -pthread -Wall -Wextra -Wpedantic
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
for entry in tools/tqandroid_run.cpp tests/android_native.cpp tests/android_packed.cpp; do
    case "$entry" in
        tools/*) output=tqandroid-run ;;
        tests/android_native.cpp) output=test-android-native ;;
        *) output=test-android-packed ;;
    esac
    "$compiler" "${flags[@]}" "$root/$entry" "${objects[@]}" -o "$build/$output"
done
"$build/test-android-native"
"$build/test-android-packed"
printf 'Built %s\n' "$build/tqandroid-run"
