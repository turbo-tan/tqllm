#!/usr/bin/env python3
"""Cross-build the native tqLLM Arm backend with an existing Android NDK."""
import argparse
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ndk', type=Path, required=True)
    ap.add_argument('--build-dir', type=Path, default=Path('build-android-arm64'))
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.jobs <= 32:
        ap.error('--jobs must be in 1..32')
    toolchain = args.ndk.resolve() / 'build/cmake/android.toolchain.cmake'
    if not toolchain.is_file():
        ap.error('NDK toolchain missing: ' + str(toolchain))
    root = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    subprocess.run(['cmake', '-S', str(root), '-B', str(build),
                    '-DCMAKE_TOOLCHAIN_FILE=' + str(toolchain), '-DANDROID_ABI=arm64-v8a',
                    '-DANDROID_PLATFORM=android-28', '-DANDROID_STL=c++_static',
                    '-DCMAKE_BUILD_TYPE=Release', '-DTQLLM_CUDA=OFF', '-DTQLLM_MLX=OFF',
                    '-DTQLLM_ANDROID=ON', '-DBUILD_TESTING=ON'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'tqandroid-run', 'tqserve', 'test-android-native', 'test-android-packed', 'test-android-batch',
                    '--parallel', str(args.jobs)], check=True)
    print('Built:', build / 'tqandroid-run', build / 'tqserve')
    print('This is an Android/arm64 binary. Execute it on the phone with adb or Termux.')


if __name__ == '__main__':
    main()
