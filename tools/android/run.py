#!/usr/bin/env python3
"""Deploy the native runner and a GGUF to an explicitly selected ADB device."""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--serial', required=True, help='Device serial from adb devices')
    ap.add_argument('--adb', default='adb')
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--prompt', default='Explain why the sky is blue in three sentences.')
    ap.add_argument('--threads', type=int, choices=range(1, 9), default=4)
    ap.add_argument('--context', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--skip-model-push', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.context <= 32768 or not 1 <= args.max_tokens < args.context:
        ap.error('invalid context or output budget')
    if not args.binary.is_file() or not args.model.is_file():
        ap.error('binary and model must be local files')
    adb = [args.adb, '-s', args.serial]
    def shell(*words):
        return subprocess.run(adb + ['shell', shlex.join(words)], check=True)
    subprocess.run(adb + ['get-state'], check=True)
    shell('mkdir', '-p', '/data/local/tmp/tqllm')
    subprocess.run(adb + ['push', str(args.binary), '/data/local/tmp/tqllm/tqandroid-run'], check=True)
    if not args.skip_model_push:
        subprocess.run(adb + ['push', str(args.model), '/data/local/tmp/tqllm/model.gguf'], check=True)
    shell('chmod', '700', '/data/local/tmp/tqllm/tqandroid-run')
    command = ['/data/local/tmp/tqllm/tqandroid-run', '--model', '/data/local/tmp/tqllm/model.gguf',
               '--context', str(args.context), '--threads', str(args.threads), '--max-tokens', str(args.max_tokens),
               '--metrics', '/data/local/tmp/tqllm/metrics.json', '--prompt', args.prompt]
    if args.thinking:
        command.append('--thinking')
    shell(*command)
    subprocess.run(adb + ['pull', '/data/local/tmp/tqllm/metrics.json', 'android-metrics.json'], check=True)


if __name__ == '__main__':
    main()
