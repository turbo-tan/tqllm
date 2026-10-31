#!/usr/bin/env python3
"""Compare TQLLM token IDs with the pinned Hugging Face tokenizers file."""
import argparse
import subprocess
from pathlib import Path

from tokenizers import Tokenizer


TEXTS = [
    "Hello, world!",
    "Say hello in one short sentence.",
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>",
    "Café — 東京 😊",
    "I'm testing\n whitespace  and\tpunctuation!!!",
    "  and",
    "foo123",
    "नमस्ते दुनिया",          # Devanagari: combining vowel signs / virama (\p{M})
    "مَرْحَبًا بِالْعَالَمِ",      # Arabic with harakat
    "Tiếng Việt có dấu",     # Vietnamese (NFC-composed)
    "<|im_start|>assistant\n<think>\n\n</think>\n\nOK<tool_call>{}</tool_call>",
    "x\u0301y",             # letter + combining mark with no precomposed form (NFC no-op)
    "int f() {\n    if (x) {\n        return 1;\n    }\n}\n",   # code indent: \s+(?!\S) keeps runs
    "a   \n\n   b\t\t\tc",   # spaces then newlines (\s*[\r\n]+), tab runs
    "trailing   ",
    "1234567 and 2026-10-24, pi 3.14159, 0007",   # digit runs: \p{N} (Qwen) vs \p{N}{1,3} (GLM)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    args = parser.parse_args()
    reference = Tokenizer.from_file(str(args.reference))
    failed = False
    for text in TEXTS:
        expected = reference.encode(text).ids
        output = subprocess.check_output([
            str(args.binary), "--tokenizer", str(args.converted), "--text", text
        ], text=True)
        actual = [int(token) for token in output.split()]
        passed = actual == expected
        print(f"{'PASS' if passed else 'FAIL'} {text!r}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
