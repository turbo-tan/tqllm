#!/bin/sh
# Downloads the WikiText-2 raw test split and drops its blank lines: the prompt text relbench.py measured with.
set -e
mkdir -p "${1:-bench/data}" && cd "${1:-bench/data}"
curl -fsSL -o wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o -j -q wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw && rm wikitext-2-raw-v1.zip
grep -v '^ *$' wiki.test.raw > wiki.test.tmp && mv wiki.test.tmp wiki.test.raw
echo "bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a  wiki.test.raw" | sha256sum -c - >/dev/null || { echo "wiki.test.raw checksum mismatch" >&2; exit 1; }
echo "$(pwd)/wiki.test.raw"
