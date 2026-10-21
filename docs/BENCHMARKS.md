# Benchmark results and reproduction

[Documentation](README.md) / [Project home](../README.md)

Run commands from the repository root. These are the existing measured results, not measurements of the monitor change.

## Performance

Measured on the model and drafter above (`Qwen3.8-27B-TQ3_4S-v2.gguf` + `Qwen3.8-27B-DFlash2-TQ3.gguf`), engine built
from the named commit, tq3 KV cache, greedy, 400 output tokens. Prompts are WikiText-2 test text cut to length with a
"summarise" instruction; each number is the median of 3 runs. tok/s is the server's own timing.

RTX 3090 24 GB, engine `de8e3c8`, context 40960:

| Prompt | Prefill tok/s | Decode, no drafter | Decode, DFlash2 drafter |
|---|---:|---:|---:|
| 2K | 2,258 | 55.5 | 155.2 |
| 8K | 2,426 | 54.5 | 149.9 |
| 32K | 2,155 | 50.6 | 131.0 |

RTX 5060 Ti 16 GB (WSL 2), engine `be4fd21`, drafter on, one run each: the memory fit chose a 16K context; 2K prompt
777 tok/s prefill / 94.5 tok/s decode, 8K prompt 821 / 76.9.

Speed with the drafter follows how often its guesses are accepted. Summaries of encyclopedia text accept about 35-40 %
of the drafted tokens; reasoning text accepts more, and the LocalMaxxing reasoning prompt runs at 256 tok/s (median of
3) on the RTX 3090 with the same files.

### Qwen3.8-Flash-Next (125B, 6B active) on one RTX 3090

The routed experts live in system RAM; the GPU keeps a cache of expert slots, misses go over PCIe by the copy engine and
the next layer's experts are prefetched while the current one runs. Native MTP drafts up to 5 tokens per step and
stops drafting when the draft is unsure; the output is identical to a fixed 3-token draft (greedy).

Same session, same file ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
IQ3_S) on both engines, RTX 3090 24 GB, greedy; [Strata](https://github.com/Niko1221/Strata) 0.2.2 (engine 0.1.47) with `--spec 4`:

| | tqLLM | Strata 0.2.2 |
|---|---:|---:|
| Short chats (5 x 512 tokens, each a fresh server's first request), tok/s | 91.6 | 89.6 |
| Time to first token, short chat | 0.32-0.41 s | 1.07-1.29 s |
| Time to first token, 32K prompt | 14.98 s | 15.14 s |
| Time to first token, 64K | 28.01 s | 29.18 s |
| Time to first token, 128K | 56.53 s | 58.25 s |
| Decode after 32K, tok/s | 76.4 | 72.2 |
| Decode after 64K | 85.1 | 63.6 |
| Decode after 128K | 77.1 | 47.2 |

On a warm server the same five chats run at 95.6 tok/s. Several users at once (`--np` up to 8) give the same text as one user alone. Run it from a manifest (`.tqfn`, one
`key=value` per line: `gguf=` per shard, `mtp=`, `experts=` (from `tqfn-pack`), `draft_vocab=`, `slots=auto`) with
`tqserve --model FILE.tqfn --tokenizer TOKENIZER.gguf`.

### Reproduce

```sh
sh bench/get_wikitext.sh                       # bench/data/wiki.test.raw (checksum-verified)
# native build: starts tqserve itself, without and with the drafter
python3 bench/relbench.py --bin build/tqserve --gpu "RTX 3090" --wiki bench/data/wiki.test.raw --out results.jsonl \
  --model v2=models/Qwen3.8-27B-TQ3_4S-v2.gguf \
  --drafter models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --vocab models/mtp-vocab/atx_65536.txt
# or against the Docker server above (start it with -e KV_FORMAT=tq3 to match)
python3 bench/relbench.py --url http://localhost:8086 --mode dflash2 --gpu "RTX 3090" \
  --wiki bench/data/wiki.test.raw --out results.jsonl --model v2=models/Qwen3.8-27B-TQ3_4S-v2.gguf
```

Each line of `results.jsonl` records the GPU, the model file, the settings and the engine commit the server reports
(`GET /health`). A second RTX 3090 (also driving a desktop display) reproduced decode within 1 % and prefill 4-15 % lower;
prefill varies more between cards than decode.
