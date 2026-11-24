# Making an FNT Flash-Next file

[Documentation](../README.md) / [Project home](../../README.md)

FNT stores Qwen3.8-Flash-Next's routed experts as trellis codes at 2 to 3.5 bits per weight
([format](../reference/formats.md#fnt-trellis-coded-moe-experts-at-2-to-35-bits-per-weight)). This page makes a file
from the original Hugging Face weights: every routed-expert tensor of the ISTA-DASLab GSQ-RCO IQ3_XXS file is encoded
again in FNT, every other tensor (attention, GDN, shared experts, router, embeddings, the n-gram table) is that file's.
The tool is `tools/convert/qwen4exp/fnt_quant.py`.

## Result (RTX 3090, 59 GiB RAM)

| | ISTA IQ3_XXS | FNT |
|---|---:|---:|
| shard 1 (experts + dense) | 47.04 GB | 41.52 GB |
| routed experts, bits per weight | 2.84 | 2.48 |
| KLD vs the Q8_0-expert reference, mean / median / 99 % | 0.1981 / 0.0155 / 2.82 | 0.1451 / 0.0093 / 2.16 |
| same top token as the reference | 88.98 % | 90.60 % |
| PPL (reference 1.9934) | 2.1578 | 2.0947 |

KLD: WikiText-2 test, 8 x 2048 tokens; the reference is llama.cpp (b12624) on a copy of the ISTA file whose routed
experts are Q8_0 of the BF16 weights (so the number measures the expert quantisation alone; llama.cpp on the ISTA file
gives 0.1960 against it, tqLLM 0.1981); `tqkld --model <manifest> --llama-base <base> --chunks 8`.

Per tensor, FNT's rate is the smallest whose output error on held-out calibration inputs is at most the ISTA tensor's
(median ratio <= 0.85 and no expert above 1.3 over 16 experts): gate / up IQ2_XXS, IQ2_XS and IQ2_S -> FNT4 (2 bits),
IQ3_XXS and IQ3_S -> FNT6 (3 bits); down GSQ Q2_0 -> FNT4, IQ4_NL -> FNT7 (3.5 bits). Measured ratios (FNT / ISTA,
median): IQ2_XS 0.32-0.39, IQ2_XXS 0.31-0.35, IQ2_S 0.82, IQ3_XXS 0.45, IQ3_S 0.45-0.50, Q2_0 0.49-0.52, IQ4_NL 0.69.

## Requirements

An NVIDIA GPU with 24 GB, 64 GB of RAM, `nvcc` on `PATH`, Python 3.10+ with `numpy` and PyTorch (CUDA), a tqLLM build
with CUDA, the BF16 weights (`hf download Qwen/Qwen3.8-Flash-Next`, 360 GB), the ISTA IQ3_XXS shards and the tqLLM
runtime files (`YTan2000/Qwen3.8-Flash-Next-tqLLM`: tokenizer, MTP layer, draft vocabulary), ~45 GB of disk per 12
layers of calibration capture and ~40 GB for the encoder output. Time on an RTX 3090: 4 captures of 25 min, ~5 min of
encoding per layer (4 h for 48 layers).

## 1. Calibration capture

Any representative text works; the file above used 642K tokens in 194 prompts of ~3.3K tokens (English and
multilingual Wikipedia, FineWeb-Edu, source code, GSM8K / MATH solutions, UltraChat dialogues and OpenThoughts
reasoning traces in the chat template). Run the IQ3_XXS file in tqserve with the capture switch and send every prompt
through `/v1/completions` with `max_tokens: 1`:

```sh
TQLLM_DEBUG=moe_dump:0-11:/data/dump/g0 ./build/tqserve --model flashnext-iq3xxs.tqfn --tokenizer flashnext-tokenizer.gguf \
  --context 16384 --max-tokens 1 --np 1 --port 8086
```

The server appends, per layer of the range, the FP16 expert inputs (`L<layer>.x16`) and the routing (`L<layer>.rt`).
Record the file sizes before the first prompt in `start.json` (`{"L0.x16": bytes, "L0.rt": bytes, ...}`) so the load-time
rows are skipped. One capture holds 12 layers (~38 GB); repeat for 12-23, 24-35, 36-47.

## 2. Encode (per layer, resumable)

```sh
Q=tools/convert/qwen4exp/fnt_quant.py
ISTA=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
python3 $Q alloc --template $ISTA          # the per-layer rates (2.476 bits per weight in all)
for l in $(seq 0 11); do python3 $Q encode --layer $l --bf16 Qwen3.8-Flash-Next --dump /data/dump/g0 --out /data/enc --template $ISTA; done
```

Per chunk of 8 experts the encoder builds each expert's Hessian from its routed inputs (shrunk toward its response to
4096 random tokens of the layer: rare experts), runs BlockLDLQ with a tail-biting Viterbi search per 128 weights, refits
the 640-group scales in closed form, and encodes the down projection on the inputs of the already-quantised gate / up.
It prints each chunk's relative output error. `--holdout` leaves out every 4th 2048-row chunk of the capture (for the
allocation checks above).

## 3. Assemble, verify, serve

```sh
python3 $Q assemble --template $ISTA --enc /data/enc --out Qwen3.8-Flash-Next-FNT-00001-of-00002.gguf
python3 $Q verify --file Qwen3.8-Flash-Next-FNT-00001-of-00002.gguf --enc /data/enc --bf16 Qwen3.8-Flash-Next
```

`verify` decodes every FNT tensor with the tool's own decoder, checks it byte for byte against the encoder output and
prints the relative weight error against BF16 (`144 matrices checked, 0 differ`). The manifest lists the FNT shard
instead of the ISTA shard 1 and keeps everything else:

```
gguf=Qwen3.8-Flash-Next-FNT-00001-of-00002.gguf
gguf=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
mtp=mtp-blk48-iq4.gguf
draft_vocab=atx_65536.txt
slots=auto
chunk=8192
```

FNT layers serve their expert cache misses on the GPU (there is no fast CPU decoder for the trellis code), and the
hot / cold expert store (`hot_map`) re-quantises cold FNT experts through the CPU reference decoder (slow at start-up).
