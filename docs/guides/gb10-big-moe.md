# MiniMax-M3.1-Flash and GLM-5.3-Flash on one GB10

Both models fit on one NVIDIA GB10 machine (DGX Spark, ASUS GX10 and similar: 128 GB of unified memory). tqLLM
serves them from the files their authors and the community publish, with no conversion step.

| Model | File you download | Size | Drafter | Slots |
|---|---|---|---|---|
| MiniMax-M3.1-Flash (428B total / 23B active) | [`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw), a Hugging Face folder (EXL3 weights + the DSpark drafter in `dspark/`) | ~115 GB | DSpark, included | 1 |
| GLM-5.3-Flash | `GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` (GSQ GGUF, e.g. from [`pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF`](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF); sha256 `a60586ce30494bf6...`) | 117 GB | none | 1 |

Build tqLLM on the GB10 first (see [installation](quickstart.md)); the commands below use `./build/tqserve`.

## MiniMax-M3.1-Flash

```sh
hf download vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw --local-dir minimax
./build/tqserve --model minimax -md minimax/dspark --context 32768 --reasoning on --port 8086
```

- `--model` takes the folder as downloaded: tqLLM reads the EXL3 trellis weights, the MXFP8 tensors, the tokenizer and
  the chat template from it.
- `-md minimax/dspark` turns on the DSpark drafter. The number of drafted tokens per step follows the measured
  acceptance; `--draft-max N` fixes it instead.
- When the context does not fit with an FP16 KV cache, tqLLM switches the cache to FP8 by itself and logs it.
- Prompts are processed in chunks on the tensor cores: about 537 tok/s on an 8K prompt at 16K context, 492 tok/s at
  32K.
- One request at a time (`--np 1`).

<!-- 1.3.0: decode row vs the vLLM + vllm-exl3 recipe goes here once measured on the release build -->

## GLM-5.3-Flash

```sh
./build/tqserve --model GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf --context 131072 --reasoning on --port 8086
```

- Sparse attention with the DSA indexer: contexts up to 128K on one GB10.
- Prompts are processed in chunks of 512 tokens (prefill about 100 tok/s).
- Turns end at GLM's end-of-turn token; the reasoning is returned separately from the answer.
- One request at a time (`--np 1`).

## Using them

Both serve the same OpenAI-compatible and Anthropic-compatible APIs as every tqLLM model
(`http://localhost:8086/v1`), and the monitor at `http://localhost:8086/`. See [coding agents](coding-agents.md) to
point an agent at them.
