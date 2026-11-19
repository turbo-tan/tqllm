# Making a TQT model

[Documentation](../README.md) / [Project home](../../README.md)

TQT is a tqLLM weight format at TQ4's size (4.125 bits per weight) that stores each block of 128 weights as a
tail-biting bitshift-trellis code ([format](../reference/formats.md#tqt-a-trellis-code-at-tq4s-size)). It is the most
accurate tqLLM format at this size: on Qwen3.8-27B its KLD against BF16 is 0.0134, against 0.0168 for the best
TQV-12 file and 0.0245 for the stock `TQ_L`, all 14.72 GB. No TQT file is published: this page builds one from the
original Hugging Face weights with `tools/convert/qwen35/tqv12_quant.py --format tqt`.

The steps are those of the [TQV-12 guide](tqv12.md) with three differences: the skeleton is made with `--format tqt`,
the quantise step uses the encoder options below, and it takes longer (about 5 h 40 min on an RTX 3090 instead of
2 h 10 min). The TQV-12 guide explains each step in more detail; this page lists the exact commands and what differs. They
are written for Qwen3.8-27B on one 24 GB GPU and work for any Qwen3.5 / Qwen3.8 dense model that has a stock tqLLM TQ
file (the layout template).

What the quantiser does: every matrix the stock `TQ_L` file stores as TQ4 is quantised again as TQT, using the inputs
those matrices see on calibration text; every other tensor (embeddings, head, MTP layer, norms) is taken from `TQ_L`
unchanged. The result has `TQ_L`'s size and layout and loads wherever `TQ_L` does (with a build that has TQT).

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 24 GB (RTX 3090 / 4090 / 5090). The capture runs `TQ_L` in tqserve (18.9 GB at 4K context); the encoder runs after the server has exited |
| System RAM | 16 GB or more (the BF16 GGUF is memory-mapped; tested with 64 GB) |
| Disk | 55 GB for the BF16 GGUF, 15 GB for `TQ_L`, 15 GB for the output, 55 GB for the Hugging Face download (can be deleted after step 2), and ~8 GB per layer of capture dumps in flight (48 GB at 6 layers per run) |
| Software | this repository built with CUDA (README, "Build"); **`nvcc` on `PATH`** (the Viterbi encoder `tools/convert/qwen35/tqt_viterbi.cu` is compiled on first use and cached in `~/.cache/tqllm`); Python 3.10+ with `numpy gguf torch triton` (PyTorch with CUDA); a llama.cpp checkout for the conversion |
| Time | about 5 h 40 min on an RTX 3090, nearly all of it in step 5 (step times below) |

```sh
python3 -m venv ~/tqv-venv && . ~/tqv-venv/bin/activate
pip install numpy gguf torch huggingface_hub   # torch from pytorch.org if your CUDA needs another build
export PATH=/usr/local/cuda/bin:$PATH          # wherever your CUDA toolkit's nvcc is
nvcc --version
```

Run the GPU steps with nothing else on the GPU (stop any running tqserve first).

## 1. Download the model and the template

```sh
mkdir -p models
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B                       # 55 GB of safetensors
hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_L.gguf --local-dir models      # the layout template, 14.7 GB
```

## 2. Convert to a BF16 GGUF (CPU, ~4 min)

```sh
git clone https://github.com/ggml-org/llama.cpp
python3 -m venv ~/llama-venv   # its own venv: the converter's requirements pin a CPU-only PyTorch
~/llama-venv/bin/pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
~/llama-venv/bin/python llama.cpp/convert_hf_to_gguf.py models/Qwen3.8-27B --outtype bf16 \
  --outfile models/Qwen3.8-27B-BF16.gguf
```

Tested with llama.cpp b12198 (54.66 GB output).

## 3. Build the skeleton (CPU, ~15 s)

```sh
Q=tools/convert/qwen35/tqv12_quant.py
python3 $Q skeleton --format tqt --bf16 models/Qwen3.8-27B-BF16.gguf --template models/Qwen3.8-27B-TQ_L.gguf \
  --out models/Qwen3.8-27B-TQT.gguf
```

This writes the output file at its final size with `TQ_L`'s tensor list and metadata plus `tqllm.tqt.code = tqt-c2-v1`:
the 400 TQ4 matrices become TQT placeholders (type 306), the other 466 tensors are written as in the TQV-12 guide.
Measured: `400 TQT tensors to quantise, 466 written (96 BF16 matrices rotated like the template)` in 13 s. The format
is kept in `models/Qwen3.8-27B-TQT.gguf.plan.json`, so the later steps need no `--format`.

## 4. Calibration text (~110K tokens)

The same 64 prompts as for TQV-12 (half WikiText-2 *train*, half this repository's source code):

```sh
mkdir -p calib
curl -fsSL -o calib/wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o -j -q calib/wt.zip wikitext-2-raw/wiki.train.raw -d calib && rm calib/wt.zip
git ls-files 'src/*.cpp' 'src/*.cu' 'src/*.hpp' 'tools/*.py' | xargs cat > calib/code.txt
python3 $Q prompts --text calib/wiki.train.raw --text calib/code.txt --out calib/prompts.json
```

## 5. Capture and quantise (GPU, the long step)

```sh
python3 $Q run --tqserve build/tqserve --model models/Qwen3.8-27B-TQ_L.gguf --prompts calib/prompts.json \
  --dir calib/dump --bf16 models/Qwen3.8-27B-BF16.gguf --out models/Qwen3.8-27B-TQT.gguf --per-run 6 \
  --damp-form median --damp-role mix:0.1,gu:0.3,down:0.3,out:0.1 --order --refit --tqt-scales 3
```

The capture is the TQV-12 one (tqserve on `TQ_L` with `TQLLM_DEBUG=calib_dump`, then H = X^T X per input). The
quantiser then runs BlockLDLQ per matrix: GPTQ with one 128-column block per step, each row's block encoded by a
tail-biting Viterbi search over the 1024-state trellis, the block's exact error fed forward to the later columns. Every
written matrix is read back and decoded by an independent decoder against the encoder's result. The options:

| option | what it does |
|---|---|
| `--damp-form median` | GPTQ dampening in units of the median of diag(H) instead of the mean. The mean is dominated by a few massive-activation channels on some inputs, so a mean-relative dampening is too strong there and too weak elsewhere |
| `--damp-role mix:0.1,gu:0.3,down:0.3,out:0.1` | the dampening per input (`mix`: attention / GDN mixer, `gu`: FFN gate + up, `down`: FFN down, `out`: attention / GDN output). Larger values regularise against calibration noise: they raise the error on the calibration text and lower it on other text |
| `--order` | block act-order: the 128-blocks in descending tr(H_bb) (the stored order does not change) |
| `--refit` | closed-form refit of each block scale under GPTQ's block metric after the codes are chosen |
| `--tqt-scales 3` | scale candidates per row and block (1..5, default 3), one Viterbi pass each; the best by GPTQ's block metric is kept |

Measured on an RTX 3090 (Qwen3.8-27B, 6 layers per run, the options above): ~290 s per layer for the encoder
(`ffn_gate` and `ffn_up` 122 s, `ffn_down` 63 s, the mixer matrices 58 s, `ssm_out` 22 s), ~3 min per run for the
capture. All 64 layers: 5 h 37 min (5 h 07 min of it in the encoder, 288 s per layer on average). The first matrix
also compiles the encoder (a few seconds). Fewer candidates are faster and less accurate: about 87 / 173 / 259 s per layer of encoder time
at `--tqt-scales 1 / 2 / 3` in a single-layer measurement.

The log prints each matrix's relative output error on the calibration inputs (0.0001-0.0025). With the dampening above
this number is higher than with plain `--damp 0.01`; judge the result by the KLD below, not by this log.

`--per-run` is limited by disk (~8 GB of dumps per layer). The command is resumable; `--layers 0-31` limits a session to
some layers.

## 6. Verify (CPU, ~1 min)

```sh
python3 $Q verify --out models/Qwen3.8-27B-TQT.gguf --bf16 models/Qwen3.8-27B-BF16.gguf \
  --template models/Qwen3.8-27B-TQ_L.gguf
```

The checks of the TQV-12 guide, with TQ4 -> 306 and the `tqllm.tqt.code` key. On Qwen3.8-27B (46 s):

```
tensors: 866 (TQT 400, byte-identical to the template 370, BF16 from the source 96)
TQT decoded and un-rotated vs the BF16 source, relative RMS: mean 0.0742 min 0.0708 max 0.0808
GPTQ relative output error on the calibration inputs: mean 0.00109 max 0.00236
ALL CHECKS PASSED
```

## 7. Serve

```sh
./build/tqserve --model models/Qwen3.8-27B-TQT.gguf --port 8086 \
  -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt
```

The drafter and vocabulary are the ones from [AGENTS.md](../../AGENTS.md) step 2; all options of `TQ_L` apply.

## Expected quality and speed

Qwen3.8-27B (original weights), RTX 3090, all files 14.72 GB, made from the same BF16 GGUF and calibration text:

| | TQ_L | TQV-12 | TQV-12, encoder options | TQT |
|---|---:|---:|---:|---:|
| KLD vs BF16, mean (99 %) | 0.0245 (0.271) | 0.0184 (0.209) | 0.0168 (0.194) | 0.0134 (0.162) |
| same top token as BF16 | 94.17 % | 94.93 % | 95.00 % | 95.60 % |
| PPL (BF16 4.880) | 4.959 | 4.956 | 4.959 | 4.917 |
| GSM8K 250, reasoning effort low, greedy | 245 | 242 | 242 | 243 |

Speed, the same build, one session, back to back (`bench/relbench.py` defaults: greedy, reasoning on, 400 output tokens,
`--kv tq3`, 40,960 context, median of 3):

| | TQ_L | TQT |
|---|---:|---:|
| decode tok/s, 2K / 8K / 32K prompt, no drafter | 51.6 / 50.5 / 47.0 | 50.6 / 49.6 / 46.2 |
| decode tok/s, 2K / 8K / 32K prompt, DFlash2-TQ3 | 140.2 / 125.9 / 119.3 | 145.3 / 116.3 / 116.5 |
| prefill tok/s, 2K / 8K / 32K prompt | 1677 / 1861 / 1700 | 1658 / 1853 / 1696 |

Without a drafter TQT decodes 1.7-2.0 % slower than TQ_L (the c2 decode does more integer work per weight than TQ4's)
and prefills within 1.1 %. With the drafter the numbers move with the acceptance, which changes
because the files generate different text.

- TQV-12 is the [TQV-12 guide](tqv12.md)'s file; "encoder options" is the same format made with
  `--damp-form median --damp-role mix:0.1,gu:0.3,down:0.3,out:0.1 --order --refit --icm 4`.
- KLD: `llama-perplexity --kl-divergence-base` from the BF16 GGUF (llama.cpp b12198), WikiText-2 test, 8 chunks of
  2048 tokens; then `tqkld --model <file> --llama-base <base> --chunks 8 --llama-prefill`.
- GSM8K: the first 250 test questions, "Solve it step by step, then give the final answer on the last line as:
  #### <number>", 3000 tokens, served with the DFlash2 drafter. 242-245 of 250 is within noise at this size.

Not every matrix gains. On a few tensors TQT's error on held-out inputs is higher than TQV-12's with the encoder options
(layer 3 `attn_output` 1.17x, early `ffn_down` up to 1.10x on the calibration inputs), against 0.80-0.88x on most others;
the cause is not known. The full-model numbers above include those tensors; taking TQV-12 for them (a mixed file) did
not change the KLD measurably.

## Troubleshooting

- **`nvcc: not found` / `FileNotFoundError: 'nvcc'` at the first matrix.** The encoder is compiled on first use; put the
  CUDA toolkit's `bin` on `PATH` (or set `NVCC=/path/to/nvcc`). `TQT_ARCH` overrides the target (default `native`).
  The compiled library is cached in `~/.cache/tqllm/` (or `$XDG_CACHE_HOME/tqllm/`), keyed by the source's hash.
- **`tqt_viterbi: CUDA error`**: another process holds the GPU memory, or the cached library was built for another GPU:
  delete `~/.cache/tqllm/tqt_viterbi_*` and run again.
- **The model does not load: `GGUF TQT tensor ... without tqllm.tqt.code` or `TQT code ... is not supported`**: the file
  was not made by this tree's quantiser (or with another code tag); requantise.
- **The model does not load: unknown tensor type 306**: the tqserve build predates TQT; rebuild from this tree.
- **CUDA out of memory, disk full, wrong nvcc picked by cmake**: as in the [TQV-12 guide](tqv12.md#troubleshooting).
