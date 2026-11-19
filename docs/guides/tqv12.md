# Making a TQV-12 model

[Documentation](../README.md) / [Project home](../../README.md)

TQV-12 is a tqLLM weight format at TQ4's size (4.125 bits per weight) that rounds 4 weights at a time to a 4096-point
codebook ([format](../reference/formats.md#tqv-12-a-4-weight-vector-codebook-at-tq4s-size)). No TQV-12 file is
published: this page builds one from the original Hugging Face weights with `tools/convert/qwen35/tqv12_quant.py`.
The steps are written for Qwen3.8-27B on one 24 GB GPU and work for any Qwen3.5 / Qwen3.8 dense model that has a
stock tqLLM TQ file (the layout template).

What the quantiser does: every matrix the stock `TQ_L` file stores as TQ4 is quantised again as TQV-12 with vector GPTQ
(Frantar et al.), using the inputs those matrices see on calibration text; every other tensor (embeddings, head, MTP
layer, norms) is taken from `TQ_L` unchanged. The result has `TQ_L`'s size and layout and loads wherever `TQ_L` does.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, 24 GB (RTX 3090 / 4090 / 5090). The capture runs `TQ_L` in tqserve (18.9 GB at 4K context); H building needs 3.4 GB and GPTQ 9.7 GB, after the server has exited |
| System RAM | 16 GB or more: the quantiser peaked at 8.5 GB resident (the BF16 GGUF is memory-mapped; tested with 64 GB) |
| Disk | 55 GB for the BF16 GGUF, 15 GB for `TQ_L`, 15 GB for the output, 55 GB for the Hugging Face download (can be deleted after step 2), and ~8 GB per layer of capture dumps in flight (64 GB at 8 layers per run) |
| Software | this repository built with CUDA (README, "Build"); Python 3.10+ with `numpy gguf torch triton` (PyTorch with CUDA; Triton ships with the Linux PyTorch wheels); a llama.cpp checkout for the conversion |
| Time | about 2 h 10 min on an RTX 3090, 2 h of it in step 5 (step times below) |

```sh
python3 -m venv ~/tqv-venv && . ~/tqv-venv/bin/activate
pip install numpy gguf torch huggingface_hub   # torch from pytorch.org if your CUDA needs another build
```

Run the GPU steps with nothing else on the GPU (stop any running tqserve first).

## 1. Download the model and the template

```sh
mkdir -p models
hf download Qwen/Qwen3.8-27B --local-dir models/Qwen3.8-27B                       # 55 GB of safetensors
hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_L.gguf --local-dir models      # the layout template, 14.7 GB
```

## 2. Convert to a BF16 GGUF (CPU, ~4 min)

The quantiser reads the weights from a BF16 GGUF with llama.cpp's tensor names (the names `TQ_L` uses):

```sh
git clone https://github.com/ggml-org/llama.cpp
python3 -m venv ~/llama-venv   # its own venv: the converter's requirements pin a CPU-only PyTorch
~/llama-venv/bin/pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
~/llama-venv/bin/python llama.cpp/convert_hf_to_gguf.py models/Qwen3.8-27B --outtype bf16 \
  --outfile models/Qwen3.8-27B-BF16.gguf
```

Tested with llama.cpp b12198 (54.66 GB output). The Hugging Face folder is not needed after this step.

## 3. Build the skeleton (CPU, ~15 s)

```sh
Q=tools/convert/qwen35/tqv12_quant.py
python3 $Q skeleton --bf16 models/Qwen3.8-27B-BF16.gguf --template models/Qwen3.8-27B-TQ_L.gguf \
  --out models/Qwen3.8-27B-TQV12.gguf
```

This writes the output file at its final size with `TQ_L`'s tensor list and metadata plus
`tqllm.tqv.codebook = tqv12-v1`: the 400 TQ4 matrices become TQV-12 placeholders, the other 466 tensors are written
(96 BF16 `ssm_alpha` / `ssm_beta` matrices from the BF16 source, rotated as `TQ_L` stores them; the rest copied from
`TQ_L`). Every written tensor is checked against the BF16 source; a template from another model is refused. Progress
is kept in `models/Qwen3.8-27B-TQV12.gguf.plan.json`.

## 4. Calibration text (~110K tokens)

64 prompts of ~6,500 characters: half WikiText-2 *train* (not the test split that the KLD below is measured on), half
this repository's source code.

```sh
mkdir -p calib
curl -fsSL -o calib/wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o -j -q calib/wt.zip wikitext-2-raw/wiki.train.raw -d calib && rm calib/wt.zip
git ls-files 'src/*.cpp' 'src/*.cu' 'src/*.hpp' 'tools/*.py' | xargs cat > calib/code.txt
python3 $Q prompts --text calib/wiki.train.raw --text calib/code.txt --out calib/prompts.json
```

Other text works too (`--text` takes any files; the prompts are split evenly between them). Text close to how the
model will be used is better calibration.

## 5. Capture and quantise (GPU, the long step)

```sh
python3 $Q run --tqserve build/tqserve --model models/Qwen3.8-27B-TQ_L.gguf --prompts calib/prompts.json \
  --dir calib/dump --bf16 models/Qwen3.8-27B-BF16.gguf --out models/Qwen3.8-27B-TQV12.gguf --per-run 8
```

For each group of `--per-run` layers it:

1. starts tqserve on `TQ_L` with `TQLLM_DEBUG=calib_dump:<layers>:calib/dump` and sends the 64 prompts. The engine
   appends the FP16 input of every quantised matrix of those layers to `calib/dump/L<layer>_<input>.x16` during prefill
   (decode steps, CUDA graph capture and the load-time prefill calibration are skipped).
2. stops tqserve, builds H = X^T X per input on the GPU and deletes the raw dumps;
3. runs vector GPTQ on each matrix (4 weights per step, damping 0.01, in the rotated basis), writes it into the output,
   reads it back and checks that the written blocks decode to the GPTQ result; then deletes that layer's H.

`TQ_L`'s matrices run in the rotated basis, so the dumps are already rotated (the `.json` beside each dump says so);
with a K-quant or BF16 capture model the plain-basis H is rotated before GPTQ (covered by the unit test; the numbers on
this page come from a `TQ_L` capture).

Measured on an RTX 3090 (Qwen3.8-27B, 8 layers per run, 116,689 dumped tokens per input): per run, tqserve starts
in ~30 s, the 64 prompts take ~2 min, H for 8 layers ~1 min; GPTQ then takes ~92 s per layer (`ffn_gate` + `ffn_up` 60 s,
`ffn_down` 17 s, the mixer and output matrices the rest). All 64 layers: 2 h 02 min. Peak GPU memory 18.9 GB (capture),
peak resident RAM 8.5 GB. The log prints each matrix's relative output error on the calibration inputs (0.0001-0.003).

`--per-run` is limited by disk, not VRAM: the dumps take ~8 GB per layer at 110K tokens. With 32 GB free use
`--per-run 4`. The command is resumable: run it again after an interruption and it continues with the first layer
not yet written (`--layers 0-31` limits a session to some layers).

## 6. Verify (CPU, ~1 min)

```sh
python3 $Q verify --out models/Qwen3.8-27B-TQV12.gguf --bf16 models/Qwen3.8-27B-BF16.gguf \
  --template models/Qwen3.8-27B-TQ_L.gguf
```

It checks tensor names, order, shapes and types against `TQ_L` (TQ4 -> 305, the rest unchanged), the metadata, that
copied tensors are byte-identical to `TQ_L`, that every TQV-12 tensor was written and decode-checked, that no block has
a zero scale, and decodes sample rows of each TQV-12 tensor back to the original basis against the BF16 source. It ends
with `ALL CHECKS PASSED`. On Qwen3.8-27B (48 s, 2.4 GB RAM):

```
tensors: 866 (TQV-12 400, byte-identical to the template 370, BF16 from the source 96)
TQV-12 decoded and un-rotated vs the BF16 source, relative RMS: mean 0.0945 min 0.0882 max 0.1187
GPTQ relative output error on the calibration inputs: mean 0.00134 max 0.00293
ALL CHECKS PASSED
```

The `.plan.json` file and the BF16 GGUF are not needed after this step.

## 7. Serve

```sh
./build/tqserve --model models/Qwen3.8-27B-TQV12.gguf --port 8086 \
  -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt
```

The drafter and vocabulary are the ones from [AGENTS.md](../../AGENTS.md) step 2; all options of `TQ_L` apply.

## Expected quality and speed

Qwen3.8-27B, the file this page makes against the stock `TQ_L` (same size: 14,721,469,248 vs 14,721,469,184 bytes),
RTX 3090, one session, same build:

| | TQ_L | TQV-12 |
|---|---:|---:|
| KLD vs BF16, mean (99 %) | 0.0245 (0.271) | 0.0184 (0.209) |
| same top token as BF16 | 94.17 % | 94.93 % |
| PPL (BF16 4.880) | 4.959 | 4.956 |
| GSM8K 250, reasoning effort low, greedy | 245 | 242 |
| decode tok/s, 2K / 8K / 32K prompt, no drafter | 51.7 / 50.8 / 47.1 | 51.5 / 50.5 / 47.2 |
| decode tok/s, 2K / 8K / 32K prompt, DFlash2-TQ3 | 140.5 / 126.5 / 119.5 | 157.3 / 130.0 / 115.2 |
| prefill tok/s, 2K / 8K / 32K prompt | 1675 / 1865 / 1696 | 1635 / 1849 / 1703 |

- KLD: `llama-perplexity --kl-divergence-base` from the BF16 GGUF (llama.cpp b12198, 20 layers on the GPU), WikiText-2
  test, 8 chunks of 2048 tokens; then `tqkld --model <file> --llama-base <base> --chunks 8 --llama-prefill`.
- GSM8K: the first 250 test questions, "Solve it step by step, then give the final answer on the last line as:
  #### <number>", 3000 tokens, served with the DFlash2 drafter. The two files disagree on 7 questions (TQV-12 alone right
  on 2, TQ_L alone on 5; exact binomial p = 0.45): no measurable difference at this size.
- Speed: `bench/relbench.py` defaults (greedy, 400 output tokens, `--kv tq3`, 40,960 context, median of 3). Without a
  drafter TQV-12 decodes within 0.6 % of TQ4; with the drafter the numbers move with the acceptance, which changes
  because the two files generate different text (TQV-12 is not token-identical to TQ_L).

## Encoder options: a better TQV-12 file, same format

Step 5 above uses plain GPTQ (damping 0.01), the encoder the numbers in the table come from. Five options of
`quantize` / `run` lower the error on text other than the calibration set without changing the file format (same type,
codebook and decoder; without them the encoder is unchanged bit for bit):

```sh
python3 $Q run ... --damp-form median --damp-role mix:0.1,gu:0.3,down:0.3,out:0.1 --order --refit --icm 4
```

| option | what it does |
|---|---|
| `--damp-form median` | dampening in units of the median of diag(H) instead of the mean (the mean is dominated by a few massive-activation channels on some inputs) |
| `--damp-role R:D,...` | dampening per input role (`mix`, `gu`, `down`, `out`); `--damp` for the roles not listed |
| `--order` | block act-order: the 128-blocks in descending tr(H_bb); 4-groups and block scales stay intact |
| `--refit` | closed-form refit of each block scale under GPTQ's block metric, exact residual fed forward |
| `--icm N` | N sweeps of local search after GPTQ (codeword / sign moves per 4-group, then a scale refit) on a dampened H |

Held-out H (39 tensors of 6 layers, inputs from prompts not in the calibration set): 0.911x the plain encoder's output
error with all five, every tensor better (worst 0.945x). Full model, Qwen3.8-27B, same calibration and BF16 base as the
table above: KLD 0.0168 (99 % 0.194, same top 95.00 %, PPL 4.959) against 0.0184, GSM8K 250 242 (same), decode
speed unchanged. Step 5 then takes about 3 h 40 min on an RTX 3090 (6 layers per run; ~190 s per layer instead of 92 s,
most of it ICM). The log's per-matrix output error
is measured on the calibration inputs and comes out higher with these options (they regularise); judge by the KLD.
The [TQT guide](tqt.md) uses the same options for the TQT format.

## Troubleshooting

- **CUDA out of memory in step 5.** Something else holds GPU memory (another tqserve, a desktop session using much
  VRAM). The capture server needs 18.9 GB and GPTQ 9.7 GB, one after the other. `nvidia-smi` must show the GPU
  nearly empty before the step.
- **"no capture for layer N"**: the capture model did not run those layers' prefill through the hooked products: use a
  qwen35 dense file (the `TQ_L` template is the tested one) and check the tqserve log the error names.
- **Why the inputs go to disk.** Accumulating H on the GPU beside the capture model ran out of memory on 24 GB (one
  `ffn_down` H is 17408^2 x 4 B = 1.2 GB, ~1.6 GB per layer), so the engine writes the raw FP16 inputs and H is built
  after tqserve has exited. A host copy is not allowed inside CUDA graph capture (it fails the capture), so the hook
  skips captured forwards; prefill chunks run eagerly and are all written. The load-time prefill calibration (synthetic
  inputs) is skipped as well. If the token counts per input differ, step 5 prints a warning.
- **Disk full during capture** (`calib_dump: short write`): lower `--per-run` or point `--dir` at a larger disk.
- **`cmake` picks the wrong nvcc** (`Compiling the CUDA compiler identification source file failed` when an old
  `/usr/bin/nvcc` is first on `PATH`): configure with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- **The model does not load: `TQV codebook ... is not supported`**: the file was made with another codebook tag than
  this build has; requantise with this tree's quantiser.
