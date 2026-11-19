# Weight formats

[Documentation](../README.md) / [Project home](../../README.md)

## TQ formats: TQ3_4S, TQ3 and TQ4

Illustrated version with bit layouts and quality charts: [formats.html](formats.html).

All three rotate the weights before quantising (the TurboQuant idea: a rotation spreads outliers so a small codebook fits
every block), then store codes plus scales. They differ in block size, scales and where they run:

| Format | Block | Scales | Bits per weight | Runs in |
|---|---|---|---:|---|
| TQ3_4S (GGUF type 46) | 32 weights, 32-point rotation, 8-level code | four 8-bit scales (one per 8 weights) | 4.0 | tqLLM and [llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) |
| TQ3 (type 301) | 128 weights, 128-point rotation, 8-level code | one FP16 scale per 128 | 3.125 | tqLLM only |
| TQ4 (type 300) | 128 weights, 128-point rotation, 16-level code | one FP16 scale per 128 | 4.125 | tqLLM only |

TQ3_4S spends a quarter of its bits on scales; TQ3 and TQ4 rotate over a 4x longer block, so one scale per 128 weights
is enough and the bits go into the codes instead. That is how the 27B fits in 11.85 GB as TQ3 (TQ_XS) against 14.5 GB as
TQ3_4S-v2, and why TQ4 at about the same size as TQ3_4S is much closer to the original (KLD vs BF16 0.022 for TQ4_L).
The release files `Qwen3.8-27B-TQ_XS.gguf` (11.85 GB), `TQ_S` (12.55 GB) and `TQ4_L` (14.97 GB) are built from these
formats. Type ids 300
and 301 are tqLLM's own (above the ggml range), so these files need tqLLM; llama.cpp cannot load them.

**Reading a TQ3 / TQ4 file: the input basis.** A TQ3 / TQ4 matrix is stored in the rotated basis, so its input must be
rotated with the same 128-point rotation before the product. Dense matrices that read the same input as TQ matrices are
stored in that rotated basis too: in the Qwen3.8 TQ_L file the 96 BF16 Gated DeltaNet `ssm_alpha` / `ssm_beta`
matrices equal the original weights times the rotation (relative error 0.006 against a Q8_0 copy rotated the same way,
1.42 unrotated). A reader that multiplies such a dense matrix by the plain input gets wrong recurrent gates: the Apple
backend did (2026-10-22): swapping only these two tensor families into a K-quant file reproduced its whole 0.53 KL gap
against the CUDA engine. Rule for any reader (CPU paths, other backends, ports): in a TQ3 / TQ4 model, rotate the input of every
matrix that shares an input with a TQ matrix, whatever its storage type.

## TQV-12: a 4-weight vector codebook at TQ4's size

TQV-12 keeps TQ4's container (128 weights in 66 bytes, 4.125 bits per weight, the same 128-point rotation and the same
FP16 scale per block) but rounds 4 weights at a time to the nearest point of a fixed 4096-entry codebook instead of each
weight to one of 16 levels. A 4-D codebook packs its points more densely than a scalar grid, so at the same size the
rounding error is lower: Qwen3.8-27B quantised with GPTQ into TQV-12 has a KLD against BF16 of 0.0184, against 0.0245
for the stock TQ_L at the same 14.72 GB (WikiText-2, 8 x 2048 tokens; decode speed within 0.6 %). tqLLM only; make a
file with the [TQV-12 guide](../guides/tqv12.md), which also lists the measurements.

| | |
|---|---|
| GGUF tensor type | 305, valid only together with the metadata key `tqllm.tqv.codebook = "tqv12-v1"` |
| Block | 128 weights along the row (rows must be a multiple of 128): FP16 scale (2 B) + 4 units of 16 B |
| Unit j (weights 32j .. 32j+31) | 4 little-endian uint32 x, y, z, w. x, y, z: a 96-bit stream (x = bits 0-31) of 8 12-bit codebook indices, group u (weights 4u .. 4u+3 of the unit) at bits 12u .. 12u+11. w: the 32 sign bits, weight 4u at bit 15-u, 4u+1 at 31-u, 4u+2 at 7-u, 4u+3 at 23-u (set = negative) |
| Weight 4u + e | scale x table[index_u][e] x (sign ? -1 : +1), in the rotated basis |
| Codebook | 4096 x 4 int8, all 0..127 (the positive orthant; signs are stored per weight) |
| Basis | TQ4's: per 128 weights H D / sqrt(128), H Sylvester-Hadamard, D the signs `tq4k::S1` (src/formats/weight/tq4.cu) |

**Codebook.** k-means (30 Lloyd iterations) of 409,600 samples of |N(0, I4)| into 4096 centroids, divided by
step = max / 127 = 4.1011205 / 127 and rounded to int8; the step is folded into the block scale. The int8 table is the
definition, not the procedure (k-means does not reproduce bit for bit across libraries): `tests/data/tqv12_codebook_i8.bin`
(16,384 bytes, entry i = bytes 4i .. 4i+3, sha256 `63437d856ae6263846036efbeedb326bc20de93ee792d480ef9f5c67b0e1f3fc`),
compiled into the engine as `src/formats/weight/tqv12_codebook.inc`. Rounding the FP32 centroids to int8 costs 0.3 % in
quantisation MSE; an int8 table lets the kernel keep it in shared memory and feed the products to integer-exact FP16 MMA.

**Type 305 and the key.** A type-305 tensor loads only when the file has `tqllm.tqv.codebook = "tqv12-v1"`; the reader
refuses any other value (a future codebook gets a new tag), and a type-305 tensor without the key does not load.

**Reading a TQV-12 file.** Exactly as TQ4: the matrix input is rotated with the 128-point rotation, and the dense
matrices beside TQV-12 matrices (`ssm_alpha` / `ssm_beta` in BF16) are stored pre-rotated. The quantiser takes every
tensor that is not TQ4 in the template file (embeddings, head, MTP layer, norms) unchanged.

**Kernels.** Decode matvec for 1 to 16 columns (as TQ4; wider products run through the column-batched matvec or the
prefill GEMM), a native prefill GEMM, and FP16 and int8 prefill dequantisation. The matvec decodes through an
int8 table in shared memory and m16n8k16 MMA, one 128-weight group per step. Every kernel variant and column count gives
the same bits, so verify widths do not change the output.

## TQT: a trellis code at TQ4's size

TQT keeps TQ4's container (128 weights in 66 bytes, 4.125 bits per weight, the same rotation and FP16 scale per block)
and stores the 128 weights as a tail-biting bitshift-trellis code (QTIP, Tseng et al. 2024): 64 bytes form a circular
bit stream, and every pair of weights is a fixed function of an 18-bit window of it. The windows of neighbouring pairs
overlap in 10 bits, so the encoder chooses all 128 weights of a block together (a Viterbi search) instead of one point
per weight or per 4 weights. On Gaussian data the MSE at 4 bits is 0.0047 (TQV-12 0.0069, the rate-distortion bound
0.0039). Qwen3.8-27B in TQT has a KLD against BF16 of 0.0134, against 0.0168 for the best TQV-12 encoder and 0.0245 for
the stock TQ_L, all 14.72 GB (WikiText-2, 8 x 2048 tokens). tqLLM only; make a file with the
[TQT guide](../guides/tqt.md), which also lists the measurements.

| | |
|---|---|
| GGUF tensor type | 306, valid only together with the metadata key `tqllm.tqt.code = "tqt-c2-v1"` |
| Block | 128 weights along the row (rows must be a multiple of 128): FP16 scale (2 B) + stream bytes B_0 .. B_63 |
| Window of pair p (weights 2p, 2p+1) | w = (B_p \| B_{p+1} << 8 \| B_{p+2} << 16) & 0x3FFFF, byte indices mod 64 (the stream wraps inside the block) |
| Code c2 (32-bit unsigned arithmetic) | x = (w * 0x9E3779B1 + 0x7F4A7C15) << 14; y = x rotated left by 16; z = y * 0x2C1B3C6D + 0x9E3779B9; z2 = z * 0x297A2D39 + 0x7F4A7C15; h(v) = (v & 0x8FFF8FFF) ^ 0x3B603B60 read as two FP16 (low half first) |
| Weights 2p, 2p+1 | the FP16 sums h(z) + h(z2) (round to nearest even; low halves = weight 2p), times the scale, in the rotated basis |
| Basis | TQ4's: per 128 weights H D / sqrt(128), H Sylvester-Hadamard, D the signs `tq4k::S1` (src/formats/weight/tq4.cu) |

**Code.** c2 is QTIP's "3INST" computed code applied twice to one hash: masking a random word with 0x8FFF and xoring
0x3B60 gives an FP16 with a random sign, 2 random exponent bits and 10 random mantissa bits; the sum of two such values
is close to Gaussian (kurtosis 2.88, the two weights of a pair uncorrelated). There is no table: the matvec decodes a
pair with 7 integer instructions and one FP16x2 add. On an RTX 3090 a table lookup in shared memory costs more than
that (random addresses make bank conflicts: 5-25 % slower than TQ4 in a standalone bench), and QTIP's one-weight
codes are ALU-bound (25-30 % slower). The tag names the code; another code or window gets a new tag, and the reader
refuses a 306 tensor without the key or with an unknown tag.

**Encoder.** For a block's targets t = w / scale the Viterbi search runs over 1024 states (the 10 bits two pairs share)
with 256 branches (the next byte) per pair; tail-biting (the windows of pairs 62 and 63 wrap to B_0, B_1) comes from a
short first pass around the wrap point that fixes the shared start / end state. With GPTQ the 128 columns of a block are
quantised together and the block's exact error goes to the later columns (BlockLDLQ); the scale is chosen per row and
block from up to five candidates (one Viterbi pass each; default 3) by the block's error under GPTQ's metric, then
optionally refit in closed form. A Qwen3.8-27B layer takes about 290 s on an RTX 3090 with 3 candidates.
`tools/convert/qwen35/tqt.py`, `tqt_viterbi.cu`.

**Kernels.** As TQV-12: decode matvec for 1 to 16 columns (the TQV-12 skeleton with the c2 decode; the last two pairs of
a 32-weight segment read the first word of the next segment, an L1 hit), a native prefill GEMM, and FP16 and int8
prefill dequantisation. Every kernel variant and column count gives the same bits. Decode cost per matrix
(`test-tqt-format --bench`, RTX 3090, us at 1 / 8 / 16 columns, TQ4 in brackets): 34816 x 5120 113.4 / 133.4 / 180.5
(109.3 / 130.0 / 205.7), 5120 x 17408 62.0 / 72.0 / 92.2 (60.1 / 63.7 / 96.2), 16384 x 5120 59.0 / 65.4 / 92.4
(56.6 / 59.9 / 103.9), 5120 x 6144 24.2 / 26.9 / 36.0 (24.1 / 26.5 / 39.1): within 4 % of TQ4 at one column, up to
13 % slower at 8 (`ffn_down`), faster at 16.
