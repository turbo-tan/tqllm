# Weight formats

[Documentation](../README.md) / [Project home](../../README.md)

## TQ formats: TQ3_4S, TQ3 and TQ4

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
