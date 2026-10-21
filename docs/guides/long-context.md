# Long context and TqCache

[Documentation](../README.md) / [Project home](../../README.md)

## Long context on a 16 GB card (TqCache)

Ask for more context than the GPU holds and the memory fit keeps the oldest KV positions in system RAM (exact: the
output is the same as with all KV on the GPU) instead of refusing the request. Reading that part over PCIe costs decode
speed, so use it for prompts past what the GPU holds; below that, the default context is faster.

RTX 5060 Ti 16 GB, Qwen3.8-27B-TQ_XS.gguf + DFlash2-TQ3, tq3 KV, decode tok/s. Default context: the memory fit caps the
context at 204,800 on this card and keeps all KV on the GPU. `--context 262144`: TqCache keeps the oldest positions in RAM.

| prompt | default context (204,800, all KV on the GPU) | `--context 262144` (TqCache) |
|---|---|---|
| 200K | 113 | 70 (81,920 of its positions read from RAM) |
| 224K | refused (longer than the context) | 42 |
| 248K | refused | 32 |

The fit reserves RAM for the whole requested context up front, so with `--context 262144` even a prompt that would fit
on the GPU reads part of its KV over PCIe (the 200K row). Ask for more context than the default only when the prompts
need it.
