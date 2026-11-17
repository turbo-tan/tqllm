# Long context and TqCache

[Documentation](../README.md) / [Project home](../../README.md)

## Long context on a 16 GB card (TqCache)

Ask for more context than the GPU holds and the memory fit keeps the oldest KV positions in system RAM (exact: the
output is the same as with all KV on the GPU) instead of refusing the request. Reading that part over PCIe costs decode
speed, so use it for prompts past what the GPU holds; below that, the default context is faster.

RTX 5060 Ti 16 GB (Linux), Qwen3.8-27B-TQ_XS + DFlash2-TQ3, tq3 KV, 200,389-token prompt, decode tok/s (11-17):

| setup | where the KV lives | decode |
|---|---|---|
| TQ_XS, `--context 204800` | 82,944 of 204,800 positions in RAM (TqCache) | 43.7 |
| TQ_XS-QSA, `--context 204800 --qsa-from 49152` | part in RAM; each token reads only the picked blocks | 109.5 |

1.0.1 measures the same 43.7 on this card (58,368 positions in RAM); the 113 tok/s row this guide used to show came from an
earlier Windows / WSL 2 setup (10-06) and does not reproduce on this card today. For 100K+
prompts on 16 GB use the QSA file ([Choose a setup](choose-your-setup.md)).

The fit reserves RAM for the whole requested context up front, so a larger `--context` moves more positions to RAM even for
prompts that would fit on the GPU. Ask for more context than the default only when the prompts
need it.
