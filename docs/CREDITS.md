# Credits

[Project home](../README.md)

tqLLM re-implements ideas from the work below (no code unless listed in NOTICE). Reference = their idea, method or
weights; improvement = their idea, extended or re-implemented by us.

| Piece | Credit |
|---|---|
| Qwen3.8-27B weights, MTP head, chat template | Qwen team, Alibaba ([Qwen](https://huggingface.co/Qwen)) |
| GGUF format, K-quant / IQ block layouts, reference engine; base of our llama.cpp-tq3 fork | ggml authors, [llama.cpp](https://github.com/ggml-org/llama.cpp), Georgi Gerganov |
| DFlash / DFlash2 drafting method and drafter weights (served here in TQ3 / TQ4 with a 65,536-token draft vocabulary) | Jian Chen, Yesheng Liang, Zhijian Liu, [z-lab](https://github.com/z-lab/dflash) ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) |
| Rotation-based quantisation, the lineage of the TQ3 / TQ4 weight formats and the tq3 KV cache | Google Research, [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) |
| turbo2 / turbo3 / turbo4 KV types, REFRACT quality metric | Tom Turney (TheTom), [turboquant_plus](https://github.com/TheTom/turboquant_plus) |
| Trellis-coded KV idea (tcq codecs; codebooks and kernels ours) | spiritbuun (buun), [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) |
| Learned rotations for quantisation | Avarok, TorQuant |
| Fixed chat template (v22, embedded in our files) | froggeric, [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) |
| FlashAttention tiling | Tri Dao et al., [flash-attention](https://github.com/Dao-AILab/flash-attention) |
| Speculative sampling | Leviathan, Kalman, Matias ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192)); Chen et al. ([arXiv 2302.01318](https://arxiv.org/abs/2302.01318)) |
| Token-tree verification (our hybrid Gated DeltaNet tree kernels build on it) | SpecInfer, Miao et al. ([arXiv 2305.09781](https://arxiv.org/abs/2305.09781)); [Medusa](https://github.com/FasterDecoding/Medusa), Cai et al.; [EAGLE](https://github.com/SafeAILab/EAGLE), Li et al. |
| Prompt-lookup drafting (`--lookup`) | Apoorv Saxena, [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) |
| Comparison baselines | [Unsloth](https://huggingface.co/unsloth) UD quants; [HyperQwen](https://github.com/syv-ai/HyperQwen); [ExLlamaV3](https://github.com/turboderp-org/exllamav3), turboderp; [vLLM](https://github.com/vllm-project/vllm); Strata, Niko1221; LocalMaxxing |

### Testers

Thank you to the testers from X who ran tqLLM on their own hardware and reported back: 0x443, Ricky Spanish,
lackonendes and GandalfOne.
