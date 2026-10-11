# Private release candidate container

Build from this source tree with `docker build --build-arg CUDA_ARCHITECTURES=86 -t tqllm:rc3-candidate .`.
This image includes binaries and the conversion script, with no weights or research artifacts.
Mount models read-only at `/models`; set `MODEL_FILE` and `TOKENIZER_FILE` to container paths.
Set `DRAFT_MODEL_FILE`, `DRAFT_VOCAB_FILE`, and `KV_FORMAT` for optional accelerated decoding.
These map to `-md`, `--draft-vocab`, and `--kv`; retired `TQLLM_*` options are rejected by the runtime.
Pass other native options after the image name, including `-ctk`, `-ctv`, `--no-tree`, `--lookup`, and `-fit`.
`CONTEXT`, `PARALLEL`, `MAX_TOKENS`, and `REASONING` retain their container defaults.
This candidate has not passed the full release gates and must not be published.
