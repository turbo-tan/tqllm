# Two GPUs: one model across both cards

`--split layer` runs one dense Qwen3.5 / 3.6 / 3.8 model (GGUF) across two NVIDIA GPUs: each GPU holds half the layers,
its share of the KV cache and recurrent state, so each card needs less memory than one card holding the whole model.
The output is the same as on one GPU (greedy text identical, token for token). It does not make one request faster:
the two GPUs run one after the other.

One `tqserve` process runs per GPU. Rank 0 serves the API; rank 1 follows it and has no HTTP port.

## One PC with two GPUs

```
CUDA_VISIBLE_DEVICES=1 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 65536 --np 4 \
    --split layer --tp-rank 1 --tp-peer 127.0.0.1 &
CUDA_VISIBLE_DEVICES=0 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 65536 --np 4 \
    --split layer --tp-rank 0
```

Both processes take the same model file and the same options (`--context`, `--np`, `-md`, `-ub`). Rank 1 waits up to
10 minutes for rank 0. Stopping rank 0 stops rank 1; if rank 1 stops, rank 0 answers the running request with an error.

## Two machines

Each machine runs one rank against its own copy of the model file. `--tp-peer` is rank 0's address on the link between
them (e.g. the 200 Gb ConnectX ports of two DGX Spark / GB10 boxes), and both machines must use the same NCCL version:

```
# machine A (rank 0)
NCCL_SOCKET_IFNAME=enp1s0f0np0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 0
# machine B (rank 1)
NCCL_SOCKET_IFNAME=enp1s0f0np0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 1 --tp-peer 10.10.0.55
```

## Flash-Next: experts over two boxes (one user faster)

With every expert resident on both boxes (`slots=512` in the manifest), `--split experts` lets the second box compute
half of each layer's experts (e % 2 == 1) for the first box's decode steps whenever the second box has no request of its
own; when it is busy, the first box computes all of them. The output is the same as one box either way. Run both behind
one address with `--peer`, so a lone request lands on the first box and is helped, and several users spread over both.
On two GB10s (NVFP4 experts, 32 unseen questions, one user at a time): through the router below, 64.4 -> 68.4 tok/s
(median, faster on all 32; several users unchanged); a dedicated pair of `--np 1` servers, 66.8 -> 70.5.

```
# both boxes: the two halves of the ConnectX port (each about 100 Gb/s), four channels per peer
export NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_NCHANNELS_PER_NET_PEER=4
# machine B (rank 1): serves its own users, helps A while idle
tqserve --model flashnext.tqfn --np 8 --context 262144 --host 10.10.0.44 --port 8080 --split experts --tp-rank 1 --tp-peer 10.10.0.55
# machine A (rank 0, the front door)
tqserve --model flashnext.tqfn --np 8 --context 262144 --port 8080 --peer http://10.10.0.44:8080 --split experts --tp-rank 0
```

If rank 1 stops, the request in flight fails and rank 0 continues on its own (every expert is resident there, same output).

## Notes

- The memory fit does not run: give `--context` and `--np` that fit each GPU.
- `--split-at N` moves the boundary (layers N and up on rank 1) when the cards differ in size.
- Not yet with `--split layer`: `--kv-host`, `--kv-compact`, image input, MoE models.
- Built from source, multi-GPU needs NCCL: a system `libnccl-dev` is found by itself, or pass `-DTQLLM_NCCL_DIR=...`.
  The Docker image includes it.
