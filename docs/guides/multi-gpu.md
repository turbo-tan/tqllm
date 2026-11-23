# Several GPUs, in one PC or several boxes

`--split layer` runs one model across two NVIDIA GPUs in the same PC: each GPU holds half the layers, its share of the
KV cache and state, so a model that does not fit one card runs on two, and a model that does fits a longer context.
The output is the same as on one GPU (greedy text identical, token for token). It does not make one request faster: the
two cards work one after the other and pass one small message per step.

Works for the dense Qwen3.5 / 3.6 / 3.8 27B GGUFs and for Flash-Next.

<!-- DRAFT for 1.7.0. Numbers: 2x RTX 4090 pod (2026-11-14), two RTX 3090 PCs over 1 GbE and two GB10s (2026-11-15),
     one run unless marked; re-measure on the tagged build before publishing. -->

## Start it

One `tqserve` process runs per GPU. Rank 0 serves the API; rank 1 follows it and has no HTTP port. Both take the same
model file and the same options (`--context`, `--np`, `-md`, `--draft-vocab`, `-ub`):

```
CUDA_VISIBLE_DEVICES=1 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 131072 --np 4 \
    --split layer --tp-rank 1 --tp-peer 127.0.0.1 &
CUDA_VISIBLE_DEVICES=0 tqserve --model Qwen3.8-27B-TQ_L.gguf --context 131072 --np 4 \
    --split layer --tp-rank 0 --port 8080
```

Start either first; rank 1 waits up to 10 minutes for rank 0 (port 29600, `--tp-port` to change it). The API answers
once both have loaded. Stopping rank 0 stops rank 1; if rank 1 stops, rank 0 answers the running request with an error.

## More than two GPUs, or GPUs in several boxes

Add `--tp-world N` and give each process its rank; every rank but 0 points `--tp-peer` at rank 0's address. The layers are
shared out by each card's free memory (a 16 GB card gets fewer than a 24 GB one); `--split-at A,B,...` sets the first layer
of ranks 1, 2, ... by hand. For Flash-Next a rank also takes no more layers than its box can hold in RAM while keeping 20 GB
available (the expert pack it pins, its share of the host K/V and, on the last rank, a full prompt's buffers); a box that
cannot keep that with even one layer is refused at start with what it would need, and when the last rank has the least room
the log asks you to start the box with the most RAM as the last rank.

```
# box C (rank 2), box B (rank 1), then box A (rank 0, serves the API); NCCL over the LAN (name the interface)
NCCL_SOCKET_IFNAME=eth0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-world 3 --tp-rank 2 --tp-peer 10.0.0.1 &
NCCL_SOCKET_IFNAME=eth0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-world 3 --tp-rank 1 --tp-peer 10.0.0.1 &
NCCL_SOCKET_IFNAME=eth0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-world 3 --tp-rank 0 --port 8080
```

Every rank must run the same build with the same NCCL version (checked when they connect). Cards of one generation give
exactly one card's text; cards of different generations (an RTX 3090 with an RTX 5060 Ti) give the same text run to run
but not that of either card alone, because each generation's kernels round differently. Measured 11-19 over 1 GbE (RTX 3090
+ RTX 3090 [+ RTX 5060 Ti]), tqLLM results 20261119_nsplit: Qwen3.8-27B TQ_L plain 50.5 tok/s on one card, 52.6 on two,
43.0 on three; Flash-Next with one user 115 / 104 / 110 / 94 / 92 tok/s on one 3090 vs 148 / 133 / 132 / 105 / 100 on two
(code / chat prompts), 128K first token 48.0 s vs 41.5 s.

## What you get (2x RTX 4090, 24 GB each)

| Model | One card | Two cards, `--split layer` |
|---|---|---|
| Qwen3.8-27B Q8_0 (29 GB) | does not load (not even at a 4K context) | loads: 1 user up to a 131K context (23.2 GB per card, 30.7 tok/s; with the DFlash2 drafter up to 194 tok/s), 4 users up to 64K (16K each, 22.3 GB per card) |
| Qwen3.8-27B TQ_L (15 GB) | 262K context, with the K/V cache, LM head and token table moved to host memory (60 tok/s) | 262K context with everything on the GPUs (20.4 GB per card, 60 tok/s) |
| Qwen3.8-27B TQ_L, speed | 61.6 tok/s; with the DFlash2 drafter up to 384 tok/s; 4 users 146.6 tok/s total | the same within 2 % (61.4, 377, 147.6 tok/s), same text |
| Flash-Next (GSQ IQ3_S) | 24 GB card + 64 GB RAM: about 106 tok/s (one user), 185 of 512 experts per layer on the card | each card holds half the layers and about twice the experts per layer: code prompts 157-181 tok/s with MTP, 180-224 with the DFlash2 drafter (PENDING re-measure with the per-request pick); chat 112-120; 64K / 128K about 100 (same text as one card) |

## How much the link costs

The two cards exchange one small message per step (a few rows of the hidden state). Between two RTX 4090s without a
direct GPU-to-GPU path (most consumer boards; NCCL's log says "P2P is disabled") it goes through host shared memory:
about 60 us for one step's message (8 rows), 80 us for 16, with a floor of about 20 us. Two messages per step add about
0.1 ms to a 20-25 ms step, so `--split layer` costs under 1 %.

Splits that talk once per layer would not be worth it on such a link: an expert split (48 exchanges each way per step)
would add 6-8 ms per step, a tensor split (two all-reduces per layer) about 10 ms. That is why one PC uses the layer
split.

## Limits

- Two ranks only (two GPUs, or one GPU in each of two machines); more than two are not supported.
- Built from source, multi-GPU needs NCCL: a system `libnccl-dev` is found by itself, or pass `-DTQLLM_NCCL_DIR=...`
  (for example the pip package `nvidia-nccl-cu13`: `<venv>/lib/python3.12/site-packages/nvidia/nccl`). The Docker image
  includes it.

## Two machines

Each machine runs one rank against its own copy of the model file. `--tp-peer` is rank 0's address on the link between
them (e.g. the 200 Gb ConnectX ports of two DGX Spark / GB10 boxes), and both machines must use the same NCCL version:

```
# machine A (rank 0)
NCCL_SOCKET_IFNAME=enp1s0f0np0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 0
# machine B (rank 1)
NCCL_SOCKET_IFNAME=enp1s0f0np0 tqserve --model MODEL.gguf --context 32768 --split layer --tp-rank 1 --tp-peer 10.10.0.55
```

Plain Ethernet works too, because the layer split sends one message per step. Two PCs with one RTX 3090 each over
1 GbE, Qwen3.8-27B TQ_L: decode 53.8 -> 53.9 tok/s, 4 users 129.8 -> 123.4 tok/s total, with the DFlash2 drafter
197.4 -> 153.8 (its 8-row step crosses the link). At a 128K context one card had to put the whole K/V cache in host
memory; split, it stays on the GPUs and decode is level, but prompts load 16-20 % slower (about 330 MB of rows cross
1 GbE per 32K-token prompt). Same text as one GPU in every case.

Flash-Next on two GB10s over the ConnectX link (NVFP4, MTP): each box runs half the layers with twice the experts per
layer on its memory, so long prompts decode faster: 32K / 64K / 128K 37.1 / 43.3 / 42.0 -> 48.8 / 53.0 / 44.8 tok/s;
short prompts 3-5 % slower (one hop per step). Same text as one box.

### Flash-Next: experts over two boxes (one user faster)

With every expert resident on both boxes (`slots=512` in the manifest; GB10-class memory, not a 24 GB card),
`--split experts` lets the second box compute half of each layer's experts (e % 2 == 1) for the first box's decode
steps whenever the second box has no request of its own; when it is busy, the first box computes all of them. The output
is the same as one box either way. Run both behind one address with `--peer`, so a lone request lands on the first box
and is helped, and several users spread over both. On two GB10s (NVFP4 experts, 32 unseen questions, one user at a
time, 1.7.0 build): through the router below, 71.1 tok/s median (one box alone 64.4); 16 users over both boxes
275.2 tok/s total.

```
# both boxes: the two halves of the ConnectX port (each about 100 Gb/s), four channels per peer
export NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_NCHANNELS_PER_NET_PEER=4
# machine B (rank 1): serves its own users, helps A while idle
tqserve --model flashnext.tqfn --np 8 --context 262144 --host 10.10.0.44 --port 8080 --split experts --tp-rank 1 --tp-peer 10.10.0.55
# machine A (rank 0, the front door)
tqserve --model flashnext.tqfn --np 8 --context 262144 --port 8080 --peer http://10.10.0.44:8080 --split experts --tp-rank 0
```

If rank 1 stops, the request in flight fails and rank 0 continues on its own (every expert is resident there, same
output). Inside one PC the expert split's per-layer exchanges cost more than they save (see the link costs above);
use `--split layer`.

## Notes

- The memory fit runs on both GPUs together: when the asked `--context` does not fit one of them, both load again at the
  largest context that leaves the `-fitt` margin free on each, and the startup log says so (`tqserve: fit: ...`). When
  nothing fits even at `-fitc` (default 4096), tqserve stops and names what each GPU had free. `-fit off` loads once at
  the asked context. Two RTX 4090s (24 GB each): unsloth Qwen3.8-27B Q8_0 (29 GB) fits 131K at one user (`--np 1`) and
  64K at four (`--np 4`); TQ_L at 262K stays all on the GPUs.
- `--split-at N` moves the boundary (layers N and up on rank 1) when the cards differ in size.
- Not yet with `--split layer`: `--kv-host`, `--kv-compact`, image input.
- Built from source, multi-GPU needs NCCL: `-DTQLLM_NCCL_DIR=...`, else a system `libnccl-dev`, else the pip package
  `nvidia-nccl-cu12` / `-cu13` of the active Python (the configure log says which: `NCCL: ...`). Every rank must run the
  same build and NCCL version (checked when the ranks connect).
  The Docker image includes it.
