# Contextual embeddings: `POST /v1/embeddings`

tqserve serves [pplx-embed-v2-context](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b) (MIT) models:
a document is embedded as one sequence (its chunks see each other through bidirectional attention) and every chunk gets
its own vector. No embedding GGUF is published: convert the Hugging Face snapshot yourself (below), then start the
server with that file; it serves `POST /v1/embeddings`, `GET /v1/models` and `GET /health`.

```sh
python3 tools/convert/pplx/pplx_context_to_gguf.py <snapshot folder> pplx-embed-v2-context-0.6b-bf16.gguf --outtype bf16
tqserve --model pplx-embed-v2-context-0.6b-bf16.gguf --port 8090
```

## Requests

OpenAI shape: `input` is a string or a list of strings, each one chunk (its own document, or a query).

```sh
curl -s localhost:8090/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["where did the cat sit?"], "task": "query"}'
```

Contextual documents: `input` is a list of documents, each a list of chunks. Every chunk gets a vector, in order, and
each item also carries `document` and `chunk`.

```sh
curl -s localhost:8090/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": [["The cat sat on the mat.", "Later, it left the room."], ["Second document."]], "task": "document"}'
```

- `task`: `document` (default) or `query`. A query is one string.
- `encoding_format`: `float` (default, L2-normalised), `base64` (the same L2-normalised vector as little-endian float32;
  the OpenAI Python client asks for this by default) or `int8` (`round(tanh(x) * 127)`, what the model's own `encode()`
  returns).
- `dimensions`: accepted only when it equals the model's (1024 for the 0.6B, 2048 for the 9B); the vectors are not
  truncated.

Limits per request (a 400 names the one exceeded): 4,096 inputs, 32,768 chunks in total, 16 MB of text, 1,048,576
tokens; each document or query must also fit the model's context. `/health` and `/v1/models` answer while a request
runs; requests share the GPU one at a time.

The server tokenizes the request's entries in parallel and runs them as segments of shared forward passes, so batching
many documents or queries into one request is faster than one request each.

## Converting a model, and memory

Every file is made by you from perplexity's Hugging Face snapshot (0.6B or 9B) with
`tools/convert/pplx/pplx_context_to_gguf.py SNAPSHOT OUT.gguf --outtype <type>`; the file names below are only examples.

| Tier | Snapshot, `--outtype` | Example output (size) | Served memory | Notes |
|---|---|---|---|---|
| small | 0.6B, `bf16` | `pplx-embed-v2-context-0.6b-bf16.gguf` (1.5 GB) | 2.4 GB | BF16 layers, F32 token table; covid-qa nDCG@10 56.17 (Hugging Face FP32 56.32) |
| 8 GB GPU | 9B, `tq4` | `pplx-embed-v2-context-9b-tq4.gguf` (4.7 GB) | 6.6 GB | 9B, TQ4 weights |
| 12 / 16 GB GPU | 9B, `q8_0` | `pplx-embed-v2-context-9b-q8_0.gguf` (8.5 GB) | 10.2 GB | 9B, Q8_0 weights |

Served memory is measured at the default 12K-token passes (weights, activations and scratch).

## Accuracy

The server runs the BF16 fast path. Against the Hugging Face model in FP32 (50 covid-qa documents with all their chunks
and 50 queries, 1,510 vectors), the 0.6B file's cosine 1st percentile is 0.99993 or better on an RTX 3090 and a GB10, and its lowest
row 0.9946 (GB10) / 0.9962 (RTX 3090), above the 0.982 floor: the lowest cosine at which retrieval (covid-qa nDCG@10,
FinChat) was still unchanged against the FP32 model. Retrieval within 0.2 nDCG@10 of FP32 on each GPU is the deciding
check; exact mode gives 1.000000.

Embedding models need a CUDA build (RTX 30 / 40 / 50, GB10); the Mac build refuses an embedding GGUF, and so do the
chat engines. The server finds an embedding GGUF from its metadata, so `--model` takes either kind: a chat GGUF serves
chat, an embedding GGUF serves only `POST /v1/embeddings` (other paths answer 404). Requests run one at a time on the
GPU; `--name` sets the model id in `/v1/models` and `--api-key` works as for chat.
