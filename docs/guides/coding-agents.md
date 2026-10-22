# Coding agents

[Documentation](../README.md) / [Project home](../../README.md)

Run the commands below from the repository root.

## tqllm launch: a coding agent on your own GPU

`tools/launch/tqllm.py` starts a local server and runs a coding agent against it, like `ollama launch`. It needs
Python 3, Docker with the NVIDIA Container Toolkit (or a native `tqserve` build) and the agent itself installed.

```sh
python3 tools/launch/tqllm.py --list                    # model catalog (local copies are marked)
python3 tools/launch/tqllm.py launch opencode           # the default model for this GPU, downloaded if missing
python3 tools/launch/tqllm.py launch codex --model qwen3.8-27b-xs
python3 tools/launch/tqllm.py launch claude --model models/Qwen3.8-27B-TQ_S.gguf \
  --drafter models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt
python3 tools/launch/tqllm.py serve --model qwen3.8-27b  # just the server, until Ctrl+C
```

- **Agents:** `opencode`, `codex` (through the server's OpenAI Responses API) and `claude` (Claude Code, through the
  server's Anthropic Messages API).
  `pi` (the pi coding agent) and `dsh` (DeepSeek Harness, `web` or `headless` profile) also run against the server.
  Arguments after `--` go to the agent, e.g. `launch opencode -- run "fix the failing test"`.
- **Models:** `--model` takes a GGUF file, a catalog name from `tools/launch/models.json`, or nothing (the default
  for the GPU's memory). Catalog files are looked up in `./models`, `~/models` and `$TQLLM_MODELS`; missing ones are
  offered for download from Hugging Face with a size and sha256 check (`--yes` skips the question).
- **Server:** the Docker image by default, or `--native build/tqserve`. Context defaults to 65536 because agent
  prompts are long; `--context`, `--port`, `--reasoning` and `--server-arg` (repeatable) pass through. A server already
  answering on the port is reused and left running; one the command started is stopped when the agent exits or the
  command gets Ctrl+C / SIGTERM.
- **Sampling and length:** agents get temperature 0.6, top-p 0.95 and top-k 20 for requests that set none (Qwen3.8's
  thinking-mode settings; Codex sends none, and greedy decoding can repeat itself on repetitive output such as ASCII art)
  and `--max-tokens 32768`. `--temperature 0` restores greedy decoding; `--top-p`, `--top-k` and `--max-tokens` override.
  The server takes the same defaults as `tqserve --temperature T --top-p P --top-k K`.
- The agents' own configuration files are never written; each gets its provider through its environment or command
  line.

### Codex CLI

![tqllm launch codex on one RTX 3090](../assets/tqllm-launch-codex.gif)

*`tqllm launch codex` on one RTX 3090 (Qwen3.8-27B TQ3_4S-v2, DFlash2 drafter): the Codex TUI writes and runs
`fizzbuzz.py`. 47 s of server start are cut from the recording.*

Codex talks to the server's OpenAI Responses API (`/v1/responses`). `launch codex` passes the provider on Codex's
command line; arguments after `--` go to Codex:

```sh
python3 tools/launch/tqllm.py launch codex --native build/tqserve            # the Codex TUI in this directory
python3 tools/launch/tqllm.py launch codex --native build/tqserve -- exec "add a unit test for utils.py"
```

To point your own Codex at a running server instead, add this to `~/.codex/config.toml` and run `codex`:

```toml
model = "tqllm"
model_provider = "tqllm"

[model_providers.tqllm]
name = "tqLLM"
base_url = "http://localhost:8086/v1"
wire_api = "responses"
```

- Needs a server with `/v1/responses`: Docker image 1.2.0 or newer, or a [native build](quickstart.md#native-build) of
  this repository.
- Codex 0.16x no longer accepts `wire_api = "chat"`. Tested with Codex 0.162.1.
