#!/usr/bin/env python3
"""Download Qwen3.8-27B weights for the tqLLM / azllm engine.

Everything lands in the layout the run scripts already expect:

    <dir>/Qwen3.8-27B-<type>.gguf
    <dir>/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
    <dir>/mtp-vocab/atx_65536.txt
    <dir>/mmproj-BF16.gguf                              (with --mmproj)

Sample runs
-----------
python3 get_models.py
    The defaults: the TQ3_4S build (14.50 GB) into ./models, plus the DFlash2
    drafter and its draft vocab, which run_3090.sh needs to speculate.

python3 get_models.py --type TQ3_4S-v2
    Identical weights, same size, but the fixed froggeric v22 chat template is
    embedded in the file. Use this one for agent and tool-calling work: the
    official template throws "Can only get item pairs from a mapping" when a
    client sends tool arguments as a JSON string. Also the only build here that
    llama.cpp (and llamAmpere) can load, and the only one with an mmproj.

python3 get_models.py --type TQ_L
    Best quality of the published builds: whole body at TQ4, 4.31 bpw, 14.72 GB,
    KLD 0.022 vs BF16. Needs --chat-template-file at serve time because it
    embeds the official template; the script prints the reminder, and
    run_3090_TQ_L.sh handles it automatically.

python3 get_models.py --type TQ_S --dir /data/models
    Fetch into another folder. Note the run scripts look for models/ next to
    themselves, so if you download elsewhere you must also set MODEL= (and
    DRAFTER=/DRAFT_VOCAB=) when serving.

python3 get_models.py --type TQ_L --mmproj --no-drafter
    TQ_L plus the vision projector, skipping the drafter because it is already
    on disk. In practice --no-drafter is unnecessary: files that are already
    there are verified and skipped rather than re-downloaded.

python3 get_models.py --type TQ_XS --no-drafter
    Smallest text-only setup, 11.85 GB, for a 16 GB card.

Which type to pick
------------------
    TQ3_4S     14.50 GB, 4.24 bpw. Loads in tqLLM AND llama.cpp-tq3.
    TQ3_4S-v2  same weights, fixed chat template. Recommended.
    TQ_XS      11.85 GB, 3.47 bpw. Smallest and fastest; weakest coding/instruct.
    TQ_S       12.55 GB, 3.67 bpw. Best quality per GB; frees ~2 GB for context.
    TQ_M       13.14 GB, 3.84 bpw. Experimental TQ3/TQ4 mix (KLD only).
    TQ_L       14.72 GB, 4.31 bpw. Closest to BF16 (KLD 0.022, 93.7% top token).

The TQ_XS/S/M/L files use GGUF types 300/301, which only tqLLM can read:

    MODEL=models/Qwen3.8-27B-TQ_L.gguf TEMPLATE=models/chat_template_v22.jinja ./run_3090.sh

No Hugging Face token is needed - these are public repos. Interrupted downloads
resume where they stopped.
"""
import argparse
import pathlib
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("needs huggingface_hub:  pip install huggingface_hub")

REPO_34S = "YTan2000/Qwen3.8-27B-TQ3_4S"
REPO_TQ = "YTan2000/Qwen3.8-27B-TQ"

# type -> (repo, filename, size, note)
FILES = {
    "TQ3_4S":    (REPO_34S, "Qwen3.8-27B-TQ3_4S.gguf",    "14.50 GB", "4.24 bpw; official chat template"),
    "TQ3_4S-v2": (REPO_34S, "Qwen3.8-27B-TQ3_4S-v2.gguf", "14.50 GB", "4.24 bpw; fixed v22 chat template"),
    "TQ_XS":     (REPO_TQ,  "Qwen3.8-27B-TQ_XS.gguf",     "11.85 GB", "3.47 bpw; tqLLM only; unfixed template"),
    "TQ_S":      (REPO_TQ,  "Qwen3.8-27B-TQ_S.gguf",      "12.55 GB", "3.67 bpw; tqLLM only; unfixed template"),
    "TQ_M":      (REPO_TQ,  "Qwen3.8-27B-TQ_M.gguf",      "13.14 GB", "3.84 bpw; experimental mix; unfixed"),
    "TQ_L":      (REPO_TQ,  "Qwen3.8-27B-TQ_L.gguf",      "14.72 GB", "4.31 bpw; closest to BF16; unfixed"),
}
DRAFT = (REPO_TQ, ["drafters/Qwen3.8-27B-DFlash2-TQ3.gguf", "mtp-vocab/atx_65536.txt"])
MMPROJ = (REPO_34S, ["mmproj-BF16.gguf"])

DEFAULT_DIR = pathlib.Path(__file__).resolve().parent / "models"


def main():
    p = argparse.ArgumentParser(
        description="Fetch Qwen3.8-27B weights for tqLLM/azllm.",
        epilog="run with no arguments for the default TQ3_4S build; see the module docstring for examples",
    )
    p.add_argument("--type", choices=sorted(FILES), default="TQ3_4S",
                   help="quant build to fetch (default: %(default)s)")
    p.add_argument("--dir", default=str(DEFAULT_DIR), metavar="DIR",
                   help="target folder (default: %(default)s)")
    p.add_argument("--no-drafter", action="store_true",
                   help="skip the DFlash2 drafter and its vocab (needed by run_3090.sh)")
    p.add_argument("--mmproj", action="store_true",
                   help="also fetch the vision projector (served with tqserve --mmproj)")
    a = p.parse_args()

    repo, name, size, note = FILES[a.type]
    target = pathlib.Path(a.dir).expanduser()
    print("model   %s  (%s, %s)\n  -> %s/%s" % (a.type, size, note, target, name), flush=True)
    snapshot_download(repo, local_dir=str(target), allow_patterns=[name])

    if not a.no_drafter:
        print("drafter DFlash2 + draft vocab (0.79 GB)", flush=True)
        snapshot_download(DRAFT[0], local_dir=str(target), allow_patterns=DRAFT[1])

    if a.mmproj:
        print("mmproj  vision projector", flush=True)
        snapshot_download(MMPROJ[0], local_dir=str(target), allow_patterns=MMPROJ[1])

    print("\ndone:\n  " + str(target / name))
    if "unfixed" in note:
        print("note: this file embeds the official chat template. For tool calling serve it with\n"
              "      TEMPLATE=models/chat_template_v22.jinja ./run_3090.sh")


if __name__ == "__main__":
    main()
