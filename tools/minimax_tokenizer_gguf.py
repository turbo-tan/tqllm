#!/usr/bin/env python3
"""MiniMax-M3.1-Flash tokenizer GGUF for tqserve --tokenizer: byte-level BPE vocab + merges (tokenizer.json), the added
tokens (special -> control, others user-defined), eos, and the chat template. pre = minimax-m3 (src/core/tokenizer.cpp).

  minimax_tokenizer_gguf.py CKPT OUT.gguf --vocab 200064
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--vocab", type=int, default=200064)
    a = ap.parse_args()
    import gguf
    t = json.load(open(os.path.join(a.ckpt, "tokenizer.json")))
    cfg = json.load(open(os.path.join(a.ckpt, "tokenizer_config.json")))
    tokens = [None] * a.vocab
    types = [5] * a.vocab
    for piece, i in t["model"]["vocab"].items():
        tokens[i] = piece; types[i] = 1
    for at in t["added_tokens"]:
        tokens[at["id"]] = at["content"]; types[at["id"]] = 3 if at.get("special") else 4
    for i in range(a.vocab):
        if tokens[i] is None: tokens[i] = f"[PAD{i}]"
    merges = [m if isinstance(m, str) else " ".join(m) for m in t["model"]["merges"]]
    ids = {tok: i for i, tok in enumerate(tokens)}
    w = gguf.GGUFWriter(a.out, "minimax-m3")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("minimax-m3")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(ids[cfg["eos_token"]])
    w.add_bos_token_id(ids[cfg["bos_token"]])
    w.add_pad_token_id(ids[cfg["pad_token"]])
    w.add_chat_template(open(os.path.join(a.ckpt, "chat_template.jinja")).read())
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {a.vocab} tokens ({sum(1 for x in types if x == 3)} control), {len(merges)} merges")


if __name__ == "__main__":
    sys.exit(main())
