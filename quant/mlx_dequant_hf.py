#!/usr/bin/env python3
"""Dense bf16 HF checkpoint from an MLX affine checkpoint, or a stock round-to-nearest baseline.

  mlx_dequant_hf.py --mlx DIR --out DIR                      dequantize an rco_to_mlx export
  mlx_dequant_hf.py --rtn-bits 4 --rtn-group 64 --model DIR --out DIR   what mlx_lm.convert would store
Both write the same tensors the scorer reads, so KLD compares like with like.
"""
import argparse, json, re, shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import save_file

SKIP = re.compile(r"(embed_tokens|lm_head|norm|conv1d|A_log|dt_bias|visual|mtp)")
TARGET = re.compile(r".*(q|k|v|o|gate|up|down)_proj\.weight$|.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$")


def to_torch(a):
    return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx")
    ap.add_argument("--model")
    ap.add_argument("--rtn-bits", type=int)
    ap.add_argument("--rtn-group", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.mlx or a.model)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    cfg = json.load(open(src / "config.json"))
    quant = cfg.pop("quantization", None)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    weights = {}
    for st in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(st)))
    result = {}
    for key, w in weights.items():
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        module = key[:-7] if key.endswith(".weight") else None
        if a.mlx and module and (module + ".scales") in weights:
            info = quant[module]
            d = mx.dequantize(w, weights[module + ".scales"], weights[module + ".biases"],
                              group_size=info["group_size"], bits=info["bits"])
            result[key] = to_torch(d.astype(mx.bfloat16))
        elif a.rtn_bits and TARGET.match(key) and not SKIP.search(key):
            wq, s, b = mx.quantize(w.astype(mx.float32), group_size=a.rtn_group, bits=a.rtn_bits)
            # stock mlx_lm stores scales/biases in the weight dtype
            d = mx.dequantize(wq, s.astype(w.dtype), b.astype(w.dtype), group_size=a.rtn_group, bits=a.rtn_bits)
            result[key] = to_torch(d.astype(mx.bfloat16))
        else:
            result[key] = to_torch(w)
    # keep 2 shards of any size: the scorer reads a directory
    save_file(result, out / "model.safetensors", metadata={"format": "pt"})
    print("wrote", len(result), "tensors to", out)


if __name__ == "__main__":
    main()
