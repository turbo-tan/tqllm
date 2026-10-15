#!/usr/bin/env python3
"""Shard-streaming MLX affine -> dense bf16 HF checkpoint (for exports too big to hold twice in RAM)."""
import argparse, json, shutil
from pathlib import Path
import mlx.core as mx, numpy as np, torch
from safetensors.torch import save_file

def to_torch(a):
    return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)

ap = argparse.ArgumentParser(); ap.add_argument("--mlx", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
src, out = Path(a.mlx), Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
        shutil.copy(f, out / f.name)
cfg = json.load(open(src / "config.json")); quant = cfg.pop("quantization")
(cfg.get("text_config") or cfg)["mtp_num_hidden_layers"] = 0
json.dump(cfg, open(out / "config.json", "w"), indent=1)
wm = {}
for i, st in enumerate(sorted(src.glob("*.safetensors"))):
    w = mx.load(str(st)); res = {}
    for k, v in w.items():
        if k.startswith("mtp.") or k.endswith(".scales") or k.endswith(".biases"):
            continue
        mod = k[:-7] if k.endswith(".weight") else None
        if mod and (mod + ".scales") in w:
            info = quant[mod]
            res[k] = to_torch(mx.dequantize(v, w[mod + ".scales"], w[mod + ".biases"], group_size=info["group_size"], bits=info["bits"]).astype(mx.bfloat16))
        else:
            res[k] = to_torch(v)
    fn = f"model-{i:05d}.safetensors"; save_file(res, out / fn, metadata={"format": "pt"})
    for k in res: wm[k] = fn
    print(fn, len(res), flush=True)
json.dump({"metadata": {}, "weight_map": wm}, open(out / "model.safetensors.index.json", "w"), indent=1)
