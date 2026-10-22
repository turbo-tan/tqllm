#!/usr/bin/env python3
"""mlx-vlm affine checkpoint (e.g. mlx-community/Qwen3.8-27B-4bit) -> dense bf16 HF checkpoint in the original naming.

Quantized matrices are dequantized from the MLX file (codes, scales, biases) and written under their HF names. Every
other tensor (norms, conv, A_log, dt_bias, vision tower) is copied from the original BF16 checkpoint, which the MLX
file stores unquantized; each is compared with the MLX copy (after the layout changes mlx-vlm applies) and reported.
Config and tokenizer files come from the original checkpoint so convert_hf_to_gguf.py sees the same model.
"""
import argparse, json, shutil
from pathlib import Path
import mlx.core as mx, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file


def hf_name(k):
    if k.startswith("language_model.model."): return "model.language_model." + k[len("language_model.model."):]
    if k.startswith("language_model.lm_head"): return "lm_head" + k[len("language_model.lm_head"):]
    if k.startswith("vision_tower."): return "model.visual." + k[len("vision_tower."):]
    return k


def to_torch(a):
    return torch.from_numpy(np.array(a.astype(mx.float32))).to(torch.bfloat16)


ap = argparse.ArgumentParser()
ap.add_argument("--mlx", required=True)
ap.add_argument("--original", required=True, help="original BF16 HF checkpoint (config, tokenizer, unquantized tensors)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
src, orig, out = Path(a.mlx), Path(a.original), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in orig.iterdir():
    if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
        shutil.copy(f, out / f.name)
cfg = json.load(open(out / "config.json"))
(cfg.get("text_config") or cfg)["mtp_num_hidden_layers"] = 0
json.dump(cfg, open(out / "config.json", "w"), indent=1)
quant = json.load(open(src / "config.json"))["quantization"]
owm = json.load(open(orig / "model.safetensors.index.json"))["weight_map"]
handles = {}


def original(name):
    fn = owm[name]
    if fn not in handles: handles[fn] = safe_open(str(orig / fn), "pt")
    return handles[fn].get_tensor(name)


wm, used, report = {}, set(), {"exact": 0, "layout": 0, "differs": []}
for i, st in enumerate(sorted(src.glob("*.safetensors"))):
    w = mx.load(str(st)); res = {}
    for k, v in w.items():
        if k.endswith(".scales") or k.endswith(".biases"): continue
        mod = k[:-7] if k.endswith(".weight") else None
        name = hf_name(k)
        if mod and (mod + ".scales") in w:
            info = quant.get(mod, quant) if isinstance(quant.get(mod, None), dict) else quant
            res[name] = to_torch(mx.dequantize(v, w[mod + ".scales"], w[mod + ".biases"],
                                               group_size=info["group_size"], bits=info["bits"]).astype(mx.bfloat16))
        else:
            if name not in owm: raise SystemExit(f"{k} -> {name}: not in the original checkpoint")
            t = original(name)
            m = np.array(v.astype(mx.float32)); o = t.float().numpy()
            if m.shape == o.shape and np.array_equal(m, o): report["exact"] += 1
            elif m.size == o.size and np.array_equal(np.sort(m.ravel()), np.sort(o.ravel())): report["layout"] += 1
            else: report["differs"].append(name)
            res[name] = t
        used.add(name)
    fn = f"model-{i:05d}.safetensors"; save_file(res, out / fn, metadata={"format": "pt"})
    for k in res: wm[k] = fn
    print(fn, len(res), flush=True)
missing = [k for k in owm if k not in used and not k.startswith("mtp.")]
if missing:
    rest = {k: original(k) for k in missing}
    fn = f"model-{i + 1:05d}.safetensors"; save_file(rest, out / fn, metadata={"format": "pt"})
    for k in rest: wm[k] = fn
json.dump({"metadata": {}, "weight_map": wm}, open(out / "model.safetensors.index.json", "w"), indent=1)
print("UNQUANTIZED", json.dumps({"exact": report["exact"], "same_values_other_layout": report["layout"],
                                 "differs": len(report["differs"]), "differs_first": report["differs"][:12],
                                 "copied_missing": len(missing)}), flush=True)
