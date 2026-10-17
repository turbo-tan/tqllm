#!/usr/bin/env python3
"""Export an RCO/GPTQ database as a stock-MLX affine checkpoint (mixed 2/3/4-bit, group 128).

RCO stores integer codes q with W = s * (q - z); MLX affine stores W = scale * q + bias, so
scales = s and biases = -s * z exactly. Nothing is re-quantized: the codes are only repacked.
Loadable with mlx_lm.load; per-matrix bit widths are recorded in config.json["quantization"].
"""
import argparse, json, re, shutil, sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def pack_codes(q: np.ndarray, bits: int) -> np.ndarray:
    """Little-endian bitstream per row, as MLX packs rows into uint32 words."""
    rows, cols = q.shape
    if bits in (2, 4, 8):
        pf = 32 // bits
        grouped = q.reshape(rows, cols // pf, pf).astype(np.uint64)
        shifts = (np.arange(pf, dtype=np.uint64) * bits)
        return (grouped << shifts).sum(-1).astype(np.uint32)
    if bits == 3:  # 8 codes = 24 bits = 3 bytes
        grouped = q.reshape(rows, cols // 8, 8).astype(np.uint32)
        v = (grouped << (np.arange(8, dtype=np.uint32) * 3)).sum(-1, dtype=np.uint32)
        by = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], -1).astype(np.uint8).reshape(rows, -1)
        return np.ascontiguousarray(by).view(np.uint32)
    raise ValueError(f"unsupported bits {bits}")


def write_safetensors(path, entries):
    header, offset = {"__metadata__": {"format": "mlx"}}, 0
    for name, (dtype, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": list(raw.shape) if dtype != "U32" or raw.ndim else [], "data_offsets": [offset, offset + raw.nbytes]}
        offset += raw.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for _, raw in entries.values():
            f.write(raw.tobytes())


def affine_from_qparams(bundle):
    if bundle.get("perm") is not None:
        raise ValueError("act-order permutation is not supported")
    if bundle["sym"] or bundle["group_size"] != 128:
        raise ValueError("expected asymmetric group-128 qparams")
    s = bundle["scales"].to(torch.float32)
    z = bundle["zeros"].to(torch.float32)
    return bundle["qweight"].numpy(), s, -s * z


def read_assignment(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or ":" not in line:
            continue
        name, bits = line.rsplit(":", 1)
        out[name.strip()] = int(bits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF dir (bf16) with the unquantized tensors")
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--config", help="RCO allocation .txt")
    ap.add_argument("--uniform-bw", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-bits", type=int, default=0)
    ap.add_argument("--head-bits", type=int, default=0)
    a = ap.parse_args()
    assign = read_assignment(a.config) if a.config else {}
    model, layers, out = Path(a.model), Path(a.layer_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in model.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        if f.is_file():
            shutil.copy(f, out / f.name)
    for old in out.glob("model-*.safetensors"):
        old.unlink()
    quant_cfg = {"group_size": 128, "bits": 4}
    shard, pending, pending_bytes, shards = 0, {}, 0, []

    def add(name, array, dtype):
        nonlocal pending_bytes
        raw = np.ascontiguousarray(array)
        pending[name] = (dtype, raw)
        pending_bytes += raw.nbytes

    def flush():
        nonlocal shard, pending, pending_bytes
        if not pending:
            return
        fn = f"model-{shard:05d}.safetensors"
        write_safetensors(out / fn, pending)
        shards.append((fn, list(pending)))
        shard, pending, pending_bytes = shard + 1, {}, 0

    def bf16(t):
        return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy()

    for st in sorted(model.glob("*.safetensors")):
        with safe_open(st, "pt") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    continue
                module = key[: -len(".weight")] if key.endswith(".weight") else None
                dbname = module.replace("model.language_model.", "model.", 1) if module else None
                bits = assign.get(dbname, a.uniform_bw) if module else None
                qfile = layers / dbname / f"{bits}_qparams.pt" if bits else None
                if qfile is not None and qfile.exists():
                    bundle = torch.load(qfile, map_location="cpu", weights_only=False)
                    q, scales, biases = affine_from_qparams(bundle)
                    add(key, pack_codes(q.astype(np.uint8), bits), "U32")
                    add(module + ".scales", bf16(scales), "BF16")
                    add(module + ".biases", bf16(biases), "BF16")
                    quant_cfg[module] = {"group_size": 128, "bits": bits}
                elif (key.endswith("embed_tokens.weight") and a.embed_bits) or (key.endswith("lm_head.weight") and a.head_bits):
                    import mlx.core as mx
                    b = a.embed_bits if "embed_tokens" in key else a.head_bits
                    t = f.get_tensor(key).to(torch.float32).numpy()
                    wq, sc, bi = mx.quantize(mx.array(t), group_size=64, bits=b)
                    add(key, np.array(wq), "U32")
                    add(module + ".scales", np.array(sc.astype(mx.bfloat16).view(mx.int16)), "BF16")
                    add(module + ".biases", np.array(bi.astype(mx.bfloat16).view(mx.int16)), "BF16")
                    quant_cfg[module] = {"group_size": 64, "bits": b}
                else:
                    t = f.get_tensor(key)
                    if t.dtype == torch.bfloat16:
                        add(key, bf16(t), "BF16")
                    elif t.dtype == torch.float32:
                        add(key, t.numpy(), "F32")
                    elif t.dtype == torch.float16:
                        add(key, t.view(torch.int16).numpy(), "F16")
                    else:
                        raise ValueError(f"unhandled dtype {t.dtype} for {key}")
                if pending_bytes > 3 << 30:
                    flush()
    flush()
    index = {"metadata": {}, "weight_map": {k: fn for fn, ks in shards for k in ks}}
    json.dump(index, open(out / "model.safetensors.index.json", "w"), indent=1)
    cfg = json.load(open(model / "config.json"))
    cfg.pop("quantization_config", None)
    cfg["quantization"] = quant_cfg
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    bits_hist = {}
    for k, v in quant_cfg.items():
        if isinstance(v, dict):
            bits_hist[v["bits"]] = bits_hist.get(v["bits"], 0) + 1
    print("matrices by bits:", bits_hist, "shards:", len(shards))


if __name__ == "__main__":
    sys.exit(main())
