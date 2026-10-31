#!/usr/bin/env python3
"""MiniMax-M3.1-Flash checkpoint dequant (own code).

Routed experts are NVFP4: weight_packed U8 [rows, cols/2] (two E2M1 codes per byte, low nibble first),
weight_scale F8_E4M3 [rows, cols/16], weight_global_scale F32; value = e2m1 * scale / global_scale.
Everything else that is quantised is MXFP8: weight F8_E4M3 [rows, cols] + weight_scale_inv U8 [rows, cols/32] (E8M0);
value = e4m3 * 2^(e8m0 - 127).

  minimax_dequant.py CKPT --layer 3 [--experts 0,1]   per-tensor stats of the dequantised weights
"""
import argparse
import json
import os
import sys

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def _e4m3_table():
    t = np.empty(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = _e4m3_table()


def e4m3_to_f32(u8):
    return E4M3[np.asarray(u8, np.uint8)]


def e8m0_to_f32(u8):
    u8 = np.asarray(u8, np.uint8)
    out = np.ldexp(np.float32(1.0), u8.astype(np.int32) - 127).astype(np.float32)
    out[u8 == 255] = np.nan
    return out


def e2m1_unpack(packed):
    """[rows, cols/2] U8 -> [rows, cols] F32, element 2j in the low nibble of byte j."""
    p = np.asarray(packed, np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), np.float32)
    out[..., 0::2] = E2M1[p & 15]
    out[..., 1::2] = E2M1[p >> 4]
    return out


def dequant_mxfp8(weight_u8, scale_u8, block=32):
    w = e4m3_to_f32(weight_u8)
    rows, cols = w.shape
    s = e8m0_to_f32(scale_u8)
    if s.shape != (rows, cols // block):
        raise ValueError(f"mxfp8: scale {s.shape} for weight {w.shape}")
    return (w.reshape(rows, cols // block, block) * s[..., None]).reshape(rows, cols)


def dequant_nvfp4(packed_u8, scale_u8, global_scale, group=16):
    q = e2m1_unpack(packed_u8)
    rows, cols = q.shape
    s = e4m3_to_f32(scale_u8) / np.float32(global_scale)
    if s.shape != (rows, cols // group):
        raise ValueError(f"nvfp4: scale {s.shape} for weight {q.shape}")
    return (q.reshape(rows, cols // group, group) * s[..., None]).reshape(rows, cols)


class Checkpoint:
    """Lazy reader over the safetensors shards; returns raw uint8 / float32 arrays (no torch needed)."""

    def __init__(self, root):
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self._headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self._headers[shard]

    def __contains__(self, name):
        return name in self.index

    def raw(self, name):
        shard = self.index[name]
        header, base = self._header(shard)
        h = header[name]
        dtype = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32, "BF16": np.uint16, "F16": np.float16}[h["dtype"]]
        a, b = h["data_offsets"]
        arr = np.fromfile(os.path.join(self.root, shard), dtype=dtype, count=(b - a) // np.dtype(dtype).itemsize,
                          offset=base + a).reshape(h["shape"])
        if h["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr

    def weight(self, prefix):
        """Dequantised F32 weight for a module prefix (…proj, …experts.E.w1, …gate)."""
        if prefix + ".weight_packed" in self:
            return dequant_nvfp4(self.raw(prefix + ".weight_packed"), self.raw(prefix + ".weight_scale"),
                                 float(self.raw(prefix + ".weight_global_scale").reshape(-1)[0]))
        if prefix + ".weight_scale_inv" in self:
            return dequant_mxfp8(self.raw(prefix + ".weight"), self.raw(prefix + ".weight_scale_inv"))
        return self.raw(prefix + ".weight").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", default="0,1")
    a = ap.parse_args()
    ck = Checkpoint(a.ckpt)
    pre = f"language_model.model.layers.{a.layer}."
    mods = sorted({k.rsplit(".", 1)[0] for k in ck.index if k.startswith(pre) and ".experts." not in k
                   and (k.endswith(".weight") or k.endswith(".weight_packed"))})
    for e in a.experts.split(","):
        mods += [f"{pre}block_sparse_moe.experts.{e}.w{i}" for i in (1, 2, 3)]
    for m in mods:
        w = ck.weight(m)
        print(f"{m[len(pre):]:48s} {str(w.shape):16s} std {w.std():.5f} absmax {np.abs(w).max():.4f} "
              f"nan {int(np.isnan(w).sum())}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
