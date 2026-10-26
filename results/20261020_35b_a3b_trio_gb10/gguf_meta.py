#!/usr/bin/env python3
"""GGUF header of one file -> architecture, layer count and the routed-expert tensor types.

Runs on the box that holds the file (piped over ssh by measure_speed.py): it only reads the
header, never the tensor data.
"""
import json
import struct
import sys

T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<", 10: "<Q", 11: "<q", 12: "<d"}
NAME = {0: "U8", 1: "I8", 2: "U16", 3: "I16", 4: "U32", 5: "I32", 6: "F32", 7: "BOOL", 8: "STRING",
        9: "ARRAY", 10: "U64", 11: "I64", 12: "F64"}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def read_value(f, t):
    if t == 8:
        return read_string(f)
    if t == 9:
        et, n = struct.unpack("<I", f.read(4))[0], struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    fmt = T[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def main():
    f = open(sys.argv[1], "rb")
    assert f.read(4) == b"GGUF", "not a GGUF"
    f.read(8)
    nkv, nti = struct.unpack("<QQ", f.read(16))
    meta, types = {}, {}
    for _ in range(nkv):
        k = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        meta[k] = read_value(f, t)
    for _ in range(nti):
        name = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        f.read(8)
        types[name] = (NAME.get(t, str(t)), dims)
    exp = {n: v[0] for n, v in types.items() if "exps.weight" in n}
    kinds = {}
    for n, ty in exp.items():
        kinds[ty] = kinds.get(ty, 0) + 1
    out = {"architecture": meta.get("general.architecture"),
           "name": meta.get("general.name"),
           "size_label": meta.get("general.size_label"),
           "layers": meta.get("general.block_count") or meta.get("qwen35moe.block_count"),
           "expert_count": None, "expert_used": None, "expert_ff": None,
           "expert_tensor_types": kinds,
           "tensor_bytes": sum(v for v in [0])}
    for pre in ("qwen35moe", "qwen35", "qwen3"):
        if meta.get(pre + ".expert_count") is not None:
            out["expert_count"] = meta[pre + ".expert_count"]
            out["expert_used"] = meta.get(pre + ".expert_used")
            out["expert_ff"] = meta.get(pre + ".expert_ff")
            out["layers"] = meta.get(pre + ".block_count", out["layers"])
            break
    print(json.dumps(out))


if __name__ == "__main__":
    main()
