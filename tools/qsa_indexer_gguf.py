#!/usr/bin/env python3
"""qsa_indexer_gguf.py INDEXERS.safetensors OUT.gguf: the QSA indexers of higa2515/Qwen3.8-27B-QSA-BF16
(qsa-indexers.safetensors, FP32, trained on the frozen Qwen3.8-27B backbone) as GGUF tensors for the Metal engine:
blk.N.qsa_q.weight [512, hidden] and blk.N.qsa_k.weight [128, hidden] FP16, blk.N.qsa_{q,k,k_block}_norm.weight [128]
FP32 with the zero-centered scale folded (1 + w). Add them to a model file with gguf_add.py (TQ4 files are not
supported: their activations are in the rotated basis)."""
import json, struct, sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from gguf_truncate import T_UINT32, w_string, w_value

GGML_F32, GGML_F16, ALIGN = 0, 1, 32


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n)); base = 8 + n
        out = {}
        for name, info in head.items():
            if name == "__metadata__": continue
            assert info["dtype"] == "F32", name
            a, b = info["data_offsets"]; f.seek(base + a)
            out[name] = np.frombuffer(f.read(b - a), dtype="<f4").reshape(info["shape"])
    return out


def main():
    src, dst = sys.argv[1:3]
    t = read_safetensors(src)
    tensors = []
    for name in sorted(t):
        layer = int(name.split(".layers.")[1].split(".")[0]); field = name.rsplit("._qsa_indexer.", 1)[1]
        p = f"blk.{layer}."
        if field == "q_proj.weight": tensors.append((p + "qsa_q.weight", t[name].astype("<f2"), GGML_F16))
        elif field == "k_proj.weight": tensors.append((p + "qsa_k.weight", t[name].astype("<f2"), GGML_F16))
        else:
            out = {"q_norm": "qsa_q_norm.weight", "k_norm": "qsa_k_norm.weight", "k_block_norm": "qsa_k_block_norm.weight"}[field]
            tensors.append((p + out, (1.0 + t[name]).astype("<f4"), GGML_F32))
    kv = [(b"qsa.budget", T_UINT32, 2048), (b"qsa.block_size", T_UINT32, 4), (b"qsa.indexer_heads", T_UINT32, 4),
          (b"qsa.indexer_head_dim", T_UINT32, 128), (b"qsa.rope_dim", T_UINT32, 64)]
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, ty, v in kv: w_string(out, k); out += struct.pack("<I", ty); w_value(out, ty, v)
    off = 0
    for name, a, ty in tensors:
        dims = list(reversed(a.shape))
        w_string(out, name.encode()); out += struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", ty, off); off += (a.nbytes + ALIGN - 1) // ALIGN * ALIGN
    out += b"\0" * ((-len(out)) % ALIGN)
    with open(dst, "wb") as o:
        o.write(out)
        for _, a, _ in tensors: o.write(a.tobytes()); o.write(b"\0" * ((-a.nbytes) % ALIGN))
    print(f"{dst}: {len(tensors)} tensors")


if __name__ == "__main__":
    main()
