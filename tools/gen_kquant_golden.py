#!/usr/bin/env python3
"""Golden K-quant / IQ dequant values from gguf-py (ggml-quants.c reference layout) for tests/mlx_kquant.cpp.
Per type: random blocks with sane FP16 scale fields, the bytes and gguf-py's FP32 values.
Output tests/data/kquant_golden.bin: repeated [u32 type][u32 blocks][u32 block bytes][u32 elements per block][bytes][f32 values]."""
import struct, sys
import numpy as np
from gguf import GGMLQuantizationType as T, GGML_QUANT_SIZES, quants

def f16_bytes(rng, n, lo, hi):
    return np.float16(rng.uniform(lo, hi, n) * rng.choice([-1, 1], n)).view(np.uint8).reshape(n, 2)

rng = np.random.default_rng(20261022)
out = open(sys.argv[1] if len(sys.argv) > 1 else "tests/data/kquant_golden.bin", "wb")
# (type, byte offsets of FP16 fields that must stay finite)
spec = [(T.Q2_K, [80, 82]), (T.Q3_K, [108]), (T.Q4_K, [0, 2]), (T.Q5_K, [0, 2]), (T.IQ3_S, [0]), (T.IQ4_XS, [0]), (T.IQ4_NL, [0])]
for qt, halves in spec:
    elems, nbytes = GGML_QUANT_SIZES[qt]
    blocks = 16 if elems == 256 else 64
    raw = rng.integers(0, 256, size=(blocks, nbytes), dtype=np.uint8)
    for off in halves: raw[:, off:off + 2] = f16_bytes(rng, blocks, 1e-3, 5e-2)
    vals = quants.dequantize(raw.reshape(-1), qt).astype(np.float32).reshape(-1)
    assert vals.size == blocks * elems and np.isfinite(vals).all()
    out.write(struct.pack("<4I", int(qt), blocks, nbytes, elems)); out.write(raw.tobytes()); out.write(vals.tobytes())
    print(qt.name, blocks, "blocks", nbytes, "bytes", "max |v|", float(np.abs(vals).max()))
