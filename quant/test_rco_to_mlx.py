"""Checks that the exporter's packing and affine mapping equal MLX's own dequantization."""
import sys, numpy as np, torch
import mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rco_to_mlx import pack_codes

rng = np.random.default_rng(0)
for bits in (2, 3, 4):
    rows, cols = 64, 512
    q = rng.integers(0, 2 ** bits, (rows, cols)).astype(np.uint8)
    s = (rng.random((rows, cols // 128)) + 0.5).astype(np.float32)
    z = rng.random((rows, cols // 128)).astype(np.float32) * (2 ** bits - 1)
    packed = pack_codes(q, bits)
    w = mx.dequantize(mx.array(packed), mx.array(s), mx.array(-s * z), group_size=128, bits=bits)
    ref = s.repeat(128, 1) * (q.astype(np.float32) - z.repeat(128, 1))
    err = np.abs(np.array(w) - ref).max()
    # and against MLX's own packer on a grid that spans every level
    q2 = q.copy(); q2[:, ::128] = 0; q2[:, 1::128] = 2 ** bits - 1
    wq, sc, bi = mx.quantize(mx.array(q2.astype(np.float32)), group_size=128, bits=bits)
    same = np.array_equal(np.array(wq), pack_codes((2 ** bits - 1 - q2).astype(np.uint8), bits))  # MLX picks a negative scale
    print(f"bits {bits}: dequant err {err:.2e}, packing equals mx.quantize: {same}")
    assert err < 1e-4 and same
print("RCO_TO_MLX_PASS")
