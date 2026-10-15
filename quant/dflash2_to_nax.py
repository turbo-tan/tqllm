#!/usr/bin/env python3
"""BF16 DFlash2 drafter GGUF (tools/dflash2_hf_to_gguf.py) -> drafter GGUF with 4-bit affine group-64 NAX projections.

Each projection is quantized as MLX's affine quantization rounds it (mx.quantize, group 64, 4 bits; the draft
precision Splash runs) and packed in the M5 matrix-unit lane order (GGUF type 317, src/mlx/nax_affine.cpp). The
selector codebooks, norms, convolution kernels and any matrix whose shape the kernel cannot tile stay as stored.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_to_nax_gguf import BLOCK, nax_pack, bf16_float  # noqa: E402

KEEP = ("selector_predecessor.weight", "selector_successor.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=os.path.expanduser("~/code/llama.cpp-tq3/gguf-py"))
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf
    import mlx.core as mx
    gguf.GGML_QUANT_SIZES[317] = BLOCK[317]

    reader = gguf.GGUFReader(a.bf16_gguf)
    writer = gguf.GGUFWriter(a.out, reader.fields["general.architecture"].contents())
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture": continue
        vt = field.types[0]
        writer.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    writer.add_uint32("tq_mlx.nax_version", 1)
    payloads, packed, kept, worst = [], 0, 0, 0.0
    for t in reader.tensors:
        ne = [int(x) for x in t.shape]
        quantize = (len(ne) == 2 and t.name.endswith(".weight") and t.name not in KEEP and ne[0] > 128 and ne[1] > 128
                    and ne[1] % 128 == 0 and ne[0] % 64 == 0
                    and t.tensor_type in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32))
        if not quantize:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            payloads.append(t.data)
            kept += 1
            continue
        N, K = ne[1], ne[0]
        raw = np.asarray(t.data)
        w = (raw.view(np.uint16).reshape(N, K).astype(np.uint32) << 16).view(np.float32) if t.tensor_type == gguf.GGMLQuantizationType.BF16 \
            else raw.reshape(N, K).astype(np.float32)
        wq, s, b = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=64, bits=4)
        ones = mx.ones(s.shape, mx.float32)
        codes = np.array(mx.dequantize(wq, ones, mx.zeros_like(ones), group_size=64, bits=4)).astype(np.uint8)
        sb = np.array(s.astype(mx.bfloat16).view(mx.uint16))
        bb = np.array(b.astype(mx.bfloat16).view(mx.uint16))
        deq = (codes.astype(np.float32).reshape(N, K // 64, 64) * bf16_float(sb)[:, :, None] + bf16_float(bb)[:, :, None]).reshape(N, K)
        ref = np.array(mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32))
        # same codes, scales and biases as MLX: only the BF16 rounding of MLX's own dequantization separates them
        err = float(np.max(np.abs(deq - ref) / (np.abs(deq) * 2.0 ** -7 + np.abs(bf16_float(bb)).max() * 2.0 ** -7 + 1e-9)))
        worst = max(worst, err)
        if err > 2.0: raise SystemExit(f"{t.name}: packed tensor differs from MLX's dequantization ({err:.3g})")
        data = nax_pack(codes, sb, bb, 4, 64)
        blk, ts = BLOCK[317]
        data = data.reshape(N, K // blk * ts)
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, 317)
        payloads.append(data)
        packed += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in payloads: writer.write_tensor_data(d)
    writer.close()
    print(f"NAX_DRAFT_DONE packed {packed} kept {kept} worst {worst:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
