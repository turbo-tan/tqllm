#!/usr/bin/env python3
"""MLX affine export -> tq_mlx NAX GGUF.

Names, metadata and the non-quantized tensors come from the BF16 GGUF that convert_hf_to_gguf.py made from the
dequantized export (mlx_dequant_stream.py). Quantized matrices are re-packed from the export's codes, scales and
biases in the lane order of the M5 matrix unit (src/mlx/nax_affine.cpp); the token embedding keeps a row-major
8-bit layout for gathers. Every converted tensor is dequantized and compared with the BF16 GGUF before writing.
"""
import argparse, json, os, re, sys
import numpy as np

TYPES = {(4, 128): 314, (6, 64): 315, (4, 64): 317}   # nax affine: (bits, group) -> GGUF type
EMBED8 = 316                                          # affine 8-bit, group 64, row-major
BLOCK = {314: (128, 68), 315: (64, 52), 317: (64, 36), 316: (64, 68)}  # elements, bytes per group of one row

GGUF_OF = {
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
}


def gguf_name(module):
    if module.endswith("lm_head"): return "output.weight"
    if module.endswith("embed_tokens"): return "token_embd.weight"
    m = re.match(r".*layers\.(\d+)\.(.+)$", module)
    if not m or m.group(2) not in GGUF_OF: return None
    return f"blk.{m.group(1)}.{GGUF_OF[m.group(2)]}.weight"


def lane_coords():
    lane = np.arange(32)
    qid = lane >> 2
    return (qid & 4) | ((lane >> 1) & 3), ((qid & 2) | (lane & 1)) * 4


def nax_pack(q, s, b, bits, gs):
    # q [N,K] uint8, s/b [N,K/gs] float16 -> payload bytes (same layout as nax_affine_pack in C++)
    N, K = q.shape
    fm, fn = lane_coords()
    words = np.zeros((N // 32, K // 16, 32, 2), np.uint32)
    hi = np.zeros((N // 32, K // 16, 32), np.uint32)
    q4 = q.reshape(N // 32, 32, K // 16, 16)
    for w in range(2):
        for i in range(8):
            qq = np.transpose(q4[:, w * 16 + fn + (i & 3), :, fm + (i >> 2) * 8], (1, 2, 0)).astype(np.uint32)
            words[..., w] |= (qq & 15) << (4 * i)
            if bits == 6: hi |= (qq >> 4) << (w * 8 + 2 * (i & 3) + 16 * (i >> 2))
    S = np.ascontiguousarray(s.reshape(N // 32, 32, K // gs).transpose(0, 2, 1))
    B = np.ascontiguousarray(b.reshape(N // 32, 32, K // gs).transpose(0, 2, 1))
    parts = [words.tobytes()] + ([hi.tobytes()] if bits == 6 else []) + [S.astype(np.float16).tobytes(), B.astype(np.float16).tobytes()]
    return np.frombuffer(b"".join(parts), np.uint8)


def v_perm(n, num_k, per_k, head):
    # convert_hf_to_gguf _reorder_v_heads: grouped-by-K-head -> tiled order; new[i] = old[perm[i]]
    return np.arange(n).reshape(num_k, per_k, head).transpose(1, 0, 2).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", required=True)
    ap.add_argument("--bf16-gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=os.path.expanduser("~/code/llama.cpp-tq3/gguf-py"))
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf
    import mlx.core as mx
    for t, (blk, ts) in BLOCK.items(): gguf.GGML_QUANT_SIZES[t] = (blk, ts)

    cfg = json.load(open(os.path.join(a.mlx, "config.json")))
    tc = cfg.get("text_config", cfg)
    qcfg = cfg["quantization"]
    nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    hk, hv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    per_k = nv // nk

    idx = json.load(open(os.path.join(a.mlx, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(idx.values()))
    quant = {}   # gguf name -> (codes uint8 [N,K], scales f16, biases f16, bits, group)
    for sh in shards:
        arrs = mx.load(os.path.join(a.mlx, sh))
        for k in [k for k in arrs if k.endswith(".scales")]:
            module = k[: -len(".scales")]
            name = gguf_name(module)
            if name is None: continue
            qc = qcfg.get(module) or qcfg.get(module.replace("model.language_model.", "language_model.model.", 1)) or {}
            bits, gs = qc.get("bits", qcfg["bits"]), qc.get("group_size", qcfg["group_size"])
            w = arrs[module + ".weight"]
            ones = mx.ones(arrs[k].shape, mx.float32)
            codes = np.array(mx.dequantize(w, ones, mx.zeros_like(ones), group_size=gs, bits=bits)).astype(np.uint8)
            s = np.array(arrs[k].astype(mx.float32)).astype(np.float16)
            b = np.array(arrs[module + ".biases"].astype(mx.float32)).astype(np.float16)
            if "linear_attn" in module and nk != nv:
                if module.endswith("in_proj_qkv"):
                    v0 = 2 * nk * hk
                    p = np.concatenate([np.arange(v0), v0 + v_perm(codes.shape[0] - v0, nk, per_k, hv)])
                    codes, s, b = codes[p], s[p], b[p]
                elif module.endswith("in_proj_z"):
                    p = v_perm(codes.shape[0], nk, per_k, hv)
                    codes, s, b = codes[p], s[p], b[p]
                elif module.endswith("out_proj"):
                    if hv % gs: raise SystemExit(f"{module}: head dim {hv} is not a multiple of group {gs}")
                    codes = codes[:, v_perm(codes.shape[1], nk, per_k, hv)]
                    gp = v_perm(s.shape[1], nk, per_k, hv // gs)
                    s, b = s[:, gp], b[:, gp]
            quant[name] = (codes, s, b, bits, gs)
        del arrs
    print("quantized matrices", len(quant), flush=True)

    reader = gguf.GGUFReader(a.bf16_gguf)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(a.out, arch)
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture": continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    writer.add_uint32("tq_mlx.nax_version", 1)

    worst = 0.0
    payloads = []
    for t in reader.tensors:
        if t.name in quant:
            codes, s, b, bits, gs = quant.pop(t.name)
            N, K = codes.shape
            ref = np.asarray(t.data)
            ref = (ref.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if t.tensor_type == gguf.GGMLQuantizationType.BF16 else ref.astype(np.float32)
            ref = ref.reshape(N, K)
            deq = (codes.astype(np.float32).reshape(N, K // gs, gs) * s.astype(np.float32)[:, :, None] + b.astype(np.float32)[:, :, None]).reshape(N, K)
            err = float(np.max(np.abs(deq - ref) / (np.abs(ref) + np.abs(s.astype(np.float32)).max(axis=1, keepdims=True).repeat(K, 1) * 0.5 + 1e-12)))
            worst = max(worst, err)
            if err > 0.02: raise SystemExit(f"{t.name}: converted tensor differs from the BF16 GGUF (rel {err:.3g})")
            if t.name == "token_embd.weight":
                if (bits, gs) != (8, 64): raise SystemExit("token embedding must be 8-bit group 64")
                data, ty = np.frombuffer(codes.tobytes() + s.tobytes() + b.tobytes(), np.uint8), EMBED8
            else:
                ty = TYPES[(bits, gs)]
                data = nax_pack(codes, s, b, bits, gs)
            blk, ts = BLOCK[ty]
            data = data.reshape(N, K // blk * ts)
            writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, ty)
            payloads.append(data)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            payloads.append(t.data)
    if quant: raise SystemExit(f"export matrices with no GGUF tensor: {sorted(quant)[:5]}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in payloads: writer.write_tensor_data(d)
    writer.close()
    print(f"NAX_GGUF_DONE worst gate error {worst:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
