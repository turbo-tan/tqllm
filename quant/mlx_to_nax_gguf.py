#!/usr/bin/env python3
"""MLX affine export -> tq_mlx NAX GGUF.

Names, metadata and the non-quantized tensors come from the BF16 GGUF that convert_hf_to_gguf.py made from the
dequantized export (mlx_dequant_stream.py). Quantized matrices are re-packed from the export's codes, scales and
biases in the lane order of the M5 matrix unit (src/backends/mlx/nax_affine.cpp); the token embedding keeps a row-major
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
    # q [N,K] uint8, s/b [N,K/gs] BF16 bits (uint16) -> payload bytes (same layout as nax_affine_pack in C++)
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
    parts = [words.tobytes()] + ([hi.tobytes()] if bits == 6 else []) + [S.astype(np.uint16).tobytes(), B.astype(np.uint16).tobytes()]
    return np.frombuffer(b"".join(parts), np.uint8)


def bf16_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def v_perm(n, num_k, per_k, head):
    # convert_hf_to_gguf _reorder_v_heads: grouped-by-K-head -> tiled order; new[i] = old[perm[i]]
    return np.arange(n).reshape(num_k, per_k, head).transpose(1, 0, 2).reshape(-1)


def q4_1_pack(codes, s, b, gs):
    # llama.cpp Q4_1 block (32 values): fp16 d, fp16 m, 16 bytes; byte i = q[i] | q[i + 16] << 4; value = d * q + m
    N, K = codes.shape
    sf, bf = bf16_float(s), bf16_float(b)
    d16, m16 = sf.astype(np.float16), bf.astype(np.float16)
    if not (np.all(np.isfinite(d16)) and np.all(np.isfinite(m16))): raise SystemExit("Q4_1: scale or bias overflows FP16")
    inexact = int(np.sum(d16.astype(np.float32) != sf) + np.sum(m16.astype(np.float32) != bf))
    Q4_1_STATS[0] += inexact; Q4_1_STATS[1] += sf.size * 2
    rep = gs // 32
    d = np.repeat(d16, rep, axis=1).reshape(N, K // 32, 1).view(np.uint8).reshape(N, K // 32, 2)
    m = np.repeat(m16, rep, axis=1).reshape(N, K // 32, 1).view(np.uint8).reshape(N, K // 32, 2)
    q = codes.reshape(N, K // 32, 32).astype(np.uint8)
    qs = (q[:, :, :16] | (q[:, :, 16:] << 4)).astype(np.uint8)
    return np.concatenate([d, m, qs], axis=2).reshape(N, K // 32 * 20)


Q4_1_STATS = [0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", required=True)
    ap.add_argument("--bf16-gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=os.path.expanduser("~/code/llama.cpp-tq3/gguf-py"))
    ap.add_argument("--draft-head", action="store_true", help="also write output_draft.weight (4-bit g64) for the drafter")
    ap.add_argument("--q4-1", action="store_true", help="write 4-bit tensors as llama.cpp Q4_1 (same codes; each group-64 scale/bias as FP16 in two 32-blocks) instead of NAX")
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

    reader = gguf.GGUFReader(a.bf16_gguf)
    by_name = {t.name: t for t in reader.tensors}
    worst = 0.0

    def convert(name, codes, s, b, bits, gs):
        # gate against the BF16 GGUF, then pack; only the packed payload is kept
        nonlocal worst
        t = by_name[name]
        N, K = codes.shape
        raw = np.asarray(t.data)
        bf16 = t.tensor_type == gguf.GGMLQuantizationType.BF16
        raw = raw.view(np.uint16).reshape(N, K) if bf16 else raw.reshape(N, K)
        sf, bf = bf16_float(s), bf16_float(b)
        err = 0.0
        for r0 in range(0, N, 4096):   # row blocks keep the FP32 temporaries small
            r1 = min(N, r0 + 4096)
            ref = (raw[r0:r1].astype(np.uint32) << 16).view(np.float32) if bf16 else raw[r0:r1].astype(np.float32)
            q = codes[r0:r1].astype(np.float32).reshape(r1 - r0, K // gs, gs)
            deq = (q * sf[r0:r1, :, None] + bf[r0:r1, :, None]).reshape(r1 - r0, K)
            # the BF16 GGUF was dequantized in BF16, so it carries one BF16 rounding of code * scale and of the sum
            mag = (q * np.abs(sf[r0:r1, :, None]) + np.abs(bf[r0:r1, :, None])).reshape(r1 - r0, K)
            err = max(err, float(np.max(np.abs(deq - ref) / (mag * 2.0 ** -7 + 1e-9))))
        worst = max(worst, err)
        if err > 1.0: raise SystemExit(f"{name}: converted tensor differs from the BF16 GGUF ({err:.3g} BF16 steps)")
        if a.q4_1:
            if bits != 4 or gs % 32: raise SystemExit(f"{name}: Q4_1 output needs 4-bit codes with groups of 32k ({bits}, {gs})")
            return q4_1_pack(codes, s, b, gs), int(gguf.GGMLQuantizationType.Q4_1)
        if name == "token_embd.weight":
            # lower-bit group-64 embeddings (mlx-community 4-bit) keep their codes, scales and biases: exact in the 8-bit layout
            if gs != 64 or bits not in (4, 6, 8): raise SystemExit("token embedding must be 4/6/8-bit group 64")
            data, ty = np.frombuffer(codes.tobytes() + s.tobytes() + b.tobytes(), np.uint8), EMBED8
        else:
            ty = TYPES[(bits, gs)]
            data = nax_pack(codes, s, b, bits, gs)
        blk, ts = BLOCK[ty]
        return data.reshape(N, K // blk * ts), ty

    idx = json.load(open(os.path.join(a.mlx, "model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(idx.values()))
    quant = {}   # gguf name -> (payload uint8, GGUF type)
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
            s = np.array(arrs[k].astype(mx.bfloat16).view(mx.uint16))
            b = np.array(arrs[module + ".biases"].astype(mx.bfloat16).view(mx.uint16))
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
            if name not in by_name: raise SystemExit(f"{name}: not in the BF16 GGUF")
            quant[name] = convert(name, codes, s, b, bits, gs)
            if name == "output.weight" and a.draft_head:
                # drafter-only copy of the head at 4-bit group 64 (MLX rounding of the dequantized head); verify keeps 6-bit
                N, K = codes.shape
                deq = (codes.astype(np.float32).reshape(N, K // gs, gs) * bf16_float(s)[:, :, None] + bf16_float(b)[:, :, None]).reshape(N, K)
                wq, s4, b4 = mx.quantize(mx.array(deq).astype(mx.bfloat16), group_size=64, bits=4)
                del deq
                ones = mx.ones(s4.shape, mx.float32)
                c4 = np.array(mx.dequantize(wq, ones, mx.zeros_like(ones), group_size=64, bits=4)).astype(np.uint8)
                data = nax_pack(c4, np.array(s4.astype(mx.bfloat16).view(mx.uint16)), np.array(b4.astype(mx.bfloat16).view(mx.uint16)), 4, 64)
                blk, ts = BLOCK[317]
                quant["output_draft.weight"] = (data.reshape(N, K // blk * ts), 317)
        del arrs
        print(sh, "converted", len(quant), "worst", round(worst, 3), flush=True)

    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(a.out, arch)
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture": continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    if not a.q4_1: writer.add_uint32("tq_mlx.nax_version", 1)
    payloads = []
    for t in reader.tensors:
        if t.name in quant:
            data, ty = quant.pop(t.name)
            writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, ty)
            payloads.append(data)
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            payloads.append(t.data)
        if t.name == "output.weight" and "output_draft.weight" in quant:
            data, ty = quant.pop("output_draft.weight")
            writer.add_tensor_info("output_draft.weight", data.shape, data.dtype, data.nbytes, ty)
            payloads.append(data)
    if quant: raise SystemExit(f"export matrices with no GGUF tensor: {sorted(quant)[:5]}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in payloads: writer.write_tensor_data(d)
    writer.close()
    if a.q4_1: print(f"Q4_1 scales/biases not exact in FP16: {Q4_1_STATS[0]} of {Q4_1_STATS[1]}", flush=True)
    print(f"NAX_GGUF_DONE worst gate error {worst:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
