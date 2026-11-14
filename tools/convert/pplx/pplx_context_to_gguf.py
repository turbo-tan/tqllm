#!/usr/bin/env python3
"""Convert perplexity-ai/pplx-embed-v2-context-* (PplxContextualModel over Qwen3_5Model, MIT) to a qwen35 GGUF that
tqserve serves on /v1/embeddings.

The text tower is written as the qwen35 engine reads Qwen3.5 (the rules of src/core/hf/arch_qwen35.cpp: every RMSNorm
stored as 1 + w except the GDN output norm, A_log -> -exp(A_log), conv1d [ch][1][k] -> [ch][k]; GDN value heads stay in
the Hugging Face order: value head j reads key head j // (value heads / key heads), as repeat_interleave does),
plus the contextual projection (FP32) and the pooling metadata (pplx.*). The vision tower is not written.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

DTYPES = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2)}


class Safetensors:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def has(self, name):
        return name in self.header

    def raw(self, name):
        """(array as stored, dtype tag): BF16 as uint16 bit patterns"""
        e = self.header[name]
        np_type, size = DTYPES[e["dtype"]]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            data = np.frombuffer(f.read(b - a), dtype=np_type)
        return data.reshape(e["shape"]), e["dtype"]

    def f32(self, name):
        a, tag = self.raw(name)
        if tag == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)


class Checkpoint:
    """model.safetensors, or the shards of model.safetensors.index.json, plus contextual_head.safetensors (the 9B)"""
    def __init__(self, snap):
        snap = Path(snap)
        files = [snap / "model.safetensors"] if (snap / "model.safetensors").exists() else []
        index = snap / "model.safetensors.index.json"
        if index.exists():
            files = sorted({snap / f for f in json.loads(index.read_text())["weight_map"].values()})
        if (snap / "contextual_head.safetensors").exists():
            files.append(snap / "contextual_head.safetensors")
        if not files:
            raise ValueError("no safetensors in " + str(snap))
        self.where = {}
        for f in files:
            st = Safetensors(f)
            for name in st.header:
                self.where[name] = st

    def has(self, name):
        return name in self.where

    def raw(self, name):
        return self.where[name].raw(name)

    def shape(self, name):
        return self.where[name].header[name]["shape"]

    def f32(self, name):
        return self.where[name].f32(name)


def tokenizer_metadata(snap, vocab):
    t = json.loads((snap / "tokenizer.json").read_text())
    c = json.loads((snap / "tokenizer_config.json").read_text())
    tokens, types = [""] * vocab, [5] * vocab                       # 5 unused
    for piece, i in t["model"]["vocab"].items():
        if i < vocab:
            tokens[i], types[i] = piece, 1                          # normal
    for a in t["added_tokens"]:
        i = a["id"]
        if i < vocab:
            tokens[i], types[i] = a["content"], 3 if a.get("special") else 4  # control : user-defined
    for i in range(vocab):
        if not tokens[i]:
            tokens[i] = f"[PAD{i}]"
    merges = [m if isinstance(m, str) else m[0] + " " + m[1] for m in t["model"]["merges"]]
    ids = {s: i for i, s in enumerate(tokens)}
    special = {}
    for key in ("eos_token", "pad_token"):
        v = c.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        if v in ids:
            special[key] = ids[v]
    return tokens, types, merges, special, ids


def full_attention_interval(layer_types):
    """the period of the full-attention layers (layer l is full when (l + 1) % n == 0, as the qwen35 engine reads it).
    pplx-embed-v2-context-0.6b says full_attention_interval 4 in its config but its layer_types alternate (n = 2):
    layer_types is what the model runs."""
    full = [t == "full_attention" for t in layer_types]
    for n in range(1, len(full) + 1):
        if full == [(l + 1) % n == 0 for l in range(len(full))]:
            return n
    raise ValueError("layer_types is not periodic: " + str(layer_types))


# TQ4 (GGUF 300, tqLLM): per 128 weights of a row, an FP16 scale in engine units then 64 bytes of 4-bit codes (low nibble =
# even column); w_rot = L[code] * scale with L the int8 Turbo4 levels, w = w_rot R per 128, R = H diag(s1) / sqrt(128).
TQ4 = 300
TQ4_LEVELS = np.array([6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127], dtype=np.float32)
TQ4_F = 127.0 / 2.7326  # engine level = F x Lloyd-Max level
_S1 = np.array([1 - 2 * ((0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9)[i // 32] >> (i % 32) & 1) for i in range(128)], dtype=np.float32)
_H = np.ones((1, 1), dtype=np.float32)
while _H.shape[0] < 128:
    _H = np.block([[_H, _H], [_H, -_H]])
TQ4_R = _H * _S1[None, :] / np.sqrt(128.0)


def tq4_pack(codes, scale_eng):
    """codes uint8 [rows][cols] (0..15), engine scales [rows][cols / 128] -> bytes [rows][cols / 128 * 66]"""
    rows, cols = codes.shape
    c = codes.reshape(rows, cols // 2, 2)
    nib = (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8).reshape(rows, cols // 128, 64)
    sc = scale_eng.astype(np.float16).view(np.uint8).reshape(rows, cols // 128, 2)
    return np.concatenate([sc, nib], axis=-1).reshape(rows, -1)


def tq4_rtn(w, chunk=1024):
    """round to nearest on the rotated rows, per-group scale searched over 0.75 .. 1.25 x RMS -> codes, engine scales"""
    rows, cols = w.shape
    lv = TQ4_LEVELS / TQ4_F
    codes = np.empty((rows, cols), dtype=np.uint8)
    scales = np.empty((rows, cols // 128), dtype=np.float32)
    for r0 in range(0, rows, chunk):
        wr = (w[r0:r0 + chunk].reshape(-1, cols // 128, 128) @ TQ4_R.T)
        rms = np.maximum(np.sqrt((wr ** 2).mean(-1, keepdims=True)), 1e-8)
        best_c = best_s = best_e = None
        for fct in np.linspace(0.75, 1.25, 11, dtype=np.float32):
            sc = rms * fct
            cc = np.abs((wr / sc)[..., None] - lv).argmin(-1)
            e = ((lv[cc] * sc - wr) ** 2).sum(-1, keepdims=True)
            if best_e is None:
                best_c, best_s, best_e = cc, sc, e
            else:
                m = e < best_e
                best_c, best_s, best_e = np.where(m, cc, best_c), np.where(m, sc, best_s), np.where(m, e, best_e)
        codes[r0:r0 + chunk] = best_c.reshape(-1, cols).astype(np.uint8)
        scales[r0:r0 + chunk] = best_s[..., 0] / TQ4_F
    return codes, scales


def convert(snap, output, outtype="auto", tq4_file=None, q8_keep=()):
    """outtype auto: matrices as stored (BF16 or F32); bf16: F32 layer matrices rounded to BF16 (nearest even), the token
    table kept F32 (the fast path gathers it in FP32; a BF16 table failed the fp32 tail rule, p1 0.999544 vs 0.999935 on the 3090); q8_0: every matrix
    (the token table included) as GGUF Q8_0 (32 int8 + an FP16 scale per block); norms, conv, gates and the projection stay F32.
    tq4: the layer matrices as TQ4 (codes from tq4_file, a tq4_gptq.py safetensors, else round to nearest); in_proj_a / b
    BF16, the token table Q8_0. q8_keep: GGUF name substrings (e.g. "ffn_down", "blk.0.") of layer matrices kept Q8_0 in a
    tq4 file (mixed precision)."""
    snap = Path(snap)
    config = json.loads((snap / "config.json").read_text())
    if config.get("model_type") != "pplx_contextual_qwen3_5":
        raise ValueError("not a pplx contextual Qwen3.5 checkpoint: " + str(config.get("model_type")))
    tc = config["text_config"]
    L, H = int(tc["num_hidden_layers"]), int(tc["hidden_size"])
    kh, vh = int(tc["linear_num_key_heads"]), int(tc["linear_num_value_heads"])
    kd, vd = int(tc["linear_key_head_dim"]), int(tc["linear_value_head_dim"])
    if vh % kh:
        raise ValueError("GDN value heads must be a multiple of the key heads")
    rope = tc["rope_parameters"]
    vocab = int(tc["vocab_size"])
    st = Checkpoint(snap)
    import gguf  # after the checks: a rejected checkpoint needs no gguf-py
    gguf.constants.GGML_QUANT_SIZES.setdefault(TQ4, (128, 66))  # tqLLM-only type: 128 weights in 66 bytes
    P = "language_model."
    w = gguf.GGUFWriter(str(output), "qwen35", use_temp_file=True)
    A = "qwen35."
    u32 = lambda k, v: w.add_uint32(A + k, int(v))
    w.add_string("general.name", "pplx-embed-v2-context")
    w.add_string("general.license", "mit")
    u32("block_count", L); u32("nextn_predict_layers", 0)
    u32("context_length", config.get("document_length", tc["max_position_embeddings"]))
    u32("embedding_length", H); u32("feed_forward_length", tc["intermediate_size"])
    u32("attention.head_count", tc["num_attention_heads"]); u32("attention.head_count_kv", tc["num_key_value_heads"])
    u32("attention.key_length", tc["head_dim"]); u32("attention.value_length", tc["head_dim"])
    w.add_float32(A + "attention.layer_norm_rms_epsilon", float(tc["rms_norm_eps"]))
    u32("ssm.conv_kernel", tc["linear_conv_kernel_dim"]); u32("ssm.state_size", vd); u32("ssm.group_count", kh)
    u32("ssm.time_step_rank", vh); u32("ssm.inner_size", vh * vd)
    u32("full_attention_interval", full_attention_interval(tc["layer_types"]))
    w.add_float32(A + "rope.freq_base", float(rope["rope_theta"]))
    u32("rope.dimension_count", int(tc["head_dim"] * rope["partial_rotary_factor"]))
    w.add_array(A + "rope.dimension_sections", [int(x) for x in rope["mrope_section"]] + [0] * (4 - len(rope["mrope_section"])))
    # contextual pooling: what tqserve's /v1/embeddings needs to build the input and pool it (modeling_pplx_contextual.py)
    w.add_uint32("pplx.embedding_length", int(config["embedding_dim"]))
    w.add_string("pplx.query_prefix", config["query_prefix"])        # prepended as its token id
    w.add_string("pplx.document_prefix", config["document_prefix"])  # tokenized as text (special tokens are split)
    w.add_string("pplx.boundary_marker", config["boundary_marker"])
    w.add_bool("pplx.attention_causal", bool(tc.get("is_causal", True)))
    tokens, types, merges, special, ids = tokenizer_metadata(snap, vocab)
    if config["query_prefix"] not in ids:
        raise ValueError("query prefix is not one token")
    w.add_uint32("pplx.query_prefix_token_id", ids[config["query_prefix"]])
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
    if "eos_token" in special: w.add_eos_token_id(special["eos_token"])
    if "pad_token" in special: w.add_pad_token_id(special["pad_token"])

    BF16, F32 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32
    tq4src = None
    if outtype == "tq4" and tq4_file:
        from safetensors import safe_open
        tq4src = safe_open(str(tq4_file), "np")
        tq4keys = set(tq4src.keys())

    def mat(dest, name):
        base = name[:-len(".weight")] if name.endswith(".weight") else name
        keep8 = outtype == "tq4" and any(k in dest for k in q8_keep)
        if outtype == "tq4" and not keep8 and dest.startswith("blk.") and not base.endswith(("in_proj_a", "in_proj_b")) and st.shape(name)[-1] % 128 == 0:
            key = next((k for k in (base, "model." + base) if tq4src is not None and k + ".tq4_codes" in tq4keys), None)
            if key:
                cb = tq4src.get_tensor(key + ".tq4_codes")
                codes = np.stack([cb & 15, cb >> 4], -1).reshape(cb.shape[0], -1).astype(np.uint8)
                scales = tq4src.get_tensor(key + ".tq4_scale").astype(np.float32) / TQ4_F
                want = tuple(st.shape(name))
                if codes.shape != want or scales.shape != (want[0], want[1] // 128):
                    raise ValueError(f"TQ4 codes for {base} in {tq4_file} do not match the checkpoint: codes {codes.shape}, "
                                     f"scales {scales.shape}; the matrix is {want} (scales {(want[0], want[1] // 128)})")
            elif tq4src is not None:
                raise ValueError("no TQ4 codes for " + base + " in " + str(tq4_file))
            else:
                codes, scales = tq4_rtn(st.f32(name))
            data = tq4_pack(codes, scales)
            rows, cols = codes.shape
            w.add_tensor(dest, data, raw_dtype=TQ4)  # logical shape from the bytes (gguf-py knows TQ4's block via convert())
            return
        if outtype in ("q8_0", "tq4") and (outtype == "q8_0" or keep8 or dest == "token_embd.weight") and st.shape(name)[-1] % 32 == 0:  # Q8_0 blocks are 32 along a row
            from gguf import quants
            Q8 = gguf.GGMLQuantizationType.Q8_0
            w.add_tensor(dest, quants.quantize(np.ascontiguousarray(st.f32(name), dtype=np.float32), Q8), raw_dtype=Q8)
            return
        a, tag = st.raw(name)
        if tag == "BF16":
            w.add_tensor(dest, a, raw_dtype=BF16)
        elif tag == "F32" and outtype == "bf16" and dest != "token_embd.weight":  # the table stays F32: fast-path inputs in FP32
            u = a.astype(np.float32).view(np.uint32)
            w.add_tensor(dest, ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16), raw_dtype=BF16)
        elif tag == "F32":
            w.add_tensor(dest, np.ascontiguousarray(a, dtype=np.float32), raw_dtype=F32)
        else:
            raise ValueError(f"{name}: {tag} weights are not handled")
    def norm1(dest, name):  # zero-centred RMSNorm: stored as 1 + w
        w.add_tensor(dest, 1.0 + st.f32(name), raw_dtype=F32)
    def f32(dest, array):
        w.add_tensor(dest, np.ascontiguousarray(array, dtype=np.float32), raw_dtype=F32)
    mat("token_embd.weight", P + "embed_tokens.weight")
    norm1("output_norm.weight", P + "norm.weight")
    full = [t == "full_attention" for t in tc["layer_types"]]
    for l in range(L):
        s, d = f"{P}layers.{l}.", f"blk.{l}."
        norm1(d + "attn_norm.weight", s + "input_layernorm.weight")
        norm1(d + "post_attention_norm.weight", s + "post_attention_layernorm.weight")
        mat(d + "ffn_gate.weight", s + "mlp.gate_proj.weight")
        mat(d + "ffn_up.weight", s + "mlp.up_proj.weight")
        mat(d + "ffn_down.weight", s + "mlp.down_proj.weight")
        if full[l]:
            for a, b in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")):
                mat(d + a + ".weight", s + "self_attn." + b + ".weight")
            norm1(d + "attn_q_norm.weight", s + "self_attn.q_norm.weight")
            norm1(d + "attn_k_norm.weight", s + "self_attn.k_norm.weight")
            continue
        la = s + "linear_attn."
        mat(d + "attn_qkv.weight", la + "in_proj_qkv.weight")
        mat(d + "attn_gate.weight", la + "in_proj_z.weight")
        mat(d + "ssm_alpha.weight", la + "in_proj_a.weight")
        mat(d + "ssm_beta.weight", la + "in_proj_b.weight")
        mat(d + "ssm_out.weight", la + "out_proj.weight")
        f32(d + "ssm_a", -np.exp(st.f32(la + "A_log")))
        f32(d + "ssm_dt.bias", st.f32(la + "dt_bias"))
        conv = st.f32(la + "conv1d.weight")
        f32(d + "ssm_conv1d.weight", conv.reshape(conv.shape[0], conv.shape[-1]))
        f32(d + "ssm_norm.weight", st.f32(la + "norm.weight"))       # the GDN output norm keeps w
    f32("contextual.projection.weight", st.f32("contextual_projection.weight"))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path, help="the Hugging Face snapshot folder (config.json, model.safetensors, tokenizer)")
    ap.add_argument("output", type=Path)
    ap.add_argument("--outtype", choices=["auto", "bf16", "q8_0", "tq4"], default="auto",
                    help="auto: matrices as stored; bf16: round F32 layer matrices to BF16 (token table stays F32); q8_0: GGUF Q8_0 matrices; tq4: TQ4 layer matrices")
    ap.add_argument("--tq4", type=Path, default=None, help="tq4_gptq.py safetensors with the layer matrices' codes and scales")
    ap.add_argument("--q8-keep", default="", help="comma-separated GGUF name substrings of layer matrices kept Q8_0 in a tq4 file")
    a = ap.parse_args()
    convert(a.snapshot, a.output, a.outtype, a.tq4, tuple(x for x in a.q8_keep.split(",") if x))
