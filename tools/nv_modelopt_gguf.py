#!/usr/bin/env python3
"""A modelopt MIXED_PRECISION checkpoint (nvidia/Qwen3.6-35B-A3B-NVFP4) as one GGUF with every tensor as NVIDIA stores it:
  NVFP4 (W4A16, group 16: routed experts, shared expert, lm_head) -> NVFP4G (323), NVIDIA's code / scale / global bytes
  FP8 (E4M3, one F32 weight_scale per tensor: attention and Gated DeltaNet projections) -> FP8T (324), NVIDIA's bytes
  BF16 (everything else)                             -> the llama.cpp converter's bytes of the same values (template)

FP8T block (128 weights, 132 bytes): the matrix's F32 weight_scale, then 128 E4M3 bytes (w = e4m3 * scale). The scale is
repeated per block so a row range of the tensor (and a row-concatenation with another FP8T tensor) keeps its own scale.

The tensor list, names, shapes, order and metadata come from TEMPLATE, the llama.cpp BF16 conversion of the original
Qwen/Qwen3.6-35B-A3B. Every BF16 tensor of NVIDIA's checkpoint is pushed through the llama.cpp transforms (norm + 1,
-exp(A_log), conv squeeze, the V-head reorder of the Gated DeltaNet tensors) and must equal the template's bytes exactly;
the template's bytes are then written. The quantized tensors get the same reorder as pure byte permutations (rows of
in_proj_qkv V / in_proj_z, columns of out_proj), so they stay NVIDIA's bytes.

  nv_modelopt_gguf.py build  TEMPLATE.gguf NVDIR OUT.gguf    write OUT (BF16 tensors checked against NVDIR)
  nv_modelopt_gguf.py verify TEMPLATE.gguf NVDIR OUT.gguf    every quantized tensor of OUT back to NVIDIA's bytes (exact),
                                                             dequantized OUT vs the template BF16 (relative error)
"""
import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn_nvfp4_gguf as nv  # noqa: E402  (GGUF header reading, NVFP4G blocks, E4M3 table)

NVFP4G, FP8T = 323, 324
nv._BLOCKS[FP8T] = (128, 132)
BF16, F32 = 30, 0


# --- safetensors folder ----------------------------------------------------------------------------------------------
class Folder:
    def __init__(self, d):
        self.d = d
        with open(os.path.join(d, "model.safetensors.index.json")) as f:
            self.wm = json.load(f)["weight_map"]
        self.files = {}
        hq = os.path.join(d, "hf_quant_config.json")
        self.qcfg = {}
        if os.path.exists(hq):
            with open(hq) as f:
                self.qcfg = json.load(f)["quantization"]
        with open(os.path.join(d, "config.json")) as f:
            self.cfg = json.load(f)

    def _file(self, fn):
        if fn not in self.files:
            p = os.path.join(self.d, fn)
            hdr, data0 = nv.safetensors_index(p)
            self.files[fn] = (hdr, data0, np.memmap(p, np.uint8, mode="r"))
        return self.files[fn]

    def has(self, k):
        return k in self.wm

    def raw(self, k):
        """(uint8 bytes, dtype, shape)"""
        hdr, data0, mm = self._file(self.wm[k])
        i = hdr[k]
        a, b = i["data_offsets"]
        return np.asarray(mm[data0 + a:data0 + b]), i["dtype"], i["shape"]

    def f32(self, k):
        """BF16 / F32 tensor as float32 (exact)."""
        b, dt, shape = self.raw(k)
        if dt == "BF16":
            return (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return b.view(np.float32).reshape(shape)
        raise ValueError(f"{k}: dtype {dt}")


def bf16_round(x):
    """float32 -> nearest bfloat16 (ties to even), returned as float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def bf16_bytes(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


# --- llama.cpp's Qwen3.5 / Qwen3-Next transforms (conversion/qwen.py) ----------------------------------------------------
class Model:
    def __init__(self, cfg):
        t = cfg.get("text_config", cfg)
        self.nk, self.nv_ = t["linear_num_key_heads"], t["linear_num_value_heads"]
        self.dk, self.dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.layers = t["num_hidden_layers"]

    def vperm(self, head_dim):
        """row order of llama.cpp's V-head reorder (grouped by K head -> tiled) for num_v_heads * head_dim rows"""
        r = self.nv_ // self.nk
        return np.arange(self.nv_ * head_dim).reshape(self.nk, r, head_dim).transpose(1, 0, 2).reshape(-1)

    def qkv_perm(self):
        qk = 2 * self.nk * self.dk
        return np.concatenate([np.arange(qk), qk + self.vperm(self.dv)])


P = "model.language_model."


def mapping(m, name):
    """GGUF tensor name -> (kind, source, transform). kind: bf16 | fp8 | nvfp4 | nvfp4_exps | mtp_exps"""
    if name == "token_embd.weight":
        return "bf16", P + "embed_tokens.weight", "id"
    if name == "output_norm.weight":
        return "bf16", P + "norm.weight", "plus1"
    if name == "output.weight":
        return "nvfp4", "lm_head", "id"
    mt = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not mt:
        raise KeyError(name)
    L, rest = int(mt.group(1)), mt.group(2)
    if L >= m.layers:  # the MTP block (BF16 in NVIDIA's file: excluded from quantization)
        q = "mtp.layers.0."
        tbl = {"nextn.eh_proj.weight": ("mtp.fc.weight", "id"), "nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", "plus1"),
               "nextn.hnorm.weight": ("mtp.pre_fc_norm_hidden.weight", "plus1"), "nextn.shared_head_norm.weight": ("mtp.norm.weight", "plus1"),
               "attn_norm.weight": (q + "input_layernorm.weight", "plus1"), "post_attention_norm.weight": (q + "post_attention_layernorm.weight", "plus1"),
               "attn_q.weight": (q + "self_attn.q_proj.weight", "id"), "attn_k.weight": (q + "self_attn.k_proj.weight", "id"),
               "attn_v.weight": (q + "self_attn.v_proj.weight", "id"), "attn_output.weight": (q + "self_attn.o_proj.weight", "id"),
               "attn_q_norm.weight": (q + "self_attn.q_norm.weight", "plus1"), "attn_k_norm.weight": (q + "self_attn.k_norm.weight", "plus1"),
               "ffn_gate_inp.weight": (q + "mlp.gate.weight", "id"), "ffn_gate_inp_shexp.weight": (q + "mlp.shared_expert_gate.weight", "id"),
               "ffn_gate_shexp.weight": (q + "mlp.shared_expert.gate_proj.weight", "id"),
               "ffn_up_shexp.weight": (q + "mlp.shared_expert.up_proj.weight", "id"),
               "ffn_down_shexp.weight": (q + "mlp.shared_expert.down_proj.weight", "id"),
               "ffn_gate_exps.weight": (q + "mlp.experts.gate_up_proj", "gate_half"),
               "ffn_up_exps.weight": (q + "mlp.experts.gate_up_proj", "up_half"),
               "ffn_down_exps.weight": (q + "mlp.experts.down_proj", "id")}
        s, t = tbl[rest]
        return "bf16", s, t
    p = f"{P}layers.{L}."
    la, sa, mlp = p + "linear_attn.", p + "self_attn.", p + "mlp."
    fp8 = {"attn_qkv.weight": (la + "in_proj_qkv", "qkv_rows"), "attn_gate.weight": (la + "in_proj_z", "v_rows"),
           "ssm_out.weight": (la + "out_proj", "v_cols"), "attn_q.weight": (sa + "q_proj", "id"),
           "attn_k.weight": (sa + "k_proj", "id"), "attn_v.weight": (sa + "v_proj", "id"),
           "attn_output.weight": (sa + "o_proj", "id")}
    if rest in fp8:
        return ("fp8",) + fp8[rest]
    if rest in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
        return "nvfp4", mlp + "shared_expert." + rest.split("_")[1] + "_proj", "id"
    me = re.match(r"^ffn_(gate|up|down)_exps\.weight$", rest)
    if me:
        return "nvfp4_exps", mlp + "experts.{}." + me.group(1) + "_proj", "id"
    bf = {"attn_norm.weight": (p + "input_layernorm.weight", "plus1"), "post_attention_norm.weight": (p + "post_attention_layernorm.weight", "plus1"),
          "ssm_a": (la + "A_log", "neg_exp_v1"), "ssm_dt.bias": (la + "dt_bias", "v1"), "ssm_conv1d.weight": (la + "conv1d.weight", "conv"),
          "ssm_alpha.weight": (la + "in_proj_a.weight", "v1"), "ssm_beta.weight": (la + "in_proj_b.weight", "v1"),
          "ssm_norm.weight": (la + "norm.weight", "id"), "attn_q_norm.weight": (sa + "q_norm.weight", "plus1"),
          "attn_k_norm.weight": (sa + "k_norm.weight", "plus1"), "ffn_gate_inp.weight": (mlp + "gate.weight", "id"),
          "ffn_gate_inp_shexp.weight": (mlp + "shared_expert_gate.weight", "id")}
    s, t = bf[rest]
    return "bf16", s, t


def bf16_values(m, src, k, tr):
    """NVIDIA's BF16 tensor through the llama.cpp transform, as candidate float32 arrays (the converter runs the arithmetic
    in BF16 or F32 depending on the op: both roundings are offered, the template decides which one it holds)."""
    x = src.f32(k)
    if tr == "id":
        return [x]
    if tr == "plus1":
        return [bf16_round(x + np.float32(1)), x + np.float32(1)]
    if tr == "v1":
        return [x[m.vperm(1)] if x.ndim == 1 else x[m.vperm(1), ...]]
    if tr == "neg_exp_v1":  # llama.cpp: -torch.exp on the F32 values (numpy's exp differs from torch's in the last ulp)
        y = x[m.vperm(1)]
        cands = [-np.exp(y), bf16_round(-np.exp(y))]
        try:
            import torch
            cands.insert(0, (-torch.exp(torch.from_numpy(np.ascontiguousarray(y)))).numpy())
        except ImportError:
            pass
        return cands
    if tr == "conv":
        y = x.reshape(x.shape[0], -1)
        qk = 2 * m.nk * m.dk
        return [np.concatenate([y[:qk], y[qk:][m.vperm(m.dv)]], 0)]
    if tr in ("gate_half", "up_half"):  # [E, 2 * ff, H] -> gate = rows [0, ff), up = [ff, 2 ff)
        ff = x.shape[1] // 2
        return [x[:, :ff] if tr == "gate_half" else x[:, ff:]]
    raise ValueError(tr)


def as_type(x, typ):
    if typ == BF16:
        return bf16_bytes(x).tobytes()
    if typ == F32:
        return np.ascontiguousarray(x, np.float32).tobytes()
    raise ValueError(f"template type {typ}")


# --- quantized tensors -------------------------------------------------------------------------------------------------
def fp8t_blocks(codes, scale):
    """codes uint8 [rows, cols] (E4M3), F32 scale -> uint8 [rows * cols / 128 * 132]"""
    rows, cols = codes.shape
    if cols % 128:
        raise ValueError(f"FP8T: {cols} columns, need a multiple of 128")
    s = np.float32(scale)
    if not np.isfinite(s):
        raise ValueError("FP8T: scale not finite")
    out = np.empty((rows, cols // 128, 132), np.uint8)
    out[:, :, :4] = np.frombuffer(s.tobytes(), np.uint8)
    out[:, :, 4:] = codes.reshape(rows, cols // 128, 128)
    return out.reshape(-1)


def fp8_source(m, src, k, tr):
    """NVIDIA FP8 tensor k (without .weight) -> (codes in GGUF row/column order, scale)"""
    c, dt, shape = src.raw(k + ".weight")
    s, sdt, sshape = src.raw(k + ".weight_scale")
    if dt != "F8_E4M3" or sdt != "F32" or int(np.prod(sshape or [1])) != 1:
        raise ValueError(f"{k}: FP8 per-tensor expected, got {dt} / {sdt} {sshape}")
    codes = c.reshape(shape)
    if tr == "qkv_rows":
        codes = codes[m.qkv_perm()]
    elif tr == "v_rows":
        codes = codes[m.vperm(m.dv)]
    elif tr == "v_cols":
        codes = codes[:, m.vperm(m.dv)]
    return np.ascontiguousarray(codes), s.view(np.float32)[0]


def nvfp4_source(src, k):
    codes, ci, cshape = src.raw(k + ".weight")
    sc, si, sshape = src.raw(k + ".weight_scale")
    g, gi, _ = src.raw(k + ".weight_scale_2")
    if ci != "U8" or si != "F8_E4M3" or gi != "F32":
        raise ValueError(f"{k}: NVFP4 dtypes {ci} {si} {gi}")
    rows, half = cshape
    return codes.reshape(rows, half), sc.reshape(rows, half // 8), g.view(np.float32)[0]


def deq_fp8(codes, scale):
    return nv.E4M3[codes] * np.float32(scale)


E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def deq_nvfp4(codes, scales, g):
    rows = codes.shape[0]
    w = np.empty((rows, codes.shape[1] * 2), np.float32)
    w[:, 0::2] = E2M1[codes & 15]
    w[:, 1::2] = E2M1[codes >> 4]
    return (w.reshape(rows, -1, 16) * (nv.E4M3[scales] * np.float32(g))[:, :, None]).reshape(rows, -1)


# --- build ---------------------------------------------------------------------------------------------------------------
def plan_types(m, tensors):
    out = []
    for name, dims, typ, off in tensors:
        kind = mapping(m, name)[0]
        nt = {"fp8": FP8T, "nvfp4": NVFP4G, "nvfp4_exps": NVFP4G}.get(kind, typ)
        out.append((name, dims, typ, off, nt))
    return out


def build(template, nvdir, out, log=print):
    src = Folder(nvdir)
    m = Model(src.cfg)
    version, n_kv, kv_bytes, tensors, align, data_start = nv.read_gguf_header(template)
    plan = plan_types(m, tensors)
    with open(out, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", version, len(plan), n_kv + 1))
        f.write(kv_bytes)
        nv._w_str(f, "tqllm.weights.source")
        f.write(struct.pack("<I", 8))
        nv._w_str(f, "nvidia modelopt checkpoint as stored (NVFP4 -> NVFP4G, FP8 -> FP8T, BF16 as converted by llama.cpp): "
                  + os.path.basename(os.path.abspath(nvdir)))
        offs, cur = [], 0
        for name, dims, _, _, nt in plan:
            cur = (cur + align - 1) // align * align
            offs.append(cur)
            cur += nv.tensor_bytes(dims, nt)
        for (name, dims, _, _, nt), o in zip(plan, offs):
            nv._w_str(f, name)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack(f"<{len(dims)}Q", *dims))
            f.write(struct.pack("<IQ", nt, o))
        base = (f.tell() + align - 1) // align * align
        f.truncate(base + cur)
        tm = np.memmap(template, np.uint8, mode="r")
        stats = {"bf16_equal": 0, "fp8": 0, "nvfp4": 0}
        for (name, dims, typ, soff, nt), o in zip(plan, offs):
            kind, k, tr = mapping(m, name)
            f.seek(base + o)
            if kind == "bf16":
                nb = nv.tensor_bytes(dims, typ)
                tb = np.asarray(tm[data_start + soff:data_start + soff + nb])
                cands = bf16_values(m, src, k, tr)
                hit = next((i for i, c in enumerate(cands) if c.size == int(np.prod(dims)) and as_type(c, typ) == tb.tobytes()), None)
                if hit is None:
                    c = cands[0].reshape(-1)
                    t = np.frombuffer(tb.tobytes(), np.float32 if typ == F32 else np.uint16)
                    t = t if typ == F32 else (t.astype(np.uint32) << 16).view(np.float32)
                    d = np.abs(c - t) if c.size == t.size else None
                    raise ValueError(f"{name}: NVIDIA {k} ({tr}) differs from the template: "
                                     + (f"{int((d > 0).sum())} of {d.size} values, max {d.max():.3g}" if d is not None else "size"))
                f.write(tb.tobytes())
                stats["bf16_equal"] += 1
            elif kind == "fp8":
                codes, s = fp8_source(m, src, k, tr)
                if [codes.shape[1], codes.shape[0]] != list(dims):
                    raise ValueError(f"{name}: template {dims}, NVIDIA {codes.shape}")
                f.write(fp8t_blocks(codes, s).tobytes())
                stats["fp8"] += 1
            elif kind == "nvfp4":
                codes, scales, g = nvfp4_source(src, k)
                if [codes.shape[1] * 2, codes.shape[0]] != list(dims):
                    raise ValueError(f"{name}: template {dims}, NVIDIA {codes.shape}")
                f.write(nv.nvfp4g_blocks(codes, scales, g).tobytes())
                stats["nvfp4"] += 1
            else:  # stacked experts [cols, rows, E]
                cols, rows, E = dims
                for e in range(E):
                    codes, scales, g = nvfp4_source(src, k.format(e))
                    if codes.shape != (rows, cols // 2):
                        raise ValueError(f"{name} expert {e}: template {dims}, NVIDIA {codes.shape}")
                    f.write(nv.nvfp4g_blocks(codes, scales, g).tobytes())
                stats["nvfp4"] += E
        log(f"built {out}: {len(plan)} tensors, {(base + cur) / 1e9:.2f} GB; {stats['bf16_equal']} BF16 tensors equal to the "
            f"template after the llama.cpp transforms, {stats['fp8']} FP8T, {stats['nvfp4']} NVFP4G matrices")


# --- verify --------------------------------------------------------------------------------------------------------------
def verify(template, nvdir, out, log=print, rel_sample=4):
    """Quantized tensors of OUT -> NVIDIA's bytes (every one, exact); dequantized vs the template BF16 (relative error, on
    every FP8T tensor and on rel_sample expert matrices per stacked tensor)."""
    src = Folder(nvdir)
    m = Model(src.cfg)
    _, _, _, otens, _, odata = nv.read_gguf_header(out)
    _, _, _, ttens, _, tdata = nv.read_gguf_header(template)
    tby = {n: (d, t, o) for n, d, t, o in ttens}
    om, tm = np.memmap(out, np.uint8, mode="r"), np.memmap(template, np.uint8, mode="r")
    bad, rels = [], {}

    def tmpl_f32(name, start=0, count=None):
        d, t, o = tby[name]
        n = int(np.prod(d)) if count is None else count
        if t != BF16:
            raise ValueError(f"{name}: template type {t}")
        b = np.asarray(tm[tdata + o + 2 * start:tdata + o + 2 * (start + n)]).view(np.uint16)
        return (b.astype(np.uint32) << 16).view(np.float32)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    inv = lambda p: np.argsort(p)  # noqa: E731
    for name, dims, typ, off in otens:
        kind, k, tr = mapping(m, name)
        if kind == "fp8":
            if typ != FP8T:
                bad.append(f"{name}: type {typ}"); continue
            cols, rows = dims
            blk = np.asarray(om[odata + off:odata + off + rows * cols // 128 * 132]).reshape(rows, cols // 128, 132)
            codes = np.ascontiguousarray(blk[:, :, 4:]).reshape(rows, cols)
            sc = np.ascontiguousarray(blk[:, :, :4]).view(np.float32).reshape(rows, cols // 128)
            back = codes[inv(m.qkv_perm())] if tr == "qkv_rows" else codes[inv(m.vperm(m.dv))] if tr == "v_rows" else \
                codes[:, inv(m.vperm(m.dv))] if tr == "v_cols" else codes
            c0, _, shape = src.raw(k + ".weight")
            s0 = src.raw(k + ".weight_scale")[0].view(np.float32)[0]
            if not np.array_equal(back, c0.reshape(shape)) or not np.all(sc == s0):
                bad.append(f"{name}: FP8T bytes differ from {k}")
            rels[name] = rel(deq_fp8(codes, sc[:, :1]), tmpl_f32(name).reshape(rows, cols))
        elif kind in ("nvfp4", "nvfp4_exps"):
            if typ != NVFP4G:
                bad.append(f"{name}: type {typ}"); continue
            cols, rows = dims[0], dims[1]
            E = dims[2] if kind == "nvfp4_exps" else 1
            per = rows * cols // 128 * 76
            pick = set(np.linspace(0, E - 1, min(E, rel_sample)).astype(int).tolist())
            for e in range(E):
                blk = np.asarray(om[odata + off + e * per:odata + off + (e + 1) * per]).reshape(rows, cols // 128, 76)
                codes = np.ascontiguousarray(blk[:, :, 12:]).reshape(rows, cols // 2)
                scales = np.ascontiguousarray(blk[:, :, 4:12]).reshape(rows, cols // 16)
                g = np.ascontiguousarray(blk[:, :, :4]).view(np.float32)
                c0, s0, g0 = nvfp4_source(src, k.format(e) if kind == "nvfp4_exps" else k)
                if not (np.array_equal(codes, c0) and np.array_equal(scales, s0) and np.all(g == g0)):
                    bad.append(f"{name} [{e}]: NVFP4G bytes differ from NVIDIA")
                if e in pick:
                    step = 16384 if rows > 65536 else rows  # lm_head: 248320 rows in slices
                    for r0 in range(0, rows, step * (8 if rows > 65536 else 1)):
                        r1 = min(rows, r0 + step)
                        w = deq_nvfp4(codes[r0:r1], scales[r0:r1], g0)
                        t = tmpl_f32(name, (e * rows + r0) * cols, (r1 - r0) * cols).reshape(r1 - r0, cols)
                        rels[f"{name}[{e}]" + (f"r{r0}" if rows > 65536 else "")] = rel(w, t)
    worst = sorted(rels.items(), key=lambda kv: -kv[1])
    by = {}
    for n, r in rels.items():
        key = re.sub(r"blk\.\d+\.", "blk.N.", n.split("[")[0])
        by.setdefault(key, []).append(r)
    for key, v in sorted(by.items()):
        log(f"rel err vs template BF16  {key:28s} n={len(v):4d}  mean {np.mean(v):.4f}  max {np.max(v):.4f}")
    log(f"worst: {worst[:3]}")
    log(f"quantized tensors back to NVIDIA's bytes: {'all exact' if not bad else f'{len(bad)} differ: {bad[:5]}'}")
    return not bad and all(r < 0.25 for r in rels.values())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build" and len(sys.argv) == 5:
        build(*sys.argv[2:])
    elif cmd == "verify" and len(sys.argv) == 5:
        sys.exit(0 if verify(*sys.argv[2:]) else 1)
    else:
        sys.exit(__doc__)
