#!/usr/bin/env python3
"""Flash-Next (qwen4_exp) MLX affine 4-bit checkpoint -> tqLLM GGUF, the same weights without requantising.

usage: mlx_to_gguf.py MLX_DIR REF_GGUF_DIR OUT_DIR [--check] [--only PATTERN] [--split-gb N] [--keep PATTERN]

MLX_DIR: an MLX checkpoint of Qwen3.8 Flash-Next (affine, 4-bit, group 32, MTP head under language_model.mtp).
REF_GGUF_DIR: a qwen4exp GGUF of the same model (the ISTA GSQ files and mtp-blk48-*.gguf): tensor names, shapes, which
  tensors the engine expects in BF16 / F32, and the model metadata come from it.
OUT_DIR: writes flashnext-mlx-0000N.gguf (the trunk, split by --split-gb), mtp-blk48-mlx.gguf and a manifest.

Every 4-bit matrix becomes GGUF Q4_1 (d = scale, m = bias per 32 weights, the MLX values: BF16 -> FP16 is checked exact),
so the engine runs the checkpoint's own weights. A tensor the reference keeps in BF16 / F32 (norms, the router, the
hyper-connection mixers, the PLE key / value projections) is dequantised once from the checkpoint into that type.
--check: converts nothing; compares every mapped tensor, dequantised, against the reference tensor (relative RMS, and
the correlation) to catch a wrong transform or transpose before a file is written.
--keep PATTERN: a mixed trunk (flashnext-mixed-q4_1.gguf, no MTP file): tensors matching PATTERN and every BF16 / F32 /
F16 tensor are copied byte for byte from the reference, the other quantised tensors come from the checkpoint as Q4_1
(e.g. "_exps|token_embd|per_layer": the reference's experts with the checkpoint's 4-bit dense matrices)."""
import json, math, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/code/llama.cpp-tq3/gguf-py"))
import gguf  # noqa: E402

GROUP = 32


# ---- safetensors (raw: numpy has no bfloat16) ----
class Shards:
    def __init__(self, folder):
        self.folder, self.where = folder, {}
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".safetensors"):
                continue
            with open(os.path.join(folder, f), "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.where[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"])

    def has(self, k):
        return k in self.where

    def raw(self, k):
        f, off, size, dt, shape = self.where[k]
        a = np.memmap(os.path.join(self.folder, f), dtype=np.uint8, mode="r", offset=off, shape=(size,))
        return a, dt, shape

    def array(self, k):
        a, dt, shape = self.raw(k)
        if dt == "BF16":
            return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "U32":
            return a.view(np.uint32).reshape(shape)
        if dt == "I64":
            return a.view(np.int64).reshape(shape)
        if dt in ("F32",):
            return a.view(np.float32).reshape(shape)
        if dt in ("F16",):
            return a.view(np.float16).astype(np.float32).reshape(shape)
        raise ValueError(f"{k}: dtype {dt}")

    def bf16_bits(self, k):
        a, dt, shape = self.raw(k)
        assert dt == "BF16", k
        return a.view(np.uint16).reshape(shape)


def bf16_bits_to_f16(bits):
    """BF16 bit patterns -> FP16; values in FP16's subnormal range round to nearest (as the engine's folder loader: at most
    3e-8 off); raises on any other inexact value (range or precision)."""
    f = (bits.astype(np.uint32) << 16).view(np.float32)
    h = f.astype(np.float16)
    off = h.astype(np.float32) != f
    if off.any():
        bad = np.flatnonzero((off & (np.abs(f) >= np.float32(6.103515625e-05))).ravel())
        if bad.size:
            raise ValueError(f"{bad.size} BF16 scales / biases are not exact in FP16 (e.g. {f.ravel()[bad[0]]!r})")
    return h


def unpack_u32(w):
    """MLX 4-bit: 8 codes per uint32, low nibble first -> uint8 codes [..., cols]."""
    s = w.shape
    q = np.empty(s[:-1] + (s[-1] * 8,), dtype=np.uint8)
    for j in range(8):
        q[..., j::8] = (w >> np.uint32(4 * j)) & np.uint32(15)
    return q


def q4_1_blocks(src, k):
    """MLX quantised tensor k (weight / scales / biases) -> GGUF Q4_1 bytes [..., rows, cols / 32 * 20], exact."""
    w = src.array(k + ".weight")
    q = unpack_u32(w)                                     # [..., rows, cols]
    d = bf16_bits_to_f16(src.bf16_bits(k + ".scales"))    # [..., rows, cols / 32]
    m = bf16_bits_to_f16(src.bf16_bits(k + ".biases"))
    cols = q.shape[-1]
    assert cols % GROUP == 0 and d.shape[-1] == cols // GROUP, (k, q.shape, d.shape)
    qb = q.reshape(q.shape[:-1] + (cols // GROUP, GROUP))
    nib = (qb[..., :16] | (qb[..., 16:] << 4)).astype(np.uint8)          # byte i: low = element i, high = i + 16
    out = np.empty(q.shape[:-1] + (cols // GROUP, 20), dtype=np.uint8)
    out[..., 0:2] = d[..., None].view(np.uint8).reshape(d.shape + (2,))
    out[..., 2:4] = m[..., None].view(np.uint8).reshape(m.shape + (2,))
    out[..., 4:] = nib
    return out.reshape(q.shape[:-1] + (cols // GROUP * 20,))


def dequant(src, k):
    """MLX tensor k (quantised: k.weight + scales + biases; else k.weight) -> float32 [..., rows, cols]."""
    if src.has(k + ".scales"):
        q = unpack_u32(src.array(k + ".weight")).astype(np.float32)
        s, b = src.array(k + ".scales"), src.array(k + ".biases")
        cols = q.shape[-1]; g = cols // s.shape[-1]
        return (q.reshape(q.shape[:-1] + (cols // g, g)) * s[..., None] + b[..., None]).reshape(q.shape)
    return src.array(k + ".weight" if src.has(k + ".weight") else k)


# ---- name map: reference GGUF name -> (MLX key, kind) ----
def mlx_source(name, n_trunk):
    """kind: 'q' quantised matrix (key without suffix), 'w' plain tensor key, or a callable (src) -> float32 array."""
    LM, MTP = "language_model.model.", "language_model.mtp."
    g = {"token_embd.weight": ("q", "language_model.model.embed_tokens"), "output.weight": ("q", "language_model.lm_head"),
         "output_hc_down.weight": ("q", LM + "hyper_connection_mixer.input_mix_weight_down"),
         "output_hc_up.weight": ("q", LM + "hyper_connection_mixer.input_mix_weight_up"),
         "output_hc_norm.weight": ("w", LM + "hyper_connection_mixer.hc_norm.weight")}
    if name in g:
        return g[name]
    if name == "per_layer_token_embd.weight":
        return ("ple", None)
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        raise KeyError(name)
    i, rest = int(m.group(1)), m.group(2)
    P = (LM + f"layers.{i}.") if i < n_trunk else (MTP + "layers.0.")
    t = {"hc_attn_down.weight": ("q", P + "attn_hyper_connection.input_mix_weight_down"),
         "hc_attn_up.weight": ("q", P + "attn_hyper_connection.input_mix_weight_up"),
         "hc_attn_inject.weight": ("q", P + "attn_hyper_connection.block_inject_weight"),
         "hc_attn_norm.weight": ("w", P + "attn_hyper_connection.hc_norm.weight"),
         "hc_ffn_down.weight": ("q", P + "mlp_hyper_connection.input_mix_weight_down"),
         "hc_ffn_up.weight": ("q", P + "mlp_hyper_connection.input_mix_weight_up"),
         "hc_ffn_inject.weight": ("q", P + "mlp_hyper_connection.block_inject_weight"),
         "hc_ffn_norm.weight": ("w", P + "mlp_hyper_connection.hc_norm.weight"),
         "attn_qkv.weight": ("q", P + "linear_attn.in_proj_qkv"), "attn_gate.weight": ("q", P + "linear_attn.in_proj_z"),
         "ssm_alpha.weight": ("q", P + "linear_attn.in_proj_a"), "ssm_beta.weight": ("q", P + "linear_attn.in_proj_b"),
         "ssm_out.weight": ("q", P + "linear_attn.out_proj"), "ssm_a": ("A_log", P + "linear_attn.A_log"),
         "ssm_dt.bias": ("w", P + "linear_attn.dt_bias"), "ssm_conv1d.weight": ("conv", P + "linear_attn.conv1d.weight"),
         "ssm_norm.weight": ("w", P + "linear_attn.norm.weight"),
         "attn_q.weight": ("q", P + "self_attn.q_proj"), "attn_k.weight": ("q", P + "self_attn.k_proj"),
         "attn_v.weight": ("q", P + "self_attn.v_proj"), "attn_output.weight": ("q", P + "self_attn.o_proj"),
         "attn_q_norm.weight": ("w", P + "self_attn.q_norm.weight"), "attn_k_norm.weight": ("w", P + "self_attn.k_norm.weight"),
         "indexer.q_proj.weight": ("rows", (P + "self_attn.indexer.index_qk_proj", 0, None)),
         "indexer.k_proj.weight": ("rows", (P + "self_attn.indexer.index_qk_proj", None, 0)),
         "indexer.q_norm.weight": ("w", P + "self_attn.indexer.q_layernorm.weight"),
         "indexer.k_norm.weight": ("w", P + "self_attn.indexer.k_layernorm.weight"),
         "ffn_gate_inp.weight": ("w", P + "mlp.gate.weight"), "ffn_gate_inp_shexp.weight": ("q", P + "mlp.shared_expert_gate"),
         "ffn_gate_shexp.weight": ("q", P + "mlp.shared_expert.gate_proj"), "ffn_up_shexp.weight": ("q", P + "mlp.shared_expert.up_proj"),
         "ffn_down_shexp.weight": ("q", P + "mlp.shared_expert.down_proj"),
         "ffn_gate_exps.weight": ("q", P + "mlp.switch_mlp.gate_proj"), "ffn_up_exps.weight": ("q", P + "mlp.switch_mlp.up_proj"),
         "ffn_down_exps.weight": ("q", P + "mlp.switch_mlp.down_proj"),
         "ple_conv1d.weight": ("conv", P + "ple.conv1d.weight"), "ple_key.weight": ("q", P + "ple.key_proj"),
         "ple_value.weight": ("q", P + "ple.value_proj"), "ple_norm_conv.weight": ("w", P + "ple.norm_conv.weight"),
         "ple_norm_key.weight": ("w", P + "ple.norm_key.weight"), "ple_norm_query.weight": ("w", P + "ple.norm_query.weight"),
         "nextn.eh_proj.weight": ("cols", (MTP + "fc_embedding", MTP + "fc_hidden")),
         "nextn.enorm.weight": ("w", MTP + "pre_fc_norm_embedding.weight"), "nextn.hnorm.weight": ("w", MTP + "pre_fc_norm_hidden.weight"),
         "nextn.hc_head_down.weight": ("q", MTP + "hyper_connection_mixer.input_mix_weight_down"),
         "nextn.hc_head_up.weight": ("q", MTP + "hyper_connection_mixer.input_mix_weight_up"),
         "nextn.hc_head_norm.weight": ("w", MTP + "hyper_connection_mixer.hc_norm.weight")}
    if rest not in t:
        raise KeyError(name)
    return t[rest]


# value transforms found by --check against the reference (same base model)
TRANSFORMS = {"A_log": "ssm_a = -exp(A_log)",
              "gdn value heads": "the reference holds value head r * KH + g where the checkpoint holds g * (VH / KH) + r"}
HEAD = 128  # GDN state dimension: one value head = 128 rows / columns


def gdn_perm(src, P):
    vh = src.array(P + "linear_attn.A_log").shape[0]
    qkv_rows = src.where[P + "linear_attn.in_proj_qkv.weight"][4][0]
    kh = (qkv_rows - vh * HEAD) // (2 * HEAD)
    r = vh // kh
    return [(j % kh) * r + j // kh for j in range(vh)], kh


def permute_blocks(a, perm, block, axis, offset=0):
    """reorder `len(perm)` blocks of `block` along axis, starting at offset: block j of the result = block perm[j]"""
    a = np.asarray(a)
    idx = np.arange(a.shape[axis])
    sec = np.concatenate([np.arange(p * block, (p + 1) * block) for p in perm]) + offset
    idx[offset: offset + len(perm) * block] = sec
    return np.take(a, idx, axis=axis)


# GGUF name suffix -> (axis, block in elements, offset in blocks of HEAD (qkv: after q and k)) for the GDN head order
GDN_POST = {"ssm_a": (0, 1, None), "ssm_dt.bias": (0, 1, None), "ssm_alpha.weight": (0, 1, None), "ssm_beta.weight": (0, 1, None),
            "attn_gate.weight": (0, HEAD, None), "attn_qkv.weight": (0, HEAD, "qk"), "ssm_conv1d.weight": (0, HEAD, "qk"),
            "ssm_out.weight": (1, HEAD, None)}


def gdn_post(src, name, arr, q41):
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m or m.group(2) not in GDN_POST:
        return arr
    P = f"language_model.model.layers.{int(m.group(1))}."
    perm, kh = gdn_perm(src, P)
    axis, block, off = GDN_POST[m.group(2)]
    offset = 2 * kh * HEAD if off == "qk" else 0
    if q41 and axis == 1:  # columns of Q4_1 bytes: 32 weights = 20 bytes
        block, offset = block // GROUP * 20, offset // GROUP * 20
    if arr.ndim == 1:
        axis = 0
    return permute_blocks(arr, perm, block, axis if arr.ndim > 1 else 0, offset)


def ple_shard_keys(src):
    ks = sorted((k for k in src.where if ".ple_embedding.ngram_embedding.shard_" in k and k.endswith(".weight")),
                key=lambda k: int(re.search(r"shard_(\d+)\.weight$", k).group(1)))
    return [k[: -len(".weight")] for k in ks]


def float_value(src, kind, key, ref_rows):
    """the mapped tensor as float32 in the reference's row-major [rows, cols] (GGUF ne reversed)"""
    if kind == "q":
        return dequant(src, key)
    if kind == "w":
        return src.array(key)
    if kind == "A_log":
        return -np.exp(src.array(key))
    if kind == "conv":
        a = src.array(key)
        return a.reshape(a.shape[0], -1)
    if kind == "rows":
        k, q_end, k_start = key
        full = dequant(src, k)
        nq = ref_rows
        return full[:nq] if q_end == 0 else full[full.shape[0] - ref_rows:]
    if kind == "cols":
        return np.concatenate([dequant(src, k) for k in key], axis=-1)
    if kind == "ple":
        return np.concatenate([dequant(src, k) for k in ple_shard_keys(src)[:1]], axis=0)  # check: shard 0 only
    raise ValueError(kind)


def q4_1_value(src, kind, key, ref_rows):
    if kind == "q":
        return q4_1_blocks(src, key)
    if kind == "rows":
        k, q_end, k_start = key
        full = q4_1_blocks(src, k)
        return full[:ref_rows] if q_end == 0 else full[full.shape[0] - ref_rows:]
    if kind == "cols":
        return np.concatenate([q4_1_blocks(src, k) for k in key], axis=-1)
    if kind == "ple":
        return None  # streamed shard by shard
    return None


def ref_tensors(ref_dir):
    out = {}
    for f in sorted(os.listdir(ref_dir)):
        if not f.endswith(".gguf") or "tokenizer" in f:
            continue
        r = gguf.GGUFReader(os.path.join(ref_dir, f))
        for t in r.tensors:
            out[t.name] = (f, t, r)
    return out


def check(src, ref_dir, only):
    refs = ref_tensors(ref_dir)
    n_trunk = int(next(r for _, _, r in refs.values()).fields["qwen4exp.block_count"].parts[-1][0])
    worst = []
    for name in sorted(refs):
        if only and not re.search(only, name):
            continue
        f, t, r = refs[name]
        try:
            kind, key = mlx_source(name, n_trunk)
        except KeyError:
            print(f"UNMAPPED {name}"); continue
        rows = int(t.shape[1]) if len(t.shape) > 1 else 1
        try:
            mine = gdn_post(src, name, float_value(src, kind, key, rows), False).astype(np.float32)
        except Exception as e:
            print(f"ERROR {name}: {e}"); continue
        theirs = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        if kind == "ple":
            theirs = theirs[: mine.shape[0]]
        if theirs.size != mine.size:
            print(f"SHAPE {name}: mlx {mine.shape} ref {theirs.shape}"); continue
        a, b = mine.ravel(), theirs.reshape(-1)
        rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        off = float(np.mean(b - a))
        worst.append((rel, name))
        print(f"{name:48s} {t.tensor_type.name:7s} rel-rms {rel:8.4f} corr {corr:7.4f} mean(ref - mlx) {off:+.4f}")
    worst.sort(reverse=True)
    print("worst:", worst[:8])


# ---- writer: GGUF v3, metadata copied byte for byte from the reference (minus split.* / quantize.*) ----
def gguf_str(s):
    b = s.encode(); return struct.pack("<Q", len(b)) + b


def kv_bytes(reader, drop=("split.", "quantize.")):
    out, n = b"", 0
    for k, f in reader.fields.items():
        if k.startswith("GGUF.") or any(k.startswith(d) for d in drop):
            continue
        out += b"".join(np.asarray(p).tobytes() for p in f.parts); n += 1
    return out, n


def float_bytes(arr, ggml_type):
    if ggml_type == gguf.GGMLQuantizationType.BF16:
        u = arr.astype(np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even
    if ggml_type == gguf.GGMLQuantizationType.F32:
        return arr.astype(np.float32)
    if ggml_type == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16)
    raise ValueError(ggml_type)


def ref_copy(t, chunk=1 << 28):
    flat = np.asarray(t.data).reshape(-1).view(np.uint8)
    return (flat[i:i + chunk] for i in range(0, flat.size, chunk))


def plan(src, refs, names, n_trunk, keep=None):
    """per tensor: (name, ne, out type, nbytes, producer() -> iterator of uint8 / typed arrays)"""
    Q41 = gguf.GGMLQuantizationType.Q4_1
    FLOAT = (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    items = []
    for name in names:
        f, t, r = refs[name]
        ne = [int(x) for x in t.shape]
        if keep is not None and (re.search(keep, name) or t.tensor_type in FLOAT):
            items.append((name, ne, t.tensor_type, int(t.n_bytes), (lambda t=t: ref_copy(t))))
            continue
        kind, key = mlx_source(name, n_trunk)
        rows = ne[1] if len(ne) > 1 else 1
        ref_type = t.tensor_type
        if ref_type in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            out_type = ref_type
            nbytes = int(np.prod(ne)) * (4 if ref_type == gguf.GGMLQuantizationType.F32 else 2)
            prod = (lambda name=name, kind=kind, key=key, rows=rows, out_type=out_type: iter([float_bytes(gdn_post(src, name, float_value(src, kind, key, rows), False), out_type)]))
        else:
            out_type = Q41
            nbytes = int(np.prod(ne)) // GROUP * 20
            if kind == "ple":
                prod = (lambda: (q4_1_blocks(src, k) for k in ple_shard_keys(src)))
            else:
                prod = (lambda name=name, kind=kind, key=key, rows=rows: iter([gdn_post(src, name, q4_1_value(src, kind, key, rows), True)]))
        items.append((name, ne, out_type, nbytes, prod))
    return items


def write_gguf(path, ref_reader, items, align=32):
    kv, nkv = kv_bytes(ref_reader)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(items), nkv) + kv
    off, infos = 0, b""
    for name, ne, out_type, nbytes, _ in items:
        infos += gguf_str(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne) + struct.pack("<IQ", int(out_type), off)
        off += (nbytes + align - 1) // align * align
    with open(path + ".part", "wb") as h:
        h.write(head + infos); h.write(b"\0" * ((-h.tell()) % align))
        for name, ne, out_type, nbytes, prod in items:
            wrote = 0
            for chunk in prod():
                b = np.ascontiguousarray(chunk).tobytes(); h.write(b); wrote += len(b)
            if wrote != nbytes:
                raise ValueError(f"{name}: wrote {wrote} bytes, expected {nbytes}")
            h.write(b"\0" * ((-wrote) % align))
            print(f"  {name} {gguf.GGMLQuantizationType(out_type).name} {nbytes / 1e6:.1f} MB", flush=True)
    os.replace(path + ".part", path)


def write(src, ref_dir, out_dir, only, keep=None):
    refs = ref_tensors(ref_dir)
    r0 = next(r for f, _, r in refs.values() if "00001" in f)
    n_trunk = int(r0.fields["qwen4exp.block_count"].parts[-1][0])
    os.makedirs(out_dir, exist_ok=True)
    trunk = [n for n in refs if "mtp" not in refs[n][0] and (not only or re.search(only, n))]
    mtp = [n for n in refs if "mtp" in refs[n][0] and (not only or re.search(only, n))]
    rm = next(r for f, _, r in refs.values() if "mtp" in f)
    # reference order: the ISTA files' own tensor order (shard 1 then 2)
    order = {n: i for i, n in enumerate(refs)}
    trunk.sort(key=order.get); mtp.sort(key=order.get)
    if keep is not None:
        write_gguf(os.path.join(out_dir, "flashnext-mixed-q4_1.gguf"), r0, plan(src, refs, trunk, n_trunk, keep))
        return
    if trunk:
        write_gguf(os.path.join(out_dir, "flashnext-mlx-q4_1.gguf"), r0, plan(src, refs, trunk, n_trunk))
    if mtp:
        write_gguf(os.path.join(out_dir, "mtp-blk48-mlx-q4_1.gguf"), rm, plan(src, refs, mtp, n_trunk))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        print(__doc__); sys.exit(2)
    only = a[a.index("--only") + 1] if "--only" in a else None
    src = Shards(a[0])
    if "--check" in a:
        check(src, a[1], only)
        sys.exit(0)
    write(src, a[1], a[2], only, a[a.index("--keep") + 1] if "--keep" in a else None)
