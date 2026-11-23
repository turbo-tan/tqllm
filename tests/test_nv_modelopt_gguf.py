"""tools/nv_modelopt_gguf.py on a tiny Qwen3.5-MoE-shaped modelopt checkpoint (2 K heads, 4 V heads: the V-head reorder
applies): FP8 projections -> FP8T with NVIDIA's bytes in llama.cpp's row / column order, NVFP4 -> NVFP4G, BF16 tensors
checked against a template written with an independent loop form of llama.cpp's transforms; a template that does not hold
NVIDIA's BF16 values is refused."""
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import nv_modelopt_gguf as conv  # noqa: E402

NK, NV, DK, DV, H, E, FF = 2, 4, 4, 128, 128, 2, 128
P = "model.language_model."


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)) + b)


def write_gguf(path, tensors):
    """tensors: [(name, dims (ne0 first), type, raw bytes)]"""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
        w_str(f, "general.alignment"); f.write(struct.pack("<II", 4, 32))
        off, offs = 0, []
        for _, _, _, raw in tensors:
            off = (off + 31) // 32 * 32; offs.append(off); off += len(raw)
        for (name, dims, t, _), o in zip(tensors, offs):
            w_str(f, name); f.write(struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, o))
        base = (f.tell() + 31) // 32 * 32
        for (_, _, _, raw), o in zip(tensors, offs):
            f.seek(base + o); f.write(raw)


def write_folder(d, arrays):
    hdr, blobs, off = {}, [], 0
    for k, (dtype, shape, raw) in arrays.items():
        hdr[k] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}; blobs.append(raw); off += len(raw)
    h = json.dumps(hdr).encode()
    with open(os.path.join(d, "model-1.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model-1.safetensors" for k in arrays}}, f)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"linear_num_key_heads": NK, "linear_num_value_heads": NV, "linear_key_head_dim": DK,
                                   "linear_value_head_dim": DV, "num_hidden_layers": 2}}, f)


def bf16(x):
    return conv.bf16_bytes(x)


def tiled(head_dim):
    """grouped V-head order (K head k, its r-th V head) -> llama.cpp's tiled order, written as a loop"""
    order = []
    for r in range(NV // NK):
        for k in range(NK):
            order += list(range((k * (NV // NK) + r) * head_dim, (k * (NV // NK) + r + 1) * head_dim))
    return np.array(order)


class NvModeloptGguf(unittest.TestCase):
    def make(self, d, rng, corrupt=False):
        arrays, tmpl = {}, []
        b16 = lambda x: (bf16(x).astype(np.uint32) << 16).view(np.float32)  # noqa: E731
        def add_bf16(k, x): arrays[k] = ("BF16", list(x.shape), bf16(x).tobytes())
        def fp8(k, rows, cols):
            codes = rng.integers(0, 0x7E, (rows, cols), dtype=np.uint8) | (rng.integers(0, 2, (rows, cols), dtype=np.uint8) << 7)
            s = np.float32(rng.uniform(1e-3, 1e-2))
            arrays[k + ".weight"] = ("F8_E4M3", [rows, cols], codes.tobytes())
            arrays[k + ".weight_scale"] = ("F32", [], s.tobytes())
            arrays[k + ".input_scale"] = ("F32", [], np.float32(1).tobytes())
            return codes, s
        def nvfp4(k, rows, cols):
            codes = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
            sc = rng.integers(0x20, 0x50, (rows, cols // 16), dtype=np.uint8)
            g = np.float32(rng.uniform(1e-4, 1e-3))
            arrays[k + ".weight"] = ("U8", [rows, cols // 2], codes.tobytes())
            arrays[k + ".weight_scale"] = ("F8_E4M3", [rows, cols // 16], sc.tobytes())
            arrays[k + ".weight_scale_2"] = ("F32", [], g.tobytes())
            return codes, sc, g
        emb = b16(rng.standard_normal((64, H)).astype(np.float32)); add_bf16(P + "embed_tokens.weight", emb)
        tmpl.append(("token_embd.weight", [H, 64], conv.BF16, bf16(emb).tobytes()))
        nrm = b16(rng.standard_normal(H).astype(np.float32) * 0.1); add_bf16(P + "norm.weight", nrm)
        tmpl.append(("output_norm.weight", [H], conv.F32, b16(nrm + 1).astype(np.float32).tobytes()))
        lc, ls, lg = nvfp4("lm_head", 64, H)
        tmpl.append(("output.weight", [H, 64], conv.BF16, bf16(conv.deq_nvfp4(lc, ls, lg)).tobytes()))
        la = P + "layers.0.linear_attn."
        qkvd = 2 * NK * DK + NV * DV
        qkv, s_qkv = fp8(la + "in_proj_qkv", qkvd, H)
        perm = np.concatenate([np.arange(2 * NK * DK), 2 * NK * DK + tiled(DV)])
        tmpl.append(("blk.0.attn_qkv.weight", [H, qkvd], conv.BF16, bf16(conv.deq_fp8(qkv[perm], s_qkv)).tobytes()))
        z, s_z = fp8(la + "in_proj_z", NV * DV, H)
        tmpl.append(("blk.0.attn_gate.weight", [H, NV * DV], conv.BF16, bf16(conv.deq_fp8(z[tiled(DV)], s_z)).tobytes()))
        o, s_o = fp8(la + "out_proj", H, NV * DV)
        tmpl.append(("blk.0.ssm_out.weight", [NV * DV, H], conv.BF16, bf16(conv.deq_fp8(o[:, tiled(DV)], s_o)).tobytes()))
        alog = b16(rng.standard_normal(NV).astype(np.float32)); add_bf16(la + "A_log", alog)
        tmpl.append(("blk.0.ssm_a", [NV], conv.F32, b16(-np.exp(alog[tiled(1)])).astype(np.float32).tobytes()))
        a = b16(rng.standard_normal((NV, H)).astype(np.float32)); add_bf16(la + "in_proj_a.weight", a)
        ta = a[tiled(1)].copy()
        if corrupt:
            ta[0, 0] += 1
        tmpl.append(("blk.0.ssm_alpha.weight", [H, NV], conv.BF16, bf16(ta).tobytes()))
        cw = b16(rng.standard_normal((qkvd, 1, 4)).astype(np.float32)); add_bf16(la + "conv1d.weight", cw)
        tmpl.append(("blk.0.ssm_conv1d.weight", [4, qkvd], conv.F32, cw.reshape(qkvd, 4)[perm].astype(np.float32).tobytes()))
        q, s_q = fp8(P + "layers.1.self_attn.q_proj", 2 * H, H)
        tmpl.append(("blk.1.attn_q.weight", [H, 2 * H], conv.BF16, bf16(conv.deq_fp8(q, s_q)).tobytes()))
        ex = []
        for e in range(E):
            ex.append(nvfp4(P + f"layers.1.mlp.experts.{e}.down_proj", H, FF))
        tmpl.append(("blk.1.ffn_down_exps.weight", [FF, H, E], conv.BF16,
                     bf16(np.concatenate([conv.deq_nvfp4(*t) for t in ex])).tobytes()))
        sh = nvfp4(P + "layers.1.mlp.shared_expert.up_proj", FF, H)
        tmpl.append(("blk.1.ffn_up_shexp.weight", [H, FF], conv.BF16, bf16(conv.deq_nvfp4(*sh)).tobytes()))
        write_folder(d, arrays)
        write_gguf(os.path.join(d, "t.gguf"), tmpl)
        return dict(qkv=(qkv, s_qkv, perm), z=(z, s_z), o=(o, s_o), lm=(lc, ls, lg), ex=ex)

    def test_build_verify(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            ref = self.make(d, rng)
            out = os.path.join(d, "o.gguf")
            conv.build(os.path.join(d, "t.gguf"), d, out, log=lambda *_: None)
            self.assertTrue(conv.verify(os.path.join(d, "t.gguf"), d, out, log=lambda *_: None))
            _, _, _, tens, _, data = conv.nv.read_gguf_header(out)
            by = {n: (dims, t, o) for n, dims, t, o in tens}
            with open(out, "rb") as f:
                raw = f.read()
            dims, t, o = by["blk.0.attn_qkv.weight"]
            self.assertEqual(t, conv.FP8T)
            qkv, s, perm = ref["qkv"]
            blk = np.frombuffer(raw, np.uint8, qkv.size // 128 * 132, data + o).reshape(qkv.shape[0], -1, 132)
            self.assertTrue(np.array_equal(blk[:, :, 4:].reshape(qkv.shape), qkv[perm]))
            self.assertTrue(np.all(blk[:, :, :4].copy().view(np.float32) == s))
            o_codes, s_o = ref["o"]
            dims, t, o = by["blk.0.ssm_out.weight"]
            blk = np.frombuffer(raw, np.uint8, o_codes.size // 128 * 132, data + o).reshape(o_codes.shape[0], -1, 132)
            self.assertTrue(np.array_equal(blk[:, :, 4:].reshape(o_codes.shape), o_codes[:, tiled(DV)]))
            dims, t, o = by["output.weight"]
            self.assertEqual(t, conv.NVFP4G)
            self.assertEqual(by["blk.1.ffn_down_exps.weight"][1], conv.NVFP4G)
            self.assertEqual(by["blk.0.ssm_alpha.weight"][1], conv.BF16)

    def test_template_mismatch_refused(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as d:
            self.make(d, rng, corrupt=True)
            with self.assertRaisesRegex(ValueError, "ssm_alpha"):
                conv.build(os.path.join(d, "t.gguf"), d, os.path.join(d, "o.gguf"), log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
