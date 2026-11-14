"""tools/convert/pplx/pplx_context_to_gguf.py on a tiny synthetic pplx-embed-v2-context checkpoint."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

try:  # every case and the converter itself need numpy: skip the whole test (ctest SKIP_RETURN_CODE 77) without it
    import numpy as np
except ImportError:
    print("pplx-converter: skipped (numpy not installed)")
    sys.exit(77)

try:
    import gguf  # noqa: F401
    HAVE_GGUF = True
except ImportError:
    HAVE_GGUF = False
NEEDS_GGUF = unittest.skipUnless(HAVE_GGUF, "gguf-py not installed")
try:
    import safetensors  # noqa: F401  (the converter's TQ4 codes file)
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False
NEEDS_SAFETENSORS = unittest.skipUnless(HAVE_SAFETENSORS, "safetensors not installed")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pplx_converter", ROOT / "tools/convert/pplx/pplx_context_to_gguf.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

H, I, HD, KD, KH = 32, 64, 32, 32, 2


def bf16(a):
    return (np.asarray(a, dtype=np.float32).view(np.uint32) >> 16).astype("<u2")


def read_gguf_tensors(path):
    """{name: (type, dims, raw bytes)} from a GGUF file; a plain reader, since gguf-py's has no TQ4 (type 300)"""
    b = Path(path).read_bytes()
    pos = 0
    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, b, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]
    def string():
        nonlocal pos
        n = take("Q"); v = b[pos:pos + n]; pos += n
        return v.decode()
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def value(t):
        if t == 8: return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(sizes[t])
    assert b[:4] == b"GGUF"; pos = 4
    take("I"); nt, nk = take("Q"), take("Q")
    kv = {}
    for _ in range(nk):
        k = string(); kv[k] = value(take("I"))
    infos = []
    for _ in range(nt):
        name = string(); nd = take("I"); dims = [take("Q") for _ in range(nd)]; ty = take("I"); off = take("Q")
        infos.append((name, dims, ty, off))
    align = kv.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    ends = sorted(base + o for *_, o in infos) + [len(b)]
    out = {}
    for name, dims, ty, off in infos:
        start = base + off
        end = next(e for e in ends if e > start)
        out[name] = (ty, dims, np.frombuffer(b[start:end], dtype=np.uint8))
    return out


def checkpoint(root, kv_heads=KH, f32=False, shards=False):
    tc = {"hidden_size": H, "intermediate_size": I, "num_hidden_layers": 2, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": HD, "rms_norm_eps": 1e-6, "vocab_size": 8,
          "linear_conv_kernel_dim": 4, "linear_key_head_dim": KD, "linear_value_head_dim": KD,
          "linear_num_key_heads": KH, "linear_num_value_heads": kv_heads, "full_attention_interval": 4,
          "layer_types": ["linear_attention", "full_attention"], "is_causal": False, "max_position_embeddings": 64,
          "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]}}
    config = {"model_type": "pplx_contextual_qwen3_5", "text_config": tc, "embedding_dim": H, "document_length": 64,
              "query_prefix": "[Q] ", "document_prefix": "[D] ", "boundary_marker": "<|chunk_sep|>"}
    rng = np.random.default_rng(0)
    conv_ch = 2 * KH * KD + kv_heads * KD
    shapes = {"contextual_projection.weight": [H, H], "language_model.embed_tokens.weight": [8, H],
              "language_model.norm.weight": [H]}
    for l, full in ((0, False), (1, True)):
        p = f"language_model.layers.{l}."
        shapes.update({p + "input_layernorm.weight": [H], p + "post_attention_layernorm.weight": [H],
                       p + "mlp.gate_proj.weight": [I, H], p + "mlp.up_proj.weight": [I, H], p + "mlp.down_proj.weight": [H, I]})
        if full:
            shapes.update({p + "self_attn.q_proj.weight": [2 * 2 * HD, H], p + "self_attn.k_proj.weight": [HD, H],
                           p + "self_attn.v_proj.weight": [HD, H], p + "self_attn.o_proj.weight": [H, 2 * HD],
                           p + "self_attn.q_norm.weight": [HD], p + "self_attn.k_norm.weight": [HD]})
        else:
            a = p + "linear_attn."
            shapes.update({a + "in_proj_qkv.weight": [conv_ch, H], a + "in_proj_z.weight": [kv_heads * KD, H],
                           a + "in_proj_a.weight": [kv_heads, H], a + "in_proj_b.weight": [kv_heads, H],
                           a + "out_proj.weight": [H, kv_heads * KD], a + "A_log": [kv_heads], a + "dt_bias": [kv_heads],
                           a + "conv1d.weight": [conv_ch, 1, 4], a + "norm.weight": [KD]})
    shapes["visual.patch_embed.proj.weight"] = [2, 2]
    values = {k: bf16(rng.standard_normal(s)) for k, s in shapes.items()}
    if f32:
        values = {k: (v.astype(np.uint32) << 16).view(np.float32) + np.float32(2.0 ** -20) for k, v in values.items()}

    def write(path, names):
        header, payload = {}, bytearray()
        for name in names:
            v = values[name]
            b = len(payload); payload.extend(v.tobytes())
            header[name] = {"dtype": "F32" if f32 else "BF16", "shape": list(v.shape), "data_offsets": [b, len(payload)]}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    if shards:  # the 9B layout: two shards + an index, the projection in contextual_head.safetensors
        names = [k for k in values if k != "contextual_projection.weight"]
        half = len(names) // 2
        write(root / "model-00001-of-00002.safetensors", names[:half])
        write(root / "model-00002-of-00002.safetensors", names[half:])
        write(root / "contextual_head.safetensors", ["contextual_projection.weight"])
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            k: ("model-00001-of-00002.safetensors" if i < half else "model-00002-of-00002.safetensors") for i, k in enumerate(names)}}))
    else:
        write(root / "model.safetensors", list(values))
    (root / "config.json").write_text(json.dumps(config))
    vocab = {"a": 0, "b": 1, "Ġ": 2, "[": 3}
    added = [{"id": 4, "content": "<|im_end|>", "special": True}, {"id": 5, "content": "[Q] ", "special": True},
             {"id": 6, "content": "[D] ", "special": True}, {"id": 7, "content": "<|chunk_sep|>", "special": False}]
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab, "merges": [["Ġ", "a"]]}, "added_tokens": added}))
    (root / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>", "pad_token": None}))
    if f32:
        return values
    return {k: (v.astype(np.uint32) << 16).view(np.float32) for k, v in values.items()}


class TestPplxConverter(unittest.TestCase):
    def test_interval_from_layer_types(self):
        self.assertEqual(converter.full_attention_interval(["linear_attention", "full_attention"] * 6), 2)
        self.assertEqual(converter.full_attention_interval((["linear_attention"] * 3 + ["full_attention"]) * 2), 4)
        with self.assertRaises(ValueError):
            converter.full_attention_interval(["full_attention", "linear_attention"])

    @NEEDS_GGUF
    def test_round_trip(self):
        import gguf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = checkpoint(root)
            converter.convert(root, root / "m.gguf")
            r = gguf.GGUFReader(str(root / "m.gguf"))
            f = lambda k: r.fields[k].contents()
            t = {x.name: np.asarray(x.data) for x in r.tensors}
            self.assertEqual(f("general.architecture"), "qwen35")
            self.assertEqual(f("qwen35.full_attention_interval"), 2)
            self.assertEqual(f("qwen35.rope.dimension_count"), HD // 2)
            self.assertEqual(f("pplx.attention_causal"), False)
            self.assertEqual(f("pplx.query_prefix_token_id"), 5)
            self.assertEqual((f("pplx.document_prefix"), f("pplx.boundary_marker")), ("[D] ", "<|chunk_sep|>"))
            self.assertEqual(list(f("tokenizer.ggml.token_type")), [1, 1, 1, 1, 3, 3, 3, 4])
            P = "language_model."
            np.testing.assert_array_equal(t["output_norm.weight"], 1 + w[P + "norm.weight"])
            np.testing.assert_array_equal(t["blk.1.attn_q_norm.weight"], 1 + w[P + "layers.1.self_attn.q_norm.weight"])
            a = P + "layers.0.linear_attn."
            np.testing.assert_allclose(t["blk.0.ssm_a"], -np.exp(w[a + "A_log"]), rtol=1e-6)
            np.testing.assert_array_equal(t["blk.0.ssm_norm.weight"], w[a + "norm.weight"])
            np.testing.assert_array_equal(t["blk.0.ssm_conv1d.weight"], w[a + "conv1d.weight"][:, 0, :])
            np.testing.assert_array_equal(t["contextual.projection.weight"], w["contextual_projection.weight"])
            self.assertFalse(any(n.startswith("visual") or "blk.1.ssm" in n for n in t))

    @NEEDS_GGUF
    def test_sharded_f32_and_value_head_groups(self):
        import gguf
        for outtype in ("auto", "bf16"):
            with self.subTest(outtype=outtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                w = checkpoint(root, kv_heads=2 * KH, f32=True, shards=True)
                converter.convert(root, root / "m.gguf", outtype)
                r = gguf.GGUFReader(str(root / "m.gguf"))
                t = {x.name: x for x in r.tensors}
                self.assertEqual(r.fields["qwen35.ssm.time_step_rank"].contents(), 2 * KH)
                self.assertEqual(r.fields["qwen35.ssm.group_count"].contents(), KH)
                q = w["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
                got = np.asarray(t["blk.0.attn_qkv.weight"].data)
                if outtype == "auto":
                    self.assertEqual(t["blk.0.attn_qkv.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
                    np.testing.assert_array_equal(got, q)
                else:
                    self.assertEqual(t["blk.0.attn_qkv.weight"].tensor_type, gguf.GGMLQuantizationType.BF16)
                    back = (got.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
                    np.testing.assert_allclose(back, q, rtol=2 ** -8)
                    # the token table stays F32: the fast path gathers it in FP32 (a BF16 table fails the fp32 tail rule)
                    self.assertEqual(t["token_embd.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
                    np.testing.assert_array_equal(np.asarray(t["token_embd.weight"].data), w["language_model.embed_tokens.weight"])
                np.testing.assert_array_equal(np.asarray(t["contextual.projection.weight"].data), w["contextual_projection.weight"])

    @NEEDS_GGUF
    def test_q8_0(self):
        import gguf
        from gguf import quants
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = checkpoint(root)
            converter.convert(root, root / "m.gguf", "q8_0")
            r = gguf.GGUFReader(str(root / "m.gguf"))
            t = {x.name: x for x in r.tensors}
            Q8 = gguf.GGMLQuantizationType.Q8_0
            for name in ("token_embd.weight", "blk.0.attn_qkv.weight", "blk.1.attn_q.weight", "blk.0.ffn_down.weight"):
                self.assertEqual(t[name].tensor_type, Q8, name)
            got = quants.dequantize(np.asarray(t["blk.0.attn_qkv.weight"].data), Q8)
            ref = w["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
            self.assertLess(float(np.abs(got - ref).max()), float(np.abs(ref).max()) / 100)
            self.assertEqual(t["contextual.projection.weight"].tensor_type, gguf.GGMLQuantizationType.F32)
            self.assertEqual(t["blk.0.ssm_norm.weight"].tensor_type, gguf.GGMLQuantizationType.F32)

    def test_tq4_pack_and_rtn(self):
        rng = np.random.default_rng(1)
        w = rng.standard_normal((8, 256)).astype(np.float32)
        codes, scales = converter.tq4_rtn(w)
        data = converter.tq4_pack(codes, scales)
        self.assertEqual(data.shape, (8, 2 * 66))
        # decode as the engine does: L[code] * scale (FP16), then R per 128
        sc = data.reshape(8, 2, 66)[..., :2].copy().view(np.float16).astype(np.float32)[..., 0]
        nib = data.reshape(8, 2, 66)[..., 2:]
        cd = np.stack([nib & 15, nib >> 4], -1).reshape(8, 2, 128)
        np.testing.assert_array_equal(cd.reshape(8, 256), codes)
        back = ((converter.TQ4_LEVELS[cd] * sc[..., None]) @ converter.TQ4_R).reshape(8, 256)
        self.assertLess(float(np.linalg.norm(back - w) / np.linalg.norm(w)), 0.1)

    @staticmethod
    def engine_tq4_tables():
        """TQ4 levels and Hadamard signs as the engine has them (pplx_embed.cu), not the converter's copies"""
        import re
        src = (ROOT / "src/models/qwen/qwen35/pplx_embed.cu").read_text()
        levels = np.array([float(x) for x in re.search(r"kTq4Levels\[16\] = \{([^}]*)\}", src).group(1).split(",")], dtype=np.float32)
        words = [int(x.strip().rstrip("u"), 16) for x in re.search(r"kTq4Signs\[4\] = \{([^}]*)\}", src).group(1).split(",")]
        signs = np.array([-1.0 if words[j // 32] >> (j % 32) & 1 else 1.0 for j in range(128)], dtype=np.float32)
        h = np.array([[1.0]], dtype=np.float32)
        while h.shape[0] < 128:
            h = np.block([[h, h], [h, -h]])
        return levels, h * signs[None, :] / np.sqrt(128.0)

    def engine_decode(self, data, rows, cols):
        levels, rot = self.engine_tq4_tables()
        blk = np.asarray(data, dtype=np.uint8).reshape(rows, cols // 128, 66)
        sc = blk[..., :2].copy().view(np.float16).astype(np.float32)[..., 0]
        nib = blk[..., 2:]
        cd = np.stack([nib & 15, nib >> 4], -1).reshape(rows, cols // 128, 128)
        return ((levels[cd] * sc[..., None]) @ rot).reshape(rows, cols), cd.reshape(rows, cols)

    def wide_checkpoint(self, root, **kw):
        global H, I
        saved = H, I
        H, I = 128, 256  # rows of 128+ columns, so the layer matrices take TQ4
        try:
            return checkpoint(root, **kw)
        finally:
            H, I = saved

    @NEEDS_GGUF
    def test_tq4_round_to_nearest_decodes_with_engine_tables(self):
        import gguf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = self.wide_checkpoint(root)
            converter.convert(root, root / "m.gguf", "tq4")
            t = read_gguf_tensors(root / "m.gguf")
            q = w["language_model.layers.0.linear_attn.in_proj_qkv.weight"]
            self.assertEqual(t["blk.0.attn_qkv.weight"][0], converter.TQ4)
            back, _ = self.engine_decode(t["blk.0.attn_qkv.weight"][2][:q.shape[0] * q.shape[1] // 128 * 66], *q.shape)
            self.assertLess(float(np.linalg.norm(back - q) / np.linalg.norm(q)), 0.1)
            self.assertEqual(t["token_embd.weight"][0], int(gguf.GGMLQuantizationType.Q8_0))
            self.assertEqual(t["blk.0.ssm_alpha.weight"][0], int(gguf.GGMLQuantizationType.BF16))  # in_proj_a stays BF16

    @NEEDS_GGUF
    @NEEDS_SAFETENSORS
    def test_tq4_codes_file_and_q8_keep(self):
        import gguf
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = self.wide_checkpoint(root)
            rng = np.random.default_rng(3)
            tensors, expect = {}, {}
            for name, v in w.items():
                base = name[:-len(".weight")]
                if v.ndim == 2 and ".layers." in name and v.shape[1] % 128 == 0 and not base.endswith(("in_proj_a", "in_proj_b")):
                    codes = rng.integers(0, 16, size=v.shape, dtype=np.uint8)
                    scale = rng.uniform(0.5, 2.0, size=(v.shape[0], v.shape[1] // 128)).astype(np.float32)
                    tensors[base + ".tq4_codes"] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
                    tensors[base + ".tq4_scale"] = scale
                    expect[base] = codes
            save_file(tensors, str(root / "codes.safetensors"))
            converter.convert(root, root / "m.gguf", "tq4", tq4_file=root / "codes.safetensors", q8_keep=("ffn_down",))
            t = read_gguf_tensors(root / "m.gguf")
            base = "language_model.layers.0.linear_attn.in_proj_qkv"
            r, c = expect[base].shape
            _, cd = self.engine_decode(t["blk.0.attn_qkv.weight"][2][:r * c // 128 * 66], r, c)
            np.testing.assert_array_equal(cd, expect[base])  # the file's codes, not round to nearest
            self.assertEqual(t["blk.0.ffn_down.weight"][0], int(gguf.GGMLQuantizationType.Q8_0))  # --q8-keep
            self.assertEqual(t["blk.0.ffn_gate.weight"][0], converter.TQ4)

            # codes whose shape is not the checkpoint matrix's are refused
            bad = dict(tensors)
            bad[base + ".tq4_codes"] = bad[base + ".tq4_codes"][:-1]
            save_file(bad, str(root / "bad.safetensors"))
            with self.assertRaisesRegex(ValueError, "TQ4 codes for .* do not match"):
                converter.convert(root, root / "bad.gguf", "tq4", tq4_file=root / "bad.safetensors")

    def test_reject_other_models_and_head_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root, kv_heads=KH + 1)
            with self.assertRaises(ValueError):
                converter.convert(root, root / "m.gguf")
            c = json.loads((root / "config.json").read_text()); c["model_type"] = "qwen3_5"
            (root / "config.json").write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                converter.convert(root, root / "m.gguf")


if __name__ == "__main__":
    unittest.main()
