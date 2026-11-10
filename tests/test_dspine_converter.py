import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

try:
    import gguf  # noqa: F401  (the converter writes through gguf-py; CPU CI may not have it)
    HAVE_GGUF = True
except ImportError:
    HAVE_GGUF = False
NEEDS_GGUF = unittest.skipUnless(HAVE_GGUF, "gguf-py not installed")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dspine_converter", ROOT / "tools/convert/dflash/dspine_hf_to_gguf.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def fixture():
    config = {
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_target_layers": 4, "vocab_size": 32, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 16, "block_size": 4,
        "rope_theta": 1000000, "rms_norm_eps": 1e-6, "model_type": "qwen3",
        "dflash_config": {
            "draft_block_causal": True, "selected_prefix_terminal": False,
            "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
            "selected_prefix_feature_layers": [1, 2], "target_layer_ids": [1, 3], "mask_token_id": 31,
            "selected_prefix_code_bridge": {"code_size": 8, "rank": 4, "eps": 1e-6},
            "selected_prefix_last_write_refine": {"kind": "exact_sender", "head_topk": 16, "version": 1},
        },
    }
    shapes = {
        "fc.weight": [32, 64], "hidden_norm.weight": [32], "norm.weight": [32],
        "selected_prefix_bridge.codebook": [32, 8], "selected_prefix_bridge.gate.weight": [4, 8],
        "selected_prefix_bridge.gate.bias": [4], "selected_prefix_bridge.predecessor_gate.weight": [4, 8],
        "selected_prefix_bridge.prev.weight": [4, 8], "selected_prefix_bridge.out.weight": [8, 4],
        "selected_prefix_bridge.early_read_weights": [1, 8, 32], "selected_prefix_bridge.read_weight": [8, 32],
        "selected_prefix_bridge.feature_write_weights": [2, 32, 4],
    }
    for layer in range(2):
        for name, shape in {
            "input_layernorm.weight": [32], "post_attention_layernorm.weight": [32],
            "self_attn.q_norm.weight": [16], "self_attn.k_norm.weight": [16],
            "self_attn.q_proj.weight": [32, 32], "self_attn.k_proj.weight": [16, 32],
            "self_attn.v_proj.weight": [16, 32], "self_attn.o_proj.weight": [32, 32],
            "mlp.gate_proj.weight": [64, 32], "mlp.up_proj.weight": [64, 32], "mlp.down_proj.weight": [32, 64],
        }.items():
            shapes[f"layers.{layer}." + name] = shape
    values = {k: np.full(shape, 0x3f80, dtype="<u2") for k, shape in shapes.items()}
    values["selected_prefix_bridge.out.weight"][:] = 0
    for key in ("draft_block_causal_version", "last_write_refine_version",
                "selected_prefix_bridge.independent_write_version", "selected_prefix_bridge.terminal_disabled_version"):
        values[key] = np.array(1, dtype="<i8")
    header, payload = {}, bytearray()
    for name, value in values.items():
        begin = len(payload)
        payload.extend(value.tobytes())
        header[name] = {"dtype": "I64" if value.dtype.itemsize == 8 else "BF16", "shape": list(value.shape),
                        "data_offsets": [begin, len(payload)]}
    return config, header, values, payload


class TestDSpineConverter(unittest.TestCase):
    def test_complete_tensor_contract(self):
        config, header, _, _ = fixture()
        plan = converter.tensor_plan(header, config)
        outputs = [out for _, out, slices in plan if out] + [out for _, _, slices in plan if slices for _, out in slices]
        self.assertIn("blk.0.dspine_read.weight", outputs)
        self.assertIn("blk.1.dspine_write.weight", outputs)
        self.assertIn("dspine_codebook.weight", outputs)
        self.assertNotIn("selected_prefix_bridge.out.weight", outputs)
        self.assertFalse(any("conv" in k or "selector_hidden" in k for k in outputs))

    def test_reject_unsupported_variants(self):
        config, _, _, _ = fixture()
        for field, value in [("selected_prefix_terminal", True), ("draft_block_causal", False),
                             ("selected_prefix_feature_write_mode", "residual"),
                             ("selected_prefix_feature_layers", [1])]:
            broken = copy.deepcopy(config)
            broken["dflash_config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                converter.parameters(broken)

    def test_second_pass_validation(self):
        config, _, _, _ = fixture()
        config["code_bridge_second_pass"] = {"mode": "serial_table", "first_exact": 1, "head_topk": 16}
        converter.parameters(config)
        for field, value in [("mode", "parallel"), ("first_exact", 0), ("head_topk", 8)]:
            broken = copy.deepcopy(config)
            broken["code_bridge_second_pass"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                converter.parameters(broken)

    @NEEDS_GGUF
    def test_target_arch_override(self):
        import gguf
        config, header, _, payload = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            raw_header = json.dumps(header).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
            converter.convert(root, root / "draft.gguf", "qwen35")
            self.assertEqual(gguf.GGUFReader(str(root / "draft.gguf")).fields["dflash.target_architecture"].contents(), "qwen35")

    def test_reject_missing_shape_and_unknown_tensors(self):
        config, header, _, _ = fixture()
        for mode in ("missing", "shape", "extra"):
            broken = copy.deepcopy(header)
            if mode == "missing": del broken["selected_prefix_bridge.gate.weight"]
            elif mode == "shape": broken["selected_prefix_bridge.codebook"]["shape"][0] += 1
            else: broken["selector_hidden.weight"] = {"shape": [32, 32], "dtype": "BF16"}
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                converter.tensor_plan(broken, config)

    @NEEDS_GGUF
    def test_round_trip_bf16_and_layer_slices(self):
        import gguf
        config, header, _, payload = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config))
            raw_header = json.dumps(header).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
            converter.convert(root, root / "draft.gguf")
            result = gguf.GGUFReader(str(root / "draft.gguf"))
            tensors = {t.name: t for t in result.tensors}
            self.assertEqual(result.fields["dflash.dspine_version"].contents(), 1)
            self.assertEqual(list(tensors["blk.0.dspine_read.weight"].shape), [32, 8])
            self.assertEqual(list(tensors["blk.1.dspine_write.weight"].shape), [4, 32])
            for tensor in tensors.values():
                self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.BF16)
                np.testing.assert_array_equal(np.asarray(tensor.data).view(np.uint16), 0x3f80)
            # Release the mapped views before Windows removes the temporary file.
            del tensor, tensors, result

    @NEEDS_GGUF
    def test_version_and_nonzero_terminal_fail_before_output(self):
        for key in ("draft_block_causal_version", "selected_prefix_bridge.out.weight"):
            config, header, _, payload = fixture()
            offset = header[key]["data_offsets"][0]
            payload[offset:offset + 2] = b"\x00\x00" if key.endswith("version") else b"\x80\x3f"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(json.dumps(config))
                raw_header = json.dumps(header).encode()
                (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
                with self.assertRaises(ValueError): converter.convert(root, root / "draft.gguf")
                self.assertFalse((root / "draft.gguf").exists())


if __name__ == "__main__":
    unittest.main()
