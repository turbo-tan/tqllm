#!/usr/bin/env python3
"""Convert Tencent's independent-write DSpine checkpoint to a BF16 draft GGUF.

The drafter borrows the target's embedding and output head. No target weights or
tokenizer are copied, and bridge weights are never silently quantized.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def parameters(config):
    cfg = config.get("dflash_config", {})
    bridge = cfg.get("selected_prefix_code_bridge", {})
    layers = int(config["num_hidden_layers"])
    refine = cfg.get("selected_prefix_last_write_refine", {})
    if (not cfg.get("draft_block_causal") or cfg.get("selected_prefix_terminal", True)
            or cfg.get("selected_prefix_terminal_residual_rank", 0)
            or cfg.get("selected_prefix_feature_write_mode") != "independent"
            or cfg.get("selected_prefix_feature_writes") is not True
            or cfg.get("selected_prefix_feature_next_token_rows", False)
            or cfg.get("selected_prefix_feature_layers") != list(range(1, layers + 1))
            or refine.get("kind") != "exact_sender" or refine.get("version") != 1
            or float(refine.get("weight", 1)) != 1):
        raise ValueError("unsupported DSpine bridge or refinement configuration")
    # newer checkpoints also declare the serial second pass; when present it must match the refinement
    second = config.get("code_bridge_second_pass")
    if second is not None and (second.get("mode") != "serial_table"
                               or int(second.get("first_exact", -1)) != layers - 1
                               or int(second.get("head_topk", -1)) != int(refine["head_topk"])):
        raise ValueError("unsupported DSpine code_bridge_second_pass")
    h, c, r = int(config["hidden_size"]), int(bridge["code_size"]), int(bridge["rank"])
    topk = int(refine["head_topk"])
    eps = float(bridge["eps"])
    block = int(config["block_size"])
    ids = cfg["target_layer_ids"]
    if (min(h, c, r, layers) < 1 or topk != 16 or block < 2 or block > 16
            or not np.isfinite(eps) or eps <= 0 or not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(i, int) or i < 0 or i >= config["num_target_layers"] for i in ids)):
        raise ValueError("invalid DSpine dimensions, feature layers, or epsilon")
    return {"hidden": h, "code": c, "rank": r, "layers": layers, "topk": topk,
            "epsilon": eps, "block": block, "features": ids}


def tensor_plan(header, config):
    """Validate the entire tensor contract before creating the output file."""
    p = parameters(config)
    h, c, r, nl = (p[k] for k in ("hidden", "code", "rank", "layers"))
    v, f = int(config["vocab_size"]), int(config["intermediate_size"])
    q, k = (int(config[x]) * int(config["head_dim"]) for x in ("num_attention_heads", "num_key_value_heads"))
    hd = int(config["head_dim"])
    entries = {}
    for name, desc in header.items():
        if name == "__metadata__":
            continue
        key = name.removeprefix("model.")
        if key in entries:
            raise ValueError("duplicate normalized tensor " + key)
        entries[key] = (name, desc)
    plan, consumed = [], set()

    def use(key, shape, out=None, slices=None, dtype="BF16"):
        if key not in entries:
            raise ValueError("missing DSpine tensor " + key)
        original, desc = entries[key]
        if desc["shape"] != shape or desc["dtype"] != dtype:
            raise ValueError("invalid DSpine tensor shape/type: " + key)
        consumed.add(key)
        plan.append((original, out, slices))

    for key in ("draft_block_causal_version", "last_write_refine_version",
                "selected_prefix_bridge.independent_write_version", "selected_prefix_bridge.terminal_disabled_version"):
        use(key, [], dtype="I64")
    use("fc.weight", [h, len(p["features"]) * h], "fc.weight")
    use("hidden_norm.weight", [h], "enc.output_norm.weight")
    use("norm.weight", [h], "output_norm.weight")
    bridge = "selected_prefix_bridge."
    for source, shape, dest in (
        ("codebook", [v, c], "dspine_codebook.weight"),
        ("gate.weight", [r, c], "dspine_gate.weight"),
        ("gate.bias", [r], "dspine_gate.bias"),
        ("predecessor_gate.weight", [r, c], "dspine_pred_gate.weight"),
        ("prev.weight", [r, c], "dspine_prev.weight"),
        ("out.weight", [c, r], None),
    ):
        use(bridge + source, shape, dest)
    use(bridge + "early_read_weights", [nl - 1, c, h], slices=[
        (i, f"blk.{i}.dspine_read.weight") for i in range(nl - 1)])
    use(bridge + "read_weight", [c, h], f"blk.{nl - 1}.dspine_read.weight")
    use(bridge + "feature_write_weights", [nl, h, r], slices=[
        (i, f"blk.{i}.dspine_write.weight") for i in range(nl)])
    mapping = {
        "input_layernorm.weight": ("attn_norm.weight", [h]),
        "post_attention_layernorm.weight": ("ffn_norm.weight", [h]),
        "self_attn.q_norm.weight": ("attn_q_norm.weight", [hd]),
        "self_attn.k_norm.weight": ("attn_k_norm.weight", [hd]),
        "self_attn.q_proj.weight": ("attn_q.weight", [q, h]),
        "self_attn.k_proj.weight": ("attn_k.weight", [k, h]),
        "self_attn.v_proj.weight": ("attn_v.weight", [k, h]),
        "self_attn.o_proj.weight": ("attn_output.weight", [h, q]),
        "mlp.gate_proj.weight": ("ffn_gate.weight", [f, h]),
        "mlp.up_proj.weight": ("ffn_up.weight", [f, h]),
        "mlp.down_proj.weight": ("ffn_down.weight", [h, f]),
    }
    for i in range(nl):
        for key, (dest, shape) in mapping.items():
            use(f"layers.{i}." + key, shape, f"blk.{i}." + dest)
    extra = set(entries) - consumed
    if extra:
        raise ValueError("unrecognized DSpine tensors: " + ", ".join(sorted(extra)))
    return plan


def convert(directory, output, target_arch=None):
    import gguf
    root = Path(directory)
    cfg = json.loads((root / "config.json").read_text())
    source = root / "model.safetensors"
    with source.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        header = json.loads(stream.read(size))
    plan = tensor_plan(header, cfg)
    p = parameters(cfg)
    data_start = 8 + size

    def read(name):
        t = header[name]
        begin, end = t["data_offsets"]
        dtype = np.dtype("<i8" if t["dtype"] == "I64" else "<u2")
        if end - begin != int(np.prod(t["shape"], dtype=np.int64)) * dtype.itemsize:
            raise ValueError("invalid safetensors payload length: " + name)
        return np.memmap(source, dtype=dtype, mode="r", offset=data_start + begin, shape=tuple(t["shape"]))

    # Validate checkpoint markers and the disabled terminal before writing anything.
    for name, out, slices in plan:
        if out is None and slices is None:
            value = read(name)
            if name.endswith("version"):
                if int(value) != 1:
                    raise ValueError("unsupported DSpine format version: " + name)
            elif np.any(np.asarray(value) & 0x7fff):
                raise ValueError("terminal-disabled DSpine has a nonzero terminal projection")
    writer = gguf.GGUFWriter(str(output), "dflash", use_temp_file=True)
    metadata = {
        "embedding_length": p["hidden"], "block_count": p["layers"], "block_size": p["block"],
        "feed_forward_length": cfg["intermediate_size"], "attention.head_count": cfg["num_attention_heads"],
        "attention.head_count_kv": cfg["num_key_value_heads"], "attention.key_length": cfg["head_dim"],
        "attention.sliding_window": cfg.get("sliding_window") or 0, "selector_top_k": p["topk"],
        "dspine_version": 1, "target_block_count": cfg["num_target_layers"],
    }
    for key, value in metadata.items():
        writer.add_uint32("dflash." + key, int(value))
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]))
    writer.add_float32("dflash.rope.freq_base", float(cfg["rope_theta"]))
    writer.add_float32("dflash.dspine_epsilon", p["epsilon"])
    writer.add_bool("dflash.draft_block_causal", True)
    # the engine guards drafter/target pairing with this string; model_type is the drafter's own
    # block type and only coincides with the target arch for Qwen3 targets
    writer.add_string("dflash.target_architecture", target_arch or cfg["model_type"])
    writer.add_array("dflash.target_layers", p["features"])
    writer.add_uint32("tokenizer.ggml.mask_token_id", cfg["dflash_config"]["mask_token_id"])
    for name, out, slices in plan:
        if out is None and slices is None:
            continue
        value = read(name)
        for dest, array in ([(out, value)] if slices is None else [(dest, value[i]) for i, dest in slices]):
            writer.add_tensor(dest, np.asarray(array), raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-arch", help="target engine architecture (e.g. qwen35 for Qwen3.8-27B); defaults to the checkpoint model_type")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.target_arch)
