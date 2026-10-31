#!/usr/bin/env python3
"""MiniMax-M3.1-Flash first-N-layer oracle: HF transformers' MiniMax-M3 text model, cut to N layers, with weights from
our own dequant (tools/minimax_dequant.py). Writes the hidden state after every layer for the engine to match.

  minimax_ref.py CKPT --layers 4 --tokens 1,2,3 --out ref4.npz [--dtype bfloat16|float32]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_dequant import Checkpoint  # noqa: E402


P = "language_model.model."


def hf_state(ck, layers, cfg):
    """checkpoint names -> HF MiniMaxM3VLTextModel names (fused gate_up as HF stores them)."""
    yield "embed_tokens.weight", ck.weight(P + "embed_tokens")
    yield "norm.weight", ck.weight(P + "norm")
    for l in range(layers):
        yield from layer_state(ck, l, cfg)


def layer_state(ck, l, cfg):
    src, dst = f"{P}layers.{l}.", f"layers.{l}."
    for n in ("input_layernorm", "post_attention_layernorm"):
        yield dst + n + ".weight", ck.weight(src + n)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        yield f"{dst}self_attn.{n}.weight", ck.weight(f"{src}self_attn.{n}")
    if src + "self_attn.index_q_proj.weight" in ck:
        for n in ("q_proj", "k_proj", "q_norm", "k_norm"):
            yield f"{dst}self_attn.indexer.{n}.weight", ck.weight(f"{src}self_attn.index_{n}")
    moe = src + "block_sparse_moe."
    if moe + "gate.weight" in ck:
        yield dst + "mlp.gate.weight", ck.weight(moe + "gate")
        yield dst + "mlp.gate.e_score_correction_bias", ck.raw(moe + "e_score_correction_bias").astype(np.float32)
        sh = moe + "shared_experts."
        yield dst + "mlp.shared_experts.gate_up_proj.weight", np.concatenate(
            [ck.weight(sh + "gate_proj"), ck.weight(sh + "up_proj")])
        yield dst + "mlp.shared_experts.down_proj.weight", ck.weight(sh + "down_proj")
        E = cfg.num_local_experts
        yield dst + "mlp.experts.gate_up_proj", lambda e, m=moe: np.concatenate(
            [ck.weight(f"{m}experts.{e}.w1"), ck.weight(f"{m}experts.{e}.w3")])
        yield dst + "mlp.experts.down_proj", lambda e, m=moe: ck.weight(f"{m}experts.{e}.w2")
    else:
        yield dst + "mlp.gate_up_proj.weight", np.concatenate(
            [ck.weight(src + "mlp.gate_proj"), ck.weight(src + "mlp.up_proj")])
        yield dst + "mlp.down_proj.weight", ck.weight(src + "mlp.down_proj")


def put(model, name, t):
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name)
    if attr in dict(mod.named_buffers(recurse=False)):
        mod.register_buffer(attr, t.float(), persistent=True)
    else:
        setattr(mod, attr, torch.nn.Parameter(t, requires_grad=False))


def tensor(value, shape, dtype):
    if callable(value):
        full = torch.empty(shape, dtype=dtype)
        for e in range(shape[0]):
            full[e] = torch.from_numpy(value(e))
        return full
    return torch.from_numpy(np.ascontiguousarray(value)).to(dtype)


def stream(model, ck, cfg, dtype, a, mm):
    from transformers.masking_utils import create_causal_mask
    ids = torch.tensor([[int(x) for x in a.tokens.split(",")]])
    emb = torch.from_numpy(ck.weight(P + "embed_tokens")[ids[0].numpy()]).to(dtype)[None]
    pos = torch.arange(ids.shape[1])[None]
    rot = mm.MiniMaxM3VLRotaryEmbedding(config=cfg)
    pe = rot(emb, position_ids=pos)
    mask = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
    h, out = emb, {}
    with torch.no_grad():
        for l in range(cfg.num_hidden_layers):
            t0 = time.time()
            layer = model.layers[l]
            shapes = {n: p.shape for n, p in list(layer.named_parameters()) + list(layer.named_buffers())}
            for name, value in layer_state(ck, l, cfg):
                local = name[len(f"layers.{l}."):]
                put(layer, local, tensor(value, shapes[local], dtype))
            h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
            out[f"h{l + 1}"] = h[0].float().numpy()
            model.layers[l] = layer.to("meta")
            print(f"layer {l} rms_last {float(h[0, -1].float().pow(2).mean().sqrt()):.3f} {time.time() - t0:.0f} s", flush=True)
        w = torch.from_numpy(ck.weight(P + "norm")).float()
        x = h[0].float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1.0 + w)
        logits = x @ torch.from_numpy(ck.weight("language_model.lm_head")).float().T
    np.savez(a.out, tokens=ids[0].numpy(), logits=logits.numpy(), **out)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    for i in range(ids.shape[1]):
        top = logits[i].topk(5)
        print(i, repr(tok.decode([int(ids[0, i])])), "->", [(repr(tok.decode([int(t)])), round(float(v), 2)) for v, t in zip(*top)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--stream", action="store_true", help="all layers, one layer in memory at a time, + lm_head top-5")
    a = ap.parse_args()
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as mm
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig

    cfg = MiniMaxM3VLConfig(**json.load(open(os.path.join(a.ckpt, "config.json")))).text_config
    if not a.stream:
        cfg.num_hidden_layers = a.layers
    cfg._attn_implementation = "eager"
    dtype = getattr(torch, a.dtype)
    torch.set_default_dtype(dtype)
    with torch.device("meta"):
        model = mm.MiniMaxM3VLTextModel(cfg)
    ck = Checkpoint(a.ckpt)
    if a.stream:
        return stream(model, ck, cfg, dtype, a, mm)
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    t0 = time.time()
    loaded = set()
    for name, value in hf_state(ck, a.layers, cfg):
        target = params[name]
        t = tensor(value, target.shape, dtype)
        if tuple(t.shape) != tuple(target.shape):
            raise ValueError(f"{name}: {tuple(t.shape)} vs HF {tuple(target.shape)}")
        put(model, name, t)
        loaded.add(name)
    missing = [n for n in params if n not in loaded and not n.endswith("inv_freq")]
    if missing:
        raise ValueError(f"not loaded: {missing[:8]}")
    model.rotary_emb = mm.MiniMaxM3VLRotaryEmbedding(config=cfg)
    print(f"loaded {len(loaded)} tensors in {time.time() - t0:.0f} s", flush=True)
    ids = torch.tensor([[int(x) for x in a.tokens.split(",")]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = [h[0].float().numpy() for h in out.hidden_states]
    np.savez(a.out, tokens=ids[0].numpy(), **{f"h{i}": h for i, h in enumerate(hs)},
             final=out.last_hidden_state[0].float().numpy())
    for i, h in enumerate(hs):
        print(f"h{i} rms {np.sqrt((h ** 2).mean(-1)).round(3).tolist()}")


if __name__ == "__main__":
    sys.exit(main())
