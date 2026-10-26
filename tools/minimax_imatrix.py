#!/usr/bin/env python3
"""Importance matrix for MiniMax-M3.1-Flash from the HF oracle (tools/minimax_ref.py), one layer in memory at a time.

For every matrix tools/minimax_convert.cpp quantises: the mean of x^2 per input column over the calibration tokens; for
the routed experts per expert over the tokens routed to it ([experts][cols], an expert no token reached gets 1.0).
Written as the .dat file minimax_convert --imatrix reads: int32 entries; per entry int32 len, name, int32 ncall = 1,
int32 nval, float values (already the mean).

  minimax_imatrix.py CKPT --calib calibration.txt --chunks 8 --ctx 512 --out imatrix.dat
"""
import argparse
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_dequant import Checkpoint  # noqa: E402
from minimax_ref import P, layer_state, put, tensor  # noqa: E402


class Stats:
    def __init__(self):
        self.sum, self.n = {}, {}

    def add(self, name, x, part=None, parts=None):
        x = x.reshape(-1, x.shape[-1]).double()
        s = x.pow(2).sum(0)
        if parts is None:
            self.sum[name] = self.sum.get(name, 0) + s
            self.n[name] = self.n.get(name, 0) + x.shape[0]
            return
        if name not in self.sum:
            self.sum[name] = torch.zeros(parts, x.shape[1], dtype=torch.float64)
            self.n[name] = torch.zeros(parts, dtype=torch.float64)
        self.sum[name][part] += s
        self.n[name][part] += x.shape[0]

    def write(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<i", len(self.sum)))
            for name, s in self.sum.items():
                n = self.n[name]
                if isinstance(n, torch.Tensor):
                    mean = torch.where(n[:, None] > 0, s / n.clamp(min=1)[:, None], torch.ones_like(s))
                else:
                    mean = s / max(n, 1)
                v = mean.float().reshape(-1).numpy()
                b = name.encode()
                f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, v.size))
                f.write(v.astype("<f4").tobytes())


def hook(stats, names):
    def pre(_mod, args):
        for n in names:
            stats.add(n, args[0])
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0)
    a = ap.parse_args()
    import json
    from tokenizers import Tokenizer
    from transformers.masking_utils import create_causal_mask
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as mm
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig

    cfg = MiniMaxM3VLConfig(**json.load(open(os.path.join(a.ckpt, "config.json")))).text_config
    cfg._attn_implementation = "eager"
    nl = a.layers or cfg.num_hidden_layers
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    with torch.device("meta"):
        model = mm.MiniMaxM3VLTextModel(cfg)
    ck = Checkpoint(a.ckpt)
    tok = Tokenizer.from_file(os.path.join(a.ckpt, "tokenizer.json"))
    ids = tok.encode(open(a.calib, encoding="utf-8", errors="ignore").read()).ids
    need = a.chunks * a.ctx
    if len(ids) < need:
        raise SystemExit(f"calibration text has {len(ids)} tokens, {need} needed")
    ids = torch.tensor(ids[:need]).view(a.chunks, a.ctx)
    stats = Stats()

    E = cfg.num_local_experts
    cur = {}

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        d = cur["prefix"]
        for expert_idx in hit:
            e = int(expert_idx[0])
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            stats.add(d + "ffn_gate_exps.weight", x, e, E)
            stats.add(d + "ffn_up_exps.weight", x, e, E)
            act = self._apply_gate(F.linear(x, self.gate_up_proj[e]))
            stats.add(d + "ffn_down_exps.weight", act, e, E)
            out = F.linear(act, self.down_proj[e]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

    mm.MiniMaxM3VLExperts.forward = experts_forward

    emb = torch.from_numpy(ck.weight(P + "embed_tokens")[ids.numpy()]).to(dtype)
    pos = torch.arange(a.ctx)[None].expand(a.chunks, -1)
    pe = mm.MiniMaxM3VLRotaryEmbedding(config=cfg)(emb, position_ids=pos)
    mask = create_causal_mask(config=cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
    h = emb
    with torch.no_grad():
        for l in range(nl):
            t0 = time.time()
            layer = model.layers[l]
            shapes = {n: p.shape for n, p in list(layer.named_parameters()) + list(layer.named_buffers())}
            for name, value in layer_state(ck, l, cfg):
                local = name[len(f"layers.{l}."):]
                put(layer, local, tensor(value, shapes[local], dtype))
            d = f"blk.{l}."
            cur["prefix"] = d
            hooks = []
            att = layer.self_attn
            for mod, names in ((att.q_proj, [d + "attn_q.weight", d + "attn_k.weight", d + "attn_v.weight"]),
                               (att.o_proj, [d + "attn_output.weight"])):
                hooks.append(mod.register_forward_pre_hook(hook(stats, names)))
            if att.indexer is not None:
                hooks.append(att.indexer.q_proj.register_forward_pre_hook(
                    hook(stats, [d + "indexer.attn_q.weight", d + "indexer.attn_k.weight"])))
            mlp = layer.mlp
            if hasattr(mlp, "shared_experts"):
                sh = mlp.shared_experts
                hooks.append(sh.gate_up_proj.register_forward_pre_hook(
                    hook(stats, [d + "ffn_gate_shexp.weight", d + "ffn_up_shexp.weight"])))
                hooks.append(sh.down_proj.register_forward_pre_hook(hook(stats, [d + "ffn_down_shexp.weight"])))
            else:
                hooks.append(mlp.gate_up_proj.register_forward_pre_hook(hook(stats, [d + "ffn_gate.weight", d + "ffn_up.weight"])))
                hooks.append(mlp.down_proj.register_forward_pre_hook(hook(stats, [d + "ffn_down.weight"])))
            h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
            for hk in hooks:
                hk.remove()
            model.layers[l] = layer.to("meta")
            print(f"layer {l} {time.time() - t0:.0f} s", flush=True)
            stats.write(a.out)  # partial file after every layer: a crash keeps the layers done
        w = torch.from_numpy(ck.weight(P + "norm")).float()
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1.0 + w)
        if nl == cfg.num_hidden_layers:
            stats.add("output.weight", x.to(dtype))
    stats.write(a.out)
    print(f"wrote {a.out}: {len(stats.sum)} entries over {need} tokens")


if __name__ == "__main__":
    sys.exit(main())
