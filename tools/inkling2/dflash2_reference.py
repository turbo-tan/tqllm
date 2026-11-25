"""Reference forward of incoai/Inkling-2-Mini-DFlash2 for tests/mlx_inkling2_dflash2.cpp.

The math follows sglang 0.5.21 srt/models/dflash.py (DFlash2DraftModel: DFlashGroupedConv / _grouped_conv, the decoder
layer's fused residual norms, CandidateSelector._score_edges + the greedy walk) with the drafter's Inkling patch
(sglang_inkling_dflash2.py: context features = the states entering applications 5 / 23 / 41 and the final-norm output,
selector softcap after the edge correction, candidates over the real vocabulary), and the draft block embedded as the
target's input (embedding + embed_norm). Weights BF16 as stored; activations rounded to BF16 where sglang's BF16 model
stores them, FP32 math in between. CPU only.

Usage: python dflash2_reference.py DRAFTER_DIR INKLING_CHECKPOINT_DIR OUT.bin
Context taps: the target's normalised input rows (embedding + embed_norm) of TOKENS, the same row at all four tap
slots (in-distribution enough that BF16 rounding does not flip the candidates; random taps did), reproduced by the C++
test; the anchor is the token after them.
"""
import json, struct, sys
import numpy as np
import torch
from safetensors import safe_open

# "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + ..." (Inkling tokenizer)
TOKENS = [1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 150003, 149934, 196982, 220, 17, 25, 198, 149860, 149862, 150583,
          149934, 198, 149862, 150583, 220]
M, ANCHOR, VALID_VOCAB = len(TOKENS), 46389, 200058


def bf(x):
    return x.to(torch.bfloat16).to(torch.float32)


def rms(x, w, eps):
    x = x.float()
    return bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def linear(x, w):
    return bf(bf(x) @ w.float().T)


def rope(x, pos, theta):  # x [n, heads, hd], NeoX halves
    n, _, hd = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.arange(pos, pos + n, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    a, b = x[..., : hd // 2], x[..., hd // 2:]
    return bf(torch.cat([a * cos - b * sin, b * cos + a * sin], -1))


def grouped_conv(h, delta, base, groups, gsize):  # sglang _grouped_conv, one block, taps 2
    blocks = h.unflatten(-1, (groups, gsize))
    coef = base.view(1, 2, groups, gsize) + delta.unsqueeze(-1)
    out = coef[:, 0] * blocks
    shifted = torch.nn.functional.pad(blocks[:-1], (0, 0, 0, 0, 1, 0))
    out = out + coef[:, 1] * shifted
    return bf(out.flatten(-2))


def main():
    ddir, cdir, out_path = sys.argv[1:4]
    cfg = json.load(open(f"{ddir}/config.json"))
    dc = cfg["dflash_config"]
    D, H, G, HD, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_hidden_layers"]
    eps, theta, gsize, block = cfg["rms_norm_eps"], cfg["rope_parameters"]["rope_theta"], dc["conv_group_size"], dc["block_size"]
    groups = D // gsize
    assert dc["target_layer_ids"] == [5, 23, 41, 48] and dc["conv_kernel_size"] == 2 and dc["selector_softcap_mode"] == "after_correction"
    W = {}
    with safe_open(f"{ddir}/model.safetensors", "pt") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).float()
    idx = json.load(open(f"{cdir}/model.safetensors.index.json"))["weight_map"]
    T = {}
    for name in ("model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight"):
        with safe_open(f"{cdir}/{idx[name]}", "pt") as f:
            T[name] = f.get_tensor(name)
    rows = rms(T["model.llm.embed.weight"][torch.tensor(TOKENS)].float(), T["model.llm.embed_norm.weight"], eps)
    taps = rows.repeat(1, 4)
    # context rows -> per layer K / V at positions 0 .. M - 1
    ctx = rms(linear(taps, W["fc.weight"]), W["hidden_norm.weight"], eps)
    ids = torch.tensor([ANCHOR] + [dc["mask_token_id"]] * (block - 1))
    x = rms(T["model.llm.embed.weight"][ids].float(), T["model.llm.embed_norm.weight"], eps)
    residual = None
    for l in range(L):
        p = f"layers.{l}."
        kc = linear(ctx, W[p + "self_attn.k_proj.weight"]).view(M, G, HD)
        vc = linear(ctx, W[p + "self_attn.v_proj.weight"]).view(M, G, HD)
        kc = rope(rms(kc, W[p + "self_attn.k_norm.weight"], eps), 0, theta)
        if residual is None:
            residual = x
        else:
            residual = bf(x + residual)
        h = rms(residual, W[p + "input_layernorm.weight"], eps)
        coef = linear(h, W[p + "attention_conv.kernel_projection.weight"]).view(block, 2, 2, groups)
        base = W[p + "attention_conv.base_kernel"]
        h = grouped_conv(h, coef[:, 0], base[0], groups, gsize)
        q = linear(h, W[p + "self_attn.q_proj.weight"]).view(block, H, HD)
        k = linear(h, W[p + "self_attn.k_proj.weight"]).view(block, G, HD)
        v = linear(h, W[p + "self_attn.v_proj.weight"]).view(block, G, HD)
        q = rope(rms(q, W[p + "self_attn.q_norm.weight"], eps), M, theta)
        k = rope(rms(k, W[p + "self_attn.k_norm.weight"], eps), M, theta)
        K, V = torch.cat([kc, k], 0), torch.cat([vc, v], 0)  # [M + block, G, HD], not causal
        K, V = K.repeat_interleave(H // G, 1), V.repeat_interleave(H // G, 1)
        s = torch.einsum("qhd,khd->hqk", q, K) / HD ** 0.5
        a = bf(torch.einsum("hqk,khd->qhd", s.softmax(-1), V)).reshape(block, H * HD)
        a = linear(a, W[p + "self_attn.o_proj.weight"])
        a = grouped_conv(a, coef[:, 1], base[1], groups, gsize)
        residual = bf(a + residual)
        h = rms(residual, W[p + "post_attention_layernorm.weight"], eps)
        coef = linear(h, W[p + "mlp_conv.kernel_projection.weight"]).view(block, 2, 2, groups)
        base = W[p + "mlp_conv.base_kernel"]
        h = grouped_conv(h, coef[:, 0], base[0], groups, gsize)
        g, u = linear(h, W[p + "mlp.gate_proj.weight"]), linear(h, W[p + "mlp.up_proj.weight"])
        h = linear(bf(bf(torch.nn.functional.silu(g)) * u), W[p + "mlp.down_proj.weight"])
        x = grouped_conv(h, coef[:, 1], base[1], groups, gsize)
    hid = rms(bf(x + residual), W["norm.weight"], eps)
    logits = hid @ T["model.llm.unembed.weight"][:VALID_VOCAB].float().T
    unary, cand = logits.topk(16, -1)  # raw: the softcap comes after the edge correction (monotone for greedy)
    gate = linear(hid, W["candidate_selector.hidden_projection.weight"])
    pred, succ = W["candidate_selector.predecessor_codebook"], W["candidate_selector.successor_codebook"]
    prev, tokens = ANCHOR, []
    for e in range(1, block):  # row e predicts the token after e - 1 drafts
        score = unary[e] + ((pred[prev] * gate[e])[None] * succ[cand[e]]).sum(-1)
        prev = int(cand[e][int(score.argmax())])
        tokens.append(prev)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<8i", 0x32464C44, M, ANCHOR, VALID_VOCAB, block, D, 16, 0))
        f.write(np.asarray(tokens, np.uint32).tobytes())
        f.write(cand.numpy().astype(np.uint32).tobytes())
        f.write(unary.numpy().astype(np.float32).tobytes())
        f.write(hid.numpy().astype(np.float32).tobytes())
    print("tokens", tokens)


if __name__ == "__main__":
    main()
