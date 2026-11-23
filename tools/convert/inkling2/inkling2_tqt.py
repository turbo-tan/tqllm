#!/usr/bin/env python3
"""TQT quantiser for the routed experts of Inkling-2-Mini (transformers PR #51050 reference, experts dequantised to BF16).

  encode   one reference forward over the calibration text per group of layers, per-expert Hessians accumulated in the
           experts' forward hook (summed over the layer's 1-3 visits), then BlockLDLQ + tail-biting Viterbi (tqt.py) of
           every expert's gate+up and down into enc/L<layer>_gu.bin / _dn.bin (66 B per 128 weights, rotated basis)
  pack     the encoded layers as one overlay GGUF (tqt_experts.gguf, engine names blk.<l>.ffn_{gate,up,down}_exps.weight, type
           306) that the engine loads next to the checkpoint folder
  sweep    each layer alone in one code against the gate (the sensitivity behind the 16 GB tier's mix of 4-bit and 2-bit layers)
  eval     fake-quant gate: the encoded experts decoded into the reference in memory, scored against
           golden/mt/lp_bf16.f16 (full-vocabulary KLD, top-1)

The routed experts are 92 % of the bytes; every other tensor is left as the checkpoint has it. Needs PyTorch with CUDA,
nvcc on PATH (tqt_viterbi.cu), transformers from the checkpoint's pylib.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen35"))
import tqt  # noqa: E402
import tqt2  # noqa: E402
import tqv12  # noqa: E402

D, I, E = 2304, 1152, 128
BLOCK = 128
CODECS = {"tqt": tqt, "tqt2": tqt2}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def parse_layers(spec):
    out = []
    for part in spec.split("+"):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


# ---------------- Hessian accumulation ----------------
class ExpertStats:
    """Per-expert second moments of one layer's inputs, summed over all of its visits.
    Hgu[e] = sum_t w_t^2 x_t x_t^T over the tokens routed to e (x: the expert's input, w: its routing weight),
    Hdn[e] the same for the input of the down projection (the activation), S[e] = sum w_t^2, n[e] = tokens,
    Hg = sum x x^T over all tokens (the layer-wide input, for experts with few tokens)."""

    def __init__(self, dev, d=D, i=I, e=E):
        import torch
        self.Hgu = torch.zeros(e, d, d, device=dev)
        self.Hdn = torch.zeros(e, i, i, device=dev)
        self.S = torch.zeros(e, device=dev, dtype=torch.float64)
        self.n = torch.zeros(e, device=dev, dtype=torch.int64)
        self.Hg = torch.zeros(d, d, device=dev)
        self.N = 0

    def add(self, hs, idx, wts, w_gu, w_dn, act):
        """hs [T, D] (the experts' input), idx [T, k] expert ids, wts [T, k] routing weights, w_gu [E, 2I, D] (gate rows
        first), w_dn [E, D, I], act(gate_up) -> [n, I]: the module's own arithmetic (BF16)"""
        import torch
        import torch.nn.functional as F
        x = hs.float()
        self.Hg += x.T @ x
        self.N += hs.shape[0]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.n.shape[0]).tolist()
        tok = order // idx.shape[1]
        w_all = wts.reshape(-1)[order].float()
        off = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            t, w = tok[off:off + c], w_all[off:off + c]
            off += c
            xe = hs[t]
            a = act(F.linear(xe, w_gu[e]))
            xw = xe.float() * w[:, None]
            aw = a.float() * w[:, None]
            self.Hgu[e] += xw.T @ xw
            self.Hdn[e] += aw.T @ aw
            self.S[e] += (w.double() ** 2).sum()
            self.n[e] += c


def blend_gu(He, Se, Hg_norm, n0):
    """normalised per-token H of an expert's gate / up input: its own moments shrunk towards the layer-wide ones
    by n0 / (S + n0) (an expert with no tokens gets the layer-wide H)"""
    if Se <= 0:
        return Hg_norm.clone()
    a = Se / (Se + n0)
    return (He / Se) * a + Hg_norm * (1.0 - a)


def final_dn(Hd, Se, nmin):
    """normalised H of the down input; an expert with fewer than nmin weighted tokens gets the identity (damping only)"""
    import torch
    if Se < nmin:
        return torch.eye(Hd.shape[0], device=Hd.device, dtype=Hd.dtype)
    return Hd / Se


# ---------------- encode ----------------
def encode_matrix(W, H, damp, nscales, codec, order=True, refit=True):
    """W [N, K] float32 CUDA (original basis), H [K, K] float32 (original basis) -> packed rows uint8 [N, K/128*bytes], rel err"""
    Wr = tqv12.rot_last(W)
    Hr = tqv12.rotate_h(H)
    s, st, Q = tqt.gptq(Wr, Hr, damp=damp, gammas=codec.gammas(nscales), damp_form="median", order=order, refit=refit, codec=codec)
    err = tqv12.rel_err(Q, Wr, Hr)
    return codec.pack(s.cpu().numpy(), st.cpu().numpy()), err


def encode_layer(stats, w_gu, w_dn, args, codec, damp_gu, damp_dn):
    Hg_norm = stats.Hg / max(stats.N, 1)
    gu = np.empty((E, 2 * I, (D // BLOCK) * codec.BLOCK_BYTES), np.uint8)
    dn = np.empty((E, D, (I // BLOCK) * codec.BLOCK_BYTES), np.uint8)
    errs = []
    t0 = time.time()
    for e in range(E):
        Se = float(stats.S[e])
        H = blend_gu(stats.Hgu[e], Se, Hg_norm, args.n0)
        gu[e], eg = encode_matrix(w_gu[e].float(), H, damp_gu, args.scales, codec)
        H = final_dn(stats.Hdn[e], Se, args.nmin)
        dn[e], ed = encode_matrix(w_dn[e].float(), H, damp_dn, args.scales, codec)
        errs.append((eg, ed))
        if e % 32 == 31:
            log(f"  {codec.__name__} expert {e + 1}/{E}: {time.time() - t0:.0f} s, rel err gu {np.mean([x[0] for x in errs]):.5f} down {np.mean([x[1] for x in errs]):.5f}")
    return gu, dn, errs


def load_model(model_dir):
    sys.path.insert(0, str(Path(model_dir) / "pylib"))
    import torch
    import transformers.models.inkling2_loop.modeling_inkling2_loop as M
    from transformers import Inkling2LoopForConditionalGeneration
    model = Inkling2LoopForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    mods = {}
    for name, mod in model.named_modules():
        if isinstance(mod, M.Inkling2LoopExperts):
            mods[int(re.search(r"layers\.(\d+)\.", name).group(1))] = mod
    assert sorted(mods) == list(range(24)), sorted(mods)
    return model, mods


def calib_ids(tok, wiki, srcdir, ntok, seqlen, seed=0):
    """half WikiText-2 train passages, half this repository's source files, seqlen tokens at most each"""
    rng = np.random.RandomState(seed)
    lines = Path(wiki).read_text().splitlines()
    passages, cur = [], ""
    for ln in lines:
        s = ln.strip()
        if s.startswith("= ") and not s.startswith("= = "):
            cur = ""
            continue
        if not s or s.startswith("= ="):
            continue
        cur = (cur + "\n" + s).strip()
        if len(cur) > 3500:
            passages.append(cur)
            cur = ""
    files = sorted(p for p in Path(srcdir).rglob("*") if p.suffix in (".cpp", ".cu", ".hpp", ".cuh", ".py", ".md") and p.stat().st_size > 2000)
    rng.shuffle(passages)
    rng.shuffle(files)
    out, tw, tc = [], 0, 0
    half = ntok // 2
    for p in passages:
        if tw >= half:
            break
        ids = tok(p)["input_ids"][:seqlen]
        out.append(ids)
        tw += len(ids)
    for f in files:
        if tc >= half:
            break
        ids = tok(f.read_text(errors="ignore"))["input_ids"][:seqlen]
        out.append(ids)
        tc += len(ids)
    log(f"calibration: {len(out)} sequences, {tw} wiki + {tc} code tokens")
    return out


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _dequant_experts(weight, scale, amax=None, denominator=None):
    """transformers PR #51050 Inkling2LoopDequantizeExperts without transformers: NVFP4 (amax given: two E2M1 per byte, low nibble
    first, E4M3 scale per 16, per-expert factor amax / denominator) or MXFP8 (E4M3 values, 2 ** (scale - 127) per 32) -> BF16"""
    import torch
    dev = weight.device
    E_, rows = weight.shape[:2]
    cols = weight.shape[2] * 2 if amax is not None else weight.shape[2]
    out = torch.empty(E_, rows, cols, dtype=torch.bfloat16, device=dev)
    if amax is not None:
        e2m1 = torch.tensor(E2M1, device=dev)
        byte = torch.arange(256, device=dev)
        b2v = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        factor = amax.to(torch.bfloat16).float() / denominator.to(torch.bfloat16).float()  # the loader casts every source to the param dtype
    for st in range(0, E_, 8):
        ex = slice(st, st + 8)
        bs = scale[ex].float()
        if amax is None:
            v = weight[ex].float().view(*bs.shape, 32) * torch.exp2(bs - 127)[..., None]
        else:
            v = b2v[weight[ex].to(torch.uint8).long()].view(*bs.shape, 16) * bs[..., None] * factor[ex, None, None, None]
        out[ex] = v.reshape(-1, rows, cols).to(torch.bfloat16)
    return out


def official_experts(model_dir, l, dev="cpu"):
    """layer l's experts as the reference loads them (the checkpoint's NVFP4 / MXFP8 dequantised to BF16, w13 de-interleaved to
    gate rows then up rows): (gate_up [E, 2I, D], down [E, D, I]); needs no model and no transformers"""
    from safetensors import safe_open
    f = safe_open(str(Path(model_dir) / "model-00001-of-00001.safetensors"), "pt", device=dev)
    keys = set(f.keys())
    out = {}
    for w in ("w13", "w2"):
        base = f"model.llm.layers.{l}.mlp.experts.{w}_weight"
        t = {k: f.get_tensor(base + k) if base + k in keys else None for k in ("", "_scale", "_amax", "_outer_scale_denominator")}
        x = _dequant_experts(t[""], t["_scale"], t["_amax"], t["_outer_scale_denominator"])
        if w == "w13":  # rows alternate gate / up: de-interleave (Interleave(dim=1))
            x = x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2]).transpose(1, 2).reshape(x.shape).contiguous()
        out[w] = x
    return out["w13"], out["w2"]


def calibrate(model, mods, group, seqs):
    import torch
    stats = {l: ExpertStats("cuda") for l in group}
    hooks = []
    for l in group:
        def pre(module, a, kw, l=l):
            hs, ti, tw = (list(a) + [kw.get(k) for k in ("hidden_states", "top_k_index", "top_k_weights")][len(a):])[:3]
            with torch.no_grad():
                stats[l].add(hs, ti, tw, module.gate_up_proj, module.down_proj, module._apply_gate)

        hooks.append(mods[l].register_forward_pre_hook(pre, with_kwargs=True))
    t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(seqs):
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
            if i % 32 == 31:
                log(f"  calibration {i + 1}/{len(seqs)} {time.time() - t0:.0f} s")
    for h in hooks:
        h.remove()
    for l in group:
        s = stats[l]
        log(f"layer {l}: tokens {s.N}, expert tokens min {int(s.n.min())} median {int(s.n.median())} max {int(s.n.max())}, "
            f"experts below 256: {int((s.n < 256).sum())}")
    return stats


def cmd_hessian(args):
    """calibrate and save the per-expert Hessians of some layers (hess/L<l>.pt, 3.4 GB each) so another machine can encode them"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    hd = Path(args.out) / "hess"
    hd.mkdir(parents=True, exist_ok=True)
    todo = [l for l in parse_layers(args.layers) if not (hd / f"L{l:02d}.pt").exists()]
    if not todo:
        return
    model, mods = load_model(args.model)
    for l in todo:  # the model-free loader must equal the reference's own weights (NVFP4 and MXFP8 layers)
        gu, dn = official_experts(args.model, l, "cuda")
        assert torch.equal(gu, mods[l].gate_up_proj.data) and torch.equal(dn, mods[l].down_proj.data), f"official_experts != the loaded model, layer {l}"
        del gu, dn
    from transformers import AutoTokenizer
    seqs = calib_ids(AutoTokenizer.from_pretrained(args.model), args.wiki, args.srcdir, args.tokens, args.seqlen)
    for g in range(0, len(todo), args.group):
        group = todo[g:g + args.group]
        stats = calibrate(model, mods, group, seqs)
        for l in group:
            s = stats[l]
            torch.save({"Hgu": s.Hgu.cpu(), "Hdn": s.Hdn.cpu(), "S": s.S.cpu(), "n": s.n.cpu(), "Hg": s.Hg.cpu(), "N": s.N}, str(hd / f"L{l:02d}.pt.tmp"))
            os.replace(str(hd / f"L{l:02d}.pt.tmp"), hd / f"L{l:02d}.pt")
            log(f"saved Hessians of layer {l}")
        del stats
        torch.cuda.empty_cache()


def cmd_encode(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = Path(args.out)
    names = args.codec.split(",")
    dgu, ddn = [float(x) for x in args.damp_gu.split(",")], [float(x) for x in args.damp_dn.split(",")]
    assert len(names) == len(dgu) == len(ddn), "one --damp-gu / --damp-dn value per codec"
    encdirs = {n: outdir / ("enc" if n == "tqt" else "enc2") for n in names}
    for d in encdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    need = {l: [n for n in names if not (encdirs[n] / f"L{l:02d}_dn.bin").exists()] for l in parse_layers(args.layers)}
    todo = [l for l, v in need.items() if v]
    log("layers to encode:", {l: need[l] for l in todo})
    if not todo:
        return

    def encode_one(l, s, w_gu, w_dn):
        for ci, n in enumerate(names):
            if n not in need[l]:
                continue
            t0 = time.time()
            with torch.no_grad():
                gu, dn, errs = encode_layer(s, w_gu, w_dn, args, CODECS[n], dgu[ci], ddn[ci])
            for name, arr in (("gu", gu), ("dn", dn)):
                p = encdirs[n] / f"L{l:02d}_{name}.bin"
                arr.tofile(str(p) + ".tmp")
                if name == "gu":
                    json.dump({"n": s.n.tolist(), "S": s.S.tolist(), "err": errs, "seconds": time.time() - t0}, open(encdirs[n] / f"L{l:02d}.json", "w"))
                os.replace(str(p) + ".tmp", p)
            log(f"layer {l} {n} done in {time.time() - t0:.0f} s, mean rel err gu {np.mean([x[0] for x in errs]):.5f} down {np.mean([x[1] for x in errs]):.5f}")

    if args.from_h:  # Hessians dumped by `hessian` (another machine), weights straight from the checkpoint: no reference model
        for l in todo:
            d = torch.load(Path(args.from_h) / f"L{l:02d}.pt")
            s = ExpertStats("cuda")
            s.Hgu, s.Hdn, s.Hg = d["Hgu"].cuda(), d["Hdn"].cuda(), d["Hg"].cuda()
            s.S, s.n, s.N = d["S"].cuda(), d["n"].cuda(), d["N"]
            gu, dn = official_experts(args.model, l, "cpu")
            encode_one(l, s, gu.cuda(), dn.cuda())
            del s, d, gu, dn
            torch.cuda.empty_cache()
        return
    model, mods = load_model(args.model)
    from transformers import AutoTokenizer
    seqs = calib_ids(AutoTokenizer.from_pretrained(args.model), args.wiki, args.srcdir, args.tokens, args.seqlen)
    for g in range(0, len(todo), args.group):
        group = todo[g:g + args.group]
        log("group", group)
        stats = calibrate(model, mods, group, seqs)
        for l in group:
            encode_one(l, stats[l], mods[l].gate_up_proj, mods[l].down_proj)
        del stats
        torch.cuda.empty_cache()


# ---------------- eval ----------------
def decode_rows_t(raw, N, K, codec, dev="cuda"):
    """encoded rows (uint8 numpy [n, N, K/128 * bytes]) -> BF16 CUDA tensor [n, N, K] in the original basis; the GPU twin of
    codec.decode followed by the un-rotation (w = q R)"""
    import torch
    nb, bb = K // BLOCK, codec.BLOCK_BYTES
    R = torch.from_numpy(tqv12.rot_matrix_np().astype(np.float32)).to(dev)
    tab = codec._table_t(dev)  # noqa: SLF001
    b = torch.from_numpy(np.ascontiguousarray(raw)).to(dev).reshape(raw.shape[0], N, nb, bb)
    sc = b[..., :2].contiguous().view(torch.float16).float()  # [n, N, nb, 1]
    v = tab[codec.windows_t(b[..., 2:])]  # [n, N, nb, 64, 2]
    q = (v.reshape(raw.shape[0], N, nb, BLOCK) * sc)
    return (q @ R).reshape(raw.shape[0], N, K).to(torch.bfloat16)


def decode_layer(encdir, l, codec=tqt):
    """-> (gate_up [E, 2I, D], down [E, D, I]) BF16 CUDA tensors in the original basis, from the encoded files"""
    import torch
    out = []
    for name, (N, K) in (("gu", (2 * I, D)), ("dn", (D, I))):
        raw = np.fromfile(Path(encdir) / f"L{l:02d}_{name}.bin", np.uint8).reshape(E, N, -1)
        out.append(torch.cat([decode_rows_t(raw[e0:e0 + 16], N, K, codec) for e0 in range(0, E, 16)]))
    return out


class Gate:
    """the held-out gate: golden/mt texts, the reference's full-vocabulary log-probabilities (lp_bf16.f16) and top-1 ids"""

    def __init__(self, model_dir):
        self.texts = json.load(open(Path(model_dir) / "golden/mt/texts.json"))
        self.top1 = np.fromfile(Path(model_dir) / "golden/mt/top1_bf16.i32", np.int32)
        self.lp = np.memmap(Path(model_dir) / "golden/mt/lp_bf16.f16", np.float16, "r")
        self.V = 200058

    def score(self, model, verbose=True):
        import torch
        pos = kl_sum = agree = n = 0
        with torch.no_grad():
            for t in self.texts:
                ids = torch.tensor([t["ids"]], device="cuda")
                T = ids.shape[1]
                lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)
                assert lp.shape[-1] == self.V, lp.shape
                ref = torch.from_numpy(np.array(self.lp[pos * self.V:(pos + T) * self.V]).reshape(T, self.V)).to("cuda").float()
                kl = (ref.exp() * (ref - lp)).sum(-1)
                kl_sum += kl.sum().item()
                agree += (lp.argmax(-1).cpu().numpy() == self.top1[pos:pos + T]).sum()
                n += T
                pos += T
                if verbose:
                    log(f"{t['name']:14s} T {T:4d} KLD {kl.mean().item():.5f}")
        return kl_sum / n, 100.0 * agree / n, n


def substitute(mods, l, codec, encdir):
    """decoded experts of layer l into the reference in memory; returns the originals"""
    import torch
    gu, dn = decode_layer(encdir, l, codec)
    m = mods[l]
    saved = (m.gate_up_proj.data.clone(), m.down_proj.data.clone())
    m.gate_up_proj.data.copy_(gu)
    m.down_proj.data.copy_(dn)
    return saved


def cmd_eval(args):
    model, mods = load_model(args.model)
    gate = Gate(args.model)
    layers = [(l, tqt, "enc") for l in parse_layers(args.layers)] + [(l, tqt2, "enc2") for l in (parse_layers(args.layers2) if args.layers2 else [])]
    for l, codec, sub in layers:
        substitute(mods, l, codec, Path(args.out) / sub)
        log("substituted layer", l, codec.__name__)
    kld, top1, n = gate.score(model)
    log(f"ALL {n} positions: KLD vs lp_bf16 {kld:.5f}, top-1 {top1:.2f} %, layers {args.layers} (tqt) {args.layers2 or '-'} (tqt2)")


def cmd_compare(args):
    """one layer encoded in several variants (--out dirs, comma separated; each holds enc/): KLD / top-1 of that layer alone"""
    model, mods = load_model(args.model)
    gate = Gate(args.model)
    l = int(args.layers)
    for d in args.out.split(","):
        saved = substitute(mods, l, tqt, Path(d) / "enc")
        kld, top1, _ = gate.score(model, verbose=False)
        mods[l].gate_up_proj.data.copy_(saved[0])
        mods[l].down_proj.data.copy_(saved[1])
        log(f"layer {l} alone, {d}: KLD {kld:.5f}, top-1 {top1:.2f} %")


def cmd_sweep(args):
    """KLD / top-1 of each layer alone in the given code (the rest as the checkpoint): the sensitivity of the tiers' allocation"""
    model, mods = load_model(args.model)
    gate = Gate(args.model)
    codec, sub = (tqt2, "enc2") if args.code == "tqt2" else (tqt, "enc")
    res = {}
    for l in parse_layers(args.layers):
        saved = substitute(mods, l, codec, Path(args.out) / sub)
        kld, top1, _ = gate.score(model, verbose=False)
        mods[l].gate_up_proj.data.copy_(saved[0])
        mods[l].down_proj.data.copy_(saved[1])
        res[l] = (kld, top1)
        log(f"layer {l} alone in {args.code}: KLD {kld:.5f}, top-1 {top1:.2f} %")
        json.dump(res, open(Path(args.out) / f"sweep_{args.code}.json", "w"))

# ---------------- pack ----------------
def _gstr(b):
    return np.uint64(len(b)).tobytes() + b


def q8_0(x):
    """float32 [rows, cols] (cols % 32 == 0) -> bytes of ggml Q8_0 blocks (FP16 scale + 32 int8 per 32 values, row-major)"""
    xb = np.ascontiguousarray(x, np.float32).reshape(-1, 32)
    amax = np.abs(xb).max(1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.clip(np.sign(xb) * np.floor(np.abs(xb * inv[:, None]) + 0.5), -127, 127).astype(np.int8)
    return np.concatenate([d.astype("<f2").view(np.uint8).reshape(-1, 2), q.view(np.uint8)], 1).tobytes()


def dense_q8_tensors(model_dir, layers=24, out_vocab=200058):
    """[(engine name, ne, Q8_0 bytes)] of the embedding, the LM head and the attention q / k / v / o matrices of the checkpoint"""
    sys.path.insert(0, str(Path(model_dir) / "pylib"))
    import torch
    from safetensors import safe_open
    f = safe_open(str(Path(model_dir) / "model-00001-of-00001.safetensors"), "pt")
    names = [("token_embd.weight", "model.llm.embed.weight", None), ("output.weight", "model.llm.unembed.weight", out_vocab)]
    for l in range(layers):
        for en, hn in (("attn_q", "wq_du"), ("attn_k", "wk_dv"), ("attn_v", "wv_dv"), ("attn_output", "wo_ud")):
            names.append((f"blk.{l}.{en}.weight", f"model.llm.layers.{l}.attn.{hn}.weight", None))
    out = []
    for en, hn, rows in names:
        t = f.get_tensor(hn)
        if rows:
            t = t[:rows]
        assert t.dtype == torch.bfloat16 and t.shape[1] % 32 == 0, (hn, t.dtype, t.shape)
        out.append((en, (t.shape[1], t.shape[0]), q8_0(t.float().numpy())))
    return out


def write_overlay(path, layers, dense=()):
    """GGUF v3, alignment 32: tensors [cols, rows, E] per layer and projection, type 306 (tqt, enc dir) or 307 (tqt2, enc2 dir);
    layers = [(layer, encdir, codec)] (stream bytes as encoded)"""
    infos, blobs, off = [], [], 0
    for l, encdir, codec in layers:
        gu = np.fromfile(Path(encdir) / f"L{l:02d}_gu.bin", np.uint8).reshape(E, 2 * I, -1)
        dn = np.fromfile(Path(encdir) / f"L{l:02d}_dn.bin", np.uint8).reshape(E, D, -1)
        for name, arr, ne in ((f"blk.{l}.ffn_gate_exps.weight", gu[:, :I], (D, I, E)), (f"blk.{l}.ffn_up_exps.weight", gu[:, I:], (D, I, E)),
                              (f"blk.{l}.ffn_down_exps.weight", dn, (I, D, E))):
            b = np.ascontiguousarray(arr).tobytes()
            assert len(b) == ne[0] // BLOCK * codec.BLOCK_BYTES * ne[1] * ne[2], (name, len(b))
            infos.append((name, ne, codec.GGUF_TYPE, off))
            pad = (-len(b)) % 32
            blobs.append(b + bytes(pad))
            off += len(b) + pad
    for name, ne, b in dense:
        infos.append((name, ne, 8, off))
        pad = (-len(b)) % 32
        blobs.append(b + bytes(pad))
        off += len(b) + pad
    kvs = [("general.architecture", b"inkling2-tqt-experts")]
    for codec in (tqt, tqt2):
        if any(c is codec for _, _, c in layers):
            kvs.append((codec.CODE_KEY, codec.CODE_TAG.encode()))
    head = b"GGUF" + np.uint32(3).tobytes() + np.uint64(len(infos)).tobytes() + np.uint64(len(kvs)).tobytes()
    for k, v in kvs:
        head += _gstr(k.encode()) + np.uint32(8).tobytes() + _gstr(v)
    for name, ne, typ, o in infos:
        head += _gstr(name.encode()) + np.uint32(3).tobytes() + b"".join(np.uint64(x).tobytes() for x in ne) + np.uint32(typ).tobytes() + np.uint64(o).tobytes()
    head += bytes((-len(head)) % 32)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return len(head) + off


def cmd_pack(args):
    out = Path(args.out)
    layers = [(l, out / "enc", tqt) for l in parse_layers(args.layers) if (out / "enc" / f"L{l:02d}_dn.bin").exists()]
    layers += [(l, out / "enc2", tqt2) for l in (parse_layers(args.layers2) if args.layers2 else []) if (out / "enc2" / f"L{l:02d}_dn.bin").exists()]
    if len({l for l, _, _ in layers}) != len(layers):
        raise SystemExit("a layer is in both --layers (tqt) and --layers2 (tqt2)")
    dense = dense_q8_tensors(args.model) if args.dense_q8 else ()
    n = write_overlay(args.gguf, sorted(layers, key=lambda x: x[0]), dense)
    log(f"{args.gguf}: {len(layers)} layers ({[l for l, _, c in layers if c is tqt]} tqt, {[l for l, _, c in layers if c is tqt2]} tqt2), {n / 1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "hessian", "eval", "pack", "sweep", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="the official checkpoint folder")
        p.add_argument("--out", required=True)
        p.add_argument("--layers", default="0-23", help="layers of the 4-bit code (pack / eval) or of this encode run")
        if name == "sweep":
            p.add_argument("--code", choices=("tqt", "tqt2"), required=True)
        if name in ("pack", "eval"):
            p.add_argument("--layers2", default="", help="layers of the 2-bit code TQT2 (their enc2 files)")
        if name == "pack":
            p.add_argument("--gguf", required=True)
            p.add_argument("--dense-q8", action="store_true", help="also the embedding, head and attention q / k / v / o as Q8_0")
        if name == "encode":
            p.add_argument("--from-h", default="", help="folder of Hessians from `hessian` (no reference model is loaded)")
        if name in ("encode", "hessian"):
            p.add_argument("--wiki", default="")
            p.add_argument("--srcdir", default="")
            p.add_argument("--tokens", type=int, default=200000)
            p.add_argument("--seqlen", type=int, default=1000)
            p.add_argument("--group", type=int, default=6)
            p.add_argument("--scales", type=int, default=3)
            p.add_argument("--codec", default="tqt", help="comma list of tqt (4-bit, enc/) and tqt2 (2-bit, enc2/): one calibration pass feeds all")
            p.add_argument("--damp-gu", required=True, help="one value per codec, comma separated")
            p.add_argument("--damp-dn", required=True, help="one value per codec, comma separated")
            p.add_argument("--n0", type=float, required=True)
            p.add_argument("--nmin", type=float, default=64)
    args = ap.parse_args()
    {"encode": cmd_encode, "hessian": cmd_hessian, "eval": cmd_eval, "pack": cmd_pack, "sweep": cmd_sweep, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
