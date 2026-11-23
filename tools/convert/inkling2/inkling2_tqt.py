#!/usr/bin/env python3
"""TQT quantiser for the routed experts of Inkling-2-Mini (transformers PR #51050 reference, experts dequantised to BF16).

  encode   one reference forward over the calibration text per group of layers, per-expert Hessians accumulated in the
           experts' forward hook (summed over the layer's 1-3 visits), then BlockLDLQ + tail-biting Viterbi (tqt.py) of
           every expert's gate+up and down into enc/L<layer>_gu.bin / _dn.bin (66 B per 128 weights, rotated basis)
  pack     the encoded layers as one overlay GGUF (tqt_experts.gguf, engine names blk.<l>.ffn_{gate,up,down}_exps.weight, type
           306) that the engine loads next to the checkpoint folder
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
import tqv12  # noqa: E402

D, I, E = 2304, 1152, 128
BLOCK = 128


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
def encode_matrix(W, H, damp, nscales, order=True, refit=True):
    """W [N, K] float32 CUDA (original basis), H [K, K] float32 (original basis) -> packed rows uint8 [N, K/128*66], rel err"""
    Wr = tqv12.rot_last(W)
    Hr = tqv12.rotate_h(H)
    s, st, Q = tqt.gptq(Wr, Hr, damp=damp, gammas=tqt.gammas(nscales), damp_form="median", order=order, refit=refit)
    err = tqv12.rel_err(Q, Wr, Hr)
    return tqt.pack(s.cpu().numpy(), st.cpu().numpy()), err


def encode_layer(stats, w_gu, w_dn, args):
    import torch
    Hg_norm = stats.Hg / max(stats.N, 1)
    gu = np.empty((E, 2 * I, (D // BLOCK) * tqt.BLOCK_BYTES), np.uint8)
    dn = np.empty((E, D, (I // BLOCK) * tqt.BLOCK_BYTES), np.uint8)
    errs = []
    t0 = time.time()
    for e in range(E):
        Se = float(stats.S[e])
        H = blend_gu(stats.Hgu[e], Se, Hg_norm, args.n0)
        gu[e], eg = encode_matrix(w_gu[e].float(), H, args.damp_gu, args.scales)
        H = final_dn(stats.Hdn[e], Se, args.nmin)
        dn[e], ed = encode_matrix(w_dn[e].float(), H, args.damp_dn, args.scales)
        errs.append((eg, ed))
        if e % 16 == 15:
            log(f"  expert {e + 1}/{E}: {time.time() - t0:.0f} s, rel err gu {np.mean([x[0] for x in errs]):.5f} down {np.mean([x[1] for x in errs]):.5f}")
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


def cmd_encode(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = Path(args.out)
    (outdir / "enc").mkdir(parents=True, exist_ok=True)
    todo = [l for l in parse_layers(args.layers) if not (outdir / "enc" / f"L{l:02d}_dn.bin").exists()]
    log("layers to encode:", todo)
    if not todo:
        return
    model, mods = load_model(args.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    seqs = calib_ids(tok, args.wiki, args.srcdir, args.tokens, args.seqlen)
    for g in range(0, len(todo), args.group):
        group = todo[g:g + args.group]
        log("group", group)
        stats = {l: ExpertStats("cuda") for l in group}
        hooks = []
        for l in group:
            m = mods[l]

            def pre(module, a, kw, l=l):
                hs, ti, tw = (list(a) + [kw.get(k) for k in ("hidden_states", "top_k_index", "top_k_weights")][len(a):])[:3]
                with torch.no_grad():
                    stats[l].add(hs, ti, tw, module.gate_up_proj, module.down_proj, module._apply_gate)

            hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))
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
            t0 = time.time()
            with torch.no_grad():
                gu, dn, errs = encode_layer(s, mods[l].gate_up_proj, mods[l].down_proj, args)
            for name, arr in (("gu", gu), ("dn", dn)):
                p = outdir / "enc" / f"L{l:02d}_{name}.bin"
                arr.tofile(str(p) + ".tmp")
                if name == "gu":
                    json.dump({"n": s.n.tolist(), "S": s.S.tolist(), "err": errs, "seconds": time.time() - t0}, open(outdir / "enc" / f"L{l:02d}.json", "w"))
                os.replace(str(p) + ".tmp", p)
            log(f"layer {l} done in {time.time() - t0:.0f} s, mean rel err gu {np.mean([x[0] for x in errs]):.5f} down {np.mean([x[1] for x in errs]):.5f}")
        del stats
        torch.cuda.empty_cache()


# ---------------- eval ----------------
def decode_layer(encdir, l):
    """-> (gate_up [E, 2I, D], down [E, D, I]) float32 in the original basis, from the encoded files"""
    R = tqv12.rot_matrix_np().astype(np.float32)
    out = []
    for name, (N, K) in (("gu", (2 * I, D)), ("dn", (D, I))):
        raw = np.fromfile(Path(encdir) / f"L{l:02d}_{name}.bin", np.uint8).reshape(E, N, -1)
        w = np.empty((E, N, K), np.float32)
        for e in range(E):
            q = tqt.decode(raw[e].tobytes(), N, K).reshape(N, K // BLOCK, BLOCK)
            w[e] = (q @ R).reshape(N, K)  # rotated rows w R^T -> w
        out.append(w)
    return out


def cmd_eval(args):
    import torch
    model, mods = load_model(args.model)
    layers = parse_layers(args.layers)
    texts = json.load(open(Path(args.model) / "golden/mt/texts.json"))
    top1_ref = np.fromfile(Path(args.model) / "golden/mt/top1_bf16.i32", np.int32)
    V = 200058
    lp_ref = np.memmap(Path(args.model) / "golden/mt/lp_bf16.f16", np.float16, "r")
    saved = {}
    for l in layers:
        gu, dn = decode_layer(Path(args.out) / "enc", l)
        m = mods[l]
        saved[l] = (m.gate_up_proj.data.clone(), m.down_proj.data.clone())
        m.gate_up_proj.data.copy_(torch.from_numpy(gu).to("cuda", torch.bfloat16))
        m.down_proj.data.copy_(torch.from_numpy(dn).to("cuda", torch.bfloat16))
        log("substituted layer", l)
    pos = 0
    kl_sum = agree = n = 0
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([t["ids"]], device="cuda")
            T = ids.shape[1]
            lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)
            assert lp.shape[-1] == V, lp.shape
            ref = torch.from_numpy(np.array(lp_ref[pos * V:(pos + T) * V]).reshape(T, V)).to("cuda").float()
            kl = (ref.exp() * (ref - lp)).sum(-1)
            kl_sum += kl.sum().item()
            agree += (lp.argmax(-1).cpu().numpy() == top1_ref[pos:pos + T]).sum()
            n += T
            pos += T
            log(f"{t['name']:14s} T {T:4d} KLD {kl.mean().item():.5f}")
    log(f"ALL {n} positions: KLD vs lp_bf16 {kl_sum / n:.5f}, top-1 {100 * agree / n:.2f} %, layers {args.layers}")


# ---------------- pack ----------------
def _gstr(b):
    return np.uint64(len(b)).tobytes() + b


def write_overlay(encdir, path, layers):
    """GGUF v3, alignment 32: tensors [cols, rows, E] of type 306 per layer and projection (stream bytes as encoded)"""
    infos, blobs, off = [], [], 0
    for l in layers:
        gu = np.fromfile(Path(encdir) / f"L{l:02d}_gu.bin", np.uint8).reshape(E, 2 * I, -1)
        dn = np.fromfile(Path(encdir) / f"L{l:02d}_dn.bin", np.uint8).reshape(E, D, -1)
        for name, arr, ne in ((f"blk.{l}.ffn_gate_exps.weight", gu[:, :I], (D, I, E)), (f"blk.{l}.ffn_up_exps.weight", gu[:, I:], (D, I, E)),
                              (f"blk.{l}.ffn_down_exps.weight", dn, (I, D, E))):
            b = np.ascontiguousarray(arr).tobytes()
            assert len(b) == ne[0] // BLOCK * tqt.BLOCK_BYTES * ne[1] * ne[2], (name, len(b))
            infos.append((name, ne, off))
            pad = (-len(b)) % 32
            blobs.append(b + bytes(pad))
            off += len(b) + pad
    head = b"GGUF" + np.uint32(3).tobytes() + np.uint64(len(infos)).tobytes() + np.uint64(2).tobytes()
    for k, v in (("general.architecture", b"inkling2-tqt-experts"), (tqt.CODE_KEY, tqt.CODE_TAG.encode())):
        head += _gstr(k.encode()) + np.uint32(8).tobytes() + _gstr(v)
    for name, ne, o in infos:
        head += _gstr(name.encode()) + np.uint32(3).tobytes() + b"".join(np.uint64(x).tobytes() for x in ne) + np.uint32(tqt.GGUF_TYPE).tobytes() + np.uint64(o).tobytes()
    head += bytes((-len(head)) % 32)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return len(head) + off


def cmd_pack(args):
    layers = [l for l in parse_layers(args.layers) if (Path(args.out) / "enc" / f"L{l:02d}_dn.bin").exists()]
    n = write_overlay(Path(args.out) / "enc", args.gguf, layers)
    log(f"{args.gguf}: layers {layers}, {n / 1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "eval", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="the official checkpoint folder")
        p.add_argument("--out", required=True)
        p.add_argument("--layers", default="0-23")
        if name == "pack":
            p.add_argument("--gguf", required=True)
        if name == "encode":
            p.add_argument("--wiki", required=True)
            p.add_argument("--srcdir", required=True)
            p.add_argument("--tokens", type=int, default=200000)
            p.add_argument("--seqlen", type=int, default=1000)
            p.add_argument("--group", type=int, default=6)
            p.add_argument("--scales", type=int, default=3)
            p.add_argument("--damp-gu", type=float, required=True)
            p.add_argument("--damp-dn", type=float, required=True)
            p.add_argument("--n0", type=float, required=True)
            p.add_argument("--nmin", type=float, default=64)
    args = ap.parse_args()
    {"encode": cmd_encode, "eval": cmd_eval, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    main()
