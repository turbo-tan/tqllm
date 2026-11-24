#!/usr/bin/env python3
"""FNT quantiser for Qwen3.8-Flash-Next routed experts (docs/guides/fnt.md; format: docs/reference/formats.md "FNT").

  alloc     print the per-layer FNT rate chosen from a template GGUF's expert types (the match-or-beat map)
  encode    one layer: BlockLDLQ + tail-biting Viterbi from the BF16 safetensors and a TQLLM_DEBUG=moe_dump capture
            (resumable per chunk of experts; writes OUT/L<layer>/e<a>-<b>.npz, then OUT/L<layer>/done)
  assemble  the template GGUF (ISTA-DASLab GSQ-RCO IQ3_XXS shard 1) with every encoded layer's ffn_{gate,up,down}_exps
            replaced by FNT tensors and tqllm.fnt.code = fnt-c2w16-v1; every other tensor and key copied byte for byte
  verify    decode every FNT tensor of an assembled file with this file's numpy decoder and compare with the encoder's
            streams (byte-identical) and the BF16 source (relative error per tensor)

Format summary: per 640 weights of a row an FP16 scale + 5 tail-biting trellis streams of KB * 8 bytes (128 weights
each; KB = 4..7 stream bits per weight pair); weight pair p of a stream reads the 16-bit window at bit p * KB and decodes
it with code c2; values in TQ4's 128-Hadamard basis. The encoder needs PyTorch with CUDA and nvcc (the Viterbi kernel
fnt_viterbi.cu is compiled on first use into ~/.cache/tqllm)."""
import argparse
import ctypes
import glob
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HID, FF, NE, TOPK = 2560, 640, 512, 10
T, SEG, L16 = 128, 640, 16
FNT_TYPE = {4: 330, 5: 331, 6: 332, 7: 333}
CODE_TAG = "fnt-c2w16-v1"
HA, HB = 0x9E3779B1, 0x7F4A7C15
C1, D1, C2, D2 = 0x2C1B3C6D, 0x9E3779B9, 0x297A2D39, 0x7F4A7C15
S1 = [0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9]  # TQ4 rotation signs (tq4k::S1)
# template (ISTA GSQ-RCO IQ3_XXS) expert type -> FNT KB: smallest rate whose held-out output error is <= the template's
# (median <= 0.85x, max <= 1.3x over 16 experts; docs/guides/fnt.md)
GU_KB = {16: 4, 17: 4, 22: 4, 18: 6, 21: 6}  # IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
DN_KB = {302: 4, 42: 4, 20: 7}  # GSQ Q2_0 (tqLLM id / file id), IQ4_NL


# ---------------- code, packing, rotation (numpy, independent of the encoder) ----------------
def code_values(w, L=L16):
    """windows (< 2^L) -> float32 [..., 2]: code c2 with an L-bit window"""
    m32 = np.uint64(0xFFFFFFFF)
    w = np.asarray(w).astype(np.uint64)
    x = ((w * np.uint64(HA) + np.uint64(HB)) << np.uint64(32 - L)) & m32
    y = ((x << np.uint64(16)) | (x >> np.uint64(16))) & m32
    z = (y * np.uint64(C1) + np.uint64(D1)) & m32
    z2 = (z * np.uint64(C2) + np.uint64(D2)) & m32
    h1 = ((z & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    h2 = ((z2 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    a = h1.view(np.uint16).reshape(*h1.shape, 2).view(np.float16).astype(np.float32)
    b = h2.view(np.uint16).reshape(*h2.shape, 2).view(np.float16).astype(np.float32)
    return (a + b).astype(np.float16).astype(np.float32)


_TAB = {}


def table(L=L16):
    if L not in _TAB:
        _TAB[L] = np.ascontiguousarray(code_values(np.arange(1 << L, dtype=np.uint32), L))
    return _TAB[L]


def code_rms(L=L16):
    return float(np.sqrt((table(L).astype(np.float64) ** 2).mean()))


def decode_stream(stream, KB, L=L16):
    """stream bytes uint8 [..., KB*8] -> unscaled values [..., 128]"""
    stream = np.asarray(stream, np.uint8)
    nbits = stream.shape[-1] * 8
    P = nbits // KB
    bits = np.unpackbits(stream, axis=-1, bitorder="little").astype(np.uint32)
    idx = (np.arange(P)[:, None] * KB + np.arange(L)[None, :]) % nbits
    w = (bits[..., idx] << np.arange(L, dtype=np.uint32)).sum(-1).astype(np.uint32)
    v = table(L)[w]
    return v.reshape(*v.shape[:-2], -1)


def sign_vec():
    return np.array([-1.0 if (S1[i // 32] >> (i % 32)) & 1 else 1.0 for i in range(128)], np.float32)


def unrot_np(y):
    """rotated rows [..., K] -> plain basis (x = D H y / sqrt(128) per 128)"""
    sh = y.shape
    v = y.reshape(*sh[:-1], sh[-1] // 128, 128).astype(np.float32)
    h = 1
    while h < 128:
        z = v.reshape(*v.shape[:-1], 128 // (2 * h), 2, h)
        v = np.stack((z[..., 0, :] + z[..., 1, :], z[..., 0, :] - z[..., 1, :]), axis=-2).reshape(*v.shape[:-1], 128)
        h *= 2
    return (v / math.sqrt(128) * sign_vec()).reshape(sh)


# ---------------- BF16 safetensors (memory-mapped, per expert) ----------------
class Safetensors:
    def __init__(self, d):
        self.d = Path(d)
        self.idx = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]
        self.hdr = {}

    def tensor(self, name):
        f = self.idx[name]
        if f not in self.hdr:
            with open(self.d / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdr[f] = (json.loads(fh.read(n)), 8 + n)
        h, base = self.hdr[f]
        m = h[name]
        if m["dtype"] != "BF16":
            raise ValueError(f"{name}: {m['dtype']}, expected BF16")
        return np.memmap(self.d / f, dtype=np.uint16, mode="r", offset=base + m["data_offsets"][0], shape=tuple(m["shape"]))

    def experts(self, layer):
        pre = f"model.language_model.layers.{layer}.mlp.experts."
        return self.tensor(pre + "gate_up_proj"), self.tensor(pre + "down_proj")  # [512, 1280, 2560] (gate rows first), [512, 2560, 640]


# ---------------- GGUF (minimal reader / writer for the template copy) ----------------
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_parse(path):
    f = open(path, "rb")

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def skip(t):
        if t in _SIZES:
            f.read(_SIZES[t])
        elif t == 8:
            f.read(u64())
        elif t == 9:
            it, n = u32(), u64()
            for _ in range(n):
                skip(it)
        else:
            raise ValueError(f"GGUF value type {t}")
    if f.read(4) != b"GGUF":
        raise ValueError(f"{path}: not a GGUF")
    ver = u32()
    nt, nkv = u64(), u64()
    kv0, align = f.tell(), 32
    for _ in range(nkv):
        k = f.read(u64()).decode()
        t = u32()
        if k == "general.alignment":
            align = u32()
        else:
            skip(t)
    kv1 = f.tell()
    tensors = []
    for _ in range(nt):
        name = f.read(u64()).decode()
        ne = [u64() for _ in range(u32())]
        typ, off = u32(), u64()
        tensors.append(dict(name=name, ne=ne, type=typ, off=off))
    base = (f.tell() + align - 1) // align * align
    return dict(ver=ver, nkv=nkv, kv=(kv0, kv1), tensors=tensors, align=align, base=base, size=os.path.getsize(path))


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def template_types(g):
    t = {}
    for x in g["tensors"]:
        p = x["name"].split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            t[(int(p[1]), p[2].split("_")[1])] = x["type"]
    return t


def layer_kb(types, layer):
    return GU_KB[types[(layer, "gate")]], DN_KB[types[(layer, "down")]]


# ---------------- encoder (PyTorch + CUDA) ----------------
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        src = Path(__file__).with_name("fnt_viterbi.cu")
        tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tqllm"
        so = cache / f"fnt_viterbi_{tag}.so"
        if not so.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = so.with_suffix(".tmp.so")
            subprocess.run([os.environ.get("NVCC", "nvcc"), "-O3", f"-arch={os.environ.get('TQT_ARCH', 'native')}", "-shared",
                            "-Xcompiler", "-fPIC", str(src), "-o", str(tmp)], check=True)
            os.replace(tmp, so)
        _LIB = ctypes.CDLL(str(so))
        _LIB.fnt_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return _LIB


def viterbi(X, KB):
    """X float32 CUDA [n, 128] (code units) -> chunks uint8 CUDA [n, 64]"""
    import torch
    X = X.contiguous().float()
    n = X.shape[0]
    out = torch.empty(n, T // 2, dtype=torch.uint8, device=X.device)
    cost = torch.empty(n, dtype=torch.float32, device=X.device)
    torch.cuda.synchronize()
    r = _lib().fnt_viterbi(X.data_ptr(), 0, n, L16, KB, T, out.data_ptr(), cost.data_ptr(), 16, 12)
    if r:
        raise RuntimeError(f"fnt_viterbi: CUDA error {r}")
    if not bool((cost > -1e38).all()):
        raise RuntimeError("fnt_viterbi: tail-biting path inconsistent")
    return out


def rot_t(x):
    import torch
    sv = torch.from_numpy(sign_vec()).to(x.device)
    sh = x.shape
    y = x.reshape(*sh[:-1], sh[-1] // 128, 128) * sv
    h = 1
    while h < 128:
        z = y.reshape(*y.shape[:-1], 128 // (2 * h), 2, h)
        y = torch.stack((z[..., 0, :] + z[..., 1, :], z[..., 0, :] - z[..., 1, :]), dim=-2).reshape(*y.shape[:-1], 128)
        h *= 2
    return (y * (1.0 / 128 ** 0.5)).reshape(sh)


def windows_t(lo, KB):
    import torch
    w = torch.zeros_like(lo, dtype=torch.int64)
    for i in range((L16 + KB - 1) // KB):
        w |= torch.roll(lo.long(), -i, -1) << (i * KB)
    return w & ((1 << L16) - 1)


def pack_t(ch, KB):
    import torch
    bits = ((ch.long()[..., :, None] >> torch.arange(KB, device=ch.device)) & 1).reshape(*ch.shape[:-1], 64 * KB)
    return (bits.reshape(*ch.shape[:-1], KB * 8, 8) << torch.arange(8, device=ch.device)).sum(-1).to(torch.uint8)


def chol_u_inv(H):
    import torch
    Lc, info = torch.linalg.cholesky_ex(H)
    if bool((info != 0).any()):
        Lc = torch.linalg.cholesky(H.double()).float()
    Hi = torch.cholesky_inverse(Lc)
    U, info = torch.linalg.cholesky_ex(Hi, upper=True)
    if bool((info != 0).any()):
        U = torch.linalg.cholesky(Hi.double(), upper=True).float()
    return U


def ldlq(W, H, KB, gamma):
    """batched BlockLDLQ: W [E, N, K] rotated, H [E, K, K] damped -> chunks [E, N, K/128, 64], scales [E, N, K/640],
    unscaled code values [E, N, K]"""
    import torch
    E, N, K = W.shape
    nb, G = K // T, K // SEG
    tab = torch.from_numpy(table()).to(W.device)
    W = W.clone()
    U = chol_u_inv(H)
    s = (W.reshape(E, N, G, SEG).pow(2).mean(3).sqrt() * (gamma / code_rms())).half().float().clamp_min(2.0 ** -24)
    chunks = torch.empty(E, N, nb, T // 2, dtype=torch.uint8, device=W.device)
    V = torch.empty_like(W)
    for b in range(nb):
        i1, i2 = b * T, (b + 1) * T
        W0 = W[:, :, i1:i2]
        sb = s[:, :, i1 // SEG]
        lo = viterbi((W0 / sb[..., None]).reshape(E * N, T), KB)
        chunks[:, :, b] = lo.view(E, N, T // 2)
        v = tab[windows_t(lo, KB)].reshape(E, N, T)
        V[:, :, i1:i2] = v
        if i2 < K:
            E1 = torch.linalg.solve_triangular(U[:, i1:i2, i1:i2], W0 - v * sb[..., None], upper=True, left=False)
            W[:, :, i2:] -= E1 @ U[:, i1:i2, i2:]
    return chunks, s, V


def refit(V, Wt, H):
    """closed-form 640-group scales per row under the H metric (G x G solve)"""
    import torch
    E, N, K = V.shape
    G = K // SEG
    out = []
    for e in range(E):
        B = torch.zeros(N, G, K, device=V.device)
        for g in range(G):
            B[:, g, g * SEG:(g + 1) * SEG] = V[e, :, g * SEG:(g + 1) * SEG]
        C = B @ H[e]
        A = C @ B.transpose(1, 2)
        A = A + 1e-12 * torch.eye(G, device=V.device) * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None]
        out.append(torch.linalg.solve(A, (C @ Wt[e][:, :, None]))[..., 0])
    return torch.stack(out)


def out_err(Q, W, H):
    D = Q - W
    return ((D @ H) * D).sum((1, 2)) / ((W @ H) * W).sum((1, 2)).clamp_min(1e-30)


def load_dump(d, layer):
    """TQLLM_DEBUG=moe_dump files -> x FP16 [N, 2560] (memmap), ids [N, 10]; rows before start.json's offsets skipped"""
    st = json.load(open(f"{d}/start.json")) if os.path.exists(f"{d}/start.json") else {}
    x = np.memmap(f"{d}/L{layer}.x16", dtype=np.float16, mode="r", offset=st.get(f"L{layer}.x16", 0)).reshape(-1, HID)
    raw = open(f"{d}/L{layer}.rt", "rb").read()[st.get(f"L{layer}.rt", 0):]
    ids, i = [], 0
    while i < len(raw):
        n = struct.unpack_from("<i", raw, i)[0]
        ids.append(np.frombuffer(raw, np.int32, n * TOPK, i + 4).reshape(n, TOPK))
        i += 4 + 4 * n * (2 * TOPK + 1)
    ids = np.concatenate(ids)
    if len(ids) != len(x):
        raise ValueError(f"layer {layer}: {len(ids)} routed rows, {len(x)} input rows")
    return x, ids


def cmd_encode(a):
    import torch
    dev = torch.device("cuda")
    t_start = time.time()
    st = Safetensors(a.bf16)
    l = a.layer
    kgu, kdn = (a.kb_gu, a.kb_dn) if a.kb_gu else layer_kb(template_types(gguf_parse(a.template)), l)
    od = Path(a.out) / f"L{l}"
    od.mkdir(parents=True, exist_ok=True)
    x, ids = load_dump(a.dump, l)
    N = len(x)
    flat = ids.reshape(-1)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(NE + 1))
    keep = (np.arange(N) // 2048) % 4 != 0 if a.holdout else np.ones(N, bool)

    def tok_of(e):
        t = order[bounds[e]:bounds[e + 1]] // TOPK
        return np.sort(t[keep[t]])
    rng = np.random.default_rng(l)
    prior = np.sort(rng.choice(np.flatnonzero(keep), size=min(a.prior_tokens, int(keep.sum())), replace=False))
    Xp = rot_t(torch.from_numpy(np.asarray(x[prior], np.float32)).to(dev))
    Hp = Xp.T @ Xp / len(prior)
    gu, dn = st.experts(l)
    bf = lambda arr: torch.from_numpy(np.ascontiguousarray(arr).astype(np.int32) << 16).to(dev).view(torch.float32)  # noqa: E731
    for b0 in range(0, NE, a.batch):
        b1 = min(NE, b0 + a.batch) - 1
        fn = od / f"e{b0:03d}-{b1:03d}.npz"
        if fn.exists():
            continue
        if time.time() - t_start > a.time_limit:
            print(f"time limit: stop before experts {b0}-{b1}", flush=True)
            return 3
        t0 = time.time()
        es = list(range(b0, b1 + 1))
        Wgu, Wdn = rot_t(bf(gu[b0:b1 + 1])), rot_t(bf(dn[b0:b1 + 1]))
        Hs, Hr, ns = [], [], []
        for e in es:
            tk = tok_of(e)
            ns.append(len(tk))
            G = torch.zeros(HID, HID, device=dev)
            if len(tk):
                Xe = rot_t(torch.from_numpy(np.asarray(x[tk], np.float32)).to(dev))
                G = Xe.T @ Xe
            Hr.append(G / max(1, len(tk)))
            Hs.append((G + a.tau * Hp) / (len(tk) + a.tau))
        H, Hr = torch.stack(Hs), torch.stack(Hr)
        md = torch.diagonal(H, dim1=1, dim2=2).mean(1)
        ch_gu, s_gu, V_gu = ldlq(Wgu, H + a.damp * md[:, None, None] * torch.eye(HID, device=dev), kgu, a.gamma)
        s2 = refit(V_gu, Wgu, H).half().float()
        s_gu = torch.where((s2 > 0) & torch.isfinite(s2), s2, s_gu)
        Qgu = V_gu * s_gu.repeat_interleave(SEG, -1)
        e_gu = out_err(Qgu, Wgu, Hr)
        Hd, Hdr = [], []
        for k, e in enumerate(es):
            tk = tok_of(e)

            def act(Xr):
                return rot_t(torch.nn.functional.silu(Xr @ Qgu[k, :FF].T) * (Xr @ Qgu[k, FF:].T))
            Ap = act(Xp)
            G = torch.zeros(FF, FF, device=dev)
            if len(tk):
                Ae = act(rot_t(torch.from_numpy(np.asarray(x[tk], np.float32)).to(dev)))
                G = Ae.T @ Ae
            Hdr.append(G / max(1, len(tk)))
            Hd.append((G + a.tau * (Ap.T @ Ap / len(prior))) / (len(tk) + a.tau))
        Hd, Hdr = torch.stack(Hd), torch.stack(Hdr)
        md = torch.diagonal(Hd, dim1=1, dim2=2).mean(1)
        ch_dn, s_dn, V_dn = ldlq(Wdn, Hd + a.damp * md[:, None, None] * torch.eye(FF, device=dev), kdn, a.gamma)
        s2 = refit(V_dn, Wdn, Hd).half().float()
        s_dn = torch.where((s2 > 0) & torch.isfinite(s2), s2, s_dn)
        e_dn = out_err(V_dn * s_dn.repeat_interleave(SEG, -1), Wdn, Hdr)
        tmp = str(fn) + ".tmp.npz"
        np.savez(tmp, experts=np.array(es), st_gu=pack_t(ch_gu, kgu).cpu().numpy(), s_gu=s_gu.half().cpu().numpy(),
                 st_dn=pack_t(ch_dn, kdn).cpu().numpy(), s_dn=s_dn.half().cpu().numpy(), n_tok=np.array(ns),
                 err_gu_raw=e_gu.cpu().numpy(), err_dn_raw=e_dn.cpu().numpy(),
                 meta=json.dumps(dict(L=L16, kb_gu=kgu, kb_dn=kdn, gamma=a.gamma, tau=a.tau, damp=a.damp, code=CODE_TAG)))
        os.replace(tmp, fn)
        print(f"L{l} e{b0}-{b1} {time.time() - t0:.1f}s tokens min {min(ns)} median {int(np.median(ns))} | "
              f"gate/up FNT{kgu} out {e_gu.mean():.4f} | down FNT{kdn} out {e_dn.mean():.4f}", flush=True)
    (od / "done").touch()
    return 0


def layer_npz(enc, l):
    zs = [np.load(f) for f in sorted(glob.glob(f"{enc}/L{l}/e*.npz"))]
    if not zs or not (np.concatenate([z["experts"] for z in zs]) == np.arange(NE)).all():
        raise ValueError(f"layer {l}: experts incomplete in {enc}")
    return zs


def fnt_rows(st, sc):
    """streams [R, nb, KB*8], scales [R, nb/5] -> GGUF rows [R, nb/5 * (2 + 40 KB)]"""
    R, nb, w = st.shape
    s2 = np.ascontiguousarray(sc.astype("<f2")).view(np.uint8).reshape(R, nb // 5, 2)
    return np.concatenate([s2, st.reshape(R, nb // 5, 5 * w)], axis=2).reshape(R, -1)


def _streams(z, key):
    """packed streams [E, R, nb, KB*8]; early encoder outputs kept one KB-bit chunk per byte ("ch_*", [E, R, nb, 64])"""
    if "st_" + key in z:
        return z["st_" + key]
    ch = z["ch_" + key]
    KB = json.loads(str(z["meta"]))["kb_dn" if key == "dn" else "kb_gu"]
    bits = ((ch[..., :, None] >> np.arange(KB, dtype=np.uint8)) & 1).reshape(*ch.shape[:-1], 64 * KB)
    return np.packbits(bits, axis=-1, bitorder="little")


def role_rows(z, role):
    if role == "down":
        return _streams(z, "dn"), z["s_dn"]
    sl = slice(0, FF) if role == "gate" else slice(FF, 2 * FF)
    return _streams(z, "gu")[:, sl], z["s_gu"][:, sl]


def cmd_assemble(a):
    g = gguf_parse(a.template)
    src = open(a.template, "rb")
    plan = {}
    for l in range(1000):
        if not (Path(a.enc) / f"L{l}" / "done").exists():
            continue
        meta = json.loads(str(np.load(sorted(glob.glob(f"{a.enc}/L{l}/e*.npz"))[0])["meta"]))
        for role in ("gate", "up", "down"):
            plan[f"blk.{l}.ffn_{role}_exps.weight"] = (l, role, meta["kb_dn" if role == "down" else "kb_gu"])
    print(f"{len(plan) // 3} layers to FNT", flush=True)
    infos, off = [], 0
    for t in g["tensors"]:
        if t["name"] in plan:
            KB = plan[t["name"]][2]
            nbytes, typ = int(np.prod(t["ne"])) // SEG * (2 + 40 * KB), FNT_TYPE[KB]
        else:
            nxt = [u["off"] for u in g["tensors"] if u["off"] > t["off"]]
            nbytes, typ = (min(nxt) if nxt else g["size"] - g["base"]) - t["off"], t["type"]
        infos.append((t, typ, off, nbytes))
        off = (off + nbytes + g["align"] - 1) // g["align"] * g["align"]
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as o:
        o.write(b"GGUF" + struct.pack("<IQQ", g["ver"], len(infos), g["nkv"] + 1))
        src.seek(g["kv"][0])
        o.write(src.read(g["kv"][1] - g["kv"][0]))
        o.write(gstr("tqllm.fnt.code") + struct.pack("<I", 8) + gstr(CODE_TAG))
        for t, typ, toff, _ in infos:
            o.write(gstr(t["name"]) + struct.pack("<I", len(t["ne"])) + b"".join(struct.pack("<Q", v) for v in t["ne"]))
            o.write(struct.pack("<IQ", typ, toff))
        pos = o.tell()
        base = (pos + g["align"] - 1) // g["align"] * g["align"]
        o.write(b"\0" * (base - pos))
        cache = {}
        for t, typ, toff, nbytes in infos:
            o.seek(base + toff)
            if t["name"] not in plan:
                src.seek(g["base"] + t["off"])
                left = nbytes
                while left:
                    b = src.read(min(left, 1 << 26))
                    o.write(b)
                    left -= len(b)
                continue
            l, role, KB = plan[t["name"]]
            if cache.get("l") != l:
                cache = dict(l=l, z=layer_npz(a.enc, l))
            for z in cache["z"]:
                st, sc = role_rows(z, role)
                E, R = st.shape[:2]
                o.write(fnt_rows(st.reshape(E * R, *st.shape[2:]), sc.reshape(E * R, -1)).tobytes())
            if o.tell() != base + toff + nbytes:
                raise RuntimeError(f"{t['name']}: wrote {o.tell() - base - toff} bytes, expected {nbytes}")
        o.truncate(base + off)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB (template {g['size'] / 1e9:.2f} GB)")


def cmd_verify(a):
    g = gguf_parse(a.file)
    st = Safetensors(a.bf16)
    f = open(a.file, "rb")
    bad, n = 0, 0
    for t in g["tensors"]:
        if t["type"] not in FNT_TYPE.values():
            continue
        KB = t["type"] - 326
        l, role = int(t["name"].split(".")[1]), t["name"].split(".")[2].split("_")[1]
        K, R = t["ne"][0], t["ne"][1]
        rowb = K // SEG * (2 + 40 * KB)
        zs = layer_npz(a.enc, l)
        gu, dn = st.experts(l)
        for e in a.experts:
            f.seek(g["base"] + t["off"] + e * R * rowb)
            raw = np.frombuffer(f.read(R * rowb), np.uint8).reshape(R, K // SEG, 2 + 40 * KB)
            z = next(z for z in zs if e in z["experts"])
            k = int(np.flatnonzero(z["experts"] == e)[0])
            c_ref, s_ref = role_rows(z, role)
            if not (np.array_equal(raw[:, :, 2:].reshape(R, -1), c_ref[k].reshape(R, -1)) and
                    np.array_equal(raw[:, :, :2].copy().view("<u2")[..., 0], s_ref[k].view("<u2"))):
                bad += 1
            sc = raw[:, :, :2].copy().view("<f2")[..., 0].astype(np.float32)
            v = decode_stream(raw[:, :, 2:].reshape(R, K // T, KB * 8), KB).reshape(R, K) * np.repeat(sc, SEG, 1)
            Wp = unrot_np(v)
            src = (dn[e] if role == "down" else gu[e, :FF] if role == "gate" else gu[e, FF:])
            W = (np.asarray(src).astype(np.uint32) << 16).view(np.float32)
            rel = float(((Wp - W) ** 2).sum() / (W ** 2).sum())
            n += 1
            print(f"{t['name']} expert {e}: FNT{KB}, relative weight error {rel:.4f}", flush=True)
    print(f"{n} matrices checked, {bad} differ from the encoder output")
    return 1 if bad else 0


def cmd_alloc(a):
    types = template_types(gguf_parse(a.template))
    tot = 0.0
    for l in range(max(k[0] for k in types) + 1):
        kg, kd = layer_kb(types, l)
        tot += (2 * kg + kd) / 6
        print(f"layer {l}: gate/up FNT{kg} ({kg / 2} bpw), down FNT{kd} ({kd / 2} bpw)")
    print(f"experts: {tot / (len(types) // 3) + 0.025:.3f} bits per weight incl. the 640-group scales")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("alloc")
    p.add_argument("--template", required=True)
    p = sub.add_parser("encode")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--bf16", required=True, help="the Hugging Face BF16 directory")
    p.add_argument("--dump", required=True, help="TQLLM_DEBUG=moe_dump directory holding this layer")
    p.add_argument("--out", required=True)
    p.add_argument("--template", help="template GGUF (the per-layer FNT rate from its expert types)")
    p.add_argument("--kb-gu", type=int, default=0)
    p.add_argument("--kb-dn", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.05, help="start scale x 640-group RMS / code RMS")
    p.add_argument("--tau", type=float, default=512.0, help="shrinkage toward the layer prior, in tokens")
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--prior-tokens", type=int, default=4096)
    p.add_argument("--holdout", action="store_true", help="leave out every 4th 2048-row chunk (allocation checks)")
    p.add_argument("--time-limit", type=float, default=1e9, help="seconds; exit 3 before the next chunk past it")
    p = sub.add_parser("assemble")
    p.add_argument("--template", required=True)
    p.add_argument("--enc", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--file", required=True)
    p.add_argument("--enc", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--experts", type=lambda s: [int(v) for v in s.split(",")], default=[0, 511])
    a = ap.parse_args()
    if a.cmd == "encode" and not a.kb_gu and not a.template:
        ap.error("encode needs --template or --kb-gu / --kb-dn")
    return {"alloc": cmd_alloc, "encode": cmd_encode, "assemble": cmd_assemble, "verify": cmd_verify}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
