"""TQT codec for the quantiser (tqv12_quant.py --format tqt): the c2 trellis code, packing, an independent decoder, and
BlockLDLQ (GPTQ with one 128-column block per step) with a Viterbi encoder. The format is in docs/reference/formats.md
("TQT"); the engine side is src/formats/weight/tqt.cpp / tqt.cu.

Per 128 weights of a row (TQ4's rotated basis): FP16 scale + 64 stream bytes B_0..B_63. Weight pair p reads the 18-bit
window w = (B_p | B_{p+1} << 8 | B_{p+2} << 16) & 0x3FFFF (indices mod 64) and decodes it with code c2 (CODE_TAG) into
two FP16 values (weights 2p, 2p+1); the weight is scale * value. The encoder finds the stream with a tail-biting Viterbi
search (tqt_viterbi.cu, built with nvcc on first use) of the targets w / scale; the scale per row and block is chosen
from a few candidates by the block's H-weighted error. The 128 columns of a block are quantised together (one trellis),
the block's exact error is fed forward to the later columns (GPTQ / LDLQ).

numpy for the codec; the encoder needs PyTorch with CUDA and nvcc."""
import ctypes
import hashlib
import math
import os
import subprocess
from pathlib import Path

import numpy as np

import tqv12  # rotation helpers (same basis as TQ4 / TQV-12)

GGUF_TYPE = 306
CODE_KEY = "tqllm.tqt.code"
CODE_TAG = "tqt-c2-v1"
BLOCK = 128
BLOCK_BYTES = 66
L = 18  # window bits of tqt-c2-v1
HA, HB = 0x9E3779B1, 0x7F4A7C15
C1, D1, C2, D2 = 0x2C1B3C6D, 0x9E3779B9, 0x297A2D39, 0x7F4A7C15
GAMMAS = (0.935, 1.01, 1.1, 1.19, 1.28)  # scale candidates x block RMS / code RMS (all five: best accuracy)
PASS1 = (16, 12)  # tail-biting first pass: steps before / after the wrap point


def gammas(n):
    """the n (1..5) central scale candidates"""
    order = (2, 1, 3, 0, 4)  # 1.1 first, then outwards
    return tuple(sorted(GAMMAS[i] for i in order[:max(1, min(5, n))]))

rot_last, rot_last_np, rot_matrix_np, rotate_h, rel_err = (tqv12.rot_last, tqv12.rot_last_np, tqv12.rot_matrix_np,
                                                          tqv12.rotate_h, tqv12.rel_err)


# ---------------- the code ----------------
def code_values(w):
    """windows (uint, < 2^L) -> float32 [..., 2] FP16-exact values of code c2"""
    m32 = np.uint64(0xFFFFFFFF)
    w = np.asarray(w).astype(np.uint64)
    x = ((w * np.uint64(HA) + np.uint64(HB)) << np.uint64(32 - L)) & m32
    y = ((x << np.uint64(16)) | (x >> np.uint64(16))) & m32
    z = (y * np.uint64(C1) + np.uint64(D1)) & m32
    z2 = (z * np.uint64(C2) + np.uint64(D2)) & m32
    h1 = ((z & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    h2 = ((z2 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
    a = h1.view(np.uint16).reshape(*h1.shape, 2).view(np.float16).astype(np.float32)  # [lo, hi] little-endian
    b = h2.view(np.uint16).reshape(*h2.shape, 2).view(np.float16).astype(np.float32)
    return (a + b).astype(np.float16).astype(np.float32)  # FP16 add: exact in FP32, one rounding (HADD2)


_TABLE = None


def table():
    """float32 [2^L, 2]: the value pair of every window"""
    global _TABLE
    if _TABLE is None:
        _TABLE = code_values(np.arange(1 << L, dtype=np.uint32))
    return _TABLE


def code_rms():
    return float(np.sqrt((table().astype(np.float64) ** 2).mean()))


def windows(stream):
    """stream bytes uint8 [..., 64] -> windows [..., 64]"""
    b = stream.astype(np.uint32)
    return (b | np.roll(b, -1, -1) << 8 | np.roll(b, -2, -1) << 16) & ((1 << L) - 1)


# ---------------- packing / decoding (numpy) ----------------
def pack(scales, stream):
    """scales fp16 [N, nb], stream uint8 [N, nb, 64] -> uint8 [N, nb * 66]"""
    scales, stream = np.asarray(scales), np.asarray(stream, np.uint8)
    N, nb = scales.shape
    sb = scales.astype("<f2").view(np.uint8).reshape(N, nb, 2)
    return np.concatenate([sb, stream], axis=2).reshape(N, nb * BLOCK_BYTES)


def decode(raw, N, K):
    """TQT rows -> float32 [N, K] in the rotated basis: value * scale in FP32 (the matvec's product; tqt_dequant_h rounds
    it to FP16). Written from the format description, not from gptq()."""
    nb = K // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(N, nb, BLOCK_BYTES)
    s = b[:, :, :2].copy().view("<f2")[..., 0].astype(np.float32)
    v = table()[windows(b[:, :, 2:])]  # [N, nb, 64, 2]
    return (v * s[:, :, None, None]).reshape(N, K)


# ---------------- encoder (CUDA) ----------------
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        src = Path(__file__).with_name("tqt_viterbi.cu")
        tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tqllm"
        so = cache / f"tqt_viterbi_{tag}_L{L}.so"
        if not so.exists():
            cache.mkdir(parents=True, exist_ok=True)
            nvcc = os.environ.get("NVCC", "nvcc")
            arch = os.environ.get("TQT_ARCH", "native")
            tmp = so.with_suffix(".tmp.so")
            subprocess.run([nvcc, "-O3", f"-arch={arch}", f"-DTQT_L={L}", "-shared", "-Xcompiler", "-fPIC", str(src), "-o", str(tmp)],
                           check=True)
            os.replace(tmp, so)
        _LIB = ctypes.CDLL(str(so))
        _LIB.tqt_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return _LIB


def viterbi(X, wt=None):
    """X [n, 128] float32 CUDA tensor in code-value units -> stream bytes uint8 [n, 64] (CUDA tensor).
    wt: optional [128] or [n, 128] weights of the squared errors in the search metric."""
    import torch  # noqa: PLC0415
    X = X.contiguous().float()
    n = X.shape[0]
    if wt is not None:
        wt = wt.float().expand(n, BLOCK).contiguous()
    out = torch.empty(n, 64, dtype=torch.uint8, device=X.device)
    cost = torch.empty(n, dtype=torch.float32, device=X.device)
    torch.cuda.synchronize(X.device)
    r = _lib().tqt_viterbi(ctypes.c_void_p(X.data_ptr()), n, ctypes.c_float(1.0), ctypes.c_void_p(out.data_ptr()),
                           ctypes.c_void_p(cost.data_ptr()), *PASS1, ctypes.c_void_p(wt.data_ptr() if wt is not None else 0))
    if r != 0:
        raise RuntimeError(f"tqt_viterbi: CUDA error {r}")
    if not bool((cost > -1e38).all()):
        raise RuntimeError("tqt_viterbi: tail-biting path inconsistent")
    return out


_TT = {}


def _table_t(dev):
    import torch  # noqa: PLC0415
    if dev not in _TT:
        _TT[dev] = torch.from_numpy(table()).to(dev)
    return _TT[dev]


DAMP_FORMS = tqv12.DAMP_FORMS


def gptq(W, H, damp=0.01, gammas=GAMMAS, damp_form="mean", order=False, refit=False):
    """BlockLDLQ in the rotated basis with one trellis per 128-weight block of a row.
    W [N, K] float32 (rotated weights), H [K, K] float32 (rotated, undamped) on one CUDA device.
    Options (the format is the same for all):
      gammas     scale candidates (x block RMS / code RMS); with several, each (row, block) keeps the one with the lowest
                 error under GPTQ's block metric (one Viterbi pass per candidate)
      damp_form  unit of `damp`: "mean" (GPTQ) or "median" of diag(H) (median, 0.1 mix / out, 0.3 gu / down
                 generalised best to held-out inputs on Qwen3.8-27B)
      order      process the 128-blocks in descending tr(H_bb) (block act-order)
      refit      closed-form FP16 scale refit per (row, block) under GPTQ's block metric (U_bb^T U_bb)^-1
    Returns scales fp16 [N, K/128], stream uint8 [N, K/128, 64], Q float32 [N, K] (decoded weights, FP32 products)."""
    import torch  # noqa: PLC0415
    N, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK
    dev = W.device
    tab = _table_t(dev)
    crms = code_rms()
    W = W.clone()
    H, dead = tqv12._damped(H, damp, damp_form)  # noqa: SLF001 (same dampening as the TQV-12 encoder)
    W[:, dead] = 0
    if order:
        bo = torch.diag(H).reshape(nb, BLOCK).sum(1).argsort(descending=True)
        perm = (bo[:, None] * BLOCK + torch.arange(BLOCK, device=dev)).reshape(-1)
        W = W[:, perm].contiguous()
        H = H[perm][:, perm].contiguous()
    U = tqv12._chol_u(H)  # noqa: SLF001
    del H
    scales = torch.empty(N, nb, dtype=torch.float16, device=dev)
    stream = torch.empty(N, nb, 64, dtype=torch.uint8, device=dev)
    Q = torch.empty_like(W)
    for i1 in range(0, K, BLOCK):
        i2 = i1 + BLOCK
        W0 = W[:, i1:i2]
        Ui = U[i1:i2, i1:i2]
        Mb = torch.cholesky_inverse(Ui, upper=True)  # GPTQ's effective metric of the block's error, (Ui^T Ui)^-1
        rms = W0.pow(2).mean(1).sqrt()
        best = None
        for g in gammas:
            sc = (rms * (g / crms)).half().float()
            sc = torch.where(sc > 0, sc, torch.full_like(sc, 2.0 ** -24))
            st = viterbi(W0 / sc[:, None])
            v = tab[windows_t(st)].reshape(N, BLOCK)
            if refit:
                vM = v @ Mb
                num, den = (vM * W0).sum(1), (vM * v).sum(1)
                s2 = (num / den.clamp_min(1e-30)).half().float()
                sc = torch.where((den > 1e-20) & (s2 > 0) & torch.isfinite(s2), s2, sc)
            q = v * sc[:, None]
            if len(gammas) == 1:
                bs, bst, bq = sc, st, q
                break
            d = W0 - q
            e = ((d @ Mb) * d).sum(1)
            if best is None:
                best, bs, bst, bq = e, sc, st, q
            else:
                k = e < best
                best = torch.where(k, e, best)
                bs = torch.where(k, sc, bs)
                bst = torch.where(k[:, None], st, bst)
                bq = torch.where(k[:, None], q, bq)
        scales[:, i1 // BLOCK] = bs.half()
        stream[:, i1 // BLOCK] = bst
        Q[:, i1:i2] = bq
        if i2 < K:
            E1 = torch.linalg.solve_triangular(Ui, W0 - bq, upper=True, left=False)
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
    if order:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(K, device=dev)
        binv = torch.empty_like(bo)
        binv[bo] = torch.arange(nb, device=dev)
        Q, scales, stream = Q[:, inv].contiguous(), scales[:, binv].contiguous(), stream[:, binv].contiguous()
    return scales, stream, Q


def windows_t(st):
    """torch version of windows()"""
    import torch  # noqa: PLC0415
    b = st.long()
    return (b | torch.roll(b, -1, -1) << 8 | torch.roll(b, -2, -1) << 16) & ((1 << L) - 1)
