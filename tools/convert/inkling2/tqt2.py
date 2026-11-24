"""TQT2: the 2-bit member of the TQT family (2.125 bits per weight, GGUF type 307, code tag tqt2-c1-v1).

Per 128 weights of a row (TQ4's rotated basis): FP16 scale + 32 stream bytes = 64 nibbles q_0..q_63 (low nibble of a byte
first, cyclic). Weight pair p (weights 2p, 2p + 1) reads the 16-bit window w_p = q_p | q_{p+1} << 4 | q_{p+2} << 8 |
q_{p+3} << 12 (indices mod 64) and decodes it with the hash of tqt-c2-v1 at L = 16:
  x = (w * 0x9E3779B1 + 0x7F4A7C15) << 16;  y = rotl(x, 16);  z = y * 0x2C1B3C6D + 0x9E3779B9;  z2 = z * 0x297A2D39 + 0x7F4A7C15;
  h(v) = (v & 0x8FFF8FFF) ^ 0x3B603B60 (two FP16);  (weight 2p, weight 2p + 1) = FP16x2 h(z) + h(z2) (round to nearest), times the scale.
The encoder is a tail-biting Viterbi search over the 4096-state trellis (tqt2_viterbi.cu); gptq() is tqt.gptq with this codec.
Same interface as tqt.py (the codec arguments of tqt.gptq)."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

import tqt  # the c2 codec: BlockLDLQ (gptq) shared through its codec argument

GGUF_TYPE = 307
CODE_KEY = "tqllm.tqt2.code"
CODE_TAG = "tqt2-c1-v1"
BLOCK = 128
STREAM_BYTES = 32
BLOCK_BYTES = 34
L = 16
HA, HB = tqt.HA, tqt.HB
C1, D1, C2, D2 = tqt.C1, tqt.D1, tqt.C2, tqt.D2
GAMMAS = (0.88, 0.97, 1.05, 1.15, 1.3)  # scale candidates x block RMS / code RMS (best single scale on Gaussian blocks: 1.05, normalised MSE 0.072)
PASS1 = (16, 12)


def gammas(n):
    order = (2, 1, 3, 0, 4)  # 1.05 first, then outwards
    return tuple(sorted(GAMMAS[i] for i in order[:max(1, min(5, n))]))


def code_values(w):
    """windows (uint, < 2^16) -> float32 [..., 2] FP16-exact values of the code"""
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


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = code_values(np.arange(1 << L, dtype=np.uint32))
    return _TABLE


def code_rms():
    return float(np.sqrt((table().astype(np.float64) ** 2).mean()))


def nibbles(stream):
    """stream bytes uint8 [..., 32] -> nibbles [..., 64] (low nibble first)"""
    s = np.asarray(stream, np.uint32)
    return np.stack([s & 15, s >> 4], -1).reshape(*s.shape[:-1], 64)


def windows(stream):
    """stream bytes uint8 [..., 32] -> windows [..., 64]"""
    q = nibbles(stream)
    return (q | np.roll(q, -1, -1) << 4 | np.roll(q, -2, -1) << 8 | np.roll(q, -3, -1) << 12) & 0xFFFF


def pack(scales, stream):
    """scales fp16 [N, nb], stream uint8 [N, nb, 32] -> uint8 [N, nb * 34]"""
    scales, stream = np.asarray(scales), np.asarray(stream, np.uint8)
    N, nb = scales.shape
    sb = scales.astype("<f2").view(np.uint8).reshape(N, nb, 2)
    return np.concatenate([sb, stream], axis=2).reshape(N, nb * BLOCK_BYTES)


def decode(raw, N, K):
    """TQT2 rows -> float32 [N, K] in the rotated basis (value * scale in FP32)"""
    nb = K // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(N, nb, BLOCK_BYTES)
    s = b[:, :, :2].copy().view("<f2")[..., 0].astype(np.float32)
    v = table()[windows(b[:, :, 2:])]
    return (v * s[:, :, None, None]).reshape(N, K)


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        src = Path(__file__).with_name("tqt2_viterbi.cu")
        tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tqllm"
        so = cache / f"tqt2_viterbi_{tag}.so"
        if not so.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = so.with_suffix(".tmp.so")
            subprocess.run([os.environ.get("NVCC", "nvcc"), "-O3", f"-arch={os.environ.get('TQT_ARCH', 'native')}", "-shared", "-Xcompiler", "-fPIC",
                            str(src), "-o", str(tmp)], check=True)
            os.replace(tmp, so)
        _LIB = ctypes.CDLL(str(so))
        _LIB.tqt2_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    return _LIB


def viterbi(X, wt=None):
    """X [n, 128] float32 CUDA tensor in code-value units -> stream bytes uint8 [n, 32] (CUDA tensor)"""
    import torch  # noqa: PLC0415
    assert wt is None
    X = X.contiguous().float()
    n = X.shape[0]
    out = torch.empty(n, STREAM_BYTES, dtype=torch.uint8, device=X.device)
    cost = torch.empty(n, dtype=torch.float32, device=X.device)
    torch.cuda.synchronize(X.device)
    r = _lib().tqt2_viterbi(ctypes.c_void_p(X.data_ptr()), n, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(cost.data_ptr()), *PASS1)
    if r != 0:
        raise RuntimeError(f"tqt2_viterbi: CUDA error {r}")
    if not bool((cost > -1e38).all()):
        raise RuntimeError("tqt2_viterbi: tail-biting path inconsistent")
    return out


_TT = {}


def _table_t(dev):
    import torch  # noqa: PLC0415
    if dev not in _TT:
        _TT[dev] = torch.from_numpy(table()).to(dev)
    return _TT[dev]


def windows_t(st):
    """torch version of windows()"""
    import torch  # noqa: PLC0415
    b = st.long()
    q = torch.stack([b & 15, b >> 4], -1).reshape(*b.shape[:-1], 64)
    return (q | torch.roll(q, -1, -1) << 4 | torch.roll(q, -2, -1) << 8 | torch.roll(q, -3, -1) << 12) & 0xFFFF


def gptq(W, H, **kw):
    return tqt.gptq(W, H, codec=__import__(__name__), **kw)
