"""TQT3: the 3-bit member of the TQT family (3.125 bits per weight, GGUF type 308, code tag tqt3-c1-v1).

Per 128 weights of a row (TQ4's rotated basis): FP16 scale + 48 stream bytes = 64 six-bit symbols s_0..s_63 (symbol i at bit
6 i of the little-endian bit stream, cyclic). Weight pair p (weights 2p, 2p + 1) reads the 18-bit window w_p = s_p | s_{p+1} << 6 |
s_{p+2} << 12 (indices mod 64) and decodes it with the code c2 of tqt-c2-v1 (the same 18-bit table as TQT 4-bit).
The encoder is a tail-biting Viterbi search over the 4096-state trellis (tqt3_viterbi.cu); gptq() is tqt.gptq with this codec.
Same interface as tqt.py (the codec arguments of tqt.gptq)."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

import tqt  # the c2 codec: BlockLDLQ (gptq) shared through its codec argument

GGUF_TYPE = 308
CODE_KEY = "tqllm.tqt3.code"
CODE_TAG = "tqt3-c1-v1"
BLOCK = 128
STREAM_BYTES = 48
BLOCK_BYTES = 50
L = tqt.L
HA, HB = tqt.HA, tqt.HB
C1, D1, C2, D2 = tqt.C1, tqt.D1, tqt.C2, tqt.D2
GAMMAS = (0.97, 1.03, 1.09, 1.15, 1.21)  # scale candidates x block RMS / code RMS (best single scale on Gaussian blocks ~1.05-1.1, normalised MSE 0.0182)
PASS1 = (16, 12)


def gammas(n):
    order = (2, 1, 3, 0, 4)
    return tuple(sorted(GAMMAS[i] for i in order[:max(1, min(5, n))]))


code_values = tqt.code_values


def table():
    return tqt.table()


def code_rms():
    return tqt.code_rms()


def symbols(stream):
    """stream bytes uint8 [..., 48] -> six-bit symbols [..., 64]"""
    bits = np.unpackbits(np.asarray(stream, np.uint8), axis=-1, bitorder="little").reshape(*np.shape(stream)[:-1], 64, 6).astype(np.uint32)
    return (bits << np.arange(6, dtype=np.uint32)).sum(-1)


def windows(stream):
    """stream bytes uint8 [..., 48] -> windows [..., 64]"""
    q = symbols(stream)
    return (q | np.roll(q, -1, -1) << 6 | np.roll(q, -2, -1) << 12) & 0x3FFFF


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
        src = Path(__file__).with_name("tqt3_viterbi.cu")
        tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tqllm"
        so = cache / f"tqt3_viterbi_{tag}.so"
        if not so.exists():
            cache.mkdir(parents=True, exist_ok=True)
            tmp = so.with_suffix(".tmp.so")
            subprocess.run([os.environ.get("NVCC", "nvcc"), "-O3", f"-arch={os.environ.get('TQT_ARCH', 'native')}", "-shared", "-Xcompiler", "-fPIC",
                            str(src), "-o", str(tmp)], check=True)
            os.replace(tmp, so)
        _LIB = ctypes.CDLL(str(so))
        _LIB.tqt3_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
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
    r = _lib().tqt3_viterbi(ctypes.c_void_p(X.data_ptr()), n, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(cost.data_ptr()), *PASS1)
    if r != 0:
        raise RuntimeError(f"tqt3_viterbi: CUDA error {r}")
    if not bool((cost > -1e38).all()):
        raise RuntimeError("tqt3_viterbi: tail-biting path inconsistent")
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
    bits = ((b[..., :, None] >> torch.arange(8, device=b.device)) & 1).reshape(*b.shape[:-1], 384).reshape(*b.shape[:-1], 64, 6)
    q = (bits << torch.arange(6, device=b.device)).sum(-1)
    return (q | torch.roll(q, -1, -1) << 6 | torch.roll(q, -2, -1) << 12) & 0x3FFFF


def gptq(W, H, **kw):
    return tqt.gptq(W, H, codec=__import__(__name__), **kw)
