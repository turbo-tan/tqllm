"""TQV-12 codec for the quantiser (tqv12_quant.py): the official codebook, TQ4's 128-point rotation, nearest-codeword
search, vector GPTQ, 66-byte block packing and an independent decoder. The format is in docs/reference/formats.md
("TQV-12"); the engine side is src/formats/weight/tqv.cpp / tqv.cu.

numpy only, except the GPTQ / search functions, which need PyTorch (and Triton for the CUDA search kernel)."""
import hashlib
import math
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CODEBOOK_INC = REPO / "src/formats/weight/tqv12_codebook.inc"
CODEBOOK_SHA256 = "63437d856ae6263846036efbeedb326bc20de93ee792d480ef9f5c67b0e1f3fc"  # of the 16,384 table bytes
CODEBOOK_TAG = "tqv12-v1"  # GGUF tqllm.tqv.codebook
GGUF_TYPE = 305
BLOCK = 128
BLOCK_BYTES = 66
# The int8 table is the k-means codebook (unit-variance |N(0, I4)| samples) divided by this step and rounded; the step
# only seeds the scale search (block_scale), the decoded weights depend on the int8 table and the stored scales alone.
CODEBOOK_STEP = 4.1011205 / 127.0
S1 = [0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9]  # tq4k::S1 in src/formats/weight/tq4.cu: the rotation's signs
# sign bit of weight j = 4u + e inside a 32-weight unit (the permutation that gives the kernel an FP16x2 mask per shift)
SIGN_BIT = [[15 - u, 31 - u, 7 - u, 23 - u][e] for u in range(8) for e in range(4)]


def load_codebook(path=CODEBOOK_INC):
    """the 4096 x 4 table (int32, values 0..127) from the engine's generated header; checks the sha256"""
    words = [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]{8})u", Path(path).read_text())]
    if len(words) != 4096:
        raise ValueError(f"{path}: {len(words)} codebook entries, expected 4096")
    raw = np.array(words, dtype="<u4").tobytes()
    if hashlib.sha256(raw).hexdigest() != CODEBOOK_SHA256:
        raise ValueError(f"{path}: codebook sha256 mismatch (this quantiser writes {CODEBOOK_TAG})")
    table = np.frombuffer(raw, np.int8).reshape(4096, 4).astype(np.int32)
    assert table.min() >= 0
    return table


# ---------------- rotation: per 128 weights x -> H D x / sqrt(128) (H Sylvester-Hadamard, D = diag of S1's signs) ------
def sign_vec():
    return np.array([-1.0 if (S1[i // 32] >> (i % 32)) & 1 else 1.0 for i in range(BLOCK)])


def _fwht_np(x):
    n = x.shape[-1]
    h = 1
    while h < n:
        y = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        x = np.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), axis=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return x


def rot_last_np(x):
    """x [..., K] (float64 / float32) -> R x per 128-block; for a weight row w this is w R^T (what the files store)"""
    sh = x.shape
    y = _fwht_np(x.reshape(*sh[:-1], sh[-1] // BLOCK, BLOCK) * sign_vec().astype(x.dtype)) / math.sqrt(BLOCK)
    return y.reshape(sh).astype(x.dtype)


def rot_matrix_np():
    return rot_last_np(np.eye(BLOCK)).T  # column j = R e_j, i.e. R


# ---------------- packing (numpy) ----------------
def pack(scales, idx, neg, rows_per_chunk=2048):
    """scales fp16 [N, K/128], idx int [N, K/4] (0..4095), neg bool [N, K] -> uint8 [N, K/128 * 66]"""
    scales, idx, neg = np.asarray(scales), np.asarray(idx), np.asarray(neg)
    N, K = neg.shape
    nb = K // BLOCK
    out = np.empty((N, nb * BLOCK_BYTES), np.uint8)
    bits = np.array(SIGN_BIT, np.uint32)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        ix = idx[r0:r1].astype(np.uint32).reshape(r1 - r0, nb, 4, 8)  # [row, block, unit, group]
        x = ix[..., 0] | (ix[..., 1] << 12) | ((ix[..., 2] & 0xFF) << 24)
        y = (ix[..., 2] >> 8) | (ix[..., 3] << 4) | (ix[..., 4] << 16) | ((ix[..., 5] & 0xF) << 28)
        z = (ix[..., 5] >> 4) | (ix[..., 6] << 8) | (ix[..., 7] << 20)
        w = np.bitwise_or.reduce(neg[r0:r1].reshape(r1 - r0, nb, 4, 32).astype(np.uint32) << bits, axis=-1)
        units = np.stack([x, y, z, w], -1).astype("<u4").reshape(r1 - r0, nb, 64 // 4).view(np.uint8)
        sb = scales[r0:r1].astype("<f2").view(np.uint8).reshape(r1 - r0, nb, 2)
        out[r0:r1] = np.concatenate([sb, units], axis=2).reshape(r1 - r0, nb * BLOCK_BYTES)
    return out


def decode(raw, N, K, table):
    """TQV-12 rows -> float32 [N, K] in the rotated basis (written from the format description, not from pack())"""
    nb = K // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(N, nb, BLOCK_BYTES)
    s = b[:, :, :2].copy().view("<f2").astype(np.float32)[..., 0]
    q = b[:, :, 2:].copy().view("<u4").reshape(N, nb, 4, 4).astype(np.uint64)  # [row, block, unit, word x y z w]
    stream = q[..., 0] | (q[..., 1] << np.uint64(32))  # bits 0-63 of the 96-bit index stream; q[..., 2] = bits 64-95
    out = np.empty((N, nb, 4, 32), np.float32)
    for u in range(8):
        p = 12 * u
        if p + 12 <= 64:
            v = stream >> np.uint64(p)
        elif p >= 64:
            v = q[..., 2] >> np.uint64(p - 64)
        else:
            v = (stream >> np.uint64(p)) | (q[..., 2] << np.uint64(64 - p))
        ix = (v & np.uint64(0xFFF)).astype(np.int64)
        for e in range(4):
            mag = table[ix, e].astype(np.float32) * s[:, :, None]
            neg = (q[..., 3] >> np.uint64(SIGN_BIT[4 * u + e])) & np.uint64(1)
            out[..., 4 * u + e] = np.where(neg == 1, -mag, mag)
    return out.reshape(N, K)


# ---------------- GPU: search and vector GPTQ (PyTorch) ----------------
def _torch():
    import torch  # noqa: PLC0415 - optional dependency, only the quantise step needs it
    return torch


def rot_last(x):
    """torch version of rot_last_np (same device / dtype as x)"""
    torch = _torch()
    sh = x.shape
    s = torch.tensor(sign_vec(), device=x.device, dtype=x.dtype)
    y = x.reshape(*sh[:-1], sh[-1] // BLOCK, BLOCK) * s
    n, h = BLOCK, 1
    while h < n:
        z = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        y = torch.stack((z[..., 0, :] + z[..., 1, :], z[..., 0, :] - z[..., 1, :]), dim=-2).reshape(*y.shape[:-1], n)
        h *= 2
    return (y * (1.0 / math.sqrt(BLOCK))).reshape(sh)


def rotate_h(H):
    """H_rot = Rb H Rb^T for a symmetric [K, K] H (Rb = blockdiag(R)): the Hessian of the rotated weights"""
    return rot_last(rot_last(H).T.contiguous())


_TRITON_KERNEL = None


def _nn_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is None:
        import triton  # noqa: PLC0415
        import triton.language as tl  # noqa: PLC0415

        @triton.jit
        def nn(A, C, CN, IDX, DIST, n, NC: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BN + tl.arange(0, BN)
            m = offs < n
            a0 = tl.load(A + offs * 4 + 0, mask=m, other=0.0)
            a1 = tl.load(A + offs * 4 + 1, mask=m, other=0.0)
            a2 = tl.load(A + offs * 4 + 2, mask=m, other=0.0)
            a3 = tl.load(A + offs * 4 + 3, mask=m, other=0.0)
            best = tl.full([BN], float("inf"), tl.float32)
            bidx = tl.zeros([BN], tl.int32)
            for c0 in range(0, NC, BC):
                co = c0 + tl.arange(0, BC)
                d = tl.load(CN + co)[None, :] - 2.0 * (a0[:, None] * tl.load(C + co * 4 + 0)[None, :]
                                                       + a1[:, None] * tl.load(C + co * 4 + 1)[None, :]
                                                       + a2[:, None] * tl.load(C + co * 4 + 2)[None, :]
                                                       + a3[:, None] * tl.load(C + co * 4 + 3)[None, :])
                mn = tl.min(d, axis=1)
                am = tl.argmin(d, axis=1).to(tl.int32) + c0
                upd = mn < best
                best = tl.where(upd, mn, best)
                bidx = tl.where(upd, am, bidx)
            tl.store(IDX + offs, bidx, mask=m)
            tl.store(DIST + offs, best + a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3, mask=m)

        _TRITON_KERNEL = (nn, triton.cdiv)
    return _TRITON_KERNEL


class Codebook:
    def __init__(self, device="cuda", path=CODEBOOK_INC):
        torch = _torch()
        self.table = load_codebook(path)
        self.step = CODEBOOK_STEP
        self.C = torch.tensor(self.table, dtype=torch.float32, device=device).contiguous()  # integer grid 0..127
        self.CN = (self.C ** 2).sum(1).contiguous()

    def nearest(self, a):
        """a [n, 4] float32 >= 0 in grid units -> (index int32 [n], squared distance [n])"""
        torch = _torch()
        a = a.contiguous()
        n = a.shape[0]
        if a.is_cuda:
            kern, cdiv = _nn_kernel()
            idx = torch.empty(n, dtype=torch.int32, device=a.device)
            dist = torch.empty(n, dtype=torch.float32, device=a.device)
            kern[(cdiv(n, 64),)](a, self.C, self.CN, idx, dist, n, NC=4096, BN=64, BC=64)
            return idx, dist.clamp_min(0)
        idx = torch.empty(n, dtype=torch.int32)
        dist = torch.empty(n, dtype=torch.float32)
        for r in range(0, n, 4096):  # CPU fallback (tests): same expansion as the kernel
            d = self.CN[None, :] - 2.0 * a[r:r + 4096] @ self.C.T
            m, i = d.min(1)
            idx[r:r + 4096] = i.int()
            dist[r:r + 4096] = (m + (a[r:r + 4096] ** 2).sum(1)).clamp_min(0)
        return idx, dist


SCALE_GRID = [0.6 + 0.05 * i for i in range(17)]


def block_scale(w, cb):
    """w [N, 128] float32 (rotated) -> per-row FP16-representable scale minimising the squared error of nearest-codeword
    rounding; candidates rms * g * step for g in SCALE_GRID"""
    torch = _torch()
    N = w.shape[0]
    rms = w.pow(2).mean(1).sqrt()
    a = w.abs()
    best_e = torch.full((N,), float("inf"), device=w.device)
    best_s = torch.zeros(N, device=w.device)
    for g in SCALE_GRID:
        s = (rms * (g * cb.step)).half().float()
        s = torch.where(s > 0, s, torch.full_like(s, 2.0 ** -24))
        _, d = cb.nearest((a / s[:, None]).reshape(-1, 4))
        e = d.reshape(N, 32).sum(1) * s * s
        m = e < best_e
        best_e = torch.where(m, e, best_e)
        best_s = torch.where(m, s, best_s)
    return best_s


DAMP_FORMS = ("mean", "median")


def _damped(H, damp, form="mean"):
    """H + damp * m * I with m the mean (GPTQ's choice) or the median of diag(H). The median is robust to the few
    massive-activation channels that dominate the mean of some inputs (attention q/k/v of early layers)."""
    torch = _torch()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    m = torch.median(torch.diag(H)) if form == "median" else torch.mean(torch.diag(H))
    H += damp * m * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    return H, dead


def _chol_u(H):
    """upper Cholesky factor of H^-1 (H damped)"""
    torch = _torch()
    L, info = torch.linalg.cholesky_ex(H)
    if info.item() != 0:
        L = torch.linalg.cholesky(H.double()).float()
    Hinv = torch.cholesky_inverse(L)
    del L
    U, info = torch.linalg.cholesky_ex(Hinv, upper=True)
    if info.item() != 0:
        U = torch.linalg.cholesky(Hinv.double(), upper=True).float()
    return U


def gptq(W, H, cb, damp=0.01, refit=False, order=False, damp_form="mean"):
    """Vector GPTQ (Frantar et al., 4 columns = one codebook group at a time) in the rotated basis.
    W [N, K] float32 (rotated weights), H [K, K] float32 (rotated, undamped) on one device.
    Encoder options (the stored format is unchanged; the defaults reproduce the PR #56 encoder exactly):
      refit  after a 128-block's codes are chosen, refit its FP16 scale in closed form under GPTQ's effective block
             metric (U_bb^T U_bb)^-1 and propagate the block's exact residual
      damp_form  "mean" (GPTQ) or "median" of diag(H) as the unit of `damp`
      order  process the 128-blocks in descending tr(H_bb) ("block act-order"; 4-groups and scales stay intact)
    Returns scales fp16 [N, K/128], idx int32 [N, K/4], neg bool [N, K], Q float32 [N, K] (the decoded weights)."""
    torch = _torch()
    N, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK
    W = W.clone()
    H, dead = _damped(H, damp, damp_form)
    W[:, dead] = 0
    if order:
        bo = torch.diag(H).reshape(nb, BLOCK).sum(1).argsort(descending=True)
        perm = (bo[:, None] * BLOCK + torch.arange(BLOCK, device=H.device)).reshape(-1)
        W = W[:, perm].contiguous()
        H = H[perm][:, perm].contiguous()
    U = _chol_u(H)
    del H
    Ad = torch.linalg.inv(torch.stack([U[i:i + 4, i:i + 4] for i in range(0, K, 4)]).double()).float()  # 4x4 blocks
    scales = torch.empty(N, nb, dtype=torch.float16, device=W.device)
    idx = torch.empty(N, K // 4, dtype=torch.int32, device=W.device)
    Q = torch.empty_like(W)
    neg = torch.empty(N, K, dtype=torch.bool, device=W.device)
    C = cb.C
    for i1 in range(0, K, BLOCK):
        i2 = i1 + BLOCK
        W0 = W[:, i1:i2].clone()
        W1 = W0.clone()
        E1 = torch.empty_like(W1)
        Ui = U[i1:i2, i1:i2]
        s = block_scale(W1, cb)
        sc = s[:, None]
        cd = torch.empty_like(W1)  # signed codewords (grid units)
        for i in range(0, BLOCK, 4):
            w = W1[:, i:i + 4]
            ix, _ = cb.nearest(w.abs() / sc)
            ng = w < 0
            c = C[ix] * torch.where(ng, -1.0, 1.0)
            q = c * sc
            idx[:, (i1 + i) // 4] = ix
            neg[:, i1 + i:i1 + i + 4] = ng
            cd[:, i:i + 4] = c
            e = (w - q) @ Ad[(i1 + i) // 4]
            W1[:, i:] -= e @ Ui[i:i + 4, i:]
            E1[:, i:i + 4] = e
        if refit:
            M = torch.cholesky_inverse(Ui, upper=True)  # (Ui^T Ui)^-1
            cM = cd @ M
            num = (cM * W0).sum(1)
            den = (cM * cd).sum(1)
            s2 = (num / den.clamp_min(1e-30)).half().float()
            s = torch.where((den > 1e-20) & (s2 > 0) & torch.isfinite(s2), s2, s)
            sc = s[:, None]
            E1 = torch.linalg.solve_triangular(Ui, W0 - cd * sc, upper=True, left=False)
        scales[:, i1 // BLOCK] = s.half()
        Q[:, i1:i2] = cd * sc
        if i2 < K:
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
    if order:
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(K, device=perm.device)
        binv = torch.empty_like(bo)
        binv[bo] = torch.arange(nb, device=bo.device)
        ginv = inv.reshape(-1, 4)[:, 0] // 4
        Q, neg = Q[:, inv].contiguous(), neg[:, inv].contiguous()
        scales, idx = scales[:, binv].contiguous(), idx[:, ginv].contiguous()
    return scales, idx, neg, Q


_KNN = {}


def icm(W, H, cb, scales, idx, neg, sweeps=2, damp=0.1, M=32, damp_form="mean"):
    """local search after GPTQ (iterated conditional modes: coordinate descent over the 4-groups) on
    (W - Q) H_d (W - Q)^T, H_d = H damped by `damp` (the regulariser against calibration noise): per 4-group, move to
    the best of the codeword nearest to the group's unconstrained optimum and its M-1 codebook neighbours (signs of the
    optimum), then a closed-form FP16 refit of every block scale. Rows are independent and run in parallel.
    In-place on scales / idx / neg; returns Q (decoded, exactly what the file decodes to)."""
    torch = _torch()
    N, K = W.shape
    nb = K // BLOCK
    C = cb.C
    if M not in _KNN:
        _KNN[M] = torch.cdist(C, C).topk(M, largest=False).indices
    nn = _KNN[M]
    H, _ = _damped(H, damp, damp_form)
    sgn = torch.where(neg, -1.0, 1.0)
    codes = (C[idx.long()].reshape(N, K) * sgn).contiguous()
    sf = scales.float()
    S = sf.repeat_interleave(BLOCK, 1)
    Q = codes * S
    R = (W - Q) @ H
    Hgg = torch.stack([H[i:i + 4, i:i + 4] for i in range(0, K, 4)])
    Hgi = torch.linalg.inv(Hgg.double()).float()
    ar = torch.arange(N, device=W.device)
    for _ in range(sweeps):
        for gi in range(K // 4):
            i = 4 * gi
            s = S[:, i]
            q = Q[:, i:i + 4]
            tg = (q + R[:, i:i + 4] @ Hgi[gi]) / s[:, None]
            ix, _ = cb.nearest(tg.abs())
            cand = nn[ix.long()]
            sg = torch.where(tg < 0, -1.0, 1.0)
            cv = torch.cat([codes[:, None, i:i + 4], C[cand] * sg[:, None, :]], 1)
            dq = cv * s[:, None, None] - q[:, None, :]
            dE = -2 * (dq * R[:, None, i:i + 4]).sum(-1) + ((dq @ Hgg[gi]) * dq).sum(-1)
            j = dE.argmin(1)
            ok = dE[ar, j] < 0
            if not bool(ok.any()):
                continue
            jj = torch.where(ok, j, torch.zeros_like(j))
            delta = dq[ar, jj]
            codes[:, i:i + 4] = cv[ar, jj]
            newix = cand[ar, (jj - 1).clamp_min(0)].int()
            idx[:, gi] = torch.where(ok, newix, idx[:, gi])
            neg[:, i:i + 4] = torch.where(ok[:, None], sg < 0, neg[:, i:i + 4])
            Q[:, i:i + 4] += delta
            R -= delta @ H[i:i + 4, :]
        for b in range(nb):
            i1, i2 = b * BLOCK, (b + 1) * BLOCK
            c = codes[:, i1:i2]
            Hb = H[i1:i2, i1:i2]
            snew = (sf[:, b] + (c * R[:, i1:i2]).sum(1) / ((c @ Hb) * c).sum(1).clamp_min(1e-30)).half().float()
            snew = torch.where((snew > 0) & torch.isfinite(snew), snew, sf[:, b])
            dq = c * (snew - sf[:, b])[:, None]
            ok = (-2 * (dq * R[:, i1:i2]).sum(1) + ((dq @ Hb) * dq).sum(1)) < 0
            dq = torch.where(ok[:, None], dq, torch.zeros_like(dq))
            sf[:, b] = torch.where(ok, snew, sf[:, b])
            Q[:, i1:i2] += dq
            R -= dq @ H[i1:i2, :]
        S = sf.repeat_interleave(BLOCK, 1)
    scales.copy_(sf.half())
    return codes * S


def rel_err(Wq, W, H):
    """tr(dW H dW^T) / tr(W H W^T): relative output error on the calibration inputs (float64 accumulation)"""
    num = den = 0.0
    Hd = H.double()
    for r in range(0, W.shape[0], 2048):
        d = (Wq[r:r + 2048] - W[r:r + 2048]).double()
        w = W[r:r + 2048].double()
        num += float(((d @ Hd) * d).sum())
        den += float(((w @ Hd) * w).sum())
    return num / den
