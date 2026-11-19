"""tools/convert/qwen35/tqv12_quant.py and tqv12.py (the TQV-12 quantiser) on synthetic data.

- The codebook the quantiser reads (the engine's tqv12_codebook.inc) is the committed table tests/data/tqv12_codebook_i8.bin
  and has the documented sha256.
- The rotation is the engine's: an emulation of tq4_rotate_h_k's butterfly gives the same vector, and it is orthogonal.
- pack() -> decode() round-trips, and decode() matches a literal reading of the bit layout in docs/reference/formats.md.
- skeleton on a tiny template / source pair: TQ4 -> type 305 with tqllm.tqv.codebook, other tensors copied, a BF16 matrix
  stored rotated in the template is written rotated; a template from another model is refused.
- With PyTorch: H from a capture dump equals X^T X; quantize (vector GPTQ, CPU) beats nearest rounding on the calibration
  inputs and its blocks decode to the GPTQ weights; verify passes on the result and fails on a corrupted copy.
- Encoder options (with PyTorch): --order / --refit / --damp-form median give blocks that decode exactly to the returned
  weights (the block permutation is undone) and keep the error below nearest rounding; --icm lowers the dampened
  objective it minimises and stays exact; --damp-role overrides --damp per role and refuses an unknown role.
Runs on the CPU in a few seconds; the PyTorch cases are skipped when it is not installed."""
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/convert/qwen35"))
import tqv12  # noqa: E402
import tqv12_quant as tq  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_GGUF = importlib.util.find_spec("gguf") is not None


def kernel_rotate(x):
    """tq4_rotate_h_k (src/formats/weight/tq4.cu) for one 128-vector: lane l holds x[4l..4l+3], signs from S1, an in-lane
    4-point butterfly, then 5 shuffle rounds, times 1/sqrt(128)"""
    v = np.zeros((32, 4))
    for lane in range(32):
        for j in range(4):
            neg = (tqv12.S1[lane // 8] >> ((lane * 4 + j) % 32)) & 1
            v[lane, j] = -x[lane * 4 + j] if neg else x[lane * 4 + j]
    for lane in range(32):
        a, b, c, d = v[lane]
        a, b, c, d = a + b, a - b, c + d, c - d
        v[lane] = [a + c, b + d, a - c, b - d]
    m = 1
    while m < 32:
        nv = v.copy()
        for lane in range(32):
            nv[lane] = v[lane ^ m] - v[lane] if lane & m else v[lane] + v[lane ^ m]
        v, m = nv, m << 1
    return (v / np.sqrt(128)).reshape(128)


def literal_decode(raw, table):
    """one 66-byte block, bit by bit as the format text says"""
    scale = float(np.frombuffer(raw[:2], "<f2")[0])
    out = []
    for j in range(4):
        x, y, z, w = struct.unpack("<4I", raw[2 + 16 * j:18 + 16 * j])
        stream = x | (y << 32) | (z << 64)
        for u in range(8):
            idx = (stream >> (12 * u)) & 0xFFF
            for e, bit in enumerate((15 - u, 31 - u, 7 - u, 23 - u)):
                v = scale * table[idx][e]
                out.append(-v if (w >> bit) & 1 else v)
    return np.array(out, np.float32)


# ---------------- tiny GGUF writer (template and source) ----------------
def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)) + b)


def write_gguf(path, tensors, kv=()):
    """tensors: [(name, dims (row length first), type, raw bytes)]; kv: [(key, string)]"""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1))
        w_str(f, "general.alignment")
        f.write(struct.pack("<II", 4, 32))
        for k, v in kv:
            w_str(f, k)
            f.write(struct.pack("<I", 8))
            w_str(f, v)
        off, offs = 0, []
        for name, dims, typ, raw in tensors:
            off = (off + 31) // 32 * 32
            offs.append(off)
            w_str(f, name)
            f.write(struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", typ, off))
            off += len(raw)
        while f.tell() % 32:
            f.write(b"\0")
        base = f.tell()
        for (_, _, _, raw), o in zip(tensors, offs):
            f.seek(base + o)
            f.write(raw)


def bf16(a):
    return tq.f32_to_bf16(np.asarray(a, np.float32)).tobytes()


class Codec(unittest.TestCase):
    def test_codebook_is_the_committed_table(self):
        table = tqv12.load_codebook()
        raw = (ROOT / "tests/data/tqv12_codebook_i8.bin").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), tqv12.CODEBOOK_SHA256)
        self.assertTrue(np.array_equal(table, np.frombuffer(raw, np.int8).reshape(4096, 4)))
        self.assertTrue(table.min() >= 0 and table.max() == 127)

    def test_rotation_is_the_engine_rotation(self):
        x = np.random.default_rng(0).standard_normal(256)
        ours = tqv12.rot_last_np(x)
        self.assertLess(np.abs(ours - np.concatenate([kernel_rotate(x[:128]), kernel_rotate(x[128:])])).max(), 1e-12)
        R = tqv12.rot_matrix_np()
        self.assertLess(np.abs(R @ R.T - np.eye(128)).max(), 1e-12)
        W = np.random.default_rng(1).standard_normal((3, 256))  # files hold W R^T, the engine feeds R x
        self.assertLess(np.abs(tqv12.rot_last_np(W) @ ours - W @ x).max(), 1e-9)

    def test_pack_decode_round_trip(self):
        rng = np.random.default_rng(2)
        N, K = 5, 256
        table = tqv12.load_codebook()
        scales = (rng.random((N, K // 128)) * 0.01 + 1e-3).astype(np.float16)
        idx = rng.integers(0, 4096, (N, K // 4))
        neg = rng.random((N, K)) < 0.5
        raw = tqv12.pack(scales, idx, neg)
        self.assertEqual(raw.shape, (N, K // 128 * 66))
        want = (table[idx].reshape(N, K) * np.repeat(scales.astype(np.float32), 128, 1) * np.where(neg, -1, 1))
        dec = tqv12.decode(raw.tobytes(), N, K, table)
        self.assertTrue(np.array_equal(dec, want.astype(np.float32)))
        blk = raw[3, 66:132].tobytes()
        self.assertTrue(np.array_equal(literal_decode(blk, table), dec[3, 128:]))


@unittest.skipUnless(HAVE_GGUF, "needs gguf (pip install gguf)")
class Pipeline(unittest.TestCase):
    """template: blk.0.ffn_up / ffn_down / ssm_out / attn_qkv as TQ4 (zero bytes), a rotated BF16 ssm_alpha, an F32 norm"""
    R, C, F = 8, 256, 384  # rows of the small matrices, hidden size, ffn size

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(3)
        self.w = {"blk.0.attn_qkv.weight": (rng.standard_normal((self.R, self.C)) * 0.02).astype(np.float32),
                  "blk.0.ssm_out.weight": (rng.standard_normal((self.C, self.C)) * 0.02).astype(np.float32),
                  "blk.0.ffn_up.weight": (rng.standard_normal((self.F, self.C)) * 0.02).astype(np.float32),
                  "blk.0.ffn_down.weight": (rng.standard_normal((self.C, self.F)) * 0.02).astype(np.float32),
                  "blk.0.ssm_alpha.weight": (rng.standard_normal((4, self.C)) * 0.1).astype(np.float32)}
        self.norm = (1 + rng.standard_normal(self.C) * 0.1).astype(np.float32)
        src, tpl = [], []
        for n, w in self.w.items():
            src.append((n, [w.shape[1], w.shape[0]], 30, bf16(w)))
            if n.endswith("ssm_alpha.weight"):
                tpl.append((n, [w.shape[1], w.shape[0]], 30, bf16(tqv12.rot_last_np(w.astype(np.float64)))))
            else:
                tpl.append((n, [w.shape[1], w.shape[0]], 300, bytes(w.size // 128 * 66)))
        src.append(("blk.0.attn_norm.weight", [self.C], 0, self.norm.tobytes()))
        tpl.append(("blk.0.attn_norm.weight", [self.C], 0, self.norm.tobytes()))
        self.src, self.tpl, self.out = self.d / "src.gguf", self.d / "tpl.gguf", self.d / "out.gguf"
        write_gguf(self.src, src, [("general.architecture", "qwen35")])
        write_gguf(self.tpl, tpl, [("general.architecture", "qwen35"), ("general.name", "tiny TQ_L")])

    def tearDown(self):
        self.tmp.cleanup()

    def skeleton(self):
        tq.main(["skeleton", "--bf16", str(self.src), "--template", str(self.tpl), "--out", str(self.out)])

    def test_skeleton_layout(self):
        self.skeleton()
        kv, tensors, _ = tq.read_header(str(self.out))
        meta = {k.decode(): v for k, _, v in kv}
        self.assertEqual(meta["tqllm.tqv.codebook"], b"tqv12-v1")
        types = {n.decode(): t for n, _, t, *_ in tensors}
        self.assertEqual(sorted(n for n, t in types.items() if t == 305),
                         sorted(n for n in self.w if not n.endswith("ssm_alpha.weight")))
        plan = tq.load_plan(self.out)
        alpha = next(p for p in plan["tensors"] if p["name"] == "blk.0.ssm_alpha.weight")
        self.assertTrue(alpha["rotated"])
        self.assertEqual(sorted(tq.tqv_tensors(plan)[0]), ["down", "gu", "mix", "out"])

    def test_template_of_another_model_is_refused(self):
        write_gguf(self.src, [(n, [w.shape[1], w.shape[0]], 30, bf16(w * 3)) for n, w in self.w.items()]
                   + [("blk.0.attn_norm.weight", [self.C], 0, self.norm.tobytes())])
        with self.assertRaises(SystemExit):
            self.skeleton()

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
    def test_capture_h_quantize_verify(self):
        import torch
        self.skeleton()
        rng = np.random.default_rng(4)
        hdir = self.d / "h"
        hdir.mkdir()
        for role, k in (("mix", self.C), ("gu", self.C), ("out", self.C), ("down", self.F)):
            X = (rng.standard_normal((700, k)) * np.linspace(0.2, 3, k)).astype(np.float16)
            X.tofile(hdir / f"L0_{role}.x16")
            n = tq.build_h(str(hdir / f"L0_{role}.x16"), k, str(hdir / f"L0_{role}.h32"), "plain")
            H, kk, tokens, basis = tq.load_h(hdir / f"L0_{role}.h32")
            X32 = X.astype(np.float32)
            self.assertEqual((n, kk, tokens, basis), (700, k, 700, 0))
            self.assertLess(np.abs(H - X32.T @ X32).max() / np.abs(H).max(), 1e-5)
        cb_cpu = tqv12.Codebook("cpu")  # the CPU search path: same expansion as the Triton kernel
        tq.main(["quantize", "--out", str(self.out), "--bf16", str(self.src), "--hdir", str(hdir), "--layers", "0",
                 "--keep-h", "--device", "cpu"])
        plan = tq.load_plan(self.out)
        self.assertEqual(len(plan["done"]), 4)
        # GPTQ beats plain nearest rounding (same scales search) on the calibration inputs
        Hn, *_ = tq.load_h(hdir / "L0_down.h32")
        Hr = tqv12.rotate_h(torch.from_numpy(Hn).double()).float()
        Wr = tqv12.rot_last(torch.from_numpy(self.w["blk.0.ffn_down.weight"]))
        Wr = torch.from_numpy(tq.bf16_to_f32(tq.f32_to_bf16(Wr.numpy())).reshape(Wr.shape))
        s = torch.stack([tqv12.block_scale(Wr[:, b:b + 128], cb_cpu) for b in range(0, self.F, 128)], 1)
        ix, _ = cb_cpu.nearest((Wr.abs() / s.repeat_interleave(128, 1)).reshape(-1, 4))
        Qr = cb_cpu.C[ix.long()].reshape(Wr.shape) * torch.where(Wr < 0, -1.0, 1.0) * s.repeat_interleave(128, 1)
        self.assertLess(plan["done"]["blk.0.ffn_down.weight"]["rel_out_err"], tqv12.rel_err(Qr, Wr, Hr))
        self.assertEqual(tq.main(["verify", "--out", str(self.out), "--bf16", str(self.src), "--template", str(self.tpl)]), 0)
        p = next(p for p in plan["tensors"] if p["name"] == "blk.0.ffn_up.weight")
        with open(self.out, "r+b") as f:  # a zero block (a tensor written only in part) fails verify
            f.seek(p["offset"])
            f.write(bytes(66))
        self.assertEqual(tq.main(["verify", "--out", str(self.out), "--bf16", str(self.src), "--template", str(self.tpl)]), 1)


class EncoderOptions(unittest.TestCase):
    def test_damp_role(self):
        import argparse
        a = argparse.Namespace(damp=0.01, damp_role="mix:0.1,down:0.3")
        self.assertEqual((tq.role_damp(a, "mix"), tq.role_damp(a, "down"), tq.role_damp(a, "gu")), (0.1, 0.3, 0.01))
        with self.assertRaises(SystemExit):
            tq.role_damp(argparse.Namespace(damp=0.01, damp_role="mlp:0.1"), "mix")

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
    def test_options_decode_exactly(self):
        import torch
        torch.manual_seed(5)
        N, K = 48, 512
        cb = tqv12.Codebook("cpu")
        X = torch.randn(900, K) * torch.linspace(0.1, 4, K)[torch.randperm(K)]
        X[:, 7] *= 40  # one massive-activation channel: the median and the mean of diag(H) differ
        H = X.T @ X
        W = torch.randn(N, K)
        s0, i0, n0, Q0 = tqv12.gptq(W, H, cb)
        e0 = tqv12.rel_err(Q0, W, H)
        for kw in (dict(order=True), dict(refit=True), dict(damp_form="median", damp=0.1),
                   dict(order=True, refit=True, damp_form="median", damp=0.1)):
            s, idx, neg, Q = tqv12.gptq(W, H, cb, **kw)
            dec = tqv12.decode(tqv12.pack(s.numpy(), idx.numpy(), neg.numpy()).tobytes(), N, K, cb.table)
            self.assertTrue(np.array_equal(dec, Q.numpy()), kw)
            self.assertLess(tqv12.rel_err(Q, W, H), 1.5 * e0, kw)
        s, idx, neg, Q = tqv12.gptq(W, H, cb, order=True, refit=True, damp_form="median", damp=0.1)
        Hd, _ = tqv12._damped(H, 0.1, "median")
        before = float(torch.einsum("nk,kl,nl->", W - Q, Hd, W - Q))
        Q2 = tqv12.icm(W, H, cb, s, idx, neg, sweeps=2, damp=0.1, damp_form="median")
        after = float(torch.einsum("nk,kl,nl->", W - Q2, Hd, W - Q2))
        self.assertLess(after, before)
        dec = tqv12.decode(tqv12.pack(s.numpy(), idx.numpy(), neg.numpy()).tobytes(), N, K, cb.table)
        self.assertTrue(np.array_equal(dec, Q2.numpy()))


if __name__ == "__main__":
    unittest.main()
