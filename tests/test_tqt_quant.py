"""tools/convert/qwen35/tqt.py (the TQT codec / encoder of tqv12_quant.py --format tqt) on synthetic data.

- code_values() equals a literal, integer-by-integer reading of code c2 in docs/reference/formats.md (Python ints, FP16
  via struct), and the table is the documented distribution (zero mean, unit-order RMS, no pair correlation).
- pack() -> decode() round-trips the scales and stream bytes; decode() reads window p from bytes p, p+1, p+2 mod 64.
- skeleton --format tqt on a tiny template / source pair: TQ4 -> type 306 with tqllm.tqt.code = tqt-c2-v1.
- With PyTorch + CUDA + nvcc: the Viterbi encoder's stream decodes to its own reconstruction, beats plain rounding to
  the nearest code pair by a wide margin on Gaussian data, and gptq() on a small matrix gives blocks that decode exactly
  to its returned weights and a lower H-weighted error than per-block encoding without error feedback.
Runs in seconds; the CUDA cases are skipped without PyTorch / a GPU / nvcc."""
import importlib.util
import shutil
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/convert/qwen35"))
import tqt  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def f16(u):
    return struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0]


def literal_pair(w):
    """docs/reference/formats.md, TQT: one window -> (weight 2p, weight 2p+1)"""
    M = 0xFFFFFFFF
    x = ((w * 0x9E3779B1 + 0x7F4A7C15) << 14) & M
    y = ((x << 16) | (x >> 16)) & M
    z = (y * 0x2C1B3C6D + 0x9E3779B9) & M
    z2 = (z * 0x297A2D39 + 0x7F4A7C15) & M
    h1, h2 = (z & 0x8FFF8FFF) ^ 0x3B603B60, (z2 & 0x8FFF8FFF) ^ 0x3B603B60
    rnd = lambda v: struct.unpack("<e", struct.pack("<e", v))[0]  # noqa: E731 - FP16 round to nearest even
    return rnd(f16(h1) + f16(h2)), rnd(f16(h1 >> 16) + f16(h2 >> 16))


def cuda_ok():
    if not HAVE_TORCH or not shutil.which("nvcc"):
        return False
    import torch  # noqa: PLC0415
    return torch.cuda.is_available()


class Codec(unittest.TestCase):
    def test_code_literal(self):
        rng = np.random.default_rng(1)
        ws = np.concatenate([rng.integers(0, 1 << 18, 400), [0, 1, (1 << 18) - 1]])
        got = tqt.code_values(ws)
        for w, g in zip(ws.tolist(), got):
            self.assertEqual(tuple(g.tolist()), literal_pair(w), w)

    def test_table_stats(self):
        t = tqt.table().astype(np.float64)
        self.assertEqual(t.shape, (1 << 18, 2))
        self.assertLess(abs(t.mean()), 0.01 * t.std())
        self.assertLess(abs((t[:, 0] * t[:, 1]).mean()) / t.var(), 0.01)
        kurt = (t ** 4).mean(0) / (t ** 2).mean(0) ** 2
        self.assertTrue(((kurt > 2.6) & (kurt < 3.2)).all(), kurt)  # near-Gaussian (3)

    def test_pack_decode(self):
        rng = np.random.default_rng(2)
        N, K = 3, 384
        sc = (rng.uniform(0.5, 2, (N, K // 128))).astype(np.float16)
        st = rng.integers(0, 256, (N, K // 128, 64), dtype=np.uint8)
        raw = tqt.pack(sc, st)
        self.assertEqual(raw.shape, (N, K // 128 * 66))
        b = raw.reshape(N, K // 128, 66)
        self.assertTrue(np.array_equal(b[:, :, 2:], st))
        self.assertTrue(np.array_equal(b[:, :, :2].copy().view("<f2")[..., 0], sc))
        dec = tqt.decode(raw.tobytes(), N, K)
        for r, blk, p in [(0, 0, 0), (1, 2, 63), (2, 1, 62), (0, 1, 17)]:
            w = (int(st[r, blk, p]) | int(st[r, blk, (p + 1) % 64]) << 8 | int(st[r, blk, (p + 2) % 64]) << 16) & 0x3FFFF
            a, c = literal_pair(w)
            s = float(sc[r, blk])
            self.assertEqual(float(dec[r, blk * 128 + 2 * p]), np.float32(a) * np.float32(s))
            self.assertEqual(float(dec[r, blk * 128 + 2 * p + 1]), np.float32(c) * np.float32(s))



sys.path.insert(0, str(ROOT / "tests"))
import test_tqv12_quant as tv  # noqa: E402 - the tiny template / source fixture
import tqv12_quant as tq  # noqa: E402


class Pipeline(tv.Pipeline):
    """tqv12_quant.py with --format tqt on the TQV-12 test's tiny template / source pair"""

    def skeleton(self):
        tq.main(["skeleton", "--bf16", str(self.src), "--template", str(self.tpl), "--out", str(self.out), "--format", "tqt"])

    def test_skeleton_layout(self):
        self.skeleton()
        kv, tensors, _ = tq.read_header(str(self.out))
        meta = {k.decode(): v for k, _, v in kv}
        self.assertEqual(meta["tqllm.tqt.code"], b"tqt-c2-v1")
        self.assertNotIn("tqllm.tqv.codebook", meta)
        types = {n.decode(): t for n, _, t, *_ in tensors}
        self.assertEqual(sorted(n for n, t in types.items() if t == 306),
                         sorted(n for n in self.w if not n.endswith("ssm_alpha.weight")))
        self.assertEqual(tq.load_plan(self.out)["format"], "tqt")

    @unittest.skipUnless(cuda_ok(), "needs PyTorch with CUDA and nvcc")
    def test_capture_h_quantize_verify(self):
        self.skeleton()
        rng = np.random.default_rng(4)
        hdir = self.d / "h"
        hdir.mkdir()
        for role, k in (("mix", self.C), ("gu", self.C), ("out", self.C), ("down", self.F)):
            X = (rng.standard_normal((700, k)) * np.linspace(0.2, 3, k)).astype(np.float16)
            X.tofile(hdir / f"L0_{role}.x16")
            tq.build_h(str(hdir / f"L0_{role}.x16"), k, str(hdir / f"L0_{role}.h32"), "plain")
        tq.main(["quantize", "--out", str(self.out), "--bf16", str(self.src), "--hdir", str(hdir), "--layers", "0",
                 "--keep-h", "--tqt-scales", "3"])
        plan = tq.load_plan(self.out)
        self.assertEqual(len(plan["done"]), 4)
        self.assertTrue(all(r["decode_exact"] for r in plan["done"].values()))
        self.assertEqual(tq.main(["verify", "--out", str(self.out), "--bf16", str(self.src), "--template", str(self.tpl)]), 0)
        p = next(p for p in plan["tensors"] if p["name"] == "blk.0.ffn_up.weight")
        with open(self.out, "r+b") as f:  # a zero block fails verify
            f.seek(p["offset"])
            f.write(bytes(66))
        self.assertEqual(tq.main(["verify", "--out", str(self.out), "--bf16", str(self.src), "--template", str(self.tpl)]), 1)




@unittest.skipUnless(cuda_ok(), "needs PyTorch with CUDA and nvcc")
class Encoder(unittest.TestCase):
    def test_viterbi(self):
        import torch  # noqa: PLC0415
        torch.manual_seed(0)
        X = torch.randn(256, 128, device="cuda") * 0.9 * tqt.code_rms()
        st = tqt.viterbi(X)
        v = torch.from_numpy(tqt.table()).cuda()[tqt.windows_t(st)].reshape(256, 128)
        mse = float(((v - X) ** 2).mean() / (X ** 2).mean())
        # Gaussian source at 4 bits: D(R) = 0.0039; the trellis reaches about 0.0047 (measured), TQV-12 0.0069
        self.assertLess(mse, 0.0053)
        st_np = st.cpu().numpy()
        dec = tqt.decode(tqt.pack(np.ones((256, 1), np.float16), st_np[:, None, :]).tobytes(), 256, 128)
        self.assertTrue(np.array_equal(dec, v.cpu().numpy()))

    def test_gptq(self):
        import torch  # noqa: PLC0415
        torch.manual_seed(1)
        N, K = 64, 512
        X = torch.randn(4096, K, device="cuda") @ (torch.eye(K, device="cuda") + 0.3 * torch.randn(K, K, device="cuda") / K ** 0.5)
        H = X.T @ X
        W = torch.randn(N, K, device="cuda") * 0.02
        s, st, Q = tqt.gptq(W, H)
        raw = tqt.pack(s.cpu().numpy(), st.cpu().numpy())
        self.assertTrue(np.array_equal(tqt.decode(raw.tobytes(), N, K), Q.cpu().numpy()))
        e_gptq = tqt.rel_err(Q, W, H)
        # no feedback: every block encoded on its own weights
        sc = (W.reshape(N, -1, 128).pow(2).mean(2).sqrt() * (1.1 / tqt.code_rms())).half().float()
        st0 = tqt.viterbi((W.reshape(N, -1, 128) / sc[..., None]).reshape(-1, 128))
        q0 = torch.from_numpy(tqt.table()).cuda()[tqt.windows_t(st0)].reshape(N, -1, 128) * sc[..., None]
        e_plain = tqt.rel_err(q0.reshape(N, K), W, H)
        self.assertLess(e_gptq, 0.9 * e_plain)


if __name__ == "__main__":
    unittest.main()
