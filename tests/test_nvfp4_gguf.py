"""tools/fn_nvfp4_gguf.py: a tiny tqLLM-style GGUF + a modelopt NVFP4 safetensors shard -> one GGUF whose expert tensors are
NVFP4G blocks (F32 global, the E4M3 scales, NVIDIA's code bytes) and whose other tensors are byte-identical."""
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fn_nvfp4_gguf as conv  # noqa: E402

E, ROWS, COLS = 2, 128, 256  # experts, out rows, in cols (gate / up); down is COLS x ROWS


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)) + b)


def write_gguf(path, tensors):
    """tensors: [(name, dims (ne0 first), float32 array)] as F32."""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
        w_str(f, "general.alignment"); f.write(struct.pack("<II", 4, 32))
        off = 0; offs = []
        for name, dims, a in tensors:
            off = (off + 31) // 32 * 32; offs.append(off); off += a.nbytes
        for (name, dims, a), o in zip(tensors, offs):
            w_str(f, name); f.write(struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", 0, o))
        base = (f.tell() + 31) // 32 * 32
        for (name, dims, a), o in zip(tensors, offs):
            f.seek(base + o); f.write(a.tobytes())


def write_safetensors(path, arrays):
    hdr, blobs, off = {}, [], 0
    for k, (dtype, shape, raw) in arrays.items():
        hdr[k] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}; blobs.append(raw); off += len(raw)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class Nvfp4Gguf(unittest.TestCase):
    def test_e4m3(self):
        self.assertEqual(conv.E4M3[0x38], 1.0)
        self.assertEqual(conv.E4M3[0x7E], 448.0)
        self.assertEqual(conv.E4M3[0xB8], -1.0)
        self.assertEqual(conv.E4M3[0x01], 2.0 ** -9)

    def test_convert(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            ours, out, st, idx = (os.path.join(d, n) for n in ("ours.gguf", "out.gguf", "s.safetensors", "index.json"))
            embd = rng.standard_normal(32 * 4).astype(np.float32)
            dense = [("token_embd.weight", [32, 4], embd)]
            exps = [(f"blk.0.ffn_{p}_exps.weight", [c, r, E], np.zeros(E * r * c, np.float32))
                    for p, r, c in (("gate", ROWS, COLS), ("up", ROWS, COLS), ("down", COLS, ROWS))]
            write_gguf(ours, dense[:1] + exps)
            arrays, expect = {}, {}
            for e in range(E):
                for p, r, c in (("gate", ROWS, COLS), ("up", ROWS, COLS), ("down", COLS, ROWS)):
                    k = f"model.language_model.layers.0.mlp.experts.{e}.{p}_proj"
                    codes = rng.integers(0, 256, (r, c // 2), dtype=np.uint8)
                    scales = rng.integers(0x20, 0x50, (r, c // 16), dtype=np.uint8)
                    g = np.float32(0.0123)
                    arrays[k + ".weight"] = ("U8", [r, c // 2], codes.tobytes())
                    arrays[k + ".weight_scale"] = ("F8_E4M3", [r, c // 16], scales.tobytes())
                    arrays[k + ".weight_scale_2"] = ("F32", [], g.tobytes())
                    arrays[k + ".input_scale"] = ("F32", [], np.float32(1).tobytes())
                    expect[(e, p)] = (codes, scales, g)
            # one matrix's scale in a second shard (NVIDIA's shards split an expert's tensors at the boundary)
            split = "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale"
            st2 = os.path.join(d, "s2.safetensors")
            write_safetensors(st, {k: v for k, v in arrays.items() if k != split})
            write_safetensors(st2, {split: arrays[split]})
            with open(idx, "w") as f:
                json.dump({"weight_map": {k: ("s2.safetensors" if k == split else "s.safetensors") for k in arrays}}, f)
            conv.plan(ours, idx, out)
            os.rename(st2, st2 + ".later")  # the second shard is not downloaded yet: that matrix waits
            conv.shard(out, st, idx)
            self.assertFalse(conv.check(out, idx))
            os.rename(st2 + ".later", st2)
            conv.shard(out, st2, idx)  # the run over the next shard (the first still on disk) completes it
            self.assertTrue(conv.check(out, idx))
            conv.shard(out, st, idx)  # a re-run (resume) writes nothing twice
            self.assertTrue(conv.check(out, idx))
            _, _, _, tensors, _, data = conv.read_gguf_header(out)
            byname = {n: (dims, t, o) for n, dims, t, o in tensors}
            with open(out, "rb") as fr:
                raw = fr.read()
            dims, t, o = byname["token_embd.weight"]
            self.assertEqual(t, 0)
            self.assertEqual(raw[data + o:data + o + embd.nbytes], embd.tobytes())
            for p, r, c in (("gate", ROWS, COLS), ("up", ROWS, COLS), ("down", COLS, ROWS)):
                dims, t, o = byname[f"blk.0.ffn_{p}_exps.weight"]
                self.assertEqual((dims, t), ([c, r, E], conv.NVFP4G))
                per = r * c // 128 * 76
                for e in range(E):
                    blk = np.frombuffer(raw, np.uint8, per, data + o + e * per).reshape(r, c // 128, 76)
                    codes, scales, g = expect[(e, p)]
                    self.assertTrue(np.array_equal(blk[:, :, 12:].reshape(r, c // 2), codes), f"{p} {e} codes")
                    self.assertTrue(np.array_equal(blk[:, :, 4:12].reshape(r, c // 16), scales), f"{p} {e} scales")
                    self.assertTrue(np.all(blk[:, :, :4].copy().view(np.float32) == g), f"{p} {e} global")

if __name__ == "__main__":
    unittest.main()
