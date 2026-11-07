#!/usr/bin/env python3
"""Flash-Next with NVIDIA's NVFP4 experts: one GGUF = every non-expert tensor (and the metadata) of a tqLLM Flash-Next GGUF
+ the routed experts of nvidia/Qwen3.8-Flash-Next-NVFP4 (modelopt NVFP4) as GGUF NVFP4G (type 323).

NVFP4G block (128 weights, 76 bytes): the matrix's F32 global scale (weight_scale_2), the 8 E4M3 group scales, then the 64 code
bytes as stored by modelopt (E2M1, low nibble = even weight): NVIDIA's bytes, nothing rounded (4.75 bits per weight on disk;
tqLLM keeps one global per row in memory, 4.5 bits per weight).

The output is written in place at final offsets, so the NVIDIA shards can be fetched and deleted one at a time:
  fn_nvfp4_gguf.py plan  OURS.gguf INDEX.json OUT.gguf     header + every non-expert tensor; experts zero
  fn_nvfp4_gguf.py shard OUT.gguf SHARD.safetensors [INDEX] converts the experts held by that shard (records them in OUT.done)
  fn_nvfp4_gguf.py check OUT.gguf INDEX.json                every expert matrix written exactly once
"""
import json
import os
import re
import struct
import sys

import numpy as np

NVFP4G = 323
ALIGN_DEFAULT = 32
EXP_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()


def nvfp4g_blocks(codes, scales, global_scale):
    """codes uint8 [rows, cols/2], scales uint8 (E4M3) [rows, cols/16], global float -> uint8 [rows * cols/128 * 76]."""
    rows, half = codes.shape
    nb = half // 64
    if half % 64 or scales.shape != (rows, half // 8):
        raise ValueError(f"shapes codes {codes.shape} scales {scales.shape}: need cols % 128 == 0")
    g = np.float32(global_scale)
    if not np.isfinite(g) or not np.all(np.isfinite(E4M3[scales])):
        raise ValueError("NVFP4G: a scale is not finite")
    out = np.empty((rows, nb, 76), np.uint8)
    out[:, :, :4] = np.frombuffer(g.tobytes(), np.uint8)
    out[:, :, 4:12] = scales.reshape(rows, nb, 8)
    out[:, :, 12:] = codes.reshape(rows, nb, 64)
    return out.reshape(-1)


# --- GGUF (v3) reading of the source header -------------------------------------------------------------------------
def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode()


_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_value(f, t):
    if t in _SCALAR:
        f.read(_SCALAR[t])
    elif t == 8:
        _read_str(f)
    elif t == 9:
        (it,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if it in _SCALAR:
            f.read(_SCALAR[it] * n)
        else:
            for _ in range(n):
                _skip_value(f, it)
    else:
        raise ValueError(f"GGUF value type {t}")


def read_gguf_header(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        (version,) = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        kv_start = f.tell()
        align = ALIGN_DEFAULT
        for _ in range(n_kv):
            key = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and t == 4:
                (align,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, t)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            typ, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, typ, off))
        data_start = (f.tell() + align - 1) // align * align
    with open(path, "rb") as f:
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)
    return version, n_kv, kv_bytes, tensors, align, data_start


# GGML block sizes (elements, bytes) for the source's tensor types
_BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 20: (32, 18),
           21: (256, 110), 23: (256, 136), 18: (256, 98), 22: (256, 82), 16: (256, 66), 17: (256, 74), 29: (256, 56),
           19: (256, 50), 10: (256, 84), 11: (256, 110), 46: (32, 16), NVFP4G: (128, 76)}


def tensor_bytes(dims, typ):
    if typ not in _BLOCKS:
        raise ValueError(f"unknown GGML type {typ}")
    e, b = _BLOCKS[typ]
    n = 1
    for d in dims:
        n *= d
    if n % e:
        raise ValueError(f"{n} elements not a multiple of the type-{typ} block")
    return n // e * b


def _w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def plan(ours, index_path, out):
    version, n_kv, kv_bytes, tensors, align, data_start = read_gguf_header(ours)
    with open(index_path) as fi:
        wm = json.load(fi)["weight_map"]
    layers = {int(m.group(1)) for k in wm for m in [EXP_RE.match(k)] if m}
    new = []
    for name, dims, typ, off in tensors:
        m = re.match(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if m and int(m.group(1)) in layers:
            new.append((name, dims, NVFP4G, None, None))
        else:
            new.append((name, dims, typ, off, tensor_bytes(dims, typ)))
    extra_key = b"tqllm.experts.source"
    with open(out, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", version, len(new), n_kv + 1))
        f.write(kv_bytes)
        _w_str(f, extra_key.decode())
        f.write(struct.pack("<I", 8))
        _w_str(f, "nvidia/Qwen3.8-Flash-Next-NVFP4 (modelopt NVFP4 experts as NVFP4G); other tensors from " + os.path.basename(ours))
        offs, cur = [], 0
        for name, dims, typ, _, _ in new:
            cur = (cur + align - 1) // align * align
            offs.append(cur)
            cur += tensor_bytes(dims, typ)
        for (name, dims, typ, _, _), o in zip(new, offs):
            _w_str(f, name)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack(f"<{len(dims)}Q", *dims))
            f.write(struct.pack("<IQ", typ, o))
        base = (f.tell() + align - 1) // align * align
        f.truncate(base + cur)
        with open(ours, "rb") as src:
            for (name, dims, typ, soff, nbytes), o in zip(new, offs):
                if soff is None:
                    continue
                src.seek(data_start + soff)
                f.seek(base + o)
                left = nbytes
                while left:
                    chunk = src.read(min(left, 1 << 26))
                    if not chunk:
                        raise IOError(f"{ours}: short read in {name}")
                    f.write(chunk)
                    left -= len(chunk)
    with open(out + ".layout.json", "w") as fl:
        json.dump({"base": base, "tensors": {n: [d, t, o] for (n, d, t, _, _), o in zip(new, offs)}}, fl)
    print(f"planned {out}: {len(new)} tensors, {sum(1 for t in new if t[2] == NVFP4G)} NVFP4G expert tensors, "
          f"{(base + cur) / 1e9:.1f} GB")


# --- safetensors ------------------------------------------------------------------------------------------------------
def safetensors_index(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def shard(out, path, index_path=None):
    """Converts the expert matrices whose codes are in `path`; a scale stored in another shard is read from that shard's
    file in the same folder (index_path maps tensor -> file). A matrix whose other shard is not on disk yet is left for the
    run over that shard (keep the previous shard until the next one is done)."""
    with open(out + ".layout.json") as fl:
        lay = json.load(fl)
    base, tens = lay["base"], lay["tensors"]
    folder = os.path.dirname(os.path.abspath(path))
    wm = {}
    if index_path:
        with open(index_path) as fi:
            wm = json.load(fi)["weight_map"]
    opened = {}

    def source(fname):
        if fname not in opened:
            p = os.path.join(folder, fname)
            if not os.path.exists(p):
                opened[fname] = None
            else:
                hdr, data0 = safetensors_index(p)
                opened[fname] = (hdr, data0, np.memmap(p, np.uint8, mode="r"))
        return opened[fname]

    here = os.path.basename(path)
    hdr0, _, _ = source(here)
    done_path = out + ".done"
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as fd:
            done = set(fd.read().split())
    n_written = n_waiting = 0
    with open(out, "r+b") as f, open(done_path, "a") as dl:
        keys = set(hdr0)
        for k, v in wm.items():  # matrices of this shard's experts whose codes live elsewhere but are now both on disk
            if v != here and EXP_RE.match(k):
                keys.add(k)
        for key in sorted(keys):
            m = EXP_RE.match(key)
            if not m or key in done:
                continue
            layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            pre = key[: -len(".weight")]

            def raw(k):
                src = source(wm.get(k, here)) if wm else source(here)
                if src is None or k not in src[0]:
                    return None, None
                hdr, data0, mm = src
                i = hdr[k]
                a, b = i["data_offsets"]
                return np.asarray(mm[data0 + a:data0 + b]), i

            parts = [raw(key), raw(pre + ".weight_scale"), raw(pre + ".weight_scale_2")]
            if any(p[0] is None for p in parts):
                n_waiting += 1
                continue
            (codes, ci), (scales, si), (g, gi) = parts
            if ci["dtype"] != "U8" or si["dtype"] != "F8_E4M3" or gi["dtype"] != "F32":
                raise ValueError(f"{key}: dtypes {ci['dtype']} {si['dtype']} {gi['dtype']}")
            rows, half = ci["shape"]
            blocks = nvfp4g_blocks(codes.reshape(rows, half), scales.reshape(rows, half // 8), g.view(np.float32)[0])
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            dims, typ, off = tens[name]
            if typ != NVFP4G or dims[0] != half * 2 or dims[1] != rows:
                raise ValueError(f"{name}: planned {dims} type {typ}, NVIDIA {rows} x {half * 2}")
            per = rows * half * 2 // 128 * 76
            f.seek(base + off + expert * per)
            f.write(blocks.tobytes())
            dl.write(key + "\n")
            done.add(key)
            n_written += 1
    print(f"{here}: {n_written} expert matrices written, {n_waiting} wait for another shard")


def check(out, index_path):
    with open(index_path) as fi:
        need = {k for k in json.load(fi)["weight_map"] if EXP_RE.match(k)}
    done = []
    if os.path.exists(out + ".done"):
        with open(out + ".done") as fd:
            done = fd.read().split()
    dup = len(done) - len(set(done))
    missing = need - set(done)
    print(f"expert matrices: {len(set(done))} of {len(need)} written, {len(missing)} missing, {dup} written twice")
    return not missing and not dup


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "plan" and len(sys.argv) == 5:
        plan(*sys.argv[2:])
    elif cmd == "shard" and len(sys.argv) in (4, 5):
        shard(*sys.argv[2:])
    elif cmd == "check" and len(sys.argv) == 4:
        sys.exit(0 if check(*sys.argv[2:]) else 1)
    else:
        sys.exit(__doc__)
