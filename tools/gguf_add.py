#!/usr/bin/env python3
"""gguf_add.py IN EXTRA OUT: copy GGUF IN with the tensors and metadata of GGUF EXTRA added (names must not repeat;
EXTRA's keys are written as <arch>.<key> of IN's architecture). Tensor data is copied byte for byte. No dependencies,
so it runs where the model is (e.g. QSA indexers from qsa_indexer_gguf.py onto a Mac model file)."""
import os, struct, sys

sys.path.insert(0, os.path.dirname(__file__))
from gguf_truncate import read_header, w_string, w_value


def main():
    src, extra, dst = sys.argv[1:4]
    kv, tensors, align = read_header(src)
    kv2, more, _ = read_header(extra)
    names = {t[0] for t in tensors}
    assert not any(t[0] in names for t in more), "EXTRA repeats a tensor of IN"
    arch = next(v for k, t, v in kv if k == b"general.architecture")
    kv = kv + [[arch + b"." + k, t, v] for k, t, v in kv2]
    tensors = tensors + more
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv: w_string(out, k); out += struct.pack("<I", t); w_value(out, t, v)
    off = 0
    for name, dims, typ, f, start, size in tensors:
        w_string(out, name); out += struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", typ, off); off += (size + align - 1) // align * align
    out += b"\0" * ((-len(out)) % align)
    with open(dst, "wb") as o:
        o.write(out)
        for name, dims, typ, f, start, size in tensors:
            f.seek(start); left = size
            while left:
                chunk = f.read(min(left, 64 << 20)); o.write(chunk); left -= len(chunk)
            o.write(b"\0" * ((-size) % align))
    print(f"{dst}: {len(tensors)} tensors ({len(more)} added), {os.path.getsize(dst) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
