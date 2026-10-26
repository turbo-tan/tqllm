#!/usr/bin/env python3
"""Compact expert pack for TQFN_KEEP_MAP: from a full TQFNEXP1 / TQFNEXPT pack and a keep map, write a pack with K experts per layer
(kept experts in compact-index order). OUT '-' streams to stdout. usage: fn_pack_slice.py FULL.tqfnexp KEEP.map OUT"""
import sys, struct, numpy as np
full, mapf, out = sys.argv[1], sys.argv[2], sys.argv[3]
f = open(full, "rb"); magic = f.read(8); L, E = struct.unpack("<ii", f.read(8)); sizes = [struct.unpack("<6Q", f.read(48)) for _ in range(L)]
m = np.fromfile(mapf, np.int32).reshape(L, E); K = int(m.max()) + 1
o = sys.stdout.buffer if out == "-" else open(out, "wb")
o.write(magic); o.write(struct.pack("<ii", L, K))
for z in sizes: o.write(struct.pack("<6Q", *z))
off = 16 + 48 * L; done = 0
for l in range(L):
    block = sum(sizes[l]); kept = np.where(m[l] >= 0)[0]; order = kept[np.argsort(m[l][kept])]
    assert len(order) == K, "layer %d keeps %d, not %d" % (l, len(order), K)
    for e in order:
        f.seek(off + int(e) * block); o.write(f.read(block))
    off += block * E; done += K * block
    print("layer %d: %d experts, %.0f MB" % (l, K, K * block / 1e6), file=sys.stderr)
o.flush(); print("compact pack %.2f GB" % (done / 1e9), file=sys.stderr)
