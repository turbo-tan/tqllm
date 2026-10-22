#!/usr/bin/env python3
"""Flash-Next routing profile for TQFN_EXPERT_PROFILE: counts every routed (layer, expert) in TQFN_ROUTE_DUMP files
(records: int32 layer, int32 n, n * topk int32 ids) and writes int32 counts [layers][experts].
usage: fn_expert_profile.py OUT.bin LAYERS EXPERTS TOPK DUMP [DUMP ...]"""
import struct, sys
import numpy as np
out, L, E, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cnt = np.zeros((L, E), np.int32)
for p in sys.argv[5:]:
    d = open(p, 'rb').read(); i = 0
    while i + 8 <= len(d):
        layer, n = struct.unpack_from('<ii', d, i); i += 8
        ids = np.frombuffer(d, np.int32, n * K, i); i += 4 * n * K
        if 0 <= layer < L: np.add.at(cnt[layer], ids[(ids >= 0) & (ids < E)], 1)
cnt.tofile(out)
print("profile %s: %d routed assignments, %d layers x %d experts" % (out, int(cnt.sum()), L, E))
