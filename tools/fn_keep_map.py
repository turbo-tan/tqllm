#!/usr/bin/env python3
"""Expert pruning map for TQFN_KEEP_MAP: from a routing profile (fn_expert_profile.py: int32 counts [layers][experts], the MTP row
ignored), keep the K most-routed experts of every trunk layer. Writes int32 [layers][experts]: compact index (ascending expert id)
or -1. usage: fn_keep_map.py PROFILE.bin LAYERS EXPERTS K OUT.map"""
import sys, numpy as np
prof, L, E, K, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
c = np.fromfile(prof, np.int32); rows = c.size // E; c = c.reshape(rows, E)[:L].astype(np.int64)
m = np.full((L, E), -1, np.int32)
for l in range(L):
    keep = np.sort(np.argsort(-c[l], kind="stable")[:K])
    m[l, keep] = np.arange(K, dtype=np.int32)
m.tofile(out)
cov = [c[l][m[l] >= 0].sum() / max(1, c[l].sum()) for l in range(L)]
print("kept %d / %d per layer; routed mass covered mean %.1f %%, worst %.1f %%" % (K, E, 100 * np.mean(cov), 100 * np.min(cov)))
