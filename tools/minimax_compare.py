#!/usr/bin/env python3
"""Engine residual stream (TQLLM_MINIMAX_DUMP) against the oracle's h1..hN (tools/minimax_ref.py --stream npz).

  minimax_compare.py engine.dump ref.npz [--head CKPT]   per layer: relative error and cosine per position;
  with --head, the oracle's logits for the cut (final norm + lm_head on h_N) to compare with the engine's top tokens.
"""
import argparse
import sys

import numpy as np


def read_dump(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                break
            layer, pos, n = np.frombuffer(head, np.int32)
            out.setdefault(int(layer), {})
            for t in range(int(n)):
                out[int(layer)][int(pos) + t] = None
            data = np.frombuffer(f.read(4 * int(n) * 6144), np.float32).reshape(int(n), 6144)
            for t in range(int(n)):
                out[int(layer)][int(pos) + t] = data[t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("ref")
    a = ap.parse_args()
    eng = read_dump(a.dump)
    ref = np.load(a.ref)
    for layer in sorted(eng):
        r = ref[f"h{layer + 1}"]
        rows = []
        for p in sorted(eng[layer]):
            if p >= r.shape[0]:
                continue
            e, o = eng[layer][p].astype(np.float64), r[p].astype(np.float64)
            rel = np.linalg.norm(e - o) / np.linalg.norm(o)
            cos = e @ o / (np.linalg.norm(e) * np.linalg.norm(o))
            rows.append((p, rel, cos))
        worst = max(rows, key=lambda x: x[1])
        print(f"layer {layer:2d}: rel err median {np.median([x[1] for x in rows]):.4f} worst {worst[1]:.4f} (pos {worst[0]}), "
              f"cos min {min(x[2] for x in rows):.5f}")


if __name__ == "__main__":
    sys.exit(main())
