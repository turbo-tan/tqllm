#!/usr/bin/env python3
"""KL divergence between two TQFN_PF_LOGITS dumps: the prefill numerics gate.

Each dump holds, per prompt position, the top-32 next-token log-probs as (int32 id, float32 logp). The first file is the
reference (e.g. fp32 accumulate), the second the candidate. KL(ref || cand) is taken over the reference's top-32 plus one
bucket for the rest (the candidate's mass outside the reference's ids), as tqkld does. Prints mean / p99 / max KL and the
top-1 agreement.

usage: pf_logits_kl.py REF.bin CAND.bin
"""
import sys
import numpy as np

K = 32


def load(path):
    raw = np.fromfile(path, dtype=np.dtype([("id", "<i4"), ("lp", "<f4")]))
    if raw.size % K:
        raise SystemExit("%s: %d records, not a multiple of %d" % (path, raw.size, K))
    return raw.reshape(-1, K)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ref, cand = load(sys.argv[1]), load(sys.argv[2])
    if ref.shape != cand.shape:
        raise SystemExit("position counts differ: %d vs %d" % (ref.shape[0], cand.shape[0]))
    n = ref.shape[0]
    kl = np.zeros(n)
    top1 = 0
    for t in range(n):
        r_ids, r_lp = ref[t]["id"], ref[t]["lp"].astype(np.float64)
        c_map = dict(zip(cand[t]["id"].tolist(), cand[t]["lp"].astype(np.float64).tolist()))
        c_rest_mass = max(1e-12, 1.0 - float(np.exp(cand[t]["lp"].astype(np.float64)).sum()))
        # candidate log-prob of a reference id outside the candidate's top-32: its share of the rest bucket, spread evenly
        c_lp = np.array([c_map.get(int(i), np.log(c_rest_mass / (248320 - K))) for i in r_ids])
        p = np.exp(r_lp)
        kl_t = float(np.sum(p * (r_lp - c_lp)))
        r_rest = max(1e-12, 1.0 - p.sum())
        c_rest_for_ref = max(1e-12, 1.0 - float(np.exp(c_lp).sum()))
        kl_t += r_rest * (np.log(r_rest) - np.log(c_rest_for_ref))
        kl[t] = kl_t
        top1 += int(r_ids[0] == cand[t]["id"][0])
    print("positions %d  KL mean %.5f  p99 %.5f  max %.5f  top-1 agreement %.4f" %
          (n, kl.mean(), np.percentile(kl, 99), kl.max(), top1 / n))


if __name__ == "__main__":
    main()
