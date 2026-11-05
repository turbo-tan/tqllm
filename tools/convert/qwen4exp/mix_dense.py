#!/usr/bin/env python3
"""Flash-Next mixed trunk from two GGUFs of the same model: every tensor from REF_DIR's trunk (e.g. the GSQ-RCO IQ3_S
experts) except the quantised dense matrices matching PATTERN, which come byte for byte from DONOR (e.g. a Q8_0
requantised to IQ4_XS with an importance matrix). Shapes must match; the donor's type is kept.

usage: mix_dense.py REF_DIR DONOR_GGUF[,DONOR_GGUF...] OUT_GGUF [PATTERN]   (a later donor wins on a shared name)
       mix_dense.py --extract SRC_DIR OUT_GGUF [PATTERN]   the matching tensors of a split GGUF in one file (metadata of
       shard 1), e.g. a Q8_0's dense matrices for llama-quantize without reading the experts"""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_to_gguf import gguf, ref_tensors, ref_copy, write_gguf  # noqa: E402

DENSE = (r"^(output\.weight|blk\.\d+\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|"
         r"ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight)$")
HC = r"^(output_hc_(down|up)|blk\.\d+\.hc_(attn|ffn)_(down|up))\.weight$"  # the hyper-connection mixers


def main(ref_dir, donor, out, pattern=DENSE):
    refs = ref_tensors(ref_dir)
    r0 = next(r for f, _, r in refs.values() if "00001" in f)
    d = {t.name: t for path in donor.split(",") for t in gguf.GGUFReader(path).tensors}
    items, taken = [], 0
    for name, (f, t, r) in refs.items():
        if "mtp" in f:
            continue
        src = t
        if re.search(pattern, name):
            src = d[name]
            if [int(x) for x in src.shape] != [int(x) for x in t.shape]:
                raise ValueError(f"{name}: donor shape {list(src.shape)} != {list(t.shape)}")
            taken += 1
        items.append((name, [int(x) for x in t.shape], src.tensor_type, int(src.n_bytes), (lambda s=src: ref_copy(s))))
    print(f"{taken} dense tensors from {donor}", flush=True)
    write_gguf(out, r0, items)


def extract(src_dir, out, pattern=DENSE):
    refs = ref_tensors(src_dir)
    r0 = gguf.GGUFReader(os.path.join(src_dir, next(f for f in sorted(os.listdir(src_dir)) if "00001" in f)))  # may hold no tensors
    items = [(n, [int(x) for x in t.shape], t.tensor_type, int(t.n_bytes), (lambda t=t: ref_copy(t)))
             for n, (f, t, r) in refs.items() if re.search(pattern, n)]
    print(f"{len(items)} tensors", flush=True)
    write_gguf(out, r0, items)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        extract(*sys.argv[2:]); sys.exit(0)
    if len(sys.argv) > 4 and sys.argv[4] in ("dense+hc", "hc"):
        sys.argv[4] = {"hc": HC, "dense+hc": f"({DENSE})|({HC})"}[sys.argv[4]]
    if len(sys.argv) > 3 and sys.argv[1] == "--extract-hc":
        extract(sys.argv[2], sys.argv[3], HC); sys.exit(0)
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    main(*sys.argv[1:])
