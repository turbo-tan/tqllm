#!/usr/bin/env python3
"""Q4 gate: compare TQLLM (tqgolden) against tests/golden fixtures. Exit 1 on any gate failure.

Gates (per fixture):
  - token ids identical to HF;
  - last-token logits rel-RMS vs HF FP32 <= max(1e-2, HF BF16's own rel-RMS vs HF FP32): a BF16-activation engine
    must be no less accurate than the reference framework at the same dtype (HF BF16 sits at 1.7-3.2e-2, so the
    ROADMAP's initial 1e-2 ceiling is below the dtype floor — see docs/EXECUTION_QUEUE.md, Q4 log);
  - 32 greedy tokens identical to HF FP32 (the numerical truth), except a characterised near-tie: the first
    divergence is accepted only if the chosen token is FP32's runner-up and the FP32 top-2 margin at that step is
    <= NEAR_TIE logits (tests/golden/greedy_margins_fp32.json). Tokens after an accepted tie are not compared.
Per-layer hidden rel-RMS (vs FP32 and vs HF BF16) is reported for diagnosis.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

LOGIT_GATE = 1e-2
NEAR_TIE = 0.1


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqgolden", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--golden", default=str(Path(__file__).parent / "golden"))
    ap.add_argument("--work", default="/tmp/tqgolden")
    a = ap.parse_args()
    g = Path(a.golden)
    man = json.loads((g / "manifest.json").read_text())
    margins = json.loads((g / "greedy_margins_fp32.json").read_text())
    Path(a.work).mkdir(parents=True, exist_ok=True)
    ok = True
    for f in man["fixtures"]:
        pre = f"{a.work}/f{f['id']}"
        src = ["--text", f["text"]] if "text" in f else ["--prompt", f["prompt"]]
        subprocess.run([a.tqgolden, "--model-dir", a.model_dir, *src, "--out", pre,
                        "--greedy", str(man["greedy_tokens"])], check=True)
        ids = [int(x) for x in open(pre + ".ids").read().split()]
        greedy = [int(x) for x in open(pre + ".greedy").read().split()]
        ref32 = np.load(g / f["float32"]["file"])
        ref16 = np.load(g / f["bfloat16"]["file"])
        L, H = ref32["hidden"].shape
        hid = np.fromfile(pre + ".hidden.f32", dtype=np.float32).reshape(L, H)
        logits = np.fromfile(pre + ".logits.f32", dtype=np.float32)
        ids_ok = ids == f["input_ids"]
        lr32 = rel_rms(logits, ref32["logits"])
        lr16 = rel_rms(logits, ref16["logits"])
        hid32 = [rel_rms(hid[l], ref32["hidden"][l]) for l in range(L)]
        hid16 = [rel_rms(hid[l], ref16["hidden"][l]) for l in range(L)]
        floor = rel_rms(ref16["logits"], ref32["logits"])
        want = f["float32"]["greedy"]
        n = min(len(want), len(greedy))
        first_diff = next((i for i in range(n) if want[i] != greedy[i]), None if len(want) == len(greedy) else n)
        g_ok = first_diff is None
        # TQLLM stops at the tokenizer EOS; HF generate may continue past it (e.g. <|im_end|> -> <|endoftext|>)
        if not g_ok and len(greedy) < man["greedy_tokens"] and greedy == want[: len(greedy)]:
            g_ok, first_diff = True, None
        tie = ""
        if first_diff is not None and first_diff < len(greedy) and first_diff < len(margins[str(f["id"])]):
            mrow = margins[str(f["id"])][first_diff]
            if greedy[first_diff] == mrow["second"] and mrow["margin"] <= NEAR_TIE:
                g_ok, tie = True, f" near-tie at {first_diff} (FP32 margin {mrow['margin']:.3f}, picked runner-up)"
        l_ok = lr32 <= max(LOGIT_GATE, floor)
        print(f"fixture {f['id']}: ids {'OK' if ids_ok else 'MISMATCH'} | logits relRMS vs fp32 {lr32:.2e} (HF-bf16 floor {floor:.2e}) "
              f"{'OK' if l_ok else 'FAIL'} | argmax {int(logits.argmax())} vs {f['float32']['argmax']} | "
              f"greedy vs HF-fp32 {('OK' + tie) if g_ok else f'DIVERGES at {first_diff}'} (vs HF-bf16 "
              f"{'same' if greedy == f['bfloat16']['greedy'] else 'differs'})")
        print(f"    hidden relRMS vs fp32 max {max(hid32):.2e} (layer {int(np.argmax(hid32))}), vs bf16 max {max(hid16):.2e}")
        ok &= ids_ok and l_ok and g_ok
    print("Q4 GOLDEN GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
