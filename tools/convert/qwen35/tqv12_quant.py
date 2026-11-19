#!/usr/bin/env python3
"""TQV-12 / TQT quantiser for Qwen3.5 / Qwen3.8 dense models (qwen35 GGUF): BF16 GGUF + a stock TQ file of the same
model as the layout template + calibration text -> a TQV-12 (or, with skeleton --format tqt, a TQT trellis-code) GGUF
that tqserve loads. Step by step: docs/guides/tqv12.md (TQT: docs/guides/tqt.md).

  prompts    calibration prompts from text files
  skeleton   the output file with the template's tensor list and metadata: every TQ4 tensor becomes TQV-12 (written
             later), the other tensors are copied from the template (checked against the BF16 source)
  capture    one tqserve run with TQLLM_DEBUG=calib_dump: the prefill inputs of some layers, then H = X^T X per input
  quantize   vector GPTQ of those layers' TQV-12 tensors into the output, with a decode self-check per tensor
  run        capture + quantize for all layers, a few layers per tqserve run (resumable)
  verify     checks the finished file against the template and the BF16 source

Needs numpy and gguf (pip install numpy gguf); capture / quantize / run also need PyTorch with CUDA and Triton."""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # tools/: gguf_truncate (GGUF header reader / writer)
import tqt  # noqa: E402
import tqv12  # noqa: E402
from gguf_truncate import T_STRING, read_header, w_string, w_value  # noqa: E402

TQ4 = 300
BF16, F32 = 30, 0
# input roles of TQLLM_DEBUG=calib_dump (src/models/qwen/qwen35/calib_dump.hpp) -> the tensors that read that input
ROLE_OF = {"attn_qkv": "mix", "attn_gate": "mix", "attn_q": "mix", "attn_k": "mix", "attn_v": "mix",
           "ffn_gate": "gu", "ffn_up": "gu", "ffn_down": "down", "ssm_out": "out", "attn_output": "out"}
ROLES = ("mix", "gu", "down", "out")
H_MAGIC = b"TQH1"
BASIS = {"plain": 0, "tq4rot128": 1}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def parse_layers(spec):
    """'0-3+8' -> [0, 1, 2, 3, 8] (the calib_dump syntax)"""
    out = []
    for part in spec.split("+"):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def layer_spec(layers):
    return "+".join(map(str, layers))


# ---------------- GGUF helpers ----------------
def block_layout(typ):
    """(weights, bytes) per block of a GGUF type, tqLLM's own ids included"""
    if typ in (TQ4, tqv12.GGUF_TYPE, tqt.GGUF_TYPE):
        return 128, 66
    if typ == 301:  # TQ3
        return 128, 50
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: PLC0415
    return GGML_QUANT_SIZES[GGMLQuantizationType(typ)]


def tensor_bytes(dims, typ):
    n = int(np.prod(dims))
    be, bb = block_layout(typ)
    assert n % be == 0
    return n // be * bb


def bf16_to_f32(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):  # round to nearest even
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Source:
    """the BF16 GGUF (any GGUF gguf-py reads): rows of a tensor as float32"""

    def __init__(self, path):
        from gguf import GGUFReader  # noqa: PLC0415
        self.t = {t.name: t for t in GGUFReader(path).tensors}

    def shape(self, name):  # GGUF order: dims[0] = row length
        return [int(d) for d in self.t[name].shape]

    def rows(self, name, rows=None):
        from gguf.quants import dequantize  # noqa: PLC0415
        t = self.t[name]
        cols = int(t.shape[0])
        a = np.asarray(t.data)
        a = a.reshape(int(np.prod(t.shape[1:])), -1)  # [rows][elements or bytes of a row]
        if rows is not None:
            a = a[rows]
        if t.tensor_type.name == "BF16":
            return bf16_to_f32(a.view(np.uint16)).reshape(-1, cols)
        if t.tensor_type.name in ("F32", "F16"):
            return a.astype(np.float32).reshape(-1, cols)
        return dequantize(a, t.tensor_type).astype(np.float32).reshape(-1, cols)


def dequant_rows(raw, typ, rows, cols):
    from gguf.constants import GGMLQuantizationType  # noqa: PLC0415
    from gguf.quants import dequantize  # noqa: PLC0415
    if typ == F32:
        return np.frombuffer(raw, np.float32).reshape(rows, cols)
    if typ == BF16:
        return bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(rows, cols)
    if typ == 1:  # F16
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(rows, cols)
    return dequantize(np.frombuffer(raw, np.uint8).reshape(rows, -1), GGMLQuantizationType(typ)).astype(np.float32)


def sample_rows(n, k=256):
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(int))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def load_plan(out):
    with open(str(out) + ".plan.json") as f:
        return json.load(f)


def save_plan(out, plan):
    p = str(out) + ".plan.json"
    with open(p + ".tmp", "w") as f:
        json.dump(plan, f, indent=0)
    os.replace(p + ".tmp", p)


def tqv_tensors(plan):
    """{layer: {role: [tensor names]}} of the TQV-12 tensors"""
    out = {}
    for p in plan["tensors"]:
        if p["kind"] != "tqv":
            continue
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", p["name"])
        if not m or m.group(2) not in ROLE_OF:
            raise SystemExit(f"{p['name']}: TQ4 in the template but not a known qwen35 projection")
        out.setdefault(int(m.group(1)), {}).setdefault(ROLE_OF[m.group(2)], []).append(p["name"])
    return out


# ---------------- formats ----------------
FORMATS = {  # skeleton --format: GGUF type, metadata key = tag (the engine refuses the type without it)
    "tqv12": dict(type=tqv12.GGUF_TYPE, key="tqllm.tqv.codebook", tag=tqv12.CODEBOOK_TAG, label="TQV-12"),
    "tqt": dict(type=tqt.GGUF_TYPE, key=tqt.CODE_KEY, tag=tqt.CODE_TAG, label="TQT"),
}


def plan_format(plan):
    return plan.get("format", "tqv12")


# ---------------- prompts ----------------
def cmd_prompts(a):
    texts = [Path(t).read_text(errors="replace") for t in a.text]
    per = [a.count // len(texts) + (i < a.count % len(texts)) for i in range(len(texts))]
    prompts = []
    for t, n in zip(texts, per):
        if len(t) < a.chars * n:
            raise SystemExit(f"a text has {len(t)} characters, {n} prompts of {a.chars} need {a.chars * n}")
        step = (len(t) - a.chars) // max(1, n - 1) if n > 1 else 0
        for i in range(n):
            s = i * step
            s = t.find(" ", s) + 1 if s else 0  # start and end on a word boundary
            e = t.rfind(" ", s, s + a.chars)
            prompts.append(t[s:e if e > s else s + a.chars])
    json.dump(prompts, open(a.out, "w"))
    log(f"wrote {len(prompts)} prompts ({sum(map(len, prompts))} characters) to {a.out}")


# ---------------- skeleton ----------------
def cmd_skeleton(a):
    kv, tensors, align = read_header(a.template)
    src = Source(a.bf16)
    fmt = FORMATS[a.format]
    kv = [list(e) for e in kv if e[0] not in (b"tqllm.tqv.codebook", tqt.CODE_KEY.encode())]
    name = (a.name or Path(a.out).stem).encode()
    if not any(e[0] == b"general.name" for e in kv):
        kv.append([b"general.name", T_STRING, name])
    for e in kv:
        if e[0] == b"general.name":
            e[1], e[2] = T_STRING, name
    kv.append([fmt["key"].encode(), T_STRING, fmt["tag"].encode()])
    plan, report, bad = [], {}, []
    for tname, dims, typ, _f, start, _size in tensors:
        n = tname.decode()
        dims = [int(d) for d in dims]
        cols, rows = dims[0], int(np.prod(dims[1:])) if len(dims) > 1 else 1
        if n not in src.t:
            raise SystemExit(f"{n}: in the template but not in {a.bf16} (is the template from the same model?)")
        if src.shape(n) != dims:
            raise SystemExit(f"{n}: template shape {dims} != source {src.shape(n)}")
        e = dict(name=n, dims=dims, rows=rows, cols=cols)
        if typ == TQ4:
            if cols % tqv12.BLOCK:
                raise SystemExit(f"{n}: row length {cols} is not a multiple of 128")
            e.update(type=fmt["type"], kind="tqv")
        elif typ == BF16 and len(dims) == 2:  # dense matrices beside TQ ones: stored in the input basis the template uses
            e.update(type=typ, kind="bf16", src=start)
        else:
            e.update(type=typ, kind="copy", src=start)
        e["nbytes"] = tensor_bytes(dims, e["type"])
        plan.append(e)
    hdr = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(plan), len(kv))
    for k, ty, v in kv:
        w_string(hdr, k)
        hdr += struct.pack("<I", ty)
        w_value(hdr, ty, v)
    off = 0
    for p in plan:
        w_string(hdr, p["name"].encode())
        hdr += struct.pack("<I", len(p["dims"])) + struct.pack("<%dQ" % len(p["dims"]), *p["dims"])
        hdr += struct.pack("<IQ", p["type"], off)
        p["rel"] = off
        off += (p["nbytes"] + align - 1) // align * align
    hdr += b"\0" * ((-len(hdr)) % align)
    for p in plan:
        p["offset"] = len(hdr) + p.pop("rel")
    total = len(hdr) + off
    ftpl = open(a.template, "rb")
    with open(a.out, "wb") as o:
        o.write(hdr)
        o.truncate(total)
        for p in plan:
            if p["kind"] == "tqv":
                continue
            ftpl.seek(p.pop("src"))
            raw = ftpl.read(p["nbytes"])
            rs = sample_rows(p["rows"])
            ref = src.rows(p["name"], rs)
            rb = p["nbytes"] // p["rows"]
            got = dequant_rows(b"".join(raw[r * rb:(r + 1) * rb] for r in rs), p["type"], len(rs), p["cols"])
            if p["kind"] == "bf16":
                full = src.rows(p["name"]).astype(np.float64)
                rot = f32_to_bf16(tqv12.rot_last_np(full).astype(np.float32))
                e_rot, e_plain = rel(bf16_to_f32(rot)[rs], got), rel(ref, got)
                p["rotated"] = bool(e_rot < e_plain)
                data = rot.tobytes() if p["rotated"] else f32_to_bf16(full.astype(np.float32)).tobytes()
                report[p["name"]] = dict(rotated=p["rotated"], vs_template=min(e_rot, e_plain))
                if min(e_rot, e_plain) > 0.02:
                    bad.append(p["name"])
            else:
                data = raw
                err = rel(got, ref)
                report[p["name"]] = dict(type=p["type"], vs_source=err)
                if err > (1e-6 if p["type"] == F32 else 0.1):
                    bad.append(p["name"])
            o.seek(p["offset"])
            o.write(data)
    nq = sum(p["kind"] == "tqv" for p in plan)
    nrot = sum(p.get("rotated", False) for p in plan)
    save_plan(a.out, dict(tensors=plan, total=total, template=str(a.template), source=str(a.bf16), report=report, done={},
                          format=a.format))
    log(f"wrote {a.out} ({total / 1e9:.2f} GB): {nq} {fmt['label']} tensors to quantise, {len(plan) - nq} written "
        f"({nrot} BF16 matrices rotated like the template)")
    if bad:
        raise SystemExit("tensors that do not match the BF16 source (template from another model?): " + ", ".join(bad))


# ---------------- capture ----------------
def build_h(x16, k, out, basis):
    """H = X^T X of the FP16 rows in x16 (FP32 accumulation) -> out: TQH1, int32 K, int64 tokens, int32 basis, K*K f32"""
    x = np.memmap(x16, dtype=np.float16, mode="r")
    x = x.reshape(-1, k)
    try:
        import torch  # noqa: PLC0415
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = False
        H = torch.zeros(k, k, device=dev)
        for r in range(0, x.shape[0], 8192):
            X = torch.from_numpy(np.array(x[r:r + 8192])).to(dev).float()
            H.addmm_(X.T, X)
        H = H.cpu().numpy()
    except ImportError:
        H = np.zeros((k, k), np.float32)
        for r in range(0, x.shape[0], 8192):
            X = np.asarray(x[r:r + 8192], np.float32)
            H += X.T @ X
    with open(out, "wb") as f:
        f.write(H_MAGIC + struct.pack("<iqi", k, x.shape[0], BASIS[basis]))
        f.write(np.ascontiguousarray(H, np.float32).tobytes())
    return x.shape[0]


def load_h(path):
    with open(path, "rb") as f:
        head = f.read(20)
        if head[:4] != H_MAGIC:
            raise SystemExit(f"{path}: not a calibration H file")
        k, tokens, basis = struct.unpack("<iqi", head[4:])
        H = np.fromfile(f, np.float32, k * k).reshape(k, k)
    return H, k, tokens, basis


def cmd_capture(a, layers=None):
    layers = layers or parse_layers(a.layers)
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(a.prompts))
    env = dict(os.environ, TQLLM_DEBUG=f"calib_dump:{layer_spec(layers)}:{d.resolve()}")
    srv_log = d / f"tqserve_{layers[0]}-{layers[-1]}.txt"
    cmd = [a.tqserve, "--model", a.model, "--host", "127.0.0.1", "--port", str(a.port), "--context", str(a.context),
           "--np", "1", "--max-tokens", "1", "--reasoning", "off", "--fit", "off"]
    t0 = time.time()
    with open(srv_log, "w") as lf:
        srv = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        while "listening" not in srv_log.read_text(errors="replace"):
            if srv.poll() is not None:
                raise SystemExit(f"tqserve exited ({srv.returncode}); see {srv_log}")
            if time.time() - t0 > a.start_timeout:
                raise SystemExit(f"tqserve did not start in {a.start_timeout} s; see {srv_log}")
            time.sleep(1)
        log(f"capture layers {layers[0]}-{layers[-1]}: tqserve up in {time.time() - t0:.0f} s")
        ntok = 0
        for i, p in enumerate(prompts):
            body = {"messages": [{"role": "user", "content": p}], "max_tokens": 1, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            ntok += r.get("usage", {}).get("prompt_tokens") or r.get("timings", {}).get("prompt_n", 0)
        log(f"  {len(prompts)} prompts, {ntok} prompt tokens in {time.time() - t0:.0f} s")
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
    counts = set()
    for L in layers:
        for role in ROLES:
            base = d / f"L{L}_{role}"
            if not Path(str(base) + ".json").exists():
                raise SystemExit(f"no capture for layer {L} input {role}: is {a.model} a qwen35 dense model? see {srv_log}")
            meta = json.load(open(str(base) + ".json"))
            if meta["basis"] not in BASIS:
                raise SystemExit(f"{base}: input basis {meta['basis']} (capture with a TQ4 or K-quant / BF16 model)")
            counts.add(meta["tokens"])
            n = build_h(str(base) + ".x16", meta["k"], str(base) + ".h32", meta["basis"])
            assert n == meta["tokens"], (base, n, meta)
            if not a.keep_x16:
                os.remove(str(base) + ".x16")
            os.remove(str(base) + ".json")
    if len(counts) != 1:
        log(f"  WARNING: token counts differ between inputs: {sorted(counts)}")
    log(f"  H written for layers {layers[0]}-{layers[-1]} ({counts.pop()} tokens each) at {time.time() - t0:.0f} s")
    srv_log.unlink()


# ---------------- quantize ----------------
def role_damp(a, role):
    """GPTQ dampening for an input role: --damp-role mix:0.1,gu:0.3,... overrides --damp"""
    spec = getattr(a, "damp_role", None)
    if spec:
        m = dict(kv.split(":") for kv in spec.split(","))
        bad = set(m) - set(ROLES)
        if bad:
            raise SystemExit(f"--damp-role: unknown role(s) {sorted(bad)} (roles: {', '.join(ROLES)})")
        if role in m:
            return float(m[role])
    return a.damp


def encoder_args(p):
    p.add_argument("--damp-role", help="per input role GPTQ dampening, e.g. mix:0.1,gu:0.3,down:0.3,out:0.1 "
                   "(held-out-tuned; the plain --damp applies to roles not listed)")
    p.add_argument("--damp-form", choices=tqv12.DAMP_FORMS, default="mean",
                   help="unit of the dampening: mean (GPTQ) or median of diag(H) (robust to massive activations)")
    p.add_argument("--refit", action="store_true", help="closed-form block-scale refit inside GPTQ")
    p.add_argument("--order", action="store_true", help="block act-order: 128-blocks in descending tr(H_bb)")
    p.add_argument("--icm", type=int, default=0, help="local-search sweeps after GPTQ (0 = off)")
    p.add_argument("--icm-damp", type=float, default=None, help="dampening of H inside ICM (default: the role's)")
    p.add_argument("--tqt-scales", type=int, default=3, help="TQT: scale candidates per block (1..5; each is one more "
                   "Viterbi pass: RTX 3090 Qwen3.8-27B layer 87 / 173 / 259 s at 1 / 2 / 3)")


def cmd_quantize(a, layers=None):
    import torch  # noqa: PLC0415
    layers = layers or parse_layers(a.layers)
    plan = load_plan(a.out)
    by_name = {p["name"]: p for p in plan["tensors"]}
    groups = tqv_tensors(plan)
    src = Source(a.bf16)
    dev = getattr(a, "device", "cuda")
    use_tqt = plan_format(plan) == "tqt"
    cb = None if use_tqt else tqv12.Codebook(dev)
    fo = open(a.out, "r+b")
    for L in layers:
        t_l = time.time()
        for role, names in groups.get(L, {}).items():
            if all(n in plan["done"] for n in names):
                continue
            hp = Path(a.hdir) / f"L{L}_{role}.h32"
            if not hp.exists():
                raise SystemExit(f"{hp} missing: run capture for layer {L} first")
            Hn, K, ntok, basis = load_h(hp)
            H = torch.from_numpy(Hn).to(dev)
            del Hn
            H = (H + H.T) / 2
            Hr = tqv12.rotate_h(H) if basis == BASIS["plain"] else H
            del H
            Ws = []
            for n in names:
                p = by_name[n]
                if p["cols"] != K:
                    raise SystemExit(f"{n}: row length {p['cols']} but its calibration input has K {K}")
                Ws.append(torch.from_numpy(src.rows(n)).to(dev))
            Wr = tqv12.rot_last(torch.cat(Ws, 0))
            del Ws
            t0 = time.time()
            damp = role_damp(a, role)
            if use_tqt:
                s, st, Q = tqt.gptq(Wr, Hr, damp=damp, gammas=tqt.gammas(getattr(a, "tqt_scales", 3)),
                                    damp_form=getattr(a, "damp_form", "mean"), order=getattr(a, "order", False),
                                    refit=getattr(a, "refit", False))
                s_np, st_np = s.cpu().numpy(), st.cpu().numpy()
            else:
                s, idx, neg, Q = tqv12.gptq(Wr, Hr, cb, damp=damp, refit=getattr(a, "refit", False),
                                            order=getattr(a, "order", False), damp_form=getattr(a, "damp_form", "mean"))
                if getattr(a, "icm", 0):
                    Q = tqv12.icm(Wr, Hr, cb, s, idx, neg, sweeps=a.icm, damp=a.icm_damp if a.icm_damp is not None else damp,
                                  damp_form=getattr(a, "damp_form", "mean"))
                s_np, idx_np, neg_np = s.cpu().numpy(), idx.cpu().numpy(), neg.cpu().numpy()
            if Q.is_cuda:
                torch.cuda.synchronize()
            tg = time.time() - t0
            r0 = 0
            for n in names:
                p = by_name[n]
                r1 = r0 + p["rows"]
                err = tqv12.rel_err(Q[r0:r1], Wr[r0:r1], Hr)
                wmse = float(((Q[r0:r1] - Wr[r0:r1]) ** 2).sum() / (Wr[r0:r1] ** 2).sum())
                raw = tqt.pack(s_np[r0:r1], st_np[r0:r1]) if use_tqt else tqv12.pack(s_np[r0:r1], idx_np[r0:r1], neg_np[r0:r1])
                assert raw.nbytes == p["nbytes"], (n, raw.nbytes, p["nbytes"])
                fo.seek(p["offset"])
                fo.write(raw.tobytes())
                fo.flush()
                # self-check: read sample rows back, decode them independently, compare with GPTQ's decoded weights
                rs = sample_rows(p["rows"], 64)
                rb = p["cols"] // tqv12.BLOCK * tqv12.BLOCK_BYTES
                fo.seek(p["offset"])
                back = np.frombuffer(fo.read(p["nbytes"]), np.uint8).reshape(p["rows"], rb)
                if use_tqt:
                    dec = tqt.decode(back[rs].tobytes(), len(rs), p["cols"])
                else:
                    dec = tqv12.decode(back[rs].tobytes(), len(rs), p["cols"], cb.table)
                exact = bool(np.array_equal(dec, Q[r0:r1][torch.from_numpy(rs).to(dev)].cpu().numpy()))
                if not exact:
                    raise SystemExit(f"{n}: the written blocks do not decode to the quantised weights")
                plan["done"][n] = dict(rel_out_err=err, rel_w_mse=wmse, h_tokens=ntok, gptq_s=round(tg, 1),
                                       decode_exact=exact, damp=damp, refit=bool(getattr(a, "refit", False)),
                                       order=bool(getattr(a, "order", False)), icm=getattr(a, "icm", 0),
                                       damp_form=getattr(a, "damp_form", "mean"))
                log(f"L{L} {n} [{p['rows']}x{p['cols']}] output error {err:.5f}, weight MSE {wmse:.5f}, GPTQ {tg:.0f} s")
                r0 = r1
            del s, Q, Wr, Hr
            if dev != "cpu":
                torch.cuda.empty_cache()
            save_plan(a.out, plan)
        if not a.keep_h:
            for role in ROLES:
                hp = Path(a.hdir) / f"L{L}_{role}.h32"
                if hp.exists():
                    hp.unlink()
        log(f"layer {L} done in {time.time() - t_l:.0f} s")
    fo.close()


def cmd_run(a):
    plan = load_plan(a.out)
    groups = tqv_tensors(plan)
    want = parse_layers(a.layers) if a.layers else sorted(groups)
    todo = [L for L in want if not all(n in plan["done"] for ns in groups.get(L, {}).values() for n in ns)]
    log(f"{len(want) - len(todo)} of {len(want)} layers already quantised; {len(todo)} to go, {a.per_run} per tqserve run")
    for i in range(0, len(todo), a.per_run):
        chunk = todo[i:i + a.per_run]
        if not all((Path(a.dir) / f"L{L}_{r}.h32").exists() for L in chunk for r in ROLES):
            cmd_capture(a, chunk)
        cmd_quantize(argparse.Namespace(out=a.out, bf16=a.bf16, hdir=a.dir, keep_h=False, damp=a.damp,
                                        damp_role=a.damp_role, damp_form=a.damp_form, refit=a.refit, order=a.order, icm=a.icm,
                                        icm_damp=a.icm_damp, tqt_scales=a.tqt_scales), chunk)
    log("all layers quantised; next: verify")


# ---------------- verify ----------------
def cmd_verify(a):
    ok = True

    def fail(*m):
        nonlocal ok
        ok = False
        print("FAIL", *m, flush=True)

    plan = load_plan(a.out)
    kv_o, t_o, _ = read_header(a.out)
    kv_t, t_t, _ = read_header(a.template)
    src = Source(a.bf16)
    fmt = FORMATS[plan_format(plan)]
    use_tqt = plan_format(plan) == "tqt"
    table = None if use_tqt else tqv12.load_codebook()
    if len(t_o) != len(t_t):
        fail(f"{len(t_o)} tensors, template {len(t_t)}")
    meta = {k.decode(): v for k, _, v in kv_o}
    if meta.get(fmt["key"]) != fmt["tag"].encode():
        fail(fmt["key"], "is not", fmt["tag"])
    tmeta = {k.decode(): v for k, _, v in kv_t}
    for k in tmeta:
        if k not in meta:
            fail("metadata key missing:", k)
        elif k != "general.name" and meta[k] != tmeta[k]:
            fail("metadata differs:", k)
    by_name = {p["name"]: p for p in plan["tensors"]}
    t_by_name = {n.decode(): (d, ty, f, s) for n, d, ty, f, s, _ in t_t}
    fo = open(a.out, "rb")
    errs, ncopy, nrot = [], 0, 0
    for name, dims, typ, _f, start, _size in t_o:
        n = name.decode()
        if n not in t_by_name:
            fail("not in the template:", n)
            continue
        tdims, ttyp, tf, tstart = t_by_name[n]
        if list(dims) != list(tdims):
            fail("shape", n)
        if typ != (fmt["type"] if ttyp == TQ4 else ttyp):
            fail("type", n, typ, ttyp)
            continue
        p = by_name[n]
        fo.seek(start)
        raw = fo.read(p["nbytes"])
        if typ == fmt["type"]:
            rec = plan["done"].get(n)
            if not rec or not rec.get("decode_exact"):
                fail("not quantised:", n)
                continue
            b = np.frombuffer(raw, np.uint8).reshape(p["rows"], p["cols"] // 128, 66)
            sc = b[:, :, :2].copy().view("<f2")[..., 0].astype(np.float32)
            if (sc == 0).any() or not np.isfinite(sc).all():
                fail("zero or non-finite scales:", n, int((sc == 0).sum()))
            rs = sample_rows(p["rows"], 16)
            dec = tqt.decode(b[rs].tobytes(), len(rs), p["cols"]) if use_tqt else tqv12.decode(b[rs].tobytes(), len(rs), p["cols"], table)
            back = (dec.reshape(len(rs), -1, 128) @ tqv12.rot_matrix_np()).reshape(len(rs), p["cols"])  # un-rotate
            e = rel(back, src.rows(n, rs))
            errs.append(e)
            if e > 0.15:
                fail(f"{n}: decoded weights {e:.3f} from the source")
        elif p["kind"] == "bf16":
            full = src.rows(n).astype(np.float64)
            ref = tqv12.rot_last_np(full) if p.get("rotated") else full
            if rel(bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(ref.shape), ref) > 0.01:
                fail("BF16 matrix differs from the source:", n)
            nrot += 1
        else:
            tf.seek(tstart)
            if tf.read(p["nbytes"]) != raw:
                fail("bytes differ from the template:", n)
            ncopy += 1
    nq = len(errs)
    print(f"tensors: {len(t_o)} ({fmt['label']} {nq}, byte-identical to the template {ncopy}, BF16 from the source {nrot})")
    if errs:
        print(f"{fmt['label']} decoded and un-rotated vs the BF16 source, relative RMS: mean {np.mean(errs):.4f} "
              f"min {np.min(errs):.4f} max {np.max(errs):.4f}")
        out_err = [r["rel_out_err"] for r in plan["done"].values()]
        print(f"GPTQ relative output error on the calibration inputs: mean {np.mean(out_err):.5f} max {np.max(out_err):.5f}")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="calibration prompts from text files")
    p.add_argument("--text", action="append", required=True, help="text file (repeat: the prompts are split evenly)")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--chars", type=int, default=6500, help="characters per prompt (~1,600-2,200 tokens)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("skeleton", help="output file with the template's layout, TQ4 -> TQV-12 placeholders")
    p.add_argument("--bf16", required=True, help="BF16 GGUF of the model (llama.cpp convert_hf_to_gguf.py)")
    p.add_argument("--template", required=True, help="stock TQ file of the same model (Qwen3.8-27B-TQ_L.gguf)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", help="general.name (default: the output file name)")
    p.add_argument("--format", choices=sorted(FORMATS), default="tqv12", help="TQ4 tensors become this format")

    def capture_args(p):
        p.add_argument("--tqserve", required=True, help="tqserve binary of this tree")
        p.add_argument("--model", required=True, help="model that runs the calibration prompts (the template)")
        p.add_argument("--prompts", required=True)
        p.add_argument("--dir", required=True, help="dump directory (about 8 GB per layer at 64 prompts)")
        p.add_argument("--port", type=int, default=8097)
        p.add_argument("--context", type=int, default=4096)
        p.add_argument("--start-timeout", type=int, default=600)
        p.add_argument("--keep-x16", action="store_true", help="keep the raw FP16 inputs")
    p = sub.add_parser("capture", help="calibration inputs of some layers -> H files")
    capture_args(p)
    p.add_argument("--layers", required=True, help="e.g. 0-7")
    p = sub.add_parser("quantize", help="vector GPTQ of some layers into the output")
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--hdir", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--keep-h", action="store_true")
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--device", default="cuda", help="torch device (cpu: slow, for tests; not for TQT)")
    encoder_args(p)
    p = sub.add_parser("run", help="capture + quantize every layer, PER_RUN layers per tqserve run (resumable)")
    capture_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--layers", help="default: every layer with TQV-12 tensors")
    p.add_argument("--per-run", type=int, default=4)
    p.add_argument("--damp", type=float, default=0.01)
    encoder_args(p)
    p = sub.add_parser("verify", help="check the finished file")
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--template", required=True)
    a = ap.parse_args(argv)
    return {"prompts": cmd_prompts, "skeleton": cmd_skeleton, "capture": cmd_capture, "quantize": cmd_quantize,
            "run": cmd_run, "verify": cmd_verify}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
