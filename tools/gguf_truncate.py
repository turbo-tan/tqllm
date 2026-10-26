#!/usr/bin/env python3
"""gguf_truncate.py IN OUT N: copy a GGUF keeping only the first N transformer blocks (blk.0 .. blk.N-1) and every
non-block tensor (embeddings, output norm, head). <arch>.block_count becomes N and every per-layer array metadata value
whose length equals the old block count is cut to N; <arch>.nextn_predict_layers becomes 0 (the NextN block is dropped).
Tensor data is copied byte for byte. IN may be any shard of a split GGUF (<prefix>-NNNNN-of-NNNNN.gguf): all shards are
read and OUT is one unsplit file.

Used to bring up an architecture against a reference engine on a model too large for one machine: both engines run
the same N-layer network, so their logits must agree."""
import os, re, struct, sys

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, \
    T_FLOAT64 = range(13)
SCALAR = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
          T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}


class Reader:
    def __init__(self, f): self.f = f
    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]
    def string(self): return self.f.read(self.u64())
    def value(self, t):
        if t == T_STRING: return self.string()
        if t == T_ARRAY:
            et, n = self.u32(), self.u64()
            return (et, [self.value(et) for _ in range(n)])
        fmt = SCALAR[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]


def w_string(out, b): out += struct.pack("<Q", len(b)) + b
def w_value(out, t, v):
    if t == T_STRING: w_string(out, v)
    elif t == T_ARRAY:
        et, items = v
        out += struct.pack("<IQ", et, len(items))
        for x in items: w_value(out, et, x)
    else: out += struct.pack(SCALAR[t], v)


def read_header(path):
    f = open(path, "rb")
    r = Reader(f)
    magic, version = f.read(4), r.u32()
    assert magic == b"GGUF" and version == 3, "GGUF v3 only"
    n_tensors, n_kv = r.u64(), r.u64()
    kv = []
    for _ in range(n_kv):
        k = r.string(); t = r.u32(); kv.append([k, t, r.value(t)])
    tensors = []
    for _ in range(n_tensors):
        name = r.string(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; typ = r.u32(); off = r.u64()
        tensors.append([name, dims, typ, off])
    align = next((v for k, t, v in kv if k == b"general.alignment"), 32)
    data0 = (f.tell() + align - 1) // align * align
    end = os.path.getsize(path)
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][3])
    sized = []
    for j, i in enumerate(order):
        nxt = tensors[order[j + 1]][3] if j + 1 < len(order) else end - data0
        name, dims, typ, off = tensors[i]
        sized.append((name, dims, typ, f, data0 + off, nxt - off))
    return kv, sized, align


def shard_paths(path, kv):
    count = next((v for k, t, v in kv if k == b"split.count"), 1)
    if count <= 1: return [path]
    m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", path)
    assert m and int(m.group(3)) == count, "split GGUF name must end -NNNNN-of-NNNNN.gguf"
    w = len(m.group(2))
    return ["%s-%0*d-of-%s.gguf" % (m.group(1), w, i + 1, m.group(3)) for i in range(count)]


def main():
    # gguf_truncate.py IN OUT N [--arch NAME]: --arch renames general.architecture and every <arch>.* key
    args = sys.argv[1:]
    new_arch = None
    if "--arch" in args:
        i = args.index("--arch"); new_arch = args[i + 1]; del args[i:i + 2]
    src, dst, keep = args[0], args[1], int(args[2])
    kv0, _, _ = read_header(src)
    paths = shard_paths(src, kv0)
    kv, tensors, align = read_header(paths[0])
    for p in paths[1:]:
        _, more, _ = read_header(p); tensors += more
    names = [t[0] for t in tensors]
    assert len(set(names)) == len(names), "a tensor is in two shards"
    kv = [e for e in kv if not e[0].startswith(b"split.")]
    arch = next(v for k, t, v in kv if k == b"general.architecture").decode()
    old_n = next(v for k, t, v in kv if k == (arch + ".block_count").encode())
    nextn = next((v for k, t, v in kv if k == (arch + ".nextn_predict_layers").encode()), 0)
    assert 0 < keep < old_n - nextn
    for e in kv:
        k, t, v = e
        if k == (arch + ".block_count").encode(): e[2] = keep
        elif k == (arch + ".nextn_predict_layers").encode(): e[2] = 0
        elif t == T_ARRAY and k.startswith(arch.encode()) and len(v[1]) in (old_n, old_n - nextn): e[2] = (v[0], v[1][:keep])
    if new_arch:
        old = (arch + ".").encode()
        for e in kv:
            if e[0] == b"general.architecture": e[2] = new_arch.encode()
            elif e[0].startswith(old): e[0] = (new_arch + ".").encode() + e[0][len(old):]
    blk = re.compile(rb"^blk\.(\d+)\.")
    kept = [t for t in tensors if not (m := blk.match(t[0])) or int(m.group(1)) < keep]
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(kept), len(kv))
    for k, t, v in kv:
        w_string(out, k); out += struct.pack("<I", t); w_value(out, t, v)
    offs, cur = [], 0
    for t in kept:
        offs.append(cur); cur += (t[5] + align - 1) // align * align
    for t, off in zip(kept, offs):
        name, dims, typ = t[0], t[1], t[2]
        w_string(out, name); out += struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", typ, off)
    out += b"\0" * ((-len(out)) % align)
    with open(dst, "wb") as o:
        o.write(out)
        for name, dims, typ, f, start, size in kept:
            f.seek(start); left = size
            while left:
                chunk = f.read(min(left, 64 << 20)); o.write(chunk); left -= len(chunk)
            o.write(b"\0" * ((-size) % align))
    print(f"{dst}: {keep} of {old_n} blocks from {len(paths)} file(s), {len(kept)} of {len(tensors)} tensors, "
          f"{os.path.getsize(dst) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
