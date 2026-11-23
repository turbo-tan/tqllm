# Draft vocabulary (--draft-vocab) from token frequencies over local text: the N most frequent token ids of a corpus,
# filled up to N from a fallback list. Format: "llama-mtp-vocab-v1 <vocab> <N>" and one id per line.
# usage: build_draft_vocab.py TQTOKENIZE MODEL.gguf VOCAB_SIZE N OUT [--fallback FILE] [--max-bytes B] GLOB...
# Qwen3.6-35B-A3B (11-17): python stdlib, /usr/include, llama.cpp sources and docs, project notes (67 MB, 51K
# distinct ids), N 65536 with the Qwen3.8 atx_65536 list as fallback: DFlash2 189.8 -> 193.4 tok/s on GB10 code prompts.
import argparse, collections, glob, os, random, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer"); ap.add_argument("model"); ap.add_argument("vocab", type=int); ap.add_argument("n", type=int)
ap.add_argument("out"); ap.add_argument("globs", nargs="+")
ap.add_argument("--fallback"); ap.add_argument("--max-bytes", type=int, default=300_000_000)
a = ap.parse_args()
files = [f for g in a.globs for f in glob.glob(os.path.expanduser(g), recursive=True)
         if os.path.isfile(f) and os.path.getsize(f) < 400_000]
random.seed(0); random.shuffle(files)
cnt, buf, blen, used = collections.Counter(), [], 0, 0
chunk = a.out + ".chunk.txt"
def flush():
    global buf, blen
    if not buf: return
    open(chunk, "w", errors="ignore").write("\n".join(buf))
    o = subprocess.run([a.tokenizer, "--tokenizer", a.model, "--text-file", chunk], capture_output=True, text=True, check=True).stdout
    cnt.update(int(x) for x in o.split()); buf, blen = [], 0
for f in files:
    try: t = open(f, errors="ignore").read()
    except OSError: continue
    buf.append(t); blen += len(t); used += len(t)
    if blen > 2_000_000: flush()
    if used > a.max_bytes: break
flush(); os.remove(chunk) if os.path.exists(chunk) else None
ids = [i for i, _ in cnt.most_common(a.n)]; seen = set(ids)
if a.fallback:
    for i in (int(x) for x in open(a.fallback).read().split()[3:]):
        if len(ids) >= a.n: break
        if i not in seen: ids.append(i); seen.add(i)
with open(a.out, "w") as f: f.write(f"llama-mtp-vocab-v1 {a.vocab} {len(ids)}\n" + "\n".join(map(str, ids)) + "\n")
print(f"{used} bytes, {sum(cnt.values())} tokens, {len(cnt)} distinct ids -> {len(ids)} ids in {a.out}")
