#pragma once
// Flash-Next runtime directional ablation (TQFN_ABLATE=<file>): per block, user-supplied unit directions are projected
// out of the block's outputs at run time (activation steering by orthogonal projection; the quantised weights stay as
// they are). With a per-block direction r and weights w:
//   attention / GDN output a:            a -= w_attn   (a . r) r
//   FFN output f (routed + shared):      f -= w_ffn    (f . r) r
//   every hyper-connection stream x_s of the block's updated residual (after the FFN update):
//                                        x_s -= w_stream sum_k (x_s . d_k) d_k   (d_k = r, or k orthonormal directions)
// Off (no file): nothing is allocated, no kernel runs and the output is unchanged. docs/reference/options.md describes
// the file and the capture workflow (TQFN_RESDUMP).
//
// TQAB file (little endian): "TQAB", int32 version (1, 2 or 3), int32 hidden, int32 entries; version 3 then int32 k
// (1..4). Per entry (trunk layer 0.., the MTP block at index `layers`): float w_attn, float w_ffn, [version >= 2: float
// w_stream,] float dir[hidden], [version 3: float sdir[k][hidden]]. Entries after the last one in the file have weight 0.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm::flashnext {

// the parsed file (host side): entries = trunk layers + 1, every array sized for all of them (missing entries zero)
struct TqabFile {
    int version = 0, k = 0;  // k: stream directions per entry (version 3), else 0
    int stored = 0;          // entries present in the file
    std::vector<float> wa, wf, ws;   // [entries]
    std::vector<float> dir;          // [entries][hidden]
    std::vector<float> sdir;         // [entries][4][hidden] (version 3)
};

inline TqabFile parse_tqab(const std::string & path, int hidden, int entries) {
    struct Closer { void operator()(FILE * f) const { std::fclose(f); } };
    std::unique_ptr<FILE, Closer> f(std::fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("TQFN_ABLATE: cannot open " + path);
    auto rd = [&](void * p, size_t bytes) { return bytes == 0 || std::fread(p, 1, bytes, f.get()) == bytes; };
    char magic[4] = {}; int32_t hdr[3] = {0, 0, 0};
    if (!rd(magic, 4) || std::memcmp(magic, "TQAB", 4) != 0 || !rd(hdr, sizeof(hdr)) || hdr[0] < 1 || hdr[0] > 3)
        throw std::runtime_error("TQFN_ABLATE: " + path + " is not a TQAB file (version 1-3)");
    if (hdr[1] != hidden || hdr[2] <= 0 || hdr[2] > entries)
        throw std::runtime_error("TQFN_ABLATE: " + path + ": hidden " + std::to_string(hdr[1]) + ", entries " + std::to_string(hdr[2]) +
                                 " (the model has hidden " + std::to_string(hidden) + ", " + std::to_string(entries) + " blocks with MTP)");
    TqabFile t; t.version = hdr[0]; t.stored = hdr[2];
    if (t.version == 3) {
        int32_t k = 0;
        if (!rd(&k, 4) || k < 1 || k > 4) throw std::runtime_error("TQFN_ABLATE: bad direction count in " + path);
        t.k = k;
    }
    const size_t H = size_t(hidden), E = size_t(entries);
    t.wa.assign(E, 0.f); t.wf.assign(E, 0.f); t.ws.assign(E, 0.f); t.dir.assign(E * H, 0.f);
    if (t.k) t.sdir.assign(E * 4 * H, 0.f);
    const size_t nw = t.version >= 2 ? 3 : 2;
    for (size_t e = 0; e < size_t(t.stored); ++e) {
        float w[3] = {0.f, 0.f, 0.f};
        if (!rd(w, nw * 4) || !rd(t.dir.data() + e * H, H * 4) || (t.k && !rd(t.sdir.data() + e * 4 * H, size_t(t.k) * H * 4)))
            throw std::runtime_error("TQFN_ABLATE: short file " + path);
        t.wa[e] = w[0]; t.wf[e] = w[1]; t.ws[e] = w[2];
    }
    char extra;
    if (std::fread(&extra, 1, 1, f.get()) == 1) throw std::runtime_error("TQFN_ABLATE: trailing bytes in " + path);
    return t;
}

struct Ablation {
    int h = 0, entries = 0;  // entries = trunk layers + 1 (MTP)
    std::vector<float> wa, wf, ws;
    float * dirs = nullptr;               // device [entries][h]
    float * sdirs = nullptr; int nsd = 0; // device [entries][4][h] (version 3; nsd = k), else unused
    bool on = false;

    bool attn(int l) const { return on && l < entries && wa[size_t(l)] != 0.f; }
    bool ffn(int l) const { return on && l < entries && wf[size_t(l)] != 0.f; }
    bool stream(int l) const { return on && l < entries && ws[size_t(l)] != 0.f; }
    const float * dir(int l) const { return dirs + size_t(l) * size_t(h); }
    // the stream ablation's directions and their count (k with a version 3 file, else the block direction)
    const float * sdir(int l) const { return nsd ? sdirs + size_t(l) * 4 * size_t(h) : dir(l); }
    int sdir_count() const { return nsd ? nsd : 1; }

    // engine load, before any graph capture: reads TQFN_ABLATE once and uploads the directions
    void load(int hidden, int layers) {
        const char * p = std::getenv("TQFN_ABLATE");
        if (!p || !*p || on) return;
        h = hidden; entries = layers + 1;
        TqabFile t = parse_tqab(p, h, entries);
        const size_t H = size_t(h), E = size_t(entries);
        if (cudaMalloc(reinterpret_cast<void **>(&dirs), E * H * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(dirs, t.dir.data(), t.dir.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("TQFN_ABLATE: device buffer");
        if (t.k && (cudaMalloc(reinterpret_cast<void **>(&sdirs), E * 4 * H * sizeof(float)) != cudaSuccess ||
                    cudaMemcpy(sdirs, t.sdir.data(), t.sdir.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess))
            throw std::runtime_error("TQFN_ABLATE: device buffer");
        wa = std::move(t.wa); wf = std::move(t.wf); ws = std::move(t.ws); nsd = t.k;
        int na = 0, nf = 0, ns = 0;
        for (size_t e = 0; e < E; ++e) { na += wa[e] != 0.f; nf += wf[e] != 0.f; ns += ws[e] != 0.f; }
        on = na + nf + ns > 0;  // an all-zero file runs nothing (the output stays the same as without it)
        std::fprintf(stderr, "tqllm: directional ablation from %s (TQAB v%d): %d entries, attention on %d blocks, FFN on %d, streams on %d (%d direction%s)\n",
                     p, t.version, t.stored, na, nf, ns, sdir_count(), sdir_count() > 1 ? "s" : "");
    }
};
inline Ablation & ablation() { static Ablation a; return a; }

}  // namespace tqllm::flashnext
