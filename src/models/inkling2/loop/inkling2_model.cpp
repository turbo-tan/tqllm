// Inkling-2-Mini resident text model (see inkling2_model.hpp). The arithmetic follows transformers PR #51050: BF16
// projections (cuBLAS, F32 accumulate, BF16 out), F32 norms / depth attention / message gates / convolutions / router,
// the residual stream rounded to BF16 between visits, experts decoded from their stored NVFP4 / MXFP8 bytes.
#include "inkling2_model.hpp"

#include "debug.hpp"
#include "gguf.hpp"
#include "parallel_blocks.hpp"
#include "inkling2_attention.cuh"
#include "inkling2_grouped.cuh"
#include "inkling2_kernels.cuh"
#include "inkling2_mtp.cuh"
#include "safetensors.hpp"
#include "inkling2_dflash2.cuh"
#include "nlohmann/json.hpp"  // tqllm::Json
#include "graph_capture.hpp"
#include "device_free_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tqllm {
namespace {
namespace K = tqllm_cuda::inkling2;
using bf16 = __nv_bfloat16;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char * what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("inkling2: cuBLAS ") + what + " " + std::to_string(int(e)));
}
float bf16_f(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; std::memcpy(&f, &u, 4); return f; }
float half_f(uint16_t h) {  // IEEE binary16 -> float (Q8_0 block scales)
    const uint32_t s = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(float(m | 1024u), int(e) - 25);
    uint32_t u; std::memcpy(&u, &v, 4);
    u |= s;
    std::memcpy(&v, &u, 4);
    return v;
}

struct Dev {
    void * p = nullptr;
    Dev() = default;
    explicit Dev(size_t n) { if (n) ck(cudaMalloc(&p, n), "malloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(Dev && o) noexcept : p(std::exchange(o.p, nullptr)) {}
    Dev & operator=(Dev && o) noexcept { if (this != &o) { if (p) cudaFree(p); p = std::exchange(o.p, nullptr); } return *this; }
    template <class T> T * as() const { return static_cast<T *>(p); }
};

// a dense projection: BF16 [rows][cols], or (int8 dense mode) INT8 rows + per-row scales in q8 / sc (d then empty)
// dense matrix: BF16 rows, or INT8 rows + per-row scales (--exact off), or TQT / TQT2 / TQT3 planes from a dense-TQT
// overlay (tq.fmt >= 3: few tokens run the one-expert TQT down kernel on the rotated input, prompt chunks decode to BF16)
struct Mat {
    Dev d, q8, sc, tqw, tqa; K::ExpertMat tq; int rows = 0, cols = 0;
    const bf16 * w() const { return d.as<bf16>(); }
    bool tqt() const { return tq.fmt >= 3; }
};
// page-locked host memory mapped into the device address space (experts that do not fit the device stream from it)
struct Pinned {
    void * p = nullptr;
    Pinned() = default;
    explicit Pinned(size_t n) { if (n) ck(cudaHostAlloc(&p, n, cudaHostAllocMapped), "host alloc"); }
    ~Pinned() { if (p) cudaFreeHost(p); }
    Pinned(Pinned && o) noexcept : p(std::exchange(o.p, nullptr)) {}
    Pinned & operator=(Pinned && o) noexcept { if (this != &o) { if (p) cudaFreeHost(p); p = std::exchange(o.p, nullptr); } return *this; }
    template <class T> T * as() const { return static_cast<T *>(p); }
};
// routed-expert stack: MXFP8 rows as stored, or NVFP4 repacked into NVF4 planes (codes / scales + global). Loaded to
// pinned host memory; place() moves the whole stack to the device, or `slots` experts of it (the rest stream from the
// host through the pointer tables wp / ap)
struct Experts {
    Dev d, aux, wp, ap; Pinned hw, ha;
    int rows = 0, cols = 0, E = 0, slots = 0;
    std::vector<int> slot_of, expert_of;  // partial residency: expert -> slot (-1: host), slot -> expert
    tqllm_cuda::inkling2::ExpertMat m;
    // BF16 prompt copies (big cards only): eg.bgu holds gate + up [E][2 rows][cols], ed.bdn the down [E][rows][cols]
    Dev bgu, bdn;
    size_t bytes() const { return (m.stride + m.aux_stride) * size_t(E); }
    size_t bf16_bytes() const { return size_t(rows) * cols * 2 * size_t(E); }
};
struct Ctrl { Mat down, fg; Dev FT, GT; };  // fg: forget then gain rows, BF16 [2 D][rank] (few tokens); FT / GT: transposed [rank][D] (prompts)

struct Layer {
    bool full = false, loop = false;
    Dev attn_q, ffn_q;  // depth-attention queries F32 [1 or 3][D], [D]
    Dev attn_norm, qn, kn, vn, ffn_norm;  // F32
    Mat wq, wk, wv, wo, wg, wr, wsink, kproj, vproj;
    Dev q_scale, rel_scale, rel_proj;  // BF16
    Dev kconv, vconv, aconv, mconv;    // F32 [C][4] (static); full layers: kconv / vconv = static_w F32 [4][C]
    Mat router;                        // BF16 [E][D] (the checkpoint's BF16 values; F32 arithmetic)
    Dev rbias;                         // F32 [E]
    Experts eg, eu, ed;
    Ctrl ctrl[4];                      // q, k, v, ffn (loop layers)
};

// KV BF16 [rows][1024]: rows = context (full layers) or a ring of window + chunk (sliding: key t in row t % ring); conv
// states F32 [C][3]: k, v, attention out, ffn out
// hist / snap: spec decode (the inputs of a verify block [8][C] BF16 and the states before it) for conv rollback
struct Slot { Dev kc, vc, conv[4], hist[4], snap[4]; int ring = 0; };
// the MTP drafter (mtp.safetensors): 3 dense layers + the rnn_hidden Markov head
struct MtpLayer {
    Dev in_norm, post_norm, dflare, qn, kn, knc;  // F32
    Mat qkv, o, gu, down, kvc;                    // BF16: [q; k; v], o, [gate; up], down, [k_ctx; v_ctx]
    Dev kc, vc; int ring = 0; bool local = false;
};

}  // namespace

struct Inkling2Model::Impl {
    GgufFile f;
    int D = 0, L = 0, H = 0, G = 0, HD = 128, DR = 16, E = 0, topk = 0, I = 0, V = 0, Vfull = 0, window = 0, ext = 0;
    int core0 = 0, core1 = 0, passes = 0, rank = 0, context = 0, chunk = 0, len = 0;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_alpha = 0, log_floor = 0, rscale = 0, gcap = 0, ucap = 0, fcap = 0;
    std::vector<Layer> Ls;
    Mat out;
    Dev embed, embed_norm, out_norm, out_q, msg_q, ucount;  // ucount: routed assignments per (layer, expert) since load
    Pinned embed_host;
    const bf16 * embed_ptr = nullptr;  // the table on the device, or mapped host memory when the experts need the room  // msg_q F32 [core layers][4][D] (q, k, v, ffn)
    std::vector<Slot> slots;
    bool dumped = false, hf_attention = false;
    std::vector<std::vector<int>> * route_log = nullptr;  // the top-k ids of every visit call, in call order
    Dev stats;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    // device copy of len (kernels of the few-token path read positions from it, so a captured decode graph stays valid),
    // pinned host staging for it, the token and the logits; the decode graph (n = 1)
    Dev dlen;
    int * h_len = nullptr, * h_tok = nullptr;
    float * h_logits = nullptr;
    cudaGraphExec_t graphs[K::kSmallN + 2] = {};  // [0]: the decode step; [n]: a verify block of n tokens; [kSmallN + 1]: greedy decode
    // greedy decode without logits: exact mode reads an INT8 copy of the LM head + per-row bounds and rescores only the rows
    // that can still win (K::head_greedy); --exact off argmaxes its own (INT8) head's logits on the device
    Dev hq8, hsc, hbound, happrox, hcells, dtok;
    int * h_next = nullptr;
    bool greedy_step = false;
    // workspace
    Dev ids, x0, hA, hB, P, B1[2], O17[2], a3, msgs, gt[3], r, m, oconv, mconv, tmp;
    Dev arn;  // attn_res_norm split-k scratch (n <= kSmallN), allocated and zeroed before any capture
    Dev xq, xk, xv, q, rs, sink, gate, kraw, vraw, kt, vt, kdyn, vdyn, att, o, xb, eids, ewts, rlog, yg, yu, act, yd, moe, z, g64, msgfa, fin, logits16,
        logits, part, gorder, gcounts, goffsets, gtile, pq, pr, gdedup;
    // BF16 expert prompt path (big cards): gathered inputs, gate+up GEMM out, down GEMM out, host counts / offsets
    Dev xg, guo, ygo, ginv;
    bool bf16_experts = false, fp4_experts = false;
    int tqt_group = 0;  // small cards: TQT prompt experts decoded G at a time (chosen at load from free memory; 0: the whole layer)
    bool tqt_experts = false;  // some layer's experts are TQT (fmt 3): few-token kernels read the rotated FP16 activations xh16 / acth16
    Dev xh16, acth16, tqt_bgu, tqt_bdn;  // rotated activations; per-visit BF16 decode of a TQT layer (cards without the whole-model BF16 copies)
    Dev q4codes, q4scales, q4row, q4acodes, q4ascales;
    int * h_gc = nullptr, * h_gf = nullptr;
    int want_rows = 0;  // logits rows allocated (grown on demand by a forward() that wants more)

    const GgufTensor & T(const std::string & n) const {
        auto it = f.tensors.find(n);
        if (it == f.tensors.end()) throw std::runtime_error("inkling2: missing tensor " + n);
        return it->second;
    }
    bool has(const std::string & n) const { return f.tensors.count(n) != 0; }
    Dev upload(const void * h, size_t n) { Dev d(n); ck(cudaMemcpy(d.p, h, n, cudaMemcpyHostToDevice), "upload"); return d; }
    std::vector<float> host_f32(const std::string & n) const {
        const auto & t = T(n);
        const auto b = f.read(t);
        std::vector<float> v;
        if (t.type == GGML_F32) { v.resize(b.size() / 4); std::memcpy(v.data(), b.data(), b.size()); }
        else if (t.type == GGML_BF16) {
            v.resize(b.size() / 2);
            for (size_t i = 0; i < v.size(); ++i) { uint16_t u; std::memcpy(&u, b.data() + 2 * i, 2); v[i] = bf16_f(u); }
        } else throw std::runtime_error("inkling2: " + n + " is not F32 / BF16");
        return v;
    }
    Dev f32(const std::string & n) { const auto v = host_f32(n); return upload(v.data(), v.size() * 4); }
    Dev raw(const std::string & n, uint32_t type) {
        const auto & t = T(n);
        if (t.type == GGML_Q8_0 && type == GGML_BF16) {  // 8-bit dense tensors of a TQT overlay (blocks of 32: FP16 scale + 32 int8) -> BF16
            const auto b = f.read(t);
            const size_t nel = size_t(t.bytes) / 34 * 32;
            std::vector<uint16_t> h(nel);
            for (size_t blk = 0; blk < nel / 32; ++blk) {
                uint16_t d16; std::memcpy(&d16, b.data() + blk * 34, 2);
                const float d = half_f(d16);
                for (size_t j = 0; j < 32; ++j) {
                    const float v = d * float(static_cast<int8_t>(b[blk * 34 + 2 + j]));
                    uint32_t u; std::memcpy(&u, &v, 4);
                    h[blk * 32 + j] = uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
                }
            }
            return upload(h.data(), h.size() * 2);
        }
        if (t.type != type) throw std::runtime_error("inkling2: " + n + " type " + std::to_string(t.type));
        const auto b = f.read(t);
        return upload(b.data(), b.size());
    }
    // dense TQT records (FP16 scale + code bytes per 128 columns) -> device planes [rows][cols * bits / 8] + scales [rows][cols / 128]
    bool tqt_dense_used = false;
    void tqt_dense(const std::string & n, const GgufTensor & t, Mat & m) {
        const int fmt = t.type == GGML_TQT ? 3 : t.type == GGML_TQT2 ? 4 : 5;
        const size_t R = size_t(m.rows), C = size_t(m.cols), nb = C / 128, cb = fmt == 3 ? 64 : fmt == 4 ? 32 : 48;
        const auto b = f.read(t);
        if (C % 128 || b.size() != R * nb * (cb + 2)) throw std::runtime_error("inkling2: " + n + " TQT bytes");
        std::vector<uint8_t> codes(R * nb * cb), sc(R * nb * 2);
        for (size_t r = 0; r < R; ++r)
            for (size_t k = 0; k < nb; ++k) {
                const uint8_t * rec = reinterpret_cast<const uint8_t *>(b.data()) + (r * nb + k) * (cb + 2);
                std::memcpy(sc.data() + (r * nb + k) * 2, rec, 2);
                std::memcpy(codes.data() + r * nb * cb + k * cb, rec + 2, cb);
            }
        m.tqw = upload(codes.data(), codes.size()); m.tqa = upload(sc.data(), sc.size());
        m.tq.w = m.tqw.as<uint8_t>(); m.tq.aux = m.tqa.as<uint8_t>(); m.tq.fmt = fmt;
        m.tq.stride = R * nb * cb; m.tq.aux_stride = R * nb * 2;
        if (!tq_xh.p) {  // made at load: decode steps run inside graph capture, which cannot allocate
            tq_xh = Dev(size_t(K::kSmallN) * 8192 * 2);
            std::vector<int> z(K::kSmallN, 0); std::vector<float> one(K::kSmallN, 1.0f);
            tq_ids = upload(z.data(), z.size() * 4); tq_wts = upload(one.data(), one.size() * 4);
        }
        tqt_dense_used = true;
    }
    // int8 dense mode (opt-in, not exact against the reference): the projections, the LM head and the message-gate downs
    // as INT8 W8A16 rows with per-row max-abs scales; the router and the norms stay BF16 / F32
    bool int8_dense = false;
    // small = projections that feed exponentials or gates (sinks, relative bias, output gate, conv projections, loop
    // controls): few bytes, error amplified -> stay BF16
    Mat mat(const std::string & n, bool small = false) {
        const auto & t = T(n);
        Mat m; m.cols = int(t.ne[0]); m.rows = int(t.ne[1]);
        if (t.type == GGML_TQT || t.type == GGML_TQT2 || t.type == GGML_TQT3) {
            if (small) throw std::runtime_error("inkling2: " + n + " is TQT but feeds a gate / exponential (kept BF16)");
            tqt_dense(n, t, m);
            return m;
        }
        m.d = raw(n, GGML_BF16);
        if (int8_dense && !small && m.cols % 16 == 0) {
            m.q8 = Dev(size_t(m.rows) * m.cols); m.sc = Dev(size_t(m.rows) * 4);
            K::quantize_rows_i8(m.w(), m.q8.as<int8_t>(), m.sc.as<float>(), m.rows, m.cols, s);
            ck(cudaStreamSynchronize(s), "quantize");
            m.d = Dev();
        }
        return m;
    }
    // NVFP4G records (per 128 columns: F32 global, 8 E4M3 scales, 64 B codes) -> NVF4 planes: codes [rows][cols / 2] and
    // aux [rows][cols / 16 scales, F32 global] (one global per row: the checkpoint has one per expert)
    Experts experts(const std::string & n) {
        const auto & t = T(n);
        Experts x; x.cols = int(t.ne[0]); x.rows = int(t.ne[1]);
        const size_t E_ = size_t(t.ne[2]), R = size_t(x.rows) * E_, C = size_t(x.cols);
        x.E = int(E_);
        const auto b = f.read(t);
        if (t.type == GGML_MXFP8) {
            if (b.size() != R * (C + C / 32)) throw std::runtime_error("inkling2: " + n + " bytes");
            x.hw = Pinned(b.size());
            std::memcpy(x.hw.p, b.data(), b.size());
            x.m.fmt = 1; x.m.stride = size_t(x.rows) * (C + C / 32);
        } else if (t.type == GGML_NVFP4G) {
            const size_t nb = C / 128, ab = C / 16 + 4;
            if (C % 128 || b.size() != R * nb * 76) throw std::runtime_error("inkling2: " + n + " bytes");
            x.hw = Pinned(R * C / 2); x.ha = Pinned(R * ab);
            uint8_t * codes = x.hw.as<uint8_t>(), * aux = x.ha.as<uint8_t>();
            std::atomic<bool> mixed{false};
            parallel_blocks(int64_t(R), [&](int64_t r) {
                const auto * row = reinterpret_cast<const uint8_t *>(b.data()) + size_t(r) * nb * 76;
                uint8_t * ar = aux + size_t(r) * ab;
                for (size_t k = 0; k < nb; ++k) {
                    const uint8_t * rec = row + k * 76;
                    if (k && std::memcmp(rec, row, 4) != 0) mixed = true;
                    std::memcpy(ar + k * 8, rec + 4, 8);
                    std::memcpy(codes + size_t(r) * C / 2 + k * 64, rec + 12, 64);
                }
                std::memcpy(ar + C / 16, row, 4);
            });
            if (mixed) throw std::runtime_error("inkling2: " + n + " has more than one NVFP4 global per row");
            x.m.fmt = 2; x.m.stride = size_t(x.rows) * C / 2; x.m.aux_stride = size_t(x.rows) * ab;
        } else if (t.type == GGML_TQT) {  // 66 B records (FP16 scale + 64 stream bytes per 128) -> stream planes [rows][cols / 2] + scales [rows][cols / 128]
            const size_t nb = C / 128;
            if (C % 128 || b.size() != R * nb * 66) throw std::runtime_error("inkling2: " + n + " bytes");
            x.hw = Pinned(R * C / 2); x.ha = Pinned(R * nb * 2);
            uint8_t * codes = x.hw.as<uint8_t>(), * sc = x.ha.as<uint8_t>();
            parallel_blocks(int64_t(R), [&](int64_t r) {
                const auto * row = reinterpret_cast<const uint8_t *>(b.data()) + size_t(r) * nb * 66;
                for (size_t k = 0; k < nb; ++k) {
                    std::memcpy(sc + (size_t(r) * nb + k) * 2, row + k * 66, 2);
                    std::memcpy(codes + size_t(r) * C / 2 + k * 64, row + k * 66 + 2, 64);
                }
            });
            x.m.fmt = 3; x.m.stride = size_t(x.rows) * C / 2; x.m.aux_stride = size_t(x.rows) * nb * 2;
            tqt_experts = true;
        } else if (t.type == GGML_TQT2) {  // 34 B records (FP16 scale + 32 stream bytes per 128) -> planes [rows][cols / 4] + scales [rows][cols / 128]
            const size_t nb = C / 128;
            if (C % 128 || b.size() != R * nb * 34) throw std::runtime_error("inkling2: " + n + " bytes");
            x.hw = Pinned(R * C / 4); x.ha = Pinned(R * nb * 2);
            uint8_t * codes = x.hw.as<uint8_t>(), * sc = x.ha.as<uint8_t>();
            parallel_blocks(int64_t(R), [&](int64_t r) {
                const auto * row = reinterpret_cast<const uint8_t *>(b.data()) + size_t(r) * nb * 34;
                for (size_t k = 0; k < nb; ++k) {
                    std::memcpy(sc + (size_t(r) * nb + k) * 2, row + k * 34, 2);
                    std::memcpy(codes + size_t(r) * C / 4 + k * 32, row + k * 34 + 2, 32);
                }
            });
            x.m.fmt = 4; x.m.stride = size_t(x.rows) * C / 4; x.m.aux_stride = size_t(x.rows) * nb * 2;
            tqt_experts = true;
        } else if (t.type == GGML_TQT3) {  // 50 B records (FP16 scale + 48 stream bytes per 128) -> planes [rows][cols * 3 / 8] + scales [rows][cols / 128]
            const size_t nb = C / 128;
            if (C % 128 || b.size() != R * nb * 50) throw std::runtime_error("inkling2: " + n + " bytes");
            x.hw = Pinned(R * C * 3 / 8); x.ha = Pinned(R * nb * 2);
            uint8_t * codes = x.hw.as<uint8_t>(), * sc = x.ha.as<uint8_t>();
            parallel_blocks(int64_t(R), [&](int64_t r) {
                const auto * row = reinterpret_cast<const uint8_t *>(b.data()) + size_t(r) * nb * 50;
                for (size_t k = 0; k < nb; ++k) {
                    std::memcpy(sc + (size_t(r) * nb + k) * 2, row + k * 50, 2);
                    std::memcpy(codes + size_t(r) * C * 3 / 8 + k * 48, row + k * 50 + 2, 48);
                }
            });
            x.m.fmt = 5; x.m.stride = size_t(x.rows) * C * 3 / 8; x.m.aux_stride = size_t(x.rows) * nb * 2;
            tqt_experts = true;
        } else {
            throw std::runtime_error("inkling2: " + n + " expert type " + std::to_string(t.type));
        }
        return x;
    }
    const uint8_t * host_dev(const Pinned & h) {  // the device address of mapped host memory
        void * d = nullptr;
        ck(cudaHostGetDevicePointer(&d, h.p, 0), "host map");
        return static_cast<const uint8_t *>(d);
    }
    // the whole stack to the device, or `slots` experts (the first `slots`, until replan_experts() picks by use)
    void place(Experts & x, int slots) {
        x.slots = std::min(slots, x.E);
        const size_t A = x.m.aux_stride, W = x.m.stride;
        if (x.slots == x.E) {
            x.d = upload(x.hw.p, W * x.E);
            if (A) x.aux = upload(x.ha.p, A * x.E);
            x.hw = Pinned(); x.ha = Pinned();
            x.m.w = x.d.as<uint8_t>(); x.m.aux = x.aux.as<uint8_t>(); x.m.wp = x.m.ap = nullptr;
            return;
        }
        x.d = Dev(W * size_t(std::max(1, x.slots))); if (A) x.aux = Dev(A * size_t(std::max(1, x.slots)));
        x.wp = Dev(size_t(x.E) * sizeof(void *)); x.ap = Dev(size_t(x.E) * sizeof(void *));
        x.slot_of.assign(size_t(x.E), -1); x.expert_of.assign(size_t(x.slots), -1);
        for (int e = 0; e < x.slots; ++e) { x.slot_of[size_t(e)] = e; x.expert_of[size_t(e)] = e; }
        x.m.w = x.m.aux = nullptr;
        x.m.wp = x.wp.as<const uint8_t *>(); x.m.ap = A ? x.ap.as<const uint8_t *>() : nullptr;
        for (int e = 0; e < x.slots; ++e) upload_slot(x, e, e);
        write_tables(x);
    }
    void upload_slot(Experts & x, int slot, int e) {
        const size_t A = x.m.aux_stride, W = x.m.stride;
        ck(cudaMemcpyAsync(x.d.as<uint8_t>() + W * slot, x.hw.as<uint8_t>() + W * e, W, cudaMemcpyHostToDevice, s), "slot");
        if (A) ck(cudaMemcpyAsync(x.aux.as<uint8_t>() + A * slot, x.ha.as<uint8_t>() + A * e, A, cudaMemcpyHostToDevice, s), "slot");
    }
    void write_tables(Experts & x) {
        const size_t A = x.m.aux_stride, W = x.m.stride;
        const uint8_t * hw = host_dev(x.hw), * ha = A ? host_dev(x.ha) : nullptr;
        std::vector<const uint8_t *> w(size_t(x.E)), a(size_t(x.E));
        for (int e = 0; e < x.E; ++e) {
            const int sl = x.slot_of[size_t(e)];
            w[size_t(e)] = sl >= 0 ? x.d.as<uint8_t>() + W * sl : hw + W * e;
            a[size_t(e)] = A ? (sl >= 0 ? x.aux.as<uint8_t>() + A * sl : ha + A * e) : nullptr;
        }
        ck(cudaMemcpyAsync(x.wp.p, w.data(), w.size() * sizeof(void *), cudaMemcpyHostToDevice, s), "tables");
        ck(cudaMemcpyAsync(x.ap.p, a.data(), a.size() * sizeof(void *), cudaMemcpyHostToDevice, s), "tables");
        ck(cudaStreamSynchronize(s), "tables");
    }
    // device memory for the routed experts after everything else: all of them when they fit, else the same share of
    // every layer's experts (the embedding table moves to mapped host memory first: one row per token crosses the bus)
    size_t expert_budget = SIZE_MAX, drafter_keep = 0;
    bool mma_experts = true, mma_dense = true;  // decode / verify kernel families, chosen per device at load (choose_kernels)
    size_t dev_mem = 0;
    int resident_share = 0;  // experts per 128 kept on the device (128: all)
    // TQT prompt path without whole-model BF16 copies: one layer's BF16 experts (gate+up, down) plus the grouped GEMM buffers
    size_t tqt_per_expert_bytes() const { return (size_t(2) * I * D + size_t(D) * I) * 2; }  // one expert's gate + up + down in BF16
    size_t tqt_fixed_bytes() const {  // the grouped GEMM buffers of one prompt chunk
        const size_t A = size_t(chunk) * topk;
        return A * D * 2 * 2 + A * 2 * I * 2 + A * 4;
    }
    // decode / verify kernel families: each family gives a token the same bits for any n (verify rows == single steps), so
    // per device the faster one runs. Measured at load on layer 0's experts and a D x D projection, 1 token and 8 tokens.
    void choose_kernels() {
        const Layer & Y = Ls[0];
        if (!K::expert_mma_ok(Y.eg.m, Y.ed.m, I, D)) { mma_experts = false; }
        const int n8 = K::kSmallN, A = n8 * topk;
        std::vector<int> bid(static_cast<size_t>(A));
        uint32_t r = 7;
        for (int t = 0; t < n8; ++t)
            for (int k = 0; k < topk; ++k) {
                int e;
                do { r = r * 1664525u + 1013904223u; e = int((r >> 8) % uint32_t(Y.eg.E)); } while (std::find(bid.begin() + t * topk, bid.begin() + t * topk + k, e) != bid.begin() + t * topk + k);
                bid[size_t(t) * topk + k] = e;
            }
        std::vector<float> w(static_cast<size_t>(A), 1.0f / topk);
        Dev did = upload(bid.data(), bid.size() * 4), dw = upload(w.data(), w.size() * 4);
        // its own scratch for 8 tokens (the model's buffers are sized for the prompt chunk, which may be 1 token)
        Dev bx(size_t(n8) * D * 2), bact(size_t(A) * I * 2), byd(size_t(A) * D * 2), bmoe(size_t(n8) * D * 2), bxh(size_t(n8) * D * 2),
            bah(size_t(A) * I * 2), bgr(size_t(1 + K::kSmallN * topk * (1 + K::kSmallN)) * 4), bout(size_t(n8) * D * 2);
        ck(cudaMemsetAsync(bx.p, 0, size_t(n8) * D * 2, s), "bench x");
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "ev"); ck(cudaEventCreate(&b), "ev");
        auto time = [&](auto && f) {
            for (int i = 0; i < 3; ++i) f();
            ck(cudaEventRecord(a, s), "ev");
            for (int i = 0; i < 20; ++i) f();
            ck(cudaEventRecord(b, s), "ev"); ck(cudaEventSynchronize(b), "ev");
            float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "ev");
            return double(ms) * 1000.0 / 20;
        };
        const bool tqt = Y.eg.m.fmt >= 3;
        auto warp = [&](int n) {
            if (tqt) {
                K::rotate_h(bx.as<bf16>(), bxh.as<__half>(), D, n, s);
                K::expert_gate_up(Y.eg.m, Y.eu.m, I, D, bx.as<bf16>(), did.as<int>(), bact.as<bf16>(), n, topk, gcap, ucap, s, bxh.as<__half>());
                K::rotate_h(bact.as<bf16>(), bah.as<__half>(), I, n * topk, s);
                K::expert_down(Y.ed.m, D, I, bact.as<bf16>(), did.as<int>(), dw.as<float>(), bmoe.as<bf16>(), n, topk, s, bah.as<__half>());
            } else if (n > 1) {
                K::expert_dedup(Y.eg.m, Y.eu.m, Y.ed.m, I, D, bx.as<bf16>(), did.as<int>(), dw.as<float>(), bgr.as<int>(), bact.as<bf16>(), byd.as<bf16>(),
                                bmoe.as<bf16>(), n, topk, gcap, ucap, s);
            } else {
                K::expert_gate_up(Y.eg.m, Y.eu.m, I, D, bx.as<bf16>(), did.as<int>(), bact.as<bf16>(), n, topk, gcap, ucap, s);
                K::expert_down(Y.ed.m, D, I, bact.as<bf16>(), did.as<int>(), dw.as<float>(), bmoe.as<bf16>(), n, topk, s);
            }
        };
        auto mma = [&](int n) {
            K::expert_mma(Y.eg.m, Y.eu.m, Y.ed.m, I, D, bx.as<bf16>(), did.as<int>(), dw.as<float>(), bgr.as<int>(), bact.as<bf16>(), byd.as<bf16>(),
                          bmoe.as<bf16>(), n, topk, gcap, ucap, s, bxh.as<__half>(), bah.as<__half>());
        };
        const bool drafter = drafter_keep > 0;
        if (mma_experts) {
            const double w1 = time([&] { warp(1); }), w8 = time([&] { warp(n8); }), m1 = time([&] { mma(1); }), m8 = time([&] { mma(n8); });
            mma_experts = K::prefer_mma(w1, w8, m1, m8, drafter);
            std::fprintf(stderr, "inkling2: experts per visit: warp kernels %.1f / %.1f us, tensor cores %.1f / %.1f us (1 / %d tokens) -> %s\n", w1, w8, m1, m8,
                         n8, mma_experts ? "tensor cores" : "warp kernels");
        }
        {   // a D x D BF16 projection (the dense projections' shape class)
            const Mat * W = nullptr;
            if (Y.wo.d.p && Y.wo.cols % 64 == 0) W = &Y.wo;
            if (W) {
                auto gemv = [&](int n, bool on) { K::set_gemv_mma(on); K::GemvSeg sg = seg(*W, bx.as<bf16>(), bout.p); K::gemv(&sg, 1, n, s); };
                const double w1 = time([&] { gemv(1, false); }), w8 = time([&] { gemv(n8, false); }), m1 = time([&] { gemv(1, true); }), m8 = time([&] { gemv(n8, true); });
                mma_dense = K::prefer_mma(w1, w8, m1, m8, drafter);
                std::fprintf(stderr, "inkling2: dense %dx%d: gemv %.1f / %.1f us, tensor cores %.1f / %.1f us -> %s\n", W->rows, W->cols, w1, w8, m1, m8,
                             mma_dense ? "tensor cores" : "gemv");
            }
            K::set_gemv_mma(mma_dense);
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
        ck(cudaStreamSynchronize(s), "choose kernels");
    }
    void place_experts() {
        size_t total = 0;
        for (const auto & Y : Ls) total += Y.eg.bytes() + Y.eu.bytes() + Y.ed.bytes();
        auto budget = [&] {
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "mem info");
            size_t keep = (size_t(1536) << 20) + drafter_keep;  // cuBLAS workspace, graphs, logits growth; the drafter enabled after the plan
            if (tqt_experts && dev_mem < (size_t(96) << 30)) {  // the per-visit BF16 decode of a TQT layer, one group at a time
                tqt_group = K::choose_tqt_group(E, fr, total, keep + tqt_fixed_bytes(), tqt_per_expert_bytes());
                keep += tqt_fixed_bytes() + size_t(tqt_group) * tqt_per_expert_bytes();
            }
            return std::min(expert_budget, fr > keep ? fr - keep : 0);
        };
        size_t b = budget();
        if (total > b && !embed_host.p) {
            const size_t eb = size_t(Vfull) * D * 2;
            embed_host = Pinned(eb);
            ck(cudaMemcpy(embed_host.p, embed.p, eb, cudaMemcpyDeviceToHost), "embed");
            embed = Dev();
            embed_ptr = reinterpret_cast<const bf16 *>(host_dev(embed_host));
            b = budget();
        }
        const int E_ = Ls[0].eg.E;
        resident_share = total <= b ? E_ : std::max(topk, int(double(b) / double(total) * E_));
        for (auto & Y : Ls) { place(Y.eg, resident_share); place(Y.eu, resident_share); place(Y.ed, resident_share); }
        ck(cudaStreamSynchronize(s), "place");
        if (tqt_group)
            std::fprintf(stderr, "inkling2: TQT prompt experts in groups of %d (%.2f GB BF16 scratch), %d of %d experts per layer resident\n", tqt_group,
                         double(size_t(tqt_group) * tqt_per_expert_bytes()) / 1e9, resident_share, E_);
        // big cards (>= 96 GB): BF16 prompt copies for the cuBLAS path, allocated after the quantized stacks are placed
        // (the pinned host copies are free again then); a failed allocation keeps the WMMA path
        // Only with room to spare: on unified memory (GB10) an allocation past what is free does not fail cleanly, the
        // system runs out (a 1.8.0 release check rebooted a GB10 at this step with another job holding ~50 GB)
        size_t bf16_need = 0, bf16_free = 0, bf16_tot = 0;
        for (const auto & Y : Ls) bf16_need += Y.eg.bf16_bytes() * 2 + Y.ed.bf16_bytes();
        bf16_need += size_t(chunk) * topk * (D * 4 + 2 * I * 2 + 4);
        // unified memory: cudaMemGetInfo counts the page cache as used, and a cgroup may cap the process below the device
        bf16_free = tqllm::device_free_bytes(&bf16_tot);
        if (int dv = 0, integ = 0; cudaGetDevice(&dv) == cudaSuccess && cudaDeviceGetAttribute(&integ, cudaDevAttrIntegrated, dv) == cudaSuccess && integ) {
            std::ifstream mi("/proc/meminfo");
            bf16_free = std::min(bf16_free, tqllm::host_available_bytes(std::string(std::istreambuf_iterator<char>(mi), {})));
        }
        const bool bf16_fits = bf16_free >= bf16_need + (size_t(16) << 30);
        if (!fp4_experts && dev_mem >= (size_t(96) << 30) && expert_budget == SIZE_MAX && !bf16_fits)
            std::fprintf(stderr, "inkling2: BF16 prompt copies skipped (%.1f GB needed + 16 GB headroom, %.1f GB free): quantized prompt path\n",
                         double(bf16_need) / 1e9, double(bf16_free) / 1e9);
        if (!fp4_experts && dev_mem >= (size_t(96) << 30) && expert_budget == SIZE_MAX && bf16_fits) {
            try {
                const size_t A = size_t(chunk) * topk;
                xg = Dev(A * D * 2); guo = Dev(A * 2 * I * 2); ygo = Dev(A * D * 2); ginv = Dev(A * 4);
                ck(cudaMallocHost(&h_gc, size_t(E) * 4), "pinned"); ck(cudaMallocHost(&h_gf, size_t(E) * 4), "pinned");
                for (auto & Y : Ls) {
                    Y.eg.bgu = Dev(Y.eg.bf16_bytes() * 2);  // gate + up in one [E][2I][D] copy (one GEMM per expert)
                    K::experts_decode_bf16(Y.eg.m, Y.eg.E, Y.eg.rows, Y.eg.cols, 2 * Y.eg.rows, Y.eg.bgu.as<bf16>(), s);
                    K::experts_decode_bf16(Y.eu.m, Y.eu.E, Y.eu.rows, Y.eu.cols, 2 * Y.eu.rows, Y.eg.bgu.as<bf16>() + size_t(Y.eg.rows) * Y.eg.cols, s);
                    Y.ed.bdn = Dev(Y.ed.bf16_bytes());
                    K::experts_decode_bf16(Y.ed.m, Y.ed.E, Y.ed.rows, Y.ed.cols, Y.ed.rows, Y.ed.bdn.as<bf16>(), s);
                }
                ck(cudaStreamSynchronize(s), "decode bf16");
                bf16_experts = true;
            } catch (...) {
                bf16_experts = false;
                xg = Dev(); guo = Dev(); ygo = Dev(); ginv = Dev();
                if (h_gc) { cudaFreeHost(h_gc); h_gc = nullptr; }
                if (h_gf) { cudaFreeHost(h_gf); h_gf = nullptr; }
                for (auto & Y : Ls) { Y.eg.bgu = Dev(); Y.ed.bdn = Dev(); }
                cudaGetLastError();
            }
        }
        if (tqt_experts && !bf16_experts) {  // TQT prompt path: one layer's BF16 decode per visit, then the cuBLAS per-expert GEMMs
            const size_t A = size_t(chunk) * topk;
            const int G = tqt_group ? tqt_group : E;
            tqt_bgu = Dev(size_t(G) * 2 * I * D * 2); tqt_bdn = Dev(size_t(G) * D * I * 2);
            xg = Dev(A * D * 2); guo = Dev(A * 2 * I * 2); ygo = Dev(A * D * 2); ginv = Dev(A * 4);
            if (!h_gc) { ck(cudaMallocHost(&h_gc, size_t(E) * 4), "pinned"); ck(cudaMallocHost(&h_gf, size_t(E) * 4), "pinned"); }
        }
        if (fp4_experts) {
            const size_t max_rows = std::max<size_t>(size_t(chunk), size_t(chunk) * topk);
            const size_t max_cols = std::max<size_t>(D, I);
            q4codes = Dev(max_rows * max_cols / 2);
            q4scales = Dev(max_rows * max_cols / 16);
            q4row = Dev(max_rows * 4);
            q4acodes = Dev(max_rows * max_cols / 2);
            q4ascales = Dev(max_rows * max_cols / 16);
        }
        ucount = Dev(size_t(L) * E_ * 4);
        ck(cudaMemset(ucount.p, 0, size_t(L) * E_ * 4), "counts");
    }
    // partial residency: per layer the resident_share experts routed most so far (since load) move to the device
    std::vector<float> route_hist;  // decayed routed counts per (layer, expert), host side
    int decode_steps = 0;           // decode forwards since the last plan (partial residency replans every kReplanEvery)
    int replan_experts() {
        if (resident_share >= Ls[0].eg.E) return 0;
        const int E_ = Ls[0].eg.E;
        std::vector<int> fresh(size_t(L) * E_);
        ck(cudaMemcpyAsync(fresh.data(), ucount.p, fresh.size() * 4, cudaMemcpyDeviceToHost, s), "counts");
        ck(cudaStreamSynchronize(s), "counts");
        ck(cudaMemsetAsync(ucount.p, 0, size_t(L) * E_ * 4, s), "counts");
        // decayed history: routes since the last plan count fully, older ones half per plan, so the resident set follows
        // what decode routes to now (a prompt's routing, or an earlier request's, fades)
        if (route_hist.size() != fresh.size()) route_hist.assign(fresh.size(), 0.0f);
        std::vector<float> cnt(fresh.size());
        for (size_t i = 0; i < fresh.size(); ++i) cnt[i] = route_hist[i] = 0.5f * route_hist[i] + float(fresh[i]);
        int moved = 0;
        for (int l = 0; l < L; ++l) {
            std::vector<int> order(static_cast<size_t>(E_));
            for (int e = 0; e < E_; ++e) order[size_t(e)] = e;
            const float * c = cnt.data() + size_t(l) * E_;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c[a] > c[b]; });
            std::vector<char> want(static_cast<size_t>(E_), 0);
            for (int i = 0; i < resident_share; ++i) want[size_t(order[size_t(i)])] = 1;
            for (Experts * x : {&Ls[l].eg, &Ls[l].eu, &Ls[l].ed}) {
                std::vector<int> free_slots;
                for (int sl = 0; sl < x->slots; ++sl)
                    if (!want[size_t(x->expert_of[size_t(sl)])]) { x->slot_of[size_t(x->expert_of[size_t(sl)])] = -1; free_slots.push_back(sl); }
                for (int e = 0; e < E_; ++e)
                    if (want[size_t(e)] && x->slot_of[size_t(e)] < 0) {
                        const int sl = free_slots.back(); free_slots.pop_back();
                        x->slot_of[size_t(e)] = sl; x->expert_of[size_t(sl)] = e;
                        upload_slot(*x, sl, e);
                        ++moved;
                    }
                write_tables(*x);
            }
        }
        return moved;
    }
    // a BF16-valued tensor as BF16 (exact: the checkpoint stores these in BF16; the reference casts them to F32), optionally
    // transposed from [rows][cols] to [cols][rows]
    Dev bf16_of(const std::string & n, bool transpose = false) {
        const auto & t = T(n);
        const auto v = host_f32(n);
        const size_t cols = size_t(t.ne[0]), rows = v.size() / cols;
        std::vector<uint16_t> h(v.size());
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c) {
                const float x = v[r * cols + c];
                uint32_t u; std::memcpy(&u, &x, 4);
                if (u & 0xffffu) throw std::runtime_error("inkling2: " + n + " is not BF16-valued");
                h[transpose ? c * rows + r : r * cols + c] = uint16_t(u >> 16);
            }
        return upload(h.data(), h.size() * 2);
    }
    // conv1d weight [C][1][4] -> F32 [C][4]; static_w [4][C] stays [4][C]
    Dev conv(const std::string & n) { return f32(n); }

    void load(const std::string & path) {
        f = gguf_open(path);
        if (f.str("general.architecture") != "inkling2-loop") throw std::runtime_error("inkling2: not an inkling2-loop model: " + path);
        const std::string A = "inkling2-loop.";
        auto num = [&](const char * k) { return f.num(A + k); };
        D = int(num("embedding_length")); L = int(num("block_count")); H = int(num("attention.head_count"));
        G = int(num("attention.head_count_kv")); E = int(num("expert_count")); topk = int(num("expert_used_count"));
        I = int(num("expert_feed_forward_length")); V = int(num("output_vocab_size")); Vfull = int(num("vocab_size"));
        window = int(num("attention.sliding_window")); ext = int(num("attention.relative_extent"));
        eps = float(num("attention.layer_norm_rms_epsilon")); qk_clip = float(num("attention.qk_clip"));
        rel_clip = float(num("attention.relative_clip")); log_alpha = float(num("attention.log_scale_alpha"));
        log_floor = float(num("attention.log_scale_floor")); rscale = float(num("expert_weights_scale"));
        gcap = float(num("expert_gate_softcap")); ucap = float(num("expert_up_softcap")); fcap = float(num("final_logit_softcapping"));
        core0 = int(num("loop.core_start")); core1 = int(num("loop.core_end")); passes = int(num("loop.num_passes"));
        rank = int(num("loop.gate_rank"));
        if (int(num("attention.key_length")) != HD || int(num("attention.relative_length")) != DR || int(num("shortconv.l_cache")) != 4 ||
            int(num("attn_res.block_size")) != 6 || topk > 16)
            throw std::runtime_error("inkling2: unsupported shape");
        const std::string full = f.str(A + "attention.full_layers");
        embed = raw("token_embd.weight", GGML_BF16);
        embed_ptr = embed.as<bf16>();
        embed_norm = f32("token_embd_norm.weight");
        out_norm = f32("output_norm.weight");
        out_q = f32("output_res_q.weight");
        out = mat("output.weight");
        {   // message queries: [core layer][q, k, v, ffn][D]
            const auto q = host_f32("loop_msg_q.weight"), kv = host_f32("loop_msg_kv_q.weight");
            const int nc = core1 - core0;
            std::vector<float> m(size_t(nc) * 4 * D);
            for (int j = 0; j < nc; ++j) {
                std::memcpy(&m[(size_t(j) * 4 + 0) * D], &q[size_t(2 * j) * D], size_t(D) * 4);
                std::memcpy(&m[(size_t(j) * 4 + 1) * D], &kv[size_t(2 * j) * D], size_t(D) * 4);
                std::memcpy(&m[(size_t(j) * 4 + 2) * D], &kv[size_t(2 * j + 1) * D], size_t(D) * 4);
                std::memcpy(&m[(size_t(j) * 4 + 3) * D], &q[size_t(2 * j + 1) * D], size_t(D) * 4);
            }
            msg_q = upload(m.data(), m.size() * 4);
        }
        Ls.resize(L);
        for (int l = 0; l < L; ++l) {
            Layer & Y = Ls[l];
            const std::string d = "blk." + std::to_string(l) + ".";
            Y.full = full[l] == '1';
            Y.loop = l >= core0 && l < core1;
            if (l > 0) Y.attn_q = f32(d + "attn_res_q.weight");
            Y.ffn_q = f32(d + "ffn_res_q.weight");
            if (Y.full) {
                Y.qn = f32(d + "attn_q_in_norm.weight"); Y.kn = f32(d + "attn_k_in_norm.weight"); Y.vn = f32(d + "attn_v_in_norm.weight");
                Y.kconv = conv(d + "attn_k_conv_static.weight"); Y.vconv = conv(d + "attn_v_conv_static.weight");
                Y.kproj = mat(d + "attn_k_conv_proj.weight", true); Y.vproj = mat(d + "attn_v_conv_proj.weight", true);
            } else {
                Y.attn_norm = f32(d + "attn_norm.weight");
                Y.kconv = conv(d + "attn_k_conv.weight"); Y.vconv = conv(d + "attn_v_conv.weight");
            }
            Y.ffn_norm = f32(d + "ffn_norm.weight");
            Y.wq = mat(d + "attn_q.weight"); Y.wk = mat(d + "attn_k.weight"); Y.wv = mat(d + "attn_v.weight");
            Y.wo = mat(d + "attn_output.weight"); Y.wg = mat(d + "attn_gate.weight", true); Y.wr = mat(d + "attn_rel.weight", true);
            Y.wsink = mat(d + "attn_sink.weight", true);
            Y.q_scale = raw(d + "attn_q_clip.weight", GGML_BF16); Y.rel_scale = raw(d + "attn_rel_clip.weight", GGML_BF16);
            Y.rel_proj = raw(d + "attn_rel_proj.weight", GGML_BF16);
            Y.aconv = conv(d + "attn_post_conv.weight"); Y.mconv = conv(d + "ffn_post_conv.weight");
            { const auto & t = T(d + "ffn_gate_inp.weight"); Y.router.cols = int(t.ne[0]); Y.router.rows = int(t.ne[1]); }
            Y.router.d = bf16_of(d + "ffn_gate_inp.weight"); Y.rbias = f32(d + "exp_probs_b.bias");
            Y.eg = experts(d + "ffn_gate_exps.weight"); Y.eu = experts(d + "ffn_up_exps.weight"); Y.ed = experts(d + "ffn_down_exps.weight");
            if (Y.loop)
                for (int c = 0; c < 4; ++c) {
                    const std::string g = std::string("loop_ctrl_") + "qkvf"[c] + (c == 3 ? "fn_" : "_");
                    Y.ctrl[c].down = mat(d + g + "down.weight", true);
                    const auto & ft = T(d + g + "forget.weight");
                    if (int(ft.ne[0]) != rank || int(ft.ne[1]) != D) throw std::runtime_error("inkling2: loop control forget shape");
                    const Dev F = bf16_of(d + g + "forget.weight"), Gm = bf16_of(d + g + "gain.weight");  // [D][rank] each
                    Mat & fg = Y.ctrl[c].fg;
                    fg.rows = 2 * D; fg.cols = rank;
                    fg.d = Dev(size_t(fg.rows) * fg.cols * 2);
                    ck(cudaMemcpy(fg.d.p, F.p, size_t(D) * rank * 2, cudaMemcpyDeviceToDevice), "fg");
                    ck(cudaMemcpy(fg.d.as<char>() + size_t(D) * rank * 2, Gm.p, size_t(D) * rank * 2, cudaMemcpyDeviceToDevice), "fg");
                    Y.ctrl[c].FT = bf16_of(d + g + "forget.weight", true);
                    Y.ctrl[c].GT = bf16_of(d + g + "gain.weight", true);
                }
        }
    }

    void alloc() {
        if (out.d.p) {  // exact BF16 head: its INT8 copy and the per-row bounds (0.46 GB)
            hq8 = Dev(size_t(V) * D); hsc = Dev(size_t(V) * 4); hbound = Dev(size_t(V) * 4);
            K::quantize_rows_i8(out.w(), hq8.as<int8_t>(), hsc.as<float>(), V, D, s);
            K::head_bound_build(out.w(), hq8.as<int8_t>(), hsc.as<float>(), hbound.as<float>(), V, D, s);
        }
        happrox = Dev(size_t(K::kSmallN) * V * 4); hcells = Dev(16); dtok = Dev(size_t(K::kSmallN) * 4);
        ck(cudaMemsetAsync(hcells.p, 0, 16, s), "cells");
        ck(cudaHostAlloc(&h_next, size_t(K::kSmallN) * 4, cudaHostAllocDefault), "pinned");
        const int n = chunk, nslots = L + (passes - 1) * (core1 - core0);
        slots.resize(nslots);
        for (int i = 0; i < nslots; ++i) {
            Slot & sl = slots[i];
            const int l = i < L ? i : core0 + (i - L) % (core1 - core0);
            sl.ring = Ls[l].full ? 0 : window + std::max(chunk, K::kSmallN);
            const size_t rows = sl.ring && sl.ring < context ? size_t(sl.ring) : size_t(context);
            if (sl.ring >= context) sl.ring = 0;
            sl.kc = Dev(rows * G * HD * 2); sl.vc = Dev(rows * G * HD * 2);
            const int C[4] = {G * HD, G * HD, D, D};
            for (int c = 0; c < 4; ++c) sl.conv[c] = Dev(size_t(C[c]) * 3 * 4);
        }
        const size_t nD = size_t(n) * D * 4;
        ids = Dev(size_t(n) * 4);
        x0 = Dev(nD); hA = Dev(nD); hB = Dev(nD); P = Dev(nD); r = Dev(nD); m = Dev(nD); oconv = Dev(nD); mconv = Dev(nD); tmp = Dev(nD);
        for (int i = 0; i < 2; ++i) { B1[i] = Dev(nD); O17[i] = Dev(nD); }
        for (auto & g : gt) g = Dev(nD);
        a3 = Dev(nD * 3); msgs = Dev(nD * 4);
        arn = Dev(size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4);
        ck(cudaMemsetAsync(arn.as<float>(), 0, size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4, s), "attn scratch");
        const size_t nD2 = size_t(n) * D * 2;
        xq = Dev(nD2); xk = Dev(nD2); xv = Dev(nD2); o = Dev(nD2); xb = Dev(nD2); moe = Dev(nD2); fin = Dev(nD2);
        q = Dev(size_t(n) * H * HD * 2); gate = Dev(size_t(n) * H * HD * 2); att = Dev(size_t(n) * H * HD * 2);
        rs = Dev(size_t(n) * H * DR * 2); sink = Dev(size_t(n) * H * 2);
        kraw = Dev(size_t(n) * G * HD * 2); vraw = Dev(size_t(n) * G * HD * 2); kt = Dev(size_t(n) * G * HD * 2); vt = Dev(size_t(n) * G * HD * 2);
        kdyn = Dev(size_t(n) * G * 4 * 2); vdyn = Dev(size_t(n) * G * 4 * 2);
        eids = Dev(size_t(n) * topk * 4); ewts = Dev(size_t(n) * topk * 4); rlog = Dev(size_t(std::max(n, K::kSmallN)) * E * 4);
        act = Dev(size_t(n) * topk * I * 2);
        gorder = Dev(size_t(n) * topk * 4); gcounts = Dev(size_t(E) * 4); goffsets = Dev(size_t(E) * 4); gtile = Dev(size_t(E + 1) * 4);
        yd = Dev(size_t(n) * topk * D * 2);
        z = Dev(size_t(n) * 2 * D * 2 * 3); g64 = Dev(size_t(n) * rank * 2 * 3);  // up to three gates at once
        msgfa = Dev(size_t(n) * 2 * D * 4 * 3);
        grow_logits(std::max(1, std::min(n, K::kSmallN)));
        int pf = tqllm_cuda::inkling2_attention_decode_floats(H, K::kSmallN, kSplitsFull);
        pq = Dev(size_t(n) * H * HD * 2); pr = Dev(size_t(n) * H * DR * 4);  // prompt attention scratch
        gdedup = Dev(size_t(1 + K::kSmallN * topk * (1 + K::kSmallN)) * 4);
        xh16 = Dev(size_t(K::kSmallN) * D * 2); acth16 = Dev(size_t(K::kSmallN) * topk * I * 2);
        if (hf_attention) pf = std::max(pf, tqllm_cuda::inkling2_attention_partial_floats(H, n, context));
        part = Dev(size_t(pf) * 4);
        dlen = Dev(4);
        ck(cudaMallocHost(&h_len, 4), "pinned"); ck(cudaMallocHost(&h_tok, size_t(K::kSmallN) * 4), "pinned");
        ck(cudaMallocHost(&h_logits, size_t(K::kSmallN) * V * 4), "pinned");
        stats = Dev(size_t(n) * H * 2 * 4);
    }

    void grow_logits(int want) {
        if (want <= want_rows) return;
        ck(cudaStreamSynchronize(s), "grow");
        logits16 = Dev(size_t(want) * V * 2); logits = Dev(size_t(want) * V * 4);
        want_rows = want;
        for (auto & g : graphs) if (g) { cudaGraphExecDestroy(g); g = nullptr; }  // they hold the old buffers
        for (auto & g : vgraphs) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
    }
    // Y[n][rows] = X[n][cols] W^T, BF16 in / out, F32 accumulate
    void gemm(const bf16 * W, int rows, int cols, const bf16 * X, void * Y, int n, bool f32 = false) {
        const float one = 1.0f, zero = 0.0f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &one, W, CUDA_R_16BF, cols, X, CUDA_R_16BF, cols,
                        &zero, Y, f32 ? CUDA_R_32F : CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm");
    }
    // rows r0 .. r0 + nr - 1 of W (BF16 or INT8 rows)
    static K::GemvSeg seg_rows(const Mat & W, int r0, int nr, const bf16 * x, void * y) {
        if (W.tqt()) throw std::runtime_error("inkling2: row slice of a TQT matrix");
        if (W.q8.p) return K::GemvSeg{reinterpret_cast<const bf16 *>(W.q8.as<int8_t>() + size_t(r0) * W.cols), x, y, nr, W.cols, W.cols, nr, 0, W.sc.as<float>() + r0};
        return K::GemvSeg{W.w() + size_t(r0) * W.cols, x, y, nr, W.cols, W.cols, nr, 0};
    }
    static K::GemvSeg seg(const Mat & W, const bf16 * x, void * y, bool f32 = false) {
        if (W.tqt()) return K::GemvSeg{nullptr, x, y, W.rows, W.cols, W.cols, W.rows, f32 ? 1 : 0, nullptr, &W};
        if (W.q8.p) return K::GemvSeg{reinterpret_cast<const bf16 *>(W.q8.p), x, y, W.rows, W.cols, W.cols, W.rows, f32 ? 1 : 0, W.sc.as<float>()};
        return K::GemvSeg{W.w(), x, y, W.rows, W.cols, W.cols, W.rows, f32 ? 1 : 0};
    }
    // projections that share nothing but the launch: few tokens -> one gemv launch, else one cuBLAS GEMM each (INT8 rows
    // dequantised to a BF16 scratch first: the same weights as the gemv)
    Dev dq_scratch;
    // dense TQT segments: few tokens -> rotate the input per 128 (FP16), the one-expert TQT down kernel (id 0, weight 1);
    // prompt chunks -> decode the matrix to the BF16 scratch, then the GEMM
    Dev tq_xh, tq_ids, tq_wts;
    void project_tqt(const K::GemvSeg & g, int n) {
        const Mat & W = *static_cast<const Mat *>(g.tq);
        if (g.f32 || g.ldx != W.cols || g.ldy != W.rows) throw std::runtime_error("inkling2: dense TQT segment layout");
        if (n <= K::kSmallN) {
            if (W.cols > 8192) throw std::runtime_error("inkling2: dense TQT cols > 8192");
            K::rotate_h(g.x, tq_xh.as<__half>(), W.cols, n, s);
            K::expert_down(W.tq, W.rows, W.cols, g.x, tq_ids.as<int>(), tq_wts.as<float>(), static_cast<bf16 *>(g.y), n, 1, s, tq_xh.as<__half>());
            return;
        }
        const size_t need = size_t(W.rows) * W.cols * 2;
        if (!dq_scratch.p || dq_bytes < need) { ck(cudaStreamSynchronize(s), "scratch"); dq_scratch = Dev(need); dq_bytes = need; }
        K::experts_decode_bf16(W.tq, 1, W.rows, W.cols, W.rows, dq_scratch.as<bf16>(), s);
        gemm(dq_scratch.as<bf16>(), W.rows, W.cols, g.x, g.y, n, false);
    }
    void project(const K::GemvSeg * sg, int ns, int n, bool head = false) {  // head: the LM head (K::gemv_rows for few tokens)
        K::GemvSeg plain[K::kMaxSegs]; int np = 0;
        for (int i = 0; i < ns; ++i) {
            if (sg[i].tq) project_tqt(sg[i], n);
            else plain[np++] = sg[i];
        }
        if (np < ns) { sg = plain; ns = np; if (!ns) return; }
        if (n <= K::kSmallN) { if (head) K::gemv_rows(sg, ns, n, s); else K::gemv(sg, ns, n, s); return; }
        for (int i = 0; i < ns; ++i) {
            const bf16 * W = sg[i].W;
            if (sg[i].scale) {
                const size_t need = size_t(sg[i].rows) * sg[i].K * 2;
                if (!dq_scratch.p || dq_bytes < need) { ck(cudaStreamSynchronize(s), "scratch"); dq_scratch = Dev(need); dq_bytes = need; }
                K::dequant_rows_i8(reinterpret_cast<const int8_t *>(sg[i].W), sg[i].scale, dq_scratch.as<bf16>(), sg[i].rows, sg[i].K, s);
                W = dq_scratch.as<bf16>();
            }
            gemm(W, sg[i].rows, sg[i].K, sg[i].x, sg[i].y, n, sg[i].f32 != 0);
        }
    }
    size_t dq_bytes = 0;
    // message gates of the streams c0 .. c0 + count - 1 (inputs h[i], messages msg[i], outputs y[i])
    void msg_gates(const Layer & Y, int c0, int count, const float * const * h, const float * const * msg, float * const * y, int n) {
        K::GemvSeg sg[4];
        for (int i = 0; i < count; ++i) {
            bf16 * zi = z.as<bf16>() + size_t(i) * n * 2 * D;
            K::msg_gate_in(h[i], msg[i], zi, n, D, eps, s);
            sg[i] = seg(Y.ctrl[c0 + i].down, zi, g64.as<bf16>() + size_t(i) * n * rank);
        }
        project(sg, count, n);
        if (n > K::kSmallN) {  // prompt chunks: the tile kernel (weights read once per 8 tokens)
            for (int i = 0; i < count; ++i)
                K::msg_gate_out(g64.as<bf16>() + size_t(i) * n * rank, Y.ctrl[c0 + i].FT.as<bf16>(), Y.ctrl[c0 + i].GT.as<bf16>(), h[i], msg[i], y[i], n, D,
                                rank, s);
            return;
        }
        // few tokens: f / a = F g, G g as one tensor-core projection per gate ([2 D] rows, F32), then y = (1 - tanh f) h + tanh(a) m
        for (int i = 0; i < count; ++i)
            sg[i] = K::GemvSeg{Y.ctrl[c0 + i].fg.w(), g64.as<bf16>() + size_t(i) * n * rank, msgfa.as<float>() + size_t(i) * n * 2 * D, 2 * D, rank, rank,
                               2 * D, 1};
        project(sg, count, n);
        for (int i = 0; i < count; ++i) K::msg_gate_combine(msgfa.as<float>() + size_t(i) * n * 2 * D, h[i], msg[i], y[i], n, D, s);
    }
    int slot_of(int l, int pass) const { return pass == 0 || l < core0 || l >= core1 ? l : L + (pass - 1) * (core1 - core0) + (l - core0); }
    // decode attention: splits of a query's span (full / windowed layers)
    static constexpr int kSplitsFull = 64, kSplitsWindow = 16;

    // one layer visit: h -> hout. B = completed blocks; msg = the four messages [4][n][D] of passes after the first.
    // Few tokens (n <= kSmallN, F32 attention) run the few-token kernels, whose per-token arithmetic does not depend on n.
    void visit(int l, int pass, const float * h, std::vector<const float *> B, const float * msg, float * hout, int n) {
        const Layer & Y = Ls[l];
        Slot & S = slots[slot_of(l, pass)];
        const bool boundary = l % 6 == 0, small = n <= K::kSmallN;
        const size_t nD = size_t(n) * D;
        float * a = a3.as<float>();
        K::States st{};
        auto states = [&](const std::vector<const float *> & v) {
            if (v.size() > size_t(K::kMaxStates)) throw std::runtime_error("inkling2: too many depth states");
            K::States x{}; x.count = int(v.size());
            for (size_t i = 0; i < v.size(); ++i) x.s[i] = v[i];
            return x;
        };
        std::vector<const float *> Bh = B; Bh.push_back(h);
        const float * sq, * sk, * sv;  // the q / k / v input streams (F32)
        const float * nq_ = Y.full ? Y.qn.as<float>() : Y.attn_norm.as<float>();
        const float * nk_ = Y.full ? Y.kn.as<float>() : Y.attn_norm.as<float>();
        const float * nv_ = Y.full ? Y.vn.as<float>() : Y.attn_norm.as<float>();
        const int nstreams = Y.full ? 3 : 1;
        // no messages: the depth attention and the input norms in one kernel (per token, any n)
        const bool fused_in = l > 0 && !msg;
        if (fused_in) {
            const float * w[3] = {nq_, nk_, nv_};
            bf16 * y[3] = {xq.as<bf16>(), xk.as<bf16>(), xv.as<bf16>()};
            K::attn_res_norm(states(Bh), Y.attn_q.as<float>(), nstreams, nullptr, w, y, n, D, eps, arn.as<float>(), s);
            sq = sk = sv = nullptr;
        }
        else if (l == 0) { sq = sk = sv = h; }
        else if (!Y.full) { st = states(Bh); K::attn_res(st, Y.attn_q.as<float>(), 1, a, n, D, arn.as<float>(), s); sq = sk = sv = a; }
        else { st = states(Bh); K::attn_res(st, Y.attn_q.as<float>(), 3, a, n, D, arn.as<float>(), s); sq = a; sk = a + nD; sv = a + 2 * nD; }
        if (msg) {
            const float * hin[3] = {sq, sk, sv}, * mi[3] = {msg, msg + nD, msg + 2 * nD};
            float * yo[3] = {gt[0].as<float>(), gt[1].as<float>(), gt[2].as<float>()};
            msg_gates(Y, 0, 3, hin, mi, yo, n);
            sq = gt[0].as<float>(); sk = gt[1].as<float>(); sv = gt[2].as<float>();
        }
        // one stream with one norm (sliding layers outside the loop passes): normalise once
        const bool one = fused_in ? !Y.full : sq == sk && sq == sv && nq_ == nk_ && nq_ == nv_;
        if (!fused_in) K::rms_norm(sq, nq_, nullptr, xq.as<bf16>(), n, D, eps, s);
        const bf16 * pxq = xq.as<bf16>(), * pxk = pxq, * pxv = pxq;
        if (!one) {
            if (!fused_in) {
                K::rms_norm(sk, nk_, nullptr, xk.as<bf16>(), n, D, eps, s);
                K::rms_norm(sv, nv_, nullptr, xv.as<bf16>(), n, D, eps, s);
            }
            pxk = xk.as<bf16>(); pxv = xv.as<bf16>();
        }
        K::GemvSeg sg[8];
        int ns = 0;
        sg[ns++] = seg(Y.wq, pxq, q.p); sg[ns++] = seg(Y.wr, pxq, rs.p); sg[ns++] = seg(Y.wsink, pxq, sink.p); sg[ns++] = seg(Y.wg, pxq, gate.p);
        sg[ns++] = seg(Y.wk, pxk, kraw.p); sg[ns++] = seg(Y.wv, pxv, vraw.p);
        if (Y.full) { sg[ns++] = seg(Y.kproj, pxk, kdyn.p); sg[ns++] = seg(Y.vproj, pxv, vdyn.p); }
        project(sg, ns, n);
        const int C = G * HD;
        if (small) {  // both convolutions and kv_finish in one launch
            K::ConvKvA cv{};
            cv.kv = K::KvFinishA{kt.as<bf16>(), vt.as<bf16>(), S.kc.as<bf16>(), S.vc.as<bf16>(), G, qk_clip, eps, dlen.as<int>(), S.ring};
            const bf16 * raw[2] = {kraw.as<bf16>(), vraw.as<bf16>()}, * dyn[2] = {kdyn.as<bf16>(), vdyn.as<bf16>()};
            const float * w[2] = {Y.kconv.as<float>(), Y.vconv.as<float>()};
            bf16 * y[2] = {kt.as<bf16>(), vt.as<bf16>()};
            for (int i = 0; i < 2; ++i) {
                cv.dyn[i] = Y.full;
                cv.s[i] = K::ConvStaticA{raw[i], w[i], S.conv[i].as<float>(), nullptr, y[i], n, C, nullptr, 0, hist(S, i)};
                cv.d[i] = K::ConvDynamicA{raw[i], w[i], dyn[i], S.conv[i].as<float>(), y[i], n, C, hist(S, i)};
            }
            K::conv_kv_finish(cv, n, s);
        } else if (Y.full) {
            K::conv_dynamic(kraw.as<bf16>(), Y.kconv.as<float>(), kdyn.as<bf16>(), S.conv[0].as<float>(), kt.as<bf16>(), n, C, s, hist(S, 0));
            K::conv_dynamic(vraw.as<bf16>(), Y.vconv.as<float>(), vdyn.as<bf16>(), S.conv[1].as<float>(), vt.as<bf16>(), n, C, s, hist(S, 1));
        } else {
            K::conv_static(kraw.as<bf16>(), Y.kconv.as<float>(), S.conv[0].as<float>(), nullptr, kt.as<bf16>(), n, C, s, nullptr, false, hist(S, 0));
            K::conv_static(vraw.as<bf16>(), Y.vconv.as<float>(), S.conv[1].as<float>(), nullptr, vt.as<bf16>(), n, C, s, nullptr, false, hist(S, 1));
        }
        if (!small) tqllm_cuda::inkling2_kv_finish(kt.as<bf16>(), vt.as<bf16>(), S.kc.as<bf16>(), S.vc.as<bf16>(), n, G, qk_clip, eps, dlen.as<int>(), S.ring, s);
        tqllm_cuda::Inkling2AttnArgs at;
        at.q = q.as<bf16>(); at.rs = rs.as<bf16>(); at.sink = sink.as<bf16>(); at.gate = gate.as<bf16>();
        at.q_scale = Y.q_scale.as<bf16>(); at.rel_scale = Y.rel_scale.as<bf16>(); at.rel_proj = Y.rel_proj.as<bf16>();
        at.kc = S.kc.as<bf16>(); at.vc = S.vc.as<bf16>(); at.out = att.as<bf16>(); at.part = part.as<float>();
        at.nq = n; at.pos0 = len; at.heads = H; at.kvheads = G; at.window = Y.full ? 0 : window; at.rel_extent = ext; at.ring = S.ring;
        at.log_scale = Y.full; at.hf_bf16 = hf_attention; at.stats = stats.as<float>(); at.qk_clip = qk_clip; at.rel_clip = rel_clip; at.log_alpha = log_alpha; at.log_floor = log_floor;
        if (hf_attention) {
            tqllm_cuda::inkling2_attention(at, s);
        } else {
            at.dpos = dlen.as<int>();
            if (small) {
                at.splits = Y.full ? kSplitsFull : kSplitsWindow;
                tqllm_cuda::inkling2_attention_decode(at, s);
            } else {
                tqllm_cuda::inkling2_attention_prompt(at, pq.as<bf16>(), pr.as<float>(), s);
            }
        }
        sg[0] = seg(Y.wo, att.as<bf16>(), o.p);
        project(sg, 1, n);
        // r = h + bf16(conv(o)) (F32), or bf16(conv(o)) alone at a block boundary
        const float * res = boundary ? oconv.as<float>() : r.as<float>();
        K::conv_static(o.as<bf16>(), Y.aconv.as<float>(), S.conv[2].as<float>(), const_cast<float *>(res), nullptr, n, D, s, boundary ? nullptr : h, false,
                       hist(S, 2));
        std::vector<const float *> Bl = B;
        if (boundary) Bl.push_back(h);
        Bl.push_back(res);
        st = states(Bl);
        if (!msg) {
            const float * w[1] = {Y.ffn_norm.as<float>()};
            bf16 * y[1] = {xb.as<bf16>()};
            K::attn_res_norm(st, Y.ffn_q.as<float>(), 1, nullptr, w, y, n, D, eps, arn.as<float>(), s);
        } else {
            K::attn_res(st, Y.ffn_q.as<float>(), 1, m.as<float>(), n, D, arn.as<float>(), s);
            const float * mi = m.as<float>();
            if (msg) {
                const float * hin[1] = {mi}, * mm[1] = {msg + 3 * nD};
                float * yo[1] = {tmp.as<float>()};
                msg_gates(Y, 3, 1, hin, mm, yo, n);
                mi = tmp.as<float>();
            }
            K::rms_norm(mi, Y.ffn_norm.as<float>(), nullptr, xb.as<bf16>(), n, D, eps, s);
        }
        sg[0] = seg(Y.router, xb.as<bf16>(), rlog.p, true);
        project(sg, 1, n);
        K::route_topk(rlog.as<float>(), Y.rbias.as<float>(), eids.as<int>(), ewts.as<float>(), n, E, topk, rscale, s, ucount.as<int>() + size_t(l) * E);
        if (route_log) {
            route_log->emplace_back(size_t(n) * topk);
            ck(cudaMemcpyAsync(route_log->back().data(), eids.p, size_t(n) * topk * 4, cudaMemcpyDeviceToHost, s), "route log");
            ck(cudaStreamSynchronize(s), "route log");
        }
        if (small && mma_experts && K::expert_mma_ok(Y.eg.m, Y.ed.m, I, D)) {  // decode and verify: tensor cores, a token's bits independent of n
            K::expert_mma(Y.eg.m, Y.eu.m, Y.ed.m, I, D, xb.as<bf16>(), eids.as<int>(), ewts.as<float>(), gdedup.as<int>(), act.as<bf16>(), yd.as<bf16>(),
                          moe.as<bf16>(), n, topk, gcap, ucap, s, xh16.as<__half>(), acth16.as<__half>());
        } else if (small && Y.eg.m.fmt >= 3 && n > 1 && Y.eg.m.fmt == Y.eu.m.fmt) {  // TQT verify blocks: each routed expert's rows read once per block
            K::rotate_h(xb.as<bf16>(), xh16.as<__half>(), D, n, s);
            K::expert_dedup_tqt(Y.eg.m, Y.eu.m, Y.ed.m, I, D, xh16.as<__half>(), eids.as<int>(), ewts.as<float>(), gdedup.as<int>(), act.as<bf16>(),
                                acth16.as<__half>(), yd.as<bf16>(), moe.as<bf16>(), n, topk, gcap, ucap, s);
        } else if (small && Y.eg.m.fmt >= 3) {  // TQT: the weights live in TQ4's rotated basis, so the activations rotate per 128 (FP16)
            K::rotate_h(xb.as<bf16>(), xh16.as<__half>(), D, n, s);
            K::expert_gate_up(Y.eg.m, Y.eu.m, I, D, xb.as<bf16>(), eids.as<int>(), act.as<bf16>(), n, topk, gcap, ucap, s, xh16.as<__half>());
            K::rotate_h(act.as<bf16>(), acth16.as<__half>(), I, n * topk, s);
            K::expert_down(Y.ed.m, D, I, act.as<bf16>(), eids.as<int>(), ewts.as<float>(), moe.as<bf16>(), n, topk, s, acth16.as<__half>());
        } else if (small && n > 1) {
            K::expert_dedup(Y.eg.m, Y.eu.m, Y.ed.m, I, D, xb.as<bf16>(), eids.as<int>(), ewts.as<float>(), gdedup.as<int>(), act.as<bf16>(), yd.as<bf16>(),
                            moe.as<bf16>(), n, topk, gcap, ucap, s);
        } else if (small) {
            K::expert_gate_up(Y.eg.m, Y.eu.m, I, D, xb.as<bf16>(), eids.as<int>(), act.as<bf16>(), n, topk, gcap, ucap, s);
            K::expert_down(Y.ed.m, D, I, act.as<bf16>(), eids.as<int>(), ewts.as<float>(), moe.as<bf16>(), n, topk, s);
        } else {  // prompt chunk: assignments grouped by expert, tensor-core GEMM
            const int A = n * topk;
            int * go = gorder.as<int>(), * gc = gcounts.as<int>(), * gf = goffsets.as<int>(), * gt_ = gtile.as<int>();
            K::group_assignments(eids.as<int>(), A, E, go, gc, gf, gt_, s);
            if (Y.eg.m.fmt >= 3) {  // TQT / TQT2: BF16 copies of the original-basis weights (resident, or decoded G experts at a time), cuBLAS per expert
                ck(cudaMemcpyAsync(h_gc, gcounts.p, size_t(E) * 4, cudaMemcpyDeviceToHost, s), "counts");
                ck(cudaMemcpyAsync(h_gf, goffsets.p, size_t(E) * 4, cudaMemcpyDeviceToHost, s), "offsets");
                ck(cudaStreamSynchronize(s), "group");
                if (bf16_experts) {
                    K::grouped_gate_up_bf16(Y.eg.bgu.as<bf16>(), I, D, xb.as<bf16>(), A, E, topk, go, h_gc, h_gf, xg.as<bf16>(), guo.as<bf16>(),
                                            act.as<bf16>(), gcap, ucap, blas, s);
                    K::grouped_down_bf16(Y.ed.bdn.as<bf16>(), D, I, act.as<bf16>(), A, E, h_gc, h_gf, ygo.as<bf16>(), blas, s);
                } else {
                    const int G = tqt_group ? tqt_group : E;
                    K::grouped_gather(xb.as<bf16>(), A, go, topk, xg.as<bf16>(), D, s);
                    for (int e0 = 0; e0 < E; e0 += G) {
                        const int g = std::min(G, E - e0);
                        int rows = 0;
                        for (int e = e0; e < e0 + g; ++e) rows += h_gc[e];
                        if (rows == 0) continue;
                        K::experts_decode_bf16(K::expert_slice(Y.eg.m, e0), g, I, D, 2 * I, tqt_bgu.as<bf16>(), s);
                        K::experts_decode_bf16(K::expert_slice(Y.eu.m, e0), g, I, D, 2 * I, tqt_bgu.as<bf16>() + size_t(I) * D, s);
                        K::experts_decode_bf16(K::expert_slice(Y.ed.m, e0), g, D, I, D, tqt_bdn.as<bf16>(), s);
                        K::grouped_gate_up_part_bf16(tqt_bgu.as<bf16>(), I, D, xg.as<bf16>(), g, h_gc + e0, h_gf + e0, guo.as<bf16>(), act.as<bf16>(),
                                                     gcap, ucap, blas, s);
                        K::grouped_down_bf16(tqt_bdn.as<bf16>(), D, I, act.as<bf16>(), A, g, h_gc + e0, h_gf + e0, ygo.as<bf16>(), blas, s);
                        if (e0 + g < E) ck(cudaStreamSynchronize(s), "tqt group");  // the scratch and the GEMM descriptors are reused by the next group
                    }
                }
                K::moe_combine_grouped(ygo.as<bf16>(), go, A, ewts.as<float>(), moe.as<bf16>(), n, D, topk, ginv.as<int>(), s);
            } else if (fp4_experts && Y.eg.m.fmt == 2 && Y.eu.m.fmt == 2 && Y.ed.m.fmt == 2 && I % 64 == 0 && D % 64 == 0) {
                K::quantize_act_nvfp4(xb.as<bf16>(), n, D, q4codes.as<uint8_t>(), q4scales.as<uint8_t>(), q4row.as<float>(), s);
                K::grouped_gate_up_fp4_q(Y.eg.m, Y.eu.m, I, D, q4codes.as<uint8_t>(), q4scales.as<uint8_t>(), q4row.as<float>(), A, E, topk, go, gc,
                                         gf, gt_, q4acodes.as<uint8_t>(), q4ascales.as<uint8_t>(), gcap, ucap, s);
                K::grouped_down_fp4(Y.ed.m, D, I, q4acodes.as<uint8_t>(), q4ascales.as<uint8_t>(), nullptr, A, E, go, gc, gf, gt_, yd.as<bf16>(), s);
                K::moe_combine(yd.as<bf16>(), ewts.as<float>(), moe.as<bf16>(), n, D, topk, s);
            } else if (bf16_experts && K::use_grouped_bf16(n)) {  // big card, big chunk: cuBLAS per expert on the BF16 copies (gather -> gate+up -> SwiGLU -> down, grouped order)
                ck(cudaMemcpyAsync(h_gc, gcounts.p, size_t(E) * 4, cudaMemcpyDeviceToHost, s), "counts");
                ck(cudaMemcpyAsync(h_gf, goffsets.p, size_t(E) * 4, cudaMemcpyDeviceToHost, s), "offsets");
                ck(cudaStreamSynchronize(s), "group");
                K::grouped_gate_up_bf16(Y.eg.bgu.as<bf16>(), I, D, xb.as<bf16>(), A, E, topk, go, h_gc, h_gf, xg.as<bf16>(), guo.as<bf16>(), act.as<bf16>(),
                                        gcap, ucap, blas, s);
                K::grouped_down_bf16(Y.ed.bdn.as<bf16>(), D, I, act.as<bf16>(), A, E, h_gc, h_gf, ygo.as<bf16>(), blas, s);
                K::moe_combine_grouped(ygo.as<bf16>(), go, A, ewts.as<float>(), moe.as<bf16>(), n, D, topk, ginv.as<int>(), s);
            } else {
                K::grouped_gate_up(Y.eg.m, Y.eu.m, I, D, xb.as<bf16>(), A, E, topk, go, gc, gf, gt_, act.as<bf16>(), gcap, ucap, s);
                K::grouped_down(Y.ed.m, D, I, act.as<bf16>(), A, E, go, gc, gf, gt_, yd.as<bf16>(), s);
                K::moe_combine(yd.as<bf16>(), ewts.as<float>(), moe.as<bf16>(), n, D, topk, s);
            }
        }
        K::conv_static(moe.as<bf16>(), Y.mconv.as<float>(), S.conv[3].as<float>(), hout, nullptr, n, D, s, res, true, hist(S, 3));  // bf16(res + bf16(conv))
    }

    // ---- MTP drafter (DSpark block of 8 + rnn_hidden Markov head; sglang inkling2_mtp.py) ----
    bool mtp_on = false;
    int mtp_block = 8, mrank = 256;
    float rope_theta = 1e6f, mtp_eps = 1e-6f;
    std::vector<MtpLayer> ml;
    Dev taps, m_alpha, m_mask, m_norm, m_fnorm, m_w1, m_rnn_b, m_fold, m_fold_b;  // alpha F32 [4][4] (softmax of fuse_weight)
    Mat m_rnn;
    Dev d_h, d_xb, d_qkv, d_att, d_ao, d_g, d_u, d_act, d_mo, d_hid, d_base, d_z, d_proj, d_state, d_lat, d_part, d_step, d_tok, d_frnn, d_cx, d_ckv, d_x0;
    void enable_mtp(const std::string & dir) {
        if (is_dflash2_folder(dir)) { enable_dflash2(dir); return; }
        const auto idx = safetensors_index_file(std::filesystem::path(dir) / "mtp.safetensors");
        auto get = [&](const std::string & n) {
            const auto it = idx.find(n);
            if (it == idx.end() || it->second.dtype != "BF16") throw std::runtime_error("inkling2 mtp: missing BF16 " + n);
            const auto b = safetensors_read(it->second);
            std::vector<uint16_t> v(b.size() / 2);
            std::memcpy(v.data(), b.data(), b.size());
            return std::make_pair(v, it->second.shape);
        };
        auto f32v = [&](const std::string & n) {
            const auto v = get(n).first;
            std::vector<float> f(v.size());
            for (size_t i = 0; i < v.size(); ++i) f[i] = bf16_f(v[i]);
            return upload(f.data(), f.size() * 4);
        };
        auto bf = [&](const std::string & n) { const auto v = get(n).first; return upload(v.data(), v.size() * 2); };
        auto cat = [&](std::initializer_list<std::string> names) {
            std::vector<uint16_t> all; int rows = 0, cols = 0;
            for (const auto & n : names) {
                const auto [v, sh] = get(n);
                rows += int(sh[0]); cols = int(sh[1]);
                all.insert(all.end(), v.begin(), v.end());
            }
            Mat m; m.rows = rows; m.cols = cols; m.d = upload(all.data(), all.size() * 2);
            return m;
        };
        const int ctx_ring = window + std::max(chunk, K::kSmallN);
        ml.resize(3);
        for (int i = 0; i < 3; ++i) {
            MtpLayer & Y = ml[size_t(i)];
            const std::string d = "layers." + std::to_string(i) + ".";
            Y.local = i != 1;  // local_layer_ids [0, 2], window 1024
            Y.in_norm = f32v(d + "input_layernorm.weight"); Y.post_norm = f32v(d + "post_attention_layernorm.weight");
            Y.dflare = f32v(d + "dflare_norm.weight");
            Y.qn = f32v(d + "self_attn.q_norm.weight"); Y.kn = f32v(d + "self_attn.k_norm.weight"); Y.knc = f32v(d + "self_attn.k_norm_ctx.weight");
            Y.qkv = cat({d + "self_attn.q_proj.weight", d + "self_attn.k_proj.weight", d + "self_attn.v_proj.weight"});
            Y.o = cat({d + "self_attn.o_proj.weight"});
            Y.gu = cat({d + "mlp.gate_proj.weight", d + "mlp.up_proj.weight"});
            Y.down = cat({d + "mlp.down_proj.weight"});
            Y.kvc = cat({d + "self_attn.k_proj_ctx.weight", d + "self_attn.v_proj_ctx.weight"});
            Y.ring = Y.local && ctx_ring < context ? ctx_ring : 0;
            const size_t rows = Y.ring ? size_t(Y.ring) : size_t(context);
            Y.kc = Dev(rows * G * HD * 2); Y.vc = Dev(rows * G * HD * 2);
        }
        {   // alpha = softmax(fuse_weight) per row (draft layers 0..2, then the Markov head)
            const auto fw = get("fuse_weight").first;
            std::vector<float> a(16);
            for (int r = 0; r < 4; ++r) {
                float mx = -1e30f, sum = 0.0f;
                for (int j = 0; j < 4; ++j) mx = std::max(mx, bf16_f(fw[size_t(r) * 4 + j]));
                for (int j = 0; j < 4; ++j) { a[size_t(r) * 4 + j] = std::exp(bf16_f(fw[size_t(r) * 4 + j]) - mx); sum += a[size_t(r) * 4 + j]; }
                for (int j = 0; j < 4; ++j) a[size_t(r) * 4 + j] /= sum;
            }
            m_alpha = upload(a.data(), a.size() * 4);
        }
        m_mask = bf("mask_embed"); m_norm = f32v("norm.weight"); m_fnorm = f32v("markov_head.dflare_norm.weight");
        m_w1 = bf("markov_head.markov_w1.weight"); m_rnn_b = bf("markov_head.rnn_proj.bias");
        m_rnn = cat({"markov_head.rnn_proj.weight"});
        if (m_rnn.rows != 3 * mrank || m_rnn.cols != 2 * mrank + 2 * D) throw std::runtime_error("inkling2 mtp: rnn_proj shape");
        {   // the Markov latent goes through hidden_out and the target LM head: fold = unembed @ hidden_out (load time)
            Dev hw = bf("markov_head.hidden_out.weight"), hb = bf("markov_head.hidden_out.bias");  // [D][R], [D]
            m_fold = Dev(size_t(V) * mrank * 2); m_fold_b = Dev(size_t(V) * 4);
            const float one = 1.0f, zero = 0.0f;
            Dev unembed;  // int8 dense mode: the LM head rows dequantised for the fold (the head the draft's base logits use)
            const bf16 * ue = out.w();
            if (out.q8.p) {
                unembed = Dev(size_t(V) * D * 2);
                K::dequant_rows_i8(out.q8.as<int8_t>(), out.sc.as<float>(), unembed.as<bf16>(), V, D, s);
                ue = unembed.as<bf16>();
            }
            cb(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, mrank, V, D, &one, hw.p, CUDA_R_16BF, mrank, ue, CUDA_R_16BF, D, &zero, m_fold.p,
                            CUDA_R_16BF, mrank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "fold");
            K::GemvSeg sg = seg(out, hb.as<bf16>(), m_fold_b.p, true);
            K::gemv(&sg, 1, 1, s);
            ck(cudaStreamSynchronize(s), "fold");
        }
        const int nb = mtp_block;
        taps = Dev(size_t(std::max(chunk, nb)) * 4 * D * 2);
        d_h = Dev(size_t(nb) * D * 4); d_xb = Dev(size_t(nb) * D * 2); d_qkv = Dev(size_t(nb) * ml[0].qkv.rows * 2);
        d_att = Dev(size_t(nb) * H * HD * 2); d_ao = Dev(size_t(nb) * D * 2);
        const int ffn = ml[0].gu.rows / 2;
        d_g = Dev(size_t(nb) * ffn * 2); d_u = Dev(size_t(nb) * ffn * 2); d_act = Dev(size_t(nb) * ffn * 2); d_mo = Dev(size_t(nb) * D * 2);
        d_hid = Dev(size_t(nb) * D * 2); d_base = Dev(size_t(nb) * V * 4); d_z = Dev(size_t(2 * mrank + 2 * D) * 2); d_proj = Dev(size_t(3 * mrank) * 4);
        d_state = Dev(size_t(mrank) * 4); d_lat = Dev(size_t(mrank) * 2); d_step = Dev(size_t(K::mtp_step_floats(V)) * 4); d_tok = Dev(size_t(nb) * 4);
        d_frnn = Dev(size_t(D) * 2); d_x0 = Dev(size_t(D) * 4);
        d_part = Dev(size_t(K::mtp_attention_floats(H, nb, kMtpSplits)) * 4);
        d_cx = Dev(size_t(std::max(chunk, nb)) * D * 2); d_ckv = Dev(size_t(std::max(chunk, nb)) * ml[0].kvc.rows * 2);
        mtp_common();
    }
    // both drafters: verify-block histories, the decode graphs recaptured with the taps
    void mtp_common() {
        if (len) throw std::runtime_error("inkling2 mtp: enable before the first forward");
        for (auto & S : slots) {
            const int C[4] = {G * HD, G * HD, D, D};
            for (int c = 0; c < 4; ++c) { S.hist[c] = Dev(size_t(K::kSmallN) * C[c] * 2); S.snap[c] = Dev(size_t(C[c]) * 12); }
        }
        for (auto & g : graphs) if (g) { cudaGraphExecDestroy(g); g = nullptr; }  // the steps now capture the taps
        for (auto & g : vgraphs) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
        mtp_on = true;
        prepare_greedy_graphs();
    }
    // capture (not run) the greedy decode graph and the greedy verify graphs of every block size now: their first use
    // would otherwise pay the capture inside a timed step (the spec policy compares step times)
    void prepare_greedy_graphs() {
        if (route_log || dump || hf_attention) return;
        grow_logits(K::kSmallN);
        *h_len = len;
        greedy_step = true;
        try {
            for (int n = 1; n <= K::kSmallN; ++n) {
                verifying = n > 1; verify_n = 0;
                cudaGraphExec_t & g = n == 1 ? graphs[K::kSmallN + 1] : vgraphs[n];
                if (!g) g = instantiate_graph(capture_graph(s, [&] { step(h_tok, n, n, nullptr); }));
            }
        } catch (...) { verifying = false; greedy_step = false; throw; }
        verifying = false; greedy_step = false;
    }
    static constexpr int kMtpSplits = 16;

    // ---- DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, -md FOLDER): five block-diffusion layers over [anchor, mask x 7]
    // whose keys come from the committed tokens' taps (fc + hidden_norm), two-tap dynamic convolutions around attention and
    // MLP, the target's LM head (its INT8 copy when there is one) for the top-16 candidates of each block row and a
    // rank-256 selector walking one path through them (sglang srt/models/dflash.py; the Metal port Inkling2MlxDFlash2).
    // It only proposes tokens; every one is verified.
    struct DfLayer { Dev in_norm, post_norm, q_norm, k_norm, attn_base, mlp_base, kc, vc; Mat qkv, o, attn_kernel, mlp_kernel, gate_up, down; int ring = 0; };
    bool dflash = false;
    std::vector<DfLayer> dfl;
    Mat df_fc, df_sel;
    Dev df_fc_norm, df_norm, df_pred, df_succ, df_mask;
    int df_pos = -1;  // draft_trace(): the block position instead of length()
    int df_H = 0, df_G = 0, df_I = 0, df_group = 0, df_window = 0, df_vocab = 0;
    float df_theta = 1e4f, df_eps = 1e-6f;
    Dev df_x, df_nf, df_n16, df_kern, df_cin, df_cin16, df_qkv, df_att, df_o16, df_of, df_gu, df_act, df_hid, df_logits, df_ids, df_vals, df_gate,
        df_tok, df_top, df_part, df_ctxf, df_ctx16, df_kv;
    static tqllm::Json read_json(const std::filesystem::path & f) {
        std::ifstream in(f, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return tqllm::Parser(text).parse();
    }
    static bool is_dflash2_folder(const std::string & dir) {
        const auto cfg = std::filesystem::path(dir) / "config.json";
        if (!std::filesystem::exists(cfg) || !std::filesystem::exists(std::filesystem::path(dir) / "model.safetensors")) return false;
        try {
            const auto j = read_json(cfg);
            const auto * a = j.get("architectures");
            return a && !a->arr.empty() && a->arr[0].str == "DFlash2DraftModel";
        } catch (const std::exception &) { return false; }
    }
    void enable_dflash2(const std::string & dir) {
        const auto cfg = read_json(std::filesystem::path(dir) / "config.json");
        auto num = [&](const tqllm::Json & j, const char * k) {
            const auto * v = j.get(k);
            if (!v) throw std::runtime_error(std::string("inkling2 DFlash2 config: missing ") + k);
            return v->num;
        };
        const auto * dcp = cfg.get("dflash_config");
        if (!dcp) throw std::runtime_error("inkling2 DFlash2: no dflash_config in " + dir);
        const auto & dc = *dcp;
        const int Dd = int(num(cfg, "hidden_size")), Ld = int(num(cfg, "num_hidden_layers")), HDd = int(num(cfg, "head_dim"));
        df_H = int(num(cfg, "num_attention_heads")); df_G = int(num(cfg, "num_key_value_heads")); df_I = int(num(cfg, "intermediate_size"));
        if (cfg.get("rms_norm_eps")) df_eps = float(num(cfg, "rms_norm_eps"));
        df_window = int(num(cfg, "sliding_window"));
        if (const auto * rp = cfg.get("rope_parameters")) if (const auto * th = rp->get("rope_theta")) df_theta = float(th->num);
        df_group = int(num(dc, "conv_group_size"));
        std::vector<int> taps_ids;
        if (const auto * t = dc.get("target_layer_ids")) for (const auto & v : t->arr) taps_ids.push_back(int(v.num));
        const auto * mode = dc.get("selector_softcap_mode");
        if (Dd != D || HDd != 128 || df_G != G || df_H % df_G || taps_ids != std::vector<int>{5, 23, 41, 48} || int(num(dc, "block_size")) != mtp_block ||
            int(num(dc, "conv_kernel_size")) != 2 || int(num(dc, "selector_top_k")) != 16 || int(num(dc, "selector_rank")) != 256 || D % df_group ||
            !mode || mode->str != "after_correction" || df_window <= 0)
            throw std::runtime_error("inkling2: unsupported DFlash2 drafter configuration in " + dir);
        const auto idx = safetensors_index_file(std::filesystem::path(dir) / "model.safetensors");
        auto get = [&](const std::string & n) {
            const auto it = idx.find(n);
            if (it == idx.end() || it->second.dtype != "BF16") throw std::runtime_error("inkling2 DFlash2: missing BF16 " + n);
            const auto b = safetensors_read(it->second);
            std::vector<uint16_t> v(b.size() / 2);
            std::memcpy(v.data(), b.data(), b.size());
            return std::make_pair(v, it->second.shape);
        };
        auto f32v = [&](const std::string & n) {
            const auto v = get(n).first;
            std::vector<float> f(v.size());
            for (size_t i = 0; i < v.size(); ++i) f[i] = bf16_f(v[i]);
            return upload(f.data(), f.size() * 4);
        };
        auto bf = [&](const std::string & n) { const auto v = get(n).first; return upload(v.data(), v.size() * 2); };
        auto cat = [&](std::initializer_list<std::string> names) {
            std::vector<uint16_t> all; int rows = 0, cols = 0;
            for (const auto & n : names) {
                const auto [v, sh] = get(n);
                rows += int(sh[0]); cols = int(sh[1]);
                all.insert(all.end(), v.begin(), v.end());
            }
            Mat m; m.rows = rows; m.cols = cols; m.d = upload(all.data(), all.size() * 2);
            if (cols % 16 == 0) {  // drafter weights as INT8 rows: half the bytes per draft; drafts are verified, so output is unchanged
                m.q8 = Dev(size_t(rows) * cols); m.sc = Dev(size_t(rows) * 4);
                K::quantize_rows_i8(m.w(), m.q8.as<int8_t>(), m.sc.as<float>(), rows, cols, s);
                ck(cudaStreamSynchronize(s), "quantize");
                m.d = Dev();
            }
            return m;
        };
        const int ring = window_ring(df_window);
        dfl.resize(size_t(Ld));
        for (int l = 0; l < Ld; ++l) {
            DfLayer & Y = dfl[size_t(l)];
            const std::string p = "layers." + std::to_string(l) + ".";
            Y.in_norm = f32v(p + "input_layernorm.weight"); Y.post_norm = f32v(p + "post_attention_layernorm.weight");
            Y.q_norm = f32v(p + "self_attn.q_norm.weight"); Y.k_norm = f32v(p + "self_attn.k_norm.weight");
            Y.attn_base = f32v(p + "attention_conv.base_kernel"); Y.mlp_base = f32v(p + "mlp_conv.base_kernel");
            Y.qkv = cat({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
            Y.o = cat({p + "self_attn.o_proj.weight"});
            Y.attn_kernel = cat({p + "attention_conv.kernel_projection.weight"}); Y.mlp_kernel = cat({p + "mlp_conv.kernel_projection.weight"});
            Y.gate_up = cat({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}); Y.down = cat({p + "mlp.down_proj.weight"});
            Y.ring = ring;
            const size_t rows = ring ? size_t(ring) : size_t(context);
            Y.kc = Dev(rows * G * HD * 2); Y.vc = Dev(rows * G * HD * 2);
            if (Y.qkv.rows != (df_H + 2 * G) * HD || Y.attn_kernel.rows != 4 * D / df_group || Y.gate_up.rows != 2 * df_I)
                throw std::runtime_error("inkling2 DFlash2: layer shapes");
        }
        df_fc = cat({"fc.weight"});
        if (df_fc.rows != D || df_fc.cols != 4 * D) throw std::runtime_error("inkling2 DFlash2: fc shape");
        df_fc_norm = f32v("hidden_norm.weight"); df_norm = f32v("norm.weight");
        df_sel = cat({"candidate_selector.hidden_projection.weight"});
        {
            const auto [pv, psh] = get("candidate_selector.predecessor_codebook");
            if (psh.size() != 2 || psh[1] != 256) throw std::runtime_error("inkling2 DFlash2: selector codebook shape");
            if (int(psh[0]) < V) throw std::runtime_error("inkling2 DFlash2: selector codebook smaller than the vocabulary");
            df_vocab = V;  // candidates among the real tokens (the codebooks are padded like the embedding)
            df_pred = upload(pv.data(), pv.size() * 2);
            df_succ = bf("candidate_selector.successor_codebook");
        }
        if (df_sel.rows != 256) throw std::runtime_error("inkling2 DFlash2: selector shape");
        {   // the mask row: the target's normalised input row of the mask token (embedding + embed_norm)
            const int mask = int(num(dc, "mask_token_id"));
            Dev mt = upload(&mask, 4), mf(size_t(D) * 4);
            K::embed(embed_ptr, mt.as<int>(), embed_norm.as<float>(), mf.as<float>(), 1, D, eps, s);
            df_mask = Dev(size_t(D) * 2);
            K::to_bf16(mf.as<float>(), df_mask.as<bf16>(), size_t(D), s);
            ck(cudaStreamSynchronize(s), "mask row");
        }
        const int nb = mtp_block, qn = df_H * HD;
        df_x = Dev(size_t(nb) * D * 4); df_nf = Dev(size_t(nb) * D * 4); df_n16 = Dev(size_t(nb) * D * 2);
        df_kern = Dev(size_t(nb) * 4 * D / df_group * 2); df_cin = Dev(size_t(nb) * D * 4); df_cin16 = Dev(size_t(nb) * D * 2);
        df_qkv = Dev(size_t(nb) * dfl[0].qkv.rows * 2); df_att = Dev(size_t(nb) * qn * 2); df_o16 = Dev(size_t(nb) * D * 2); df_of = Dev(size_t(nb) * D * 4);
        df_gu = Dev(size_t(nb) * 2 * df_I * 2); df_act = Dev(size_t(nb) * df_I * 2); df_hid = Dev(size_t(nb) * D * 2);
        df_logits = Dev(size_t(nb) * V * 4); df_ids = Dev(size_t(nb) * 16 * 4); df_vals = Dev(size_t(nb) * 16 * 4); df_gate = Dev(size_t(nb) * 256 * 4);
        df_tok = Dev(size_t(nb) * 4); df_top = Dev(size_t(K::dflash_top16_floats(nb, df_vocab)) * 4);
        df_part = Dev(size_t(K::mtp_attention_floats(df_H, nb, kMtpSplits)) * 4);
        const int cm = std::max(chunk, nb);
        df_ctxf = Dev(size_t(cm) * D * 4); df_ctx16 = Dev(size_t(cm) * D * 2); df_kv = Dev(size_t(cm) * 2 * G * HD * 2);
        taps = Dev(size_t(cm) * 4 * D * 2);
        d_tok = Dev(size_t(nb) * 4); d_x0 = Dev(size_t(D) * 4);
        dflash = true;
        {   // warm-up: one context row and one block on zero taps, so every drafter kernel is loaded now (a lazily loaded module
            // needs device memory, which the expert plan may have taken by the first real draft); the rows are overwritten
            // by the first committed tokens
            ck(cudaMemsetAsync(taps.p, 0, size_t(4) * D * 2, s), "warm-up taps");
            dflash_context(0, 1);
            df_pos = 1;
            std::vector<int> tok;
            dflash_draft(0, tok, mtp_block - 1);
            df_pos = -1;
            for (auto & Y : dfl) {
                ck(cudaMemsetAsync(Y.kc.p, 0, size_t(Y.ring ? Y.ring : context) * G * HD * 2, s), "warm-up kv");
                ck(cudaMemsetAsync(Y.vc.p, 0, size_t(Y.ring ? Y.ring : context) * G * HD * 2, s), "warm-up kv");
            }
            ck(cudaStreamSynchronize(s), "DFlash2 warm-up");
        }
        std::fprintf(stderr, "inkling2: DFlash2 drafter %s: %d layers, %d heads, window %d, selector over %d tokens\n", dir.c_str(), Ld, df_H, df_window, df_vocab);
        mtp_common();
    }
    // the reference check (tools/inkling2/dflash2_reference.py): context taps = the normalised input rows of `tokens`
    // (the same row at all four taps) at 0 .., one block after `anchor`; the final normed rows, top-16 ids, the path
    void dflash_trace(const std::vector<int> & tokens, int anchor, std::vector<float> & hidden, std::vector<int> & ids, std::vector<int> & path) {
        if (!dflash) throw std::runtime_error("inkling2: draft_trace needs a DFlash2 drafter");
        const int m = int(tokens.size()), nb = mtp_block;
        if (m < 1 || m > std::max(chunk, nb)) throw std::runtime_error("inkling2: draft_trace context");
        Dev t = upload(tokens.data(), size_t(m) * 4), rf(size_t(m) * D * 4), rb(size_t(m) * D * 2);
        K::embed(embed_ptr, t.as<int>(), embed_norm.as<float>(), rf.as<float>(), m, D, eps, s);
        K::to_bf16(rf.as<float>(), rb.as<bf16>(), size_t(m) * D, s);
        for (int r = 0; r < m; ++r)
            for (int k = 0; k < 4; ++k)
                ck(cudaMemcpyAsync(taps.as<bf16>() + (size_t(r) * 4 + k) * D, rb.as<bf16>() + size_t(r) * D, size_t(D) * 2, cudaMemcpyDeviceToDevice, s), "taps");
        dflash_context(0, m);
        df_pos = m;
        dflash_draft(anchor, path, nb - 1);
        df_pos = -1;
        std::vector<uint16_t> h(size_t(nb) * D);
        ids.resize(size_t(nb) * 16);
        ck(cudaMemcpy(h.data(), df_hid.p, h.size() * 2, cudaMemcpyDeviceToHost), "trace");
        ck(cudaMemcpy(ids.data(), df_ids.p, ids.size() * 4, cudaMemcpyDeviceToHost), "trace");
        hidden.resize(h.size());
        for (size_t i = 0; i < h.size(); ++i) hidden[i] = bf16_f(h[i]);
    }
    int window_ring(int w) const { const int r = w + std::max(chunk, K::kSmallN); return r < context ? r : 0; }
    // draft context of the m committed tokens at pos0 .. (their taps): fc(taps) -> hidden_norm -> every layer's K / V
    void dflash_context(int pos0, int m) {
        const int kn = G * HD, qn = df_H * HD;
        K::GemvSeg sg = seg(df_fc, taps.as<bf16>(), df_ctx16.p);
        project(&sg, 1, m);
        K::to_f32(df_ctx16.as<bf16>(), df_ctxf.as<float>(), size_t(m) * D, s);
        K::rms_norm(df_ctxf.as<float>(), df_fc_norm.as<float>(), nullptr, df_ctx16.as<bf16>(), m, D, df_eps, s);
        for (auto & Y : dfl) {
            K::GemvSeg kv = seg_rows(Y.qkv, qn, 2 * kn, df_ctx16.as<bf16>(), df_kv.p);
            project(&kv, 1, m);
            K::mtp_head_norm_rope(df_kv.as<bf16>(), 2 * kn, m, G, Y.k_norm.as<float>(), pos0, df_theta, df_eps, s);
            K::mtp_store_kv(df_kv.as<bf16>(), df_kv.as<bf16>() + kn, 2 * kn, Y.kc.as<bf16>(), Y.vc.as<bf16>(), m, G, pos0, Y.ring, s);
        }
    }
    void dflash_draft(int anchor, std::vector<int> & out_tokens, int steps, std::vector<float> * conf = nullptr) {
        const int nb = mtp_block, P = df_pos >= 0 ? df_pos : len, qn = df_H * HD, kn = G * HD;
        ck(cudaMemcpyAsync(d_tok.p, &anchor, 4, cudaMemcpyHostToDevice, s), "anchor");
        K::embed(embed_ptr, d_tok.as<int>(), embed_norm.as<float>(), d_x0.as<float>(), 1, D, eps, s);
        K::to_bf16(d_x0.as<float>(), df_n16.as<bf16>(), size_t(D), s);
        K::to_f32(df_n16.as<bf16>(), d_x0.as<float>(), size_t(D), s);
        K::mtp_block_init(d_x0.as<float>(), df_mask.as<bf16>(), df_x.as<float>(), nb, D, s);
        auto gemv = [&](const Mat & W, const bf16 * x, void * y, bool f32) { K::GemvSeg sg = seg(W, x, y, f32); K::gemv(&sg, 1, nb, s); };
        for (auto & Y : dfl) {
            // attention: conv(side 0) of the normed rows in, conv(side 1) of its output + the residual out
            K::rms_norm(df_x.as<float>(), Y.in_norm.as<float>(), nullptr, df_n16.as<bf16>(), nb, D, df_eps, s);
            K::to_f32(df_n16.as<bf16>(), df_nf.as<float>(), size_t(nb) * D, s);
            gemv(Y.attn_kernel, df_n16.as<bf16>(), df_kern.p, false);
            K::dflash_conv(df_nf.as<float>(), df_kern.as<bf16>(), Y.attn_base.as<float>(), nb, D, df_group, 0, nullptr, df_cin.as<float>(), s);
            K::to_bf16(df_cin.as<float>(), df_cin16.as<bf16>(), size_t(nb) * D, s);
            gemv(Y.qkv, df_cin16.as<bf16>(), df_qkv.p, false);
            const int ld = Y.qkv.rows;
            bf16 * q = df_qkv.as<bf16>(), * k = q + qn, * v = k + kn;
            K::mtp_head_norm_rope(q, ld, nb, df_H, Y.q_norm.as<float>(), P, df_theta, df_eps, s);
            K::mtp_head_norm_rope(k, ld, nb, G, Y.k_norm.as<float>(), P, df_theta, df_eps, s);
            K::mtp_attention(q, k, v, ld, ld, Y.kc.as<bf16>(), Y.vc.as<bf16>(), Y.ring, P, nb, df_H, G, df_window, kMtpSplits, df_part.as<float>(),
                             df_att.as<bf16>(), s);
            gemv(Y.o, df_att.as<bf16>(), df_o16.p, false);
            K::to_f32(df_o16.as<bf16>(), df_of.as<float>(), size_t(nb) * D, s);
            K::dflash_conv(df_of.as<float>(), df_kern.as<bf16>(), Y.attn_base.as<float>(), nb, D, df_group, 1, df_x.as<float>(), df_x.as<float>(), s);
            // MLP, the same wrapping
            K::rms_norm(df_x.as<float>(), Y.post_norm.as<float>(), nullptr, df_n16.as<bf16>(), nb, D, df_eps, s);
            K::to_f32(df_n16.as<bf16>(), df_nf.as<float>(), size_t(nb) * D, s);
            gemv(Y.mlp_kernel, df_n16.as<bf16>(), df_kern.p, false);
            K::dflash_conv(df_nf.as<float>(), df_kern.as<bf16>(), Y.mlp_base.as<float>(), nb, D, df_group, 0, nullptr, df_cin.as<float>(), s);
            K::to_bf16(df_cin.as<float>(), df_cin16.as<bf16>(), size_t(nb) * D, s);
            gemv(Y.gate_up, df_cin16.as<bf16>(), df_gu.p, false);
            K::dflash_silu_mul(df_gu.as<bf16>(), df_act.as<bf16>(), nb, df_I, s);
            gemv(Y.down, df_act.as<bf16>(), df_o16.p, false);
            K::to_f32(df_o16.as<bf16>(), df_of.as<float>(), size_t(nb) * D, s);
            K::dflash_conv(df_of.as<float>(), df_kern.as<bf16>(), Y.mlp_base.as<float>(), nb, D, df_group, 1, df_x.as<float>(), df_x.as<float>(), s);
        }
        K::rms_norm(df_x.as<float>(), df_norm.as<float>(), nullptr, df_hid.as<bf16>(), nb, D, df_eps, s);
        {   // candidates: the target head over the real tokens (its INT8 copy when there is one)
            K::GemvSeg sg = seg(out, df_hid.as<bf16>(), df_logits.p, true);
            if (hq8.p) sg = K::GemvSeg{reinterpret_cast<const bf16 *>(hq8.p), df_hid.as<bf16>(), df_logits.p, V, D, D, V, 1, hsc.as<float>()};
            K::gemv(&sg, 1, nb, s);
        }
        K::dflash_top16(df_logits.as<float>(), nb, df_vocab, V, df_ids.as<int>(), df_vals.as<float>(), df_top.as<float>(), s);
        gemv(df_sel, df_hid.as<bf16>(), df_gate.p, false);
        K::dflash_select(df_ids.as<int>(), df_vals.as<float>(), df_gate.as<bf16>(), df_succ.as<bf16>(), df_pred.as<bf16>(), d_tok.as<int>(), steps,
                         df_vocab, df_tok.as<int>(), s);
        out_tokens.resize(size_t(steps));
        // a drafter kernel that did not launch (e.g. its module could not load: no device memory left) must fail loudly,
        // not leave stale candidates that verify rejects forever
        ck(cudaPeekAtLastError(), "DFlash2 draft launch");
        ck(cudaMemcpyAsync(out_tokens.data(), df_tok.p, size_t(steps) * 4, cudaMemcpyDeviceToHost, s), "draft");
        std::vector<int> cid; std::vector<float> cv;
        if (conf) {  // rows 1 .. steps of the top-16 candidates (step i walks row i)
            cid.resize(size_t(nb) * 16); cv.resize(size_t(nb) * 16);
            ck(cudaMemcpyAsync(cid.data(), df_ids.p, cid.size() * 4, cudaMemcpyDeviceToHost, s), "draft ids");
            ck(cudaMemcpyAsync(cv.data(), df_vals.p, cv.size() * 4, cudaMemcpyDeviceToHost, s), "draft vals");
        }
        ck(cudaStreamSynchronize(s), "draft");
        ck(cudaGetLastError(), "DFlash2 draft");
        if (conf) {
            conf->assign(size_t(steps), 0.0f);
            for (int i = 0; i < steps; ++i) {
                const int * ids = cid.data() + size_t(i + 1) * 16; const float * v = cv.data() + size_t(i + 1) * 16;
                double z = 0, w = 0;
                for (int c = 0; c < 16; ++c) { const double e = std::exp(double(v[c]) - double(v[0])); z += e; if (ids[c] == out_tokens[size_t(i)]) w = e; }
                (*conf)[size_t(i)] = float(w / z);  // 0 when the walk left the candidates (its token is then the vocab stop)
            }
        }
    }
    // draft context K / V of the m committed tokens at positions pos0.. (their taps are in `taps`), and f_rnn of the last
    void mtp_context(int pos0, int m) {
        if (dflash) { dflash_context(pos0, m); return; }
        for (int i = 0; i < 3; ++i) {
            MtpLayer & Y = ml[size_t(i)];
            K::mtp_mix_norm(taps.as<bf16>(), 4, m_alpha.as<float>() + i * 4, Y.dflare.as<float>(), d_cx.as<bf16>(), m, D, mtp_eps, s);
            K::GemvSeg sg = seg(Y.kvc, d_cx.as<bf16>(), d_ckv.p);
            project(&sg, 1, m);
            const int ld = Y.kvc.rows;
            K::mtp_head_norm_rope(d_ckv.as<bf16>(), ld, m, G, Y.knc.as<float>(), pos0, rope_theta, mtp_eps, s);
            K::mtp_store_kv(d_ckv.as<bf16>(), d_ckv.as<bf16>() + G * HD, ld, Y.kc.as<bf16>(), Y.vc.as<bf16>(), m, G, pos0, Y.ring, s);
        }
        K::mtp_mix_norm(taps.as<bf16>() + size_t(m - 1) * 4 * D, 4, m_alpha.as<float>() + 12, m_fnorm.as<float>(), d_frnn.as<bf16>(), 1, D, mtp_eps, s);
    }
    // the block at position len: [anchor, mask x 7] -> 7 greedy draft tokens
    void draft(int anchor, std::vector<int> & out_tokens, int steps) {
        if (!mtp_on) throw std::runtime_error("inkling2: draft needs enable_mtp()");
        if (dflash) { dflash_draft(anchor, out_tokens, std::max(1, std::min(steps, mtp_block - 1))); return; }  // -md DFlash2 folder
        const int nb = mtp_block, P = len;
        steps = std::max(1, std::min(steps, nb - 1));
        ck(cudaMemcpyAsync(d_tok.p, &anchor, 4, cudaMemcpyHostToDevice, s), "anchor");
        K::embed(embed_ptr, d_tok.as<int>(), embed_norm.as<float>(), d_x0.as<float>(), 1, D, eps, s);
        K::mtp_block_init(d_x0.as<float>(), m_mask.as<bf16>(), d_h.as<float>(), nb, D, s);
        for (int i = 0; i < 3; ++i) {
            MtpLayer & Y = ml[size_t(i)];
            const int ldq = Y.qkv.rows;
            K::rms_norm(d_h.as<float>(), Y.in_norm.as<float>(), nullptr, d_xb.as<bf16>(), nb, D, mtp_eps, s);
            K::GemvSeg sg[2];
            sg[0] = seg(Y.qkv, d_xb.as<bf16>(), d_qkv.p);
            K::gemv(sg, 1, nb, s);
            bf16 * q = d_qkv.as<bf16>(), * kb = q + H * HD, * vb = kb + G * HD;
            K::mtp_head_norm_rope(q, ldq, nb, H, Y.qn.as<float>(), P, rope_theta, mtp_eps, s);
            K::mtp_head_norm_rope(kb, ldq, nb, G, Y.kn.as<float>(), P, rope_theta, mtp_eps, s);
            K::mtp_attention(q, kb, vb, ldq, ldq, Y.kc.as<bf16>(), Y.vc.as<bf16>(), Y.ring, P, nb, H, G, Y.local ? window : 0, kMtpSplits,
                             d_part.as<float>(), d_att.as<bf16>(), s);
            sg[0] = seg(Y.o, d_att.as<bf16>(), d_ao.p);
            K::gemv(sg, 1, nb, s);
            K::mtp_add(d_h.as<float>(), d_ao.as<bf16>(), size_t(nb) * D, s);
            K::rms_norm(d_h.as<float>(), Y.post_norm.as<float>(), nullptr, d_xb.as<bf16>(), nb, D, mtp_eps, s);
            const int ffn = Y.gu.rows / 2;
            sg[0] = K::GemvSeg{Y.gu.w(), d_xb.as<bf16>(), d_g.p, ffn, D, D, ffn, 0};
            sg[1] = K::GemvSeg{Y.gu.w() + size_t(ffn) * D, d_xb.as<bf16>(), d_u.p, ffn, D, D, ffn, 0};
            K::gemv(sg, 2, nb, s);
            K::mtp_silu_mul(d_g.as<bf16>(), d_u.as<bf16>(), d_act.as<bf16>(), size_t(nb) * ffn, s);
            sg[0] = seg(Y.down, d_act.as<bf16>(), d_mo.p);
            K::gemv(sg, 1, nb, s);
            K::mtp_add(d_h.as<float>(), d_mo.as<bf16>(), size_t(nb) * D, s);
        }
        K::rms_norm(d_h.as<float>(), m_norm.as<float>(), nullptr, d_hid.as<bf16>(), nb, D, mtp_eps, s);
        {   // base logits of the 7 masked positions through the target LM head (its INT8 copy when there is one: the drafts
            // only propose tokens, every one is verified with the BF16 head)
            K::GemvSeg sg = seg(out, d_hid.as<bf16>() + D, d_base.p, true);
            if (hq8.p) sg = K::GemvSeg{reinterpret_cast<const bf16 *>(hq8.p), d_hid.as<bf16>() + D, d_base.p, V, D, D, V, 1, hsc.as<float>()};
            K::gemv(&sg, 1, steps, s);
        }
        ck(cudaMemsetAsync(d_state.p, 0, size_t(mrank) * 4, s), "state");
        for (int i = 0; i < steps; ++i) {
            K::mtp_rnn_inputs(d_state.as<float>(), d_tok.as<int>() + i, m_w1.as<bf16>(), d_hid.as<bf16>() + size_t(i + 1) * D, d_frnn.as<bf16>(), d_z.as<bf16>(),
                              mrank, D, s);
            K::GemvSeg sg = seg(m_rnn, d_z.as<bf16>(), d_proj.p, true);
            K::gemv(&sg, 1, 1, s);
            K::mtp_rnn_gates(d_proj.as<float>(), m_rnn_b.as<bf16>(), d_state.as<float>(), d_lat.as<bf16>(), mrank, s);
            K::mtp_step_argmax(d_base.as<float>() + size_t(i) * V, m_fold.as<bf16>(), m_fold_b.as<float>(), d_lat.as<bf16>(), V, mrank, fcap,
                               d_step.as<float>(), d_tok.as<int>() + i + 1, s);
        }
        out_tokens.resize(size_t(steps));
        ck(cudaMemcpyAsync(out_tokens.data(), d_tok.as<int>() + 1, size_t(steps) * 4, cudaMemcpyDeviceToHost, s), "draft");
        ck(cudaStreamSynchronize(s), "draft");
    }
    // TQLLM_DEBUG=inkling2_dump:FILE: the first forward() writes x0 and every visit's output ([n][D] F32 each, in visit order)
    std::FILE * dump = nullptr;
    void dump_rows(const float * x, int n) {
        if (!dump) return;
        std::vector<float> h(size_t(n) * D);
        ck(cudaMemcpyAsync(h.data(), x, h.size() * 4, cudaMemcpyDeviceToHost, s), "dump");
        ck(cudaStreamSynchronize(s), "dump");
        std::fwrite(h.data(), 4, h.size(), dump);
    }
    void forward(const int * tokens, int n, int want, std::vector<float> & logits_out) {
        // partial residency: re-plan the resident experts from the recent routing every kReplanEvery decode steps
        constexpr int kReplanEvery = 32;
        if (n == 1 && resident_share < Ls[0].eg.E && ++decode_steps >= kReplanEvery) { decode_steps = 0; replan_experts(); }
        if (len == 0 && !dumped) { if (const char * d = debug_cstr("inkling2_dump")) dump = std::fopen(d, "wb"); dumped = true; }
        if (n <= 0 || n > chunk || want > n) throw std::runtime_error("inkling2: chunk size");
        grow_logits(want);
        if (len + n > context) throw std::runtime_error("inkling2: context full");
        *h_len = len;
        logits_out.resize(size_t(want) * V);
        if (verifying && (n > K::kSmallN || want != n)) throw std::runtime_error("inkling2: a verify block is n <= 8 tokens with all logits");
        if (n > K::kSmallN) {
            step(tokens, n, want, logits_out.data());
        } else {
            std::memcpy(h_tok, tokens, size_t(n) * 4);
            // the decode step (and a verify block) as one CUDA graph: every launch reads positions from dlen and host data
            // through pinned buffers
            if ((n == 1 || verifying) && want == n && !route_log && !dump && !hf_attention) {
                cudaGraphExec_t & g = graphs[verifying ? n : 0];
                if (!g) g = instantiate_graph(capture_graph(s, [&] { step(h_tok, n, want, h_logits); }));
                ck(cudaGraphLaunch(g, s), "graph launch");
            } else {
                step(h_tok, n, want, h_logits);
            }
            ck(cudaStreamSynchronize(s), "forward");
            std::memcpy(logits_out.data(), h_logits, logits_out.size() * 4);
        }
        ck(cudaStreamSynchronize(s), "forward");
        if (mtp_on && !verifying) mtp_context(len, n);  // every committed token becomes draft context
        len += n;
    }
    // verify block of n tokens (as verify()), greedy tokens of its n rows into next[n], no logits
    cudaGraphExec_t vgraphs[K::kSmallN + 1] = {};
    void verify_greedy(const int * tokens, int n, int * next) {
        if (!mtp_on) throw std::runtime_error("inkling2: verify needs enable_mtp()");
        if (n < 1 || n > K::kSmallN) throw std::runtime_error("inkling2: a verify block is n <= 8 tokens");
        if (len + n > context) throw std::runtime_error("inkling2: context full");
        grow_logits(n);
        verifying = true; verify_n = n; greedy_step = true;
        *h_len = len;
        std::memcpy(h_tok, tokens, size_t(n) * 4);
        try {
            if (!route_log && !dump && !hf_attention) {
                cudaGraphExec_t & g = vgraphs[n];
                if (!g) g = instantiate_graph(capture_graph(s, [&] { step(h_tok, n, n, nullptr); }));
                ck(cudaGraphLaunch(g, s), "graph launch");
            } else {
                step(h_tok, n, n, nullptr);
            }
        } catch (...) { verifying = false; greedy_step = false; throw; }
        verifying = false; greedy_step = false;
        ck(cudaStreamSynchronize(s), "verify_greedy");
        std::memcpy(next, h_next, size_t(n) * 4);
        len += n;
    }
    // one decode token at length(): the greedy next token (argmax of the softcapped logits, lowest index on ties), no logits
    int forward_greedy(int token) {
        constexpr int kReplanEvery = 32;
        if (resident_share < Ls[0].eg.E && ++decode_steps >= kReplanEvery) { decode_steps = 0; replan_experts(); }
        if (len == 0 && !dumped) { if (const char * d = debug_cstr("inkling2_dump")) dump = std::fopen(d, "wb"); dumped = true; }
        if (len + 1 > context) throw std::runtime_error("inkling2: context full");
        if (verifying) throw std::runtime_error("inkling2: forward_greedy inside a verify block");
        grow_logits(1);
        *h_len = len; *h_tok = token;
        greedy_step = true;
        try {
            if (!route_log && !dump && !hf_attention) {
                cudaGraphExec_t & g = graphs[K::kSmallN + 1];
                if (!g) g = instantiate_graph(capture_graph(s, [&] { step(h_tok, 1, 1, nullptr); }));
                ck(cudaGraphLaunch(g, s), "graph launch");
            } else {
                step(h_tok, 1, 1, nullptr);
            }
        } catch (...) { greedy_step = false; throw; }
        greedy_step = false;
        ck(cudaStreamSynchronize(s), "forward_greedy");
        const int next = *h_next;
        if (mtp_on) mtp_context(len, 1);
        len += 1;
        return next;
    }
    // spec decode: forward() of a verify block (states before it kept), then commit(keep) keeps its first keep tokens
    bool verifying = false;
    int verify_n = 0;
    __nv_bfloat16 * hist(Slot & S, int c) { return verifying ? S.hist[c].as<bf16>() : nullptr; }
    void verify(const int * tokens, int n, std::vector<float> & lg) {
        if (!mtp_on) throw std::runtime_error("inkling2: verify needs enable_mtp()");
        verifying = true; verify_n = n;
        try { forward(tokens, n, n, lg); } catch (...) { verifying = false; throw; }
        verifying = false;
    }
    void commit(int keep) {
        if (verify_n <= 0 || keep < 1 || keep > verify_n) throw std::runtime_error("inkling2: commit without a verify block");
        const int base = len - verify_n;
        if (keep < verify_n)
            for (auto & S : slots) {
                const int C[4] = {G * HD, G * HD, D, D};
                for (int c = 0; c < 4; ++c) K::conv_rollback(S.snap[c].as<float>(), S.hist[c].as<bf16>(), S.conv[c].as<float>(), keep, C[c], s);
            }
        len = base + keep;
        mtp_context(base, keep);
        ck(cudaStreamSynchronize(s), "commit");
        verify_n = 0;
    }
    // embed .. logits for n tokens at position *h_len: tokens / logits_dst are host memory (pinned on the few-token path)
    void step(const int * tokens, int n, int want, float * logits_dst) {
        const size_t nD = size_t(n) * D, bytes = nD * 4;
        ck(cudaMemcpyAsync(dlen.p, h_len, 4, cudaMemcpyHostToDevice, s), "len");
        if (verifying)  // the conv states before the block, for commit()'s rollback
            for (auto & S : slots) {
                const int C[4] = {G * HD, G * HD, D, D};
                for (int c = 0; c < 4; ++c) ck(cudaMemcpyAsync(S.snap[c].p, S.conv[c].p, size_t(C[c]) * 12, cudaMemcpyDeviceToDevice, s), "snap");
            }
        ck(cudaMemcpyAsync(ids.p, tokens, size_t(n) * 4, cudaMemcpyHostToDevice, s), "ids");
        K::embed(embed_ptr, ids.as<int>(), embed_norm.as<float>(), x0.as<float>(), n, D, eps, s);
        float * cur = hA.as<float>(), * nxt = hB.as<float>();
        ck(cudaMemcpyAsync(cur, x0.p, bytes, cudaMemcpyDeviceToDevice, s), "copy");
        dump_rows(x0.as<float>(), n);
        const float * X0 = x0.as<float>(), * Pp = P.as<float>();
        std::vector<const float *> B;
        for (int l = 0; l < core0; ++l) {
            if (mtp_on && l == 5) K::mtp_tap(cur, n, D, taps.as<bf16>(), 0, 4, s);  // MTP taps: the partial entering visits 5, 23, 41
            visit(l, 0, cur, B, nullptr, nxt, n);
            dump_rows(nxt, n);
            if (l == 0) B = {X0};
            std::swap(cur, nxt);
        }
        ck(cudaMemcpyAsync(P.p, cur, bytes, cudaMemcpyDeviceToDevice, s), "copy");
        const int mid = 12;  // the core's inner block boundary (core0 + 6)
        for (int p = 0; p < passes; ++p) {
            float * b1 = B1[p % 2].as<float>(), * o17 = O17[p % 2].as<float>();
            ck(cudaMemcpyAsync(cur, Pp, bytes, cudaMemcpyDeviceToDevice, s), "copy");
            B = {X0};
            for (int l = core0; l < core1; ++l) {
                const float * msg = nullptr;
                if (p > 0) {
                    K::States st{}; st.count = 2;
                    st.s[0] = B1[(p - 1) % 2].as<float>(); st.s[1] = O17[(p - 1) % 2].as<float>();
                    K::attn_res(st, msg_q.as<float>() + size_t(l - core0) * 4 * D, 4, msgs.as<float>(), n, D, arn.as<float>(), s);
                    msg = msgs.as<float>();
                }
                if (l == mid) ck(cudaMemcpyAsync(b1, cur, bytes, cudaMemcpyDeviceToDevice, s), "copy");
                if (mtp_on && ((p == 1 && l == 11) || (p == 2 && l == 17))) K::mtp_tap(cur, n, D, taps.as<bf16>(), p, 4, s);
                visit(l, p, cur, B, msg, nxt, n);
                dump_rows(nxt, n);
                if (l == core0) B = {X0, Pp};
                if (l == mid) B = {X0, Pp, b1};
                std::swap(cur, nxt);
            }
            ck(cudaMemcpyAsync(o17, cur, bytes, cudaMemcpyDeviceToDevice, s), "copy");
        }
        const float * last17 = O17[(passes - 1) % 2].as<float>();
        for (int l = core1; l < L; ++l) {
            visit(l, 0, cur, B, nullptr, nxt, n);
            dump_rows(nxt, n);
            if (l == core1) B.push_back(last17);
            std::swap(cur, nxt);
        }
        std::vector<const float *> Bh = B; Bh.push_back(cur);
        K::States st{}; st.count = int(Bh.size());
        for (size_t i = 0; i < Bh.size(); ++i) st.s[i] = Bh[i];
        K::attn_res(st, out_q.as<float>(), 1, m.as<float>(), n, D, arn.as<float>(), s);
        K::rms_norm(m.as<float>(), out_norm.as<float>(), tmp.as<float>(), fin.as<bf16>(), n, D, eps, s);
        if (mtp_on) K::mtp_tap(tmp.as<float>(), n, D, taps.as<bf16>(), 3, 4, s);  // and the final normed state (visit 48)
        dump_rows(tmp.as<float>(), n);
        if (dump) { std::fclose(dump); dump = nullptr; }
        const bf16 * last = fin.as<bf16>() + size_t(n - want) * D;
        if (greedy_step) {  // the greedy token of each of the `want` rows, no logits
            if (hq8.p) {
                const K::HeadBound hb{out.w(), hq8.as<int8_t>(), hsc.as<float>(), hbound.as<float>(), V, D, V};
                K::head_greedy(hb, last, want, fcap, happrox.as<float>(), hcells.as<unsigned long long>(), dtok.as<int>(), nullptr, s);
            } else {
                K::GemvSeg sg = seg(out, last, logits16.p);
                project(&sg, 1, want, true);
                K::to_f32(logits16.as<bf16>(), logits.as<float>(), size_t(want) * V, s);
                K::softcap(logits.as<float>(), size_t(want) * V, fcap, s);
                for (int i = 0; i < want; ++i)
                    K::argmax_greedy(logits.as<float>() + size_t(i) * V, size_t(V), hcells.as<unsigned long long>(), dtok.as<int>() + i, nullptr, s);
            }
            ck(cudaMemcpyAsync(h_next, dtok.p, size_t(want) * 4, cudaMemcpyDeviceToHost, s), "token");
            return;
        }
        K::GemvSeg sg = seg(out, last, logits16.p);
        project(&sg, 1, want, true);
        K::to_f32(logits16.as<bf16>(), logits.as<float>(), size_t(want) * V, s);
        K::softcap(logits.as<float>(), size_t(want) * V, fcap, s);
        ck(cudaMemcpyAsync(logits_dst, logits.p, size_t(want) * V * 4, cudaMemcpyDeviceToHost, s), "logits");
    }

    // one step per visit from the reference's own states: hs = [x0, the outputs of visits 0 .. V - 2, final] ([n][D] each);
    // each visit v runs on its reference input (the prelude output at a pass start), the reference blocks and messages,
    // and its output is compared to hs[v + 1]: the error of that visit alone, nothing carried over
    bool quiet = false;
    void visits(const std::vector<float> & hs, int n) {
        const size_t nD = size_t(n) * D;
        const int nh = int(hs.size() / nD);
        Dev dh(hs.size() * 4);
        ck(cudaMemcpy(dh.p, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice), "hs");
        auto H_ = [&](int i) { return dh.as<float>() + size_t(i) * nD; };
        reset();
        std::vector<float> out(nD);
        const int nc = core1 - core0;
        for (int v = 0; v + 1 < nh - 1 && v < core0 + passes * nc + (L - core1); ++v) {
            int l, p = 0;
            if (v < core0) l = v; else if (v < core0 + passes * nc) { p = (v - core0) / nc; l = core0 + (v - core0) % nc; } else l = core1 + (v - core0 - passes * nc);
            const float * X0 = H_(0), * P_ = H_(core0);
            const float * hin = (l == core0) ? P_ : H_(v);
            std::vector<const float *> B;
            if (l > 0 && l < core0) B = {X0};
            else if (l == core0) B = {X0};
            else if (l > core0 && l <= 12 && l < core1) B = {X0, P_};
            else if (l > 12 && l < core1) B = {X0, P_, H_(12 + nc * p)};
            else if (l == core1) B = {X0, P_, H_(12 + nc * (passes - 1))};
            else if (l > core1) B = {X0, P_, H_(12 + nc * (passes - 1)), H_(core0 + passes * nc)};
            const float * msg = nullptr;
            if (p > 0 && l >= core0 && l < core1) {
                K::States st{}; st.count = 2;
                st.s[0] = H_(12 + nc * (p - 1)); st.s[1] = H_(core0 + nc * p);
                K::attn_res(st, msg_q.as<float>() + size_t(l - core0) * 4 * D, 4, msgs.as<float>(), n, D, arn.as<float>(), s);
                msg = msgs.as<float>();
            }
            len = 0;
            *h_len = 0;
            ck(cudaMemcpyAsync(dlen.p, h_len, 4, cudaMemcpyHostToDevice, s), "len");
            visit(l, p, hin, B, msg, hB.as<float>(), n);
            ck(cudaMemcpyAsync(out.data(), hB.p, nD * 4, cudaMemcpyDeviceToHost, s), "out");
            ck(cudaStreamSynchronize(s), "visit");
            const float * ref = hs.data() + size_t(v + 1) * nD;
            double e = 0, r = 0, worst = 0; int wt = 0;
            for (int t = 0; t < n; ++t) {
                double et = 0, rt = 0;
                for (int c = 0; c < D; ++c) { const double d = out[size_t(t) * D + c] - ref[size_t(t) * D + c]; et += d * d; rt += double(ref[size_t(t) * D + c]) * ref[size_t(t) * D + c]; }
                e += et; r += rt;
                if (std::sqrt(et / (rt + 1e-30)) > worst) { worst = std::sqrt(et / (rt + 1e-30)); wt = t; }
            }
            if (!quiet) std::printf("visit %2d (layer %2d pass %d): rel err %.4f, worst token %2d %.4f\n", v, l, p, std::sqrt(e / r), wt, worst);
        }
        len = 0;
        reset();
    }

    // conversation reuse: the conv states, the position and the window of K / V rows before it in every sliding-window
    // ring (the rows later tokens may overwrite; full-attention rows sit at their position and are only appended to).
    // restore() puts all of it back, however many tokens came since. The buffers are allocated on the first snapshot.
    std::vector<Dev> keep_conv, keep_kv;
    int keep_len = -1;
    // the window's rows [p0, len) of a ring cache as at most two contiguous slot ranges: (first slot, rows) pairs
    void ring_ranges(int ring, int p0, int n, int (&r)[2][2]) const {
        const int s0 = p0 % ring, a = std::min(n, ring - s0);
        r[0][0] = s0; r[0][1] = a; r[1][0] = 0; r[1][1] = n - a;
    }
    void copy_window(bool save) {
        const size_t rb = size_t(G) * HD * 2;  // bytes per cache row
        size_t k = 0;
        auto one = [&](Dev & kc, Dev & vc, int ring, int win) {  // the last `win` rows of a ring cache
            if (!ring) return;
            const int p0 = std::max(0, keep_len - win), n = keep_len - p0;
            if (save && keep_kv.size() < k + 2) { keep_kv.emplace_back(size_t(win) * rb); keep_kv.emplace_back(size_t(win) * rb); }
            int r[2][2];
            ring_ranges(ring, p0, n, r);
            size_t off = 0;
            for (int j = 0; j < 2; ++j) {
                if (!r[j][1]) continue;
                const size_t bytes = size_t(r[j][1]) * rb, at = size_t(r[j][0]) * rb;
                char * kk = keep_kv[k].as<char>() + off, * kv = keep_kv[k + 1].as<char>() + off;
                if (save) {
                    ck(cudaMemcpyAsync(kk, kc.as<char>() + at, bytes, cudaMemcpyDeviceToDevice, s), "keep kv");
                    ck(cudaMemcpyAsync(kv, vc.as<char>() + at, bytes, cudaMemcpyDeviceToDevice, s), "keep kv");
                } else {
                    ck(cudaMemcpyAsync(kc.as<char>() + at, kk, bytes, cudaMemcpyDeviceToDevice, s), "restore kv");
                    ck(cudaMemcpyAsync(vc.as<char>() + at, kv, bytes, cudaMemcpyDeviceToDevice, s), "restore kv");
                }
                off += bytes;
            }
            k += 2;
        };
        for (auto & S : slots) one(S.kc, S.vc, S.ring, window);
        for (auto & Y : ml) one(Y.kc, Y.vc, Y.ring, window);
        for (auto & Y : dfl) one(Y.kc, Y.vc, Y.ring, df_window);
    }
    void snapshot() {
        if (verify_n) throw std::runtime_error("inkling2: snapshot inside a verify block");
        const int C[4] = {G * HD, G * HD, D, D};
        if (keep_conv.empty())
            for (size_t i = 0; i < slots.size(); ++i) for (int c = 0; c < 4; ++c) keep_conv.emplace_back(size_t(C[c]) * 12);
        size_t k = 0;
        for (auto & S : slots) for (int c = 0; c < 4; ++c, ++k) ck(cudaMemcpyAsync(keep_conv[k].p, S.conv[c].p, size_t(C[c]) * 12, cudaMemcpyDeviceToDevice, s), "keep");
        keep_len = len;
        copy_window(true);
        ck(cudaStreamSynchronize(s), "keep");
    }
    bool restore() {
        if (keep_len < 0 || len < keep_len || verify_n) return false;
        const int C[4] = {G * HD, G * HD, D, D};
        size_t k = 0;
        for (auto & S : slots) for (int c = 0; c < 4; ++c, ++k) ck(cudaMemcpyAsync(S.conv[c].p, keep_conv[k].p, size_t(C[c]) * 12, cudaMemcpyDeviceToDevice, s), "restore");
        copy_window(false);
        ck(cudaStreamSynchronize(s), "restore");
        len = keep_len;
        return true;
    }

    void reset() {
        const int C[4] = {G * HD, G * HD, D, D};
        for (auto & sl : slots)
            for (int c = 0; c < 4; ++c) ck(cudaMemsetAsync(sl.conv[c].p, 0, size_t(C[c]) * 3 * 4, s), "reset");
        ck(cudaStreamSynchronize(s), "reset");
        len = 0;
    }
};

int inkling2_prompt_chunk() {
    int dev = 0; cudaDeviceProp pr{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&pr, dev) != cudaSuccess) return 2048;
    return pr.totalGlobalMem >= (size_t(64) << 30) ? 8192 : 2048;
}
int Inkling2Model::chunk() const { return p_->chunk; }

Inkling2Model::Inkling2Model(const std::string & path, int context, int chunk, bool hf_attention, size_t expert_budget, bool exact, size_t drafter_bytes)
    : p_(std::make_unique<Impl>()) {
    {
        int dev = 0; cudaDeviceProp pr{};
        ck(cudaGetDevice(&dev), "device"); ck(cudaGetDeviceProperties(&pr, dev), "device");
        p_->int8_dense = inkling2_int8_dense(exact, pr.major, pr.minor);
        // --exact off on sm_12x (Blackwell, GB10): prompt experts on the native FP4 block-scaled tensor cores
        p_->fp4_experts = !exact && pr.major >= 12 && K::fp4_mma_available();
        p_->dev_mem = pr.totalGlobalMem;
    }
    p_->hf_attention = hf_attention;
    p_->context = context;
    p_->chunk = chunk;
    ck(cudaStreamCreate(&p_->s), "stream");
    cb(cublasCreate(&p_->blas), "create");
    cb(cublasSetStream(p_->blas, p_->s), "stream");
    p_->expert_budget = expert_budget;
    p_->drafter_keep = drafter_bytes;
    p_->load(path);
    p_->alloc();
    p_->place_experts();
    p_->choose_kernels();
    p_->reset();
}
Inkling2Model::~Inkling2Model() {
    for (auto & g : p_->graphs) if (g) cudaGraphExecDestroy(g);
    for (auto & g : p_->vgraphs) if (g) cudaGraphExecDestroy(g);
    if (p_->h_len) cudaFreeHost(p_->h_len);
    if (p_->h_tok) cudaFreeHost(p_->h_tok);
    if (p_->h_logits) cudaFreeHost(p_->h_logits);
    if (p_->h_gc) cudaFreeHost(p_->h_gc);
    if (p_->h_gf) cudaFreeHost(p_->h_gf);
    if (p_->h_next) cudaFreeHost(p_->h_next);
    if (p_->blas) cublasDestroy(p_->blas);
    if (p_->s) cudaStreamDestroy(p_->s);
}
void Inkling2Model::forward(const int * tokens, int n, int want, std::vector<float> & logits) { p_->forward(tokens, n, want, logits); }
void Inkling2Model::reset() { p_->reset(); }
void Inkling2Model::debug_visits(const std::vector<float> & hs, int n) { p_->visits(hs, n); }
void Inkling2Model::log_routes(std::vector<std::vector<int>> * log) { p_->route_log = log; }
int Inkling2Model::length() const { return p_->len; }
int Inkling2Model::replan_experts() { return p_->replan_experts(); }
void Inkling2Model::enable_mtp(const std::string & dir) { p_->enable_mtp(dir); }
void Inkling2Model::draft(int anchor, std::vector<int> & tokens, int steps) { p_->draft(anchor, tokens, steps); }
bool Inkling2Model::dflash_drafter() const { return p_->dflash; }
void Inkling2Model::draft(int anchor, std::vector<int> & tokens, int steps, std::vector<float> & conf) {
    conf.clear();
    if (p_->dflash) p_->dflash_draft(anchor, tokens, std::max(1, std::min(steps, p_->mtp_block - 1)), &conf);
    else p_->draft(anchor, tokens, steps);
}
size_t Inkling2Model::drafter_bytes(const std::string & dir) {
    namespace fs = std::filesystem;
    if (dir.empty()) return 0;
    const fs::path df = fs::path(dir) / "model.safetensors", mtp = fs::path(dir) / "mtp.safetensors";
    std::error_code ec;
    const size_t w = fs::exists(df, ec) ? size_t(fs::file_size(df, ec)) : fs::exists(mtp, ec) ? size_t(fs::file_size(mtp, ec)) : 0;
    // BF16 weights (+ their INT8 copy while converting), KV caches, block scratch
    return w * 3 / 2 + (size_t(384) << 20);
}
void Inkling2Model::draft_trace(const std::vector<int> & tokens, int anchor, std::vector<float> & hidden, std::vector<int> & ids, std::vector<int> & path) {
    p_->dflash_trace(tokens, anchor, hidden, ids, path);
}
void Inkling2Model::verify(const int * tokens, int n, std::vector<float> & logits) { p_->verify(tokens, n, logits); }
void Inkling2Model::commit(int keep) { p_->commit(keep); }
int Inkling2Model::resident_experts() const { return p_->resident_share; }
int Inkling2Model::vocab() const { return p_->V; }
int Inkling2Model::forward_greedy(int token) { return p_->forward_greedy(token); }
void Inkling2Model::snapshot() { p_->snapshot(); }
void Inkling2Model::verify_greedy(const int * tokens, int n, int * next) { p_->verify_greedy(tokens, n, next); }
bool Inkling2Model::restore() { return p_->restore(); }

}  // namespace tqllm
