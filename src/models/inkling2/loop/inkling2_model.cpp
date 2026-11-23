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
#include "graph_capture.hpp"

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
struct Mat { Dev d, q8, sc; int rows = 0, cols = 0; const bf16 * w() const { return d.as<bf16>(); } };
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
struct Ctrl { Mat down; Dev FT, GT; };  // FT / GT: forget / gain weights transposed, BF16 [rank][D]

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
    cudaGraphExec_t graphs[K::kSmallN + 1] = {};  // [0]: the decode step; [n]: a verify block of n tokens
    // workspace
    Dev ids, x0, hA, hB, P, B1[2], O17[2], a3, msgs, gt[3], r, m, oconv, mconv, tmp;
    Dev arn;  // attn_res_norm split-k scratch (n <= kSmallN), allocated and zeroed before any capture
    Dev xq, xk, xv, q, rs, sink, gate, kraw, vraw, kt, vt, kdyn, vdyn, att, o, xb, eids, ewts, rlog, yg, yu, act, yd, moe, z, g64, fin, logits16,
        logits, part, gorder, gcounts, goffsets, gtile, pq, pr, gdedup;
    // BF16 expert prompt path (big cards): gathered inputs, gate+up GEMM out, down GEMM out, host counts / offsets
    Dev xg, guo, ygo, ginv;
    bool bf16_experts = false, fp4_experts = false;
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
        if (t.type != type) throw std::runtime_error("inkling2: " + n + " type " + std::to_string(t.type));
        const auto b = f.read(t);
        return upload(b.data(), b.size());
    }
    // int8 dense mode (opt-in, not exact against the reference): the projections, the LM head and the message-gate downs
    // as INT8 W8A16 rows with per-row max-abs scales; the router and the norms stay BF16 / F32
    bool int8_dense = false;
    // small = projections that feed exponentials or gates (sinks, relative bias, output gate, conv projections, loop
    // controls): few bytes, error amplified -> stay BF16
    Mat mat(const std::string & n, bool small = false) {
        const auto & t = T(n);
        Mat m; m.cols = int(t.ne[0]); m.rows = int(t.ne[1]);
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
    size_t expert_budget = SIZE_MAX;
    size_t dev_mem = 0;
    int resident_share = 0;  // experts per 128 kept on the device (128: all)
    void place_experts() {
        size_t total = 0;
        for (const auto & Y : Ls) total += Y.eg.bytes() + Y.eu.bytes() + Y.ed.bytes();
        auto budget = [&] {
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "mem info");
            const size_t keep = size_t(1536) << 20;  // cuBLAS workspace, graphs, logits growth
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
        // big cards (>= 96 GB): BF16 prompt copies for the cuBLAS path, allocated after the quantized stacks are placed
        // (the pinned host copies are free again then); a failed allocation keeps the WMMA path
        if (!fp4_experts && dev_mem >= (size_t(96) << 30) && expert_budget == SIZE_MAX) {
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
    int replan_experts() {
        if (resident_share >= Ls[0].eg.E) return 0;
        const int E_ = Ls[0].eg.E;
        std::vector<int> cnt(size_t(L) * E_);
        ck(cudaMemcpyAsync(cnt.data(), ucount.p, cnt.size() * 4, cudaMemcpyDeviceToHost, s), "counts");
        ck(cudaStreamSynchronize(s), "counts");
        int moved = 0;
        for (int l = 0; l < L; ++l) {
            std::vector<int> order(static_cast<size_t>(E_));
            for (int e = 0; e < E_; ++e) order[size_t(e)] = e;
            const int * c = cnt.data() + size_t(l) * E_;
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
                    Y.ctrl[c].FT = bf16_of(d + g + "forget.weight", true);
                    Y.ctrl[c].GT = bf16_of(d + g + "gain.weight", true);
                }
        }
    }

    void alloc() {
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
        grow_logits(std::max(1, std::min(n, K::kSmallN)));
        int pf = tqllm_cuda::inkling2_attention_decode_floats(H, K::kSmallN, kSplitsFull);
        pq = Dev(size_t(n) * H * HD * 2); pr = Dev(size_t(n) * H * DR * 4);  // prompt attention scratch
        gdedup = Dev(size_t(1 + K::kSmallN * topk * (1 + K::kSmallN)) * 4);
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
    }
    // Y[n][rows] = X[n][cols] W^T, BF16 in / out, F32 accumulate
    void gemm(const bf16 * W, int rows, int cols, const bf16 * X, void * Y, int n, bool f32 = false) {
        const float one = 1.0f, zero = 0.0f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &one, W, CUDA_R_16BF, cols, X, CUDA_R_16BF, cols,
                        &zero, Y, f32 ? CUDA_R_32F : CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm");
    }
    static K::GemvSeg seg(const Mat & W, const bf16 * x, void * y, bool f32 = false) {
        if (W.q8.p) return K::GemvSeg{reinterpret_cast<const bf16 *>(W.q8.p), x, y, W.rows, W.cols, W.cols, W.rows, f32 ? 1 : 0, W.sc.as<float>()};
        return K::GemvSeg{W.w(), x, y, W.rows, W.cols, W.cols, W.rows, f32 ? 1 : 0};
    }
    // projections that share nothing but the launch: few tokens -> one gemv launch, else one cuBLAS GEMM each (INT8 rows
    // dequantised to a BF16 scratch first: the same weights as the gemv)
    Dev dq_scratch;
    void project(const K::GemvSeg * sg, int ns, int n) {
        if (n <= K::kSmallN) { K::gemv(sg, ns, n, s); return; }
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
        for (int i = 0; i < count; ++i)
            K::msg_gate_out(g64.as<bf16>() + size_t(i) * n * rank, Y.ctrl[c0 + i].FT.as<bf16>(), Y.ctrl[c0 + i].GT.as<bf16>(), h[i], msg[i], y[i], n, D, rank, s);
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
        if (Y.full) {
            K::conv_dynamic(kraw.as<bf16>(), Y.kconv.as<float>(), kdyn.as<bf16>(), S.conv[0].as<float>(), kt.as<bf16>(), n, C, s, hist(S, 0));
            K::conv_dynamic(vraw.as<bf16>(), Y.vconv.as<float>(), vdyn.as<bf16>(), S.conv[1].as<float>(), vt.as<bf16>(), n, C, s, hist(S, 1));
        } else {
            K::conv_static(kraw.as<bf16>(), Y.kconv.as<float>(), S.conv[0].as<float>(), nullptr, kt.as<bf16>(), n, C, s, nullptr, false, hist(S, 0));
            K::conv_static(vraw.as<bf16>(), Y.vconv.as<float>(), S.conv[1].as<float>(), nullptr, vt.as<bf16>(), n, C, s, nullptr, false, hist(S, 1));
        }
        tqllm_cuda::inkling2_kv_finish(kt.as<bf16>(), vt.as<bf16>(), S.kc.as<bf16>(), S.vc.as<bf16>(), n, G, qk_clip, eps, dlen.as<int>(), S.ring, s);
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
        K::route_topk(rlog.as<float>(), Y.rbias.as<float>(), eids.as<int>(), ewts.as<float>(), n, E, topk, rscale, s);
        K::count_routes(eids.as<int>(), n * topk, ucount.as<int>() + size_t(l) * E, s);
        if (route_log) {
            route_log->emplace_back(size_t(n) * topk);
            ck(cudaMemcpyAsync(route_log->back().data(), eids.p, size_t(n) * topk * 4, cudaMemcpyDeviceToHost, s), "route log");
            ck(cudaStreamSynchronize(s), "route log");
        }
        if (small && n > 1) {
            K::expert_dedup(Y.eg.m, Y.eu.m, Y.ed.m, I, D, xb.as<bf16>(), eids.as<int>(), ewts.as<float>(), gdedup.as<int>(), act.as<bf16>(), yd.as<bf16>(),
                            moe.as<bf16>(), n, topk, gcap, ucap, s);
        } else if (small) {
            K::expert_gate_up(Y.eg.m, Y.eu.m, I, D, xb.as<bf16>(), eids.as<int>(), act.as<bf16>(), n, topk, gcap, ucap, s);
            K::expert_down(Y.ed.m, D, I, act.as<bf16>(), eids.as<int>(), ewts.as<float>(), moe.as<bf16>(), n, topk, s);
        } else {  // prompt chunk: assignments grouped by expert, tensor-core GEMM
            const int A = n * topk;
            int * go = gorder.as<int>(), * gc = gcounts.as<int>(), * gf = goffsets.as<int>(), * gt_ = gtile.as<int>();
            K::group_assignments(eids.as<int>(), A, E, go, gc, gf, gt_, s);
            if (fp4_experts && Y.eg.m.fmt == 2 && Y.eu.m.fmt == 2 && Y.ed.m.fmt == 2 && I % 64 == 0 && D % 64 == 0) {
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
        for (auto & S : slots) {
            const int C[4] = {G * HD, G * HD, D, D};
            for (int c = 0; c < 4; ++c) { S.hist[c] = Dev(size_t(K::kSmallN) * C[c] * 2); S.snap[c] = Dev(size_t(C[c]) * 12); }
        }
        for (auto & g : graphs) if (g) { cudaGraphExecDestroy(g); g = nullptr; }  // the steps now capture the taps
        mtp_on = true;
        if (len) throw std::runtime_error("inkling2 mtp: enable before the first forward");
    }
    static constexpr int kMtpSplits = 16;
    // draft context K / V of the m committed tokens at positions pos0.. (their taps are in `taps`), and f_rnn of the last
    void mtp_context(int pos0, int m) {
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
        {   // base logits of the 7 masked positions through the target LM head
            K::GemvSeg sg = seg(out, d_hid.as<bf16>() + D, d_base.p, true);
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
        K::GemvSeg sg = seg(out, last, logits16.p);
        project(&sg, 1, want);
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

Inkling2Model::Inkling2Model(const std::string & path, int context, int chunk, bool hf_attention, size_t expert_budget, bool exact)
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
    p_->load(path);
    p_->alloc();
    p_->place_experts();
    p_->reset();
}
Inkling2Model::~Inkling2Model() {
    for (auto & g : p_->graphs) if (g) cudaGraphExecDestroy(g);
    if (p_->h_len) cudaFreeHost(p_->h_len);
    if (p_->h_tok) cudaFreeHost(p_->h_tok);
    if (p_->h_logits) cudaFreeHost(p_->h_logits);
    if (p_->h_gc) cudaFreeHost(p_->h_gc);
    if (p_->h_gf) cudaFreeHost(p_->h_gf);
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
void Inkling2Model::verify(const int * tokens, int n, std::vector<float> & logits) { p_->verify(tokens, n, logits); }
void Inkling2Model::commit(int keep) { p_->commit(keep); }
int Inkling2Model::resident_experts() const { return p_->resident_share; }
int Inkling2Model::vocab() const { return p_->V; }

}  // namespace tqllm
