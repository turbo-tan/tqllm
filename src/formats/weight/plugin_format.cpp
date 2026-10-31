#include "plugin_format.hpp"
#include "dequant.hpp"
#include "format_contract.hpp"
#include "gguf.hpp"
#include "weight_format.hpp"
#include "tqllm/plugin.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>

namespace tqllm {
namespace {

void call(int32_t rc, const tq_format_v2 & d, const char * what) {
    if (rc) throw std::runtime_error(std::string("plugin format ") + d.name + ": " + what + " failed (" + std::to_string(rc) + ")");
}

// A new GGUF type: the GGUF blocks go to the device unchanged, the plugin's kernels read them.
struct PluginFormat final : WeightFormat {
    tq_format_v2 d;
    explicit PluginFormat(const tq_format_v2 & desc) : d(desc) {}
    const char * name() const override { return d.name; }
    Basis basis() const override { return Basis::Plain; }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t) const override {
        return {up(raw.data(), raw.size()), d.gguf_type, rows, cols, nullptr, this};
    }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        call(d.gpu_matvec(m.p, m.aux, m.rows, m.cols, x, y, n, s), d, "matvec");
    }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        call(d.gpu_dequant_f16(m.p, m.aux, m.rows, m.cols, out, s), d, "dequant");
    }
    size_t row_bytes(int cols) const override { return size_t(cols) / d.block_values * d.block_bytes; }
};

// A patch: the plugin's dequant / matvec over the built-in format's device layout; everything else is the built-in's.
struct PluginPatch final : WeightFormat {
    std::unique_ptr<WeightFormat> base;
    tq_format_v2 d;
    PluginPatch(std::unique_ptr<WeightFormat> b, const tq_format_v2 & desc) : base(std::move(b)), d(desc) {}
    const char * name() const override { return base->name(); }
    Basis basis() const override { return base->basis(); }
    Mat load(const std::vector<char> & raw, int rows, int cols, const Upload & up, cudaStream_t s) const override {
        Mat m = base->load(raw, rows, cols, up, s);
        if (m.fmt == base.get()) m.fmt = this;
        return m;
    }
    int max_matvec_cols() const override { return base->max_matvec_cols(); }
    int max_batch_cols() const override { return base->max_batch_cols(); }
    void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override {
        if (d.gpu_matvec) call(d.gpu_matvec(m.p, m.aux, m.rows, m.cols, x, y, n, s), d, "matvec");
        else base->matvec(m, x, y, n, s);
    }
    void matvec_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const override { base->matvec_f32(m, x, y, n, s); }
    // a patched matvec replaces the strided variant too, so every decode matvec runs the plugin's kernel
    bool matvec_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        return !d.gpu_matvec && base->matvec_ld(m, x, y, n, ldy, s);
    }
    bool has_matvec_ld() const override { return !d.gpu_matvec && base->has_matvec_ld(); }
    bool expert_gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int a, const int * b, const int * c, int e, int f, int g, bool h,
                     cudaStream_t s) const override { return base->expert_gemm(m, x, y, a, b, c, e, f, g, h, s); }
    bool expert_gemm_i8(const Mat & m, const int8_t * xq, const float * xs, const int16_t * xsum, __nv_bfloat16 * y, int a, const int * b,
                        const int * c, int e, int f, int g, bool h, cudaStream_t s, int mode) const override {
        return base->expert_gemm_i8(m, xq, xs, xsum, y, a, b, c, e, f, g, h, s, mode);
    }
    bool expert_gemm_gateup(const Mat & g, const Mat & u, const __half * x, __half * y, int rows, const int * counts, const int * groups,
                            int capacity, int experts, int topk, cudaStream_t s) const override {
        return base->expert_gemm_gateup(g, u, x, y, rows, counts, groups, capacity, experts, topk, s);
    }
    bool has_expert_matvec() const override { return base->has_expert_matvec(); }
    void expert_matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * ids, int n, int topk, bool per,
                       cudaStream_t s) const override { base->expert_matvec(m, x, y, rows, ids, n, topk, per, s); }
    int u4h_id() const override { return base->u4h_id(); }
    void dequant_h(const Mat & m, __half * out, cudaStream_t s) const override {
        if (d.gpu_dequant_f16) call(d.gpu_dequant_f16(m.p, m.aux, m.rows, m.cols, out, s), d, "dequant");
        else base->dequant_h(m, out, s);
    }
    bool dequant_i8(const Mat & m, int8_t * w8, float * rs, cudaStream_t s) const override { return base->dequant_i8(m, w8, rs, s); }
    size_t row_bytes(int cols) const override { return base->row_bytes(cols); }
    size_t aux_row_bytes(int cols) const override { return base->aux_row_bytes(cols); }
    bool gemm(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override { return base->gemm(m, x, y, n, s); }
    bool has_gemm() const override { return base->has_gemm(); }
    int gemm_f32_id() const override { return base->gemm_f32_id(); }
    int fused_gemm_cols() const override { return base->fused_gemm_cols(); }
    bool gemm_ld(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, int ldy, cudaStream_t s) const override {
        return base->gemm_ld(m, x, y, n, ldy, s);
    }
    const std::vector<Mat> * parts() const override { return base->parts(); }
};

// run-time formats a patch may target, and the GGUF type whose loader produces matrices of it (the contract loads
// through that loader)
uint32_t patch_loader_type(uint32_t target) { return target == TQ_PATCH_I8H ? 8 : UINT32_MAX; }

int device_sm() {
    int dev = 0; cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
    return p.major * 10 + p.minor;
}

void run_contract(uint32_t type, const WeightFormat & loader, const WeightFormat * runtime, const char * name) {
    std::mt19937 rng(11);
    for (int pass = 0; pass < 2; ++pass) {
        const auto r = contract_pass(type, loader, runtime, pass, rng);
        if (!r.error.empty()) throw std::runtime_error(std::string("plugin format ") + name + ": " + r.error);
        if (!r.ok)
            throw std::runtime_error(std::string("plugin format ") + name + " fails the format contract (pass " + (pass ? "B" : "A") + ": " +
                                     std::to_string(r.bad) + " of " + std::to_string(r.compared) + " values off, max |diff| / group max " +
                                     std::to_string(r.worst_rel) + ", matvec rel-rms " + std::to_string(r.matvec_rel) + ")");
    }
}

std::vector<PluginInfo> & plugins() { static std::vector<PluginInfo> p; return p; }

}  // namespace

PluginInfo load_plugin(const std::string & path) {
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);  // never closed: formats keep its callbacks
    if (!h) throw std::runtime_error("plugin " + path + ": " + dlerror());
    auto init = reinterpret_cast<tq_plugin_init_v2_fn>(dlsym(h, "tq_plugin_init_v2"));
    if (!init) throw std::runtime_error("plugin " + path + ": no tq_plugin_init_v2 (ABI 2)");
    std::vector<tq_format_v2> descs;
    tq_host_v2 host{sizeof(tq_host_v2), TQ_ABI_VERSION_2, &descs, [](void * ctx, const tq_format_v2 * f, tq_error * err) -> int32_t {
        const char * why = !f ? "null descriptor" : f->struct_size < sizeof(tq_format_v2) ? "descriptor too small"
                         : f->abi_version != TQ_ABI_VERSION_2 ? "ABI version" : !f->name || !*f->name ? "no name" : nullptr;
        if (why) {
            if (err && err->struct_size >= sizeof(tq_error)) { err->code = 1; std::snprintf(err->message, sizeof(err->message), "%s", why); }
            return 1;
        }
        static_cast<std::vector<tq_format_v2> *>(ctx)->push_back(*f);
        return 0;
    }};
    tq_error err{sizeof(tq_error), 0, {}};
    if (init(&host, &err))
        throw std::runtime_error("plugin " + path + ": initialization failed" + (err.message[0] ? std::string(": ") + err.message : std::string()));
    (void) weight_format(0);  // the built-in table, before any patch
    PluginInfo info; info.path = path;
    const int sm = device_sm();
    for (const auto & d : descs) {
        const std::string label = std::string(d.name) + " (type " + std::to_string(d.gguf_type) + ")";
        if (d.gguf_type >= TQ_PLUGIN_TYPE_MIN && d.gguf_type <= TQ_PLUGIN_TYPE_MAX) {
            if (!d.cpu_dequant || !d.gpu_dequant_f16 || !d.gpu_matvec || !d.block_values || !d.block_bytes)
                throw std::runtime_error("plugin format " + label + ": a new format needs block sizes, cpu_dequant, gpu_dequant_f16 and gpu_matvec");
            if (weight_format(d.gguf_type)) throw std::runtime_error("plugin format " + label + ": type already registered");
            if (d.sm_min && (sm < int(d.sm_min) || sm > int(d.sm_max)))
                throw std::runtime_error("plugin format " + label + ": kernels for sm " + std::to_string(d.sm_min) + "-" + std::to_string(d.sm_max) +
                                         ", this GPU is sm " + std::to_string(sm));
            register_tensor_type(d.gguf_type, d.block_values, d.block_bytes);
            const auto cpu = d.cpu_dequant;
            register_host_dequant(d.gguf_type, [cpu, label](const void * src, float * dst, int64_t n) {
                if (cpu(src, dst, uint64_t(n))) throw std::runtime_error("plugin format " + label + ": cpu_dequant failed");
            });
            auto f = std::make_unique<PluginFormat>(d);
            run_contract(d.gguf_type, *f, nullptr, d.name);
            register_weight_format(d.gguf_type, std::move(f));
            info.formats.push_back(label);
            continue;
        }
        const uint32_t loader_type = patch_loader_type(d.gguf_type);
        if (loader_type == UINT32_MAX)
            throw std::runtime_error("plugin format " + label + ": a new type needs an id in " + std::to_string(TQ_PLUGIN_TYPE_MIN) + "-" +
                                     std::to_string(TQ_PLUGIN_TYPE_MAX) + "; patchable run-time formats: " + std::to_string(TQ_PATCH_I8H) + " (I8H)");
        if (!d.gpu_dequant_f16 && !d.gpu_matvec) throw std::runtime_error("plugin format " + label + ": a patch replaces dequant, matvec or both");
        if (d.sm_min && (sm < int(d.sm_min) || sm > int(d.sm_max))) {
            info.skipped.push_back(label + " (sm " + std::to_string(d.sm_min) + "-" + std::to_string(d.sm_max) + ", this GPU sm " + std::to_string(sm) + ")");
            continue;
        }
        const WeightFormat * loader = weight_format(loader_type);
        const WeightFormat * builtin = weight_format(d.gguf_type);
        // check the patch on a stand-in that forwards to the built-in, then install it
        struct Borrow final : WeightFormat {  // the built-in by reference, as PluginPatch's base
            const WeightFormat * f;
            explicit Borrow(const WeightFormat * x) : f(x) {}
            const char * name() const override { return f->name(); }
            Basis basis() const override { return f->basis(); }
            Mat load(const std::vector<char> & r, int a, int b, const Upload & u, cudaStream_t s) const override { return f->load(r, a, b, u, s); }
            int max_matvec_cols() const override { return f->max_matvec_cols(); }
            void matvec(const Mat & m, const __half * x, __nv_bfloat16 * y, int n, cudaStream_t s) const override { f->matvec(m, x, y, n, s); }
            void dequant_h(const Mat & m, __half * o, cudaStream_t s) const override { f->dequant_h(m, o, s); }
        };
        PluginPatch trial(std::make_unique<Borrow>(builtin), d);
        run_contract(loader_type, *loader, &trial, d.name);
        std::mt19937 rng(23);
        const size_t diff = contract_identical(loader_type, *loader, *builtin, trial, d.gpu_matvec != nullptr, rng);
        if (d.exact && diff)
            throw std::runtime_error("plugin format " + label + " claims exact but differs from the built-in on " + std::to_string(diff) + " values");
        info.inexact |= diff != 0;
        patch_weight_format(d.gguf_type, [&](std::unique_ptr<WeightFormat> old) { return std::make_unique<PluginPatch>(std::move(old), d); });
        info.patches.push_back(std::string(d.name) + " -> type " + std::to_string(d.gguf_type) + (diff ? " (not exact)" : " (exact)"));
    }
    plugins().push_back(info);
    return info;
}

const std::vector<PluginInfo> & loaded_plugins() { return plugins(); }

}  // namespace tqllm
