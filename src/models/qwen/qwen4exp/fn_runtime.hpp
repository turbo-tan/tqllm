#pragma once
// Flash-Next engine runtime: CUDA error checks, scratch arena, prefill tuning and the shared per-process state.
#include <unordered_map>
// Private stage replay and native short-context chain diagnostics, not a generation or speed benchmark.
#include "debug.hpp"
#include "../src/cuda_recover.hpp"
#include "dev_alloc.cuh"
#include "tq3_expert_cache.hpp"
#include "prefill_plan.hpp"
#include "kv_store.hpp"
#include "kv_format.hpp"
#include "qwen4exp.hpp"
#include "dequant.hpp"
#include "moe_kernels.cuh"
#include "qwen4exp_kernels.cuh"
#include "gsq_activation.cuh"
#include "quant_kernels.cuh"
#include "vision_kernels.cuh"
#include "device_free_cuda.hpp"
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <future>
#include <thread>
#include <atomic>
#include <random>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <cuda_profiler_api.h>
// TQFN_* switches that won their A/Bs are on by default; set NAME=0 to turn one off
inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }
// Expert slots per trunk layer: TQFN_LAYER_SLOTS=c0,c1,... or, by default, EXPERTS.slots beside the packed experts (one
// line of comma-separated counts from docs/steering/tqllm/flashnext/tools/alloc.py, fitted on a routing trace of other
// prompts). Counts fitted for another --slots are scaled to keep the same total; without either, every layer gets --slots.
inline void apply_layer_slots(tqllm::MixedExpertCache & cache, const std::string & experts_path, int slots) {
    std::string text, from;
    if (const char * v = std::getenv("TQFN_LAYER_SLOTS"); v && *v) { text = v; from = "TQFN_LAYER_SLOTS"; }
    else if (!experts_path.empty()) {
        std::ifstream f(experts_path + ".slots");
        if (f && std::getline(f, text)) from = experts_path + ".slots";
    }
    if (text.empty()) return;
    std::vector<int> caps; size_t at = 0;
    while (at <= text.size()) { const size_t e = std::min(text.find(',', at), text.size()); caps.push_back(std::stoi(text.substr(at, e - at))); at = e + 1; }
    long total = 0; for (int c : caps) total += c;
    const long want = long(slots) * long(caps.size());
    if (total != want && total > 0)
        for (int & c : caps) c = std::max(1, std::min(cache.experts(), int(std::lround(double(c) * double(want) / double(total)))));
    cache.set_layer_slots(caps);
    std::printf("DECODE per-layer expert slots %d..%d from %s\n", *std::min_element(caps.begin(), caps.end()), cache.slots(), from.c_str());
}
// expert slots hold lane-contiguous tiles: only with a packed experts file (cold GGUF uploads stay untiled); --pack-experts
// writes tiled blocks unless TQFN_TILED=0
inline bool g_tile_experts = false;
// attention KV positions per sequence: 2048 = the QSA indexer budget (dense attention is exact below it); the server raises
// it for long contexts (dense attention over every position, see TQFN_SERVE_ENGINE)
inline int g_kv_capacity = 2048;
// slots=auto on a card with < 16 GB free after the dense weights (12 / 16 GB cards): CPU expert pool, small MTP bank and
// banks sized up to a 512 MB keep (fn_engine_load); TQFN_* variables still override each part
inline bool g_small_card = false;
inline bool g_cpu_owned = false;  // the CPU expert pool computes the owned experts (decode_rules.hpp cpu_owned); set by the decode pool

using namespace tqllm;
inline void ck(cudaError_t rc, const char * file = __builtin_FILE(), int line = __builtin_LINE()) {
    if (rc == cudaSuccess) return;
    const char * base = std::strrchr(file, '/');
    throw std::runtime_error(std::string(cudaGetErrorString(rc)) + " (" + (base ? base + 1 : file) + ":" + std::to_string(line) + ")");
}
inline void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
inline bool exact_f32() { static const bool on = std::getenv("TQFN_EXACT_F32") != nullptr; return on; }
inline void f32_to_half_host(const float * src, __half * dst, size_t count) {
    std::vector<float> v(count); ck(cudaMemcpy(v.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<__half> h(count); for (size_t i = 0; i < count; ++i) h[i] = __float2half_rn(v[i]);
    ck(cudaMemcpy(dst, h.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
}
inline bool bf16_act() { static const bool on = std::getenv("TQFN_BF16_ACT") != nullptr; return on; }
inline void round_bf16_device(float * p, size_t count) {
    std::vector<float> v(count); ck(cudaMemcpy(v.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto & x : v) { uint32_t u; std::memcpy(&u, &x, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u; std::memcpy(&x, &u, 4); }
    ck(cudaMemcpy(p, v.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

inline bool round32_gpu_side(const std::string & name) {
    static const char * mode = std::getenv("TQFN_ROUND32");
    if (!mode) return false;
    if (std::string(mode) == "all") return true;
    for (const char * cpu : {"ssm_alpha", "ssm_beta", "attn_qkv", "attn_gate", "shexp", "ffn_gate_inp", "hc_", "ple", "per_layer"})
        if (name.find(cpu) != std::string::npos) return false;
    return true;
}

inline void cpu_q8_round(std::vector<float> & values) {
    require(values.size() % 32 == 0, "Q8 diagnostic needs complete blocks");
    for (size_t b = 0; b < values.size(); b += 32) {
        float maximum = 0.f;
        for (size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(values[b + j]));
        const float scale = maximum / 127.f, inverse = scale ? 1.f / scale : 0.f;
        const float stored_scale = __half2float(__float2half_rn(scale));
        for (size_t j = 0; j < 32; ++j) values[b + j] = std::round(values[b + j] * inverse) * stored_scale;
    }
}
inline void cpu_q8k_round(std::vector<float> & values) {
    require(values.size() % 256 == 0, "Q8_K diagnostic needs complete blocks");
    for (size_t b = 0; b < values.size(); b += 256) {
        float extremum = 0.f;
        for (size_t j = 0; j < 256; ++j)
            if (std::abs(values[b + j]) > std::abs(extremum)) extremum = values[b + j];
        if (!extremum) continue;
        const float inverse = -127.f / extremum, scale = 1.f / inverse;
        for (size_t j = 0; j < 256; ++j)
            values[b + j] = std::min(127.f, std::round(inverse * values[b + j])) * scale;
    }
}
// Decode scratch arena: while active (and not suspended by PersistentAlloc), Device temporaries are bump-allocated from
// one buffer that is reset per decode step, so a token makes no allocation calls. State that outlives a step must be
// allocated under PersistentAlloc.
struct ScratchArena { char * base = nullptr; size_t size = 0, used = 0, high = 0; };
inline bool g_prefill_log = std::getenv("TQFN_PREFILL_LOG") != nullptr; inline double g_qsa_ms = 0;  // prefill profile (syncs)
inline ScratchArena * g_arena = nullptr; inline int g_persistent = 0;
inline bool g_profile_decode_req = false;  // TQFN_PROFILE_DECODE: the profiled request is in its decode
inline int g_qsa_block = 2048;  // queries per QSA selection block (prefill): the plan lowers it (1024, 512) before halving the chunk
inline int g_sub_req = 0;       // the request's dense sub-chunk cap: follows the block (the arena's users scale with it), 0 = --prefill-sub
inline int g_prefill_epoch = 0;  // bumped when a prefill releases decode scratch: graph capture warms up again
inline std::unique_ptr<tqllm_cuda::QsaBatchScratch> g_qsa_batch;  // QSA selection scratch, sized by g_qsa_block
struct PersistentAlloc { PersistentAlloc() { ++g_persistent; } ~PersistentAlloc() { --g_persistent; } };
// --dense-cache-mb: one layer's fp16 dense weight images, dequantised once and reused by every
// 256-token sub-chunk of that layer (the pool is a prefill borrow: counted in the bank eviction, freed with the prefill buffers)
struct DenseHalfCache { char * base = nullptr; size_t cap = 0, used = 0, high = 0; int layer = -2; std::unordered_map<const void *, __half *> map; };
inline DenseHalfCache g_dhc;
// layer-major prefill tuning (tqserve parameters --prefill-*, --dense-cache-mb; set before the engine starts)
struct FnTune { int chunk = 16384, sub = 2048, stage = 192, dense_min = 256, borrow_count = 0; long reserve_mb = 1000, arena_mb = 3800, dense_cache_mb = 600;
                tqllm::BorrowMode borrow = tqllm::BorrowMode::Auto;
                bool dense_cache_set = false, chunk_set = false, stage_set = false; };  // set: given on the command line
inline FnTune g_tune;
inline size_t dense_half_pool_bytes() { return size_t(std::max(0L, g_tune.dense_cache_mb)) << 20; }
inline void dense_half_release() {
    if (g_dhc.base && std::getenv("TQFN_PREFILL_LOG")) std::fprintf(stderr, "tqllm: dense fp16 pool high-water %.0f of %.0f MB\n", g_dhc.high / 1048576.0, g_dhc.cap / 1048576.0);
    if (g_dhc.base) cudaFree(g_dhc.base); g_dhc = DenseHalfCache{};
}
// hands back the arena temporaries of a scope (anything allocated before it stays valid)
// release_prefill trims the memory pool only when free device memory is below this (the refill and the MTP bank need
// it); the trim costs 60-100 ms of idle GPU, so a request whose refill fits without it skips it
inline size_t g_trim_below = SIZE_MAX;
struct ArenaMark { size_t at = g_arena ? g_arena->used : 0; ~ArenaMark() { if (g_arena) g_arena->used = at; } };
template<class T> struct Device {
    T * p = nullptr; size_t count; bool owned = true, plain = false;
    struct View {};
    Device(View, T * q, size_t n) : p(q), count(n), owned(false) {}  // non-owning view of persistent memory
    explicit Device(size_t n) : count(n) {
        const size_t bytes = (std::max<size_t>(n, 1) * sizeof(T) + 255) / 256 * 256;
        if (g_arena && !g_persistent) {
            if (g_arena->used + bytes <= g_arena->size) {
                p = reinterpret_cast<T *>(g_arena->base + g_arena->used); g_arena->used += bytes; owned = false;
                g_arena->high = std::max(g_arena->high, g_arena->used); return;
            }
            // the arena is sized by a measured rule; past it the pool serves the request (same values, another address)
            static bool warned = false;
            if (!warned) { warned = true; std::fprintf(stderr, "tqllm: scratch arena full (%zu of %zu MB): allocating from the pool\n", g_arena->used >> 20, g_arena->size >> 20); }
        }
        // during a CUDA graph capture (decode graphs) a pool allocation would become graph-owned memory, remapped at every
        // launch: state allocated lazily there (raw keys, carries) is taken outside the graph instead
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed; ck(cudaThreadExchangeStreamCaptureMode(&mode));
            const cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&p), bytes);
            ck(cudaThreadExchangeStreamCaptureMode(&mode));
            if (e != cudaSuccess) { cudaGetLastError(); throw std::runtime_error(std::string(cudaGetErrorString(e)) + ": device buffer of " + std::to_string(bytes >> 20) + " MiB in a graph capture"); }
            plain = true; return;
        }
        if (const cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&p), bytes, nullptr); e != cudaSuccess) {
            cudaGetLastError(); size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
            throw std::runtime_error(std::string(cudaGetErrorString(e)) + ": device buffer of " + std::to_string(bytes >> 20) + " MiB (" +
                                     (g_persistent ? "persistent" : "temporary") + "), " + std::to_string(fr >> 20) + " MiB free");
        }
    }
    ~Device() {
        if (!owned) return;
        // a free inside a CUDA graph capture (a buffer replaced by a larger one in a captured step) is invalid for pool
        // memory and synchronizes for cudaMalloc memory: both wait until the capture is closed
        static std::vector<std::pair<void *, bool>> later;
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) { later.emplace_back(p, plain); return; }
        for (auto [q, sync] : later) { if (sync) cudaFree(q); else cudaFreeAsync(q, nullptr); }
        later.clear();
        if (plain) cudaFree(p); else cudaFreeAsync(p, nullptr);
    }
    void put(const std::vector<T> & v) {
        require(v.size() == count, "device upload size differs");
        ck(cudaMemcpy(p, v.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(count); ck(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost)); return values;
    }
};
inline void validate_round(const Device<__half> & actual, std::vector<float> expected, bool q8k, const std::string & name) {
    if (q8k) cpu_q8k_round(expected); else cpu_q8_round(expected);
    const auto output = actual.get();
    require(output.size() == expected.size(), "rounding validation shape differs");
    for (size_t j = 0; j < output.size(); ++j) {
        const __half h = __float2half_rn(expected[j]);
        if (__half2float(output[j]) != __half2float(h))
            throw std::runtime_error("GSQ device rounding differs: " + name + " index=" + std::to_string(j) +
                                     " cpu=" + std::to_string(__half2float(h)) +
                                     " gpu=" + std::to_string(__half2float(output[j])));
    }
}
