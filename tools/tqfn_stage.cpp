#include "../src/prefill_observer_cuda.hpp"
// Private stage replay and native short-context chain diagnostics, not a generation or speed benchmark.
#include "../src/tq3_expert_cache.hpp"
#include "../src/qwen4exp.hpp"
#include "../src/dequant.hpp"
#include "moe_kernels.cuh"
#include "qwen4exp_kernels.cuh"
#include "gsq_activation.cuh"
#include "quant_kernels.cuh"
#include "vision_kernels.cuh"
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
static inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }
// TQFN_LAYER_SLOTS=c0,c1,...: expert slots per trunk layer (from docs/steering/tqllm/flashnext/tools/alloc.py on a routing
// trace, same total bytes as --slots); the GPU state still has one table size, so device resolve stays uniform.
static void apply_layer_slots(tqllm::MixedExpertCache & cache) {
    const char * v = std::getenv("TQFN_LAYER_SLOTS");
    if (!v || !*v) return;
    std::vector<int> caps; std::string t(v); size_t at = 0;
    while (at <= t.size()) { const size_t e = std::min(t.find(',', at), t.size()); caps.push_back(std::stoi(t.substr(at, e - at))); at = e + 1; }
    cache.set_layer_slots(caps);
    std::printf("DECODE per-layer expert slots %d..%d (TQFN_LAYER_SLOTS)\n", *std::min_element(caps.begin(), caps.end()), cache.slots());
}
// expert slots hold lane-contiguous tiles: only with a packed experts file (cold GGUF uploads stay untiled); --pack-experts
// writes tiled blocks unless TQFN_TILED=0
static bool g_tile_experts = false;
// attention KV positions per sequence: 2048 = the QSA indexer budget (dense attention is exact below it); the server raises
// it for long contexts (dense attention over every position, see TQFN_SERVE_ENGINE)
static int g_kv_capacity = 2048;

using namespace tqllm;
static void ck(cudaError_t rc, int line = __builtin_LINE()) { if (rc != cudaSuccess) throw std::runtime_error(std::string(cudaGetErrorString(rc)) + " (tqfn_stage.cpp:" + std::to_string(line) + ")"); }
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
static bool exact_f32() { static const bool on = std::getenv("TQFN_EXACT_F32") != nullptr; return on; }
static void f32_to_half_host(const float * src, __half * dst, size_t count) {
    std::vector<float> v(count); ck(cudaMemcpy(v.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<__half> h(count); for (size_t i = 0; i < count; ++i) h[i] = __float2half_rn(v[i]);
    ck(cudaMemcpy(dst, h.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
}
static bool bf16_act() { static const bool on = std::getenv("TQFN_BF16_ACT") != nullptr; return on; }
static void round_bf16_device(float * p, size_t count) {
    std::vector<float> v(count); ck(cudaMemcpy(v.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto & x : v) { uint32_t u; std::memcpy(&u, &x, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u; std::memcpy(&x, &u, 4); }
    ck(cudaMemcpy(p, v.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

static bool round32_gpu_side(const std::string & name) {
    static const char * mode = std::getenv("TQFN_ROUND32");
    if (!mode) return false;
    if (std::string(mode) == "all") return true;
    for (const char * cpu : {"ssm_alpha", "ssm_beta", "attn_qkv", "attn_gate", "shexp", "ffn_gate_inp", "hc_", "ple", "per_layer"})
        if (name.find(cpu) != std::string::npos) return false;
    return true;
}

static void cpu_q8_round(std::vector<float> & values) {
    require(values.size() % 32 == 0, "Q8 diagnostic needs complete blocks");
    for (size_t b = 0; b < values.size(); b += 32) {
        float maximum = 0.f;
        for (size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(values[b + j]));
        const float scale = maximum / 127.f, inverse = scale ? 1.f / scale : 0.f;
        const float stored_scale = __half2float(__float2half_rn(scale));
        for (size_t j = 0; j < 32; ++j) values[b + j] = std::round(values[b + j] * inverse) * stored_scale;
    }
}
static void cpu_q8k_round(std::vector<float> & values) {
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
static bool g_prefill_log = std::getenv("TQFN_PREFILL_LOG") != nullptr; static double g_qsa_ms = 0;  // prefill profile (syncs)
static ScratchArena * g_arena = nullptr; static int g_persistent = 0;
struct PersistentAlloc { PersistentAlloc() { ++g_persistent; } ~PersistentAlloc() { --g_persistent; } };
// hands back the arena temporaries of a scope (anything allocated before it stays valid)
struct ArenaMark { size_t at = g_arena ? g_arena->used : 0; ~ArenaMark() { if (g_arena) g_arena->used = at; } };
template<class T> struct Device {
    T * p = nullptr; size_t count; bool owned = true;
    struct View {};
    Device(View, T * q, size_t n) : p(q), count(n), owned(false) {}  // non-owning view of persistent memory
    explicit Device(size_t n) : count(n) {
        const size_t bytes = (std::max<size_t>(n, 1) * sizeof(T) + 255) / 256 * 256;
        if (g_arena && !g_persistent) {
            if (g_arena->used + bytes > g_arena->size)
                throw std::runtime_error("decode scratch arena exhausted: " + std::to_string(bytes) + " bytes requested, " +
                                         std::to_string(g_arena->used) + " of " + std::to_string(g_arena->size) + " used");
            p = reinterpret_cast<T *>(g_arena->base + g_arena->used); g_arena->used += bytes; owned = false;
            g_arena->high = std::max(g_arena->high, g_arena->used); return;
        }
        ck(cudaMallocAsync(reinterpret_cast<void **>(&p), bytes, nullptr));
    }
    ~Device() { if (owned) cudaFreeAsync(p, nullptr); }
    void put(const std::vector<T> & v) {
        require(v.size() == count, "device upload size differs");
        ck(cudaMemcpy(p, v.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> values(count); ck(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost)); return values;
    }
};
static void validate_round(const Device<__half> & actual, std::vector<float> expected, bool q8k, const std::string & name) {
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
struct Stage { std::string file, name; size_t elements; std::vector<uint64_t> shape; };
class Fixtures {
    std::filesystem::path root_;
    std::map<std::string, std::vector<Stage>> stages_;
public:
    Fixtures() = default;  // no fixtures (serving): has() is always false
    explicit Fixtures(const std::string & root) : root_(root) {
        std::ifstream input(root_ / "tensors.tsv"); require(bool(input), "fixture manifest missing");
        std::string line;
        while (std::getline(input, line)) {
            std::stringstream fields(line); Stage stage;
            require(bool(std::getline(fields, stage.file, '\t') && std::getline(fields, stage.name, '\t')), "malformed manifest");
            size_t elements = 1, bytes = 0;
            for (int axis = 0; axis < 4; ++axis) {
                uint64_t n; require(bool(fields >> n) && n > 0 && n <= (32 * 1024 * 1024) / 4 / elements, "invalid fixture dimensions");
                stage.shape.push_back(n); elements *= n;
            }
            require(bool(fields >> bytes) && bytes == elements * 4 && std::filesystem::path(stage.file).filename() == stage.file,
                    "fixture size/path differs");
            stage.elements = elements; stages_[stage.name].push_back(stage);
        }
    }
    const Stage & stage(const std::string & name, size_t occurrence = 0) const {
        return stages_.at(name).at(occurrence);
    }
    bool has(const std::string & name) const { return stages_.count(name); }
    template<class T> std::vector<T> read(const Stage & stage) const {
        const bool integer = std::is_integral<T>::value;
        require(std::filesystem::path(stage.file).extension() == (integer ? ".i32" : ".f32"), "fixture type differs");
        std::ifstream input(root_ / stage.file, std::ios::binary); std::vector<T> data(stage.elements);
        require(bool(input.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T))), "fixture read failed");
        require(input.peek() == EOF, "fixture has extra bytes"); return data;
    }
    std::vector<float> values(const std::string & name, size_t occurrence = 0) const { return read<float>(stage(name, occurrence)); }
    std::vector<int> ids(const std::string & name) const { return read<int>(stage(name)); }
    std::vector<int> tokens() const {
        std::ifstream input(root_ / "token-ids.txt"); std::vector<int> values; int id;
        while (input >> id) values.push_back(id);
        require(!values.empty(), "token IDs missing"); return values;
    }
    bool cpu_f32_router() const {
        std::ifstream input(root_ / "precision.txt"); std::string mode;
        return bool(std::getline(input, mode)) && mode == "router=cpu-f32";
    }
};
class Checkpoint {
    std::vector<std::unique_ptr<MappedGguf>> files_;
    std::map<std::string, size_t> owners_;
public:
    Qwen4ExpConfig config;
    double epsilon = 0.;
    int key_heads = 0, value_heads = 0, state_dim = 0, conv_kernel = 0;
    int attention_heads = 0, attention_kvheads = 0, attention_dim = 0, rotary = 0;
    float theta = 0.f;
    bool cpu_q8_projection = false;
    // Decode mode: projections keep their loaded matrices, dense vectors and the head on the device.
    bool resident = false;
    struct Resident { Mat m; std::vector<std::unique_ptr<Device<char>>> mem; std::unique_ptr<Device<float>> dense; };
    mutable std::map<std::string, std::unique_ptr<Resident>> resident_;
    mutable std::map<std::string, std::unique_ptr<Device<float>>> dense_dev_;
    mutable cublasHandle_t blas_ = nullptr;
    // Projections that share one input convert it to FP16 once inside a SharedInput scope (same values as per call).
    mutable const float * half_src_ = nullptr; mutable size_t half_count_ = 0; mutable const __half * half_ = nullptr;
    struct SharedInput {
        const Checkpoint & cp; std::unique_ptr<Device<__half>> buf; bool set = false;
        // have: x already converted to FP16 by the caller (same conversion), reused instead of converting again
        SharedInput(const Checkpoint & c, const float * x, size_t count, __half * have = nullptr) : cp(c) {
            if (c.cpu_q8_projection || !c.resident) return;
            __half * h = have;
            if (!h) { buf = std::make_unique<Device<__half>>(count); tqllm_cuda::vis_f32_to_h(x, buf->p, int64_t(count), nullptr); h = buf->p; }
            c.half_src_ = x; c.half_count_ = count; c.half_ = h; set = true;
        }
        ~SharedInput() { if (set) { cp.half_src_ = nullptr; cp.half_ = nullptr; } }
    };
    bool validate_gsq_round = false;
    bool f32_q6_projection = false;
    bool f32_expert_output = false;
    int trace_layer = 0;
    Checkpoint(int argc, char ** argv) {
        std::set<int> parts; int expected = 0;
        GgufDialect dialect = GgufDialect::Native;
        for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--gsq-rco") dialect = GgufDialect::GsqRco;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--gsq-rco") continue;
            if (std::string(argv[i]) == "--check-gsq-round") { validate_gsq_round = true; continue; }
            if (std::string(argv[i]) == "--gsq-q6-f32" || std::string(argv[i]) == "--gsq-exact-q6") {
                f32_q6_projection = true; continue;
            }
            if (std::string(argv[i]) == "--gsq-f32-expert-output") { f32_expert_output = true; continue; }
            if (std::string(argv[i]) == "--mtp") {  // MTP block (blk.N + nextn.*) from its own native GGUF, outside the split set
                require(++i < argc, "--mtp requires a file");
                mtp_files_.push_back(std::make_unique<MappedGguf>(argv[i], GgufDialect::Native));
                continue;
            }
            if (std::string(argv[i]) == "--trace-layer") {
                require(++i < argc, "--trace-layer requires a layer number");
                const std::string value = argv[i];
                require(!value.empty() && std::all_of(value.begin(), value.end(), ::isdigit), "invalid trace layer");
                trace_layer = std::stoi(value);
                continue;
            }
            auto file = std::make_unique<MappedGguf>(argv[i], dialect); const auto & metadata = file->file();
            if (files_.empty()) {
                config = qwen4exp_config(metadata); expected = int(metadata.num("split.count", 1));
                epsilon = metadata.num("qwen4exp.attention.layer_norm_rms_epsilon");
                key_heads = int(metadata.num("qwen4exp.ssm.group_count"));
                value_heads = int(metadata.num("qwen4exp.ssm.time_step_rank"));
                state_dim = int(metadata.num("qwen4exp.ssm.state_size"));
                conv_kernel = int(metadata.num("qwen4exp.ssm.conv_kernel"));
                attention_heads = int(metadata.num("qwen4exp.attention.head_count"));
                attention_kvheads = int(metadata.num("qwen4exp.attention.head_count_kv"));
                attention_dim = int(metadata.num("qwen4exp.attention.key_length"));
                rotary = int(metadata.num("qwen4exp.rope.dimension_count"));
                theta = float(metadata.num("qwen4exp.rope.freq_base"));
            } else if (metadata.has("general.architecture")) require(qwen4exp_config(metadata) == config, "shard architecture differs");
            require(metadata.num("split.count", 1) == expected, "shard count differs");
            const int part = int(metadata.num("split.no", 0));
            require(part >= 0 && part < expected && parts.insert(part).second, "invalid shard ID");
            for (const auto & [name, tensor] : metadata.tensors)
                require(owners_.emplace(name, files_.size()).second, "duplicate tensor " + name);
            files_.push_back(std::move(file));
        }
        require(int(files_.size()) == expected && std::isfinite(epsilon) && epsilon > 0., "incomplete shards or invalid norm epsilon");
        for (auto & m : mtp_files_) {
            for (const auto & [name, tensor] : m->file().tensors)
                require(owners_.emplace(name, files_.size()).second, "duplicate tensor " + name);
            files_.push_back(std::move(m)); has_mtp = true;
        }
        mtp_files_.clear();
    }
    bool has_mtp = false;
    std::vector<std::unique_ptr<MappedGguf>> mtp_files_;
    const MappedGguf & file(const std::string & name) const { return *files_.at(owners_.at(name)); }
    bool has_tensor(const std::string & name) const { return owners_.count(name) > 0; }
    std::vector<float> dense(const std::string & name, size_t max_bytes = 32 * 1024 * 1024) const {
        const auto & f = file(name); const auto & t = f.file().tensors.at(name);
        size_t count = 1; for (auto dimension : t.ne) {
            require(dimension > 0 && dimension <= max_bytes / sizeof(float) / count,
                    "dense stage weight too large: " + name + " " + gguf_shape_string(t)); count *= dimension;
        }
        std::vector<float> data(count); const auto raw = f.tensor(name);
        dequantize_host(t.type, raw.data, data.data(), int64_t(count)); return data;
    }
    std::array<GgufBytes, 3> expert(int layer, int id) const {
        std::array<GgufBytes, 3> result;
        const std::array<std::string, 3> suffix{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (int i = 0; i < 3; ++i) {
            const auto name = "blk." + std::to_string(layer) + "." + suffix[i]; const auto & f = file(name);
            const auto & t = f.file().tensors.at(name);
            require(t.type == GGML_TQ3_4S && t.ne == std::vector<uint64_t>{uint64_t(i == 2 ? config.expert_ff : config.hidden),
                    uint64_t(i == 2 ? config.hidden : config.expert_ff), uint64_t(config.experts)}, "expert shape/type differs");
            result[i] = f.expert(name, id);
        }
        return result;
    }
    std::array<ExpertPayload, 3> expert_payload(int layer, int id) const {
        std::array<ExpertPayload, 3> result;
        const std::array<std::string, 3> suffix{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (int i = 0; i < 3; ++i) {
            const auto name = "blk." + std::to_string(layer) + "." + suffix[i]; const auto & f = file(name);
            const auto & t = f.file().tensors.at(name);
            require(t.ne == std::vector<uint64_t>{uint64_t(i == 2 ? config.expert_ff : config.hidden),
                uint64_t(i == 2 ? config.hidden : config.expert_ff), uint64_t(config.experts)}, "mixed expert shape differs");
            result[i] = {t.type, f.expert(name, id)};
        }
        return result;
    }
    std::vector<float> head_logits(const std::vector<float> & hidden) const {
        const auto & f = file("output.weight"); const auto & t = f.file().tensors.at("output.weight");
        require(t.ne.size() == 2 && t.ne[0] == uint64_t(config.hidden) && hidden.size() == t.ne[0], "head shape differs");
        const auto * format = weight_format(t.type);
        require(format && format->basis() == Basis::Plain, "chain diagnostic needs a native plain-basis head format");
        const auto raw = f.tensor("output.weight");
        std::vector<std::unique_ptr<Device<char>>> allocations;
        Upload upload = [&](const void * data, size_t bytes) {
            auto buffer = std::make_unique<Device<char>>(bytes);
            ck(cudaMemcpy(buffer->p, data, bytes, cudaMemcpyHostToDevice));
            const void * result = buffer->p; allocations.push_back(std::move(buffer)); return result;
        };
        const auto matrix = format->load(std::vector<char>(raw.data, raw.data + raw.size), int(t.ne[1]), config.hidden, upload, nullptr);
        Device<__half> input(hidden.size()); std::vector<__half> half(hidden.size());
        for (size_t j = 0; j < hidden.size(); ++j) half[j] = __float2half_rn(hidden[j]);
        input.put(half); Device<float> output(t.ne[1]);
        matrix.fmt->matvec_f32(matrix, input.p, output.p, 1, nullptr);
        return output.get();
    }
    // head logits for several hidden rows (row-major, `hidden` floats each) with one load of the output matrix
    // Resident raw BF16 tensor on the device (decode GEMVs read it directly).
    const __nv_bfloat16 * bf16_device(const std::string & name) const {
        const auto & f = file(name); const auto & t = f.file().tensors.at(name);
        require(t.type == GGML_BF16, "not a BF16 tensor: " + name);
        auto & r = resident_[name]; if (!r) r = std::make_unique<Resident>();
        if (r->mem.empty()) {
            PersistentAlloc keep; const auto raw = f.tensor(name); auto buffer = std::make_unique<Device<char>>(raw.size);
            ck(cudaMemcpy(buffer->p, raw.data, raw.size, cudaMemcpyHostToDevice)); r->mem.push_back(std::move(buffer));
        }
        return reinterpret_cast<const __nv_bfloat16 *>(r->mem[0]->p);
    }
    const float * dense_device(const std::string & name) const {
        auto & d = dense_dev_[name];
        if (!d) { PersistentAlloc keep; const auto v = dense(name); d = std::make_unique<Device<float>>(v.size()); d->put(v); }
        return d->p;
    }
    // Greedy head for the last row of a device [count][hidden] F32 block; the matrix stays resident.
    // Greedy head for every row of a device [count][hidden] F32 block in one weight read (speculative verify).
    std::vector<int> head_argmax_rows(const float * rows, int count) const {
        const auto * r = &head_resident();
        const size_t V = r->m.rows, H = config.hidden;
        Device<__half> input(size_t(count) * H); Device<float> output(size_t(count) * V);
        tqllm_cuda::vis_f32_to_h(rows, input.p, int64_t(count) * H, nullptr);
        const int step = std::max(1, r->m.fmt->max_matvec_cols());
        for (int b = 0; b < count; b += step)
            r->m.fmt->matvec_f32(r->m, input.p + size_t(b) * H, output.p + size_t(b) * V, std::min(step, count - b), nullptr);
        return argmax_rows(output.p, int(V), count);
    }
    // Device argmax per row (ties -> lowest index, as std::max_element): only the ids leave the GPU.
    static std::vector<int> argmax_rows(const float * logits, int V, int count) {
        std::vector<int> best(count);
        Device<uint32_t> ids(count), scratch(size_t(count) * 128 * 2);
        tqllm_cuda::argmax_cols(logits, V, count, ids.p, scratch.p, nullptr);
        const auto got = ids.get(); for (int t = 0; t < count; ++t) best[t] = int(got[t]);
        return best;
    }
    std::vector<float> head_logits_device(const float * rows, int count) const {
        const auto * r = &head_resident();
        const size_t V = r->m.rows, H = config.hidden;
        Device<__half> input(size_t(count) * H); Device<float> output(size_t(count) * V);
        tqllm_cuda::vis_f32_to_h(rows, input.p, int64_t(count) * H, nullptr);
        const int step = std::max(1, r->m.fmt->max_matvec_cols());
        for (int b = 0; b < count; b += step)
            r->m.fmt->matvec_f32(r->m, input.p + size_t(b) * H, output.p + size_t(b) * V, std::min(step, count - b), nullptr);
        return output.get();
    }
    int head_argmax_device(const float * rows, int count) const {
        const auto & r = head_resident();
        const size_t V = r.m.rows;
        Device<__half> input(config.hidden); Device<float> output(V);
        tqllm_cuda::vis_f32_to_h(rows + size_t(count - 1) * config.hidden, input.p, config.hidden, nullptr);
        r.m.fmt->matvec_f32(r.m, input.p, output.p, 1, nullptr);
        return argmax_rows(output.p, int(V), 1)[0];
    }
    // Draft-only head over a token subset (TQFN_DRAFT_VOCAB = "llama-mtp-vocab-v1 V K" header + K ids): the MTP draft
    // argmax reads K rows instead of the full vocabulary. Verification keeps the full head, so output is unchanged.
    const std::vector<int> * draft_vocab() const {
        static std::vector<int> ids; static bool tried = false;
        if (!tried) {
            tried = true;
            if (const char * path = std::getenv("TQFN_DRAFT_VOCAB")) {
                std::ifstream in(path); std::string magic; size_t V = 0, K = 0; in >> magic >> V >> K;
                require(magic == "llama-mtp-vocab-v1" && K > 0, std::string("bad draft vocabulary file ") + path);
                ids.resize(K); for (auto & v : ids) in >> v;
                require(bool(in), "draft vocabulary file truncated");
            }
        }
        return ids.empty() ? nullptr : &ids;
    }
    int draft_head_argmax_device(const float * rows, int count) const {
        const auto * ids = draft_vocab();
        if (!ids) return head_argmax_device(rows, count);
        auto & r = resident_["draft_head"]; if (!r) r = std::make_unique<Resident>();
        if (!r->m.fmt) {
            PersistentAlloc keep;
            const auto & f = file("output.weight"); const auto & t = f.file().tensors.at("output.weight");
            const auto raw = f.tensor("output.weight"); const size_t row_bytes = raw.size / size_t(t.ne[1]);
            std::vector<char> sub(ids->size() * row_bytes);
            for (size_t i = 0; i < ids->size(); ++i) {
                require((*ids)[i] >= 0 && uint64_t((*ids)[i]) < t.ne[1], "draft vocabulary id out of range");
                std::memcpy(sub.data() + i * row_bytes, raw.data + size_t((*ids)[i]) * row_bytes, row_bytes);
            }
            Upload upload = [&](const void * data, size_t bytes) {
                auto buffer = std::make_unique<Device<char>>(bytes);
                ck(cudaMemcpy(buffer->p, data, bytes, cudaMemcpyHostToDevice));
                const void * ptr = buffer->p; r->mem.push_back(std::move(buffer)); return ptr;
            };
            r->m = weight_format(t.type)->load(sub, int(ids->size()), config.hidden, upload, nullptr);
        }
        Device<__half> input(config.hidden); Device<float> output(ids->size());
        tqllm_cuda::vis_f32_to_h(rows + size_t(count - 1) * config.hidden, input.p, config.hidden, nullptr);
        r->m.fmt->matvec_f32(r->m, input.p, output.p, 1, nullptr);
        return (*ids)[size_t(argmax_rows(output.p, int(ids->size()), 1)[0])];
    }
    const Resident & head_resident() const {
        const auto & f = file("output.weight"); const auto & t = f.file().tensors.at("output.weight");
        auto & r = resident_["output.weight"]; if (!r) r = std::make_unique<Resident>();
        if (!r->m.fmt) {
            PersistentAlloc keep;
            const auto * format = weight_format(t.type);
            require(format && format->basis() == Basis::Plain, "decode needs a native plain-basis head format");
            Upload upload = [&](const void * data, size_t bytes) {
                auto buffer = std::make_unique<Device<char>>(bytes);
                ck(cudaMemcpy(buffer->p, data, bytes, cudaMemcpyHostToDevice));
                const void * ptr = buffer->p; r->mem.push_back(std::move(buffer)); return ptr;
            };
            const auto raw = f.tensor("output.weight");
            r->m = format->load(std::vector<char>(raw.data, raw.data + raw.size), int(t.ne[1]), config.hidden, upload, nullptr);
        }
        return *r;
    }
    std::vector<float> head_logits_rows(const std::vector<float> & rows, int count) const {
        const auto & f = file("output.weight"); const auto & t = f.file().tensors.at("output.weight");
        require(t.ne.size() == 2 && t.ne[0] == uint64_t(config.hidden) && rows.size() == size_t(count) * t.ne[0], "head shape differs");
        const auto * format = weight_format(t.type);
        require(format && format->basis() == Basis::Plain, "chain diagnostic needs a native plain-basis head format");
        const auto raw = f.tensor("output.weight");
        std::vector<std::unique_ptr<Device<char>>> allocations;
        Upload upload = [&](const void * data, size_t bytes) {
            auto buffer = std::make_unique<Device<char>>(bytes);
            ck(cudaMemcpy(buffer->p, data, bytes, cudaMemcpyHostToDevice));
            const void * result = buffer->p; allocations.push_back(std::move(buffer)); return result;
        };
        const auto matrix = format->load(std::vector<char>(raw.data, raw.data + raw.size), int(t.ne[1]), config.hidden, upload, nullptr);
        const size_t h = t.ne[0], V = t.ne[1];
        std::vector<float> out(size_t(count) * V);
        Device<__half> input(h); Device<float> output(V); std::vector<__half> half(h);
        for (int r = 0; r < count; ++r) {
            for (size_t j = 0; j < h; ++j) half[j] = __float2half_rn(rows[size_t(r) * h + j]);
            input.put(half);
            matrix.fmt->matvec_f32(matrix, input.p, output.p, 1, nullptr);
            const auto o = output.get();
            std::copy(o.begin(), o.end(), out.begin() + size_t(r) * V);
        }
        return out;
    }
    void project(const std::string & name, const float * input, float * output, int cols, int rows, int tokens) const {
        const auto & f = file(name); const auto & t = f.file().tensors.at(name);
        require(t.ne == std::vector<uint64_t>{uint64_t(cols), uint64_t(rows)},
                "stage projection shape/type differs: " + name);
        require(!owners_.count(name.substr(0, name.size() - 6) + "input_scale"), "HC input scale needs an explicit implementation");
        if (f32_q6_projection && t.type == GGML_Q6_K) {
            require(cols % 256 == 0, "Q6_K diagnostic needs 256-column blocks");
            std::vector<float> values(size_t(cols) * tokens);
            ck(cudaMemcpy(values.data(), input, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
            cpu_q8k_round(values);
            Device<float> x(values.size()), w(size_t(cols) * rows);
            x.put(values); w.put(dense(name, 128 * 1024 * 1024));
            cublasHandle_t handle = nullptr;
            require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cannot create Q6_K diagnostic handle");
            require(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) == CUBLAS_STATUS_SUCCESS,
                    "cannot select Q6_K diagnostic math mode");
            const float alpha = 1.f, beta = 0.f;
            const auto rc = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols, &alpha,
                                       w.p, cols, x.p, cols, &beta, output, rows);
            const auto finished = cudaDeviceSynchronize(); cublasDestroy(handle);
            require(rc == CUBLAS_STATUS_SUCCESS, "Q6_K diagnostic projection failed"); ck(finished);
            return;
        }
        if (t.type != GGML_TQ3_4S && t.type != GGML_F32 && t.type != GGML_BF16 && t.type != GGML_F16 && !exact_f32()) {
            const auto * format = weight_format(t.type);
            require(format && format->basis() == Basis::Plain, "native stage projection format unsupported: " + name);
            Resident local; Resident * slot = &local;
            if (resident) { auto & r = resident_[name]; if (!r) r = std::make_unique<Resident>(); slot = r.get(); }
            if (!slot->m.fmt) {
                PersistentAlloc keep_resident;
                Upload upload = [&](const void * data, size_t bytes) {
                    auto buffer = std::make_unique<Device<char>>(bytes);
                    ck(cudaMemcpy(buffer->p, data, bytes, cudaMemcpyHostToDevice));
                    const void * ptr = buffer->p; slot->mem.push_back(std::move(buffer)); return ptr;
                };
                const auto raw = f.tensor(name);
                slot->m = format->load(std::vector<char>(raw.data, raw.data + raw.size), rows, cols, upload, nullptr);
            }
            const Mat & matrix = slot->m;
            require(matrix.fmt && matrix.fmt->basis() == Basis::Plain, "stage loader changed basis");
            const bool shared = half_src_ == input && half_count_ == size_t(cols) * tokens;
            Device<__half> x(shared ? 1 : size_t(cols) * tokens);
            const __half * xp = shared ? half_ : x.p;
            if (shared) {
            } else if (cpu_q8_projection) {
                if (cols % 256 == 0 && t.type != GGML_IQ4_NL && !round32_gpu_side(name)) tqllm_cuda::gsq_q8k_round_h(input, x.p, x.count, nullptr);
                else tqllm_cuda::gsq_q8_round_h(input, x.p, x.count, nullptr);
                if (validate_gsq_round) {
                    std::vector<float> values(x.count);
                    ck(cudaMemcpy(values.data(), input, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    validate_round(x, std::move(values), cols % 256 == 0 && t.type != GGML_IQ4_NL, name);
                }
            } else {
                tqllm_cuda::vis_f32_to_h(input, x.p, int64_t(x.count), nullptr);
            }
            // TQFN_PROJ_GEMM (default on): prefill chunks above 64 tokens use the u4h-family GEMM (weights once per 64 tokens);
            // decode, verify and short chunks keep the exact matvec path
            if (tokens > 64 && matrix.fmt->gemm_f32(matrix, xp, output, tokens, nullptr)) {
                if (!resident) ck(cudaDeviceSynchronize());
                return;
            }
            const int step = matrix.fmt->max_matvec_cols(); require(step > 0, "invalid native matvec width");
            for (int base = 0; base < tokens; base += step)
                matrix.fmt->matvec_f32(matrix, xp + size_t(base) * cols, output + size_t(base) * rows, std::min(step, tokens - base), nullptr);
            if (!resident) ck(cudaDeviceSynchronize());
            return;
        }
        const auto raw = f.tensor(name); Device<char> weights(resident && t.type != GGML_TQ3_4S ? 1 : raw.size);
        if (t.type == GGML_TQ3_4S || (t.type == GGML_F32 && !resident)) ck(cudaMemcpy(weights.p, raw.data, raw.size, cudaMemcpyHostToDevice));
        if (t.type != GGML_TQ3_4S) {
            if (resident && t.type == GGML_BF16 && !exact_f32() && !bf16_act()) {
                // BF16 weights stay BF16; activations round to BF16 (llama.cpp's BF16 matmul does the same), F32 accumulate.
                auto & r = resident_[name]; if (!r) r = std::make_unique<Resident>();
                if (r->mem.empty()) {
                    PersistentAlloc keep; auto buffer = std::make_unique<Device<char>>(raw.size);
                    ck(cudaMemcpy(buffer->p, raw.data, raw.size, cudaMemcpyHostToDevice)); r->mem.push_back(std::move(buffer));
                }
                if (cols % 8 == 0 && tokens <= 64) {  // decode, verify blocks and short prefill chunks: one weight read per 4 tokens
                    tqllm_cuda::fn_gemv_bf16_tokens(reinterpret_cast<const __nv_bfloat16 *>(r->mem[0]->p), input, output, rows, cols, tokens, nullptr);
                    return;
                }
                Device<__nv_bfloat16> xb(size_t(cols) * tokens);
                tqllm_cuda::f32_to_bf16(input, xb.p, xb.count, nullptr);
                cublasHandle_t handle = blas_;
                if (!handle) { require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cannot create BF16 projection handle"); blas_ = handle; }
                const float alpha = 1.f, beta = 0.f;
                require(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols, &alpha, r->mem[0]->p, CUDA_R_16BF, cols,
                                     xb.p, CUDA_R_16BF, cols, &beta, output, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                        == CUBLAS_STATUS_SUCCESS, "BF16 stage projection failed: " + name);
                return;
            }
            std::unique_ptr<Device<float>> expanded;
            const float * matrix = reinterpret_cast<const float *>(weights.p);
            if (resident) {
                auto & r = resident_[name]; if (!r) r = std::make_unique<Resident>();
                if (!r->dense) { PersistentAlloc keep; r->dense = std::make_unique<Device<float>>(size_t(rows) * cols); r->dense->put(dense(name, 512 * 1024 * 1024)); }
                matrix = r->dense->p;
            } else if (t.type != GGML_F32) {
                expanded = std::make_unique<Device<float>>(size_t(rows) * cols);
                expanded->put(dense(name, 128 * 1024 * 1024)); matrix = expanded->p;
            }
            std::unique_ptr<Device<float>> rounded_input;
            if (bf16_act() && t.type == GGML_BF16) {
                rounded_input = std::make_unique<Device<float>>(size_t(cols) * tokens);
                ck(cudaMemcpy(rounded_input->p, input, size_t(cols) * tokens * sizeof(float), cudaMemcpyDeviceToDevice));
                round_bf16_device(rounded_input->p, rounded_input->count); input = rounded_input->p;
            }
            cublasHandle_t handle = blas_;
            if (!handle) require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cannot create F32 projection handle");
            const float alpha = 1.f, beta = 0.f;
            const auto rc = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols, &alpha,
                                      matrix, cols, input, cols, &beta, output, rows);
            const auto finished = resident ? cudaSuccess : cudaDeviceSynchronize();
            if (resident) blas_ = handle; else cublasDestroy(handle);
            require(rc == CUBLAS_STATUS_SUCCESS, "F32 stage projection failed"); ck(finished); return;
        }
        Device<float> rotated(size_t(cols) * tokens);
        tqllm_cuda::tq3_rotate_f32(input, rotated.p, cols, tokens, nullptr);
        if (cpu_q8_projection) {
            // Diagnostic CPU-reference contract: Q8 codes use the F32 scale, then store it as F16.
            auto values = rotated.get();
            cpu_q8_round(values);
            rotated.put(values);
        }
        tqllm_cuda::tq3_matvec_f32_raw(weights.p, rotated.p, output, rows, cols, tokens, nullptr);
        ck(cudaDeviceSynchronize());
    }
};
static void compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
    require(actual.size() == expected.size(), name + ": size differs");
    double squared = 0., norm = 0., maximum = 0.;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), name + ": nonfinite result");
        const double delta = double(actual[i]) - expected[i]; squared += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta));
    }
    const double relative = std::sqrt(squared / std::max(norm, 1e-30));
    std::cout << "STAGE " << name << " elements=" << actual.size() << " relative_l2=" << relative << " max_abs=" << maximum << '\n';
    require(relative <= tolerance, name + ": relative L2 exceeds prespecified screen tolerance");
}
static std::vector<float> bf16_values(const Device<__nv_bfloat16> & device) {
    const auto raw = device.get(); std::vector<float> result(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) result[i] = __bfloat162float(raw[i]); return result;
}
struct RecurrentState {
    std::unique_ptr<Device<float>> history, state;
    // speculative mode: state after every token, the block's conv inputs and the history before it (rollback)
    bool spec = false; int last_n = 0; std::unique_ptr<Device<float>> qkv_saved, old_history; std::unique_ptr<Device<float>> state2, history2;  // double buffers
    // TQFN_GDN_REPLAY: keep the block's GDN inputs; accept(m) reruns the first m tokens from state2 (the pre-block state)
    std::unique_ptr<Device<float>> conv_saved, a_saved, b_saved, replay_out;
    const float * neg_a = nullptr, * bias = nullptr; int kh = 0, vh = 0, dim = 0; float eps = 0.f;
};
#if defined(__x86_64__)
static const int8_t kCpuIq4nl[16] = {
#include "../kernels/kvalues_iq4nl.inc"
};
__attribute__((target("avx2,fma"))) static float cpu_dot_f32(const float * a, const float * b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(); int k = 0;
    for (; k + 16 <= n; k += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    q = _mm_add_ps(q, _mm_movehl_ps(q, q)); q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 1));
    float r = _mm_cvtss_f32(q); for (; k < n; ++k) r += a[k] * b[k]; return r;
}
// multi-column IQ4_NL: decode each block once for up to 4 activation vectors
__attribute__((target("avx2,fma"))) static void cpu_dot_iq4nl_q8_mc(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCpuIq4nl)));
    const __m256i low = _mm256_set1_epi8(0x0f), ones = _mm256_set1_epi16(1);
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0; b < nblocks; ++b, w += 18) {
        uint16_t dh; std::memcpy(&dh, w, 2); __half_raw hr; hr.x = dh; const float d = __half2float(__half(hr));
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + 2));
        const __m256i wv = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), low));
        const __m256i wa = _mm256_sign_epi8(wv, wv);
        for (int c = 0; c < nc; ++c) {
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xqc[c] + b * 32));
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(xv, wv)), ones);
            acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(d * xdc[c][b]), _mm256_cvtepi32_ps(p32), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc[c]), _mm256_extractf128_ps(acc[c], 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1)); out[c] = _mm_cvtss_f32(s);
    }
}
// x -> fp16-rounded fp32 and int8 per 32-block (vectorised; n % 32 == 0)
__attribute__((target("avx2,fma,f16c"))) static void cpu_prepare_x(const float * x, float * xr, int8_t * q, float * d, int n) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int b = 0; b < n / 32; ++b) {
        __m256 v[4], m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(x + b * 32 + 8 * i), _MM_FROUND_TO_NEAREST_INT));
            _mm256_storeu_ps(xr + b * 32 + 8 * i, v[i]); m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 l = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        l = _mm_max_ps(l, _mm_movehl_ps(l, l)); l = _mm_max_ss(l, _mm_shuffle_ps(l, l, 1));
        const float mx = _mm_cvtss_f32(l), sc = mx / 127.f; d[b] = sc;
        const __m256 inv = _mm256_set1_ps(sc > 0.f ? 1.f / sc : 0.f);
        __m256i i32[4]; for (int i = 0; i < 4; ++i) i32[i] = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], inv));
        const __m256i p16a = _mm256_packs_epi32(i32[0], i32[1]), p16b = _mm256_packs_epi32(i32[2], i32[3]);
        const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(q + b * 32), p8);
    }
}
// x as int8 per 32-block with one scale (CPU numerics)
static void cpu_quant_q8(const float * x, int n, int8_t * q, float * d) {
    for (int b = 0; b < n / 32; ++b) {
        float m = 0.f; for (int j = 0; j < 32; ++j) m = std::max(m, std::fabs(x[b * 32 + j]));
        const float s = m / 127.f, inv = s > 0.f ? 1.f / s : 0.f; d[b] = s;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = int8_t(std::lrintf(x[b * 32 + j] * inv));
    }
}
// one IQ4_NL row (18-byte blocks: fp16 d, 16 bytes of nibbles; low nibbles are elements 0-15) against a q8 vector
__attribute__((target("avx2,fma"))) static float cpu_dot_iq4nl_q8(const uint8_t * w, const int8_t * xq, const float * xd, int nblocks) {
    const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCpuIq4nl)));
    const __m256i low = _mm256_set1_epi8(0x0f), ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b, w += 18) {
        uint16_t dh; std::memcpy(&dh, w, 2);
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + 2));
        const __m256i nib = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), low);
        const __m256i wv = _mm256_shuffle_epi8(table, nib);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq + b * 32));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        const __m256i p32 = _mm256_madd_epi16(p16, ones);
        __half_raw hr; hr.x = dh;
        acc = _mm256_fmadd_ps(_mm256_set1_ps(__half2float(__half(hr)) * xd[b]), _mm256_cvtepi32_ps(p32), acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
static const uint32_t kCpuIq3xxs[256] = {
#include "../kernels/iq3xxs_grid.inc"
};
static const uint32_t kCpuIq3s[512] = {
#include "../kernels/iq3s_grid.inc"
};
static const uint64_t kCpuIq2s[1024] = {
#include "../kernels/iq2s_grid.inc"
};
// 32 sign bits -> 32 bytes of 0xff (negative) / 0x00
__attribute__((target("avx2"))) static inline __m256i cpu_sign_mask(uint32_t bits) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(bits)),
        _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0x0000000000000000LL));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ULL));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}
// unsigned magnitudes (32 bytes) with sign mask against 32 int8 activations -> 8 int32 partial sums (4 bytes each)
__attribute__((target("avx2"))) static inline __m256i cpu_mag_dot(__m256i mag, __m256i neg, const int8_t * xq) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq));
    const __m256i xs = _mm256_sub_epi8(_mm256_xor_si256(x, neg), neg);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(mag, xs), _mm256_set1_epi16(1));
}
__attribute__((target("avx2,fma"))) static inline float cpu_hsum(__m256 acc) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1)); return _mm_cvtss_f32(s);
}
static inline float cpu_h2f(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); __half_raw r; r.x = v; return __half2float(__half(r)); }
// IQ3_XXS row (98-byte blocks of 256) against q8 x
__attribute__((target("avx2,fma"))) static void cpu_dot_iq3xxs_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 98) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            uint32_t a; std::memcpy(&a, w + 66 + 4 * ib, 4);
            uint32_t sbits = 0;
            for (int l = 0; l < 4; ++l) { const uint32_t i7 = (a >> 7 * l) & 127; sbits |= (i7 | ((__builtin_popcount(i7) & 1) << 7)) << (8 * l); }
            const uint8_t * q = qs + ib * 8;
            const __m256i mag = _mm256_set_epi32(int(kCpuIq3xxs[q[7]]), int(kCpuIq3xxs[q[6]]), int(kCpuIq3xxs[q[5]]), int(kCpuIq3xxs[q[4]]),
                                                 int(kCpuIq3xxs[q[3]]), int(kCpuIq3xxs[q[2]]), int(kCpuIq3xxs[q[1]]), int(kCpuIq3xxs[q[0]]));
            const float db = d * (0.5f + float(a >> 28)) * 0.5f; const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(db * xdc[c][sb]), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// IQ3_S row (110-byte blocks)
__attribute__((target("avx2,fma"))) static void cpu_dot_iq3s_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 110) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2, * qh = w + 66, * signs = w + 74, * sc = w + 106;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            const uint8_t * q = qs + ib * 8; const int hb = qh[ib];
            int g[8]; for (int k = 0; k < 8; ++k) g[k] = int(kCpuIq3s[q[k] | (((hb >> k) & 1) << 8)]);
            const __m256i mag = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(g));
            uint32_t sbits; std::memcpy(&sbits, signs + ib * 4, 4);
            const float db = d * float(1 + 2 * ((sc[ib / 2] >> 4 * (ib % 2)) & 0xf)); const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_set1_ps(db * xdc[c][sb]), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// IQ2_S row (82-byte blocks); two scales per 32 (one per 16 weights)
__attribute__((target("avx2,fma"))) static void cpu_dot_iq2s_q8(const uint8_t * w, const int8_t * const * xqc, const float * const * xdc, int nc, int nblocks, float * out) {
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int b = 0, sb = 0; b < nblocks; ++b, w += 82) {
        const float d = cpu_h2f(w); const uint8_t * qs = w + 2, * signs = w + 34, * qh = w + 66, * sc = w + 74;
        for (int ib = 0; ib < 8; ++ib, ++sb) {
            int64_t g[4]; for (int l = 0; l < 4; ++l) g[l] = int64_t(kCpuIq2s[qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)]);
            const __m256i mag = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(g));
            uint32_t sbits; std::memcpy(&sbits, signs + ib * 4, 4);
            const float d0 = d * (0.5f + float(sc[ib] & 0xf)) * 0.25f, d1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
            const __m256 dd = _mm256_setr_ps(d0, d0, d0, d0, d1, d1, d1, d1); const __m256i neg = cpu_sign_mask(sbits);
            for (int c = 0; c < nc; ++c)
                acc[c] = _mm256_fmadd_ps(_mm256_mul_ps(dd, _mm256_set1_ps(xdc[c][sb])), _mm256_cvtepi32_ps(cpu_mag_dot(mag, neg, xqc[c] + sb * 32)), acc[c]);
        }
    }
    for (int c = 0; c < nc; ++c) out[c] = cpu_hsum(acc[c]);
}
// GSQ Q2_0 row (18-byte blocks of 64: fp16 d, 16 bytes of 2-bit codes, value (q - 1) * d; byte k holds elements 4k..4k+3)
// against q8 x permuted per 32 to [s * 8 + k] = x[4k + s], with xs = per-32 sum of the int8 x
__attribute__((target("avx2,fma"))) static float cpu_dot_q2_0_q8p(const uint8_t * w, const int8_t * xq, const float * xd, const int * xs, int nblocks) {
    __m256 acc = _mm256_setzero_ps(); const uint64_t m = 0x0303030303030303ULL; float tail = 0.f;
    for (int b = 0; b < nblocks; ++b, w += 18) {
        const float d = cpu_h2f(w);
        for (int h = 0; h < 2; ++h, xq += 32, ++xd, ++xs) {
            uint64_t v; std::memcpy(&v, w + 2 + 8 * h, 8);
            const __m256i q = _mm256_set_epi64x(int64_t((v >> 6) & m), int64_t((v >> 4) & m), int64_t((v >> 2) & m), int64_t(v & m));
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xq));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), _mm256_set1_epi16(1));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d * xd[0]), _mm256_cvtepi32_ps(p), acc);
            tail -= d * xd[0] * float(xs[0]);
        }
    }
    return cpu_hsum(acc) + tail;
}
static void cpu_permute_q2(const int8_t * xq, int n, int8_t * out, int * sums) {
    for (int b = 0; b < n / 32; ++b) {
        int sum = 0;
        for (int k = 0; k < 8; ++k) for (int s2 = 0; s2 < 4; ++s2) { const int8_t v = xq[b * 32 + 4 * k + s2]; out[b * 32 + s2 * 8 + k] = v; sum += v; }
        sums[b] = sum;
    }
}
// The CPU pool reads missed experts from the pinned packed copy (device layout, optionally tiled) instead of the GGUF
// mapping, so no file pages are touched: each row is rebuilt into its GGUF block bytes for the int8 kernels.
struct CpuPackedView {
    const char * map = nullptr; std::vector<uint64_t> base, block; std::vector<std::array<uint64_t, 6>> sizes;
    std::vector<std::array<int, 3>> fmt; bool tiled = false; int layers = 0;
};
static CpuPackedView * g_cpu_packed = nullptr;
static inline int64_t cpu_block_off(int r, int s, int nb, int bpb, bool tiled) {
    if (!tiled) return (int64_t(r) * nb + s) * bpb;
    return (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
}
// GGUF bytes of row r (cols weights) of one device-layout matrix; returns nullptr if the format is not handled
static const char * cpu_gguf_row(int fmt, const char * dev, const char * aux, int r, int cols, bool tiled, char * out) {
    const int nb = cols / 32, kb = cols / 256;
    auto blk32 = [&](int s, int bpb) { return reinterpret_cast<const uint8_t *>(dev + cpu_block_off(r, s, nb, bpb, tiled)); };
    if (fmt == 5) {  // IQ3_XXS 98 B: d, qs[64], signs/scales[32]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 98; const char * rec = aux + (int64_t(r) * kb + b) * 4;
            std::memcpy(o, rec, 2);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 12); std::memcpy(o + 2 + ib * 8, d, 8); std::memcpy(o + 66 + ib * 4, d + 8, 4); }
        }
        return out;
    }
    if (fmt == 4) {  // IQ3_S 110 B: d, qs[64], qh[8], signs[32], scales[4]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 110; const char * rec = aux + (int64_t(r) * kb + b) * 16;
            std::memcpy(o, rec, 2); std::memcpy(o + 66, rec + 2, 8); std::memcpy(o + 106, rec + 10, 4);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 12); std::memcpy(o + 2 + ib * 8, d, 8); std::memcpy(o + 74 + ib * 4, d + 8, 4); }
        }
        return out;
    }
    if (fmt == 10) {  // IQ2_S 82 B: d, qs[32], signs[32], qh[8], scales[8]
        for (int b = 0; b < kb; ++b) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + b * 82; const char * rec = aux + (int64_t(r) * kb + b) * 20;
            std::memcpy(o, rec, 2); std::memcpy(o + 66, rec + 2, 16);
            for (int ib = 0; ib < 8; ++ib) { const uint8_t * d = blk32(b * 8 + ib, 8); std::memcpy(o + 2 + ib * 4, d, 4); std::memcpy(o + 34 + ib * 4, d + 4, 4); }
        }
        return out;
    }
    if (fmt == 14) {  // IQ4_NL 18 B per 32: d, qs[16] (qs[i] = e[i] | e[i + 16] << 4); device: natural nibble pairs + fp16 d
        const auto * dh = reinterpret_cast<const uint16_t *>(aux) + int64_t(r) * nb;
        for (int s = 0; s < nb; ++s) {
            uint8_t * o = reinterpret_cast<uint8_t *>(out) + s * 18; const uint8_t * d = blk32(s, 16);
            std::memcpy(o, dh + s, 2);
            auto e = [&](int i) { return (d[i / 2] >> (4 * (i & 1))) & 0xF; };
            for (int i = 0; i < 16; ++i) o[2 + i] = uint8_t(e(i) | (e(i + 16) << 4));
        }
        return out;
    }
    return nullptr;
}
// TQFN_CPU_MISS=T: experts that are not resident in the GPU cache are computed by T pinned CPU threads while the GPU
// runs the hits (full routing, CPU numerics: not bit-identical to the GPU path). Rows come from the GGUF bytes.
class CpuMissPool {
public:
    struct Job { const char * w[3]; uint32_t type[3]; size_t row_bytes[3]; int cols_t[8]; int ncols = 0; int slot = 0;
                 const char * dev[3] = {nullptr, nullptr, nullptr}, * aux[3] = {nullptr, nullptr, nullptr}; int fmt[3] = {-1, -1, -1}; bool tiled = false; };
    CpuMissPool(int threads, int hidden, int ff) : hidden_(hidden), ff_(ff) {
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this, i] { run(i); });
        for (int i = 0; i < threads; ++i) {
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(1 + i, &set);
            pthread_setaffinity_np(workers_[i].native_handle(), sizeof(set), &set);
        }
        cpu_set_t host; CPU_ZERO(&host); CPU_SET(0, &host); pthread_setaffinity_np(pthread_self(), sizeof(host), &host);
    }
    ~CpuMissPool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; gen_.fetch_add(1); } cv_.notify_all(); for (auto & w : workers_) w.join();
        if (calls_) std::printf("CPU_MISS pool %.3f ms per layer call busy, host blocked %.3f ms per call (%llu calls)\n",
                                busy_ms_ / calls_, blocked_ms_ / calls_, (unsigned long long) calls_);
        for (auto & [t, e] : check_gu_) std::printf("CPU_CHECK gate/up type %u int8 vs fp32: max abs diff %.3g, max |ref| %.3g\n", t, e.first, e.second);
        if (std::getenv("TQFN_CPU_CHECK")) std::printf("CPU_CHECK packed-row rebuild mismatches vs GGUF: %llu\n", (unsigned long long) row_mismatch_);
        if (check_mag_ > 0) std::printf("CPU_CHECK down int8 vs fp32: max abs diff %.3g, max |ref| %.3g\n", check_err_, check_mag_);
    }
    // x: n tokens x hidden (fp32, rounded to fp16 like the GPU input); out: one bf16 row of `hidden` per job column.
    void start(const float * x, int n, std::vector<Job> jobs, __nv_bfloat16 * out) {
        x_.assign(size_t(n) * hidden_, 0.f);
        xq_.resize(x_.size()); xd_.resize(x_.size() / 32);
        cpu_prepare_x(x, x_.data(), xq_.data(), xd_.data(), int(x_.size()));
        jobs_ = std::move(jobs); out_ = out; int cols = 0;
        for (auto & j : jobs_) { j.slot = cols; cols += j.ncols; }
        act_.assign(size_t(cols) * ff_, 0.f);
        // about two items per worker in each phase, whatever the number of experts (load balance)
        const int k = std::max<int>(1, int(jobs_.size())), tw = 2 * int(workers_.size());
        chunks_[0] = std::clamp((tw + k - 1) / k, 4, 64); chunks_[1] = std::clamp((tw + k - 1) / k, 4, 128);
        phase_items_[0] = int(jobs_.size()) * chunks_[0]; phase_items_[1] = int(jobs_.size()) * chunks_[1];
        next_.store(0); done_.store(0); phase_.store(0); pending_ = !jobs_.empty();
        if (pending_) { t_start_ = std::chrono::steady_clock::now(); { std::lock_guard<std::mutex> l(m_); gen_.fetch_add(1, std::memory_order_release); } if (sleepers_.load()) cv_.notify_all(); }
    }
    void wait() {
        if (!pending_) return;
        const auto t0 = std::chrono::steady_clock::now();
        while (done_.load(std::memory_order_acquire) < phase_items_[0] + phase_items_[1]) _mm_pause();
        const auto t1 = std::chrono::steady_clock::now();
        busy_ms_ += std::chrono::duration<double, std::milli>(t_last_.load() > t_start_ ? t_last_.load() - t_start_ : t1 - t_start_).count();
        blocked_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count(); ++calls_;
        pending_ = false;
    }
private:
    int chunks_[2] = {20, 32};
    // the GGUF bytes of row r of matrix m: rebuilt from the pinned packed copy when available, else the GGUF mapping
    const char * row_src(const Job & j, int m, int r, int cols) {
        const char * gguf = j.w[m] + size_t(r) * j.row_bytes[m];
        if (!j.dev[m]) return gguf;
        if (j.type[m] == GGML_GSQ_Q2_0) return j.dev[m] + size_t(r) * j.row_bytes[m];  // device layout == GGUF rows
        thread_local std::vector<char> rb[3]; rb[m].resize(j.row_bytes[m]);
        const char * q = cpu_gguf_row(j.fmt[m], j.dev[m], j.aux[m], r, cols, j.tiled, rb[m].data());
        if (!q) return gguf;
        static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
        if (check && std::memcmp(q, gguf, j.row_bytes[m]) != 0) { std::lock_guard<std::mutex> l(m_); ++row_mismatch_; }
        return q;
    }
    void work(int item) {
        const bool down = item >= phase_items_[0];
        const int it = down ? item - phase_items_[0] : item, chunks = down ? chunks_[1] : chunks_[0];
        const Job & j = jobs_[it / chunks]; const int part = it % chunks;
        thread_local std::vector<float> row; row.resize(down ? ff_ : hidden_);
        if (!down) {
            const int r0 = part * ff_ / chunks, r1 = (part + 1) * ff_ / chunks;
            thread_local std::vector<float> g; g.assign(size_t(j.ncols) * (r1 - r0), 0.f);
            using DotQ8 = void (*)(const uint8_t *, const int8_t * const *, const float * const *, int, int, float *);
            const int8_t * xqc[8]; const float * xdc[8];
            for (int c = 0; c < j.ncols; ++c) { xqc[c] = xq_.data() + size_t(j.cols_t[c]) * hidden_; xdc[c] = xd_.data() + size_t(j.cols_t[c]) * hidden_ / 32; }
            for (int m = 0; m < 2; ++m) {
                const DotQ8 fast = hidden_ % 256 ? nullptr : j.type[m] == 18 ? cpu_dot_iq3xxs_q8 : j.type[m] == 21 ? cpu_dot_iq3s_q8 : j.type[m] == 22 ? cpu_dot_iq2s_q8 : nullptr;
                for (int r = r0; r < r1; ++r) {
                    const char * wr = row_src(j, m, r, hidden_);
                    float fo[8];
                    if (fast) for (int c0 = 0; c0 < j.ncols; c0 += 4) fast(reinterpret_cast<const uint8_t *>(wr), xqc + c0, xdc + c0, std::min(4, j.ncols - c0), hidden_ / 256, fo + c0);
                    else dequantize_host(j.type[m], wr, row.data(), hidden_);
                    for (int c = 0; c < j.ncols; ++c) {
                        const int t = j.cols_t[c];
                        const float acc = fast ? fo[c] : cpu_dot_f32(row.data(), x_.data() + size_t(t) * hidden_, hidden_);
                        static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
                        if (check && fast) {
                            std::vector<float> ref(hidden_); dequantize_host(j.type[m], wr, ref.data(), hidden_);
                            const float want = cpu_dot_f32(ref.data(), x_.data() + size_t(t) * hidden_, hidden_);
                            std::lock_guard<std::mutex> l(m_); auto & e = check_gu_[j.type[m]];
                            e.first = std::max(e.first, double(std::fabs(acc - want))); e.second = std::max(e.second, double(std::fabs(want)));
                        }
                        const float v = __bfloat162float(__float2bfloat16_rn(acc));
                        float & g_or_out = g[size_t(c) * (r1 - r0) + (r - r0)];
                        if (m == 0) g_or_out = v;
                        else act_[size_t(j.slot + c) * ff_ + r] = __half2float(__float2half_rn(g_or_out / (1.f + std::exp(-g_or_out)) * v));
                    }
                }
            }
        } else {
            const int r0 = part * hidden_ / chunks, r1 = (part + 1) * hidden_ / chunks;
            const bool nl = j.type[2] == 20 && ff_ % 32 == 0, q2 = j.type[2] == GGML_GSQ_Q2_0 && ff_ % 64 == 0;
            thread_local std::vector<int8_t> aq, ap; thread_local std::vector<float> ad; thread_local std::vector<int> as;
            if (nl || q2) {
                aq.resize(size_t(j.ncols) * ff_); ad.resize(size_t(j.ncols) * ff_ / 32);
                for (int c = 0; c < j.ncols; ++c) cpu_quant_q8(act_.data() + size_t(j.slot + c) * ff_, ff_, aq.data() + size_t(c) * ff_, ad.data() + size_t(c) * ff_ / 32);
            }
            if (q2) {
                ap.resize(aq.size()); as.resize(ad.size());
                for (int c = 0; c < j.ncols; ++c) cpu_permute_q2(aq.data() + size_t(c) * ff_, ff_, ap.data() + size_t(c) * ff_, as.data() + size_t(c) * ff_ / 32);
            }
            for (int r = r0; r < r1; ++r) {
                const char * wr = row_src(j, 2, r, ff_);
                if (!nl && !q2) dequantize_host(j.type[2], wr, row.data(), ff_);
                float nlo[8];
                if (nl) {
                    const int8_t * xqc[8]; const float * xdc[8];
                    for (int c = 0; c < j.ncols; ++c) { xqc[c] = aq.data() + size_t(c) * ff_; xdc[c] = ad.data() + size_t(c) * ff_ / 32; }
                    for (int c0 = 0; c0 < j.ncols; c0 += 4) cpu_dot_iq4nl_q8_mc(reinterpret_cast<const uint8_t *>(wr), xqc + c0, xdc + c0, std::min(4, j.ncols - c0), ff_ / 32, nlo + c0);
                }
                for (int c = 0; c < j.ncols; ++c) {
                    const auto * wu = reinterpret_cast<const uint8_t *>(wr);
                    const float acc = nl ? nlo[c]
                                    : q2 ? cpu_dot_q2_0_q8p(wu, ap.data() + size_t(c) * ff_, ad.data() + size_t(c) * ff_ / 32, as.data() + size_t(c) * ff_ / 32, ff_ / 64)
                                         : cpu_dot_f32(row.data(), act_.data() + size_t(j.slot + c) * ff_, ff_);
                    out_[size_t(j.slot + c) * hidden_ + r] = __float2bfloat16_rn(acc);
                    static const bool check = std::getenv("TQFN_CPU_CHECK") != nullptr;
                    if (check && (nl || q2)) {  // int8 kernel vs dequantize_host + fp32 dot
                        std::vector<float> ref(ff_); dequantize_host(j.type[2], wr, ref.data(), ff_);
                        const float want = cpu_dot_f32(ref.data(), act_.data() + size_t(j.slot + c) * ff_, ff_);
                        std::lock_guard<std::mutex> l(m_); check_err_ = std::max(check_err_, double(std::fabs(acc - want)));
                        check_mag_ = std::max(check_mag_, double(std::fabs(want)));
                    }
                }
            }
        }
    }
    void run(int) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            const auto idle0 = std::chrono::steady_clock::now();
            for (int spin = 0; (g = gen_.load(std::memory_order_acquire)) == seen; ++spin) {
                _mm_pause();
                // decode calls arrive every ~0.5 ms; after 2 ms idle (prefill, between prompts) sleep instead of spinning
                static const int spin_ms = std::getenv("TQFN_CPU_SPIN_MS") ? std::atoi(std::getenv("TQFN_CPU_SPIN_MS")) : 2;
                if ((spin & 1023) == 1023 && std::chrono::steady_clock::now() - idle0 > std::chrono::milliseconds(spin_ms)) {
                    std::unique_lock<std::mutex> l(m_); ++sleepers_;
                    cv_.wait(l, [&] { return gen_.load() != seen; }); --sleepers_;
                }
            }
            seen = g; if (stop_) return;
            for (;;) {
                const int ph = phase_.load(std::memory_order_acquire);
                const int total = ph == 0 ? phase_items_[0] : phase_items_[0] + phase_items_[1];
                const int item = next_.fetch_add(1);
                if (item >= phase_items_[0] + phase_items_[1]) break;
                if (item >= phase_items_[0])  // down needs every gate/up item finished
                    while (done_.load(std::memory_order_acquire) < phase_items_[0]) _mm_pause();
                (void) total; work(item);
                if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == phase_items_[0] + phase_items_[1]) t_last_.store(std::chrono::steady_clock::now());
            }
        }
    }
    int hidden_, ff_; std::vector<std::thread> workers_; std::atomic<uint64_t> gen_{0}; std::atomic<int> next_{0}, done_{0}, phase_{0};
    std::atomic<bool> stop_{false}; bool pending_ = false; int phase_items_[2] = {0, 0};
    std::mutex m_; std::condition_variable cv_; std::atomic<int> sleepers_{0};
    std::chrono::steady_clock::time_point t_start_; std::atomic<std::chrono::steady_clock::time_point> t_last_{};
    double busy_ms_ = 0, blocked_ms_ = 0, check_err_ = 0, check_mag_ = 0; uint64_t calls_ = 0, row_mismatch_ = 0;
    std::vector<float> x_, act_, xd_; std::vector<int8_t> xq_; std::map<uint32_t, std::pair<double, double>> check_gu_; std::vector<Job> jobs_; __nv_bfloat16 * out_ = nullptr;
};
#else  // not x86: no AVX2 CPU expert kernels; the CPU miss pool is unavailable (TQFN_CPU_MISS refuses to start)
struct CpuPackedView { const char * map = nullptr; std::vector<uint64_t> base, block; std::vector<std::array<uint64_t, 6>> sizes;
                       std::vector<std::array<int, 3>> fmt; bool tiled = false; int layers = 0; };
static CpuPackedView * g_cpu_packed = nullptr;
class CpuMissPool {
public:
    struct Job { const char * w[3]; uint32_t type[3]; size_t row_bytes[3]; int cols_t[8]; int ncols = 0; int slot = 0;
                 const char * dev[3] = {nullptr, nullptr, nullptr}, * aux[3] = {nullptr, nullptr, nullptr}; int fmt[3] = {-1, -1, -1}; bool tiled = false; };
    CpuMissPool(int, int, int) { throw std::runtime_error("TQFN_CPU_MISS needs an x86 CPU with AVX2"); }
    void start(const float *, int, std::vector<Job>, __nv_bfloat16 *) {}
    void wait() {}
};
#endif
struct AttentionState { std::unique_ptr<Device<__nv_bfloat16>> k, v; int len = 0;
                        std::unique_ptr<tqllm_cuda::QsaIndexer> qsa; };  // QSA indexer keys (beyond the indexer budget)
// pre: the row-wise projections of this sequence's rows were done for a whole batch (qkv, z, alpha, beta slices);
// raw_out: write the recurrence output there and stop before the gated norm and the output projection (also batched)
struct GdnPre { const float * qkv, * z, * a, * b; };
static void recurrent_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, RecurrentState * carry, const GdnPre * pre = nullptr, float * raw_out = nullptr);
static std::vector<float> recurrent_forward(const Checkpoint & checkpoint, const std::vector<float> & mixed,
                                        int layer, const Fixtures * fixtures = nullptr, RecurrentState * carry = nullptr) {
    const int n = int(mixed.size() / checkpoint.config.hidden);
    Device<float> input(mixed.size()), out(size_t(n) * checkpoint.config.hidden); input.put(mixed);
    recurrent_core(checkpoint, input.p, out.p, n, layer, fixtures, carry); return out.get();
}
static void recurrent_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, RecurrentState * carry, const GdnPre * pre, float * raw_out) {
    const auto & c = checkpoint.config;
    const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
    const int kh = checkpoint.key_heads, vh = checkpoint.value_heads, dim = checkpoint.state_dim;
    const int channels = (2 * kh + vh) * dim, taps = checkpoint.conv_kernel;
    require(kh > 0 && vh % kh == 0 && dim == 128 && taps == 4, "unsupported recurrent configuration");
    struct In { const float * p; } input{input_p};
    auto mk = [](const float * p, size_t count) {
        return p ? std::make_unique<Device<float>>(Device<float>::View{}, const_cast<float *>(p), count) : std::make_unique<Device<float>>(count);
    };
    auto qkv_d = mk(pre ? pre->qkv : nullptr, size_t(n) * channels), z_d = mk(pre ? pre->z : nullptr, size_t(n) * vh * dim);
    auto a_d = mk(pre ? pre->a : nullptr, size_t(n) * vh), b_d = mk(pre ? pre->b : nullptr, size_t(n) * vh);
    auto & qkv = *qkv_d; auto & z = *z_d; auto & a = *a_d; auto & b = *b_d;
    if (!pre) {
        const Checkpoint::SharedInput shared_in(checkpoint, input.p, size_t(n) * c.hidden);
        checkpoint.project(prefix + "attn_qkv.weight", input.p, qkv.p, c.hidden, channels, n);
        checkpoint.project(prefix + "attn_gate.weight", input.p, z.p, c.hidden, vh * dim, n);
        checkpoint.project(prefix + "ssm_alpha.weight", input.p, a.p, c.hidden, vh, n);
        checkpoint.project(prefix + "ssm_beta.weight", input.p, b.p, c.hidden, vh, n);
    }
    if (fixtures && fixtures->has("linear_attn_qkv_mixed" + suffix)) {
        compare("gdn_qkv_projection" + suffix, qkv.get(), fixtures->values("linear_attn_qkv_mixed" + suffix), .02);
        compare("gdn_z_projection" + suffix, z.get(), fixtures->values("z" + suffix), .02);
        const double dense_tolerance = checkpoint.cpu_q8_projection ? .002 : 2e-5;
        compare("gdn_alpha_projection" + suffix, a.get(), fixtures->values("alpha" + suffix), dense_tolerance);
        compare("gdn_beta_projection" + suffix, b.get(), fixtures->values("beta" + suffix), dense_tolerance);
    }
    const float * conv_weights = checkpoint.dense_device(prefix + "ssm_conv1d.weight");
    Device<float> history(size_t(taps - 1) * channels), next_history(history.count), convolved(qkv.count);
    if (carry && !carry->history) {
        PersistentAlloc keep;
        carry->history = std::make_unique<Device<float>>(history.count);
        ck(cudaMemset(carry->history->p, 0, history.count * sizeof(float)));
    }
    if (!carry) ck(cudaMemsetAsync(history.p, 0, history.count * sizeof(float), nullptr));
    if (carry && carry->spec) {
        PersistentAlloc keep;
        if (!carry->old_history) carry->old_history = std::make_unique<Device<float>>(history.count);
        if (!carry->qkv_saved || carry->qkv_saved->count < qkv.count) carry->qkv_saved = std::make_unique<Device<float>>(qkv.count);
        ck(cudaMemcpyAsync(carry->old_history->p, carry->history->p, history.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(carry->qkv_saved->p, qkv.p, qkv.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        carry->last_n = n;
    }
    // The conv history and recurrent state go to a second buffer and the pointers swap (no D2D copy).
    if (carry && !carry->history2) { PersistentAlloc keep; carry->history2 = std::make_unique<Device<float>>(carry->history->count); }
    float * hist_out = carry ? carry->history2->p : next_history.p;
    tqllm_cuda::ple_conv(qkv.p, carry ? carry->history->p : history.p, conv_weights, convolved.p, hist_out, channels, n, taps, 1, nullptr);
    if (carry) std::swap(carry->history, carry->history2);
    if (fixtures && fixtures->has("conv_output_silu" + suffix))
        compare("gdn_convolution" + suffix, convolved.get(), fixtures->values("conv_output_silu" + suffix), .02);
    const size_t state_count = size_t(vh) * dim * dim, temp_count = carry && !fixtures ? 1 : state_count;  // unused with carry + swap
    Device<float> state(temp_count), next_state(temp_count);
    auto out_d = raw_out ? std::make_unique<Device<float>>(Device<float>::View{}, raw_out, size_t(n) * vh * dim) : std::make_unique<Device<float>>(size_t(n) * vh * dim);
    auto & out = *out_d;
    const float * neg_a = checkpoint.dense_device(prefix + "ssm_a"), * bias = checkpoint.dense_device(prefix + "ssm_dt.bias");
    if (carry && !carry->state) {
        PersistentAlloc keep;
        carry->state = std::make_unique<Device<float>>(state_count);
        ck(cudaMemset(carry->state->p, 0, state_count * sizeof(float)));
    }
    if (!carry) ck(cudaMemsetAsync(state.p, 0, state.count * sizeof(float), nullptr));
    if (carry && !carry->state2) { PersistentAlloc keep; carry->state2 = std::make_unique<Device<float>>(state_count); }
    float * state_out = carry ? carry->state2->p : next_state.p;
    if (carry && carry->spec) {
        PersistentAlloc keep;
        auto keep_copy = [](std::unique_ptr<Device<float>> & dst, const Device<float> & src) {
            if (!dst || dst->count < src.count) dst = std::make_unique<Device<float>>(src.count);
            ck(cudaMemcpyAsync(dst->p, src.p, src.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        };
        keep_copy(carry->conv_saved, convolved); keep_copy(carry->a_saved, a); keep_copy(carry->b_saved, b);
        if (!carry->replay_out || carry->replay_out->count < out.count) carry->replay_out = std::make_unique<Device<float>>(out.count);
        carry->neg_a = neg_a; carry->bias = bias; carry->kh = kh; carry->vh = vh; carry->dim = dim; carry->eps = float(checkpoint.epsilon);
        tqllm_cuda::fn_gdn(convolved.p, a.p, b.p, neg_a, bias, carry->state->p, state_out, out.p,
                          kh, vh, dim, n, float(checkpoint.epsilon), nullptr);
    } else
    tqllm_cuda::fn_gdn(convolved.p, a.p, b.p, neg_a, bias, carry ? carry->state->p : state.p, state_out, out.p,
                      kh, vh, dim, n, float(checkpoint.epsilon), nullptr);
    if (carry) std::swap(carry->state, carry->state2);
    if (fixtures) compare("gdn_recurrent_output" + suffix, out.get(), fixtures->values("attn_output" + suffix), .02);
    if (fixtures && fixtures->has("new_state" + suffix))
        compare("gdn_recurrent_state" + suffix, next_state.get(), fixtures->values("new_state" + suffix), .02);
    if (raw_out) return;
    Device<float> gated(out.count); struct Out { float * p; } projected{result};
    tqllm_cuda::fn_gated_rms(out.p, z.p, checkpoint.dense_device(prefix + "ssm_norm.weight"), gated.p, vh, dim, n, float(checkpoint.epsilon), nullptr);
    if (fixtures && fixtures->has("final_output" + suffix))
        compare("gdn_sigmoid_norm" + suffix, gated.get(), fixtures->values("final_output" + suffix), .02);
    checkpoint.project(prefix + "ssm_out.weight", gated.p, projected.p, vh * dim, c.hidden, n);
    if (const char * dump = std::getenv("TQFN_DUMP_GDN")) {
        std::vector<float> pv(size_t(n) * c.hidden); ck(cudaMemcpy(pv.data(), result, pv.size() * 4, cudaMemcpyDeviceToHost));
        for (const auto & [tag, values] : {std::pair<std::string, std::vector<float>>{"gated", gated.get()}, {"projected", pv}}) {
            std::ofstream file(std::filesystem::path(dump) / ("gdn-" + tag + suffix + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
        }
    }
    if (fixtures) compare("gdn_projected_output" + suffix, [&] { std::vector<float> v(size_t(n) * c.hidden); ck(cudaMemcpy(v.data(), result, v.size() * 4, cudaMemcpyDeviceToHost)); return v; }(), fixtures->values("linear_attn_out" + suffix), .02);
}
// pre: q (with gate), k, v of this sequence's rows projected for a whole batch; raw_out: write the attention output
// there and stop before the gate and the output projection (both row-wise, done for the batch)
struct AttnPre { const float * qg, * k, * v; };
static void attention_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, AttentionState * carry, const AttnPre * pre = nullptr, float * raw_out = nullptr);
static std::vector<float> attention_forward(const Checkpoint & checkpoint, const std::vector<float> & mixed,
                                        int layer, const Fixtures * fixtures = nullptr, AttentionState * carry = nullptr) {
    const int n = int(mixed.size() / checkpoint.config.hidden);
    Device<float> input(mixed.size()), out(size_t(n) * checkpoint.config.hidden); input.put(mixed);
    attention_core(checkpoint, input.p, out.p, n, layer, fixtures, carry); return out.get();
}
static void attention_core(const Checkpoint & checkpoint, const float * input_p, float * result, int n,
                           int layer, const Fixtures * fixtures, AttentionState * carry, const AttnPre * pre, float * raw_out) {
    const auto & c = checkpoint.config;
    const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
    const int heads = checkpoint.attention_heads, kvheads = checkpoint.attention_kvheads, dim = checkpoint.attention_dim;
    require(n > 0 && n <= 2048, "dense fixture exceeds QSA budget");
    struct In { const float * p; } input{input_p};
    auto mk = [](const float * p, size_t count) {
        return p ? std::make_unique<Device<float>>(Device<float>::View{}, const_cast<float *>(p), count) : std::make_unique<Device<float>>(count);
    };
    auto qg_d = mk(pre ? pre->qg : nullptr, size_t(n) * heads * dim * 2), k_d = mk(pre ? pre->k : nullptr, size_t(n) * kvheads * dim);
    auto v_d = mk(pre ? pre->v : nullptr, size_t(n) * kvheads * dim);
    auto & qg = *qg_d; auto & k = *k_d; auto & v = *v_d;
    Device<float> q(size_t(n) * heads * dim), kn(k.count);
    if (!pre) {
        const Checkpoint::SharedInput shared_in(checkpoint, input.p, size_t(n) * c.hidden);
        checkpoint.project(prefix + "attn_q.weight", input.p, qg.p, c.hidden, heads * dim * 2, n);
        checkpoint.project(prefix + "attn_k.weight", input.p, k.p, c.hidden, kvheads * dim, n);
        checkpoint.project(prefix + "attn_v.weight", input.p, v.p, c.hidden, kvheads * dim, n);
    }
    if (fixtures && fixtures->has("Qcur_full" + suffix)) {
        compare("attention_query_projection" + suffix, qg.get(), fixtures->values("Qcur_full" + suffix), .02);
        compare("attention_key_projection" + suffix, k.get(), fixtures->values("Kcur" + suffix, 0), .02);
        compare("attention_value_projection" + suffix, v.get(), fixtures->values("Vcur" + suffix, 0), .02);
    }
    const int pos = carry ? carry->len : 0;
    tqllm_cuda::fn_norm_rope(qg.p, checkpoint.dense_device(prefix + "attn_q_norm.weight"), q.p, heads, dim, n, pos, checkpoint.rotary, checkpoint.theta,
                            float(checkpoint.epsilon), true, nullptr);
    tqllm_cuda::fn_norm_rope(k.p, checkpoint.dense_device(prefix + "attn_k_norm.weight"), kn.p, kvheads, dim, n, pos, checkpoint.rotary, checkpoint.theta,
                            float(checkpoint.epsilon), false, nullptr);
    Device<__nv_bfloat16> kb(k.count), vb(v.count);
    tqllm_cuda::f32_to_bf16(kn.p, kb.p, kn.count, nullptr);
    tqllm_cuda::f32_to_bf16(v.p, vb.p, v.count, nullptr);
    auto out_d = raw_out ? std::make_unique<Device<float>>(Device<float>::View{}, raw_out, q.count) : std::make_unique<Device<float>>(q.count);
    auto & out = *out_d; struct Out { float * p; } projected{result};
    if (carry) {
        const size_t row = size_t(kvheads) * dim;
        if (!carry->k) { PersistentAlloc keep; carry->k = std::make_unique<Device<__nv_bfloat16>>(size_t(g_kv_capacity) * row); carry->v = std::make_unique<Device<__nv_bfloat16>>(size_t(g_kv_capacity) * row); }
        require(pos + n <= g_kv_capacity, "decode context exceeds the attention KV capacity");
        ck(cudaMemcpyAsync(carry->k->p + size_t(pos) * row, kb.p, kb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpyAsync(carry->v->p + size_t(pos) * row, vb.p, vb.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, nullptr));
        // QSA (the model's own long-context attention): every token's indexer key is kept; once the context passes the
        // indexer budget each query attends to its top-k selected positions (dense attention is the same below it)
        static const bool qsa_on = tqfn_default_on("TQFN_QSA");
        const bool has_indexer = qsa_on && checkpoint.has_tensor(prefix + "indexer.k_proj.weight");
        const int top_k = 2048, ratio = 4, iheads = 4, idim = 128;
        if (has_indexer) {
            ArenaMark scratch;
            if (!carry->qsa) {
                // the indexer uses plain cudaMalloc: hand back what the async pool keeps reserved first
                cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
                PersistentAlloc keep; carry->qsa = std::make_unique<tqllm_cuda::QsaIndexer>(g_kv_capacity, ratio, top_k, iheads, idim,
                                   std::min(checkpoint.rotary, idim), checkpoint.theta, float(checkpoint.epsilon)); }
            if (carry->qsa->size() > pos) carry->qsa->truncate(pos);  // speculative rollback
            Device<float> ik(size_t(n) * idim);
            checkpoint.project(prefix + "indexer.k_proj.weight", input.p, ik.p, c.hidden, idim, n);
            carry->qsa->append(ik.p, n, checkpoint.dense_device(prefix + "indexer.k_norm.weight"), nullptr);
        }
        if (has_indexer && pos + n > top_k) {
            ArenaMark scratch;
            struct QsaTimer { std::chrono::steady_clock::time_point t0;
                QsaTimer() { if (g_prefill_log) { cudaDeviceSynchronize(); t0 = std::chrono::steady_clock::now(); } }
                ~QsaTimer() { if (g_prefill_log) { cudaDeviceSynchronize(); g_qsa_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } } qsa_timer;
            Device<float> iq(size_t(n) * iheads * idim);
            checkpoint.project(prefix + "indexer.q_proj.weight", input.p, iq.p, c.hidden, iheads * idim, n);
            const float * qnorm = checkpoint.dense_device(prefix + "indexer.q_norm.weight");
            static std::unique_ptr<tqllm_cuda::QsaBatchScratch> batch;  // shared by every layer (one stream)
            {
                if (!batch || batch->max_queries < n) {
                    batch.reset(); cudaMemPool_t pool; ck(cudaDeviceSynchronize()); ck(cudaDeviceGetDefaultMemPool(&pool, 0)); ck(cudaMemPoolTrimTo(pool, 0));
                    batch = std::make_unique<tqllm_cuda::QsaBatchScratch>(std::max(n, 16), g_kv_capacity, ratio, top_k);
                }
                const auto sel = carry->qsa->select_batch(iq.p, qnorm, pos, n, *batch, nullptr);
                if (n >= 32) {  // grouped kernel, K/V read once per kv head  // few tokens (3090 chunk 16, verify): the per-head kernel has more blocks
                    Device<float> sc(size_t(n) * heads * sel.width), inv(size_t(n) * heads);
                    tqllm_cuda::qsa_attention_grouped(q.p, carry->k->p, carry->v->p, sel, sc.p, inv.p, out.p, n, pos + 1, heads, kvheads, dim, nullptr);
                } else tqllm_cuda::qsa_attention_direct(q.p, carry->k->p, carry->v->p, sel, out.p, n, pos + 1, heads, kvheads, dim, nullptr);
            }
        } else tqllm_cuda::fn_dense_attention(q.p, carry->k->p, carry->v->p, out.p, heads, kvheads, dim, n, pos, nullptr);
        carry->len += n;
    } else tqllm_cuda::fn_dense_attention(q.p, kb.p, vb.p, out.p, heads, kvheads, dim, n, 0, nullptr);
    if (fixtures && fixtures->has("attn_pregate" + suffix))
        compare("attention_causal_output" + suffix, out.get(), fixtures->values("attn_pregate" + suffix), .02);
    if (raw_out) return;
    tqllm_cuda::fn_attention_gate(out.p, qg.p, heads, dim, n, nullptr);
    if (fixtures && fixtures->has("attn_gated" + suffix))
        compare("attention_gated_output" + suffix, out.get(), fixtures->values("attn_gated" + suffix), .02);
    checkpoint.project(prefix + "attn_output.weight", out.p, projected.p, heads * dim, c.hidden, n);
    if (fixtures) compare("attention_projected_output" + suffix, [&] { std::vector<float> v2(size_t(n) * c.hidden); ck(cudaMemcpy(v2.data(), result, v2.size() * 4, cudaMemcpyDeviceToHost)); return v2; }(), fixtures->values("attn_output" + suffix), .02);
}
static void run_recurrent(int argc, char ** argv, bool cpu_q8k = false) {
    require(argc >= 3, "usage: tqfn-stage --recurrent fixture-directory shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); int layers = 0;
    checkpoint.cpu_q8_projection = cpu_q8k;
    for (int layer = 0; layer < checkpoint.config.layers; ++layer) {
        if (checkpoint.config.compress_ratios[layer]) continue;
        recurrent_forward(checkpoint, fixtures.values("hc_mixed-" + std::to_string(layer), 0), layer, &fixtures);
        ++layers;
    }
    std::cout << "NATIVE_RECURRENT_REPLAY_DONE layers=" << layers << " full_forward=false\n";
}
static void run_attention(int argc, char ** argv, bool cpu_q8 = false) {
    require(argc >= 3, "usage: tqfn-stage --attention fixture-directory shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); int layers = 0;
    checkpoint.cpu_q8_projection = cpu_q8;
    std::cout << "PROJECTION_ARITHMETIC " << (cpu_q8 ? "cpu-q8-rht-diagnostic" : "native-f32-rht") << '\n';
    for (int layer = 0; layer < checkpoint.config.layers; ++layer) {
        if (!checkpoint.config.compress_ratios[layer]) continue;
        attention_forward(checkpoint, fixtures.values("hc_mixed-" + std::to_string(layer), 0), layer, &fixtures);
        ++layers;
    }
    std::cout << "NATIVE_ATTENTION_REPLAY_DONE layers=" << layers << " QSA=false context_below_budget=true full_forward=false\n";
}
struct MixerResult { std::vector<float> mixed, inject; };
// Device GR Read for decode: mixed [n][h] and inject [n][streams] stay on the device.
static void mix_device(const Checkpoint & checkpoint, const std::string & prefix, const float * normalized, int n,
                       float * mixed, float * inject) {
    const auto & c = checkpoint.config; const int h = c.hidden, streams = c.hc, width = h * streams;
    Device<float> low(size_t(n) * c.hc_low_rank), logits(size_t(n) * width);
    checkpoint.project(prefix + "down.weight", normalized, low.p, width, c.hc_low_rank, n);
    tqllm_cuda::hc_low_silu(low.p, int(low.count), streams, nullptr);
    checkpoint.project(prefix + "up.weight", low.p, logits.p, c.hc_low_rank, width, n);
    tqllm_cuda::hc_mix(normalized, logits.p, mixed, h, streams, n, nullptr);
    if (inject) checkpoint.project(prefix + "inject.weight", normalized, inject, width, streams, n);
}
static MixerResult replay_mix(const Fixtures & fixtures, const Checkpoint & checkpoint, const std::string & suffix,
                              size_t occurrence, const Device<float> & normalized, bool head = false, bool screen = true) {
    const auto & c = checkpoint.config;
    const int h = c.hidden, streams = c.hc, width = h * streams, n = int(normalized.count / width);
    const std::string prefix = head ? "output_hc_" : "blk." + suffix.substr(1) + (occurrence ? ".hc_ffn_" : ".hc_attn_");
    Device<float> low(size_t(n) * c.hc_low_rank), logits(normalized.count), mixed(size_t(n) * h);
    checkpoint.project(prefix + "down.weight", normalized.p, low.p, width, c.hc_low_rank, n);
    tqllm_cuda::hc_low_silu(low.p, int(low.count), streams, nullptr);
    checkpoint.project(prefix + "up.weight", low.p, logits.p, c.hc_low_rank, width, n);
    auto gates = logits.get();
    for (auto & value : gates) value = 1.f / (1.f + std::exp(-value));
    const std::string tag = head ? "head" : suffix + "/" + std::to_string(occurrence);
    // A 2 percent stage screen covers the reference's Q8 activation rounding, not full-model KLD.
    if (screen) compare("hc_projected_gate" + tag, gates, fixtures.values("hc_gate" + suffix, occurrence), .02);
    tqllm_cuda::hc_mix(normalized.p, logits.p, mixed.p, h, streams, n, nullptr);
    MixerResult result{mixed.get(), {}};
    if (head && fixtures.has("result_norm")) {
        if (screen) compare("hc_head_collapse", result.mixed, fixtures.values("result_norm"), .02);
    } else if (!head) {
        if (screen) compare("hc_projected_mix" + tag, result.mixed, fixtures.values("hc_mixed" + suffix, occurrence), .02);
        Device<float> inject(size_t(n) * streams);
        checkpoint.project(prefix + "inject.weight", normalized.p, inject.p, width, streams, n);
        result.inject = inject.get();
        if (screen) compare("hc_projected_inject" + tag, result.inject, fixtures.values("hc_inject" + suffix, occurrence), .02);
    }
    return result;
}
// Complete short-context chain. Fixtures supply only token IDs and final expected logits.
// Temporary transfers/allocations are diagnostic; this is not the production session or speed path.
// --chain-kld: tokens from a llama.cpp KL base instead of the fixture, no trace files, final residual handed back
static const std::vector<int> * g_chain_tokens = nullptr;
static bool g_chain_quiet = false;
static std::vector<float> g_chain_residual;
static bool g_chain_f16_act = false;
static void run_chain(int argc, char ** argv, bool cpu_q8, bool f32_experts = false, bool gsq_q8k = false) {
    require(argc >= 3, "usage: tqfn-stage --chain[-cpu-q8] fixture-dir shard1.gguf [more shards]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); checkpoint.cpu_q8_projection = cpu_q8 || gsq_q8k;
    if (g_chain_f16_act) checkpoint.cpu_q8_projection = false;
    const auto & c = checkpoint.config; const int h = c.hidden, width = h * c.hc;
    require(checkpoint.trace_layer < c.layers, "trace layer exceeds model depth");
    const std::string mode = gsq_q8k ? std::string("gsq-q8k") + (checkpoint.f32_expert_output ? "-expert-f32-out" : "")
                                      : std::string(cpu_q8 ? "q8" : "f32") + (f32_experts ? "-expert-f32" : "");
    const auto tokens = g_chain_tokens ? *g_chain_tokens : fixtures.tokens(); const int nt = int(tokens.size());
    require(nt > 0 && nt <= 64, "chain diagnostic currently bounded to 64 tokens");
    auto trace = [&](const std::string & name, const std::vector<float> & data) {
        if (g_chain_quiet) return;
        std::ofstream file(std::filesystem::path(argv[1]) / ("trace-" + mode + "-" + name + ".f32"), std::ios::binary);
        require(bool(file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float))), "cannot write native trace");
    };
    std::vector<uint64_t> token_rows;
    for (int token : tokens) { require(token >= 0, "negative token ID"); token_rows.push_back(uint64_t(token)); }
    auto embedding = checkpoint.file("token_embd.weight").gather_rows("token_embd.weight", token_rows);
    trace("embedding", embedding);
    Device<float> de(embedding.size()), init(size_t(width) * nt); de.put(embedding);
    tqllm_cuda::hc_init(de.p, init.p, h, c.hc, nt, nullptr); auto residual = init.get();
    PleTokenHistory history(c.ple, tokens.size()); const auto ids = history.append_batch(tokens);
    std::vector<uint64_t> ple_rows(ids.begin(), ids.end());
    auto ple = checkpoint.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
    const int slots = std::min(c.experts, nt * c.topk);
    std::unique_ptr<Tq3ExpertCache> tq_cache;
    std::unique_ptr<MixedExpertCache> mixed_cache;
    const bool mixed_formats = checkpoint.expert_payload(0, 0)[0].type != GGML_TQ3_4S;
    std::cout << "CHAIN_EXPERT_LAYOUT " << (mixed_formats ? "native-mixed-plain" : "TQ3-PRMT-rotated") << '\n';
    require(!gsq_q8k || mixed_formats, "Q8_K chain diagnostic requires native mixed formats");
    require(!checkpoint.f32_expert_output || (gsq_q8k && mixed_formats), "F32 expert output needs mixed GSQ chain");
    require(!mixed_formats || (!f32_experts && !cpu_q8), "mixed chain uses native grouped formats, not TQ3 precision modes");
    if (mixed_formats) mixed_cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, h, c.expert_ff, slots,
        size_t(2) * 1024 * 1024 * 1024, [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    else tq_cache = std::make_unique<Tq3ExpertCache>(c.layers, c.experts, h, c.expert_ff, slots,
        size_t(2) * 1024 * 1024 * 1024, [&](int layer, int expert) { return checkpoint.expert(layer, expert); }, nullptr);
    auto gate_matrix = [&] { return mixed_formats ? mixed_cache->gate() : tq_cache->gate(); };
    auto up_matrix = [&] { return mixed_formats ? mixed_cache->up() : tq_cache->up(); };
    auto down_matrix = [&] { return mixed_formats ? mixed_cache->down() : tq_cache->down(); };
    for (int layer = 0; layer < c.layers; ++layer) {
        const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
        const bool selected_trace = layer == checkpoint.trace_layer;
        auto trace_selected = [&](const std::string & name, const std::vector<float> & data) {
            trace(layer ? name + suffix : name, data);
        };
        if (std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end()) {
            require(ple.size() == size_t(h) * nt, "PLE gather width differs");
            Device<float> dp(ple.size()), key(residual.size()), value(size_t(h) * nt), query(residual.size());
            Device<float> kn(key.count), qn(query.count), gamma(width), gates(size_t(c.hc) * nt), gated(residual.size());
            dp.put(ple); query.put(residual);
            checkpoint.project(prefix + "ple_key.weight", dp.p, key.p, h, width, nt);
            checkpoint.project(prefix + "ple_value.weight", dp.p, value.p, h, h, nt);
            gamma.put(checkpoint.dense(prefix + "ple_norm_key.weight"));
            tqllm_cuda::hc_norm(key.p, gamma.p, kn.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            gamma.put(checkpoint.dense(prefix + "ple_norm_query.weight"));
            tqllm_cuda::hc_norm(query.p, gamma.p, qn.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_gate(kn.p, qn.p, gates.p, h, c.hc, nt, nullptr);
            tqllm_cuda::ple_broadcast(value.p, gates.p, gated.p, h, c.hc, nt, nullptr);
            gamma.put(checkpoint.dense(prefix + "ple_norm_conv.weight"));
            Device<float> norm(residual.size()), kernel{checkpoint.dense(prefix + "ple_conv1d.weight").size()};
            kernel.put(checkpoint.dense(prefix + "ple_conv1d.weight"));
            const int hist = (c.ple_kernel - 1) * c.ple.ngram;
            Device<float> past(size_t(hist) * width), next(past.count), conv(residual.size()), out(residual.size());
            past.put(std::vector<float>(past.count, 0.f));
            tqllm_cuda::hc_norm(gated.p, gamma.p, norm.p, h, c.hc, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_conv(norm.p, past.p, kernel.p, conv.p, next.p, width, nt, c.ple_kernel, c.ple.ngram, nullptr);
            tqllm_cuda::ple_apply(query.p, gated.p, conv.p, out.p, int(out.count), nullptr); residual = out.get();
        }
        int n = int(residual.size() / width);
        Device<float> dr(residual.size()), norm(residual.size()), gamma(width); dr.put(residual);
        gamma.put(checkpoint.dense(prefix + "hc_attn_norm.weight"));
        tqllm_cuda::hc_norm(dr.p, gamma.p, norm.p, h, c.hc, n, float(checkpoint.epsilon), nullptr);
        auto attention_mix = replay_mix(fixtures, checkpoint, suffix, 0, norm, false, false);
        if (selected_trace) trace_selected("attention_mixed", attention_mix.mixed);
        auto block = c.compress_ratios[layer] ? attention_forward(checkpoint, attention_mix.mixed, layer)
                                             : recurrent_forward(checkpoint, attention_mix.mixed, layer);
        if (selected_trace) trace_selected("attention_out", block);
        if (layer + 1 == c.layers && !g_chain_quiet) {
            residual.erase(residual.begin(), residual.end() - width);
            block.erase(block.begin(), block.end() - h);
            attention_mix.inject.erase(attention_mix.inject.begin(), attention_mix.inject.end() - c.hc); n = 1;
        }
        Device<float> ar(residual.size()), ao(block.size()), ai(attention_mix.inject.size()), combined(residual.size());
        ar.put(residual); ao.put(block); ai.put(attention_mix.inject);
        tqllm_cuda::hc_combine(ar.p, ao.p, ai.p, combined.p, h, c.hc, n, nullptr); residual = combined.get();
        if (selected_trace) trace_selected("attention_combined", residual);
        Device<float> fr(residual.size()), fn(residual.size()); fr.put(residual);
        gamma.put(checkpoint.dense(prefix + "hc_ffn_norm.weight"));
        tqllm_cuda::hc_norm(fr.p, gamma.p, fn.p, h, c.hc, n, float(checkpoint.epsilon), nullptr);
        const auto mix = replay_mix(fixtures, checkpoint, suffix, 1, fn, false, false);
        if (selected_trace) trace_selected("ffn_mixed", mix.mixed);
        Device<float> plain(mix.mixed.size()), router{checkpoint.dense(prefix + "ffn_gate_inp.weight").size()}, shared_gate(h);
        plain.put(mix.mixed); router.put(checkpoint.dense(prefix + "ffn_gate_inp.weight"));
        shared_gate.put(checkpoint.dense(prefix + "ffn_gate_inp_shexp.weight"));
        Device<float> logits(size_t(n) * (c.experts + 1)), weights(size_t(n) * c.topk), shared_weights(n);
        Device<int> routed_ids(weights.count);
        Device<float> router_input(plain.count);
        ck(cudaMemcpy(router_input.p, plain.p, plain.count * sizeof(float), cudaMemcpyDeviceToDevice));
        if (bf16_act()) round_bf16_device(router_input.p, router_input.count);
        tqllm_cuda::moe_router_f32(router.p, shared_gate.p, router_input.p, logits.p, h, c.experts, n, nullptr);
        tqllm_cuda::moe_topk(logits.p, routed_ids.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
        const auto actual_ids = routed_ids.get();
        const auto slot_ids = mixed_formats ? mixed_cache->prepare(layer, actual_ids) : tq_cache->prepare(layer, actual_ids); routed_ids.put(slot_ids);
        auto mixed_f32_projection = [&](Mat matrix, const __half * x, int rows, bool per_expert) {
            Device<float> output(size_t(n) * c.topk * rows);
            const size_t data_stride = size_t(rows) * matrix.fmt->row_bytes(matrix.cols);
            const size_t aux_stride = size_t(rows) * matrix.fmt->aux_row_bytes(matrix.cols);
            for (int assignment = 0; assignment < n * c.topk; ++assignment) {
                Mat view = matrix;
                view.p = static_cast<const char *>(matrix.p) + size_t(slot_ids[assignment]) * data_stride;
                if (matrix.aux) view.aux = static_cast<const char *>(matrix.aux) + size_t(slot_ids[assignment]) * aux_stride;
                view.rows = rows;
                const int token = per_expert ? assignment : assignment / c.topk;
                view.fmt->matvec_f32(view, x + size_t(token) * matrix.cols,
                                     output.p + size_t(assignment) * rows, 1, nullptr);
            }
            return output.get();
        };
        auto f32_projection = [&](const Mat & matrix, const std::vector<float> & values, int rows, bool per_expert) {
            const int columns = per_expert ? n * c.topk : n;
            Device<float> input(values.size()), rotated(values.size()), output(size_t(n) * c.topk * rows); input.put(values);
            tqllm_cuda::tq3_rotate_f32(input.p, rotated.p, matrix.cols, columns, nullptr);
            if (cpu_q8) { auto rounded = rotated.get(); cpu_q8_round(rounded); rotated.put(rounded); }
            const size_t stride = size_t(rows) * matrix.cols / 2;
            for (int assignment = 0; assignment < n * c.topk; ++assignment)
                tqllm_cuda::tq3_matvec_f32_repacked(static_cast<const char *>(matrix.p) + size_t(slot_ids[assignment]) * stride,
                    rotated.p + size_t(per_expert ? assignment : assignment / c.topk) * matrix.cols,
                    output.p + size_t(assignment) * rows, rows, matrix.cols, 1, nullptr);
            return output.get();
        };
        std::vector<float> gate, up, routed;
        if (f32_experts) {
            gate = f32_projection(gate_matrix(), mix.mixed, c.expert_ff, false);
            up = f32_projection(up_matrix(), mix.mixed, c.expert_ff, false);
        } else {
            Device<__nv_bfloat16> xb(plain.count), eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count);
            Device<__half> xr(plain.count);
            if (gsq_q8k) {
                if (exact_f32()) f32_to_half_host(plain.p, xr.p, plain.count);
                else if (round32_gpu_side("ffn_gate_exps")) tqllm_cuda::gsq_q8_round_h(plain.p, xr.p, plain.count, nullptr);
                else tqllm_cuda::gsq_q8k_round_h(plain.p, xr.p, plain.count, nullptr);
                if (checkpoint.validate_gsq_round) validate_round(xr, mix.mixed, true, prefix + "expert_input");
            } else {
                tqllm_cuda::f32_to_bf16(plain.p, xb.p, plain.count, nullptr);
                if (mixed_formats) tqllm_cuda::bf16_to_h(xb.p, xr.p, int(xb.count), nullptr);
                else tqllm_cuda::tq3_rotate_h(xb.p, xr.p, h, n, nullptr);
            }
            if (checkpoint.f32_expert_output) {
                gate = mixed_f32_projection(gate_matrix(), xr.p, c.expert_ff, false);
                up = mixed_f32_projection(up_matrix(), xr.p, c.expert_ff, false);
            } else {
                auto matrix = gate_matrix(); matrix.fmt->expert_matvec(matrix, xr.p, eg.p, c.expert_ff, routed_ids.p, n, c.topk, false, nullptr);
                matrix = up_matrix(); matrix.fmt->expert_matvec(matrix, xr.p, eu.p, c.expert_ff, routed_ids.p, n, c.topk, false, nullptr);
                gate = bf16_values(eg); up = bf16_values(eu);
            }
        }
        std::vector<float> activation(gate.size());
        for (size_t j = 0; j < activation.size(); ++j) activation[j] = gate[j] / (1.f + std::exp(-gate[j])) * up[j];
        if (selected_trace) { trace_selected("expert_gate", gate); trace_selected("expert_up", up); trace_selected("expert_activation", activation); }
        if (f32_experts) routed = f32_projection(down_matrix(), activation, h, true);
        else {
            Device<float> af(activation.size()); af.put(activation); Device<__nv_bfloat16> ab(activation.size());
            Device<__half> ax(activation.size()); tqllm_cuda::f32_to_bf16(af.p, ab.p, af.count, nullptr);
            if (gsq_q8k) {
                if (exact_f32()) f32_to_half_host(af.p, ax.p, af.count);
                else tqllm_cuda::gsq_q8_round_h(af.p, ax.p, af.count, nullptr);
                if (checkpoint.validate_gsq_round && !exact_f32()) validate_round(ax, activation, false, prefix + "expert_down_input");
            } else if (mixed_formats) tqllm_cuda::bf16_to_h(ab.p, ax.p, int(ab.count), nullptr);
            else tqllm_cuda::tq3_rotate_h(ab.p, ax.p, c.expert_ff, n * c.topk, nullptr);
            if (checkpoint.f32_expert_output) routed = mixed_f32_projection(down_matrix(), ax.p, h, true);
            else {
                Device<__nv_bfloat16> ed(size_t(n) * c.topk * h); auto matrix = down_matrix();
                matrix.fmt->expert_matvec(matrix, ax.p, ed.p, h, routed_ids.p, n, c.topk, true, nullptr);
                routed = bf16_values(ed);
            }
        }
        Device<float> sg(size_t(n) * c.shared_ff), su(sg.count), sa(sg.count), sd(size_t(n) * h);
        checkpoint.project(prefix + "ffn_gate_shexp.weight", plain.p, sg.p, h, c.shared_ff, n);
        checkpoint.project(prefix + "ffn_up_shexp.weight", plain.p, su.p, h, c.shared_ff, n);
        auto shared_activation = sg.get(), shared_up = su.get();
        for (size_t j = 0; j < shared_activation.size(); ++j)
            shared_activation[j] = shared_activation[j] / (1.f + std::exp(-shared_activation[j])) * shared_up[j];
        sa.put(shared_activation); checkpoint.project(prefix + "ffn_down_shexp.weight", sa.p, sd.p, c.shared_ff, h, n);
        const auto shared = sd.get(), rw = weights.get(), sw = shared_weights.get();
        if (selected_trace) { trace_selected("expert_down", routed); trace_selected("shared", shared); trace_selected("routing_weights", rw); trace_selected("shared_weights", sw); }
        std::vector<float> result(size_t(n) * h);
        for (int t = 0; t < n; ++t) for (int j = 0; j < h; ++j) {
            float value = 0.f;
            for (int k = 0; k < c.topk; ++k) value += rw[t * c.topk + k] * routed[(size_t(t) * c.topk + k) * h + j];
            result[size_t(t) * h + j] = value + sw[t] * shared[size_t(t) * h + j];
        }
        if (selected_trace) trace_selected("ffn_output", result);
        Device<float> fo(result.size()), fi(mix.inject.size()), updated(residual.size()); fo.put(result); fi.put(mix.inject);
        tqllm_cuda::hc_combine(fr.p, fo.p, fi.p, updated.p, h, c.hc, n, nullptr); residual = updated.get();
        require(std::all_of(residual.begin(), residual.end(), [](float x) { return std::isfinite(x); }), "nonfinite chained residual");
        if (!g_chain_quiet) {
            std::ofstream state_file(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-layer-" + std::to_string(layer) + ".f32"), std::ios::binary);
            require(bool(state_file.write(reinterpret_cast<const char *>(residual.data()), residual.size() * sizeof(float))), "cannot save chained layer trace");
            std::ofstream route_file(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-routes-" + std::to_string(layer) + ".i32"), std::ios::binary);
            require(bool(route_file.write(reinterpret_cast<const char *>(actual_ids.data()), actual_ids.size() * sizeof(int))), "cannot save native routing trace");
            std::cout << "NATIVE_CHAIN_LAYER " << layer << " tokens=" << n << std::endl;
        }
    }
    if (g_chain_quiet) { g_chain_residual = residual; return; }
    Device<float> last(residual.size()), norm(residual.size()), gamma(width); last.put(residual);
    gamma.put(checkpoint.dense("output_hc_norm.weight"));
    tqllm_cuda::hc_norm(last.p, gamma.p, norm.p, h, c.hc, 1, float(checkpoint.epsilon), nullptr);
    const auto head = replay_mix(fixtures, checkpoint, "", 0, norm, true, false);
    trace("head", head.mixed);
    const auto logits = checkpoint.head_logits(head.mixed), expected = fixtures.values("result_output");
    require(logits.size() == expected.size(), "expected last-token logit dimensions differ");
    std::ofstream raw(std::filesystem::path(argv[1]) / ("native-chain-" + mode + "-logits.f32"), std::ios::binary);
    require(bool(raw.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float))), "cannot save chained logits");
    double zp = 0., zq = 0., mean = 0.;
    const float mp = *std::max_element(expected.begin(), expected.end()), mq = *std::max_element(logits.begin(), logits.end());
    for (size_t i = 0; i < logits.size(); ++i) { zp += std::exp(double(expected[i]) - mp); zq += std::exp(double(logits[i]) - mq); }
    for (size_t i = 0; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]) && std::isfinite(expected[i]), "nonfinite chained logits");
        const double probability = std::exp(double(expected[i]) - mp) / zp;
        mean += probability * (double(expected[i]) - mp - double(logits[i]) + mq);
    }
    const double kld = mean + std::log(zq / zp);
    const auto actual_top = std::max_element(logits.begin(), logits.end()) - logits.begin();
    const auto reference_top = std::max_element(expected.begin(), expected.end()) - expected.begin();
    std::cout << "NATIVE_CHAIN_DONE layers=" << c.layers << " input_tokens=" << nt << " fixture_hidden_inputs=0"
              << " cpu_q8_diagnostic=" << cpu_q8 << " gsq_q8k_diagnostic=" << gsq_q8k
              << " f32_q6_diagnostic=" << checkpoint.f32_q6_projection
              << " f32_expert_output_diagnostic=" << checkpoint.f32_expert_output
              << " last_token_kld=" << kld << " top1=" << actual_top
              << " reference_top1=" << reference_top << " expert_intermediates="
              << (gsq_q8k ? "f32-q8k/q8-h" : f32_experts ? "f32" : "bf16")
              << " production_session=false" << std::endl;
    require(kld <= .01, "chained last-token KLD exceeds .01 screen");
}
// Many-token quality gate: teacher-forced native chain over the chunks of a llama.cpp --kl-divergence-base file (chunks of
// <= 64 tokens), positions n_ctx/2 .. n_ctx-2 scored with llama.cpp's KLD (same math as tqkld --llama-base).
struct KldSums {
    double sum_nll = 0, sum_nll2 = 0, sum_nll_base = 0, sum_nll_base2 = 0, sum_nll_nll_base = 0;
    double sum_kld = 0, sum_kld2 = 0, sum_p_diff = 0, sum_p_diff2 = 0, sum_p_diff4 = 0;
    float max_p_diff = 0;
    size_t n_same_top = 0, count = 0;
};
std::pair<double, float> kld_token(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, KldSums & kld) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_logit) { max_logit = logits[i]; imax = i; }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = log(sum_exp);
    const float * d = reinterpret_cast<const float *>(base_log_prob);
    const float scale = d[0], min_log_prob = d[1];
    base_log_prob += 4;
    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll += nll;
    kld.sum_nll2 += nll * nll;
    const float nll_base = -(scale * base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base += nll_base;
    kld.sum_nll_base2 += nll_base * nll_base;
    kld.sum_nll_nll_base += nll * nll_base;
    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) { p_log_base_max = p_log_base; imax_base = i; }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld += sum;
    kld.sum_kld2 += sum * sum;
    ++kld.count;
    if (imax == imax_base) ++kld.n_same_top;
    const float p_base = expf(-nll_base), p = expf(-nll), p_diff = p - p_base;
    kld.sum_p_diff += p_diff;
    const double p_diff2 = p_diff * p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2 * p_diff2;
    kld.max_p_diff = std::max(kld.max_p_diff, std::fabs(p_diff));
    return {sum, p_diff};
}


// Greedy decode prototype: the validated stage operations with carried state (GDN conv + recurrent state, BF16 KV,
// PLE conv history), resident weights and an LRU expert cache. Plain precision: FP16 activations into native formats.
struct PinnedInts { int * p = nullptr; size_t n = 0; void need(size_t k) { if (k > n) { if (p) cudaFreeHost(p); ck(cudaMallocHost(reinterpret_cast<void **>(&p), k * sizeof(int))); n = k; } } };
static double g_host_ms[3];  // decode host time blocked in: routed ids D2H, cache prepare, prefetch issue
struct DecodeModel {
    const Checkpoint & cp; const Fixtures & fx; const Qwen4ExpConfig & c; int h, width;
    MixedExpertCache & experts; MixedExpertCache * mtp_experts = nullptr;
    std::vector<RecurrentState> rec; std::vector<AttentionState> att; std::vector<std::unique_ptr<Device<float>>> ple_past;
    PleTokenHistory history; int position = 0;
    bool prefetch_on = std::getenv("TQFN_NO_PREFETCH") == nullptr; cudaStream_t copy_stream = nullptr, miss_stream = nullptr;
    // speculative decoding: per-token rollback state, final wide residual rows of the last step, MTP head state
    bool spec = false; int spec_pos0 = 0, spec_n = 0; size_t spec_hist0 = 0;
    std::vector<std::unique_ptr<Device<float>>> ple_saved, ple_old;
    std::unique_ptr<Device<float>> pingpong[2];  // prefill residual between layers (layer scratch is reset)
    std::unique_ptr<Device<float>> res_keep;  // [n][width] of the last step, persistent (grows with the prefill chunk)
    AttentionState mtp_att; std::unique_ptr<Device<float>> mtp_h;  // MTP block KV and its last wide output
    DecodeModel(const Checkpoint & k, const Fixtures & f, MixedExpertCache & e, size_t capacity)
        : cp(k), fx(f), c(k.config), h(c.hidden), width(c.hidden * c.hc), experts(e), rec(c.layers), att(c.layers),
          ple_past(c.layers), history(c.ple, capacity), ple_saved(c.layers), ple_old(c.layers) {
        ck(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        ck(cudaStreamCreateWithFlags(&miss_stream, cudaStreamNonBlocking)); experts.set_miss_stream(miss_stream);
        PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(64) * width); mtp_h = std::make_unique<Device<float>>(width);
    }
    ~DecodeModel() { if (copy_stream) { cudaStreamSynchronize(copy_stream); cudaStreamDestroy(copy_stream); }
                     if (miss_stream) { experts.set_miss_stream(nullptr); if (mtp_experts) mtp_experts->set_miss_stream(nullptr);
                                        cudaStreamSynchronize(miss_stream); cudaStreamDestroy(miss_stream); } }
    void set_spec(bool on) { spec = on; for (auto & r : rec) r.spec = on; }

    // One block (trunk layer 0..L-1, or the MTP block L) on n tokens; residual [n][width] is replaced.
    // a batched step over several sequences: rows [off, off + n) belong to model m (its PLE history, KV, recurrent state)
    struct Seg { DecodeModel * m; int off, n; };
    void layer_forward(int layer, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows,
                       AttentionState * attn, MixedExpertCache & cache, const std::vector<Seg> * segs = nullptr) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const bool trunk = layer < c.layers, attention = !trunk || c.compress_ratios[layer];
        Device<float> norm(size_t(width) * n), mixed(size_t(h) * n), inject(size_t(c.hc) * n), block(size_t(h) * n), combined(size_t(width) * n);
        if (trunk && ple_rows && std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end()) {
            Device<float> key(size_t(width) * n), value(size_t(h) * n), kn(key.count), qn(key.count), gates(size_t(c.hc) * n), gated(key.count);
            {
                const Checkpoint::SharedInput shared_in(cp, ple_rows, size_t(n) * h);
                cp.project(prefix + "ple_key.weight", ple_rows, key.p, h, width, n);
                cp.project(prefix + "ple_value.weight", ple_rows, value.p, h, h, n);
            }
            tqllm_cuda::hc_norm(key.p, cp.dense_device(prefix + "ple_norm_key.weight"), kn.p, h, c.hc, n, float(cp.epsilon), nullptr);
            tqllm_cuda::hc_norm(residual->p, cp.dense_device(prefix + "ple_norm_query.weight"), qn.p, h, c.hc, n, float(cp.epsilon), nullptr);
            tqllm_cuda::ple_gate(kn.p, qn.p, gates.p, h, c.hc, n, nullptr);
            tqllm_cuda::ple_broadcast(value.p, gates.p, gated.p, h, c.hc, n, nullptr);
            const int hist = (c.ple_kernel - 1) * c.ple.ngram;
            Device<float> pn(key.count), conv(key.count);
            auto out = std::make_unique<Device<float>>(key.count);
            tqllm_cuda::hc_norm(gated.p, cp.dense_device(prefix + "ple_norm_conv.weight"), pn.p, h, c.hc, n, float(cp.epsilon), nullptr);
            auto ple_state = [&](DecodeModel & m, int off, int nn) {  // the conv history is per sequence
                if (!m.ple_past[layer]) { PersistentAlloc keep; m.ple_past[layer] = std::make_unique<Device<float>>(size_t(hist) * width); ck_zero(*m.ple_past[layer]); }
                Device<float> next(m.ple_past[layer]->count);
                const size_t rows = size_t(nn) * width; const float * pseg = pn.p + size_t(off) * width;
                if (m.spec) {
                    PersistentAlloc keep;
                    if (!m.ple_old[layer]) m.ple_old[layer] = std::make_unique<Device<float>>(m.ple_past[layer]->count);
                    if (!m.ple_saved[layer] || m.ple_saved[layer]->count < rows) m.ple_saved[layer] = std::make_unique<Device<float>>(rows);
                    ck_copy_n(*m.ple_old[layer], m.ple_past[layer]->p, m.ple_past[layer]->count);
                    ck_copy_n(*m.ple_saved[layer], pseg, rows);
                }
                tqllm_cuda::ple_conv(pseg, m.ple_past[layer]->p, cp.dense_device(prefix + "ple_conv1d.weight"), conv.p + size_t(off) * width, next.p,
                                     width, nn, c.ple_kernel, c.ple.ngram, nullptr);
                ck_copy(*m.ple_past[layer], next);
            };
            if (segs) for (const auto & sg : *segs) ple_state(*sg.m, sg.off, sg.n);
            else ple_state(*this, 0, n);
            tqllm_cuda::ple_apply(residual->p, gated.p, conv.p, out->p, int(out->count), nullptr); residual = std::move(out);
        }
        tqllm_cuda::hc_norm(residual->p, cp.dense_device(prefix + "hc_attn_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        mix_device(cp, prefix + "hc_attn_", norm.p, n, mixed.p, inject.p);
        if (segs) {  // row-wise projections once for every sequence of the batch
            // row-wise projections once for all sequences; only the stateful middle (conv + recurrence, KV + attention) per sequence
            const std::string pf = "blk." + std::to_string(layer) + ".";
            if (attention) {
                const int heads = cp.attention_heads, kvh = cp.attention_kvheads, ad = cp.attention_dim;
                Device<float> qgB(size_t(n) * heads * ad * 2), kB(size_t(n) * kvh * ad), vB(kB.count), aoB(size_t(n) * heads * ad);
                {
                    const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                    cp.project(pf + "attn_q.weight", mixed.p, qgB.p, h, heads * ad * 2, n);
                    cp.project(pf + "attn_k.weight", mixed.p, kB.p, h, kvh * ad, n);
                    cp.project(pf + "attn_v.weight", mixed.p, vB.p, h, kvh * ad, n);
                }
                for (const auto & sg : *segs) {
                    const AttnPre pre{qgB.p + size_t(sg.off) * heads * ad * 2, kB.p + size_t(sg.off) * kvh * ad, vB.p + size_t(sg.off) * kvh * ad};
                    attention_core(cp, mixed.p + size_t(sg.off) * h, nullptr, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att,
                                   &pre, aoB.p + size_t(sg.off) * heads * ad);
                }
                tqllm_cuda::fn_attention_gate(aoB.p, qgB.p, heads, ad, n, nullptr);
                cp.project(pf + "attn_output.weight", aoB.p, block.p, heads * ad, h, n);
            } else {
                const int kh = cp.key_heads, vh = cp.value_heads, sd = cp.state_dim, channels = (2 * kh + vh) * sd;
                Device<float> qkvB(size_t(n) * channels), zB(size_t(n) * vh * sd), aB(size_t(n) * vh), bB(aB.count), roB(zB.count), gB(zB.count);
                {
                    const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h);
                    cp.project(pf + "attn_qkv.weight", mixed.p, qkvB.p, h, channels, n);
                    cp.project(pf + "attn_gate.weight", mixed.p, zB.p, h, vh * sd, n);
                    cp.project(pf + "ssm_alpha.weight", mixed.p, aB.p, h, vh, n);
                    cp.project(pf + "ssm_beta.weight", mixed.p, bB.p, h, vh, n);
                }
                for (const auto & sg : *segs) {
                    const GdnPre pre{qkvB.p + size_t(sg.off) * channels, zB.p + size_t(sg.off) * vh * sd, aB.p + size_t(sg.off) * vh, bB.p + size_t(sg.off) * vh};
                    recurrent_core(cp, nullptr, nullptr, sg.n, layer, nullptr, &sg.m->rec[layer], &pre, roB.p + size_t(sg.off) * vh * sd);
                }
                tqllm_cuda::fn_gated_rms(roB.p, zB.p, cp.dense_device(pf + "ssm_norm.weight"), gB.p, vh, sd, n, float(cp.epsilon), nullptr);
                cp.project(pf + "ssm_out.weight", gB.p, block.p, vh * sd, h, n);
            }
        } else if (segs) {
            for (const auto & sg : *segs) {
                const size_t o = size_t(sg.off) * h;
                if (attention) attention_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, trunk ? &sg.m->att[layer] : &sg.m->mtp_att);
                else recurrent_core(cp, mixed.p + o, block.p + o, sg.n, layer, nullptr, &sg.m->rec[layer]);
            }
        } else if (attention) attention_core(cp, mixed.p, block.p, n, layer, nullptr, attn ? attn : &att[layer]);
        else recurrent_core(cp, mixed.p, block.p, n, layer, nullptr, &rec[layer]);
        tqllm_cuda::hc_combine(residual->p, block.p, inject.p, combined.p, h, c.hc, n, nullptr);
        tqllm_cuda::hc_norm(combined.p, cp.dense_device(prefix + "hc_ffn_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        mix_device(cp, prefix + "hc_ffn_", norm.p, n, mixed.p, inject.p);
        Device<float> logits(size_t(n) * (c.experts + 1)), weights(size_t(n) * c.topk), shared_weights(n);
        Device<int> routed(weights.count);
        const bool predict = prefetch_on && n <= 4 && trunk && layer + 1 < c.layers;
        const std::string np = "blk." + std::to_string(layer + 1) + ".";
        std::unique_ptr<Device<float>> nl;
        if (predict) nl = std::make_unique<Device<float>>(size_t(n) * (c.experts + 1));
        router(prefix, mixed.p, logits.p, n);
        if (predict) router(np, mixed.p, nl->p, n);
        tqllm_cuda::moe_topk(logits.p, routed.p, weights.p, shared_weights.p, c.experts, c.topk, n, 1.f, nullptr);
        std::unique_ptr<Device<int>> next_ids;
        if (predict) {
            Device<float> nw(size_t(n) * c.topk), nsw(n); next_ids = std::make_unique<Device<int>>(nw.count);
            tqllm_cuda::moe_topk(nl->p, next_ids->p, nw.p, nsw.p, c.experts, c.topk, n, 1.f, nullptr);
        }
        using clk = std::chrono::steady_clock; auto h0 = clk::now();
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        Device<float> sg(size_t(n) * c.shared_ff), su(sg.count), sa(sg.count), sd(size_t(n) * h), fo(size_t(n) * h);
        auto shared_expert = [&](__half * have = nullptr) {
            const Checkpoint::SharedInput shared_in(cp, mixed.p, size_t(n) * h, have);
            cp.project(prefix + "ffn_gate_shexp.weight", mixed.p, sg.p, h, c.shared_ff, n);
            cp.project(prefix + "ffn_up_shexp.weight", mixed.p, su.p, h, c.shared_ff, n);
            tqllm_cuda::fn_swiglu_f32(sg.p, su.p, sa.p, int64_t(sg.count), nullptr);
            cp.project(prefix + "ffn_down_shexp.weight", sa.p, sd.p, c.shared_ff, h, n);
        };
        // TQFN_PREFILL_WAVES (default on): a prefill chunk that routes to more experts than the cache holds runs its MoE in
        // waves of at most `slots` experts; each expert is loaded once per layer per chunk (not once per small chunk).
        std::vector<int> wave_ids;
        if (trunk && n * c.topk > cache.slots(layer)) {  // incl. the short last chunk of a prompt
            wave_ids = routed.get();  // one wave when the unique experts fit: the split path's miss accounting assumes small steps
        }
        if (!wave_ids.empty()) {
            const int A = n * c.topk; constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
            std::vector<int> u(wave_ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
            Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
            Device<__nv_bfloat16> eg(size_t(A) * c.expert_ff), eu(eg.count), ed(size_t(A) * h);
            Device<__half> ax(eg.count);
            const int W = cache.slots(layer);
            for (size_t w0 = 0; w0 < u.size(); w0 += size_t(W)) {
                const std::vector<int> wave(u.begin() + long(w0), u.begin() + long(std::min(u.size(), w0 + size_t(W))));
                const auto wslots = cache.prepare(layer, wave);
                if (cache.split()) cache.wait_misses();
                std::vector<int> slot_of(size_t(c.experts), -1);
                for (size_t i = 0; i < wave.size(); ++i) slot_of[size_t(wave[i])] = wslots[i];
                std::vector<int> hm(size_t(2) * A * DS); int groups[2] = {0, 0};
                for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                    int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(W) + 1, -1); int g = 0;
                    for (int a = 0; a < A; ++a) {
                        const int sl = slot_of[size_t(wave_ids[size_t(a)])];
                        if (sl < 0) continue;
                        int & at = open_at[size_t(sl)];
                        if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = sl; out[at * DS + 1] = 0; }
                        int * e = out + at * DS; const int col = e[1]++;
                        e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                    }
                    groups[part] = g;
                }
                Device<int> dmap(hm.size()); dmap.put(hm);
                // formats without a dedup kernel (GSQ Q2_0) take the grouped GEMM path: per-slot lists of this wave
                std::vector<int> cnt(size_t(W), 0); int wcap = 1;
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) wcap = std::max(wcap, ++cnt[size_t(sl)]); }
                std::vector<int> grp(size_t(W) * wcap, 0), fill(size_t(W), 0);
                for (int a = 0; a < A; ++a) { const int sl = slot_of[size_t(wave_ids[size_t(a)])]; if (sl >= 0) grp[size_t(sl) * wcap + size_t(fill[size_t(sl)]++)] = a; }
                Device<int> dcnt(cnt.size()), dgrp(grp.size()); dcnt.put(cnt); dgrp.put(grp);
                // many tokens per expert (GB10 chunk 1024: ~20): grouped GEMM, weights read once per 64 tokens instead of
                // once per 4; few (3090 chunk 128: ~3) keep the dedup kernel. TQFN_WAVE_GEMM_AVG: the switch-over average
                static const int gemm_avg = std::getenv("TQFN_WAVE_GEMM_AVG") ? std::atoi(std::getenv("TQFN_WAVE_GEMM_AVG")) : 8;
                int wave_a = 0; for (int x : cnt) wave_a += x;
                const bool wave_gemm = gemm_avg > 0 && wave_a >= gemm_avg * int(wave.size());
                auto gemm_fmt = [](const Mat & mm) {
                    const int id = mm.fmt->u4h_id();
                    return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
                };
                auto proj = [&](const Mat & mm, const __half * x, __nv_bfloat16 * y, int rows, bool per_expert) {
                    if (mm.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(mm))) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_expert ? size_t(A) * DS : 0), groups[per_expert], nullptr);
                    else require(mm.fmt->expert_gemm(mm, x, y, rows, dcnt.p, dgrp.p, wcap, W, c.topk, per_expert, nullptr),
                                 std::string("prefill waves: no grouped kernel for ") + mm.fmt->name());
                };
                auto m = cache.gate(), u2 = cache.up();
                if (m.fmt == u2.fmt && m.fmt->has_expert_dedup() && !(wave_gemm && gemm_fmt(m))) m.fmt->expert_matvec_dedup2(m, u2, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
                else { proj(m, xr.p, eg.p, c.expert_ff, false); proj(u2, xr.p, eu.p, c.expert_ff, false); }
                tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
                proj(cache.down(), ax.p, ed.p, h, true);
                ck(cudaStreamSynchronize(nullptr));  // the next wave's prepare may evict these slots
            }
            shared_expert();
            tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        } else if (cache.split()) {
            // Shared expert and hit experts run while the host prepares the cache and misses land on the miss stream.
            const int A = n * c.topk;
            static PinnedInts ids_host; ids_host.need(size_t(2) * A);
            // The routed ids go to the host on their own stream, so the default stream runs the FP16 input and the shared
            // expert during the copy instead of waiting on a copy-engine -> SM dependency.
            static cudaStream_t ids_stream = [] { cudaStream_t q = nullptr; ck(cudaStreamCreateWithFlags(&q, cudaStreamNonBlocking)); return q; }();
            static cudaEvent_t routed_done = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
            ck(cudaEventRecord(routed_done, nullptr)); ck(cudaStreamWaitEvent(ids_stream, routed_done, 0));
            const cudaStream_t cs = ids_stream;
            ck(cudaMemcpyAsync(ids_host.p, routed.p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
            if (predict) ck(cudaMemcpyAsync(ids_host.p + A, next_ids->p, size_t(A) * 4, cudaMemcpyDeviceToHost, cs));
            static cudaEvent_t ids_ready = [] { cudaEvent_t e; ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; }();
            static const int cpu_miss = std::getenv("TQFN_CPU_MISS") ? std::atoi(std::getenv("TQFN_CPU_MISS")) : 0;
            static std::unique_ptr<CpuMissPool> cpu_pool = cpu_miss > 0 ? std::make_unique<CpuMissPool>(cpu_miss, h, c.expert_ff) : nullptr;
            static float * x_host = nullptr; static __nv_bfloat16 * cpu_out = nullptr;
            const bool use_cpu = cpu_pool && trunk && n <= 4;
            if (use_cpu) {
                if (!x_host) { ck(cudaMallocHost(reinterpret_cast<void **>(&x_host), size_t(4) * h * 4));
                               ck(cudaMallocHost(reinterpret_cast<void **>(&cpu_out), size_t(4) * c.topk * h * 2)); }
                ck(cudaMemcpyAsync(x_host, mixed.p, size_t(n) * h * 4, cudaMemcpyDeviceToHost, cs));
            }
            ck(cudaEventRecord(ids_ready, cs));
            Device<__half> xr(mixed.count);  // one FP16 copy of the FFN input for the shared and routed experts
            tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr); shared_expert(xr.p);
            ck(cudaEventSynchronize(ids_ready)); auto h1 = clk::now();
            std::vector<int> routed_ids(ids_host.p, ids_host.p + A);
            // TQFN_TAIL_SKIP=R (APPROXIMATE, changes the output; idea: Strata v0.1.42 route-tail-skip): an expert that is
            // not resident and that every token of this call ranks R-th or lower is neither copied nor computed. Its
            // assignments get weight 0 (other weights unchanged) and point at the token's resident first choice.
            static const int tail_skip = std::getenv("TQFN_TAIL_SKIP") ? std::atoi(std::getenv("TQFN_TAIL_SKIP")) : 0;
            if (tail_skip > 0 && trunk && n <= 4) {
                std::vector<int> best_rank(c.experts, c.topk);
                for (int a = 0; a < A; ++a) best_rank[routed_ids[a]] = std::min(best_rank[routed_ids[a]], a % c.topk);
                for (int a = 0; a < A; ++a) {
                    const int e = routed_ids[a];
                    if (best_rank[e] < tail_skip - 1 || cache.resident(layer, e)) continue;
                    const int first = routed_ids[(a / c.topk) * c.topk];
                    if (best_rank[first] >= tail_skip - 1 && !cache.resident(layer, first)) continue;  // first choice itself skipped: keep
                    routed_ids[a] = first;
                    ck(cudaMemsetAsync(weights.p + a, 0, sizeof(float), nullptr));
                    static uint64_t skipped = 0; ++skipped;
                    static struct Report { ~Report() { std::printf("TAIL_SKIP assignments skipped %llu\n", (unsigned long long) skipped); } } report;
                }
            }
            if (static FILE * dump = std::getenv("TQFN_ROUTE_DUMP") ? std::fopen(std::getenv("TQFN_ROUTE_DUMP"), "ab") : nullptr; dump && trunk) {
                const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
            }
            // Routed and predicted next-layer ids per trunk layer call, for offline cache simulation.
            if (static FILE * dump = std::getenv("TQFN_PRED_DUMP") ? std::fopen(std::getenv("TQFN_PRED_DUMP"), "ab") : nullptr; dump && trunk) {
                const int32_t hdr[3] = {layer, n, predict ? A : 0}; std::fwrite(hdr, 4, 3, dump);
                std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
                if (predict) std::fwrite(ids_host.p + A, 4, size_t(A), dump);
            }
            // TQFN_CPU_MISS: a non-resident expert goes to the CPU pool; its GPU assignment points at a resident expert of the
            // same token (output row overwritten before the combine), so the cache neither copies nor inserts it.
            std::vector<std::pair<int, int>> cpu_rows;  // (assignment, row in cpu_out)
            std::vector<int> cpu_experts;  // TQFN_CPU_REFILL: copied into the cache after this layer, off the critical path
            if (use_cpu) {
                std::vector<CpuMissPool::Job> jobs; std::map<int, int> job_of;
                for (int a = 0; a < A; ++a) {
                    const int e = routed_ids[a], t = a / c.topk;
                    if (cache.resident(layer, e)) continue;
                    int sub = -1;
                    for (int q = 0; q < c.topk && sub < 0; ++q) if (cache.resident(layer, routed_ids[t * c.topk + q])) sub = routed_ids[t * c.topk + q];
                    if (sub < 0) continue;  // nothing resident for this token: the GPU miss path serves it
                    // TQFN_CPU_MISS_MAX=K: at most K distinct experts per layer call on the CPU, the rest are GPU misses
                    static const int cpu_max = std::getenv("TQFN_CPU_MISS_MAX") ? std::atoi(std::getenv("TQFN_CPU_MISS_MAX")) : 1 << 30;
                    if (!job_of.count(e) && int(jobs.size()) >= cpu_max) continue;
                    auto [it, fresh] = job_of.try_emplace(e, int(jobs.size()));
                    if (fresh) {
                        CpuMissPool::Job j; const auto pl = cp.expert_payload(layer, e);
                        const int rows[3] = {c.expert_ff, c.expert_ff, h};
                        for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                        if (const auto * pv = g_cpu_packed; pv && layer < pv->layers) {
                            const char * blk = pv->map + pv->base[layer] + uint64_t(e) * pv->block[layer]; uint64_t off = 0;
                            for (int m = 0; m < 3; ++m) {
                                j.dev[m] = blk + off; j.aux[m] = blk + off + pv->sizes[layer][2 * m]; j.fmt[m] = pv->fmt[layer][m];
                                off += pv->sizes[layer][2 * m] + pv->sizes[layer][2 * m + 1];
                            }
                            j.tiled = pv->tiled;
                        }
                        jobs.push_back(j);
                    }
                    if (fresh) cpu_experts.push_back(e);
                    auto & j = jobs[it->second]; j.cols_t[j.ncols++] = t;
                    cpu_rows.push_back({a, it->second * 1000 + j.ncols - 1});
                    routed_ids[a] = sub;
                }
                std::vector<int> base(jobs.size(), 0);
                for (size_t q = 1; q < jobs.size(); ++q) base[q] = base[q - 1] + jobs[q - 1].ncols;
                for (auto & [a, r] : cpu_rows) r = base[r / 1000] + r % 1000;
                static uint64_t cpu_assign = 0, all_assign = 0; cpu_assign += cpu_rows.size(); all_assign += A;
                static struct Report { ~Report() { std::printf("CPU_MISS assignments on the CPU %llu of %llu\n", (unsigned long long) cpu_assign, (unsigned long long) all_assign); } } report;
                cpu_pool->start(x_host, n, std::move(jobs), cpu_out);
            }
            const auto slots = cache.prepare(layer, routed_ids); const auto & missed = cache.last_missed();
            require(cache.gate().fmt->has_expert_matvec(), "split expert path needs a grouped expert kernel");
            static std::vector<PinnedInts> slot_host(c.layers + 1); auto & sh = slot_host[layer]; sh.need(size_t(3) * A);
            int hit_n = 0, miss_n = 0;
            for (int a = 0; a < A; ++a) {
                sh.p[a] = slots[a];
                sh.p[A + a] = missed[a] ? -1 : slots[a]; hit_n += !missed[a];
                sh.p[2 * A + a] = missed[a] ? slots[a] : -1; miss_n += missed[a];
            }
            int * slots_p = nullptr, * hit_p = nullptr, * miss_p = nullptr;  // set from the parameter maps below
            auto h2 = clk::now();
            auto issue_prefetch = [&] {
                if (!predict) return;
                // TQFN_PF_K: prefetch only each token's k best predicted experts (ids are in descending score order)
                static const int pf_k = std::getenv("TQFN_PF_K") ? std::atoi(std::getenv("TQFN_PF_K")) : 6;  // 6 of 10: +3.3 % vs all (10-11)
                std::vector<int> want; want.reserve(A);
                constexpr int high = 3;  // ranks 1-3 are admitted first, ranks 4..pf_k only over stale or rarely used victims
                for (int t = 0; t < n; ++t) for (int j = 0; j < std::min(high, c.topk); ++j) want.push_back(ids_host.p[A + t * c.topk + j]);
                cache.set_prefetch_strict(int(want.size()));
                for (int t = 0; t < n; ++t) for (int j = high; j < std::min(pf_k, c.topk); ++j) want.push_back(ids_host.p[A + t * c.topk + j]);
                cache.prefetch(layer + 1, want, copy_stream);
            };
            const auto h3 = clk::now();
            g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
            Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
            Device<__half> ax(eg.count);
            // Dedup: one block per unique expert for all verify tokens that route to it (weights read once).
            const bool dedup_gu = n > 1 && cache.gate().fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup();
            const bool dedup_dn = n > 1 && cache.down().fmt->has_expert_dedup();
            constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
            static std::vector<PinnedInts> dedup_host(c.layers + 1); auto & dh = dedup_host[layer]; dh.need(size_t(3) * A * DS);
            int groups[3] = {0, 0, 0};  // hit, miss, all (down)
            if (dedup_gu || dedup_dn) {
                auto build = [&](int which, int * out, bool per_expert) {
                    int g = 0;
                    std::vector<int> open_at(c.experts + 1, -1);
                    for (int a = 0; a < A; ++a) {
                        if ((which == 0 && missed[a]) || (which == 1 && !missed[a])) continue;
                        const int slot = slots[a]; int & at = open_at[slot];
                        if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slot; out[at * DS + 1] = 0; }
                        int * e = out + at * DS; const int col = e[1]++;
                        e[2 + col] = per_expert ? a : a / c.topk; e[2 + DC + col] = a;
                    }
                    return g;
                };
                if (dedup_gu) { groups[0] = build(0, dh.p, false); groups[1] = build(1, dh.p + A * DS, false); }
                if (dedup_dn) groups[2] = build(2, dh.p + 2 * A * DS, true);
            }
            int * dmap_p = nullptr;
            const int A4 = (A + 3) / 4 * 4;
            // slot and dedup maps travel in one kernel's parameters (no host-memory reads, which crawl while demand-miss
            // copies hold the SMs)
            Device<int> maps(size_t(3) * A4 + size_t(3) * A * DS + 4);
            {
                static std::vector<int> blob; blob.assign(size_t(3) * A4 + ((dedup_gu || dedup_dn) ? size_t(3) * A * DS : 0), -1);
                for (int q = 0; q < 3; ++q) std::memcpy(blob.data() + size_t(q) * A4, sh.p + size_t(q) * A, size_t(A) * 4);
                if (dedup_gu || dedup_dn) std::memcpy(blob.data() + size_t(3) * A4, dh.p, size_t(3) * A * DS * 4);
                blob.resize((blob.size() + 3) / 4 * 4, -1);
                tqllm_cuda::param_copy(blob.data(), blob.size() * 4, maps.p, nullptr);
                slots_p = maps.p; hit_p = maps.p + A4; miss_p = maps.p + 2 * A4; dmap_p = maps.p + 3 * A4;
            }
            auto gate_up = [&](const int * which, int part) {
                auto m = cache.gate(), u = cache.up();
                if (dedup_gu) {
                    if (!groups[part]) return;
                    if (m.fmt == u.fmt) { m.fmt->expert_matvec_dedup2(m, u, xr.p, eg.p, eu.p, c.expert_ff, dmap_p + part * A * DS, groups[part], nullptr); return; }  // gate + up in one launch
                    m.fmt->expert_matvec_dedup(m, xr.p, eg.p, c.expert_ff, dmap_p + part * A * DS, groups[part], nullptr);
                    u.fmt->expert_matvec_dedup(u, xr.p, eu.p, c.expert_ff, dmap_p + part * A * DS, groups[part], nullptr);
                    return;
                }
                m.fmt->expert_matvec(m, xr.p, eg.p, c.expert_ff, which, n, c.topk, false, nullptr);
                u.fmt->expert_matvec(u, xr.p, eu.p, c.expert_ff, which, n, c.topk, false, nullptr);
            };
            if (hit_n) gate_up(hit_p, 0);
            issue_prefetch();  // after the hit launches: the GPU has work while the host issues copies
            cache.wait_misses();
            if (miss_n) gate_up(miss_p, 1);
            tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
            auto m = cache.down();
            if (dedup_dn) m.fmt->expert_matvec_dedup(m, ax.p, ed.p, h, dmap_p + 2 * A * DS, groups[2], nullptr);
            else m.fmt->expert_matvec(m, ax.p, ed.p, h, slots_p, n, c.topk, true, nullptr);
            if (!cpu_rows.empty()) {
                cpu_pool->wait();
                std::vector<tqllm_cuda::ScatterItem> items;
                for (auto [a, r] : cpu_rows) items.push_back({reinterpret_cast<const char *>(cpu_out + size_t(r) * h), reinterpret_cast<char *>(ed.p + size_t(a) * h), size_t(h) * 2});
                tqllm_cuda::scatter_copy(items.data(), int(items.size()), nullptr, 1);
            }
            tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
            static const bool cpu_refill = std::getenv("TQFN_CPU_REFILL") != nullptr;
            if (cpu_refill && !cpu_experts.empty()) {  // the cache still learns the experts the CPU served (copies after this layer)
                cache.set_prefetch_strict(1 << 30); cache.prefetch(layer, cpu_experts, copy_stream);
            }
        } else {
            const auto routed_ids = routed.get(); auto h1 = clk::now();
            if (static FILE * dump = std::getenv("TQFN_ROUTE_DUMP") ? std::fopen(std::getenv("TQFN_ROUTE_DUMP"), "ab") : nullptr; dump && trunk) {
                const int32_t hdr[2] = {layer, n}; std::fwrite(hdr, 4, 2, dump); std::fwrite(routed_ids.data(), 4, routed_ids.size(), dump);
            }
            const auto slots = cache.prepare(layer, routed_ids); routed.put(slots); auto h2 = clk::now();
            if (predict) cache.prefetch(layer + 1, next_ids->get(), copy_stream);
            const auto h3 = clk::now();
            g_host_ms[0] += ms(h0, h1); g_host_ms[1] += ms(h1, h2); g_host_ms[2] += ms(h2, h3);
            Device<__half> xr(mixed.count); tqllm_cuda::vis_f32_to_h(mixed.p, xr.p, int64_t(mixed.count), nullptr);
            Device<__nv_bfloat16> eg(size_t(n) * c.topk * c.expert_ff), eu(eg.count), ed(size_t(n) * c.topk * h);
            Device<__half> ax(eg.count);
            // Formats without a grouped expert kernel (MTP's Q8_0 -> I8H): one row-sliced matvec per assignment.
            auto sliced = [&](Mat mat, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
                const size_t data = size_t(rows) * mat.fmt->row_bytes(cols), aux = size_t(rows) * mat.fmt->aux_row_bytes(cols);
                for (int a = 0; a < n * c.topk; ++a) {
                    Mat v = mat; v.rows = rows;
                    v.p = static_cast<const char *>(mat.p) + size_t(slots[a]) * data;
                    if (mat.aux) v.aux = static_cast<const char *>(mat.aux) + size_t(slots[a]) * aux;
                    v.fmt->matvec(v, x + size_t(per_assignment ? a : a / c.topk) * cols, y + size_t(a) * rows, 1, nullptr);
                }
            };
            auto m = cache.gate();
            // TQFN_NS_DEDUP=0: one block per assignment. Default: one block per unique expert (weights read once per step)
            const bool dd = n > 1 && m.fmt->has_expert_dedup() && cache.up().fmt->has_expert_dedup() && cache.down().fmt->has_expert_dedup();
            // TQFN_EXPERT_GEMM (default on, chunks of TQFN_EXPERT_GEMM_MIN+ tokens): grouped GEMM per cache slot, each expert's
            // weights read once per 64 of its tokens (the dedup kernel reads them once per 4). Prefill only: decode and the batched
            // verify (<= 64 rows) keep the exact dedup path.
            static const int egemm_min = std::getenv("TQFN_EXPERT_GEMM_MIN") ? std::atoi(std::getenv("TQFN_EXPERT_GEMM_MIN")) : 65;  // above the 64-row batched verify
            auto gemm_ok = [](const Mat & mm) {
                const int id = mm.fmt->u4h_id();
                if (std::string(mm.fmt->name()) == "GSQ_Q2_0") return mm.cols % 64 == 0;
                return (id == 1 || id == 4 || id == 5 || id == 6 || id == 7 || id == 10 || id == 13 || id == 14) && mm.cols % (id == 14 ? 128 : 256) == 0;
            };
            (void) dd;
            // per projection: 0 grouped GEMM, 1 dedup, 2 one block per assignment, 3 row-sliced matvecs (a format without a
            // grouped kernel no longer pulls the layer's other projections onto the slow path)
            // per-slot token lists built on the host with stride = the largest count (not n): the grouped kernels' grids
            // then cover only real tokens (n = 1024 routes ~20 tokens to an expert, not 1024)
            const bool gav = n >= egemm_min;  // the grouped loader reads TQFN_TILED banks too (tests/expert_gemm_tiled.cu)
            int gcap = 1; std::vector<int> hcnt, hgrp;
            if (gav) {
                hcnt.assign(size_t(cache.slots()), 0);
                for (int a = 0; a < n * c.topk; ++a) gcap = std::max(gcap, ++hcnt[size_t(slots[a])]);
                hgrp.assign(size_t(cache.slots()) * gcap, 0); std::vector<int> fill(size_t(cache.slots()), 0);
                for (int a = 0; a < n * c.topk; ++a) hgrp[size_t(slots[a]) * gcap + size_t(fill[size_t(slots[a])]++)] = a;
            }
            Device<int> gcounts(gav ? hcnt.size() : 1), ggroups(gav ? hgrp.size() : 1);
            if (gav) { gcounts.put(hcnt); ggroups.put(hgrp); }
            const Mat mg = cache.gate(), mu = cache.up(), md = cache.down();
            auto mode = [&](const Mat & mm) {
                if (gav && gemm_ok(mm)) return 0;
                if (n > 1 && mm.fmt->has_expert_dedup()) return 1;
                return mm.fmt->has_expert_matvec() ? 2 : 3;
            };
            const int kg = mode(mg), ku = mode(mu), kd = mode(md);

            constexpr int DS = tqllm_cuda::kDedupStride, DC = tqllm_cuda::kDedupCols;
            const int A = n * c.topk; std::vector<int> hm; int groups[2] = {0, 0};
            if (kg == 1 || ku == 1 || kd == 1) {
                hm.assign(size_t(2) * A * DS, 0);
                for (int part = 0; part < 2; ++part) {  // 0: gate/up (token input), 1: down (assignment input)
                    int * out = hm.data() + size_t(part) * A * DS; std::vector<int> open_at(size_t(cache.slots()) + 1, -1); int g = 0;
                    for (int a = 0; a < A; ++a) {
                        int & at = open_at[slots[a]];
                        if (at < 0 || out[at * DS + 1] == DC) { at = g++; out[at * DS] = slots[a]; out[at * DS + 1] = 0; }
                        int * e = out + at * DS; const int col = e[1]++;
                        e[2 + col] = part ? a : a / c.topk; e[2 + DC + col] = a;
                    }
                    groups[part] = g;
                }
            }
            Device<int> dmap(hm.empty() ? 1 : hm.size()); if (!hm.empty()) dmap.put(hm);
            auto run = [&](const Mat & mm, int k, const __half * x, __nv_bfloat16 * y, int rows, int cols, bool per_assignment) {
                if (k == 0) require(mm.fmt->expert_gemm(mm, x, y, rows, gcounts.p, ggroups.p, gcap, cache.slots(), c.topk, per_assignment, nullptr),
                                   "expert_gemm unsupported");
                else if (k == 1) mm.fmt->expert_matvec_dedup(mm, x, y, rows, dmap.p + (per_assignment ? size_t(A) * DS : 0), groups[per_assignment], nullptr);
                else if (k == 2) mm.fmt->expert_matvec(mm, x, y, rows, routed.p, n, c.topk, per_assignment, nullptr);
                else sliced(mm, x, y, rows, cols, per_assignment);
            };
            if (kg == 1 && ku == 1 && mg.fmt == mu.fmt) mg.fmt->expert_matvec_dedup2(mg, mu, xr.p, eg.p, eu.p, c.expert_ff, dmap.p, groups[0], nullptr);
            else { run(mg, kg, xr.p, eg.p, c.expert_ff, h, false); run(mu, ku, xr.p, eu.p, c.expert_ff, h, false); }
            tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(eg.count), nullptr);
            run(md, kd, ax.p, ed.p, h, c.expert_ff, true);
            shared_expert();
            tqllm_cuda::fn_moe_combine(ed.p, weights.p, sd.p, shared_weights.p, fo.p, h, c.topk, n, nullptr);
        }
        auto updated = std::make_unique<Device<float>>(size_t(width) * n);
        tqllm_cuda::hc_combine(combined.p, fo.p, inject.p, updated->p, h, c.hc, n, nullptr); residual = std::move(updated);
    }

    // Runs n new tokens through the trunk; returns the head-input rows [n][hidden]; keeps the wide rows in res_keep.
    std::unique_ptr<Device<float>> step(const std::vector<int> & tokens) {
        const int n = int(tokens.size());
        if (g_arena) g_arena->used = 0;
        spec_pos0 = position; spec_n = n; spec_hist0 = history.size();
        std::vector<uint64_t> rows(tokens.begin(), tokens.end());
        const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()); de.put(embedding);
        auto residual = std::make_unique<Device<float>>(size_t(width) * n);
        tqllm_cuda::hc_init(de.p, residual->p, h, c.hc, n, nullptr);
        const auto ids = history.append_batch(tokens);
        std::vector<uint64_t> ple_rows(ids.begin(), ids.end());
        const auto ple = cp.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
        Device<float> dp(ple.size()); dp.put(ple);
        // prefill: hand back each layer's scratch; the residual moves through two persistent buffers (speculative steps keep
        // their per-layer rollback state in the arena, so they keep one arena span per step)
        const bool layer_reset = g_arena && !spec && !std::getenv("TQFN_NO_LAYER_RESET");
        const size_t base = g_arena ? g_arena->used : 0;
        for (int layer = 0; layer < c.layers; ++layer) {
            layer_forward(layer, residual, n, dp.p, nullptr, experts);
            if (!layer_reset) continue;
            auto & pp = pingpong[layer & 1];
            if (!pp || pp->count < size_t(n) * width) { PersistentAlloc keep; pp = std::make_unique<Device<float>>(size_t(n) * width); }
            ck(cudaMemcpyAsync(pp->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            residual = std::make_unique<Device<float>>(Device<float>::View{}, pp->p, size_t(n) * width);
            g_arena->used = base;
        }
        position += n;
        if (res_keep->count < size_t(n) * width) { PersistentAlloc keep; res_keep = std::make_unique<Device<float>>(size_t(n) * width); }
        ck(cudaMemcpyAsync(res_keep->p, residual->p, size_t(n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        Device<float> norm(size_t(width) * n);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h) * n);
        mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }

    // One verify pass over several sequences (this model first): row-independent work (projections, router, experts)
    // runs once for all rows, PLE history and the attention / recurrent cores run per sequence. Returns the head-input
    // rows of every sequence, concatenated in the order given; each model keeps its wide rows in res_keep.
    std::unique_ptr<Device<float>> step_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks) {
        if (g_arena) g_arena->used = 0;
        std::vector<Seg> segs; std::vector<uint64_t> rows, ple_rows; int n = 0;
        for (size_t i = 0; i < ms.size(); ++i) {
            DecodeModel & m = *ms[i]; const int nn = int(toks[i].size());
            m.spec_pos0 = m.position; m.spec_n = nn; m.spec_hist0 = m.history.size();
            segs.push_back({&m, n, nn}); n += nn;
            rows.insert(rows.end(), toks[i].begin(), toks[i].end());
            const auto ids = m.history.append_batch(toks[i]); ple_rows.insert(ple_rows.end(), ids.begin(), ids.end());
        }
        const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()); de.put(embedding);
        auto residual = std::make_unique<Device<float>>(size_t(width) * n);
        tqllm_cuda::hc_init(de.p, residual->p, h, c.hc, n, nullptr);
        const auto ple = cp.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", ple_rows);
        Device<float> dp(ple.size()); dp.put(ple);
        for (int layer = 0; layer < c.layers; ++layer) layer_forward(layer, residual, n, dp.p, nullptr, experts, &segs);
        for (const auto & sg : segs) {
            DecodeModel & m = *sg.m; m.position += sg.n;
            if (m.res_keep->count < size_t(sg.n) * width) { PersistentAlloc keep; m.res_keep = std::make_unique<Device<float>>(size_t(sg.n) * width); }
            ck(cudaMemcpyAsync(m.res_keep->p, residual->p + size_t(sg.off) * width, size_t(sg.n) * width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        }
        Device<float> norm(size_t(width) * n);
        tqllm_cuda::hc_norm(residual->p, cp.dense_device("output_hc_norm.weight"), norm.p, h, c.hc, n, float(cp.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h) * n);
        mix_device(cp, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }

    // Keep the first m of the last step's n tokens (speculative verify): restore every carried state to token m-1.
    void accept(int m) {
        const int n = spec_n;
        if (m >= n) return;
        require(spec && m >= 1, "accept: speculative state was not recorded");
        for (int layer = 0; layer < c.layers; ++layer) {
            auto & r = rec[layer];
            if (r.state && r.conv_saved) {
                const size_t ch = r.qkv_saved->count / size_t(n), taps1 = r.history->count / ch;
                // rerun the same recurrence for m tokens from the pre-block state (bit-identical to a per-token state)
                tqllm_cuda::fn_gdn(r.conv_saved->p, r.a_saved->p, r.b_saved->p, r.neg_a, r.bias, r.state2->p, r.state->p,
                                   r.replay_out->p, r.kh, r.vh, r.dim, m, r.eps, nullptr);
                for (size_t j = 0; j < taps1; ++j) {  // history = last taps-1 rows of (old history ++ block inputs[0..m))
                    const size_t row = size_t(m) + j;
                    const float * src = row < taps1 ? r.old_history->p + row * ch : r.qkv_saved->p + (row - taps1) * ch;
                    ck(cudaMemcpyAsync(r.history->p + j * ch, src, ch * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                }
            }
            if (ple_past[layer] && ple_saved[layer]) {
                const size_t hist_rows = ple_past[layer]->count / size_t(width);
                for (size_t j = 0; j < hist_rows; ++j) {
                    const size_t row = size_t(m) + j;
                    const float * src = row < hist_rows ? ple_old[layer]->p + row * width : ple_saved[layer]->p + (row - hist_rows) * width;
                    ck(cudaMemcpyAsync(ple_past[layer]->p + j * width, src, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                }
            }
            if (layer < int(att.size()) && att[layer].k) att[layer].len = spec_pos0 + m;
        }
        history.rollback(spec_hist0 + size_t(m));
        position = spec_pos0 + m;
    }

    // MTP block on n (token, wide residual) pairs: appends n MTP KV entries; returns the argmax after the last pair and
    // keeps its wide output in mtp_h (input of the next chained draft).
    int mtp_batch(const std::vector<int> & tokens, const float * h_rows) {
        require(mtp_experts, "MTP not loaded (--mtp)");
        const int n = int(tokens.size()), L = c.layers;
        const std::string nx = "blk." + std::to_string(L) + ".nextn.";
        std::vector<uint64_t> rows(tokens.begin(), tokens.end());
        const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()), en(embedding.size()), hn(size_t(n) * width), cat(size_t(n) * c.hc * 2 * h); de.put(embedding);
        tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, n, float(cp.epsilon), nullptr);
        tqllm_cuda::hc_norm(h_rows, cp.dense_device(nx + "hnorm.weight"), hn.p, h, c.hc, n, float(cp.epsilon), nullptr);
        for (int sidx = 0; sidx < c.hc; ++sidx)
            ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                 size_t(h) * sizeof(float), size_t(n), cudaMemcpyDeviceToDevice, nullptr));
        ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                             size_t(n) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
        auto residual = std::make_unique<Device<float>>(size_t(n) * width);
        cp.project(nx + "eh_proj.weight", cat.p, residual->p, 2 * h, h, n * c.hc);
        layer_forward(L, residual, n, nullptr, &mtp_att, *mtp_experts);
        ck(cudaMemcpyAsync(mtp_h->p, residual->p + size_t(n - 1) * width, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        Device<float> norm{size_t(width)}, out{size_t(h)};
        tqllm_cuda::hc_norm(residual->p + size_t(n - 1) * width, cp.dense_device(nx + "hc_head_norm.weight"), norm.p, h, c.hc, 1, float(cp.epsilon), nullptr);
        mix_device(cp, nx + "hc_head_", norm.p, 1, out.p, nullptr);
        return cp.draft_head_argmax_device(out.p, 1);
    }
    // Chained draft from the MTP's own last output; its KV entry is dropped (the next catch-up recomputes it).
    int mtp_chain(int token) {
        Device<float> hcopy{size_t(width)};
        ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
        const int d = mtp_batch({token}, hcopy.p);
        mtp_att.len -= 1;
        return d;
    }
    // mtp_batch over several sequences (this model first): one MTP-layer pass, per-sequence MTP KV and mtp_h. Returns each
    // sequence's argmax after its last pair. Sequences are grouped so the routed experts of a group fit the MTP cache.
    std::vector<int> mtp_batch_multi(const std::vector<DecodeModel *> & ms, const std::vector<std::vector<int>> & toks,
                                     const std::vector<const float *> & hs) {
        require(mtp_experts, "MTP not loaded (--mtp)");
        std::vector<int> result(ms.size());
        const int cap = std::max(1, mtp_experts->slots() / c.topk);
        for (size_t g0 = 0; g0 < ms.size();) {
            size_t g1 = g0; int N = 0;
            while (g1 < ms.size() && (g1 == g0 || N + int(toks[g1].size()) <= cap)) N += int(toks[g1++].size());
            const int L = c.layers; const std::string nx = "blk." + std::to_string(L) + ".nextn.";
            std::vector<Seg> segs; std::vector<uint64_t> rows;
            for (size_t i = g0, off = 0; i < g1; ++i) { segs.push_back({ms[i], int(off), int(toks[i].size())}); off += toks[i].size(); rows.insert(rows.end(), toks[i].begin(), toks[i].end()); }
            const auto embedding = cp.file("token_embd.weight").gather_rows("token_embd.weight", rows);
            Device<float> de(embedding.size()), en(embedding.size()), hn(size_t(N) * width), cat(size_t(N) * c.hc * 2 * h); de.put(embedding);
            tqllm_cuda::hc_norm(de.p, cp.dense_device(nx + "enorm.weight"), en.p, h, 1, N, float(cp.epsilon), nullptr);
            for (size_t i = g0; i < g1; ++i) {
                const auto & sg = segs[i - g0];
                tqllm_cuda::hc_norm(hs[i], cp.dense_device(nx + "hnorm.weight"), hn.p + size_t(sg.off) * width, h, c.hc, sg.n, float(cp.epsilon), nullptr);
            }
            for (int sidx = 0; sidx < c.hc; ++sidx)
                ck(cudaMemcpy2DAsync(cat.p + size_t(sidx) * 2 * h, size_t(c.hc) * 2 * h * sizeof(float), en.p, size_t(h) * sizeof(float),
                                     size_t(h) * sizeof(float), size_t(N), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpy2DAsync(cat.p + h, size_t(2) * h * sizeof(float), hn.p, size_t(h) * sizeof(float), size_t(h) * sizeof(float),
                                 size_t(N) * c.hc, cudaMemcpyDeviceToDevice, nullptr));
            auto residual = std::make_unique<Device<float>>(size_t(N) * width);
            cp.project(nx + "eh_proj.weight", cat.p, residual->p, 2 * h, h, N * c.hc);
            layer_forward(L, residual, N, nullptr, nullptr, *mtp_experts, &segs);
            for (size_t i = g0; i < g1; ++i) {
                const auto & sg = segs[i - g0]; DecodeModel & m = *sg.m;
                const float * last = residual->p + size_t(sg.off + sg.n - 1) * width;
                ck(cudaMemcpyAsync(m.mtp_h->p, last, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                Device<float> norm{size_t(width)}, out{size_t(h)};
                tqllm_cuda::hc_norm(last, cp.dense_device(nx + "hc_head_norm.weight"), norm.p, h, c.hc, 1, float(cp.epsilon), nullptr);
                mix_device(cp, nx + "hc_head_", norm.p, 1, out.p, nullptr);
                result[i] = cp.draft_head_argmax_device(out.p, 1);
            }
            g0 = g1;
        }
        return result;
    }
    // Drafts d2..dD from d1 for several sequences at once (same chain as mtp_drafts, one MTP pass per chain step).
    std::vector<std::vector<int>> mtp_drafts_multi(const std::vector<DecodeModel *> & ms, const std::vector<int> & d1, int count) {
        std::vector<std::vector<int>> d(ms.size());
        for (size_t i = 0; i < ms.size(); ++i) d[i] = {d1[i]};
        std::vector<std::unique_ptr<Device<float>>> hc(ms.size());
        for (auto & x : hc) x = std::make_unique<Device<float>>(size_t(width));
        for (int k = 1; k < count; ++k) {
            std::vector<std::vector<int>> t(ms.size()); std::vector<const float *> hp(ms.size());
            for (size_t i = 0; i < ms.size(); ++i) {
                ck(cudaMemcpyAsync(hc[i]->p, ms[i]->mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
                t[i] = {d[i].back()}; hp[i] = hc[i]->p;
            }
            const auto r = mtp_batch_multi(ms, t, hp);
            for (size_t i = 0; i < ms.size(); ++i) d[i].push_back(r[i]);
        }
        for (auto * m : ms) m->mtp_att.len -= count - 1;
        return d;
    }

    // Drafts d2..dD from d1: each chain step sees the previous drafts' MTP KV; all chain entries are dropped at the end.
    std::vector<int> mtp_drafts(int d1, int count) {
        std::vector<int> d{d1};
        Device<float> hcopy{size_t(width)};
        for (int i = 1; i < count; ++i) {
            ck(cudaMemcpyAsync(hcopy.p, mtp_h->p, size_t(width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            d.push_back(mtp_batch({d.back()}, hcopy.p));
        }
        mtp_att.len -= count - 1;
        return d;
    }

    // Router logits [n][experts + 1] (last = shared-expert gate). BF16 router weights: per-token GEMVs on the raw weights.
    void router(const std::string & prefix, const float * x, float * logits, int n) {
        const bool bf16 = cp.file(prefix + "ffn_gate_inp.weight").file().tensors.at(prefix + "ffn_gate_inp.weight").type == GGML_BF16;
        if (bf16 && std::getenv("TQFN_F32_ROUTER") == nullptr && n > 64) {
            // prefill chunks only (decode / batched verify keep the per-token GEMVs, faster at <= 64 tokens); same sums, in place
            tqllm_cuda::fn_gemv_bf16_chunk(cp.bf16_device(prefix + "ffn_gate_inp.weight"), x, logits, c.experts, h, n, c.experts + 1, nullptr);
            tqllm_cuda::fn_gemv_bf16_chunk(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), x, logits + c.experts, 1, h, n, c.experts + 1, nullptr);
        } else if (bf16 && std::getenv("TQFN_F32_ROUTER") == nullptr) {
            Device<float> rl{size_t(n) * c.experts}, sl{size_t(n)};
            tqllm_cuda::fn_gemv_bf16_tokens(cp.bf16_device(prefix + "ffn_gate_inp.weight"), x, rl.p, c.experts, h, n, nullptr);
            tqllm_cuda::fn_gemv_bf16_tokens(cp.bf16_device(prefix + "ffn_gate_inp_shexp.weight"), x, sl.p, 1, h, n, nullptr);
            ck(cudaMemcpy2DAsync(logits, size_t(c.experts + 1) * sizeof(float), rl.p, size_t(c.experts) * sizeof(float), size_t(c.experts) * sizeof(float), size_t(n), cudaMemcpyDeviceToDevice, nullptr));
            ck(cudaMemcpy2DAsync(logits + c.experts, size_t(c.experts + 1) * sizeof(float), sl.p, sizeof(float), sizeof(float), size_t(n), cudaMemcpyDeviceToDevice, nullptr));
        } else tqllm_cuda::moe_router_f32(cp.dense_device(prefix + "ffn_gate_inp.weight"), cp.dense_device(prefix + "ffn_gate_inp_shexp.weight"),
                                          x, logits, h, c.experts, n, nullptr);
    }
    static void ck_zero(Device<float> & d) { ck(cudaMemset(d.p, 0, d.count * sizeof(float))); }
    static void ck_copy(Device<float> & dst, const Device<float> & src) { ck(cudaMemcpy(dst.p, src.p, src.count * sizeof(float), cudaMemcpyDeviceToDevice)); }
    static void ck_copy_n(Device<float> & dst, const float * src, size_t count) { ck(cudaMemcpyAsync(dst.p, src, count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr)); }
};

// Packed expert file (TQFNEXP1): header, then per layer [6 x uint64 sizes: gate data/aux, up data/aux, down data/aux],
// then every expert of every layer as one contiguous block in the decode cache's row layout (no repack on a miss).
struct PackedExperts {
    int layers = 0, experts = 0; std::vector<std::array<uint64_t, 6>> sizes; std::vector<uint64_t> base, block;
    const char * map = nullptr; size_t bytes = 0; bool tiled = false;
    const char * at(int layer, int id) const { return map + base[layer] + uint64_t(id) * block[layer]; }
};
static PackedExperts open_packed(const std::string & path) {
    PackedExperts pe; FILE * f = std::fopen(path.c_str(), "rb"); require(f, "cannot open " + path);
    char magic[8]; int32_t dims[2];
    require(std::fread(magic, 1, 8, f) == 8 && (std::memcmp(magic, "TQFNEXP1", 8) == 0 || std::memcmp(magic, "TQFNEXPT", 8) == 0) &&
            std::fread(dims, 4, 2, f) == 2, "not a TQFNEXP1 / TQFNEXPT file");
    pe.tiled = std::memcmp(magic, "TQFNEXPT", 8) == 0;
    pe.layers = dims[0]; pe.experts = dims[1]; pe.sizes.resize(pe.layers);
    for (auto & z : pe.sizes) require(std::fread(z.data(), 8, 6, f) == 6, "packed header truncated");
    uint64_t off = 16 + uint64_t(pe.layers) * 48;
    for (int l = 0; l < pe.layers; ++l) { uint64_t b = 0; for (auto v : pe.sizes[l]) b += v; pe.base.push_back(off); pe.block.push_back(b); off += b * pe.experts; }
    std::fseek(f, 0, SEEK_END); pe.bytes = size_t(std::ftell(f)); std::fclose(f);
    require(pe.bytes == off, "packed file size differs from its header");
    const int fd = ::open(path.c_str(), O_RDONLY); require(fd >= 0, "cannot open " + path);
    void * m = ::mmap(nullptr, pe.bytes, PROT_READ, MAP_SHARED, fd, 0); ::close(fd);
    require(m != MAP_FAILED, "cannot map " + path); pe.map = static_cast<const char *>(m); return pe;
}
// u4h kernel id of each projection's device format (what the grouped kernels see), from one expert's load
static std::array<int, 3> expert_u4h_ids(const Checkpoint & checkpoint, int layer) {
    const auto & c = checkpoint.config; std::array<int, 3> ids{}; const auto payload = checkpoint.expert_payload(layer, 0);
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? c.expert_ff : c.hidden, rows = p == 2 ? c.hidden : c.expert_ff;
        std::vector<std::vector<char>> keep_bytes;
        Upload keep = [&](const void * data, size_t bytes) {
            keep_bytes.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes); return keep_bytes.back().data(); };
        ids[p] = weight_format(payload[p].type)->load(
            std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, keep, nullptr).fmt->u4h_id();
    }
    return ids;
}
// One expert in the device layout (data, aux per projection), as the packed file and the MTP prepack store it.
static std::array<std::vector<char>, 6> repack_expert(const Checkpoint & checkpoint, int layer, int id) {
    const auto & c = checkpoint.config;
    std::array<std::vector<char>, 6> parts; const auto payload = checkpoint.expert_payload(layer, id);
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? c.expert_ff : c.hidden, rows = p == 2 ? c.hidden : c.expert_ff;
        std::vector<std::vector<char>> pieces; std::vector<const void *> where; pieces.reserve(4);
        Upload keep = [&](const void * data, size_t bytes) {
            pieces.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes);
            where.push_back(pieces.back().data()); return pieces.back().data();
        };
        const auto m = weight_format(payload[p].type)->load(
            std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, keep, nullptr);
        for (size_t k = 0; k < pieces.size(); ++k) {
            if (where[k] == m.p) parts[2 * p] = std::move(pieces[k]);
            else if (where[k] == m.aux) parts[2 * p + 1] = std::move(pieces[k]);
        }
        require(!parts[2 * p].empty(), "loader produced no row data");
        // TQFN_TILED: lane-contiguous code tiles for the grouped expert kernels (bytes permuted, values unchanged)
        const int bpb = g_tile_experts ? tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id()) : 0;
        if (bpb && rows % 16 == 0 && (cols / 32) % 4 == 0 && parts[2 * p].size() == size_t(rows) * (cols / 32) * bpb) {
            std::vector<char> t(parts[2 * p].size());
            tqllm_cuda::tile_expert_rows(parts[2 * p].data(), t.data(), rows, cols, bpb);
            parts[2 * p] = std::move(t);
        }
    }
    return parts;
}
static void run_pack_experts(int argc, char ** argv) {
    require(argc >= 3, "usage: tqfn-stage --pack-experts OUT.bin shard1.gguf [more] [--gsq-rco]");
    const std::string out = argv[1];
    std::vector<char *> args{argv[0], argv[0]}; for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
    Checkpoint checkpoint(int(args.size()), args.data()); const auto & c = checkpoint.config;
    auto repack = [&](int layer, int id) { return repack_expert(checkpoint, layer, id); };
    FILE * f = std::fopen(out.c_str(), "wb"); require(f, "cannot create " + out);
    g_tile_experts = true;
    std::fwrite(g_tile_experts ? "TQFNEXPT" : "TQFNEXP1", 1, 8, f); int32_t dims[2] = {c.layers, c.experts}; std::fwrite(dims, 4, 2, f);
    std::vector<std::array<uint64_t, 6>> sizes(c.layers);
    for (int l = 0; l < c.layers; ++l) { const auto parts = repack(l, 0); for (int k = 0; k < 6; ++k) sizes[l][k] = parts[k].size(); std::fwrite(sizes[l].data(), 8, 6, f); }
    for (int l = 0; l < c.layers; ++l) {
        for (int e = 0; e < c.experts; ++e) {
            const auto parts = repack(l, e);
            for (int k = 0; k < 6; ++k) { require(parts[k].size() == sizes[l][k], "expert layout varies within a layer"); std::fwrite(parts[k].data(), 1, parts[k].size(), f); }
        }
        std::printf("packed layer %d\n", l); std::fflush(stdout);
    }
    require(std::fclose(f) == 0, "write failed: " + out);
    std::printf("PACK_DONE %s\n", out.c_str());
}

static void run_decode(int argc, char ** argv) {
    require(argc >= 5, "usage: tqfn-stage --decode fixture-dir prompt-ids.txt N shard1.gguf [more] [--gsq-rco] [--slots S] [--budget-gb G] [--chunk C] [--check]");
    const std::string ids_file = argv[2]; const int max_new = std::stoi(argv[3]);
    int slots = 160, chunk = 8; double budget_gb = 14; bool check = false, spec_on = false; int drafts = 2; std::string packed_path;
    std::vector<char *> args{argv[0], argv[1]};
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slots" && i + 1 < argc) { slots = std::stoi(argv[++i]); continue; }
        if (a == "--budget-gb" && i + 1 < argc) { budget_gb = std::stod(argv[++i]); continue; }
        if (a == "--chunk" && i + 1 < argc) { chunk = std::stoi(argv[++i]); continue; }
        if (a == "--check") { check = true; continue; }
        if (a == "--experts" && i + 1 < argc) { packed_path = argv[++i]; continue; }
        if (a == "--spec") { spec_on = true; continue; }
        if (a == "--drafts" && i + 1 < argc) { drafts = std::stoi(argv[++i]); continue; }
        args.push_back(argv[i]);
    }
    { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
      ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
    // TQFN_ARENA_MB: scratch arena size (default 1024); the high-water mark is printed at exit
    static ScratchArena arena; arena.size = size_t(std::getenv("TQFN_ARENA_MB") ? std::atoi(std::getenv("TQFN_ARENA_MB")) : 512) << 20;
    static struct ArenaReport { ~ArenaReport() { if (g_arena) std::printf("ARENA high-water %.1f MB of %.1f MB\n", g_arena->high / 1048576.0, g_arena->size / 1048576.0); } } arena_report; ck(cudaMalloc(&arena.base, arena.size)); g_arena = &arena;
    Fixtures fixtures(args[1]); Checkpoint checkpoint(int(args.size()), args.data()); checkpoint.resident = true;
    const auto & c = checkpoint.config;
    // TQFN_CPU_BENCH=threads: time the CPU miss pool alone on real expert bytes (no GPU work), K experts x n tokens per call
    // TQFN_PACK_CHECK=packed.tqfnexp: is each packed expert part byte-identical to its GGUF slice? (CPU pool could read it)
    if (const char * pc = std::getenv("TQFN_PACK_CHECK")) {
        const auto pk = open_packed(pc); std::map<std::string, int> tally;
        for (int l = 0; l < c.layers; ++l) {
            const auto pl = checkpoint.expert_payload(l, 7); const char * blk = pk.at(l, 7); size_t off = 0;
            for (int m = 0; m < 3; ++m) {
                const size_t d = pk.sizes[l][2 * m], a = pk.sizes[l][2 * m + 1];
                const bool same = a == 0 && d == pl[m].bytes.size && std::memcmp(blk + off, pl[m].bytes.data, d) == 0;
                ++tally["type " + std::to_string(pl[m].type) + " u4h " + std::to_string(expert_u4h_ids(checkpoint, l)[m]) + " tbpb " +
                        std::to_string(tqllm_cuda::tiled_block_bytes(expert_u4h_ids(checkpoint, l)[m])) + (m == 2 ? " down" : " gate/up") + (same ? " SAME" : " DIFFERS") +
                        " (gguf " + std::to_string(pl[m].bytes.size) + ", data " + std::to_string(d) + ", aux " + std::to_string(a) + ")"];
                off += d + a;
            }
        }
        for (auto & [k, v] : tally) std::printf("PACK_CHECK %s x%d\n", k.c_str(), v);
        std::fflush(stdout); std::_Exit(0);
    }
    if (const char * cb = std::getenv("TQFN_CPU_BENCH")) {
        CpuMissPool pool(std::atoi(cb), c.hidden, c.expert_ff);
        std::vector<float> x(size_t(4) * c.hidden); for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.37f * float(i)) * 0.5f;
        std::vector<__nv_bfloat16> out(size_t(4) * c.topk * c.hidden);
        for (int k : {1, 2, 3, 4}) for (int n : {1, 4}) {
            double total = 0; int calls = 0;
            for (int rep = 0; rep < 3; ++rep) for (int layer = 0; layer < c.layers; ++layer) {
                std::vector<CpuMissPool::Job> jobs;
                for (int e = 0; e < k; ++e) {
                    CpuMissPool::Job j; const auto pl = checkpoint.expert_payload(layer, (37 * e + 11 * layer) % c.experts);
                    const int rows[3] = {c.expert_ff, c.expert_ff, c.hidden};
                    for (int m = 0; m < 3; ++m) { j.w[m] = pl[m].bytes.data; j.type[m] = pl[m].type; j.row_bytes[m] = pl[m].bytes.size / size_t(rows[m]); }
                    j.ncols = n == 1 ? 1 : 2; for (int q = 0; q < j.ncols; ++q) j.cols_t[q] = q;  // a verify window: ~2 tokens per missed expert
                    jobs.push_back(j);
                }
                const auto t0 = std::chrono::steady_clock::now(); pool.start(x.data(), n, std::move(jobs), out.data()); pool.wait();
                if (rep) { total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); ++calls; }
            }
            std::printf("CPU_BENCH threads %s experts %d tokens %d: %.3f ms per call\n", cb, k, n, total / calls);
        }
        std::fflush(stdout); std::_Exit(0);
    }
    // A ".list" file names several token-id files: run them in order on one cache (like a server), report each + mean.
    std::vector<std::pair<std::string, std::vector<int>>> prompts;
    auto read_ids = [](const std::string & f) { std::vector<int> v; std::ifstream in(f); for (int id; in >> id;) v.push_back(id); return v; };
    if (ids_file.size() > 5 && ids_file.substr(ids_file.size() - 5) == ".list") {
        std::ifstream in(ids_file); for (std::string line; std::getline(in, line);) if (!line.empty()) prompts.push_back({line, read_ids(line)});
    } else prompts.push_back({ids_file, read_ids(ids_file)});
    for (const auto & [name, ids] : prompts)
        require(!ids.empty() && int(ids.size()) + max_new <= 2048, "prompt empty or beyond the dense-attention budget: " + name);
    const auto prompt = prompts[0].second;
    require(chunk * c.topk <= slots, "prefill chunk needs more unique experts than slots");
    MixedExpertCache cache(c.layers, c.experts, c.hidden, c.expert_ff, slots, size_t(budget_gb * 1024 * 1024 * 1024),
                           [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
    apply_layer_slots(cache);
    PackedExperts packed;
    if (!packed_path.empty()) {
        packed = open_packed(packed_path);
        require(packed.layers == c.layers && packed.experts == c.experts, "packed experts do not match the model");
        const bool pin = std::getenv("TQFN_NO_PIN") == nullptr;
        if (pin) {
            // File-backed mappings cannot be registered: copy the file into pinned anonymous memory once, then drop the
            // page-cache copy so RAM does not hold it twice.
            const auto t0 = std::chrono::steady_clock::now();
            char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), packed.bytes));
            const int fd = ::open(packed_path.c_str(), O_RDONLY); require(fd >= 0, "cannot reopen " + packed_path);
            for (size_t done = 0; done < packed.bytes;) {
                const ssize_t r = ::pread(fd, host + done, std::min<size_t>(size_t(1) << 28, packed.bytes - done), off_t(done));
                require(r > 0, "short read of " + packed_path); done += size_t(r);
            }
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd);
            ::munmap(const_cast<char *>(packed.map), packed.bytes); packed.map = host;
            std::printf("DECODE packed experts in pinned memory (%.1f GB) in %.1f s\n", packed.bytes / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        // TQFN_TILED with a plain TQFNEXP1 file: tile the code parts in place in host memory (no second 50 GB file)
        if (!packed.tiled) {
            require(pin, "TQFN_TILED on a TQFNEXP1 file needs the pinned copy (unset TQFN_NO_PIN)");
            const auto t0 = std::chrono::steady_clock::now(); char * base = const_cast<char *>(packed.map);
            for (int l = 0; l < packed.layers; ++l) {
                size_t off = 0; const auto ids = expert_u4h_ids(checkpoint, l);
                for (int pp = 0; pp < 3; ++pp) {
                    const int cols = pp == 2 ? c.expert_ff : c.hidden, rows = pp == 2 ? c.hidden : c.expert_ff;
                    const int bpb = tqllm_cuda::tiled_block_bytes(ids[pp]);  // the device format
                    const size_t dsz = packed.sizes[l][2 * pp];
                    if (bpb) require(rows % 16 == 0 && (cols / 32) % 4 == 0 && dsz == size_t(rows) * (cols / 32) * bpb,
                                     "tileable expert part with an unexpected layout");
                    if (bpb) {
                        static int tiled_parts = 0; ++tiled_parts;
                        static struct Rep { ~Rep() { std::printf("DECODE tiled expert parts %d\n", tiled_parts); } } rep;
                        const int nt = 16; std::vector<std::thread> th;
                        for (int w = 0; w < nt; ++w) th.emplace_back([&, w] {
                            std::vector<char> tmp(dsz);
                            for (int e = w; e < packed.experts; e += nt) {
                                char * d = base + packed.base[l] + uint64_t(e) * packed.block[l] + off;
                                tqllm_cuda::tile_expert_rows(d, tmp.data(), rows, cols, bpb); std::memcpy(d, tmp.data(), dsz);
                            }
                        });
                        for (auto & t : th) t.join();
                    }
                    off += dsz + packed.sizes[l][2 * pp + 1];
                }
            }
            packed.tiled = true;
            std::printf("DECODE expert code parts tiled in host memory in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        if (packed.tiled) { tqllm_cuda::set_tiled_experts(true); g_tile_experts = true; std::printf("DECODE tiled expert layout\n"); }
        if (std::getenv("TQFN_CPU_MISS") && pin && std::getenv("TQFN_CPU_GGUF") == nullptr) {  // CPU pool reads the pinned copy
            static CpuPackedView pv; pv.map = packed.map; pv.base = packed.base; pv.block = packed.block; pv.sizes = packed.sizes;
            pv.tiled = packed.tiled; pv.layers = packed.layers;
            for (int l = 0; l < packed.layers; ++l) pv.fmt.push_back(expert_u4h_ids(checkpoint, l));
            g_cpu_packed = &pv; std::printf("DECODE CPU pool reads the pinned packed experts\n");
        }
        cache.set_prepacked([&](int layer, int id) -> const char * { return layer < packed.layers ? packed.at(layer, id) : nullptr; }, pin);
        std::printf("DECODE experts from %s (%.1f GB, no repack on a miss)\n", packed_path.c_str(), packed.bytes / 1e9);
    }
    std::unique_ptr<MixedExpertCache> mtp_cache; char * mtp_packed = nullptr;
    if (checkpoint.has_mtp) {
        const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : 256;
        mtp_cache = std::make_unique<MixedExpertCache>(c.layers + 1, c.experts, c.hidden, c.expert_ff, mtp_slots, size_t(mtp_slots) * (6u << 20) + (64u << 20),
            [&](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
        std::printf("DECODE MTP block loaded (blk.%d)%s\n", c.layers, spec_on ? ", speculative verify of 2 drafts" : "");
        if (!packed_path.empty() && std::getenv("TQFN_NO_PIN") == nullptr) {
            // MTP experts in the device layout in pinned memory: a miss is a kernel copy, not a CPU repack + upload.
            const auto t0 = std::chrono::steady_clock::now();
            const auto first = repack_expert(checkpoint, c.layers, 0); size_t block = 0; for (const auto & p : first) block += p.size();
            char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), block * size_t(c.experts)));
            std::vector<std::thread> pool; std::atomic<int> next{0};
            for (int t = 0; t < 16; ++t) pool.emplace_back([&] {
                for (int e; (e = next++) < c.experts;) {
                    const auto parts = repack_expert(checkpoint, c.layers, e); size_t off = 0;
                    for (const auto & p : parts) { std::memcpy(host + size_t(e) * block + off, p.data(), p.size()); off += p.size(); }
                    require(off == block, "MTP expert layout varies");
                }
            });
            for (auto & t : pool) t.join();
            mtp_packed = host;
            mtp_cache->set_prepacked([&, block](int layer, int id) -> const char * { return layer == c.layers ? mtp_packed + size_t(id) * block : nullptr; }, true);
            std::printf("DECODE MTP experts prepacked in pinned memory (%.2f GB) in %.1f s\n", block * double(c.experts) / 1e9,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    uint64_t spec_steps = 0, spec_tokens = 0;
    auto run = [&](const std::vector<int> & prompt_ids, int generate, std::vector<int> * all_argmax) {
        DecodeModel model(checkpoint, fixtures, cache, prompt_ids.size() + generate + 16);
        model.mtp_experts = mtp_cache.get();
        if (mtp_cache && model.miss_stream) mtp_cache->set_miss_stream(model.miss_stream);  // MTP experts on the split grouped path
        if (const char * tf = std::getenv("TQFN_TF_LOGITS")) {
            // Teacher-forced decode probe: prefill 64 tokens, then feed the prompt's remaining tokens in verify-width blocks
            // of 3 through the decode kernels; writes float32 logits [blocks * 3][vocab] to compare numerics changes.
            const size_t P0 = 64; require(prompt_ids.size() >= P0 + 3, "TQFN_TF_LOGITS needs a prompt of 67+ tokens");
            for (size_t b = 0; b < P0; b += size_t(chunk)) { auto r = model.step(std::vector<int>(prompt_ids.begin() + b, prompt_ids.begin() + std::min(P0, b + size_t(chunk)))); (void) r; }
            std::FILE * f = std::fopen(tf, "wb"); require(f != nullptr, std::string("cannot open ") + tf); int blocks = 0;
            for (size_t b = P0; b + 3 <= prompt_ids.size(); b += 3, ++blocks) {
                auto rows = model.step(std::vector<int>(prompt_ids.begin() + b, prompt_ids.begin() + b + 3));
                const auto lg = checkpoint.head_logits_device(rows->p, 3);
                require(std::fwrite(lg.data(), sizeof(float), lg.size(), f) == lg.size(), "short write of teacher-forced logits");
            }
            std::fclose(f); std::printf("TF_LOGITS %s blocks %d vocab %zu\n", tf, blocks, size_t(checkpoint.head_resident().m.rows));
            return std::vector<int>{};
        }
        const bool use_spec = spec_on && mtp_cache && !all_argmax && generate > 1;
        std::unique_ptr<Device<float>> prompt_res;
        if (use_spec) { PersistentAlloc keep; prompt_res = std::make_unique<Device<float>>(prompt_ids.size() * size_t(model.width)); }
        std::vector<int> out;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t b = 0; b < prompt_ids.size(); b += chunk) {
            std::vector<int> part(prompt_ids.begin() + b, prompt_ids.begin() + std::min(prompt_ids.size(), b + chunk));
            auto rows = model.step(part);
            if (prompt_res) ck(cudaMemcpyAsync(prompt_res->p + b * size_t(model.width), model.res_keep->p, part.size() * size_t(model.width) * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            if (all_argmax) for (int i = 0; i < int(part.size()); ++i) all_argmax->push_back(checkpoint.head_argmax_device(rows->p + size_t(i) * c.hidden, 1));
            if (b + chunk >= prompt_ids.size()) out.push_back(checkpoint.head_argmax_device(rows->p, int(part.size())));
        }
        ck(cudaDeviceSynchronize());
        if (std::getenv("TQFN_STEP_PROBE") && !all_argmax) {
            // Cost of one warm step carrying k tokens (Plan A: multi-token verify). Advances the state; timing only.
            for (int k = 1; k <= 4; ++k) {
                for (int w = 0; w < 4; ++w) { auto r = model.step(std::vector<int>(k, out.back())); (void) r; }
                ck(cudaDeviceSynchronize()); const auto a = std::chrono::steady_clock::now();
                for (int w = 0; w < 16; ++w) { auto r = model.step(std::vector<int>(k, out.back())); checkpoint.head_argmax_device(r->p, k); }
                ck(cudaDeviceSynchronize());
                std::printf("STEP_PROBE k=%d %.2f ms/step\n", k, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / 16);
                std::fflush(stdout);
            }
            return out;
        }
        const auto t1 = std::chrono::steady_clock::now(); g_host_ms[0] = g_host_ms[1] = g_host_ms[2] = 0;
        const auto stats0 = cache.stats();
        static const bool stop_eos = std::getenv("TQFN_NO_EOS_STOP") == nullptr && !all_argmax;
        auto is_eos = [](int t) { return t == 248046 || t == 248044; };
        if (use_spec) {
            // MTP catch-up over the prompt: pairs (token i+1, residual i), the last pair uses the first generated token.
            const int P = int(prompt_ids.size()); int d1 = 0;
            for (int b = 0; b < P; b += 4) {
                std::vector<int> toks;
                for (int i = b; i < std::min(P, b + 4); ++i) toks.push_back(i + 1 < P ? prompt_ids[i + 1] : out[0]);
                d1 = model.mtp_batch(toks, prompt_res->p + size_t(b) * model.width);
            }
            model.set_spec(true);
            static const bool st = std::getenv("TQFN_SPEC_TRACE") != nullptr; double tt[4] = {0, 0, 0, 0};
            auto mark = [&](int k, std::chrono::steady_clock::time_point & a) {
                if (!st) return; ck(cudaDeviceSynchronize()); const auto b = std::chrono::steady_clock::now();
                tt[k] += std::chrono::duration<double, std::milli>(b - a).count(); a = b; };
            while (int(out.size()) < generate && !(stop_eos && is_eos(out.back()))) {
                auto a = std::chrono::steady_clock::now();
                const int y = out.back();
                const std::vector<int> dr = model.mtp_drafts(d1, drafts);
                mark(0, a);
                std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
                auto rows = model.step(block);
                const auto v = checkpoint.head_argmax_rows(rows->p, int(block.size()));
                mark(1, a);
                int m = 1; while (m <= drafts && v[m - 1] == dr[m - 1]) ++m;  // m - 1 drafts accepted, plus the verified next token
                for (int i = 0; i < m && int(out.size()) < generate; ++i) { out.push_back(v[i]); if (stop_eos && is_eos(v[i])) { m = i + 1; break; } }
                model.accept(m); mark(2, a);
                ++spec_steps; spec_tokens += m;
                d1 = model.mtp_batch(std::vector<int>(v.begin(), v.begin() + m), model.res_keep->p); mark(3, a);
            }
            if (st && spec_steps) std::printf("SPEC_TRACE per cycle: chain %.2f ms, verify %.2f, accept %.2f, mtp catch-up %.2f\n",
                tt[0] / spec_steps, tt[1] / spec_steps, tt[2] / spec_steps, tt[3] / spec_steps);
            model.set_spec(false);
        } else
        while (int(out.size()) < generate && !(stop_eos && is_eos(out.back()))) {
            auto rows = model.step({out.back()}); out.push_back(checkpoint.head_argmax_device(rows->p, 1));
        }
        ck(cudaDeviceSynchronize());
        const auto t2 = std::chrono::steady_clock::now();
        const auto stats = cache.stats();
        const double pre = std::chrono::duration<double>(t1 - t0).count(), dec = std::chrono::duration<double>(t2 - t1).count();
        const double hits = double(stats.hits - stats0.hits), misses = double(stats.misses - stats0.misses);
        if (generate > 1 && out.size() > 1)
            std::printf("DECODE prompt %zu prefill %.2f s (%.1f tok/s) | generated %d decode %.2f s = %.2f tok/s | expert hit rate %.1f %% (%.0f / %.0f) transfer %.2f GB prefetched %llu (used %llu)\n",
                        prompt_ids.size(), pre, prompt_ids.size() / pre, int(out.size()) - 1, dec, (int(out.size()) - 1) / dec,
                        100 * hits / std::max(1.0, hits + misses), hits, hits + misses, double(stats.transfer_bytes - stats0.transfer_bytes) / 1e9,
                        (unsigned long long) cache.prefetched(), (unsigned long long) cache.prefetch_used());
        std::printf("DECODE prefetch overlap: %llu prepares with queued copies, %llu still running, %llu waited for unread experts\n",
                    (unsigned long long) (stats.pf_pending - stats0.pf_pending), (unsigned long long) (stats.pf_running - stats0.pf_running),
                    (unsigned long long) (stats.pf_waited_unread - stats0.pf_waited_unread));
        { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0); uint64_t res = 0, used = 0;
          cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res); cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
          std::printf("VRAM used %.0f MiB of %.0f MiB; pool reserved %.0f MiB, in use %.0f MiB\n", (tot - fr) / 1048576.0, tot / 1048576.0, res / 1048576.0, used / 1048576.0); }
        std::printf("HOST per generated token: ids %.2f ms, prepare %.2f ms, prefetch %.2f ms\n", g_host_ms[0] / std::max<size_t>(1, out.size() - 1),
                    g_host_ms[1] / std::max<size_t>(1, out.size() - 1), g_host_ms[2] / std::max<size_t>(1, out.size() - 1));
        if (use_spec && spec_steps) std::printf("SPEC steps %llu tokens %llu = %.2f tokens per verify\n", (unsigned long long) spec_steps,
                                                (unsigned long long) spec_tokens, double(spec_tokens) / spec_steps);
        return out;
    };
    if (prompts.size() > 1) {
        double total = 0;
        for (const auto & [name, ids] : prompts) {
            const auto t0 = std::chrono::steady_clock::now(); const auto out = run(ids, max_new, nullptr); (void) t0;
            int stop = int(out.size()); for (int i = 0; i < int(out.size()); ++i) if (out[i] == 248046 || out[i] == 248044) { stop = i + 1; break; }
            std::printf("PROMPT %s generated %zu (first stop at %d)\n", name.c_str(), out.size(), stop);
            std::printf("TOKENS %s", name.c_str()); for (int t : out) std::printf(" %d", t); std::printf("\n"); std::fflush(stdout);
            (void) total;
        }
        return;
    }
    if (std::getenv("TQFN_TF_LOGITS")) { run(prompt, 0, nullptr); return; }
    const auto generated = run(prompt, max_new, nullptr);
    std::printf("TOKENS"); for (int t : generated) std::printf(" %d", t); std::printf("\n"); std::fflush(stdout);
    if (check) {
        std::vector<int> forced = prompt; forced.insert(forced.end(), generated.begin(), generated.end() - 1);
        std::vector<int> argmax; run(forced, 1, &argmax);
        int bad = 0;
        for (size_t i = 0; i + 1 < generated.size(); ++i) {
            const int expected = generated[i], got = argmax[prompt.size() - 1 + i];
            if (expected != got) { ++bad; std::printf("CHECK mismatch at generated %zu: step %d batch %d\n", i, expected, got); }
        }
        std::printf("CHECK %s: %d of %zu generated tokens differ from the teacher-forced batch argmax\n", bad ? "FAIL" : "PASS", bad, generated.size() - 1);
    }
}

static void run_chain_kld(int argc, char ** argv) {
    require(argc >= 5, "usage: tqfn-stage --chain-kld-gsq BASE.kld fixture-dir shard1.gguf [more shards] [--gsq-rco] [--chunks N] [--first C] [--pos-out FILE] [--logits-out FILE] [--act-f16]");
    const std::string base = argv[1];
    int max_chunks = 0, first_chunk = 0; std::string pos_out, logits_out;
    std::vector<char *> args{argv[0]};
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--chunks" && i + 1 < argc) { max_chunks = std::stoi(argv[++i]); continue; }
        if (std::string(argv[i]) == "--act-f16") { g_chain_f16_act = true; continue; }
        if (std::string(argv[i]) == "--first" && i + 1 < argc) { first_chunk = std::stoi(argv[++i]); continue; }
        if (std::string(argv[i]) == "--pos-out" && i + 1 < argc) { pos_out = argv[++i]; continue; }
        if (std::string(argv[i]) == "--logits-out" && i + 1 < argc) { logits_out = argv[++i]; continue; }
        args.push_back(argv[i]);
    }
    std::ifstream in(base, std::ios::binary);
    char check[9] = {};
    in.read(check, 8);
    require(!in.fail() && std::strncmp("_logits_", check, 8) == 0, base + " is not a llama.cpp log-prob file");
    uint32_t n_ctx = 0; int n_vocab = 0, n_chunk = 0;
    in.read(reinterpret_cast<char *>(&n_ctx), 4); in.read(reinterpret_cast<char *>(&n_vocab), 4); in.read(reinterpret_cast<char *>(&n_chunk), 4);
    require(n_ctx >= 4 && n_ctx <= 64, "chain KLD needs a base made with -c <= 64");
    std::vector<int32_t> tokens(size_t(n_ctx) * n_chunk);
    in.read(reinterpret_cast<char *>(tokens.data()), std::streamsize(tokens.size() * 4));
    require(!in.fail(), "cannot read the base header / tokens");
    if (max_chunks > 0) n_chunk = std::min(n_chunk, first_chunk + max_chunks);
    require(first_chunk >= 0 && first_chunk < n_chunk, "--first beyond the base");
    Fixtures fixtures(args[1]); Checkpoint checkpoint(int(args.size()), args.data()); checkpoint.cpu_q8_projection = true;
    const auto & c = checkpoint.config; const int h = c.hidden, width = h * c.hc;
    const int nv = 2 * ((n_vocab + 1) / 2) + 4, first = int(n_ctx) / 2, nscore = int(n_ctx) - 1 - first;
    std::vector<uint16_t> lp(size_t(nscore) * nv);
    KldSums kld;
    std::vector<int> chunk(n_ctx);
    g_chain_tokens = &chunk; g_chain_quiet = true;
    std::ofstream logits_file;
    if (!logits_out.empty()) { logits_file.open(logits_out, std::ios::binary); require(bool(logits_file), "cannot open " + logits_out); }
    std::ofstream pos_file;
    if (!pos_out.empty()) { pos_file.open(pos_out); require(bool(pos_file), "cannot open " + pos_out); }
    in.seekg(std::streamoff(lp.size()) * 2 * first_chunk, std::ios::cur);
    for (int ci = first_chunk; ci < n_chunk; ++ci) {
        std::copy(tokens.begin() + size_t(ci) * n_ctx, tokens.begin() + size_t(ci + 1) * n_ctx, chunk.begin());
        run_chain(int(args.size()), args.data(), false, false, true);
        require(g_chain_residual.size() == size_t(width) * n_ctx, "chain residual size differs");
        Device<float> last(g_chain_residual.size()), norm(g_chain_residual.size()), gamma(width); last.put(g_chain_residual);
        gamma.put(checkpoint.dense("output_hc_norm.weight"));
        tqllm_cuda::hc_norm(last.p, gamma.p, norm.p, h, c.hc, int(n_ctx), float(checkpoint.epsilon), nullptr);
        const auto head = replay_mix(fixtures, checkpoint, "", 0, norm, true, false);
        std::vector<float> rows(head.mixed.begin() + size_t(first) * h, head.mixed.begin() + size_t(first + nscore) * h);
        const auto logits = checkpoint.head_logits_rows(rows, nscore);
        const size_t V = logits.size() / size_t(nscore);
        require(V >= size_t(n_vocab), "head vocabulary smaller than the base");
        in.read(reinterpret_cast<char *>(lp.data()), std::streamsize(lp.size() * 2));
        require(!in.fail(), "failed reading log-probs of chunk " + std::to_string(ci));
        std::vector<float> row(n_vocab);
        if (logits_file) for (int i = 0; i < nscore; ++i)
            logits_file.write(reinterpret_cast<const char *>(logits.data() + size_t(i) * V), std::streamsize(size_t(n_vocab) * sizeof(float)));
        for (int i = 0; i < nscore; ++i) {
            std::copy(logits.begin() + size_t(i) * V, logits.begin() + size_t(i) * V + n_vocab, row.begin());
            const double before = kld.sum_kld; const size_t same_before = kld.n_same_top;
            kld_token(n_vocab, row.data(), lp.data() + size_t(i) * nv, chunk[first + i + 1], kld);
            if (pos_file) pos_file << ci << '\t' << first + i << '\t' << kld.sum_kld - before << '\t' << kld.n_same_top - same_before << '\n';
        }
        std::printf("chunk %d  mean KLD %.5f  same top %.4f  positions %zu\n", ci, kld.sum_kld / kld.count,
                    double(kld.n_same_top) / kld.count, kld.count);
        std::fflush(stdout);
    }
    g_chain_tokens = nullptr; g_chain_quiet = false;
    std::printf("NATIVE_CHAIN_KLD positions %zu  mean KLD %.5f  same top %.4f  PPL %.4f  PPL(base) %.4f\n", kld.count,
                kld.sum_kld / kld.count, double(kld.n_same_top) / kld.count, std::exp(kld.sum_nll / kld.count),
                std::exp(kld.sum_nll_base / kld.count));
}
static void run(int argc, char ** argv) {
    require(argc >= 3, "usage: tqfn-stage fixture-dir shard1.gguf [shard2.gguf ...]");
    Fixtures fixtures(argv[1]); Checkpoint checkpoint(argc, argv); const auto & c = checkpoint.config;
    require(fixtures.cpu_f32_router(), "F32 router projection gate requires router=cpu-f32 fixtures; the default reference uses custom TF32 MMF");
    const int h = c.hidden, streams = c.hc, width = h * streams;
    const auto tokens = fixtures.tokens(); const int nt = int(tokens.size());
    PleTokenHistory history(c.ple, tokens.size()); const auto row_ids = history.append_batch(tokens);
    std::vector<uint64_t> rows(row_ids.begin(), row_ids.end());
    auto gathered = checkpoint.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight", rows);
    const auto ple_expected = fixtures.values("ple_embd");
    require(gathered.size() == ple_expected.size() && !std::memcmp(gathered.data(), ple_expected.data(), gathered.size() * 4),
            "PLE hash/gather differs from independent reference forward");
    std::cout << "PLE_HASH_GATHER_BIT_EXACT rows=" << rows.size() << '\n';
    std::vector<float> residual = fixtures.values("hc_init");
    Device<float> embedding(h * nt), initialized(width * nt); embedding.put(fixtures.values("model.input_embed"));
    tqllm_cuda::hc_init(embedding.p, initialized.p, h, streams, nt, nullptr);
    compare("hc_init", initialized.get(), residual, 0.);
    Tq3ExpertCache cache(c.layers, c.experts, h, c.expert_ff, 64, 256 * 1024 * 1024,
                         [&](int layer, int expert) { return checkpoint.expert(layer, expert); }, nullptr);
    for (int layer = 0; layer < c.layers; ++layer) {
        const std::string suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
        const bool ple_layer = std::find(c.ple_layers.begin(), c.ple_layers.end(), layer) != c.ple_layers.end();
        if (ple_layer) {
            require(gathered.size() == size_t(h) * nt, "PLE concatenated embedding width differs");
            Device<float> dp(gathered.size()), key(size_t(width) * nt), value(size_t(h) * nt), query(residual.size());
            Device<float> key_norm(key.count), query_norm(query.count), gates(size_t(streams) * nt), native_gated(residual.size());
            Device<float> norm_weight{size_t(width)}; dp.put(gathered); query.put(residual);
            checkpoint.project(prefix + "ple_key.weight", dp.p, key.p, h, width, nt);
            checkpoint.project(prefix + "ple_value.weight", dp.p, value.p, h, h, nt);
            norm_weight.put(checkpoint.dense(prefix + "ple_norm_key.weight"));
            tqllm_cuda::hc_norm(key.p, norm_weight.p, key_norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            norm_weight.put(checkpoint.dense(prefix + "ple_norm_query.weight"));
            tqllm_cuda::hc_norm(query.p, norm_weight.p, query_norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_gate(key_norm.p, query_norm.p, gates.p, h, streams, nt, nullptr);
            tqllm_cuda::ple_broadcast(value.p, gates.p, native_gated.p, h, streams, nt, nullptr);
            compare("ple_projected_gate" + suffix, gates.get(), fixtures.values("ple_gate" + suffix), .02);
            auto gated = native_gated.get(), conv_expected = fixtures.values("ple_conv_out" + suffix);
            compare("ple_projected_gated_value" + suffix, gated, fixtures.values("ple_gated_value" + suffix), .02);
            auto gamma = checkpoint.dense(prefix + "ple_norm_conv.weight"), kernel = checkpoint.dense(prefix + "ple_conv1d.weight");
            Device<float> dg(gated.size()), dw(width), dk(kernel.size()), norm(gated.size()), conv(gated.size());
            const int hist = (c.ple_kernel - 1) * c.ple.ngram;
            Device<float> dh(width * hist), next(width * hist), dr(residual.size()), after(residual.size());
            dg.put(gated); dw.put(gamma); dk.put(kernel); dh.put(std::vector<float>(width * hist, 0.f)); dr.put(residual);
            tqllm_cuda::hc_norm(dg.p, dw.p, norm.p, h, streams, nt, float(checkpoint.epsilon), nullptr);
            tqllm_cuda::ple_conv(norm.p, dh.p, dk.p, conv.p, next.p, width, nt, c.ple_kernel, c.ple.ngram, nullptr);
            compare("ple_conv" + suffix, conv.get(), conv_expected, .02);
            tqllm_cuda::ple_apply(dr.p, dg.p, conv.p, after.p, int(residual.size()), nullptr); residual = after.get();
        }
        const auto attention_norm = fixtures.values("hc_norm" + suffix, 0);
        const int attn_tokens = int(attention_norm.size() / width);
        Device<float> dr(residual.size()), dg(width), dn(residual.size()); dr.put(residual);
        dg.put(checkpoint.dense(prefix + "hc_attn_norm.weight"));
        tqllm_cuda::hc_norm(dr.p, dg.p, dn.p, h, streams, attn_tokens, float(checkpoint.epsilon), nullptr);
        compare("hc_attn_norm" + suffix, dn.get(), attention_norm, ple_layer ? .02 : 2e-5);
        const auto attn_mix = replay_mix(fixtures, checkpoint, suffix, 0, dn);
        // Last-layer output selection happens between attention and the FFN mixer.
        auto attention_out = fixtures.values((fixtures.has("linear_attn_out" + suffix) ? "linear_attn_out" : "attn_output") + suffix);
        auto attention_inject = attn_mix.inject;
        const auto combined_expected = fixtures.values("hc_combine" + suffix);
        const int selected = int(combined_expected.size() / width);
        require(residual.size() >= combined_expected.size() && attention_out.size() >= size_t(selected) * h &&
                attention_inject.size() >= size_t(selected) * streams, "attention output selection dimensions differ");
        const auto last = [](std::vector<float> & values, size_t count) { values.erase(values.begin(), values.end() - count); };
        last(residual, size_t(selected) * width); last(attention_out, size_t(selected) * h); last(attention_inject, size_t(selected) * streams);
        Device<float> dares(residual.size()), dao(attention_out.size()), dai(attention_inject.size()), dac(residual.size());
        dares.put(residual); dao.put(attention_out); dai.put(attention_inject);
        tqllm_cuda::hc_combine(dares.p, dao.p, dai.p, dac.p, h, streams, selected, nullptr);
        compare("hc_attn_combine" + suffix, dac.get(), combined_expected, .02);
        residual = combined_expected;
        const int n = int(residual.size() / width);
        Device<float> dres(residual.size()), dn2(residual.size()); dres.put(residual);
        dg.put(checkpoint.dense(prefix + "hc_ffn_norm.weight"));
        tqllm_cuda::hc_norm(dres.p, dg.p, dn2.p, h, streams, n, float(checkpoint.epsilon), nullptr);
        compare("hc_ffn_norm" + suffix, dn2.get(), fixtures.values("hc_norm" + suffix, 1), 2e-5);
        const auto ffn_mix = replay_mix(fixtures, checkpoint, suffix, 1, dn2);
        const auto mixed = fixtures.values("hc_mixed" + suffix, 1);
        const auto reference_ids = fixtures.ids("ffn_moe_topk" + suffix);
        const auto reference_logits = fixtures.values("ffn_moe_logits" + suffix);
        require(reference_ids.size() == size_t(n * c.topk) && reference_logits.size() == size_t(n * c.experts), "routing fixture shape differs");
        std::vector<float> logits(size_t(n) * (c.experts + 1));
        for (int token = 0; token < n; ++token)
            std::copy_n(reference_logits.data() + token * c.experts, c.experts, logits.data() + token * (c.experts + 1));
        Device<float> dl(logits.size()), dwroute(reference_ids.size()), ds(n); Device<int> di(reference_ids.size()); dl.put(logits);
        tqllm_cuda::moe_topk(dl.p, di.p, dwroute.p, ds.p, c.experts, c.topk, n, 1.f, nullptr);
        require(di.get() == reference_ids, "native top-k differs from reference selected experts at " + std::to_string(layer));
        compare("router_weights" + suffix, dwroute.get(), fixtures.values("ffn_moe_weights_norm" + suffix), 3e-6);
        const auto router = checkpoint.dense(prefix + "ffn_gate_inp.weight");
        const auto shared_gate = checkpoint.dense(prefix + "ffn_gate_inp_shexp.weight");
        require(router.size() == size_t(h) * c.experts && shared_gate.size() == size_t(h), "router weight shape differs");
        Device<float> drouter(router.size()), dshared(shared_gate.size()), plain(mixed.size());
        drouter.put(router); dshared.put(shared_gate); plain.put(mixed);
        tqllm_cuda::moe_router_f32(drouter.p, dshared.p, plain.p, dl.p, h, c.experts, n, nullptr);
        const auto projected = dl.get(); std::vector<float> dense_logits(reference_logits.size()), dense_shared(n);
        for (int token = 0; token < n; ++token) {
            std::copy_n(projected.data() + token * (c.experts + 1), c.experts, dense_logits.data() + token * c.experts);
            dense_shared[token] = projected[token * (c.experts + 1) + c.experts];
        }
        compare("router_logits" + suffix, dense_logits, reference_logits, 2e-5);
        compare("shared_gate_logits" + suffix, dense_shared, fixtures.values("shared_expert_gate" + suffix), 2e-5);
        tqllm_cuda::moe_topk(dl.p, di.p, dwroute.p, ds.p, c.experts, c.topk, n, 1.f, nullptr);
        require(di.get() == reference_ids, "projected native router changed selected experts at " + std::to_string(layer));
        const auto physical = cache.prepare(layer, reference_ids); di.put(physical);
        const auto cold = cache.stats(); require(cache.prepare(layer, reference_ids) == physical, "warm expert placement differs");
        require(cache.stats().misses == cold.misses && cache.stats().transfer_bytes == cold.transfer_bytes, "warm expert replay transferred weights");
        Device<float> dx(mixed.size()); dx.put(mixed); Device<__nv_bfloat16> db(mixed.size()); Device<__half> rotated(mixed.size());
        tqllm_cuda::f32_to_bf16(dx.p, db.p, int64_t(mixed.size()), nullptr);
        tqllm_cuda::tq3_rotate_h(db.p, rotated.p, h, n, nullptr);
        for (int projection = 0; projection < 2; ++projection) {
            const auto matrix = projection == 0 ? cache.gate() : cache.up(); Device<__nv_bfloat16> out(size_t(n) * c.topk * c.expert_ff);
            matrix.fmt->expert_matvec(matrix, rotated.p, out.p, c.expert_ff, di.p, n, c.topk, false, nullptr);
            compare((projection == 0 ? "expert_gate" : "expert_up") + suffix, bf16_values(out),
                    fixtures.values((projection == 0 ? "ffn_moe_gate" : "ffn_moe_up") + suffix), .02);
        }
        // Isolate the down projection using the reference activation, rather than hiding earlier errors.
        auto activation = fixtures.values("ffn_moe_swiglu" + suffix);
        Device<float> da(activation.size()); da.put(activation); Device<__nv_bfloat16> dab(activation.size()); Device<__half> dar(activation.size());
        tqllm_cuda::f32_to_bf16(da.p, dab.p, int64_t(activation.size()), nullptr);
        tqllm_cuda::tq3_rotate_h(dab.p, dar.p, c.expert_ff, n * c.topk, nullptr);
        const auto matrix = cache.down(); Device<__nv_bfloat16> down(size_t(n) * c.topk * h);
        matrix.fmt->expert_matvec(matrix, dar.p, down.p, h, di.p, n, c.topk, true, nullptr);
        compare("expert_down" + suffix, bf16_values(down), fixtures.values("ffn_moe_down" + suffix), .02);
        auto block = fixtures.values("ffn_out" + suffix), inject = ffn_mix.inject;
        Device<float> dout(block.size()), dinject(inject.size()), combined(residual.size()); dout.put(block); dinject.put(inject);
        tqllm_cuda::hc_combine(dres.p, dout.p, dinject.p, combined.p, h, streams, n, nullptr);
        residual = fixtures.values("l_last" + suffix); compare("hc_ffn_combine" + suffix, combined.get(), residual, .02);
    }
    Device<float> final_residual(residual.size()), head_norm(residual.size()), head_gamma{size_t(width)};
    final_residual.put(residual); head_gamma.put(checkpoint.dense("output_hc_norm.weight"));
    tqllm_cuda::hc_norm(final_residual.p, head_gamma.p, head_norm.p, h, streams, int(residual.size() / width),
                       float(checkpoint.epsilon), nullptr);
    compare("hc_head_norm", head_norm.get(), fixtures.values("hc_norm"), 2e-5);
    replay_mix(fixtures, checkpoint, "", 0, head_norm, true);
    const auto stats = cache.stats();
    std::cout << "NATIVE_STAGE_REPLAY_DONE layers=" << c.layers << " cache_device_bytes=" << stats.device_bytes
              << " misses=" << stats.misses << " hits=" << stats.hits << " evictions=" << stats.evictions
              << " transfer_bytes=" << stats.transfer_bytes << " full_forward=false\n";
}
#ifndef TQFN_SERVE_ENGINE
int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--chain-f32-experts") run_chain(argc - 1, argv + 1, false, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-cpu-q8-f32-experts") run_chain(argc - 1, argv + 1, true, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain") run_chain(argc - 1, argv + 1, false);
        else if (argc > 1 && std::string(argv[1]) == "--chain-cpu-q8") run_chain(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-gsq-cpu-q8k") run_chain(argc - 1, argv + 1, false, false, true);
        else if (argc > 1 && std::string(argv[1]) == "--chain-kld-gsq") run_chain_kld(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--decode") run_decode(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--pack-experts") run_pack_experts(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--recurrent-cpu-q8k") run_recurrent(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--recurrent") run_recurrent(argc - 1, argv + 1);
        else if (argc > 1 && std::string(argv[1]) == "--attention-cpu-q8") run_attention(argc - 1, argv + 1, true);
        else if (argc > 1 && std::string(argv[1]) == "--attention") run_attention(argc - 1, argv + 1);
        else run(argc, argv);
        return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
#endif

#ifdef TQFN_SERVE_ENGINE
// Flash-Next behind the tqllm Engine interface (tqserve). The model is a manifest file NAME.tqfn of key=value lines:
//   gguf=<shard> (one line per shard, in order), mtp=<MTP gguf>, experts=<packed experts>, draft_vocab=<file>,
//   slots=176, budget_gb=18, drafts=3, chunk=8, gsq_rco=1
// One slot, greedy MTP speculative decoding (exact: tokens identical to plain greedy). A fresh DecodeModel per request
// resets the recurrent, PLE, attention and MTP state; the expert caches persist across requests.
#include "../src/engine.hpp"
namespace {
struct FnServe {
    std::vector<std::string> ggufs; std::string mtp, experts, draft_vocab;
    int slots = 176, drafts = 3, chunk = 8; double budget_gb = 18; bool gsq_rco = true;
    std::vector<std::string> arg_store; std::vector<char *> argv;
    std::unique_ptr<Fixtures> fixtures; std::unique_ptr<Checkpoint> cp;
    std::unique_ptr<MixedExpertCache> cache, mtp_cache; PackedExperts packed; char * mtp_packed = nullptr; size_t mtp_block = 0;
    ScratchArena arena;
};
class FlashNextEngine final : public tqllm::Engine {
    FnServe f_; uint32_t context_;
    // one sequence per slot (KV, recurrent and MTP state, sampling); weights, expert caches and scratch are shared
    struct SlotState {
        std::unique_ptr<DecodeModel> model_; std::unique_ptr<Device<float>> prompt_res_;
        std::vector<int> out_; int generate_ = 0, d1_ = 0; bool done_ = true; uint64_t steps_ = 0, toks_ = 0; uint64_t prefill_us_ = 0, decode_us_ = 0;
        std::chrono::steady_clock::time_point t_begin_, t_decode_;
        float temp_ = 0.f, top_p_ = 1.f; int top_k_ = 0; std::mt19937_64 rng_{1};
    };
    std::vector<SlotState> slots_; int cur_ = 0;
    SlotState & S() { return slots_[size_t(cur_)]; }
    uint32_t slot_context() const { return context_; }  // Model passes the per-slot context (tqserve: --context / --np)
    // the shared caches copy misses on the active slot's stream
    void bind() { if (S().model_ && S().model_->miss_stream) { f_.cache->set_miss_stream(S().model_->miss_stream); f_.mtp_cache->set_miss_stream(S().model_->miss_stream); } }
    static bool is_eos(int t) { return t == 248046 || t == 248044; }
    // T > 0: sample each verify position from the target's own distribution; a draft is accepted only if it equals that
    // sample, so the output follows the target exactly (greedy drafts just lower the acceptance)
    std::vector<int> pick(const float * rows, int n) {
        auto & checkpoint = *f_.cp;
        if (S().temp_ <= 0.f) return checkpoint.head_argmax_rows(rows, n);
        const auto lg = checkpoint.head_logits_device(rows, n); const size_t V = lg.size() / size_t(n);
        std::vector<int> r(n); std::vector<std::pair<float, int>> cand;
        for (int i = 0; i < n; ++i) {
            const float * l = lg.data() + size_t(i) * V; float mx = l[0]; for (size_t v = 1; v < V; ++v) mx = std::max(mx, l[v]);
            cand.clear();
            for (size_t v = 0; v < V; ++v) { const float p = std::exp((l[v] - mx) / S().temp_); if (p > 1e-9f) cand.push_back({p, int(v)}); }
            std::sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first > b.first; });
            if (S().top_k_ > 0 && int(cand.size()) > S().top_k_) cand.resize(size_t(S().top_k_));
            double total = 0; for (auto & c : cand) total += c.first;
            if (S().top_p_ < 1.f) { double acc = 0; size_t keep = 0; for (; keep < cand.size(); ++keep) { acc += cand[keep].first / total; if (acc >= S().top_p_) { ++keep; break; } }
                                cand.resize(std::max<size_t>(1, std::min(keep, cand.size()))); total = 0; for (auto & c : cand) total += c.first; }
            double u = std::uniform_real_distribution<double>(0, total)(S().rng_), acc = 0; r[i] = cand.back().second;
            for (auto & c : cand) { acc += c.first; if (acc >= u) { r[i] = c.second; break; } }
        }
        return r;
    }
public:
    FlashNextEngine(const std::string & manifest, uint32_t context, int nslots) : context_(context), slots_(size_t(std::max(1, nslots))) {
        // above 8 concurrent slots some requests end after one token and np 16 fails (10-13, P8 build too): refuse until fixed
        require(nslots <= 8 || std::getenv("TQFN_ALLOW_NP_ABOVE_8") != nullptr,
                "Flash-Next serves up to --np 8 for now (a bug above 8 slots is being fixed); TQFN_ALLOW_NP_ABOVE_8=1 for diagnostics");
        auto & f = f_; std::ifstream in(manifest); require(bool(in), "cannot open Flash-Next manifest " + manifest);
        const auto dir = std::filesystem::path(manifest).parent_path();
        auto resolve = [&](const std::string & v) { return std::filesystem::path(v).is_absolute() ? v : (dir / v).string(); };
        for (std::string line; std::getline(in, line);) {
            if (line.empty() || line[0] == '#') continue;
            const auto eq = line.find('='); require(eq != std::string::npos, "manifest line without '=': " + line);
            const auto k = line.substr(0, eq), v = line.substr(eq + 1);
            if (k == "gguf") f.ggufs.push_back(resolve(v)); else if (k == "mtp") f.mtp = resolve(v);
            else if (k == "experts") f.experts = resolve(v); else if (k == "draft_vocab") f.draft_vocab = resolve(v);
            else if (k == "slots") f.slots = std::stoi(v); else if (k == "budget_gb") f.budget_gb = std::stod(v);
            else if (k == "drafts") f.drafts = std::stoi(v); else if (k == "chunk") f.chunk = std::stoi(v);
            else if (k == "gsq_rco") f.gsq_rco = v != "0";
            else throw std::runtime_error("unknown Flash-Next manifest key " + k);
        }
        require(!f.ggufs.empty() && !f.mtp.empty(), "manifest needs gguf and mtp lines");
        if (!f.draft_vocab.empty()) setenv("TQFN_DRAFT_VOCAB", f.draft_vocab.c_str(), 1);
        g_kv_capacity = int(context);  // per slot
        { cudaMemPool_t pool; ck(cudaDeviceGetDefaultMemPool(&pool, 0)); uint64_t keep = UINT64_MAX;
          ck(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep)); }
        // per-step temporaries grow with the prefill chunk (one layer at a time in prefill: measured 117 MB at chunk 16-64, 245 MB at 256; decode verify steps keep one span); TQFN_ARENA_MB overrides
        f.arena.size = size_t(std::getenv("TQFN_ARENA_MB") ? std::atoi(std::getenv("TQFN_ARENA_MB")) : std::max({256, 2 * f.chunk + 128,
                       slots_.size() > 1 ? 26 * std::min(64, int(slots_.size()) * (f.drafts + 1)) + 128 : 0})) << 20;  // + one span per batched verify (~26 MB a row)
        ck(cudaMalloc(&f.arena.base, f.arena.size)); g_arena = &f.arena;
        f.arg_store = {"tqserve", "-"}; for (const auto & g : f.ggufs) f.arg_store.push_back(g);
        f.arg_store.push_back("--mtp"); f.arg_store.push_back(f.mtp); if (f.gsq_rco) f.arg_store.push_back("--gsq-rco");
        for (auto & a : f.arg_store) f.argv.push_back(a.data());
        f.fixtures = std::make_unique<Fixtures>();
        f.cp = std::make_unique<Checkpoint>(int(f.argv.size()), f.argv.data()); f.cp->resident = true;
        const auto & c = f.cp->config; auto & checkpoint = *f.cp;
        f.cache = std::make_unique<MixedExpertCache>(c.layers, c.experts, c.hidden, c.expert_ff, f.slots, size_t(f.budget_gb * 1024 * 1024 * 1024),
                    [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
        apply_layer_slots(*f.cache);
        // packed experts in pinned memory, tiled in place (same steps as --decode). Without an experts file (a GB10 with
        // every expert resident) the cache loads experts from the GGUF once and keeps them.
        if (!f.experts.empty()) {
        f.packed = open_packed(f.experts);
        require(f.packed.layers == c.layers && f.packed.experts == c.experts, "packed experts do not match the model");
        char * host = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&host), f.packed.bytes));
        { const int fd = ::open(f.experts.c_str(), O_RDONLY); require(fd >= 0, "cannot reopen " + f.experts);
          for (size_t done = 0; done < f.packed.bytes;) {
              const ssize_t r = ::pread(fd, host + done, std::min<size_t>(size_t(1) << 28, f.packed.bytes - done), off_t(done));
              require(r > 0, "short read of " + f.experts); done += size_t(r);
          }
          ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
        ::munmap(const_cast<char *>(f.packed.map), f.packed.bytes); f.packed.map = host;
        if (!f.packed.tiled) {
            for (int l = 0; l < f.packed.layers; ++l) {
                size_t off = 0; const auto ids = expert_u4h_ids(checkpoint, l);
                for (int pp = 0; pp < 3; ++pp) {
                    const int cols = pp == 2 ? c.expert_ff : c.hidden, rows = pp == 2 ? c.hidden : c.expert_ff;
                    const int bpb = tqllm_cuda::tiled_block_bytes(ids[pp]); const size_t dsz = f.packed.sizes[l][2 * pp];
                    if (bpb) {
                        require(rows % 16 == 0 && (cols / 32) % 4 == 0 && dsz == size_t(rows) * (cols / 32) * bpb, "tileable expert part with an unexpected layout");
                        std::vector<std::thread> th;
                        for (int w = 0; w < 16; ++w) th.emplace_back([&, w] {
                            std::vector<char> tmp(dsz);
                            for (int e = w; e < f.packed.experts; e += 16) {
                                char * d = host + f.packed.base[l] + uint64_t(e) * f.packed.block[l] + off;
                                tqllm_cuda::tile_expert_rows(d, tmp.data(), rows, cols, bpb); std::memcpy(d, tmp.data(), dsz);
                            }
                        });
                        for (auto & t : th) t.join();
                    }
                    off += dsz + f.packed.sizes[l][2 * pp + 1];
                }
            }
            f.packed.tiled = true;
        }
        if (f.packed.tiled) { tqllm_cuda::set_tiled_experts(true); g_tile_experts = true; }
        PackedExperts * pk = &f.packed;
        f.cache->set_prepacked([pk](int layer, int id) -> const char * { return layer < pk->layers ? pk->at(layer, id) : nullptr; }, true);
        }
        // MTP experts prepacked in pinned memory
        // all experts resident (GB10): the MTP block's experts too, so its prefill catch-up can run whole chunks
        const int mtp_slots = std::getenv("TQFN_MTP_SLOTS") ? std::atoi(std::getenv("TQFN_MTP_SLOTS")) : (f.slots >= c.experts ? c.experts : 256);
        f.mtp_cache = std::make_unique<MixedExpertCache>(c.layers + 1, c.experts, c.hidden, c.expert_ff, mtp_slots, size_t(mtp_slots) * (6u << 20) + (64u << 20),
                        [&checkpoint](int layer, int expert) { return checkpoint.expert_payload(layer, expert); }, nullptr);
        const auto first = repack_expert(checkpoint, c.layers, 0); for (const auto & p : first) f.mtp_block += p.size();
        ck(cudaMallocHost(reinterpret_cast<void **>(&f.mtp_packed), f.mtp_block * size_t(c.experts)));
        { std::vector<std::thread> pool; std::atomic<int> next{0};
          for (int t = 0; t < 16; ++t) pool.emplace_back([&] {
              for (int e; (e = next++) < c.experts;) {
                  const auto parts = repack_expert(checkpoint, c.layers, e); size_t off = 0;
                  for (const auto & p : parts) { std::memcpy(f.mtp_packed + size_t(e) * f.mtp_block + off, p.data(), p.size()); off += p.size(); }
              }
          });
          for (auto & t : pool) t.join(); }
        char * mp = f.mtp_packed; const size_t mb = f.mtp_block; const int L = c.layers;
        f.mtp_cache->set_prepacked([mp, mb, L](int layer, int id) -> const char * { return layer == L ? mp + size_t(id) * mb : nullptr; }, true);
        std::fprintf(stderr, "tqllm: Flash-Next ready: %d layers, %d experts, %d slots/layer, context %u, %d drafts\n",
                     c.layers, c.experts, f.slots, context, f.drafts);
    }
    void spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int) override {
        auto & checkpoint = *f_.cp; const auto & c = checkpoint.config;
        require(!prompt.empty() && prompt.size() + count + 16 <= slot_context(), "prompt plus max_tokens exceeds the slot context");
        S().model_.reset(); S().prompt_res_.reset();
        if (g_arena) g_arena->used = 0;
        S().model_ = std::make_unique<DecodeModel>(checkpoint, *f_.fixtures, *f_.cache, prompt.size() + count + 16);
        S().model_->mtp_experts = f_.mtp_cache.get();
        bind();
        const std::vector<int> ids(prompt.begin(), prompt.end());
        // TQFN_PROFILE_REQ=N: cudaProfilerStart at the N-th request's prefill, stop after it (nsys --capture-range=cudaProfilerApi)
        static const int prof_req = std::getenv("TQFN_PROFILE_REQ") ? std::atoi(std::getenv("TQFN_PROFILE_REQ")) : 0; static int req_no = 0;
        const bool profiling = prof_req > 0 && ++req_no == prof_req;
        if (profiling) ck(cudaProfilerStart());
        S().t_begin_ = std::chrono::steady_clock::now(); S().prefill_us_ = S().decode_us_ = 0;
        S().out_.clear(); S().generate_ = int(count); S().steps_ = S().toks_ = 0;
        // prefill chunk by chunk; the MTP catch-up of each chunk runs right after it from res_keep (no prompt-sized buffer)
        const int P = int(ids.size());
        using pclk = std::chrono::steady_clock; double trunk_ms = 0, mtp_ms = 0; g_qsa_ms = 0; const auto p0 = pclk::now();
        auto lap = [&](double & acc, pclk::time_point & t) { if (!g_prefill_log) return; ck(cudaDeviceSynchronize()); const auto u = pclk::now();
                                                             acc += std::chrono::duration<double, std::milli>(u - t).count(); t = u; };
        for (int b = 0; b < P; b += f_.chunk) {
            const int e = std::min(P, b + f_.chunk);
            auto tl = pclk::now();
            auto rows = S().model_->step(std::vector<int>(ids.begin() + b, ids.begin() + e));
            lap(trunk_ms, tl);
            if (e == P) S().out_.push_back(pick(rows->p + size_t(e - b - 1) * checkpoint.config.hidden, 1)[0]);
            // TQFN_MTP_CATCHUP: tokens per MTP catch-up batch (each needs topk unique MTP slots)
            // a cache holding every expert takes any width (the whole chunk: grouped GEMM path); else unique experts <= slots
            static const int cw = f_.mtp_cache->slots() >= checkpoint.config.experts
                ? (std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : f_.chunk)
                : std::max(1, std::min(std::getenv("TQFN_MTP_CATCHUP") ? std::atoi(std::getenv("TQFN_MTP_CATCHUP")) : 16,
                                       f_.mtp_cache->slots() / checkpoint.config.topk));
            for (int q = b; q < e; q += cw) {
                std::vector<int> toks;
                for (int i = q; i < std::min(e, q + cw); ++i) toks.push_back(i + 1 < P ? ids[i + 1] : S().out_[0]);
                if (g_arena) g_arena->used = 0;  // the catch-up only reads res_keep (persistent)
                S().d1_ = S().model_->mtp_batch(toks, S().model_->res_keep->p + size_t(q - b) * S().model_->width);
            }
            lap(mtp_ms, tl);
            tqllm::prefill_completed(nullptr,uint32_t(e),uint32_t(P));
        }
        if (g_prefill_log) std::fprintf(stderr, "tqllm: prefill %d tokens %.1f s: trunk %.1f s (QSA %.1f s), MTP catch-up %.1f s\n", P,
            std::chrono::duration<double>(pclk::now() - p0).count(), trunk_ms / 1000, g_qsa_ms / 1000, mtp_ms / 1000);
        S().model_->set_spec(true);
        ck(cudaDeviceSynchronize()); S().t_decode_ = std::chrono::steady_clock::now();
        S().prefill_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(S().t_decode_ - S().t_begin_).count());
        if (profiling) ck(cudaProfilerStop());
        if (g_arena && std::getenv("TQFN_ARENA_LOG")) std::fprintf(stderr, "tqllm: prefill %d tokens, arena high %.0f MB of %.0f\n", P, g_arena->high / 1048576.0, g_arena->size / 1048576.0);
        S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
        (void) c;
    }
    bool spec_advance() override {
        if (S().done_) return false;
        bind();
        auto & checkpoint = *f_.cp; const int drafts = f_.drafts;
        const int y = S().out_.back();
        const std::vector<int> dr = S().model_->mtp_drafts(S().d1_, drafts);
        std::vector<int> block{y}; block.insert(block.end(), dr.begin(), dr.end());
        auto rows = S().model_->step(block);
        const auto v = pick(rows->p, int(block.size()));
        int m = 1; while (m <= drafts && v[m - 1] == dr[m - 1]) ++m;
        for (int i = 0; i < m && int(S().out_.size()) < S().generate_; ++i) { S().out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
        S().model_->accept(m); ++S().steps_; S().toks_ += m;
        S().d1_ = S().model_->mtp_batch(std::vector<int>(v.begin(), v.begin() + m), S().model_->res_keep->p);
        S().done_ = int(S().out_.size()) >= S().generate_ || is_eos(S().out_.back());
        if (S().done_) { S().model_->set_spec(false); ck(cudaDeviceSynchronize());
                     S().decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - S().t_decode_).count()); }
        return !S().done_;
    }
    std::vector<uint32_t> spec_finish(uint64_t * stats) override {
        bind();
        while (spec_advance()) {}
        // steps, drafted, accepted drafts, decode us, prefill us (tqserve timings)
        if (stats) { stats[0] = S().steps_; stats[1] = S().steps_ * uint64_t(f_.drafts); stats[2] = S().toks_ - S().steps_; stats[3] = S().decode_us_; stats[4] = S().prefill_us_; }
        std::vector<uint32_t> r(S().out_.begin(), S().out_.end());
        S().model_.reset(); S().prompt_res_.reset();
        return r;
    }
    std::vector<uint32_t> spec_tokens(int s) const override {
        const auto & o = slots_[size_t(s)].out_; return std::vector<uint32_t>(o.begin(), o.end());
    }
    int slots() const override { return int(slots_.size()); }
    void select_slot(int s) override { require(s >= 0 && s < int(slots_.size()), "Flash-Next slot out of range"); cur_ = s; }
    // one step for every listed slot: each drafts with its own MTP state, one verify pass covers all of them
    // (TQFN_SLOT_BATCH=0, a single active slot, more than 64 rows or more experts than cache slots: one slot at a time)
    void spec_step(const int * ids, int n, bool * done) override {
        const int drafts = f_.drafts, topk = f_.cp->config.topk;
        std::vector<int> act;
        for (int i = 0; i < n; ++i) { done[i] = slots_[size_t(ids[i])].done_; if (!done[i]) act.push_back(i); }
        const int rows = int(act.size()) * (drafts + 1);
        if (act.size() < 2 || rows > 64 || (f_.cache->slots() < f_.cp->config.experts && rows * topk > f_.cache->slots())) {
            for (int i : act) { select_slot(ids[i]); done[i] = !spec_advance(); }
            return;
        }
        // every slot drafts and catches up in shared MTP passes
        std::vector<DecodeModel *> ms; std::vector<std::vector<int>> blocks, drs; std::vector<int> d1s;
        for (int i : act) { select_slot(ids[i]); ms.push_back(S().model_.get()); d1s.push_back(S().d1_); }
        select_slot(ids[act[0]]); bind();
        drs = S().model_->mtp_drafts_multi(ms, d1s, drafts);
        for (size_t a = 0; a < act.size(); ++a) {
            std::vector<int> block{slots_[size_t(ids[act[a]])].out_.back()}; block.insert(block.end(), drs[a].begin(), drs[a].end());
            blocks.push_back(block);
        }
        select_slot(ids[act[0]]); bind();
        const auto rows_d = S().model_->step_multi(ms, blocks);
        std::vector<std::vector<int>> vs(act.size()); std::vector<int> ms_acc(act.size());
        size_t off = 0;
        for (size_t a = 0; a < act.size(); ++a) {  // pick and roll back each slot before any MTP catch-up reuses the arena
            select_slot(ids[act[a]]); auto & sl = S();
            const auto v = pick(rows_d->p + off * size_t(f_.cp->config.hidden), int(blocks[a].size())); off += blocks[a].size();
            int m = 1; while (m <= drafts && v[m - 1] == drs[a][m - 1]) ++m;
            for (int i = 0; i < m && int(sl.out_.size()) < sl.generate_; ++i) { sl.out_.push_back(v[i]); if (is_eos(v[i])) { m = i + 1; break; } }
            sl.model_->accept(m); ++sl.steps_; sl.toks_ += m; vs[a] = v; ms_acc[a] = m;
        }
        std::vector<int> d1n(act.size());
        {
            std::vector<std::vector<int>> t(act.size()); std::vector<const float *> hp(act.size());
            for (size_t a = 0; a < act.size(); ++a) { t[a].assign(vs[a].begin(), vs[a].begin() + ms_acc[a]); hp[a] = ms[a]->res_keep->p; }
            select_slot(ids[act[0]]); bind();
            d1n = S().model_->mtp_batch_multi(ms, t, hp);
        }
        for (size_t a = 0; a < act.size(); ++a) {
            select_slot(ids[act[a]]); bind(); auto & sl = S();
            sl.d1_ = d1n[a];
            sl.done_ = int(sl.out_.size()) >= sl.generate_ || is_eos(sl.out_.back());
            if (sl.done_) { sl.model_->set_spec(false); ck(cudaDeviceSynchronize());
                            sl.decode_us_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sl.t_decode_).count()); }
            done[act[a]] = sl.done_;
        }
    }
    bool incremental() const override { return true; }
    bool has_drafter() const override { return true; }
    void set_sampling(float temperature, float top_p, int top_k, uint64_t seed) override {
        S().temp_ = temperature; S().top_p_ = top_p > 0.f && top_p <= 1.f ? top_p : 1.f; S().top_k_ = top_k; S().rng_.seed(seed ? seed : 0x9e3779b97f4a7c15ull);
    }
    uint32_t step(uint32_t, uint32_t, bool, float *) override { throw std::runtime_error("Flash-Next serves through spec_begin / spec_advance"); }
    void verify(const uint32_t *, int, uint32_t, uint32_t *) override { throw std::runtime_error("Flash-Next: verify is internal"); }
    void commit(int) override { throw std::runtime_error("Flash-Next: commit is internal"); }
    const float * logits() const override { return nullptr; }
    void synchronize() override { ck(cudaDeviceSynchronize()); }
    uint32_t layers() const override { return uint32_t(f_.cp->config.layers); }
    uint32_t hidden() const override { return uint32_t(f_.cp->config.hidden); }
    uint32_t vocab() const override { return uint32_t(f_.cp->head_resident().m.rows); }
    uint32_t eos() const override { return 248046; }
    uint32_t context() const override { return context_; }
};
std::unique_ptr<tqllm::Engine> make_flashnext(const std::string & manifest, const std::string &, uint32_t context, int slots,
                                              const tqllm::EngineOptions &) {
    return std::make_unique<FlashNextEngine>(manifest, context, slots);
}
const bool registered_flashnext = (tqllm::register_flashnext_engine(&make_flashnext), true);
}  // namespace
#endif
