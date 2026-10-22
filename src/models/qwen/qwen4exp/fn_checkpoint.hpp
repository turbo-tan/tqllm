#pragma once
// Flash-Next weights: the GGUF checkpoint, resident tensors and the dense projection dispatch; stage fixtures.
#include "decode_rules.hpp"
#include "fn_runtime.hpp"
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
    // after the experts are in pinned memory: drop the trunk expert pages, keep every tensor read per token cached
    void release_expert_pages() const { for (const auto & [name, i] : owners_) if (tqllm::flashnext::drop_after_pack(name)) files_[i]->release_tensor(name); }
    size_t warm_tensors(int threads) const {
        size_t n = 0; for (const auto & [name, i] : owners_) if (tqllm::flashnext::warm_at_start(name)) n += files_[i]->warm_tensor(name, threads);
        return n;
    }
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
    // Parking: a resident matrix's device buffers copied to pinned host memory and freed, put back (byte for byte, kernels
    // tuned by shape) before its next use. Small cards park the LM heads through a layer-outer prefill.
    struct Parked { std::vector<char *> host; std::vector<size_t> bytes; std::vector<const char *> old; };
    mutable std::map<std::string, Parked> parked_;
public:
    size_t park(const std::string & key) const {
        const auto it = resident_.find(key);
        if (it == resident_.end() || !it->second || !it->second->m.fmt || it->second->dense || it->second->mem.empty() || parked_.count(key)) return 0;
        auto & r = *it->second; Parked pk; size_t total = 0;
        ck(cudaDeviceSynchronize());
        for (auto & buf : r.mem) {
            const size_t n = buf->count; char * h = nullptr; ck(cudaMallocHost(reinterpret_cast<void **>(&h), n));
            ck(cudaMemcpy(h, buf->p, n, cudaMemcpyDeviceToHost));
            pk.host.push_back(h); pk.bytes.push_back(n); pk.old.push_back(buf->p); total += n;
        }
        r.mem.clear();
        parked_[key] = std::move(pk);
        return total;
    }
    void unpark(const std::string & key) const {
        const auto it = parked_.find(key); if (it == parked_.end()) return;
        auto & r = *resident_.at(key); auto & pk = it->second; std::vector<const char *> fresh;
        { PersistentAlloc keep;
          for (size_t i = 0; i < pk.host.size(); ++i) {
              auto buf = std::make_unique<Device<char>>(pk.bytes[i]);
              ck(cudaMemcpy(buf->p, pk.host[i], pk.bytes[i], cudaMemcpyHostToDevice));
              fresh.push_back(buf->p); r.mem.push_back(std::move(buf));
          } }
        auto repoint = [&](const void * q) -> const void * {
            const char * c = static_cast<const char *>(q);
            for (size_t i = 0; i < pk.old.size(); ++i) if (c >= pk.old[i] && c < pk.old[i] + pk.bytes[i]) return fresh[i] + (c - pk.old[i]);
            return q;
        };
        require(!r.m.fmt->parts(), "park: composite heads are not supported");
        r.m.p = repoint(r.m.p); r.m.aux = repoint(r.m.aux);
        for (char * h : pk.host) ck(cudaFreeHost(h));
        parked_.erase(it);
    }
    size_t park_heads() const { return park("output.weight") + park("draft_head"); }
    void unpark_heads() const { unpark("output.weight"); unpark("draft_head"); }
    mutable std::map<std::string, std::unique_ptr<Device<float>>> dense_dev_;
    mutable cublasHandle_t blas_ = nullptr;
    cublasHandle_t blas() const { if (!blas_) require(cublasCreate(&blas_) == CUBLAS_STATUS_SUCCESS, "cannot create cuBLAS handle"); return blas_; }
    // Projections that share one input convert it to FP16 once inside a SharedInput scope (same values as per call).
    mutable const float * half_src_ = nullptr; mutable size_t half_count_ = 0; mutable const __half * half_ = nullptr;
    // a BF16 image of `bf16_src_` made by its producer; BF16 projections of that input use it instead of converting
    mutable const float * bf16_src_ = nullptr; mutable size_t bf16_count_ = 0; mutable const __nv_bfloat16 * bf16_img_ = nullptr;
    struct SharedBf16 {
        const Checkpoint & cp;
        SharedBf16(const Checkpoint & c, const float * x, size_t count, const __nv_bfloat16 * img) : cp(c) { c.bf16_src_ = x; c.bf16_count_ = count; c.bf16_img_ = img; }
        ~SharedBf16() { cp.bf16_src_ = nullptr; cp.bf16_img_ = nullptr; }
    };
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
    // TQLLM_DEBUG=mem: device bytes of the resident projections and dense vectors by GGUF type, against their file bytes
    void resident_report() const {
        std::map<int, std::array<double, 3>> by;  // type -> device, file, count
        auto add = [&](const std::string & name, double dev) {
            const auto & t = file(name).file().tensors.at(name); auto & e = by[int(t.type)];
            e[0] += dev; e[1] += double(file(name).tensor(name).size); e[2] += 1;
        };
        for (const auto & [name, r] : resident_) {
            if (!owners_.count(name)) continue;
            double dev = 0; for (const auto & b : r->mem) dev += double(b->count); if (r->dense) dev += double(r->dense->count) * 4;
            add(name, dev);
        }
        for (const auto & [name, d] : dense_dev_) if (owners_.count(name)) add(name, double(d->count) * 4);
        double dt = 0, ft = 0;
        for (const auto & [type, e] : by) { dt += e[0]; ft += e[1];
            std::fprintf(stderr, "RESIDENT type %d: %.0f tensors, device %.0f MiB, file %.0f MiB\n", type, e[2], e[0] / 1048576.0, e[1] / 1048576.0); }
        std::fprintf(stderr, "RESIDENT total: device %.0f MiB, file %.0f MiB\n", dt / 1048576.0, ft / 1048576.0);
    }
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
    bool is_bf16(const std::string & name) const { return file(name).file().tensors.at(name).type == GGML_BF16; }
    const float * dense_device(const std::string & name) const {
        auto & d = dense_dev_[name];
        if (!d) { PersistentAlloc keep; const auto v = dense(name); d = std::make_unique<Device<float>>(v.size()); d->put(v); }
        return d->p;
    }
    // Greedy head for the last row of a device [count][hidden] F32 block; the matrix stays resident.
    // Greedy head for every row of a device [count][hidden] F32 block in one weight read (speculative verify).
    std::vector<int> head_argmax_rows(const float * rows, int count) const {
        static const bool req_log = std::getenv("TQFN_PROFILE_REQ") != nullptr;  // diagnostic: where a slow first token goes
        const auto t0 = std::chrono::steady_clock::now();
        const auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
        const auto * r = &head_resident();
        const double t_res = ms();
        const size_t V = r->m.rows, H = config.hidden;
        Device<__half> input(size_t(count) * H); Device<float> output(size_t(count) * V);
        const double t_alloc = ms();
        tqllm_cuda::vis_f32_to_h(rows, input.p, int64_t(count) * H, nullptr);
        const int step = std::max(1, r->m.fmt->max_matvec_cols());
        for (int b = 0; b < count; b += step)
            r->m.fmt->matvec_f32(r->m, input.p + size_t(b) * H, output.p + size_t(b) * V, std::min(step, count - b), nullptr);
        const double t_launch = ms();
        auto best = argmax_rows(output.p, int(V), count);
        if (req_log) std::fprintf(stderr, "tqllm: head argmax %d rows: resident %.0f ms, alloc %.0f ms, launches %.0f ms, argmax+sync %.0f ms\n",
                                  count, t_res, t_alloc - t_res, t_launch - t_alloc, ms() - t_launch);
        return best;
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
    // conf (optional): the draft's top softmax probability over the draft vocabulary (TQFN_DRAFT_TAU)
    int draft_head_argmax_device(const float * rows, int count, float * conf = nullptr) const {
        const auto * ids = draft_vocab();
        if (!ids) { if (conf) *conf = 1.f; return head_argmax_device(rows, count); }
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
        if (conf) { Device<float> pk(1); tqllm_cuda::fn_softmax_peak(output.p, int(ids->size()), pk.p, nullptr); *conf = pk.get()[0]; }
        return (*ids)[size_t(argmax_rows(output.p, int(ids->size()), 1)[0])];
    }
    const Resident & head_resident() const {
        static const bool req_log = std::getenv("TQFN_PROFILE_REQ") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        const auto & f = file("output.weight"); const auto & t = f.file().tensors.at("output.weight");
        auto & r = resident_["output.weight"]; if (!r) r = std::make_unique<Resident>();
        if (req_log) std::fprintf(stderr, "tqllm: head_resident: file lookup %.0f ms, %s\n",
                                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), r->m.fmt ? "resident" : "LOADING");
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
            if (req_log) std::fprintf(stderr, "tqllm: head_resident: rebuilt in %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
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
            // TQFN_DENSE_INT8: int8 activations for the dense prefill projections (1 per-32 scales, 2 per token)
            static const int dense_int8 = std::getenv("TQFN_DENSE_INT8") ? std::atoi(std::getenv("TQFN_DENSE_INT8")) : 0;
            if (dense_int8 && tokens > 64 && cols % 64 == 0) {
                Device<int8_t> xq8(size_t(cols) * tokens); Device<float> xs8(size_t(tokens) * (cols / 32)); Device<int16_t> xsum8(size_t(tokens) * (cols / 32) * 2);
                tqllm_cuda::k_quant_act_rows(xp, nullptr, nullptr, 0, 0, tokens, cols, xq8.p, xs8.p, xsum8.p, nullptr, dense_int8);
                if (matrix.fmt->gemm_i8_f32(matrix, xq8.p, xs8.p, xsum8.p, output, tokens, nullptr, dense_int8)) {
                    if (!resident) ck(cudaDeviceSynchronize());
                    return;
                }
            }
            // dense FP16 cache (Strata's dense path, --dense-cache-mb): chunks of >= 256 tokens dequantise the weight to fp16 once and run one
            // cuBLAS fp16 tensor-core GEMM (fp32 accumulate); the weight decode is amortised over the whole chunk
            const int dense_deq = g_tune.dense_cache_mb > 0 ? g_tune.dense_min : 0;
            if (dense_deq > 0 && tokens >= dense_deq) {
                __half * whp = nullptr; std::unique_ptr<Device<__half>> wh;
                if (const size_t pool = dense_half_pool_bytes()) {
                    int layer = -1; std::sscanf(name.c_str(), "blk.%d.", &layer);
                    if (layer != g_dhc.layer) { g_dhc.used = 0; g_dhc.map.clear(); g_dhc.layer = layer; }
                    if (auto it = g_dhc.map.find(matrix.p); it != g_dhc.map.end()) whp = it->second;
                    else {
                        const size_t b = (size_t(rows) * cols * sizeof(__half) + 255) / 256 * 256;
                        if (!g_dhc.base && cudaMalloc(reinterpret_cast<void **>(&g_dhc.base), pool) == cudaSuccess) g_dhc.cap = pool;
                        cudaGetLastError();
                        if (g_dhc.base && g_dhc.used + b <= g_dhc.cap) {
                            whp = reinterpret_cast<__half *>(g_dhc.base + g_dhc.used); g_dhc.used += b; g_dhc.high = std::max(g_dhc.high, g_dhc.used);
                            matrix.fmt->dequant_h(matrix, whp, nullptr); g_dhc.map[matrix.p] = whp;
                        }
                    }
                }
                if (!whp) { wh = std::make_unique<Device<__half>>(size_t(rows) * cols); matrix.fmt->dequant_h(matrix, wh->p, nullptr); whp = wh->p; }
                const float alpha = 1.f, beta = 0.f;
                require(cublasGemmEx(blas(), CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols, &alpha, whp, CUDA_R_16F, cols,
                                     xp, CUDA_R_16F, cols, &beta, output, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                        == CUBLAS_STATUS_SUCCESS, "dense fp16 GEMM failed: " + name);
                if (!resident) ck(cudaDeviceSynchronize());
                return;
            }
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
                const bool have_img = bf16_img_ && bf16_src_ == input && bf16_count_ == size_t(cols) * tokens;
                Device<__nv_bfloat16> xb(have_img ? 1 : size_t(cols) * tokens);
                if (!have_img) tqllm_cuda::f32_to_bf16(input, xb.p, xb.count, nullptr);
                const void * xbp = have_img ? static_cast<const void *>(bf16_img_) : static_cast<const void *>(xb.p);
                cublasHandle_t handle = blas_;
                if (!handle) { require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cannot create BF16 projection handle"); blas_ = handle; }
                const float alpha = 1.f, beta = 0.f;
                require(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, cols, &alpha, r->mem[0]->p, CUDA_R_16BF, cols,
                                     xbp, CUDA_R_16BF, cols, &beta, output, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
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
inline void compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected, double tolerance) {
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
inline std::vector<float> bf16_values(const Device<__nv_bfloat16> & device) {
    const auto raw = device.get(); std::vector<float> result(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) result[i] = __bfloat162float(raw[i]); return result;
}
