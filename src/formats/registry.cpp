#include "weight_format.hpp"

#include <map>
#include <mutex>
#include <stdexcept>

namespace tqllm {

void WeightFormat::matvec_f32(const Mat &, const __half *, float *, int, cudaStream_t) const {
    throw std::runtime_error(std::string("weight format ") + name() + " has no FP32-output matvec");
}

namespace formats {  // built-in formats; a new format adds one declaration and one call below
void register_tq3_4s();
void register_q4();
void register_tq4();
void register_tq3();
void register_i8h();
}

namespace {
std::map<uint32_t, std::unique_ptr<WeightFormat>> & table() {
    static std::map<uint32_t, std::unique_ptr<WeightFormat>> t;
    return t;
}
void builtins() {
    static std::once_flag once;
    std::call_once(once, [] {
        formats::register_tq3_4s();
        formats::register_q4();
        formats::register_tq4();
        formats::register_tq3();
        formats::register_i8h();
    });
}
}  // namespace

void register_weight_format(uint32_t gguf_type, std::unique_ptr<WeightFormat> f) {
    if (!table().emplace(gguf_type, std::move(f)).second)
        throw std::runtime_error("weight format id " + std::to_string(gguf_type) + " registered twice");
}

const WeightFormat * weight_format(uint32_t gguf_type) {
    builtins();
    const auto it = table().find(gguf_type);
    return it == table().end() ? nullptr : it->second.get();
}

}  // namespace tqllm
