#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace tqllm {
class HostAttentionCache {
public:
    HostAttentionCache(int context, int width) : width_(width), context_(context) {
        if (context <= 0 || width <= 0 || size_t(context) > size_t(-1) / sizeof(float) / size_t(width))
            throw std::invalid_argument("host KV dimensions");
        keys_.resize(size_t(context) * width); values_.resize(keys_.size());
    }
    void write(int pos, const std::vector<float> & k, const std::vector<float> & v) {
        if (pos < 0 || pos >= context_ || k.size() != size_t(width_) || v.size() != k.size())
            throw std::out_of_range("host KV write");
        std::copy(k.begin(), k.end(), keys_.begin() + size_t(pos) * width_);
        std::copy(v.begin(), v.end(), values_.begin() + size_t(pos) * width_);
    }
    const float * key(int pos, int offset) const { return keys_.data() + size_t(pos) * width_ + offset; }
    const float * value(int pos, int offset) const { return values_.data() + size_t(pos) * width_ + offset; }
    // Rows [0, count) of keys then values, as stored (prompt-state files).
    float * keys(int count) { check(count); return keys_.data(); }
    float * values(int count) { check(count); return values_.data(); }
    size_t row_floats() const { return size_t(width_); }
private:
    void check(int count) const { if (count < 0 || count > context_) throw std::out_of_range("host KV rows"); }
    int width_, context_;
    std::vector<float> keys_, values_;
};
}
