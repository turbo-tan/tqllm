#include "qwen4exp.hpp"
#include "qwen4exp_kernels.cuh"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t count) { ck(cudaMalloc(&p, count * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & values) {
        ck(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t count) const {
        std::vector<T> values(count);
        ck(cudaMemcpy(values.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

static void check_case(tqllm_cuda::QsaSelector & selector, const std::vector<float> & scores,
                       int cells, int position, int ratio, int top_k) {
    Device<float> device_scores(scores.size());
    Device<int32_t> device_positions(cells);
    Device<uint8_t> device_valid(cells);
    device_scores.put(scores);
    const int width = selector.select(device_scores.p, cells, position,
                                      device_positions.p, device_valid.p, nullptr);
    const auto expected = tqllm::qsa_select_visible(scores, cells, position, ratio, top_k);
    if (width != int(expected.positions.size()) || device_positions.get(width) != expected.positions ||
        device_valid.get(width) != expected.valid)
        throw std::runtime_error("QSA device selection differs from CPU oracle");
}

static void check_scores(int blocks, int heads, int dim) {
    std::vector<float> keys(size_t(blocks) * dim), query(size_t(heads) * dim);
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = float(int(i * 37 % 31) - 15) / 31.f;
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i * 19 % 23) - 11) / 23.f;
    Device<float> dk(keys.size()), dq(query.size()), ds(blocks);
    dk.put(keys); dq.put(query);
    tqllm_cuda::qsa_score_blocks(dk.p, dq.p, ds.p, blocks, heads, dim, nullptr);
    const auto got = ds.get(blocks);
    for (int b = 0; b < blocks; ++b) {
        double expected = 0.;
        for (int h = 0; h < heads; ++h) {
            double dot = 0.;
            for (int d = 0; d < dim; ++d) dot += double(keys[size_t(b) * dim + d]) * query[size_t(h) * dim + d];
            expected += std::max(dot, 0.);
        }
        if (std::fabs(got[b] - expected) > 2e-5 * std::max(1., expected))
            throw std::runtime_error("QSA block score differs from CPU oracle");
    }
    if (blocks > 1) {
        const int cells = blocks * 4, top_k = std::min(2048, cells);
        tqllm_cuda::QsaSelector selector(cells, 4, top_k);
        Device<int32_t> positions(cells);
        Device<uint8_t> valid(cells);
        const int width = selector.select(ds.p, cells, cells - 1, positions.p, valid.p, nullptr);
        const auto expected = tqllm::qsa_select_visible(got, cells, cells - 1, 4, top_k);
        if (width != int(expected.positions.size()) || positions.get(width) != expected.positions ||
            valid.get(width) != expected.valid)
            throw std::runtime_error("QSA scored selection differs from CPU oracle");
    }
}

static void check_raw_cache(int cells, int ratio, int dim) {
    std::vector<float> keys(size_t(cells) * dim);
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = float(int(i * 17 % 43) - 21) / 43.f;
    Device<float> dk(keys.size()); dk.put(keys);
    std::vector<float> weights(dim);
    for (int d = 0; d < dim; ++d) weights[d] = 1.f + float(d % 7) / 10.f;
    Device<float> dw(weights.size()); dw.put(weights);
    tqllm_cuda::QsaRawKeyCache cache(cells, ratio, dim);
    for (int at = 0; at < cells;) {
        const int chunk = std::min(cells - at, (at % 5) + 1);
        cache.append(dk.p + int64_t(at) * dim, chunk, nullptr);
        cache.normalize_new(dw.p, dim, 10000.f, 1e-6f, nullptr);
        at += chunk;
        if (cache.size() != at || cache.complete_blocks() != at / ratio)
            throw std::runtime_error("QSA raw-key cache size differs");
    }
    auto compare = [&](int complete) {
        std::vector<float> got(size_t(complete) * dim), norm(got.size());
        if (!got.empty()) ck(cudaMemcpy(got.data(), cache.pooled(), got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (!norm.empty()) ck(cudaMemcpy(norm.data(), cache.normalized(), norm.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int b = 0; b < complete; ++b) {
            std::vector<float> block(dim);
            for (int d = 0; d < dim; ++d) {
                float expected = keys[(int64_t(b) * ratio) * dim + d];
                for (int r = 1; r < ratio; ++r)
                    expected += keys[(int64_t(b) * ratio + r) * dim + d];
                expected *= 1.f / ratio;
                block[d] = expected;
                if (std::fabs(got[int64_t(b) * dim + d] - expected) > 1e-6f)
                    throw std::runtime_error("QSA raw-key pooled value differs");
            }
            double squares = 0.;
            for (float value : block) squares += double(value) * value;
            const double inv = 1. / std::sqrt(squares / dim + 1e-6);
            for (int d = 0; d < dim; ++d) {
                const int other = d < dim / 2 ? d + dim / 2 : d - dim / 2;
                const double angle = double(b * ratio) / std::pow(10000., double(2 * (d % (dim / 2))) / dim);
                const double value = double(block[d]) * inv * weights[d];
                const double paired = double(block[other]) * inv * weights[other];
                const double expected = d < dim / 2 ? value * std::cos(angle) - paired * std::sin(angle)
                                                    : value * std::cos(angle) + paired * std::sin(angle);
                if (std::fabs(norm[int64_t(b) * dim + d] - expected) > 1e-4)
                    throw std::runtime_error("QSA pooled key norm/RoPE differs from CPU oracle");
            }
        }
    };
    compare(cells / ratio);
    if (cells >= ratio + 1) {
        const int keep = ratio - 1;
        cache.truncate(keep);
        for (size_t i = size_t(keep) * dim; i < keys.size(); ++i) keys[i] += .25f;
        dk.put(keys);
        cache.append(dk.p + int64_t(keep) * dim, cells - keep, nullptr);
        cache.normalize_new(dw.p, dim, 10000.f, 1e-6f, nullptr);
        compare(cells / ratio);
    }
}

static std::vector<double> cpu_norm_rope(const std::vector<float> & raw, const std::vector<float> & weight,
                                         int position, float epsilon) {
    const int dim = int(raw.size()), half = dim / 2;
    double sum = 0.;
    for (float v : raw) sum += double(v) * v;
    const double inv = 1. / std::sqrt(sum / dim + epsilon);
    std::vector<double> out(dim);
    for (int d = 0; d < dim; ++d) {
        const int other = d < half ? d + half : d - half;
        const double angle = double(position) / std::pow(10000., double(2 * (d % half)) / dim);
        const double value = double(raw[d]) * inv * weight[d];
        const double paired = double(raw[other]) * inv * weight[other];
        out[d] = d < half ? value * std::cos(angle) - paired * std::sin(angle)
                          : value * std::cos(angle) + paired * std::sin(angle);
    }
    return out;
}

static void check_indexer() {
    constexpr int cells = 14, ratio = 4, top_k = 8, heads = 4, dim = 128;
    std::vector<float> keys(size_t(cells) * dim), query(size_t(heads) * dim), kn(dim), qn(dim);
    for (size_t i = 0; i < keys.size(); ++i) keys[i] = std::sin(float(i * 17 + 3) / 31.f);
    for (size_t i = 0; i < query.size(); ++i) query[i] = std::cos(float(i * 13 + 7) / 19.f);
    for (int d = 0; d < dim; ++d) { kn[d] = 1.f + float(d % 11) / 20.f; qn[d] = 1.f + float(d % 7) / 10.f; }
    Device<float> dk(keys.size()), dq(query.size()), dkn(kn.size()), dqn(qn.size());
    dk.put(keys); dq.put(query); dkn.put(kn); dqn.put(qn);
    tqllm_cuda::QsaIndexer indexer(cells, ratio, top_k, heads, dim, dim, 10000.f, 1e-6f);
    indexer.append(dk.p, 3, dkn.p, nullptr);
    indexer.append(dk.p + int64_t(3) * dim, cells - 3, dkn.p, nullptr);
    auto check = [&](int position) {
        std::vector<float> scores((position + 1) / ratio);
        for (int b = 0; b < int(scores.size()); ++b) {
            std::vector<float> pooled(dim);
            for (int d = 0; d < dim; ++d) {
                pooled[d] = keys[(int64_t(b) * ratio) * dim + d];
                for (int r = 1; r < ratio; ++r)
                    pooled[d] += keys[(int64_t(b) * ratio + r) * dim + d];
                pooled[d] *= 1.f / ratio;
            }
            const auto k = cpu_norm_rope(pooled, kn, b * ratio, 1e-6f);
            double score = 0.;
            for (int h = 0; h < heads; ++h) {
                const std::vector<float> query_head(query.begin() + int64_t(h) * dim,
                                                    query.begin() + int64_t(h + 1) * dim);
                const auto q = cpu_norm_rope(query_head, qn, position, 1e-6f);
                double dot = 0.;
                for (int d = 0; d < dim; ++d) dot += k[d] * q[d];
                score += std::max(dot, 0.);
            }
            scores[b] = float(score);
        }
        const auto expected = tqllm::qsa_select_visible(scores, cells, position, ratio, top_k);
        const auto selected = indexer.select(dq.p, dqn.p, position, nullptr);
        std::vector<int32_t> got(selected.width);
        std::vector<uint8_t> valid(selected.width);
        ck(cudaMemcpy(got.data(), selected.positions, got.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(valid.data(), selected.valid, valid.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        if (got != expected.positions || valid != expected.valid)
            throw std::runtime_error("QSA indexer output differs from CPU oracle");
    };
    for (int position : {0, 3, 4, 7, 9, 13}) check(position);
    indexer.truncate(5);
    for (size_t i = size_t(5) * dim; i < keys.size(); ++i) keys[i] += .4f;
    dk.put(keys);
    indexer.append(dk.p + int64_t(5) * dim, cells - 5, dkn.p, nullptr);
    check(13);
}

static void check_gather(int width) {
    constexpr int cells = 2052, heads = 2, dim = 256, row = heads * dim;
    std::vector<__nv_bfloat16> keys(size_t(cells) * row), values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = __float2bfloat16(float(int(i * 11 % 191) - 95) / 37.f);
        values[i] = __float2bfloat16(float(int(i * 29 % 173) - 86) / 41.f);
    }
    std::vector<int32_t> positions(width);
    std::vector<uint8_t> valid(width);
    for (int i = 0; i < width; ++i) {
        valid[i] = i % 5 != 0;
        positions[i] = valid[i] ? (i * 137) % cells : std::numeric_limits<int32_t>::max();
    }
    Device<__nv_bfloat16> dk(keys.size()), dv(values.size()), outk(size_t(width) * row), outv(size_t(width) * row);
    Device<int32_t> dp(width); Device<uint8_t> dm(width);
    dk.put(keys); dv.put(values); dp.put(positions); dm.put(valid);
    tqllm_cuda::qsa_gather_flat_bf16(dk.p, dv.p, {dp.p, dm.p, width}, outk.p, outv.p,
                                      cells, heads, dim, nullptr);
    const auto gotk = outk.get(size_t(width) * row), gotv = outv.get(size_t(width) * row);
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < row; ++j) {
            const float expected_k = valid[i] ? __bfloat162float(keys[int64_t(positions[i]) * row + j]) : 0.f;
            const float expected_v = valid[i] ? __bfloat162float(values[int64_t(positions[i]) * row + j]) : 0.f;
            if (__bfloat162float(gotk[int64_t(i) * row + j]) != expected_k ||
                __bfloat162float(gotv[int64_t(i) * row + j]) != expected_v)
                throw std::runtime_error("QSA flat KV gather differs from independent row oracle");
        }
    constexpr int attention_heads = 4;
    std::vector<float> query(attention_heads * dim);
    for (size_t i = 0; i < query.size(); ++i) query[i] = std::sin(float(i * 7 + 1) / 29.f);
    Device<float> dq(query.size()), dout(query.size()); dq.put(query);
    tqllm_cuda::qsa_attention_selected(dq.p, outk.p, outv.p, dm.p, dout.p,
                                        width, attention_heads, heads, dim, nullptr);
    const auto actual = dout.get(query.size());
    for (int h = 0; h < attention_heads; ++h) {
        const int kh = h / (attention_heads / heads);
        std::vector<double> scores(width, -INFINITY);
        double maximum = -INFINITY;
        for (int i = 0; i < width; ++i) if (valid[i]) {
            double dot = 0.;
            for (int d = 0; d < dim; ++d)
                dot += double(query[h * dim + d]) * __bfloat162float(keys[(int64_t(positions[i]) * heads + kh) * dim + d]);
            scores[i] = dot / std::sqrt(double(dim));
            maximum = std::max(maximum, scores[i]);
        }
        double denominator = 0.;
        for (int i = 0; i < width; ++i) if (valid[i]) denominator += std::exp(scores[i] - maximum);
        for (int d = 0; d < dim; ++d) {
            double expected = 0.;
            for (int i = 0; i < width; ++i) if (valid[i])
                expected += std::exp(scores[i] - maximum) *
                            __bfloat162float(values[(int64_t(positions[i]) * heads + kh) * dim + d]);
            expected /= denominator;
            if (std::fabs(actual[h * dim + d] - expected) > 2e-4)
                throw std::runtime_error("QSA selected attention differs from CPU softmax oracle");
        }
    }
}

static void check_dense_parity() {
    constexpr int cells = 16, heads = 4, kv_heads = 2, dim = 256;
    std::vector<__nv_bfloat16> keys(cells * kv_heads * dim), values(keys.size());
    std::vector<float> query(heads * dim);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = __float2bfloat16(std::sin(float(i * 3 + 1) / 17.f));
        values[i] = __float2bfloat16(std::cos(float(i * 7 + 2) / 23.f));
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = std::sin(float(i * 11 + 5) / 31.f);
    std::vector<int32_t> positions(cells);
    std::vector<uint8_t> valid(cells, 1);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    Device<__nv_bfloat16> dk(keys.size()), dv(values.size()), gk(keys.size()), gv(values.size());
    Device<float> dq(query.size()), selected(query.size()), dense(query.size());
    Device<int32_t> dp(cells); Device<uint8_t> dm(cells);
    dk.put(keys); dv.put(values); dq.put(query); dp.put(positions); dm.put(valid);
    tqllm_cuda::qsa_gather_flat_bf16(dk.p, dv.p, {dp.p, dm.p, cells}, gk.p, gv.p,
                                      cells, kv_heads, dim, nullptr);
    tqllm_cuda::qsa_attention_selected(dq.p, gk.p, gv.p, dm.p, selected.p,
                                        cells, heads, kv_heads, dim, nullptr);
    tqllm_cuda::fn_dense_attention(dq.p, dk.p, dv.p, dense.p, heads, kv_heads, dim, 1, cells - 1, nullptr);
    const auto a = selected.get(query.size()), b = dense.get(query.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > 2e-4f)
            throw std::runtime_error("QSA all-selected attention differs from dense attention");
}


// Decode attention over a compact copy of the selected rows (QsaGather, used when K/V pages live in host memory) gives
// the same output bits as reading the rows in place: same rows, same order, same causal cut.
static void check_compact_attention() {
    constexpr int cells = 5000, kvh = 2, heads = 24, dim = 256, tokens = 6, width = 2048, first = cells - tokens + 1, row = kvh * dim;
    std::vector<__nv_bfloat16> keys(size_t(cells) * row), values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = __float2bfloat16(float(int(i * 13 % 197) - 98) / 43.f);
        values[i] = __float2bfloat16(float(int(i * 31 % 181) - 90) / 47.f);
    }
    std::vector<float> query(size_t(tokens) * heads * dim);
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(int(i * 7 % 113) - 56) / 29.f;
    std::vector<int32_t> positions(size_t(tokens) * width); std::vector<uint8_t> valid(positions.size());
    for (int t = 0; t < tokens; ++t) for (int w = 0; w < width; ++w) {  // overlapping selections, some invalid, some past the causal cut
        const size_t i = size_t(t) * width + w;
        valid[i] = (w + t) % 9 != 0;
        positions[i] = w < width - 3 ? (w * 2 + t * 5) % (first + t) : first + t + w % 3;
    }
    Device<__nv_bfloat16> dk(keys.size()), dv(values.size()); Device<float> dq(query.size()), o1(query.size()), o2(query.size());
    Device<int32_t> dp(positions.size()); Device<uint8_t> dm(valid.size());
    dk.put(keys); dv.put(values); dq.put(query); dp.put(positions); dm.put(valid);
    tqllm_cuda::qsa_attention_direct(dq.p, dk.p, dv.p, {dp.p, dm.p, width}, o1.p, tokens, first, heads, kvh, dim, nullptr);
    tqllm_cuda::QsaGather gather(cells, size_t(row) * sizeof(__nv_bfloat16));
    gather.run(dp.p, dm.p, tokens, width, first, dk.p, dv.p, nullptr);
    tqllm_cuda::qsa_attention_direct(dq.p, static_cast<const __nv_bfloat16 *>(gather.keys()), static_cast<const __nv_bfloat16 *>(gather.values()),
                                     {gather.positions(), dm.p, width}, o2.p, tokens, tokens * width + 1, heads, kvh, dim, nullptr);
    ck(cudaDeviceSynchronize());
    const auto a = o1.get(query.size()), b = o2.get(query.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) throw std::runtime_error("QSA compact attention differs at " + std::to_string(i));
}

int main() {
    try {
        check_raw_cache(3, 4, 128);
        check_raw_cache(9, 4, 128);
        check_raw_cache(2051, 4, 128);
        check_indexer();
        check_gather(7);
        check_gather(2051);
        check_dense_parity();
        check_compact_attention();
        check_scores(1, 4, 128);
        check_scores(2051, 4, 128);
        check_scores(65536, 4, 128);
        constexpr int cells = 2052, ratio = 4, top_k = 2048;
        std::vector<float> scores((cells + ratio - 1) / ratio);
        for (size_t i = 0; i < scores.size(); ++i) scores[i] = float((i * 37) % 101) / 17.f;
        tqllm_cuda::QsaSelector selector(cells, ratio, top_k);
        for (int position : {0, 1, 3, 4, 5, 7, 9, 2047, 2048, 2051})
            check_case(selector, scores, cells, position, ratio, top_k);
        std::fill(scores.begin(), scores.end(), 1.f);
        check_case(selector, scores, cells, 2051, ratio, top_k);
        constexpr int long_cells = 262144;
        std::vector<float> long_scores(long_cells / ratio);
        for (size_t i = 0; i < long_scores.size(); ++i)
            long_scores[i] = float((i * 139) % 1009) / 97.f;
        tqllm_cuda::QsaSelector long_selector(long_cells, ratio, top_k);
        check_case(long_selector, long_scores, long_cells, long_cells - 1, ratio, top_k);
        std::puts("QSA device selector: PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
