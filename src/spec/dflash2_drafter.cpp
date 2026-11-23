#include "dflash2_drafter.hpp"

#include "dequant.hpp"
#include "dflash_kernels.cuh"
#include "gguf.hpp"
#include "quant_kernels.cuh"
#include "attention_gen.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tqllm {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("DFlash2 drafter: ") + cudaGetErrorString(e)); }
}

void * Dflash2Drafter::dev(size_t bytes) {
    void * p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(bytes, 16)));
    owned_.push_back(p); bytes_ += bytes;
    return p;
}

Dflash2Drafter::~Dflash2Drafter() { for (void * p : owned_) cudaFree(p); }

Dflash2Drafter::Dflash2Drafter(const std::string & path, int target_hidden, int vocab, float eps, cudaStream_t stream)
    : s_(stream), hidden_(target_hidden), vocab_(vocab), eps_(eps) {
    const auto f = gguf_open(path);
    const std::string A = "dflash.";
    if (int(f.num(A + "embedding_length")) != hidden_) throw std::runtime_error("DFlash2 drafter: hidden size differs from the target");
    if (f.tensors.count("markov_w1.weight")) throw std::runtime_error("DFlash2 drafter: a DSpark file (not supported by this module)");
    ffn_ = int(f.num(A + "feed_forward_length")); heads_ = int(f.num(A + "attention.head_count"));
    kvheads_ = int(f.num(A + "attention.head_count_kv")); hd_ = int(f.num(A + "attention.key_length"));
    block_ = int(f.num(A + "block_size")); group_ = int(f.num(A + "conv_group_size"));
    window_ = int(f.num(A + "attention.sliding_window", 0)); rank_ = int(f.num(A + "selector_rank"));
    theta_ = float(f.num(A + "rope.freq_base"));
    mask_id_ = uint32_t(f.num("tokenizer.ggml.mask_token_id"));
    if (hd_ != 128 || int(f.num(A + "selector_top_k")) != 16 || int(f.num(A + "conv_kernel_size")) != 2 || block_ > 8 || window_ <= 0)
        throw std::runtime_error("DFlash2 drafter: unsupported shape");
    for (double v : f.kv.at(A + "target_layers").arr) layers_.push_back(int(v) - 1);  // one-based in the GGUF
    const int H = hidden_, nf = int(layers_.size()), qd = heads_ * hd_, kd = kvheads_ * hd_, nl = int(f.num(A + "block_count"));

    const Upload up = [this](const void * h, size_t b) { void * d = dev(b); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); return static_cast<const void *>(d); };
    auto format = [](uint32_t type) {
        const WeightFormat * wf = weight_format(type);
        if (!wf || wf->basis() != Basis::Plain)
            throw std::runtime_error("DFlash2 drafter: weight type " + std::to_string(type) + " not supported (Q4_0 / Q8_0 / K-quant files)");
        return wf;
    };
    // row-concatenated tensors -> one device matrix per run of equal types (Plain-basis formats only: no input rotation)
    auto rows_of = [&](const std::vector<std::string> & names, int cols) {
        Rows out; out.cols = cols;
        std::vector<char> raw; int rows = 0; uint32_t type = 0;
        auto flush = [&] { if (rows) out.parts.push_back(format(type)->load(raw, rows, cols, up, s_)); raw.clear(); rows = 0; };
        for (const auto & n : names) {
            const auto & t = f.tensors.at(n);
            if (t.ne.size() != 2 || int(t.ne[0]) != cols) throw std::runtime_error("DFlash2 drafter: unexpected shape for " + n);
            if (rows && t.type != type) flush();
            type = t.type; rows += int(t.ne[1]); out.rows += int(t.ne[1]);
            const auto b = f.read(t); raw.insert(raw.end(), b.begin(), b.end());
        }
        flush();
        return out;
    };
    auto mat = [&](const std::string & name, int cols) {
        Rows r = rows_of({name}, cols);
        return r.parts.at(0);
    };
    auto f32 = [&](const std::string & n, size_t count) {
        const auto & t = f.tensors.at(n); const auto b = f.read(t);
        std::vector<float> v(count); dequantize_host(t.type, b.data(), v.data(), int64_t(count));
        return static_cast<const float *>(up(v.data(), count * 4));
    };
    auto bf16_table = [&](const std::string & n) {  // [rank][vocab] in the GGUF -> [vocab][rank] BF16
        const auto & t = f.tensors.at(n); const auto b = f.read(t);
        const int64_t rk = int64_t(t.ne[0]), vc = int64_t(t.ne[1]);
        if (rk != rank_) throw std::runtime_error("DFlash2 drafter: selector rank differs in " + n);
        std::vector<float> v(size_t(rk * vc)); dequantize_host_rows(t.type, b.data(), v.data(), vc, rk);
        std::vector<__nv_bfloat16> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = __float2bfloat16_rn(v[i]);
        return static_cast<const __nv_bfloat16 *>(up(o.data(), o.size() * 2));
    };
    fc_ = mat("fc.weight", nf * H);
    enc_norm_ = f32("enc.output_norm.weight", size_t(H)); out_norm_ = f32("output_norm.weight", size_t(H));
    sel_hidden_ = mat("selector_hidden.weight", H);
    sel_prev_ = bf16_table("selector_predecessor.weight"); sel_next_ = bf16_table("selector_successor.weight");
    std::vector<std::string> kv_names;
    // the ring holds the window plus a block and slack (multiple of 64, as attention_gen128 requires)
    ring_ = (window_ + 64 + block_ + 64 + 63) / 64 * 64;
    for (int l = 0; l < nl; ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        Layer y;
        y.attn_norm = f32(p + "attn_norm.weight", size_t(H)); y.q_norm = f32(p + "attn_q_norm.weight", size_t(hd_));
        y.k_norm = f32(p + "attn_k_norm.weight", size_t(hd_)); y.ffn_norm = f32(p + "ffn_norm.weight", size_t(H));
        y.attn_conv_base = f32(p + "attn_conv_base", size_t(H) * 4); y.ffn_conv_base = f32(p + "ffn_conv_base", size_t(H) * 4);
        y.conv_attn = mat(p + "attn_conv_proj.weight", H); y.conv_ffn = mat(p + "ffn_conv_proj.weight", H);
        y.w_qkv = rows_of({p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}, H);
        y.wo = mat(p + "attn_output.weight", qd);
        y.w_gu = rows_of({p + "ffn_gate.weight", p + "ffn_up.weight"}, H);
        y.down = mat(p + "ffn_down.weight", ffn_);
        y.kc = static_cast<__nv_bfloat16 *>(dev(size_t(ring_) * kd * 2)); y.vc = static_cast<__nv_bfloat16 *>(dev(size_t(ring_) * kd * 2));
        L_.push_back(y);
        kv_names.push_back(p + "attn_k.weight"); kv_names.push_back(p + "attn_v.weight");
    }
    w_kv_all_ = rows_of(kv_names, H);  // [l0 k | l0 v | l1 k | ...] for the injection's one projection
    conv_rows_ = L_[0].conv_attn.rows;
    const size_t B = size_t(block_), bh = B * H * 2;
    x_ = static_cast<__nv_bfloat16 *>(dev(bh)); r_ = static_cast<__nv_bfloat16 *>(dev(bh)); a_ = static_cast<__nv_bfloat16 *>(dev(bh));
    b_ = static_cast<__nv_bfloat16 *>(dev(bh)); te_ = static_cast<__nv_bfloat16 *>(dev(bh));
    t_ = static_cast<__nv_bfloat16 *>(dev(B * size_t(std::max(L_[0].w_qkv.rows, L_[0].w_gu.rows)) * 2));
    o_ = static_cast<__nv_bfloat16 *>(dev(B * qd * 2)); q_ = static_cast<__nv_bfloat16 *>(dev(B * qd * 2));
    act_ = static_cast<__nv_bfloat16 *>(dev(B * ffn_ * 2));
    dyn_ = static_cast<__nv_bfloat16 *>(dev(B * conv_rows_ * 2)); dynf_ = static_cast<__nv_bfloat16 *>(dev(B * conv_rows_ * 2));
    gate_ = static_cast<__nv_bfloat16 *>(dev(B * rank_ * 2));
    part_ = static_cast<float *>(dev(size_t(tqllm_cuda::attention_gen_partial_floats(heads_, ring_, block_)) * 4));
    logits_ = static_cast<float *>(dev(B * size_t(vocab_) * 4));
    vals_ = static_cast<float *>(dev(B * 16 * 4)); idx_ = static_cast<uint32_t *>(dev(B * 16 * 4));
    score_ = static_cast<float *>(dev(B * 256 * 4)); tok_ = static_cast<uint32_t *>(dev(B * 4));
    topk_scratch_ = dev(B * 64 * 16 * 8); pos_ = static_cast<int *>(dev(2 * sizeof(int)));
}

void Dflash2Drafter::mm(const Mat & m, const __nv_bfloat16 * x, __nv_bfloat16 * y, int n) {
    const size_t need = size_t(n) * m.cols;
    if (xh_cap_ < need) { xh_ = static_cast<__half *>(dev(need * 2)); xh_cap_ = need; }
    tqllm_cuda::bf16_to_h(x, xh_, int(need), s_);
    for (int c0 = 0; c0 < n; c0 += 8)
        m.fmt->matvec(m, xh_ + size_t(c0) * m.cols, y + size_t(c0) * m.rows, std::min(8, n - c0), s_);
}

void Dflash2Drafter::mm(const Rows & m, const __nv_bfloat16 * x, __nv_bfloat16 * y, int n) {
    if (m.parts.size() == 1) { mm(m.parts[0], x, y, n); return; }
    size_t most = 0; for (const auto & p : m.parts) most = std::max(most, size_t(p.rows));
    if (part_y_cap_ < most * n) { part_y_ = static_cast<__nv_bfloat16 *>(dev(most * n * 2)); part_y_cap_ = most * n; }
    int off = 0;
    for (const auto & p : m.parts) {  // each part's [n][p.rows] into its column range of y [n][m.rows]
        mm(p, x, part_y_, n);
        ck(cudaMemcpy2DAsync(y + off, size_t(m.rows) * 2, part_y_, size_t(p.rows) * 2, size_t(p.rows) * 2, size_t(n), cudaMemcpyDeviceToDevice, s_));
        off += p.rows;
    }
}

void Dflash2Drafter::inject(const __nv_bfloat16 * feats, int n, uint32_t pos0) {
    if (n <= 0) return;
    const int H = hidden_, kd = kvheads_ * hd_, ts = w_kv_all_.rows;
    if (inj_cap_ < n) {
        inj_g_ = static_cast<__nv_bfloat16 *>(dev(size_t(n) * H * 2)); inj_kv_ = static_cast<__nv_bfloat16 *>(dev(size_t(n) * ts * 2)); inj_cap_ = n;
    }
    const int p = int(pos0);
    ck(cudaMemcpyAsync(pos_ + 1, &p, sizeof(int), cudaMemcpyHostToDevice, s_));
    mm(fc_, feats, inj_g_, n);
    tqllm_cuda::dflash_rms(inj_g_, H, 0, enc_norm_, inj_g_, H, 0, H, 1, n, eps_, s_);
    mm(w_kv_all_, inj_g_, inj_kv_, n);
    for (size_t l = 0; l < L_.size(); ++l) {
        __nv_bfloat16 * K = inj_kv_ + l * 2 * kd;
        tqllm_cuda::dflash_rms(K, ts, hd_, L_[l].k_norm, K, ts, hd_, hd_, kvheads_, n, eps_, s_);
        tqllm_cuda::dflash_rope(K, ts, n, kvheads_, hd_, hd_, pos_ + 1, theta_, s_);
        tqllm_cuda::dflash_cache_write(K, K + kd, ts, n, L_[l].kc, L_[l].vc, pos_ + 1, kvheads_, hd_, s_, ring_);
    }
}

std::vector<uint32_t> Dflash2Drafter::draft(uint32_t anchor, uint32_t pos, int want, const EmbedFn & embed, const HeadFn & head) {
    const int H = hidden_, B = block_, qd = heads_ * hd_, kd = kvheads_ * hd_;
    want = std::min(want, B - 1);
    if (want <= 0) return {};
    std::vector<uint32_t> blk(size_t(B), mask_id_); blk[0] = anchor;
    ck(cudaMemcpyAsync(tok_, blk.data(), size_t(B) * 4, cudaMemcpyHostToDevice, s_));
    const int p = int(pos);
    ck(cudaMemcpyAsync(pos_, &p, sizeof(int), cudaMemcpyHostToDevice, s_));
    embed(blk, x_, s_);
    for (const auto & y : L_) {
        ck(cudaMemcpyAsync(r_, x_, size_t(B) * H * 2, cudaMemcpyDeviceToDevice, s_));
        tqllm_cuda::dflash_rms(x_, H, 0, y.attn_norm, a_, H, 0, H, 1, B, eps_, s_);
        mm(y.conv_attn, a_, dyn_, B);
        tqllm_cuda::dflash_conv(a_, dyn_, conv_rows_, y.attn_conv_base, 0, b_, B, H, group_, s_);
        mm(y.w_qkv, b_, t_, B);
        const int rows = y.w_qkv.rows; __nv_bfloat16 * K0 = t_ + qd;
        tqllm_cuda::dflash_rms(t_, rows, hd_, y.q_norm, q_, qd, hd_, hd_, heads_, B, eps_, s_);
        tqllm_cuda::dflash_rms(K0, rows, hd_, y.k_norm, K0, rows, hd_, hd_, kvheads_, B, eps_, s_);
        tqllm_cuda::dflash_rope(q_, qd, B, heads_, hd_, hd_, pos_, theta_, s_);
        tqllm_cuda::dflash_rope(K0, rows, B, kvheads_, hd_, hd_, pos_, theta_, s_);
        tqllm_cuda::dflash_cache_write(K0, K0 + kd, rows, B, y.kc, y.vc, pos_, kvheads_, hd_, s_, ring_);
        int bucket = 256; while (bucket < p + B) bucket *= 2;
        tqllm_cuda::attention_gen128(q_, y.kc, y.vc, o_, part_, pos_, B, bucket, heads_, kvheads_, 1.f / std::sqrt(float(hd_)), false,
                                     window_, s_, ring_);
        mm(y.wo, o_, a_, B);
        tqllm_cuda::dflash_conv(a_, dyn_, conv_rows_, y.attn_conv_base, 1, b_, B, H, group_, s_);
        tqllm_cuda::dflash_residual(b_, r_, B * H, s_);
        tqllm_cuda::dflash_rms(b_, H, 0, y.ffn_norm, a_, H, 0, H, 1, B, eps_, s_);
        mm(y.conv_ffn, a_, dynf_, B);
        tqllm_cuda::dflash_conv(a_, dynf_, conv_rows_, y.ffn_conv_base, 0, x_, B, H, group_, s_);
        mm(y.w_gu, x_, t_, B);
        tqllm_cuda::dflash_swiglu(t_, B, ffn_, act_, s_);
        mm(y.down, act_, a_, B);
        tqllm_cuda::dflash_conv(a_, dynf_, conv_rows_, y.ffn_conv_base, 1, x_, B, H, group_, s_);
        tqllm_cuda::dflash_residual(x_, b_, B * H, s_);
    }
    tqllm_cuda::dflash_rms(x_, H, 0, out_norm_, te_, H, 0, H, 1, B, eps_, s_);
    head(te_, B, logits_, s_);
    tqllm_cuda::topk16_big(logits_, vocab_, B, vals_, idx_, topk_scratch_, s_);
    mm(sel_hidden_, te_, gate_, B);
    tqllm_cuda::dflash_lattice(idx_, vals_, gate_, rank_, tqllm_cuda::SelTable{sel_next_}, tqllm_cuda::SelTable{sel_prev_}, anchor, B, rank_, score_, s_);
    tqllm_cuda::dflash_chain(idx_, score_, want, tok_, s_);
    std::vector<uint32_t> out(static_cast<size_t>(want)), cand(size_t(B) * 16); std::vector<float> val(size_t(B) * 16);
    ck(cudaMemcpyAsync(out.data(), tok_, size_t(want) * 4, cudaMemcpyDeviceToHost, s_));
    if (tau_ > 0.f) {
        ck(cudaMemcpyAsync(cand.data(), idx_, cand.size() * 4, cudaMemcpyDeviceToHost, s_));
        ck(cudaMemcpyAsync(val.data(), vals_, val.size() * 4, cudaMemcpyDeviceToHost, s_));
    }
    ck(cudaStreamSynchronize(s_));
    // the chain stops at the first draft whose probability among its position's 16 best is below tau_ (at least one draft):
    // a block of 7 drafts of which ~3 are accepted made every verify step route 8 rows through the experts
    if (tau_ > 0.f)
        for (int j = 1; j < want; ++j) {
            const float * v = val.data() + size_t(j + 1) * 16; const uint32_t * c = cand.data() + size_t(j + 1) * 16;
            float mx = v[0], sum = 0.f, mine = 0.f;
            for (int q = 1; q < 16; ++q) mx = std::max(mx, v[q]);
            for (int q = 0; q < 16; ++q) { const float e = std::exp(v[q] - mx); sum += e; if (c[q] == out[size_t(j)]) mine = e; }
            if (mine < tau_ * sum) { out.resize(size_t(j)); break; }
        }
    return out;
}

}  // namespace tqllm
