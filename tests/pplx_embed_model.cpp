// PplxEmbedder against the Hugging Face reference: cosine of every chunk and query vector >= 0.999.
// TQLLM_PPLX_GGUF = converted model; TQLLM_PPLX_TOKENS = {"docs": [{"chunks"}], "queries": [{"text"}]};
// TQLLM_PPLX_MODE=exact (default) | bf16 (BF16 products, a test mode) | fast (the served BF16 forward; queries batched as
// segments of one pass). Gates against the HF fp32 reference (tqDoc docs/EMBED-ENGINE.md): exact: min >= 0.999;
// bf16 / fast: p1 >= 0.999922 (the worse HF bf16 run's p1 vs fp32) and min >= 0.982 (see kTailMin; retrieval is the
// deciding gate, tqDoc docs/EMBED-ENGINE.md; our BF16
// forward may not be noisier than HF's own; short chunks such as covid-qa doc 17's last are bf16-sensitive).
// TQLLM_PPLX_REF = float32 rows (dim each): every chunk of every doc in order, then every query.
#include "pplx_embed.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

int main() {
    // ctest's pplx-embed-model-fast sets TQLLM_PPLX_MODE=fast and reads the served file from TQLLM_PPLX_FAST_GGUF (BF16; the
    // exact test needs the F32 conversion in TQLLM_PPLX_GGUF)
    const bool fast_test = std::getenv("TQLLM_PPLX_FAST_TEST") != nullptr;
    const char * gguf = std::getenv(fast_test ? "TQLLM_PPLX_FAST_GGUF" : "TQLLM_PPLX_GGUF");
    const char * fixture = std::getenv("TQLLM_PPLX_TOKENS");
    const char * ref = std::getenv("TQLLM_PPLX_REF");
    if (!gguf || !fixture || !ref) {
        std::cout << "pplx embed model skipped (set " << (fast_test ? "TQLLM_PPLX_FAST_GGUF" : "TQLLM_PPLX_GGUF") << ", TQLLM_PPLX_TOKENS, TQLLM_PPLX_REF)\n";
        return 77;
    }
    tqllm::Tokenizer tok;
    tok.load(gguf);
    if (const char * li = std::getenv("TQLLM_PPLX_LAYERS")) {  // "doc:chunk": per-layer hidden cosine, fast vs exact, over that chunk's tokens
        std::stringstream fs; fs << std::ifstream(fixture).rdbuf();
        const std::string ftext = fs.str();
        const auto fj = tqllm::Parser(ftext).parse();
        const int d = std::atoi(li), c = std::atoi(std::strchr(li, ':') + 1);
        std::vector<std::string> chunks;
        for (const auto & x : fj.get("docs")->arr[size_t(d)].get("chunks")->arr) chunks.push_back(x.str);
        std::vector<std::vector<float>> la, lb;
        tqllm::EmbedInput in;
        { auto fast = tqllm::PplxEmbedder::load(gguf, 0, tqllm::PplxEmbedder::Mode::fast); in = tqllm::embed_document_input(tok, fast->prompt(), chunks); fast->embed(in, &la); }
        { auto exact = tqllm::PplxEmbedder::load(gguf, 0, tqllm::PplxEmbedder::Mode::exact); exact->embed(in, &lb); }
        const size_t H = la[0].size() / in.ids.size();
        const auto [b0, e0] = in.spans[size_t(c)];
        if (const char * pre = std::getenv("TQLLM_PPLX_LAYERS_DUMP")) {  // [layers + 1][tokens][H] float32 each
            for (const auto & [name, ls] : {std::make_pair(std::string("_fast.f32"), &la), std::make_pair(std::string("_exact.f32"), &lb)}) {
                std::ofstream o(std::string(pre) + name, std::ios::binary);
                for (const auto & l : *ls) o.write(reinterpret_cast<const char *>(l.data()), std::streamsize(l.size() * 4));
            }
        }
        std::cout << "doc " << d << " chunk " << c << ": tokens " << b0 << ".." << e0 << " of " << in.ids.size() << "\n";
        for (size_t l = 0; l < la.size(); ++l) {
            double mn = 2, rel = 0, all = 2;
            for (size_t t = 0; t < in.ids.size(); ++t) {
                double ab = 0, aa = 0, bb = 0, dd = 0;
                for (size_t h = 0; h < H; ++h) { const double a = la[l][t * H + h], b = lb[l][t * H + h]; ab += a * b; aa += a * a; bb += b * b; dd += (a - b) * (a - b); }
                const double cs = ab / std::sqrt(aa * bb);
                all = std::min(all, cs);
                if (t >= b0 && t < e0) { mn = std::min(mn, cs); rel = std::max(rel, std::sqrt(dd / bb)); }
            }
            std::cout << (l + 1 < la.size() ? "layer " + std::to_string(l) : std::string("final norm")) << ": chunk min cos " << mn
                      << ", max rel err " << rel << "; whole doc min cos " << all << "\n";
        }
        return 0;
    }
    if (std::getenv("TQLLM_PPLX_LONG")) {  // one document above 65,535 tokens (grid-y limits): fast vs exact
        std::stringstream fs; fs << std::ifstream(fixture).rdbuf();
        const std::string ftext = fs.str();
        const auto fj = tqllm::Parser(ftext).parse();
        std::vector<std::string> chunks;
        tqllm::EmbedInput in;
        for (const auto & d : fj.get("docs")->arr) {
            for (const auto & c : d.get("chunks")->arr) chunks.push_back(c.str);
            in = tqllm::embed_document_input(tok, tqllm::EmbedPrompt{"[D] ", "<|chunk_sep|>", tok.special_id("[Q] ")}, chunks);
            if (in.ids.size() > 70000) break;
        }
        std::vector<float> a, b;
        { auto fast = tqllm::PplxEmbedder::load(gguf, 0, tqllm::PplxEmbedder::Mode::fast); in = tqllm::embed_document_input(tok, fast->prompt(), chunks); a = fast->embed(in); }
        { auto exact = tqllm::PplxEmbedder::load(gguf, 0, tqllm::PplxEmbedder::Mode::exact); b = exact->embed(in); }
        const int dim = int(a.size() / in.spans.size());
        double mn = 2, sum = 0;
        for (size_t v = 0; v < in.spans.size(); ++v) {
            double ab = 0, aa = 0, bb = 0;
            for (int c = 0; c < dim; ++c) { ab += double(a[v * dim + c]) * b[v * dim + c]; aa += double(a[v * dim + c]) * a[v * dim + c]; bb += double(b[v * dim + c]) * b[v * dim + c]; }
            const double cs = ab / std::sqrt(aa * bb);
            mn = std::min(mn, cs); sum += cs;
        }
        const double mean = sum / in.spans.size();
        std::cout << "long document: " << in.ids.size() << " tokens, " << in.spans.size() << " chunks; fast vs exact cosine min " << mn
                  << ", mean " << mean << "\n";
        const bool ok = in.ids.size() > 65535 && mean >= 0.9999 && mn >= 0.99;
        std::cout << (ok ? "pplx embed long: OK\n" : "pplx embed long: FAILED\n");
        return ok ? 0 : 1;
    }
    const std::string mode = std::getenv("TQLLM_PPLX_MODE") ? std::getenv("TQLLM_PPLX_MODE") : "exact";
    using Mode = tqllm::PplxEmbedder::Mode;
    const Mode m = mode == "fast" ? Mode::fast : mode == "bf16" ? Mode::bf16_products : Mode::exact;
    const bool bf16 = m != Mode::exact;
    auto e = tqllm::PplxEmbedder::load(gguf, 0, m);
    const int dim = e->dim();
    std::stringstream ss; ss << std::ifstream(fixture).rdbuf();
    const std::string text = ss.str();
    const auto j = tqllm::Parser(text).parse();
    std::ifstream rf(ref, std::ios::binary | std::ios::ate);
    if (!rf) { std::cerr << "FAIL: cannot read " << ref << "\n"; return 1; }
    std::vector<float> want(size_t(rf.tellg()) / 4);
    rf.seekg(0);
    rf.read(reinterpret_cast<char *>(want.data()), std::streamsize(want.size() * 4));
    const size_t rows = want.size() / dim;
    size_t row = 0, worst = 0, tokens = 0;
    double min_cos = 2, sum = 0;
    std::vector<double> all;
    std::ofstream dump;  // TQLLM_PPLX_DUMP=file: every vector (float32 rows, reference order) for offline comparison
    if (const char * d = std::getenv("TQLLM_PPLX_DUMP")) dump.open(d, std::ios::binary);
    std::vector<float> dumped(want.size());
    auto score = [&](const std::vector<float> & got) {
        if (dump) std::copy(got.begin(), got.end(), dumped.begin() + row * dim);
        for (size_t v = 0; v * dim < got.size() && row < rows; ++v, ++row) {
            double ab = 0, aa = 0, bb = 0;
            for (int c = 0; c < dim; ++c) { const double a = got[v * dim + c], b = want[row * dim + c]; ab += a * b; aa += a * a; bb += b * b; }
            const double cos = ab / std::sqrt(aa * bb);
            sum += cos;
            all.push_back(cos);
            if (!(cos >= min_cos)) { min_cos = cos; worst = row; }
        }
    };
    size_t chunk_rows = 0;
    for (const auto & d : j.get("docs")->arr) chunk_rows += d.get("chunks")->arr.size();
    if (rows != chunk_rows + j.get("queries")->arr.size() || want.size() != rows * dim) {
        std::cerr << "FAIL: the reference has " << rows << " rows, the items " << chunk_rows << " chunks + " << j.get("queries")->arr.size() << " queries\n";
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    size_t nd = 0, nq = 0;
    // TQLLM_PPLX_DOC_BATCH=k: k documents per embed_many call (segments of one pass in fast mode)
    const size_t batch = std::getenv("TQLLM_PPLX_DOC_BATCH") ? std::max(1, std::atoi(std::getenv("TQLLM_PPLX_DOC_BATCH"))) : 1;
    // TQLLM_PPLX_LANES=2: a second embedder on a second host thread takes every other document batch (two passes in flight)
    const int lanes = std::getenv("TQLLM_PPLX_LANES") ? std::max(1, std::atoi(std::getenv("TQLLM_PPLX_LANES"))) : 1;
    std::unique_ptr<tqllm::PplxEmbedder> e2 = lanes > 1 ? tqllm::PplxEmbedder::load(gguf, 0, m) : nullptr;
    std::vector<tqllm::EmbedInput> docs;
    std::vector<std::vector<tqllm::EmbedInput>> pending;
    auto run_docs = [&] {
        if (!e2) { for (const auto & v : e->embed_many(docs)) score(v); docs.clear(); return; }
        pending.push_back(std::move(docs)); docs.clear();
    };
    for (const auto & d : j.get("docs")->arr) {
        std::vector<std::string> chunks;
        for (const auto & c : d.get("chunks")->arr) chunks.push_back(c.str);
        docs.push_back(tqllm::embed_document_input(tok, e->prompt(), chunks));
        tokens += docs.back().ids.size();
        ++nd;
        if (docs.size() == batch) run_docs();
    }
    if (!docs.empty()) run_docs();
    if (e2) {  // batches alternate between the two embedders, each on its own thread; scored in order afterwards
        std::vector<std::vector<std::vector<float>>> got(pending.size());
        std::thread t2([&] { for (size_t i = 1; i < pending.size(); i += 2) got[i] = e2->embed_many(pending[i]); });
        for (size_t i = 0; i < pending.size(); i += 2) got[i] = e->embed_many(pending[i]);
        t2.join();
        for (const auto & g : got) for (const auto & v : g) score(v);
    }
    const auto tq = std::chrono::steady_clock::now();
    std::vector<tqllm::EmbedInput> qs;
    for (const auto & q : j.get("queries")->arr) {
        qs.push_back(tqllm::embed_query_input(tok, e->prompt(), q.get("text")->str));
        tokens += qs.back().ids.size();
        ++nq;
    }
    for (const auto & v : e->embed_many(qs)) score(v);
    const double qsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tq).count();
    std::cout << "queries: " << nq << " in " << qsec << " s (" << nq / qsec << " /s)\n";
    if (dump) dump.write(reinterpret_cast<const char *>(dumped.data()), std::streamsize(dumped.size() * 4));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "pplx embed: " << nd << " docs, " << nq << " queries, " << rows << " vectors, " << tokens << " tokens in " << s
              << " s (" << tokens / s << " tok/s); cosine min " << min_cos << " (row " << worst << "), mean " << sum / rows << "\n";
    std::sort(all.begin(), all.end());
    const double p1 = all.empty() ? 0 : all[all.size() / 100];
    // kTailP1: the worse HF bf16 run's p1 vs HF fp32 on this fixture (it separates files on both GPUs we gate).
    // kTailMin: the lowest min at which retrieval still held: the BF16-token-table 0.6B file on an RTX 3090 had min 0.982383
    // (row 483, a 71-character chunk) and covid-qa nDCG@10 56.34 vs HF fp32 56.32, FinChat 53.53 / 47.00 vs 52.56 / 46.54
    // (same client, 11-14); the previous floor (HF bf16 CPU's 0.995478) failed files whose retrieval did not move.
    constexpr double kTailMin = 0.982, kTailP1 = 0.999922;
    std::cout << "cosine p1 " << p1 << (bf16 ? " (" + mode + "; gate min >= 0.982, p1 >= 0.999922)\n" : " (FP32; gate min >= 0.999)\n");
    const bool finite = std::all_of(all.begin(), all.end(), [](double c) { return std::isfinite(c); });
    if (!finite) std::cout << "NaN / inf vectors\n";
    const bool ok = finite && (bf16 ? (p1 >= kTailP1 && min_cos >= kTailMin) : min_cos >= 0.999) && all.size() == rows;
    std::cout << (ok ? "pplx embed model: OK\n" : "pplx embed model: FAILED\n");
    return ok ? 0 : 1;
}
