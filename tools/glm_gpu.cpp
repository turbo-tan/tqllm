// glm-gpu: the GPU GLM-5.3-Flash engine with glm-ref's command line and output, for a line-by-line comparison.
//   glm-gpu MODEL.gguf --tokens 1,2,3 [--gen N | --force a,b,c] [--context N] [--step] [--spec D [--draft ids]]
// --spec D decodes through verify() blocks of D+1 tokens and commits only the accepted prefix (the KDA state and conv
// history roll back), so its tokens must be identical to the plain decode's. GLM_DUMP=1 adds the per-stage sums.
#include "engine.hpp"

#include <fstream>
#include <iterator>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<uint32_t> parse_ids(const std::string & arg) {  // "1,2,3" or "@FILE" holding that list (long prompts)
    std::string s = arg;
    if (!s.empty() && s[0] == '@') {
        std::ifstream in(s.substr(1));
        if (!in) throw std::runtime_error("cannot read " + s.substr(1));
        s.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    }
    std::vector<uint32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        v.push_back(uint32_t(std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i))));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: glm-gpu MODEL.gguf --tokens 1,2,3 [--gen N] [--force ids] [--context N] [--step] [--spec D] [--draft ids]");
        std::vector<uint32_t> prompt, force, draft;
        int gen = 8;
        uint32_t context = 2048;
        bool stepwise = false;
        int spec = 0;
        std::string dspark;
        tqllm::TpConfig tp;
        std::string hot_map;  // decode through verify blocks of spec + 1 tokens, committing the accepted prefix
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) prompt = parse_ids(argv[++i]);
            else if (a == "--gen" && i + 1 < argc) gen = std::stoi(argv[++i]);
            else if (a == "--force" && i + 1 < argc) force = parse_ids(argv[++i]);
            else if (a == "--context" && i + 1 < argc) context = uint32_t(std::stoul(argv[++i]));
            else if (a == "--step") stepwise = true;  // prompt one token at a time (default: chunked verify)
            else if (a == "--spec" && i + 1 < argc) spec = std::stoi(argv[++i]);
            else if (a == "--draft" && i + 1 < argc) draft = parse_ids(argv[++i]);
            else if (a == "--dspark" && i + 1 < argc) dspark = argv[++i];  // MiniMax: DSpark draft GGUF, --spec D drafts
            else if (a == "--tp-rank" && i + 1 < argc) tp.rank = std::stoi(argv[++i]), tp.world = 2;  // MiniMax on two linked boxes
            else if (a == "--tp-peer" && i + 1 < argc) tp.peer = argv[++i];
            else if (a == "--tp-port" && i + 1 < argc) tp.port = std::stoi(argv[++i]);
            else if (a == "--hot-map" && i + 1 < argc) hot_map = argv[++i];  // MiniMax: routing counts, hot experts preloaded
            else throw std::runtime_error("unknown option " + a);
        }
        if (prompt.empty()) throw std::runtime_error("--tokens is empty");
        auto e = tqllm::is_minimax_gguf(argv[1]) ? tqllm::make_minimax_engine(argv[1], context, dspark, "", tp, hot_map)
                                                  : tqllm::make_glm5next_engine(argv[1], context);
        std::vector<float> lg(e->vocab());
        auto top4 = [&](const char * tag, size_t i, uint32_t in) {
            if (cudaMemcpy(lg.data(), e->logits(), lg.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("logits copy failed");
            const float mx = *std::max_element(lg.begin(), lg.end());
            double z = 0;
            for (float v : lg) z += std::exp(double(v) - mx);
            const double lz = mx + std::log(z);
            std::vector<uint32_t> idx(lg.size());
            for (size_t k = 0; k < idx.size(); ++k) idx[k] = uint32_t(k);
            std::partial_sort(idx.begin(), idx.begin() + 4, idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] > lg[b]; });
            std::printf("%s %zu in %u:", tag, i, in);
            for (int k = 0; k < 4; ++k) std::printf(" (%u, %.3f)", idx[size_t(k)], double(lg[idx[size_t(k)]]) - lz);
            std::printf("\n");
            std::fflush(stdout);
            return idx[0];
        };
        if (tp.rank > 0) { e->follow(); return 0; }  // rank 1 mirrors rank 0's forwards
        if (!dspark.empty()) {  // the engine's own speculative loop with the DSpark draft
            uint64_t st[5] = {};
            const auto toks = e->generate_spec(prompt, uint32_t(gen), spec, st);
            std::printf("gen:");
            for (auto t : toks) std::printf(" %u", t);
            std::printf("\ndecode: %zu tokens in %.3f s = %.2f tok/s\nspec: %llu steps, %llu drafted, %llu accepted (%.2f tokens per step)\n",
                        toks.size(), st[3] / 1e6, toks.size() / (st[3] / 1e6), (unsigned long long)st[0], (unsigned long long)st[1],
                        (unsigned long long)st[2], st[0] ? double(toks.size()) / double(st[0]) : 0.0);
            return 0;
        }
        uint32_t pos = 0, next = 0;
        const auto p0 = std::chrono::steady_clock::now();
        if (stepwise || !force.empty()) {
            for (size_t i = 0; i < prompt.size(); ++i) { e->step(prompt[i], pos++, true, nullptr); next = top4("prompt", i, prompt[i]); }
        } else {
            std::vector<uint32_t> nx(prompt.size());
            e->verify(prompt.data(), int(prompt.size()), 0, nx.data());
            pos = uint32_t(prompt.size());
            top4("prompt(chunked)", prompt.size() - 1, prompt.back());
            next = nx.back();  // the engine's argmax (first index on an exact tie)
            if (lg[next] != *std::max_element(lg.begin(), lg.end())) throw std::runtime_error("verify argmax is not a maximum of its logits");
        }
        const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
        std::printf("prefill: %zu tokens in %.3f s = %.1f tok/s\n", prompt.size(), ps, double(prompt.size()) / ps);
        if (!force.empty()) {
            for (size_t i = 0; i + 1 < force.size(); ++i) { e->step(force[i], pos++, true, nullptr); top4("forced", i, force[i]); }
            return 0;
        }
        cudaProfilerStart();  // nsys --capture-range=cudaProfilerApi: decode only
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("gen:");
        long committed = 0;
        int blocks = 0, partial = 0, largest = 0;
        if (spec > 0) {
            // A trivial drafter: blk[0] is the token the previous block settled on, the rest are the predictions the last
            // verify made past its accepted prefix, so they are stale and mostly wrong. With --draft the block's tail comes
            // from the caller instead: the right ids accept (a full commit), a wrong one forces a mid-block rollback.
            // Either way the accepted prefix is exactly the greedy chain, so the printed tokens must equal the plain
            // decode's while commit(k) keeps rolling the KDA state and the conv history back into the middle of a block.
            std::vector<uint32_t> blk(size_t(spec) + 1), nx(blk.size()), keep(blk.size());
            std::fill(blk.begin(), blk.end(), 0u);
            blk[0] = next;
            uint32_t printed = 0;
            size_t di = 0;
            while (printed < uint32_t(gen)) {
                if (pos >= context) break;
                const int n = int(std::min<size_t>(blk.size(), size_t(context - pos)));
                const size_t d0 = di;  // --draft is indexed by the token position, so a commit of k consumes k proposals
                for (int j = 1; j < n && !draft.empty(); ++j) blk[size_t(j)] = di < draft.size() ? draft[di++] : 0u;
                e->verify(blk.data(), n, pos, nx.data());
                int k = 1;
                while (k < n && blk[size_t(k)] == nx[size_t(k) - 1]) ++k;
                e->commit(k);
                ++blocks;
                committed += k;
                largest = std::max(largest, n);
                if (k < n) ++partial;
                for (int j = 0; j < k && printed < uint32_t(gen); ++j) {
                    std::printf(" %u", blk[size_t(j)]);
                    std::fflush(stdout);
                    ++printed;
                }
                pos += uint32_t(k);
                if (!draft.empty()) di = d0 + size_t(k);
                blk[0] = nx[size_t(k) - 1];
                if (draft.empty()) {
                    keep = blk;
                    for (int j = 1; j < n; ++j)  // the rejected tail's (stale) predictions, or the older draft past the block
                        blk[size_t(j)] = (k - 1 + j < n) ? nx[size_t(k - 1 + j)] : keep[size_t(j)];
                }
            }
        } else
            for (int g = 0; g < gen; ++g) {
                std::printf(" %u", next);
                std::fflush(stdout);
                if (g + 1 < gen) next = e->step(next, pos++, true, nullptr);
            }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cudaProfilerStop();
        std::printf("\ndecode: %d tokens in %.3f s = %.2f tok/s\n", gen - 1, s, gen > 1 ? (gen - 1) / s : 0.0);
        if (spec > 0)
            std::printf("spec: %d verify blocks (largest %d), %ld tokens committed, %d committed part-way\n", blocks, largest,
                        committed, partial);
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "glm-gpu: %s\n", ex.what());
        return 1;
    }
}
