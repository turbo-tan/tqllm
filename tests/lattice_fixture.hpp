#pragma once
// The recorded 8K selector lattice (tests/data/lattice8k.bin, from the j1540 dump) for host tests: the loader, a lattice
// view per step for the tree selectors, and the acceptance a tree would get against the recorded greedy continuation.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "draft_tree.hpp"

namespace lattice_fixture {
constexpr int kTopK = 16, kRank = 256;
struct Step {
    int produced = 0;
    uint32_t anchor = 0;
    int count = 0;
    std::vector<uint32_t> truth;
    int rows = 0;
    std::vector<uint32_t> tokens;   // [rows * topk]
    std::vector<float> unary;       // [rows * topk]
    std::vector<float> gate;        // [rows * rank]
    std::vector<float> successor;   // [rows * topk * rank]
    std::vector<uint32_t> codebook_tokens;
    std::vector<float> predecessor; // [codebook_tokens * rank]
};

template <class T>
inline std::vector<T> read_array(std::ifstream& in, size_t count) {
    std::vector<T> out(count);
    in.read(reinterpret_cast<char*>(out.data()), std::streamsize(sizeof(T) * count));
    if (!in) throw std::runtime_error("fixture truncated");
    return out;
}
inline std::vector<Step> load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open fixture " + path);
    char magic[4];
    int version = 0, steps = 0, topk = 0, rank = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), sizeof(int));
    in.read(reinterpret_cast<char*>(&steps), sizeof(int));
    in.read(reinterpret_cast<char*>(&topk), sizeof(int));
    in.read(reinterpret_cast<char*>(&rank), sizeof(int));
    if (std::memcmp(magic, "DFLT", 4) || version != 1 || topk != kTopK || rank != kRank)
        throw std::runtime_error("bad fixture header");
    std::vector<Step> out;
    for (int s = 0; s < steps; ++s) {
        Step step;
        in.read(reinterpret_cast<char*>(&step.produced), sizeof(int));
        in.read(reinterpret_cast<char*>(&step.anchor), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&step.count), sizeof(int));
        in.read(reinterpret_cast<char*>(&step.rows), sizeof(int));
        if (!in || step.count < 1 || step.count > 15 || step.rows < step.count + 1 || step.rows > 16)
            throw std::runtime_error("bad fixture step");
        step.truth = read_array<uint32_t>(in, size_t(step.count));
        // the fixture carries every block row (row 0 is the anchor); the selector uses rows 1..count
        step.tokens = read_array<uint32_t>(in, size_t(step.rows) * kTopK);
        step.unary = read_array<float>(in, size_t(step.rows) * kTopK);
        step.gate = read_array<float>(in, size_t(step.rows) * kRank);
        step.successor = read_array<float>(in, size_t(step.rows) * kTopK * kRank);
        int codebook = 0;
        in.read(reinterpret_cast<char*>(&codebook), sizeof(int));
        if (!in || codebook < 1) throw std::runtime_error("bad fixture codebook");
        step.codebook_tokens = read_array<uint32_t>(in, size_t(codebook));
        step.predecessor = read_array<float>(in, size_t(codebook) * kRank);
        out.push_back(std::move(step));
    }
    return out;
}

// the recorded block row r of a step (r = 0 is the anchor row, rows 1..count are the drafted positions)
struct StepView {
    const Step& step;
    std::map<uint32_t, const float*> rows;
    StepView(const Step& s) : step(s) {
        for (size_t i = 0; i < step.codebook_tokens.size(); ++i)
            rows[step.codebook_tokens[i]] = step.predecessor.data() + i * kRank;
    }
    const float* predecessor_row(uint32_t token) const {
        auto it = rows.find(token);
        return it == rows.end() ? nullptr : it->second;
    }
    tqllm::DraftLattice lattice() {
        tqllm::DraftLattice l;
        l.positions = step.count;
        l.topk = kTopK;
        l.rank = kRank;
        l.tokens = step.tokens.data() + kTopK;              // skip the anchor row
        l.unary = step.unary.data() + kTopK;
        l.gate = step.gate.data() + kRank;                  // skip the anchor row
        l.successor = step.successor.data() + size_t(kTopK) * kRank;
        l.predecessor = &callback;
        l.context = this;
        return l;
    }
    static const float* callback(void* context, uint32_t token) {
        return static_cast<StepView*>(context)->predecessor_row(token);
    }
};

// deepest tree path that is a prefix of the recorded continuation, plus the bonus token
inline int accepted(const tqllm::DraftTree& tree, const std::vector<uint32_t>& truth) {
    std::vector<int> depth(tree.tokens.size(), 0);
    for (size_t row = 0; row < tree.tokens.size(); ++row)
        depth[row] = tree.parents[row] < 0 ? 0 : depth[size_t(tree.parents[row])] + 1;
    int best = 0;
    for (size_t row = 0; row < tree.tokens.size(); ++row) {
        std::vector<uint32_t> path;
        for (int node = int(row); node >= 0; node = tree.parents[size_t(node)]) path.push_back(tree.tokens[size_t(node)]);
        if (int(path.size()) <= int(truth.size()) &&
            std::equal(path.rbegin(), path.rend(), truth.begin())) {
            best = std::max(best, int(path.size()));
        }
    }
    return best + 1;
}
}  // namespace lattice_fixture
