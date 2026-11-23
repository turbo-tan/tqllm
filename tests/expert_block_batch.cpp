#include "expert_block_batch.hpp"
#include <array>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    const int ids[]{3, 1, 2, 0};
    std::vector<char> out(16, 0);
    auto load = [](int id) { return std::array<std::vector<char>, 2>{std::vector<char>(2, char(id)), std::vector<char>(2, char(id + 10))}; };
    bool caught = false;
    try {
        tqllm::expert_block_batch(ids, 4, out.data(), 4, [&](int id) {
            if (id == 2) throw std::runtime_error("injected expert read failure");
            return load(id);
        });
    } catch (const std::runtime_error & e) { caught = std::string(e.what()) == "injected expert read failure"; }
    if (!caught) return 1;
    tqllm::expert_block_batch(ids, 4, out.data(), 4, load);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (out[i * 4 + j] != char(ids[i] + (j >= 2 ? 10 : 0))) return 2;
    for (size_t block : {size_t(3), size_t(5)}) {
        bool rejected = false;
        try { tqllm::expert_block_batch(ids, 1, out.data(), block, load); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) return 3;
    }
    tqllm::expert_block_batch(ids, 0, nullptr, 4, load);
    std::puts("PASS expert block batch exception recovery and byte order");
    return 0;
}
